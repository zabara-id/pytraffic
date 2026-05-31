import numpy as np
import warnings

from graph import CSRGraph
from typing import Tuple

def stop_criterion(flow, time, sp_cost):
    """
    Остановка по относительной ошибке
    """
    cur_cost = np.dot(flow, time)
    return max(abs((cur_cost - sp_cost) / cur_cost), 0.0)

# ============================================================
# All-or-Nothing assignment (FW direction)
# ============================================================
def aon_assign(csr: CSRGraph, weight: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    All-or-Nothing loading when ALL nodes are zones.

    csr    : CSRGraph with n nodes
    weight : (m,) edge weights
    D      : (n,n) OD matrix over nodes 0..n-1

    return:
      y       : (m,) AON flows
      sp_cost : sum_{o,d} D[o,d] * dist_o[d]
      gradient: градиент AON решение по матрице корреспонденции 
                (необходимо в расчете градиента оптимальной точки в модели Бэкмана)
    """

    weight = np.asarray(weight, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))

    solution = np.zeros(csr.m, dtype=np.float64)
    total_assignment_cost = 0.0

    for origin in range(csr.n):
        # 1) Считаем кратчайшие пути 
        # для всех остальных вершин
        dist, previous_edges = csr.dijkstra(weight, origin)
        
        # Понимаем, а какая корреспонденция 
        # должна по ним ехать
        corresp_to_assign = D[origin, :]
        # Расчет итоговой стоимости системы
        total_assignment_cost += float(np.dot(corresp_to_assign, dist))

        # 2) Пересчет потоков по путям в потоки по рёбрам
        # 
        # Этот пересчет можно сделать при помощи матричного умножения:
        # solution = PATH_TO_EDGE_INSIDENCE @ assign(corresp_to_assign),
        # где assign(corresp_to_assing) = вектор, в котором корреспонденция поставлена на нужные пути 
        # но для этого нужно хранить матрицу + модифицировать выход алгоритма Дейкстры
        for destanation, correspodence in enumerate(corresp_to_assign):
            cur = int(destanation)
            # раскручивание кратчайшего пути для 
            # добавления соотвествующей корреспонденции на ребро
            while cur != origin:
                e = int(previous_edges[cur])
                if e < 0:
                    break
                solution[e] += correspodence
                # Предполагаемый счет градиента
                gradient[e, origin * csr.n + destanation] = 1
                cur = int(csr.tail[e])

    return solution, total_assignment_cost, gradient
    

# ============================================================
# Frank–Wolfe (Beckmann UE)
# ============================================================
def fw_beckmann(
    csr: CSRGraph,
    edge_cost,
    D : np.ndarray,
    max_iter=500,
    rgap_target=1e-4,
    verbose=True,
):
    """
    Метод Франк-Вульфа для решения задачи TA по модели Бэкманна. 
    Считает само решение и его градиент по матрице корреспонденции для классческий модели Бэкманна

    csr: CSRGraph -- представление графа в формате CSR для нахождения исходящих рёбер,
    edge_cost: callable -- функция стоимости ребра в зависимости от нагрузки,
    D: np.ndarray (n, n), n -- количество вершин в графе (они все и origin, и destination),
    use_numba: bool -- использовать ускоренный AON через numba (если установлена),

    """
    flow = np.zeros(csr.m, dtype=np.float64)
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))

    for k in range(1, max_iter + 1):
        # Решаем ЛП на заданном множестве
        edge_cost_field = edge_cost(flow)
        y, total_cost_k, gradient_k = aon_assign(csr, edge_cost_field, D)

        # Шаг аглоритма Франка-Вульфа
        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y
        gradient = (1.0 - gamma) * gradient + gamma * gradient_k

        new_edge_cost_field = edge_cost(flow)
        rg = stop_criterion(flow, new_edge_cost_field, total_cost_k)

        # Это просто полезный вывод
        if verbose and (k == 1 or k % 10 == 0 or rg <= rgap_target):
            print(f"iter={k:4d}  gamma={gamma:.6f}  rgap={rg:.3e} grad_norm={np.linalg.norm(gradient):.6f} step_diff={np.linalg.norm(gamma / (1 - gamma) * (y - flow)) / np.linalg.norm(flow)}")

        if rg <= rgap_target:
            break

    return flow, gradient


def fw_beckmann_regularized(
    csr: CSRGraph,
    edge_cost,
    D : np.ndarray,
    f_hat, 
    experiment_mask,
    alpha = 1e-2,
    max_iter=500,
    rgap_target=1e-4,
    verbose=True,
):
    """
    Метод Франк-Вульфа для решения задачи TA по модели Бэкманна c регуляризацией. 
    Регуляризация записывается в виде $beckman_potential + \alpha \| f - \hat{f}\|_{2}^2$ и фактически означает,
    что на каждой итерации в модели Бэкманна будет происходить смена весов на рэбрах графа.
    Считает само решение и его градиент по матрице корреспонденции.

    csr: CSRGraph -- представление графа в формате CSR для нахождения исходящих рёбер,
    edge_cost: callable -- функция стоимости ребра в зависимости от нагрузки,
    D: np.ndarray (n, n), n -- количество вершин в графе (они все и origin, и destination),
    f_hat: np.ndarray (m, ) -- вектор из известных потоков на рёбрах, входит в регуляризатор
    experiment_mask: np.ndarray (m, ) -- вектора-маска на те рёбра, на которых нам реально известны потоки
    alpha: np.float64 -- параметр регуляризации к модели Бэкманна
    """

    # приближение к решению задачи минимизации 
    # (начальное приближение можно брать нулём, а можно как решение равновесной задачи)
    # мне кажется, что взять с качестве начального приближения решение модели бэкмана будет нормальной темой,
    # так как этот вектор сразу в допустимом множестве, а нулевой -- нет
    flow, _ = fw_beckmann(csr, edge_cost, D, max_iter=500, rgap_target=1e-4, verbose=False)
    # приближение к градиенту решения задачи минимизации по матрице корреспонденции
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))
    # приближение к градиенту оптимального значения миниизируемой функции по матрице корреспонденции
    gradint_of_optimal_potential_value = np.zeros(shape=csr.n * csr.n)

    for k in range(1, max_iter + 1):
        # Решаем ЛП на заданном множестве 
        # Из-за регуляризации как раз приедет коррекция стоимости рёбер на величину alpha * experimental_mask * (f - f_hat)
        # Получается, что если поток f на ребре не дотягивает до экспериментального, то стоимость движения по ребру уменьшается,
        # что дает больший поток на этом ребре (вероятно). Если f больше экспериментального, то стоимость увеличивается и поездка по ребру становится "дорогой",
        # так что на этой итерации тут мало кто поедет
        
        # TODO: Возможно исходя из такой трактовки следует как-то перепридумать регуляризатор, но пока это в будущем

        # Новый рачет поля стоимости на ребрах, с учетом регуляризации
        additional_cost =  alpha * experiment_mask * (flow - f_hat)
        edge_cost_field = edge_cost(flow) + additional_cost

        # АААААААААААААААААА: В этом ифе сделана дофольно маргинальная вещь
        if (np.min(edge_cost_field) < 0):
            idx = np.argmin(edge_cost_field)
            warnings.warn(f"стоимость ребра {idx} на {k} итерации оказалась отрицательной, t(f_e) = {edge_cost(flow)[idx]}, add_cost = {additional_cost[idx]}. \
                             Возможно, необходимо уменьшить регуляризатор alpha")
            # Если уж стоимость ребра оказалась отрицательной, то будет считать, что по этому ребру стоимость просто нуль
            # ААААААААААААААААААААААААААА: Но это влияет на функционал, он должен быть модифицирован
            edge_cost_field[edge_cost_field < 0] = 0
    
        y, total_cost_k, gradient_k = aon_assign(csr, edge_cost_field, D)

        # Шаг аглоритма Франк-Вульфа
        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y
        gradient = (1.0 - gamma) * gradient + gamma * gradient_k

        new_edge_cost_field = edge_cost(flow) + alpha * experiment_mask * (flow - f_hat)
        # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA: И вот следующая строка тоже очень маргинальная
        new_edge_cost_field[new_edge_cost_field < 0] = 0
        rg = stop_criterion(flow, new_edge_cost_field, total_cost_k)

        # Это просто полезный вывод
        if verbose and (k == 1 or k % 10 == 0 or rg <= rgap_target):
            print(f"iter={k:4d}  gamma={gamma:.6f}  rgap={rg:.3e} grad_norm={np.linalg.norm(gradient):.6f} step_diff={np.linalg.norm(gamma / (1 - gamma) * (y - flow)) / np.linalg.norm(flow)}")

        if rg <= rgap_target:
            break

    # Расчет градиента от функции F(D)
    beckmann_part =  edge_cost(flow)
    additional_part = alpha * experiment_mask * (flow - f_hat)
    gradint_of_optimal_potential_value = ((beckmann_part + additional_part) @ gradient)
    
    return flow, gradient, gradint_of_optimal_potential_value

def fw_beckmann_regularized_new_gradient(
    csr: CSRGraph,
    edge_cost,
    D : np.ndarray,
    f_hat, 
    experiment_mask,
    alpha = 1e-2,
    max_iter=500,
    rgap_target=1e-4,
    verbose=True,
):
    """
    Метод Франк-Вульфа для решения задачи TA по модели Бэкманна c регуляризацией. 
    Регуляризация записывается в виде $beckman_potential + \alpha \| f - \hat{f}\|_{2}^2$ и фактически означает,
    что на каждой итерации в модели Бэкманна будет происходить смена весов на рэбрах графа.
    Считает само решение и его градиент (новым способом) по матрице корреспонденции.

    csr: CSRGraph -- представление графа в формате CSR для нахождения исходящих рёбер,
    edge_cost: callable -- функция стоимости ребра в зависимости от нагрузки,
    D: np.ndarray (n, n), n -- количество вершин в графе (они все и origin, и destination),
    f_hat: np.ndarray (m, ) -- вектор из известных потоков на рёбрах, входит в регуляризатор
    experiment_mask: np.ndarray (m, ) -- вектора-маска на те рёбра, на которых нам реально известны потоки
    alpha: np.float64 -- параметр регуляризации к модели Бэкманна
    """

    # приближение к решению задачи минимизации 
    # (начальное приближение можно брать нулём, а можно как решение равновесной задачи)
    # мне кажется, что взять с качестве начального приближения решение модели бэкмана будет нормальной темой,
    # так как этот вектор сразу в допустимом множестве, а нулевой -- нет
    flow, _ = fw_beckmann(csr, edge_cost, D, max_iter=500, rgap_target=1e-4, verbose=False)
    # приближение к градиенту решения задачи минимизации по матрице корреспонденции
    gradient = np.zeros(shape=(csr.m, csr.n * csr.n))

    for k in range(1, max_iter + 1):
        # Решаем ЛП на заданном множестве 
        # Из-за регуляризации как раз приедет коррекция стоимости рёбер на величину alpha * experimental_mask * (f - f_hat)
        # Получается, что если поток f на ребре не дотягивает до экспериментального, то стоимость движения по ребру уменьшается,
        # что дает больший поток на этом ребре (вероятно). Если f больше экспериментального, то стоимость увеличивается и поездка по ребру становится "дорогой",
        # так что на этой итерации тут мало кто поедет
        
        # TODO: Возможно исходя из такой трактовки следует как-то перепридумать регуляризатор, но пока это в будущем

        # Новый рачет поля стоимости на ребрах, с учетом регуляризации
        additional_cost =  alpha * experiment_mask * (flow - f_hat)
        edge_cost_field = edge_cost(flow) + additional_cost

        # АААААААААААААААААА: В этом ифе сделана дофольно маргинальная вещь
        if (np.min(edge_cost_field) < 0):
            idx = np.argmin(edge_cost_field)
            warnings.warn(f"стоимость ребра {idx} на {k} итерации оказалась отрицательной, t(f_e) = {edge_cost(flow)[idx]}, add_cost = {additional_cost[idx]}. \
                             Возможно, необходимо уменьшить регуляризатор alpha")
            # Если уж стоимость ребра оказалась отрицательной, то будет считать, что по этому ребру стоимость просто нуль
            # ААААААААААААААААААААААААААА: Но это влияет на функционал, он должен быть модифицирован
            edge_cost_field[edge_cost_field < 0] = 0
    
        y, total_cost_k, gradient_k = aon_assign(csr, edge_cost_field, D)

        # Шаг аглоритма Франк-Вульфа
        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y
        gradient = (1.0 - gamma) * gradient + gamma * gradient_k

        new_edge_cost_field = edge_cost(flow) + alpha * experiment_mask * (flow - f_hat)
        # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA: И вот следующая строка тоже очень маргинальная
        new_edge_cost_field[new_edge_cost_field < 0] = 0
        rg = stop_criterion(flow, new_edge_cost_field, total_cost_k)

        # Это просто полезный вывод
        if verbose and (k == 1 or k % 10 == 0 or rg <= rgap_target):
            print(f"iter={k:4d}  gamma={gamma:.6f}  rgap={rg:.3e} grad_norm={np.linalg.norm(gradient):.6f} step_diff={np.linalg.norm(gamma / (1 - gamma) * (y - flow)) / np.linalg.norm(flow)}")

        if rg <= rgap_target:
            break

    # Расчет градиента от функции F(D)
    # 1. Считаем равновесные цена ребёр, с учетом модификации по невязке
    equilibrium_edge_costs = edge_cost(flow) + alpha * experiment_mask * (flow - f_hat)
    # 2. Считаем кратчайшие пути между всеми вершинами
    gradint_of_optimal_potential_value = csr.all_pairs_shortest_distances(equilibrium_edge_costs)
    # Это и есть градиент по матрице корреспонденции, потому что добавленная малая корреспонденция delta_{ij}
    # поедет по кратчайшему пути между вершинами i и j, а значит, увеличатся потоки на инцидентных этому пути 
    # рёбрах и величины sigma_i будет изменяться на edge_k_cost(f_k + delta_{ij}).
    # 
    # Это верно в случае, когда оптимальный путь уникален. Если оптимальных путей больше, то кажется, что функция
    # недифференцируема в этом месте.
    
    return flow, gradient, gradint_of_optimal_potential_value
