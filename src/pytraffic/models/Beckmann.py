import numpy as np
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
def aon_assign(csr: CSRGraph, weight: np.ndarray, D: np.ndarray, use_numba: bool = True) -> Tuple[np.ndarray, float]:
    """
    All-or-Nothing loading when ALL nodes are zones.

    csr    : CSRGraph with n nodes
    weight : (m,) edge weights
    D      : (n,n) OD matrix over nodes 0..n-1

    return:
      y       : (m,) AON flows
      sp_cost : sum_{o,d} D[o,d] * dist_o[d]
    """
    weight = np.asarray(weight, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    if D.shape != (csr.n, csr.n):
        raise ValueError(f"D must have shape ({csr.n},{csr.n}), got {D.shape}")

    return _aon_assign_py(csr, weight, D)


def _aon_assign_py(csr: CSRGraph, weight: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Pure NumPy implementation (fallback when numba is unavailable).
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
    use_numba=True,
):
    """
    Метод Франка-Вульфа для решения задачи TA по модели Бэкманна. Считает само решение и его 
    градиент по матрице корреспонденции.

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
        y, total_cost_k, gradient_k = aon_assign(csr, edge_cost_field, D, use_numba=use_numba)

        # Шаг аглоритма Франка-Вульфа
        gamma = 2.0 / (k + 2.0)
        flow = (1.0 - gamma) * flow + gamma * y
        gradient = (1.0 - gamma) * gradient + gamma * gradient_k

        new_edge_cost_field = edge_cost(flow)
        rg = stop_criterion(flow, new_edge_cost_field, total_cost_k)

        # Это просто полезный вывод
        if verbose and (k == 1 or k % 10 == 0 or rg <= rgap_target):
            print(f"iter={k:4d}  gamma={gamma:.6f}  rgap={rg:.3e} grad_norm={np.linalg.norm(gradient):.6f}")

        if rg <= rgap_target:
            break

    return flow, gradient
