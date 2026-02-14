import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import wrightomega

from pytraffic.graph.CSRGraph import CSRGraph
from pytraffic.models.BRPCost import BRP
import pytraffic.models.Beckmann as beckmann


# ---------------------------
# Построение операторов маргиналов для row-major vec(D)
# ---------------------------
def build_L_row_col(n: int):
    """
    Для d = vec(D) в row-major: k = i*n + j.

    L_row @ d = суммы по строкам (ell_hat), размер (n,)
    L_col @ d = суммы по столбцам (w_hat), размер (n,)
    """
    nn = n * n
    i = np.repeat(np.arange(n), n)      # индекс строки в D
    j = np.tile(np.arange(n), n)        # индекс столбца в D
    k = i * n + j                       # индекс в vec

    data = np.ones(nn, dtype=np.float64)

    L_row = sp.coo_matrix((data, (i, k)), shape=(n, nn)).tocsr()  # сумма по j для каждого i
    L_col = sp.coo_matrix((data, (j, k)), shape=(n, nn)).tocsr()  # сумма по i для каждого j
    return L_row, L_col


# ---------------------------
# Прокс энтропии + неотрицательности (поэлементно), используется в z-обновлении ADMM
# ---------------------------
def prox_entropy_nonneg(y: np.ndarray, rho: float, gamma: float) -> np.ndarray:
    """
    z = argmin_{z>=0} gamma * sum z log z + (rho/2) ||z - y||^2

    Формула в замкнутом виде:
      z = (gamma/rho) * W( (rho/gamma) * exp( (rho/gamma)*y - 1 ) )

    Численно устойчивая реализация:
      W(exp(a)) = wrightomega(a), чтобы избежать overflow при больших положительных a.
    """
    assert rho > 0 and gamma > 0
    scale = rho / gamma
    log_arg = np.log(scale) + scale * y - 1.0
    z = (gamma / rho) * np.real(wrightomega(log_arg))
    z = np.maximum(z, 0.0)
    return z


# ---------------------------
# ADMM-прокс для g(d): мягкие маргиналы, неотрицательность и нулевая диагональ
# ---------------------------
def prox_g_admm(
    v: np.ndarray,          # (n^2,)
    eta: float,
    gamma: float,
    alpha: float,
    ell: np.ndarray,        # (n,)
    w: np.ndarray,          # (n,)
    L_row: sp.csr_matrix,
    L_col: sp.csr_matrix,
    rho: float = 1.0,
    max_iter: int = 2000,
    tol_pri: float = 1e-5,
    tol_dual: float = 1e-5,
    diag_zero: bool = True,
):
    """
    Решает задачу:
      min_d  (1/(2eta))||d - v||^2 + gamma*sum d log d + alpha(||L_row d - ell||^2 + ||L_col d - w||^2)
      при d >= 0 и diag(D)=0

    ADMM-разбиение: d (квадратичная часть) и z (энтропия + неотрицательность), связь d=z.

    d-обновление: решение разреженной квадратичной задачи;
    z-обновление: поэлементный LambertW;
    u-обновление: стандартное для ADMM.
    """
    n = ell.size
    nn = n * n
    assert v.shape == (nn,)
    assert eta > 0 and gamma > 0 and alpha >= 0 and rho > 0

    ell = np.asarray(ell, float).reshape(n)
    w = np.asarray(w, float).reshape(n)

    # Маска диагонали (если нужна): решаем только по внедиагональным переменным
    diag_mask = np.zeros(nn, dtype=bool)
    if diag_zero:
        for i in range(n):
            diag_mask[i * n + i] = True
    free = ~diag_mask
    idx = np.where(free)[0]
    m = idx.size  # = n^2 - n

    # Редуцированные операторы:
    # L_row_red = L_row[:, idx], аналогично для столбцов
    Lr = L_row[:, idx].tocsr()
    Lc = L_col[:, idx].tocsr()

    # Квадратичная матрица для d-обновления:
    # H = (1/eta) I + 2alpha (Lr^T Lr + Lc^T Lc) + rho I
    I = sp.eye(m, format="csr")
    AtA = (Lr.T @ Lr) + (Lc.T @ Lc)
    H = (1.0 / eta) * I + 2.0 * alpha * AtA + rho * I
    H_factor = spla.factorized(H.tocsc())  # быстрые решения

    # Правая часть для штрафа по маргиналам: 2alpha (Lr^T ell + Lc^T w)
    b_marg = 2.0 * alpha * (Lr.T @ ell + Lc.T @ w)  # (m,)

    # Инициализация
    d = np.maximum(v, 0.0).copy()
    if diag_zero:
        d[diag_mask] = 0.0
    d_red = d[idx]

    z_red = d_red.copy()
    u_red = np.zeros_like(d_red)

    pri_hist = []
    dual_hist = []

    for _ in range(max_iter):
        z_old = z_red.copy()

        # ---- d-обновление (квадратичное решение) ----
        rhs = (1.0 / eta) * v[idx] + b_marg + rho * (z_red - u_red)
        d_red = H_factor(rhs)

        # ---- z-обновление (энтропия + неотрицательность) ----
        y = d_red + u_red
        z_red = prox_entropy_nonneg(y, rho=rho, gamma=gamma)

        # ---- u-обновление ----
        u_red = u_red + (d_red - z_red)

        # Невязки ADMM
        pri = np.linalg.norm(d_red - z_red)
        dual = rho * np.linalg.norm(z_red - z_old)
        pri_hist.append(pri)
        dual_hist.append(dual)

        if pri < tol_pri and dual < tol_dual:
            break

    # Собираем полный вектор d
    d_out = np.zeros(nn, dtype=float)
    d_out[idx] = z_red  # берем z как примал (гарантирует неотрицательность)
    if diag_zero:
        d_out[diag_mask] = 0.0

    info = {
        "admm_iters": len(pri_hist),
        "pri": pri_hist[-1] if pri_hist else None,
        "dual": dual_hist[-1] if dual_hist else None,
    }
    return d_out, info


def init_outer_marginals(ell: np.ndarray, w: np.ndarray, diag_zero: bool = True) -> np.ndarray:
    n = ell.size
    D0 = np.outer(ell, w) / (ell.sum() + 1e-12)
    if diag_zero:
        np.fill_diagonal(D0, 0.0)
        D0 *= ell.sum() / (D0.sum() + 1e-12)
        np.fill_diagonal(D0, 0.0)
    return D0.reshape(-1)


def safe_semilogy(x, y, **kwargs):
    y = np.asarray(y)
    if np.any(y > 0):
        plt.semilogy(x, np.maximum(y, 1e-300), **kwargs)
    else:
        plt.plot(x, y, **kwargs)


if __name__ == "__main__":
    # =========================
    # Пример данных
    # =========================
    n_nodes = 4

    tail = np.array([0, 1, 0, 2, 1, 2, 1, 3, 2, 3], dtype=np.int32)
    head = np.array([1, 0, 2, 0, 2, 1, 3, 1, 3, 2], dtype=np.int32)
    csr = CSRGraph.from_edges(n_nodes, tail, head)

    cap = np.array([20, 20, 15, 15, 15, 15, 20, 20, 15, 15], dtype=np.float64)
    t0  = np.array([6, 4, 5, 5, 4, 4, 6, 4, 5, 5], dtype=np.float64)
    alpha_bpr = np.full_like(t0, 0.15)
    beta_bpr  = np.full_like(t0, 4.0)
    edge_cost = BRP(cap, t0, alpha_bpr, beta_bpr)

    D_true = np.array([
        [0, 100, 50,  0],
        [80,  0,  20, 10],
        [40, 10,  0, 30],
        [0,  30, 10,  0],
    ], dtype=np.float64)

    f_hat = np.zeros_like(cap)
    f_hat[0] = 90
    f_hat[1] = 67
    # При наблюдении только двух рёбер обратная задача недоопределена:
    # можно хорошо согласовать маргиналы и измеренные потоки, но D_true не единственна.
    experiment_mask = np.zeros_like(f_hat)
    experiment_mask[0] = 1
    experiment_mask[1] = 1

    # Целевые маргиналы (мягкое соблюдение через штраф)
    ell = D_true.sum(axis=1)
    w = D_true.sum(axis=0)

    # Операторы суммирования по строкам/столбцам
    L_row, L_col = build_L_row_col(n_nodes)

    # =========================
    # Параметры прокс-градиента
    # =========================
    K = 120
    eta0 = 1e-4          # базовый шаг (с backtracking)
    gamma = 1e-2         # вес энтропии
    alpha = 300.0        # сильный штраф для согласования маргиналов
    rho = 1.0            # штраф ADMM

    admm_max_iter = 2000
    admm_tol_pri = 1e-6
    admm_tol_dual = 1e-6

    # Начальное приближение
    d = init_outer_marginals(ell, w, diag_zero=True)
    prev_d = d.copy()

    # Логи метрик
    hist_eta = []
    hist_step = []
    hist_gnorm = []
    hist_row_err = []
    hist_col_err = []
    hist_flow_misfit = []
    hist_f0 = []
    hist_f1 = []
    hist_rel_fro = []
    hist_admm_iters = []
    hist_grad_step = []
    hist_prox_correction = []
    hist_net_step_abs = []

    norm_true = np.linalg.norm(D_true) + 1e-12

    for k in range(K):
        d_before_update = d.copy()
        Dmat = d.reshape(n_nodes, n_nodes)

        # ---- Градиент части Бекмана ----
        flow, gradient, grad_F_over_D = beckmann.fw_beckmann_regularized_marginal(
            csr, edge_cost, Dmat, f_hat, experiment_mask, alpha=0.9, max_iter=5000, verbose=False
        )
        g = grad_F_over_D.reshape(n_nodes, n_nodes)
        np.fill_diagonal(g, 0.0)
        gnorm = np.linalg.norm(g)

        # ---- Backtracking по eta (принимаем, если prox-решение корректно) ----
        eta = eta0
        accepted = False
        info = None
        accepted_v = None

        for _trial in range(20):
            v = d_before_update - eta * g.reshape(-1)

            d_try, info = prox_g_admm(
                v=v,
                eta=eta, gamma=gamma, alpha=alpha,
                ell=ell, w=w,
                L_row=L_row, L_col=L_col,
                rho=rho,
                max_iter=admm_max_iter,
                tol_pri=admm_tol_pri,
                tol_dual=admm_tol_dual,
                diag_zero=True
            )

            # Критерий принятия: конечные значения и разумная суммарная масса
            if np.isfinite(d_try).all() and d_try.sum() > 0.2 * ell.sum():
                d = d_try
                accepted_v = v
                accepted = True
                break
            eta *= 0.5

        if not accepted:
            # Если не удалось принять шаг, уменьшаем базовый eta0
            eta0 *= 0.5

        # ---- Метрики ----
        step = np.linalg.norm(d - prev_d) / (np.linalg.norm(prev_d) + 1e-12)
        prev_d = d.copy()
        grad_step = np.linalg.norm(eta * g.reshape(-1))
        prox_correction = np.linalg.norm(d - accepted_v) if accepted_v is not None else np.nan
        net_step_abs = np.linalg.norm(d - d_before_update)

        Drec = d.reshape(n_nodes, n_nodes)
        row_err = np.linalg.norm(Drec.sum(axis=1) - ell) / (np.linalg.norm(ell) + 1e-12)
        col_err = np.linalg.norm(Drec.sum(axis=0) - w) / (np.linalg.norm(w) + 1e-12)

        diff = (flow - f_hat) * experiment_mask
        flow_misfit = np.linalg.norm(diff) / (np.linalg.norm(f_hat * experiment_mask) + 1e-12)

        rel_fro = np.linalg.norm(Drec - D_true) / norm_true

        hist_eta.append(eta)
        hist_step.append(step)
        hist_gnorm.append(gnorm)
        hist_row_err.append(row_err)
        hist_col_err.append(col_err)
        hist_flow_misfit.append(flow_misfit)
        hist_f0.append(flow[0])
        hist_f1.append(flow[1])
        hist_rel_fro.append(rel_fro)
        hist_admm_iters.append(info["admm_iters"] if info is not None else 0)
        hist_grad_step.append(grad_step)
        hist_prox_correction.append(prox_correction)
        hist_net_step_abs.append(net_step_abs)

        print(
            f"iter={k:03d} eta={eta:.1e} ||G||={gnorm:.3e} relStep={step:.3e} "
            f"rowErr={row_err:.2e} colErr={col_err:.2e} flowMisfit={flow_misfit:.3e} "
            f"admmIters={hist_admm_iters[-1]} gradStep={grad_step:.3e} "
            f"proxCorr={prox_correction:.3e} netStep={net_step_abs:.3e}"
        )

    # =========================
    # Графики
    # =========================
    it = np.arange(K)

    plt.figure()
    safe_semilogy(it, hist_step)
    plt.xlabel("итерация")
    plt.ylabel("||d_{k+1}-d_k|| / ||d_k||")
    plt.title("Относительный размер обновления")
    plt.grid(True)

    plt.figure()
    safe_semilogy(it, hist_gnorm)
    plt.xlabel("итерация")
    plt.ylabel("||grad F||_F")
    plt.title("Норма градиента части Бекмана")
    plt.grid(True)

    plt.figure()
    safe_semilogy(it, hist_row_err, label="ошибка по строкам")
    safe_semilogy(it, hist_col_err, label="ошибка по столбцам")
    plt.xlabel("итерация")
    plt.ylabel("относительная ошибка маргиналов")
    plt.title("Ошибки маргиналов (мягкий штраф, ADMM-прокс)")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(it, hist_flow_misfit)
    plt.xlabel("итерация")
    plt.ylabel("относительная невязка (по маске)")
    plt.title("Согласование с наблюдаемыми потоками на рёбрах")
    plt.grid(True)

    plt.figure()
    plt.plot(it, hist_f0, label="поток на ребре 0")
    plt.plot(it, hist_f1, label="поток на ребре 1")
    plt.axhline(f_hat[0], linestyle="--", label="наблюдение f_hat[0]")
    plt.axhline(f_hat[1], linestyle="--", label="наблюдение f_hat[1]")
    plt.xlabel("итерация")
    plt.ylabel("поток")
    plt.title("Согласование по каждому наблюдаемому ребру")
    plt.legend()
    plt.grid(True)

    plt.figure()
    safe_semilogy(it, hist_rel_fro)
    plt.xlabel("итерация")
    plt.ylabel("||D-D_true||_F / ||D_true||_F")
    plt.title("Ошибка восстановления OD-матрицы")
    plt.grid(True)

    plt.figure()
    safe_semilogy(it, hist_eta)
    plt.xlabel("итерация")
    plt.ylabel("использованное eta")
    plt.title("Размер шага после backtracking")
    plt.grid(True)

    plt.figure()
    plt.plot(it, hist_admm_iters)
    plt.xlabel("итерация")
    plt.ylabel("число итераций ADMM")
    plt.title("Затраты ADMM на один prox-шаг")
    plt.grid(True)

    plt.show()

    np.set_printoptions(precision=6, suppress=True)
    print("\nD_true (истинная матрица):\n", D_true)
    print("\nD_recovered (восстановленная матрица):\n", d.reshape(n_nodes, n_nodes))
    print("\nВосстановленные суммы по строкам:", d.reshape(n_nodes, n_nodes).sum(axis=1))
    print("Восстановленные суммы по столбцам:", d.reshape(n_nodes, n_nodes).sum(axis=0))
