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
# Прокс-утилиты (как в прототипе)
# ---------------------------
def build_L_row_col(n: int):
    nn = n * n
    i = np.repeat(np.arange(n), n)
    j = np.tile(np.arange(n), n)
    k = i * n + j
    data = np.ones(nn, dtype=np.float64)
    L_row = sp.coo_matrix((data, (i, k)), shape=(n, nn)).tocsr()
    L_col = sp.coo_matrix((data, (j, k)), shape=(n, nn)).tocsr()
    return L_row, L_col


def prox_entropy_nonneg(y: np.ndarray, rho: float, gamma: float) -> np.ndarray:
    assert rho > 0 and gamma > 0
    scale = rho / gamma
    log_arg = np.log(scale) + scale * y - 1.0
    z = (gamma / rho) * np.real(wrightomega(log_arg))
    z = np.maximum(z, 0.0)
    return z


def prox_g_admm(
    v: np.ndarray,
    eta: float,
    gamma: float,
    alpha: float,
    ell: np.ndarray,
    w: np.ndarray,
    L_row: sp.csr_matrix,
    L_col: sp.csr_matrix,
    rho: float = 1.0,
    max_iter: int = 1500,
    tol_pri: float = 1e-6,
    tol_dual: float = 1e-6,
    diag_zero: bool = True,
):
    n = ell.size
    nn = n * n
    assert v.shape == (nn,)
    assert eta > 0 and gamma > 0 and alpha >= 0 and rho > 0

    ell = np.asarray(ell, float).reshape(n)
    w = np.asarray(w, float).reshape(n)

    diag_mask = np.zeros(nn, dtype=bool)
    if diag_zero:
        for i in range(n):
            diag_mask[i * n + i] = True
    idx = np.where(~diag_mask)[0]
    m = idx.size

    Lr = L_row[:, idx].tocsr()
    Lc = L_col[:, idx].tocsr()

    I = sp.eye(m, format="csr")
    AtA = (Lr.T @ Lr) + (Lc.T @ Lc)
    H = (1.0 / eta) * I + 2.0 * alpha * AtA + rho * I
    H_factor = spla.factorized(H.tocsc())
    b_marg = 2.0 * alpha * (Lr.T @ ell + Lc.T @ w)

    d_red = np.maximum(v[idx], 0.0).copy()
    z_red = d_red.copy()
    u_red = np.zeros_like(d_red)

    pri_hist = []
    dual_hist = []
    for _ in range(max_iter):
        z_old = z_red.copy()
        rhs = (1.0 / eta) * v[idx] + b_marg + rho * (z_red - u_red)
        d_red = H_factor(rhs)
        z_red = prox_entropy_nonneg(d_red + u_red, rho=rho, gamma=gamma)
        u_red = u_red + (d_red - z_red)

        pri = np.linalg.norm(d_red - z_red)
        dual = rho * np.linalg.norm(z_red - z_old)
        pri_hist.append(pri)
        dual_hist.append(dual)
        if pri < tol_pri and dual < tol_dual:
            break

    d_out = np.zeros(nn, dtype=float)
    d_out[idx] = z_red
    if diag_zero:
        d_out[diag_mask] = 0.0

    info = {
        "admm_iters": len(pri_hist),
        "pri": pri_hist[-1] if pri_hist else None,
        "dual": dual_hist[-1] if dual_hist else None,
    }
    return d_out, info


def init_outer_marginals(ell: np.ndarray, w: np.ndarray, diag_zero: bool = True) -> np.ndarray:
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


# ---------------------------
# Генерация более интересного synthetic-case
# ---------------------------
def _add_bidirectional(edges, u: int, v: int) -> None:
    edges.append((u, v))
    edges.append((v, u))


def build_synthetic_city():
    n_nodes = 8
    edges = []

    # Кольцевая магистраль
    for u in range(n_nodes):
        _add_bidirectional(edges, u, (u + 1) % n_nodes)

    # Межрайонные связки и диагональные "быстрые" ребра
    for u, v in [(0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3), (4, 6), (5, 7)]:
        _add_bidirectional(edges, u, v)

    # Убираем дубли с сохранением порядка
    unique_edges = []
    seen = set()
    for e in edges:
        if e not in seen:
            seen.add(e)
            unique_edges.append(e)

    tail = np.array([u for u, _ in unique_edges], dtype=np.int32)
    head = np.array([v for _, v in unique_edges], dtype=np.int32)
    csr = CSRGraph.from_edges(n_nodes, tail, head)
    m = csr.m

    # Параметры ребер: кольцо медленнее, магистрали быстрее, мосты вместительнее
    cap = np.full(m, 25.0, dtype=np.float64)
    t0 = np.full(m, 4.0, dtype=np.float64)
    bridge_pairs = {(0, 4), (1, 5), (2, 6), (3, 7)}
    for e, (u, v) in enumerate(unique_edges):
        pair = tuple(sorted((u, v)))
        is_ring = abs(u - v) == 1 or abs(u - v) == n_nodes - 1
        if is_ring:
            t0[e] = 5.0
            cap[e] = 35.0
        else:
            t0[e] = 3.0
            cap[e] = 20.0
        if pair in bridge_pairs:
            t0[e] = 2.5
            cap[e] = 40.0

    edge_cost = BRP(cap, t0, np.full_like(t0, 0.15), np.full_like(t0, 4.0))
    return n_nodes, unique_edges, csr, edge_cost, t0


def build_synthetic_od(csr: CSRGraph, t0: np.ndarray):
    n = csr.n
    pop = np.array([140, 120, 90, 80, 130, 110, 100, 70], dtype=np.float64)
    attract = np.array([100, 120, 110, 90, 130, 115, 105, 95], dtype=np.float64)

    D_true = np.zeros((n, n), dtype=np.float64)
    for o in range(n):
        dist, _ = csr.dijkstra(t0, o)
        for d in range(n):
            if o == d:
                continue
            val = pop[o] * attract[d] * np.exp(-dist[d] / 9.0)
            # Усиливаем межрайонные поездки (между половинами узлов)
            if (o < n // 2) != (d < n // 2):
                val *= 1.25
            D_true[o, d] = val

    D_true *= 3200.0 / (D_true.sum() + 1e-12)
    np.fill_diagonal(D_true, 0.0)
    return D_true


def build_noisy_observations(flow_true: np.ndarray, rng: np.random.Generator):
    m = flow_true.size
    top_frac = 0.4
    total_frac = 0.6
    noise_std = 0.03

    n_top = max(1, int(top_frac * m))
    n_obs = max(n_top, int(total_frac * m))

    sorted_idx = np.argsort(flow_true)[::-1]
    observed = set(sorted_idx[:n_top].tolist())
    rest = [i for i in range(m) if i not in observed]
    if len(observed) < n_obs:
        extra = rng.choice(rest, size=n_obs - len(observed), replace=False)
        observed.update(int(i) for i in extra)
    observed = np.array(sorted(observed), dtype=int)

    mask = np.zeros(m, dtype=np.float64)
    mask[observed] = 1.0

    noise = rng.normal(0.0, noise_std, size=m)
    f_hat = np.zeros(m, dtype=np.float64)
    f_hat[observed] = np.maximum(flow_true[observed] * (1.0 + noise[observed]), 0.0)
    return f_hat, mask, observed


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n_nodes, edges, csr, edge_cost, t0 = build_synthetic_city()
    D_true = build_synthetic_od(csr, t0)

    # Генерируем "наблюдения" из истинной OD-матрицы
    flow_true, _ = beckmann.fw_beckmann(
        csr, edge_cost, D_true, max_iter=1500, rgap_target=5e-5, verbose=False
    )
    f_hat, experiment_mask, observed_edges = build_noisy_observations(flow_true, rng)

    ell = D_true.sum(axis=1)
    w = D_true.sum(axis=0)
    L_row, L_col = build_L_row_col(n_nodes)

    # Параметры внешнего prox-grad цикла
    K = 1000
    eta0 = 8e-5
    gamma = 5e-3
    alpha = 220.0
    rho = 1.0
    admm_max_iter = 1200
    admm_tol_pri = 1e-6
    admm_tol_dual = 1e-6

    d = init_outer_marginals(ell, w, diag_zero=True)
    prev_d = d.copy()

    hist_step = []
    hist_gnorm = []
    hist_row_err = []
    hist_col_err = []
    hist_flow_misfit = []
    hist_rel_fro = []
    hist_eta = []
    hist_admm_iters = []
    hist_grad_step = []
    hist_prox_correction = []
    hist_net_step_abs = []

    norm_true = np.linalg.norm(D_true) + 1e-12

    print(
        f"Graph: n_nodes={n_nodes}, directed_edges={csr.m}, "
        f"observed_edges={int(experiment_mask.sum())}, total_demand={D_true.sum():.1f}"
    )

    for k in range(K):
        d_before_update = d.copy()
        Dmat = d.reshape(n_nodes, n_nodes)

        flow, _, grad_F_over_D = beckmann.fw_beckmann_regularized_marginal(
            csr,
            edge_cost,
            Dmat,
            f_hat,
            experiment_mask,
            alpha=0.9,
            max_iter=2000,
            rgap_target=5e-5,
            verbose=False,
        )
        g = grad_F_over_D.reshape(n_nodes, n_nodes)
        np.fill_diagonal(g, 0.0)
        gvec = g.reshape(-1)
        gnorm = np.linalg.norm(g)

        eta = eta0
        accepted = False
        accepted_v = None
        info = None
        for _trial in range(20):
            v = d_before_update - eta * gvec
            d_try, info = prox_g_admm(
                v=v,
                eta=eta,
                gamma=gamma,
                alpha=alpha,
                ell=ell,
                w=w,
                L_row=L_row,
                L_col=L_col,
                rho=rho,
                max_iter=admm_max_iter,
                tol_pri=admm_tol_pri,
                tol_dual=admm_tol_dual,
                diag_zero=True,
            )
            if np.isfinite(d_try).all() and d_try.sum() > 0.4 * ell.sum():
                d = d_try
                accepted_v = v
                accepted = True
                break
            eta *= 0.5

        if not accepted:
            eta0 *= 0.5

        step = np.linalg.norm(d - prev_d) / (np.linalg.norm(prev_d) + 1e-12)
        prev_d = d.copy()
        grad_step = np.linalg.norm(eta * gvec)
        prox_correction = np.linalg.norm(d - accepted_v) if accepted_v is not None else np.nan
        net_step_abs = np.linalg.norm(d - d_before_update)

        Drec = d.reshape(n_nodes, n_nodes)
        row_err = np.linalg.norm(Drec.sum(axis=1) - ell) / (np.linalg.norm(ell) + 1e-12)
        col_err = np.linalg.norm(Drec.sum(axis=0) - w) / (np.linalg.norm(w) + 1e-12)
        flow_misfit = np.linalg.norm((flow - f_hat) * experiment_mask) / (
            np.linalg.norm(f_hat * experiment_mask) + 1e-12
        )
        rel_fro = np.linalg.norm(Drec - D_true) / norm_true

        hist_step.append(step)
        hist_gnorm.append(gnorm)
        hist_row_err.append(row_err)
        hist_col_err.append(col_err)
        hist_flow_misfit.append(flow_misfit)
        hist_rel_fro.append(rel_fro)
        hist_eta.append(eta)
        hist_admm_iters.append(info["admm_iters"] if info is not None else 0)
        hist_grad_step.append(grad_step)
        hist_prox_correction.append(prox_correction)
        hist_net_step_abs.append(net_step_abs)

        print(
            f"iter={k:03d} eta={eta:.1e} ||G||={gnorm:.3e} relStep={step:.3e} "
            f"rowErr={row_err:.2e} colErr={col_err:.2e} flowMisfit={flow_misfit:.3e} "
            f"relOD={rel_fro:.3e} admmIters={hist_admm_iters[-1]} "
            f"gradStep={grad_step:.3e} proxCorr={prox_correction:.3e} netStep={net_step_abs:.3e}"
        )

    D_rec = d.reshape(n_nodes, n_nodes)
    flow_rec, _ = beckmann.fw_beckmann(
        csr, edge_cost, D_rec, max_iter=1500, rgap_target=5e-5, verbose=False
    )

    # ---------------------------
    # Графики
    # ---------------------------
    it = np.arange(len(hist_step))

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
    plt.title("Сходимость по маргиналам")
    plt.legend()
    plt.grid(True)

    plt.figure()
    safe_semilogy(it, hist_flow_misfit, label="невязка по наблюдаемым потокам")
    safe_semilogy(it, hist_rel_fro, label="ошибка OD-матрицы")
    plt.xlabel("итерация")
    plt.ylabel("относительная ошибка")
    plt.title("Качество восстановления")
    plt.legend()
    plt.grid(True)

    plt.figure()
    safe_semilogy(it, hist_grad_step, label="||eta * grad||")
    safe_semilogy(it, hist_prox_correction, label="||d_new - v||")
    safe_semilogy(it, hist_net_step_abs, label="||d_new - d_old||")
    plt.xlabel("итерация")
    plt.ylabel("абсолютная величина шага")
    plt.title("Кто двигает итерацию: градиент или prox")
    plt.legend()
    plt.grid(True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    im0 = ax[0].imshow(D_true, cmap="viridis")
    ax[0].set_title("D_true")
    ax[0].set_xlabel("destination")
    ax[0].set_ylabel("origin")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(D_rec, cmap="viridis")
    ax[1].set_title("D_recovered")
    ax[1].set_xlabel("destination")
    ax[1].set_ylabel("origin")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    plt.tight_layout()

    plt.figure()
    obs = observed_edges
    x = np.arange(obs.size)
    plt.plot(x, flow_true[obs], "o-", label="истинный поток")
    plt.plot(x, f_hat[obs], "s--", label="наблюдение f_hat")
    plt.plot(x, flow_rec[obs], "d-", label="восстановленный поток")
    plt.xlabel("индекс наблюдаемого ребра")
    plt.ylabel("поток")
    plt.title("Сопоставление потоков на наблюдаемых рёбрах")
    plt.legend()
    plt.grid(True)

    np.set_printoptions(precision=4, suppress=True)
    print("\nFinal metrics:")
    print(f"  rowErr={hist_row_err[-1]:.4e} colErr={hist_col_err[-1]:.4e}")
    print(f"  flowMisfit={hist_flow_misfit[-1]:.4e} relOD={hist_rel_fro[-1]:.4e}")
    print(f"  observed_edges={obs.size}/{csr.m}")
    print(f"  observed_edge_indices={obs.tolist()}")
    print("\nD_true:")
    print(D_true)
    print("\nD_recovered:")
    print(D_rec)
    print("\nRow sums (true):", D_true.sum(axis=1))
    print("Row sums (recovered):", D_rec.sum(axis=1))
    print("\nCol sums (true):", D_true.sum(axis=0))
    print("Col sums (recovered):", D_rec.sum(axis=0))

    plt.show()
