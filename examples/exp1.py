import csv
import io
import re
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import pytraffic.models.Beckmann as beckmann
from pytraffic.graph.CSRGraph import CSRGraph
from pytraffic.models.BRPCost import BRP


EPS = 1e-12
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "SiouxFalls"
SIOUX_FALLS_NET_URL = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_net.tntp"
SIOUX_FALLS_OD_URL = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/CSV-data/SiouxFalls_od.csv"

# Данные. Можно поставить URL-ы выше вместо локальных файлов, если хочется читать напрямую с GitHub.
NET_FILE = DEFAULT_DATA_DIR / "SiouxFalls_net.tntp"
DEMAND_FILE = DEFAULT_DATA_DIR / "SiouxFalls_od.csv"

# Воспроизводимость и начальное приближение.
SEED = 42
INITIAL_NOISE = 1.25

# Внешний Adam по полной матрице D.
OUTER_ITERS = 500
LEARNING_RATE = 0.2
LEARNING_RATE_DECAY = 0.03
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8

# Веса трех слагаемых целевой функции.
ALPHA = 1e-4
GAMMA = 1e-2
LAMBDA_MARGINAL = 1e-2

# Настройки внутренних расчетов Бекмана.
FW_REFERENCE_ITERS = 1000
FW_INNER_ITERS = 200
FW_RGAP = 5e-5

# Вывод.
PRINT_EVERY = 5
SHOW_PLOTS = True


def read_text_source(source: str | Path) -> str:
    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        with urlopen(source_str) as response:
            return response.read().decode("utf-8")

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text()


def _metadata_int(line: str, key: str) -> int | None:
    match = re.search(rf"<{re.escape(key)}>\s+(\d+)", line, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_sioux_falls_network(source: str | Path) -> tuple[CSRGraph, BRP]:
    """Reads a TNTP network file and builds the graph and BPR edge costs."""
    n_nodes: int | None = None
    tail: list[int] = []
    head: list[int] = []
    capacity: list[float] = []
    free_flow_time: list[float] = []
    b_param: list[float] = []
    power: list[float] = []

    for raw_line in read_text_source(source).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        n_nodes = _metadata_int(line, "NUMBER OF NODES") or n_nodes
        if line.startswith("<") or line.startswith("~"):
            continue

        fields = line.replace(";", " ").split()
        if len(fields) < 7:
            continue

        tail.append(int(fields[0]) - 1)
        head.append(int(fields[1]) - 1)
        capacity.append(float(fields[2]))
        free_flow_time.append(float(fields[4]))
        b_param.append(float(fields[5]))
        power.append(float(fields[6]))

    if n_nodes is None:
        n_nodes = max(max(tail), max(head)) + 1
    if not tail:
        raise ValueError(f"No links were parsed from {source}")

    csr = CSRGraph.from_edges(
        n_nodes,
        np.asarray(tail, dtype=np.int32),
        np.asarray(head, dtype=np.int32),
    )
    edge_cost = BRP(
        np.asarray(capacity, dtype=np.float64),
        np.asarray(free_flow_time, dtype=np.float64),
        np.asarray(b_param, dtype=np.float64),
        np.asarray(power, dtype=np.float64),
    )
    return csr, edge_cost


def load_sioux_falls_od_csv(source: str | Path) -> np.ndarray:
    """Reads CSV-data/SiouxFalls_od.csv with columns O,D,Ton."""
    rows: list[tuple[int, int, float]] = []
    reader = csv.DictReader(io.StringIO(read_text_source(source)))

    for row in reader:
        origin = int(row["O"]) - 1
        destination = int(row["D"]) - 1
        demand = float(row["Ton"])
        rows.append((origin, destination, demand))

    if not rows:
        raise ValueError(f"No OD pairs were parsed from {source}")

    n_zones = 1 + max(max(origin, destination) for origin, destination, _ in rows)
    D = np.zeros((n_zones, n_zones), dtype=np.float64)
    for origin, destination, demand in rows:
        D[origin, destination] = demand
    np.fill_diagonal(D, 0.0)
    return D


def load_sioux_falls(net_source: str | Path, demand_source: str | Path) -> tuple[CSRGraph, BRP, np.ndarray]:
    csr, edge_cost = load_sioux_falls_network(net_source)
    D_true = load_sioux_falls_od_csv(demand_source)
    if D_true.shape != (csr.n, csr.n):
        raise ValueError(f"OD matrix shape {D_true.shape} does not match graph nodes {csr.n}")
    return csr, edge_cost, D_true


def ipf_project_to_marginals(
    D: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> np.ndarray:
    """IPF projection onto known row and column marginals, with zero diagonal fixed."""
    projected = np.maximum(np.asarray(D, dtype=np.float64), EPS)
    np.fill_diagonal(projected, 0.0)

    for _ in range(max_iter):
        projected *= (row_target / np.maximum(projected.sum(axis=1), EPS))[:, None]
        projected *= (col_target / np.maximum(projected.sum(axis=0), EPS))[None, :]
        np.fill_diagonal(projected, 0.0)

        row_error = np.max(np.abs(projected.sum(axis=1) - row_target))
        col_error = np.max(np.abs(projected.sum(axis=0) - col_target))
        if max(row_error, col_error) <= tol:
            break

    return projected


def make_noisy_initial_od(
    D_true: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    rng: np.random.Generator,
    *,
    relative_noise: float,
) -> np.ndarray:
    """Starts from D_true + additive noise and projects the result to the true marginals."""
    positive_mean = float(D_true[D_true > 0.0].mean())
    noisy = D_true + rng.normal(0.0, relative_noise * positive_mean, size=D_true.shape)
    noisy = np.maximum(noisy, EPS)
    np.fill_diagonal(noisy, 0.0)
    return ipf_project_to_marginals(noisy, row_target, col_target)


def entropy_gradient(D: np.ndarray, gamma: float) -> np.ndarray:
    """Gradient of gamma * sum_ij D_ij log(D_ij)."""
    return gamma * (np.log(np.maximum(D, EPS)) + 1.0)


def marginal_gradient(D: np.ndarray, row_target: np.ndarray, col_target: np.ndarray, penalty: float) -> np.ndarray:
    """Gradient of penalty * (||D 1 - row||_2^2 + ||D^T 1 - col||_2^2)."""
    row_residual = D.sum(axis=1) - row_target
    col_residual = D.sum(axis=0) - col_target
    return 2.0 * penalty * (row_residual[:, None] + col_residual[None, :])


def project_nonnegative_zero_diag(D: np.ndarray, *adam_states: np.ndarray) -> None:
    D[D < 0.0] = EPS
    np.fill_diagonal(D, 0.0)
    for state in adam_states:
        np.fill_diagonal(state, 0.0)


def relative_l1_error(D: np.ndarray, D_true: np.ndarray) -> float:
    return float(np.sum(np.abs(D - D_true)) / (np.sum(np.abs(D_true)) + EPS))


def relative_marginal_errors(D: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> tuple[float, float]:
    row_error = np.linalg.norm(D.sum(axis=1) - row_target, ord=1) / (np.linalg.norm(row_target, ord=1) + EPS)
    col_error = np.linalg.norm(D.sum(axis=0) - col_target, ord=1) / (np.linalg.norm(col_target, ord=1) + EPS)
    return float(row_error), float(col_error)


def run_experiment() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    csr, edge_cost, D_true = load_sioux_falls(NET_FILE, DEMAND_FILE)

    row_target = D_true.sum(axis=1)
    col_target = D_true.sum(axis=0)
    D = make_noisy_initial_od(D_true, row_target, col_target, rng, relative_noise=INITIAL_NOISE)
    D_initial = D.copy()

    reference_flow, _ = beckmann.fw_beckmann(
        csr,
        edge_cost,
        D_true,
        max_iter=FW_REFERENCE_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )
    edge_mask = np.ones(csr.m, dtype=np.float64)

    first_moment = np.zeros_like(D)
    second_moment = np.zeros_like(D)
    trajectory = [D.copy()]
    rel_l1_history = [relative_l1_error(D, D_true)]
    best_rel_l1 = rel_l1_history[-1]
    best_iteration = 0
    best_D = D.copy()

    row_error_history: list[float] = []
    col_error_history: list[float] = []

    print(
        f"Sioux Falls: nodes={csr.n}, edges={csr.m}, total_demand={D_true.sum():.1f}, "
        f"initial_rel_l1={rel_l1_history[-1]:.4e}"
    )

    for iteration in range(1, OUTER_ITERS + 1):
        _, _, beckmann_grad_flat = beckmann.fw_beckmann_regularized(
            csr,
            edge_cost,
            D,
            reference_flow,
            edge_mask,
            alpha=ALPHA,
            max_iter=FW_INNER_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )

        beckmann_grad = beckmann_grad_flat.reshape(D.shape)
        ent_grad = entropy_gradient(D, GAMMA)
        marg_grad = marginal_gradient(D, row_target, col_target, LAMBDA_MARGINAL)
        grad = beckmann_grad + ent_grad + marg_grad
        np.fill_diagonal(grad, 0.0)

        first_moment = BETA1 * first_moment + (1.0 - BETA1) * grad
        second_moment = BETA2 * second_moment + (1.0 - BETA2) * grad * grad
        first_unbiased = first_moment / (1.0 - BETA1**iteration)
        second_unbiased = second_moment / (1.0 - BETA2**iteration)

        learning_rate = LEARNING_RATE / np.sqrt(1.0 + LEARNING_RATE_DECAY * (iteration - 1))
        step = learning_rate * first_unbiased / (np.sqrt(second_unbiased) + ADAM_EPS)
        D -= step
        project_nonnegative_zero_diag(D, first_moment, second_moment)

        rel_l1 = relative_l1_error(D, D_true)
        row_error, col_error = relative_marginal_errors(D, row_target, col_target)

        trajectory.append(D.copy())
        rel_l1_history.append(rel_l1)
        row_error_history.append(float(row_error))
        col_error_history.append(float(col_error))

        if rel_l1 < best_rel_l1:
            best_rel_l1 = rel_l1
            best_iteration = iteration
            best_D = D.copy()

        if iteration == 1 or iteration % PRINT_EVERY == 0 or iteration == OUTER_ITERS:
            print(
                f"iter={iteration:03d} lr={learning_rate:.2e} "
                f"rel_l1={rel_l1:.4e} best={best_rel_l1:.4e}@{best_iteration:03d} "
                f"row_err={row_error:.2e} col_err={col_error:.2e}"
            )

    return {
        "D_true": D_true,
        "D_initial": D_initial,
        "D_recovered": best_D,
        "D_final": D,
        "best_iteration": best_iteration,
        "best_rel_l1": best_rel_l1,
        "reference_flow": reference_flow,
        "trajectory": np.stack(trajectory),
        "rel_l1_history": rel_l1_history,
        "row_error_history": row_error_history,
        "col_error_history": col_error_history,
    }


def plot_results(result: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    rel_l1_history = np.asarray(result["rel_l1_history"], dtype=np.float64)
    iterations = np.arange(rel_l1_history.size)
    best_iteration = int(result["best_iteration"])

    plt.figure(figsize=(7, 4))
    plt.semilogy(iterations, np.maximum(rel_l1_history, 1e-300), linewidth=2.0)
    plt.scatter(
        [best_iteration],
        [rel_l1_history[best_iteration]],
        color="tab:red",
        zorder=3,
        label=f"best iter={best_iteration}",
    )
    plt.xlabel("iteration")
    plt.ylabel(r"$\|D_k - D_{true}\|_1 / \|D_{true}\|_1$")
    plt.title("Sioux Falls OD recovery")
    plt.legend()
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    D_initial = np.asarray(result["D_initial"])
    D_true = np.asarray(result["D_true"])
    D_recovered = np.asarray(result["D_recovered"])

    _, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    vmin = min(float(D_initial.min()), float(D_recovered.min()), float(D_true.min()))
    vmax = max(float(D_initial.max()), float(D_recovered.max()), float(D_true.max()))

    axes[0].imshow(D_initial, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("initial")
    axes[0].set_xlabel("destination")
    axes[0].set_ylabel("origin")

    axes[1].imshow(D_recovered, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"recovered, iter={best_iteration}")
    axes[1].set_xlabel("destination")
    axes[1].set_ylabel("origin")

    axes[2].imshow(D_true, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title("true")
    axes[2].set_xlabel("destination")
    axes[2].set_ylabel("origin")

    plt.show()


if __name__ == "__main__":
    experiment_result = run_experiment()
    print("\nFinal relative L1 error:", f"{experiment_result['rel_l1_history'][-1]:.6e}")
    print(
        "Best relative L1 error:",
        f"{experiment_result['best_rel_l1']:.6e}",
        f"at iteration {experiment_result['best_iteration']}",
    )
    if SHOW_PLOTS:
        plot_results(experiment_result)
