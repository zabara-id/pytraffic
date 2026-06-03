"""Постановка 1 модифицированная: потоковая невязка + энтропия без референсной матрицы."""

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
INITIAL_NOISE = 0.6

# Внешний Adam по прямой невязке потоков. Маргиналии держим жестко через IPF.
OUTER_ITERS = 600
LEARNING_RATE = 4.0
LEARNING_RATE_DECAY = 0.01
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8

# Веса слагаемых целевой функции.
GAMMA = 1e-2
IPF_PROJECT_ITERS = 500
IPF_PROJECT_TOL = 1e-7

# Настройки внутренних расчетов Бекмана.
FW_REFERENCE_ITERS = 1000
FW_INNER_ITERS = 200
FW_RGAP = 5e-5

# Counterfactual-метрика: удаляем по одному ребру и сравниваем потоки на измененных графах.
ROBUST_METRIC_NUM_EDGES = 12
ROBUST_METRIC_EVERY = 5
ROBUST_METRIC_FW_ITERS = 200

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


def is_strongly_connected(csr: CSRGraph) -> bool:
    """Checks that every OD pair remains reachable in a directed graph."""
    if csr.n == 0:
        return True

    def visit_from(start: int, tail: np.ndarray, head: np.ndarray) -> np.ndarray:
        seen = np.zeros(csr.n, dtype=bool)
        stack = [start]
        seen[start] = True
        adjacency = [[] for _ in range(csr.n)]
        for u, v in zip(tail, head):
            adjacency[int(u)].append(int(v))

        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        return seen

    forward_seen = visit_from(0, csr.tail, csr.head)
    backward_seen = visit_from(0, csr.head, csr.tail)
    return bool(forward_seen.all() and backward_seen.all())


def edge_cost_without_edge(edge_cost: BRP, keep_mask: np.ndarray) -> BRP:
    return BRP(
        np.asarray(edge_cost.cap)[keep_mask],
        np.asarray(edge_cost.t0)[keep_mask],
        np.asarray(edge_cost.alpha)[keep_mask],
        np.asarray(edge_cost.beta)[keep_mask],
    )


def build_deleted_edge_scenarios(
    csr: CSRGraph,
    edge_cost: BRP,
    rng: np.random.Generator,
    *,
    num_edges: int,
) -> list[tuple[int, CSRGraph, BRP]]:
    candidates = []

    for edge_id in range(csr.m):
        keep_mask = np.ones(csr.m, dtype=bool)
        keep_mask[edge_id] = False
        candidate_csr = CSRGraph.from_edges(csr.n, csr.tail[keep_mask], csr.head[keep_mask])
        if is_strongly_connected(candidate_csr):
            candidates.append((edge_id, candidate_csr, edge_cost_without_edge(edge_cost, keep_mask)))

    if not candidates:
        return []

    if len(candidates) <= num_edges:
        return candidates

    chosen = rng.choice(len(candidates), size=num_edges, replace=False)
    return [candidates[int(i)] for i in np.sort(chosen)]


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


def entropy_value(D: np.ndarray, gamma: float) -> float:
    """Convex negative-entropy regularizer gamma * sum_ij D_ij(log D_ij - 1)."""
    positive = np.maximum(D[D > 0.0], EPS)
    return float(gamma * np.sum(positive * (np.log(positive) - 1.0)))


def entropy_gradient(D: np.ndarray, gamma: float) -> np.ndarray:
    """Gradient of gamma * sum_ij D_ij(log D_ij - 1)."""
    return gamma * np.log(np.maximum(D, EPS))


def project_nonnegative_zero_diag(D: np.ndarray, *optimizer_states: np.ndarray) -> None:
    D[D < 0.0] = EPS
    np.fill_diagonal(D, 0.0)
    for state in optimizer_states:
        np.fill_diagonal(state, 0.0)


def relative_l1_error(D: np.ndarray, D_true: np.ndarray) -> float:
    return float(np.sum(np.abs(D - D_true)) / (np.sum(np.abs(D_true)) + EPS))


def flow_mismatch_value(flow: np.ndarray, reference_flow: np.ndarray) -> float:
    residual = flow - reference_flow
    return float(0.5 * np.dot(residual, residual))


def relative_flow_error(flow: np.ndarray, reference_flow: np.ndarray) -> float:
    return float(np.linalg.norm(flow - reference_flow) / (np.linalg.norm(reference_flow) + EPS))


def deleted_edge_metric(
    scenarios: list[tuple[int, CSRGraph, BRP]],
    D: np.ndarray,
    reference_flows: list[np.ndarray],
) -> float:
    if not scenarios:
        return np.nan

    errors = []
    for (_, scenario_csr, scenario_edge_cost), reference_flow in zip(scenarios, reference_flows):
        flow, _ = beckmann.fw_beckmann(
            scenario_csr,
            scenario_edge_cost,
            D,
            max_iter=ROBUST_METRIC_FW_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )
        errors.append(relative_flow_error(flow, reference_flow))

    return float(np.mean(errors))


def relative_marginal_errors(D: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> tuple[float, float]:
    row_error = np.linalg.norm(D.sum(axis=1) - row_target, ord=1) / (np.linalg.norm(row_target, ord=1) + EPS)
    col_error = np.linalg.norm(D.sum(axis=0) - col_target, ord=1) / (np.linalg.norm(col_target, ord=1) + EPS)
    return float(row_error), float(col_error)


def print_identifiability_warning(csr: CSRGraph, D: np.ndarray) -> None:
    n = D.shape[0]
    od_variables = n * (n - 1)
    independent_marginals = 2 * n - 1
    flow_equations = csr.m
    underdetermined = max(od_variables - independent_marginals - flow_equations, 0)

    print(
        "Identifiability check: "
        f"OD variables={od_variables}, edge counts={flow_equations}, "
        f"independent marginals={independent_marginals}, "
        f"underdetermined directions >= {underdetermined}"
    )


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
    robust_scenarios = build_deleted_edge_scenarios(
        csr,
        edge_cost,
        rng,
        num_edges=ROBUST_METRIC_NUM_EDGES,
    )
    robust_reference_flows = [
        beckmann.fw_beckmann(
            scenario_csr,
            scenario_edge_cost,
            D_true,
            max_iter=ROBUST_METRIC_FW_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )[0]
        for _, scenario_csr, scenario_edge_cost in robust_scenarios
    ]

    flow, flow_jacobian = beckmann.fw_beckmann(
        csr,
        edge_cost,
        D,
        max_iter=FW_INNER_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )

    first_moment = np.zeros_like(D)
    second_moment = np.zeros_like(D)
    trajectory = [D.copy()]
    rel_l1_history = [relative_l1_error(D, D_true)]
    flow_error_history = [relative_flow_error(flow, reference_flow)]
    deleted_edge_error_history = [deleted_edge_metric(robust_scenarios, D, robust_reference_flows)]
    deleted_edge_iterations = [0]
    objective_history = [flow_mismatch_value(flow, reference_flow) + entropy_value(D, GAMMA)]

    best_objective = objective_history[-1]
    best_rel_l1 = rel_l1_history[-1]
    best_rel_l1_iteration = 0
    best_iteration = 0
    best_D = D.copy()

    row_error, col_error = relative_marginal_errors(D, row_target, col_target)
    row_error_history: list[float] = [row_error]
    col_error_history: list[float] = [col_error]

    print(
        f"Sioux Falls: nodes={csr.n}, edges={csr.m}, total_demand={D_true.sum():.1f}, "
        f"initial_rel_l1={rel_l1_history[-1]:.4e}, initial_flow_err={flow_error_history[-1]:.4e}"
    )
    print(
        f"Deleted-edge validation scenarios: {len(robust_scenarios)} "
        f"edges={[edge_id for edge_id, _, _ in robust_scenarios]}"
    )
    print_identifiability_warning(csr, D_true)

    for iteration in range(1, OUTER_ITERS + 1):
        flow_residual = flow - reference_flow
        flow_grad = (flow_jacobian.T @ flow_residual).reshape(D.shape)
        ent_grad = entropy_gradient(D, GAMMA)
        grad = flow_grad + ent_grad
        np.fill_diagonal(grad, 0.0)

        first_moment = BETA1 * first_moment + (1.0 - BETA1) * grad
        second_moment = BETA2 * second_moment + (1.0 - BETA2) * grad * grad
        first_unbiased = first_moment / (1.0 - BETA1**iteration)
        second_unbiased = second_moment / (1.0 - BETA2**iteration)

        learning_rate = LEARNING_RATE / np.sqrt(1.0 + LEARNING_RATE_DECAY * (iteration - 1))
        D -= learning_rate * first_unbiased / (np.sqrt(second_unbiased) + ADAM_EPS)
        project_nonnegative_zero_diag(D, first_moment, second_moment)
        D = ipf_project_to_marginals(
            D,
            row_target,
            col_target,
            max_iter=IPF_PROJECT_ITERS,
            tol=IPF_PROJECT_TOL,
        )

        flow, flow_jacobian = beckmann.fw_beckmann(
            csr,
            edge_cost,
            D,
            max_iter=FW_INNER_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )

        rel_l1 = relative_l1_error(D, D_true)
        flow_error = relative_flow_error(flow, reference_flow)
        objective = flow_mismatch_value(flow, reference_flow) + entropy_value(D, GAMMA)
        row_error, col_error = relative_marginal_errors(D, row_target, col_target)

        trajectory.append(D.copy())
        rel_l1_history.append(rel_l1)
        flow_error_history.append(flow_error)
        objective_history.append(objective)
        row_error_history.append(float(row_error))
        col_error_history.append(float(col_error))

        if iteration % ROBUST_METRIC_EVERY == 0 or iteration == OUTER_ITERS:
            deleted_edge_iterations.append(iteration)
            deleted_edge_error_history.append(deleted_edge_metric(robust_scenarios, D, robust_reference_flows))

        if objective < best_objective:
            best_objective = objective
            best_iteration = iteration
            best_D = D.copy()

        if rel_l1 < best_rel_l1:
            best_rel_l1 = rel_l1
            best_rel_l1_iteration = iteration

        if iteration == 1 or iteration % PRINT_EVERY == 0 or iteration == OUTER_ITERS:
            print(
                f"iter={iteration:03d} lr={learning_rate:.2e} "
                f"flow_err={flow_error:.4e} best_obj={best_objective:.4e}@{best_iteration:03d} "
                f"rel_l1={rel_l1:.4e} best_l1={best_rel_l1:.4e}@{best_rel_l1_iteration:03d} "
                f"row_err={row_error:.2e} col_err={col_error:.2e}"
            )

    return {
        "D_true": D_true,
        "D_initial": D_initial,
        "D_recovered": best_D,
        "D_final": D,
        "best_iteration": best_iteration,
        "best_objective": best_objective,
        "best_rel_l1": best_rel_l1,
        "best_rel_l1_iteration": best_rel_l1_iteration,
        "reference_flow": reference_flow,
        "final_flow": flow,
        "trajectory": np.stack(trajectory),
        "rel_l1_history": rel_l1_history,
        "flow_error_history": flow_error_history,
        "deleted_edge_iterations": deleted_edge_iterations,
        "deleted_edge_error_history": deleted_edge_error_history,
        "objective_history": objective_history,
        "row_error_history": row_error_history,
        "col_error_history": col_error_history,
    }


def plot_results(result: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    rel_l1_history = np.asarray(result["rel_l1_history"], dtype=np.float64)
    flow_error_history = np.asarray(result["flow_error_history"], dtype=np.float64)
    deleted_edge_iterations = np.asarray(result["deleted_edge_iterations"], dtype=np.int64)
    deleted_edge_error_history = np.asarray(result["deleted_edge_error_history"], dtype=np.float64)
    iterations = np.arange(rel_l1_history.size)
    best_iteration = int(result["best_iteration"])
    best_rel_l1_iteration = int(result["best_rel_l1_iteration"])

    plt.figure(figsize=(7, 4))
    plt.semilogy(iterations, np.maximum(flow_error_history, 1e-300), linewidth=2.0, label="flow error")
    plt.semilogy(
        deleted_edge_iterations,
        np.maximum(deleted_edge_error_history, 1e-300),
        "o-",
        linewidth=2.0,
        label="deleted-edge flow error",
    )
    plt.scatter(
        [best_iteration],
        [flow_error_history[best_iteration]],
        color="tab:red",
        zorder=3,
        label=f"best objective iter={best_iteration}",
    )
    plt.xlabel("iteration")
    plt.ylabel("relative error")
    plt.title("Sioux Falls OD recovery")
    plt.legend()
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.semilogy(iterations, np.maximum(rel_l1_history, 1e-300), linewidth=2.0, label="OD L1 error")
    plt.scatter(
        [best_rel_l1_iteration],
        [rel_l1_history[best_rel_l1_iteration]],
        color="tab:red",
        zorder=3,
        label=f"best L1 iter={best_rel_l1_iteration}",
    )
    plt.xlabel("iteration")
    plt.ylabel(r"$\|D_k-D_{true}\|_1/\|D_{true}\|_1$")
    plt.title("OD matrix recovery error")
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
    print("Final relative flow error:", f"{experiment_result['flow_error_history'][-1]:.6e}")
    print("Final deleted-edge flow error:", f"{experiment_result['deleted_edge_error_history'][-1]:.6e}")
    print(
        "Best relative L1 error:",
        f"{experiment_result['best_rel_l1']:.6e}",
        f"at iteration {experiment_result['best_rel_l1_iteration']}",
    )
    best_objective_iteration = int(experiment_result["best_iteration"])
    print(
        "Best objective iteration:",
        best_objective_iteration,
        "flow error:",
        f"{experiment_result['flow_error_history'][best_objective_iteration]:.6e}",
    )
    if SHOW_PLOTS:
        plot_results(experiment_result)
