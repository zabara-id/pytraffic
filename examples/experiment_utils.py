"""Общая инфраструктура экспериментов по восстановлению OD-матрицы."""

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen

import numpy as np

import pytraffic.models.Beckmann as beckmann
from pytraffic.graph.CSRGraph import CSRGraph
from pytraffic.models.BRPCost import BRP


EPS = 1e-12
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "SiouxFalls"
DEFAULT_NET_FILE = DEFAULT_DATA_DIR / "SiouxFalls_net.tntp"
DEFAULT_DEMAND_FILE = DEFAULT_DATA_DIR / "SiouxFalls_od.csv"


@dataclass
class ExperimentData:
    csr: CSRGraph
    edge_cost: BRP
    D_true: np.ndarray
    D_initial: np.ndarray
    row_target: np.ndarray
    col_target: np.ndarray
    reference_flow: np.ndarray
    robust_scenarios: list[tuple[int, CSRGraph, BRP]]
    robust_reference_flows: list[np.ndarray]


@dataclass
class ExperimentResult:
    method_name: str
    D_true: np.ndarray
    D_initial: np.ndarray
    D_recovered: np.ndarray
    D_final: np.ndarray
    final_flow: np.ndarray
    best_iteration: int
    best_objective: float
    best_rel_l1: float
    best_rel_l1_iteration: int
    trajectory: list[np.ndarray] | np.ndarray = field(default_factory=list)
    rel_l1_history: list[float] = field(default_factory=list)
    flow_error_history: list[float] = field(default_factory=list)
    deleted_edge_iterations: list[int] = field(default_factory=list)
    deleted_edge_error_history: list[float] = field(default_factory=list)
    objective_history: list[float] = field(default_factory=list)
    row_error_history: list[float] = field(default_factory=list)
    col_error_history: list[float] = field(default_factory=list)


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

    if not tail:
        raise ValueError(f"No links were parsed from {source}")
    if n_nodes is None:
        n_nodes = max(max(tail), max(head)) + 1

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


def _is_strongly_connected(csr: CSRGraph) -> bool:
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

    return bool(visit_from(0, csr.tail, csr.head).all() and visit_from(0, csr.head, csr.tail).all())


def _edge_cost_without_edge(edge_cost: BRP, keep_mask: np.ndarray) -> BRP:
    return BRP(
        np.asarray(edge_cost.cap)[keep_mask],
        np.asarray(edge_cost.t0)[keep_mask],
        np.asarray(edge_cost.alpha)[keep_mask],
        np.asarray(edge_cost.beta)[keep_mask],
    )


def _build_deleted_edge_scenarios(
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
        if _is_strongly_connected(candidate_csr):
            candidates.append((edge_id, candidate_csr, _edge_cost_without_edge(edge_cost, keep_mask)))

    if len(candidates) <= num_edges:
        return candidates

    chosen = rng.choice(len(candidates), size=num_edges, replace=False)
    return [candidates[int(i)] for i in np.sort(chosen)]


def _ipf_project_to_marginals(
    D: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Проецирует OD-матрицу на заданные маргиналии, сохраняя нулевую диагональ."""
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


def _make_noisy_initial_od(
    D_true: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    rng: np.random.Generator,
    *,
    relative_noise: float,
) -> np.ndarray:
    positive_mean = float(D_true[D_true > 0.0].mean())
    noisy = D_true + rng.normal(0.0, relative_noise * positive_mean, size=D_true.shape)
    noisy = np.maximum(noisy, EPS)
    np.fill_diagonal(noisy, 0.0)
    return _ipf_project_to_marginals(noisy, row_target, col_target)


def prepare_experiment(
    net_file: str | Path,
    demand_file: str | Path,
    *,
    seed: int,
    initial_noise: float,
    fw_reference_iters: int,
    fw_rgap: float,
    robust_metric_num_edges: int,
    robust_metric_fw_iters: int,
) -> ExperimentData:
    rng = np.random.default_rng(seed)
    csr, edge_cost, D_true = load_sioux_falls(net_file, demand_file)
    row_target = D_true.sum(axis=1)
    col_target = D_true.sum(axis=0)
    D_initial = _make_noisy_initial_od(
        D_true,
        row_target,
        col_target,
        rng,
        relative_noise=initial_noise,
    )
    reference_flow, _ = beckmann.fw_beckmann(
        csr,
        edge_cost,
        D_true,
        max_iter=fw_reference_iters,
        rgap_target=fw_rgap,
        verbose=False,
    )
    robust_scenarios = _build_deleted_edge_scenarios(
        csr,
        edge_cost,
        rng,
        num_edges=robust_metric_num_edges,
    )
    robust_reference_flows = [
        beckmann.fw_beckmann(
            scenario_csr,
            scenario_edge_cost,
            D_true,
            max_iter=robust_metric_fw_iters,
            rgap_target=fw_rgap,
            verbose=False,
        )[0]
        for _, scenario_csr, scenario_edge_cost in robust_scenarios
    ]
    return ExperimentData(
        csr=csr,
        edge_cost=edge_cost,
        D_true=D_true,
        D_initial=D_initial,
        row_target=row_target,
        col_target=col_target,
        reference_flow=reference_flow,
        robust_scenarios=robust_scenarios,
        robust_reference_flows=robust_reference_flows,
    )


def relative_l1_error(D: np.ndarray, D_true: np.ndarray) -> float:
    return float(np.sum(np.abs(D - D_true)) / (np.sum(np.abs(D_true)) + EPS))


def relative_flow_error(flow: np.ndarray, reference_flow: np.ndarray) -> float:
    return float(np.linalg.norm(flow - reference_flow) / (np.linalg.norm(reference_flow) + EPS))


def relative_marginal_errors(D: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> tuple[float, float]:
    row_error = np.linalg.norm(D.sum(axis=1) - row_target, ord=1) / (np.linalg.norm(row_target, ord=1) + EPS)
    col_error = np.linalg.norm(D.sum(axis=0) - col_target, ord=1) / (np.linalg.norm(col_target, ord=1) + EPS)
    return float(row_error), float(col_error)


def _deleted_edge_metric(
    data: ExperimentData,
    D: np.ndarray,
    *,
    fw_iters: int,
    fw_rgap: float,
) -> float:
    if not data.robust_scenarios:
        return np.nan

    errors = []
    for (_, scenario_csr, scenario_edge_cost), reference_flow in zip(
        data.robust_scenarios,
        data.robust_reference_flows,
    ):
        flow, _ = beckmann.fw_beckmann(
            scenario_csr,
            scenario_edge_cost,
            D,
            max_iter=fw_iters,
            rgap_target=fw_rgap,
            verbose=False,
        )
        errors.append(relative_flow_error(flow, reference_flow))
    return float(np.mean(errors))


def create_result(
    method_name: str,
    data: ExperimentData,
    D: np.ndarray,
    flow: np.ndarray,
    objective: float,
    *,
    robust_metric_fw_iters: int,
    fw_rgap: float,
) -> ExperimentResult:
    rel_l1 = relative_l1_error(D, data.D_true)
    row_error, col_error = relative_marginal_errors(D, data.row_target, data.col_target)
    return ExperimentResult(
        method_name=method_name,
        D_true=data.D_true,
        D_initial=data.D_initial,
        D_recovered=D.copy(),
        D_final=D.copy(),
        final_flow=flow.copy(),
        best_iteration=0,
        best_objective=float(objective),
        best_rel_l1=rel_l1,
        best_rel_l1_iteration=0,
        trajectory=[D.copy()],
        rel_l1_history=[rel_l1],
        flow_error_history=[relative_flow_error(flow, data.reference_flow)],
        deleted_edge_iterations=[0],
        deleted_edge_error_history=[
            _deleted_edge_metric(data, D, fw_iters=robust_metric_fw_iters, fw_rgap=fw_rgap)
        ],
        objective_history=[float(objective)],
        row_error_history=[row_error],
        col_error_history=[col_error],
    )


def record_iteration(
    result: ExperimentResult,
    data: ExperimentData,
    D: np.ndarray,
    flow: np.ndarray,
    objective: float,
    iteration: int,
    *,
    outer_iters: int,
    robust_metric_every: int,
    robust_metric_fw_iters: int,
    fw_rgap: float,
) -> tuple[float, float, float, float]:
    rel_l1 = relative_l1_error(D, data.D_true)
    flow_error = relative_flow_error(flow, data.reference_flow)
    row_error, col_error = relative_marginal_errors(D, data.row_target, data.col_target)

    result.trajectory.append(D.copy())
    result.rel_l1_history.append(rel_l1)
    result.flow_error_history.append(flow_error)
    result.objective_history.append(float(objective))
    result.row_error_history.append(row_error)
    result.col_error_history.append(col_error)

    if iteration % robust_metric_every == 0 or iteration == outer_iters:
        result.deleted_edge_iterations.append(iteration)
        result.deleted_edge_error_history.append(
            _deleted_edge_metric(data, D, fw_iters=robust_metric_fw_iters, fw_rgap=fw_rgap)
        )

    result.D_final = D.copy()
    result.final_flow = flow.copy()
    if objective < result.best_objective:
        result.best_objective = float(objective)
        result.best_iteration = iteration
        result.D_recovered = D.copy()
    if rel_l1 < result.best_rel_l1:
        result.best_rel_l1 = rel_l1
        result.best_rel_l1_iteration = iteration

    return rel_l1, flow_error, row_error, col_error


def finish_result(result: ExperimentResult) -> ExperimentResult:
    result.trajectory = np.stack(result.trajectory)
    return result


def print_experiment_header(data: ExperimentData, initial_flow: np.ndarray) -> None:
    print(
        f"Sioux Falls: nodes={data.csr.n}, edges={data.csr.m}, total_demand={data.D_true.sum():.1f}, "
        f"initial_rel_l1={relative_l1_error(data.D_initial, data.D_true):.4e}, "
        f"initial_flow_err={relative_flow_error(initial_flow, data.reference_flow):.4e}"
    )
    print(
        f"Deleted-edge validation scenarios: {len(data.robust_scenarios)} "
        f"edges={[edge_id for edge_id, _, _ in data.robust_scenarios]}"
    )

    n = data.D_true.shape[0]
    od_variables = n * (n - 1)
    independent_marginals = 2 * n - 1
    underdetermined = max(od_variables - independent_marginals - data.csr.m, 0)
    print(
        "Identifiability check: "
        f"OD variables={od_variables}, edge counts={data.csr.m}, "
        f"independent marginals={independent_marginals}, "
        f"underdetermined directions >= {underdetermined}"
    )


def print_result_summary(result: ExperimentResult) -> None:
    print("\nSummary")
    print("Final relative L1 error:", f"{result.rel_l1_history[-1]:.6e}")
    print("Final relative flow error:", f"{result.flow_error_history[-1]:.6e}")
    print("Final deleted-edge flow error:", f"{result.deleted_edge_error_history[-1]:.6e}")
    print(
        "Final marginal errors:",
        f"row={result.row_error_history[-1]:.3e}",
        f"col={result.col_error_history[-1]:.3e}",
    )
    print("Best relative L1 error:", f"{result.best_rel_l1:.6e}", f"at iteration {result.best_rel_l1_iteration}")
    print("Best objective:", f"{result.best_objective:.6e}", f"at iteration {result.best_iteration}")


def plot_result(result: ExperimentResult, *, title: str) -> None:
    import matplotlib.pyplot as plt

    rel_l1_history = np.asarray(result.rel_l1_history, dtype=np.float64)
    flow_error_history = np.asarray(result.flow_error_history, dtype=np.float64)
    deleted_edge_iterations = np.asarray(result.deleted_edge_iterations, dtype=np.int64)
    deleted_edge_error_history = np.asarray(result.deleted_edge_error_history, dtype=np.float64)
    iterations = np.arange(rel_l1_history.size)

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
        [result.best_iteration],
        [flow_error_history[result.best_iteration]],
        color="tab:red",
        zorder=3,
        label=f"best objective iter={result.best_iteration}",
    )
    plt.xlabel("iteration")
    plt.ylabel("relative error")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    plt.figure(figsize=(7, 4))
    plt.semilogy(iterations, np.maximum(rel_l1_history, 1e-300), linewidth=2.0, label="OD L1 error")
    plt.scatter(
        [result.best_rel_l1_iteration],
        [rel_l1_history[result.best_rel_l1_iteration]],
        color="tab:red",
        zorder=3,
        label=f"best L1 iter={result.best_rel_l1_iteration}",
    )
    plt.xlabel("iteration")
    plt.ylabel(r"$\|D_k-D_{true}\|_1/\|D_{true}\|_1$")
    plt.title("OD matrix recovery error")
    plt.legend()
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    matrices = [
        ("initial", result.D_initial),
        ("recovered", result.D_recovered),
        ("true", result.D_true),
    ]
    vmin = min(float(matrix.min()) for _, matrix in matrices)
    vmax = max(float(matrix.max()) for _, matrix in matrices)

    _, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, (matrix_title, matrix) in zip(axes, matrices):
        axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(matrix_title)
        axis.set_xlabel("destination")
        axis.set_ylabel("origin")

    plt.show()
