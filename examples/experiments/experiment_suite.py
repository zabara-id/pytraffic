"""Shared runner for the masked Sioux Falls experiments based on ``exp1.py``."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from examples import exp1
from examples import experiment_utils
import pytraffic.models.Beckmann as beckmann


InitialODBuilder = Callable[[np.ndarray, np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class ObservationScenario:
    """One deterministic choice of observed edges."""

    name: str
    mask: np.ndarray


@dataclass
class ExperimentContext:
    """Data shared by all scenarios in one experiment script."""

    csr: object
    edge_cost: object
    D_true: np.ndarray
    D_initial: np.ndarray
    row_target: np.ndarray
    col_target: np.ndarray
    reference_flow: np.ndarray
    robust_scenarios: list[tuple[int, object, object]]
    robust_reference_flows: list[np.ndarray]


def _edge_count(fraction: float, num_edges: int) -> int:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Observed fraction must lie in (0, 1], got {fraction}")
    return min(num_edges, max(1, int(round(fraction * num_edges))))


def mask_from_indices(num_edges: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(num_edges, dtype=np.float64)
    mask[np.asarray(indices, dtype=np.int64)] = 1.0
    return mask


def build_nested_random_scenarios(
    num_edges: int,
    fractions: Sequence[float],
    rng: np.random.Generator,
) -> list[ObservationScenario]:
    """Build nested random masks: every smaller mask is a subset of every larger one."""
    ordered_fractions = sorted(float(fraction) for fraction in fractions)
    permutation = rng.permutation(num_edges)
    return [
        ObservationScenario(
            name=f"random {fraction:.0%}",
            mask=mask_from_indices(num_edges, permutation[: _edge_count(fraction, num_edges)]),
        )
        for fraction in ordered_fractions
    ]


def build_flow_rank_scenarios(
    reference_flow: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> list[ObservationScenario]:
    """Build equal-sized masks for the largest, smallest, and random reference flows."""
    reference_flow = np.asarray(reference_flow, dtype=np.float64)
    num_edges = reference_flow.size
    count = _edge_count(fraction, num_edges)
    ascending = np.argsort(reference_flow, kind="stable")

    return [
        ObservationScenario(
            name=f"top {fraction:.0%}",
            mask=mask_from_indices(num_edges, ascending[-count:]),
        ),
        ObservationScenario(
            name=f"bottom {fraction:.0%}",
            mask=mask_from_indices(num_edges, ascending[:count]),
        ),
        ObservationScenario(
            name=f"random {fraction:.0%}",
            mask=mask_from_indices(num_edges, rng.permutation(num_edges)[:count]),
        ),
    ]


def build_top_flow_scenarios(
    reference_flow: np.ndarray,
    fractions: Sequence[float],
) -> list[ObservationScenario]:
    """Build nested masks containing the edges with the largest reference flows."""
    reference_flow = np.asarray(reference_flow, dtype=np.float64)
    num_edges = reference_flow.size
    descending = np.argsort(-reference_flow, kind="stable")

    return [
        ObservationScenario(
            name=f"top {fraction:.0%}",
            mask=mask_from_indices(num_edges, descending[: _edge_count(float(fraction), num_edges)]),
        )
        for fraction in sorted(float(fraction) for fraction in fractions)
    ]


def make_multiplicative_noisy_initial_od(
    D_true: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    rng: np.random.Generator,
    *,
    target_relative_l1: float,
    max_bisection_iters: int = 80,
) -> np.ndarray:
    """Create ``D_true * (1 + noise)`` with a prescribed L1 error after IPF."""
    if target_relative_l1 < 0.0:
        raise ValueError("target_relative_l1 must be nonnegative")
    if target_relative_l1 == 0.0:
        return D_true.copy()

    noise_direction = rng.normal(0.0, 1.0, size=D_true.shape)
    np.fill_diagonal(noise_direction, 0.0)

    def candidate(scale: float) -> tuple[np.ndarray, float]:
        noisy = D_true * np.maximum(1.0 + scale * noise_direction, experiment_utils.EPS)
        np.fill_diagonal(noisy, 0.0)
        projected = exp1.ipf_project_to_marginals(
            noisy,
            row_target,
            col_target,
        )
        return projected, experiment_utils.relative_l1_error(projected, D_true)

    lower_scale = 0.0
    upper_scale = 1.0
    upper_D, upper_error = candidate(upper_scale)
    while upper_error < target_relative_l1 and upper_scale < 1e6:
        lower_scale = upper_scale
        upper_scale *= 2.0
        upper_D, upper_error = candidate(upper_scale)

    if upper_error < target_relative_l1:
        raise RuntimeError(
            "Could not construct the requested noisy initial OD matrix: "
            f"target={target_relative_l1:.6e}, largest error={upper_error:.6e}"
        )

    best_D = upper_D
    best_error = upper_error
    for _ in range(max_bisection_iters):
        middle_scale = 0.5 * (lower_scale + upper_scale)
        middle_D, middle_error = candidate(middle_scale)
        if abs(middle_error - target_relative_l1) < abs(best_error - target_relative_l1):
            best_D = middle_D
            best_error = middle_error
        if middle_error < target_relative_l1:
            lower_scale = middle_scale
        else:
            upper_scale = middle_scale

    return best_D


def setup_experiment(
    *,
    initial_od_builder: InitialODBuilder | None = None,
    seed: int = exp1.SEED,
    fw_reference_iters: int = exp1.FW_REFERENCE_ITERS,
    fw_rgap: float = exp1.FW_RGAP,
    robust_metric_num_edges: int = exp1.ROBUST_METRIC_NUM_EDGES,
    robust_metric_fw_iters: int = exp1.ROBUST_METRIC_FW_ITERS,
) -> ExperimentContext:
    """Load data and compute references shared by all masks in a script."""
    data = experiment_utils.prepare_experiment(
        exp1.NET_FILE,
        exp1.DEMAND_FILE,
        seed=seed,
        initial_noise=exp1.INITIAL_NOISE,
        fw_reference_iters=fw_reference_iters,
        fw_rgap=fw_rgap,
        robust_metric_num_edges=robust_metric_num_edges,
        robust_metric_fw_iters=robust_metric_fw_iters,
    )
    D_initial = data.D_initial
    if initial_od_builder is not None:
        D_initial = initial_od_builder(
            data.D_true,
            data.row_target,
            data.col_target,
            np.random.default_rng(seed),
        )

    context = ExperimentContext(
        csr=data.csr,
        edge_cost=data.edge_cost,
        D_true=data.D_true,
        D_initial=D_initial,
        row_target=data.row_target,
        col_target=data.col_target,
        reference_flow=data.reference_flow,
        robust_scenarios=data.robust_scenarios,
        robust_reference_flows=data.robust_reference_flows,
    )
    print_context(context)
    return context


def print_context(context: ExperimentContext) -> None:
    initial_flow, _ = beckmann.fw_beckmann(
        context.csr,
        context.edge_cost,
        context.D_initial,
        max_iter=exp1.FW_INNER_ITERS,
        rgap_target=exp1.FW_RGAP,
        verbose=False,
    )
    print(
        f"Sioux Falls: nodes={context.csr.n}, edges={context.csr.m}, "
        f"total_demand={context.D_true.sum():.1f}, "
        f"initial_rel_l1={experiment_utils.relative_l1_error(context.D_initial, context.D_true):.4e}, "
        f"initial_flow_err={experiment_utils.relative_flow_error(initial_flow, context.reference_flow):.4e}"
    )
    print(
        f"Deleted-edge validation scenarios: {len(context.robust_scenarios)} "
        f"edges={[edge_id for edge_id, _, _ in context.robust_scenarios]}"
    )
    n = context.D_true.shape[0]
    od_variables = n * (n - 1)
    independent_marginals = 2 * n - 1
    underdetermined = max(od_variables - independent_marginals - context.csr.m, 0)
    print(
        "Identifiability check: "
        f"OD variables={od_variables}, edge counts={context.csr.m}, "
        f"independent marginals={independent_marginals}, "
        f"underdetermined directions >= {underdetermined}"
    )


def deleted_edge_metric(context: ExperimentContext, D: np.ndarray) -> float:
    if not context.robust_scenarios:
        return np.nan

    errors = []
    for (_, scenario_csr, scenario_edge_cost), reference_flow in zip(
        context.robust_scenarios,
        context.robust_reference_flows,
    ):
        flow, _ = beckmann.fw_beckmann(
            scenario_csr,
            scenario_edge_cost,
            D,
            max_iter=exp1.ROBUST_METRIC_FW_ITERS,
            rgap_target=exp1.FW_RGAP,
            verbose=False,
        )
        errors.append(experiment_utils.relative_flow_error(flow, reference_flow))
    return float(np.mean(errors))


def masked_flow_mismatch_value(
    flow: np.ndarray,
    reference_flow: np.ndarray,
    observation_mask: np.ndarray,
) -> float:
    residual = observation_mask * (flow - reference_flow)
    return float(0.5 * np.dot(residual, residual))


def _validated_mask(observation_mask: np.ndarray, reference_flow: np.ndarray) -> np.ndarray:
    mask = np.asarray(observation_mask, dtype=np.float64)
    if mask.shape != reference_flow.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match flow shape {reference_flow.shape}")
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ValueError("Observation mask must contain only zeros and ones")
    if not np.any(mask):
        raise ValueError("At least one edge must be observed")
    return mask


def run_masked_experiment(
    context: ExperimentContext,
    scenario: ObservationScenario,
    *,
    outer_iters: int = exp1.OUTER_ITERS,
    learning_rate_base: float = exp1.LEARNING_RATE,
    fw_inner_iters: int = exp1.FW_INNER_ITERS,
    fw_rgap: float = exp1.FW_RGAP,
    robust_metric_every: int = exp1.ROBUST_METRIC_EVERY,
    print_every: int = exp1.PRINT_EVERY,
) -> dict[str, object]:
    """Run the exact Adam/IPF formulation from exp1 with a masked flow residual."""
    observation_mask = _validated_mask(scenario.mask, context.reference_flow)
    D = context.D_initial.copy()
    flow, flow_jacobian = beckmann.fw_beckmann(
        context.csr,
        context.edge_cost,
        D,
        max_iter=fw_inner_iters,
        rgap_target=fw_rgap,
        verbose=False,
    )

    first_moment = np.zeros_like(D)
    second_moment = np.zeros_like(D)
    trajectory = [D.copy()]
    rel_l1_history = [experiment_utils.relative_l1_error(D, context.D_true)]
    flow_error_history = [experiment_utils.relative_flow_error(flow, context.reference_flow)]
    deleted_edge_error_history = [deleted_edge_metric(context, D)]
    deleted_edge_iterations = [0]
    objective_history = [
        masked_flow_mismatch_value(flow, context.reference_flow, observation_mask)
        + exp1.entropy_value(D)
    ]

    best_objective = objective_history[-1]
    best_rel_l1 = rel_l1_history[-1]
    best_rel_l1_iteration = 0
    best_iteration = 0
    best_D = D.copy()
    row_error, col_error = experiment_utils.relative_marginal_errors(
        D,
        context.row_target,
        context.col_target,
    )
    row_error_history = [row_error]
    col_error_history = [col_error]

    print(
        f"\nScenario {scenario.name!r}: observed_edges={int(observation_mask.sum())}/{observation_mask.size} "
        f"({observation_mask.mean():.1%})"
    )

    for iteration in range(1, outer_iters + 1):
        flow_residual = observation_mask * (flow - context.reference_flow)
        flow_grad = (flow_jacobian.T @ flow_residual).reshape(D.shape)
        grad = flow_grad + exp1.entropy_gradient(D)
        np.fill_diagonal(grad, 0.0)

        first_moment = exp1.BETA1 * first_moment + (1.0 - exp1.BETA1) * grad
        second_moment = exp1.BETA2 * second_moment + (1.0 - exp1.BETA2) * grad * grad
        first_unbiased = first_moment / (1.0 - exp1.BETA1**iteration)
        second_unbiased = second_moment / (1.0 - exp1.BETA2**iteration)

        learning_rate = learning_rate_base / np.sqrt(
            1.0 + exp1.LEARNING_RATE_DECAY * (iteration - 1)
        )
        D -= learning_rate * first_unbiased / (np.sqrt(second_unbiased) + exp1.ADAM_EPS)
        D[D < 0.0] = experiment_utils.EPS
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(first_moment, 0.0)
        np.fill_diagonal(second_moment, 0.0)
        D = exp1.ipf_project_to_marginals(D, context.row_target, context.col_target)

        flow, flow_jacobian = beckmann.fw_beckmann(
            context.csr,
            context.edge_cost,
            D,
            max_iter=fw_inner_iters,
            rgap_target=fw_rgap,
            verbose=False,
        )

        rel_l1 = experiment_utils.relative_l1_error(D, context.D_true)
        flow_error = experiment_utils.relative_flow_error(flow, context.reference_flow)
        objective = (
            masked_flow_mismatch_value(flow, context.reference_flow, observation_mask)
            + exp1.entropy_value(D)
        )
        row_error, col_error = experiment_utils.relative_marginal_errors(
            D,
            context.row_target,
            context.col_target,
        )

        trajectory.append(D.copy())
        rel_l1_history.append(rel_l1)
        flow_error_history.append(flow_error)
        objective_history.append(objective)
        row_error_history.append(row_error)
        col_error_history.append(col_error)

        if iteration % robust_metric_every == 0 or iteration == outer_iters:
            deleted_edge_iterations.append(iteration)
            deleted_edge_error_history.append(
                deleted_edge_metric(context, D)
            )

        if objective < best_objective:
            best_objective = objective
            best_iteration = iteration
            best_D = D.copy()

        if rel_l1 < best_rel_l1:
            best_rel_l1 = rel_l1
            best_rel_l1_iteration = iteration

        if iteration == 1 or iteration % print_every == 0 or iteration == outer_iters:
            print(
                f"iter={iteration:03d} lr={learning_rate:.2e} "
                f"flow_err={flow_error:.4e} best_obj={best_objective:.4e}@{best_iteration:03d} "
                f"rel_l1={rel_l1:.4e} best_l1={best_rel_l1:.4e}@{best_rel_l1_iteration:03d} "
                f"row_err={row_error:.2e} col_err={col_error:.2e}"
            )

    result: dict[str, object] = {
        "scenario_name": scenario.name,
        "observation_mask": observation_mask.copy(),
        "D_true": context.D_true,
        "D_initial": context.D_initial,
        "D_recovered": best_D,
        "D_final": D,
        "best_iteration": best_iteration,
        "best_objective": best_objective,
        "best_rel_l1": best_rel_l1,
        "best_rel_l1_iteration": best_rel_l1_iteration,
        "reference_flow": context.reference_flow,
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
    print_result_summary(result)
    return result


def print_result_summary(result: dict[str, object]) -> None:
    print("Final relative L1 error:", f"{result['rel_l1_history'][-1]:.6e}")
    print("Final relative flow error:", f"{result['flow_error_history'][-1]:.6e}")
    print("Final deleted-edge flow error:", f"{result['deleted_edge_error_history'][-1]:.6e}")
    print(
        "Best relative L1 error:",
        f"{result['best_rel_l1']:.6e}",
        f"at iteration {result['best_rel_l1_iteration']}",
    )
    best_iteration = int(result["best_iteration"])
    print(
        "Best objective iteration:",
        best_iteration,
        "flow error:",
        f"{result['flow_error_history'][best_iteration]:.6e}",
    )


def plot_results(results: Sequence[dict[str, object]], *, title: str) -> None:
    """Plot the same flow, deleted-edge, OD, and heatmap diagnostics as exp1."""
    import matplotlib.pyplot as plt

    if not results:
        return

    num_results = len(results)
    colors = plt.get_cmap("tab10").colors

    flow_figure, (flow_axis, deleted_axis) = plt.subplots(
        1,
        2,
        figsize=(15, 5),
        constrained_layout=True,
    )

    for index, result in enumerate(results):
        color = colors[index % len(colors)]
        name = str(result["scenario_name"])
        flow_errors = np.asarray(result["flow_error_history"], dtype=np.float64)
        deleted_iterations = np.asarray(result["deleted_edge_iterations"], dtype=np.int64)
        deleted_errors = np.asarray(result["deleted_edge_error_history"], dtype=np.float64)
        iterations = np.arange(flow_errors.size)
        best_iteration = int(result["best_iteration"])

        flow_axis.semilogy(
            iterations,
            np.maximum(flow_errors, 1e-300),
            color=color,
            linewidth=2.0,
            label=name,
        )
        deleted_axis.semilogy(
            deleted_iterations,
            np.maximum(deleted_errors, 1e-300),
            "o--",
            color=color,
            linewidth=1.5,
            markersize=3,
            label=name,
        )
        flow_axis.scatter(
            [best_iteration],
            [flow_errors[best_iteration]],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    flow_axis.set_title("Full-flow error")
    deleted_axis.set_title("Deleted-edge error")
    for axis in (flow_axis, deleted_axis):
        axis.set_xlabel("iteration")
        axis.set_ylabel("relative error")
        axis.legend(fontsize=8)
        axis.grid(True, which="both", alpha=0.35)

    flow_figure.suptitle(f"{title}: full-flow and deleted-edge errors")

    od_figure, od_axis = plt.subplots(
        figsize=(8, 5),
        constrained_layout=True,
    )

    for index, result in enumerate(results):
        color = colors[index % len(colors)]
        name = str(result["scenario_name"])
        rel_l1 = np.asarray(result["rel_l1_history"], dtype=np.float64)
        iterations = np.arange(rel_l1.size)
        best_iteration = int(result["best_rel_l1_iteration"])
        od_axis.semilogy(
            iterations,
            np.maximum(rel_l1, 1e-300),
            color=color,
            linewidth=2.0,
            label=name,
        )
        od_axis.scatter(
            [best_iteration],
            [rel_l1[best_iteration]],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    od_axis.set_xlabel("iteration")
    od_axis.set_ylabel(r"$\|D_k-D_{true}\|_1/\|D_{true}\|_1$")
    od_axis.set_title(f"{title}: OD matrix relative L1 error")
    od_axis.legend(fontsize=8)
    od_axis.grid(True, which="both", alpha=0.35)

    _, axes = plt.subplots(
        num_results,
        3,
        figsize=(14, max(4.0, 3.6 * num_results)),
        squeeze=False,
        constrained_layout=True,
    )
    matrices = [
        np.asarray(result[key])
        for result in results
        for key in ("D_initial", "D_recovered", "D_true")
    ]
    vmin = min(float(matrix.min()) for matrix in matrices)
    vmax = max(float(matrix.max()) for matrix in matrices)

    for row, result in enumerate(results):
        name = str(result["scenario_name"])
        best_iteration = int(result["best_iteration"])
        row_matrices = (
            np.asarray(result["D_initial"]),
            np.asarray(result["D_recovered"]),
            np.asarray(result["D_true"]),
        )
        row_titles = (
            f"{name}: initial",
            f"{name}: recovered, iter={best_iteration}",
            f"{name}: true",
        )
        for column, (matrix, matrix_title) in enumerate(zip(row_matrices, row_titles)):
            axes[row, column].imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row, column].set_title(matrix_title)
            axes[row, column].set_xlabel("destination")
            axes[row, column].set_ylabel("origin")

    plt.show()
