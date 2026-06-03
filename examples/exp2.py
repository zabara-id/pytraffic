import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from examples import exp1
import pytraffic.models.Beckmann as beckmann


EPS = exp1.EPS

# Same data and same initial approximation as in exp1.py.
NET_FILE = exp1.NET_FILE
DEMAND_FILE = exp1.DEMAND_FILE
SEED = exp1.SEED
INITIAL_NOISE = exp1.INITIAL_NOISE

# Same outer experiment scale as in exp1.py.
OUTER_ITERS = exp1.OUTER_ITERS
LEARNING_RATE_DECAY = exp1.LEARNING_RATE_DECAY
BETA1 = exp1.BETA1
BETA2 = exp1.BETA2
ADAM_EPS = exp1.ADAM_EPS

# The modified formulation is exactly the one currently used in exp1.py.
MODIFIED_LEARNING_RATE = exp1.LEARNING_RATE
MODIFIED_GAMMA = exp1.GAMMA

# The original formulation has different units in the gradient, so its Adam step is tuned separately.
ORIGINAL_LEARNING_RATE = 2.0
ORIGINAL_ALPHA = 1e-4
ORIGINAL_GAMMA = exp1.GAMMA
ORIGINAL_LAMBDA = 1.0

# Same Beckmann and validation settings as in exp1.py.
FW_REFERENCE_ITERS = exp1.FW_REFERENCE_ITERS
FW_INNER_ITERS = exp1.FW_INNER_ITERS
FW_RGAP = exp1.FW_RGAP
ROBUST_METRIC_NUM_EDGES = exp1.ROBUST_METRIC_NUM_EDGES
ROBUST_METRIC_EVERY = exp1.ROBUST_METRIC_EVERY
ROBUST_METRIC_FW_ITERS = exp1.ROBUST_METRIC_FW_ITERS
IPF_PROJECT_ITERS = exp1.IPF_PROJECT_ITERS
IPF_PROJECT_TOL = exp1.IPF_PROJECT_TOL

PRINT_EVERY = exp1.PRINT_EVERY
SHOW_PLOTS = True


def bpr_potential_value(edge_cost, flow: np.ndarray) -> float:
    flow = np.maximum(np.asarray(flow, dtype=np.float64), 0.0)
    cap = np.asarray(edge_cost.cap, dtype=np.float64)
    t0 = np.asarray(edge_cost.t0, dtype=np.float64)
    alpha_bpr = np.asarray(edge_cost.alpha, dtype=np.float64)
    beta_bpr = np.asarray(edge_cost.beta, dtype=np.float64)
    x = flow / cap
    value = t0 * (flow + alpha_bpr * cap * x ** (beta_bpr + 1.0) / (beta_bpr + 1.0))
    return float(np.sum(value))


def marginal_penalty_value(
    D: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    penalty_weight: float,
) -> float:
    row_residual = D.sum(axis=1) - row_target
    col_residual = D.sum(axis=0) - col_target
    return float(0.5 * penalty_weight * (np.dot(row_residual, row_residual) + np.dot(col_residual, col_residual)))


def marginal_penalty_gradient(
    D: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    penalty_weight: float,
) -> np.ndarray:
    row_residual = D.sum(axis=1) - row_target
    col_residual = D.sum(axis=0) - col_target
    return penalty_weight * (row_residual[:, None] + col_residual[None, :])


def original_objective_value(
    edge_cost,
    D: np.ndarray,
    regularized_flow: np.ndarray,
    reference_flow: np.ndarray,
    experiment_mask: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
) -> float:
    flow_residual = experiment_mask * (regularized_flow - reference_flow)
    beckmann_value = bpr_potential_value(edge_cost, regularized_flow)
    flow_penalty = 0.5 * ORIGINAL_ALPHA * np.dot(flow_residual, flow_residual)
    entropy = exp1.entropy_value(D, ORIGINAL_GAMMA)
    marginal_penalty = marginal_penalty_value(D, row_target, col_target, ORIGINAL_LAMBDA)
    return float(beckmann_value + flow_penalty + entropy + marginal_penalty)


def deleted_edge_metric(
    scenarios,
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
        errors.append(exp1.relative_flow_error(flow, reference_flow))

    return float(np.mean(errors))


def setup_common_experiment() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    csr, edge_cost, D_true = exp1.load_sioux_falls(NET_FILE, DEMAND_FILE)

    row_target = D_true.sum(axis=1)
    col_target = D_true.sum(axis=0)
    D_initial = exp1.make_noisy_initial_od(
        D_true,
        row_target,
        col_target,
        rng,
        relative_noise=INITIAL_NOISE,
    )

    reference_flow, _ = beckmann.fw_beckmann(
        csr,
        edge_cost,
        D_true,
        max_iter=FW_REFERENCE_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )

    robust_scenarios = exp1.build_deleted_edge_scenarios(
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

    return {
        "csr": csr,
        "edge_cost": edge_cost,
        "D_true": D_true,
        "D_initial": D_initial,
        "row_target": row_target,
        "col_target": col_target,
        "reference_flow": reference_flow,
        "robust_scenarios": robust_scenarios,
        "robust_reference_flows": robust_reference_flows,
    }


def evaluate_standard_metrics(common: dict[str, object], D: np.ndarray) -> tuple[np.ndarray, float, float]:
    csr = common["csr"]
    edge_cost = common["edge_cost"]
    reference_flow = common["reference_flow"]
    robust_scenarios = common["robust_scenarios"]
    robust_reference_flows = common["robust_reference_flows"]

    flow, _ = beckmann.fw_beckmann(
        csr,
        edge_cost,
        D,
        max_iter=FW_INNER_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )
    flow_error = exp1.relative_flow_error(flow, reference_flow)
    robust_error = deleted_edge_metric(robust_scenarios, D, robust_reference_flows)
    return flow, flow_error, robust_error


def empty_result(method_name: str, D: np.ndarray, objective: float, common: dict[str, object]) -> dict[str, object]:
    D_true = common["D_true"]
    row_target = common["row_target"]
    col_target = common["col_target"]
    flow, flow_error, robust_error = evaluate_standard_metrics(common, D)
    row_error, col_error = exp1.relative_marginal_errors(D, row_target, col_target)

    return {
        "method_name": method_name,
        "D_recovered": D.copy(),
        "D_final": D.copy(),
        "best_iteration": 0,
        "best_objective": objective,
        "best_rel_l1": exp1.relative_l1_error(D, D_true),
        "best_rel_l1_iteration": 0,
        "final_flow": flow,
        "trajectory": [D.copy()],
        "rel_l1_history": [exp1.relative_l1_error(D, D_true)],
        "flow_error_history": [flow_error],
        "deleted_edge_iterations": [0],
        "deleted_edge_error_history": [robust_error],
        "objective_history": [objective],
        "row_error_history": [row_error],
        "col_error_history": [col_error],
    }


def update_best_state(result: dict[str, object], D: np.ndarray, objective: float, rel_l1: float, iteration: int) -> None:
    if objective < result["best_objective"]:
        result["best_objective"] = float(objective)
        result["best_iteration"] = iteration
        result["D_recovered"] = D.copy()

    if rel_l1 < result["best_rel_l1"]:
        result["best_rel_l1"] = float(rel_l1)
        result["best_rel_l1_iteration"] = iteration


def append_iteration_metrics(
    result: dict[str, object],
    common: dict[str, object],
    D: np.ndarray,
    flow: np.ndarray,
    objective: float,
    iteration: int,
) -> tuple[float, float, float, float]:
    D_true = common["D_true"]
    row_target = common["row_target"]
    col_target = common["col_target"]
    reference_flow = common["reference_flow"]
    robust_scenarios = common["robust_scenarios"]
    robust_reference_flows = common["robust_reference_flows"]

    rel_l1 = exp1.relative_l1_error(D, D_true)
    flow_error = exp1.relative_flow_error(flow, reference_flow)
    row_error, col_error = exp1.relative_marginal_errors(D, row_target, col_target)

    result["trajectory"].append(D.copy())
    result["rel_l1_history"].append(rel_l1)
    result["flow_error_history"].append(flow_error)
    result["objective_history"].append(float(objective))
    result["row_error_history"].append(row_error)
    result["col_error_history"].append(col_error)

    if iteration % ROBUST_METRIC_EVERY == 0 or iteration == OUTER_ITERS:
        result["deleted_edge_iterations"].append(iteration)
        result["deleted_edge_error_history"].append(deleted_edge_metric(robust_scenarios, D, robust_reference_flows))

    result["D_final"] = D.copy()
    result["final_flow"] = flow.copy()
    update_best_state(result, D, objective, rel_l1, iteration)

    return rel_l1, flow_error, row_error, col_error


def adam_step(
    D: np.ndarray,
    grad: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    iteration: int,
    learning_rate_base: float,
) -> tuple[np.ndarray, np.ndarray]:
    first_moment = BETA1 * first_moment + (1.0 - BETA1) * grad
    second_moment = BETA2 * second_moment + (1.0 - BETA2) * grad * grad
    first_unbiased = first_moment / (1.0 - BETA1**iteration)
    second_unbiased = second_moment / (1.0 - BETA2**iteration)

    learning_rate = learning_rate_base / np.sqrt(1.0 + LEARNING_RATE_DECAY * (iteration - 1))
    D -= learning_rate * first_unbiased / (np.sqrt(second_unbiased) + ADAM_EPS)
    exp1.project_nonnegative_zero_diag(D, first_moment, second_moment)
    return first_moment, second_moment


def run_modified_formulation(common: dict[str, object]) -> dict[str, object]:
    csr = common["csr"]
    edge_cost = common["edge_cost"]
    row_target = common["row_target"]
    col_target = common["col_target"]
    reference_flow = common["reference_flow"]

    D = common["D_initial"].copy()
    flow, flow_jacobian = beckmann.fw_beckmann(
        csr,
        edge_cost,
        D,
        max_iter=FW_INNER_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )
    objective = exp1.flow_mismatch_value(flow, reference_flow) + exp1.entropy_value(D, MODIFIED_GAMMA)
    result = empty_result("modified: direct flow mismatch + hard marginals", D, objective, common)

    first_moment = np.zeros_like(D)
    second_moment = np.zeros_like(D)

    print("\nModified formulation")
    for iteration in range(1, OUTER_ITERS + 1):
        flow_residual = flow - reference_flow
        flow_grad = (flow_jacobian.T @ flow_residual).reshape(D.shape)
        grad = flow_grad + exp1.entropy_gradient(D, MODIFIED_GAMMA)
        np.fill_diagonal(grad, 0.0)

        first_moment, second_moment = adam_step(
            D,
            grad,
            first_moment,
            second_moment,
            iteration,
            MODIFIED_LEARNING_RATE,
        )
        D = exp1.ipf_project_to_marginals(
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
        objective = exp1.flow_mismatch_value(flow, reference_flow) + exp1.entropy_value(D, MODIFIED_GAMMA)
        rel_l1, flow_error, row_error, col_error = append_iteration_metrics(
            result,
            common,
            D,
            flow,
            objective,
            iteration,
        )

        if iteration == 1 or iteration % PRINT_EVERY == 0 or iteration == OUTER_ITERS:
            learning_rate = MODIFIED_LEARNING_RATE / np.sqrt(1.0 + LEARNING_RATE_DECAY * (iteration - 1))
            print(
                f"modified iter={iteration:03d} lr={learning_rate:.2e} "
                f"flow_err={flow_error:.4e} rel_l1={rel_l1:.4e} "
                f"row_err={row_error:.2e} col_err={col_error:.2e}"
            )

    result["trajectory"] = np.stack(result["trajectory"])
    return result


def run_original_formulation(common: dict[str, object]) -> dict[str, object]:
    csr = common["csr"]
    edge_cost = common["edge_cost"]
    row_target = common["row_target"]
    col_target = common["col_target"]
    reference_flow = common["reference_flow"]
    experiment_mask = np.ones_like(reference_flow)

    D = common["D_initial"].copy()
    regularized_flow, _, regularized_grad = beckmann.fw_beckmann_regularized_new_gradient(
        csr,
        edge_cost,
        D,
        reference_flow,
        experiment_mask,
        alpha=ORIGINAL_ALPHA,
        max_iter=FW_INNER_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )
    objective = original_objective_value(
        edge_cost,
        D,
        regularized_flow,
        reference_flow,
        experiment_mask,
        row_target,
        col_target,
    )
    result = empty_result("original: regularized Beckmann, new gradient + entropy + marginal penalty", D, objective, common)

    first_moment = np.zeros_like(D)
    second_moment = np.zeros_like(D)

    print("\nOriginal formulation")
    for iteration in range(1, OUTER_ITERS + 1):
        grad = regularized_grad.reshape(D.shape)
        grad += exp1.entropy_gradient(D, ORIGINAL_GAMMA)
        grad += marginal_penalty_gradient(D, row_target, col_target, ORIGINAL_LAMBDA)
        np.fill_diagonal(grad, 0.0)

        first_moment, second_moment = adam_step(
            D,
            grad,
            first_moment,
            second_moment,
            iteration,
            ORIGINAL_LEARNING_RATE,
        )

        flow, _ = beckmann.fw_beckmann(
            csr,
            edge_cost,
            D,
            max_iter=FW_INNER_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )
        regularized_flow, _, regularized_grad = beckmann.fw_beckmann_regularized_new_gradient(
            csr,
            edge_cost,
            D,
            reference_flow,
            experiment_mask,
            alpha=ORIGINAL_ALPHA,
            max_iter=FW_INNER_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )
        objective = original_objective_value(
            edge_cost,
            D,
            regularized_flow,
            reference_flow,
            experiment_mask,
            row_target,
            col_target,
        )
        rel_l1, flow_error, row_error, col_error = append_iteration_metrics(
            result,
            common,
            D,
            flow,
            objective,
            iteration,
        )

        if iteration == 1 or iteration % PRINT_EVERY == 0 or iteration == OUTER_ITERS:
            learning_rate = ORIGINAL_LEARNING_RATE / np.sqrt(1.0 + LEARNING_RATE_DECAY * (iteration - 1))
            print(
                f"original iter={iteration:03d} lr={learning_rate:.2e} "
                f"flow_err={flow_error:.4e} rel_l1={rel_l1:.4e} "
                f"row_err={row_error:.2e} col_err={col_error:.2e}"
            )

    result["trajectory"] = np.stack(result["trajectory"])
    return result


def run_comparison() -> tuple[dict[str, object], list[dict[str, object]]]:
    common = setup_common_experiment()
    csr = common["csr"]
    D_true = common["D_true"]
    D_initial = common["D_initial"]
    robust_scenarios = common["robust_scenarios"]
    reference_flow = common["reference_flow"]
    initial_flow, initial_flow_error, initial_robust_error = evaluate_standard_metrics(common, D_initial)

    print(
        f"Sioux Falls: nodes={csr.n}, edges={csr.m}, total_demand={D_true.sum():.1f}, "
        f"initial_rel_l1={exp1.relative_l1_error(D_initial, D_true):.4e}, "
        f"initial_flow_err={initial_flow_error:.4e}, initial_deleted_edge_err={initial_robust_error:.4e}"
    )
    print(
        f"Deleted-edge validation scenarios: {len(robust_scenarios)} "
        f"edges={[edge_id for edge_id, _, _ in robust_scenarios]}"
    )
    exp1.print_identifiability_warning(csr, D_true)

    common["initial_flow"] = initial_flow
    results = [
        run_modified_formulation(common),
        run_original_formulation(common),
    ]
    return common, results


def plot_comparison(common: dict[str, object], results: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = ["tab:blue", "tab:orange"]

    plt.figure(figsize=(9, 4.8))
    for result, color in zip(results, colors):
        name = result["method_name"]
        flow_error_history = np.asarray(result["flow_error_history"], dtype=np.float64)
        deleted_edge_iterations = np.asarray(result["deleted_edge_iterations"], dtype=np.int64)
        deleted_edge_error_history = np.asarray(result["deleted_edge_error_history"], dtype=np.float64)
        iterations = np.arange(flow_error_history.size)
        best_iteration = int(result["best_iteration"])

        plt.semilogy(iterations, np.maximum(flow_error_history, 1e-300), color=color, linewidth=2.0, label=f"{name}: flow")
        plt.semilogy(
            deleted_edge_iterations,
            np.maximum(deleted_edge_error_history, 1e-300),
            "o--",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=f"{name}: deleted-edge",
        )
        plt.scatter(
            [best_iteration],
            [flow_error_history[best_iteration]],
            color=color,
            edgecolor="black",
            zorder=3,
        )

    plt.xlabel("iteration")
    plt.ylabel("relative error")
    plt.title("Sioux Falls OD recovery: flow metrics")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    plt.figure(figsize=(9, 4.8))
    for result, color in zip(results, colors):
        name = result["method_name"]
        rel_l1_history = np.asarray(result["rel_l1_history"], dtype=np.float64)
        iterations = np.arange(rel_l1_history.size)
        best_rel_l1_iteration = int(result["best_rel_l1_iteration"])

        plt.semilogy(iterations, np.maximum(rel_l1_history, 1e-300), color=color, linewidth=2.0, label=name)
        plt.scatter(
            [best_rel_l1_iteration],
            [rel_l1_history[best_rel_l1_iteration]],
            color=color,
            edgecolor="black",
            zorder=3,
        )

    plt.xlabel("iteration")
    plt.ylabel(r"$\|D_k-D_{true}\|_1/\|D_{true}\|_1$")
    plt.title("Sioux Falls OD recovery: OD matrix error")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()

    D_initial = np.asarray(common["D_initial"])
    D_true = np.asarray(common["D_true"])
    matrices = [
        ("initial", D_initial),
        ("modified recovered", np.asarray(results[0]["D_recovered"])),
        ("original recovered", np.asarray(results[1]["D_recovered"])),
        ("true", D_true),
    ]
    vmin = min(float(matrix.min()) for _, matrix in matrices)
    vmax = max(float(matrix.max()) for _, matrix in matrices)

    _, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
    for axis, (title, matrix) in zip(axes, matrices):
        axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("destination")
        axis.set_ylabel("origin")

    plt.show()


if __name__ == "__main__":
    common_result, comparison_results = run_comparison()

    print("\nSummary")
    for result in comparison_results:
        best_iteration = int(result["best_iteration"])
        print(
            result["method_name"],
            "\n  final flow error:",
            f"{result['flow_error_history'][-1]:.6e}",
            "\n  final deleted-edge flow error:",
            f"{result['deleted_edge_error_history'][-1]:.6e}",
            "\n  final marginal errors:",
            f"row={result['row_error_history'][-1]:.3e}",
            f"col={result['col_error_history'][-1]:.3e}",
            "\n  best objective iteration:",
            best_iteration,
            "flow error:",
            f"{result['flow_error_history'][best_iteration]:.6e}",
        )

    if SHOW_PLOTS:
        plot_comparison(common_result, comparison_results)
