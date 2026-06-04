"""Постановка 2: регуляризованная задача Бекмана и мягкие маргиналии.

Внешняя задача:

    min_D  Phi(f_alpha(D))
           + alpha/2 * ||M(f_alpha(D) - f_hat)||^2
           + gamma * sum D_ij(log D_ij - 1)
           + lambda/2 * (||D 1 - l||^2 + ||D^T 1 - w||^2),

где f_alpha(D) решает регуляризованную задачу Бекмана. В отличие от первой
постановки, известные маргиналии входят в цель как мягкий квадратичный штраф.
"""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

import pytraffic.models.Beckmann as beckmann
from examples.experiment_utils import (
    DEFAULT_DEMAND_FILE,
    DEFAULT_NET_FILE,
    ExperimentResult,
    create_result,
    finish_result,
    plot_result,
    prepare_experiment,
    print_experiment_header,
    print_result_summary,
    record_iteration,
)
from pytraffic.models.BRPCost import BRP


EPS = 1e-12

# Данные и начальное приближение совпадают с постановкой 1.
NET_FILE = DEFAULT_NET_FILE
DEMAND_FILE = DEFAULT_DEMAND_FILE
SEED = 42
INITIAL_NOISE = 0.6

# Внешний Adam. Шаг настроен отдельно из-за другого масштаба градиента.
OUTER_ITERS = 600
LEARNING_RATE = 2.0
LEARNING_RATE_DECAY = 0.01
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8

# Параметры второй постановки.
FLOW_REGULARIZATION = 1e-4
ENTROPY_WEIGHT = 1e-2
MARGINAL_PENALTY_WEIGHT = 1.0

# Внутренние задачи Бекмана и контрольная метрика с удалением ребер.
FW_REFERENCE_ITERS = 1000
FW_INNER_ITERS = 200
FW_RGAP = 5e-5
ROBUST_METRIC_NUM_EDGES = 12
ROBUST_METRIC_EVERY = 5
ROBUST_METRIC_FW_ITERS = 200

PRINT_EVERY = 5
SHOW_PLOTS = True


def entropy_value(D: np.ndarray) -> float:
    """Выпуклая отрицательная энтропия gamma * sum D_ij(log D_ij - 1)."""
    positive = np.maximum(D[D > 0.0], EPS)
    return float(ENTROPY_WEIGHT * np.sum(positive * (np.log(positive) - 1.0)))


def entropy_gradient(D: np.ndarray) -> np.ndarray:
    return ENTROPY_WEIGHT * np.log(np.maximum(D, EPS))


def bpr_potential_value(edge_cost: BRP, flow: np.ndarray) -> float:
    """Значение потенциала Бекмана для BPR-функций стоимости."""
    flow = np.maximum(np.asarray(flow, dtype=np.float64), 0.0)
    capacity = np.asarray(edge_cost.cap, dtype=np.float64)
    free_flow_time = np.asarray(edge_cost.t0, dtype=np.float64)
    bpr_weight = np.asarray(edge_cost.alpha, dtype=np.float64)
    bpr_power = np.asarray(edge_cost.beta, dtype=np.float64)
    relative_flow = flow / capacity
    edge_potential = free_flow_time * (
        flow + bpr_weight * capacity * relative_flow ** (bpr_power + 1.0) / (bpr_power + 1.0)
    )
    return float(np.sum(edge_potential))


def marginal_penalty_value(D: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> float:
    """Квадратичный штраф за отклонение сумм по строкам и столбцам."""
    row_residual = D.sum(axis=1) - row_target
    col_residual = D.sum(axis=0) - col_target
    return float(
        0.5
        * MARGINAL_PENALTY_WEIGHT
        * (np.dot(row_residual, row_residual) + np.dot(col_residual, col_residual))
    )


def marginal_penalty_gradient(D: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> np.ndarray:
    row_residual = D.sum(axis=1) - row_target
    col_residual = D.sum(axis=0) - col_target
    return MARGINAL_PENALTY_WEIGHT * (row_residual[:, None] + col_residual[None, :])


def objective_value(
    edge_cost: BRP,
    D: np.ndarray,
    regularized_flow: np.ndarray,
    reference_flow: np.ndarray,
    experiment_mask: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
) -> float:
    """Целевая функция второй постановки."""
    observed_flow_residual = experiment_mask * (regularized_flow - reference_flow)
    flow_penalty = 0.5 * FLOW_REGULARIZATION * np.dot(observed_flow_residual, observed_flow_residual)
    return float(
        bpr_potential_value(edge_cost, regularized_flow)
        + flow_penalty
        + entropy_value(D)
        + marginal_penalty_value(D, row_target, col_target)
    )


def objective_gradient(
    D: np.ndarray,
    regularized_objective_gradient: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
) -> np.ndarray:
    """Градиент второй целевой функции по OD-матрице."""
    gradient = regularized_objective_gradient.reshape(D.shape).copy()
    gradient += entropy_gradient(D)
    gradient += marginal_penalty_gradient(D, row_target, col_target)
    np.fill_diagonal(gradient, 0.0)
    return gradient


def adam_step(
    D: np.ndarray,
    gradient: np.ndarray,
    first_moment: np.ndarray,
    second_moment: np.ndarray,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    first_moment = BETA1 * first_moment + (1.0 - BETA1) * gradient
    second_moment = BETA2 * second_moment + (1.0 - BETA2) * gradient * gradient
    first_unbiased = first_moment / (1.0 - BETA1**iteration)
    second_unbiased = second_moment / (1.0 - BETA2**iteration)

    learning_rate = LEARNING_RATE / np.sqrt(1.0 + LEARNING_RATE_DECAY * (iteration - 1))
    D -= learning_rate * first_unbiased / (np.sqrt(second_unbiased) + ADAM_EPS)
    D[D < 0.0] = EPS
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(first_moment, 0.0)
    np.fill_diagonal(second_moment, 0.0)
    return first_moment, second_moment, learning_rate


def run_experiment() -> ExperimentResult:
    data = prepare_experiment(
        NET_FILE,
        DEMAND_FILE,
        seed=SEED,
        initial_noise=INITIAL_NOISE,
        fw_reference_iters=FW_REFERENCE_ITERS,
        fw_rgap=FW_RGAP,
        robust_metric_num_edges=ROBUST_METRIC_NUM_EDGES,
        robust_metric_fw_iters=ROBUST_METRIC_FW_ITERS,
    )
    # Сейчас наблюдаем потоки на всех ребрах. Маска позволяет ограничить набор наблюдений.
    experiment_mask = np.ones_like(data.reference_flow)
    D = data.D_initial.copy()
    regularized_flow, _, regularized_gradient = beckmann.fw_beckmann_regularized_new_gradient(
        data.csr,
        data.edge_cost,
        D,
        data.reference_flow,
        experiment_mask,
        alpha=FLOW_REGULARIZATION,
        max_iter=FW_INNER_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )
    flow, _ = beckmann.fw_beckmann(
        data.csr,
        data.edge_cost,
        D,
        max_iter=FW_INNER_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )
    print_experiment_header(data, flow)
    objective = objective_value(
        data.edge_cost,
        D,
        regularized_flow,
        data.reference_flow,
        experiment_mask,
        data.row_target,
        data.col_target,
    )
    result = create_result(
        "regularized Beckmann + soft marginals",
        data,
        D,
        flow,
        objective,
        robust_metric_fw_iters=ROBUST_METRIC_FW_ITERS,
        fw_rgap=FW_RGAP,
    )
    first_moment = np.zeros_like(D)
    second_moment = np.zeros_like(D)

    for iteration in range(1, OUTER_ITERS + 1):
        gradient = objective_gradient(
            D,
            regularized_gradient,
            data.row_target,
            data.col_target,
        )
        first_moment, second_moment, learning_rate = adam_step(
            D, gradient, first_moment, second_moment, iteration
        )

        regularized_flow, _, regularized_gradient = beckmann.fw_beckmann_regularized_new_gradient(
            data.csr,
            data.edge_cost,
            D,
            data.reference_flow,
            experiment_mask,
            alpha=FLOW_REGULARIZATION,
            max_iter=FW_INNER_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )
        flow, _ = beckmann.fw_beckmann(
            data.csr,
            data.edge_cost,
            D,
            max_iter=FW_INNER_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )
        objective = objective_value(
            data.edge_cost,
            D,
            regularized_flow,
            data.reference_flow,
            experiment_mask,
            data.row_target,
            data.col_target,
        )
        rel_l1, flow_error, row_error, col_error = record_iteration(
            result,
            data,
            D,
            flow,
            objective,
            iteration,
            outer_iters=OUTER_ITERS,
            robust_metric_every=ROBUST_METRIC_EVERY,
            robust_metric_fw_iters=ROBUST_METRIC_FW_ITERS,
            fw_rgap=FW_RGAP,
        )

        if iteration == 1 or iteration % PRINT_EVERY == 0 or iteration == OUTER_ITERS:
            print(
                f"iter={iteration:03d} lr={learning_rate:.2e} "
                f"flow_err={flow_error:.4e} rel_l1={rel_l1:.4e} "
                f"row_err={row_error:.2e} col_err={col_error:.2e}"
            )

    return finish_result(result)


if __name__ == "__main__":
    experiment_result = run_experiment()
    print_result_summary(experiment_result)
    if SHOW_PLOTS:
        plot_result(experiment_result, title="Постановка 2: регуляризованный Бекман")
