"""Постановка 1: прямая невязка равновесных потоков и жесткие маргиналии.

Внешняя задача:

    min_D  1/2 ||f(D) - f_hat||^2 + gamma * sum D_ij(log D_ij - 1),

где f(D) — равновесные потоки из задачи Бекмана. После каждого шага Adam
OD-матрица проецируется на известные суммы по строкам и столбцам методом IPF.
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


EPS = 1e-12

# Данные и начальное приближение.
NET_FILE = DEFAULT_NET_FILE
DEMAND_FILE = DEFAULT_DEMAND_FILE
SEED = 42
INITIAL_NOISE = 0.6

# Внешний Adam.
OUTER_ITERS = 600
LEARNING_RATE = 4.0
LEARNING_RATE_DECAY = 0.01
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8

# Вес отрицательной энтропии и точность проекции на маргиналии.
GAMMA = 1e-2
IPF_PROJECT_ITERS = 500
IPF_PROJECT_TOL = 1e-7

# Внутренние задачи Бекмана и контрольная метрика с удалением ребер.
FW_REFERENCE_ITERS = 1000
FW_INNER_ITERS = 200
FW_RGAP = 5e-5
ROBUST_METRIC_NUM_EDGES = 12
ROBUST_METRIC_EVERY = 5
ROBUST_METRIC_FW_ITERS = 200

PRINT_EVERY = 5
SHOW_PLOTS = True


def flow_mismatch_value(flow: np.ndarray, reference_flow: np.ndarray) -> float:
    residual = flow - reference_flow
    return float(0.5 * np.dot(residual, residual))


def entropy_value(D: np.ndarray) -> float:
    """Выпуклая отрицательная энтропия gamma * sum D_ij(log D_ij - 1)."""
    positive = np.maximum(D[D > 0.0], EPS)
    return float(GAMMA * np.sum(positive * (np.log(positive) - 1.0)))


def entropy_gradient(D: np.ndarray) -> np.ndarray:
    return GAMMA * np.log(np.maximum(D, EPS))


def objective_value(D: np.ndarray, flow: np.ndarray, reference_flow: np.ndarray) -> float:
    """Целевая функция первой постановки."""
    return flow_mismatch_value(flow, reference_flow) + entropy_value(D)


def objective_gradient(
    D: np.ndarray,
    flow: np.ndarray,
    reference_flow: np.ndarray,
    flow_jacobian: np.ndarray,
) -> np.ndarray:
    """Градиент первой целевой функции по OD-матрице."""
    flow_gradient = (flow_jacobian.T @ (flow - reference_flow)).reshape(D.shape)
    gradient = flow_gradient + entropy_gradient(D)
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


def ipf_project_to_marginals(D: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> np.ndarray:
    """Проекция на известные маргиналии с фиксированной нулевой диагональю."""
    projected = np.maximum(np.asarray(D, dtype=np.float64), EPS)
    np.fill_diagonal(projected, 0.0)

    for _ in range(IPF_PROJECT_ITERS):
        projected *= (row_target / np.maximum(projected.sum(axis=1), EPS))[:, None]
        projected *= (col_target / np.maximum(projected.sum(axis=0), EPS))[None, :]
        np.fill_diagonal(projected, 0.0)

        row_error = np.max(np.abs(projected.sum(axis=1) - row_target))
        col_error = np.max(np.abs(projected.sum(axis=0) - col_target))
        if max(row_error, col_error) <= IPF_PROJECT_TOL:
            break

    return projected


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
    D = data.D_initial.copy()
    flow, flow_jacobian = beckmann.fw_beckmann(
        data.csr,
        data.edge_cost,
        D,
        max_iter=FW_INNER_ITERS,
        rgap_target=FW_RGAP,
        verbose=False,
    )
    print_experiment_header(data, flow)
    result = create_result(
        "direct flow mismatch + hard marginals",
        data,
        D,
        flow,
        objective_value(D, flow, data.reference_flow),
        robust_metric_fw_iters=ROBUST_METRIC_FW_ITERS,
        fw_rgap=FW_RGAP,
    )
    first_moment = np.zeros_like(D)
    second_moment = np.zeros_like(D)

    for iteration in range(1, OUTER_ITERS + 1):
        gradient = objective_gradient(D, flow, data.reference_flow, flow_jacobian)
        first_moment, second_moment, learning_rate = adam_step(
            D, gradient, first_moment, second_moment, iteration
        )
        D = ipf_project_to_marginals(D, data.row_target, data.col_target)

        flow, flow_jacobian = beckmann.fw_beckmann(
            data.csr,
            data.edge_cost,
            D,
            max_iter=FW_INNER_ITERS,
            rgap_target=FW_RGAP,
            verbose=False,
        )
        objective = objective_value(D, flow, data.reference_flow)
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
        plot_result(experiment_result, title="Постановка 1: прямая невязка потоков")
