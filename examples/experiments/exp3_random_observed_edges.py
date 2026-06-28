"""Эксперимент 3: вариация доли случайно наблюдаемых потоков на рёбрах.

Сравниваются пять вложенных случайных наборов: 10%, 30%, 50%, 70% и 90% рёбер
Sioux Falls. Каждый следующий набор содержит все рёбра предыдущего. В невязку
потоков и её градиент входят только наблюдаемые рёбра.

Оптимизатор, энтропийное слагаемое, жёсткая IPF-проекция на маргиналии и все
гиперпараметры совпадают с ``examples/exp1.py``. Метрики такие же.
"""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from examples import exp1
from examples.experiments import experiment_suite as suite


OBSERVED_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90)
MASK_SEED = exp1.SEED + 1_003
SHOW_PLOTS = True


def run_experiment() -> list[dict[str, object]]:
    context = suite.setup_experiment()
    scenarios = suite.build_nested_random_scenarios(
        context.reference_flow.size,
        OBSERVED_FRACTIONS,
        np.random.default_rng(MASK_SEED),
    )
    return [suite.run_masked_experiment(context, scenario) for scenario in scenarios]


if __name__ == "__main__":
    experiment_results = run_experiment()
    if SHOW_PLOTS:
        suite.plot_results(
            experiment_results,
            title="Random observed edges",
        )
