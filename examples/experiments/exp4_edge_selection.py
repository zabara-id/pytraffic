"""Эксперимент 4: сравнение трёх способов выбрать 50% наблюдаемых рёбер.

Сравниваются рёбра с максимальными референсными потоками, рёбра с минимальными
референсными потоками и случайные рёбра. Во всех сценариях наблюдается ровно
половина сети; только выбранные рёбра входят в невязку и её градиент.

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


OBSERVED_FRACTION = 0.50
MASK_SEED = exp1.SEED + 1_004
SHOW_PLOTS = True


def run_experiment() -> list[dict[str, object]]:
    context = suite.setup_experiment()
    scenarios = suite.build_flow_rank_scenarios(
        context.reference_flow,
        OBSERVED_FRACTION,
        np.random.default_rng(MASK_SEED),
    )
    return [suite.run_masked_experiment(context, scenario) for scenario in scenarios]


if __name__ == "__main__":
    experiment_results = run_experiment()
    if SHOW_PLOTS:
        suite.plot_results(
            experiment_results,
            title="Top, bottom, and random observed edges",
        )
