"""Эксперимент 5: вариация доли наблюдаемых рёбер с наибольшими потоками.

Наблюдаемые наборы содержат top-50%, top-70% и top-90% рёбер по величине
референсного равновесного потока и являются вложенными. Только выбранные рёбра
входят в невязку потоков и её градиент.

Оптимизатор, энтропийное слагаемое, жёсткая IPF-проекция на маргиналии и все
гиперпараметры совпадают с ``examples/exp1.py``. Метрики такие же.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from examples.experiments import experiment_suite as suite


OBSERVED_FRACTIONS = (0.50, 0.70, 0.90)
SHOW_PLOTS = True


def run_experiment() -> list[dict[str, object]]:
    context = suite.setup_experiment()
    scenarios = suite.build_top_flow_scenarios(
        context.reference_flow,
        OBSERVED_FRACTIONS,
    )
    return [suite.run_masked_experiment(context, scenario) for scenario in scenarios]


if __name__ == "__main__":
    experiment_results = run_experiment()
    if SHOW_PLOTS:
        suite.plot_results(
            experiment_results,
            title="Largest-flow observed edges",
        )
