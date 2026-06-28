"""Эксперимент 6: top-10% потоков и начальная OD-матрица с ошибкой 10%.

Начальная матрица строится как ``D_true * (1 + noise)``, проецируется на точные
строчные и столбцовые маргиналии и калибруется до относительной L1-ошибки 0.1.
В целевой функции наблюдаются только 10% рёбер с наибольшими референсными
потоками.

Оптимизатор, энтропийное слагаемое, жёсткая IPF-проекция на маргиналии и все
остальные гиперпараметры совпадают с ``examples/exp1.py``. Для устойчивости
этого запуска используется отдельный уменьшенный шаг Adam. Метрики те же.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from examples.experiments import experiment_suite as suite


OBSERVED_FRACTION = 0.10
INITIAL_RELATIVE_L1 = 0.10
OUTER_ITERS = 2000
LEARNING_RATE = 0.05
SHOW_PLOTS = True


def build_initial_od(D_true, row_target, col_target, rng):
    return suite.make_multiplicative_noisy_initial_od(
        D_true,
        row_target,
        col_target,
        rng,
        target_relative_l1=INITIAL_RELATIVE_L1,
    )


def run_experiment() -> list[dict[str, object]]:
    context = suite.setup_experiment(initial_od_builder=build_initial_od)
    scenario = suite.build_top_flow_scenarios(
        context.reference_flow,
        (OBSERVED_FRACTION,),
    )[0]
    return [
        suite.run_masked_experiment(
            context,
            scenario,
            outer_iters=OUTER_ITERS,
            learning_rate_base=LEARNING_RATE,
        )
    ]


if __name__ == "__main__":
    experiment_results = run_experiment()
    if SHOW_PLOTS:
        suite.plot_results(
            experiment_results,
            title=(
                f"Top {OBSERVED_FRACTION:.0%} edges with "
                f"{INITIAL_RELATIVE_L1:.0%}-error initial OD"
            ),
        )
