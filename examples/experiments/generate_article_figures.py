"""Reproduce the numerical results and convergence figures used in the article.

The script runs only experiments 3--5.  Experiment 6 is intentionally absent.
For a full reproduction from the repository root run

    PYTHONPATH=src:. python3 examples/experiments/generate_article_figures.py

The run writes compact numerical histories to ``article/results``, CSV summary
tables to the same directory, and PDF/EPS/PNG figures to ``article/figures``.
Once the result archives exist, figures can be rebuilt without rerunning the
optimizer by adding ``--mode plot``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))

from examples import exp1
from examples.experiments import exp3_random_observed_edges as exp3
from examples.experiments import exp4_edge_selection as exp4
from examples.experiments import exp5_top_flow_fractions as exp5
from examples.experiments import experiment_suite as suite


ARTICLE_DIR = ROOT_DIR / "article"
DEFAULT_RESULTS_DIR = ARTICLE_DIR / "results"
DEFAULT_FIGURES_DIR = ARTICLE_DIR / "figures"
ROBUST_METRIC_EVERY = 25
PRINT_EVERY = 500

FIGURE_FILENAMES = {
    "exp3": "exp3_random_fraction_convergence",
    "exp4": "exp4_selection_convergence",
    "exp5": "exp5_top_fraction_convergence",
}

COLORS = ("#000000", "#0072B2", "#D55E00", "#009E73", "#7E57C2")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "P")


def build_scenarios(experiment: str, context: suite.ExperimentContext):
    if experiment == "exp3":
        return suite.build_nested_random_scenarios(
            context.reference_flow.size,
            exp3.OBSERVED_FRACTIONS,
            np.random.default_rng(exp3.MASK_SEED),
        )
    if experiment == "exp4":
        return suite.build_flow_rank_scenarios(
            context.reference_flow,
            exp4.OBSERVED_FRACTION,
            np.random.default_rng(exp4.MASK_SEED),
        )
    if experiment == "exp5":
        return suite.build_top_flow_scenarios(
            context.reference_flow,
            exp5.OBSERVED_FRACTIONS,
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def run_experiment(experiment: str):
    context = suite.setup_experiment()
    scenarios = build_scenarios(experiment, context)
    results = [
        suite.run_masked_experiment(
            context,
            scenario,
            robust_metric_every=ROBUST_METRIC_EVERY,
            print_every=PRINT_EVERY,
        )
        for scenario in scenarios
    ]
    return context, results


def result_archive_path(results_dir: Path, experiment: str) -> Path:
    return results_dir / f"{experiment}_article_results.npz"


def save_results(
    results_dir: Path,
    experiment: str,
    context: suite.ExperimentContext,
    results: list[dict[str, object]],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": experiment,
        "seed": exp1.SEED,
        "mask_seed": (
            exp3.MASK_SEED
            if experiment == "exp3"
            else exp4.MASK_SEED if experiment == "exp4" else None
        ),
        "outer_iterations": exp1.OUTER_ITERS,
        "learning_rate": exp1.LEARNING_RATE,
        "learning_rate_decay": exp1.LEARNING_RATE_DECAY,
        "beta1": exp1.BETA1,
        "beta2": exp1.BETA2,
        "adam_epsilon": exp1.ADAM_EPS,
        "entropy_weight": exp1.GAMMA,
        "fw_reference_iterations": exp1.FW_REFERENCE_ITERS,
        "fw_inner_iterations": exp1.FW_INNER_ITERS,
        "fw_stopping_tolerance": exp1.FW_RGAP,
        "ipf_iterations": exp1.IPF_PROJECT_ITERS,
        "ipf_tolerance": exp1.IPF_PROJECT_TOL,
        "deleted_edge_iterations": exp1.ROBUST_METRIC_FW_ITERS,
        "deleted_edge_evaluation_period": ROBUST_METRIC_EVERY,
        "deleted_edge_ids_zero_based": [
            int(edge_id) for edge_id, _, _ in context.robust_scenarios
        ],
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "names": np.asarray([str(result["scenario_name"]) for result in results]),
        "D_true": np.asarray(context.D_true, dtype=np.float64),
        "D_initial": np.asarray(context.D_initial, dtype=np.float64),
        "reference_flow": np.asarray(context.reference_flow, dtype=np.float64),
    }
    for index, result in enumerate(results):
        arrays[f"mask_{index}"] = np.asarray(result["observation_mask"], dtype=np.float64)
        arrays[f"D_recovered_{index}"] = np.asarray(
            result["D_recovered"], dtype=np.float64
        )
        arrays[f"D_final_{index}"] = np.asarray(result["D_final"], dtype=np.float64)
        arrays[f"flow_error_{index}"] = np.asarray(
            result["flow_error_history"], dtype=np.float64
        )
        arrays[f"deleted_iterations_{index}"] = np.asarray(
            result["deleted_edge_iterations"], dtype=np.int64
        )
        arrays[f"deleted_error_{index}"] = np.asarray(
            result["deleted_edge_error_history"], dtype=np.float64
        )
        arrays[f"masked_mismatch_{index}"] = np.asarray(
            result["masked_mismatch_history"], dtype=np.float64
        )
        arrays[f"objective_{index}"] = np.asarray(
            result["objective_history"], dtype=np.float64
        )
        arrays[f"relative_l1_{index}"] = np.asarray(
            result["rel_l1_history"], dtype=np.float64
        )
        arrays[f"row_error_{index}"] = np.asarray(
            result["row_error_history"], dtype=np.float64
        )
        arrays[f"column_error_{index}"] = np.asarray(
            result["col_error_history"], dtype=np.float64
        )
        arrays[f"best_iteration_{index}"] = np.asarray(result["best_iteration"])
        arrays[f"best_deleted_error_{index}"] = np.asarray(
            result["best_deleted_edge_error"]
        )

    np.savez_compressed(result_archive_path(results_dir, experiment), **arrays)
    write_summary_csv(results_dir, experiment, results)


def load_results(results_dir: Path, experiment: str) -> list[dict[str, object]]:
    with np.load(result_archive_path(results_dir, experiment), allow_pickle=False) as archive:
        names = archive["names"]
        results: list[dict[str, object]] = []
        for index, name in enumerate(names):
            results.append(
                {
                    "scenario_name": str(name),
                    "observation_mask": archive[f"mask_{index}"].copy(),
                    "D_recovered": archive[f"D_recovered_{index}"].copy(),
                    "D_final": archive[f"D_final_{index}"].copy(),
                    "flow_error_history": archive[f"flow_error_{index}"].copy(),
                    "deleted_edge_iterations": archive[
                        f"deleted_iterations_{index}"
                    ].copy(),
                    "deleted_edge_error_history": archive[
                        f"deleted_error_{index}"
                    ].copy(),
                    "masked_mismatch_history": archive[
                        f"masked_mismatch_{index}"
                    ].copy(),
                    "objective_history": archive[f"objective_{index}"].copy(),
                    "rel_l1_history": archive[f"relative_l1_{index}"].copy(),
                    "row_error_history": archive[f"row_error_{index}"].copy(),
                    "col_error_history": archive[f"column_error_{index}"].copy(),
                    "best_iteration": int(archive[f"best_iteration_{index}"]),
                    "best_deleted_edge_error": float(
                        archive[f"best_deleted_error_{index}"]
                    ),
                }
            )
    return results


def summary_rows(experiment: str, results: list[dict[str, object]]):
    for result in results:
        flow_error = np.asarray(result["flow_error_history"], dtype=np.float64)
        deleted_error = np.asarray(
            result["deleted_edge_error_history"], dtype=np.float64
        )
        mismatch = np.asarray(result["masked_mismatch_history"], dtype=np.float64)
        relative_l1 = np.asarray(result["rel_l1_history"], dtype=np.float64)
        row_error = np.asarray(result["row_error_history"], dtype=np.float64)
        column_error = np.asarray(result["col_error_history"], dtype=np.float64)
        best_iteration = int(result["best_iteration"])
        mask = np.asarray(result["observation_mask"], dtype=np.float64)
        yield {
            "experiment": experiment,
            "scenario": str(result["scenario_name"]),
            "observed_edges": int(mask.sum()),
            "observed_fraction": float(mask.mean()),
            "best_objective_iteration": best_iteration,
            "initial_full_flow_error": float(flow_error[0]),
            "best_objective_full_flow_error": float(flow_error[best_iteration]),
            "best_objective_deleted_edge_error": float(
                result["best_deleted_edge_error"]
            ),
            "final_full_flow_error": float(flow_error[-1]),
            "final_deleted_edge_error": float(deleted_error[-1]),
            "minimum_full_flow_error_iteration": int(np.argmin(flow_error)),
            "minimum_full_flow_error": float(np.min(flow_error)),
            "initial_masked_mismatch": float(mismatch[0]),
            "best_objective_masked_mismatch": float(mismatch[best_iteration]),
            "final_masked_mismatch": float(mismatch[-1]),
            "best_objective_relative_l1": float(relative_l1[best_iteration]),
            "final_relative_row_marginal_error": float(row_error[-1]),
            "final_relative_column_marginal_error": float(column_error[-1]),
        }


def write_summary_csv(
    results_dir: Path,
    experiment: str,
    results: list[dict[str, object]],
) -> None:
    rows = list(summary_rows(experiment, results))
    path = results_dir / f"{experiment}_article_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def russian_label(experiment: str, name: str, observed_edges: int) -> str:
    if experiment in {"exp3", "exp5"}:
        percentage = int(name.rsplit(" ", maxsplit=1)[-1].removesuffix("%"))
        return f"{percentage} % ({observed_edges} рёбер)"
    return {
        "top 50%": "наибольшие потоки",
        "bottom 50%": "наименьшие потоки",
        "random 50%": "случайный выбор",
    }[name]


def plot_results(
    figures_dir: Path,
    experiment: str,
    results: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (full_axis, deleted_axis) = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.75),
        constrained_layout=True,
    )

    for index, result in enumerate(results):
        flow_error = np.asarray(result["flow_error_history"], dtype=np.float64)
        deleted_iterations = np.asarray(
            result["deleted_edge_iterations"], dtype=np.int64
        )
        deleted_error = np.asarray(
            result["deleted_edge_error_history"], dtype=np.float64
        )
        iterations = np.arange(flow_error.size)
        displayed_iterations = iterations[::ROBUST_METRIC_EVERY]
        displayed_flow_error = flow_error[::ROBUST_METRIC_EVERY]
        best_iteration = int(result["best_iteration"])
        observed_edges = int(np.asarray(result["observation_mask"]).sum())
        label = russian_label(
            experiment,
            str(result["scenario_name"]),
            observed_edges,
        )
        style = {
            "color": COLORS[index],
            "linestyle": LINESTYLES[index],
            "marker": MARKERS[index],
            "linewidth": 1.35,
            "markersize": 2.8,
            "label": label,
        }
        full_axis.plot(
            displayed_iterations,
            np.maximum(displayed_flow_error, 1e-300),
            markevery=max(1, 500 // ROBUST_METRIC_EVERY),
            **style,
        )
        deleted_axis.plot(
            deleted_iterations,
            np.maximum(deleted_error, 1e-300),
            markevery=max(1, 500 // ROBUST_METRIC_EVERY),
            **style,
        )
        full_axis.scatter(
            [best_iteration],
            [flow_error[best_iteration]],
            s=36,
            marker=MARKERS[index],
            facecolor=COLORS[index],
            edgecolor="white" if index == 0 else "black",
            linewidth=0.7,
            zorder=5,
        )
        deleted_axis.scatter(
            [best_iteration],
            [float(result["best_deleted_edge_error"])],
            s=36,
            marker=MARKERS[index],
            facecolor=COLORS[index],
            edgecolor="white" if index == 0 else "black",
            linewidth=0.7,
            zorder=5,
        )

    full_axis.set_title(r"(а) Ошибка полного потока $E_{\rm full}$")
    deleted_axis.set_title(r"(б) Ошибка при удалении ребра $E_{\rm del}$")
    for axis in (full_axis, deleted_axis):
        axis.set_yscale("log")
        axis.set_xlabel("Внешняя итерация")
        axis.set_ylabel("Относительная ошибка")
        axis.set_xlim(0, exp1.OUTER_ITERS)
        axis.grid(True, which="major", color="#c7c7c7", linewidth=0.55)
        axis.legend(framealpha=1.0, fancybox=False)

    output_base = figures_dir / FIGURE_FILENAMES[experiment]
    save_options = {"bbox_inches": "tight", "facecolor": "white"}
    figure.savefig(output_base.with_suffix(".pdf"), **save_options)
    figure.savefig(output_base.with_suffix(".eps"), **save_options)
    figure.savefig(output_base.with_suffix(".png"), dpi=300, **save_options)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("all", "exp3", "exp4", "exp5"),
        default="all",
    )
    parser.add_argument("--mode", choices=("run", "plot"), default="run")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = (
        ("exp3", "exp4", "exp5")
        if args.experiment == "all"
        else (args.experiment,)
    )
    for experiment in experiments:
        if args.mode == "run":
            context, results = run_experiment(experiment)
            save_results(args.results_dir, experiment, context, results)
        else:
            results = load_results(args.results_dir, experiment)
        plot_results(args.figures_dir, experiment, results)


if __name__ == "__main__":
    main()
