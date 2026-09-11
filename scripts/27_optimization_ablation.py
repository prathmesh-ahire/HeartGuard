"""EXP-F2 -- the optimization ablation (Phase 75). Emits T19, G23 and G24.

Reads the per-fold metrics EXP-A1 and EXP-A2 already wrote and assembles the
five optimization stages, each as an incremental delta over the one before it.
**No model is fitted here.** The comparison's whole claim is "identical folds",
and the only way to guarantee that is to use the runs that produced them.

    python scripts/27_optimization_ablation.py
    python scripts/27_optimization_ablation.py --out-dir outputs/09_ablation/_scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/27_optimization_ablation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("optimization_ablation")

COMMAND = "python scripts/27_optimization_ablation.py"
SECTION = "outputs/09_ablation"

SOURCES = (
    "outputs/06_binary_results/EXP-A1/per_fold_metrics.csv",
    "outputs/06_binary_results/EXP-A2/per_fold_metrics.csv",
    "outputs/05_search_optimization/SO-05/weight_stability.csv",
    "outputs/01_dataset_audit/subject_split_map.csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="27_optimization_ablation")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def build_g23(weights, command: str = ""):
    """G23 -- what each weight-optimization method actually chose."""
    import numpy as np

    from src.reporting.graphs import Graph, GraphSpec, subplots
    from src.reporting.plot_style import class_color

    def draw(data):
        methods = list(dict.fromkeys(data["method"]))
        members = list(dict.fromkeys(data["member"]))
        fig, axes = subplots("double", ncols=1, nrows=1)
        ax = np.atleast_1d(axes)[0]

        width = 0.8 / max(len(members), 1)
        for index, member in enumerate(members):
            block = data[data["member"] == member].set_index("method")
            values = [float(block.loc[m, "mean_weight"]) for m in methods]
            errors = [float(block.loc[m, "std_weight"]) for m in methods]
            positions = np.arange(len(methods)) + index * width - 0.4 + width / 2
            ax.bar(
                positions,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                color=class_color(index),
                edgecolor="black",
                linewidth=0.4,
                label=member,
                error_kw={"elinewidth": 0.6},
            )

        # Equal weighting is the thing every method is being compared against,
        # so it is drawn, not left for the reader to infer from three bars that
        # happen to sit at the same height.
        ax.axhline(
            1.0 / max(len(members), 1),
            color="black",
            linestyle="--",
            linewidth=0.8,
            label="equal weight",
        )
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel("Ensemble weight (mean over folds)")
        ax.set_xlabel("Weight optimization method")
        ax.legend(frameon=False, ncol=2, fontsize=6)
        return fig

    spec = GraphSpec(
        figure_id="G23",
        title="Ensemble Weight Comparison",
        caption=(
            "Mean ensemble weight per member under each weight-optimization "
            "method, with the fold-to-fold SD as the error bar, over the 25 "
            "outer folds. The dashed line is equal weighting. Three of the four "
            "methods land on equal weights or within a few percent of them; only "
            "the log-loss objective moves substantially, and it scores lower "
            "than the methods that do not."
        ),
        sources=SOURCES,
        exp_id="EXP-F2",
        objective="O3 (optimization)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=(
            "A weight search that returns equal weights is a result, not a "
            "failure: it says the three members are close enough in quality "
            "that the data does not support preferring one.",
            "Weights are fitted inside each training fold. A weight vector is "
            "never chosen using the outer test fold it is then scored on.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return Graph(spec=spec, frame=weights, draw=draw)


def build_g24(stages, per_fold, command: str = ""):
    """G24 -- the five stages, sensitivity and balanced accuracy side by side."""
    import numpy as np

    from src.reporting.graphs import Graph, GraphSpec, subplots
    from src.reporting.plot_style import class_color

    def draw(data):
        fig, axes = subplots("wide", ncols=2, nrows=1)
        left, right = np.atleast_1d(axes)[0], np.atleast_1d(axes)[1]
        # Single-line labels, rotated. Multi-line tick labels eat the vertical
        # band `annotate_source` reserves for the source stamp, and the stamp
        # then lands on top of them.
        labels = [str(row["stage"]).replace("_", " ") for row in data.to_dict("records")]
        positions = np.arange(len(data))

        for ax, metric, title in (
            (left, "sensitivity", "Sensitivity"),
            (right, "balanced_accuracy", "Balanced accuracy"),
        ):
            values = data[metric].to_numpy(dtype=float)
            errors = data[metric + "_sd"].to_numpy(dtype=float)
            colors = [class_color(i) for i in range(len(values))]
            ax.bar(
                positions,
                values,
                yerr=errors,
                capsize=3,
                color=colors,
                edgecolor="black",
                linewidth=0.4,
                error_kw={"elinewidth": 0.6},
            )
            # The baseline every cumulative delta is measured from.
            ax.axhline(
                float(values[0]), color="black", linestyle="--", linewidth=0.8,
            )
            best = int(np.nanargmax(values))
            ax.annotate(
                "best",
                xy=(positions[best], values[best]),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize="small",
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6)
            ax.set_ylabel(title)
            ax.set_ylim(
                max(0.0, float(np.nanmin(values - errors)) - 0.03),
                min(1.0, float(np.nanmax(values + errors)) + 0.05),
            )
        return fig

    spec = GraphSpec(
        figure_id="G24",
        title="Baseline versus Optimized",
        caption=(
            "The five optimization stages over the identical DA-07 repeated 5x5 "
            "subject-grouped fold map, with the fold SD as the error bar. The "
            "dashed line is the untuned single-model baseline. Read the two "
            "panels together: the stage that maximises sensitivity is not the "
            "stage that maximises balanced accuracy, and neither is the last one."
        ),
        sources=SOURCES,
        exp_id="EXP-F2",
        objective="O3 (optimization), O5 (ablation evidence)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=(
            "Error bars are the SD across 25 folds, not a confidence interval "
            "on the mean, and they overlap between every pair of stages. No "
            "stage in this figure is distinguishable from its neighbour on this "
            "evidence; Phase 81's paired tests are where that question is "
            "settled.",
            "The stages are the optimization pipeline's order, not a ranking.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return Graph(spec=spec, frame=stages, draw=draw)


def main(argv: list[str] | None = None) -> int:
    from src.evaluation import optimization_ablation as oa
    from src.reporting.ablation_report import build_t19
    from src.reporting.graphs import write_graph
    from src.reporting.tables import write_table
    from src.utils.io import save_csv
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    section = Path(args.out_dir) if args.out_dir else Path(SECTION)
    section.mkdir(parents=True, exist_ok=True)

    run = start_run("optimization_ablation")
    try:
        stages, per_fold = oa.build_stages()
        save_csv(stages, section / "optimization_ablation_stages.csv")
        save_csv(per_fold, section / "optimization_ablation_per_fold.csv")

        # A9 and A10 as named, self-contained comparisons (T75.1, T75.2).
        a9 = oa.pairwise_comparison(per_fold, "tuned_single", "tuned_ensemble_equal")
        a10 = oa.pairwise_comparison(
            per_fold, "tuned_ensemble_equal", "tuned_ensemble_optimized"
        )
        a9_default = oa.pairwise_comparison(per_fold, "default_single", "default_ensemble")
        save_csv(a9, section / "A9_individual_vs_ensemble_tuned.csv")
        save_csv(a9_default, section / "A9_individual_vs_ensemble_default.csv")
        save_csv(a10, section / "A10_equal_vs_optimized_weights.csv")

        table = build_t19(stages, sources=SOURCES, command=COMMAND)
        written = write_table(table, section)
        log.info("T19 -> %s", written["csv"].name)

        # G23/G24 go to the figures directory with every other G-figure, not
        # beside the ablation CSVs: a second figure_registry.csv in 09_ablation
        # numbered them 1 and 2 in a series of their own (fixed Phase 93).
        weights = oa.ensemble_weight_frame()
        for graph in (build_g23(weights, COMMAND), build_g24(stages, per_fold, COMMAND)):
            paths = write_graph(graph, args.out_dir)
            log.info("%s -> %s", graph.spec.figure_id, paths["png"].name)

        for row in stages.to_dict("records"):
            log.info(
                "%d %-26s %-7s sensitivity %.4f  incr %+.4f  cum %+.4f",
                row["stage_order"],
                row["stage"],
                row["model_id"],
                row["sensitivity"],
                row["sensitivity_incremental"],
                row["sensitivity_cumulative"],
            )
        run.set("optimization_stages", len(stages))
    except BaseException:
        run.finish("failed")
        raise

    run.finish("ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
