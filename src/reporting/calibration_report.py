"""T23, G33 and G34 -- calibration and confidence (Phase 78).

Three artifacts from one frame each, assembled by
:mod:`src.evaluation.calibration_analysis`.

**T23** is the per-model summary: Brier and ECE as mean +/- SD over that run's
folds, beside the two numbers that say *which direction* the miscalibration runs
-- mean confidence and the accuracy of the argmax prediction. ECE is an absolute
value and throws the sign away; a screening prototype that is systematically
over-confident and one that is under-confident are different problems, so the
signed gap travels with it.

**G33** is the confidence histogram: where a model's confidence actually sits,
split into the predictions it got right and the ones it got wrong. A model whose
errors cluster at high confidence is the dangerous case, and no scalar metric
shows it.

**G34** is the reliability diagram, one panel per run. The dashed diagonal is
perfect calibration. The horizontal floor is 1/n_classes -- the confidence of a
uniform probability vector, below which nothing can fall -- drawn because a
4-class panel whose curve starts at 0.25 is not "badly calibrated at low
confidence", it is at the bottom of its possible range.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.reporting.graphs import Graph, GraphSpec, subplots
from src.reporting.plot_style import class_color
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "HEADLINE_RUN",
    "build_t23",
    "build_g33",
    "build_g34",
]

log = get_logger("reporting.calibration")

#: The run G33 draws in detail. EXP-A2 is the headline binary run -- nested
#: search, 25 folds -- and is the one every binary table reports.
HEADLINE_RUN = "EXP-A2"

_DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support "
    "prototype, not a diagnostic tool. A calibrated probability is not a "
    "clinical risk."
)


def _svm_note(comparison: Any, record: dict[str, Any]) -> str:
    """T78.4's finding, written from the comparison frame rather than by hand."""
    if comparison is None or not len(comparison):
        return (
            "The sigmoid-versus-isotonic comparison for the SVM did not run; see "
            "outputs/missing_outputs_report.txt."
        )
    n_folds = int(comparison["fold_label"].nunique())
    means = comparison.groupby("method")[["brier", "ece"]].mean()
    parts = [
        str(method)
        + " brier "
        + format(float(row["brier"]), ".4f")
        + " / ECE "
        + format(float(row["ece"]), ".4f")
        for method, row in means.iterrows()
    ]
    wins = record.get("brier_fold_wins") or {}
    win_text = ", ".join(str(k) + " " + str(v) for k, v in wins.items())
    return (
        "T78.4 -- the SVM's calibrator, compared on the IDENTICAL "
        + str(n_folds)
        + "-fold map with everything else held fixed (n="
        + str(n_folds)
        + "): "
        + "; ".join(parts)
        + ". Lower is better for both. Fold wins on Brier: "
        + win_text
        + ". The configured method is "
        + str(record.get("svm_calibration_method"))
        + ". THE FINAL BINARY MODEL IS "
        + str(record.get("final_model_id"))
        + ", a logistic regression that contains no SVM and no post-hoc "
        "calibrator, so this comparison is evidence about the M6/M7 ensemble "
        "members and not about the deployed estimator."
    )


def build_t23(
    summary: Any,
    comparison: Any,
    record: dict[str, Any],
    sources: tuple[str, ...],
    command: str = "",
) -> Table:
    """T23 -- calibration and confidence summary, one row per (run, model)."""
    frame = summary.copy()
    frame = frame.sort_values(["run", "model_id"]).reset_index(drop=True)

    columns = (
        Column("run", "Run"),
        Column("task", "Task"),
        Column("model_id", "Model"),
        Column("n_bins", "ECE bins", kind="integer"),
        Column("n_folds", "Folds", kind="count"),
        Column("n_records_total", "Scored rows", kind="count"),
        Column("brier_mean", "Brier mean", kind="metric"),
        Column("brier_sd", "Brier SD", kind="metric"),
        Column("ece_mean", "ECE mean", kind="metric"),
        Column("ece_sd", "ECE SD", kind="metric"),
        Column("mean_confidence_mean", "Confidence mean", kind="metric"),
        Column("argmax_accuracy_mean", "Argmax accuracy", kind="metric"),
        Column("confidence_gap_mean", "Confidence - accuracy", kind="metric"),
        Column("confidence_gap_sd", "Gap SD", kind="metric"),
    )

    spec = TableSpec(
        table_id="T23",
        title="Calibration and Confidence Summary",
        caption=(
            "Brier score and expected calibration error for every model on every "
            "task, computed inside each fold from the stored out-of-fold "
            "probabilities and averaged over folds. ECE is measured over the "
            "confidence of the predicted class in equal-width bins, the same "
            "definition src/evaluation/metrics.py uses, so it is defined "
            "identically for the binary and the multiclass tracks. 'Confidence - "
            "accuracy' is signed: positive means the model claims more certainty "
            "than its own hit rate justifies."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-A1, EXP-A2, EXP-B1, EXP-B2, EXP-C1, EXP-C2, EXP-D1",
        objective="O3 (model comparison), O4 (reliability)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        notes=(
            "ECE IS NOT BIN-COUNT INVARIANT. The reported value uses the bin "
            "count in the 'ECE bins' column; "
            "calibration_bin_sensitivity.csv carries the same predictions "
            "re-binned at 5, 10, 15 and 20 bins. An ECE quoted without its bin "
            "count is half a number.",
            "BRIER IS NOT COMPARABLE ACROSS LABEL SPACES. The multiclass form "
            "sums squared error over all classes, so its range is 0-2 where the "
            "binary form's is 0-1. Compare models within a run, never a PASCAL A "
            "Brier against a PhysioNet one.",
            "Reliability is measured against the ARGMAX of each model's own "
            "probability vector, not against its stored y_pred. M6 and M7 decide "
            "by an in-fold selected threshold rather than by argmax, so the two "
            "legitimately differ; 'Argmax accuracy' is therefore not the accuracy "
            "those two models' tables report.",
            "EXP-D1 is ONE external evaluation, not cross-validation: n_folds=1 "
            "and every SD is undefined. It is adult-to-paediatric transfer and "
            "the population mismatch dominates it.",
            _svm_note(comparison, record),
            _DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)


def _focus_model(models: list[str]) -> str:
    """The deployed model where the manifest names one that ran in this run.

    Falls back to the first model id rather than raising: a figure is not the
    place to discover that the final model has not been selected yet, and the
    panel title prints whichever it used.
    """
    import json

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "models_saved" / "binary" / "final" / "manifest.json"
    if path.is_file():
        selected = str(json.loads(path.read_text(encoding="utf-8")).get("selected_model_id", ""))
        if selected in models:
            return selected
    return models[0]


def build_g33(histogram: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G33 -- confidence histogram, right answers against wrong ones."""

    def draw(data: Any) -> Any:
        headline = data[data["run"] == HEADLINE_RUN]
        if not len(headline):  # pragma: no cover - guarded by the caller
            headline = data

        fig, axes = subplots((10.0, 3.6), ncols=2, nrows=1)
        left, right = np.atleast_1d(axes)[0], np.atleast_1d(axes)[1]

        models = sorted(set(headline["model_id"]))
        centres_by_model = {}
        for model in models:
            block = headline[headline["model_id"] == model].sort_values("bin")
            centres_by_model[model] = (
                (block["bin_low"].to_numpy() + block["bin_high"].to_numpy()) / 2.0,
                block,
            )

        # The x-axis starts where predictions actually start. A binary model's
        # confidence cannot fall below 0.5, and half an empty axis makes the
        # occupied half half as legible.
        occupied = headline[headline["n_predictions"] > 0]
        low = float(occupied["bin_low"].min()) if len(occupied) else 0.0

        focus = _focus_model(models)
        centres, block = centres_by_model[focus]
        width = float(block["bin_high"].iloc[0] - block["bin_low"].iloc[0]) * 0.9
        correct = block["n_correct"].to_numpy(dtype=float)
        incorrect = block["n_incorrect"].to_numpy(dtype=float)
        left.bar(
            centres,
            correct,
            width=width,
            color=class_color(0),
            edgecolor="black",
            linewidth=0.3,
            label="correct",
        )
        left.bar(
            centres,
            incorrect,
            width=width,
            bottom=correct,
            color=class_color(3),
            edgecolor="black",
            linewidth=0.3,
            label="incorrect",
        )
        left.set_xlabel("confidence of the predicted class")
        left.set_ylabel("predictions (pooled over folds)")
        left.set_xlim(low, 1.0)
        left.set_title(
            HEADLINE_RUN + " " + focus + " (deployed model): where the errors sit", fontsize=8
        )
        left.legend(fontsize=6)

        for index, model in enumerate(models):
            centres, block = centres_by_model[model]
            right.step(
                centres,
                block["share_of_predictions"].to_numpy(dtype=float),
                where="mid",
                linewidth=1.1,
                color=class_color(index),
                label=model,
            )
        right.set_xlabel("confidence of the predicted class")
        right.set_ylabel("share of predictions")
        right.set_xlim(low, 1.0)
        right.set_title(HEADLINE_RUN + ": confidence distribution per model", fontsize=8)
        right.legend(fontsize=6, ncol=2)

        fig.suptitle("Confidence distribution of the predicted class")
        fig.text(
            0.5,
            0.90,
            "Counts are pooled over the 25 repeated-CV folds, so each record "
            "contributes once per repeat. The shape is the point, not the height.",
            fontsize=6,
            ha="center",
        )
        return fig

    spec = GraphSpec(
        figure_id="G33",
        title="Confidence Histogram",
        caption=(
            "Left: every out-of-fold prediction of the deployed model on the "
            + HEADLINE_RUN
            + " run, binned by the confidence of its predicted class and stacked "
            "into the ones it got right and the ones it got wrong. Errors at high "
            "confidence are the failure mode no scalar metric shows. Right: the "
            "same distribution as a share, for every model of that run. Counts "
            "are pooled over the 25 folds of the repeated 5x5 map, so each record "
            "contributes once per repeat -- the shape is comparable, the absolute "
            "height is five times the corpus."
        ),
        sources=sources,
        exp_id=HEADLINE_RUN,
        objective="O3 (model comparison), O4 (reliability)",
        dataset="D1 PhysioNet 2016",
        command=command,
        size="wide",
    )
    return Graph(spec=spec, frame=histogram, draw=draw)


def build_g34(reliability: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G34 -- reliability diagrams, one panel per run, all its models overlaid."""

    def draw(data: Any) -> Any:
        runs = sorted(set(data["run"]))
        ncols = 4
        nrows = int(np.ceil(len(runs) / ncols))
        fig, axes = subplots((10.0, 2.7 * nrows), ncols=ncols, nrows=nrows, squeeze=False)

        for position, run in enumerate(runs):
            axis = axes[position // ncols][position % ncols]
            block = data[data["run"] == run]
            n_classes = int(block["n_classes"].iloc[0])
            axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.7)
            floor = 1.0 / n_classes
            axis.axvline(floor, color="grey", linewidth=0.5, linestyle=":")
            for index, model in enumerate(sorted(set(block["model_id"]))):
                part = block[block["model_id"] == model].sort_values("bin")
                axis.plot(
                    part["confidence_mean"].to_numpy(dtype=float),
                    part["accuracy_mean"].to_numpy(dtype=float),
                    marker="o",
                    markersize=2.5,
                    linewidth=1.0,
                    color=class_color(index),
                    label=model,
                )
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_title(run + " (" + str(n_classes) + "-class)", fontsize=7)
            axis.tick_params(labelsize=6)
            axis.legend(fontsize=5, ncol=2, loc="upper left")
            if position % ncols == 0:
                axis.set_ylabel("observed accuracy", fontsize=7)
            if position // ncols == nrows - 1:
                axis.set_xlabel("mean confidence", fontsize=7)

        for position in range(len(runs), nrows * ncols):
            axes[position // ncols][position % ncols].axis("off")

        fig.suptitle("Reliability: confidence of the predicted class against its accuracy")
        fig.subplots_adjust(top=0.86)
        fig.text(
            0.5,
            0.925,
            "Dashed diagonal is perfect calibration; a curve below it is "
            "over-confident. The dotted vertical is 1/n_classes, the confidence "
            "of a uniform vector -- nothing can fall to its left.",
            fontsize=6,
            ha="center",
        )
        return fig

    spec = GraphSpec(
        figure_id="G34",
        title="Calibration Curve",
        caption=(
            "Reliability diagrams for every run that holds out-of-fold "
            "probabilities, one panel per run and one line per model. The x-axis "
            "is the confidence of the predicted class, the y-axis the accuracy of "
            "those predictions, binned in equal-width bins inside each fold and "
            "averaged over the folds that had records in the bin. A curve below "
            "the dashed diagonal is over-confident. The dotted vertical marks "
            "1/n_classes, the confidence of a uniform probability vector: no "
            "point can lie to its left, so a 4-class panel starting at 0.25 is at "
            "the bottom of its possible range rather than badly calibrated. This "
            "is NOT the positive-class reliability curve the dashboard shows -- "
            "that one is defined only for the binary task."
        ),
        sources=sources,
        exp_id="EXP-A1, EXP-A2, EXP-B1, EXP-B2, EXP-C1, EXP-C2, EXP-D1",
        objective="O3 (model comparison), O4 (reliability)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        command=command,
        size="tall",
    )
    return Graph(spec=spec, frame=reliability, draw=draw)
