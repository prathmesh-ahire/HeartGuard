"""G11-G17, G20-G22 and G28 -- the result figures (Phases 92-93).

Every figure here is drawn from numbers that already exist on disk. Nothing is
refitted, and with one exception nothing is recomputed: the frame each figure
plots is a re-shaping of a committed results file, written to CSV beside the
PNG by the graph engine (T90.2), so the T92.7/T93.7 gates can reconcile every
plotted value against the file it claims to come from.

===  ==========================================  ================================
G11  Individual Model Comparison                 T08 (mean, SD across folds)
G12  Binary ROC Curve                            EXP-A1/A2 roc_pr_curve_points + T08
G13  Binary Precision Recall Curve               EXP-A1/A2 roc_pr_curve_points
G14  Binary Confusion Matrix                     EXP-A2 confusion_matrices.json
G15  PASCAL A Confusion Matrix                   EXP-B1 confusion_matrices.json + T11
G16  PASCAL B Confusion Matrix                   EXP-B2 confusion_matrices.json + T12
G17  Multiclass One Vs Rest ROC                  EXP-B1/B2 predictions.parquet
G20  Search Convergence Plot                     SO-01/02/03a/03b convergence.csv
G21  All Features Versus Selected Features       SO-04 all_features_vs_selected.csv
G22  F1 And Accuracy Versus Feature Count        SO-04 feature_count_curve.csv
G28  Dataset Wise Generalization                 T08, T11-T14, T16, T-S5
===  ==========================================  ================================

The exception is G17. A one-vs-rest curve needs per-class probabilities, which
exist only in the gitignored ``predictions.parquet``; G17 computes the curves
from it exactly as ``curves.py`` does for the binary task (per fold,
interpolated onto a common grid, averaged -- never pooled), and its written CSV
is what the gate reads. Its per-class AUCs are reconciled against the
experiment's own ``ovr_auc_macro`` so the recomputation is checked against a
second code path.

The AUC in a legend is the mean of the per-fold AUCs
----------------------------------------------------
Not the area under the drawn mean curve. Averaging curves and averaging areas
do not commute, and every table in the project reports the mean of per-fold
areas; a legend that disagreed with T08 in the third decimal would be a second,
unexplained number for the same thing. The gate checks the legend against T08.

Confusion matrices: why the raw panel shows ONE repeat
------------------------------------------------------
The ``total`` matrix in ``confusion_matrices.json`` sums all folds of a repeated
map, so every record is counted once per repeat -- 16,200 predictions for a
3,240-record corpus. Printed as "counts" that reads as a corpus five times its
size. The raw panel therefore sums the folds of repeat 0 only, a complete
partition in which every record appears exactly once, so its row totals are
the class counts in DA-02 and the gate checks exactly that. The normalized
panel uses every fold, where the repeat multiplicity cancels in the ratio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.reporting.data_graphs import _outputs, _read, _relative
from src.reporting.graphs import Graph, GraphSpec, class_color, subplots
from src.utils.logging_setup import get_logger

__all__ = [
    "RESULT_GRAPH_IDS",
    "NEEDS_PARQUET",
    "BINARY_RUNS",
    "MODEL_ORDER",
    "G11_METRICS",
    "G14_COMPARATOR",
    "GRID_POINTS",
    "T08_FILE",
    "T11_FILE",
    "T12_FILE",
    "T13_FILE",
    "T14_FILE",
    "T16_FILE",
    "TS5_FILE",
    "class_order",
    "multiclass_models",
    "build_result_graphs",
]

log = get_logger("reporting.result_graphs")

RESULT_GRAPH_IDS: tuple[str, ...] = (
    "G11",
    "G12",
    "G13",
    "G14",
    "G15",
    "G16",
    "G17",
    "G20",
    "G21",
    "G22",
    "G28",
)

#: G17 reads per-class probabilities from the gitignored predictions.parquet.
NEEDS_PARQUET: tuple[str, ...] = ("G17",)

BINARY_RUNS: tuple[str, ...] = ("EXP-A1", "EXP-A2")

#: One colour per model across every figure in this module, so M1 is the same
#: colour in G11, G12 and G13. Seven models fit the eight-colour palette.
MODEL_ORDER: tuple[str, ...] = ("M1", "M3", "M4", "M5", "M6", "M7", "M8")

#: The four T08 metrics that carry an SD. Rule 6: never accuracy alone.
G11_METRICS: tuple[str, ...] = ("sensitivity", "specificity", "balanced_accuracy", "roc_auc")

METRIC_LABELS = {
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "balanced_accuracy": "balanced accuracy",
    "roc_auc": "ROC-AUC",
    "macro_f1": "macro-F1",
    "accuracy": "accuracy",
}

#: G14 shows the final model beside M6, the equal-weight ensemble it is
#: statistically tied with (p = 0.797 on sensitivity; see note.md). Declared
#: here, not chosen from the numbers.
G14_COMPARATOR = "M6"

#: Same grid as curves.py, so G17's class curves and G12's are comparable.
GRID_POINTS = 101

T08_FILE = "T08_physionet_individual_model_comparison.csv"
T11_FILE = "T11_pascal_a_multiclass_results.csv"
T12_FILE = "T12_pascal_b_multiclass_results.csv"
T13_FILE = "T13_circor_murmur_classification_results.csv"
T14_FILE = "T14_circor_clinical_outcome_results.csv"
T16_FILE = "T16_cross_dataset_generalization.csv"
TS5_FILE = "T-S5_leave_one_source_out_generalization.csv"

DISCLAIMER = "PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic tool."

WITHIN_CORPUS = (
    "All PhysioNet figures are within the PhysioNet 2016 corpus: leave-one-"
    "sub-collection-out (EXP-F3) takes AUC to chance, so none of them is a "
    "claim of generalization."
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, places: int = 3) -> str:
    """T85.6: three decimals for a metric, formatted here and never in a caller."""
    return format(float(value), "." + str(places) + "f")


def _model_color(model_id: str) -> str:
    index = MODEL_ORDER.index(model_id) if model_id in MODEL_ORDER else len(MODEL_ORDER)
    return class_color(index)


def class_order(task: str) -> list[str]:
    """The fixed class order for a task: the code order of its label map."""
    from src.utils.constants import LABEL_MAPS

    mapping = LABEL_MAPS[task]
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


def _t08() -> tuple[Path, Any]:
    source = _outputs("binary_results") / T08_FILE
    return source, _read(source)


def _confusion(path: Path, task: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("source not found: " + str(path))
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected = class_order(task)
    if list(payload.get("class_names") or []) != expected:
        raise ValueError(
            str(path)
            + ": class order "
            + str(payload.get("class_names"))
            + " is not the fixed order "
            + str(expected)
            + " for "
            + task
        )
    if list(payload.get("labels") or []) != list(range(len(expected))):
        raise ValueError(str(path) + ": labels are not 0..n-1 in class order")
    return payload


def multiclass_models(table: Any) -> list[str]:
    """Models a multiclass confusion figure shows, by a rule rather than a pick.

    The best model by macro-F1 (how T11/T12 rank), and the best by accuracy if
    that is a different model. On PASCAL B the two disagree and the accuracy
    winner never predicts `extrastole`; showing only the macro-F1 winner would
    hide the one confusion matrix that makes the accuracy trap visible.
    """
    by_f1 = str(table.loc[table["macro_f1_mean"].astype(float).idxmax(), "model_id"])
    by_accuracy = str(table.loc[table["accuracy_mean"].astype(float).idxmax(), "model_id"])
    return list(dict.fromkeys([by_f1, by_accuracy]))


# ---------------------------------------------------------------------------
# G11 -- individual model comparison (T92.1)
# ---------------------------------------------------------------------------


def build_g11(command: str = "") -> Graph:
    import pandas as pd

    source, t08 = _t08()
    rows = []
    for record in t08.to_dict("records"):
        for metric in G11_METRICS:
            rows.append(
                {
                    "run": record["run"],
                    "tuning": record["tuning"],
                    "rank": int(record["rank"]),
                    "model_id": record["model_id"],
                    "model_name": record["model_name"],
                    "n_folds": int(record["n_folds"]),
                    "metric": metric,
                    "mean": float(record[metric + "_mean"]),
                    "sd": float(record[metric + "_sd"]),
                }
            )
    frame = pd.DataFrame(rows)
    n_folds = int(frame["n_folds"].max())

    def draw(data: Any) -> Any:
        runs = list(dict.fromkeys(data["run"]))
        fig, axes = subplots("wide", ncols=len(runs), sharey=True)
        axes = np.atleast_1d(axes)
        width = 0.8 / len(G11_METRICS)
        for axis, run in zip(axes, runs, strict=True):
            block = data[data["run"] == run]
            models = list(block.sort_values("rank")["model_id"].drop_duplicates())
            positions = np.arange(len(models))
            for index, metric in enumerate(G11_METRICS):
                part = block[block["metric"] == metric].set_index("model_id").loc[models]
                axis.bar(
                    positions + (index - (len(G11_METRICS) - 1) / 2) * width,
                    part["mean"],
                    width,
                    yerr=part["sd"],
                    capsize=1.5,
                    color=class_color(index),
                    label=METRIC_LABELS[metric],
                    error_kw={"elinewidth": 0.6},
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(models)
            axis.set_title(run + " (" + str(block["tuning"].iloc[0]) + ")", fontsize=9)
            axis.set_xlabel("model, in T08 rank order")
            axis.set_ylim(0.0, 1.22)
            axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axes[0].set_ylabel("score (fold mean, SD whiskers)")
        axes[0].legend(fontsize=6.5, ncol=4, loc="upper left", frameon=False)
        return fig

    spec = GraphSpec(
        figure_id="G11",
        title="Individual Model Comparison",
        caption=(
            "Every binary model on the two PhysioNet runs, as the mean over the "
            + str(n_folds)
            + " outer folds of the repeated 5x5 subject-grouped map, with one SD "
            "across folds as the whisker. Models are in T08's rank order -- "
            "sensitivity, then balanced accuracy -- not by accuracy, which on this "
            "majority-normal corpus ranks the models almost in reverse. Whiskers overlap "
            "for most pairs; T28 holds the paired tests. " + WITHIN_CORPUS
        ),
        sources=(_relative(source),),
        exp_id="EXP-A1, EXP-A2",
        objective="O3 (heterogeneous ensemble), O1 (binary screening)",
        dataset="D1 PhysioNet 2016",
        size="wide",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G12 / G13 -- binary ROC and precision-recall (T92.2, T92.3)
# ---------------------------------------------------------------------------


def _binary_curves(kind: str) -> tuple[list[str], Any]:
    import pandas as pd

    source, t08 = _t08()
    sources = [_relative(source)]
    blocks = []
    for run in BINARY_RUNS:
        run_dir = _outputs("binary_results") / run
        curve_path = run_dir / "roc_pr_curve_points.csv"
        aggregate_path = run_dir / "aggregate_metrics.csv"
        curves = _read(curve_path)
        aggregate = _read(aggregate_path).set_index("model_id")
        sources.append(_relative(curve_path))
        if kind == "pr":
            sources.append(_relative(aggregate_path))
        ranked = t08[t08["run"] == run].sort_values("rank")
        for record in ranked.to_dict("records"):
            model_id = str(record["model_id"])
            part = curves[(curves["model_id"] == model_id) & (curves["kind"] == kind)]
            if part.empty:
                raise ValueError(run + " has no " + kind + " curve for " + model_id)
            part = part.sort_values("x")[["x", "mean", "sd", "n_folds"]].copy()
            part.insert(0, "model_name", record["model_name"])
            part.insert(0, "model_id", model_id)
            part.insert(0, "rank", int(record["rank"]))
            part.insert(0, "run", run)
            if kind == "roc":
                auc, sd = float(record["roc_auc_mean"]), float(record["roc_auc_sd"])
                part["roc_auc_mean"] = auc
                part["roc_auc_sd"] = sd
                part["legend"] = model_id + " " + str(record["model_name"]) + "  AUC " + _fmt(auc)
            else:
                row = aggregate.loc[model_id]
                auc, sd = float(row["pr_auc_mean"]), float(row["pr_auc_sd"])
                prevalence = float(row["n_positive_mean"]) / float(row["support_mean"])
                part["pr_auc_mean"] = auc
                part["pr_auc_sd"] = sd
                part["prevalence"] = prevalence
                part["legend"] = model_id + " " + str(record["model_name"]) + "  AP " + _fmt(auc)
            blocks.append(part)
    return sources, pd.concat(blocks, ignore_index=True)


def _draw_curves(data: Any, kind: str) -> Any:
    runs = list(dict.fromkeys(data["run"]))
    fig, axes = subplots("wide", ncols=len(runs), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, run in zip(axes, runs, strict=True):
        block = data[data["run"] == run]
        for model_id in block.sort_values("rank")["model_id"].drop_duplicates():
            part = block[block["model_id"] == model_id].sort_values("x")
            x = part["x"].to_numpy(float)
            mean = part["mean"].to_numpy(float)
            style = "--" if model_id == "M7" else "-"
            axis.plot(
                x,
                mean,
                style,
                color=_model_color(str(model_id)),
                linewidth=1.1,
                label=str(part["legend"].iloc[0]),
            )
            if model_id == "M1":
                sd = part["sd"].to_numpy(float)
                axis.fill_between(
                    x,
                    np.clip(mean - sd, 0, 1),
                    np.clip(mean + sd, 0, 1),
                    color=_model_color("M1"),
                    alpha=0.15,
                    linewidth=0,
                )
        if kind == "roc":
            axis.plot([0, 1], [0, 1], ":", color="#777777", linewidth=0.8)
            axis.set_xlabel("false-positive rate (1 - specificity)")
            loc = "lower right"
        else:
            prevalence = float(block["prevalence"].mean())
            axis.axhline(prevalence, linestyle=":", color="#777777", linewidth=0.8)
            axis.set_xlabel("recall (sensitivity)")
            loc = "lower left"
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.02)
        axis.set_title(run, fontsize=9)
        axis.legend(fontsize=5.8, loc=loc, frameon=False)
    axes[0].set_ylabel("true-positive rate (sensitivity)" if kind == "roc" else "precision")
    return fig


def build_g12(command: str = "") -> Graph:
    sources, frame = _binary_curves("roc")
    n_folds = int(frame["n_folds"].max())

    spec = GraphSpec(
        figure_id="G12",
        title="Binary ROC Curve",
        caption=(
            "Cross-validated ROC curves for every binary model on both PhysioNet "
            "runs. Each curve is computed inside a fold on that fold's held-out "
            "records, interpolated onto a common grid and averaged over "
            + str(n_folds)
            + " folds; folds are never pooled, because the repeated map counts "
            "every record once per repeat. The AUC in the legend is the mean of "
            "the per-fold AUCs -- exactly T08's ROC-AUC column -- and not the "
            "area under the drawn mean curve. The band is one SD across folds "
            "for M1, the final model; M7 is dashed because it lies on M6. " + WITHIN_CORPUS
        ),
        sources=tuple(sources),
        exp_id="EXP-A1, EXP-A2",
        objective="O1 (binary screening)",
        dataset="D1 PhysioNet 2016",
        size="wide",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=lambda data: _draw_curves(data, "roc"))


def build_g13(command: str = "") -> Graph:
    sources, frame = _binary_curves("pr")
    n_folds = int(frame["n_folds"].max())

    spec = GraphSpec(
        figure_id="G13",
        title="Binary Precision Recall Curve",
        caption=(
            "Cross-validated precision-recall curves, built the same way as G12: "
            "per fold, interpolated on recall (precision is not monotone in the "
            "threshold, so the other direction is meaningless), averaged over "
            + str(n_folds)
            + " folds. The legend's AP is the mean per-fold average precision "
            "from each run's aggregate metrics. The dotted line is the abnormal "
            "prevalence, the precision of a model that ranks at random; it is "
            "the right baseline here, not 0.5. " + WITHIN_CORPUS
        ),
        sources=tuple(sources),
        exp_id="EXP-A1, EXP-A2",
        objective="O1 (binary screening)",
        dataset="D1 PhysioNet 2016",
        size="wide",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=lambda data: _draw_curves(data, "pr"))


# ---------------------------------------------------------------------------
# G14 / G15 / G16 -- confusion matrices (T92.4, T92.5)
# ---------------------------------------------------------------------------


def _matrix_rows(
    task: str, run: str, model_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    names = class_order(task)
    entry = payload["models"].get(model_id)
    if entry is None:
        raise KeyError(run + " has no confusion matrix for " + model_id)
    per_fold = entry["per_fold"]
    first_repeat = sorted(label for label in per_fold if label.startswith("r0f"))
    if not first_repeat:
        raise ValueError(run + " " + model_id + ": no repeat-0 folds")
    raw = np.sum([np.asarray(per_fold[label], dtype=np.int64) for label in first_repeat], axis=0)
    total = np.asarray(entry["total"], dtype=np.float64)
    shares = total / total.sum(axis=1, keepdims=True)
    rows = []
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            rows.append(
                {
                    "task": task,
                    "run": run,
                    "model_id": model_id,
                    "true_index": i,
                    "true_class": true_name,
                    "pred_index": j,
                    "pred_class": pred_name,
                    "count_repeat0": int(raw[i, j]),
                    "count_all_folds": int(total[i, j]),
                    "row_share_all_folds": float(shares[i, j]),
                    "n_folds_repeat0": len(first_repeat),
                    "n_folds_all": len(per_fold),
                }
            )
    return rows


def _draw_matrices(data: Any) -> Any:
    models = list(dict.fromkeys(data["model_id"]))
    n_classes = int(data["true_index"].max()) + 1
    height = 2.5 + 0.35 * n_classes
    fig, axes = subplots((7.2, height * len(models)), nrows=len(models), ncols=2, squeeze=False)
    for row, model_id in enumerate(models):
        block = data[data["model_id"] == model_id]
        names = list(block.sort_values("true_index")["true_class"].drop_duplicates())
        for column, (value, label) in enumerate(
            (
                ("count_repeat0", "counts, repeat 0 (each record once)"),
                ("row_share_all_folds", "row-normalized, all folds"),
            )
        ):
            axis = axes[row][column]
            matrix = (
                block.pivot(index="true_index", columns="pred_index", values=value)
                .sort_index()
                .sort_index(axis=1)
                .to_numpy(float)
            )
            scale = matrix / matrix.sum(axis=1, keepdims=True) if column == 0 else matrix
            axis.imshow(scale, cmap="Blues", vmin=0.0, vmax=1.0)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = (
                        format(int(matrix[i, j]), ",")
                        if column == 0
                        else _fmt(100.0 * matrix[i, j], 1) + "%"
                    )
                    axis.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if scale[i, j] > 0.6 else "black",
                    )
            axis.set_xticks(range(len(names)))
            axis.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
            axis.set_yticks(range(len(names)))
            axis.set_yticklabels(names, fontsize=7)
            axis.grid(visible=False)
            axis.set_xlabel("predicted", fontsize=7)
            if column == 0:
                axis.set_ylabel("true", fontsize=7)
            axis.set_title(
                str(block["run"].iloc[0]) + " " + str(model_id) + " -- " + label, fontsize=8
            )
    return fig


def build_g14(command: str = "") -> Graph:
    import pandas as pd

    selection_path = _outputs("binary_results") / "final_model_selection.csv"
    selection = _read(selection_path)
    final = str(selection.sort_values("rank")["model_id"].iloc[0])
    path = _outputs("binary_results") / "EXP-A2" / "confusion_matrices.json"
    payload = _confusion(path, "binary")
    rows: list[dict[str, Any]] = []
    for model_id in dict.fromkeys([final, G14_COMPARATOR]):
        rows.extend(_matrix_rows("binary", "EXP-A2", model_id, payload))
    frame = pd.DataFrame(rows)
    n_records = int(frame[frame["model_id"] == final]["count_repeat0"].sum())

    spec = GraphSpec(
        figure_id="G14",
        title="Binary Confusion Matrix",
        caption=(
            "Confusion matrices of the final binary model ("
            + final
            + ") and the equal-weight ensemble "
            + G14_COMPARATOR
            + ", which it is statistically tied with, on EXP-A2. Left: counts "
            "summed over the five folds of repeat 0, a complete partition in "
            "which each of the "
            + format(n_records, ",")
            + " records is predicted exactly once, so the row totals are the "
            "class counts. Right: row-normalized over all folds of the repeated "
            "map; the diagonal is sensitivity (abnormal) and specificity "
            "(normal). Classes are in the fixed label-map order, normal then "
            "abnormal. " + WITHIN_CORPUS
        ),
        sources=(_relative(path), _relative(selection_path)),
        exp_id="EXP-A2",
        objective="O1 (binary screening)",
        dataset="D1 PhysioNet 2016",
        size="tall",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=_draw_matrices)


def _pascal_matrix(figure_id: str, task: str, run: str, table_file: str, command: str) -> Graph:
    import pandas as pd

    table_path = _outputs("multiclass_results") / table_file
    table = _read(table_path)
    models = multiclass_models(table)
    path = _outputs("multiclass_results") / run / "confusion_matrices.json"
    payload = _confusion(path, task)
    rows: list[dict[str, Any]] = []
    for model_id in models:
        rows.extend(_matrix_rows(task, run, model_id, payload))
    frame = pd.DataFrame(rows)
    n_records = int(frame[frame["model_id"] == models[0]]["count_repeat0"].sum())
    names = class_order(task)
    corpus = "PASCAL A (set_a)" if task == "pascal_a" else "PASCAL B (set_b)"

    chosen = (
        models[0] + ", the best model by macro-F1"
        if len(models) == 1
        else models[0]
        + ", the best model by macro-F1, and "
        + models[1]
        + ", the best by accuracy -- shown together because they are not the same model"
    )
    extra = (
        " `artifact` is a recording-quality label, not a cardiac class, so this "
        "is not a four-class cardiac classifier."
        if task == "pascal_a"
        else " A model can lead on accuracy here while never predicting the "
        "smallest class; the right-hand panels show whether it does."
    )
    spec = GraphSpec(
        figure_id=figure_id,
        title=("PASCAL A" if task == "pascal_a" else "PASCAL B") + " Confusion Matrix",
        caption=(
            corpus
            + " confusion matrices on "
            + run
            + " for "
            + chosen
            + ". Left: counts over the folds of repeat 0, in which each of the "
            + format(n_records, ",")
            + " labelled records is predicted once. Right: row-normalized over "
            "every fold; the diagonal is per-class recall. Classes are in the "
            "fixed order " + ", ".join(names) + "." + extra
        ),
        sources=(_relative(path), _relative(table_path)),
        exp_id=run,
        objective="O6 (multiclass event classification)",
        dataset="D2 PASCAL A" if task == "pascal_a" else "D3 PASCAL B",
        size="tall",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=_draw_matrices)


def build_g15(command: str = "") -> Graph:
    return _pascal_matrix("G15", "pascal_a", "EXP-B1", T11_FILE, command)


def build_g16(command: str = "") -> Graph:
    return _pascal_matrix("G16", "pascal_b", "EXP-B2", T12_FILE, command)


# ---------------------------------------------------------------------------
# G17 -- multiclass one-vs-rest ROC (T92.5)
# ---------------------------------------------------------------------------


def _ovr_rows(task: str, run: str, model_id: str, predictions: Any) -> list[dict[str, Any]]:
    """Per-class OvR ROC for one model, per fold, averaged.

    Only folds holding every class are used, for every class. That is the fold
    set on which the experiment's ``ovr_auc_macro`` is defined, so the mean of
    the per-class AUCs here reproduces it -- which is how the gate checks this
    recomputation against a second code path.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    names = class_order(task)
    labels = list(range(len(names)))
    part = predictions[predictions["model_id"] == model_id]
    if part.empty:
        raise ValueError(run + " has no predictions for " + model_id)
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    curves: dict[int, list[np.ndarray]] = {c: [] for c in labels}
    areas: dict[int, list[float]] = {c: [] for c in labels}
    macro: list[float] = []
    n_total = int(part["fold_label"].nunique())
    for _fold, rows in part.groupby("fold_label", sort=True):
        truth = rows["y_true"].to_numpy(int)
        if set(np.unique(truth)) != set(labels):
            continue
        fold_areas = []
        for c in labels:
            target = (truth == c).astype(int)
            score = rows["proba_" + str(c)].to_numpy(float)
            fpr, tpr, _ = roc_curve(target, score)
            curves[c].append(np.interp(grid, fpr, tpr))
            area = float(roc_auc_score(target, score))
            areas[c].append(area)
            fold_areas.append(area)
        macro.append(float(np.mean(fold_areas)))
    if not macro:
        raise ValueError(run + " " + model_id + ": no fold holds every class")
    out = []
    for c in labels:
        block = np.vstack(curves[c])
        mean = block.mean(axis=0)
        sd = block.std(axis=0, ddof=1) if block.shape[0] > 1 else np.zeros_like(mean)
        auc_mean = float(np.mean(areas[c]))
        auc_sd = float(np.std(areas[c], ddof=1)) if len(areas[c]) > 1 else 0.0
        for index, x in enumerate(grid):
            out.append(
                {
                    "task": task,
                    "run": run,
                    "model_id": model_id,
                    "class_index": c,
                    "class_name": names[c],
                    "x": round(float(x), 4),
                    "mean": float(mean[index]),
                    "sd": float(sd[index]),
                    "n_folds": int(block.shape[0]),
                    "n_folds_total": n_total,
                    "auc_mean": auc_mean,
                    "auc_sd": auc_sd,
                    "ovr_auc_macro_mean": float(np.mean(macro)),
                    "legend": names[c] + "  AUC " + _fmt(auc_mean),
                }
            )
    return out


def build_g17(command: str = "") -> Graph:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for task, run, table_file in (
        ("pascal_a", "EXP-B1", T11_FILE),
        ("pascal_b", "EXP-B2", T12_FILE),
    ):
        table_path = _outputs("multiclass_results") / table_file
        table = _read(table_path)
        model_id = multiclass_models(table)[0]
        parquet = _outputs("multiclass_results") / run / "predictions.parquet"
        if not parquet.is_file():
            raise FileNotFoundError(
                str(parquet) + " is gitignored and absent here; G17 cannot be rebuilt"
            )
        predictions = pd.read_parquet(parquet)
        rows.extend(_ovr_rows(task, run, model_id, predictions))
        sources += [_relative(parquet), _relative(table_path)]
    frame = pd.DataFrame(rows)

    def draw(data: Any) -> Any:
        tasks = list(dict.fromkeys(data["task"]))
        fig, axes = subplots("wide", ncols=len(tasks), sharey=True)
        axes = np.atleast_1d(axes)
        for axis, task in zip(axes, tasks, strict=True):
            block = data[data["task"] == task]
            for c in sorted(set(block["class_index"])):
                part = block[block["class_index"] == c].sort_values("x")
                x = part["x"].to_numpy(float)
                mean = part["mean"].to_numpy(float)
                sd = part["sd"].to_numpy(float)
                axis.plot(
                    x, mean, color=class_color(int(c)), linewidth=1.2, label=part["legend"].iloc[0]
                )
                axis.fill_between(
                    x,
                    np.clip(mean - sd, 0, 1),
                    np.clip(mean + sd, 0, 1),
                    color=class_color(int(c)),
                    alpha=0.12,
                    linewidth=0,
                )
            axis.plot([0, 1], [0, 1], ":", color="#777777", linewidth=0.8)
            first = block.iloc[0]
            axis.set_title(
                ("PASCAL A" if task == "pascal_a" else "PASCAL B")
                + ", "
                + str(first["run"])
                + " "
                + str(first["model_id"])
                + " (macro OvR AUC "
                + _fmt(first["ovr_auc_macro_mean"])
                + ", "
                + str(int(first["n_folds"]))
                + " of "
                + str(int(first["n_folds_total"]))
                + " folds)",
                fontsize=8,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.02)
            axis.set_xlabel("false-positive rate (this class vs the rest)")
            axis.legend(fontsize=6.5, loc="lower right", frameon=False)
        axes[0].set_ylabel("true-positive rate")
        return fig

    spec = GraphSpec(
        figure_id="G17",
        title="Multiclass One Vs Rest ROC",
        caption=(
            "One-vs-rest ROC curves per class for the best model by macro-F1 on "
            "each PASCAL track, computed per fold from out-of-fold probabilities "
            "and averaged on a common grid with a one-SD band. Only folds that "
            "hold every class are used, the same fold set on which the "
            "experiment's macro OvR AUC is defined; the title shows how many "
            "that is. The class AUCs are means of per-fold AUCs, and their mean "
            "equals the experiment's macro OvR AUC. PASCAL A is the smallest "
            "track in the project (T02 gives its class counts), so its curves are "
            "noisy by construction, and its `artifact` class is a recording-"
            "quality label, not a cardiac one."
        ),
        sources=tuple(sources),
        exp_id="EXP-B1, EXP-B2",
        objective="O6 (multiclass event classification)",
        dataset="D2 PASCAL A, D3 PASCAL B",
        size="wide",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G20 -- search convergence (T93.1)
# ---------------------------------------------------------------------------

SEARCHES: tuple[tuple[str, str, str], ...] = (
    ("SO-01", "SO-01 Randomized", "hyperparameter"),
    ("SO-02", "SO-02 Bayesian", "hyperparameter"),
    ("SO-03a", "SO-03a Genetic", "feature_mask"),
    ("SO-03b", "SO-03b PSO", "feature_mask"),
)


def build_g20(command: str = "") -> Graph:
    """Two panels, never one axis: see ``search_report``'s module docstring.

    SO-01/SO-02's ``best_so_far`` is a rising balanced accuracy and SO-03a/b's
    is a falling J. The ranges overlap, so a shared axis would read as the mask
    searches converging "below" the others. The figure is drawn with nrows=2.
    """
    import pandas as pd

    section = _outputs("search_optimization")
    blocks = []
    sources = []
    for exp, label, family in SEARCHES:
        path = section / exp / "convergence.csv"
        trace = _read(path)
        sources.append(_relative(path))
        step_column = "trial" if family == "hyperparameter" else "generation"
        block = pd.DataFrame(
            {
                "search": exp,
                "label": label,
                "family": family,
                "objective": (
                    "balanced accuracy, maximised" if family == "hyperparameter" else "J, minimised"
                ),
                "model_id": trace["model_id"],
                "step": trace[step_column].astype(int) + 1,
                "best_so_far": trace["best_so_far"].astype(float),
            }
        )
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)

    def draw(data: Any) -> Any:
        fig, axes = subplots("tall", nrows=2, sharex=False)
        top, bottom = axes
        hyper = data[data["family"] == "hyperparameter"]
        # Colour is the model, line style the method: the same model searched
        # two ways reads as one colour, solid against dashed.
        for (label, model_id), block in hyper.groupby(["label", "model_id"], sort=False):
            block = block.sort_values("step")
            top.plot(
                block["step"],
                block["best_so_far"],
                color=_model_color(str(model_id)),
                linestyle="-" if str(label).startswith("SO-01") else "--",
                linewidth=1.1,
                label=str(label) + " - " + str(model_id),
            )
        top.set_ylabel("best balanced\naccuracy so far", fontsize=8)
        top.set_xlabel("trial", fontsize=8)
        top.set_title(
            "Hyperparameter search - MAXIMISING balanced accuracy (higher is better)",
            loc="left",
            fontsize=9,
        )
        top.legend(fontsize=5.5, ncol=3, frameon=False, loc="lower right")

        mask = data[data["family"] == "feature_mask"]
        for index, (label, block) in enumerate(mask.groupby("label", sort=False)):
            block = block.sort_values("step")
            bottom.plot(
                block["step"],
                block["best_so_far"],
                color=class_color(index + 4),
                linewidth=1.3,
                marker="o",
                markersize=2.5,
                label=str(label),
            )
        bottom.set_ylabel("best J so far", fontsize=8)
        bottom.set_xlabel("generation / iteration", fontsize=8)
        bottom.set_title(
            "Feature-mask search - MINIMISING the multi-objective score J (lower is better)",
            loc="left",
            fontsize=9,
        )
        bottom.legend(fontsize=7, frameon=False, loc="upper right")
        for axis in (top, bottom):
            axis.tick_params(labelsize=7)
        return fig

    spec = GraphSpec(
        figure_id="G20",
        title="Search Convergence Plot",
        caption=(
            "Running best of all four searches on outer fold r0f0. TWO PANELS, "
            "NOT ONE AXIS: the hyperparameter searches (randomized, Bayesian) "
            "maximise balanced accuracy; the feature-mask searches (genetic, "
            "PSO) minimise J, a cost. The two ranges overlap numerically, so a "
            "shared axis would read as the mask searches converging below the "
            "others. Every curve is best-so-far: PSO keeps no elite, so its "
            "per-iteration best is not monotone and only the running best is "
            "comparable. Single-fold search traces, not cross-validated results."
        ),
        sources=tuple(sources),
        exp_id="SO-01, SO-02, SO-03a, SO-03b",
        objective="O3 (search-based optimization)",
        dataset="D1 PhysioNet 2016",
        size="tall",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G21 -- all features versus the selected subsets (T93.2)
# ---------------------------------------------------------------------------

G21_METRICS: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "sensitivity", "specificity")


def build_g21(command: str = "") -> Graph:
    source = _outputs("search_optimization") / "SO-04" / "all_features_vs_selected.csv"
    folds = _read(source)
    grouped = folds.groupby("configuration")
    frame = grouped[list(G21_METRICS)].mean().add_suffix("_mean")
    errors = grouped[list(G21_METRICS)].sem().add_suffix("_se")
    frame = frame.join(errors)
    frame["n_features"] = grouped["n_features"].mean()
    frame["n_folds"] = grouped["outer_fold"].nunique()
    frame = frame.sort_values("balanced_accuracy_mean").reset_index()

    def draw(data: Any) -> Any:
        fig, axis = subplots("double")
        positions = np.arange(len(data))
        width = 0.2
        for index, metric in enumerate(G21_METRICS):
            axis.barh(
                positions + (index - 1.5) * width,
                data[metric + "_mean"],
                height=width,
                xerr=data[metric + "_se"],
                color=class_color(index),
                label=METRIC_LABELS.get(metric, metric),
                error_kw={"elinewidth": 0.6, "capsize": 1.5},
            )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [
                str(name).replace("_", " ") + "  (" + str(round(float(k))) + " features)"
                for name, k in zip(data["configuration"], data["n_features"], strict=True)
            ],
            fontsize=7,
        )
        axis.set_xlabel("score on the held-out outer folds (mean +/- SE)", fontsize=8)
        axis.set_xlim(0.6, 1.0)
        axis.tick_params(labelsize=7)
        axis.legend(
            fontsize=7, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01)
        )
        axis.set_title("All 138 features versus the selected subsets", loc="left", pad=22)
        return fig

    spec = GraphSpec(
        figure_id="G21",
        title="All Features Versus Selected Features",
        caption=(
            "Every SO-04 configuration scored on the held-out outer folds: all "
            "138 features against each selected subset, mean with the standard "
            "error across folds. Every arm was selected inside its own fold's "
            "training rows and scored once on that fold's test rows. The x-axis "
            "starts at 0.6 to separate arms that differ by a few points; the "
            "bars are therefore not proportional to the score. Sorted by "
            "balanced accuracy."
        ),
        sources=(_relative(source),),
        exp_id="SO-04",
        objective="O3 (search-based feature selection)",
        dataset="D1 PhysioNet 2016",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G22 -- F1 and accuracy versus feature count (T93.3)
# ---------------------------------------------------------------------------

#: T93.3 names F1 and accuracy. Balanced accuracy is drawn beside them because
#: accuracy on a majority-normal corpus is exactly the metric rule 6 forbids alone.
G22_METRICS: tuple[str, ...] = ("macro_f1", "accuracy", "balanced_accuracy")


def build_g22(command: str = "") -> Graph:
    section = _outputs("search_optimization") / "SO-04"
    source = section / "feature_count_curve.csv"
    settings_path = section / "so04_settings.json"
    curve = _read(source)
    columns = ["ranker", "k", "n_evaluations"]
    for metric in G22_METRICS:
        columns += [metric + "_mean", metric + "_std"]
    frame = curve[columns].sort_values(["ranker", "k"]).reset_index(drop=True)
    chosen: dict[str, Any] = {}
    if settings_path.is_file():
        chosen = json.loads(settings_path.read_text(encoding="utf-8")).get("chosen", {}) or {}
    frame["chosen_ranker"] = str(chosen.get("ranker", ""))
    frame["chosen_k"] = int(chosen["k"]) if "k" in chosen else -1

    def draw(data: Any) -> Any:
        fig, axes = subplots("tall", nrows=len(G22_METRICS), sharex=True)
        top_k = int(data["k"].max())
        for index, metric in enumerate(G22_METRICS):
            axis = axes[index]
            for position, (ranker, block) in enumerate(data.groupby("ranker", sort=True)):
                block = block.sort_values("k")
                axis.errorbar(
                    block["k"],
                    block[metric + "_mean"],
                    yerr=block[metric + "_std"] / np.sqrt(block["n_evaluations"]),
                    color=class_color(position),
                    marker="o",
                    markersize=3,
                    linewidth=1.1,
                    capsize=2,
                    label=str(ranker),
                )
            full = float(data[data["k"] == top_k][metric + "_mean"].mean())
            axis.axhline(full, color="#888888", linestyle="--", linewidth=0.9)
            if int(data["chosen_k"].iloc[0]) > 0:
                axis.axvline(
                    int(data["chosen_k"].iloc[0]), color="#B00020", linestyle=":", linewidth=1.1
                )
            axis.set_ylabel(METRIC_LABELS[metric], fontsize=8)
            axis.tick_params(labelsize=7)
            if index == 0:
                axis.legend(fontsize=6.5, ncol=4, frameon=False, loc="lower right")
                axis.set_title(
                    "Performance versus feature count (inner folds, mean +/- SE)",
                    loc="left",
                    fontsize=9,
                )
        axes[-1].set_xlabel("features kept (k)", fontsize=8)
        return fig

    marker = ""
    if chosen:
        marker = (
            " The red dotted line is the shipped subset, "
            + str(chosen.get("ranker", ""))
            + " at k = "
            + str(chosen.get("k", ""))
            + ", chosen by J under a "
            + str(chosen.get("selection_rule", "")).replace("_", " ")
            + " performance guard."
        )
    spec = GraphSpec(
        figure_id="G22",
        title="F1 And Accuracy Versus Feature Count",
        caption=(
            "Macro-F1, accuracy and balanced accuracy against the number of "
            "features kept, one curve per ranker, from the T57.4 sweep: mean over "
            "the inner validation folds with the standard error, so no point on "
            "these curves saw a test row. The dashed line is the mean at the "
            "largest k. Accuracy is shown because T93.3 asks for it and balanced "
            "accuracy beside it because accuracy alone flatters a model that "
            "under-calls the minority class." + marker
        ),
        sources=(_relative(source), _relative(settings_path)),
        exp_id="SO-04",
        objective="O3 (search-based feature selection)",
        dataset="D1 PhysioNet 2016",
        size="tall",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G28 -- dataset-wise generalization (T93.6)
# ---------------------------------------------------------------------------

#: The five label spaces, in the order used everywhere else (G02). Rule 4:
#: they sit side by side on one axis but each keeps its own chance level.
G28_TASKS: tuple[tuple[str, str, str, str], ...] = (
    ("binary", "D1 PhysioNet\nbinary", "binary_results", T08_FILE),
    ("pascal_a", "D2 PASCAL A\n4-class", "multiclass_results", T11_FILE),
    ("pascal_b", "D3 PASCAL B\n3-class", "multiclass_results", T12_FILE),
    ("circor_murmur", "D4 CirCor\nmurmur 3-class", "circor_external_validation", T13_FILE),
    ("circor_outcome", "D4 CirCor\noutcome", "circor_external_validation", T14_FILE),
)

G28_BINARY_METRICS: tuple[str, ...] = ("sensitivity", "specificity", "balanced_accuracy")


def _task_rows(task: str, table: Any) -> Any:
    """The rows of a headline table that describe the task's in-domain run."""
    if task == "binary":
        return table[table["run"] == "EXP-A2"]
    if task == "circor_murmur":
        return table[(table["run"] == "EXP-C1-three_class") & (table["level"] == "recording")]
    if task == "circor_outcome":
        return table[table["level"] == "recording"]
    return table


def build_g28(command: str = "") -> Graph:
    import pandas as pd

    from src.utils.constants import LABEL_MAPS

    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for task, label, key, table_file in G28_TASKS:
        path = _outputs(key) / table_file
        table = _task_rows(task, _read(path))
        if table.empty:
            raise ValueError(table_file + " has no in-domain rows for " + task)
        sources.append(_relative(path))
        for record in table.to_dict("records"):
            rows.append(
                {
                    "panel": "within_corpus",
                    "task": task,
                    "label": label,
                    "regime": "within-corpus CV",
                    "model_id": str(record["model_id"]),
                    "metric": "balanced_accuracy",
                    "value": float(record["balanced_accuracy_mean"]),
                    "chance": 1.0 / len(LABEL_MAPS[task]),
                    "source_table": table_file,
                }
            )

    final = "M1"
    t08 = _read(_outputs("binary_results") / T08_FILE)
    pooled = t08[(t08["run"] == "EXP-A2") & (t08["model_id"] == final)].iloc[0]
    ts5_path = _outputs("ablation") / TS5_FILE
    ts5 = _read(ts5_path).set_index("model_id").loc[final]
    t16_path = _outputs("circor_external_validation") / T16_FILE
    t16 = _read(t16_path)
    t16 = t16[(t16["level"] == "recording") & (t16["rule"] == "none")].set_index("metric")
    sources += [_relative(ts5_path), _relative(t16_path)]
    regimes = (
        ("pooled 5x5 CV, tuned (EXP-A2)", lambda m: float(pooled[m + "_mean"]), T08_FILE),
        ("pooled 5x5 CV, defaults (EXP-A1)", lambda m: float(ts5[m + "_pooled"]), TS5_FILE),
        ("unseen sub-collection (EXP-F3)", lambda m: float(ts5[m + "_holdout"]), TS5_FILE),
        ("unseen corpus, CirCor (EXP-D1)", lambda m: float(t16.loc[m, "external_value"]), T16_FILE),
    )
    for regime, value, table_file in regimes:
        for metric in G28_BINARY_METRICS:
            rows.append(
                {
                    "panel": "beyond_corpus",
                    "task": "binary",
                    "label": "D1 PhysioNet binary",
                    "regime": regime,
                    "model_id": final,
                    "metric": metric,
                    "value": value(metric),
                    "chance": 0.5,
                    "source_table": table_file,
                }
            )
    frame = pd.DataFrame(rows)

    def draw(data: Any) -> Any:
        fig, (left, right) = subplots("wide", ncols=2, width_ratios=[1.15, 1.0])
        inner = data[data["panel"] == "within_corpus"]
        tasks = list(dict.fromkeys(inner["task"]))
        for position, task in enumerate(tasks):
            block = inner[inner["task"] == task].sort_values("model_id")
            offsets = np.linspace(-0.25, 0.25, len(block)) if len(block) > 1 else np.zeros(1)
            for offset, record in zip(offsets, block.to_dict("records"), strict=True):
                left.scatter(
                    position + offset,
                    record["value"],
                    s=16,
                    color=_model_color(str(record["model_id"])),
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=3,
                )
            chance = float(block["chance"].iloc[0])
            left.hlines(
                chance,
                position - 0.4,
                position + 0.4,
                colors="#555555",
                linestyles=":",
                linewidth=1.0,
            )
        for model_id in MODEL_ORDER:
            if model_id in set(inner["model_id"]):
                left.scatter(
                    [],
                    [],
                    s=16,
                    color=_model_color(model_id),
                    edgecolor="black",
                    linewidth=0.3,
                    label=model_id,
                )
        left.plot([], [], ":", color="#555555", label="chance (1 / classes)")
        left.set_xticks(range(len(tasks)))
        left.set_xticklabels(
            [inner[inner["task"] == t]["label"].iloc[0] for t in tasks], fontsize=6.5
        )
        left.set_ylim(0.0, 1.0)
        left.set_ylabel("balanced accuracy (fold mean)")
        left.set_title("Within each corpus: every model, every task", fontsize=8.5)
        left.legend(fontsize=5.5, ncol=4, frameon=False, loc="lower left")

        outer = data[data["panel"] == "beyond_corpus"]
        regimes_ = list(dict.fromkeys(outer["regime"]))
        width = 0.8 / len(G28_BINARY_METRICS)
        positions = np.arange(len(regimes_))
        for index, metric in enumerate(G28_BINARY_METRICS):
            part = outer[outer["metric"] == metric].set_index("regime").loc[regimes_]
            right.bar(
                positions + (index - 1) * width,
                part["value"],
                width,
                color=class_color(index),
                label=METRIC_LABELS[metric],
            )
        right.axhline(0.5, color="#555555", linestyle=":", linewidth=1.0)
        right.set_xticks(positions)
        right.set_xticklabels(
            [r.replace(", ", "\n").replace(" (", "\n(") for r in regimes_], fontsize=6
        )
        right.set_ylim(0.0, 1.05)
        right.set_title("Binary model " + final + " beyond its training data", fontsize=8.5)
        right.legend(fontsize=6, ncol=3, frameon=False, loc="upper right")
        return fig

    spec = GraphSpec(
        figure_id="G28",
        title="Dataset Wise Generalization",
        caption=(
            "Left: balanced accuracy of every model on each of the five label "
            "spaces under that corpus's own subject-grouped cross-validation, "
            "each against its own chance level -- the tasks share an axis but "
            "not a target, and are never merged. Right: the binary model "
            + final
            + " moving progressively further from its training data: pooled "
            "cross-validation (tuned and default hyperparameters), a PhysioNet "
            "sub-collection never seen in training (EXP-F3), and a different "
            "corpus (CirCor, EXP-D1, recording level). The collapse to near "
            "chance on an unseen sub-collection is a dataset limitation -- "
            "acquisition setup and label are entangled in PhysioNet -- and the "
            "CirCor drop also carries an adult-to-paediatric population shift. "
            "Neither is evidence that the method generalizes, and the pooled "
            "numbers are within-corpus only."
        ),
        sources=tuple(dict.fromkeys(sources)),
        exp_id="EXP-A1, EXP-A2, EXP-B1, EXP-B2, EXP-C1, EXP-C2, EXP-D1, EXP-F3",
        objective="O1 (screening), O6 (multiclass), external validation",
        dataset="D1, D2, D3, D4",
        size="wide",
        notes=(DISCLAIMER,),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_BUILDERS = {
    "G11": build_g11,
    "G12": build_g12,
    "G13": build_g13,
    "G14": build_g14,
    "G15": build_g15,
    "G16": build_g16,
    "G17": build_g17,
    "G20": build_g20,
    "G21": build_g21,
    "G22": build_g22,
    "G28": build_g28,
}


def build_result_graphs(
    figure_ids: tuple[str, ...] = RESULT_GRAPH_IDS, *, command: str = ""
) -> list[Graph]:
    """Build the requested figures without writing anything."""
    built: list[Graph] = []
    for figure_id in figure_ids:
        if figure_id not in _BUILDERS:
            raise KeyError("unknown result graph: " + figure_id)
        graph = _BUILDERS[figure_id](command)
        log.info("%s built (%d plotted rows)", figure_id, len(graph.frame))
        built.append(graph)
    return built
