"""T21 and G31 -- duration robustness (Phase 73, EXP-E2).

Arranged like the noise track and for the same reason: an ``analysis`` column
separating what the corpus happens to show from what shortening a recording
actually causes.

The confound that must travel with the observational half is stated on the
artifact itself, because it is severe here. **Duration is almost perfectly
confounded with corpus.** Every recording under 5 s is PASCAL; PhysioNet has
none. So a short-band deficit measured across corpora is a PASCAL result wearing
a duration label, and the truncation study exists precisely because that
comparison cannot settle anything on its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.reporting.graphs import Graph, GraphSpec, subplots
from src.reporting.plot_style import class_color
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "HEADLINE_METRICS",
    "summarise_bands",
    "build_t21",
    "build_g31",
]

log = get_logger("reporting.duration")

HEADLINE_METRICS = (
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "macro_f1",
    "accuracy",
)


def summarise_bands(per_fold: Any) -> Any:
    """mean +/- SD over folds for each ``(run, model, duration band)``."""
    import pandas as pd

    keys = ["run", "task", "model_id", "duration_band"]
    metrics = [m for m in HEADLINE_METRICS if m in per_fold.columns]
    rows: list[dict[str, Any]] = []
    for values, block in per_fold.groupby(keys, sort=True):
        row = dict(zip(keys, values, strict=True))
        row["n_records_corpus"] = int(block["n_records_corpus"].iloc[0])
        row["reported"] = bool(block["reported"].all())
        row["n_folds"] = len(block)
        row["n_units_mean"] = float(np.mean(block["n_units"]))
        row["median_duration_sec"] = float(np.median(block["median_duration_sec"]))
        row["min_duration_sec"] = float(np.min(block["min_duration_sec"]))
        for metric in metrics:
            series = np.asarray(block[metric], dtype=float)
            finite = series[np.isfinite(series)]
            row[metric + "_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[metric + "_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _clip_label(value: float) -> str:
    return "full (untruncated)" if not np.isfinite(value) else format(value, ".0f") + " s clip"


def _combine(summary: Any, truncation: Any) -> Any:
    """The observational rows, then the interventional ones, in one long frame."""
    import pandas as pd

    metrics = [m for m in HEADLINE_METRICS if m + "_mean" in summary.columns]
    observational = summary.copy()
    observational.insert(0, "analysis", "observational_bands")
    observational = observational.rename(columns={"duration_band": "group"})

    rows: list[dict[str, Any]] = []
    for _, row in truncation.iterrows():
        entry: dict[str, Any] = {
            "analysis": "interventional_truncation",
            "run": "EXP-E2-TRUNC",
            "task": "binary",
            "model_id": row["model_id"],
            "group": _clip_label(float(row["clip_seconds"])),
            "n_records_corpus": int(row["n_units"]),
            "reported": True,
            "n_folds": 1,
            "n_units_mean": float(row["n_units"]),
            "median_duration_sec": float(row["realised_seconds_mean"]),
            "min_duration_sec": float(row["realised_seconds_min"]),
        }
        for metric in metrics:
            entry[metric + "_mean"] = (
                float(row[metric]) if metric in truncation.columns else float("nan")
            )
            entry[metric + "_sd"] = float("nan")
        rows.append(entry)

    combined = pd.concat([observational, pd.DataFrame(rows)], ignore_index=True)
    order = ["analysis", "run", "task", "model_id", "group", "n_records_corpus", "reported"]
    order += ["n_folds", "n_units_mean", "median_duration_sec", "min_duration_sec"]
    order += [m + s for m in metrics for s in ("_mean", "_sd")]
    return combined[[c for c in order if c in combined.columns]].reset_index(drop=True)


def build_t21(
    summary: Any,
    truncation: Any,
    diagnostics: Any,
    sources: tuple[str, ...],
    command: str = "",
) -> Table:
    """T21 -- duration robustness, both halves, with the short-record finding."""
    frame = _combine(summary, truncation)
    metrics = [m for m in HEADLINE_METRICS if m + "_mean" in frame.columns]

    columns = (
        Column("analysis", "Analysis"),
        Column("run", "Run"),
        Column("task", "Task"),
        Column("model_id", "Model"),
        Column("group", "Group"),
        Column("n_records_corpus", "Records", kind="count"),
        Column("reported", "Reportable"),
        Column("n_folds", "Folds", kind="count"),
        Column("n_units_mean", "Units/fold", kind="metric", places=1),
        Column("median_duration_sec", "Median duration (s)", kind="seconds"),
        Column("min_duration_sec", "Shortest (s)", kind="seconds"),
        *[
            Column(m + s, m.replace("_", " ") + (" mean" if s == "_mean" else " SD"), kind="metric")
            for m in metrics
            for s in ("_mean", "_sd")
        ],
    )

    shortest = diagnostics.iloc[0] if len(diagnostics) else {}
    short_note = (
        "The shortest recording in the corpus is "
        + format(float(shortest.get("duration_sec", float("nan"))), ".3f")
        + " s ("
        + str(shortest.get("record_uid", "?"))
        + "). The extractor returns NO NaNs at any duration -- measured over all "
        "7,536 records -- so nothing breaks. What does happen is subtler: "
        "env_peak_rate counts heart SOUNDS at roughly 2.4/s on this corpus, so a "
        "window that short can physically contain only about "
        + format(float(shortest.get("implied_n_heart_sounds", float("nan"))), ".1f")
        + " of them, and a rate estimated from one or two events carries an "
        "enormous relative error while looking perfectly finite. Treat every "
        "frame-count-dependent feature on a sub-2-second record as an estimate "
        "with no usable precision, not as a measurement."
    )

    spec = TableSpec(
        table_id="T21",
        title="Duration Robustness",
        caption=(
            "Two kinds of evidence about recording length, labelled by the "
            "Analysis column. observational_bands partitions each experiment's "
            "stored out-of-fold predictions by the T18.3 bands (short under 5 s, "
            "medium 5-20 s, long over 20 s), loaded from the same assignment T18 "
            "used. interventional_truncation clips the same PhysioNet recordings "
            "to 10, 5 and 3 seconds and re-extracts them, scored by a model "
            "refitted without their subjects. Only the second half can attribute "
            "anything to duration."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-E2",
        objective="O4 (robustness)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        notes=(
            "DURATION IS CONFOUNDED WITH CORPUS. Every recording under 5 s is "
            "PASCAL; PhysioNet has none and CirCor has none. A short-band deficit "
            "compared across corpora is therefore a PASCAL result wearing a "
            "duration label, which is why the truncation study exists.",
            short_note,
            "Clips are taken from a randomly placed window, not from the start. "
            "The opening seconds of a recording carry the transducer being "
            "settled, so always clipping the head would confound length with "
            "position. The window start is drawn from the seeded generator.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return build_table(spec, frame)


def build_g31(summary: Any, truncation: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G31 -- band comparison beside the truncation curve."""
    frame = _combine(summary, truncation)

    def draw(data: Any) -> Any:
        observational = data[data["analysis"] == "observational_bands"]
        interventional = data[data["analysis"] == "interventional_truncation"].copy()

        fig, axes = subplots("wide", ncols=2, nrows=1)
        left, right = np.atleast_1d(axes)[0], np.atleast_1d(axes)[1]

        runs = sorted(set(observational["run"]))
        bands = [b for b in ("short", "medium", "long") if b in set(observational["group"])]
        width = 0.8 / max(len(bands), 1)
        for index, band in enumerate(bands):
            values, errors, hatches = [], [], []
            for run in runs:
                block = observational[
                    (observational["run"] == run) & (observational["group"] == band)
                ]
                values.append(float(block["balanced_accuracy_mean"].mean()))
                errors.append(float(block["balanced_accuracy_mean"].std(ddof=1)))
                hatches.append("" if len(block) and bool(block["reported"].all()) else "//")
            positions = np.arange(len(runs)) + index * width - 0.4 + width / 2
            left.bar(
                positions,
                values,
                width=width,
                yerr=errors,
                capsize=2,
                color=class_color(index),
                edgecolor="black",
                linewidth=0.4,
                hatch=hatches,
                label=band,
                error_kw={"elinewidth": 0.6},
            )
        left.axhline(0.5, color="grey", linewidth=0.6, linestyle="--")
        left.set_xticks(np.arange(len(runs)))
        left.set_xticklabels(runs, rotation=20, ha="right", fontsize=6)
        left.set_ylabel("balanced accuracy (mean over models and folds)")
        left.set_ylim(0, 1)
        left.set_title("Observational: T18.3 duration bands", fontsize=8)
        left.legend(fontsize=6, title="band", title_fontsize=6)

        interventional["seconds"] = [
            float("inf") if g.startswith("full") else float(g.split()[0])
            for g in interventional["group"]
        ]
        finite = interventional[np.isfinite(interventional["seconds"])].sort_values("seconds")
        full = interventional[~np.isfinite(interventional["seconds"])]
        curves = ("sensitivity", "specificity", "balanced_accuracy", "roc_auc")
        for index, metric in enumerate(curves):
            column = metric + "_mean"
            if column not in finite.columns:
                continue
            right.plot(
                finite["seconds"],
                finite[column],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=class_color(index),
                label=metric.replace("_", " "),
            )
            if len(full):
                right.axhline(
                    float(full[column].iloc[0]),
                    color=class_color(index),
                    linewidth=0.6,
                    linestyle=":",
                )
        right.axhline(0.5, color="grey", linewidth=0.6, linestyle="--")
        right.set_xlabel("clip length in seconds (shorter to the left)")
        right.set_ylim(0, 1)
        right.set_title("Interventional: the same recordings, truncated", fontsize=8)
        right.legend(fontsize=6)

        fig.suptitle("Duration robustness: what the corpus shows, and what shortening causes")
        fig.text(
            0.5,
            0.92,
            "Dotted lines are the same recordings untruncated. Left is confounded "
            "with corpus -- every sub-5 s record is PASCAL; hatched bars are below "
            "the 30-record floor.",
            fontsize=6,
            ha="center",
        )
        return fig

    spec = GraphSpec(
        figure_id="G31",
        title="Duration Wise Performance",
        caption=(
            "Left: balanced accuracy by T18.3 duration band for each experiment, "
            "averaged over that experiment's models and folds. Duration is "
            "confounded with corpus here -- every recording under 5 s is PASCAL "
            "-- and hatched bars fall below the 30-record reporting floor. Right: "
            "the same PhysioNet recordings clipped to 10, 5 and 3 seconds and "
            "re-extracted, scored by a model refitted without their subjects; "
            "dotted lines mark the untruncated control."
        ),
        sources=sources,
        exp_id="EXP-E2",
        objective="O4 (robustness)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        command=command,
        size="wide",
    )
    return Graph(spec=spec, frame=frame, draw=draw)
