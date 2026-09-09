"""T20 and G30 -- noise robustness (Phase 72, EXP-E1).

One table and one figure carrying **two different kinds of evidence**, labelled
as such on the artifact itself:

``observational_pp08``
    The stored out-of-fold predictions partitioned by the PP-08 quality flags.
    Nothing is altered. The groups differ in more than noise, so a gap is an
    association and never a causal claim.

``interventional_awgn``
    The same recordings at 20/15/10/5/0 dB added white Gaussian noise, scored by
    a model refitted without those recordings' subjects. Only the noise varies,
    so a gap here is causal -- for additive white noise, which is not the noise a
    stethoscope actually picks up.

They are in one table because a reader needs both to conclude anything, and they
carry an ``analysis`` column because reading one as the other is the specific
mistake this arrangement prevents.
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
    "summarise_quality",
    "build_t20",
    "build_g30",
]

log = get_logger("reporting.robustness")

HEADLINE_METRICS = (
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "macro_f1",
    "accuracy",
)


def summarise_quality(per_fold: Any) -> Any:
    """mean +/- SD over folds for each ``(run, model, quality group)``."""
    import pandas as pd

    keys = ["run", "task", "model_id", "quality_group"]
    metrics = [m for m in HEADLINE_METRICS if m in per_fold.columns]
    rows: list[dict[str, Any]] = []
    for values, block in per_fold.groupby(keys, sort=True):
        row = dict(zip(keys, values, strict=True))
        row["n_records_corpus"] = int(block["n_records_corpus"].iloc[0])
        row["reported"] = bool(block["reported"].all())
        row["n_folds"] = len(block)
        row["n_units_mean"] = float(np.mean(block["n_units"]))
        row["mean_snr_proxy_db"] = float(np.nanmean(block["mean_snr_proxy_db"]))
        for metric in metrics:
            series = np.asarray(block[metric], dtype=float)
            finite = series[np.isfinite(series)]
            row[metric + "_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[metric + "_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _sweep_label(value: float) -> str:
    return "clean (no noise added)" if not np.isfinite(value) else format(value, ".0f") + " dB SNR"


def _combine(summary: Any, sweep: Any) -> Any:
    """One long frame: the observational rows, then the interventional ones."""
    import pandas as pd

    metrics = [m for m in HEADLINE_METRICS if m + "_mean" in summary.columns]
    observational = summary.copy()
    observational.insert(0, "analysis", "observational_pp08")
    observational = observational.rename(columns={"quality_group": "group"})

    rows: list[dict[str, Any]] = []
    for _, row in sweep.iterrows():
        entry: dict[str, Any] = {
            "analysis": "interventional_awgn",
            "run": "EXP-E1-AWGN",
            "task": "binary",
            "model_id": row["model_id"],
            "group": _sweep_label(float(row["snr_db"])),
            "n_records_corpus": int(row["n_units"]),
            "reported": True,
            "n_folds": 1,
            "n_units_mean": float(row["n_units"]),
            "mean_snr_proxy_db": float(row["realised_snr_db_mean"]),
        }
        for metric in metrics:
            entry[metric + "_mean"] = (
                float(row[metric]) if metric in sweep.columns else float("nan")
            )
            entry[metric + "_sd"] = float("nan")
        rows.append(entry)

    combined = pd.concat([observational, pd.DataFrame(rows)], ignore_index=True)
    order = ["analysis", "run", "task", "model_id", "group", "n_records_corpus", "reported"]
    order += ["n_folds", "n_units_mean", "mean_snr_proxy_db"]
    order += [m + s for m in metrics for s in ("_mean", "_sd")]
    return combined[[c for c in order if c in combined.columns]].reset_index(drop=True)


def build_t20(
    summary: Any,
    sweep: Any,
    sqi: Any,
    sources: tuple[str, ...],
    command: str = "",
) -> Table:
    """T20 -- noise robustness, both halves, with the SQI agreement in the notes."""
    frame = _combine(summary, sweep)
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
        Column("mean_snr_proxy_db", "Mean SNR (dB)", kind="metric", places=2),
        *[
            Column(m + s, m.replace("_", " ") + (" mean" if s == "_mean" else " SD"), kind="metric")
            for m in metrics
            for s in ("_mean", "_sd")
        ],
    )

    row = sqi.iloc[0] if len(sqi) else {}
    sqi_note = (
        "Cross-checked against PhysioNet's own REFERENCE-SQI annotation over "
        + str(int(row.get("n_records", 0)))
        + " records: agreement "
        + format(float(row.get("agreement", float("nan"))), ".4f")
        + ", but recall of the SQI-poor records only "
        + format(float(row.get("recall_of_sqi_poor", float("nan"))), ".4f")
        + ". The two are measuring related but DIFFERENT things, so a PP-08 group "
        "must never be described as 'the noisy recordings' without saying whose "
        "definition of noisy. PP-08's own SNR proxy does separate the two SQI "
        "classes in the expected direction (mean "
        + format(float(row.get("mean_snr_proxy_db_sqi_good", float("nan"))), ".2f")
        + " dB good vs "
        + format(float(row.get("mean_snr_proxy_db_sqi_poor", float("nan"))), ".2f")
        + " dB poor)."
    )

    spec = TableSpec(
        table_id="T20",
        title="Noise Robustness",
        caption=(
            "Two kinds of evidence about noise, labelled by the Analysis column. "
            "observational_pp08 partitions each experiment's stored out-of-fold "
            "predictions by the PP-08 quality flags -- nothing is altered, the "
            "groups differ in more than noise, and a gap is an association. "
            "interventional_awgn adds white Gaussian noise at fixed SNRs to a "
            "stratified PhysioNet sample and scores it with a model refitted "
            "without those recordings' subjects -- only the noise varies, so a "
            "gap is causal for additive white noise specifically. The five label "
            "spaces are reported separately and never pooled."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-E1",
        objective="O4 (robustness)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        notes=(
            sqi_note,
            "Additive white Gaussian noise is a laboratory stressor, not a model "
            "of stethoscope noise -- real contamination is coloured, "
            "non-stationary and often in-band (breathing, bowel sounds, rubbing). "
            "The AWGN curve bounds sensitivity to a broadband disturbance; it "
            "does not predict field performance.",
            "SNR is set against the RAW recording, before the 20-400 Hz bandpass. "
            "The filter then removes the out-of-band share, so the realised "
            "post-filter SNR is higher than the nominal level and is reported "
            "beside it rather than assumed.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return build_table(spec, frame)


def build_g30(summary: Any, sweep: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G30 -- the observational gap and the interventional curve, side by side."""
    frame = _combine(summary, sweep)

    def draw(data: Any) -> Any:
        observational = data[data["analysis"] == "observational_pp08"]
        interventional = data[data["analysis"] == "interventional_awgn"].copy()

        fig, axes = subplots("wide", ncols=2, nrows=1)
        left, right = np.atleast_1d(axes)[0], np.atleast_1d(axes)[1]

        # Left: balanced accuracy per quality group, one cluster per run.
        runs = sorted(set(observational["run"]))
        groups = [
            g
            for g in ("clean", "noisy", "low_quality_other")
            if g in set(observational["group"])
        ]
        width = 0.8 / max(len(groups), 1)
        for index, group in enumerate(groups):
            values = []
            errors = []
            for run in runs:
                block = observational[
                    (observational["run"] == run) & (observational["group"] == group)
                ]
                values.append(float(block["balanced_accuracy_mean"].mean()))
                errors.append(float(block["balanced_accuracy_mean"].std(ddof=1)))
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
                label=group,
                error_kw={"elinewidth": 0.6},
            )
        left.axhline(0.5, color="grey", linewidth=0.6, linestyle="--")
        left.set_xticks(np.arange(len(runs)))
        left.set_xticklabels(runs, rotation=20, ha="right", fontsize=6)
        left.set_ylabel("balanced accuracy (mean over models and folds)")
        left.set_ylim(0, 1)
        left.set_title("Observational: PP-08 quality groups", fontsize=8)
        left.legend(fontsize=6, title="group", title_fontsize=6)

        # Right: the AWGN degradation curve. Clean is drawn at the left edge as a
        # reference band rather than on the SNR axis, which has no position for it.
        interventional["snr"] = [
            float("inf") if g.startswith("clean") else float(g.split()[0])
            for g in interventional["group"]
        ]
        finite = interventional[np.isfinite(interventional["snr"])].sort_values(
            "snr", ascending=False
        )
        clean = interventional[~np.isfinite(interventional["snr"])]
        curves = ("sensitivity", "specificity", "balanced_accuracy", "roc_auc")
        for index, metric in enumerate(curves):
            column = metric + "_mean"
            if column not in finite.columns:
                continue
            right.plot(
                finite["snr"],
                finite[column],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=class_color(index),
                label=metric.replace("_", " "),
            )
            if len(clean):
                right.axhline(
                    float(clean[column].iloc[0]),
                    color=class_color(index),
                    linewidth=0.6,
                    linestyle=":",
                )
        right.invert_xaxis()
        right.axhline(0.5, color="grey", linewidth=0.6, linestyle="--")
        right.set_xlabel("added noise, dB SNR against the raw signal (worse to the right)")
        right.set_ylim(0, 1)
        right.set_title("Interventional: added white Gaussian noise", fontsize=8)
        right.legend(fontsize=6)

        fig.suptitle("Noise robustness: what the corpus shows, and what noise causes")
        fig.text(
            0.5,
            0.92,
            "Dotted lines are the same model on the untouched recordings. Left "
            "is an association; only the right panel isolates noise.",
            fontsize=6,
            ha="center",
        )
        return fig

    spec = GraphSpec(
        figure_id="G30",
        title="Noise Level Robustness",
        caption=(
            "Left: balanced accuracy by PP-08 quality group for each experiment, "
            "averaged over that experiment's models and folds -- observational, "
            "and the groups differ in more than noise. Right: the same PhysioNet "
            "recordings at 20/15/10/5/0 dB added white Gaussian noise, scored by "
            "a model refitted without their subjects; dotted lines mark the "
            "untouched control. Additive white noise is a laboratory stressor and "
            "not a model of real stethoscope contamination."
        ),
        sources=sources,
        exp_id="EXP-E1",
        objective="O4 (robustness)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        command=command,
        size="wide",
    )
    return Graph(spec=spec, frame=frame, draw=draw)
