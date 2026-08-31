"""T22 and G32 -- the CirCor auscultation-location analysis (Phase 70, EXP-C3).

EXP-C3 is ``analysis_only``. It trains nothing and re-scores nothing: it takes
the out-of-fold predictions EXP-C1 and EXP-C2 already produced and asks a
question the pooled metrics cannot answer -- **does where the stethoscope was
placed change what the model gets right?**

Two things make that worth a table of its own.

*The label is a patient's, the prediction is a recording's.* Phase 69 propagates
each patient's ``Present`` / ``Abnormal`` label onto every recording taken from
them. A murmur audible only at the pulmonary area therefore contributes four
recordings labelled ``Present`` that contain nothing audible. That mislabelling
cannot be uniform across locations, so a per-location sensitivity is the closest
thing to a direct measurement of the cost.

*``Phc`` has four recordings in the entire corpus.* It is carried through every
row with ``reported=False`` and the reason attached, never quoted as a result.
Four records over five folds is under one per fold; the balanced accuracy it
produces (0.83) is a coin landing the same way twice, not a finding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.reporting.graphs import Graph, GraphSpec, subplots
from src.reporting.plot_style import class_color
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "T22_FILENAME",
    "HEADLINE_METRICS",
    "summarise_locations",
    "build_t22",
    "build_g32",
    "audible_summary",
]

log = get_logger("reporting.location")

T22_FILENAME = "T22_auscultation_location_analysis.csv"

#: Reported for every location. Rule 6: never accuracy on its own.
HEADLINE_METRICS = (
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
)


def summarise_locations(per_fold: Any) -> Any:
    """mean +/- SD over folds for each ``(run, model, location)``.

    The ``reported`` flag and its reason ride along unchanged from
    :func:`src.evaluation.location.stratified_metrics`, so a consumer of this
    table can never see an excluded location's number without also seeing that
    it was excluded.
    """
    import pandas as pd

    keys = ["run", "task", "model_id", "location"]
    metrics = [m for m in HEADLINE_METRICS if m in per_fold.columns]
    rows: list[dict[str, Any]] = []
    for values, block in per_fold.groupby(keys, sort=True):
        row = dict(zip(keys, values, strict=True))
        row["n_records_corpus"] = int(block["n_records_corpus"].iloc[0])
        row["reported"] = bool(block["reported"].iloc[0])
        row["n_folds"] = len(block)
        row["n_units_mean"] = float(np.mean(block["n_units"]))
        for metric in metrics:
            series = np.asarray(block[metric], dtype=float)
            finite = series[np.isfinite(series)]
            row[metric + "_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[metric + "_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        row["exclusion_reason"] = str(block["exclusion_reason"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def _relative(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def build_t22(summary: Any, sources: tuple[str, ...], command: str = "") -> Table:
    """T22 -- every location, every model, with the excluded one still visible."""
    metrics = [m for m in HEADLINE_METRICS if m + "_mean" in summary.columns]
    order = [
        "run",
        "task",
        "model_id",
        "location",
        "n_records_corpus",
        "reported",
        "n_folds",
        "n_units_mean",
    ]
    order += [m + suffix for m in metrics for suffix in ("_mean", "_sd")]
    order += ["exclusion_reason"]
    frame = summary[[c for c in order if c in summary.columns]].copy()

    columns = (
        Column("run", "Run"),
        Column("task", "Task"),
        Column("model_id", "Model"),
        Column("location", "Location"),
        Column("n_records_corpus", "Recordings", kind="count"),
        Column("reported", "Reportable"),
        Column("n_folds", "Folds", kind="count"),
        Column("n_units_mean", "Units/fold", kind="metric", places=1),
        *[
            Column(m + s, m.replace("_", " ") + (" mean" if s == "_mean" else " SD"), kind="metric")
            for m in metrics
            for s in ("_mean", "_sd")
        ],
        Column("exclusion_reason", "Why not reportable"),
    )

    spec = TableSpec(
        table_id="T22",
        title="Auscultation Location Analysis",
        caption=(
            "CirCor out-of-fold predictions stratified by the chest position the "
            "recording was taken from, for the murmur task in both label spaces "
            "and for the clinical outcome task. Nothing is retrained: this "
            "re-scores the predictions EXP-C1 and EXP-C2 already produced. "
            "Phc holds four recordings in the whole corpus and is carried with "
            "Reportable = False; its numbers are arithmetic noise and must not "
            "be quoted. Mean and SD are over the five patient-grouped folds."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-C3",
        objective="O5 (external validation)",
        dataset="D4 CirCor 2022",
        notes=(
            "Location read from metadata_master.recording_location and "
            "cross-checked against the record_uid, which encodes it "
            "independently; 43 recordings carry a repeat suffix (_1.._3) that a "
            "naive suffix split misreads.",
            "The patient-level label is propagated to every recording of that "
            "patient, so per-location sensitivity is measured against a label "
            "that may not be audible at that position. See "
            "circor_label_propagation.md.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return build_table(spec, frame)


def build_g32(summary: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G32 -- balanced accuracy by location, one panel per run.

    The excluded location is drawn, hatched and labelled, rather than dropped.
    A reader who sees four bars and no explanation cannot tell whether the fifth
    position was never recorded or was quietly removed.
    """
    from src.evaluation.location import LOCATIONS

    metric = "balanced_accuracy_mean"
    error = "balanced_accuracy_sd"
    frame = summary[summary[metric].notna()].copy()
    frame["location"] = frame["location"].astype(str)
    frame = frame.sort_values(["run", "model_id", "location"]).reset_index(drop=True)

    def draw(data: Any) -> Any:
        runs = sorted(set(data["run"]))
        fig, axes = subplots("wide", ncols=len(runs), nrows=1, sharey=True)
        axes = np.atleast_1d(axes)
        models = sorted(set(data["model_id"]))
        present = [loc for loc in LOCATIONS if loc in set(data["location"])]
        width = 0.8 / max(len(models), 1)

        for axis, run in zip(axes, runs, strict=True):
            block = data[data["run"] == run]
            for index, model_id in enumerate(models):
                rows = block[block["model_id"] == model_id].set_index("location")
                values = [float(rows[metric].get(loc, np.nan)) for loc in present]
                errors = [float(rows[error].get(loc, np.nan)) for loc in present]
                flagged = [not bool(rows["reported"].get(loc, True)) for loc in present]
                positions = np.arange(len(present)) + index * width - 0.4 + width / 2
                axis.bar(
                    positions,
                    values,
                    width=width,
                    yerr=errors,
                    capsize=2,
                    color=class_color(index),
                    edgecolor="black",
                    linewidth=0.4,
                    hatch=["//" if f else "" for f in flagged],
                    label=model_id,
                    error_kw={"elinewidth": 0.6},
                )
            axis.axhline(0.5, color="grey", linewidth=0.6, linestyle="--")
            axis.set_xticks(np.arange(len(present)))
            axis.set_xticklabels(
                [loc + (" *" if loc in set(block.loc[~block["reported"], "location"]) else "")
                 for loc in present],
                fontsize=7,
            )
            axis.set_title(run, fontsize=8)
            axis.set_ylim(0, 1)
        axes[0].set_ylabel("balanced accuracy (mean +/- SD over folds)")
        axes[-1].legend(fontsize=6, ncol=2, loc="upper right", frameon=True)
        fig.suptitle("CirCor performance by auscultation location")
        # Placed under the title, not at the figure foot: write_graph stamps the
        # provenance line there and the two would overlap.
        fig.text(
            0.5,
            0.915,
            "* hatched: not reportable -- Phc holds 4 recordings in the entire "
            "corpus, under one per fold. The dashed line is chance.",
            fontsize=6,
            ha="center",
        )
        return fig

    spec = GraphSpec(
        figure_id="G32",
        title="Location Performance",
        caption=(
            "Balanced accuracy at each auscultation position for every CirCor "
            "model, in both murmur label spaces and for clinical outcome. Bars "
            "are the mean over the five patient-grouped folds with SD whiskers. "
            "The hatched position (Phc, 4 recordings corpus-wide) is shown for "
            "completeness and is not a result."
        ),
        sources=sources,
        exp_id="EXP-C3",
        objective="O5 (external validation)",
        dataset="D4 CirCor 2022",
        notes=(
            "Balanced accuracy is used rather than accuracy because murmur is "
            "20.5% positive at recording level and accuracy rewards a model for "
            "rarely predicting Present.",
        ),
        command=command,
        size="wide",
    )
    return Graph(spec=spec, frame=frame, draw=draw)


def audible_summary(agreement: Any) -> Any:
    """T70.5 -- per-model summary of the ``Most audible location`` cross-check.

    ``Most audible location`` is a human annotation the model never sees. A
    positive ``delta`` means the model assigned a higher ``Present`` probability
    at the position where a clinician judged the murmur loudest than at that same
    patient's other positions -- evidence it is responding to the murmur rather
    than to a per-patient recording confound.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for model_id, block in agreement.groupby("model_id", sort=True):
        deltas = np.asarray(block["delta"], dtype=float)
        rows.append(
            {
                "model_id": model_id,
                "n_patients": len(block),
                "proba_at_most_audible": float(np.mean(block["proba_at_most_audible"])),
                "proba_elsewhere": float(np.mean(block["proba_elsewhere"])),
                "delta_mean": float(deltas.mean()),
                "delta_sd": float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan"),
                "share_positive": float(np.mean(deltas > 0)),
            }
        )
    return pd.DataFrame(rows)
