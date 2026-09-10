"""T17, T18 and T19 -- the ablation tables (Phases 74-75).

Three deliverables, two experiments:

**T17** (:func:`build_t17`) is EXP-F1's feature-family ablation: what each
family is worth, per model, with the delta against A7 (all 138). Ranked by
sensitivity, because rule 6 ranks by sensitivity.

**T18** (:func:`build_t18`) is the narrow question T17 buries: all 138 features
versus the SO-04 selected 20. A7 and A8 side by side with the delta and the
feature-count ratio, so "does the subset cost anything" has a table of its own
rather than two rows a reader has to find and subtract.

**T19** (:func:`build_t19`) is EXP-F2's optimization ablation: each stage of the
optimization pipeline as an incremental delta over the one before it.

All three read the same per-fold rows their experiments wrote. None of them
recompute a metric -- a table that recomputes is a second implementation of the
metric module, and the two will disagree eventually.
"""

from __future__ import annotations

from typing import Any

from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = ["build_t17", "build_t18", "build_t19"]

log = get_logger("reporting.ablation")

DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic tool."
)

_METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "accuracy",
)


def _metric_columns(frame: Any, *, deltas: bool = True) -> tuple[Column, ...]:
    columns: list[Column] = []
    for metric in _METRICS:
        if metric not in frame.columns:
            continue
        label = metric.replace("_", " ")
        columns.append(Column(metric, label, kind="metric"))
        if metric + "_sd" in frame.columns:
            columns.append(Column(metric + "_sd", label + " SD", kind="metric"))
        if deltas and metric + "_delta_vs_A7" in frame.columns:
            columns.append(Column(metric + "_delta_vs_A7", label + " Δ vs A7", kind="metric"))
    return tuple(columns)


# ---------------------------------------------------------------------------
# T17 -- feature-family ablation
# ---------------------------------------------------------------------------


def build_t17(summary: Any, per_fold: Any, sources: tuple[str, ...], command: str = "") -> Table:
    """EXP-F1's eight arms x five models, ranked within each arm by sensitivity."""
    frame = summary.copy()
    frame = frame.sort_values(
        ["config_id", "sensitivity"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)

    n_folds = sorted(set(per_fold.groupby(["config_id", "model_id"]).size().tolist()))
    columns = (
        Column("config_id", "Arm"),
        Column("families", "Feature families"),
        Column("n_features", "Features", kind="count"),
        Column("model_id", "Model"),
        Column("n_folds", "Folds", kind="count"),
        *_metric_columns(frame),
    )

    spec = TableSpec(
        table_id="T17",
        title="Feature Family Ablation",
        caption=(
            "Eight feature configurations over the identical DA-07 repeated 5x5 "
            "subject-grouped fold map, identical seed and identical pipeline. Every "
            "arm is a column subset of the single FE-03 matrix, so no arm can "
            "disagree with another about what a feature value means. A7 is the full "
            "138-feature representation and is the reference every delta is taken "
            "against; it re-runs EXP-A1's exact configuration and is verified "
            "against EXP-A1's committed per-fold numbers before this table is "
            "written. Ranked by sensitivity within each arm."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-F1",
        objective="O2 (representation), O5 (ablation evidence)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=(
            "Chroma (24) and envelope (5) have no solo arm. The eight "
            "configurations are the ones T74.1-T74.5 name; those two families "
            "appear only inside A7, so this table says nothing about either on "
            "its own and no claim may be made about them from it.",
            "The ensembles M6 and M7 are not run here. EXP-F1 varies the feature "
            "representation with the model held fixed; what ensembling is worth "
            "is EXP-F2's subject (T19), measured on the full 138 where it "
            "belongs.",
            "A8's 20 features are FE-12's shipped subset, read from disk. "
            "Selection was NOT re-run inside the ablation -- doing so would have "
            "fitted it across all 25 folds at once and made A8 the one arm that "
            "saw its own test rows.",
            "Five of the 24 time features are constants by construction on "
            "z-normalized signals (time_mean, time_std, time_var, time_rms, "
            "time_hjorth_activity), so A1's effective width is 19, not 24.",
            "Every value is the mean over "
            + ("/".join(str(n) for n in n_folds) if n_folds else "the")
            + " folds with its sample SD beside it; no number here is pooled "
            "across folds.",
            DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)


# ---------------------------------------------------------------------------
# T18 -- all features versus selected
# ---------------------------------------------------------------------------


def build_t18(summary: Any, per_fold: Any, sources: tuple[str, ...], command: str = "") -> Table:
    """A7 (138) against A8 (the SO-04 subset), per model, with the delta.

    Paired per fold, not merely compared: A7 and A8 ran the same 25 folds with
    the same seed, so the per-fold difference is a paired sample and its SD is
    the honest spread of the effect. An unpaired comparison of two 25-fold means
    would throw that structure away and overstate the uncertainty.
    """
    import numpy as np
    import pandas as pd

    both = summary[summary["config_id"].isin(["A7", "A8"])]
    if set(both["config_id"]) != {"A7", "A8"}:
        raise ValueError("T18 needs both A7 and A8; found " + str(sorted(set(both["config_id"]))))

    wide = both.pivot(index="model_id", columns="config_id")
    rows = []
    for model_id in sorted(wide.index):
        row: dict[str, Any] = {
            "model_id": model_id,
            "n_features_all": int(wide[("n_features", "A7")][model_id]),
            "n_features_selected": int(wide[("n_features", "A8")][model_id]),
        }
        row["feature_reduction"] = 1.0 - row["n_features_selected"] / row["n_features_all"]
        for metric in _METRICS:
            if (metric, "A7") not in wide.columns:
                continue
            row[metric + "_all"] = float(wide[(metric, "A7")][model_id])
            row[metric + "_selected"] = float(wide[(metric, "A8")][model_id])
            row[metric + "_delta"] = row[metric + "_selected"] - row[metric + "_all"]

            # The paired per-fold difference, which is what a later signed-rank
            # test would use. Reported here so T18 carries its own spread.
            paired = _paired_differences(per_fold, model_id, metric)
            row[metric + "_delta_sd"] = (
                float(np.std(paired, ddof=1)) if paired.size > 1 else float("nan")
            )
            row[metric + "_folds_selected_wins"] = int(np.sum(paired > 0))
            row[metric + "_n_folds_paired"] = int(paired.size)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        "sensitivity_selected", ascending=False, kind="mergesort"
    ).reset_index(drop=True)

    columns: list[Column] = [
        Column("model_id", "Model"),
        Column("n_features_all", "Features (all)", kind="count"),
        Column("n_features_selected", "Features (selected)", kind="count"),
        Column("feature_reduction", "Reduction", kind="metric"),
    ]
    for metric in _METRICS:
        if metric + "_all" not in frame.columns:
            continue
        label = metric.replace("_", " ")
        columns.extend(
            [
                Column(metric + "_all", label + " (all 138)", kind="metric"),
                Column(metric + "_selected", label + " (selected)", kind="metric"),
                Column(metric + "_delta", label + " Δ", kind="metric"),
                Column(metric + "_delta_sd", label + " Δ SD", kind="metric"),
                Column(
                    metric + "_folds_selected_wins",
                    label + " folds won", kind="count",
                ),
            ]
        )

    spec = TableSpec(
        table_id="T18",
        title="All Features versus Selected Features",
        caption=(
            "A7 (all 138 features) against A8 (the SO-04 selected subset) on the "
            "identical DA-07 repeated 5x5 subject-grouped fold map, identical seed "
            "and identical pipeline. A positive delta means the subset scored "
            "higher than the full representation. Deltas are PAIRED per fold -- the "
            "two arms ran the same folds -- so the SD is the spread of the "
            "difference itself, and 'folds won' counts the folds on which the "
            "subset beat all 138."
        ),
        sources=sources,
        columns=tuple(columns),
        exp_id="EXP-F1",
        objective="O5 (feature selection evidence)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=(
            "The subset was chosen ONCE, by FE-12, under a one-standard-error "
            "performance guard, and is read from disk here. It was not re-selected "
            "per fold inside this comparison; a subset re-chosen with the test "
            "folds visible would beat all 138 by construction.",
            "A reduction figure is not a speed claim. The cost of computing the "
            "20 retained features is not 20/138 of the cost of computing all 138 "
            "-- sample entropy dominates extraction and lives in the time family. "
            "T24-T26 carry the measured timings.",
            DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)


def _paired_differences(per_fold: Any, model_id: str, metric: str) -> Any:
    """A8 minus A7 on each fold both arms ran, in fold order."""
    import numpy as np

    if metric not in per_fold.columns:
        return np.array([], dtype=float)
    block = per_fold[per_fold["model_id"] == model_id]
    keys = ["repeat", "fold"]
    left = block[block["config_id"] == "A8"].set_index(keys)[metric]
    right = block[block["config_id"] == "A7"].set_index(keys)[metric]
    shared = left.index.intersection(right.index)
    if not len(shared):
        return np.array([], dtype=float)
    return (
        left.loc[shared].to_numpy(dtype=float) - right.loc[shared].to_numpy(dtype=float)
    )


# ---------------------------------------------------------------------------
# T19 -- optimization ablation
# ---------------------------------------------------------------------------


def build_t19(stages: Any, sources: tuple[str, ...], command: str = "") -> Table:
    """EXP-F2's stages, each as an incremental delta over the previous one."""
    frame = stages.copy()

    columns: list[Column] = [
        Column("stage_order", "#", kind="count"),
        Column("stage", "Stage"),
        Column("comparison", "Comparison"),
        Column("source_run", "Source run"),
        Column("model_id", "Model"),
        Column("n_folds", "Folds", kind="count"),
    ]
    for metric in _METRICS:
        if metric not in frame.columns:
            continue
        label = metric.replace("_", " ")
        columns.append(Column(metric, label, kind="metric"))
        if metric + "_sd" in frame.columns:
            columns.append(Column(metric + "_sd", label + " SD", kind="metric"))
        if metric + "_incremental" in frame.columns:
            columns.append(
                Column(metric + "_incremental", label + " Δ vs previous", kind="metric")
            )
        if metric + "_cumulative" in frame.columns:
            columns.append(
                Column(metric + "_cumulative", label + " Δ vs baseline", kind="metric")
            )

    spec = TableSpec(
        table_id="T19",
        title="Search and Optimization Ablation",
        caption=(
            "Each optimization stage as an incremental delta over the stage before "
            "it, plus the cumulative delta over the untuned single-model baseline. "
            "Every stage is scored on the identical DA-07 repeated 5x5 "
            "subject-grouped fold map with the identical seed, and every number is "
            "read from the stage's own stored per-fold metrics rather than "
            "recomputed here. A9 is the best individual model against the "
            "ensemble; A10 is the equal-weight ensemble against the "
            "optimized-weight ensemble."
        ),
        sources=sources,
        columns=tuple(columns),
        exp_id="EXP-F2",
        objective="O3 (optimization), O5 (ablation evidence)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=(
            "An incremental delta is a difference of means over shared folds, not "
            "a significance test. Phase 81 owns the paired tests; nothing here "
            "may be described as significant.",
            "Tuning improved every individual model and made the ensembles "
            "slightly worse. 'Tuned' is not a synonym for 'better' in this "
            "table, and the stage ordering is the pipeline's order, not a "
            "ranking.",
            "The nested search behind the tuned stage ran at a reduced budget -- "
            "12 Bayesian trials per model per outer fold, 5 for M5, 3 inner "
            "splits. Write it up as that, never as 'tuned' unqualified.",
            DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)
