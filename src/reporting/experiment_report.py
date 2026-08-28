"""T08 and T10 -- the model comparison and the fold-wise results (T64.6).

Both are views of ``per_fold_metrics.csv`` and nothing else. That is the point:
the per-fold table is the measurement, and every summary in the write-up has to
be recomputable from it. A number that appears in T08 but cannot be re-derived
from the 25 rows behind it is exactly the kind of value this project exists to
not produce.

Two files rather than one, because the phase asks for two different things:

``T08_individual_model_comparison.csv``
    One row per model, every reported metric as ``mean +/- SD``, ranked by the
    documented selection rule. The comparison table.

``T10_fold_wise_results.csv``
    Every (model, fold) value, kept. Phase 81's Wilcoxon and Friedman tests are
    paired across folds; aggregating these away would delete the statistics
    chapter's input. ``T10_fold_wise_summary.csv`` sits beside it with the same
    mean and SD in long form, one row per (model, metric).

**The SD is the sample SD over folds (ddof=1), not a standard error.** Folds of
a repeated CV are not independent -- the 25 training sets overlap heavily -- so
dividing by sqrt(25) would produce a confidence interval that is much too
narrow. The spread across folds is reported as a spread, and the significance
testing is left to Phase 81 where the dependence is handled explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "T08_FILENAME",
    "T10_FILENAME",
    "T10_SUMMARY_FILENAME",
    "HEADLINE_METRICS",
    "format_mean_sd",
    "build_t08",
    "build_t10",
    "build_t10_summary",
]

log = get_logger("reporting.experiment_report")

T08_FILENAME = "T08_individual_model_comparison.csv"
T10_FILENAME = "T10_fold_wise_results.csv"
T10_SUMMARY_FILENAME = "T10_fold_wise_summary.csv"

#: Reported for every binary run, in the order they belong in a table. Research
#: rule 6: sensitivity and balanced accuracy lead, accuracy is present but never
#: alone and never first.
HEADLINE_METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "f1",
    "precision",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "mcc",
    "brier",
    "ece",
)


def format_mean_sd(mean: float, sd: float, *, places: int = 4) -> str:
    """``0.8588 +/- 0.0255``. A NaN SD renders as ``n/a``, never as 0."""
    if not np.isfinite(mean):
        return "n/a"
    if not np.isfinite(sd):
        return format(mean, "." + str(places) + "f") + " +/- n/a"
    return format(mean, "." + str(places) + "f") + " +/- " + format(sd, "." + str(places) + "f")


def _model_name(model_id: str) -> str:
    from src.models import registry as reg

    try:
        return reg.entry(model_id).name
    except Exception:  # noqa: BLE001 - a missing registry entry is not fatal here
        return model_id


def _metrics_present(per_fold: Any, requested: tuple[str, ...] | None) -> list[str]:
    import pandas as pd

    candidates = list(requested) if requested else list(HEADLINE_METRICS)
    return [
        metric
        for metric in candidates
        if metric in per_fold.columns and pd.api.types.is_numeric_dtype(per_fold[metric])
    ]


def build_t08(
    per_fold: Any,
    *,
    metrics: tuple[str, ...] | None = None,
    rule: tuple[str, ...] = ("sensitivity", "balanced_accuracy"),
) -> Any:
    """One row per model: ``mean +/- SD`` per metric, ranked by the selection rule."""
    import pandas as pd

    frame = pd.DataFrame(per_fold)
    if frame.empty:
        raise ValueError("no per-fold metrics to summarise")
    columns = _metrics_present(frame, metrics)

    rows = []
    for model_id, block in frame.groupby("model_id", sort=False):
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_name": _model_name(str(model_id)),
            "exp_id": str(block["exp_id"].iloc[0]) if "exp_id" in block else "",
            "n_folds": len(block),
        }
        for metric in columns:
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            mean = float(np.mean(finite)) if finite.size else float("nan")
            sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            row[metric] = format_mean_sd(mean, sd)
            row[metric + "_mean"] = mean
            row[metric + "_sd"] = sd
            # Kept so a reader can see when a metric was undefined in some folds
            # -- an OvR AUC missing from three folds is a different claim from
            # one averaged over all 25.
            row[metric + "_n"] = int(finite.size)
        for timing in ("fit_seconds", "predict_seconds"):
            if timing in block.columns:
                row[timing + "_mean"] = float(block[timing].mean())
        rows.append(row)

    table = pd.DataFrame(rows)
    ordering = [metric + "_mean" for metric in rule if metric + "_mean" in table.columns]
    if ordering:
        table = table.sort_values(ordering, ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    table["ranked_by"] = ", ".join(rule)
    return table


def build_t10(per_fold: Any, *, metrics: tuple[str, ...] | None = None) -> Any:
    """Every (model, fold) value that the paired tests in Phase 81 will consume."""
    import pandas as pd

    frame = pd.DataFrame(per_fold)
    columns = _metrics_present(frame, metrics)
    keep = [
        column
        for column in (
            "exp_id", "model_id", "repeat", "fold", "fold_label", "n_train", "n_test",
        )
        if column in frame.columns
    ]
    table = frame[[*keep, *columns]].copy()
    return table.sort_values(["model_id", "repeat", "fold"]).reset_index(drop=True)


def build_t10_summary(per_fold: Any, *, metrics: tuple[str, ...] | None = None) -> Any:
    """Long form: one row per (model, metric), with the fold count behind it."""
    import pandas as pd

    frame = pd.DataFrame(per_fold)
    columns = _metrics_present(frame, metrics)

    rows = []
    for model_id, block in frame.groupby("model_id", sort=False):
        for metric in columns:
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            mean = float(np.mean(finite)) if finite.size else float("nan")
            sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            rows.append(
                {
                    "exp_id": str(block["exp_id"].iloc[0]) if "exp_id" in block else "",
                    "model_id": model_id,
                    "metric": metric,
                    "n_folds": len(block),
                    "n_folds_defined": int(finite.size),
                    "mean": mean,
                    "sd": sd,
                    "min": float(np.min(finite)) if finite.size else float("nan"),
                    "max": float(np.max(finite)) if finite.size else float("nan"),
                    "mean_pm_sd": format_mean_sd(mean, sd),
                }
            )
    return pd.DataFrame(rows)


def write_binary_tables(
    exp_id: str, *, out_dir: str | Path | None = None, command: str = ""
) -> dict[str, Path]:
    """Emit T08, T10 and T10's summary for one experiment, and register them."""
    from src.evaluation.experiment import Experiment, load_per_fold_metrics
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv

    exp = Experiment.load(exp_id)
    per_fold = load_per_fold_metrics(exp_id, out_dir=out_dir)
    section = exp.output_dir(out_dir).parent
    rule = exp.selection_rule or ("sensitivity", "balanced_accuracy")

    written = {
        T08_FILENAME: save_csv(build_t08(per_fold, rule=rule), section / T08_FILENAME),
        T10_FILENAME: save_csv(build_t10(per_fold), section / T10_FILENAME),
        T10_SUMMARY_FILENAME: save_csv(
            build_t10_summary(per_fold), section / T10_SUMMARY_FILENAME
        ),
    }

    register_evidence(
        "T08",
        written[T08_FILENAME],
        metric_or_asset="Individual-model comparison, mean +/- SD over the "
        + exp.cv
        + " map, ranked by "
        + ", ".join(rule),
        experiment_id=exp_id,
        dataset=exp.dataset,
        model=", ".join(exp.models),
        source_data=str(exp.output_dir(out_dir) / "per_fold_metrics.csv"),
        command=command,
    )
    register_evidence(
        "T10",
        written[T10_FILENAME],
        metric_or_asset="Fold-wise results, every (model, fold) value retained for "
        "the paired tests in Phase 81",
        experiment_id=exp_id,
        dataset=exp.dataset,
        source_data=str(exp.output_dir(out_dir) / "per_fold_metrics.csv"),
        command=command,
    )
    for name, path in written.items():
        log.info("%s -> %s", name, path)
    return written
