"""T11 and T12 -- the PASCAL A (4-class) and PASCAL B (3-class) result tables.

Two things separate this from the binary builder in ``experiment_report``:

**Confidence intervals are mandatory, not decoration.** PASCAL A is 124 records
with 19 in one class (T66.6/T66.7); PASCAL B is 461 records with 46 in one class
(T67.5). A point estimate on that many records invites a reader to believe a
two-point difference. Every headline metric therefore carries an interval.

**Two intervals, because they answer different questions.** The *fold* interval
is a Student-t interval over the per-fold values and expresses how much the
score moves when the split moves. The *record* interval is a percentile
bootstrap over records and expresses how much it moves when the sample moves.
At n=124 the second is much the wider of the two, and reporting only the first
would understate the uncertainty by design rather than by accident.

The record bootstrap resamples **one repeat at a time**. Every record is tested
exactly once per repeat, so a repeat is a complete, duplicate-free set of
out-of-fold predictions. Pooling all repeats first would enter each record five
times and shrink the interval by roughly sqrt(5) -- an interval narrowed by the
CV protocol rather than by evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "T11_FILENAME",
    "T12_FILENAME",
    "MULTICLASS_HEADLINE",
    "fold_interval",
    "record_interval",
    "confusion_metrics",
    "coverage",
    "build_multiclass_table",
    "build_per_class_table",
    "write_multiclass_tables",
    "build_tuning_comparison",
]

log = get_logger("reporting.multiclass")

T11_FILENAME = "T11_pascal_a_results.csv"
T12_FILENAME = "T12_pascal_b_results.csv"

#: Rule 6 for the multiclass tracks: macro-F1 and per-class recall lead, and
#: accuracy is present only so a reader can see how misleading it is on
#: 320/95/46 (PASCAL B) and 40/34/31/19 (PASCAL A).
MULTICLASS_HEADLINE: tuple[str, ...] = (
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "ovr_auc_macro",
    "accuracy",
)

_PER_CLASS_KINDS = ("precision", "recall", "f1", "support")


# ---------------------------------------------------------------------------
# intervals
# ---------------------------------------------------------------------------


def fold_interval(values: Any, *, alpha: float = 0.05) -> dict[str, float]:
    """Student-t interval over per-fold values -- ``n`` is the fold count, not the record count.

    Reported with its own ``n`` because the statistics rule for this project
    requires every interval to state what it was computed over (Docs/note.md,
    the underpowered-statistics entry).
    """
    from scipy import stats

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    n = int(finite.size)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean = float(finite.mean())
    if n < 2:
        return {"mean": mean, "lo": float("nan"), "hi": float("nan"), "n": n}
    sem = float(stats.sem(finite))
    if not np.isfinite(sem) or sem == 0.0:
        return {"mean": mean, "lo": mean, "hi": mean, "n": n}
    half = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1)) * sem
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "n": n}


def confusion_metrics(
    y_true: Any, y_pred: Any, *, labels: Any
) -> dict[str, float]:
    """Every metric in :data:`MULTICLASS_HEADLINE` from one confusion matrix.

    Written because the obvious implementation -- call ``f1_score`` and friends
    once per bootstrap draw -- costs six sklearn round trips per draw, and at
    2000 draws x 5 repeats x 6 models that is 360,000 calls and roughly ten
    minutes for one table. Counting once with ``np.bincount`` and deriving all
    six from the counts is the same arithmetic, two orders of magnitude faster.

    The sklearn conventions are reproduced exactly, and
    ``test_the_fast_metrics_match_sklearn_exactly`` holds them to it:

    * macro precision/recall/F1 average over **every declared label**, scoring an
      undefined class 0 (``zero_division=0``), not over the labels that happen to
      appear in the draw;
    * weighted F1 weights by true support, so an absent class contributes
      nothing;
    * balanced accuracy averages recall over the classes **present in y_true**
      only -- that one differs from the macro metrics, and silently getting it
      wrong would shift every interval.
    """
    order = np.asarray(list(labels))
    n = int(order.size)
    position = {int(label): index for index, label in enumerate(order)}
    true = np.asarray([position[int(v)] for v in np.asarray(y_true)], dtype=np.int64)
    pred = np.asarray([position[int(v)] for v in np.asarray(y_pred)], dtype=np.int64)

    counts = np.bincount(true * n + pred, minlength=n * n).reshape(n, n)
    total = float(counts.sum())
    tp = np.diag(counts).astype(float)
    support = counts.sum(axis=1).astype(float)
    predicted = counts.sum(axis=0).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2.0 * precision * recall / denominator, 0.0)

    present = support > 0
    return {
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else float("nan"),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / total) if total else float("nan"),
    }


def coverage(predictions: Any, *, labels: Any) -> dict[str, Any]:
    """Which classes a model actually emits, and whether it collapsed.

    Added in Phase 67 after EXP-B2 produced two results a metrics table cannot
    show on its own:

    * **M3 on config defaults predicted `normal` for all 461 records.** Its
      macro-F1 of 0.2732 is, to four decimals, the always-predict-the-majority
      baseline. It appears in a results table as an ordinary row.
    * **M6 never predicted `extrastole` at any point** -- and M6 has the *highest
      accuracy* of the six models, because on a 320/95/46 corpus refusing to use
      the smallest class is rewarded.

    A model that cannot produce a class has not learned that class, whatever its
    accuracy says. ``degenerate`` marks the extreme case of a single constant
    output; ``missing_classes`` names anything never emitted.
    """
    import pandas as pd

    frame = pd.DataFrame(predictions)
    declared = [int(label) for label in labels]
    emitted = {int(value) for value in frame["y_pred"]}
    missing = [label for label in declared if label not in emitted]
    return {
        "n_classes_declared": len(declared),
        "n_classes_predicted": len(emitted & set(declared)),
        "predicts_all_classes": not missing,
        "missing_classes": ";".join(str(label) for label in missing),
        "degenerate": len(emitted) <= 1,
    }


def record_interval(
    predictions: Any,
    *,
    labels: Any,
    metrics: Any = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap over records, computed within each repeat and averaged.

    Returns ``{metric: {"point", "lo", "hi", "n_repeats", "n_records"}}`` -- every
    requested metric from one shared resampling pass, because drawing separate
    samples per metric would cost six times as much and buy nothing.
    """
    import pandas as pd

    frame = pd.DataFrame(predictions)
    wanted = list(metrics) if metrics is not None else list(_RECORD_METRICS)
    collected: dict[str, dict[str, list[float]]] = {
        name: {"point": [], "lo": [], "hi": []} for name in wanted
    }
    n_records = 0
    n_repeats = 0

    for repeat, block in frame.groupby("repeat", sort=True):
        y_true = np.asarray(block["y_true"], dtype=int)
        y_pred = np.asarray(block["y_pred"], dtype=int)
        size = int(y_true.size)
        if size == 0:
            continue
        n_records = max(n_records, size)
        n_repeats += 1

        rng = np.random.default_rng(int(seed) + int(repeat))
        draws: dict[str, list[float]] = {name: [] for name in wanted}
        for _ in range(int(n_resamples)):
            idx = rng.integers(0, size, size)
            scored = confusion_metrics(y_true[idx], y_pred[idx], labels=labels)
            for name in wanted:
                value = scored.get(name, float("nan"))
                # A draw missing a whole class leaves balanced accuracy undefined.
                # It is EXCLUDED, never replaced by a substitute value.
                if np.isfinite(value):
                    draws[name].append(value)

        point = confusion_metrics(y_true, y_pred, labels=labels)
        for name in wanted:
            if not draws[name]:
                continue
            collected[name]["point"].append(float(point.get(name, float("nan"))))
            collected[name]["lo"].append(
                float(np.percentile(draws[name], 100.0 * alpha / 2.0))
            )
            collected[name]["hi"].append(
                float(np.percentile(draws[name], 100.0 * (1.0 - alpha / 2.0)))
            )

    result: dict[str, dict[str, float]] = {}
    for name in wanted:
        bucket = collected[name]
        if not bucket["point"]:
            result[name] = {
                "point": float("nan"),
                "lo": float("nan"),
                "hi": float("nan"),
                "n_repeats": 0.0,
                "n_records": 0.0,
            }
            continue
        result[name] = {
            "point": float(np.mean(bucket["point"])),
            "lo": float(np.mean(bucket["lo"])),
            "hi": float(np.mean(bucket["hi"])),
            "n_repeats": float(n_repeats),
            "n_records": float(n_records),
        }
    return result


def _format_ci(lo: float, hi: float, *, places: int = 4) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    fmt = "." + str(places) + "f"
    return "[" + format(lo, fmt) + ", " + format(hi, fmt) + "]"


def _model_name(model_id: str) -> str:
    from src.models import registry as reg

    try:
        return reg.entry(model_id).name
    except Exception:  # noqa: BLE001 - a missing registry entry is not fatal here
        return model_id


#: Only label-based metrics are bootstrapped over records. ``ovr_auc_macro``
#: needs the probability columns and is left to the fold interval rather than
#: given a second interval computed on a different footing.
_RECORD_METRICS: tuple[str, ...] = (
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "accuracy",
    "macro_precision",
    "macro_recall",
)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def build_multiclass_table(
    per_fold: Any,
    *,
    predictions: Any = None,
    labels: Any = None,
    metrics: tuple[str, ...] = MULTICLASS_HEADLINE,
    rule: str = "macro_f1",
    n_resamples: int = 2000,
) -> Any:
    """One row per model: mean +/- SD, the fold interval, and the record interval.

    ``predictions`` and ``labels`` are optional only so the table can still be
    built from a metrics CSV alone; without them the record-interval column is
    written as ``n/a`` rather than omitted, so its absence is visible.
    """
    import pandas as pd

    frame = pd.DataFrame(per_fold)
    present = [m for m in metrics if m in frame.columns]
    preds = None if predictions is None else pd.DataFrame(predictions)

    rows: list[dict[str, Any]] = []
    for model_id, block in frame.groupby("model_id", sort=True):
        row: dict[str, Any] = {
            "model_id": str(model_id),
            "model_name": _model_name(str(model_id)),
            "n_folds": len(block),
            "ranked_by": rule,
        }
        for metric in present:
            values = np.asarray(block[metric], dtype=float)
            finite = values[np.isfinite(values)]
            interval = fold_interval(values)
            row[metric + "_mean"] = interval["mean"]
            row[metric + "_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            row[metric + "_fold_ci"] = _format_ci(interval["lo"], interval["hi"])
            row[metric + "_record_ci"] = "n/a"

        if preds is not None and labels is not None:
            block_preds = preds[preds["model_id"] == model_id]
            wanted = [metric for metric in present if metric in _RECORD_METRICS]
            if wanted and not block_preds.empty:
                # One resampling pass serves every metric -- see `record_interval`.
                intervals = record_interval(
                    block_preds,
                    labels=labels,
                    metrics=wanted,
                    n_resamples=n_resamples,
                )
                for metric, interval in intervals.items():
                    row[metric + "_record_ci"] = _format_ci(interval["lo"], interval["hi"])
                    row["n_records"] = int(interval.get("n_records", 0) or 0)
            if not block_preds.empty:
                row.update(coverage(block_preds, labels=labels))
        rows.append(row)

    table = pd.DataFrame(rows)
    if rule + "_mean" in table.columns:
        table = table.sort_values(rule + "_mean", ascending=False).reset_index(drop=True)
    return table


def build_per_class_table(per_fold: Any, class_names: Any) -> Any:
    """One row per (model, class): mean +/- SD and the fold interval, per class.

    Per-class **recall** is the column rule 6 cares about on these tracks -- it is
    what a 19-record or 46-record class actually tests.
    """
    import pandas as pd

    frame = pd.DataFrame(per_fold)
    rows: list[dict[str, Any]] = []
    for model_id, block in frame.groupby("model_id", sort=True):
        for name in class_names:
            row: dict[str, Any] = {
                "model_id": str(model_id),
                "model_name": _model_name(str(model_id)),
                "class_name": str(name),
            }
            for kind in _PER_CLASS_KINDS:
                column = kind + "_" + str(name)
                if column not in block.columns:
                    continue
                values = np.asarray(block[column], dtype=float)
                finite = values[np.isfinite(values)]
                if kind == "support":
                    row["support_mean"] = float(finite.mean()) if finite.size else float("nan")
                    continue
                interval = fold_interval(values)
                row[kind + "_mean"] = interval["mean"]
                row[kind + "_sd"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
                )
                row[kind + "_fold_ci"] = _format_ci(interval["lo"], interval["hi"])
            rows.append(row)
    return pd.DataFrame(rows)



def build_tuning_comparison(
    tuned: Any, untuned: Any, *, metrics: tuple[str, ...] = ("macro_f1", "balanced_accuracy")
) -> Any:
    """Per-model, per-fold comparison of a nested-tuned run against config defaults.

    Written because on a 124-record corpus a search inside each training fold can
    make things *worse*: the inner split is small enough that its score is mostly
    noise, and the point that maximises noise does not generalise. Reporting the
    tuned run without this table would leave that unstated.

    Both runs share one fold map, so the comparison is paired: ``wins`` counts
    folds where the tuned run scored higher on the same split, which is the only
    honest way to compare two runs at this sample size.
    """
    import pandas as pd

    left = pd.DataFrame(tuned)
    right = pd.DataFrame(untuned)
    rows: list[dict[str, Any]] = []
    for model_id in sorted(set(left["model_id"]) & set(right["model_id"])):
        a = left[left["model_id"] == model_id].set_index("fold_label")
        b = right[right["model_id"] == model_id].set_index("fold_label")
        shared = sorted(set(a.index) & set(b.index))
        row: dict[str, Any] = {
            "model_id": model_id,
            "model_name": _model_name(model_id),
            "n_paired_folds": len(shared),
        }
        for metric in metrics:
            if metric not in a.columns or metric not in b.columns:
                continue
            tuned_values = np.asarray(a.loc[shared, metric], dtype=float)
            plain_values = np.asarray(b.loc[shared, metric], dtype=float)
            difference = tuned_values - plain_values
            row[metric + "_tuned"] = float(np.nanmean(tuned_values))
            row[metric + "_defaults"] = float(np.nanmean(plain_values))
            row[metric + "_delta"] = float(np.nanmean(difference))
            row[metric + "_tuned_wins"] = int(np.nansum(difference > 0))
        rows.append(row)
    table = pd.DataFrame(rows)
    key = metrics[0] + "_delta"
    if key in table.columns:
        table = table.sort_values(key, ascending=False).reset_index(drop=True)
    return table

def write_multiclass_tables(
    directory: Any,
    *,
    filename: str,
    per_fold: Any,
    predictions: Any,
    labels: Any,
    class_names: Any,
    n_resamples: int = 2000,
) -> dict[str, Any]:
    """Write the headline table and its per-class companion beside each other."""
    from pathlib import Path

    from src.utils.io import ensure_dir

    target = Path(ensure_dir(directory))
    headline = build_multiclass_table(
        per_fold, predictions=predictions, labels=labels, n_resamples=n_resamples
    )
    per_class = build_per_class_table(per_fold, class_names)

    written: dict[str, Any] = {}
    headline_path = target / filename
    headline.to_csv(headline_path, index=False)
    written[filename] = headline_path

    companion = filename.replace(".csv", "_per_class.csv")
    per_class_path = target / companion
    per_class.to_csv(per_class_path, index=False)
    written[companion] = per_class_path

    log.info("%s: %d model row(s), %d per-class row(s)", filename, len(headline), len(per_class))
    return written
