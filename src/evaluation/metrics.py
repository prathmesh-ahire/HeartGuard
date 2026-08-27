"""Classification metrics (Phase 45).

Two rules from the project brief are enforced here rather than left to callers.

**Never accuracy alone (research rule 6).** :func:`binary_metrics` always returns
sensitivity, specificity, F1, balanced accuracy and AUC alongside accuracy;
:func:`multiclass_metrics` always returns macro-F1 and per-class recall. There is
no code path that produces an accuracy on its own, because on a corpus that is
77% normal, accuracy alone is 0.77 for a model that has learned nothing.

**Class ordering is fixed and explicit, never inferred from the data.** Every
function takes ``labels``. Deriving them from ``np.unique(y_true)`` is the
classic silent failure: a test fold that happens to contain no ``extrahls``
produces a 3x3 confusion matrix where every other fold produced 4x4, the rows
stop meaning the same thing, and averaging across folds quietly compares
different quantities. A label absent from a fold must appear as a zero row, not
vanish.

Specificity is computed here rather than taken from scikit-learn, which does not
expose it: it is the recall of the negative class, TN / (TN + FP). For screening
it matters as much as sensitivity, and rule 6 names it explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "DEFAULT_BOOTSTRAP",
    "ECE_BINS",
    "BINARY_KEYS",
    "confusion",
    "confusion_normalized",
    "binary_metrics",
    "multiclass_metrics",
    "specificity_score",
    "brier_score",
    "expected_calibration_error",
    "bootstrap_ci",
    "metrics_frame",
]

log = get_logger("evaluation.metrics")

DEFAULT_BOOTSTRAP = 2000
ECE_BINS = 10

#: What a binary report always contains. Asserted in the tests so a future edit
#: cannot quietly drop specificity or AUC from the reported set.
BINARY_KEYS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "tp",
    "fp",
    "tn",
    "fn",
    "support",
    "n_positive",
    "n_negative",
)


def _as_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred disagree on shape: " + str((true.shape, pred.shape))
        )
    if true.size == 0:
        raise ValueError("no samples to score")
    return true, pred


# ---------------------------------------------------------------------------
# T45.4 -- confusion matrices
# ---------------------------------------------------------------------------


def confusion(y_true: Any, y_pred: Any, labels: Sequence[Any]) -> np.ndarray:
    """Raw counts, rows = true, columns = predicted, ordered by ``labels``.

    ``labels`` is required, not optional. See the module docstring.
    """
    from sklearn.metrics import confusion_matrix

    true, pred = _as_arrays(y_true, y_pred)
    if len(labels) < 2:
        raise ValueError("need at least two labels")
    if len(set(map(str, labels))) != len(labels):
        raise ValueError("labels contains duplicates: " + str(list(labels)))

    unknown = set(np.unique(pred).tolist()) - set(labels)
    if unknown:
        raise ValueError(
            "y_pred contains labels absent from the fixed ordering: " + str(sorted(unknown))
        )
    return np.asarray(
        confusion_matrix(true, pred, labels=list(labels)), dtype=np.int64
    )


def confusion_normalized(
    y_true: Any, y_pred: Any, labels: Sequence[Any], *, axis: str = "true"
) -> np.ndarray:
    """Row- or column-normalized confusion matrix.

    A row of a class with no support normalizes to zeros rather than NaN: the
    class was never present, which is a count of zero, not an undefined rate.
    """
    matrix = confusion(y_true, y_pred, labels).astype(np.float64)
    if axis == "true":
        totals = matrix.sum(axis=1, keepdims=True)
    elif axis == "pred":
        totals = matrix.sum(axis=0, keepdims=True)
    else:
        raise ValueError("axis must be 'true' or 'pred', got " + str(axis))

    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    return normalized


# ---------------------------------------------------------------------------
# T45.1 / T45.2 -- binary
# ---------------------------------------------------------------------------


def specificity_score(
    y_true: Any, y_pred: Any, labels: Sequence[Any], positive_label: Any
) -> float:
    """TN / (TN + FP). Not available from scikit-learn directly."""
    negative_label = _negative_label(labels, positive_label)
    matrix = confusion(y_true, y_pred, [negative_label, positive_label])
    tn, fp = float(matrix[0, 0]), float(matrix[0, 1])
    denominator = tn + fp
    return float(tn / denominator) if denominator else float("nan")


def _negative_label(labels: Sequence[Any], positive_label: Any) -> Any:
    others = [label for label in labels if label != positive_label]
    if len(labels) != 2 or len(others) != 1:
        raise ValueError(
            "binary metrics need exactly two labels, got "
            + str(list(labels))
            + " with positive="
            + str(positive_label)
        )
    return others[0]


def binary_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    *,
    labels: Sequence[Any] = (0, 1),
    positive_label: Any = 1,
) -> dict[str, float]:
    """The full binary report. Never returns accuracy without its companions.

    ``y_proba`` is the probability **of the positive class** -- a 1-D vector. If
    a 2-D ``(n, 2)`` array is passed it is reduced using ``labels``, so the
    caller cannot silently hand over the wrong column.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true, pred = _as_arrays(y_true, y_pred)
    negative_label = _negative_label(labels, positive_label)
    ordering = [negative_label, positive_label]
    matrix = confusion(true, pred, ordering)
    tn, fp, fn, tp = (float(value) for value in matrix.ravel())

    sensitivity = float(
        recall_score(true, pred, pos_label=positive_label, zero_division=0)
    )
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    result: dict[str, float] = {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(
            precision_score(true, pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(f1_score(true, pred, pos_label=positive_label, zero_division=0)),
        "mcc": (
            float(matthews_corrcoef(true, pred))
            if len(set(true.tolist())) > 1
            else float("nan")
        ),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": float(true.size),
        "n_positive": float((true == positive_label).sum()),
        "n_negative": float((true == negative_label).sum()),
    }

    if y_proba is not None:
        scores = _positive_scores(y_proba, ordering, positive_label)
        binary_true = (true == positive_label).astype(int)
        if binary_true.min() != binary_true.max():
            result["roc_auc"] = float(roc_auc_score(binary_true, scores))
            result["pr_auc"] = float(average_precision_score(binary_true, scores))
        else:
            # One class in the fold: AUC is undefined, not 0.5 and not 1.0.
            log.warning("AUC undefined: y_true holds a single class")

    return result


def _positive_scores(
    y_proba: Any, ordering: Sequence[Any], positive_label: Any
) -> np.ndarray:
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.ndim == 2 and proba.shape[1] == len(ordering):
        return proba[:, list(ordering).index(positive_label)]
    raise ValueError(
        "y_proba must be 1-D positive-class scores or an (n, 2) array, got shape "
        + str(proba.shape)
    )


# ---------------------------------------------------------------------------
# T45.3 -- multiclass
# ---------------------------------------------------------------------------


def multiclass_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    *,
    labels: Sequence[Any],
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Macro/weighted averages plus per-class precision, recall and F1.

    Per-class recall is always present because rule 6 requires it: a macro-F1
    can look respectable while one minority class is never predicted at all, and
    PASCAL A has a class with 19 records.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    true, pred = _as_arrays(y_true, y_pred)
    ordering = list(labels)
    names = list(class_names) if class_names is not None else [str(x) for x in ordering]
    if len(names) != len(ordering):
        raise ValueError("class_names and labels differ in length")

    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=ordering, zero_division=0
    )

    result: dict[str, float] = {
        "accuracy": float((true == pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(
            f1_score(true, pred, labels=ordering, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(true, pred, labels=ordering, average="weighted", zero_division=0)
        ),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "support": float(true.size),
        "n_classes": float(len(ordering)),
        "ovr_auc_macro": float("nan"),
    }
    for index, name in enumerate(names):
        result["precision_" + name] = float(precision[index])
        result["recall_" + name] = float(recall[index])
        result["f1_" + name] = float(f1[index])
        result["support_" + name] = float(support[index])

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        if proba.shape != (true.size, len(ordering)):
            raise ValueError(
                "y_proba must be (n_samples, n_labels) = "
                + str((true.size, len(ordering)))
                + ", got "
                + str(proba.shape)
            )
        present = np.unique(true)
        if len(present) == len(ordering):
            result["ovr_auc_macro"] = float(
                roc_auc_score(true, proba, multi_class="ovr", average="macro", labels=ordering)
            )
        else:
            # One-vs-rest AUC is undefined when a class has no true samples in
            # this fold. Reported as NaN and aggregated with nanmean later,
            # rather than dropped or silently replaced with a plausible number.
            log.warning(
                "OvR AUC undefined: %d of %d classes present in y_true",
                len(present),
                len(ordering),
            )

    return result


# ---------------------------------------------------------------------------
# T45.5 -- calibration
# ---------------------------------------------------------------------------


def brier_score(y_true: Any, y_proba: Any, *, labels: Sequence[Any]) -> float:
    """Multiclass Brier score: mean squared error against the one-hot truth.

    For two classes this reduces to the familiar binary Brier score, so one
    function serves both and there is no chance of the two definitions being
    mixed between tracks.
    """
    true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    ordering = list(labels)

    if proba.ndim == 1:
        if len(ordering) != 2:
            raise ValueError("1-D probabilities require exactly two labels")
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape != (true.size, len(ordering)):
        raise ValueError(
            "y_proba must be (n_samples, n_labels), got " + str(proba.shape)
        )

    one_hot = np.zeros_like(proba)
    index = {label: position for position, label in enumerate(ordering)}
    for row, value in enumerate(true.tolist()):
        if value not in index:
            raise ValueError("y_true holds label " + str(value) + " absent from labels")
        one_hot[row, index[value]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Any, y_proba: Any, *, labels: Sequence[Any], n_bins: int = ECE_BINS
) -> float:
    """ECE over the predicted-class confidence, in ``n_bins`` equal-width bins.

    Empty bins contribute nothing rather than counting as perfectly calibrated:
    the weighting is by the number of samples in the bin, so an empty bin has
    weight zero by construction.
    """
    true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    ordering = list(labels)

    if proba.ndim == 1:
        if len(ordering) != 2:
            raise ValueError("1-D probabilities require exactly two labels")
        proba = np.column_stack([1.0 - proba, proba])
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    confidence = proba.max(axis=1)
    predicted = np.asarray(ordering, dtype=object)[proba.argmax(axis=1)]
    correct = (predicted == true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `right=True` on all but the first bin so 1.0 lands in the last bin rather
    # than in an out-of-range index.
    assignment = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)

    error = 0.0
    for index in range(n_bins):
        mask = assignment == index
        if not mask.any():
            continue
        weight = float(mask.sum()) / confidence.size
        error += weight * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


# ---------------------------------------------------------------------------
# T45.6 -- bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    metric: Callable[..., float],
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, float]:
    """Percentile bootstrap CI for any scalar metric.

    Resampling is with replacement over *records*, seeded at 42 so the interval
    is reproducible -- an unseeded CI moves every time it is computed, and a
    number in a paper that changes on re-run is not a result.

    Resamples that cannot be scored (a draw containing only one class, so AUC is
    undefined) are counted and excluded rather than replaced by a substitute
    value; ``n_valid`` reports how many contributed.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    proba = None if y_proba is None else np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(seed)

    def score(indices: np.ndarray) -> float:
        if proba is None:
            return float(metric(true[indices], pred[indices], **kwargs))
        return float(metric(true[indices], pred[indices], proba[indices], **kwargs))

    point = score(np.arange(true.size))
    samples: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, true.size, size=true.size)
        try:
            value = score(indices)
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            samples.append(value)

    if not samples:
        return {
            "point": point,
            "lower": float("nan"),
            "upper": float("nan"),
            "n_valid": 0.0,
            "n_resamples": float(n_resamples),
        }

    values = np.asarray(samples, dtype=float)
    lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "lower": float(lower),
        "upper": float(upper),
        "n_valid": float(values.size),
        "n_resamples": float(n_resamples),
    }


# ---------------------------------------------------------------------------
# tabulation
# ---------------------------------------------------------------------------


def metrics_frame(per_fold: Sequence[dict[str, float]]) -> Any:
    """Aggregate per-fold metric dicts into a mean/std/min/max table.

    ``nanmean`` and ``nanstd`` throughout: a fold whose AUC was undefined
    contributes nothing to the AUC row rather than dragging it to zero, and the
    ``n`` column states how many folds each row is actually based on -- which
    the statistical plan requires every p-value to declare anyway.
    """
    import pandas as pd

    if not per_fold:
        raise ValueError("no folds to aggregate")

    keys: list[str] = []
    for fold in per_fold:
        for key in fold:
            if key not in keys:
                keys.append(key)

    rows = []
    for key in keys:
        values = np.array(
            [float(fold.get(key, np.nan)) for fold in per_fold], dtype=float
        )
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "metric": key,
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
                "min": float(np.min(finite)) if finite.size else float("nan"),
                "max": float(np.max(finite)) if finite.size else float("nan"),
                "n": int(finite.size),
                "n_folds": len(per_fold),
            }
        )
    return pd.DataFrame(rows)
