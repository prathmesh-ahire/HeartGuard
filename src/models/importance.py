"""Feature importance -- impurity-based and permutation-based (T48.3, T48.4).

Two measures, and the difference between them is the whole point.

**Impurity importance** is free: a tree records how much each split reduced
impurity while it was being built, and the total per feature is available the
moment the forest is fitted. It is also **measured on the training rows**, and it
is biased toward features with many possible split points -- continuous,
high-cardinality features look important partly because they offer the tree more
places to cut. On this project's matrix that bias has a specific target: 7 of the
138 features carry no unique information (5 are constants by construction), and
the 24 wavelet and 39 MFCC columns are strongly correlated within their families.
Impurity importance spreads a family's true contribution across its members
arbitrarily, and gives correlated features a share each rather than a share
between them.

**Permutation importance** asks a different question: shuffle one column and see
how much the score falls. It measures what the *fitted model actually uses*, and
because it is computed by re-scoring, it can be computed **on rows the model
never saw**. That is what makes it trustworthy and it is why T48.4 calls it "the
more trustworthy alternative" -- and why Phase 81 reports it as the primary
measure.

Computing permutation importance on training rows is the failure mode this
module is built to prevent. It looks identical, runs identically, and produces
numbers that say how much the model relies on each feature *to memorise its
training set*. An overfitted forest scores every feature as important. So
:func:`permutation_importance` here **requires** an explicit held-out index and
refuses to run without one -- there is no default that silently uses everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "IMPORTANCE_FILENAME",
    "ImportanceError",
    "ImportanceResult",
    "supports_impurity_importance",
    "impurity_importance",
    "permutation_importance",
    "importance_frame",
    "family_importance",
]

log = get_logger("models.importance")

IMPORTANCE_FILENAME = "feature_importance.csv"


class ImportanceError(ValueError):
    """The importance cannot be computed as asked, or would not mean what it says."""


@dataclass
class ImportanceResult:
    """One importance measurement over the full feature list."""

    kind: str
    model_id: str
    feature_names: tuple[str, ...]
    values: np.ndarray
    std: np.ndarray | None = None
    scoring: str = ""
    n_repeats: int = 0
    n_scored_rows: int = 0
    computed_on: str = ""

    def __post_init__(self) -> None:
        if len(self.feature_names) != self.values.size:
            raise ImportanceError(
                "importance has " + str(self.values.size) + " values for "
                + str(len(self.feature_names)) + " feature names"
            )

    def top(self, n: int = 20) -> list[tuple[str, float]]:
        order = np.argsort(self.values)[::-1][:n]
        return [(self.feature_names[index], float(self.values[index])) for index in order]

    def as_frame(self) -> Any:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "model_id": self.model_id,
                "kind": self.kind,
                "feature": list(self.feature_names),
                "importance": self.values,
            }
        )
        if self.std is not None:
            frame["importance_std"] = self.std
        frame["scoring"] = self.scoring
        frame["n_repeats"] = self.n_repeats
        frame["n_scored_rows"] = self.n_scored_rows
        frame["computed_on"] = self.computed_on
        frame["rank"] = frame["importance"].rank(ascending=False, method="min").astype(int)
        return frame.sort_values("rank", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# T48.3 -- impurity
# ---------------------------------------------------------------------------


def _final_estimator(model: Any) -> Any:
    """Unwrap a Pipeline and this project's own estimator wrappers -- only those.

    Unwrapping by duck-typing on ``estimator_`` is wrong, and wrong in a way that
    fails quietly. ``RandomForestClassifier`` has an ``estimator_`` attribute: the
    single unfitted ``DecisionTreeClassifier`` it clones each tree from. A generic
    "follow ``estimator_`` until it stops" walks straight past the fitted forest
    into that prototype, which has no importances at all -- and on a wrapper that
    *did* have them, it would silently report one tree's opinion as the forest's.

    So the wrapper types are named explicitly. A new wrapper has to be added here
    deliberately, which is the point.
    """
    from src.models.calibration import CalibratedSVM
    from src.models.weighting import ClassWeightedClassifier

    current = model
    for _ in range(8):  # a wrapper chain deeper than this is a bug, not a design
        steps = getattr(current, "named_steps", None)
        if steps is not None and "estimator" in steps:
            current = steps["estimator"]
            continue
        if isinstance(current, ClassWeightedClassifier):
            inner = getattr(current, "estimator_", None)
            current = inner if inner is not None else current.estimator
            continue
        if isinstance(current, CalibratedSVM):
            # A calibrated model is an average over k refitted copies; there is
            # no single fitted estimator to take importances from, and pretending
            # otherwise would report one of the k as if it were the model.
            return current
        return current
    return current


def supports_impurity_importance(model: Any) -> bool:
    return hasattr(_final_estimator(model), "feature_importances_")


def impurity_importance(
    model: Any, feature_names: tuple[str, ...] | list[str], *, model_id: str = ""
) -> ImportanceResult:
    """The fitted model's own ``feature_importances_``, named and ranked.

    Labelled ``computed_on="train"`` in the result, not as a formality: every
    consumer of this table needs to know it was measured while the model was
    being built, on rows it was fitted to.
    """
    estimator = _final_estimator(model)
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        raise ImportanceError(
            type(estimator).__name__ + " exposes no feature_importances_; use "
            "permutation_importance, which works for any model"
        )

    names = tuple(str(name) for name in feature_names)
    return ImportanceResult(
        kind="impurity",
        model_id=model_id or type(estimator).__name__,
        feature_names=names,
        values=np.asarray(values, dtype=float),
        scoring="mean decrease in impurity",
        computed_on="train",
    )


# ---------------------------------------------------------------------------
# T48.4 -- permutation, on held-out rows only
# ---------------------------------------------------------------------------


def permutation_importance(
    model: Any,
    X: Any,
    y: Any,
    *,
    held_out_index: Any,
    feature_names: tuple[str, ...] | list[str],
    model_id: str = "",
    scoring: str = "balanced_accuracy",
    n_repeats: int = 10,
    seed: int = 42,
    train_index: Any = None,
    n_jobs: int = 1,
) -> ImportanceResult:
    """Permutation importance, computed **only** on the rows named by ``held_out_index``.

    ``held_out_index`` is required and has no default. Passing ``train_index`` as
    well is optional but recommended: when both are given, the two are checked
    for overlap and the call fails loudly if they share a row. That check is the
    one thing standing between this function and a table of numbers that look
    reasonable and mean nothing.

    ``scoring`` defaults to balanced accuracy rather than accuracy, for the same
    reason research rule 6 does: on a 79/21 split, permuting a feature that only
    helps identify the minority class barely moves plain accuracy, so an
    accuracy-scored permutation ranking systematically under-weights exactly the
    features a screening model depends on.
    """
    from sklearn.inspection import permutation_importance as sk_permutation

    held = np.asarray(held_out_index, dtype=int)
    if held.size == 0:
        raise ImportanceError("held_out_index is empty; nothing to score")

    if train_index is not None:
        overlap = np.intersect1d(held, np.asarray(train_index, dtype=int))
        if overlap.size:
            raise ImportanceError(
                "permutation importance would be computed on "
                + str(overlap.size)
                + " row(s) the model was trained on, e.g. positions "
                + str(overlap[:5].tolist())
                + "; that measures memorisation, not use"
            )

    features = np.asarray(X)
    targets = np.asarray(y)
    names = tuple(str(name) for name in feature_names)
    if features.shape[1] != len(names):
        raise ImportanceError(
            "X has " + str(features.shape[1]) + " columns for "
            + str(len(names)) + " feature names"
        )

    result = sk_permutation(
        model,
        features[held],
        targets[held],
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return ImportanceResult(
        kind="permutation",
        model_id=model_id or type(_final_estimator(model)).__name__,
        feature_names=names,
        values=np.asarray(result.importances_mean, dtype=float),
        std=np.asarray(result.importances_std, dtype=float),
        scoring=scoring,
        n_repeats=int(n_repeats),
        n_scored_rows=int(held.size),
        computed_on="held-out fold",
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def importance_frame(results: list[ImportanceResult]) -> Any:
    """Stack several measurements into one long table."""
    import pandas as pd

    if not results:
        raise ImportanceError("no importance results to tabulate")
    return pd.concat([result.as_frame() for result in results], ignore_index=True)


def family_importance(result: ImportanceResult) -> Any:
    """Roll importance up to the six feature families.

    Individual importances are unstable when features are correlated -- the 39
    MFCC columns split one contribution between them, and which member wins is
    close to arbitrary. The family total is the stable quantity, and it is the
    one that answers the question the thesis actually asks: which kind of
    acoustic measurement carries the signal.
    """
    import pandas as pd

    from src.feature_extraction.registry import spec_for

    families = []
    for name in result.feature_names:
        try:
            families.append(spec_for(name).family)
        except Exception:  # noqa: BLE001 - a non-registry column is not fatal here
            families.append("unknown")

    frame = pd.DataFrame(
        {"family": families, "importance": result.values, "feature": list(result.feature_names)}
    )
    # Permutation importance can be negative: shuffling a feature sometimes
    # *improves* the score, which means the model was being actively misled by
    # it. Impurity importance cannot. So the share has to be taken over the
    # positive part only -- dividing a family's signed total by a signed grand
    # total produces shares above 1 and below 0 that look like arithmetic and
    # mean nothing. The negative totals are kept in their own column, because a
    # family that costs the model accuracy is a finding, not a rounding error.
    frame["positive_importance"] = frame["importance"].clip(lower=0.0)
    grouped = (
        frame.groupby("family", as_index=False)
        .agg(
            total_importance=("importance", "sum"),
            positive_importance=("positive_importance", "sum"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
            n_negative=("importance", lambda values: int((values < 0).sum())),
            n_features=("feature", "count"),
        )
        .sort_values("total_importance", ascending=False, kind="mergesort")
    )
    positive_total = float(grouped["positive_importance"].sum())
    grouped["share_of_positive"] = (
        grouped["positive_importance"] / positive_total if positive_total else np.nan
    )
    grouped["net_is_negative"] = grouped["total_importance"] < 0
    grouped.insert(0, "kind", result.kind)
    grouped.insert(0, "model_id", result.model_id)
    return grouped.reset_index(drop=True)
