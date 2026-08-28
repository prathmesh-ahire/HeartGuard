"""Feature rankers and threshold selectors, all fitted on training rows only.

Four rankers (T57.1, T57.2) and two threshold selectors, behind one interface:
give them a training matrix and they return a score per column, high is better.
Nothing here decides *how many* features to keep -- that is the sweep's job
(T57.4) -- so the ranking can be computed once per fold and cut at every k in
the grid without refitting the ranker.

FOLD SAFETY IS THE WHOLE POINT
------------------------------
A ranker is a fitted thing. `mutual_info_classif` reads the labels; a random
forest's `feature_importances_` is the result of training. Rank once on the full
matrix and cut per fold and you have chosen your features using the labels of
rows you are about to be tested on -- the single most common way a feature
selection study reports a score it cannot reproduce. Every function here takes
the rows it may look at as an argument and has no way to reach the others.

`mutual_info_classif` is also stochastic: it estimates entropy from k-nearest
neighbour distances with random tie-breaking noise. Left unseeded it returns a
different ranking on every call, which would break research rule 5 quietly --
the numbers would just wobble. It is seeded here, always.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "RankingError",
    "FeatureRanking",
    "RANKERS",
    "THRESHOLD_SELECTORS",
    "rank_features",
    "top_k",
    "threshold_select",
    "rfecv_select",
]


class RankingError(ValueError):
    """A ranking cannot be computed as asked, or would not mean what it says."""


@dataclass(frozen=True)
class FeatureRanking:
    """One ranker's scores over the columns of one training block.

    ``order`` is the column positions best-first, so ``order[:k]`` is the top-k
    subset at any k without recomputing anything.
    """

    kind: str
    scores: np.ndarray
    order: np.ndarray
    seconds: float = 0.0

    def top(self, k: int) -> np.ndarray:
        """The best ``k`` column positions, in ascending position order.

        Sorted by position rather than by score so that two subsets holding the
        same columns compare equal, and so a written-out subset reads in matrix
        order rather than in an order that encodes the ranking.
        """
        if k < 1:
            raise RankingError("k must be >= 1, got " + str(k))
        if k > self.order.size:
            raise RankingError(
                "k=" + str(k) + " exceeds the " + str(self.order.size) + " columns ranked"
            )
        return np.sort(self.order[:k])


# ---------------------------------------------------------------------------
# T57.1 -- filter rankers
# ---------------------------------------------------------------------------


def _mutual_info(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif

    return np.asarray(mutual_info_classif(x, y, random_state=int(seed)), dtype=float)


def _anova_f(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.feature_selection import f_classif

    del seed
    scores, _ = f_classif(x, y)
    # A constant column inside this fold gives F = nan, which would sort ahead
    # of every real score under a naive argsort. Zero is the honest value: a
    # column that does not vary here separates nothing here.
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# T57.2 -- embedded rankers
# ---------------------------------------------------------------------------


def _tree_importance(model_id: str) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    def rank(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        from sklearn.base import clone

        from src.models import estimators as est
        from src.models.pipeline import build_pipeline

        estimator = est.build_estimator(model_id)
        pipeline = build_pipeline(
            clone(estimator), config=None, y=y, n_features=int(x.shape[1])
        )
        pipeline.fit(x, y)
        final = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline
        importances = getattr(final, "feature_importances_", None)
        if importances is None:
            raise RankingError(
                model_id + " exposes no feature_importances_; it cannot rank embedded"
            )
        del seed
        return np.asarray(importances, dtype=float)

    return rank


#: kind -> (X_train, y_train, seed) -> score per column, high is better.
RANKERS: dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "mutual_info": _mutual_info,
    "anova_f": _anova_f,
    "rf_importance": _tree_importance("M4"),
    "gb_importance": _tree_importance("M5"),
}


def rank_features(
    kind: str,
    x_train: Any,
    y_train: Any,
    *,
    seed: int | None = None,
) -> FeatureRanking:
    """Score every column of ``x_train`` under ``kind``. Training rows only."""
    import time

    if kind not in RANKERS:
        raise RankingError(
            "unknown ranker " + repr(kind) + "; expected one of " + ", ".join(sorted(RANKERS))
        )
    if seed is None:
        from src.utils.seed import GLOBAL_SEED

        seed = GLOBAL_SEED

    features = np.asarray(x_train, dtype=float)
    targets = np.asarray(y_train)
    if features.ndim != 2:
        raise RankingError("x_train must be 2-D, got shape " + str(features.shape))
    if features.shape[0] != targets.shape[0]:
        raise RankingError(
            "x_train has " + str(features.shape[0]) + " rows and y_train "
            + str(targets.shape[0])
        )
    if len(np.unique(targets)) < 2:
        raise RankingError(kind + ": the training rows hold a single class")

    # The rankers see the matrix as the pipeline's estimator would: imputed.
    # Not scaled -- mutual information and impurity importance are invariant to
    # a monotone rescale, and ANOVA F is a ratio of variances, so scaling would
    # change nothing but would put a fitted transformer between the ranker and
    # the data for no reason.
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(features)

    started = time.perf_counter()
    scores = RANKERS[kind](imputed, targets, int(seed))
    seconds = time.perf_counter() - started

    scores = np.asarray(scores, dtype=float)
    if scores.shape != (features.shape[1],):
        raise RankingError(
            kind + " returned " + str(scores.shape) + " scores for "
            + str(features.shape[1]) + " columns"
        )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    # Ties broken by column position, deterministically: `-scores` with a stable
    # sort gives best-first and leaves equal scores in matrix order, so two runs
    # of the same command produce the same subset (rule 5).
    order = np.argsort(-scores, kind="stable").astype(int)
    return FeatureRanking(kind=kind, scores=scores, order=order, seconds=float(seconds))


def top_k(ranking: FeatureRanking, k: int) -> np.ndarray:
    """``ranking.top(k)``, as a free function for readability at call sites."""
    return ranking.top(k)


# ---------------------------------------------------------------------------
# T57.2 -- embedded selection by IMPORTANCE THRESHOLD, not by count
# ---------------------------------------------------------------------------

#: name -> the sklearn threshold string it passes to SelectFromModel.
THRESHOLD_SELECTORS: dict[str, str] = {
    "rf_threshold_mean": "mean",
    "rf_threshold_median": "median",
    "gb_threshold_mean": "mean",
    "gb_threshold_median": "median",
}


def threshold_select(
    name: str,
    x_train: Any,
    y_train: Any,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Embedded selection at an importance threshold; returns (columns, threshold).

    Distinct from ``rank_features`` + ``top_k`` in one way that matters: the
    threshold decides the subset SIZE from the data instead of being told it.
    That is the point of T57.2 -- it is a genuinely different question from "the
    best 40" and it answers with a different number per fold, which is itself
    worth reporting.

    The threshold is computed here rather than delegated to
    ``SelectFromModel(threshold="mean")``. Not a preference: with ``prefit=True``
    that class exposes its resolved ``threshold_`` only through an attribute it
    sets during a ``fit`` it is being told to skip, so reading the number back is
    version-dependent. The mean and the median of an importance vector are not
    something worth a compatibility shim.
    """
    if name not in THRESHOLD_SELECTORS:
        raise RankingError(
            "unknown threshold selector " + repr(name) + "; expected one of "
            + ", ".join(sorted(THRESHOLD_SELECTORS))
        )
    model_id = "M4" if name.startswith("rf_") else "M5"
    ranked = rank_features(model_id_to_ranker(model_id), x_train, y_train, seed=seed)

    statistic = THRESHOLD_SELECTORS[name]
    threshold = (
        float(np.mean(ranked.scores)) if statistic == "mean"
        else float(np.median(ranked.scores))
    )
    columns = np.flatnonzero(ranked.scores >= threshold).astype(int)
    if columns.size == 0:
        # Only reachable if every importance is below the mean, which needs a
        # degenerate vector. Falling back to the single best feature keeps the
        # caller from having to handle an empty subset it cannot score.
        columns = np.asarray([int(ranked.order[0])], dtype=int)
    return np.sort(columns), threshold


def model_id_to_ranker(model_id: str) -> str:
    """The embedded ranker backed by ``model_id``."""
    mapping = {"M4": "rf_importance", "M5": "gb_importance"}
    if model_id not in mapping:
        raise RankingError(model_id + " has no embedded ranker")
    return mapping[model_id]


# ---------------------------------------------------------------------------
# T57.3 -- RFECV
# ---------------------------------------------------------------------------


def rfecv_select(
    x_train: Any,
    y_train: Any,
    groups: Any = None,
    *,
    step: int = 5,
    min_features_to_select: int = 10,
    inner_cv: int = 3,
    seed: int | None = None,
    scoring: str = "balanced_accuracy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recursive feature elimination with its own cross-validated stopping point.

    The estimator is logistic regression, not the sweep's evaluation model, and
    that is a compute decision the task names outright (T57.3): RFECV refits its
    estimator once per elimination step per CV fold, so 138 features at step 5
    over 3 folds is ~84 fits. At a 3.5 s Random Forest that is five minutes per
    outer fold; at a 0.25 s logistic regression it is twenty seconds.

    ``groups`` is used when given: the inner CV must not straddle a subject any
    more than the outer map does.
    """
    from sklearn.feature_selection import RFECV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if seed is None:
        from src.utils.seed import GLOBAL_SEED

        seed = GLOBAL_SEED

    features = np.asarray(x_train, dtype=float)
    targets = np.asarray(y_train)
    prepared = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(features)
    )

    if groups is None:
        splitter: Any = StratifiedKFold(
            n_splits=int(inner_cv), shuffle=True, random_state=int(seed)
        )
        split_groups = None
    else:
        splitter = StratifiedGroupKFold(
            n_splits=int(inner_cv), shuffle=True, random_state=int(seed)
        )
        split_groups = np.asarray(groups, dtype=object)

    selector = RFECV(
        estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        step=int(step),
        min_features_to_select=int(min_features_to_select),
        cv=splitter.split(prepared, targets, split_groups),
        scoring=scoring,
        n_jobs=1,
    )
    selector.fit(prepared, targets)
    columns = np.sort(np.asarray(selector.get_support(indices=True), dtype=int))
    grid = selector.cv_results_.get("mean_test_score")
    detail: dict[str, Any] = {
        "n_selected": int(columns.size),
        "step": int(step),
        "min_features_to_select": int(min_features_to_select),
        "inner_cv": int(inner_cv),
        "scoring": scoring,
        "best_cv_score": float(np.max(grid)) if grid is not None else float("nan"),
    }
    return columns, detail


def selected_names(columns: Sequence[int], feature_names: Sequence[str]) -> tuple[str, ...]:
    """Column positions to feature names, in matrix order."""
    names = list(feature_names)
    out_of_range = [int(c) for c in columns if not 0 <= int(c) < len(names)]
    if out_of_range:
        raise RankingError("columns outside the matrix: " + str(out_of_range[:5]))
    return tuple(names[int(c)] for c in sorted(int(c) for c in columns))
