"""The fold-safe pipeline (Phase 44).

Every transformation that learns anything from the data -- the median used to
impute, the mean and standard deviation used to scale, the scores used to select
features -- is a **step inside an sklearn ``Pipeline``**, never something applied
to the matrix beforehand. That is the whole design, and it exists to make
research rule 2 structurally true rather than a thing to remember.

The distinction matters more than it looks. These two lines produce nearly
identical numbers and differ completely in what they mean::

    X = StandardScaler().fit_transform(X)     # leak: fitted on all folds
    Pipeline([("scaler", StandardScaler()), ...])   # fitted per training fold

The first computes a mean over records the model is about to be tested on. The
resulting metric is optimistic by a small, plausible, unnoticeable amount -- the
kind of error that survives review because nothing looks wrong. Once every such
step lives inside the pipeline, ``cross_val_*`` and the Phase 43 driver refit
them per fold automatically, and getting it wrong requires deliberately taking a
step back out.

**Imbalance is handled by class weights, not resampling, by default.** The
primary binary track is 77% normal, and SMOTE-style oversampling on 138
correlated acoustic features invents records that no chest ever produced. Class
weighting changes the loss without inventing data. Resampling stays available
behind a config flag, and if it is ever switched on it must also happen inside
the fold -- resampling before splitting duplicates a record into both train and
test, which is leakage that looks like a large improvement.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "SELECTOR_KINDS",
    "PipelineError",
    "build_pipeline",
    "make_imputer",
    "make_scaler",
    "make_selector",
    "class_weight_for",
    "imbalance_ratio",
    "supports_class_weight",
    "fitted_steps",
]

log = get_logger("models.pipeline")

SELECTOR_KINDS: tuple[str, ...] = (
    "none",
    "mutual_info",
    "anova_f",
    "rfecv",
    "fixed_subset",
)


class PipelineError(ValueError):
    """The requested pipeline cannot be built as specified."""


# ---------------------------------------------------------------------------
# T44.2 -- imputer and scaler
# ---------------------------------------------------------------------------


def make_imputer(settings: dict[str, Any] | None = None) -> Any:
    """Median imputation by default.

    Median rather than mean because several of the 138 have long tails -- the
    Phase 41 outlier report names them -- and a mean imputed into a skewed
    feature lands somewhere no real record sits.
    """
    from sklearn.impute import SimpleImputer

    config = dict(settings or {})
    strategy = str(config.get("strategy", "median"))
    if strategy not in {"mean", "median", "most_frequent", "constant"}:
        raise PipelineError("unknown imputer strategy: " + strategy)
    return SimpleImputer(strategy=strategy, keep_empty_features=True)


def make_scaler(settings: dict[str, Any] | None = None) -> Any:
    """StandardScaler by default; ``robust`` and ``none`` are also available."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    config = dict(settings or {})
    kind = str(config.get("kind", "standard")).lower()

    if kind in {"none", "off"}:
        return None
    if kind == "standard":
        return StandardScaler(
            with_mean=bool(config.get("with_mean", True)),
            with_std=bool(config.get("with_std", True)),
        )
    if kind == "robust":
        return RobustScaler()
    raise PipelineError("unknown scaler kind: " + kind)


# ---------------------------------------------------------------------------
# T44.3 -- the selector
# ---------------------------------------------------------------------------


def make_selector(
    settings: dict[str, Any] | None = None, *, n_features: int | None = None
) -> Any:
    """The feature-selection step: off, a fixed subset, or a search in-fold.

    All three variants are sklearn transformers, so all three fit on the
    training fold only. A "fixed subset" is the one case that could legitimately
    be applied outside the pipeline -- the columns are decided in advance and
    nothing is learned -- but it stays a step anyway, so that switching between
    the three modes never changes where fitting happens.
    """
    config = dict(settings or {})
    if not config.get("enabled", False):
        return None

    kind = str(config.get("kind", "none")).lower()
    if kind in {"none", "off"}:
        return None
    if kind not in SELECTOR_KINDS:
        raise PipelineError(
            "unknown selector kind: " + kind + "; expected one of " + str(SELECTOR_KINDS)
        )

    k = int(config.get("k", 60))
    if n_features is not None:
        k = min(k, int(n_features))
    if k < 1:
        raise PipelineError("selector k must be >= 1, got " + str(k))

    if kind == "fixed_subset":
        columns = config.get("columns")
        if not columns:
            raise PipelineError("selector kind 'fixed_subset' needs a 'columns' list")
        return _FixedSubset(list(columns))

    if kind in {"mutual_info", "anova_f"}:
        from sklearn.feature_selection import (
            SelectKBest,
            f_classif,
            mutual_info_classif,
        )

        score_func = mutual_info_classif if kind == "mutual_info" else f_classif
        return SelectKBest(score_func=score_func, k=k)

    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression

    # RFECV runs its own inner CV. That inner CV is over the training fold only,
    # because the whole selector is fitted inside the outer fold -- a nested
    # arrangement, which is what makes in-fold search legitimate.
    return RFECV(
        estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        min_features_to_select=k,
        cv=int(config.get("inner_cv", 3)),
        n_jobs=1,
    )


class _FixedSubset:
    """Select a fixed set of column positions. A transformer, so it fits in-fold."""

    def __init__(self, columns: list[int]) -> None:
        self.columns = list(columns)

    def fit(self, X: Any, y: Any = None) -> _FixedSubset:
        width = np.asarray(X).shape[1]
        out_of_range = [index for index in self.columns if not 0 <= index < width]
        if out_of_range:
            raise PipelineError(
                "fixed_subset references columns outside the matrix: "
                + str(out_of_range[:5])
            )
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X)[:, self.columns]

    def get_support(self, indices: bool = False) -> np.ndarray:
        if indices:
            return np.asarray(self.columns, dtype=int)
        raise PipelineError("boolean support needs the matrix width; pass indices=True")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"columns": self.columns}

    def set_params(self, **params: Any) -> _FixedSubset:
        if "columns" in params:
            self.columns = list(params["columns"])
        return self


# ---------------------------------------------------------------------------
# T44.4 -- imbalance
# ---------------------------------------------------------------------------


def imbalance_ratio(y: Any) -> float:
    """Majority count divided by minority count. 1.0 is perfectly balanced."""
    values, counts = np.unique(np.asarray(y), return_counts=True)
    if values.size < 2:
        raise PipelineError("cannot measure imbalance with fewer than two classes")
    return float(counts.max() / counts.min())


def supports_class_weight(estimator: Any) -> bool:
    """True when the estimator accepts a ``class_weight`` parameter.

    KNN and plain gradient boosting do not, so a weight silently passed to them
    would either raise or -- worse -- be accepted and ignored.
    """
    try:
        return "class_weight" in estimator.get_params(deep=False)
    except (AttributeError, TypeError):
        return hasattr(estimator, "class_weight")


def class_weight_for(
    estimator: Any, y: Any, settings: dict[str, Any] | None = None
) -> str | dict[Any, float] | None:
    """The class_weight to apply, or ``None`` when it cannot or should not be.

    Returns ``None`` rather than raising for an estimator that has no such
    parameter: the imbalance is then handled by whatever that model offers
    instead, and the caller is told through the log rather than crashing a
    25-fold run at fold 19.
    """
    config = dict(settings or {})
    strategy = config.get("strategy", "balanced")

    if strategy in (None, "none", False):
        return None
    if not supports_class_weight(estimator):
        log.info(
            "%s takes no class_weight; imbalance left to the estimator",
            type(estimator).__name__,
        )
        return None
    if strategy == "balanced":
        return "balanced"
    if isinstance(strategy, dict):
        return {key: float(value) for key, value in strategy.items()}
    raise PipelineError("unknown class_weight strategy: " + str(strategy))


# ---------------------------------------------------------------------------
# T44.1 -- the pipeline
# ---------------------------------------------------------------------------


def build_pipeline(
    estimator: Any,
    *,
    config: dict[str, Any] | None = None,
    y: Any = None,
    n_features: int | None = None,
    apply_class_weight: bool = True,
) -> Any:
    """Assemble imputer -> scaler -> [selector] -> estimator.

    ``estimator`` is cloned, so the object handed in is never fitted and cannot
    carry state between folds. The Phase 43 driver builds a fresh pipeline per
    fold on top of that; the two protections are independent on purpose.
    """
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    settings = _pipeline_config(config)
    steps: list[tuple[str, Any]] = []

    imputer = make_imputer(settings.get("imputer"))
    if imputer is not None:
        steps.append(("imputer", imputer))

    scaler = make_scaler(settings.get("scaler"))
    if scaler is not None:
        steps.append(("scaler", scaler))

    selector = make_selector(settings.get("selector"), n_features=n_features)
    if selector is not None:
        steps.append(("selector", selector))

    final = clone(estimator)
    if apply_class_weight and y is not None:
        weight = class_weight_for(final, y, settings.get("class_weight"))
        if weight is not None:
            final.set_params(class_weight=weight)

    if settings.get("resampling", {}).get("enabled", False):
        # Not silently ignored and not silently applied. Resampling belongs
        # inside the fold (imblearn's Pipeline, not sklearn's); until that is
        # built and tested, switching the flag on must fail loudly rather than
        # produce numbers that look like an improvement.
        raise PipelineError(
            "resampling is enabled in config but not implemented; it must be a "
            "step inside the fold (imblearn Pipeline), never applied to the "
            "matrix before splitting -- see Phase 44 in Docs/todo.md"
        )

    steps.append(("estimator", final))
    return Pipeline(steps)


def _pipeline_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    from src.utils.config import load_config

    loaded = load_config("models").get("pipeline")
    return dict(loaded) if loaded else {}


def fitted_steps(pipeline: Any) -> dict[str, Any]:
    """The fitted statistics of each learned step, for the fold-safety tests.

    Exposed so a test can assert *what the pipeline learned* -- the imputer's
    medians, the scaler's mean -- rather than only that its outputs look right.
    An output check cannot tell a scaler fitted on 3,240 records from one fitted
    on 2,593; the learned statistics can.
    """
    found: dict[str, Any] = {}
    for name, step in getattr(pipeline, "named_steps", {}).items():
        if hasattr(step, "statistics_"):
            found[name + ".statistics_"] = np.asarray(step.statistics_)
        if hasattr(step, "mean_") and step.mean_ is not None:
            found[name + ".mean_"] = np.asarray(step.mean_)
        if hasattr(step, "scale_") and step.scale_ is not None:
            found[name + ".scale_"] = np.asarray(step.scale_)
        if hasattr(step, "n_samples_seen_"):
            found[name + ".n_samples_seen_"] = step.n_samples_seen_
    return found
