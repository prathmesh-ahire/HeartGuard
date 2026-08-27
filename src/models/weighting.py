"""Class weighting for estimators that do not have a ``class_weight`` parameter.

Three of the models this project needs cannot be told their classes are
imbalanced through the pipeline: ``GradientBoostingClassifier`` has no
``class_weight`` at all, and ``XGBClassifier`` offers only ``scale_pos_weight``,
which is binary-only and therefore useless for the PASCAL and CirCor multiclass
tracks. Every one of them accepts ``sample_weight`` at ``fit`` time instead.

That sounds like a small difference and is not one. The Phase 43 driver calls
``estimator.fit(X, y)`` with no fit parameters, and the Phase 44 pipeline sets
class weights by ``set_params(class_weight=...)`` on its final step. An estimator
with neither route ends up with **no imbalance handling whatsoever** while
looking exactly like the ones that have it -- and on the primary binary track,
which is 79% normal, that is the difference between a model that finds abnormal
recordings and one that quietly learns to say "normal".

:class:`ClassWeightedClassifier` closes the gap by exposing ``class_weight`` as a
real constructor parameter and converting it to a per-row ``sample_weight``
vector at fit time. ``"balanced"`` reproduces sklearn's own definition exactly --
``n_samples / (n_classes * count(class))`` -- so a wrapped estimator and a native
``class_weight="balanced"`` estimator are weighted identically, and a comparison
between them measures the model rather than the plumbing.

**Weights are computed from the rows passed to ``fit``**, which inside the
pipeline are the training fold's rows and nothing else. Computing them from the
full matrix would leak the test fold's class balance into training -- a small,
plausible, invisible leak of exactly the kind research rule 2 exists to stop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if

from src.utils.logging_setup import get_logger

__all__ = [
    "WeightingError",
    "ClassWeightedClassifier",
    "balanced_sample_weight",
    "needs_sample_weight_wrapper",
]

log = get_logger("models.weighting")


class WeightingError(ValueError):
    """The requested weighting cannot be applied."""


def _inner_has(method: str) -> Any:
    """Predicate for ``available_if``: does the estimator being wrapped have this?

    Checked against the fitted inner estimator when there is one, and against the
    constructor argument before ``fit`` -- because sklearn inspects an *unfitted*
    estimator to decide which response method it will use later.
    """

    def check(wrapper: Any) -> bool:
        inner = getattr(wrapper, "estimator_", None) or getattr(wrapper, "estimator", None)
        return inner is not None and hasattr(inner, method)

    return check


def balanced_sample_weight(
    y: Any, class_weight: str | dict[Any, float] | None = "balanced"
) -> np.ndarray | None:
    """Per-row weights matching scikit-learn's ``class_weight`` semantics.

    Returns ``None`` for "no weighting", so a caller can pass the result straight
    through to ``fit(sample_weight=...)`` without branching.
    """
    if class_weight in (None, "none", False):
        return None

    targets = np.asarray(y)
    classes, counts = np.unique(targets, return_counts=True)
    if classes.size < 2:
        raise WeightingError("cannot weight classes with fewer than two present")

    if class_weight == "balanced":
        # sklearn's compute_class_weight, spelled out: n / (k * count).
        weights = targets.size / (classes.size * counts.astype(float))
        lookup = dict(zip(classes.tolist(), weights.tolist(), strict=True))
    elif isinstance(class_weight, dict):
        missing = [value for value in classes.tolist() if value not in class_weight]
        if missing:
            raise WeightingError(
                "class_weight dict is missing weight(s) for class(es) " + str(missing)
            )
        lookup = {key: float(value) for key, value in class_weight.items()}
    else:
        raise WeightingError("unknown class_weight: " + repr(class_weight))

    return np.asarray([lookup[value] for value in targets.tolist()], dtype=float)


def needs_sample_weight_wrapper(estimator: Any) -> bool:
    """True when the estimator takes ``sample_weight`` but not ``class_weight``."""
    try:
        params = estimator.get_params(deep=False)
    except (AttributeError, TypeError):
        return False
    if "class_weight" in params:
        return False
    return hasattr(estimator, "fit")


class ClassWeightedClassifier(BaseEstimator, ClassifierMixin):
    """Give any ``sample_weight``-accepting classifier a ``class_weight`` parameter.

    A real sklearn estimator: constructor arguments are stored unmodified, so it
    clones correctly and cannot carry a fitted model from one fold into the next.

    An explicit ``sample_weight`` passed to :meth:`fit` **multiplies** the class
    weights rather than replacing them. That is the only composition that keeps
    both meanings -- a per-record confidence weight and a per-class imbalance
    correction are different statements about the same row, and picking one to
    discard would silently drop whichever the caller cared about.
    """

    def __init__(
        self,
        estimator: Any = None,
        *,
        class_weight: str | dict[Any, float] | None = "balanced",
    ) -> None:
        self.estimator = estimator
        self.class_weight = class_weight

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None
    ) -> ClassWeightedClassifier:
        if self.estimator is None:
            raise WeightingError("ClassWeightedClassifier needs an estimator to wrap")

        weights = balanced_sample_weight(y, self.class_weight)
        if sample_weight is not None:
            supplied = np.asarray(sample_weight, dtype=float)
            weights = supplied if weights is None else weights * supplied

        self.estimator_ = clone(self.estimator)
        if weights is None:
            self.estimator_.fit(X, y)
        else:
            self.estimator_.fit(X, y, sample_weight=weights)

        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = getattr(
            self.estimator_, "n_features_in_", np.asarray(X).shape[1]
        )
        return self

    def _fitted(self) -> Any:
        inner = getattr(self, "estimator_", None)
        if inner is None:
            raise WeightingError(
                "ClassWeightedClassifier is not fitted yet; call fit first"
            )
        return inner

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self._fitted().predict(X))

    @available_if(_inner_has("predict_proba"))
    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self._fitted().predict_proba(X))

    @available_if(_inner_has("decision_function"))
    def decision_function(self, X: Any) -> np.ndarray:
        """Only present when the wrapped estimator actually has one.

        A wrapper that declares this method unconditionally is a trap. sklearn
        decides how to read a classifier's scores with ``hasattr``, and it
        prefers ``decision_function`` when it finds one -- so an unconditional
        stub makes ``CalibratedClassifierCV`` choose that path and then die on
        ``'XGBClassifier' object has no attribute 'decision_function'`` deep
        inside a fit. ``available_if`` makes ``hasattr`` tell the truth.
        """
        return np.asarray(self._fitted().decision_function(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        """Forwarded so the wrapper is transparent to the importance module."""
        return np.asarray(self._fitted().feature_importances_)

    def __sklearn_tags__(self) -> Any:  # pragma: no cover - sklearn plumbing
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
