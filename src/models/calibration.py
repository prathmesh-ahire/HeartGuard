"""Explicit probability calibration for the SVM (Phase 47).

An RBF SVM produces a signed distance to a hyperplane, not a probability. The
soft-voting ensemble fuses probability vectors, so that distance has to be
turned into one -- and *how* it is turned into one is a modelling decision that
this project makes visibly rather than delegating to a keyword argument.

**Why not ``SVC(probability=True)``.** It fits an internal 5-fold Platt scaling
whose folds are sklearn's, not ours: not grouped by subject, not seeded from the
project seed, and not visible in any output. Worse, sklearn documents that its
``predict`` and ``predict_proba`` can disagree on the same row, because
``predict`` uses the raw decision function while ``predict_proba`` uses the
separately-fitted sigmoid. A member of a voting ensemble whose hard and soft
outputs contradict each other is not a member anyone can reason about.

**What this does instead.** :class:`CalibratedSVM` wraps
``CalibratedClassifierCV`` and adds the two things the bare wrapper is missing
for this project:

* a ``class_weight`` parameter that reaches the *inner* SVC. The Phase 44
  pipeline sets class weights on its final step; without this the weight would
  land on the calibrator, which has no such parameter, and the imbalance
  handling would silently disappear for M3 alone.
* the ability to take a **subject-grouped** calibration split. The calibrator's
  internal CV divides the training fold; PhysioNet has multiple recordings per
  subject, so an ungrouped split puts two recordings of one subject on both
  sides of it. That does not touch the outer test fold -- the reported metric
  stays honest either way -- but it makes the fitted sigmoid slightly optimistic
  about how separable the scores are. :func:`grouped_calibration_cv` builds the
  grouped split from the training fold's own groups.

Both calibration methods required by T47.2 are supported. ``sigmoid`` (Platt) is
the default: it fits two parameters and is stable on a few hundred rows.
``isotonic`` is non-parametric, strictly more flexible, and overfits hard on
small folds -- PASCAL A's training folds are ~99 records, where isotonic
regression on 99 points is fitting noise. :func:`recommended_method` states that
threshold rather than leaving it to whoever writes the next config.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from src.utils.logging_setup import get_logger

__all__ = [
    "CALIBRATION_METHODS",
    "ISOTONIC_MIN_SAMPLES",
    "CalibrationError",
    "CalibratedSVM",
    "calibration_settings",
    "recommended_method",
    "grouped_calibration_cv",
]

log = get_logger("models.calibration")

CALIBRATION_METHODS: tuple[str, ...] = ("sigmoid", "isotonic")

#: Below this many training rows, isotonic calibration is fitting noise. The
#: number is the rule of thumb from Niculescu-Mizil & Caruana (2005), who put
#: isotonic's crossover with Platt scaling at roughly one thousand samples.
ISOTONIC_MIN_SAMPLES = 1000


class CalibrationError(ValueError):
    """The calibration cannot be configured or fitted as specified."""


def calibration_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``calibration`` block of ``configs/models.yaml``."""
    if config is None:
        from src.utils.config import load_config

        config = load_config("models").get("calibration") or {}

    method = str(config.get("method", "sigmoid")).lower()
    if method not in CALIBRATION_METHODS:
        raise CalibrationError(
            "unknown calibration method " + repr(method)
            + "; expected one of " + str(CALIBRATION_METHODS)
        )
    return {
        "method": method,
        "cv": int(config.get("cv", 3)),
        "fit_inside_fold": bool(config.get("fit_inside_fold", True)),
        "ensemble": bool(config.get("ensemble", True)),
    }


def recommended_method(n_train: int) -> str:
    """Which calibrator suits a training fold of this size, and why."""
    return "isotonic" if n_train >= ISOTONIC_MIN_SAMPLES else "sigmoid"


def grouped_calibration_cv(
    groups: Any, y: Any, *, n_splits: int = 3, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """A subject-grouped calibration split, as a materialised index list.

    Returned as a concrete list of ``(train_index, calibration_index)`` pairs
    rather than a splitter object because ``CalibratedClassifierCV.fit`` receives
    no ``groups`` argument -- there is nowhere to pass the subject ids through.
    Precomputing the split is the one way to make the calibrator's internal
    division subject-aware without depending on sklearn's metadata routing.

    Indices are positional **within the array handed to fit**, so this must be
    built from the training fold's own groups, in the training fold's own row
    order. Building it from the full matrix and passing it to a fold's fit is a
    leak; the caller owns that correspondence.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    group_values = np.asarray(groups)
    targets = np.asarray(y)
    if group_values.shape[0] != targets.shape[0]:
        raise CalibrationError(
            "groups and y disagree on length: "
            + str((group_values.shape[0], targets.shape[0]))
        )

    n_groups = len(np.unique(group_values))
    if n_groups < n_splits:
        raise CalibrationError(
            "cannot build a " + str(n_splits) + "-way grouped calibration split "
            "from " + str(n_groups) + " group(s)"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    splits = [
        (train, held)
        for train, held in splitter.split(
            np.zeros((targets.shape[0], 1)), targets, groups=group_values
        )
    ]

    for train, held in splits:
        shared = set(group_values[train].tolist()) & set(group_values[held].tolist())
        if shared:  # pragma: no cover - StratifiedGroupKFold guarantees this
            raise CalibrationError(
                "grouped calibration split leaked " + str(len(shared)) + " group(s)"
            )
    return splits


class CalibratedSVM(BaseEstimator, ClassifierMixin):
    """``CalibratedClassifierCV`` that forwards ``class_weight`` to the inner model.

    Written as a real sklearn estimator -- ``get_params``/``set_params`` from
    ``BaseEstimator``, constructor arguments stored unmodified -- so it clones
    correctly. That matters more than it sounds: the Phase 43 driver builds a
    fresh estimator per fold and the Phase 44 pipeline clones what it is given,
    and a wrapper that mutated its arguments in ``__init__`` would carry state
    between folds while passing every test that only checks outputs.

    Not SVM-specific in its mechanics; named for its only current use.
    """

    def __init__(
        self,
        estimator: Any = None,
        *,
        method: str = "sigmoid",
        cv: Any = 3,
        class_weight: Any = None,
        ensemble: bool = True,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.class_weight = class_weight
        self.ensemble = ensemble

    # -- sklearn plumbing --------------------------------------------------

    def _base_estimator(self) -> Any:
        if self.estimator is None:
            from sklearn.svm import SVC

            return SVC(kernel="rbf", random_state=42)
        return clone(self.estimator)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> CalibratedSVM:
        from sklearn.calibration import CalibratedClassifierCV

        if self.method not in CALIBRATION_METHODS:
            raise CalibrationError(
                "unknown calibration method " + repr(self.method)
                + "; expected one of " + str(CALIBRATION_METHODS)
            )

        inner = self._base_estimator()
        if getattr(inner, "probability", False) is True:
            # `is True` and not truthiness: scikit-learn 1.9 defaults this to
            # the string sentinel "deprecated", which is truthy.
            raise CalibrationError(
                "the inner estimator has probability=True; that fits a second, "
                "hidden calibration whose predict and predict_proba can disagree"
            )
        if self.class_weight is not None and "class_weight" in inner.get_params():
            inner.set_params(class_weight=self.class_weight)

        n_train = int(np.asarray(y).shape[0])
        if self.method == "isotonic" and n_train < ISOTONIC_MIN_SAMPLES:
            log.warning(
                "isotonic calibration on %d training rows is below the %d-row "
                "threshold where it overfits Platt scaling; see recommended_method",
                n_train,
                ISOTONIC_MIN_SAMPLES,
            )

        self.calibrated_ = CalibratedClassifierCV(
            estimator=inner,
            method=self.method,
            cv=self.cv,
            ensemble=self.ensemble,
        )
        self.calibrated_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.calibrated_.classes_
        self.n_features_in_ = getattr(
            self.calibrated_, "n_features_in_", np.asarray(X).shape[1]
        )
        return self

    def _check_fitted(self) -> Any:
        calibrated = getattr(self, "calibrated_", None)
        if calibrated is None:
            raise CalibrationError("CalibratedSVM is not fitted yet; call fit first")
        return calibrated

    def predict(self, X: Any) -> np.ndarray:
        """The argmax of the calibrated probabilities -- deliberately.

        Delegating to the calibrator's own ``predict`` gives the same answer and
        makes the guarantee implicit. Computing it here makes it structural:
        ``predict`` and ``predict_proba`` cannot disagree, because there is only
        one source of truth. This is precisely the property
        ``SVC(probability=True)`` does not have.
        """
        proba = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self._check_fitted().predict_proba(X))

    def decision_function(self, X: Any) -> np.ndarray:
        return np.asarray(self._check_fitted().decision_function(X))

    def __sklearn_tags__(self) -> Any:  # pragma: no cover - sklearn plumbing
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
