"""Model factories -- one unfitted estimator per model id, built from config.

Every model in this project is identified by the id the source documents use
(M1..M9) and defined in ``configs/models.yaml``. This module turns an id into an
unfitted estimator. It deliberately does **not** fit anything, does not touch a
matrix, and does not know what a fold is: those belong to the Phase 43 driver
and the Phase 44 pipeline, and keeping them apart is what lets the same factory
be reused by the smoke run, the search, and the final refit without any of the
three being able to change what the others get.

Two things that are easy to get wrong and are handled here rather than left to
each caller:

* **``predict_proba`` is not optional.** The soft-voting ensemble (M6/M7) fuses
  probability vectors, so a member that only produces hard labels cannot join
  it. :func:`probability_report` checks the actual output -- shape, range, row
  sums, finiteness -- rather than merely checking that the method exists.
* **Deprecated parameters are not silently carried.** ``LogisticRegression``'s
  ``penalty`` was deprecated in scikit-learn 1.8 and is removed in 1.10; the M1
  configuration is written against ``l1_ratio`` instead, and a ``penalty`` left
  in an override is translated with a warning rather than passed through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "IMPLEMENTED_MODELS",
    "EstimatorError",
    "ProbabilityReport",
    "model_spec",
    "model_defaults",
    "build_estimator",
    "make_m1",
    "make_m2",
    "make_m3",
    "make_m4",
    "make_m5",
    "make_m8",
    "Capability",
    "m8_capability",
    "has_predict_proba",
    "probability_report",
    "row_sum_tolerance",
    "assert_probabilities_well_formed",
]

log = get_logger("models.estimators")

#: Ids this module can build today. Grows as Part V proceeds; a caller asking
#: for one that is not here gets a clear error naming the phase that adds it.
IMPLEMENTED_MODELS: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5", "M8")

_ADDED_IN_PHASE: dict[str, str] = {
    "M6": "Phase 50",
    "M7": "Phase 50",
    "M9": "Phase 52 (optional, undecided)",
}

#: Old name -> new name for parameters scikit-learn has renamed under us.
_PENALTY_TO_L1_RATIO: dict[Any, float] = {"l2": 0.0, "l1": 1.0}


class EstimatorError(ValueError):
    """The requested estimator cannot be built as specified."""


# ---------------------------------------------------------------------------
# config access
# ---------------------------------------------------------------------------


def model_spec(model_id: str) -> dict[str, Any]:
    """One model's whole config block."""
    from src.utils.config import load_config

    spec = load_config("models").get("models." + model_id)
    if spec is None:
        raise EstimatorError("no model " + repr(model_id) + " in configs/models.yaml")
    return dict(spec)


def model_defaults(model_id: str) -> dict[str, Any]:
    """The untuned baseline parameters -- what EXP-A1 reports."""
    return dict(model_spec(model_id).get("defaults") or {})


def _global_setting(name: str, fallback: Any) -> Any:
    from src.utils.config import load_config

    value = load_config("models").get("global." + name)
    return fallback if value is None else value


# ---------------------------------------------------------------------------
# M1 -- Logistic Regression (T46.1)
# ---------------------------------------------------------------------------


def make_m1(**overrides: Any) -> Any:
    """Logistic regression with balanced class weights.

    The linear baseline the whole comparison is anchored on: if a 138-dimension
    ensemble cannot beat a logistic regression on the same features, the ensemble
    is not earning its complexity. Balanced class weights because the primary
    binary track is 79% normal, and an unweighted fit on that ratio maximises
    accuracy by under-calling the abnormal class -- which is the one that matters
    for screening (research rule 6).

    ``max_iter`` is 2000, not sklearn's 100. On 138 standardised features lbfgs
    has not converged at 100 and emits a ConvergenceWarning, and a
    non-converged fit is a different model from one iteration to the next.
    """
    from sklearn.linear_model import LogisticRegression

    params = model_defaults("M1")
    params.update(overrides)
    params = _translate_penalty(params)
    params.setdefault("random_state", _global_setting("random_state", 42))
    return LogisticRegression(**params)


def _translate_penalty(params: dict[str, Any]) -> dict[str, Any]:
    """Accept the deprecated ``penalty`` and express it as ``l1_ratio``.

    Not a silent rewrite: anything that still passes ``penalty`` is doing so from
    older code or an older config, and it is told once, loudly, that the value
    has moved. ``penalty=None`` (no regularisation) has no ``l1_ratio`` spelling
    -- it is ``C=inf`` -- so it is refused rather than mistranslated to L2.
    """
    if "penalty" not in params:
        return params

    translated = dict(params)
    penalty = translated.pop("penalty")
    if penalty is None:
        raise EstimatorError(
            "penalty=None has no l1_ratio equivalent; use C=float('inf') for an "
            "unregularised fit"
        )
    if penalty not in _PENALTY_TO_L1_RATIO:
        raise EstimatorError(
            "penalty=" + repr(penalty) + " cannot be translated; set l1_ratio "
            "directly (0.0 == l2, 1.0 == l1, between == elasticnet)"
        )
    ratio = _PENALTY_TO_L1_RATIO[penalty]
    log.warning(
        "LogisticRegression 'penalty' is deprecated in scikit-learn 1.8 and "
        "removed in 1.10; translating penalty=%r to l1_ratio=%s",
        penalty,
        ratio,
    )
    translated["l1_ratio"] = ratio
    return translated


# ---------------------------------------------------------------------------
# M2 -- K-Nearest Neighbours (T46.3)
# ---------------------------------------------------------------------------


def make_m2(**overrides: Any) -> Any:
    """K-nearest neighbours. Optional baseline, and the one distance-based model.

    Two properties make it worth keeping despite being optional. It is the only
    model here whose decision depends directly on distances in the 138-dimension
    feature space, so it is the one that reacts to the scaling and the redundancy
    that Phase 41 measured. And it has **no ``class_weight``**: it cannot be told
    the classes are imbalanced, so its sensitivity is a floor for what the
    feature space gives you before any imbalance handling. That is informative,
    not a defect -- but it does mean M2's sensitivity is not comparable to the
    others' on equal terms, and the write-up has to say so.
    """
    from sklearn.neighbors import KNeighborsClassifier

    params = model_defaults("M2")
    params.update(overrides)
    params.setdefault("n_jobs", _global_setting("n_jobs", -1))
    return KNeighborsClassifier(**params)


# ---------------------------------------------------------------------------
# M3 -- SVM-RBF (T47.1)
# ---------------------------------------------------------------------------


def make_m3(
    *,
    calibrated: bool = True,
    calibration_cv: Any = None,
    calibration_method: str | None = None,
    **overrides: Any,
) -> Any:
    """SVM with an RBF kernel, wrapped for explicit probability calibration.

    ``probability=False`` on the inner SVC, always. sklearn's ``probability=True``
    runs an *internal* 5-fold Platt scaling that is not under our control: its
    folds are not the project's folds, it is not grouped by subject, and its
    ``predict`` and ``predict_proba`` can disagree with each other on the same
    row. Phase 47 replaces it with :class:`~src.models.calibration.CalibratedSVM`,
    where the calibration split is ours, visible, and can be made subject-aware.

    Pass ``calibrated=False`` to get the bare SVC -- useful for timing the
    uncalibrated fit, and for nothing else: an uncalibrated SVC cannot join the
    soft-voting ensemble.

    ``calibration_cv`` takes a materialised split from
    :func:`~src.models.calibration.grouped_calibration_cv`, which is how a
    caller that knows the training fold's subject groups makes the calibrator's
    internal division subject-aware. It has to be supplied per fold -- a factory
    that has never seen the fold cannot build it -- and on D1 fold 0 it is worth
    a measurable amount: see ``svm_calibration_benchmark.csv``.
    """
    from sklearn.svm import SVC

    params = model_defaults("M3")
    params.update(overrides)
    params.setdefault("random_state", _global_setting("random_state", 42))

    # `probability` is not passed through at all. It was deprecated in
    # scikit-learn 1.9 (removed in 1.11) and now defaults to a sentinel, so even
    # passing the harmless `probability=False` raises a FutureWarning on every
    # fit. The config still declares it, because declaring the intent is the
    # point -- it is read here as an assertion and then dropped.
    if params.pop("probability", False):
        raise EstimatorError(
            "M3 must be built with probability=False; sklearn's internal Platt "
            "scaling is ungrouped and uninspectable -- calibrate explicitly via "
            "src.models.calibration instead"
        )

    svc = SVC(**params)
    if not calibrated:
        return svc

    from src.models.calibration import CalibratedSVM, calibration_settings

    settings = calibration_settings()
    return CalibratedSVM(
        estimator=svc,
        method=calibration_method or settings["method"],
        cv=settings["cv"] if calibration_cv is None else calibration_cv,
        class_weight=params.get("class_weight"),
        ensemble=settings["ensemble"],
    )


# ---------------------------------------------------------------------------
# M4 -- Random Forest (T48.1)
# ---------------------------------------------------------------------------


def make_m4(**overrides: Any) -> Any:
    """Random forest. The bagged tree ensemble, and the project's importance source.

    Two reasons it is a mandatory ensemble member rather than another baseline.
    It is the only model here that produces a feature ranking as a by-product of
    being fitted, which is what Phase 81's explainability rests on. And it fails
    differently from the SVM: an RBF kernel measures distance in the whole
    138-dimension space at once, a forest asks a sequence of one-feature
    questions, so the two make uncorrelated mistakes -- which is the entire
    reason averaging them (M6/M7) can beat either.

    ``n_jobs`` comes from config and reaches every fit. Thread count does **not**
    change a forest's output -- verified bit-identical between 1 and all cores on
    the real D1 matrix -- because each tree is seeded independently from
    ``random_state``. That is worth knowing rather than assuming: it is why the
    forest can be parallelised inside a search without breaking research rule 5.
    """
    from sklearn.ensemble import RandomForestClassifier

    params = model_defaults("M4")
    params.update(overrides)
    params.setdefault("random_state", _global_setting("random_state", 42))
    params.setdefault("n_jobs", _global_setting("n_jobs", -1))
    return RandomForestClassifier(**params)


# ---------------------------------------------------------------------------
# M5 -- Gradient Boosting (T49.1)
# ---------------------------------------------------------------------------


def make_m5(*, implementation: str | None = None, **overrides: Any) -> Any:
    """Boosted trees, in whichever of the two scikit-learn implementations config names.

    ``classic`` is ``GradientBoostingClassifier`` -- the exact estimator both
    source documents specify. ``hist`` is ``HistGradientBoostingClassifier``,
    which bins the features first and is roughly six times faster on this matrix.
    T49.1 permits either; ``configs/models.yaml`` chooses, and
    ``gradient_boosting_choice.csv`` records what the choice cost or saved.

    **Classic gradient boosting has no ``class_weight``.** That is not a detail:
    it is the only mandatory ensemble member that cannot be told the primary
    track is 79% normal, and left alone it would be the one model in the
    comparison with no imbalance handling at all -- while still being reported
    beside models that have it. It is therefore wrapped in
    :class:`~src.models.weighting.ClassWeightedClassifier`, which converts
    ``class_weight`` into the per-row ``sample_weight`` the estimator does
    accept, using scikit-learn's own definition of "balanced". The histogram
    implementation has a native ``class_weight`` and is left unwrapped.
    """
    spec = model_spec("M5")
    params = model_defaults("M5")
    params.update(overrides)
    params.setdefault("random_state", _global_setting("random_state", 42))

    kind = (implementation or spec.get("implementation") or "classic").lower()
    if kind not in {"classic", "hist"}:
        raise EstimatorError(
            "unknown M5 implementation " + repr(kind) + "; expected 'classic' or 'hist'"
        )

    class_weight = params.pop("class_weight", "balanced")

    if kind == "hist":
        from sklearn.ensemble import HistGradientBoostingClassifier

        renamed = _to_hist_params(params)
        return HistGradientBoostingClassifier(class_weight=class_weight, **renamed)

    from sklearn.ensemble import GradientBoostingClassifier

    from src.models.weighting import ClassWeightedClassifier

    return ClassWeightedClassifier(
        estimator=GradientBoostingClassifier(**params), class_weight=class_weight
    )


#: The two implementations do not share a parameter vocabulary. Translating is
#: safer than maintaining two config blocks that drift apart, but only for the
#: names that genuinely mean the same thing -- `subsample` has no histogram
#: equivalent and is dropped with a warning rather than mapped onto something
#: that merely sounds similar.
_HIST_RENAMES: dict[str, str] = {
    "n_estimators": "max_iter",
    "min_samples_leaf": "min_samples_leaf",
    "max_depth": "max_depth",
    "learning_rate": "learning_rate",
    "random_state": "random_state",
}


def _to_hist_params(params: dict[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for name, value in params.items():
        if name in _HIST_RENAMES:
            translated[_HIST_RENAMES[name]] = value
        else:
            log.warning(
                "HistGradientBoostingClassifier has no equivalent of %r; dropping it "
                "rather than mapping it onto a parameter that only sounds similar",
                name,
            )
    return translated


# ---------------------------------------------------------------------------
# M8 -- external gradient boosting, behind a capability check (T49.3)
# ---------------------------------------------------------------------------


@dataclass
class Capability:
    """Whether an optional dependency is usable, and the reason if it is not."""

    model_id: str
    available: bool
    backend: str = ""
    version: str = ""
    reason: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "available": self.available,
            "backend": self.backend,
            "version": self.version,
            "reason": self.reason,
        }


def m8_capability() -> Capability:
    """Which external booster is usable: XGBoost, LightGBM, or neither.

    Import is not enough to call a package available. Both of these ship native
    binaries, and the failure this check exists for is the one where the wheel
    installs, the module imports, and the first ``fit`` dies on a missing
    ``libomp`` or an unsupported CPU. So the check builds a tiny model and fits
    it, and reports the exception text as the reason when that fails.
    """
    import numpy as np

    spec = model_spec("M8")
    candidates = [
        ("xgboost", str(spec.get("estimator") or "")),
        ("lightgbm", str(spec.get("fallback_estimator") or "")),
    ]

    reasons: list[str] = []
    for package, path in candidates:
        if not path:
            continue
        try:
            estimator_class = _import_from_path(path)
        except Exception as error:  # noqa: BLE001 - any import failure is a decline
            reasons.append(package + ": " + type(error).__name__ + " " + str(error)[:120])
            continue
        try:
            import importlib

            version = str(getattr(importlib.import_module(package), "__version__", ""))
            rng = np.random.default_rng(42)
            probe_x = rng.normal(size=(40, 4))
            probe_y = (probe_x[:, 0] > 0).astype(int)
            estimator_class(n_estimators=5, **_quiet_params(package)).fit(probe_x, probe_y)
        except Exception as error:  # noqa: BLE001 - a broken binary is a decline
            reasons.append(package + ": " + type(error).__name__ + " " + str(error)[:120])
            continue
        return Capability(
            model_id="M8", available=True, backend=package, version=version
        )

    return Capability(
        model_id="M8",
        available=False,
        reason="; ".join(reasons) or "no estimator path configured",
    )


def _quiet_params(package: str) -> dict[str, Any]:
    """LightGBM prints per-fit chatter unless told not to. XGBoost does not."""
    return {"verbose": -1} if package == "lightgbm" else {}


def _import_from_path(path: str) -> Any:
    import importlib

    module_name, _, attribute = path.rpartition(".")
    if not module_name:
        raise EstimatorError("not an importable path: " + repr(path))
    return getattr(importlib.import_module(module_name), attribute)


def make_m8(**overrides: Any) -> Any:
    """XGBoost, or LightGBM, or a clear refusal naming why neither could be used.

    XGBoost has no ``class_weight`` -- only ``scale_pos_weight``, which is
    binary-only and so useless for the PASCAL and CirCor multiclass tracks -- so
    it is wrapped in :class:`~src.models.weighting.ClassWeightedClassifier` for a
    single weighting mechanism that works on every task. LightGBM has a native
    ``class_weight`` and is left alone.

    ``deterministic`` is **not** passed to XGBoost. It is not an XGBoost
    parameter: the booster accepts it silently and logs ``Parameters:
    {"deterministic"} are not used``.

    Determinism is bought by pinning ``n_jobs`` instead, and that is not
    belt-and-braces. XGBoost draws the ``subsample`` row mask per thread block,
    so with ``subsample < 1.0`` the thread count decides which rows each tree
    sees: measured on 2026-08-27, ``n_jobs=1`` and ``n_jobs=4`` differ by up to
    **0.062 in predicted probability**. ``colsample_bytree`` does not do this and
    neither does ``subsample=1.0``; the whole effect belongs to row subsampling,
    which is in M8's search space and so cannot simply be turned off. A global
    ``n_jobs=-1`` would therefore make M8 produce different numbers on a machine
    with a different core count -- research rule 5, broken silently. M8 declares
    its own integer instead, and this factory refuses to guess one.
    """
    capability = m8_capability()
    if not capability.available:
        raise EstimatorError(
            "M8 is unavailable: " + capability.reason + " -- record this in "
            "outputs/missing_outputs_report.txt rather than substituting a model"
        )

    spec = model_spec("M8")
    path = str(
        spec.get("estimator")
        if capability.backend == "xgboost"
        else spec.get("fallback_estimator")
    )
    estimator_class = _import_from_path(path)

    params = model_defaults("M8")
    params.update(overrides)
    params.setdefault("random_state", _global_setting("random_state", 42))
    # Deliberately NOT falling back to the global n_jobs, which is -1. See the
    # docstring: with subsample < 1.0 the thread count changes the model, and -1
    # resolves to whatever the current machine has.
    if "n_jobs" not in params:
        raise EstimatorError(
            "M8 must declare an explicit integer n_jobs in configs/models.yaml; "
            "the global -1 resolves to the core count of whichever machine runs "
            "it, and XGBoost's subsample draw is per-thread"
        )

    if params.pop("deterministic", False):
        log.info(
            "M8 config declares deterministic=true; XGBoost has no such parameter "
            "and ignores it. Determinism is verified by measurement instead -- see "
            "the Phase 48-49 entry in Docs/note.md"
        )

    class_weight = params.pop("class_weight", "balanced")

    if capability.backend == "lightgbm":
        params.pop("tree_method", None)
        params.setdefault("verbose", -1)
        return estimator_class(class_weight=class_weight, **params)

    from src.models.weighting import ClassWeightedClassifier

    return ClassWeightedClassifier(
        estimator=estimator_class(**params), class_weight=class_weight
    )


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_FACTORIES = {
    "M1": make_m1,
    "M2": make_m2,
    "M3": make_m3,
    "M4": make_m4,
    "M5": make_m5,
    "M8": make_m8,
}


def build_estimator(model_id: str, **overrides: Any) -> Any:
    """An unfitted estimator for ``model_id``, with config defaults applied."""
    factory = _FACTORIES.get(model_id)
    if factory is None:
        phase = _ADDED_IN_PHASE.get(model_id)
        if phase:
            raise EstimatorError(
                model_id + " is not implemented yet; it is built in " + phase
            )
        raise EstimatorError(
            "unknown model id " + repr(model_id) + "; implemented: "
            + ", ".join(IMPLEMENTED_MODELS)
        )
    return factory(**overrides)


# ---------------------------------------------------------------------------
# probability checks (T46.5, T47.5)
# ---------------------------------------------------------------------------


def has_predict_proba(estimator: Any) -> bool:
    """Whether the estimator advertises ``predict_proba``.

    A necessary condition, never a sufficient one -- a wrapper can expose the
    method and still return something the ensemble cannot fuse. Use
    :func:`probability_report` on real output before trusting it.
    """
    return callable(getattr(estimator, "predict_proba", None))


@dataclass
class ProbabilityReport:
    """What a probability matrix actually looks like, measured rather than assumed."""

    n_rows: int
    n_classes: int
    min_value: float
    max_value: float
    max_row_sum_error: float
    n_nan: int
    n_inf: int
    agrees_with_predict: bool | None = None
    #: Tolerance the row sums are judged against, derived from the dtype the
    #: model actually returned. Not a fixed constant: XGBoost returns float32,
    #: whose epsilon is 1.2e-07, so a 1e-9 threshold fails a matrix that is
    #: exactly as correct as float32 can be. See :func:`row_sum_tolerance`.
    row_sum_tolerance: float = 1e-9

    @property
    def in_unit_interval(self) -> bool:
        return self.min_value >= 0.0 and self.max_value <= 1.0

    @property
    def rows_sum_to_one(self) -> bool:
        return self.max_row_sum_error <= self.row_sum_tolerance

    @property
    def is_finite(self) -> bool:
        return self.n_nan == 0 and self.n_inf == 0

    @property
    def is_well_formed(self) -> bool:
        return (
            self.is_finite
            and self.in_unit_interval
            and self.rows_sum_to_one
            and self.agrees_with_predict is not False
        )

    def problems(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.n_nan:
            found.append(str(self.n_nan) + " NaN")
        if self.n_inf:
            found.append(str(self.n_inf) + " Inf")
        if not self.in_unit_interval:
            found.append(
                "values outside [0, 1]: min " + repr(self.min_value)
                + ", max " + repr(self.max_value)
            )
        if not self.rows_sum_to_one:
            found.append(
                "row sums off by up to " + repr(self.max_row_sum_error)
                + " (tolerance " + repr(self.row_sum_tolerance) + ")"
            )
        if self.agrees_with_predict is False:
            found.append("argmax of predict_proba disagrees with predict")
        return tuple(found)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_classes": self.n_classes,
            "proba_min": self.min_value,
            "proba_max": self.max_value,
            "max_row_sum_error": self.max_row_sum_error,
            "row_sum_tolerance": self.row_sum_tolerance,
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "agrees_with_predict": self.agrees_with_predict,
            "well_formed": self.is_well_formed,
        }


def row_sum_tolerance(dtype: Any, n_classes: int) -> float:
    """How far from 1.0 a row of probabilities may legitimately sum.

    Derived from the dtype the model returned, not fixed. XGBoost hands back
    float32, whose epsilon is 1.19e-07; summing ``n_classes`` of them accumulates
    a few multiples of that. A blanket 1e-9 threshold -- reasonable for float64 --
    rejects a float32 matrix that is as exact as its dtype allows, which is a
    failing test that reports a bug in the model instead of in the test.

    Loosening a tolerance to make a test pass is normally the worst thing to do
    in this project. This is the exception that proves the rule: the tolerance
    was wrong, not the assertion, because it demanded precision the number could
    not carry. The float64 case is unchanged at 1e-9 -- roughly 4.5 million times
    its own epsilon -- so nothing real is let through.
    """
    try:
        eps = float(np.finfo(np.dtype(dtype)).eps)
    except (TypeError, ValueError):
        return 1e-9
    return max(1e-9, 8.0 * eps * max(int(n_classes), 1))


def probability_report(
    proba: Any,
    *,
    n_classes: int | None = None,
    y_pred: Any = None,
    classes: Any = None,
) -> ProbabilityReport:
    """Measure a probability matrix. Reports; does not raise.

    ``agrees_with_predict`` is the check that catches a *calibrated* model going
    wrong in the way calibration specifically can: a monotone recalibration of a
    binary score cannot move the argmax, so if ``predict`` and
    ``argmax(predict_proba)`` disagree the two are coming from different fitted
    objects. sklearn's own ``SVC(probability=True)`` is documented to do exactly
    this, which is why M3 does not use it.
    """
    original_dtype = getattr(np.asarray(proba), "dtype", np.dtype(float))
    matrix = np.asarray(proba, dtype=float)
    if matrix.ndim != 2:
        raise EstimatorError(
            "predict_proba must return a 2-D (n_samples, n_classes) array, got shape "
            + str(matrix.shape)
        )
    if n_classes is not None and matrix.shape[1] != n_classes:
        raise EstimatorError(
            "predict_proba returned " + str(matrix.shape[1]) + " columns for "
            + str(n_classes) + " classes"
        )

    finite = np.isfinite(matrix)
    n_nan = int(np.isnan(matrix).sum())
    n_inf = int((~finite & ~np.isnan(matrix)).sum())
    usable = matrix[finite]

    agrees: bool | None = None
    if y_pred is not None and classes is not None and matrix.shape[0]:
        ordering = np.asarray(classes)
        argmax_labels = ordering[np.argmax(matrix, axis=1)]
        agrees = bool(np.array_equal(argmax_labels, np.asarray(y_pred)))

    return ProbabilityReport(
        n_rows=int(matrix.shape[0]),
        n_classes=int(matrix.shape[1]),
        min_value=float(usable.min()) if usable.size else float("nan"),
        max_value=float(usable.max()) if usable.size else float("nan"),
        max_row_sum_error=(
            float(np.max(np.abs(matrix.sum(axis=1) - 1.0))) if finite.all() else float("inf")
        ),
        n_nan=n_nan,
        n_inf=n_inf,
        agrees_with_predict=agrees,
        row_sum_tolerance=row_sum_tolerance(original_dtype, matrix.shape[1]),
    )


def assert_probabilities_well_formed(
    proba: Any,
    *,
    n_classes: int | None = None,
    y_pred: Any = None,
    classes: Any = None,
    context: str = "",
) -> ProbabilityReport:
    """:func:`probability_report`, but raising on anything malformed."""
    report = probability_report(
        proba, n_classes=n_classes, y_pred=y_pred, classes=classes
    )
    if not report.is_well_formed:
        raise EstimatorError(
            (context + ": " if context else "")
            + "malformed probabilities -- "
            + "; ".join(report.problems())
        )
    return report
