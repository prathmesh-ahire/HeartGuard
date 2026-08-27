"""Gradient boosting M5 and the external baseline M8 (Phase 49, gate T49.7).

The gate: smoke-run M5 and M8; if the external baseline is unavailable, confirm
the capability check degrades gracefully and the reason is recorded.

Both models arrive with the same defect and it is not a small one:
``GradientBoostingClassifier`` has no ``class_weight``, and ``XGBClassifier`` has
only ``scale_pos_weight``, which is binary-only. Through this project's pipeline
-- which sets weights by ``set_params(class_weight=...)`` and calls
``fit(X, y)`` with no fit parameters -- both would end up with **no imbalance
handling at all** while sitting in a results table beside models that have it.
The tests below check that the wrapper closing that gap weights identically to
scikit-learn's own definition, rather than merely that it runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models import estimators as est
from src.models import spaces
from src.models import weighting as wt

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture
def imbalanced() -> tuple[np.ndarray, np.ndarray]:
    """Roughly the primary track's 79/21 split, at a size that fits in a test."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + rng.normal(scale=0.7, size=n) > 1.0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# the weighting wrapper
# ---------------------------------------------------------------------------


def test_balanced_weights_match_scikit_learns_own_definition(imbalanced: Any):
    """If these drift apart, a wrapped and an unwrapped model stop being comparable."""
    from sklearn.utils.class_weight import compute_sample_weight

    _, y = imbalanced
    ours = wt.balanced_sample_weight(y, "balanced")
    theirs = compute_sample_weight("balanced", y)
    assert np.allclose(ours, theirs)


def test_no_weighting_returns_none_rather_than_ones(imbalanced: Any):
    _, y = imbalanced
    assert wt.balanced_sample_weight(y, None) is None
    assert wt.balanced_sample_weight(y, "none") is None


def test_a_weight_dict_must_cover_every_present_class(imbalanced: Any):
    _, y = imbalanced
    assert wt.balanced_sample_weight(y, {0: 1.0, 1: 3.0}) is not None
    with pytest.raises(wt.WeightingError, match="missing weight"):
        wt.balanced_sample_weight(y, {0: 1.0})


def test_wrapping_reproduces_a_native_class_weight_model(imbalanced: Any):
    """The wrapper's whole claim: same weights in, same model out."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = imbalanced
    native = RandomForestClassifier(
        n_estimators=50, class_weight="balanced", random_state=42, n_jobs=1
    ).fit(X, y)
    wrapped = wt.ClassWeightedClassifier(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        class_weight="balanced",
    ).fit(X, y)
    assert np.allclose(native.predict_proba(X), wrapped.predict_proba(X))


def test_an_explicit_sample_weight_multiplies_rather_than_replaces(imbalanced: Any):
    from sklearn.ensemble import RandomForestClassifier

    X, y = imbalanced
    supplied = np.full(len(y), 2.0)
    wrapper = wt.ClassWeightedClassifier(
        estimator=RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1),
        class_weight="balanced",
    )
    wrapper.fit(X, y, sample_weight=supplied)
    expected = wt.balanced_sample_weight(y, "balanced") * supplied
    assert expected is not None and expected.max() > wt.balanced_sample_weight(y).max()


def test_the_wrapper_hides_decision_function_when_the_inner_model_has_none():
    """sklearn picks its response method by hasattr; an unconditional stub crashes it."""
    import xgboost
    from sklearn.ensemble import GradientBoostingClassifier

    xgb = wt.ClassWeightedClassifier(estimator=xgboost.XGBClassifier(n_estimators=5))
    gbm = wt.ClassWeightedClassifier(estimator=GradientBoostingClassifier(n_estimators=5))
    assert not hasattr(xgb, "decision_function")
    assert hasattr(gbm, "decision_function")
    assert hasattr(xgb, "predict_proba")


def test_the_wrapper_clones_without_carrying_fitted_state(imbalanced: Any):
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier

    X, y = imbalanced
    original = wt.ClassWeightedClassifier(
        estimator=RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    )
    original.fit(X, y)
    fresh = clone(original)
    assert not hasattr(fresh, "estimator_")
    with pytest.raises(wt.WeightingError, match="not fitted"):
        fresh.predict(X)


# ---------------------------------------------------------------------------
# T49.1 / T49.2 -- M5
# ---------------------------------------------------------------------------


def test_m5_is_wrapped_because_gradient_boosting_has_no_class_weight():
    from sklearn.ensemble import GradientBoostingClassifier

    model = est.make_m5()
    assert isinstance(model, wt.ClassWeightedClassifier)
    assert isinstance(model.estimator, GradientBoostingClassifier)
    assert model.class_weight == "balanced"
    assert "class_weight" not in GradientBoostingClassifier().get_params()


def test_m5_hist_uses_its_native_class_weight_instead():
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = est.make_m5(implementation="hist")
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.class_weight == "balanced"
    assert model.max_iter == est.model_defaults("M5")["n_estimators"]


def test_an_untranslatable_parameter_is_dropped_not_guessed(caplog: Any):
    """`subsample` has no histogram equivalent; mapping it to something similar lies."""
    with caplog.at_level("WARNING", logger="models.estimators"):
        est.make_m5(implementation="hist")
    assert any("subsample" in record.message for record in caplog.records)


def test_an_unknown_m5_implementation_is_refused():
    with pytest.raises(est.EstimatorError, match="expected 'classic' or 'hist'"):
        est.make_m5(implementation="lightning")


def test_m5_space_holds_the_five_declared_dimensions():
    space = spaces.load_space("M5")
    assert set(space.names) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "min_samples_leaf",
    }


# ---------------------------------------------------------------------------
# T49.3 / T49.4 -- M8 and its capability check
# ---------------------------------------------------------------------------


def test_the_capability_check_reports_a_backend_or_a_reason():
    """Never silently absent, and never silently substituted."""
    capability = est.m8_capability()
    assert capability.model_id == "M8"
    if capability.available:
        assert capability.backend in {"xgboost", "lightgbm"}
        assert capability.version
        assert capability.reason == ""
    else:
        assert capability.reason, "an unavailable M8 must say why"
        assert capability.backend == ""


def test_the_capability_check_fits_rather_than_only_importing():
    """These ship native binaries; import success does not mean fit success."""
    import inspect

    source = inspect.getsource(est.m8_capability)
    assert ".fit(" in source, "the probe must actually fit a model"


def test_m8_refuses_clearly_when_unavailable(monkeypatch: Any):
    """The gate's second clause: degrade gracefully, with the reason recorded."""
    monkeypatch.setattr(
        est,
        "m8_capability",
        lambda: est.Capability(
            model_id="M8", available=False, reason="xgboost: ImportError no libomp"
        ),
    )
    with pytest.raises(est.EstimatorError) as caught:
        est.make_m8()
    message = str(caught.value)
    assert "no libomp" in message, "the technical reason must survive to the caller"
    assert "missing_outputs_report" in message
    assert "substituting" in message


def test_m8_space_holds_the_six_declared_dimensions():
    space = spaces.load_space("M8")
    assert set(space.names) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    }


def test_m8_does_not_pass_the_non_existent_deterministic_flag_to_xgboost():
    """XGBoost accepts it silently and logs `Parameters: {...} are not used`."""
    capability = est.m8_capability()
    if not capability.available or capability.backend != "xgboost":
        pytest.skip("XGBoost is not the active M8 backend")

    assert est.model_defaults("M8").get("deterministic") is True
    model = est.make_m8()
    inner = model.estimator if isinstance(model, wt.ClassWeightedClassifier) else model
    assert "deterministic" not in inner.get_params()


def test_m8_is_wrapped_for_multiclass_capable_weighting():
    capability = est.m8_capability()
    if not capability.available:
        pytest.skip("M8 unavailable: " + capability.reason)
    model = est.make_m8()
    if capability.backend == "xgboost":
        assert isinstance(model, wt.ClassWeightedClassifier), (
            "scale_pos_weight is binary-only and cannot serve the multiclass tracks"
        )
    else:
        assert model.get_params()["class_weight"] == "balanced"


# ---------------------------------------------------------------------------
# T49.5 -- probability behaviour
# ---------------------------------------------------------------------------


def test_m5_and_m8_probabilities_are_well_formed(imbalanced: Any):
    X, y = imbalanced
    for model_id in ("M5", "M8"):
        try:
            model = est.build_estimator(model_id, n_estimators=20)
        except est.EstimatorError as error:
            pytest.skip(model_id + " unavailable: " + str(error))
        fitted = model.fit(X, y)
        proba = fitted.predict_proba(X)
        est.assert_probabilities_well_formed(
            proba, n_classes=2, y_pred=fitted.predict(X),
            classes=fitted.classes_, context=model_id,
        )


def test_m5_and_m8_are_reproducible(imbalanced: Any):
    X, y = imbalanced
    for model_id in ("M5", "M8"):
        try:
            first = est.build_estimator(model_id, n_estimators=20).fit(X, y)
            second = est.build_estimator(model_id, n_estimators=20).fit(X, y)
        except est.EstimatorError as error:
            pytest.skip(model_id + " unavailable: " + str(error))
        assert np.array_equal(first.predict_proba(X), second.predict_proba(X)), model_id


def test_m8_pins_its_thread_count_rather_than_inheriting_the_global(imbalanced: Any):
    """XGBoost's `subsample` draw is per thread block, so n_jobs changes the model.

    Measured 2026-08-27: with `subsample=0.9`, one thread against four differ by
    up to 0.062 in predicted probability -- a different model, not float noise.
    `colsample_bytree` alone does not do it and `subsample=1.0` does not either.
    A global `n_jobs=-1` resolves to the running machine's core count, so M8
    would produce different numbers here and in CI. Rule 5 requires the pin.
    """
    capability = est.m8_capability()
    if not capability.available:
        pytest.skip("M8 unavailable: " + capability.reason)

    configured = est.model_defaults("M8").get("n_jobs")
    assert isinstance(configured, int) and configured > 0, (
        "M8 must declare a fixed positive n_jobs, got " + repr(configured)
    )

    X, y = imbalanced
    first = est.make_m8(n_estimators=60).fit(X, y).predict_proba(X)
    second = est.make_m8(n_estimators=60).fit(X, y).predict_proba(X)
    assert np.array_equal(first, second)


def test_the_thread_sensitivity_that_forced_the_pin_is_real(imbalanced: Any):
    """The measurement the pin rests on -- so a future n_jobs=-1 fails loudly here."""
    capability = est.m8_capability()
    if not capability.available or capability.backend != "xgboost":
        pytest.skip("XGBoost is not the active M8 backend")

    import xgboost

    X, y = imbalanced

    def fit(n_jobs: int, **kwargs: Any) -> np.ndarray:
        model = xgboost.XGBClassifier(
            n_estimators=60, tree_method="hist", n_jobs=n_jobs,
            random_state=42, **kwargs,
        )
        return model.fit(X, y).predict_proba(X)

    assert np.array_equal(fit(1), fit(4)), "no subsampling should be thread-stable"
    assert not np.array_equal(fit(1, subsample=0.9), fit(4, subsample=0.9)), (
        "if this now passes, XGBoost has changed its subsample RNG and the "
        "n_jobs pin can be revisited -- do not just delete the assertion"
    )


# ---------------------------------------------------------------------------
# T49.6 / T49.7 -- the gate, on real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def boosting_on_fold_zero() -> dict[str, Any]:
    from src.models import smoke as sm

    try:
        data = sm.load_task_data("binary")
        fold = sm._fold_zero("binary", data)
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 fold 0 unavailable (" + type(error).__name__ + "): " + str(error))

    results: dict[str, Any] = {}
    for model_id in ("M5", "M8"):
        try:
            results[model_id] = sm.smoke_fold(
                model_id, lambda mid=model_id: est.build_estimator(mid), data, fold
            )
        except est.EstimatorError as error:
            results[model_id] = error
    return results


def test_m5_smoke_runs_clean_on_the_real_matrix(boosting_on_fold_zero: Any):
    result = boosting_on_fold_zero["M5"]
    assert not isinstance(result, Exception), "M5 is mandatory and must run"
    assert result.probability["n_nan"] == 0
    assert result.probability["well_formed"] is True
    assert result.model_bytes > 0
    for name, value in result.metrics.items():
        assert np.isfinite(value), "M5." + name + " is " + str(value)


def test_m8_either_runs_or_declines_with_a_recorded_reason(boosting_on_fold_zero: Any):
    """The gate: both outcomes are acceptable; silence is not."""
    result = boosting_on_fold_zero["M8"]
    if isinstance(result, Exception):
        assert "M8 is unavailable" in str(result)
        assert "missing_outputs_report" in str(result)
        return
    assert result.probability["n_nan"] == 0
    assert result.probability["well_formed"] is True
    assert result.model_bytes > 0
    for name, value in result.metrics.items():
        assert np.isfinite(value), "M8." + name + " is " + str(value)


def test_neither_booster_scores_suspiciously_well(boosting_on_fold_zero: Any):
    for model_id, result in boosting_on_fold_zero.items():
        if isinstance(result, Exception):
            continue
        balanced = result.metrics["balanced_accuracy"]
        assert 0.5 < balanced < 0.95, (
            model_id + " scored " + str(round(balanced, 4)) + " on one untuned fold; "
            "a near-perfect result is a leak until proven otherwise"
        )
