"""SVM-RBF (M3) and its explicit calibration (Phase 47, gate T47.7).

The gate: confirm M3's calibrated probabilities lie in [0, 1] and sum to 1, and
record the fit time so the search budget can be planned.

The probability checks are the easy half. The half worth writing carefully is
everything around them -- that the inner SVC never fits sklearn's hidden Platt
scaling, that ``class_weight`` survives the calibration wrapper instead of
landing on an object that ignores it, that the calibration split can be made
subject-aware, and that ``predict`` and ``predict_proba`` cannot disagree. Each
of those is a way for a calibrated model to look completely healthy while being
the wrong model.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.svm import SVC

from src.models import calibration as cal
from src.models import estimators as est

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An imbalanced two-class problem with repeated subjects, like the real one."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + rng.normal(scale=0.8, size=n) > 0.9).astype(int)
    groups = np.array(["s" + str(index // 3) for index in range(n)])
    return X, y, groups


# ---------------------------------------------------------------------------
# T47.1 -- the estimator
# ---------------------------------------------------------------------------


def test_m3_is_a_calibrated_wrapper_around_an_rbf_svc():
    model = est.make_m3()
    assert isinstance(model, cal.CalibratedSVM)
    inner = model.estimator
    assert isinstance(inner, SVC)
    assert inner.kernel == "rbf"
    assert inner.random_state == 42


def test_the_inner_svc_never_gets_sklearns_own_probability_fitting(toy: Any):
    """`probability=True` fits a second, hidden, ungrouped calibration."""
    X, y, _ = toy
    with pytest.raises(est.EstimatorError, match="probability=False"):
        est.make_m3(probability=True)
    with pytest.raises(cal.CalibrationError, match="probability=True"):
        cal.CalibratedSVM(estimator=SVC(probability=True)).fit(X, y)


def test_building_and_fitting_m3_emits_no_deprecation_warning(toy: Any):
    """`probability` is deprecated in 1.9; passing even False warns on every fit."""
    X, y, _ = toy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        est.make_m3().fit(X[:120], y[:120])
    messages = [str(entry.message) for entry in caught if "probability" in str(entry.message)]
    assert not messages, messages


def test_calibrated_false_returns_the_bare_svc():
    bare = est.make_m3(calibrated=False)
    assert isinstance(bare, SVC)
    assert not est.has_predict_proba(bare) or not hasattr(bare, "predict_proba")


# ---------------------------------------------------------------------------
# T47.2 -- class weight and the calibration split
# ---------------------------------------------------------------------------


def test_class_weight_reaches_the_inner_svc_not_the_calibrator(toy: Any):
    """Without the forwarding, M3 alone would silently lose its imbalance handling."""
    X, y, _ = toy
    fitted = cal.CalibratedSVM(
        estimator=SVC(kernel="rbf", random_state=42), class_weight="balanced"
    ).fit(X, y)
    inner = fitted.calibrated_.calibrated_classifiers_[0].estimator
    assert inner.class_weight == "balanced"


def test_the_pipeline_can_set_class_weight_on_the_wrapper(toy: Any):
    """`build_pipeline` calls set_params(class_weight=...) on its final step."""
    from src.models import pipeline as pl

    X, y, _ = toy
    assert pl.supports_class_weight(est.make_m3())
    built = pl.build_pipeline(est.make_m3(class_weight=None), y=y)
    assert built.named_steps["estimator"].class_weight == "balanced"
    built.fit(X, y)


def test_both_calibration_methods_are_available(toy: Any):
    X, y, _ = toy
    for method in cal.CALIBRATION_METHODS:
        fitted = est.make_m3(calibration_method=method).fit(X, y)
        assert fitted.method == method
        proba = fitted.predict_proba(X[:20])
        est.assert_probabilities_well_formed(proba, n_classes=2, context=method)


def test_isotonic_on_a_small_fold_says_so_rather_than_failing_quietly(
    toy: Any, caplog: Any
):
    """PASCAL A's training folds are ~99 rows; isotonic there is fitting noise."""
    X, y, _ = toy
    with caplog.at_level("WARNING", logger="models.calibration"):
        est.make_m3(calibration_method="isotonic").fit(X[:200], y[:200])
    assert any("isotonic" in record.message for record in caplog.records), (
        "a below-threshold isotonic fit must be logged, not silent"
    )
    assert cal.recommended_method(cal.ISOTONIC_MIN_SAMPLES - 1) == "sigmoid"
    assert cal.recommended_method(cal.ISOTONIC_MIN_SAMPLES) == "isotonic"


def test_the_grouped_calibration_split_keeps_subjects_on_one_side(toy: Any):
    _, y, groups = toy
    splits = cal.grouped_calibration_cv(groups, y, n_splits=3, seed=42)
    assert len(splits) == 3
    for train, held in splits:
        assert not (set(groups[train].tolist()) & set(groups[held].tolist()))
        assert not (set(train.tolist()) & set(held.tolist()))
    covered = np.concatenate([held for _, held in splits])
    assert sorted(covered.tolist()) == list(range(len(y)))


def test_m3_accepts_a_grouped_calibration_split(toy: Any):
    X, y, groups = toy
    splits = cal.grouped_calibration_cv(groups, y, n_splits=3, seed=42)
    fitted = est.make_m3(calibration_cv=splits).fit(X, y)
    est.assert_probabilities_well_formed(
        fitted.predict_proba(X), n_classes=2, context="grouped"
    )


def test_a_grouped_split_is_refused_when_there_are_too_few_groups(toy: Any):
    _, y, _ = toy
    with pytest.raises(cal.CalibrationError, match="grouped calibration split"):
        cal.grouped_calibration_cv(np.array(["a"] * len(y)), y, n_splits=3)


# ---------------------------------------------------------------------------
# sklearn contract
# ---------------------------------------------------------------------------


def test_the_wrapper_clones_without_carrying_fitted_state(toy: Any):
    """The CV driver builds one per fold; a wrapper that mutates its own args leaks."""
    from sklearn.base import clone

    X, y, _ = toy
    original = est.make_m3()
    original.fit(X, y)
    fresh = clone(original)
    assert not hasattr(fresh, "calibrated_")
    assert fresh.get_params(deep=False)["method"] == original.method
    with pytest.raises(cal.CalibrationError, match="not fitted"):
        fresh.predict_proba(X)


def test_predict_cannot_disagree_with_predict_proba(toy: Any):
    """The property `SVC(probability=True)` is documented NOT to have."""
    X, y, _ = toy
    fitted = est.make_m3().fit(X, y)
    proba = fitted.predict_proba(X)
    predicted = fitted.predict(X)
    assert np.array_equal(np.asarray(fitted.classes_)[proba.argmax(axis=1)], predicted)


def test_two_fits_of_m3_produce_identical_probabilities(toy: Any):
    X, y, _ = toy
    first = est.make_m3().fit(X, y).predict_proba(X)
    second = est.make_m3().fit(X, y).predict_proba(X)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# T47.5 -- the probability report itself
# ---------------------------------------------------------------------------


def test_the_probability_report_catches_what_it_claims_to():
    good = np.array([[0.2, 0.8], [0.6, 0.4]])
    assert est.probability_report(good).is_well_formed

    assert not est.probability_report(np.array([[0.2, 0.9], [0.6, 0.4]])).rows_sum_to_one
    assert not est.probability_report(np.array([[-0.1, 1.1], [0.6, 0.4]])).in_unit_interval
    assert not est.probability_report(np.array([[np.nan, 1.0], [0.6, 0.4]])).is_finite

    disagreeing = est.probability_report(
        good, n_classes=2, y_pred=np.array([0, 0]), classes=np.array([0, 1])
    )
    assert disagreeing.agrees_with_predict is False
    assert not disagreeing.is_well_formed

    with pytest.raises(est.EstimatorError, match="2-D"):
        est.probability_report(np.array([0.2, 0.8]))
    with pytest.raises(est.EstimatorError, match="malformed"):
        est.assert_probabilities_well_formed(np.array([[0.2, 0.9]]))


# ---------------------------------------------------------------------------
# T47.6 / T47.7 -- the gate, on real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def m3_on_fold_zero() -> Any:
    from src.models import smoke as sm

    try:
        data = sm.load_task_data("binary")
        fold = sm._fold_zero("binary", data)
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 fold 0 unavailable (" + type(error).__name__ + "): " + str(error))
    return sm.smoke_fold("M3", lambda: est.build_estimator("M3"), data, fold)


def test_m3_calibrated_probabilities_lie_in_the_unit_interval(m3_on_fold_zero: Any):
    """The gate's first clause, on the real 138-feature matrix."""
    probability = m3_on_fold_zero.probability
    assert probability["has_predict_proba"] is True
    assert probability["proba_min"] >= 0.0
    assert probability["proba_max"] <= 1.0
    assert probability["n_nan"] == 0
    assert probability["n_inf"] == 0


def test_m3_calibrated_probabilities_sum_to_one_per_row(m3_on_fold_zero: Any):
    """The gate's second clause. The tolerance is float noise, not a slack budget."""
    assert m3_on_fold_zero.probability["max_row_sum_error"] <= 1e-9
    assert m3_on_fold_zero.probability["well_formed"] is True


def test_m3_fit_time_is_recorded(m3_on_fold_zero: Any):
    """The gate's third clause -- the number the search budget is planned from."""
    assert m3_on_fold_zero.fit_seconds > 0.0
    assert np.isfinite(m3_on_fold_zero.fit_seconds)


def test_m3_produces_no_nan_metric_on_the_real_fold(m3_on_fold_zero: Any):
    for name, value in m3_on_fold_zero.metrics.items():
        assert np.isfinite(value), name + " is " + str(value)


def test_m3_does_not_score_suspiciously_well(m3_on_fold_zero: Any):
    balanced = m3_on_fold_zero.metrics["balanced_accuracy"]
    assert 0.5 < balanced < 0.95, (
        "M3 scored " + str(round(balanced, 4)) + " on one untuned fold; a near-perfect "
        "result is a leak until proven otherwise"
    )
