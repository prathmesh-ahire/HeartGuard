"""Baseline models M1 and M2 (Phase 46, gate T46.7).

The gate: smoke-run M1 and M2 on D1 fold 0; confirm both expose
``predict_proba`` and produce no NaN.

Run against the **real** feature matrix, not a synthetic blob. A synthetic
Gaussian says nothing about a matrix with seven constant-by-construction columns,
features spanning fifteen orders of magnitude, and the specific NaN pattern FE-04
itemises -- and those are precisely what a scaler and a distance metric react to.
The real-data tests skip cleanly when FE-03 is absent so the suite still runs on
a fresh clone; the space and factory tests never need it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models import estimators as est
from src.models import spaces

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# ---------------------------------------------------------------------------
# T46.1 / T46.3 -- the factories
# ---------------------------------------------------------------------------


def test_m1_is_a_balanced_logistic_regression():
    model = est.make_m1()
    params = model.get_params()
    assert type(model).__name__ == "LogisticRegression"
    assert params["class_weight"] == "balanced"
    assert params["random_state"] == 42
    assert params["max_iter"] >= 1000, "138 features do not converge at sklearn's 100"


def test_m1_does_not_carry_the_deprecated_penalty_parameter():
    """`penalty` was deprecated in scikit-learn 1.8 and is removed in 1.10."""
    model = est.make_m1()
    assert "penalty" not in est.model_defaults("M1")
    assert model.get_params()["l1_ratio"] == 0.0  # 0.0 is the old penalty="l2"


def test_a_supplied_penalty_is_translated_not_passed_through():
    assert est.make_m1(penalty="l1").get_params()["l1_ratio"] == 1.0
    assert est.make_m1(penalty="l2").get_params()["l1_ratio"] == 0.0
    with pytest.raises(est.EstimatorError, match="C=float"):
        est.make_m1(penalty=None)


def test_m2_is_knn_and_has_no_class_weight():
    """Not a defect -- a documented asymmetry the write-up has to state."""
    model = est.make_m2()
    assert type(model).__name__ == "KNeighborsClassifier"
    assert "class_weight" not in model.get_params()


def test_an_unimplemented_model_names_the_phase_that_builds_it():
    """Update the id here as phases land -- never the assertion."""
    with pytest.raises(est.EstimatorError, match="Phase 50"):
        est.build_estimator("M6")
    with pytest.raises(est.EstimatorError, match="unknown model id"):
        est.build_estimator("M99")


def test_the_implemented_list_matches_what_can_actually_be_built():
    """Stops IMPLEMENTED_MODELS drifting away from the factory table."""
    for model_id in est.IMPLEMENTED_MODELS:
        try:
            est.build_estimator(model_id)
        except est.EstimatorError as error:
            # An optional model whose package is absent is a legitimate decline;
            # a missing factory is not.
            assert "unavailable" in str(error), model_id + ": " + str(error)


def test_both_baselines_are_deterministic_across_two_builds():
    """Rule 5: two runs of the same command produce identical numbers."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 10))
    y = (X[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    for model_id in ("M1", "M2"):
        first = est.build_estimator(model_id).fit(X, y).predict_proba(X)
        second = est.build_estimator(model_id).fit(X, y).predict_proba(X)
        assert np.array_equal(first, second), model_id + " is not reproducible"


# ---------------------------------------------------------------------------
# T46.2 / T46.4 -- the search spaces
# ---------------------------------------------------------------------------


def test_m1_space_holds_the_four_declared_dimensions():
    space = spaces.load_space("M1")
    assert set(space.names) == {"C", "l1_ratio", "solver", "max_iter"}
    assert space.dimension("C").kind == "log_uniform"


def test_m2_space_holds_the_four_declared_dimensions():
    space = spaces.load_space("M2")
    assert set(space.names) == {"n_neighbors", "weights", "metric", "p"}
    assert space.dimension("n_neighbors").kind == "int_uniform"


def test_yaml_bounds_parsed_as_numbers_not_strings():
    """PyYAML is YAML 1.1: `1.0e3` without a signed exponent parses as a STRING.

    Four bounds in configs/models.yaml were silently strings until 2026-08-27.
    This is the guard that stops it recurring in any model's space.
    """
    for model_id in ("M1", "M2", "M3", "M4", "M5", "M8"):
        for dimension in spaces.load_space(model_id).dimensions:
            for bound in (dimension.low, dimension.high):
                assert not isinstance(bound, str), (
                    model_id + "." + dimension.name + " bound is a string; "
                    "write the exponent signed, e.g. 1.0e+3"
                )


def test_m1_rejects_solver_penalty_combinations_sklearn_would_refuse():
    space = spaces.load_space("M1")
    assert not space.is_valid({"solver": "lbfgs", "l1_ratio": 1.0})
    assert not space.is_valid({"solver": "liblinear", "l1_ratio": 0.5})
    assert space.is_valid({"solver": "saga", "l1_ratio": 0.5})
    assert space.is_valid({"solver": "lbfgs", "l1_ratio": 0.0})


def test_every_sampled_m1_point_actually_fits():
    """The constraint is only worth having if it matches what sklearn accepts."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    X = StandardScaler().fit_transform(rng.normal(size=(150, 6)))
    y = (X[:, 0] > 0).astype(int)

    space = spaces.load_space("M1")
    for params in space.sample_many(25, 42):
        est.make_m1(**params).fit(X, y)  # must not raise


def test_m2_pins_p_off_the_minkowski_metric():
    space = spaces.load_space("M2")
    repaired = space.repair({"metric": "chebyshev", "p": 1, "n_neighbors": 5})
    assert repaired["p"] == 2
    for params in space.sample_many(50, 7):
        if params["metric"] != "minkowski":
            assert params["p"] == 2


def test_sampling_is_reproducible_from_the_seed():
    space = spaces.load_space("M1")
    assert space.sample_many(10, 42) == space.sample_many(10, 42)


def test_a_malformed_dimension_is_refused_at_load():
    with pytest.raises(spaces.SpaceError, match="unknown dimension kind"):
        spaces.Dimension(name="C", kind="gaussian", low=0.0, high=1.0)
    with pytest.raises(spaces.SpaceError, match="strictly positive"):
        spaces.Dimension(name="C", kind="log_uniform", low=0.0, high=1.0)
    with pytest.raises(spaces.SpaceError, match="non-empty choices"):
        spaces.Dimension(name="solver", kind="categorical")


# ---------------------------------------------------------------------------
# T46.5 / T46.7 -- the gate, on real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fold_zero() -> tuple[Any, Any]:
    """D1's binary task and its repeat-0 fold-0 split, or a skip."""
    from src.models import smoke as sm

    try:
        data = sm.load_task_data("binary")
        fold = sm._fold_zero("binary", data)
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 fold 0 unavailable (" + type(error).__name__ + "): " + str(error))
    return data, fold


@pytest.fixture(scope="module")
def smoke_results(fold_zero: tuple[Any, Any]) -> dict[str, Any]:
    from src.models import smoke as sm

    data, fold = fold_zero
    return {
        model_id: sm.smoke_fold(
            model_id, lambda mid=model_id: est.build_estimator(mid), data, fold
        )
        for model_id in ("M1", "M2")
    }


def test_the_real_matrix_is_the_one_the_audit_describes(fold_zero: tuple[Any, Any]):
    data, fold = fold_zero
    assert data.n_features == 138
    assert data.n_records == 3240, "D1 is 3,240 records; see Docs/note.md"
    assert fold.label == "r0f0"
    assert fold.n_train + fold.n_test == data.n_records


def test_both_baselines_expose_predict_proba(smoke_results: dict[str, Any]):
    """The gate's first clause."""
    for model_id, result in smoke_results.items():
        assert result.probability["has_predict_proba"] is True, model_id


def test_neither_baseline_produces_a_nan(smoke_results: dict[str, Any]):
    """The gate's second clause -- in the metrics and in the probabilities."""
    for model_id, result in smoke_results.items():
        assert result.probability["n_nan"] == 0, model_id
        assert result.probability["n_inf"] == 0, model_id
        for name in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "pr_auc",
            "mcc",
        ):
            value = result.metrics[name]
            assert np.isfinite(value), model_id + "." + name + " is " + str(value)


def test_probabilities_are_well_formed(smoke_results: dict[str, Any]):
    for model_id, result in smoke_results.items():
        assert result.probability["well_formed"] is True, model_id
        assert result.probability["proba_min"] >= 0.0
        assert result.probability["proba_max"] <= 1.0
        assert result.probability["max_row_sum_error"] <= 1e-9
        assert result.probability["agrees_with_predict"] is True


def test_no_subject_appears_in_both_sides_of_fold_zero(fold_zero: tuple[Any, Any]):
    """Research rule 3, checked against the rows the model was actually given."""
    data, fold = fold_zero
    train = set(data.groups[fold.train_index].tolist())
    test = set(data.groups[fold.test_index].tolist())
    assert not (train & test)


def test_the_smoke_metrics_are_plausible_not_perfect(smoke_results: dict[str, Any]):
    """A near-1.0 metric on this corpus is a leak until proven otherwise.

    The ceiling is deliberately below the published state of the art for
    PhysioNet 2016 (~0.95 balanced accuracy with far heavier machinery). An
    untuned baseline on one fold reaching that would mean something is wrong
    with the fold, not that the baseline is excellent.
    """
    for model_id, result in smoke_results.items():
        balanced = result.metrics["balanced_accuracy"]
        assert 0.5 < balanced < 0.95, (
            model_id + " scored " + str(round(balanced, 4)) + " balanced accuracy on "
            "one untuned fold; investigate before recording it"
        )
