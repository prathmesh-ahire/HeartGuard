"""Random Forest M4 and the two importance measures (Phase 48, gate T48.7).

The gate: smoke-run M4; confirm impurity and permutation importances are both
produced and that permutation importance is computed on the held-out fold only.

That last clause is the one worth real machinery. Permutation importance run on
training rows produces a table that looks completely normal -- same shape, same
column names, plausible magnitudes -- and reports how much the forest relies on
each feature to reproduce data it has already memorised. Nothing about the output
reveals which rows it used. So the tests below check the *inputs*: that the
function refuses to run without an explicit held-out index, and that it refuses
outright when the rows it is given overlap the ones the model was trained on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models import estimators as est
from src.models import importance as imp
from src.models import spaces

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A separable problem where only the first two columns carry signal."""
    rng = np.random.default_rng(42)
    n = 240
    X = rng.normal(size=(n, 10))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.4, size=n) > 0.5).astype(int)
    train = np.arange(0, 160)
    test = np.arange(160, n)
    return X, y, train, test


# ---------------------------------------------------------------------------
# T48.1 / T48.2 -- the estimator and its space
# ---------------------------------------------------------------------------


def test_m4_is_a_seeded_random_forest_with_n_jobs_from_config():
    model = est.make_m4()
    params = model.get_params()
    assert type(model).__name__ == "RandomForestClassifier"
    assert params["random_state"] == 42
    assert params["n_jobs"] is not None
    assert params["class_weight"] == "balanced"


def test_m4_space_holds_the_five_declared_dimensions():
    space = spaces.load_space("M4")
    assert set(space.names) == {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    }
    # max_depth's "no limit" is None, not a large number; it must survive the
    # categorical round trip rather than being coerced to the string "None".
    assert None in space.dimension("max_depth").choices


def test_every_sampled_m4_point_actually_fits(toy: Any):
    X, y, train, _ = toy
    space = spaces.load_space("M4")
    for params in space.sample_many(8, 42):
        small = {key: value for key, value in params.items() if key != "n_estimators"}
        est.make_m4(n_estimators=10, **small).fit(X[train], y[train])


def test_thread_count_does_not_change_the_forest(toy: Any):
    """Rule 5: parallelising a search must not change what it finds."""
    X, y, train, test = toy
    single = est.make_m4(n_estimators=60, n_jobs=1).fit(X[train], y[train])
    parallel = est.make_m4(n_estimators=60, n_jobs=-1).fit(X[train], y[train])
    assert np.array_equal(single.predict_proba(X[test]), parallel.predict_proba(X[test]))


# ---------------------------------------------------------------------------
# T48.3 -- impurity importance
# ---------------------------------------------------------------------------


def test_impurity_importance_is_produced_and_named(toy: Any):
    X, y, train, _ = toy
    names = tuple("f" + str(index) for index in range(X.shape[1]))
    fitted = est.make_m4(n_estimators=60).fit(X[train], y[train])

    result = imp.impurity_importance(fitted, names, model_id="M4")
    assert result.kind == "impurity"
    assert result.computed_on == "train"
    assert result.values.size == len(names)
    assert set(dict(result.top(2))) <= {"f0", "f1"}, "the planted signal should rank first"


def test_impurity_importance_comes_from_the_forest_not_its_prototype_tree(toy: Any):
    """`RandomForestClassifier.estimator_` is the unfitted template it clones.

    A generic "follow estimator_ until it stops" unwrapper walks past the fitted
    forest into that prototype, which has no importances -- the exact bug this
    module hit on 2026-08-27.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, train, _ = toy
    forest = est.make_m4(n_estimators=40)
    built = Pipeline([("scaler", StandardScaler()), ("estimator", forest)])
    built.fit(X[train], y[train])

    final = imp._final_estimator(built)
    assert type(final).__name__ == "RandomForestClassifier"
    assert final.n_estimators == 40


def test_impurity_importance_refuses_a_model_that_has_none(toy: Any):
    X, y, train, _ = toy
    names = tuple("f" + str(index) for index in range(X.shape[1]))
    knn = est.make_m2().fit(X[train], y[train])
    with pytest.raises(imp.ImportanceError, match="no feature_importances_"):
        imp.impurity_importance(knn, names)


# ---------------------------------------------------------------------------
# T48.4 / T48.7 -- permutation importance, held-out only
# ---------------------------------------------------------------------------


def test_permutation_importance_is_produced_on_held_out_rows(toy: Any):
    X, y, train, test = toy
    names = tuple("f" + str(index) for index in range(X.shape[1]))
    fitted = est.make_m4(n_estimators=60).fit(X[train], y[train])

    result = imp.permutation_importance(
        fitted, X, y,
        held_out_index=test, train_index=train,
        feature_names=names, model_id="M4", n_repeats=4,
    )
    assert result.kind == "permutation"
    assert result.computed_on == "held-out fold"
    assert result.n_scored_rows == test.size
    assert result.std is not None
    assert set(dict(result.top(2))) <= {"f0", "f1"}


def test_permutation_importance_has_no_default_row_set():
    """It cannot be called without saying which rows to score. The gate's core."""
    import inspect

    signature = inspect.signature(imp.permutation_importance)
    parameter = signature.parameters["held_out_index"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_permutation_importance_refuses_rows_the_model_was_trained_on(toy: Any):
    """The failure this whole module exists to prevent."""
    X, y, train, _ = toy
    names = tuple("f" + str(index) for index in range(X.shape[1]))
    fitted = est.make_m4(n_estimators=30).fit(X[train], y[train])

    with pytest.raises(imp.ImportanceError, match="trained on"):
        imp.permutation_importance(
            fitted, X, y,
            held_out_index=train, train_index=train,
            feature_names=names, n_repeats=2,
        )
    with pytest.raises(imp.ImportanceError, match="trained on"):
        imp.permutation_importance(
            fitted, X, y,
            held_out_index=np.arange(len(y)), train_index=train,
            feature_names=names, n_repeats=2,
        )


def test_permutation_importance_scores_balanced_accuracy_by_default():
    """Accuracy would under-rank the features that find the minority class."""
    import inspect

    assert inspect.signature(imp.permutation_importance).parameters[
        "scoring"
    ].default == "balanced_accuracy"


def test_permutation_and_impurity_disagree_on_training_data_by_construction(toy: Any):
    """Both must exist; they are different measurements, not two spellings of one."""
    X, y, train, test = toy
    names = tuple("f" + str(index) for index in range(X.shape[1]))
    fitted = est.make_m4(n_estimators=60).fit(X[train], y[train])

    impurity = imp.impurity_importance(fitted, names, model_id="M4")
    permutation = imp.permutation_importance(
        fitted, X, y, held_out_index=test, train_index=train,
        feature_names=names, model_id="M4", n_repeats=4,
    )
    assert impurity.computed_on != permutation.computed_on
    # Impurity is a non-negative share of total impurity reduction; permutation
    # is a score delta and may go negative. Conflating the two scales is how a
    # figure ends up with a y-axis that means two things.
    assert (impurity.values >= 0).all()
    assert impurity.values.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# family roll-up
# ---------------------------------------------------------------------------


def test_family_share_is_taken_over_the_positive_part_only():
    """Negative permutation values made the old signed share exceed 1 and go below 0."""
    from src.feature_extraction.registry import feature_names

    names = feature_names()
    values = np.zeros(len(names))
    values[0] = 1.0
    values[1] = -0.5

    result = imp.ImportanceResult(
        kind="permutation", model_id="M4", feature_names=names, values=values
    )
    frame = imp.family_importance(result)
    assert frame["share_of_positive"].between(0.0, 1.0).all()
    assert frame["share_of_positive"].sum() == pytest.approx(1.0)
    assert frame["n_negative"].sum() == 1


# ---------------------------------------------------------------------------
# T48.5 / T48.6 / T48.7 -- the gate, on real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def m4_on_fold_zero() -> tuple[Any, Any, Any]:
    from src.models import smoke as sm

    try:
        data = sm.load_task_data("binary")
        fold = sm._fold_zero("binary", data)
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 fold 0 unavailable (" + type(error).__name__ + "): " + str(error))
    result = sm.smoke_fold(
        "M4", lambda: est.build_estimator("M4"), data, fold, keep_pipeline=True
    )
    return result, data, fold


def test_m4_smoke_runs_clean_on_the_real_matrix(m4_on_fold_zero: Any):
    result, _, _ = m4_on_fold_zero
    assert result.probability["has_predict_proba"] is True
    assert result.probability["n_nan"] == 0
    assert result.probability["well_formed"] is True
    for name, value in result.metrics.items():
        assert np.isfinite(value), name + " is " + str(value)


def test_m4_model_size_is_recorded(m4_on_fold_zero: Any):
    """T48.6 -- the number the complexity table (T26) needs."""
    result, _, _ = m4_on_fold_zero
    assert result.model_bytes > 0
    assert result.fit_seconds > 0


def test_both_importances_are_produced_on_the_real_138(m4_on_fold_zero: Any):
    """The gate, stated directly."""
    result, data, fold = m4_on_fold_zero

    impurity = imp.impurity_importance(result.pipeline, data.feature_names, model_id="M4")
    permutation = imp.permutation_importance(
        result.pipeline, data.X, data.y,
        held_out_index=fold.test_index, train_index=fold.train_index,
        feature_names=data.feature_names, model_id="M4", n_repeats=2, n_jobs=-1,
    )

    assert impurity.values.size == 138
    assert permutation.values.size == 138
    assert np.isfinite(impurity.values).all()
    assert np.isfinite(permutation.values).all()
    assert permutation.n_scored_rows == len(fold.test_uids)
    assert permutation.n_scored_rows < data.n_records, "scored the whole matrix"


def test_m4_does_not_score_suspiciously_well(m4_on_fold_zero: Any):
    result, _, _ = m4_on_fold_zero
    balanced = result.metrics["balanced_accuracy"]
    assert 0.5 < balanced < 0.95, (
        "M4 scored " + str(round(balanced, 4)) + " on one untuned fold; a near-perfect "
        "result is a leak until proven otherwise"
    )
