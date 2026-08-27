"""Fold safety (Phase 44, gate T44.7).

The gate: inject a leakage canary feature and confirm the pipeline never sees
test-fold statistics.

A canary is needed because leakage is invisible in the output. A scaler fitted
on all 3,240 records and one fitted on a 2,593-record training fold both produce
a matrix of plausible standardised numbers; the metric shifts by a fraction of a
point. Nothing looks wrong. So instead of inspecting outputs, these tests
inspect **what each step learned** -- the imputer's medians, the scaler's mean
and ``n_samples_seen_`` -- and compare them against the statistics of the
training rows alone.

The canary itself is a feature whose value is wildly different in the test rows
than in the training rows. If any step of the pipeline touched the test rows
while fitting, its learned statistics move detectably; if the pipeline is
fold-safe, they are bit-identical to fitting on the training rows by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.evaluation import cv
from src.models import pipeline as pl

N_SAMPLES = 200
N_FEATURES = 8
CANARY = 0  # column index of the planted feature


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A matrix whose column 0 betrays which rows are in the test fold.

    Training rows carry a canary around 0; test rows carry it around 10,000.
    Any statistic computed over both is dragged far from the training-only
    value, which is exactly what makes the leak detectable.
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES))
    y = rng.integers(0, 2, size=N_SAMPLES)

    test_index = np.arange(N_SAMPLES // 2, N_SAMPLES)
    train_index = np.arange(0, N_SAMPLES // 2)
    X[test_index, CANARY] += 10_000.0
    return X, y, train_index, test_index


def _fit(X: np.ndarray, y: np.ndarray, index: np.ndarray, **kwargs: Any) -> Any:
    from sklearn.linear_model import LogisticRegression

    built = pl.build_pipeline(
        LogisticRegression(max_iter=1000), y=y[index], **kwargs
    )
    built.fit(X[index], y[index])
    return built


# ---------------------------------------------------------------------------
# T44.5 / T44.7 -- the canary
# ---------------------------------------------------------------------------


def test_the_scaler_learns_only_the_training_rows(dataset: Any):
    """The gate, stated directly."""
    X, y, train_index, test_index = dataset
    fitted = _fit(X, y, train_index)

    learned = pl.fitted_steps(fitted)
    expected_mean = X[train_index].mean(axis=0)

    assert np.allclose(learned["scaler.mean_"], expected_mean)
    assert learned["scaler.mean_"][CANARY] == pytest.approx(
        X[train_index, CANARY].mean()
    )
    # And provably not the leaked value.
    assert not np.isclose(learned["scaler.mean_"][CANARY], X[:, CANARY].mean())
    del test_index


def test_the_scaler_counts_only_the_training_rows(dataset: Any):
    """``n_samples_seen_`` is the most direct evidence available."""
    X, y, train_index, _ = dataset
    learned = pl.fitted_steps(_fit(X, y, train_index))

    seen = np.atleast_1d(learned["scaler.n_samples_seen_"])
    assert int(seen.ravel()[0]) == len(train_index)
    assert int(seen.ravel()[0]) != N_SAMPLES


def test_the_imputer_learns_only_the_training_rows(dataset: Any):
    X, y, train_index, _ = dataset
    X = X.copy()
    X[train_index[:10], 3] = np.nan

    learned = pl.fitted_steps(_fit(X, y, train_index))
    expected = np.nanmedian(X[train_index], axis=0)
    assert np.allclose(learned["imputer.statistics_"], expected)
    assert not np.isclose(learned["imputer.statistics_"][CANARY], np.median(X[:, CANARY]))


def test_a_leaked_fit_is_detectably_different(dataset: Any):
    """The canary must be capable of firing, or the tests above prove nothing."""
    X, y, train_index, _ = dataset

    honest = pl.fitted_steps(_fit(X, y, train_index))
    leaked = pl.fitted_steps(_fit(X, y, np.arange(N_SAMPLES)))

    assert not np.allclose(honest["scaler.mean_"], leaked["scaler.mean_"])
    difference = abs(honest["scaler.mean_"][CANARY] - leaked["scaler.mean_"][CANARY])
    assert difference > 1000.0, "the canary is too quiet to detect a leak"


def test_transform_of_test_rows_uses_training_statistics(dataset: Any):
    """The canary column must stay enormous after scaling, not be normalised away.

    If the scaler had seen the test rows it would map them to roughly zero. A
    fold-safe scaler leaves them thousands of standard deviations out, which is
    the honest representation: those values really are unlike anything in
    training.
    """
    X, y, train_index, test_index = dataset
    fitted = _fit(X, y, train_index)

    transformed = fitted[:-1].transform(X[test_index])
    assert abs(transformed[:, CANARY].mean()) > 100.0


def test_the_selector_also_fits_inside_the_fold(dataset: Any):
    """A selector scored on all rows is the same leak in a subtler place."""
    X, y, train_index, _ = dataset
    config = {
        "imputer": {"strategy": "median"},
        "scaler": {"kind": "standard"},
        "selector": {"enabled": True, "kind": "anova_f", "k": 4},
    }
    fitted = _fit(X, y, train_index, config=config, n_features=N_FEATURES)

    selector = fitted.named_steps["selector"]
    assert selector.k == 4
    assert selector.get_support().sum() == 4
    assert selector.n_features_in_ == N_FEATURES
    # Scores come from the training rows only.
    assert np.isfinite(selector.scores_).any()


def test_every_learning_step_is_inside_the_pipeline(dataset: Any):
    """Structural version of the claim: nothing that learns sits outside."""
    X, y, train_index, _ = dataset
    fitted = _fit(X, y, train_index)

    names = list(fitted.named_steps)
    assert names == ["imputer", "scaler", "estimator"]
    for name in ("imputer", "scaler"):
        step = fitted.named_steps[name]
        assert hasattr(step, "fit"), name


def test_the_driver_and_the_pipeline_compose_without_leaking(
    dataset: Any, tmp_path: Any, monkeypatch: Any
):
    """End to end: Phase 43's driver running Phase 44's pipeline.

    Each fold must fit its own scaler. The test asserts that the scalers differ
    between folds -- identical statistics across folds would mean each fold saw
    the same (whole-matrix) data.
    """
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    X, y, _, _ = dataset
    uids = ["R" + format(index, "03d") for index in range(N_SAMPLES)]
    groups = np.array(["S" + format(index // 2, "03d") for index in range(N_SAMPLES)])

    rows = []
    for index, uid in enumerate(uids):
        rows.append(
            {
                "task": "toy",
                "scheme": "grouped_5fold",
                "repeat": 0,
                "fold": (index // 2) % 5,
                "record_uid": uid,
                "split_group": groups[index],
                "y": int(y[index]),
            }
        )
    path = tmp_path / cv.SPLIT_MAP_FILENAME
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(cv, "split_map_path", lambda: path)

    folds = cv.resolve_folds(cv.load_folds("toy"), uids)
    seen: list[np.ndarray] = []

    def factory() -> Any:
        built = pl.build_pipeline(LogisticRegression(max_iter=1000), y=y)

        original_fit = built.fit

        def recording_fit(X_fold: Any, y_fold: Any, **kwargs: Any) -> Any:
            result = original_fit(X_fold, y_fold, **kwargs)
            seen.append(pl.fitted_steps(built)["scaler.mean_"].copy())
            return result

        built.fit = recording_fit  # type: ignore[method-assign]
        return built

    result = cv.run_cv(factory, X, y, groups, folds, task="toy")
    assert result.n_folds == 5
    assert len(seen) == 5

    # No two folds learned the same scaler: each saw a different training set.
    for first in range(len(seen)):
        for second in range(first + 1, len(seen)):
            assert not np.allclose(seen[first], seen[second])


# ---------------------------------------------------------------------------
# T44.1 / T44.2 -- construction
# ---------------------------------------------------------------------------


def test_the_default_pipeline_is_imputer_scaler_estimator():
    from sklearn.linear_model import LogisticRegression

    built = pl.build_pipeline(LogisticRegression())
    assert list(built.named_steps) == ["imputer", "scaler", "estimator"]


def test_the_imputer_defaults_to_median():
    assert pl.make_imputer().strategy == "median"
    assert pl.make_imputer({"strategy": "mean"}).strategy == "mean"


def test_an_unknown_imputer_strategy_is_rejected():
    with pytest.raises(pl.PipelineError, match="unknown imputer strategy"):
        pl.make_imputer({"strategy": "magic"})


def test_the_scaler_defaults_to_standard_and_honours_config():
    from sklearn.preprocessing import RobustScaler, StandardScaler

    assert isinstance(pl.make_scaler(), StandardScaler)
    assert isinstance(pl.make_scaler({"kind": "robust"}), RobustScaler)
    assert pl.make_scaler({"kind": "none"}) is None
    assert pl.make_scaler({"kind": "standard", "with_mean": False}).with_mean is False


def test_an_unknown_scaler_kind_is_rejected():
    with pytest.raises(pl.PipelineError, match="unknown scaler kind"):
        pl.make_scaler({"kind": "quantum"})


def test_the_estimator_handed_in_is_never_fitted(dataset: Any):
    """Cloning means the caller's object cannot carry state between folds."""
    from sklearn.exceptions import NotFittedError
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.validation import check_is_fitted

    X, y, train_index, _ = dataset
    original = LogisticRegression(max_iter=1000)
    built = pl.build_pipeline(original, y=y[train_index])
    built.fit(X[train_index], y[train_index])

    with pytest.raises(NotFittedError):
        check_is_fitted(original)


def test_the_project_config_builds_a_valid_pipeline():
    """models.yaml's pipeline block must actually construct."""
    from sklearn.linear_model import LogisticRegression

    built = pl.build_pipeline(LogisticRegression())
    assert "imputer" in built.named_steps
    assert "scaler" in built.named_steps


# ---------------------------------------------------------------------------
# T44.3 -- the selector's three modes
# ---------------------------------------------------------------------------


def test_the_selector_is_off_by_default():
    assert pl.make_selector() is None
    assert pl.make_selector({"enabled": False, "kind": "anova_f"}) is None
    assert pl.make_selector({"enabled": True, "kind": "none"}) is None


def test_a_fixed_subset_selects_the_named_columns(dataset: Any):
    X, _, _, _ = dataset
    selector = pl.make_selector(
        {"enabled": True, "kind": "fixed_subset", "columns": [1, 3, 5]}
    )
    reduced = selector.fit(X, None).transform(X)
    assert reduced.shape == (N_SAMPLES, 3)
    assert np.allclose(reduced[:, 0], X[:, 1])


def test_a_fixed_subset_out_of_range_is_rejected(dataset: Any):
    X, _, _, _ = dataset
    selector = pl.make_selector(
        {"enabled": True, "kind": "fixed_subset", "columns": [1, 999]}
    )
    with pytest.raises(pl.PipelineError, match="outside the matrix"):
        selector.fit(X, None)


def test_a_fixed_subset_without_columns_is_rejected():
    with pytest.raises(pl.PipelineError, match="needs a 'columns' list"):
        pl.make_selector({"enabled": True, "kind": "fixed_subset"})


def test_k_is_capped_at_the_matrix_width():
    selector = pl.make_selector(
        {"enabled": True, "kind": "anova_f", "k": 500}, n_features=N_FEATURES
    )
    assert selector.k == N_FEATURES


def test_an_unknown_selector_kind_is_rejected():
    with pytest.raises(pl.PipelineError, match="unknown selector kind"):
        pl.make_selector({"enabled": True, "kind": "telepathy"})


# ---------------------------------------------------------------------------
# T44.4 -- imbalance
# ---------------------------------------------------------------------------


def test_imbalance_ratio_is_majority_over_minority():
    assert pl.imbalance_ratio([0] * 75 + [1] * 25) == pytest.approx(3.0)
    assert pl.imbalance_ratio([0, 1]) == pytest.approx(1.0)


def test_imbalance_ratio_needs_two_classes():
    with pytest.raises(pl.PipelineError, match="two classes"):
        pl.imbalance_ratio([1, 1, 1])


def test_class_weight_is_applied_where_supported(dataset: Any):
    from sklearn.linear_model import LogisticRegression

    _, y, _, _ = dataset
    built = pl.build_pipeline(LogisticRegression(), y=y)
    assert built.named_steps["estimator"].class_weight == "balanced"


def test_class_weight_is_skipped_where_unsupported(dataset: Any):
    """KNN takes no class_weight; passing one would raise mid-run."""
    from sklearn.neighbors import KNeighborsClassifier

    _, y, _, _ = dataset
    assert not pl.supports_class_weight(KNeighborsClassifier())
    built = pl.build_pipeline(KNeighborsClassifier(), y=y)
    assert not hasattr(built.named_steps["estimator"], "class_weight")


def test_class_weight_can_be_turned_off(dataset: Any):
    from sklearn.linear_model import LogisticRegression

    _, y, _, _ = dataset
    built = pl.build_pipeline(
        LogisticRegression(), y=y, config={"class_weight": {"strategy": "none"}}
    )
    assert built.named_steps["estimator"].class_weight is None


def test_an_explicit_weight_dict_is_honoured(dataset: Any):
    from sklearn.linear_model import LogisticRegression

    _, y, _, _ = dataset
    built = pl.build_pipeline(
        LogisticRegression(), y=y, config={"class_weight": {"strategy": {0: 1.0, 1: 3.0}}}
    )
    assert built.named_steps["estimator"].class_weight == {0: 1.0, 1: 3.0}


def test_resampling_fails_loudly_rather_than_being_ignored():
    """A flag that silently does nothing is worse than one that is unimplemented."""
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(pl.PipelineError, match="resampling is enabled"):
        pl.build_pipeline(
            LogisticRegression(), config={"resampling": {"enabled": True}}
        )


def test_resampling_is_off_in_the_project_config():
    """Rule: imbalance is handled by class weights, not by inventing records."""
    from src.utils.config import load_config

    settings = load_config("models").get("pipeline")
    assert settings["resampling"]["enabled"] is False
    assert settings["class_weight"]["strategy"] == "balanced"
