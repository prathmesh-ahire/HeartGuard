"""Model registry and persistence (Phase 51, gate T51.7).

The gate: every saved model reloads and reproduces identical predictions;
confirm size and timing are recorded.

"Identical" means bit-identical, not close. A reloaded model that agrees to six
decimal places has lost something, and whatever it lost will show up as an
unexplainable difference between a metric computed during training and the same
metric computed by the API months later.

The other half of this module is the failure joblib cannot catch. A matrix with
the right 138 columns in the wrong order has the right *shape*, so a reloaded
pipeline consumes it without complaint and returns confident nonsense -- every
row wrong, no exception, no warning, and no metric on that run capable of
revealing it. That is why a saved model carries its feature-name list and why
:func:`load_model` compares against it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.models import registry as reg

pytestmark = [
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    # `measure_inference` predicts one row at a time against a forest built with
    # n_jobs=-1, and sklearn warns once per call that its own `delayed` is being
    # used outside its `Parallel`. That is internal chatter about a real cost --
    # spinning up parallel machinery for a single row is exactly the deployment
    # penalty T51.4 exists to measure -- so the measurement stays and the twelve
    # thousand identical warnings do not.
    pytest.mark.filterwarnings(
        "ignore:`sklearn.utils.parallel.delayed` should be used:UserWarning"
    ),
]


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] + rng.normal(scale=0.6, size=n) > 0.7).astype(int)
    names = tuple("feature_" + str(index).zfill(2) for index in range(8))
    return X, y, names


def _fit(model_id: str, X: np.ndarray, y: np.ndarray) -> Any:
    from src.models import pipeline as pl

    built = pl.build_pipeline(reg.build(model_id), y=y)
    built.fit(X, y)
    return built


# ---------------------------------------------------------------------------
# T51.1 -- the registry
# ---------------------------------------------------------------------------


def test_the_registry_describes_every_declared_model():
    ids = reg.model_ids()
    assert ids[:9] == ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9")
    assert len(reg.entries()) == len(ids)


def test_the_registry_describes_models_it_cannot_build():
    """A report must be able to list M9 without a deep-learning stack installed."""
    entry = reg.entry("M9")
    assert entry.name == "1D-CNN"
    assert entry.implemented is False
    assert entry.unavailable_reason, "an unavailable model must say why"
    assert not reg.available("M9")


def test_m9s_reason_points_at_the_missing_outputs_report():
    """T52.6: the exclusion is recorded, not implied."""
    reason = reg.entry("M9").unavailable_reason
    assert "missing_outputs_report" in reason
    report = Path("outputs/missing_outputs_report.txt")
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        assert "T52" in text and "1D-CNN" in text


def test_the_registry_knows_which_models_are_ensembles():
    for model_id in ("M6", "M7"):
        entry = reg.entry(model_id)
        assert entry.is_ensemble
        assert entry.members == ("M3", "M4", "M5")
    assert not reg.entry("M4").is_ensemble


def test_the_registry_reports_each_models_search_size():
    assert reg.entry("M1").n_search_dimensions == 4
    assert reg.entry("M1").search_constraints == ("M1_SOLVER_PENALTY",)
    assert reg.entry("M6").n_search_dimensions == 0


def test_the_registry_table_has_one_row_per_model():
    frame = reg.registry_frame()
    assert len(frame) == len(reg.model_ids())
    assert frame["model_id"].is_unique
    assert frame.loc[frame["model_id"] == "M9", "implemented"].item() is np.False_ or (
        not frame.loc[frame["model_id"] == "M9", "implemented"].item()
    )


# ---------------------------------------------------------------------------
# T51.2 / T51.6 / T51.7 -- the round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["M1", "M2", "M4"])
def test_a_reloaded_model_predicts_identically(
    model_id: str, dataset: Any, tmp_path: Path
):
    """The gate, stated directly. Bit-identical, not close."""
    X, y, names = dataset
    fitted = _fit(model_id, X, y)
    before_proba = fitted.predict_proba(X)
    before_pred = fitted.predict(X)

    reg.save_model(
        fitted, model_id=model_id, task="binary", feature_names=names, root=tmp_path
    )
    reloaded, _ = reg.load_model("binary", model_id, tmp_path, feature_names=names)

    assert np.array_equal(reloaded.predict_proba(X), before_proba)
    assert np.array_equal(reloaded.predict(X), before_pred)


def test_a_reloaded_ensemble_keeps_its_weights_and_threshold(
    dataset: Any, tmp_path: Path
):
    """The in-fold decision rule has to survive serialisation or it means nothing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    from src.ensemble.soft_voting import SoftVotingEnsemble

    X, y, names = dataset
    groups = np.array(["s" + str(index // 2) for index in range(len(y))])
    ensemble = SoftVotingEnsemble(
        [
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ],
        weights="optimized", groups=groups, inner_cv=3,
    ).fit(X, y)

    reg.save_model(
        ensemble, model_id="M7", task="binary", feature_names=names, root=tmp_path
    )
    reloaded, _ = reg.load_model("binary", "M7", tmp_path, feature_names=names)

    assert np.array_equal(reloaded.weights_, ensemble.weights_)
    assert reloaded.threshold_ == ensemble.threshold_
    assert np.array_equal(reloaded.predict(X), ensemble.predict(X))


# ---------------------------------------------------------------------------
# T51.5 -- the feature list is the alignment guard
# ---------------------------------------------------------------------------


def test_saving_without_feature_names_is_refused(dataset: Any, tmp_path: Path):
    X, y, _ = dataset
    fitted = _fit("M1", X, y)
    with pytest.raises(reg.RegistryError, match="without its feature names"):
        reg.save_model(
            fitted, model_id="M1", task="binary", feature_names=[], root=tmp_path
        )


def test_a_feature_list_of_the_wrong_length_is_refused(dataset: Any, tmp_path: Path):
    X, y, names = dataset
    fitted = _fit("M1", X, y)
    with pytest.raises(reg.RegistryError, match="fitted on 8 columns"):
        reg.save_model(
            fitted, model_id="M1", task="binary",
            feature_names=names[:5], root=tmp_path,
        )


def test_reordered_columns_are_caught_and_named(dataset: Any, tmp_path: Path):
    """The failure joblib cannot catch: right shape, wrong meaning, no exception."""
    X, y, names = dataset
    fitted = _fit("M1", X, y)
    reg.save_model(
        fitted, model_id="M1", task="binary", feature_names=names, root=tmp_path
    )

    swapped = (names[1], names[0], *names[2:])
    with pytest.raises(reg.RegistryError) as caught:
        reg.load_model("binary", "M1", tmp_path, feature_names=swapped)

    message = str(caught.value)
    assert "DIFFERENT" in message and "ORDER" in message
    assert "position 0" in message
    assert "silently wrong" in message


def test_a_different_feature_set_is_caught_and_itemised(dataset: Any, tmp_path: Path):
    X, y, names = dataset
    fitted = _fit("M1", X, y)
    reg.save_model(
        fitted, model_id="M1", task="binary", feature_names=names, root=tmp_path
    )
    with pytest.raises(reg.RegistryError, match="missing"):
        reg.load_model(
            "binary", "M1", tmp_path,
            feature_names=(*names[:-1], "something_else"),
        )


def test_loading_without_a_feature_list_still_works(dataset: Any, tmp_path: Path):
    """The check is opt-in; not passing names must not silently pass a wrong matrix."""
    X, y, names = dataset
    fitted = _fit("M1", X, y)
    reg.save_model(
        fitted, model_id="M1", task="binary", feature_names=names, root=tmp_path
    )
    model, manifest = reg.load_model("binary", "M1", tmp_path)
    assert model is not None
    assert tuple(manifest["feature_names"]) == names, (
        "the names must still be on disk so a caller can check later"
    )


def test_loading_a_model_that_is_not_there_says_so(tmp_path: Path):
    with pytest.raises(reg.RegistryError, match="no saved model"):
        reg.load_model("binary", "M4", tmp_path)


# ---------------------------------------------------------------------------
# T51.3 / T51.4 -- size and timing
# ---------------------------------------------------------------------------


def test_size_and_timings_are_recorded(dataset: Any, tmp_path: Path):
    """The gate's second clause -- what the complexity table (T26) reads."""
    X, y, names = dataset
    fitted = _fit("M4", X, y)
    saved = reg.save_model(
        fitted, model_id="M4", task="binary", feature_names=names,
        root=tmp_path, fit_seconds=1.25, X_sample=X[:10],
    )

    assert saved.size_bytes > 0
    assert saved.manifest["model_mb"] > 0
    assert saved.manifest["fit_seconds"] == 1.25
    assert saved.manifest["inference_seconds_per_record"] > 0
    assert saved.manifest["inference_seconds_per_record_batched"] > 0
    assert saved.manifest["n_features"] == len(names)


def test_the_layout_is_models_saved_task_model_id(dataset: Any, tmp_path: Path):
    """T51.2 names the layout explicitly."""
    X, y, names = dataset
    fitted = _fit("M1", X, y)
    saved = reg.save_model(
        fitted, model_id="M1", task="pascal_a", feature_names=names, root=tmp_path
    )
    assert saved.path == tmp_path / "pascal_a" / "M1" / reg.MODEL_FILENAME
    assert (tmp_path / "pascal_a" / "M1" / reg.MANIFEST_FILENAME).is_file()


def test_single_record_inference_is_not_batch_time_divided_by_batch_size(
    dataset: Any,
):
    """A 500-tree forest amortises across a batch; POST /predict gets one row."""
    X, y, _ = dataset
    fitted = _fit("M4", X, y)
    measured = reg.measure_inference(fitted, X[:20], repeats=2)

    assert measured["inference_seconds_per_record"] > 0
    assert measured["inference_seconds_per_record_batched"] > 0
    assert measured["inference_batch_speedup"] > 1.0, (
        "batched inference should be cheaper per record; if it is not, the two "
        "numbers are measuring the same thing and one of them is wrong"
    )


def test_measuring_inference_needs_real_rows(dataset: Any):
    X, y, _ = dataset
    fitted = _fit("M1", X, y)
    with pytest.raises(reg.RegistryError, match="at least one row"):
        reg.measure_inference(fitted, np.zeros((0, 8)))


def test_saved_models_are_discoverable_with_their_manifests(
    dataset: Any, tmp_path: Path
):
    X, y, names = dataset
    for model_id in ("M1", "M2"):
        reg.save_model(
            _fit(model_id, X, y), model_id=model_id, task="binary",
            feature_names=names, root=tmp_path, fit_seconds=0.5,
        )
    found = {saved.model_id: saved for saved in reg.saved_models(tmp_path)}
    assert set(found) == {"M1", "M2"}
    for saved in found.values():
        assert saved.size_bytes > 0
        assert saved.feature_names == names


def test_the_manifest_records_the_package_versions(dataset: Any, tmp_path: Path):
    """Rule 5: a model reloaded under a different sklearn is not the same model."""
    X, y, names = dataset
    saved = reg.save_model(
        _fit("M1", X, y), model_id="M1", task="binary",
        feature_names=names, root=tmp_path,
    )
    versions = saved.manifest["package_versions"]
    assert "scikit-learn" in versions or "sklearn" in str(versions)


# ---------------------------------------------------------------------------
# the real matrix
# ---------------------------------------------------------------------------


def test_every_committed_model_reloads_and_matches_its_manifest():
    """Whatever the driver script last wrote must still load and be self-consistent."""
    saved = reg.saved_models()
    if not saved:
        pytest.skip("no models saved yet; run scripts/04_model_smoke.py")

    for model in saved:
        if not model.path.is_file():
            pytest.skip("model binaries are gitignored; " + str(model.path) + " absent")
        loaded, manifest = reg.load_model(
            model.task, model.model_id, feature_names=model.feature_names
        )
        assert manifest["n_features"] == len(model.feature_names)
        assert manifest["model_bytes"] > 0
        assert getattr(loaded, "n_features_in_", len(model.feature_names)) == len(
            model.feature_names
        )
