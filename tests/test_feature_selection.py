"""T57.7 -- SO-04 feature selection: the rankers, the score J, and the emitted files.

Two halves. The first runs the code against the REAL D1 matrix and the real DA-07
fold map -- a ranker that returns 138 finite numbers from a synthetic Gaussian
blob says nothing about a corpus where whole columns are constant inside some
folds. The second is a gate over what is on disk, which skips rather than passes
when the run has not happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# fixtures over the real matrix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data() -> Any:
    """The real D1 matrix, or a skip.

    Skipped rather than failed when the matrix is absent: it is a parquet file
    and `*.parquet` is gitignored, so it never reaches CI. Every test below
    depends on this fixture, so the skip cascades and CI reports "skipped"
    instead of erroring on a file it was never going to have. Same guard as
    ``real_d1`` in tests/test_search_no_leakage.py.
    """
    from src.models import smoke as sm

    try:
        return sm.load_task_data("binary")
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 matrix unavailable (" + type(error).__name__ + "): " + str(error))


@pytest.fixture(scope="module")
def fold(data: Any) -> Any:
    from src.optimization import driver as od

    try:
        return od.outer_folds_for("binary", data, repeats=[0], folds=[0])[0]
    except Exception as error:  # noqa: BLE001 - a missing DA-07 map is a skip
        pytest.skip("DA-07 fold 0 unavailable (" + type(error).__name__ + "): " + str(error))


@pytest.fixture(scope="module")
def train_block(data: Any, fold: Any) -> Any:
    train = np.asarray(fold.train_index, dtype=int)
    return np.asarray(data.X, dtype=float)[train], np.asarray(data.y)[train]


# ---------------------------------------------------------------------------
# T57.1 / T57.2 -- the rankers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mutual_info", "anova_f", "rf_importance", "gb_importance"])
def test_every_ranker_scores_every_column_finitely(train_block: Any, kind: str) -> None:
    from src.feature_selection import ranking as fr

    x_train, y_train = train_block
    ranked = fr.rank_features(kind, x_train, y_train)
    assert ranked.scores.shape == (x_train.shape[1],)
    assert np.isfinite(ranked.scores).all()
    assert sorted(ranked.order.tolist()) == list(range(x_train.shape[1]))


@pytest.mark.parametrize("kind", ["mutual_info", "anova_f"])
def test_rankers_are_reproducible_under_the_fixed_seed(train_block: Any, kind: str) -> None:
    """Rule 5. `mutual_info_classif` is stochastic and would drift unseeded."""
    from src.feature_selection import ranking as fr

    x_train, y_train = train_block
    first = fr.rank_features(kind, x_train, y_train, seed=42)
    second = fr.rank_features(kind, x_train, y_train, seed=42)
    assert np.array_equal(first.order, second.order)
    assert np.allclose(first.scores, second.scores)


def test_mutual_info_actually_depends_on_its_seed(train_block: Any) -> None:
    """If it did not, the seeding above would be proving nothing."""
    from src.feature_selection import ranking as fr

    x_train, y_train = train_block
    a = fr.rank_features("mutual_info", x_train, y_train, seed=42)
    b = fr.rank_features("mutual_info", x_train, y_train, seed=7)
    assert not np.allclose(a.scores, b.scores)


def test_top_k_is_a_prefix_nest(train_block: Any) -> None:
    """top-20 must be contained in top-40, or the sweep is not a sweep."""
    from src.feature_selection import ranking as fr

    x_train, y_train = train_block
    ranked = fr.rank_features("anova_f", x_train, y_train)
    assert set(ranked.top(20).tolist()) <= set(ranked.top(40).tolist())
    assert ranked.top(20).size == 20


def test_threshold_selectors_choose_their_own_size(train_block: Any) -> None:
    """T57.2: the point of a threshold is that it is not told a k."""
    from src.feature_selection import ranking as fr

    x_train, y_train = train_block
    columns, threshold = fr.threshold_select("rf_threshold_mean", x_train, y_train)
    assert 0 < columns.size < x_train.shape[1]
    assert np.isfinite(threshold)


# ---------------------------------------------------------------------------
# fold safety
# ---------------------------------------------------------------------------


def test_a_ranker_fitted_on_two_different_folds_disagrees(data: Any) -> None:
    """Proof that ranking is fold-dependent, so ranking once globally would leak.

    If a ranker returned the same order regardless of which rows it saw, fitting
    it inside the fold would be a formality. It does not.
    """
    from src.feature_selection import ranking as fr
    from src.optimization import driver as od

    folds = od.outer_folds_for("binary", data, repeats=[0], folds=[0, 1])
    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    orders = [
        fr.rank_features(
            "anova_f",
            features[np.asarray(item.train_index, dtype=int)],
            targets[np.asarray(item.train_index, dtype=int)],
        ).order
        for item in folds
    ]
    assert not np.array_equal(orders[0], orders[1])


def test_select_per_fold_uses_only_that_folds_training_rows(data: Any) -> None:
    """The per-fold subset must not change when other folds' rows change.

    Selected inside fold 0 with the whole matrix present, then again with every
    row outside fold 0's training block replaced by noise. Identical output is
    the only outcome consistent with the selector never reading those rows.
    """
    from dataclasses import replace

    from src.feature_selection import sweep as fs
    from src.optimization import driver as od

    folds = od.outer_folds_for("binary", data, repeats=[0], folds=[0])
    before = fs.select_per_fold(data, folds, "anova_f", 30, seed=42)

    features = np.asarray(data.X, dtype=float).copy()
    train = set(np.asarray(folds[0].train_index, dtype=int).tolist())
    outside = [row for row in range(features.shape[0]) if row not in train]
    rng = np.random.default_rng(0)
    features[outside] = rng.normal(size=(len(outside), features.shape[1]))
    poisoned = replace(data, X=features)

    after = fs.select_per_fold(poisoned, folds, "anova_f", 30, seed=42)
    assert before == after


# ---------------------------------------------------------------------------
# the multi-objective score J
# ---------------------------------------------------------------------------


def test_j_is_exactly_the_documented_formula() -> None:
    """T61.1's formula, checked term by term rather than against a stored number."""
    from src.feature_extraction import registry
    from src.optimization import multi_objective as mo

    weights = mo.load_weights()
    cost_model = mo.load_cost_model()
    names = list(registry.feature_names())[:40]

    scored = mo.score_j(0.83, names, weights=weights, cost_model=cost_model)
    expected = (
        weights.alpha * (1.0 - 0.83)
        + weights.beta * (40.0 / weights.n_features_total)
        + weights.gamma * cost_model.normalized(mo.families_needed(names))
    )
    assert scored.value == pytest.approx(expected)


def test_every_term_of_j_is_bounded_in_the_unit_interval() -> None:
    """T61.3 for the time term, and the same claim for the other two."""
    from src.feature_extraction import registry
    from src.optimization import multi_objective as mo

    cost_model = mo.load_cost_model()
    all_names = list(registry.feature_names())

    assert cost_model.normalized(mo.families_needed(all_names)) == pytest.approx(1.0)
    for family in sorted(cost_model.seconds):
        subset = [name for name in all_names if registry.family_of(name) == family]
        value = cost_model.normalized(mo.families_needed(subset))
        assert 0.0 <= value <= 1.0


def test_the_inference_time_term_is_family_wise_not_feature_wise() -> None:
    """One MFCC coefficient costs the whole MFCC stack; that is the modelling claim."""
    from src.feature_extraction import registry
    from src.optimization import multi_objective as mo

    cost_model = mo.load_cost_model()
    mfcc = [name for name in registry.feature_names() if registry.family_of(name) == "mfcc"]
    one = cost_model.normalized(mo.families_needed(mfcc[:1]))
    many = cost_model.normalized(mo.families_needed(mfcc))
    assert one == pytest.approx(many)


def test_an_empty_subset_has_no_j() -> None:
    from src.optimization import multi_objective as mo

    with pytest.raises(mo.MultiObjectiveError):
        mo.score_j(0.9, [])


# ---------------------------------------------------------------------------
# T57.7 -- the gate over what is on disk
# ---------------------------------------------------------------------------


def _section() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.search_optimization"))


def _load(name: str) -> Any:
    import pandas as pd

    path = _section() / "SO-04" / name
    if not path.exists():
        pytest.skip(
            str(path) + " does not exist; run scripts/06_run_feature_selection.py first"
        )
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def curve() -> Any:
    return _load("feature_count_curve.csv")


@pytest.fixture(scope="module")
def settings() -> Any:
    return _load("so04_settings.json")


@pytest.fixture(scope="module")
def subset() -> Any:
    import pandas as pd

    from src.utils.config import load_config

    path = Path(load_config("paths").require("outputs.features")) / "selected_feature_subset.csv"
    if not path.exists():
        pytest.skip(str(path) + " does not exist; run scripts/06_run_feature_selection.py")
    return pd.read_csv(path)


def test_the_sweep_produced_a_performance_curve(curve: Any) -> None:
    """T57.7: a curve, not a point -- several feature counts per ranker."""
    assert len(curve) > 0
    for ranker, block in curve.groupby("ranker"):
        assert len(block) >= 3, ranker + " has only " + str(len(block)) + " point(s)"
        assert block["k"].is_unique
        assert block["macro_f1_mean"].notna().all()
        assert block["macro_f1_mean"].between(0.0, 1.0).all()


def test_the_curve_is_averaged_over_more_than_one_fold(curve: Any) -> None:
    assert int(curve["n_evaluations"].min()) >= 2


def test_the_selected_subset_is_smaller_than_138(subset: Any) -> None:
    """T57.7, the part that fails loudest if the selection quietly did nothing."""
    from src.feature_extraction import registry

    assert 0 < len(subset) < len(registry.feature_names())


def test_the_selected_subset_names_real_features_exactly_once(subset: Any) -> None:
    from src.feature_extraction import registry

    known = set(registry.feature_names())
    names = subset["feature"].tolist()
    assert set(names) <= known, sorted(set(names) - known)[:5]
    assert len(names) == len(set(names))


def test_the_selected_subset_records_its_stability(subset: Any, settings: Any) -> None:
    """FE-12 carries how many outer folds chose each feature, not just the names."""
    assert "selected_in_folds" in subset.columns
    n_folds = int(subset["n_folds"].iloc[0])
    assert n_folds == len(settings["outer_folds"])
    assert subset["selected_in_folds"].between(1, n_folds).all()


def test_the_selected_subset_is_reproducible_from_the_per_fold_table(subset: Any) -> None:
    """T57.7 'reproducible': re-derive FE-12 from per_fold_selection.csv and compare."""
    from src.feature_extraction import registry
    from src.feature_selection import sweep as fs

    per_fold_frame = _load("per_fold_selection.csv")
    per_fold = {
        str(label): tuple(int(c) for c in block["column"])
        for label, block in per_fold_frame.groupby("outer_fold")
    }
    columns, _ = fs.consensus_subset(per_fold, len(subset), registry.feature_names())
    assert sorted(columns) == sorted(int(c) for c in subset["column"])


def test_the_comparison_scored_all_138_and_the_subset_on_the_same_folds() -> None:
    """T57.6: same folds, same seed, both arms present."""
    comparison = _load("all_features_vs_selected.csv")
    configurations = set(comparison["configuration"].unique())
    assert {"all_features", "selected_subset"} <= configurations

    folds_by_configuration = {
        name: set(block["outer_fold"]) for name, block in comparison.groupby("configuration")
    }
    reference = folds_by_configuration["all_features"]
    for name, folds in folds_by_configuration.items():
        assert folds == reference, name + " was scored on different folds"
    assert (comparison.loc[comparison.configuration == "all_features", "n_features"] == 138).all()


def test_rfecv_stopped_somewhere_short_of_the_full_matrix() -> None:
    """T57.3. If RFECV kept all 138 on every fold it did not run as configured."""
    frame = _load("rfecv_selection.csv")
    assert len(frame) > 0
    assert frame["n_selected"].between(1, 138).all()
    assert int(frame["n_selected"].max()) < 138


def test_the_settings_file_records_the_j_weighting_that_chose_the_subset(
    settings: Any,
) -> None:
    """No number in a deliverable may be traceable only to a default in code."""
    weights = settings["j_weights"]
    for key in ("alpha", "beta", "gamma", "n_features_total"):
        assert key in weights
    assert settings["chosen"]["k"] == settings["chosen"]["n_selected"]
    assert settings["chosen"]["ranker"] in settings["settings"]["rankers"]


def test_the_settings_file_records_what_pure_performance_would_have_chosen(
    settings: Any,
) -> None:
    """The J choice must be auditable against the choice it overruled.

    J trades performance for compactness by construction. Recording only the
    winner would hide how much was traded, which is exactly the failure the
    2026-08-28 search-objective entry in Docs/note.md is about.
    """
    chosen = settings["chosen"]
    for key in ("mean_macro_f1", "best_performance_ranker", "best_performance_k",
                "best_performance_macro_f1"):
        assert key in chosen, key
    given_up = float(chosen["best_performance_macro_f1"]) - float(chosen["mean_macro_f1"])
    assert np.isfinite(given_up)
    # Not asserted to be small -- how much J trades away is a finding, not a
    # requirement. Asserted to be RECORDED, and non-negative: the pure-performance
    # winner cannot score below the J winner on the metric it was chosen by.
    assert given_up >= -1e-12
