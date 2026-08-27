"""T54.6/T54.7 -- the outer test fold is never seen by a search, and every trial is logged.

The point of this file is that the leakage claim is a **measurement**, not an
assertion in a docstring. ``src.optimization.base.RowLedger`` records every row
index a trial fitted on or scored, so after a search has run the question "did
it touch the test fold?" is a set intersection over real indices, taken from the
arrays the estimators were actually handed.

Most of the file runs on a small grouped synthetic matrix, because a fast test
that exercises every branch of the budget and the ledger is worth having. But a
synthetic matrix cannot show that the *real* DA-07 map resolves correctly against
the *real* feature matrix, so the last section repeats the leakage assertion
against D1 fold 0 exactly as the pipeline loads it, and skips only if the feature
matrix has not been built yet.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.optimization import base as ob
from src.optimization.randomized import RandomizedSearch

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class _Data:
    """The TaskData shape the search consumes, over a synthetic-but-grouped matrix."""

    def __init__(self, n: int = 240, n_features: int = 12, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Two recordings per subject, so a grouped splitter has something to do
        # and an ungrouped one would visibly leak.
        subjects = np.repeat(np.arange(n // 2), 2)
        signal = rng.normal(size=(n, n_features))
        y = (signal[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        self.task = "binary"
        self.X = signal
        self.y = y
        self.groups = np.array(["s" + str(int(value)) for value in subjects], dtype=object)
        self.record_uids = tuple("r" + str(index) for index in range(n))
        self.feature_names = tuple("f" + str(index) for index in range(n_features))


class _Fold:
    """A resolved outer fold, in the shape ``cv.resolve_folds`` produces."""

    def __init__(self, train_index: Any, test_index: Any, repeat: int = 0, fold: int = 0) -> None:
        self.task = "binary"
        self.scheme = "test"
        self.repeat = repeat
        self.fold = fold
        self.train_index = np.asarray(train_index, dtype=int)
        self.test_index = np.asarray(test_index, dtype=int)
        self.label = "r" + str(repeat) + "f" + str(fold)


@pytest.fixture(scope="module")
def data() -> _Data:
    return _Data()


@pytest.fixture(scope="module")
def outer_fold(data: _Data) -> _Fold:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train, test = next(iter(splitter.split(data.X, data.y, data.groups)))
    return _Fold(train, test)


def _search(data: _Data, budget: ob.Budget) -> RandomizedSearch:
    from src.models import spaces

    return RandomizedSearch(
        "M1",
        spaces.load_space("M1"),
        ob.objective_for("binary"),
        budget,
        seed=42,
    )


# ---------------------------------------------------------------------------
# inner folds
# ---------------------------------------------------------------------------


def test_inner_folds_never_include_an_outer_test_row(data: _Data, outer_fold: _Fold) -> None:
    forbidden = set(outer_fold.test_index.tolist())
    for split in ob.inner_folds(outer_fold, data.y, data.groups, n_splits=3):
        assert not set(split.train_index.tolist()) & forbidden
        assert not set(split.val_index.tolist()) & forbidden


def test_inner_folds_cover_the_training_block_exactly(data: _Data, outer_fold: _Fold) -> None:
    """Every training row is validated exactly once across the inner folds."""
    splits = ob.inner_folds(outer_fold, data.y, data.groups, n_splits=3)
    validated = np.concatenate([split.val_index for split in splits])
    assert sorted(validated.tolist()) == sorted(outer_fold.train_index.tolist())


def test_inner_folds_do_not_split_a_subject(data: _Data, outer_fold: _Fold) -> None:
    groups = np.asarray(data.groups, dtype=object)
    for split in ob.inner_folds(outer_fold, data.y, data.groups, n_splits=3):
        assert not set(groups[split.train_index]) & set(groups[split.val_index])


def test_inner_folds_are_deterministic(data: _Data, outer_fold: _Fold) -> None:
    first = ob.inner_folds(outer_fold, data.y, data.groups, n_splits=3)
    second = ob.inner_folds(outer_fold, data.y, data.groups, n_splits=3)
    for left, right in zip(first, second, strict=True):
        assert left.train_index.tolist() == right.train_index.tolist()
        assert left.val_index.tolist() == right.val_index.tolist()


def test_inner_folds_refuse_an_unresolved_fold(data: _Data) -> None:
    class _Unresolved:
        train_index = None
        label = "r0f0"

    with pytest.raises(ob.SearchError, match="RESOLVED"):
        ob.inner_folds(_Unresolved(), data.y, data.groups)


# ---------------------------------------------------------------------------
# T54.7 -- the search itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result(data: _Data, outer_fold: _Fold) -> ob.SearchResult:
    return _search(data, ob.Budget(max_trials=8)).run(data, outer_fold, n_inner_splits=3)


def test_search_never_touched_an_outer_test_row(
    result: ob.SearchResult, outer_fold: _Fold
) -> None:
    """The headline assertion of T54.7, taken from the ledger, not from the code path."""
    forbidden = frozenset(int(value) for value in outer_fold.test_index)
    assert not (result.ledger.fitted & forbidden)
    assert not (result.ledger.scored & forbidden)
    result.assert_no_outer_leakage()


def test_search_did_use_the_training_rows(result: ob.SearchResult, outer_fold: _Fold) -> None:
    """The mirror of the leakage test: a search that touched nothing also passes it."""
    assert result.ledger.touched == frozenset(int(v) for v in outer_fold.train_index)


def test_every_trial_is_logged(result: ob.SearchResult) -> None:
    frame = result.trials_frame()
    assert len(frame) == len(result.trials) == 8
    assert frame["trial"].tolist() == list(range(8))
    for column in ("score", "seconds", "status", "n_inner_folds", "best_so_far"):
        assert column in frame.columns
    for name in result.trials[0].params:
        assert "param_" + name in frame.columns
    # Every successful trial carries one score per inner fold, not an average only.
    for trial in result.successful:
        assert len(trial.fold_scores) == 3
        # ...and the sensitivity/specificity behind each of those scores, so a
        # different linear objective can be re-ranked off the log without a re-run.
        assert len(trial.fold_components) == 3
        for scored in trial.fold_components:
            assert {"sensitivity", "specificity"} <= set(scored)
    for position in range(3):
        assert "inner_fold_" + str(position) + "_sensitivity" in frame.columns
        assert "inner_fold_" + str(position) + "_specificity" in frame.columns


def test_ledger_catches_a_deliberate_leak(data: _Data, outer_fold: _Fold) -> None:
    """A poisoned ledger must fail -- otherwise the passing tests above prove nothing."""
    ledger = ob.RowLedger()
    ledger.record(outer_fold.test_index[:1], outer_fold.train_index[:1])
    with pytest.raises(ob.SearchError, match="touched the outer test fold"):
        ledger.assert_disjoint_from(outer_fold.test_index)


def test_best_is_the_best_logged_trial(result: ob.SearchResult) -> None:
    best = result.best
    assert best is not None
    assert best.score == max(trial.score for trial in result.successful)
    assert result.best_params == best.params


def test_search_is_reproducible(data: _Data, outer_fold: _Fold) -> None:
    """Rule 5 -- the same command twice gives the same trials in the same order."""
    first = _search(data, ob.Budget(max_trials=5)).run(data, outer_fold, n_inner_splits=3)
    second = _search(data, ob.Budget(max_trials=5)).run(data, outer_fold, n_inner_splits=3)
    assert [trial.params for trial in first.trials] == [trial.params for trial in second.trials]
    assert [trial.score for trial in first.trials] == [trial.score for trial in second.trials]


def test_trial_params_are_always_legal(result: ob.SearchResult) -> None:
    from src.models import spaces

    space = spaces.load_space("M1")
    for trial in result.trials:
        assert space.is_valid(trial.params), trial.params


# ---------------------------------------------------------------------------
# T54.5 -- the budget
# ---------------------------------------------------------------------------


def test_trial_budget_is_exact(data: _Data, outer_fold: _Fold) -> None:
    run = _search(data, ob.Budget(max_trials=3)).run(data, outer_fold, n_inner_splits=3)
    assert len(run.trials) == 3
    assert "trial budget exhausted" in run.stop_reason


def test_wall_clock_budget_terminates_gracefully(data: _Data, outer_fold: _Fold) -> None:
    """A time-limited search stops between trials, and the trials it did run are complete."""
    run = _search(data, ob.Budget(max_trials=500, max_seconds=2.0)).run(
        data, outer_fold, n_inner_splits=3
    )
    assert 0 < len(run.trials) < 500
    assert "budget" in run.stop_reason
    for trial in run.trials:
        assert trial.status in {"ok", "error", "invalid"}
        if trial.status == "ok":
            assert len(trial.fold_scores) == 3


def test_budget_rejects_nonsense() -> None:
    with pytest.raises(ob.SearchError):
        ob.Budget(max_trials=0)
    with pytest.raises(ob.SearchError):
        ob.Budget(max_trials=5, max_seconds=0)


def test_truncated_search_is_a_prefix_of_the_longer_one(data: _Data, outer_fold: _Fold) -> None:
    """A stopped search must be the first k trials of the full one, not a different search."""
    short = _search(data, ob.Budget(max_trials=3)).run(data, outer_fold, n_inner_splits=3)
    long = _search(data, ob.Budget(max_trials=6)).run(data, outer_fold, n_inner_splits=3)
    assert [trial.params for trial in short.trials] == [
        trial.params for trial in long.trials[:3]
    ]


# ---------------------------------------------------------------------------
# T54.2 -- the objective
# ---------------------------------------------------------------------------


def test_objective_matches_the_configured_scoring() -> None:
    """Pins the 2026-08-28 decision, not just "whatever config happens to say".

    The binary objective was ``balanced_accuracy_plus_sensitivity`` until the
    first SO-01/SO-02 runs measured that it picks a different -- and on balanced
    accuracy worse -- point than plain balanced accuracy in 7 of 8 searches. It
    is the same objective Phase 50 rejected for the M6/M7 threshold. Changing
    this literal should require reading the note.md entry first.
    """
    assert ob.objective_for("binary").name == "balanced_accuracy"
    assert ob.objective_for("pascal_a").name == "macro_f1"


@pytest.mark.parametrize(
    "name", ["balanced_accuracy", "balanced_accuracy_plus_sensitivity", "sensitivity", "youden"]
)
def test_binary_objective_is_the_ensembles_objective(name: str) -> None:
    """One definition of the objective, not two that happen to share a name."""
    from src.ensemble.soft_voting import OBJECTIVES

    objective = ob.Objective(name=name, kind="binary")
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    expected = OBJECTIVES[name](2 / 3, 2 / 3)
    assert objective(y_true, y_pred, (0, 1)) == pytest.approx(expected)


def test_objective_rejects_an_unknown_name() -> None:
    with pytest.raises(ob.SearchError):
        ob.Objective(name="accuracy", kind="binary")
    with pytest.raises(ob.SearchError):
        ob.Objective(name="macro_f1", kind="binary")


# ---------------------------------------------------------------------------
# the randomized sampler is the one RandomizedSearchCV would have used
# ---------------------------------------------------------------------------


def test_sampled_values_stay_inside_the_declared_distributions() -> None:
    """SO-01 does not use sklearn's class, so its draws are checked against the space.

    ``to_distributions`` is the scipy mapping ``RandomizedSearchCV`` would have
    been given. Sampling both and comparing point-for-point is impossible -- the
    two consume different RNG streams -- so what is pinned instead is that every
    drawn value lies in the same declared support, and that the constrained
    dimensions the sklearn class cannot express are respected.
    """
    from src.models import spaces

    for model_id in ("M1", "M3", "M4", "M5", "M8"):
        space = spaces.load_space(model_id)
        distributions = space.to_distributions()
        assert set(distributions) == set(space.names)
        rng = np.random.default_rng(42)
        for point in space.sample_many(40, rng):
            assert space.is_valid(point), (model_id, point)
            for name, value in point.items():
                assert space.dimension(name).contains(value)


def test_m1_constraint_holds_over_the_whole_draw() -> None:
    """The reason SO-01 is not RandomizedSearchCV: illegal combinations are excluded."""
    from src.models import spaces

    space = spaces.load_space("M1")
    rng = np.random.default_rng(7)
    for point in space.sample_many(200, rng):
        if point["solver"] == "lbfgs":
            assert point["l1_ratio"] == 0.0
        if point["solver"] == "liblinear":
            assert point["l1_ratio"] in {0.0, 1.0}


# ---------------------------------------------------------------------------
# the same assertion, against the real D1 matrix and the real DA-07 fold map
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_d1() -> tuple[Any, Any]:
    from src.models import smoke as sm

    try:
        loaded = sm.load_task_data("binary")
        fold = sm._fold_zero("binary", loaded)
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 fold 0 unavailable (" + type(error).__name__ + "): " + str(error))
    return loaded, fold


def test_real_d1_search_never_touches_the_outer_test_fold(real_d1: tuple[Any, Any]) -> None:
    """The claim that matters, on the matrix and fold map the results come from.

    Two trials of the cheapest model: this is a leakage assertion, not a search.
    """
    from src.models import spaces

    loaded, fold = real_d1
    search = RandomizedSearch(
        "M1",
        spaces.load_space("M1"),
        ob.objective_for("binary"),
        ob.Budget(max_trials=2),
        seed=42,
    )
    run = search.run(loaded, fold, n_inner_splits=3)

    forbidden = frozenset(int(value) for value in np.asarray(fold.test_index))
    assert forbidden, "fold 0 has no test rows"
    assert not (run.ledger.touched & forbidden)
    assert run.ledger.touched == frozenset(int(v) for v in np.asarray(fold.train_index))
    assert len(run.trials) == 2


def test_real_d1_inner_folds_keep_subjects_whole(real_d1: tuple[Any, Any]) -> None:
    """Subject leakage, on real subject ids rather than invented ones (rule 3)."""
    loaded, fold = real_d1
    groups = np.asarray(loaded.groups, dtype=object)
    test_groups = set(groups[np.asarray(fold.test_index, dtype=int)])
    for split in ob.inner_folds(fold, loaded.y, loaded.groups, n_splits=3):
        train_groups = set(groups[split.train_index])
        val_groups = set(groups[split.val_index])
        assert not train_groups & val_groups
        assert not (train_groups | val_groups) & test_groups
