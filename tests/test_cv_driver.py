"""The CV driver (Phase 43, gate T43.7).

The gate has two clauses:

1. the driver **loads folds from DA-07 rather than re-deriving them**;
2. out-of-fold predictions **cover every record exactly once**.

Clause 1 is the awkward one to test, because "it did not compute the folds" is a
statement about what the code *didn't* do. Asserting that ``load_folds`` returns
the right answer proves nothing -- a re-derivation would return the same answer
on the same data, which is exactly why the bug would be invisible in practice.
So it is tested three ways: the map is rewritten with deliberately *wrong* folds
and the driver must follow them; the map is deleted and the driver must fail
rather than fall back to computing splits; and the fold count is cross-checked
against ``experiments.yaml``.

Everything except the final section runs on synthetic data with known groups, so
the leakage assertions can be exercised on splits that genuinely leak.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.evaluation import cv

TASK = "toy"
# The toy map declares a scheme experiments.yaml really contains, with
# matching counts: 5 repeats x 2 folds. A fixture that disagreed with
# config would trip the cross-check before reaching what is being tested.
SCHEME = "repeated_5x2_stratified"


# ---------------------------------------------------------------------------
# a fabricated DA-07 map
# ---------------------------------------------------------------------------


def _write_map(
    path: Path,
    *,
    n_records: int = 20,
    n_folds: int = 2,
    n_repeats: int = 5,
    scheme: str = SCHEME,
    leak_group: bool = False,
) -> Any:
    """Write a split map in DA-07's own format.

    Groups pair up records (two recordings per subject), which is what makes a
    grouped split meaningful: a splitter that ignored groups would separate the
    two halves of a subject and the leakage assertion would catch it.
    """
    import pandas as pd

    rows = []
    for repeat in range(n_repeats):
        for index in range(n_records):
            group = "S" + format(index // 2, "02d")
            # leak_group moves one recording away from its subject's fold, so
            # subject S00 straddles two folds and the invariant must fire.
            leaked = leak_group and index == 1
            fold = (index // 2 + 1) % n_folds if leaked else (index // 2) % n_folds
            rows.append(
                {
                    "task": TASK,
                    "scheme": scheme,
                    "dataset_source": "D1",
                    "n_splits": n_folds,
                    "n_repeats": n_repeats,
                    "repeat": repeat,
                    "fold": fold,
                    "record_uid": "R" + format(index, "03d"),
                    "split_group": group,
                    "subject_id": group,
                    "subject_derived": True,
                    "y": index % 2,
                    "class_name": "abnormal" if index % 2 else "normal",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


@pytest.fixture
def fake_map(tmp_path: Path, monkeypatch: Any) -> Path:
    path = tmp_path / cv.SPLIT_MAP_FILENAME
    _write_map(path)
    monkeypatch.setattr(cv, "split_map_path", lambda: path)
    return path


class CountingEstimator:
    """A trivial classifier that records how many times it was constructed."""

    built = 0

    def __init__(self) -> None:
        type(self).built += 1
        self.seen_rows = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> CountingEstimator:
        self.classes_ = np.unique(y)
        self.seen_rows = len(X)
        self._majority = self.classes_[np.argmax([np.sum(y == k) for k in self.classes_])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._majority)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, int(np.where(self.classes_ == self._majority)[0][0])] = 1.0
        return proba


def _xyg(frame: Any, n_features: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Feature matrix aligned to the map's record order within one repeat."""
    single = frame[frame["repeat"] == 0].reset_index(drop=True)
    rng = np.random.default_rng(42)
    uids = single["record_uid"].astype(str).tolist()
    X = rng.normal(size=(len(uids), n_features))
    y = single["y"].to_numpy()
    groups = single["split_group"].astype(str).to_numpy()
    return X, y, groups, uids


# ---------------------------------------------------------------------------
# T43.1 / clause 1 -- the folds come from the file
# ---------------------------------------------------------------------------


def test_folds_are_read_from_the_map(fake_map: Path):
    folds = cv.load_folds(TASK, validate=False)
    assert len(folds) == 10  # 5 repeats x 2 folds
    assert {fold.repeat for fold in folds} == {0, 1, 2, 3, 4}
    assert {fold.fold for fold in folds} == {0, 1}
    assert folds[0].task == TASK
    assert folds[0].scheme == SCHEME


def test_the_driver_follows_a_deliberately_wrong_map(tmp_path: Path, monkeypatch: Any):
    """Clause 1, stated as a falsifiable claim.

    The map is rewritten so every record sits in fold 0. No splitter would ever
    produce that. If ``load_folds`` re-derived folds it would disagree; because
    it reads the file, it reproduces the nonsense exactly.
    """
    import pandas as pd

    path = tmp_path / cv.SPLIT_MAP_FILENAME
    frame = _write_map(path)
    frame["fold"] = 0
    frame.to_csv(path, index=False)
    monkeypatch.setattr(cv, "split_map_path", lambda: path)

    folds = cv.load_folds(TASK, validate=False)
    assert len(folds) == 5  # one "fold" per repeat, exactly as written
    assert all(fold.fold == 0 for fold in folds)
    assert all(fold.n_train == 0 for fold in folds)
    assert pd.read_csv(path)["fold"].nunique() == 1


def test_a_missing_map_is_an_error_not_a_fallback(tmp_path: Path, monkeypatch: Any):
    """There must be no code path that computes folds when the file is absent."""
    monkeypatch.setattr(cv, "split_map_path", lambda: tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="never re-derived"):
        cv.load_folds(TASK)


def test_an_unknown_task_is_rejected_with_the_available_ones(fake_map: Path):
    with pytest.raises(ValueError, match="no folds for task"):
        cv.load_folds("not_a_task")


def test_train_is_the_complement_within_the_same_repeat(fake_map: Path):
    """Across repeats it would put a record in its own training set."""
    folds = cv.load_folds(TASK, validate=False)
    for fold in folds:
        assert set(fold.train_uids) & set(fold.test_uids) == set()
        assert fold.n_train + fold.n_test == 20


def test_the_fold_count_is_cross_checked_against_config(tmp_path: Path, monkeypatch: Any):
    """DA-07 and experiments.yaml disagreeing must stop the run, not be resolved."""
    path = tmp_path / cv.SPLIT_MAP_FILENAME
    _write_map(path, n_folds=3, n_repeats=1, scheme="grouped_5fold")
    monkeypatch.setattr(cv, "split_map_path", lambda: path)

    with pytest.raises(ValueError, match="config declares"):
        cv.load_folds(TASK)


def test_a_scheme_config_never_heard_of_is_rejected(tmp_path: Path, monkeypatch: Any):
    """A map and a config that have drifted apart must stop the run."""
    path = tmp_path / cv.SPLIT_MAP_FILENAME
    _write_map(path, scheme="invented_scheme")
    monkeypatch.setattr(cv, "split_map_path", lambda: path)

    with pytest.raises(ValueError, match="does not declare"):
        cv.load_folds(TASK)


# ---------------------------------------------------------------------------
# T43.5 -- the leakage invariant
# ---------------------------------------------------------------------------


def test_a_leaked_group_is_rejected_at_load(tmp_path: Path, monkeypatch: Any):
    path = tmp_path / cv.SPLIT_MAP_FILENAME
    _write_map(path, leak_group=True)
    monkeypatch.setattr(cv, "split_map_path", lambda: path)

    with pytest.raises(cv.LeakageError, match="both train and test"):
        cv.load_folds(TASK)


def test_leakage_error_is_an_assertion_error():
    """It must not be swallowed by a bare ``except Exception`` further up."""
    assert issubclass(cv.LeakageError, AssertionError)


def test_run_cv_checks_the_rows_it_is_actually_given(fake_map: Path):
    """A correct map resolved against the wrong ordering still leaks.

    The map-level check passes here; only the runtime check on the arrays that
    reach the estimator catches it.
    """
    import dataclasses

    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK, validate=False), uids)

    tampered = dataclasses.replace(
        folds[0], train_index=np.append(folds[0].train_index, folds[0].test_index[0])
    )
    with pytest.raises(cv.LeakageError, match="both train and test"):
        cv.run_cv(CountingEstimator, X, y, groups, [tampered])


def test_run_cv_catches_a_group_shared_through_the_group_array(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, _, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK, validate=False), uids)

    # Every record now belongs to one subject: any split leaks.
    with pytest.raises(cv.LeakageError, match="group"):
        cv.run_cv(CountingEstimator, X, y, np.full(len(y), "S00"), folds[:1])


# ---------------------------------------------------------------------------
# T43.2 -- the driver
# ---------------------------------------------------------------------------


def test_run_cv_returns_a_result_per_fold(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    result = cv.run_cv(CountingEstimator, X, y, groups, folds, task=TASK)
    assert result.n_folds == len(folds)
    assert result.repeats == (0, 1, 2, 3, 4)
    assert result.n_records == len(y)
    assert all(item.y_pred.shape == (item.y_true.shape[0],) for item in result.folds)


def test_a_fresh_estimator_is_built_for_every_fold(fake_map: Path):
    """Reusing one instance would carry fold k's fitted state into fold k+1."""
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    CountingEstimator.built = 0
    cv.run_cv(CountingEstimator, X, y, groups, folds)
    assert CountingEstimator.built == len(folds)


def test_the_estimator_only_ever_sees_training_rows(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    result = cv.run_cv(CountingEstimator, X, y, groups, folds)
    for item, fold in zip(result.folds, folds, strict=True):
        assert item.n_train == fold.n_train
        assert item.n_train == len(y) - len(item.test_uids)


def test_probabilities_are_captured_when_available(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    result = cv.run_cv(CountingEstimator, X, y, groups, folds)
    for item in result.folds:
        assert item.y_proba is not None
        assert item.y_proba.shape == (len(item.test_uids), len(item.classes))
        assert np.allclose(item.y_proba.sum(axis=1), 1.0)


def test_a_hard_classifier_is_accepted_without_probabilities(fake_map: Path):
    class HardOnly(CountingEstimator):
        predict_proba = None  # type: ignore[assignment]

        def __getattribute__(self, name: str) -> Any:
            if name == "predict_proba":
                raise AttributeError(name)
            return super().__getattribute__(name)

    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    result = cv.run_cv(HardOnly, X, y, groups, folds)
    assert all(item.y_proba is None for item in result.folds)


def test_unresolved_folds_are_rejected(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, _ = _xyg(frame)
    with pytest.raises(ValueError, match="resolved"):
        cv.run_cv(CountingEstimator, X, y, groups, cv.load_folds(TASK))


def test_mismatched_lengths_are_rejected(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)
    with pytest.raises(ValueError, match="disagree on length"):
        cv.run_cv(CountingEstimator, X, y[:-1], groups, folds)


def test_resolve_folds_refuses_a_matrix_missing_records(fake_map: Path):
    folds = cv.load_folds(TASK, validate=False)
    with pytest.raises(ValueError, match="absent from the matrix"):
        cv.resolve_folds(folds, ["R000", "R001"])


def test_resolve_folds_rejects_duplicate_uids(fake_map: Path):
    folds = cv.load_folds(TASK, validate=False)
    with pytest.raises(ValueError, match="duplicates"):
        cv.resolve_folds(folds, ["R000", "R000"])


# ---------------------------------------------------------------------------
# clause 2 -- out-of-fold coverage
# ---------------------------------------------------------------------------


def test_oof_predictions_cover_every_record_exactly_once_per_repeat(fake_map: Path):
    """T43.7, second clause -- verbatim, and per repeat rather than overall."""
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    oof = cv.run_cv(CountingEstimator, X, y, groups, folds).oof_frame()
    for repeat, group in oof.groupby("repeat"):
        assert len(group) == len(uids)
        assert set(group["record_uid"]) == set(uids)
        assert group["record_uid"].is_unique, "repeat " + str(repeat)


def test_repeats_are_kept_separate_rather_than_collapsed(fake_map: Path):
    """Five repeats give a record five OOF predictions; the pairing needs all of them."""
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    oof = cv.run_cv(CountingEstimator, X, y, groups, folds).oof_frame()
    assert len(oof) == 5 * len(uids)
    assert oof.groupby("record_uid").size().eq(5).all()


def test_oof_y_true_matches_the_supplied_labels(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    oof = cv.run_cv(CountingEstimator, X, y, groups, folds).oof_frame()
    truth = dict(zip(uids, y, strict=True))
    for row in oof.itertuples(index=False):
        assert row.y_true == truth[row.record_uid]


def test_a_record_tested_twice_in_one_repeat_is_caught(fake_map: Path):
    import dataclasses

    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = list(cv.resolve_folds(cv.load_folds(TASK, validate=False), uids))

    first = folds[0]
    duplicate = dataclasses.replace(
        folds[1],
        test_uids=first.test_uids,
        test_index=first.test_index,
        train_uids=first.train_uids,
        train_index=first.train_index,
        train_groups=first.train_groups,
        test_groups=first.test_groups,
    )
    with pytest.raises(cv.LeakageError, match="tested more than once"):
        cv.run_cv(CountingEstimator, X, y, groups, [first, duplicate])


# ---------------------------------------------------------------------------
# T43.3 -- persistence
# ---------------------------------------------------------------------------


def test_predictions_and_membership_are_persisted(fake_map: Path, tmp_path: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)
    result = cv.run_cv(CountingEstimator, X, y, groups, folds, task=TASK)

    written = cv.save_cv_result(result, tmp_path / "cv", folds=folds)
    assert set(written) == {"oof", "timings", "membership"}
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0

    restored = cv.load_oof_predictions(tmp_path / "cv")
    assert len(restored) == len(result.oof_frame())
    assert set(restored["record_uid"]) == set(uids)


def test_membership_records_both_sides_of_every_fold(fake_map: Path, tmp_path: Path):
    """Paired tests in Part VIII are only valid if the folds are provably identical."""
    import pandas as pd

    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)
    result = cv.run_cv(CountingEstimator, X, y, groups, folds)

    cv.save_cv_result(result, tmp_path / "cv", folds=folds)
    membership = pd.read_parquet(tmp_path / "cv" / "fold_membership.parquet")
    assert set(membership["split"]) == {"train", "test"}
    assert len(membership) == sum(fold.n_train + fold.n_test for fold in folds)


def test_fold_timings_are_recorded(fake_map: Path):
    frame = cv.load_split_map(TASK)
    X, y, groups, uids = _xyg(frame)
    folds = cv.resolve_folds(cv.load_folds(TASK), uids)

    timings = cv.run_cv(CountingEstimator, X, y, groups, folds).fold_frame()
    assert len(timings) == len(folds)
    assert (timings["fit_seconds"] >= 0).all()
    assert (timings["n_train"] > 0).all()
    assert (timings["n_test"] > 0).all()


# ---------------------------------------------------------------------------
# T43.4 -- the real DA-07 map
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_every_real_task_loads_with_the_scheme_the_config_declares():
    expected = {
        "binary": ("repeated_5x5_grouped", 25),
        "pascal_a": ("repeated_5x2_stratified", 10),
        "pascal_b": ("grouped_5fold", 5),
        "circor_murmur": ("patient_grouped_5fold", 5),
        "circor_outcome": ("patient_grouped_5fold", 5),
    }
    assert set(cv.available_tasks()) == set(expected)

    for task, (scheme, n_folds) in expected.items():
        folds = cv.load_folds(task)
        assert cv.scheme_for(task) == scheme, task
        assert len(folds) == n_folds, task


@pytest.mark.needs_data
def test_the_primary_track_is_5x5_and_pascal_a_is_5x2():
    """T43.4 -- the two repeated schemes, checked as repeats x splits."""
    binary = cv.load_folds("binary")
    assert len({fold.repeat for fold in binary}) == 5
    assert len({fold.fold for fold in binary}) == 5

    pascal = cv.load_folds("pascal_a")
    assert len({fold.repeat for fold in pascal}) == 5
    assert len({fold.fold for fold in pascal}) == 2


@pytest.mark.needs_data
def test_no_real_fold_shares_a_group_between_train_and_test():
    """Research rule 3 across every fold of every task that exists."""
    for task in cv.available_tasks():
        for fold in cv.load_folds(task, validate=False):
            cv.assert_group_disjoint(fold)


@pytest.mark.needs_data
def test_real_folds_partition_their_repeat_completely():
    for task in cv.available_tasks():
        folds = cv.load_folds(task)
        by_repeat: dict[int, list[Any]] = {}
        for fold in folds:
            by_repeat.setdefault(fold.repeat, []).append(fold)

        for repeat, group in by_repeat.items():
            tested: list[str] = []
            for fold in group:
                tested.extend(fold.test_uids)
            assert len(tested) == len(set(tested)), task + " repeat " + str(repeat)
            assert set(tested) == set(group[0].train_uids) | set(group[0].test_uids)


@pytest.mark.needs_data
def test_fold_sizes_match_the_published_da07_summary():
    """The audit published fold sizes; the driver must reproduce them exactly."""
    import pandas as pd

    from src.utils.config import load_config

    path = (
        Path(load_config("paths").require("outputs.dataset_audit")) / cv.SUMMARY_FILENAME
    )
    summary = pd.read_csv(path)

    for task in cv.available_tasks():
        published = summary[summary["task"].astype(str) == task]
        if published.empty:
            continue
        sizes = {
            (int(row.repeat), int(row.fold)): int(row.n_records)
            for row in published.itertuples(index=False)
        }
        for fold in cv.load_folds(task, validate=False):
            assert fold.n_test == sizes[fold.key], (
                task + " " + fold.label + ": driver says " + str(fold.n_test)
            )
