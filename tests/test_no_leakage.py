"""DA-07 leakage gate (T20.6, T20.7).

Rule 3, checked against the artifact rather than the generator. Every
experiment in Part VII loads ``subject_split_map.csv``; if a subject spans two
folds in that file, nothing in this project will crash -- the metrics will
simply come back higher than they should, and stay wrong all the way into the
paper. So this is the file the tests read.

Three properties:

1. Zero subject overlap between any two folds of any repeat, in every task.
2. The 301 PhysioNet ``validation/`` records appear in no fold.
3. Two runs of the generator produce identical maps (rule 5).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.data_loader import splits as sp

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> Any:
    from src.data_loader.master import build_master

    return build_master()


@pytest.fixture(scope="module")
def split_map(master: Any) -> Any:
    return sp.build_split_maps(master)


# ===========================================================================
# schemes
# ===========================================================================


def test_every_task_has_a_scheme() -> None:
    from src.utils.constants import TASKS

    assert set(sp.TASK_SCHEMES) == set(TASKS)


def test_schemes_resolve_from_the_config() -> None:
    """The fold counts T20.2 -- T20.4 ask for, read from the config not hardcoded."""
    expected = {
        "binary": (5, 5, 25),
        "pascal_a": (2, 5, 10),
        "pascal_b": (5, 1, 5),
        "circor_murmur": (5, 1, 5),
        "circor_outcome": (5, 1, 5),
    }
    for task, (n_splits, n_repeats, total) in expected.items():
        scheme = sp.load_scheme(sp.TASK_SCHEMES[task])
        assert (scheme.n_splits, scheme.n_repeats, scheme.total_folds) == (
            n_splits,
            n_repeats,
            total,
        ), task
        assert scheme.shuffle is True
        assert scheme.random_state == 42


def test_unknown_scheme_raises() -> None:
    with pytest.raises(KeyError, match="unknown cv scheme"):
        sp.load_scheme("no_such_scheme")


# ===========================================================================
# T20.7 -- THE GATE: zero subject overlap
# ===========================================================================


@pytest.mark.needs_data
def test_zero_subject_overlap_in_every_fold_of_every_task(split_map: Any) -> None:
    """The assertion T20.6 names, spelled out here rather than delegated."""
    for task in sorted(set(split_map["task"])):
        subset = split_map[split_map["task"] == task]
        for repeat, block in subset.groupby("repeat"):
            folds = sorted(set(block["fold"]))
            seen: dict[str, int] = {}
            for fold in folds:
                for group in block.loc[block["fold"] == fold, "split_group"]:
                    if group in seen and seen[group] != fold:
                        raise AssertionError(
                            task + " repeat " + str(repeat) + ": subject " + group
                            + " is in folds " + str(seen[group]) + " and " + str(fold)
                        )
                    seen[group] = fold
    sp.assert_no_leakage(split_map)


@pytest.mark.needs_data
def test_train_and_test_never_share_a_subject(split_map: Any) -> None:
    """The same rule stated the way an experiment will actually use the map."""
    for task in sorted(set(split_map["task"])):
        groups = dict(
            zip(split_map["record_uid"], split_map["split_group"], strict=True)
        )
        for repeat, fold, train, test in sp.iter_folds(split_map, task):
            shared = {groups[u] for u in train} & {groups[u] for u in test}
            assert not shared, (task, repeat, fold, sorted(shared)[:3])
            assert not set(train) & set(test)


def test_assert_no_leakage_catches_a_planted_leak() -> None:
    """A check that has only seen clean data proves nothing -- so break it."""
    import pandas as pd

    clean = pd.DataFrame(
        {
            "task": ["binary"] * 4,
            "repeat": [0, 0, 0, 0],
            "fold": [0, 0, 1, 1],
            "record_uid": ["r1", "r2", "r3", "r4"],
            "split_group": ["s1", "s1", "s2", "s2"],
        }
    )
    sp.assert_no_leakage(clean)

    leaked = clean.copy()
    leaked.loc[1, "fold"] = 1  # subject s1 now spans folds 0 and 1
    with pytest.raises(ValueError, match="rule 3 violated"):
        sp.assert_no_leakage(leaked)

    doubled = clean.copy()
    doubled.loc[3, "record_uid"] = "r1"  # one record assigned twice
    with pytest.raises(ValueError, match="more than one fold"):
        sp.assert_no_leakage(doubled)


# ===========================================================================
# T20.5 / T20.7 -- the 301 validation records are in no fold
# ===========================================================================


@pytest.mark.needs_data
def test_the_301_validation_records_appear_in_no_cv_fold(
    split_map: Any, master: Any
) -> None:
    validation = set(master.loc[master["subset"] == "validation", "record_uid"])
    assert len(validation) == 301
    assert not validation & set(split_map["record_uid"])
    sp.assert_validation_excluded(split_map, master)


@pytest.mark.needs_data
def test_excluding_the_copies_did_not_exclude_the_originals(
    split_map: Any, master: Any
) -> None:
    """The 301 training records the validation folder duplicates stay in the map.

    Dropping both halves of each duplicate pair would silently shrink the
    primary track from 3,240 records to 2,939.
    """
    twins = set(master.loc[master["subset"] == "validation", "duplicate_of"]) - {""}
    assert len(twins) == 301
    binary = set(split_map.loc[split_map["task"] == "binary", "record_uid"])
    assert twins <= binary


@pytest.mark.needs_data
def test_assert_validation_excluded_catches_a_planted_validation_record(
    split_map: Any, master: Any
) -> None:
    import pandas as pd

    intruder = master[master["subset"] == "validation"].iloc[0]
    planted = pd.concat(
        [
            split_map,
            split_map.head(1).assign(record_uid=intruder["record_uid"]),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="validation record"):
        sp.assert_validation_excluded(planted, master)


# ===========================================================================
# coverage and shape
# ===========================================================================


@pytest.mark.needs_data
def test_every_supervised_record_is_assigned_exactly_once_per_repeat(
    split_map: Any, master: Any
) -> None:
    from src.data_loader.master import task_frame

    for task in sp.TASK_SCHEMES:
        expected = set(task_frame(master, task)["record_uid"])
        subset = split_map[split_map["task"] == task]
        for repeat, block in subset.groupby("repeat"):
            assert set(block["record_uid"]) == expected, (task, repeat)
            assert not block["record_uid"].duplicated().any(), (task, repeat)


@pytest.mark.needs_data
def test_fold_counts_match_the_scheme(split_map: Any) -> None:
    for task, scheme_name in sp.TASK_SCHEMES.items():
        scheme = sp.load_scheme(scheme_name)
        subset = split_map[split_map["task"] == task]
        assert subset["repeat"].nunique() == scheme.n_repeats, task
        assert set(subset["fold"]) == set(range(scheme.n_splits)), task
        assert len(list(sp.iter_folds(split_map, task))) == scheme.total_folds, task


@pytest.mark.needs_data
def test_every_fold_contains_every_class(split_map: Any) -> None:
    """Stratification actually stratified.

    A fold missing a class makes per-class recall undefined for that fold, and
    the aggregate silently averages over a different number of folds per class.
    """
    summary = sp.fold_summary(split_map)
    expected_classes = {
        "binary": 2,
        "pascal_a": 4,
        "pascal_b": 3,
        "circor_murmur": 3,
        "circor_outcome": 2,
    }
    for task, n_classes in expected_classes.items():
        rows = summary[summary["task"] == task]
        assert (rows["n_classes"] == n_classes).all(), task
        assert (rows["min_class_count"] >= 1).all(), task


@pytest.mark.needs_data
def test_the_pascal_a_duplicate_pair_is_never_split(split_map: Any) -> None:
    """Phase 17's contradictory pair shares a group, so it shares a fold."""
    pair = ["D2_set_a_extrahls__201104021355", "D2_set_a_murmur__201104021355"]
    rows = split_map[
        (split_map["task"] == "pascal_a") & split_map["record_uid"].isin(pair)
    ]
    assert len(rows) == 10  # 2 records x 5 repeats
    for repeat, block in rows.groupby("repeat"):
        assert block["fold"].nunique() == 1, repeat


@pytest.mark.needs_data
def test_circor_folds_are_patient_wise_and_the_two_tasks_split_independently(
    split_map: Any,
) -> None:
    """T20.4 -- murmur and outcome get their own maps, not one shared map."""
    murmur = split_map[split_map["task"] == "circor_murmur"]
    outcome = split_map[split_map["task"] == "circor_outcome"]
    assert set(murmur["record_uid"]) == set(outcome["record_uid"])
    # Grouped on the native patient id, not on the recording.
    assert (murmur["subject_derived"] == False).all()  # noqa: E712
    assert murmur["split_group"].nunique() == 942
    # Independently stratified: the same patient need not land in the same fold
    # for both tasks, and requiring that would couple two separate label spaces.
    merged = murmur.merge(outcome, on="record_uid", suffixes=("_m", "_o"))
    assert (merged["fold_m"] != merged["fold_o"]).any()


# ===========================================================================
# rule 5 -- determinism
# ===========================================================================


@pytest.mark.needs_data
def test_regenerating_the_map_reproduces_it_exactly(master: Any, split_map: Any) -> None:
    again = sp.build_split_maps(master)
    assert list(again.columns) == list(split_map.columns)
    assert again.equals(split_map)


@pytest.mark.needs_data
def test_each_repeat_produces_a_different_partition(split_map: Any) -> None:
    """Five repeats of the same shuffle would make n=25 a fiction.

    The repeated 5x5 protocol exists because Wilcoxon on n=5 folds cannot reach
    significance; if the repeats were identical, n would still be 5.
    """
    binary = split_map[split_map["task"] == "binary"]
    signatures = {
        repeat: tuple(
            block.sort_values("record_uid")["fold"].tolist()
        )
        for repeat, block in binary.groupby("repeat")
    }
    assert len(set(signatures.values())) == 5


# ===========================================================================
# T20.6 -- the written artifact
# ===========================================================================


@pytest.mark.needs_data
def test_written_map_round_trips_and_still_passes_the_leakage_check(
    split_map: Any, master: Any, tmp_path: Any
) -> None:
    path = sp.write_split_map(split_map, tmp_path)
    assert path.is_file() and path.stat().st_size > 0

    reloaded = sp.load_split_map(path)
    assert list(reloaded.columns) == list(sp.SPLIT_MAP_COLUMNS)
    assert len(reloaded) == len(split_map)
    sp.assert_no_leakage(reloaded)
    sp.assert_validation_excluded(reloaded, master)
