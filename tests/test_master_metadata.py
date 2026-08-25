"""DA-08 master metadata gate (T19.6, T19.7).

Three properties this table has to hold, because everything from Part III on
loads it instead of walking the dataset tree:

1. ``record_uid`` is globally unique and follows the T19.3 rule.
2. Every ``file_path`` resolves on disk.
3. **No record carries a label for a task it does not belong to** -- rule 4,
   asserted rather than assumed, in both directions.

The counts asserted here come from DA-02 ``class_distribution.csv``, which is
itself generated. If a count moves, one of the loaders changed and the class
distribution in the paper moved with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.data_loader import master as ms

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> Any:
    return ms.build_master()


# ===========================================================================
# T19.1 / T19.4 -- schema and dtypes
# ===========================================================================


def test_schema_columns_are_the_t19_1_list_in_order() -> None:
    assert ms.MASTER_SCHEMA == (
        "record_uid",
        "dataset_source",
        "subset",
        "subject_id",
        "subject_derived",
        "record_id",
        "file_path",
        "original_fs",
        "duration_sec",
        "n_samples",
        "recording_location",
        "binary_label",
        "multiclass_label",
        "murmur_label",
        "outcome_label",
        "diagnosis_class",
        "quality_flags",
        "is_duplicate",
        "is_unlabeled",
        "split_group",
    )


def test_every_declared_column_has_a_dtype() -> None:
    for column in ms.MASTER_SCHEMA + ms.EXTENSION_COLUMNS:
        assert column in ms.MASTER_DTYPES, column


def test_label_columns_are_nullable_integers() -> None:
    """A missing label must be NA, never 0.

    ``normal`` is class 0 in three of the five label spaces. A plain ``int64``
    column would have to fill missing values with something, and every candidate
    collides with a real class.
    """
    for column in ("binary_label", "multiclass_label", "murmur_label", "outcome_label"):
        assert ms.MASTER_DTYPES[column] == "Int64"


@pytest.mark.needs_data
def test_master_passes_its_own_validator(master: Any) -> None:
    ms.validate_master(master)
    assert list(master.columns)[: len(ms.MASTER_SCHEMA)] == list(ms.MASTER_SCHEMA)
    for column, expected in ms.MASTER_DTYPES.items():
        assert str(master[column].dtype) == expected, column


# ===========================================================================
# T19.3 -- record_uid
# ===========================================================================


def test_make_record_uid_joins_the_three_parts() -> None:
    assert ms.make_record_uid("D1", "training-a", "a0001") == "D1_training-a_a0001"
    assert ms.make_record_uid("D4", "training_data", "50782_MV") == (
        "D4_training_data_50782_MV"
    )


def test_make_record_uid_refuses_an_empty_part() -> None:
    for args in (("", "s", "r"), ("D1", "", "r"), ("D1", "s", "  ")):
        with pytest.raises(ValueError, match="non-empty"):
            ms.make_record_uid(*args)


@pytest.mark.needs_data
def test_record_uids_are_unique_and_follow_the_rule(master: Any) -> None:
    assert not master["record_uid"].duplicated().any()
    rebuilt = [
        ms.make_record_uid(row.dataset_source, row.subset, row.record_id)
        for row in master.itertuples(index=False)
    ]
    assert list(master["record_uid"]) == rebuilt


# ===========================================================================
# T19.7 -- every file path resolves on disk
# ===========================================================================


@pytest.mark.needs_data
@pytest.mark.slow
def test_every_file_path_resolves_on_disk(master: Any) -> None:
    root = Path(ms._project_root())
    missing = [
        row.record_uid
        for row in master.itertuples(index=False)
        if not (root / str(row.file_path)).is_file()
    ]
    assert missing == []


@pytest.mark.needs_data
def test_file_paths_are_relative_and_inside_the_dataset_tree(master: Any) -> None:
    """Nothing absolute, nothing outside ``dataset/``.

    An absolute path in a committed CSV pins the table to this machine; a path
    outside ``dataset/`` means a loader picked up something that is not input
    data.
    """
    paths = master["file_path"].astype(str)
    assert not paths.str.contains(r"^[A-Za-z]:", regex=True).any()
    assert not paths.str.startswith("/").any()
    assert paths.str.startswith("dataset/").all()


# ===========================================================================
# T19.2 / T19.7 -- THE GATE: no cross-task label bleed
# ===========================================================================


@pytest.mark.needs_data
def test_no_record_carries_a_label_for_a_task_it_does_not_belong_to(
    master: Any,
) -> None:
    """The hard assertion T19.7 names. Checked column by column, not by helper."""
    d1 = master["dataset_source"] == "D1"
    d23 = master["dataset_source"].isin(["D2", "D3"])
    d4 = master["dataset_source"] == "D4"

    assert master.loc[~d1, "binary_label"].isna().all()
    assert master.loc[~d23, "multiclass_label"].isna().all()
    assert master.loc[~d4, "murmur_label"].isna().all()
    assert master.loc[~d4, "outcome_label"].isna().all()
    assert (master.loc[~d1, "diagnosis_class"].fillna("") == "").all()

    ms.assert_no_cross_task_bleed(master)


@pytest.mark.needs_data
def test_pascal_a_and_pascal_b_are_distinguishable(master: Any) -> None:
    """The reason ``multiclass_task`` exists.

    Both label spaces number ``normal`` 0 and ``murmur`` 1, so the numeric
    column alone cannot tell them apart. Every labelled PASCAL record must say
    which of the two it is in.
    """
    labelled = master[master["multiclass_label"].notna()]
    assert (labelled["multiclass_task"] != "").all()
    assert set(labelled["multiclass_task"]) == {"pascal_a", "pascal_b"}
    a = labelled[labelled["multiclass_task"] == "pascal_a"]
    b = labelled[labelled["multiclass_task"] == "pascal_b"]
    assert set(a["dataset_source"]) == {"D2"}
    assert set(b["dataset_source"]) == {"D3"}
    # The overlap that would be invisible without the task column.
    assert set(a["multiclass_label"]) == {0, 1, 2, 3}
    assert set(b["multiclass_label"]) == {0, 1, 2}


def test_assert_no_cross_task_bleed_catches_a_planted_bleed() -> None:
    """Never weaken a test to make it pass -- so prove the check can fail.

    A validator that has only ever seen clean data is not evidence of anything.
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "record_uid": [
                "D1_training-a_a0001",
                "D2_set_a_normal__201101070538",
                "D4_training_data_50782_MV",
            ],
            "dataset_source": ["D1", "D2", "D4"],
            "use_in_supervised": [True, True, True],
            "binary_label": pd.array([1, pd.NA, pd.NA], dtype="Int64"),
            "multiclass_label": pd.array([pd.NA, 0, pd.NA], dtype="Int64"),
            "murmur_label": pd.array([pd.NA, pd.NA, 0], dtype="Int64"),
            "outcome_label": pd.array([pd.NA, pd.NA, 1], dtype="Int64"),
            "multiclass_task": ["", "pascal_a", ""],
        }
    )
    ms.assert_no_cross_task_bleed(frame)

    bled = frame.copy()
    bled.loc[2, "binary_label"] = 1  # a CirCor record given a PhysioNet label
    with pytest.raises(ValueError, match="label spaces have been merged"):
        ms.assert_no_cross_task_bleed(bled)

    ambiguous = frame.copy()
    ambiguous.loc[1, "multiclass_task"] = ""  # a label with no label space
    with pytest.raises(ValueError, match="ambiguous"):
        ms.assert_no_cross_task_bleed(ambiguous)

    swapped = frame.copy()
    swapped.loc[1, "multiclass_task"] = "pascal_b"  # set_a row in set_b's space
    with pytest.raises(ValueError, match="PASCAL A and PASCAL B have been merged"):
        ms.assert_no_cross_task_bleed(swapped)


@pytest.mark.needs_data
def test_task_frame_returns_only_that_tasks_rows(master: Any) -> None:
    expected = {
        "binary": ("D1", 3240, {0: 2575, 1: 665}),
        "pascal_a": ("D2", 124, {0: 31, 1: 34, 2: 19, 3: 40}),
        "pascal_b": ("D3", 461, {0: 320, 1: 95, 2: 46}),
        "circor_murmur": ("D4", 3163, {0: 2391, 1: 616, 2: 156}),
        "circor_outcome": ("D4", 3163, {0: 1632, 1: 1531}),
    }
    for task, (dataset, n_records, counts) in expected.items():
        frame = ms.task_frame(master, task)
        assert len(frame) == n_records, task
        assert set(frame["dataset_source"]) == {dataset}, task
        assert {int(k): int(v) for k, v in frame["y"].value_counts().items()} == counts
        assert frame["y"].notna().all()


def test_task_frame_rejects_an_unknown_task() -> None:
    import pandas as pd

    with pytest.raises(KeyError, match="unknown task"):
        ms.task_frame(pd.DataFrame(), "pascal_c")


# ===========================================================================
# counts, cross-checked against DA-02 and the audited dataset map
# ===========================================================================


@pytest.mark.needs_data
def test_row_counts_match_the_audited_corpus(master: Any) -> None:
    assert len(master) == 7536
    counts = master["dataset_source"].value_counts().to_dict()
    assert counts == {"D1": 3541, "D4": 3163, "D3": 656, "D2": 176}
    assert int(master["use_in_supervised"].sum()) == 6988


@pytest.mark.needs_data
def test_the_301_validation_records_are_marked_duplicate_and_excluded(
    master: Any,
) -> None:
    """Phase 09's finding, carried into DA-08 rather than re-derived downstream."""
    validation = master[master["subset"] == "validation"]
    assert len(validation) == 301
    assert validation["is_duplicate"].all()
    assert (validation["duplicate_of"] != "").all()
    assert not validation["use_in_supervised"].any()
    assert (validation["split_group"] == "").all()
    # Every twin is a training record that is itself kept.
    twins = set(validation["duplicate_of"])
    kept = master[master["record_uid"].isin(twins)]
    assert len(kept) == 301
    assert kept["use_in_supervised"].all()


@pytest.mark.needs_data
def test_unlabelled_pascal_files_carry_no_label(master: Any) -> None:
    unlabelled = master[master["is_unlabeled"]]
    assert len(unlabelled) == 247  # 52 in set_a + 195 in set_b
    assert unlabelled["multiclass_label"].isna().all()
    assert (unlabelled["multiclass_task"] == "").all()
    assert not unlabelled["use_in_supervised"].any()


@pytest.mark.needs_data
def test_split_group_is_present_exactly_where_cv_will_use_it(master: Any) -> None:
    supervised = master[master["use_in_supervised"]]
    assert (supervised["split_group"] != "").all()
    assert (supervised["split_group"] == supervised["subject_id"]).all()
    assert (master.loc[~master["use_in_supervised"], "split_group"] == "").all()


@pytest.mark.needs_data
def test_the_pascal_a_duplicate_pair_shares_one_split_group(master: Any) -> None:
    """Phase 17's finding: one recording under two class labels.

    Both rows are kept (the dataset cannot say which label is the error) but
    they must share a group so grouped CV can never put them on opposite sides
    of a split.
    """
    pair = master[
        master["record_uid"].isin(
            [
                "D2_set_a_extrahls__201104021355",
                "D2_set_a_murmur__201104021355",
            ]
        )
    ]
    assert len(pair) == 2
    assert pair["split_group"].nunique() == 1
    assert pair["multiclass_label"].nunique() == 2  # the contradiction, preserved


@pytest.mark.needs_data
def test_quality_flags_carry_the_phase_16_findings(master: Any) -> None:
    """One silent recording, 63 clipped, nothing unreadable or zero-length."""
    flags = master["quality_flags"].fillna("")
    assert not flags.str.contains("unreadable").any()
    assert not flags.str.contains("is_zero_length").any()
    assert not flags.str.contains("is_truncated").any()
    assert int(flags.str.contains("is_silent").sum()) == 1
    assert int(flags.str.contains("is_clipped").sum()) == 63


# ===========================================================================
# T19.5 -- the written artifacts
# ===========================================================================


@pytest.mark.needs_data
def test_write_and_reload_round_trips(master: Any, tmp_path: Path) -> None:
    csv_path, parquet_path = ms.write_master(master, tmp_path)
    assert csv_path.is_file() and parquet_path.is_file()
    assert csv_path.stat().st_size > 0

    from_parquet = ms.load_master(parquet_path)
    ms.validate_master(from_parquet)
    assert list(from_parquet.columns) == list(master.columns)
    assert len(from_parquet) == len(master)
    assert list(from_parquet["record_uid"]) == list(master["record_uid"])

    from_csv = ms.load_master(csv_path)
    assert len(from_csv) == len(master)
    for column in ("binary_label", "multiclass_label", "murmur_label", "outcome_label"):
        # The NA/0 distinction has to survive a CSV round trip, or "no label"
        # becomes class 0 the moment anyone reads the committed file.
        assert list(from_csv[column].isna()) == list(master[column].isna()), column
        assert list(from_csv[column].dropna()) == list(master[column].dropna()), column
