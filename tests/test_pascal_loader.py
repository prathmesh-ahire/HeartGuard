"""PASCAL loader gates (T12.7, T13.7).

Same split as the PhysioNet suite: the ``needs_data`` tests run the loader over
the real 832-file corpus and assert the audited figures exactly; the rest are
pure-function tests over hand-written filenames and run everywhere.

The pure-function half matters more here than usual. Every hard problem in this
dataset is a *filename* problem -- three spellings of the same recording, a
missing prefix in one CSV, a doubled prefix in the other, and 149 files with a
noise qualifier that breaks the obvious regex. Those are exactly the cases a
count-based test cannot see: 149 mis-parsed files still produce 832 rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.data_loader import pascal as pa
from src.utils.constants import TASK_PASCAL_A, TASK_PASCAL_B, label_names

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table() -> Any:
    """The full record table, built once -- 832 WAV headers is not free."""
    return pa.load_pascal()


@pytest.fixture(scope="module")
def audit_files(table: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("dataset_audit")
    return {
        "conflicts": pa.write_label_conflicts(table, out),
        "subjects": pa.write_subject_derivation(table, out),
    }


def _read_csv(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


# ===========================================================================
# T12.7 -- loader gate
# ===========================================================================


@pytest.mark.needs_data
def test_labelled_counts_land_exactly(table: Any) -> None:
    """T12.6/T12.7 -- per class, not just on the totals."""
    labelled = table[table["use_in_supervised"]]

    set_a = labelled[labelled["subset"] == pa.SET_A]
    assert len(set_a) == 124
    assert set_a["class_folder"].value_counts().to_dict() == {
        "artifact": 40,
        "extrahls": 19,
        "murmur": 34,
        "normal": 31,
    }

    set_b = labelled[labelled["subset"] == pa.SET_B]
    assert len(set_b) == 461
    assert set_b["class_folder"].value_counts().to_dict() == {
        "extrastole": 46,
        "murmur": 95,
        "normal": 320,
    }


@pytest.mark.needs_data
def test_every_recording_is_found_and_labelled(table: Any) -> None:
    assert len(table) == 832
    assert table["subset"].value_counts().to_dict() == {"set_b": 656, "set_a": 176}
    assert not table["record_uid"].duplicated().any()
    assert not table["canonical_key"].duplicated().any()
    assert table["class_folder"].notna().all()
    assert (table["label_source"].str.startswith("Heartbeat_Sound/")).all()


@pytest.mark.needs_data
def test_the_normalizer_resolves_every_csv_row(table: Any) -> None:
    """T12.3/T12.7 -- both CSVs, both failure modes, zero unresolved rows."""
    keys = set(table["canonical_key"])
    for dataset, n_rows in ((pa.SET_A, 176), (pa.SET_B, 656)):
        csv = pa.load_set_csv(dataset)
        assert len(csv) == n_rows, dataset
        unresolved = set(csv["canonical_key"]) - keys
        assert unresolved == set(), dataset

    # set_b.csv's raw filenames match nothing at all without normalisation --
    # the reason a normalizer exists rather than a str.replace at the call site.
    raw = pa.load_set_csv(pa.SET_B)["fname"].str.rsplit("/", n=1).str[-1]
    on_disk = set(table.loc[table["subset"] == pa.SET_B, "record_id"] + ".wav")
    assert set(raw) & on_disk == set()


@pytest.mark.needs_data
def test_folder_labels_agree_with_the_csvs(
    table: Any, audit_files: dict[str, Path]
) -> None:
    """T12.4 -- the cross-check ran, and the conflict file records its result."""
    logged = set(_read_csv(audit_files["conflicts"]).get("record_uid", []))
    assert set(table.loc[table["label_conflict"], "record_uid"]) == logged
    assert audit_files["conflicts"].is_file()
    assert len(logged) == 0

    matched = table[table["csv_label"] != ""]
    assert len(matched) == 832
    assert (matched["csv_label"] == matched["class_folder"]).all()


@pytest.mark.needs_data
def test_unlabelled_records_are_excluded_but_kept(table: Any) -> None:
    """T12.5 -- out of the supervised tracks, still in the metadata."""
    unlabelled = table[table["is_unlabeled"]]
    assert len(unlabelled) == 247
    assert unlabelled["subset"].value_counts().to_dict() == {"set_b": 195, "set_a": 52}
    assert not unlabelled["use_in_supervised"].any()
    assert unlabelled["multiclass_label"].isna().all()
    assert (unlabelled["multiclass_label_name"] == "").all()

    # Still carries its file path and header facts, so Phases 16-18 can audit it.
    assert unlabelled["file_path"].notna().all()
    assert (unlabelled["duration_sec"] > 0).all()


@pytest.mark.needs_data
def test_the_two_label_spaces_are_never_merged(table: Any) -> None:
    """Rule 4 -- set_a is pascal_a, set_b is pascal_b, and neither borrows."""
    labelled = table[table["use_in_supervised"]]
    assert set(labelled.loc[labelled["subset"] == pa.SET_A, "task"]) == {TASK_PASCAL_A}
    assert set(labelled.loc[labelled["subset"] == pa.SET_B, "task"]) == {TASK_PASCAL_B}

    for subset, task in ((pa.SET_A, TASK_PASCAL_A), (pa.SET_B, TASK_PASCAL_B)):
        rows = labelled[labelled["subset"] == subset]
        assert set(rows["class_folder"]) <= set(label_names(task)), subset

    # The sharpest case: code 2 is `extrahls` in one task and `extrastole` in the
    # other, so a shared integer column would silently merge two different
    # cardiac phenomena.
    a_two = set(
        labelled.loc[
            (labelled["subset"] == pa.SET_A) & (labelled["multiclass_label"] == 2),
            "class_folder",
        ]
    )
    b_two = set(
        labelled.loc[
            (labelled["subset"] == pa.SET_B) & (labelled["multiclass_label"] == 2),
            "class_folder",
        ]
    )
    assert a_two == {"extrahls"}
    assert b_two == {"extrastole"}


@pytest.mark.needs_data
def test_heartbeat_sound_is_the_label_index_and_nothing_else(table: Any) -> None:
    """832 files, a complete cover of set_a + set_b, never a training source."""
    index = pa.load_label_index()
    assert len(index) == 832
    assert set(index) == set(table["canonical_key"])

    # No row's audio path points into Heartbeat_Sound/ -- including it as a
    # source is what doubles the corpus and puts a recording in both splits.
    assert not table["file_path"].str.contains("Heartbeat_Sound").any()
    assert set(table["file_path"].str.split("/").str[-2]) == {"set_a", "set_b"}


# ===========================================================================
# T13.7 -- subject derivation gate
# ===========================================================================


@pytest.mark.needs_data
def test_set_b_subject_groups(table: Any) -> None:
    """T13.1/T13.7 -- 165 subjects, and the 167 the task quotes are sessions."""
    labelled = table[(table["subset"] == pa.SET_B) & table["use_in_supervised"]]
    assert len(labelled) == 461
    assert labelled["subject_id"].nunique() == pa.N_SET_B_SUBJECTS == 165
    assert labelled["session_id"].nunique() == pa.N_SET_B_SESSIONS == 167
    assert labelled["subject_derived"].all()

    # The two subjects the difference is made of: same person, two sessions.
    # Grouping on the session would place them on both sides of a fold.
    sessions_per_subject = labelled.groupby("subject_id")["session_id"].nunique()
    multi = set(sessions_per_subject[sessions_per_subject > 1].index)
    assert len(multi) == 2
    assert multi <= {"b_109", "b_240", "b_245"}

    # Across all 656 files, including the unlabelled ones.
    assert table[table["subset"] == pa.SET_B]["subject_id"].nunique() == 171


@pytest.mark.needs_data
def test_the_149_record_outlier_group_does_not_exist(table: Any) -> None:
    """T13.3 -- it was a regex fallback bucket, not a subject.

    149 set_b files carry a noise qualifier and a single separator. Under the
    strict ``<label>__(\\d+)_...`` reading they all fail to match and land in one
    fallback, which reads as a single 149-recording subject. Parsed properly they
    scatter across their real subjects and the largest group is 10.
    """
    set_b = table[table["subset"] == pa.SET_B]
    noisy = set_b[set_b["noise_qualifier"] != ""]
    assert len(noisy) == pa.N_NOISY_SET_B_FILES == 149
    assert noisy["noise_qualifier"].value_counts().to_dict() == {
        "noisynormal": 120,
        "noisymurmur": 29,
    }

    # They scatter across 84 real subjects, at most 5 apiece, rather than
    # pooling into one. The largest set_b group of any kind is 10 recordings.
    assert noisy["subject_id"].nunique() == 84
    assert noisy["subject_id"].value_counts().max() == 5
    assert set_b["subject_id"].value_counts().max() == 10


@pytest.mark.needs_data
def test_recording_location_is_present_wherever_the_filename_gives_one(
    table: Any,
) -> None:
    """T13.2/T13.7 -- every set_b record has a site; set_a filenames have none."""
    set_b = table[table["subset"] == pa.SET_B]
    assert (set_b["recording_location"] != "").all()
    assert set(set_b["recording_location"]) == {"A", "B", "C", "D", "E", "F"}

    set_a = table[table["subset"] == pa.SET_A]
    assert (set_a["recording_location"] == "").all()

    # The repeat index is separate from the site, so `D1` does not become a
    # sixth body site.
    repeats = set(set_b["location_repeat"])
    assert "" in repeats
    assert repeats <= {"", "1", "2", "3", "4", "31"}


@pytest.mark.needs_data
def test_set_a_has_no_subject_information(table: Any) -> None:
    """T13.4 -- record-level only, and it says so."""
    set_a = table[table["subset"] == pa.SET_A]
    assert len(set_a) == 176
    assert not set_a["subject_derived"].any()
    assert set_a["subject_id"].nunique() == 176
    assert (set_a["session_id"] == "").all()
    assert set(set_a["subject_pattern"]) == {
        "set_a: timestamp-only filename -- record is its own group"
    }


@pytest.mark.needs_data
def test_set_a_timing_table(table: Any) -> None:
    """T13.5 -- 21 recordings, 195 S1 + 195 S2, indices inside the file."""
    timing = pa.load_set_a_timing(table)
    assert len(timing) == 390
    assert timing["canonical_key"].nunique() == 21
    assert timing["sound"].value_counts().to_dict() == {"S1": 195, "S2": 195}
    assert timing["cycle"].min() == 1

    # Every annotated recording is a real set_a file, and every index lands
    # inside it. The column is a sample position at 44.1 kHz, not a body site.
    lengths = dict(zip(table["canonical_key"], table["n_samples"], strict=False))
    rates = dict(zip(table["canonical_key"], table["original_fs"], strict=False))
    for key, index in zip(timing["canonical_key"], timing["sample_index"], strict=False):
        assert key in lengths
        assert 0 <= index < lengths[key]
        assert rates[key] == pa.TIMING_FS

    assert timing["time_sec"].max() < 10.0


@pytest.mark.needs_data
def test_subject_derivation_csv(table: Any, audit_files: dict[str, Path]) -> None:
    """T13.6 -- record, pattern, subject, session, site, confidence flag."""
    written = _read_csv(audit_files["subjects"])
    assert len(written) == 832
    for column in (
        "record_uid",
        "subject_pattern",
        "subject_id",
        "session_id",
        "recording_location",
        "subject_derived",
    ):
        assert column in written.columns
    assert written["subject_pattern"].notna().all()
    assert written["subject_id"].nunique() == table["subject_id"].nunique()


# ===========================================================================
# pure-function tests -- no dataset required
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "dataset", "expected"),
    [
        # the plain set_b recording, in its three spellings
        ("set_b/Btraining_extrastole_127_1306764300147_C2.wav", pa.SET_B,
         "extrastole_127_1306764300147_C2"),
        ("extrastole__127_1306764300147_C2.wav", pa.SET_B,
         "extrastole_127_1306764300147_C2"),
        ("extrastole__127_1306764300147_C2.wav", None,
         "extrastole_127_1306764300147_C2"),
        # the noisy form, where set_b.csv doubles the Btraining_ prefix
        ("set_b/Btraining_normal_Btraining_noisynormal_125_1306332456645_C.wav",
         pa.SET_B, "normal_noisynormal_125_1306332456645_C"),
        ("normal_noisynormal_125_1306332456645_C.wav", pa.SET_B,
         "normal_noisynormal_125_1306332456645_C"),
        # set_b unlabelled: CSV drops one underscore, disk keeps two
        ("set_b/Bunlabelledtest_101_1305030823364_A.wav", pa.SET_B,
         "Bunlabelledtest_101_1305030823364_A"),
        ("Bunlabelledtest__101_1305030823364_A.wav", pa.SET_B,
         "Bunlabelledtest_101_1305030823364_A"),
        # set_a labelled: the CSV is faithful here
        ("set_a/artifact__201012172012.wav", pa.SET_A, "artifact_201012172012"),
        # set_a unlabelled: the CSV drops the prefix entirely
        ("set_a/__201012172010.wav", pa.SET_A, "Aunlabelledtest_201012172010"),
        ("Aunlabelledtest__201012172010.wav", pa.SET_A,
         "Aunlabelledtest_201012172010"),
    ],
)
def test_normalize_pascal_name(name: str, dataset: str | None, expected: str) -> None:
    assert pa.normalize_pascal_name(name, dataset) == expected


def test_normalize_refuses_to_guess_a_missing_prefix() -> None:
    """The set_a.csv unlabelled rows cannot be resolved without knowing the set."""
    with pytest.raises(ValueError, match="label prefix is missing"):
        pa.normalize_pascal_name("set_a/__201012172010.wav")


@pytest.mark.parametrize(
    ("name", "subject", "timestamp", "noise", "location", "repeat"),
    [
        ("extrastole__127_1306764300147_C2.wav", "127", "1306764300147", "", "C", "2"),
        ("murmur__112_1306243000964_A.wav", "112", "1306243000964", "", "A", ""),
        ("normal_noisynormal_125_1306332456645_A1.wav",
         "125", "1306332456645", "noisynormal", "A", "1"),
        ("murmur_noisymurmur_162_1307101835989_B_1.wav",
         "162", "1307101835989", "noisymurmur", "B", "1"),
        ("Bunlabelledtest__268_1309368960960_E.wav",
         "268", "1309368960960", "", "E", ""),
        ("normal__103_1305031931979_D1.wav", "103", "1305031931979", "", "D", "1"),
    ],
)
def test_parse_set_b_name(
    name: str, subject: str, timestamp: str, noise: str, location: str, repeat: str
) -> None:
    parsed = pa.parse_set_b_name(name)
    assert parsed.subject == subject
    assert parsed.timestamp == timestamp
    assert parsed.noise_qualifier == noise
    assert parsed.location == location
    assert parsed.location_repeat == repeat


def test_noisy_and_plain_recordings_of_one_subject_group_together() -> None:
    """The 149-file trap, as a unit test.

    A strict `__` regex sends the noisy file to a fallback, so these two land in
    different groups even though they are the same person.
    """
    plain = pa.derive_subject_pascal("normal__125_1306332456645_C", pa.SET_B)
    noisy = pa.derive_subject_pascal("normal_noisynormal_125_1306332456645_A1", pa.SET_B)
    assert plain.subject_id == noisy.subject_id == "b_125"
    assert plain.recording_location == "C"
    assert noisy.recording_location == "A"


def test_two_sessions_of_one_subject_share_a_subject_but_not_a_session() -> None:
    """Subject 109, recorded twice five minutes apart on 2011-05-17."""
    first = pa.derive_subject_pascal("Bunlabelledtest__109_1305653646620_B", pa.SET_B)
    second = pa.derive_subject_pascal("Bunlabelledtest__109_1305653972028_B", pa.SET_B)
    assert first.subject_id == second.subject_id == "b_109"
    assert first.session_id != second.session_id


def test_derive_subject_pascal_set_a_is_record_level() -> None:
    result = pa.derive_subject_pascal("artifact__201012172012", pa.SET_A)
    assert result.subject_id == "a_artifact__201012172012"
    assert result.subject_derived is False
    assert result.session_id == ""
    assert result.recording_location == ""


def test_derive_subject_pascal_rejects_an_unknown_set() -> None:
    with pytest.raises(ValueError, match="unknown PASCAL set"):
        pa.derive_subject_pascal("whatever", "set_c")


def test_parse_set_b_name_rejects_an_unparseable_name() -> None:
    with pytest.raises(ValueError, match="does not parse"):
        pa.parse_set_b_name("nothing_numeric_here.wav")


def test_expected_counts_are_internally_consistent() -> None:
    for dataset, total in pa.EXPECTED_LABELLED_TOTALS.items():
        assert sum(pa.EXPECTED_CLASS_COUNTS[dataset].values()) == total
    assert sum(pa.EXPECTED_LABELLED_TOTALS.values()) == 585
    assert sum(pa.EXPECTED_UNLABELLED_COUNTS.values()) == 247
    assert 585 + 247 == 832
