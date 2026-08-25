"""CirCor loader gates (T14.7, T15.7).

Same split as the PhysioNet and PASCAL suites: ``needs_data`` tests run the
loader over the real 942-patient tree and assert the audited figures exactly;
pure-function tests over hand-written files run everywhere.

The checksum verification is additionally marked ``slow`` -- it hashes 585 MB
and takes around 40 seconds. It still runs; ``--runslow`` includes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.data_loader import circor as ci
from src.utils.constants import (
    CIRCOR_LOCATIONS,
    TASK_CIRCOR_MURMUR,
    TASK_CIRCOR_OUTCOME,
    label_names,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table() -> Any:
    """The full record table, built once -- 3,163 headers plus 3,162 TSVs."""
    return ci.load_circor()


@pytest.fixture(scope="module")
def patients() -> Any:
    return ci.build_patient_table()


@pytest.fixture(scope="module")
def audit_files(
    table: Any, patients: Any, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("dataset_audit")
    return {
        "conflicts": ci.write_demographics_conflicts(patients, None, out),
        "unsegmented": ci.write_unsegmented_report(table, None, out),
    }


def _read_csv(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


# ===========================================================================
# T14.7 -- loader gate
# ===========================================================================


@pytest.mark.needs_data
def test_patient_and_recording_counts(table: Any, patients: Any) -> None:
    assert len(patients) == ci.N_PATIENTS == 942
    assert len(table) == ci.N_RECORDINGS == 3163
    assert table["patient_id"].nunique() == 942
    assert not table["record_uid"].duplicated().any()
    assert not patients["patient_id"].duplicated().any()


@pytest.mark.needs_data
def test_murmur_counts(patients: Any) -> None:
    """T14.6/T14.7 -- 695 Absent / 179 Present / 68 Unknown."""
    assert patients["murmur"].value_counts().to_dict() == {
        "Absent": 695,
        "Present": 179,
        "Unknown": 68,
    }
    assert set(patients["murmur"]) == set(label_names(TASK_CIRCOR_MURMUR))


@pytest.mark.needs_data
def test_outcome_is_parsed_from_the_txt_files_not_the_csv(patients: Any) -> None:
    """T14.3/T14.7 -- the headline trap of this dataset.

    ``training_data.csv`` has no Outcome column at all. A pipeline that reads
    demographics from the CSV and stops there has no outcome task, and nothing
    anywhere raises to say so.
    """
    demographics = ci.load_demographics()
    assert "Outcome" not in demographics.columns
    assert len(demographics) == 942

    assert patients["outcome"].value_counts().to_dict() == {
        "Normal": 486,
        "Abnormal": 456,
    }
    assert set(patients["outcome"]) == set(label_names(TASK_CIRCOR_OUTCOME))
    assert patients["outcome"].notna().all()
    assert (patients["outcome"] != "").all()
    assert set(patients["outcome_source"]) == {
        "patient .txt #Outcome (absent from training_data.csv)"
    }


@pytest.mark.needs_data
def test_location_counts(table: Any) -> None:
    """T14.6/T14.7 -- AV 800 / PV 766 / TV 732 / MV 861 / Phc 4."""
    assert table["recording_location"].value_counts().to_dict() == {
        "MV": 861,
        "AV": 800,
        "PV": 766,
        "TV": 732,
        "Phc": 4,
    }
    assert set(table["recording_location"]) <= set(CIRCOR_LOCATIONS)
    assert sum(ci.EXPECTED_LOCATION_COUNTS.values()) == ci.N_RECORDINGS


@pytest.mark.needs_data
def test_all_metadata_keys_and_murmur_descriptors_are_parsed(patients: Any) -> None:
    """T14.3/T14.4 -- 21 keys in every patient file, 10 of them descriptors."""
    assert len(ci.METADATA_KEYS) == 21
    assert len(ci.MURMUR_DESCRIPTORS) == 10

    for key in ci.METADATA_KEYS:
        if key in ("Murmur", "Outcome"):
            continue
        column = key.lower().replace(" ", "_")
        assert column in patients.columns, key

    # Descriptors are populated for murmur patients and blank otherwise -- the
    # source writes "nan" there, which the loader turns into an empty string.
    present = patients[patients["murmur"] == "Present"]
    assert (present["systolic_murmur_timing"] != "").sum() > 0
    absent = patients[patients["murmur"] == "Absent"]
    assert (absent["systolic_murmur_timing"] == "").all()
    assert (absent["murmur_locations"] == "").all()


@pytest.mark.needs_data
def test_murmur_location_fields(table: Any) -> None:
    """T14.4 -- murmur locations and the most audible one, per recording."""
    present = table[table["murmur"] == "Present"]
    assert present["is_murmur_location"].any()
    assert present["is_most_audible"].any()

    # A recording flagged as the most audible location is always one of the
    # patient's murmur locations.
    audible = present[present["is_most_audible"]]
    assert audible["is_murmur_location"].all()

    # Absent-murmur patients have neither.
    absent = table[table["murmur"] == "Absent"]
    assert not absent["is_murmur_location"].any()
    assert not absent["is_most_audible"].any()


@pytest.mark.needs_data
def test_demographics_join_and_the_age_conflict(
    patients: Any, audit_files: dict[str, Path]
) -> None:
    """T14.5 -- patient ids agree; Age is the one field that does not."""
    demographics = ci.load_demographics()
    assert set(demographics["patient_id"]) == set(patients["patient_id"])

    conflicts = _read_csv(audit_files["conflicts"])
    assert len(conflicts) == 73
    assert set(conflicts["field"]) == {"Age"}
    assert set(conflicts["authoritative"]) == {"patient_txt"}

    # "Young Adult" exists only in the CSV; the txt files never use it. Worth
    # pinning, because it is the difference between CirCor being wholly
    # paediatric and containing 7 adults -- which EXP-D1 rests on.
    assert "Young Adult" in set(demographics["Age"])
    assert "Young Adult" not in set(patients["age"])


@pytest.mark.needs_data
def test_patient_labels_broadcast_to_every_recording(table: Any, patients: Any) -> None:
    """Both tasks are diagnosed per patient, which is why patient_id groups."""
    by_patient = patients.set_index("patient_id")
    for patient_id, group in table.groupby("patient_id"):
        assert group["murmur"].nunique() == 1, patient_id
        assert group["outcome"].nunique() == 1, patient_id
        assert group["murmur"].iloc[0] == by_patient.at[patient_id, "murmur"]

    assert (table["subject_id"] == table["patient_id"]).all()
    assert not table["subject_derived"].any()   # native ids, nothing derived


@pytest.mark.needs_data
def test_headers_agree_with_the_audio(table: Any) -> None:
    """T15.1 -- fs and sample count cross-checked, not trusted."""
    assert (table["original_fs"] == 4000).all()
    assert (table["header_fs"] == table["original_fs"]).all()
    assert (table["header_n_samples"] == table["n_samples"]).all()
    assert (table["n_channels"] == 1).all()
    assert (table["header_location"] == table["recording_location"]).all()
    assert (table["duration_sec"] > 0).all()


# ===========================================================================
# T15.7 -- segmentation gate
# ===========================================================================


@pytest.mark.needs_data
def test_every_segmentation_parses_with_valid_states(table: Any) -> None:
    """T15.7 -- states within 0-4 across all 3,162 real segmentation files."""
    segments = ci.build_segmentation_table(table)
    assert len(segments) > 250_000
    assert set(segments["state"]) == set(ci.SEGMENTATION_STATES)
    assert segments["state"].between(0, 4).all()
    assert set(segments["state_name"]) == set(ci.SEGMENTATION_STATES.values())
    assert "invalid" not in set(segments["state_name"])
    assert segments["record_uid"].nunique() == 3162


@pytest.mark.needs_data
def test_segment_times_are_monotonic_and_inside_the_recording(table: Any) -> None:
    """T15.7 -- per segment, and per file, with a one-sample tolerance.

    The tolerance is not slack for convenience: six recordings end their last
    segment up to 201 us past the WAV, against a 250 us sample period at 4 kHz.
    That is float formatting in the source, and a stricter check would reject
    six perfectly good files while catching nothing real.
    """
    segments = ci.build_segmentation_table(table)
    tolerance = 1.0 / 4000.0

    assert (segments["end_sec"] >= segments["start_sec"]).all()
    assert (segments["start_sec"] >= 0).all()

    durations = dict(zip(table["record_uid"], table["duration_sec"], strict=False))
    grouped = segments.groupby("record_uid")
    assert (grouped["end_sec"].max() - grouped["record_uid"].first().map(durations)
            <= tolerance).all()

    # Within a file, segments are ordered and non-overlapping beyond tolerance.
    for uid, group in grouped:
        starts = group["start_sec"].tolist()
        assert starts == sorted(starts), uid


@pytest.mark.needs_data
def test_the_two_duplicated_segmentation_files_are_repaired_losslessly(
    table: Any,
) -> None:
    """Exactly two files list every segment twice, appended out of order.

    Deduplicating and sorting resolves both to zero overlaps. No value changes
    and nothing is dropped that is not a byte-identical copy of a kept row, so
    this is a repair rather than an edit.
    """
    repaired = table[
        table["segmentation_issues"].str.contains("exact duplicate", na=False)
    ]
    assert set(repaired["record_id"]) == {"50690_MV_2", "50690_TV"}

    root = ci.circor_root()
    for record_id, unique_rows in (("50690_MV_2", 53), ("50690_TV", 39)):
        parsed = ci.load_segmentation(root / (record_id + ".tsv"))
        assert parsed.n_segments == unique_rows
        starts = [s for s, _, _ in parsed.segments]
        assert starts == sorted(starts)
        assert any("exact duplicate" in issue for issue in parsed.issues)
        assert not any("overlap" in issue for issue in parsed.issues)


@pytest.mark.needs_data
def test_overlaps_are_reported_with_their_magnitude(table: Any) -> None:
    """93 overlaps: 90 are sub-millisecond rounding, one is real.

    Pinned so the one genuine 11.7 ms overlap in ``50150_MV`` cannot get lost
    among the rounding noise if the corpus is ever re-processed.
    """
    import re

    magnitudes = [
        float(m)
        for issues in table["segmentation_issues"]
        for m in re.findall(r"overlap by ([\d.]+) s", str(issues))
    ]
    assert len(magnitudes) == 93
    assert sum(1 for m in magnitudes if m >= 0.001) == 3
    assert sum(1 for m in magnitudes if m >= 0.002) == 1
    assert max(magnitudes) == pytest.approx(0.01168, abs=1e-6)

    worst = table[table["segmentation_issues"].str.contains("0.011680", na=False)]
    assert list(worst["record_id"]) == ["50150_MV"]


@pytest.mark.needs_data
def test_annotated_fraction_and_cycle_counts(table: Any) -> None:
    """T15.3 -- both derived, both sane."""
    segmented = table[table["has_segmentation"]]
    assert len(segmented) == 3162
    assert (segmented["annotated_fraction"] > 0).all()
    assert (segmented["n_cycles"] > 0).all()
    assert segmented["n_cycles"].max() == 94

    # Annotated time cannot exceed the recording, give or take one sample.
    # Stated in seconds rather than as a fraction because the fraction hides the
    # scale: `50782_MV_2` is annotated end to end and its last segment overhangs
    # by 38 us, which is 1.000006 as a ratio and plainly fine as a duration.
    tolerance = 1.0 / 4000.0
    assert (
        segmented["annotated_sec"] <= segmented["duration_sec"] + tolerance
    ).all()
    assert segmented["annotated_fraction"].max() < 1.001

    # The parts add up to the whole, within float noise.
    total = (
        segmented["s1_sec"]
        + segmented["systole_sec"]
        + segmented["s2_sec"]
        + segmented["diastole_sec"]
    )
    assert ((total - segmented["annotated_sec"]).abs() < 1e-3).all()


@pytest.mark.needs_data
def test_unsegmented_recordings_are_reported(
    table: Any, audit_files: dict[str, Path]
) -> None:
    """T15.5 -- one recording without a TSV, one TSV without a recording."""
    assert int((~table["has_segmentation"]).sum()) == 1
    assert list(table.loc[~table["has_segmentation"], "record_id"]) == ["50782_MV_1"]

    report = _read_csv(audit_files["unsegmented"])
    assert len(report) == 2
    kinds = report["kind"].tolist()
    assert "recording without usable segmentation" in kinds
    assert "segmentation without a recording" in kinds

    # The orphan is reported, never adopted as the missing one -- and it is
    # unusable anyway: one zero-length row carrying state 28.
    orphan = report[report["kind"] == "segmentation without a recording"].iloc[0]
    assert orphan["record_id"] == "50782_MV"
    assert "invalid state 28" in orphan["segmentation_issues"]
    assert "50782_MV" not in set(table["record_id"])


@pytest.mark.needs_data
def test_segmentation_artifact_is_written_not_discarded(table: Any) -> None:
    """T15.4 -- the per-segment table Phase 80 and T113.6 consume."""
    segments = ci.build_segmentation_table(table)
    for column in (
        "record_uid",
        "patient_id",
        "segment_index",
        "start_sec",
        "end_sec",
        "duration_sec",
        "state",
        "state_name",
    ):
        assert column in segments.columns
    assert (segments["duration_sec"] >= 0).all()
    assert set(segments["record_uid"]) <= set(table["record_uid"])


@pytest.mark.needs_data
@pytest.mark.slow
def test_manifest_integrity(table: Any) -> None:
    """T15.6 -- RECORDS and SHA256SUMS, resolved against the right base.

    The expected outcome is *not* "everything matches". All 9,489 WAV/HEA/TSV
    files match their published SHA-256; all 942 patient ``.txt`` files do not.
    0 of 942 is a stale manifest, not corruption -- and it is consistent with
    ``training_data.csv`` still matching its own checksum while disagreeing with
    those same txt files about Age.
    """
    report = ci.verify_integrity()

    assert (report["status"] != "missing").all()
    assert int((report["manifest"] == "RECORDS").sum()) == 3163

    checksummed = report[report["manifest"] == "SHA256SUMS"]
    by_type = checksummed.groupby(["file_type", "status"]).size().to_dict()
    assert by_type[(".wav", "ok")] == 3163
    assert by_type[(".hea", "ok")] == 3163
    assert by_type[(".tsv", "ok")] == 3163
    assert by_type[(".csv", "ok")] == 1
    assert by_type[(".txt", "checksum_mismatch")] == 942
    assert (".wav", "checksum_mismatch") not in by_type
    assert (".hea", "checksum_mismatch") not in by_type
    assert (".tsv", "checksum_mismatch") not in by_type


# ===========================================================================
# pure-function tests -- no dataset required
# ===========================================================================


def _write_patient(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "12345.txt"
    path.write_text(body, encoding="utf-8")
    return path


_MINIMAL_METADATA = "\n".join(
    "#" + key + ": " + value
    for key, value in [
        ("Age", "Child"),
        ("Sex", "Male"),
        ("Height", "98.0"),
        ("Weight", "15.9"),
        ("Pregnancy status", "False"),
        ("Murmur", "Present"),
        ("Murmur locations", "TV"),
        ("Most audible location", "TV"),
        ("Systolic murmur timing", "Holosystolic"),
        ("Systolic murmur shape", "Plateau"),
        ("Systolic murmur grading", "I/VI"),
        ("Systolic murmur pitch", "Low"),
        ("Systolic murmur quality", "Blowing"),
        ("Diastolic murmur timing", "nan"),
        ("Diastolic murmur shape", "nan"),
        ("Diastolic murmur grading", "nan"),
        ("Diastolic murmur pitch", "nan"),
        ("Diastolic murmur quality", "nan"),
        ("Outcome", "Abnormal"),
        ("Campaign", "CC2015"),
        ("Additional ID", "nan"),
    ]
)


def test_parse_patient_file(tmp_path: Path) -> None:
    path = _write_patient(
        tmp_path,
        "12345 2 4000\n"
        "AV 12345_AV.hea 12345_AV.wav 12345_AV.tsv\n"
        "TV 12345_TV.hea 12345_TV.wav 12345_TV.tsv\n" + _MINIMAL_METADATA + "\n",
    )
    parsed = ci.parse_patient_file(path)
    assert parsed.patient_id == "12345"
    assert parsed.n_locations == 2
    assert parsed.fs_hz == 4000
    assert [r.location for r in parsed.recordings] == ["AV", "TV"]
    assert parsed.recordings[0].record_id == "12345_AV"
    assert parsed.murmur == "Present"
    assert parsed.outcome == "Abnormal"
    assert len(parsed.metadata) == 21


def test_parse_patient_file_accepts_a_location_line_with_no_tsv(tmp_path: Path) -> None:
    """The 50782 case: a real recording whose patient file names no TSV.

    A strict four-field parse rejects a valid patient over a missing
    segmentation, which is the wrong trade.
    """
    path = _write_patient(
        tmp_path,
        "12345 1 4000\n"
        "MV 12345_MV_1.hea 12345_MV_1.wav\n" + _MINIMAL_METADATA + "\n",
    )
    parsed = ci.parse_patient_file(path)
    assert parsed.recordings[0].tsv_file == ""
    assert parsed.recordings[0].record_id == "12345_MV_1"


def test_parse_patient_file_rejects_a_filename_content_mismatch(tmp_path: Path) -> None:
    path = _write_patient(
        tmp_path,
        "99999 1 4000\nAV 99999_AV.hea 99999_AV.wav 99999_AV.tsv\n"
        + _MINIMAL_METADATA + "\n",
    )
    with pytest.raises(ValueError, match="filename and content disagree"):
        ci.parse_patient_file(path)


def test_parse_patient_file_rejects_an_unknown_location(tmp_path: Path) -> None:
    path = _write_patient(
        tmp_path,
        "12345 1 4000\nXX 12345_XX.hea 12345_XX.wav 12345_XX.tsv\n"
        + _MINIMAL_METADATA + "\n",
    )
    with pytest.raises(ValueError, match="unknown auscultation location"):
        ci.parse_patient_file(path)


def test_parse_patient_file_rejects_a_missing_metadata_key(tmp_path: Path) -> None:
    trimmed = "\n".join(
        line for line in _MINIMAL_METADATA.splitlines()
        if not line.startswith("#Outcome")
    )
    path = _write_patient(
        tmp_path,
        "12345 1 4000\nAV 12345_AV.hea 12345_AV.wav 12345_AV.tsv\n" + trimmed + "\n",
    )
    with pytest.raises(ValueError, match="missing metadata key"):
        ci.parse_patient_file(path)


def test_load_segmentation(tmp_path: Path) -> None:
    path = tmp_path / "12345_AV.tsv"
    path.write_text(
        "0\t1.0\t0\n1.0\t1.15\t1\n1.15\t1.35\t2\n1.35\t1.45\t3\n1.45\t1.80\t4\n",
        encoding="utf-8",
    )
    parsed = ci.load_segmentation(path, duration_sec=2.0)
    assert parsed.n_segments == 5
    assert parsed.issues == ()

    summary = ci.segmentation_summary(parsed, 2.0)
    assert summary["n_cycles"] == 1
    assert summary["annotated_sec"] == pytest.approx(0.80)
    assert summary["annotated_fraction"] == pytest.approx(0.40)
    assert summary["unannotated_sec"] == pytest.approx(1.0)
    assert summary["has_segmentation"] is True


def test_load_segmentation_reports_an_invalid_state(tmp_path: Path) -> None:
    """The orphan file's shape: one zero-length row carrying state 28."""
    path = tmp_path / "50782_MV.tsv"
    path.write_text("0\t0\t28\n", encoding="utf-8")
    parsed = ci.load_segmentation(path)
    assert any("invalid state 28" in issue for issue in parsed.issues)
    assert ci.segmentation_summary(parsed, 10.0)["has_segmentation"] is False


def test_load_segmentation_deduplicates_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "x.tsv"
    path.write_text(
        "1.0\t1.2\t1\n1.2\t1.4\t2\n1.0\t1.2\t1\n1.2\t1.4\t2\n", encoding="utf-8"
    )
    parsed = ci.load_segmentation(path)
    assert parsed.n_segments == 2
    assert any("exact duplicate" in issue for issue in parsed.issues)
    assert not any("overlap" in issue for issue in parsed.issues)


def test_load_segmentation_reports_a_real_overlap(tmp_path: Path) -> None:
    path = tmp_path / "y.tsv"
    path.write_text("1.0\t1.5\t4\n1.4\t1.6\t1\n", encoding="utf-8")
    parsed = ci.load_segmentation(path)
    assert any("overlap by 0.100000 s" in issue for issue in parsed.issues)


def test_load_segmentation_rejects_a_malformed_row(tmp_path: Path) -> None:
    path = tmp_path / "z.tsv"
    path.write_text("1.0\t1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="field"):
        ci.load_segmentation(path)


def test_parse_circor_header(tmp_path: Path) -> None:
    path = tmp_path / "13918_AV.hea"
    path.write_text(
        "13918_AV 1 4000 41152\n13918_AV.wav 16+44 1 16 0 0 0 0 AV\n",
        encoding="utf-8",
    )
    header = ci.parse_circor_header(path)
    assert header.record == "13918_AV"
    assert header.fs_hz == 4000
    assert header.n_samples == 41152
    assert header.location == "AV"


def test_state_alphabet_matches_the_task_description() -> None:
    assert ci.SEGMENTATION_STATES == {
        0: "unannotated",
        1: "S1",
        2: "systole",
        3: "S2",
        4: "diastole",
    }
    assert ci.ANNOTATED_STATES == (1, 2, 3, 4)


def test_expected_counts_are_internally_consistent() -> None:
    assert sum(ci.EXPECTED_MURMUR_COUNTS.values()) == ci.N_PATIENTS
    assert sum(ci.EXPECTED_OUTCOME_COUNTS.values()) == ci.N_PATIENTS
    assert sum(ci.EXPECTED_LOCATION_COUNTS.values()) == ci.N_RECORDINGS
    assert set(ci.EXPECTED_LOCATION_COUNTS) == set(CIRCOR_LOCATIONS)
