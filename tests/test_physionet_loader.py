"""PhysioNet loader gates (T09.7, T10.7, T11.7).

Split in two by design.

The ``needs_data`` tests run the loader over the real 3,541-record tree and
assert the audited figures exactly. They are the actual gates; CI skips them
because the 1.3 GB corpus is gitignored and never reaches GitHub.

The rest are pure-function tests over hand-built inputs -- header text, record
names -- and run everywhere. They exist because the derivation rules have eight
known edge cases (four in training-a, four in training-b) whose whole point is
that they look like ordinary records. A test that only ever sees the aggregate
count would pass with those eight silently mis-grouped, since 60 subjects is
what you get either way once the fallback invents its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.data_loader import physionet as pn

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table() -> Any:
    """The full record table, built once. Reading 3,541 headers is not free."""
    return pn.load_physionet()


@pytest.fixture(scope="module")
def audit_files(table: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The three Part II audit CSVs, written to a throwaway directory."""
    out = tmp_path_factory.mktemp("dataset_audit")
    return {
        "conflicts": pn.write_label_conflicts(table, out),
        "unannotated": pn.write_unannotated_report(table, out),
        "subjects": pn.write_subject_derivation(table, out),
    }


def _read_csv(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


# ===========================================================================
# T09.7 -- loader gate
# ===========================================================================


@pytest.mark.needs_data
def test_all_records_load(table: Any) -> None:
    assert len(table) == pn.N_ALL_ROWS == 3541
    assert set(table["subset"]) == set(pn.ALL_SUBSETS)
    counts = table["subset"].value_counts().to_dict()
    assert counts == pn.EXPECTED_RECORD_COUNTS


@pytest.mark.needs_data
def test_every_wav_has_a_label_and_every_label_a_wav(table: Any) -> None:
    # build_record_table raises on either gap, so reaching here already proves
    # it. Asserted anyway: a future refactor that downgrades that raise to a
    # warning would otherwise pass this file unchanged.
    assert table["reference_value"].isin([-1, 1]).all()
    assert table["binary_label"].isin([0, 1]).all()
    assert table["file_path"].notna().all()
    assert not table["record_uid"].duplicated().any()

    for subset in pn.ALL_SUBSETS:
        reference = pn.load_reference(subset)
        on_disk = set(table.loc[table["subset"] == subset, "record_id"])
        assert on_disk == set(reference), subset


@pytest.mark.needs_data
def test_header_comment_agrees_with_reference(table: Any, audit_files: dict[str, Path]) -> None:
    """T09.6/T09.7 -- agreement everywhere except the rows actually logged."""
    with_header = table[table["header_source"] == "hea"]
    assert len(with_header) == pn.N_TRAINING_RECORDS

    logged = set(_read_csv(audit_files["conflicts"]).get("record_uid", []))
    disagreeing = set(
        with_header.loc[
            with_header["hea_label"] != with_header["binary_label"], "record_uid"
        ]
    )
    assert disagreeing == logged

    # As it happens the conflict file is empty -- the two label sources agree on
    # all 3,240 training records. The file is still written, so "the check ran
    # and found nothing" is distinguishable from "the check never ran".
    assert audit_files["conflicts"].is_file()
    assert len(logged) == 0


@pytest.mark.needs_data
def test_validation_has_no_headers_and_falls_back_to_the_wav(table: Any) -> None:
    validation = table[table["subset"] == pn.VALIDATION_SUBSET]
    assert (validation["header_source"] == "wav").all()
    assert (validation["header_path"] == "").all()
    assert (validation["hea_comment"] == "").all()
    assert validation["hea_label"].isna().all()
    assert (validation["original_fs"] == 2000).all()


@pytest.mark.needs_data
def test_only_training_a_carries_ecg_and_only_for_405_of_409(table: Any) -> None:
    """T09.3 -- the flag is per-record, not per-subset."""
    ecg = table[table["has_ecg_channel"]]
    assert set(ecg["subset"]) == {"training-a"}
    assert len(ecg) == 405

    without = set(
        table.loc[
            (table["subset"] == "training-a") & ~table["has_ecg_channel"], "record_id"
        ]
    )
    assert without == {"a0041", "a0117", "a0220", "a0233"}
    assert (ecg["n_signals"] == 2).all()
    assert (ecg["ecg_signal_file"].str.endswith(".dat")).all()


@pytest.mark.needs_data
def test_signal_quality_column(table: Any) -> None:
    """T09.5 -- SQI where the file exists, absent where it does not."""
    training = table[table["subset"].isin(pn.TRAINING_SUBSETS)]
    assert training["sqi_available"].all()
    assert training["sqi"].isin([0, 1]).all()
    assert int((training["sqi"] == 1).sum()) == 2876
    assert int((training["sqi"] == 0).sum()) == 364

    validation = table[table["subset"] == pn.VALIDATION_SUBSET]
    assert not validation["sqi_available"].any()


@pytest.mark.needs_data
def test_validation_is_a_byte_identical_copy_of_training(table: Any) -> None:
    """The finding that makes 3,541 a double count. See the module docstring."""
    validation = table[table["subset"] == pn.VALIDATION_SUBSET]
    assert len(validation) == 301
    assert validation["is_duplicate"].all()
    assert not validation["use_in_supervised"].any()

    by_uid = dict(zip(table["record_uid"], table["binary_label"], strict=False))
    for uid, twin in zip(validation["record_uid"], validation["duplicate_of"], strict=False):
        assert twin.startswith("D1_training-"), uid
        assert by_uid[twin] == by_uid[uid], uid

    unique = table[table["use_in_supervised"]]
    assert len(unique) == pn.N_TRAINING_RECORDS == 3240
    assert unique["binary_label_name"].value_counts().to_dict() == {
        "normal": 2575,
        "abnormal": 665,
    }


# ===========================================================================
# T10.7 -- annotation enrichment gate
# ===========================================================================


@pytest.mark.needs_data
def test_appendix_rows_joined(table: Any) -> None:
    training = table[table["subset"].isin(pn.TRAINING_SUBSETS)]
    assert int(training["appendix_matched"].sum()) == pn.N_APPENDIX_ROWS == 3153
    assert len(training) == 3240

    matched = training[training["appendix_matched"]]
    assert matched["original_record_name"].notna().all()
    assert (matched["appendix_class"] == matched["reference_value"]).all()


@pytest.mark.needs_data
def test_twelve_diagnosis_categories_are_populated(table: Any) -> None:
    """T10.4 -- 12 after the whitespace strip, not the 13 the raw column shows."""
    matched = table[table["appendix_matched"]]
    categories = set(matched["diagnosis_class"].dropna())
    assert len(categories) == pn.N_DIAGNOSIS_CLASSES == 12
    assert {"Normal", "CAD", "MVP", "Benign", "Pathologic", "Controls"} <= categories
    assert "Normal " not in categories

    # T10.3 -- every category resolves to a clinical meaning, including the
    # training-c "Controls" / "Normal" alias.
    assert matched["diagnosis_meaning"].notna().all()


@pytest.mark.needs_data
def test_unmatched_records_are_listed_not_dropped(
    table: Any, audit_files: dict[str, Path]
) -> None:
    """T10.6 -- the 87 unannotated training-e records."""
    unannotated = _read_csv(audit_files["unannotated"])
    assert len(unannotated) == 87
    assert set(unannotated["subset"]) == {"training-e"}

    # Still present in the record table, still labelled, still usable for the
    # binary task -- "unannotated" is not "dropped".
    still_present = table[table["record_uid"].isin(set(unannotated["record_uid"]))]
    assert len(still_present) == 87
    assert still_present["binary_label"].isin([0, 1]).all()
    assert still_present["use_in_supervised"].all()


@pytest.mark.needs_data
def test_coded_annotation_columns(table: Any) -> None:
    """T10.5 -- codes and their decoded names, populated for training-b."""
    training_b = table[table["subset"] == "training-b"]
    legends = pn.annotation_code_maps()

    for short in pn.ORDINAL_ANNOTATION_COLUMNS:
        codes = training_b[short + "_code"]
        labels = training_b[short + "_label"]
        assert codes.notna().all(), short
        # A label exists for exactly the codes the legend defines, and nowhere
        # else. Both halves matter: a missing label on a legal code is a broken
        # legend parse, and a present label on an illegal code is a guess.
        in_legend = codes.isin(list(legends[short])).fillna(False).astype(bool)
        assert (labels.notna() == in_legend).all(), short

    assert set(training_b["murmur_label"].dropna()) <= {
        "None",
        "Weak",
        "Strong",
        "Unclear",
    }


@pytest.mark.needs_data
def test_undocumented_zero_codes_are_kept_but_left_unlabelled(table: Any) -> None:
    """Every legend starts at 2, yet every column also contains 0.

    Pinned per column so the gap cannot widen unnoticed. Filling these in would
    mean inventing a meaning the appendix never states.
    """
    training_b = table[table["subset"] == "training-b"]
    expected = {
        "murmur": 0,
        "murmur_location": 415,
        "arrhythmia": 26,
        "respiration_noise": 36,
        "ambient_noise": 37,
        "recording_noise": 36,
        "abdominal_sounds": 35,
    }
    legends = pn.annotation_code_maps()
    for short, count in expected.items():
        codes = training_b[short + "_code"]
        outside = codes.notna() & ~codes.isin(list(legends[short]))
        assert int(outside.sum()) == count, short
        assert training_b.loc[outside, short + "_label"].isna().all(), short


# ===========================================================================
# T11.7 -- subject derivation gate
# ===========================================================================


@pytest.mark.needs_data
def test_every_record_has_a_subject_id(table: Any) -> None:
    assert table["subject_id"].notna().all()
    assert (table["subject_id"].str.len() > 0).all()
    assert table["subject_derived"].isin([True, False]).all()

    # A subject must never straddle two subsets: the ids are prefixed per subset
    # precisely so that `id17` in training-c cannot collide with anything else.
    spread = table.groupby("subject_id")["subset"].nunique()
    straddling = spread[spread > 1]
    assert set(straddling.index) <= set(
        table.loc[table["is_duplicate"], "subject_id"]
    )


@pytest.mark.needs_data
def test_training_b_matches_the_106_native_subject_ids(table: Any) -> None:
    """T11.2 -- derived versus ground truth, per record, not just in aggregate."""
    training_b = table[table["subset"] == "training-b"]
    assert training_b["native_subject_id"].notna().all()
    assert training_b["subject_id"].nunique() == 106
    assert training_b["native_subject_id"].nunique() == 106
    assert training_b["subject_derived"].all()

    expected = ["b_S" + str(int(v)) for v in training_b["native_subject_id"]]
    assert list(training_b["subject_id"]) == expected


@pytest.mark.needs_data
def test_the_eight_edge_case_records_join_existing_subjects(table: Any) -> None:
    """The four ``C..S..b`` and four ``S34f*.e4k`` records.

    An end-anchored regex misses all eight and invents a singleton group for each
    -- for people whose other recordings are already in the corpus. The aggregate
    subject counts do not move when that happens, so it is checked directly.
    """
    by_record = dict(zip(table["record_id"], table["subject_id"], strict=False))
    sizes = table[table["use_in_supervised"]]["subject_id"].value_counts().to_dict()

    for record, subject in (
        ("a0067", "a_C16"),
        ("a0251", "a_C12"),
        ("a0265", "a_C16"),
        ("a0385", "a_C13"),
        ("b0267", "b_S34"),
        ("b0385", "b_S34"),
        ("b0390", "b_S34"),
        ("b0413", "b_S34"),
    ):
        assert by_record[record] == subject, record
        assert sizes[subject] > 1, record


@pytest.mark.needs_data
def test_training_e_groups_by_raw_record_not_by_record(table: Any) -> None:
    """The deviation from T11.4, agreed 2026-08-25. See derive_subject_physionet."""
    training_e = table[table["subset"] == "training-e"]
    assert len(training_e) == 2141

    derived = training_e[training_e["subject_derived"]]
    assert len(derived) == 2054
    assert derived["subject_id"].nunique() == 404
    # 15, not the 16 a raw `# Raw record` tally reports: the largest tally (284)
    # is two cohorts sharing a number, and namespacing splits it 15 + 1.
    assert derived["subject_id"].value_counts().max() == 15

    # Every group is label-pure. The cohort namespacing exists for exactly this
    # reason: raw-record numbers are reused between the normal and CAD cohorts,
    # and keying on the bare number merges a normal subject with an abnormal one.
    per_group = derived.groupby("subject_id")["binary_label"].nunique()
    assert int(per_group.max()) == 1

    # The 87 with no appendix row keep their own group and say so.
    fallback = training_e[~training_e["subject_derived"]]
    assert len(fallback) == 87
    assert fallback["subject_id"].nunique() == 87


@pytest.mark.needs_data
def test_subject_derived_is_false_wherever_no_pattern_applied(table: Any) -> None:
    """T11.5 -- training-d and training-f, plus the unannotated training-e rows."""
    for subset, expected in (("training-d", 55), ("training-f", 114)):
        rows = table[table["subset"] == subset]
        assert len(rows) == expected
        assert not rows["subject_derived"].any(), subset
        assert rows["subject_id"].nunique() == expected, subset

    undeclared = table[
        table["use_in_supervised"] & ~table["subject_derived"]
    ]
    assert set(undeclared["subset"]) == {"training-d", "training-e", "training-f"}
    assert len(undeclared) == 55 + 114 + 87

    for subset in ("training-a", "training-b", "training-c"):
        assert table.loc[table["subset"] == subset, "subject_derived"].all(), subset


@pytest.mark.needs_data
def test_subject_derivation_csv(table: Any, audit_files: dict[str, Path]) -> None:
    """T11.6 -- record, pattern used, subject id, confidence flag."""
    written = _read_csv(audit_files["subjects"])
    assert len(written) == pn.N_ALL_ROWS
    for column in ("record_uid", "subject_pattern", "subject_id", "subject_derived"):
        assert column in written.columns
    assert written["subject_pattern"].notna().all()
    assert written["subject_id"].nunique() == table["subject_id"].nunique()


# ===========================================================================
# pure-function tests -- no dataset required
# ===========================================================================


def test_parse_header_two_channel(tmp_path: Path) -> None:
    path = tmp_path / "a0001.hea"
    path.write_text(
        "a0001 2 2000 71332\n"
        "a0001.wav 16+44 1 16 0 0 0 0 PCG\n"
        "a0001.dat 16 1000 16 0 0 367 0 ECG\n"
        "# Abnormal\n",
        encoding="utf-8",
    )
    header = pn.parse_header(path)
    assert header.record == "a0001"
    assert header.n_signals == 2
    assert header.fs_hz == 2000
    assert header.n_samples == 71332
    assert header.comment == "Abnormal"
    assert header.has_ecg_channel
    assert header.ecg_signal_file == "a0001.dat"
    assert header.duration_sec == pytest.approx(35.666, abs=1e-3)
    assert header.source == "hea"


def test_parse_header_single_channel(tmp_path: Path) -> None:
    path = tmp_path / "b0001.hea"
    path.write_text(
        "b0001 1 2000 16000\nb0001.wav 16+44 1 16 0 0 0 0 PCG\n# Normal\n",
        encoding="utf-8",
    )
    header = pn.parse_header(path)
    assert not header.has_ecg_channel
    assert header.ecg_signal_file == ""
    assert header.comment == "Normal"


def test_parse_header_rejects_a_miscounted_signal_list(tmp_path: Path) -> None:
    path = tmp_path / "x.hea"
    path.write_text(
        "x 2 2000 100\nx.wav 16+44 1 16 0 0 0 0 PCG\n# Normal\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="declares 2 signal"):
        pn.parse_header(path)


def test_subset_dir_rejects_an_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unknown PhysioNet subset"):
        pn.subset_dir("training-z", Path("/nowhere"))


@pytest.mark.parametrize(
    ("record", "subset", "original", "raw", "subject", "derived"),
    [
        ("a0001", "training-a", "C45S1", None, "a_C45", True),
        ("a0067", "training-a", "C16S0b", None, "a_C16", True),
        ("a9999", "training-a", "", None, "a_rec_a9999", False),
        ("b0001", "training-b", "S98f2_data", None, "b_S98", True),
        ("b0267", "training-b", "S34f2.e4k_data", None, "b_S34", True),
        ("c0001", "training-c", "id17", None, "c_id17", True),
        ("e00002", "training-e", "e01430", 198.0, "e_nR198", True),
        ("e01062", "training-e", "14", 14.0, "e_pR14", True),
        ("e00001", "training-e", None, None, "e_rec_e00001", False),
        ("d0001", "training-d", "C20PSPE", None, "d_rec_d0001", False),
        ("f0001", "training-f", "a80", None, "f_rec_f0001", False),
    ],
)
def test_derive_subject_physionet(
    record: str,
    subset: str,
    original: str | None,
    raw: float | None,
    subject: str,
    derived: bool,
) -> None:
    result = pn.derive_subject_physionet(record, subset, original, raw)
    assert result.subject_id == subject
    assert result.subject_derived is derived
    assert result.pattern


def test_training_e_cohorts_do_not_collide() -> None:
    """Raw record 14 is two different people in two different numbering schemes."""
    normal = pn.derive_subject_physionet("e00019", "training-e", "e00079", 14.0)
    cad = pn.derive_subject_physionet("e01062", "training-e", "14", 14.0)
    assert normal.subject_id != cad.subject_id


def test_expected_counts_sum_to_the_audited_figures() -> None:
    assert pn.N_TRAINING_RECORDS == 3240
    assert pn.N_ALL_ROWS == 3541
