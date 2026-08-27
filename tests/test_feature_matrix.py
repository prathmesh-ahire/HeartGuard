"""FE-03, the merged feature matrix (Phase 40, gate T40.7).

The gate: one row per non-excluded record, 138 feature columns, a clean join to
master metadata on ``record_uid``, and a recorded wall time.

The join guard is tested on fabricated frames as well as on the real matrix,
because the interesting cases -- a short extraction, an orphan record, a
duplicate uid -- must not require breaking the real cache to exercise. A guard
that has never been seen to fail is not known to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import matrix as mx
from src.feature_extraction.registry import FEATURE_NAMES

EXPECTED_TOTAL = 138

#: The audited corpus (CLAUDE.md), not a count re-derived from whatever is on
#: disk. If extraction silently covered fewer records, this is what catches it.
EXPECTED_PER_DATASET = {"D1": 3541, "D2": 176, "D3": 656, "D4": 3163}
EXPECTED_CORPUS = 7536


# ---------------------------------------------------------------------------
# the join guard, on fabricated frames
# ---------------------------------------------------------------------------


def _frames(master_uids: list[str], feature_uids: list[str]) -> tuple[Any, Any]:
    import pandas as pd

    master = pd.DataFrame({"record_uid": master_uids, "dataset_source": "D2"})
    features = pd.DataFrame({"record_uid": feature_uids})
    return master, features


def test_a_complete_join_passes_the_guard():
    master, features = _frames(["a", "b", "c"], ["c", "a", "b"])
    mx._check_join(master, features)  # order is irrelevant; coverage is not


def test_an_incomplete_extraction_is_rejected_by_name():
    """The failure this guard exists for: fewer features than records."""
    master, features = _frames(["a", "b", "c"], ["a", "b"])
    with pytest.raises(mx.MatrixError, match="incomplete"):
        mx._check_join(master, features)


def test_the_missing_records_are_named_not_just_counted():
    master, features = _frames(["a", "b", "c"], ["a"])
    with pytest.raises(mx.MatrixError) as info:
        mx._check_join(master, features)
    message = str(info.value)
    assert "2 master records" in message
    assert "b" in message and "c" in message


def test_a_record_absent_from_master_is_rejected():
    master, features = _frames(["a", "b"], ["a", "b", "ghost"])
    with pytest.raises(mx.MatrixError, match="absent from master"):
        mx._check_join(master, features)


def test_a_duplicated_uid_is_rejected():
    """Two rows for one record would silently double that record's weight."""
    master, features = _frames(["a", "b"], ["a", "b", "b"])
    with pytest.raises(mx.MatrixError, match="duplicate"):
        mx._check_join(master, features)


def test_an_unknown_dataset_is_rejected():
    with pytest.raises(mx.MatrixError, match="unknown dataset"):
        mx.build_matrix(["D9"])


# ---------------------------------------------------------------------------
# the real matrix -- the gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fe03() -> Any:
    if not mx.matrix_path().is_file():
        pytest.skip("FE-03 not built; run scripts/03_feature_reports.py")
    return mx.load_matrix()


@pytest.mark.needs_data
def test_one_row_per_non_excluded_record(fe03: Any, master_frame: Any):
    """T40.7, first clause."""
    assert len(fe03) == len(master_frame)
    assert len(fe03) == EXPECTED_CORPUS
    assert fe03["record_uid"].is_unique


@pytest.mark.needs_data
def test_every_master_record_is_present_and_no_others(fe03: Any, master_frame: Any):
    got = set(fe03["record_uid"].astype(str))
    expected = set(master_frame["record_uid"].astype(str))
    assert got == expected, "symmetric difference: " + str(
        sorted(got ^ expected)[:10]
    )


@pytest.mark.needs_data
def test_per_dataset_counts_match_the_audit(fe03: Any):
    counts = fe03["dataset_source"].astype(str).value_counts().to_dict()
    assert {key: int(value) for key, value in counts.items()} == EXPECTED_PER_DATASET


@pytest.mark.needs_data
def test_the_138_are_present_in_registry_order(fe03: Any):
    """T40.7, second clause -- and order, which a column count would not catch."""
    present = [column for column in fe03.columns if column in set(FEATURE_NAMES)]
    assert present == list(FEATURE_NAMES)
    assert len(present) == EXPECTED_TOTAL
    # The features are the last 138 columns; metadata precedes them.
    assert list(fe03.columns[-EXPECTED_TOTAL:]) == list(FEATURE_NAMES)


@pytest.mark.needs_data
def test_feature_columns_are_numeric_float64(fe03: Any):
    for name in FEATURE_NAMES:
        assert np.issubdtype(fe03[name].dtype, np.floating), name


@pytest.mark.needs_data
def test_master_metadata_survived_the_join(fe03: Any, master_frame: Any):
    """T40.7, third clause -- the labels are actually attached, not just the shape."""
    for column in (
        "dataset_source",
        "subject_id",
        "binary_label",
        "murmur_label",
        "outcome_label",
        "duration_sec",
        "use_in_supervised",
    ):
        assert column in fe03.columns, column

    left = fe03.set_index("record_uid")["duration_sec"].astype(float)
    right = master_frame.set_index("record_uid")["duration_sec"].astype(float)
    aligned = right.reindex(left.index)
    assert np.allclose(left.to_numpy(), aligned.to_numpy())


@pytest.mark.needs_data
def test_labels_are_not_silently_shifted_by_the_join(fe03: Any, master_frame: Any):
    """A join on the wrong key still produces 7,536 rows -- with wrong labels."""
    left = fe03.set_index("record_uid")["binary_label"]
    right = master_frame.set_index("record_uid")["binary_label"].reindex(left.index)
    both = left.notna() & right.notna()
    assert bool(both.any())
    assert (left[both].astype(float) == right[both].astype(float)).all()


@pytest.mark.needs_data
def test_extraction_bookkeeping_came_through(fe03: Any):
    for column in ("n_missing", "flags", "failed_families", "extract_seconds"):
        assert column in fe03.columns, column
    assert (fe03["extract_seconds"].astype(float) > 0).all()
    assert set(fe03.columns) >= {"sec_time", "sec_mfcc", "sec_dwt"}


@pytest.mark.needs_data
def test_no_family_failed_across_the_corpus(fe03: Any):
    """Every record produced all six families, or this says which did not."""
    failed = fe03[fe03["failed_families"].astype(str).str.strip() != ""]
    assert failed.empty, (
        str(len(failed))
        + " records had a family fail, e.g. "
        + str(failed[["record_uid", "failed_families"]].head(5).to_dict("records"))
    )


# ---------------------------------------------------------------------------
# T40.7, fourth clause -- wall time
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_wall_time_is_recorded_per_dataset(fe03: Any):
    table = mx.wall_time_table(fe03)
    assert set(table["dataset"]) == set(EXPECTED_PER_DATASET)
    assert (table["n_records"] > 0).all()
    assert (table["total_extract_sec"] > 0).all()
    assert table["share_of_total"].sum() == pytest.approx(1.0)
    assert list(table["n_records"]) == [
        EXPECTED_PER_DATASET[name] for name in table["dataset"]
    ]


@pytest.mark.needs_data
def test_the_wall_time_csv_exists_beside_the_matrix():
    path = mx.matrix_path().parent / mx.WALL_TIME_FILENAME
    if not mx.matrix_path().is_file():
        pytest.skip("FE-03 not built; run scripts/03_feature_reports.py")

    assert path.is_file(), "T40.1/T40.5 wall time not recorded: " + str(path)
    assert path.stat().st_size > 0


@pytest.mark.needs_data
def test_the_run_manifest_records_the_extraction_wall_clock():
    """The per-record seconds are CPU time; the manifest holds the wall clock."""
    import json

    from src.utils.config import load_config

    if not mx.matrix_path().is_file():
        pytest.skip("FE-03 not built; run scripts/03_feature_reports.py")

    root = Path(load_config("paths").require("outputs.evidence_index"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    runs = manifest if isinstance(manifest, list) else manifest.get("runs", [])
    full = [
        run
        for run in runs
        if run.get("name") == "feature_extraction"
        and run.get("extra", {}).get("extract_mode") == "full"
    ]
    assert full, "no full feature_extraction run in the manifest"
    seconds = float(full[-1]["extra"]["extract_wall_seconds"])
    assert seconds > 0


@pytest.mark.needs_data
def test_fe03_is_registered_in_the_evidence_index():
    from src.utils.evidence import read_evidence
    if not mx.matrix_path().is_file():
        pytest.skip("FE-03 not built; run scripts/03_feature_reports.py")

    rows = {row["evidence_id"]: row for row in read_evidence()}
    assert "FE-03" in rows, "FE-03 not registered"
    assert rows["FE-03"]["status"] == "ok"
    assert rows["FE-03"]["filename"].endswith(mx.FE03_FILENAME)
