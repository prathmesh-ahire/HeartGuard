"""Assembly of the 138 (Phase 38, gate T38.7).

The gate: the vector is length 138, ordered per the registry, and every value is
finite on real records from all four datasets.

The order check is the one that earns its place. A 138-long vector of finite
floats in the *wrong order* passes every count and finiteness test ever written,
trains a model that works, and produces a SHAP plot whose labels are all wrong.
So the assertion here is against the registry's own name tuple, not against a
length.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import registry as reg
from src.feature_extraction.extractor import (
    FE_ARTIFACTS,
    TIMING_FILENAME,
    AssemblyError,
    benchmark_families,
    extract_all,
    family_summary,
    feature_inventory,
    write_feature_artifacts,
)

FS = 2000
EXPECTED_TOTAL = 138


# ---------------------------------------------------------------------------
# T38.1 / T38.2 -- shape and order
# ---------------------------------------------------------------------------


def test_the_vector_is_138_long_and_in_registry_order(synthetic_signal: np.ndarray):
    result = extract_all(synthetic_signal, FS, record_uid="synthetic")

    assert len(result.values) == EXPECTED_TOTAL
    assert result.vector.shape == (EXPECTED_TOTAL,)
    assert result.names == reg.FEATURE_NAMES
    assert list(result.values) == list(reg.FEATURE_NAMES)


def test_all_values_are_finite_on_the_synthetic_signal(synthetic_signal: np.ndarray):
    """T38.5 -- the synthetic-signal assertion."""
    result = extract_all(synthetic_signal, FS, record_uid="synthetic")
    assert result.failed_families == (), result.errors
    assert result.n_missing == 0, result.missing_names()
    assert np.isfinite(result.vector).all()


def test_the_vector_is_float64_throughout(synthetic_signal: np.ndarray):
    vector = extract_all(synthetic_signal, FS).vector
    assert vector.dtype == np.float64


def test_column_index_maps_back_to_the_registry_spec(synthetic_signal: np.ndarray):
    """Column 47 must mean what the registry says column 47 means."""
    result = extract_all(synthetic_signal, FS)
    for index, name in enumerate(result.names):
        assert reg.index_of(name) == index
        assert reg.spec_for(name).index == index


def test_a_short_or_misordered_vector_is_rejected():
    """The T38.2 guard, exercised directly -- it must be able to fail."""
    from src.feature_extraction import extractor as ex

    good = dict.fromkeys(reg.FEATURE_NAMES, 0.0)
    ex._validate(good)  # must not raise

    short = dict(good)
    short.pop(reg.FEATURE_NAMES[-1])
    with pytest.raises(AssemblyError, match="137"):
        ex._validate(short)

    shuffled = dict.fromkeys(reversed(reg.FEATURE_NAMES), 0.0)
    with pytest.raises(AssemblyError, match="registry order"):
        ex._validate(shuffled)


def test_extraction_is_deterministic(synthetic_signal: np.ndarray):
    """Rule 5: two runs of the same command produce identical numbers."""
    first = extract_all(synthetic_signal, FS).vector
    second = extract_all(synthetic_signal, FS).vector
    assert np.array_equal(first, second)


def test_as_row_carries_the_uid_the_features_and_the_bookkeeping(
    synthetic_signal: np.ndarray,
):
    row = extract_all(synthetic_signal, FS, record_uid="R-1").as_row()
    assert row["record_uid"] == "R-1"
    assert row["n_missing"] == 0
    assert row["failed_families"] == ""
    assert row["extract_seconds"] > 0
    assert all(name in row for name in reg.FEATURE_NAMES)
    # uid + 138 + n_missing + flags + failed_families + extract_seconds
    assert len(row) == EXPECTED_TOTAL + 5


# ---------------------------------------------------------------------------
# a skipped family is NaN in the right columns, never a short row
# ---------------------------------------------------------------------------


def test_restricting_families_still_returns_138_columns(synthetic_signal: np.ndarray):
    """EXP-F1 subsets columns of a full matrix; it never changes the row width."""
    result = extract_all(synthetic_signal, FS, families=["time", "envelope"])

    assert len(result.values) == EXPECTED_TOTAL
    assert result.names == reg.FEATURE_NAMES

    for name in reg.feature_names("time"):
        assert np.isfinite(result.values[name]), name
    for name in reg.feature_names("mfcc"):
        assert np.isnan(result.values[name]), name

    assert "mfcc_not_run" in result.flags
    assert "time_not_run" not in result.flags
    assert set(result.timings) == {"time", "envelope"}


def test_an_unknown_family_is_rejected(synthetic_signal: np.ndarray):
    with pytest.raises(AssemblyError, match="unknown family"):
        extract_all(synthetic_signal, FS, families=["spectrogram"])


def test_a_failing_family_is_named_rather_than_silently_empty():
    """One sample: MFCC and chroma cannot run, the other four can."""
    result = extract_all(np.array([0.5]), FS, record_uid="single_sample")

    assert len(result.values) == EXPECTED_TOTAL
    assert result.n_missing > 0
    assert result.flags, "a family that could not run must leave a flag"
    # Every missing value belongs to a family that reported a flag or an error.
    for name in result.missing_names():
        family = reg.family_of(name)
        explained = family in result.failed_families or any(
            flag.startswith(family + ":") for flag in result.flags
        )
        assert explained, name + " is NaN with nothing explaining it"


# ---------------------------------------------------------------------------
# T38.3 / T38.4 -- FE-01 and FE-02
# ---------------------------------------------------------------------------


def test_fe01_inventory_has_one_row_per_feature_fully_populated():
    inventory = feature_inventory()

    assert len(inventory) == EXPECTED_TOTAL
    assert list(inventory.columns) == [
        "index",
        "name",
        "family",
        "extractor",
        "equation",
        "unit",
        "description",
    ]
    assert list(inventory["name"]) == list(reg.FEATURE_NAMES)
    assert list(inventory["index"]) == list(range(EXPECTED_TOTAL))
    assert inventory["name"].is_unique
    for column in ("name", "family", "extractor", "equation", "description"):
        assert inventory[column].astype(str).str.strip().ne("").all(), column


def test_fe02_summary_reports_the_locked_counts():
    summary = family_summary()

    families = summary[summary["family"] != "TOTAL"]
    assert list(families["family"]) == list(reg.FAMILY_ORDER)
    assert list(families["n_features"]) == [24, 22, 39, 24, 24, 5]
    assert list(families["expected_count"]) == [24, 22, 39, 24, 24, 5]
    assert bool(summary["matches_expected"].all())

    total = summary[summary["family"] == "TOTAL"].iloc[0]
    assert int(total["n_features"]) == EXPECTED_TOTAL
    assert int(total["expected_count"]) == EXPECTED_TOTAL


def test_fe02_first_index_marks_where_each_family_block_starts():
    summary = family_summary()
    families = summary[summary["family"] != "TOTAL"]
    starts = list(families["first_index"])
    assert starts == [0, 24, 46, 85, 109, 133]
    for family, start in zip(families["family"], starts, strict=True):
        assert reg.FEATURE_NAMES[start] == reg.feature_names(family)[0]


def test_artifacts_are_written_and_reproducible(tmp_path: Path,
                                                synthetic_signal: np.ndarray):
    first = write_feature_artifacts(
        tmp_path / "a", benchmark_signal=synthetic_signal, repeats=1
    )
    assert set(first) == {"FE-01", "FE-02", "timing"}
    assert first["FE-01"].name == FE_ARTIFACTS["FE-01"]
    assert first["FE-02"].name == FE_ARTIFACTS["FE-02"]
    assert first["timing"].name == TIMING_FILENAME

    second = write_feature_artifacts(tmp_path / "b")
    # FE-01 and FE-02 describe the registry, so they are byte-stable. The timing
    # file is a measurement and is deliberately NOT compared.
    assert first["FE-01"].read_bytes() == second["FE-01"].read_bytes()
    assert first["FE-02"].read_bytes() == second["FE-02"].read_bytes()


# ---------------------------------------------------------------------------
# T38.6 -- the per-family benchmark
# ---------------------------------------------------------------------------


def test_benchmark_covers_every_family_and_sums_to_one(synthetic_signal: np.ndarray):
    table = benchmark_families(synthetic_signal, FS, repeats=2)

    assert set(table["family"]) == set(reg.FAMILY_ORDER)
    assert list(table["n_features"]) == [
        reg.EXPECTED_FAMILY_COUNTS[family] for family in table["family"]
    ]
    assert (table["min_seconds"] > 0).all()
    assert (table["min_seconds"] <= table["mean_seconds"]).all()
    assert (table["mean_seconds"] <= table["max_seconds"]).all()
    assert float(table["share_of_total"].sum()) == pytest.approx(1.0)
    # Sorted most expensive first, so the complexity table reads top-down.
    assert list(table["min_seconds"]) == sorted(table["min_seconds"], reverse=True)


def test_the_time_family_dominates_the_extraction_budget(synthetic_signal: np.ndarray):
    """Sample entropy is ~98% of the cost of a record; pin that so it stays visible.

    If this ever fails because the time family got cheap, that is good news and
    open decision 7 can be closed. If it fails because another family got
    expensive, something regressed.
    """
    table = benchmark_families(synthetic_signal, FS, repeats=2)
    top = table.iloc[0]
    assert top["family"] == "time"
    assert float(top["share_of_total"]) > 0.8


# ---------------------------------------------------------------------------
# real records -- the gate (T38.7)
# ---------------------------------------------------------------------------


def _preprocessed(path: Path, uid: str) -> np.ndarray:
    from src.preprocessing.pipeline import preprocess

    return preprocess(path, record_uid=uid).signal


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_yield_138_finite_values_in_order(
    sample_records: Any, dataset: str
):
    """The gate, verbatim: 138, ordered, all finite, on every dataset."""
    picked = sample_records(dataset, 3)
    assert picked, "no records for " + dataset

    for uid, path in picked:
        result = extract_all(_preprocessed(path, uid), FS, record_uid=uid)

        assert len(result.values) == EXPECTED_TOTAL, dataset + " " + uid
        assert result.names == reg.FEATURE_NAMES, dataset + " " + uid
        assert result.failed_families == (), (
            dataset + " " + uid + ": " + str(result.errors)
        )
        assert result.n_missing == 0, (
            dataset + " " + uid + " non-finite: " + ", ".join(result.missing_names())
        )


@pytest.mark.needs_data
def test_duration_extremes_assemble_cleanly(master_frame: Any, project_root: Path):
    """0.76 s and 122 s. Both degrade in at least one family; neither loses a column."""
    shortest = master_frame.loc[master_frame["duration_sec"].idxmin()]
    longest = master_frame.loc[master_frame["duration_sec"].idxmax()]

    for row in (shortest, longest):
        uid = str(row["record_uid"])
        signal = _preprocessed(project_root / str(row["file_path"]), uid)
        result = extract_all(signal, FS, record_uid=uid)

        assert len(result.values) == EXPECTED_TOTAL, uid
        assert result.names == reg.FEATURE_NAMES, uid
        assert result.failed_families == (), uid + ": " + str(result.errors)
        assert result.n_missing == 0, uid + ": " + ", ".join(result.missing_names())


@pytest.mark.needs_data
def test_every_family_contributes_a_timing_on_a_real_record(sample_records: Any):
    uid, path = sample_records("D1", 1)[0]
    result = extract_all(_preprocessed(path, uid), FS, record_uid=uid)

    assert set(result.timings) == set(reg.FAMILY_ORDER)
    assert all(seconds > 0 for seconds in result.timings.values())
    assert result.seconds == pytest.approx(sum(result.timings.values()))


@pytest.mark.needs_data
def test_fe01_and_fe02_exist_in_the_features_output_directory():
    from src.feature_extraction.extractor import features_dir

    for evidence_id, filename in FE_ARTIFACTS.items():
        path = features_dir() / filename
        assert path.is_file(), evidence_id + " missing: " + str(path)
        assert path.stat().st_size > 0, evidence_id + " is empty"
