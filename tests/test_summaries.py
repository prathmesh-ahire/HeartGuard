"""Duration, sampling-rate and class-distribution gate (T18.7).

The audited figures these tests check against were measured on 2026-08-22 and
are recorded in ``Docs/note.md``. They are asserted here to four decimal places
because that is what a summary table is for: if a duration statistic moves, a
file changed, and every downstream duration band and robustness stratum moved
with it.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.data_loader import summaries as sm

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> Any:
    from src.data_loader.catalog import build_catalog

    return build_catalog()


@pytest.fixture(scope="module")
def durations(catalog: Any) -> Any:
    return sm.duration_summary(catalog)


def _row(frame: Any, **filters: Any) -> Any:
    subset = frame
    for column, value in filters.items():
        subset = subset[subset[column] == value]
    assert len(subset) == 1, (filters, len(subset))
    return subset.iloc[0]


# ===========================================================================
# T18.7 -- the gate
# ===========================================================================


@pytest.mark.needs_data
def test_physionet_duration_matches_the_audit(durations: Any) -> None:
    """PhysioNet median 20.83 s, min 5.31 s, max 122.00 s.

    Checked on the *supervised* population -- the 3,240 training records. The
    301 validation rows are byte-identical copies of records already counted, so
    including them would report a corpus this project never uses.
    """
    row = _row(durations, scope="supervised", dataset_source="D1", **{"class": "ALL"})
    assert int(row["n"]) == 3240
    assert float(row["median"]) == pytest.approx(20.83, abs=0.01)
    assert float(row["min"]) == pytest.approx(5.31, abs=0.01)
    assert float(row["max"]) == pytest.approx(122.00, abs=0.01)


@pytest.mark.needs_data
def test_pascal_durations_match_the_audit(durations: Any) -> None:
    """set_a 0.94 / 8.88 / 9.00 s, set_b 0.76 / 4.95 / 27.87 s.

    **The two audited medians were measured on different populations**, which
    the 2026-08-22 table does not say. set_a's 8.88 is the 124 *labelled*
    records (all 176 give 8.75); set_b's 4.95 is *all* 656 (the 461 labelled
    give 5.09). Both scopes are asserted here so the ambiguity cannot come back.

    set_b's 0.76 s minimum is the number the whole preprocessing chain is
    designed around: 1,520 samples at the 2 kHz target, below what a 5-level db4
    decomposition or an MFCC delta window wants.
    """
    set_a_labelled = _row(
        durations, scope="supervised", dataset_source="D2", **{"class": "ALL"}
    )
    assert int(set_a_labelled["n"]) == 124
    assert float(set_a_labelled["min"]) == pytest.approx(0.94, abs=0.01)
    assert float(set_a_labelled["median"]) == pytest.approx(8.88, abs=0.01)
    assert float(set_a_labelled["max"]) == pytest.approx(9.00, abs=0.01)

    set_a_all = _row(
        durations, scope="all_records", dataset_source="D2", **{"class": "ALL"}
    )
    assert int(set_a_all["n"]) == 176
    assert float(set_a_all["median"]) == pytest.approx(8.75, abs=0.01)

    set_b_all = _row(
        durations, scope="all_records", dataset_source="D3", **{"class": "ALL"}
    )
    assert int(set_b_all["n"]) == 656
    assert float(set_b_all["min"]) == pytest.approx(0.76, abs=0.01)
    assert float(set_b_all["median"]) == pytest.approx(4.95, abs=0.01)
    assert float(set_b_all["max"]) == pytest.approx(27.87, abs=0.01)

    set_b_labelled = _row(
        durations, scope="supervised", dataset_source="D3", **{"class": "ALL"}
    )
    assert int(set_b_labelled["n"]) == 461
    assert float(set_b_labelled["median"]) == pytest.approx(5.09, abs=0.01)


@pytest.mark.needs_data
def test_circor_duration_matches_the_audit(durations: Any) -> None:
    """CirCor 5.15 / 21.46 / 64.51 s."""
    row = _row(durations, scope="supervised", dataset_source="D4", **{"class": "ALL"})
    assert int(row["n"]) == 3163
    assert float(row["min"]) == pytest.approx(5.15, abs=0.01)
    assert float(row["median"]) == pytest.approx(21.46, abs=0.01)
    assert float(row["max"]) == pytest.approx(64.51, abs=0.01)


@pytest.mark.needs_data
def test_three_native_sampling_rates(catalog: Any) -> None:
    """T18.5/T18.7 -- 2000, 4000 and 44100 Hz, and the 22.05x that follows."""
    summary = sm.sampling_rate_summary(catalog)
    assert set(summary["original_fs"]) == {2000, 4000, 44100}
    assert set(summary["converted_fs"]) == {2000}

    by_dataset = summary.set_index("dataset_source")
    assert int(by_dataset.at["D1", "original_fs"]) == 2000
    assert int(by_dataset.at["D2", "original_fs"]) == 44100
    assert int(by_dataset.at["D3", "original_fs"]) == 4000
    assert int(by_dataset.at["D4", "original_fs"]) == 4000

    assert by_dataset.at["D1", "conversion"] == "none"
    assert float(by_dataset.at["D2", "factor"]) == pytest.approx(22.05)
    assert float(by_dataset.at["D3", "factor"]) == pytest.approx(2.0)
    assert int(summary["n_records"].sum()) == 7536


@pytest.mark.needs_data
def test_every_record_gets_a_duration_band(catalog: Any) -> None:
    """T18.3 -- short / medium / long, none left unassigned."""
    banded = sm.assign_duration_bands(catalog)
    assert set(banded["duration_band"]) <= set(sm.DURATION_BAND_NAMES)
    assert banded["duration_band"].notna().all()
    assert len(banded) == len(catalog)

    thresholds_short = banded[banded["duration_band"] == "short"]
    thresholds_long = banded[banded["duration_band"] == "long"]
    assert (thresholds_short["duration_sec"] < 5.0).all()
    assert (thresholds_long["duration_sec"] > 20.0).all()

    # All three bands are populated -- the strata EXP-E reports against are not
    # hypothetical on this corpus.
    assert banded["duration_band"].nunique() == 3


@pytest.mark.needs_data
def test_class_distribution_is_per_task_and_never_pooled(catalog: Any) -> None:
    """T18.6 -- rule 4, in the one table most likely to break it."""
    distribution = sm.class_distribution(catalog)
    supervised = distribution[distribution["scope"] == "supervised"]

    by_task = supervised.groupby("task")["n_records"].sum().to_dict()
    assert by_task["binary"] == 3240
    assert by_task["pascal_a"] == 124
    assert by_task["pascal_b"] == 461
    assert by_task["circor_murmur"] == 942 or by_task["circor_murmur"] == 3163
    assert by_task["circor_outcome"] == by_task["circor_murmur"]

    # extrahls belongs to pascal_a and extrastole to pascal_b, and neither
    # appears under the other's task.
    pascal_a = set(supervised.loc[supervised["task"] == "pascal_a", "class"])
    pascal_b = set(supervised.loc[supervised["task"] == "pascal_b", "class"])
    assert "extrahls" in pascal_a and "extrahls" not in pascal_b
    assert "extrastole" in pascal_b and "extrastole" not in pascal_a

    # Every task's classes come only from that task's vocabulary.
    from src.utils.constants import label_names

    for task, group in supervised.groupby("task"):
        assert set(group["class"]) <= set(label_names(str(task))), task


@pytest.mark.needs_data
def test_class_distribution_counts_and_imbalance(catalog: Any) -> None:
    """The audited class counts, and a majority class that reads 1.0."""
    distribution = sm.class_distribution(catalog)
    supervised = distribution[distribution["scope"] == "supervised"]

    binary = supervised[supervised["task"] == "binary"].set_index("class")
    assert int(binary.at["normal", "n_records"]) == 2575
    assert int(binary.at["abnormal", "n_records"]) == 665
    assert float(binary.at["normal", "imbalance_ratio"]) == pytest.approx(1.0)
    assert float(binary.at["abnormal", "imbalance_ratio"]) == pytest.approx(
        2575 / 665, abs=1e-3
    )

    pascal_a = supervised[supervised["task"] == "pascal_a"].set_index("class")
    assert int(pascal_a.at["artifact", "n_records"]) == 40
    assert int(pascal_a.at["extrahls", "n_records"]) == 19
    assert int(pascal_a.at["murmur", "n_records"]) == 34
    assert int(pascal_a.at["normal", "n_records"]) == 31

    pascal_b = supervised[supervised["task"] == "pascal_b"].set_index("class")
    assert int(pascal_b.at["extrastole", "n_records"]) == 46
    assert int(pascal_b.at["murmur", "n_records"]) == 95
    assert int(pascal_b.at["normal", "n_records"]) == 320

    # Shares sum to 1 within each task.
    for _, group in supervised.groupby(["dataset_source", "task"]):
        assert float(group["share"].sum()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.needs_data
def test_the_three_summaries_are_written(catalog: Any, tmp_path: Any) -> None:
    """T18.4, T18.5, T18.6 -- DA-03, DA-04, DA-02."""
    import pandas as pd

    results = sm.run_summaries(catalog, write_outputs=True, out_dir=tmp_path)
    for name in ("duration", "sampling_rate", "class_distribution"):
        assert len(results[name]) > 0, name

    for filename in (
        "recording_duration_summary.csv",
        "sampling_rate_summary.csv",
        "class_distribution.csv",
    ):
        target = tmp_path / filename
        assert target.is_file(), filename
        assert len(pd.read_csv(target)) > 0, filename


# ===========================================================================
# pure-function tests -- no dataset required
# ===========================================================================


def _tiny_catalog() -> Any:
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "record_uid": "D1_x_a",
                "dataset_source": "D1",
                "subject_id": "s1",
                "duration_sec": 2.0,
                "original_fs": 2000,
                "use_in_supervised": True,
                "binary_class": "normal",
                "pascal_a_class": "",
                "pascal_b_class": "",
                "circor_murmur_class": "",
                "circor_outcome_class": "",
            },
            {
                "record_uid": "D1_x_b",
                "dataset_source": "D1",
                "subject_id": "s2",
                "duration_sec": 12.0,
                "original_fs": 2000,
                "use_in_supervised": True,
                "binary_class": "abnormal",
                "pascal_a_class": "",
                "pascal_b_class": "",
                "circor_murmur_class": "",
                "circor_outcome_class": "",
            },
            {
                "record_uid": "D1_x_c",
                "dataset_source": "D1",
                "subject_id": "s3",
                "duration_sec": 40.0,
                "original_fs": 2000,
                "use_in_supervised": False,
                "binary_class": "normal",
                "pascal_a_class": "",
                "pascal_b_class": "",
                "circor_murmur_class": "",
                "circor_outcome_class": "",
            },
        ]
    )


def test_duration_bands_use_the_configured_edges() -> None:
    banded = sm.assign_duration_bands(_tiny_catalog(), short_below=5.0, long_above=20.0)
    assert list(banded["duration_band"]) == ["short", "medium", "long"]


def test_band_edges_are_exclusive_on_both_sides() -> None:
    """Exactly 5.0 s is medium, not short; exactly 20.0 s is medium, not long."""
    import pandas as pd

    frame = pd.DataFrame({"duration_sec": [4.999, 5.0, 20.0, 20.001]})
    banded = sm.assign_duration_bands(frame, short_below=5.0, long_above=20.0)
    assert list(banded["duration_band"]) == ["short", "medium", "medium", "long"]


def test_scope_separates_supervised_from_all_records() -> None:
    summary = sm.duration_summary(_tiny_catalog())
    supervised = _row(summary, scope="supervised", dataset_source="D1", **{"class": "ALL"})
    everything = _row(
        summary, scope="all_records", dataset_source="D1", **{"class": "ALL"}
    )
    assert int(supervised["n"]) == 2
    assert int(everything["n"]) == 3
    assert float(supervised["max"]) == 12.0
    assert float(everything["max"]) == 40.0


def test_single_record_group_reports_no_standard_deviation() -> None:
    """One record has no spread; NaN is the honest answer, not 0."""
    import math

    summary = sm.duration_summary(_tiny_catalog())
    row = _row(summary, scope="supervised", dataset_source="D1", **{"class": "normal"})
    assert int(row["n"]) == 1
    assert math.isnan(float(row["sd"]))


def test_class_distribution_counts_subjects_as_well_as_records() -> None:
    distribution = sm.class_distribution(_tiny_catalog())
    supervised = distribution[distribution["scope"] == "supervised"].set_index("class")
    assert int(supervised.at["normal", "n_records"]) == 1
    assert int(supervised.at["normal", "n_subjects"]) == 1
    assert set(distribution["task"]) == {"binary"}
