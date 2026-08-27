"""Feature quality assurance (Phase 41, gate T41.7).

The gate has two clauses:

1. Re-extract 50 random records and assert **bit-identical** values against the
   cache. Not ``approx``: research rule 5 says two runs of the same command
   produce identical numbers, and any tolerance here would hide the exact drift
   -- an unseeded window, a library default, a dict ordering -- that the rule is
   written to catch.
2. FE-04 accounts for **every** NaN and Inf with a named record. The test
   re-derives the corpus total from the report and compares it against the
   matrix, so a report that undercounts fails rather than reading as a clean
   bill of health.

The whole-corpus checks are marked slow because clause 1 re-runs the real
extractor on 50 real recordings, and the time family costs about a second each.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.feature_extraction import matrix as mx
from src.feature_extraction import quality as qa
from src.feature_extraction.registry import FEATURE_NAMES

REPRO_SAMPLE = 50


@pytest.fixture(scope="module")
def fe03() -> Any:
    if not mx.matrix_path().is_file():
        pytest.skip("FE-03 not built; run scripts/03_feature_reports.py")
    return mx.load_matrix()


# ---------------------------------------------------------------------------
# the reports, on a fabricated matrix -- so the logic is testable without data
# ---------------------------------------------------------------------------


def _fake_matrix(n_rows: int = 40) -> Any:
    """A matrix with known defects planted in known columns."""
    import pandas as pd

    rng = np.random.default_rng(42)
    data: dict[str, Any] = {
        "record_uid": ["R" + format(index, "03d") for index in range(n_rows)],
        "dataset_source": ["D1"] * (n_rows // 2) + ["D4"] * (n_rows - n_rows // 2),
        "duration_sec": rng.uniform(5.0, 30.0, n_rows),
        "binary_label": [0, 1] * (n_rows // 2),
        "binary_label_name": ["normal", "abnormal"] * (n_rows // 2),
        "use_in_supervised": [True] * n_rows,
    }
    for name in FEATURE_NAMES:
        data[name] = rng.normal(size=n_rows)

    frame = pd.DataFrame(data)
    frame.loc[0, FEATURE_NAMES[0]] = np.nan
    frame.loc[3, FEATURE_NAMES[0]] = np.nan
    frame.loc[7, FEATURE_NAMES[5]] = np.inf
    frame[FEATURE_NAMES[9]] = 1.0  # constant
    frame.loc[1, FEATURE_NAMES[12]] = 1e9  # a wild tail
    return frame


def test_fe04_names_the_records_responsible():
    frame = _fake_matrix()
    report = qa.missing_value_report(frame)

    assert len(report) == len(FEATURE_NAMES)
    row = report[report["feature"] == FEATURE_NAMES[0]].iloc[0]
    assert int(row["n_nan"]) == 2
    assert int(row["n_inf"]) == 0
    assert "R000" in row["record_uids"] and "R003" in row["record_uids"]

    inf_row = report[report["feature"] == FEATURE_NAMES[5]].iloc[0]
    assert int(inf_row["n_inf"]) == 1
    assert int(inf_row["n_nan"]) == 0
    assert "R007" in inf_row["record_uids"]


def test_fe04_counts_nan_and_inf_separately():
    """An Inf is not a missing value; conflating them hides an overflow."""
    report = qa.missing_value_report(_fake_matrix())
    assert int(report["n_nan"].sum()) == 2
    assert int(report["n_inf"].sum()) == 1
    assert int(report["n_nonfinite"].sum()) == 3


def test_fe04_totals_reconcile_with_the_matrix():
    """The gate's second clause, on planted defects."""
    frame = _fake_matrix()
    report = qa.missing_value_report(frame)
    values = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    assert int(report["n_nonfinite"].sum()) == int((~np.isfinite(values)).sum())


def test_variance_report_finds_the_constant_column():
    report = qa.variance_report(_fake_matrix())
    constant = report[report["verdict"] == "constant"]
    assert list(constant["feature"]) == [FEATURE_NAMES[9]]
    assert int(constant.iloc[0]["n_unique"]) == 1


def test_variance_report_does_not_drop_anything():
    """T41.2 is explicitly report-only; dropping happens inside the fold."""
    report = qa.variance_report(_fake_matrix())
    assert list(report["feature"]) == list(FEATURE_NAMES)


def test_outlier_report_flags_the_wild_tail_and_states_a_policy():
    report = qa.outlier_report(_fake_matrix())
    row = report[report["feature"] == FEATURE_NAMES[12]].iloc[0]
    assert int(row["n_extreme"]) >= 1
    assert row["verdict"] in {"heavy_tail", "unbounded_tail"}
    assert row["policy"] != "none"
    assert float(row["max"]) == pytest.approx(1e9)


def test_the_clipping_policy_says_it_is_fitted_inside_the_fold():
    """A policy that forgot this would be a leak written down as a decision."""
    assert "training fold" in qa.CLIPPING_POLICY
    assert "rule 2" in qa.CLIPPING_POLICY


def test_domain_shift_reports_one_row_per_feature_with_both_datasets():
    report = qa.domain_shift_report(_fake_matrix())
    assert len(report) == len(FEATURE_NAMES)
    assert set(report["feature"]) == set(FEATURE_NAMES)
    for column in ("mean_D1", "mean_D4", "smd_D1_vs_D4"):
        assert column in report.columns

    # Sorted descending with NaN last. The NaN is correct and expected: the
    # planted constant feature has zero variance in both datasets, so its pooled
    # denominator is zero and no standardised difference exists. pandas reports
    # is_monotonic_decreasing as False whenever any NaN is present, so the order
    # is checked on the defined values and the NaN's position separately.
    ordered = report["abs_smd_D1_vs_D4"]
    assert ordered.dropna().is_monotonic_decreasing
    assert ordered.isna().sum() == 1
    assert bool(np.isnan(ordered.iloc[-1]))
    assert report.iloc[-1]["feature"] == FEATURE_NAMES[9]


def test_correlation_matrix_is_square_and_in_registry_order():
    corr = qa.correlation_matrix(_fake_matrix())
    assert list(corr.columns) == list(FEATURE_NAMES)
    assert list(corr.index) == list(FEATURE_NAMES)
    diagonal = np.diag(corr.to_numpy(dtype=float))
    finite = diagonal[np.isfinite(diagonal)]
    assert np.allclose(finite, 1.0)


def test_quality_artifacts_are_all_written(tmp_path: Any):
    written = qa.write_quality_artifacts(_fake_matrix(), tmp_path)
    for key in ("FE-04", "FE-10", "variance", "outliers", "domain_shift", "policy"):
        assert written[key].is_file(), key
        assert written[key].stat().st_size > 0, key
    assert written["FE-04"].name == qa.QA_ARTIFACTS["FE-04"]
    assert written["FE-10"].name == qa.QA_ARTIFACTS["FE-10"]


# ---------------------------------------------------------------------------
# the gate, on the real matrix
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_fe04_accounts_for_every_nonfinite_value_in_the_corpus(fe03: Any):
    """T41.7, second clause -- verbatim."""
    report = qa.missing_value_report(fe03)
    values = fe03[list(FEATURE_NAMES)].to_numpy(dtype=float)

    total_nonfinite = int((~np.isfinite(values)).sum())
    assert int(report["n_nonfinite"].sum()) == total_nonfinite
    assert int(report["n_nan"].sum()) == int(np.isnan(values).sum())
    assert int(report["n_inf"].sum()) == int(np.isinf(values).sum())

    # Every feature with a non-finite value names at least one record.
    offenders = report[report["n_nonfinite"] > 0]
    for row in offenders.itertuples(index=False):
        assert row.record_uids.strip(), row.feature + " has NaN but names no record"


@pytest.mark.needs_data
def test_the_named_records_really_hold_the_nonfinite_values(fe03: Any):
    """A named record that is actually finite would be a fabricated attribution."""
    report = qa.missing_value_report(fe03)
    indexed = fe03.set_index("record_uid")

    for row in report[report["n_nonfinite"] > 0].itertuples(index=False):
        for uid in [part for part in row.record_uids.split("; ") if part]:
            value = float(indexed.loc[uid, row.feature])
            assert not np.isfinite(value), (
                uid + " is named for " + row.feature + " but holds " + str(value)
            )


@pytest.mark.needs_data
def test_every_nonfinite_value_is_explained_by_a_flag(fe03: Any):
    """The base.py contract, enforced across the whole corpus this time.

    NaN means "this could not be computed" and is never left unexplained: the
    record either records a failed family or carries a flag naming the measure.
    """
    values = fe03[list(FEATURE_NAMES)].to_numpy(dtype=float)
    bad_rows = np.where(~np.isfinite(values).all(axis=1))[0]

    unexplained = []
    for index in bad_rows:
        row = fe03.iloc[index]
        if str(row["failed_families"]).strip() or str(row["flags"]).strip():
            continue
        unexplained.append(str(row["record_uid"]))
    assert not unexplained, (
        str(len(unexplained))
        + " records hold an unexplained non-finite value, e.g. "
        + ", ".join(unexplained[:5])
    )


@pytest.mark.needs_data
def test_variance_and_outlier_reports_cover_the_real_matrix(fe03: Any):
    variance = qa.variance_report(fe03)
    outliers = qa.outlier_report(fe03)
    assert list(variance["feature"]) == list(FEATURE_NAMES)
    assert list(outliers["feature"]) == list(FEATURE_NAMES)
    assert set(variance["verdict"]) <= {
        "ok",
        "constant",
        "near_zero_variance",
        "all_nonfinite",
    }


@pytest.mark.needs_data
def test_domain_shift_between_physionet_and_circor_is_measured(fe03: Any):
    """EXP-D1's premise, quantified before any model runs."""
    report = qa.domain_shift_report(fe03)
    smd = report["abs_smd_D1_vs_D4"].dropna()
    assert not smd.empty
    assert float(smd.max()) > 0.0


@pytest.mark.slow
@pytest.mark.needs_data
def test_reextracting_50_random_records_is_bit_identical(fe03: Any):
    """T41.7, first clause -- verbatim, and exact."""
    result = qa.reproducibility_check(fe03, REPRO_SAMPLE, seed=42)

    assert result["n_checked"] == REPRO_SAMPLE
    assert result["n_values"] == REPRO_SAMPLE * len(FEATURE_NAMES)
    assert result["identical"], (
        str(result["n_mismatches"])
        + " of "
        + str(result["n_values"])
        + " values differ, e.g. "
        + str(result["mismatches"][:3])
    )


@pytest.mark.needs_data
def test_fe04_and_fe10_exist_on_disk_and_are_registered():
    from src.utils.evidence import read_evidence

    directory = mx.matrix_path().parent
    if not mx.matrix_path().is_file():
        pytest.skip("FE-03 not built; run scripts/03_feature_reports.py")
    for filename in qa.QA_ARTIFACTS.values():
        path = directory / filename
        assert path.is_file(), "missing " + str(path)
        assert path.stat().st_size > 0

    rows = {row["evidence_id"]: row for row in read_evidence()}
    for evidence_id in qa.QA_ARTIFACTS:
        assert evidence_id in rows, evidence_id + " not registered"
        assert rows[evidence_id]["status"] == "ok"

    assert (directory / qa.EXTRA_ARTIFACTS["correlation_csv"]).is_file()
