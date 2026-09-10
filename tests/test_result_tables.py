"""The T87.7 gate: T08-T15 exist, render their CSVs, and match the runs behind them.

Two halves, and the second is the one the gate names.

**The audit.** ``result_tables.audit_table`` is the single definition of a
finished table: all four renderings plus provenance, every rendering showing
exactly the CSV, unique headers, and every source repo-relative, present and
current by content digest. If a digest check fails the table is stale --
``python scripts/37_result_tables.py`` regenerates it. That is never a reason to
relax an assertion.

**Reconciliation.** Each table's numbers are re-derived here from the experiment
outputs they came from -- the per-fold files the runs wrote, and for T08 the
aggregate file the experiment runner wrote independently -- and compared at
1e-9. No expected value in this file is typed; every one is read from a CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.reporting.result_tables import CORE_TABLE_IDS, audit_table, location_of, table_frame

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
TOL = 1e-9
BINARY_METRICS = ("sensitivity", "specificity", "balanced_accuracy", "f1", "roc_auc", "accuracy")

#: The working CSVs Phases 64-69 wrote, which scripts 13/14/15/18, the Phase 82
#: statistics and several tests read by name. The deliverables sit beside them
#: under different names, never on top of them.
WORKING_FILES = {
    "T08": "06_binary_results/T08_individual_model_comparison.csv",
    "T09": "06_binary_results/T09_ensemble_weight_comparison.csv",
    "T10": "06_binary_results/T10_fold_wise_results.csv",
    "T11": "07_multiclass_results/T11_pascal_a_results.csv",
    "T12": "07_multiclass_results/T12_pascal_b_results.csv",
    "T13": "08_circor_external_validation/T13_circor_murmur_results.csv",
    "T14": "08_circor_external_validation/T14_circor_outcome_results.csv",
    "T15": "08_circor_external_validation/T15_recording_vs_patient_level.csv",
}


def _csv(relative: str) -> pd.DataFrame:
    path = OUTPUTS / relative
    if not path.is_file():
        pytest.skip(relative + " is not in this checkout")
    return pd.read_csv(path)


def _table(table_id: str) -> pd.DataFrame:
    directory, _ = location_of(table_id)
    if not directory.is_dir():
        pytest.skip(str(directory) + " is not in this checkout")
    return table_frame(table_id)


def _close(actual: Any, expected: Any, label: str) -> None:
    a, e = float(actual), float(expected)
    if np.isnan(e):
        assert np.isnan(a), label + ": expected n/a, table has " + repr(a)
        return
    assert abs(a - e) <= TOL, label + ": table " + repr(a) + " != source " + repr(e)


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_id", CORE_TABLE_IDS)
def test_table_passes_the_audit(table_id: str) -> None:
    directory, _ = location_of(table_id)
    if not directory.is_dir():
        pytest.skip(str(directory) + " is not in this checkout")
    problems = audit_table(table_id)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("table_id", CORE_TABLE_IDS)
def test_the_deliverable_sits_beside_the_working_csv_not_on_it(table_id: str) -> None:
    working = WORKING_FILES[table_id]
    assert len(_csv(working)), working + " is empty"
    _, stem = location_of(table_id)
    assert stem + ".csv" != Path(working).name, table_id + " overwrote its working CSV"


# ---------------------------------------------------------------------------
# T87.7 -- each metric matches the experiment output it came from
# ---------------------------------------------------------------------------


def test_t08_and_t10_carry_both_binary_runs() -> None:
    """The shipped model comes from EXP-A2; the working T08 held EXP-A1 alone."""
    for table_id in ("T08", "T10"):
        assert set(_table(table_id)["run"]) == {"EXP-A1", "EXP-A2"}, table_id


def test_t08_matches_each_runs_independently_written_aggregate() -> None:
    table = _table("T08")
    for run in ("EXP-A1", "EXP-A2"):
        aggregate = _csv("06_binary_results/" + run + "/aggregate_metrics.csv").set_index(
            "model_id"
        )
        block = table[table["run"] == run]
        assert set(block["model_id"]) == set(aggregate.index), run + " model set differs"
        for row in block.itertuples(index=False):
            for metric in BINARY_METRICS:
                _close(
                    getattr(row, metric + "_mean"),
                    aggregate.loc[row.model_id, metric + "_mean"],
                    "T08 " + run + " " + row.model_id + " " + metric,
                )
            for metric in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc"):
                _close(
                    getattr(row, metric + "_sd"),
                    aggregate.loc[row.model_id, metric + "_sd"],
                    "T08 " + run + " " + row.model_id + " " + metric + " SD",
                )


def test_t08_ranks_by_sensitivity_then_balanced_accuracy_within_each_run() -> None:
    table = _table("T08")
    for run, block in table.groupby("run"):
        ordered = block.sort_values("rank")
        keys = list(
            zip(ordered["sensitivity_mean"], ordered["balanced_accuracy_mean"], strict=True)
        )
        assert keys == sorted(keys, reverse=True), str(run) + " is not ranked by the rule"
        assert list(ordered["rank"]) == list(range(1, len(ordered) + 1))


def test_t09_matches_the_per_fold_pairs_and_ablation_a10() -> None:
    table = _table("T09").set_index("metric")
    per_fold = _csv("06_binary_results/EXP-A2/per_fold_metrics.csv")
    a10 = _csv("09_ablation/A10_equal_vs_optimized_weights.csv")
    keys = ["repeat", "fold"]
    m6 = per_fold[per_fold["model_id"] == "M6"].set_index(keys).sort_index()
    m7 = per_fold[per_fold["model_id"] == "M7"].set_index(keys).sort_index()
    for metric in BINARY_METRICS:
        _close(table.loc[metric, "m6_mean"], m6[metric].mean(), "T09 M6 " + metric)
        _close(table.loc[metric, "m7_mean"], m7[metric].mean(), "T09 M7 " + metric)
        delta = (m7[metric] - m6[metric]).to_numpy()
        _close(table.loc[metric, "delta_mean"], delta.mean(), "T09 delta " + metric)
        _close(table.loc[metric, "delta_sd"], np.std(delta, ddof=1), "T09 delta SD " + metric)
        assert int(table.loc[metric, "n_folds_identical"]) == int(np.sum(delta == 0))
        # EXP-F2's A10 is the same comparison, written by a different script.
        _close(table.loc[metric, "delta_mean"], a10[metric + "_delta"].mean(), "A10 " + metric)


def test_t09_p_values_are_the_phase_82_tests() -> None:
    table = _table("T09")
    tests = _csv("12_statistics/paired_fold_tests_interpreted.csv")
    for row in table.itertuples(index=False):
        match = tests[
            (tests["run"] == "EXP-A2")
            & (tests["metric"] == row.metric)
            & (tests["model_a"] == "M6")
            & (tests["model_b"] == "M7")
        ]
        assert len(match) == 1, "no Phase 82 test for M6 vs M7 on " + row.metric
        _close(row.p_holm, match["p_holm"].iloc[0], "T09 Holm p " + row.metric)


def test_t10_matches_the_per_fold_metrics_of_both_runs() -> None:
    table = _table("T10")
    for run in ("EXP-A1", "EXP-A2"):
        per_fold = _csv("06_binary_results/" + run + "/per_fold_metrics.csv")
        block = table[table["run"] == run]
        assert len(block) == per_fold["model_id"].nunique() * len(BINARY_METRICS)
        for row in block.itertuples(index=False):
            values = per_fold.loc[per_fold["model_id"] == row.model_id, row.metric].dropna()
            label = "T10 " + run + " " + row.model_id + " " + row.metric
            _close(row.mean, values.mean(), label + " mean")
            _close(row.sd, values.std(ddof=1), label + " SD")
            _close(row.min, values.min(), label + " min")
            _close(row.max, values.max(), label + " max")
            assert int(row.n_folds_defined) == len(values)


@pytest.mark.parametrize(
    ("table_id", "run", "classes"),
    [
        ("T11", "EXP-B1", ("normal", "murmur", "extrahls", "artifact")),
        ("T12", "EXP-B2", ("normal", "murmur", "extrastole")),
    ],
)
def test_t11_t12_match_the_multiclass_runs(
    table_id: str, run: str, classes: tuple[str, ...]
) -> None:
    table = _table(table_id)
    per_fold = _csv("07_multiclass_results/" + run + "/per_fold_metrics.csv")
    assert set(table["model_id"]) == set(per_fold["model_id"])
    for row in table.itertuples(index=False):
        block = per_fold[per_fold["model_id"] == row.model_id]
        label = table_id + " " + row.model_id
        _close(row.macro_f1_mean, block["macro_f1"].mean(), label + " macro-F1")
        _close(row.balanced_accuracy_mean, block["balanced_accuracy"].mean(), label + " bal acc")
        _close(row.accuracy_mean, block["accuracy"].mean(), label + " accuracy")
        for name in classes:
            _close(
                getattr(row, "recall_" + name),
                block["recall_" + name].mean(),
                label + " recall " + name,
            )


def _by_level(run: str) -> pd.DataFrame:
    return _csv("08_circor_external_validation/" + run + "/per_fold_by_level.csv")


@pytest.mark.parametrize(
    ("table_id", "runs", "metrics"),
    [
        (
            "T13",
            ("EXP-C1-three_class", "EXP-C1-two_class"),
            ("balanced_accuracy", "macro_f1", "recall_Present", "sensitivity", "accuracy"),
        ),
        (
            "T14",
            ("EXP-C2",),
            ("sensitivity", "specificity", "balanced_accuracy", "f1", "roc_auc", "accuracy"),
        ),
    ],
)
def test_t13_t14_match_the_per_fold_scores_at_both_levels(
    table_id: str, runs: tuple[str, ...], metrics: tuple[str, ...]
) -> None:
    table = _table(table_id)
    if "run" not in table.columns:
        table = table.assign(run=runs[0])
    checked = 0
    for run in runs:
        scored = _by_level(run)
        for (model_id, level, rule), block in scored.groupby(["model_id", "level", "rule"]):
            row = table[
                (table["run"] == run)
                & (table["model_id"] == model_id)
                & (table["level"] == level)
                & (table["rule"] == rule)
            ]
            where = " ".join((table_id, run, str(model_id), str(level), str(rule)))
            assert len(row) == 1, where + " is missing from the table"
            for metric in metrics:
                if metric not in block.columns:
                    continue
                values = block[metric].dropna()
                expected = values.mean() if len(values) else float("nan")
                _close(row.iloc[0][metric + "_mean"], expected, where + " " + metric)
                checked += 1
    assert checked, table_id + " reconciled nothing"


def test_t15_deltas_are_patient_minus_recording_from_the_per_fold_scores() -> None:
    table = _table("T15")
    checked = 0
    for run in ("EXP-C1-three_class", "EXP-C1-two_class", "EXP-C2"):
        scored = _by_level(run)
        recording = (
            scored[scored["level"] == "recording"].groupby("model_id").mean(numeric_only=True)
        )
        patient = (
            scored[scored["level"] == "patient"]
            .groupby(["model_id", "rule"])
            .mean(numeric_only=True)
        )
        for row in table[table["run"] == run].itertuples(index=False):
            base = recording.loc[row.model_id, "balanced_accuracy"]
            value = patient.loc[(row.model_id, row.rule), "balanced_accuracy"]
            label = "T15 " + run + " " + row.model_id + " " + row.rule
            _close(row.balanced_accuracy_recording, base, label + " recording")
            _close(row.balanced_accuracy_patient, value, label + " patient")
            _close(row.balanced_accuracy_delta, value - base, label + " delta")
            checked += 1
    assert checked == len(table), "T15 has rows no run explains"
