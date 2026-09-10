"""The T88.7 gate: T16-T23 exist, render their CSVs, and the ablation rows reconcile.

T16-T23 were produced by the analysis phases that own them (70-78). This gate
holds them to the same definition of a finished table as T08-T15 --
``result_tables.audit_table`` -- and then does what T88.7 names: re-derives every
ablation row from the EXP-F1 and EXP-F2 outputs it came from.

**Why the audit matters here in particular.** Before Phase 87 only T01-T07 had a
render-agreement check. T20, T21 and T22 did not render their own CSV under the
rules their provenance recorded (``places`` overrides were never written to the
meta), and T23 recorded ``D:/Projects/HeartGuard/...`` source paths that cannot
resolve on CI. All four were regenerated through their owning scripts' table
stages; see the 2026-09-10 Phase 87 and Phase 88 entries in Docs/note.md.

**Reconciliation, not restatement.** T17 is checked against each arm's OWN run
directory (``EXP-F1-A*/per_fold_metrics.csv``), not against
``feature_ablation_per_fold.csv`` the table was built from -- two files written by
two code paths agreeing is evidence; a table agreeing with its own input is not.
T19 is checked against the EXP-A1/A2 runs its stages read, and against the three
A9/A10 comparison files EXP-F2 wrote separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.reporting.result_tables import audit_table, location_of, table_frame

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
TOL = 1e-9

PHASE_88 = ("T16", "T17", "T18", "T19", "T20", "T21", "T22", "T23")
ABLATION_METRICS = ("sensitivity", "specificity", "f1", "balanced_accuracy", "roc_auc", "accuracy")


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


def _arm(config_id: str) -> pd.DataFrame:
    return _csv("09_ablation/EXP-F1-" + config_id + "/per_fold_metrics.csv")


# ---------------------------------------------------------------------------
# T16-T23 exist and are finished tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_id", PHASE_88)
def test_table_passes_the_audit(table_id: str) -> None:
    directory, _ = location_of(table_id)
    if not directory.is_dir():
        pytest.skip(str(directory) + " is not in this checkout")
    problems = audit_table(table_id)
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# T88.7 -- T17 and T18 reconcile with EXP-F1
# ---------------------------------------------------------------------------


def test_t17_covers_every_declared_arm_and_every_model() -> None:
    from src.evaluation import feature_ablation as fa

    table = _table("T17")
    assert set(table["config_id"]) == set(fa.configuration_ids())
    models = set(_arm("A7")["model_id"])
    for config_id, block in table.groupby("config_id"):
        assert set(block["model_id"]) == models, str(config_id) + " is missing a model"


def test_t17_every_row_is_its_arms_own_run() -> None:
    """Mean, SD and delta vs A7, re-derived from EXP-F1-<arm>/per_fold_metrics.csv."""
    table = _table("T17")
    reference = _arm("A7")
    for config_id, block in table.groupby("config_id"):
        arm = _arm(str(config_id))
        for row in block.itertuples(index=False):
            folds = arm[arm["model_id"] == row.model_id]
            base = reference[reference["model_id"] == row.model_id]
            assert len(folds) == int(row.n_folds), str(config_id) + " " + row.model_id
            for metric in ABLATION_METRICS:
                label = "T17 " + str(config_id) + " " + row.model_id + " " + metric
                _close(getattr(row, metric), folds[metric].mean(), label)
                _close(getattr(row, metric + "_sd"), folds[metric].std(ddof=1), label + " SD")
                _close(
                    getattr(row, metric + "_delta_vs_A7"),
                    folds[metric].mean() - base[metric].mean(),
                    label + " delta",
                )


def test_t17_feature_counts_are_the_arms_declared_widths() -> None:
    from src.evaluation import feature_ablation as fa

    table = _table("T17")
    for config_id, block in table.groupby("config_id"):
        expected = len(fa.configuration_columns(str(config_id)))
        assert set(block["n_features"]) == {expected}, str(config_id)


def test_t18_reconciles_with_the_a7_and_a8_runs_fold_by_fold() -> None:
    table = _table("T18")
    a7, a8 = _arm("A7"), _arm("A8")
    keys = ["repeat", "fold"]
    for row in table.itertuples(index=False):
        selected = a8[a8["model_id"] == row.model_id].set_index(keys).sort_index()
        full = a7[a7["model_id"] == row.model_id].set_index(keys).sort_index()
        assert list(selected.index) == list(full.index), row.model_id + " arms ran different folds"
        for metric in ABLATION_METRICS:
            label = "T18 " + row.model_id + " " + metric
            paired = (selected[metric] - full[metric]).to_numpy()
            _close(getattr(row, metric + "_all"), full[metric].mean(), label + " all")
            _close(getattr(row, metric + "_selected"), selected[metric].mean(), label + " sel")
            _close(getattr(row, metric + "_delta"), paired.mean(), label + " delta")
            _close(getattr(row, metric + "_delta_sd"), np.std(paired, ddof=1), label + " SD")
            wins = int(getattr(row, metric + "_folds_selected_wins"))
            assert wins == int(np.sum(paired > 0)), label + " folds won"


def test_t17_and_t18_agree_where_they_overlap() -> None:
    """A7 and A8 appear in both tables; the two builders must not disagree."""
    t17 = _table("T17").set_index(["config_id", "model_id"])
    for row in _table("T18").itertuples(index=False):
        for metric in ABLATION_METRICS:
            _close(getattr(row, metric + "_all"), t17.loc[("A7", row.model_id), metric], metric)
            _close(
                getattr(row, metric + "_selected"), t17.loc[("A8", row.model_id), metric], metric
            )


# ---------------------------------------------------------------------------
# T88.7 -- T19 reconciles with EXP-F2
# ---------------------------------------------------------------------------


def test_t19_every_stage_is_its_source_runs_per_fold_mean() -> None:
    table = _table("T19")
    for row in table.itertuples(index=False):
        per_fold = _csv("06_binary_results/" + row.source_run + "/per_fold_metrics.csv")
        folds = per_fold[per_fold["model_id"] == row.model_id]
        assert len(folds) == int(row.n_folds), row.stage
        for metric in ABLATION_METRICS:
            label = "T19 " + row.stage + " " + metric
            _close(getattr(row, metric), folds[metric].mean(), label)
            _close(getattr(row, metric + "_sd"), folds[metric].std(ddof=1), label + " SD")


def test_t19_increments_telescope() -> None:
    table = _table("T19").sort_values("stage_order").reset_index(drop=True)
    for metric in ABLATION_METRICS:
        values = table[metric].to_numpy(dtype=float)
        _close(table.loc[0, metric + "_cumulative"], 0.0, "T19 baseline " + metric)
        for position in range(1, len(values)):
            _close(
                table.loc[position, metric + "_incremental"],
                values[position] - values[position - 1],
                "T19 incremental " + metric,
            )
            _close(
                table.loc[position, metric + "_cumulative"],
                values[position] - values[0],
                "T19 cumulative " + metric,
            )


@pytest.mark.parametrize(
    "filename",
    [
        "A9_individual_vs_ensemble_default.csv",
        "A9_individual_vs_ensemble_tuned.csv",
        "A10_equal_vs_optimized_weights.csv",
    ],
)
def test_t19_matches_the_a9_and_a10_comparisons_exp_f2_wrote(filename: str) -> None:
    stages = _table("T19").set_index("stage")
    comparison = _csv("09_ablation/" + filename)
    left, right = comparison["left_stage"].iloc[0], comparison["right_stage"].iloc[0]
    assert {left, right} <= set(stages.index), filename + " names a stage T19 lacks"
    for metric in ABLATION_METRICS:
        _close(
            stages.loc[right, metric] - stages.loc[left, metric],
            comparison[metric + "_delta"].mean(),
            filename + " " + metric,
        )


def test_t19_matches_the_optimization_ablation_per_fold_file() -> None:
    table = _table("T19")
    per_fold = _csv("09_ablation/optimization_ablation_per_fold.csv")
    for row in table.itertuples(index=False):
        folds = per_fold[per_fold["stage"] == row.stage]
        assert len(folds) == int(row.n_folds), row.stage
        for metric in ABLATION_METRICS:
            _close(getattr(row, metric), folds[metric].mean(), "T19 " + row.stage + " " + metric)


# ---------------------------------------------------------------------------
# the regenerated tables still say what they said
# ---------------------------------------------------------------------------


def test_t20_t21_t22_record_the_places_they_render_at() -> None:
    """The defect Phase 88 regenerated them for: overrides the meta did not record."""
    import json

    for table_id in ("T20", "T21", "T22"):
        directory, stem = location_of(table_id)
        meta = json.loads((directory / (stem + ".meta.json")).read_text(encoding="utf-8"))
        assert meta.get("column_places"), table_id + " still records no column places"


def test_t22_renders_an_absent_exclusion_reason_as_n_a_not_blank() -> None:
    from src.reporting import tables as tb

    directory, stem = location_of("T22")
    rendered = tb.read_markdown_table(directory / (stem + ".md"))
    # Rendered columns are the CSV's columns in order, so the CSV names it.
    position = list(table_frame("T22").columns).index("exclusion_reason")
    cells = [row[position] for row in rendered[1:]]
    assert "" not in cells, "T22 renders a missing exclusion reason as a blank"
    assert tb.NA_TEXT in cells, "no row without an exclusion reason -- the check is vacuous"
