"""The T89.7 gate: T24-T30 exist, and T29 names a real file for all six objectives.

T24-T28 were produced by Phases 79-83 and regenerated in Phase 89 through their
owning scripts' table stages, because they recorded ``D:/Projects/HeartGuard/...``
source paths and unrecorded ``places`` overrides. The audit holds them to the
same definition of a finished table as T08-T23.

T29 and T30 are new. The checks on them are the ones that would catch the
failures they are most exposed to:

* **a paraphrased objective** -- T29's wording must equal the locked text exactly;
* **evidence that is only promised** -- every file T29 names must exist, and an
  objective with outstanding evidence must say ``partial`` and name what is
  missing;
* **a typed conclusion** -- T30's numbers are re-derived here from the files its
  rows name, so a finding that drifted from its source fails;
* **overclaiming** -- T30 may not contain the phrases research rule 7 and the
  EXP-F3 limitation forbid.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.reporting import tables as tb
from src.reporting.result_tables import audit_table, location_of, table_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"

PHASE_89 = ("T24", "T25", "T26", "T27", "T28", "T29", "T30")


def _table(table_id: str) -> pd.DataFrame:
    directory, _ = location_of(table_id)
    if not directory.is_dir():
        pytest.skip(str(directory) + " is not in this checkout")
    return table_frame(table_id)


def _csv(relative: str) -> pd.DataFrame:
    path = OUTPUTS / relative
    if not path.is_file():
        pytest.skip(relative + " is not in this checkout")
    return pd.read_csv(path)


def _row(table: pd.DataFrame, objective: int) -> pd.Series:
    return table[table["objective"] == objective].iloc[0]


# ---------------------------------------------------------------------------
# T24-T30 exist and are finished tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_id", PHASE_89)
def test_table_passes_the_audit(table_id: str) -> None:
    directory, _ = location_of(table_id)
    if not directory.is_dir():
        pytest.skip(str(directory) + " is not in this checkout")
    problems = audit_table(table_id)
    assert not problems, "\n".join(problems)


def test_t28_never_prints_a_p_value_as_zero() -> None:
    """2.7e-14 printed as 0.0000 before the p_value kind existed."""
    table = _table("T28")
    directory, stem = location_of("T28")
    rendered = tb.read_markdown_table(directory / (stem + ".md"))
    for column in ("p_value", "p_holm", "p_bh"):
        position = list(table.columns).index(column)
        cells = [row[position] for row in rendered[1:]]
        assert not any(re.fullmatch(r"0\.0+", cell) for cell in cells), column
    assert (table["p_value"] < 1e-4).any(), "the case this test exists for is absent"


# ---------------------------------------------------------------------------
# T89.7 -- T29 maps all six objectives to real files
# ---------------------------------------------------------------------------


def test_t29_quotes_all_six_locked_objectives_verbatim() -> None:
    from src.reporting.objectives import OBJECTIVES

    table = _table("T29").sort_values("objective")
    assert list(table["objective"]) == [o.number for o in OBJECTIVES]
    for row, objective in zip(table.itertuples(index=False), OBJECTIVES, strict=True):
        assert row.wording == objective.wording, "T29 paraphrases objective " + str(
            objective.number
        )


def test_t29_names_a_real_file_for_every_objective() -> None:
    """The T89.7 clause: a real file path, not a directory and not a promise."""
    table = _table("T29")
    for row in table.itertuples(index=False):
        paths = [p.strip() for p in str(row.evidence_files).split(";") if p.strip()]
        assert paths, "objective " + str(row.objective) + " names no evidence file"
        for relative in paths:
            assert (PROJECT_ROOT / relative).is_file(), (
                "objective " + str(row.objective) + " names a missing file: " + relative
            )
        for module in str(row.modules).split(";"):
            assert (PROJECT_ROOT / module.strip()).is_file(), module


def test_t29_every_evidence_file_is_also_a_recorded_source() -> None:
    """So the staleness check covers what the mapping cites, not just a sample."""
    import json

    directory, stem = location_of("T29")
    meta = json.loads((directory / (stem + ".meta.json")).read_text(encoding="utf-8"))
    recorded = {source["path"] for source in meta["sources"]}
    for row in _table("T29").itertuples(index=False):
        for relative in str(row.evidence_files).split(";"):
            assert relative.strip() in recorded, relative


def test_t29_partial_objectives_say_what_is_missing() -> None:
    table = _table("T29")
    assert set(table["status"]) <= {"evidence produced", "partial"}
    for row in table.itertuples(index=False):
        if row.status == "partial":
            assert str(row.outstanding).strip().lower() not in ("", "none", "nan"), (
                "objective " + str(row.objective) + " is partial with no reason"
            )
        else:
            assert str(row.outstanding).strip().lower() in ("none", "nan")


def test_t29_status_tracks_the_filesystem() -> None:
    """Objective 2 stays partial until a literature table exists, and no longer."""
    has_literature = any(PROJECT_ROOT.glob("outputs/**/*literature*.csv"))
    status = _row(_table("T29"), 2)["status"]
    assert status == ("evidence produced" if has_literature else "partial")


# ---------------------------------------------------------------------------
# T30 -- conclusions re-derived from their sources
# ---------------------------------------------------------------------------


def test_t30_concludes_on_every_objective_from_files_that_exist() -> None:
    table = _table("T30")
    assert sorted(table["objective"]) == [1, 2, 3, 4, 5, 6]
    for row in table.itertuples(index=False):
        for column in ("finding", "verdict", "rule", "limitation"):
            assert str(getattr(row, column)).strip(), (
                "objective " + str(row.objective) + " has no " + column
            )
        for relative in str(row.sources).split(";"):
            assert (PROJECT_ROOT / relative.strip()).is_file(), relative


def test_t30_objective_1_quotes_the_final_models_own_numbers() -> None:
    selection = _csv("06_binary_results/final_model_selection.csv").sort_values("rank")
    final = str(selection.iloc[0]["model_id"])
    aggregate = _csv("06_binary_results/EXP-A2/aggregate_metrics.csv").set_index("model_id")
    loso = _csv("09_ablation/T-S5_leave_one_source_out_generalization.csv").set_index("model_id")
    row = _row(_table("T30"), 1)
    for metric in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc"):
        value = tb.format_value(aggregate.loc[final, metric + "_mean"], "metric")
        assert value in row["finding"], metric + " " + value + " not in objective 1's finding"
    holdout = tb.format_value(loso.loc[final, "roc_auc_holdout"], "metric")
    assert holdout in row["limitation"], "the EXP-F3 held-out AUC is missing"
    assert "within-corpus" in row["verdict"].lower() or "within corpus" in row["verdict"].lower()


def test_t30_objective_3_quotes_the_weight_search_and_the_tests() -> None:
    t09 = _table("T09").set_index("metric")
    identical = str(int(t09.loc["balanced_accuracy", "n_folds_identical"]))
    finding = _row(_table("T30"), 3)["finding"]
    assert "equal M6 on " + identical + " of" in finding
    assert "Holm p=" in finding, "objective 3 states no corrected p-value"


def test_t30_objective_6_quotes_the_multiclass_tables() -> None:
    top = _table("T11").iloc[0]
    finding = _row(_table("T30"), 6)["finding"]
    assert tb.format_value(top["macro_f1_mean"], "metric") in finding
    assert str(top["macro_f1_record_ci"]) in finding


def test_t30_objective_2_verdict_matches_t29_status() -> None:
    """The two tables must not disagree about whether objective 2 is complete."""
    status = _row(_table("T29"), 2)["status"]
    verdict = _row(_table("T30"), 2)["verdict"].lower()
    assert ("partially" in verdict) == (status == "partial")


def test_t30_keeps_to_screening_language() -> None:
    """Rule 7, and the EXP-F3 limitation: no diagnosis, no generalization claim."""
    text = " ".join(_table("T30").astype(str).to_numpy().ravel()).lower()
    for phrase in (
        "diagnoses",
        "diagnostic accuracy",
        "replaces a",
        "clinically validated",
        "generalizes",
        "generalises",
        "outperform",
        "state-of-the-art performance",
    ):
        assert phrase not in text, "T30 contains the claim '" + phrase + "'"
