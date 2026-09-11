"""Phase 103 gate: the Q1 / IEEE paper asset pack.

T103.7: every Q1 asset exists, and every number in ``Q1_results_narrative.docx``
traces to a generated file rather than to prose. The same re-derivation is what
MEGA TEST 4's T105.5 relies on, so it is asserted here against the committed
pack and, separately, against a fresh pack written to a scratch directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reporting import q1_pack as q1
from src.reporting.evidence_pack import Q1_ASSETS
from src.utils.evidence import PROJECT_ROOT, read_evidence


def _sources_present() -> bool:
    blocks = [b for t in q1.q1_tables() for b in t.blocks] + [
        b for g in q1.q1_graphs() for b in g.blocks
    ]
    return all((PROJECT_ROOT / b.source).is_file() for b in blocks)


@pytest.fixture(scope="module")
def committed() -> Path:
    target = q1.q1_dir()
    if not (target / "Q1_results_narrative.docx").is_file():
        pytest.skip("Q1 pack not generated; run scripts/43_q1_assets.py")
    return target


def test_t103_7_every_q1_asset_exists_and_is_registered(committed: Path) -> None:
    rows = {row["evidence_id"]: row for row in read_evidence()}
    for asset_id, _ in Q1_ASSETS:
        assert asset_id in rows, asset_id + " is not in the evidence index"
        assert (PROJECT_ROOT / rows[asset_id]["filename"]).is_file(), asset_id
        assert Path(rows[asset_id]["filename"]).parent.name == "Q1_PAPER_ASSETS", asset_id


def test_t103_7_every_q1_number_matches_its_source(committed: Path) -> None:
    problems = {asset: issues for asset, issues in q1.verify_q1_pack().items() if issues}
    assert not problems, problems


def test_t103_7_narrative_has_no_untraced_number(committed: Path) -> None:
    from docx import Document

    document = Document(str(committed / "Q1_results_narrative.docx"))
    text = " ".join(paragraph.text for paragraph in document.paragraphs)
    assert q1.numbers_in(text), "the narrative carries no numbers at all"
    assert "screening" in text and "does not diagnose" in text
    assert "outperform" not in text.replace("not a model that outperformed", "")


def test_q1_tables_name_their_source_on_every_row(committed: Path) -> None:
    import pandas as pd

    for table in q1.q1_tables():
        frame = pd.read_csv(committed / (q1._table_spec(table).slug() + ".csv"))
        assert frame["source_file"].notna().all(), table.table_id
        assert set(frame["source_file"]) <= {block.source for block in table.blocks}


def test_a_fresh_pack_reverifies_in_a_scratch_directory(tmp_path: Path) -> None:
    if not _sources_present():
        pytest.skip("result files behind the Q1 pack are not present")
    q1.write_q1_pack(tmp_path, evidence_index=tmp_path / "evidence_index.csv")
    problems = {asset: issues for asset, issues in q1.verify_q1_pack(tmp_path).items() if issues}
    assert not problems, problems


def test_a_tampered_q1_value_is_caught(tmp_path: Path) -> None:
    if not _sources_present():
        pytest.skip("result files behind the Q1 pack are not present")
    q1.write_q1_pack(tmp_path, evidence_index=tmp_path / "evidence_index.csv")
    table = q1.q1_tables()[3]  # Q1_T04, the binary headline table
    path = tmp_path / (q1._table_spec(table).slug() + ".csv")
    text = path.read_text(encoding="utf-8").splitlines()
    cells = text[1].split(",")
    cells[4] = "0.999"  # sensitivity_mean of the first model
    text[1] = ",".join(cells)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    assert q1.verify_q1_pack(tmp_path)["Q1_T04"], "a changed number went unnoticed"


def test_numbers_in_ignores_identifiers() -> None:
    assert q1.numbers_in(" M1 EXP-A2 Q1_T04 outputs/06_binary T-S5 0.865 3,240 25") == [
        "0.865",
        "3,240",
        "25",
    ]
