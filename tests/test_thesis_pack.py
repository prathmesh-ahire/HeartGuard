"""Phase 104 gate: the thesis asset pack.

T104.7: every chapter folder is populated and the counts reconcile -- 30
tables, 35 graphs, 20 diagrams, 20 algorithms -- and every copy is still
byte-identical to the file it was copied from.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.reporting import thesis_pack as tp


@pytest.fixture(scope="module")
def manifest_rows() -> list[dict[str, str]]:
    path = tp.thesis_dir() / tp.MANIFEST
    if not path.is_file():
        pytest.skip("thesis pack not generated; run scripts/44_thesis_assets.py")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_t104_7_folders_populated_counts_reconcile_and_copies_identical(
    manifest_rows: list[dict[str, str]],
) -> None:
    assert tp.verify_thesis_pack() == []


@pytest.mark.parametrize(("kind", "count"), sorted(tp.EXPECTED.items()))
def test_t104_7_count(manifest_rows: list[dict[str, str]], kind: str, count: int) -> None:
    assert len({row["asset_id"] for row in manifest_rows if row["kind"] == kind}) == count


def test_every_asset_sits_in_its_chapter(manifest_rows: list[dict[str, str]]) -> None:
    for row in manifest_rows:
        if row["kind"] in ("table", "graph", "diagram", "algorithm"):
            assert row["chapter"] == tp.chapter_of(row["asset_id"]), row["file"]


def test_t104_5_equations_reference_and_t104_6_matrices_are_placed(
    manifest_rows: list[dict[str, str]],
) -> None:
    files = {row["file"] for row in manifest_rows}
    assert "Ch3_Methodology/equations_reference.docx" in files
    assert "Ch4_Implementation/reproducibility_appendix.docx" in files
    for table_id in ("T29", "T30"):
        assert any(
            r["asset_id"] == table_id and r["chapter"] == "Ch6_Conclusion" for r in manifest_rows
        )


def test_reproducibility_appendix_states_the_seed_and_the_disclaimer(
    manifest_rows: list[dict[str, str]],
) -> None:
    from docx import Document

    document = Document(
        str(tp.thesis_dir() / "Ch4_Implementation" / "reproducibility_appendix.docx")
    )
    text = " ".join(paragraph.text for paragraph in document.paragraphs)
    assert "Global seed: 42" in text
    assert "screening" in text and "does not diagnose" in text


def test_chapter_mapping_covers_every_id_exactly_once() -> None:
    ids = (
        ["T" + str(n).zfill(2) for n in range(1, 31)]
        + ["G" + str(n).zfill(2) for n in range(1, 36)]
        + ["F" + str(n).zfill(2) for n in range(1, 21)]
        + ["ALG-" + str(n).zfill(2) for n in range(1, 21)]
    )
    chapters = {tp.chapter_of(asset_id) for asset_id in ids}
    assert chapters <= set(tp.CHAPTERS)
    assert chapters == set(tp.CHAPTERS) - {"Ch2_Literature"}


def test_a_changed_copy_is_detected(tmp_path: Path) -> None:
    rows = [
        {
            "chapter": c,
            "asset_id": "x",
            "kind": "appendix",
            "file": c + "/a.txt",
            "source": "",
            "sha256": "",
        }
        for c in tp.CHAPTERS
    ]
    for row in rows:
        (tmp_path / row["chapter"]).mkdir()
        (tmp_path / row["file"]).write_text("changed", encoding="utf-8")
    tp._write_csv(tmp_path / tp.MANIFEST, rows)
    problems = tp.verify_thesis_pack(tmp_path)
    assert any(p.startswith("changed since packing") for p in problems)
    assert any(p.startswith("table: 0 assets") for p in problems)
