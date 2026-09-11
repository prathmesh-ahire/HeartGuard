"""Phase 102 gate: the evidence index, its workbook and the completeness audit.

T102.7 asks two things: every evidence-index row resolves to a real file on
disk, and every mandatory item from both source documents appears in the index.
"Appears" is read strictly -- an item is either present, or it is not produced
and ``missing_outputs_report.txt`` says why. An item that is simply absent is
the failure this gate exists to catch.

A row whose file is gitignored (the feature matrix is) cannot exist on CI; such
rows are exempt from the on-disk check only when git confirms the ignore, so a
fresh clone reports what it lacks without failing on what it was never given.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from src.reporting import evidence_pack as ep
from src.utils.evidence import EVIDENCE_COLUMNS, PROJECT_ROOT, read_evidence


def _gitignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    found = read_evidence()
    if not found:
        pytest.skip("no evidence index to audit")
    return found


@pytest.fixture(scope="module")
def workbook() -> Path:
    path = PROJECT_ROOT / "outputs" / "00_evidence_index" / ep.WORKBOOK_NAME
    if not path.is_file():
        pytest.skip("evidence_index.xlsx not generated; run scripts/42_evidence_index.py")
    return path


# ---------------------------------------------------------------------------
# T102.7 -- the gate
# ---------------------------------------------------------------------------


def test_t102_7_every_evidence_row_resolves_to_a_real_file(rows: list[dict[str, str]]) -> None:
    missing = [row for row in rows if ep.row_status(row)[0] == "failed"]
    unexplained = [
        row["evidence_id"] + " -> " + row["filename"]
        for row in missing
        if not _gitignored(row["filename"])
    ]
    assert not unexplained, "evidence rows with no file on disk: " + ", ".join(unexplained)


def test_t102_7_every_mandatory_item_is_present_or_accounted_for(
    rows: list[dict[str, str]],
) -> None:
    results = ep.check_mandatory(rows)
    unexplained = [
        item["item_id"] + ": " + item["detail"]
        for item in results
        if item["status"] == "failed" and not ep.fails_only_on_ignored_files(item, rows)
    ]
    assert not unexplained, "mandatory items missing with no declared reason:\n" + "\n".join(
        unexplained
    )


def test_t102_6_every_unproduced_item_is_in_the_missing_outputs_report(
    rows: list[dict[str, str]],
) -> None:
    report = ep.missing_report_path().read_text(encoding="utf-8")
    assert report.count(ep.REPORT_BEGIN) == 1 and report.count(ep.REPORT_END) == 1
    block = report.split(ep.REPORT_BEGIN, 1)[1].split(ep.REPORT_END, 1)[0]
    for item in ep.check_mandatory(rows):
        if item["status"] == "present":
            continue
        if item["status"] == "failed" and ep.fails_only_on_ignored_files(item, rows):
            continue  # a fresh clone lacks the file; the report was written where it exists
        assert "] " + item["item_id"] + " -- " in block, (
            item["item_id"] + " is absent from the report"
        )
        assert item["detail"] in block, item["item_id"] + "'s reason is not in the report"


def test_t102_1_workbook_holds_every_row_with_the_spec_fields(
    rows: list[dict[str, str]], workbook: Path
) -> None:
    sheet = ep.read_workbook_rows(workbook, "evidence_index")
    assert [row["evidence_id"] for row in sheet] == [row["evidence_id"] for row in rows]
    assert set(EVIDENCE_COLUMNS) <= set(sheet[0])
    assert {row["status"] for row in sheet} <= {"verified", "generated", "failed"}
    for row in sheet:
        assert row["filename"] and row["command"] is not None


def test_t102_4_workbook_lists_every_mandatory_item(workbook: Path) -> None:
    sheet = ep.read_workbook_rows(workbook, "mandatory_items")
    assert [row["item_id"] for row in sheet] == [item.item_id for item in ep.MANDATORY]
    assert {row["status"] for row in sheet} <= {"present", "not produced", "failed"}


def test_t102_5_manifest_is_finalized_with_environment_seed_and_timing() -> None:
    from src.utils.config import load_config

    path = Path(load_config("paths").require("outputs.run_manifest"))
    # conftest.py redirects the manifest for tests; the gate reads the real one.
    real = PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json"
    data = json.loads((real if real.is_file() else path).read_text(encoding="utf-8"))
    if "final" not in data:
        pytest.skip("run_manifest.json not finalized; run scripts/42_evidence_index.py")
    final = data["final"]
    assert final["seed"]["global_seed"] == 42
    assert final["environment"]["python_version"]
    assert final["packages"]
    assert final["runs"]["total"] == len(data["runs"])
    assert final["stage_timing"]
    assert sum(stage["n_runs"] for stage in final["stage_timing"]) == len(data["runs"])


# ---------------------------------------------------------------------------
# the catalogue itself
# ---------------------------------------------------------------------------


def test_catalogue_ids_are_unique() -> None:
    ids = [item.item_id for item in ep.MANDATORY]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("prefix", "count"),
    [
        ("SPEC-DA-", 9),
        ("SPEC-PP-", 9),
        ("SPEC-FE-", 12),
        (
            "SPEC-T",
            30 + 6 + 3,
        ),  # 30 tables + 6 thesis-chapter items + the repro/matrices/screenshots items
        ("SPEC-G", 35),
        ("SPEC-F", 20 + 12),  # 20 figures + FE-01..FE-12
        ("SPEC-ALG-", 20),
        ("SPEC-SS", 13),
        ("SPEC-Q1_", 20),
        ("BP-TABLE", 20),
        ("BP-G", 20),
        ("BP-ABL-A", 10),
    ],
)
def test_catalogue_counts_match_the_source_documents(prefix: str, count: int) -> None:
    assert sum(1 for item in ep.MANDATORY if item.item_id.startswith(prefix)) == count


def test_every_declared_reason_is_technical_and_every_item_has_an_answer() -> None:
    for item in ep.MANDATORY:
        assert item.evidence_ids or item.paths or item.reason, item.item_id
        if item.reason:
            assert len(item.reason) > 40, item.item_id


# ---------------------------------------------------------------------------
# the mechanics, on scratch files
# ---------------------------------------------------------------------------


def test_row_status_uses_the_spec_vocabulary(tmp_path: Path) -> None:
    artifact = tmp_path / "a.csv"
    artifact.write_text("x\n1\n", encoding="utf-8")
    base = dict.fromkeys(EVIDENCE_COLUMNS, "")
    assert ep.row_status({**base, "filename": str(tmp_path / "nope.csv")})[0] == "failed"
    assert ep.row_status({**base, "filename": str(artifact)})[0] == "generated"
    assert (
        ep.row_status(
            {**base, "filename": str(artifact), "source_data": "outputs/no_such_file_x.csv"}
        )[0]
        == "generated"
    )
    assert (
        ep.row_status(
            {**base, "filename": str(artifact), "source_data": "README.md; feature registry"}
        )[0]
        == "verified"
    )
    assert ep.source_paths("dataset/ (read-only input); feature registry") == ["dataset/"]


def test_missing_report_block_is_rewritten_in_place_and_hand_entries_survive(
    tmp_path: Path,
) -> None:
    report = tmp_path / "missing.txt"
    report.write_text("HEADER\n[2026-08-26] T20.5 -- hand entry\n", encoding="utf-8")
    items = [
        {
            "item_id": "X-1",
            "description": "thing",
            "source_document": "spec",
            "section": "9",
            "status": "not produced",
            "detail": "a technical reason long enough to be one",
        }
    ]
    ep.update_missing_report(items, report)
    ep.update_missing_report(items, report)
    text = report.read_text(encoding="utf-8")
    assert text.startswith("HEADER\n[2026-08-26] T20.5 -- hand entry\n")
    assert text.count(ep.REPORT_BEGIN) == 1 and text.count("X-1 -- thing") == 1


def test_sidecars_merge_without_overwriting_the_main_index(tmp_path: Path) -> None:
    from src.utils.evidence import _write_rows

    def row(evidence_id: str, filename: str) -> dict[str, str]:
        return {
            **dict.fromkeys(EVIDENCE_COLUMNS, ""),
            "evidence_id": evidence_id,
            "filename": filename,
        }

    _write_rows([row("A", "main.csv")], tmp_path / "evidence_index.csv")
    _write_rows([row("A", "side.csv"), row("B", "b.csv")], tmp_path / "evidence_index_window2.csv")
    manifest = {
        "schema": 1,
        "config_snapshots": {"h1": {}},
        "runs": [{"run_id": "r1", "started_utc": "2"}],
    }
    side = {
        "config_snapshots": {"h2": {}},
        "runs": [{"run_id": "r0", "started_utc": "1"}, {"run_id": "r1"}],
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "run_manifest_window2.json").write_text(json.dumps(side), encoding="utf-8")

    report = ep.merge_sidecars(tmp_path)

    merged = read_evidence(tmp_path / "evidence_index.csv")
    assert [(r["evidence_id"], r["filename"]) for r in merged] == [
        ("A", "main.csv"),
        ("B", "b.csv"),
    ]
    data = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert [run["run_id"] for run in data["runs"]] == ["r0", "r1"]
    assert set(data["config_snapshots"]) == {"h1", "h2"}
    assert report["evidence_added"] == ["B"] and report["runs_added"] == 1
    assert not (tmp_path / "evidence_index_window2.csv").exists()


def test_backfill_timestamps_are_file_times_and_never_duplicate(tmp_path: Path) -> None:
    index = tmp_path / "evidence_index.csv"
    added = ep.backfill_unregistered(index)
    assert added == [item.evidence_id for item in ep.BACKFILL]
    assert ep.backfill_unregistered(index) == []
    for row in read_evidence(index):
        artifact = PROJECT_ROOT / row["filename"]
        if artifact.is_file():
            from datetime import UTC, datetime

            expected = datetime.fromtimestamp(os.path.getmtime(artifact), tz=UTC).isoformat()
            assert row["timestamp"] == expected
