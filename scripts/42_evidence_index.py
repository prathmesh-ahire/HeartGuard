"""Assemble the evidence index and audit it against both source documents (Phase 102).

Writes ``outputs/00_evidence_index/evidence_index.xlsx``, finalizes
``run_manifest.json`` and regenerates the completeness block of
``outputs/missing_outputs_report.txt``. Re-run after Phases 103 and 104 so their
assets are counted.

Usage
-----
    python scripts/42_evidence_index.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/42_evidence_index.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.evidence_pack import assemble
from src.utils.logging_setup import get_logger

log = get_logger("evidence_index")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="42_evidence_index",
        description="Assemble evidence_index.xlsx and audit every mandatory item.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when any row's file is missing or any mandatory item failed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = assemble()

    merged = result["merged"]
    print(
        "sidecars merged: evidence +"
        + str(len(merged["evidence_added"]))
        + ", runs +"
        + str(merged["runs_added"])
    )
    print("backfilled: " + (", ".join(result["backfilled"]) or "none"))
    print("rows: " + str(result["rows"]) + "  " + str(result["row_status"]))
    print("mandatory items: " + str(result["items"]))
    for item in result["failed_items"]:
        print("  FAILED " + item["item_id"] + ": " + item["detail"])
    for row in result["failed_rows"]:
        print("  MISSING FILE " + row["evidence_id"] + ": " + row["filename"])
    print("workbook: " + str(result["workbook"]))
    print("manifest stages timed: " + str(result["stages"]))
    failed = bool(result["failed_items"] or result["failed_rows"])
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
