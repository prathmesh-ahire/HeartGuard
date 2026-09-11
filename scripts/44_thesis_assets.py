"""Assemble outputs/THESIS_ASSETS/ by chapter (Phase 104).

Copies the 30 tables, 35 graphs, 20 diagrams, 20 algorithms, the equations
reference and the literature review into Ch1-Ch6, writes the reproducibility
appendix, and verifies the counts and every copy's digest.

Run ``python scripts/42_evidence_index.py`` first (the appendix reads the
manifest's final block), and again afterwards so the chapters are indexed.

Usage
-----
    python scripts/44_thesis_assets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/44_thesis_assets.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.thesis_pack import verify_thesis_pack, write_thesis_pack
from src.utils.logging_setup import get_logger

log = get_logger("thesis_assets")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="44_thesis_assets", description="Assemble the thesis asset pack."
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    run = start_run("thesis_assets")
    result = write_thesis_pack(args.out_dir, evidence_index=args.evidence_index)
    run.record_artifact(result["manifest"])
    problems = verify_thesis_pack(args.out_dir)
    print("files: " + str(result["files"]) + "  assets: " + str(result["assets"]))
    for problem in problems:
        print("  PROBLEM " + problem)
    run.finish(status="ok" if not problems else "failed")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
