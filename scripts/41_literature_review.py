"""Write the Objective 2 literature review (LIT-01) and the indicative comparison (LIT-02).

Phase 101. The studies are declared in ``src/reporting/literature.py``; PV-MEPCG's
own numbers in LIT-02 are read from its result files at build time.

Usage
-----
    python scripts/41_literature_review.py
    python scripts/41_literature_review.py --out-dir <scratch>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/41_literature_review.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.literature import GROUPS, STUDIES, write_literature_tables
from src.utils.logging_setup import get_logger

log = get_logger("literature_review")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="41_literature_review",
        description="Write LIT-01 (literature review) and LIT-02 (published comparison).",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    run = start_run("literature_review")
    written = write_literature_tables(args.out_dir, evidence_index=args.evidence_index)
    for paths in written.values():
        for path in paths.values():
            run.record_artifact(path)

    print()
    for key, label in GROUPS.items():
        print(f"{label:<32} {sum(1 for s in STUDIES if s.group == key):>3} row(s)")
    print()
    for table_id, paths in written.items():
        print(table_id + ": " + ", ".join(sorted(p.name for p in paths.values())))
    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
