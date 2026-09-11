"""Export the section 11 equations reference as LaTeX, DOCX and CSV (Phase 100).

Every formula is read from ``src/reporting/equations.py`` and cross-referenced to
the source line that computes it; see ``src/reporting/equations_reference.py``.

Usage
-----
    python scripts/40_export_equations.py
    python scripts/40_export_equations.py --out-dir <scratch>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/40_export_equations.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.equations_reference import reference_rows, write_equations_reference
from src.utils.logging_setup import get_logger

log = get_logger("export_equations")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="40_export_equations",
        description="Write the blueprint section 11 equations as LaTeX, DOCX and CSV.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    run = start_run("export_equations")
    written = write_equations_reference(args.out_dir, evidence_index=args.evidence_index)
    for path in written.values():
        run.record_artifact(path)

    print()
    print(f"{'#':>2}  {'Equation':<24} Implemented at")
    print("-" * 72)
    for row in reference_rows():
        print(f"{row['number']:>2}  {row['name']:<24} {row['implemented_in']}:{row['line']}")
    print()
    for kind, path in written.items():
        print(f"{kind:<5} {str(path).replace(chr(92), '/')}")
    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
