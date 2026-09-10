"""Generate the numbered result tables T08-T15 and T29-T30 (Phases 87 and 89).

T16-T28 are produced by the analysis scripts that own them (21-24, 26-27 and
30-35); this script reports whether each exists and is current, and does not
rebuild them -- rebuilding would re-run analyses whose numbers are already the
evidence. Each table is written beside the data it summarises; see
``src.reporting.result_tables.LOCATIONS``.

Usage
-----
    python scripts/37_result_tables.py
    python scripts/37_result_tables.py --tables T08 T11 --formats csv md
    python scripts/37_result_tables.py --check
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/37_result_tables.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.tables import WRITERS
from src.utils.logging_setup import get_logger

log = get_logger("result_tables")

#: The tables this script builds. T16-T28 belong to the analysis scripts.
BUILT_HERE: tuple[str, ...] = (
    "T08",
    "T09",
    "T10",
    "T11",
    "T12",
    "T13",
    "T14",
    "T15",
    "T29",
    "T30",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="37_result_tables",
        description="Generate T08-T15 and T29-T30 in CSV, Markdown, DOCX and LaTeX.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(BUILT_HERE),
        metavar="ID",
        help="which tables to build (default: every table this script owns)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(WRITERS),
        metavar="FMT",
        choices=list(WRITERS),
    )
    parser.add_argument("--out-dir", default=None, help="write every table here instead")
    parser.add_argument("--evidence-index", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report which of T08-T30 exist and whether their sources are current",
    )
    return parser.parse_args(argv)


def _builders(table_ids: tuple[str, ...], command: str) -> dict[str, Any]:
    from src.reporting.result_tables import CORE_TABLE_IDS, build_core_tables

    built: dict[str, Any] = {}
    core = tuple(t for t in table_ids if t in CORE_TABLE_IDS)
    if core:
        built.update(build_core_tables(core, command=command))
    closing = tuple(t for t in table_ids if t in ("T29", "T30"))
    if closing:
        # Imported by name: T29/T30 are Phase 89's, and this script also has to
        # run (and type-check) in a checkout that predates them.
        module = importlib.import_module("src.reporting.conclusion_tables")
        built.update(module.build_conclusion_tables(closing, command=command))
    return built


def main(argv: list[str] | None = None) -> int:
    from src.reporting.result_tables import RESULT_TABLE_IDS, audit_table, location_of
    from src.reporting.tables import write_table
    from src.utils.run_manifest import start_run

    args = parse_args(argv)

    if args.check:
        failing = 0
        print()
        for table_id in RESULT_TABLE_IDS:
            problems = audit_table(table_id)
            failing += bool(problems)
            print(f"{table_id:<5} " + ("ok" if not problems else problems[0]))
            for problem in problems[1:]:
                print("      " + problem)
        return 1 if failing else 0

    unknown = [t for t in args.tables if t not in BUILT_HERE]
    if unknown:
        log.error("%s are built by their own analysis scripts, not here", ", ".join(unknown))
        return 2

    command = "python scripts/37_result_tables.py --tables " + " ".join(args.tables)
    run = start_run("result_tables")
    run.set("tables", list(args.tables))
    run.set("formats", list(args.formats))

    built = _builders(tuple(args.tables), command)

    rows: list[tuple[str, str, int, str]] = []
    for table_id, table in built.items():
        target = Path(args.out_dir) if args.out_dir else location_of(table_id)[0]
        written = write_table(
            table,
            target,
            formats=tuple(args.formats),
            evidence_index=args.evidence_index,
        )
        for path in written.values():
            run.record_artifact(path)
        rows.append((table_id, table.spec.title, len(table.frame), str(target).replace("\\", "/")))

    print()
    print(f"{'ID':<5} {'Table':<48} {'Rows':>5}  Written to")
    print("-" * 110)
    for table_id, title, n_rows, where in rows:
        print(f"{table_id:<5} {title:<48} {n_rows:>5}  {where}")
    print()
    print("formats: " + ", ".join(args.formats) + "  (+ .meta.json provenance)")

    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
