"""Generate T01-T07, the setup tables, in all four formats (Phase 86).

Each table is written beside the data it summarises -- T01/T02/T03 into
``outputs/01_dataset_audit/``, T04 into ``02_preprocessing/``, T05 into
``03_features/``, T06 into ``04_models/``, T07 into ``05_search_optimization/``
-- which is the convention T64.6 and T66.6 already set for T08-T15. Phases 103
and 104 collect them into the paper and thesis asset packs.

Usage
-----
    python scripts/18_setup_tables.py
    python scripts/18_setup_tables.py --tables T01 T05 --formats csv md

``--evidence-index`` exists because this repository's evidence index is a
read-modify-write of one shared CSV. Two pipeline processes registering
artifacts at the same time silently drop each other's rows, so a parallel
session points this at a sidecar that is merged afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/18_setup_tables.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.setup_tables import SETUP_TABLE_IDS
from src.reporting.tables import WRITERS
from src.utils.logging_setup import get_logger

log = get_logger("setup_tables")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="18_setup_tables",
        description="Generate the T01-T07 setup tables in CSV, Markdown, DOCX and LaTeX.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(SETUP_TABLE_IDS),
        metavar="ID",
        help="which tables to build (default: all seven)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(WRITERS),
        metavar="FMT",
        choices=list(WRITERS),
        help="which writers to run (default: all four)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write every table here instead of beside its source data",
    )
    parser.add_argument(
        "--evidence-index",
        default=None,
        help="register artifacts in this index instead of the shared one",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.reporting.setup_tables import build_setup_tables, destination_for
    from src.reporting.tables import write_table
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    command = "python scripts/18_setup_tables.py --tables " + " ".join(args.tables)

    run = start_run("setup_tables")
    run.set("tables", list(args.tables))
    run.set("formats", list(args.formats))

    built = build_setup_tables(tuple(args.tables), command=command)

    rows: list[tuple[str, str, int, str]] = []
    for table_id, table in built.items():
        target = Path(args.out_dir) if args.out_dir else destination_for(table_id)
        written = write_table(
            table,
            target,
            formats=tuple(args.formats),
            evidence_index=args.evidence_index,
        )
        for path in written.values():
            run.record_artifact(path)
        rows.append(
            (
                table_id,
                table.spec.title,
                len(table.frame),
                str(target).replace("\\", "/"),
            )
        )

    print()
    print(f"{'ID':<5} {'Table':<42} {'Rows':>5}  Written to")
    print("-" * 100)
    for table_id, title, n_rows, target in rows:
        print(f"{table_id:<5} {title:<42} {n_rows:>5}  {target}")
    print()
    print("formats: " + ", ".join(args.formats) + "  (+ .meta.json provenance)")

    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
