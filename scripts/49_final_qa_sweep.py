"""The final QA sweep across all six areas (Phase 124, T124.1-T124.7).

Re-verifies dataset integrity, split safety, feature completeness, model
discipline, metric coverage, search safety and the dashboard audit -- each by
recomputing from a committed file, never by trusting an earlier report.

Exits nonzero if any area fails, which is T124.7's gate.

Usage
-----
    python scripts/49_final_qa_sweep.py
    python scripts/49_final_qa_sweep.py --area split --area search
    python scripts/49_final_qa_sweep.py --no-write      # print, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/49_final_qa_sweep.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.reporting.final_qa import AREAS

    parser = argparse.ArgumentParser(
        prog="49_final_qa_sweep",
        description="Re-verify the six QA areas against the committed files.",
    )
    parser.add_argument(
        "--area",
        action="append",
        choices=[area for area, _ in AREAS],
        help="run only this area; repeatable (default: all six)",
    )
    parser.add_argument("--no-write", action="store_true", help="print only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.reporting.final_qa import AREAS, run_sweep, write_report
    from src.utils.run_manifest import start_run

    run = start_run("final_qa_sweep")
    try:
        report = run_sweep(args.area)
    except Exception:
        run.finish(status="failed")
        raise

    for area, description in AREAS:
        if args.area and area not in args.area:
            continue
        print(f"\n{area:10s} {report.area_status(area):16s} {description}")
        for check in report.for_area(area):
            print(f"  {check.status:15s} {check.check_id:18s} {check.title}")
            print(f"  {'':15s} {'':18s} {check.detail}")

    counts = report.to_dict()["counts"]
    print(
        f"\n{counts['pass']} passed, {counts['fail']} failed, "
        f"{counts['skipped']} not verifiable here, {counts['not_applicable']} not applicable"
    )

    if not args.no_write:
        json_path, md_path = write_report(report)
        print(f"wrote {json_path.name} and {md_path.name}")

    run.finish(status="ok" if report.ok else "failed")

    if not report.ok:
        print("\nFAILED: the QA sweep found " + str(len(report.failures)) + " failing check(s)")
        return 1
    print("\nQA sweep green across all six areas")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
