"""Compliance and claims review across the whole deliverable (Phase 125).

Six checks -- diagnostic language, the disclaimer, implausible perfection, the
locked objectives, documented counts, and whether any published number was
hand-entered. Exits nonzero on any finding, which is T125.7's gate.

Usage
-----
    python scripts/50_compliance_review.py
    python scripts/50_compliance_review.py --check language --check generated
    python scripts/50_compliance_review.py --no-write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/50_compliance_review.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.reporting.compliance import CHECKS

    parser = argparse.ArgumentParser(
        prog="50_compliance_review",
        description="Review every claim the deliverable makes.",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=[name for name, _ in CHECKS],
        help="run only this check; repeatable (default: all six)",
    )
    parser.add_argument("--no-write", action="store_true", help="print only")
    parser.add_argument(
        "--limit", type=int, default=25, help="findings printed per check (default 25)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.reporting.compliance import CHECKS, run_review, write_report
    from src.utils.run_manifest import start_run

    run = start_run("compliance_review")
    try:
        report = run_review(args.check)
    except Exception:
        run.finish(status="failed")
        raise

    for name, description in CHECKS:
        if args.check and name not in args.check:
            continue
        findings = report.for_check(name)
        print(f"\n{name:12s} {len(findings):4d} finding(s)   {description}")
        if report.notes.get(name):
            print(f"{'':12s}      {report.notes[name]}")
        for finding in findings[: args.limit]:
            print(f"  {finding.where}:{finding.line}  {finding.text}")
            print(f"      -> {finding.why}")
        if len(findings) > args.limit:
            print(f"  ... {len(findings) - args.limit} more")

    if not args.no_write:
        json_path, md_path = write_report(report)
        print(f"\nwrote {json_path.name} and {md_path.name}")

    run.finish(status="ok" if report.ok else "failed")

    if not report.ok:
        print(f"\nFAILED: {len(report.findings)} compliance finding(s)")
        return 1
    print("\ncompliance review clean across all six checks")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
