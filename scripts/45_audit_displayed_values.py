"""Audit the BUILT dashboard against the files that produced it (T119.3, T119.5).

Runs after every `npm run build` (the `postbuild` hook) and fails the build on
any finding. See `src/reporting/display_audit.py` for the four checks.

`todo.md` places the guard rail at `scripts/07_`; the numbering drifted as for
scripts 16 and 17, so the displayed-value audit lands at `45_`.

Usage
-----
    python scripts/45_audit_displayed_values.py            # audit and stamp
    python scripts/45_audit_displayed_values.py --gate     # T119.4: refuse unless this build passed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/45_audit_displayed_values.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="45_audit_displayed_values",
        description="Diff every value the built dashboard renders against its source files.",
    )
    parser.add_argument("--out-dir", default=None, help="the static export (default frontend/out)")
    parser.add_argument("--generated", default=None, help="frontend/lib/generated")
    parser.add_argument("--report", default=None, help="where to write the stamp")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="do not audit; exit 1 unless the site on disk is the one that passed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from src.reporting.display_audit import (
        AuditGateError,
        require_passed_audit,
        run_audit,
        write_report,
    )

    if args.gate:
        try:
            recorded = require_passed_audit(out_dir=args.out_dir, report_path=args.report)
        except AuditGateError as error:
            print("SCREENSHOT GATE CLOSED: " + str(error))
            return 1
        print(
            "screenshot gate open: site "
            + str(recorded["site_digest"])[:12]
            + " passed at "
            + str(recorded["audited_utc"])
        )
        return 0

    report = run_audit(out_dir=args.out_dir, generated_dir=args.generated)
    path = write_report(report, args.report)
    print("displayed-value audit -- " + str(report.checked.get("pages", 0)) + " pages")
    for key, value in report.checked.items():
        print(f"  checked {key:<24} {value:>8,}")
    for check, findings in report.findings.items():
        state = "ok " if not findings else "FAIL"
        print(f"  {state} {check:<24} {len(findings):>4} finding(s)")
        for finding in findings[:10]:
            print("       " + finding)
    print("report: " + path.as_posix())
    if not report.passed:
        print(
            "\nThe dashboard renders a value its sources do not support. "
            "Nothing may be screenshotted."
        )
        return 1
    print("audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
