"""Capture the thirteen dashboard screenshots (Phase 120).

Three steps, one command:

1. Write ``outputs/15_dashboard_screenshots/capture_plan.json`` from the plan
   declared in ``src/reporting/screenshots.py``.
2. Run ``npm run screenshots``, which is Playwright against the BUILT site in
   ``frontend/out/`` served by the real FastAPI process, with the real inference
   API answering ``/predict``. Its ``globalSetup`` runs the displayed-value
   audit's gate first (T119.4), so an unaudited or rebuilt site aborts the run
   before a browser opens.
3. Caption and index what the browser wrote, and register ``SS-01``..``SS-13``
   in the evidence index (T120.6).

``--finalize-only`` runs step 3 alone, for when the capture was driven by hand.

Usage
-----
    python scripts/47_dashboard_screenshots.py
    python scripts/47_dashboard_screenshots.py --plan-only
    python scripts/47_dashboard_screenshots.py --finalize-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/47_dashboard_screenshots.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.screenshots import CAPTURE_PLAN, finalize, write_plan
from src.utils.logging_setup import get_logger

log = get_logger("dashboard_screenshots")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="47_dashboard_screenshots",
        description="Write the capture plan, run the gated Playwright capture, and "
        "caption, index and register the thirteen screenshots.",
    )
    parser.add_argument("--plan-only", action="store_true", help="write the plan and stop")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="caption, index and register screenshots already on disk",
    )
    parser.add_argument(
        "--grep", default=None, help="pass a -g filter through to Playwright (e.g. SS-08)"
    )
    return parser.parse_args(argv)


def run_capture(grep: str | None = None) -> int:
    """Drive the Playwright capture through the project's own npm script."""
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        log.error("npm is not on PATH; the capture needs Node to drive Playwright")
        return 1
    command = [npm, "run", "screenshots"]
    if grep:
        command += ["--", "-g", grep]
    log.info("capture: %s (cwd=frontend)", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT / "frontend", check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.finalize_only:
        plan = write_plan()
        print("capture plan: " + str(len(CAPTURE_PLAN)) + " shots -> " + str(plan))
        if args.plan_only:
            return 0
        status = run_capture(args.grep)
        if status != 0:
            print("FAILED: the capture run exited " + str(status))
            return status

    result = finalize()
    print("captions: " + str(result.captions))
    print("index: " + str(result.index) + " (" + str(len(result.rows)) + " rows)")
    for row in result.rows:
        print(
            "  " + row["screenshot_id"] + "  " + row["filename"]
            + "  " + str(row["bytes"]) + " B"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
