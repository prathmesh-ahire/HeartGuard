"""Reproduce the whole project with one command (T122.1, T122.2).

Runs every stage declared in ``src/pipeline/run_all.py`` in dependency order --
dataset audit, preprocessing, feature extraction, models, search, every
experiment, every analysis, every table, figure, algorithm and asset pack, the
evidence index, then the frontend chain (export, ``npm ci``, ``npm run build``,
the hard-coded-metric guard, the bundle budget and the displayed-value audit)
and the thirteen gated dashboard screenshots.

**Read this before starting a full run.** A complete run from an empty
``outputs/`` is measured at roughly **67 hours of CPU** on this machine (no GPU;
see CLAUDE.md), of which EXP-A2 alone is ~23.6 h and EXP-C1 ~16.4 h.
``--estimate`` prints the measured breakdown from the committed run manifest.
``--smoke`` runs the same chain on the reduced paths the scripts already declare
and finishes in minutes; CLAUDE.md requires it before any long run.

``--resume`` skips a stage whose declared outputs already exist, so an
interrupted run continues rather than restarting.

Usage
-----
    python scripts/00_run_everything.py --list
    python scripts/00_run_everything.py --estimate
    python scripts/00_run_everything.py --smoke
    python scripts/00_run_everything.py --resume
    python scripts/00_run_everything.py --from exp_a1 --skip-frontend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/00_run_everything.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.run_all import STAGES, estimate_from_manifest, run_stages, select


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="00_run_everything",
        description="Run the full PV-MEPCG pipeline from raw data to final assets.",
    )
    parser.add_argument("--list", action="store_true", help="print the stage list and stop")
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="print the measured wall time of a complete run, from the run manifest",
    )
    parser.add_argument("--smoke", action="store_true", help="run the reduced path of every stage")
    parser.add_argument(
        "--resume", action="store_true", help="skip stages whose declared outputs exist"
    )
    parser.add_argument("--dry-run", action="store_true", help="print what would run and stop")
    parser.add_argument("--from", dest="start", default=None, help="start at this stage id")
    parser.add_argument(
        "--only", action="append", default=None, help="run only this stage id; repeatable"
    )
    parser.add_argument(
        "--skip-frontend", action="store_true", help="stop after the evidence index"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list:
        print(str(len(STAGES)) + " stages:")
        for stage in STAGES:
            mark = "  [frontend]" if stage.frontend else ""
            smoke = "" if stage.smoke_argv is not None else "  (no smoke form)"
            print("  " + stage.stage_id.ljust(24) + stage.title + mark + smoke)
        return 0

    if args.estimate:
        measured = estimate_from_manifest()
        print(
            "measured full-pipeline wall time: "
            + str(measured["total_hours"])
            + " h over "
            + str(measured["n_runs_counted"])
            + " recorded runs ("
            + measured["source"]
            + ")"
        )
        for name, hours in list(measured["by_stage_hours"].items())[:15]:
            print("  " + str(hours).rjust(7) + " h  " + name)
        return 0

    stages = select(only=args.only, start=args.start, skip_frontend=args.skip_frontend)
    payload = run_stages(
        stages, smoke=args.smoke, resume=args.resume, dry_run=args.dry_run
    )

    for entry in payload["stages"]:
        print(
            entry["status"].ljust(22)
            + str(round(entry["seconds"], 1)).rjust(10)
            + " s  "
            + entry["stage_id"]
        )
    print(
        payload["mode"]
        + " run: "
        + str(payload["n_ok"])
        + " ok, "
        + str(payload["n_failed"])
        + " failed, "
        + str(payload["n_skipped"])
        + " skipped, "
        + str(payload["total_hours"])
        + " h total"
    )
    if "manifest" in payload:
        print("manifest: " + payload["manifest"])
    if payload["n_failed"]:
        failed = [item for item in payload["stages"] if item["status"] == "failed"]
        print("FAILED: " + json.dumps(failed[-1]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
