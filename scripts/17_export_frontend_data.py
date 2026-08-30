"""Export `outputs/` to `frontend/lib/generated/` (Phase 109, T109.1).

The correctness boundary of the dashboard. Every precomputed number the browser
shows is formatted here, in Python, and the frontend imports the result and
nothing else. See `src/reporting/frontend_export.py` for the rules.

`todo.md` and CLAUDE.md name this `scripts/06_export_frontend_data.py`. The `06`
slot was taken by `06_run_feature_selection.py` in Phase 57, so it lands as
`17_` -- the same numbering drift already recorded for the metric guard rail,
which is `16_` rather than `07_`.

Usage
-----
    python scripts/17_export_frontend_data.py
    python scripts/17_export_frontend_data.py --include-results
    python scripts/17_export_frontend_data.py --out-dir build/generated --check

`--check` re-verifies an existing export without rewriting it, for a gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/17_export_frontend_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.frontend_export import GENERATED_FILES
from src.utils.logging_setup import get_logger

log = get_logger("export_frontend_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="17_export_frontend_data",
        description="Emit frontend/lib/generated/ from outputs/. Nothing else may.",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help=(
            "also read 06_binary_results and 07_multiclass_results. Off by "
            "default: a CSV being rewritten by another process reads as a valid "
            "CSV with a truncated row count."
        ),
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--public-dir", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing export parses as strict JSON; write nothing",
    )
    return parser.parse_args(argv)


def _check(out_dir: str | None) -> int:
    from src.reporting.frontend_export import generated_dir, verify_strict_json

    target = generated_dir(out_dir)
    missing = [name for name in GENERATED_FILES if not (target / name).is_file()]
    if missing:
        print("MISSING from " + str(target) + ": " + ", ".join(missing))
        return 1
    for name in GENERATED_FILES:
        if name.endswith(".json"):
            info = verify_strict_json(target / name)
            print(f"  ok  {name:<16} {info['bytes']:>9,} bytes  strict JSON")
        else:
            print(f"  ok  {name:<16} {(target / name).stat().st_size:>9,} bytes")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        return _check(args.out_dir)

    from src.reporting.frontend_export import export_all
    from src.utils.run_manifest import start_run

    command = "python scripts/17_export_frontend_data.py" + (
        " --include-results" if args.include_results else ""
    )
    run = start_run("export_frontend_data")
    run.set("include_results", bool(args.include_results))

    result = export_all(
        out_dir=args.out_dir,
        public_dir=args.public_dir,
        include_results=args.include_results,
        command=command,
    )
    for path in result.written:
        run.record_artifact(path)

    print()
    print("generated: " + str(result.generated).replace("\\", "/"))
    for path in result.written:
        print(f"  {path.name:<16} {path.stat().st_size:>9,} bytes")
    print()
    print(f"tables:  {len(result.tables):>3}  ({', '.join(sorted(result.tables))})")
    print(f"figures: {len(result.figures):>3}  ({', '.join(sorted(result.figures))})")
    print(f"sources: {len(result.manifest['sources']):>3}  fingerprinted")
    for excluded in result.manifest["excluded_dirs"]:
        print("EXCLUDED " + excluded["dir"] + ": " + excluded["reason"].split(".")[0])
    for omission in result.skipped:
        print("NOT INLINED " + omission["artifact"] + ": " + omission["reason"])

    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
