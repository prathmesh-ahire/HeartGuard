"""Write the Q1 / IEEE paper asset pack to outputs/Q1_PAPER_ASSETS/ (Phase 103).

Nine tables, five graphs, two figures, three algorithms and the results
narrative, every number selected from a generated result file. Then re-derive
every Q1 CSV from its sources and report any mismatch.

Usage
-----
    python scripts/43_q1_assets.py
    python scripts/43_q1_assets.py --out-dir <scratch>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/43_q1_assets.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.q1_pack import verify_q1_pack, write_q1_pack
from src.utils.logging_setup import get_logger

log = get_logger("q1_assets")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="43_q1_assets", description="Write the Q1 asset pack.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    run = start_run("q1_assets")
    written = write_q1_pack(args.out_dir, evidence_index=args.evidence_index)
    for paths in written.values():
        for path in paths:
            run.record_artifact(path)

    problems = verify_q1_pack(args.out_dir)
    failed = {asset: issues for asset, issues in problems.items() if issues}
    for asset in sorted(written):
        print(
            f"{asset:<14} {len(written[asset])} file(s)  "
            + ("OK" if asset not in failed else "MISMATCH")
        )
    for asset, issues in failed.items():
        for issue in issues[:10]:
            print("  " + asset + ": " + issue)
    run.finish(status="ok" if not failed else "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
