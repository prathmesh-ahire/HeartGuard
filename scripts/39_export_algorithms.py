"""Export the ALG pseudocode deliverables from the code they describe (Phases 98-99).

Each algorithm is written to ``outputs/14_algorithms/`` as plain text and DOCX,
with its parameters read from the configs and constants the implementation runs
with and every step citing the ``path:line`` that performs it. See
``src/reporting/algorithms.py`` for why nothing in them is typed twice.

Usage
-----
    python scripts/39_export_algorithms.py
    python scripts/39_export_algorithms.py --ids ALG-03 ALG-08
    python scripts/39_export_algorithms.py --expect 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/39_export_algorithms.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.algorithms import ALGORITHMS, algorithms_dir, write_algorithms
from src.utils.logging_setup import get_logger

log = get_logger("export_algorithms")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="39_export_algorithms",
        description="Write the ALG pseudocode as TXT and DOCX from the implementation.",
    )
    parser.add_argument("--ids", nargs="+", default=None, metavar="ALG")
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="fail unless exactly this many algorithms are written",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    declared = [algorithm.alg_id for algorithm in ALGORITHMS]
    wanted = args.ids or declared
    unknown = [alg_id for alg_id in wanted if alg_id not in declared]
    if unknown:
        print("unknown algorithm id(s): " + ", ".join(unknown), file=sys.stderr)
        return 2

    run = start_run("export_algorithms")
    run.set("algorithms", wanted)
    written = write_algorithms(wanted, args.out_dir, evidence_index=args.evidence_index)
    for paths in written.values():
        for path in paths.values():
            run.record_artifact(path)

    print()
    print(f"{'ID':<7} {'Title':<52} Files")
    print("-" * 80)
    titles = {algorithm.alg_id: algorithm.title for algorithm in ALGORITHMS}
    for alg_id, paths in written.items():
        print(
            f"{alg_id:<7} {titles[alg_id][:52]:<52} "
            + ", ".join(sorted(p.suffix for p in paths.values()))
        )
    print()
    print("wrote " + str(len(written)) + " of " + str(len(declared)) + " declared algorithms")
    print("into " + str(algorithms_dir(args.out_dir)).replace("\\", "/"))

    if args.expect is not None and len(written) != args.expect:
        print(
            "FAILED: expected " + str(args.expect) + " algorithms, wrote " + str(len(written)),
            file=sys.stderr,
        )
        run.finish(status="failed")
        return 1
    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
