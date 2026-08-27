"""Extract the 138 features for one or more datasets (Phase 39).

Reads preprocessed signals from ``cache/preprocessed/`` (Phase 27), runs the six
feature families, and writes one Parquet shard per dataset under
``cache/features/<digest>/``.

Usage
-----
    python scripts/02_extract_features.py --smoke              # 20 records per dataset
    python scripts/02_extract_features.py --dataset D2 D3      # two datasets
    python scripts/02_extract_features.py                      # everything
    python scripts/02_extract_features.py --force              # ignore checkpoints

The run is **resumable**: each chunk of ``extraction.checkpoint_every`` records is
written to a checkpoint the moment it completes, and a restart continues from
there rather than beginning again. This matters -- the full corpus is several
CPU-hours, dominated by sample entropy.

``--smoke`` writes into a separate ``_smoke`` subtree, never where a full run
writes. A 20-row shard sitting where a 3,240-row shard belongs reads exactly like
a complete one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/02_extract_features.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extraction.batch import DATASETS, run_extraction
from src.utils.logging_setup import get_logger
from src.utils.timing import format_duration

log = get_logger("extract")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="02_extract_features",
        description="Extract the locked 138 features into per-dataset Parquet shards.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=[*DATASETS, "all"],
        default=["all"],
        help="datasets to extract (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="extract at most N records per dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        metavar="N",
        help="parallel workers; -1 uses every core (default: -1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="discard existing shards and checkpoints and re-extract",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="fast validation: 20 records per dataset, written to a _smoke subtree",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write extraction_errors.csv somewhere other than outputs/03_features",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress the per-chunk progress bar (for CI and log files)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    datasets = DATASETS if "all" in args.dataset else tuple(args.dataset)

    run = start_run("feature_extraction" + ("_smoke" if args.smoke else ""))
    run.set("extract_datasets", list(datasets))
    run.set("extract_mode", "smoke" if args.smoke else "full")
    run.set("extract_limit", args.limit)
    run.set("extract_workers", args.workers)

    try:
        summary = run_extraction(
            datasets,
            n_jobs=args.workers,
            force=args.force,
            limit=args.limit,
            smoke=args.smoke,
            progress=not args.no_progress,
            out_dir=args.out_dir,
        )
    except BaseException:
        # A run that died is a fact worth recording, not an ambiguous "running".
        run.finish("failed")
        raise

    run.set("extract_records", summary.n_records)
    run.set("extract_failures", summary.n_failed)
    run.set("extract_digest", summary.digest)
    run.set("extract_wall_seconds", round(summary.seconds, 3))
    run.finish()

    _report(summary)
    return 1 if summary.n_failed else 0


def _report(summary: Any) -> None:
    log.info("-" * 62)
    log.info(
        "%s extraction: %d records in %s",
        "SMOKE" if summary.smoke else "FULL",
        summary.n_records,
        format_duration(summary.seconds),
    )
    for dataset, count in summary.datasets.items():
        log.info("  %-3s %5d records -> %s", dataset, count, summary.shards[dataset])
    log.info("  cache digest: %s", summary.digest)
    if summary.n_failed:
        log.warning("  %d family failures -> %s", summary.n_failed, summary.errors_path)
    else:
        log.info("  no failures; empty error report at %s", summary.errors_path)
    log.info("-" * 62)


if __name__ == "__main__":
    raise SystemExit(main())
