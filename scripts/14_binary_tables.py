"""Emit T08 and T10 from a completed binary run (T64.6).

Reads ``per_fold_metrics.csv`` and writes the comparison and fold-wise tables
beside it. It computes nothing the run did not already measure -- every value is
an aggregate of the 25 rows the experiment produced, and the per-fold values are
kept rather than replaced.

Usage
-----
    python scripts/14_binary_tables.py --exp EXP-A1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/14_binary_tables.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("binary_tables")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="14_binary_tables",
        description="Emit T08 (model comparison) and T10 (fold-wise results).",
    )
    parser.add_argument("--exp", default="EXP-A1", metavar="EXP-ID")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.reporting.experiment_report import T08_FILENAME, write_binary_tables
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    command = "python scripts/14_binary_tables.py --exp " + args.exp

    run = start_run("binary_tables_" + args.exp)
    run.set("exp", args.exp)
    written = write_binary_tables(args.exp, out_dir=args.out_dir, command=command)
    for path in written.values():
        run.record_artifact(path)

    table = pd.read_csv(written[T08_FILENAME])
    columns = [
        column
        for column in (
            "rank", "model_id", "model_name", "n_folds",
            "sensitivity", "specificity", "balanced_accuracy", "f1", "roc_auc", "accuracy",
        )
        if column in table.columns
    ]
    print()
    print(table[columns].to_string(index=False))
    print()
    print("ranked by: " + str(table["ranked_by"].iloc[0]))
    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
