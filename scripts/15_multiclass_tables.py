"""Emit T11 (PASCAL A) or T12 (PASCAL B) from a completed multiclass run (T66.5, T67.4).

Reads the experiment's ``per_fold_metrics.csv`` and ``predictions.parquet`` and
writes the headline table plus its per-class companion beside them. It computes
nothing the run did not already measure, with one deliberate exception: the
**record-level bootstrap intervals**, which are recomputed here from the stored
out-of-fold predictions because a per-fold metrics row cannot carry them.

Both PASCAL tracks are small enough that a point estimate alone would mislead --
124 records with 19 in one class, and 461 with 46 -- so every headline metric
carries an interval. See ``src/reporting/multiclass_report`` for why there are
two of them and why the bootstrap runs within a repeat rather than across all of
them.

Usage
-----
    python scripts/15_multiclass_tables.py --exp EXP-B1
    python scripts/15_multiclass_tables.py --exp EXP-B2
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/15_multiclass_tables.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("multiclass_tables")

#: Which table each experiment emits. Declared here rather than inferred from
#: the class count so a future 4-class track cannot silently overwrite T11.
TABLE_FOR_EXPERIMENT = {
    "EXP-B1": "T11_pascal_a_results.csv",
    "EXP-B2": "T12_pascal_b_results.csv",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="15_multiclass_tables",
        description="Emit T11 (PASCAL A) or T12 (PASCAL B) with confidence intervals.",
    )
    parser.add_argument("--exp", default="EXP-B1", metavar="EXP-ID")
    parser.add_argument("--variant", default="", metavar="NAME")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--compare-variant",
        default=None,
        metavar="NAME",
        help=(
            "also emit a paired tuned-vs-<NAME> comparison, e.g. --compare-variant "
            "defaults. Both runs must share the fold map."
        ),
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=2000,
        help="bootstrap draws per repeat for the record-level interval",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.evaluation.experiment import Experiment
    from src.reporting.multiclass_report import (
        build_tuning_comparison,
        write_multiclass_tables,
    )
    from src.reporting.pascal_statements import write_statement
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    filename = TABLE_FOR_EXPERIMENT.get(args.exp)
    if filename is None:
        log.error(
            "%s emits no declared multiclass table; known: %s",
            args.exp,
            ", ".join(sorted(TABLE_FOR_EXPERIMENT)),
        )
        return 2

    experiment = Experiment.load(args.exp)
    if args.variant:
        experiment = dataclasses.replace(experiment, variant=args.variant)
    directory = experiment.output_dir(args.out_dir)

    per_fold_path = directory / "per_fold_metrics.csv"
    predictions_path = directory / "predictions.parquet"
    if not per_fold_path.is_file():
        log.error("no run to report: %s is missing", per_fold_path)
        return 2

    per_fold = pd.read_csv(per_fold_path)
    predictions = pd.read_parquet(predictions_path) if predictions_path.is_file() else None
    if predictions is None:
        log.warning("%s is missing; the record-level intervals will read n/a", predictions_path)

    label_space = dict(experiment.label_space)
    class_names = tuple(sorted(label_space, key=lambda name: label_space[name]))
    labels = tuple(label_space[name] for name in class_names)

    run = start_run("multiclass_tables_" + experiment.run_id)
    run.set("exp", experiment.run_id)
    run.set("class_names", list(class_names))
    run.set("bootstrap_resamples", int(args.resamples))

    # The headline table sits beside the experiment folders in
    # outputs/07_multiclass_results/. A variant writes inside its OWN folder --
    # otherwise the comparison run would overwrite the table it is compared to,
    # which is the trap the EXP-A2 two-pass run fell into (Docs/note.md).
    target = directory if experiment.variant else directory.parent
    written = write_multiclass_tables(
        target,
        filename=filename,
        per_fold=per_fold,
        predictions=predictions,
        labels=labels,
        class_names=class_names,
        n_resamples=int(args.resamples),
    )
    if args.compare_variant:
        other_dir = dataclasses.replace(
            experiment, variant=args.compare_variant
        ).output_dir(args.out_dir)
        other_path = other_dir / "per_fold_metrics.csv"
        if not other_path.is_file():
            log.error(
                "no %s run to compare against: %s is missing",
                args.compare_variant,
                other_path,
            )
            return 2
        comparison = build_tuning_comparison(per_fold, pd.read_csv(other_path))
        comparison_path = target / filename.replace(".csv", "_tuning_comparison.csv")
        comparison.to_csv(comparison_path, index=False)
        written[comparison_path.name] = comparison_path
        print()
        print(comparison.to_string(index=False))

    # T66.6 / T67.5 / T67.6 -- the caveats are part of the deliverable, not a
    # covering note. They are generated from the fold map so their numbers
    # cannot drift from the table they sit beside, and the rule-4 claim is
    # verified before it is written.
    statement = write_statement(
        target,
        task=experiment.task,
        filename=filename.replace(".csv", "_caveats.md"),
        per_fold=per_fold,
        predictions=predictions,
        label_space=label_space,
    )
    written[statement.name] = statement

    for path in written.values():
        run.record_artifact(path)

    headline = pd.read_csv(written[filename])
    columns = [
        column
        for column in (
            "model_id",
            "model_name",
            "n_folds",
            "macro_f1_mean",
            "macro_f1_fold_ci",
            "macro_f1_record_ci",
            "balanced_accuracy_mean",
            "accuracy_mean",
        )
        if column in headline.columns
    ]
    print()
    print(headline[columns].to_string(index=False))
    print()
    print("ranked by: " + str(headline["ranked_by"].iloc[0]))
    print("classes:   " + ", ".join(class_names))
    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
