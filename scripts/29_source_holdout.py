"""EXP-F3 -- leave-one-sub-collection-out on the binary track. Emits T-S5.

The deployment question: does the shipping binary model still work on audio from
a recording setup it has never seen? The standard DA-07 protocol cannot answer
it -- every one of its 25 folds trains on all six PhysioNet sub-collections, so
a model that has learned the six acquisition signatures is never tested without
them. This holds out a whole collection at a time.

    python scripts/29_source_holdout.py
    python scripts/29_source_holdout.py --models M1 M6

Six folds, so it costs about a quarter of an hour rather than the hours a
25-fold run takes. Read the per-fold table, not the mean: the collections are
wildly unequal and two of the six folds are noisy in opposite directions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/29_source_holdout.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("source_holdout")

COMMAND = "python scripts/29_source_holdout.py"
SECTION = "outputs/09_ablation"
RUN_DIR = "EXP-F3"

SOURCES = (
    "outputs/03_features/all_features_matrix.parquet",
    "outputs/01_dataset_audit/metadata_master.csv",
    "outputs/06_binary_results/EXP-A1/aggregate_metrics.csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="29_source_holdout")
    parser.add_argument("--models", nargs="+", default=None, metavar="ID")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def build_ts5(comparison, per_fold, baseline, command=""):
    """T-S5 -- pooled CV against leave-one-source-out, per model."""
    from src.reporting.tables import Column, TableSpec, build_table

    columns = [
        Column("model_id", "Model"),
        Column("n_folds_holdout", "Folds", kind="count"),
    ]
    for metric in ("sensitivity", "balanced_accuracy", "specificity", "roc_auc", "f1"):
        label = metric.replace("_", " ")
        for suffix, header in (
            ("_pooled", label + " (pooled 25-fold)"),
            ("_holdout", label + " (leave-one-source-out)"),
            ("_holdout_sd", label + " SD across sources"),
            ("_drop", label + " drop"),
        ):
            if metric + suffix in comparison.columns:
                columns.append(Column(metric + suffix, header, kind="metric"))

    baseline_sens = float(baseline["sensitivity"].mean())
    baseline_ba = float(baseline["balanced_accuracy"].mean())

    spec = TableSpec(
        table_id="T-S5",
        title="Leave-One-Source-Out Generalization",
        caption=(
            "The same models, features and seed evaluated two ways. 'Pooled "
            "25-fold' is the DA-07 protocol, where every training fold contains "
            "records from all six PhysioNet sub-collections. 'Leave-one-source-"
            "out' holds out an entire sub-collection and trains on the other "
            "five, six times, so the model is tested on a recording setup it has "
            "never seen. The drop is the cost of that. Subjects never span "
            "sub-collections, so both protocols are subject-disjoint."
        ),
        sources=SOURCES,
        columns=tuple(columns),
        exp_id="EXP-F3",
        objective="O4 (robustness), deployment realism",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects, 6 sub-collections)",
        notes=(
            "A source-only predictor scored on these same six folds reaches "
            "sensitivity " + format(baseline_sens, ".4f") + " and balanced "
            "accuracy " + format(baseline_ba, ".4f") + ". Read every model "
            "number against that, not against 0.5.",
            "**The mean over six folds hides a great deal.** training-e alone is "
            "2,141 of the 3,240 records and is 8.5% abnormal; training-c is 31 "
            "records and 77.4% abnormal. Holding out training-e leaves a "
            "training set with a completely different class balance, and holding "
            "out training-c gives a 31-record test fold. The per-fold table is "
            "the deliverable.",
            "A drop here is evidence that the model depends on the acquisition "
            "signature. It is NOT proof: holding out a collection also changes "
            "the class balance and the patient population, the same confounds "
            "EXP-D1 documents. The gap between a model and the source-only "
            "baseline on the same fold is the part neither explains.",
            "This experiment exists because Phase 76 found that a no-audio "
            "source-only predictor reaches balanced accuracy 0.719 on the pooled "
            "binary task. See Docs/note.md, 2026-09-10.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return build_table(spec, comparison)


def main(argv: list[str] | None = None) -> int:
    from src.evaluation import source_holdout as sh
    from src.reporting.tables import write_table
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    section = Path(args.out_dir) if args.out_dir else Path(SECTION)
    root = section / RUN_DIR
    root.mkdir(parents=True, exist_ok=True)

    run = start_run("source_holdout")
    try:
        result = sh.run_holdout(models=tuple(args.models) if args.models else None)
        per_fold = result["per_fold"]
        baseline = result["baseline"]
        comparison = sh.compare_with_pooled(per_fold)

        save_csv(per_fold, root / "per_fold_metrics.csv")
        save_csv(baseline, root / "source_only_baseline.csv")
        save_csv(comparison, root / "pooled_vs_holdout.csv")
        save_json(
            {
                "exp_id": sh.EXP_ID,
                "task": sh.TASK,
                "scheme": sh.SCHEME,
                "n_folds": result["n_folds"],
                "question": (
                    "Does the binary model still work on audio from a recording "
                    "setup it has never seen?"
                ),
                "command": COMMAND,
            },
            root / "run_manifest.json",
        )

        table = build_ts5(comparison, per_fold, baseline, COMMAND)
        written = write_table(table, section)
        log.info("T-S5 -> %s", written["csv"].name)

        log.info("--- pooled 25-fold vs leave-one-source-out ---")
        for row in comparison.to_dict("records"):
            log.info(
                "%-3s sensitivity %.4f -> %.4f (drop %+.4f)   balanced %.4f -> %.4f (drop %+.4f)",
                row["model_id"],
                row.get("sensitivity_pooled", float("nan")),
                row["sensitivity_holdout"],
                -row.get("sensitivity_drop", float("nan")),
                row.get("balanced_accuracy_pooled", float("nan")),
                row["balanced_accuracy_holdout"],
                -row.get("balanced_accuracy_drop", float("nan")),
            )
        log.info(
            "source-only baseline on the same folds: sensitivity %.4f, balanced %.4f",
            float(baseline["sensitivity"].mean()),
            float(baseline["balanced_accuracy"].mean()),
        )
        run.set("holdout_models", len(comparison))
    except BaseException:
        run.finish("failed")
        raise

    run.finish("ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
