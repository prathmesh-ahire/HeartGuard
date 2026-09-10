"""EXP-G1 -- the PhysioNet diagnosis multiclass track (Phase 76).

Supplementary evidence for Objective 6 on a far larger sample than PASCAL A:
665 abnormal PhysioNet recordings over 348 subjects, eight diagnosis classes
merged to five, subject-grouped 5-fold, balanced class weighting, and a
confidence interval on every per-class metric.

    python scripts/28_diagnosis_track.py --stage map      # build + check the fold map
    python scripts/28_diagnosis_track.py                  # map, run, tables

The merge policy is applied by code and written out as
``diagnosis_class_merge_policy.csv`` -- T76.2 asks for it to be explicit, and a
policy that lives only in a docstring is not auditable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/28_diagnosis_track.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("diagnosis_track")

COMMAND = "python scripts/28_diagnosis_track.py"
SECTION = "outputs/07_multiclass_results"
RUN_DIR = "EXP-G1"

SOURCES = (
    "outputs/01_dataset_audit/metadata_master.csv",
    "outputs/01_dataset_audit/diagnosis_split_map.csv",
    "outputs/03_features/all_features_matrix.parquet",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="28_diagnosis_track")
    parser.add_argument("--stage", default="all", choices=("map", "run", "tables", "all"))
    parser.add_argument("--models", nargs="+", default=None, metavar="ID")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--within-source",
        action="store_true",
        help=(
            "the de-confounded variant: training-a only, where every recording "
            "shares one sub-collection so the source shortcut is unavailable"
        ),
    )
    return parser.parse_args(argv)


def build_ts1(summary, per_class, classes, n_records, n_subjects,
              baseline_f1, baseline_ba, beaten, within_source=False, command=""):
    """T-S1 -- the diagnosis track headline, one row per model.

    A **supplementary** id, not a number in the locked T01-T30 thesis series:
    T13 and T14 are CirCor murmur and outcome, and EXP-G1 is declared
    supplementary in configs/experiments.yaml. Same `-S` convention as the
    PP-S1..PP-S3 supporting artifacts.
    """
    from src.evaluation import diagnosis_track as dt
    from src.reporting.tables import Column, TableSpec, build_table

    columns = [
        Column("model_id", "Model"),
        Column("model_name", "Name"),
        Column("n_folds", "Folds", kind="count"),
    ]
    for metric in ("macro_f1", "balanced_accuracy", "macro_recall", "macro_precision",
                   "weighted_f1", "accuracy"):
        label = metric.replace("_", " ")
        for suffix, header in (
            ("_mean", label),
            ("_sd", label + " SD"),
            ("_fold_ci", label + " 95% CI (folds)"),
            ("_record_ci", label + " 95% CI (records)"),
        ):
            if metric + suffix in summary.columns:
                columns.append(Column(metric + suffix, header,
                                      kind="metric" if suffix in ("_mean", "_sd") else None))
    if "n_classes_predicted" in summary.columns:
        columns.append(Column("n_classes_predicted", "Classes predicted", kind="count"))
    if "degenerate" in summary.columns:
        columns.append(Column("degenerate", "Degenerate"))

    spec = TableSpec(
        table_id="T-S3" if within_source else "T-S1",
        title=(
            "PhysioNet Diagnosis Multiclass (within one source)"
            if within_source
            else "PhysioNet Diagnosis Multiclass"
        ),
        caption=(
            "Supplementary multiclass evidence for Objective 6 on "
            + str(n_records) + " abnormal PhysioNet recordings over "
            + str(n_subjects) + " subjects, subject-grouped 5-fold, balanced class "
            "weighting. Eight diagnosis classes were merged to "
            + str(len(classes)) + " by the policy in "
            "diagnosis_class_merge_policy.csv. Two intervals are reported on "
            "every metric because they answer different questions: the fold "
            "interval is a Student-t interval over 5 folds, the record interval "
            "a percentile bootstrap over recordings. Ranked by macro-F1."
        ),
        sources=SOURCES,
        columns=tuple(columns),
        exp_id="EXP-G1",
        objective="O6 (multiclass classification)",
        dataset="D1 PhysioNet 2016, abnormal recordings only",
        notes=(
            (
                "**THE RECORDING SOURCE IS CONSTANT IN THIS TABLE.** Every "
                "record is from " + str(dt.WITHIN_SOURCE_SUBSET) + ", so the "
                "sub-collection shortcut that EXP-G1's full five-class track "
                "was found to be taking is arithmetically unavailable here: a "
                "source-only predictor degenerates to always predicting the "
                "majority class, and scores macro-F1 "
                + format(baseline_f1, ".4f") + " / balanced accuracy "
                + format(baseline_ba, ".4f") + ". "
                + (
                    "Model(s) beating that trivial baseline: "
                    + ", ".join(beaten) + "."
                    if beaten
                    else "NO MODEL BEATS THE TRIVIAL BASELINE."
                )
                + " This is the de-confounded evidence; the five-class table "
                "(T-S1) is reported beside it as the negative result."
            )
            if within_source
            else "**A SOURCE-ONLY BASELINE SCORES MACRO-F1 "
            + format(baseline_f1, ".4f") + " AND BALANCED ACCURACY "
            + format(baseline_ba, ".4f") + " USING NO AUDIO AT ALL** -- it "
            "predicts each recording's class from which PhysioNet "
            "sub-collection it came from, fitted inside each training fold. "
            + (
                "Model(s) beating it: " + ", ".join(beaten) + "."
                if beaten
                else "NO MODEL IN THIS TABLE BEATS IT."
            )
            + " The diagnosis labels are nearly nested inside the "
            "sub-collections (see class_by_sub_collection.csv), so this track "
            "does NOT establish audio-based diagnosis classification and no "
            "number in it may be reported as if it did. The de-confounded "
            "within-source variant is T-S3.",
            "This track covers the ABNORMAL recordings only. The 2,575 normal "
            "recordings are not a class here: including them would make the "
            "largest class 39 times the smallest and would restate the binary "
            "task, which EXP-A1 answers on all 3,240 records.",
            "'" + ", ".join(str(c) for c in classes) + "' are the classes. "
            "'Other pathologic' pools four conditions that are individually too "
            "small to estimate (MPC, AD, MR, AS). It is a statistical "
            "convenience and NOT a diagnosis -- no number for it may be read as "
            "a claim about any one of the four.",
            "n = 5 folds. Every interval here is wide and most of them overlap "
            "between models; this is supplementary evidence on a small sample, "
            "not a model ranking that separates its models.",
            "Nine subjects carry recordings under more than one diagnosis "
            "class. Grouping is by subject, so those subjects contribute to two "
            "classes within the same fold; no subject crosses a fold boundary.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return build_table(spec, summary)


def build_ts2(per_class, classes, within_source=False, command=""):
    """T-S2 -- per-class recall, precision and F1 with intervals (T76.6)."""
    from src.reporting.tables import Column, TableSpec, build_table

    columns = [
        Column("model_id", "Model"),
        Column("class_name", "Class"),
        Column("support_mean", "Support/fold", kind="metric", places=1),
    ]
    for kind in ("recall", "precision", "f1"):
        for suffix, header in (
            ("_mean", kind),
            ("_sd", kind + " SD"),
            ("_fold_ci", kind + " 95% CI (folds)"),
        ):
            if kind + suffix in per_class.columns:
                columns.append(
                    Column(kind + suffix, header,
                           kind="metric" if suffix in ("_mean", "_sd") else None)
                )

    spec = TableSpec(
        table_id="T-S4" if within_source else "T-S2",
        title=(
            "Diagnosis Per-Class Results (within one source)"
            if within_source
            else "Diagnosis Track Per-Class Results"
        ),
        caption=(
            "Per-class recall, precision and F1 for every model, each with a "
            "95% Student-t interval over the 5 folds and the mean per-fold "
            "support beside it. Rule 6: a macro-F1 can look respectable while "
            "one class is never predicted, so the per-class recall is the column "
            "that matters here."
        ),
        sources=SOURCES,
        columns=tuple(columns),
        exp_id="EXP-G1",
        objective="O6 (multiclass classification)",
        dataset="D1 PhysioNet 2016, abnormal recordings only",
        notes=(
            "Intervals are over n = 5 folds. With 12-13 test records per fold "
            "for the smallest classes, an interval spanning 0.3 or more is "
            "expected and is the honest width, not a defect in the model.",
            "'Other pathologic' is a pooled bucket of four unrelated "
            "conditions, not a diagnosis. Its recall says how often a rare "
            "pathology was placed in the rare-pathology bin, nothing more.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return build_table(spec, per_class)


def main(argv: list[str] | None = None) -> int:

    from src.evaluation import diagnosis_track as dt
    from src.reporting.multiclass_report import (
        build_multiclass_table,
        build_per_class_table,
    )
    from src.reporting.tables import write_table
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    section = Path(args.out_dir) if args.out_dir else Path(SECTION)
    variant = "-within_source" if args.within_source else ""
    root = section / (RUN_DIR + variant)
    root.mkdir(parents=True, exist_ok=True)

    run = start_run("diagnosis_track" + ("_within_source" if args.within_source else ""))
    run.set("diagnosis_stage", args.stage)
    try:
        # T76.2 -- the merge policy, written before anything uses it.
        policy = dt.merge_report(dt.diagnosis_records(within_source=args.within_source))
        save_csv(policy, root / "diagnosis_class_merge_policy.csv")
        log.info(
            "merge policy: %d class(es) retained, %d merged into %r",
            int(policy["retained"].sum()),
            int((~policy["retained"]).sum()),
            dt.OTHER_CLASS,
        )

        # T76.3 -- the fold map, leakage-checked on the way out.
        dt.write_split_map(within_source=args.within_source)
        if args.stage == "map":
            run.finish("ok")
            return 0

        # T76.4 -- the mandatory model set.
        result = dt.run_track(
            models=tuple(args.models) if args.models else None,
            within_source=args.within_source,
        )
        per_fold = result["per_fold"]
        classes = result["classes"]
        save_csv(per_fold, root / "per_fold_metrics.csv")
        per_fold.to_parquet(root / "predictions.parquet", index=False)
        result["predictions"].to_parquet(root / "oof_predictions.parquet", index=False)

        # T76.6 -- intervals on every metric, fold-level and record-level.
        summary = build_multiclass_table(
            per_fold,
            predictions=result["predictions"],
            labels=list(range(len(classes))),
        )
        per_class = build_per_class_table(per_fold, classes)
        save_csv(summary, root / "aggregate_metrics.csv")
        save_csv(per_class, root / "per_class_metrics.csv")

        # The confound baseline, and the confound itself. Written BEFORE the
        # tables so the tables can quote it: on this track a source-only lookup
        # outscores every model, and an EXP-G1 number without it beside it
        # would read as audio-based diagnosis classification.
        baseline = dt.source_confound_baseline(within_source=args.within_source)
        save_csv(baseline, root / "source_confound_baseline.csv")
        if not args.within_source:
            save_csv(dt.class_by_source_table(), root / "class_by_sub_collection.csv")
        baseline_f1 = float(baseline["macro_f1"].mean())
        baseline_ba = float(baseline["balanced_accuracy"].mean())
        beaten = summary[summary["macro_f1_mean"] > baseline_f1]["model_id"].tolist()

        save_json(
            {
                "exp_id": dt.EXP_ID,
                "task": dt.TASK,
                "scheme": dt.SCHEME,
                "classes": list(classes),
                "merge_threshold_records": dt.MIN_CLASS_RECORDS,
                "other_class": dt.OTHER_CLASS,
                "n_records": result["n_records"],
                "n_subjects": result["n_subjects"],
                "supplementary": True,
                "command": COMMAND,
            },
            root / "run_manifest.json",
        )

        for table in (
            build_ts1(summary, per_class, classes, result["n_records"],
                      result["n_subjects"], baseline_f1, baseline_ba, beaten,
                      args.within_source, COMMAND),
            build_ts2(per_class, classes, args.within_source, COMMAND),
        ):
            written = write_table(table, section)
            log.info("%s -> %s", table.spec.table_id, written["csv"].name)

        ranked = summary.sort_values("macro_f1_mean", ascending=False)
        for row in ranked.to_dict("records"):
            log.info(
                "%-3s macro-F1 %.4f %s  balanced %.4f",
                row["model_id"],
                row["macro_f1_mean"],
                row.get("macro_f1_fold_ci", ""),
                row["balanced_accuracy_mean"],
            )
        run.set("diagnosis_models", len(summary))
        run.set("diagnosis_classes", list(classes))
    except BaseException:
        run.finish("failed")
        raise

    run.finish("ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
