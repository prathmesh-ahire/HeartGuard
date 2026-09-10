"""Calibration analysis (Phase 78). Emits T23, G33 and G34.

Two stages, because one of them fits models and the other does not.

``--stage svm``
    T78.4 -- refits M3 under sigmoid and under isotonic calibration on the
    identical 25-fold binary map and scores both. ~50 fits, a few minutes. This
    is the only place Phase 78 fits anything; everything else reads the stored
    out-of-fold probabilities, because a calibration number that does not belong
    to the model the tables report is worse than no number.

``--stage tables`` (default: both)
    Collects out-of-fold probabilities from every run, computes Brier and ECE per
    fold at four bin counts, builds the reliability and confidence frames, and
    writes T23, G33 and G34.

    python scripts/29_calibration_analysis.py --stage svm
    python scripts/29_calibration_analysis.py --stage tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/29_calibration_analysis.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("calibration_analysis")

SECTION = "outputs/10_robustness"
RUN_DIR = "calibration"
COMMAND = "python scripts/29_calibration_analysis.py"

SVM_COMPARISON_CSV = "svm_calibration_method_comparison.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="29_calibration_analysis")
    parser.add_argument("--stage", default="both", choices=("svm", "tables", "both"))
    parser.add_argument(
        "--force-svm",
        action="store_true",
        help="re-run the sigmoid/isotonic fits even if the CSV already exists",
    )
    return parser.parse_args(argv)


def _root() -> Path:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION / RUN_DIR


def run_svm(args) -> int:
    from src.evaluation.calibration_analysis import svm_calibration_comparison
    from src.utils.io import save_csv

    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / SVM_COMPARISON_CSV
    if target.is_file() and not args.force_svm:
        log.info("%s already exists; pass --force-svm to refit", target)
        return 0

    comparison = svm_calibration_comparison()
    save_csv(comparison, target)
    log.info("wrote %d row(s) -> %s", len(comparison), target)
    print(
        comparison.groupby("method")[["brier", "ece", "sensitivity", "balanced_accuracy"]]
        .mean()
        .round(4)
        .to_string()
    )
    return 0 if len(comparison) else 2


def run_tables(args) -> int:
    import pandas as pd

    from src.evaluation.calibration_analysis import (
        BIN_SWEEP,
        RELIABILITY_BINS,
        confidence_histogram,
        final_model_calibration,
        load_predictions,
        per_fold_calibration,
        reliability_frame,
        run_specs,
        summarise_calibration,
    )
    from src.reporting.calibration_report import build_g33, build_g34, build_t23
    from src.reporting.graphs import write_graph
    from src.reporting.tables import write_table
    from src.utils.evidence import PROJECT_ROOT
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run

    section = PROJECT_ROOT / SECTION
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    run = start_run("calibration_analysis")
    written: dict[str, Path] = {}

    per_fold_all: list[pd.DataFrame] = []
    reliability_all: list[pd.DataFrame] = []
    histogram_all: list[pd.DataFrame] = []
    covered: list[dict[str, object]] = []

    for spec in run_specs():
        frame = load_predictions(spec)
        if frame is None:
            covered.append({"run": spec.run, "status": "no predictions.parquet"})
            continue

        per_fold = per_fold_calibration(frame, spec.labels, bins=BIN_SWEEP)
        per_fold.insert(0, "run", spec.run)
        per_fold.insert(1, "task", spec.task)
        per_fold_all.append(per_fold)

        reliability = reliability_frame(frame, spec.labels, n_bins=RELIABILITY_BINS)
        reliability.insert(0, "run", spec.run)
        reliability.insert(1, "task", spec.task)
        reliability["n_classes"] = len(spec.labels)
        reliability_all.append(reliability)

        histogram = confidence_histogram(frame, spec.labels)
        histogram.insert(0, "run", spec.run)
        histogram.insert(1, "task", spec.task)
        histogram["n_classes"] = len(spec.labels)
        histogram_all.append(histogram)

        covered.append(
            {
                "run": spec.run,
                "task": spec.task,
                "exp_id": spec.exp_id,
                "variant": spec.variant or "",
                "n_classes": len(spec.labels),
                "class_names": "; ".join(spec.class_names),
                "n_models": int(frame["model_id"].nunique()),
                "n_folds": int(frame["fold_label"].nunique()),
                "n_rows": len(frame),
                "status": "ok",
                "note": spec.note,
            }
        )

    if not per_fold_all:
        log.error("no run had stored out-of-fold probabilities")
        return 2

    per_fold = pd.concat(per_fold_all, ignore_index=True)
    reliability = pd.concat(reliability_all, ignore_index=True)
    histogram = pd.concat(histogram_all, ignore_index=True)
    sweep = summarise_calibration(per_fold)
    summary = sweep[sweep["n_bins"] == RELIABILITY_BINS].reset_index(drop=True)

    written["coverage"] = save_csv(pd.DataFrame(covered), root / "calibration_coverage.csv")
    written["per-fold calibration"] = save_csv(per_fold, root / "calibration_per_fold.csv")
    written["bin sensitivity"] = save_csv(sweep, root / "calibration_bin_sensitivity.csv")
    written["summary"] = save_csv(summary, root / "calibration_summary.csv")
    written["reliability points"] = save_csv(reliability, root / "reliability_points.csv")
    written["confidence histogram"] = save_csv(histogram, root / "confidence_histogram.csv")

    comparison = None
    comparison_path = root / SVM_COMPARISON_CSV
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
    record = final_model_calibration(comparison=comparison)
    written["final-model calibration"] = save_json(
        record, root / "final_model_calibration.json"
    )

    sources = tuple(
        str(p).replace("\\", "/")
        for p in (
            written["summary"],
            written["per-fold calibration"],
            written["reliability points"],
            written["confidence histogram"],
            comparison_path,
        )
    )

    table = build_t23(summary, comparison, record, sources, command=COMMAND)
    for fmt, path in write_table(table, section).items():
        written["T23 " + fmt] = path

    for builder, label in ((build_g33, "G33"), (build_g34, "G34")):
        graph = builder(
            histogram if label == "G33" else reliability, sources, command=COMMAND
        )
        for fmt, path in write_graph(graph, formats=("png", "svg")).items():
            written[label + " " + fmt] = path

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    columns = [
        "run",
        "model_id",
        "brier_mean",
        "ece_mean",
        "mean_confidence_mean",
        "argmax_accuracy_mean",
        "confidence_gap_mean",
    ]
    print(summary[columns].round(4).to_string(index=False))
    print()
    for name, path in written.items():
        print(f"{name:34s} -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("svm", "both"):
        code = run_svm(args)
        if code:
            return code
    if args.stage in ("tables", "both"):
        return run_tables(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
