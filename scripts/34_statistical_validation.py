"""Statistical validation (Phase 82). Writes the raw test results.

Fold-wise means and SDs, bootstrap confidence intervals for ROC-AUC, McNemar over
paired predictions, Wilcoxon and paired t on fold-level pairs with a documented
normality check, Friedman with a Nemenyi post-hoc, and an effect size beside
every p-value.

Nothing is fitted: every test reads the stored per-fold metrics and out-of-fold
predictions. Seconds, not minutes.

Phase 83's `scripts/35_statistical_reporting.py` turns these into T28, the
significance matrix and the summary DOCX.

    python scripts/34_statistical_validation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/34_statistical_validation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("statistical_validation")

SECTION = "outputs/12_statistics"
COMMAND = "python scripts/34_statistical_validation.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.evaluation.statistics import BOOTSTRAP_RESAMPLES

    parser = argparse.ArgumentParser(prog="34_statistical_validation")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.evaluation import statistics as st
    from src.utils.evidence import PROJECT_ROOT
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    root = PROJECT_ROOT / SECTION
    root.mkdir(parents=True, exist_ok=True)
    run = start_run("statistical_validation")
    written: dict[str, Path] = {}

    fold_blocks: list[pd.DataFrame] = []
    prediction_blocks: list[pd.DataFrame] = []
    coverage: list[dict[str, object]] = []

    for spec in st.RUNS:
        metrics = st.load_fold_metrics(spec)
        predictions = st.partition_predictions(spec)
        if metrics is not None:
            fold_blocks.append(metrics)
        if predictions is not None:
            predictions = predictions.copy()
            predictions.insert(0, "run", spec.run)
            prediction_blocks.append(predictions)
        coverage.append(
            {
                "run": spec.run,
                "task": spec.task,
                "n_folds": int(metrics["fold_label"].nunique()) if metrics is not None else 0,
                "n_models": int(metrics["model_id"].nunique()) if metrics is not None else 0,
                "n_partition_records": (
                    int(predictions["record_uid"].nunique()) if predictions is not None else 0
                ),
                "minimum_achievable_p_wilcoxon": st.minimum_achievable_p(
                    "wilcoxon",
                    int(metrics["fold_label"].nunique()) if metrics is not None else 0,
                ),
                "note": spec.note,
            }
        )

    if not fold_blocks:
        log.error("no run had per-fold metrics to test")
        return 2

    per_fold = pd.concat(fold_blocks, ignore_index=True)
    written["coverage"] = save_csv(pd.DataFrame(coverage), root / "statistics_coverage.csv")
    written["per-fold metrics"] = save_csv(per_fold, root / "per_fold_metrics_all_runs.csv")

    summary = st.foldwise_summary(per_fold)
    written["fold-wise summary"] = save_csv(summary, root / "foldwise_mean_sd.csv")

    paired = st.correct_p_values(st.paired_fold_tests(per_fold))
    written["paired fold tests"] = save_csv(paired, root / "paired_fold_tests.csv")

    omnibus, posthoc = st.friedman_test(per_fold)
    written["friedman"] = save_csv(omnibus, root / "friedman_omnibus.csv")
    if len(posthoc):
        written["nemenyi"] = save_csv(posthoc, root / "nemenyi_posthoc.csv")

    predictions = (
        pd.concat(prediction_blocks, ignore_index=True) if prediction_blocks else None
    )
    if predictions is not None:
        bootstrap = st.bootstrap_auc(predictions, n_resamples=args.resamples)
        written["bootstrap auc"] = save_csv(bootstrap, root / "bootstrap_auc_ci.csv")
        mcnemar = st.correct_p_values(st.mcnemar_matrix(predictions), group_by=("run",))
        written["mcnemar"] = save_csv(mcnemar, root / "mcnemar_paired_predictions.csv")

    written["design"] = save_json(
        {
            "alpha": st.ALPHA,
            "partition_repeat": st.PARTITION_REPEAT,
            "bootstrap_resamples": int(args.resamples),
            "bootstrap_seed": 42,
            "normality_check": (
                "Shapiro-Wilk on the paired differences at alpha "
                + str(st.ALPHA)
                + "; the paired t-test where they are normal, Wilcoxon "
                "signed-rank otherwise. Both tests are computed and stored for "
                "every pair, so the choice is auditable."
            ),
            "n_description": st.N_DESCRIPTION,
            "why_one_repeat_for_record_level_tests": (
                "The repeated 5x5 map holds every record out once per repeat, so "
                "pooling all 25 folds would put each record in the sample five "
                "times. McNemar and the bootstrap therefore run on repeat "
                + str(st.PARTITION_REPEAT)
                + " alone, which is a complete partition of the corpus."
            ),
            "why_all_folds_for_fold_level_tests": (
                "There the unit of observation is the fold, not the record, and "
                "25 paired folds is what makes the primary track powered at all."
            ),
        },
        root / "statistical_design.json",
    )

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    headline = paired[(paired["run"] == "EXP-A2") & (paired["metric"] == "sensitivity")]
    print(
        headline[
            [
                "model_a",
                "model_b",
                "n",
                "mean_difference",
                "chosen_test",
                "p_value",
                "p_holm",
                "effect_size",
                "underpowered",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    for name, path in written.items():
        print(f"{name:22s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
