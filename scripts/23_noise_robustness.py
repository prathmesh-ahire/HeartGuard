"""EXP-E1 -- noise robustness (Phase 72). Emits T20 and G30.

Two stages, run independently so the long one can be resumed:

``--stage sweep``
    Adds white Gaussian noise at 20/15/10/5/0 dB (plus an untouched control) to
    a stratified sample of PhysioNet recordings, re-runs the whole
    filter/normalize/extract chain on each, and checkpoints after every batch.
    This is the expensive stage -- roughly 2.7 s per record per level.

``--stage tables`` (default: both)
    Partitions the stored out-of-fold predictions by the PP-08 quality flags,
    cross-checks the grouping against PhysioNet's REFERENCE-SQI, scores the
    sweep, and writes T20 and G30.

    python scripts/23_noise_robustness.py --stage sweep
    python scripts/23_noise_robustness.py --stage tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python scripts/23_noise_robustness.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("noise_robustness")

SECTION = "outputs/09_robustness_analysis"
RUN_DIR = "EXP-E1"
COMMAND = "python scripts/23_noise_robustness.py"

#: The stored out-of-fold predictions that get partitioned by quality group.
#: One entry per label space -- rule 4, they are never pooled.
#: The experiment id is carried explicitly rather than derived from the run
#: directory: "EXP-C1-two_class".split("-")[0] is "EXP", and a variant folder
#: name is not an experiment id.
RUNS = (
    ("EXP-A2", "outputs/06_binary_results/EXP-A2", "binary", [0, 1], None, "EXP-A2"),
    ("EXP-B1", "outputs/07_multiclass_results/EXP-B1", "pascal_a", None, None, "EXP-B1"),
    ("EXP-B2", "outputs/07_multiclass_results/EXP-B2", "pascal_b", None, None, "EXP-B2"),
    (
        "EXP-C1-two_class",
        "outputs/08_circor_external_validation/EXP-C1-two_class",
        "circor_murmur",
        [0, 1],
        "two_class",
        "EXP-C1",
    ),
    (
        "EXP-C2",
        "outputs/08_circor_external_validation/EXP-C2",
        "circor_outcome",
        [0, 1],
        None,
        "EXP-C2",
    ),
)

SWEEP_CHECKPOINT = "awgn_sweep_features.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="23_noise_robustness")
    parser.add_argument("--stage", default="both", choices=("sweep", "tables", "both"))
    parser.add_argument("--n-records", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def sample_records(n_records: int, seed: int = 42):
    """A stratified PhysioNet sample: proportional over (subset, class).

    Stratified by **subset** as well as by class because PhysioNet's six
    sub-collections behave like six different datasets -- class balance runs
    8.5% to 71.4% abnormal and the top features reverse sign between them. A
    sample drawn without that stratification would measure noise sensitivity
    confounded with which sub-collection happened to be drawn.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    master = pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv", low_memory=False
    )
    d1 = master[
        (master["dataset_source"] == "D1")
        & (master["use_in_supervised"])
        & (master["binary_label"].notna())
    ].copy()
    d1["y"] = d1["binary_label"].astype(int)

    share = n_records / len(d1)
    chunks = []
    for _, block in d1.groupby(["subset", "y"], sort=True):
        take = max(1, round(len(block) * share))
        chunks.append(block.sample(n=min(take, len(block)), random_state=seed))
    sample = pd.concat(chunks).sort_values("record_uid").reset_index(drop=True)
    log.info(
        "sample: %d record(s), %d subject(s), %.1f%% abnormal",
        len(sample),
        sample["subject_id"].nunique(),
        100 * float(sample["y"].mean()),
    )
    return sample[["record_uid", "file_path", "y", "subject_id", "duration_sec", "subset"]]


def fit_holdout_model(sample):
    """Refit the finalized configuration WITHOUT the sampled records' subjects.

    The deployed model saw all 3,240 PhysioNet records, so scoring it on any of
    them is in-sample. Excluding by ``subject_id`` rather than by record keeps
    rule 3: a subject with two recordings must not contribute one to training and
    one to the noise sweep.
    """
    import json
    import time

    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models.smoke import load_task_data
    from src.utils.evidence import PROJECT_ROOT

    manifest = json.loads(
        (PROJECT_ROOT / "models_saved" / "binary" / "final" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    model_id = str(manifest["selected_model_id"])
    params = dict(manifest["hyperparameters"])

    data = load_task_data("binary")
    held = set(sample["subject_id"].astype(str))
    mask = ~np.isin(data.groups.astype(str), list(held))
    estimator = est.build_estimator(model_id, **params)
    built = pl.build_pipeline(estimator, y=data.y[mask], n_features=data.X.shape[1])
    started = time.perf_counter()
    built.fit(data.X[mask], data.y[mask])
    log.info(
        "holdout model %s: fitted on %d of %d record(s) in %.1f s (%d subject(s) excluded)",
        model_id,
        int(mask.sum()),
        len(mask),
        time.perf_counter() - started,
        len(held),
    )
    return built, model_id, params, int(mask.sum())


def run_sweep(args) -> int:
    from src.evaluation.robustness import SNR_LEVELS, sweep_features
    from src.utils.io import save_json

    root = Path(SECTION) / RUN_DIR
    root.mkdir(parents=True, exist_ok=True)
    sample = sample_records(args.n_records)
    sample.to_csv(root / "awgn_sample.csv", index=False)

    _, model_id, params, n_fitted = fit_holdout_model(sample)
    save_json(
        {
            "model_id": model_id,
            "hyperparameters": params,
            "n_records_fitted": n_fitted,
            "n_sample_records": len(sample),
            "n_sample_subjects": int(sample["subject_id"].nunique()),
            "snr_levels_db": [None if not np.isfinite(v) else v for v in SNR_LEVELS],
            "why_refitted": (
                "The deployed model was fitted on all 3,240 PhysioNet records, so "
                "scoring it on any of them is in-sample. This model excludes every "
                "subject appearing in the sweep sample."
            ),
        },
        root / "awgn_model.json",
    )

    table = sweep_features(
        sample,
        checkpoint=root / SWEEP_CHECKPOINT,
        workers=args.workers,
    )
    log.info("sweep rows: %d", len(table))
    return 0 if len(table) else 2


def score_sweep(root: Path):
    """Metrics at each SNR level, from the held-out model."""
    import pandas as pd

    from src.evaluation import metrics as mt
    from src.feature_extraction.registry import feature_names

    path = root / SWEEP_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(str(path) + "; run --stage sweep first")

    sweep = pd.read_parquet(path)
    sample = pd.read_csv(root / "awgn_sample.csv")
    model, model_id, _, _ = fit_holdout_model(sample)
    names = list(feature_names())

    rows = []
    for level, block in sweep.groupby("snr_db", sort=False):
        ordered = block.sort_values("record_uid")
        X = ordered[names].to_numpy(dtype=float)
        y = ordered["y"].to_numpy(dtype=int)
        proba = model.predict_proba(X)
        row = {
            "model_id": model_id,
            "snr_db": float(level),
            "realised_snr_db_mean": float(ordered["realised_snr_db"].mean()),
            "n_units": len(ordered),
            "n_nan_features": int(np.isnan(X).sum()),
        }
        row.update(
            mt.binary_metrics(y, model.predict(X), proba, labels=[0, 1], positive_label=1)
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("snr_db", ascending=False).reset_index(drop=True)

    clean = frame[~np.isfinite(frame["snr_db"])]
    if len(clean):
        reference = clean.iloc[0]
        for metric in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc"):
            frame[metric + "_delta_vs_clean"] = frame[metric] - float(reference[metric])
    return frame


def run_tables(args) -> int:
    import pandas as pd

    from src.evaluation.experiment import Experiment
    from src.evaluation.robustness import sqi_crosscheck, stratify_by_quality
    from src.reporting.graphs import write_graph
    from src.reporting.robustness_report import build_g30, build_t20, summarise_quality
    from src.reporting.tables import write_table
    from src.utils.run_manifest import start_run

    section = Path(SECTION)
    root = section / RUN_DIR
    root.mkdir(parents=True, exist_ok=True)
    run = start_run("noise_robustness")
    written: dict[str, Path] = {}
    every: list[pd.DataFrame] = []

    for run_id, directory, task, labels, variant, exp_id in RUNS:
        path = Path(directory) / "predictions.parquet"
        if not path.is_file():
            log.warning("%s missing; skipping %s", path, run_id)
            continue
        experiment = Experiment.load(exp_id)
        if variant:
            experiment = experiment.for_variant(variant)
        per_fold = stratify_by_quality(
            pd.read_parquet(path),
            labels=labels or experiment.labels,
            class_names=experiment.class_names,
        )
        per_fold.insert(0, "run", run_id)
        per_fold.insert(1, "task", task)
        every.append(per_fold)
        fold_path = Path(directory) / "per_fold_by_quality.csv"
        per_fold.to_csv(fold_path, index=False)
        written["per_fold_by_quality (" + run_id + ")"] = fold_path

    combined = pd.concat(every, ignore_index=True)
    summary = summarise_quality(combined)

    sqi_summary, sqi_detail = sqi_crosscheck()
    sqi_path = root / "sqi_crosscheck.csv"
    sqi_summary.to_csv(sqi_path, index=False)
    written["SQI cross-check"] = sqi_path
    detail_path = root / "sqi_crosscheck_records.csv"
    sqi_detail.to_csv(detail_path, index=False)
    written["SQI cross-check detail"] = detail_path

    sweep = score_sweep(root)
    sweep_path = root / "awgn_sweep_metrics.csv"
    sweep.to_csv(sweep_path, index=False)
    written["AWGN sweep metrics"] = sweep_path

    sources = (
        "outputs/02_preprocessing/signal_quality_flags.csv",
        str(sweep_path).replace("\\", "/"),
        str(sqi_path).replace("\\", "/"),
    )
    table = build_t20(summary, sweep, sqi_summary, sources, command=COMMAND)
    for fmt, path in write_table(table, section).items():
        written["T20 " + fmt] = path

    graph = build_g30(summary, sweep, sources, command=COMMAND)
    for fmt, path in write_graph(graph, formats=("png", "svg")).items():
        written["G30 " + fmt] = path

    summary_path = root / "per_group_summary.csv"
    summary.to_csv(summary_path, index=False)
    written["quality group summary"] = summary_path

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    columns = ["snr_db", "sensitivity", "specificity", "balanced_accuracy", "roc_auc"]
    print(sweep[columns].round(4).to_string(index=False))
    print()
    for name, path in written.items():
        print(f"{name:34s} -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("sweep", "both"):
        code = run_sweep(args)
        if code:
            return code
    if args.stage in ("tables", "both"):
        return run_tables(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
