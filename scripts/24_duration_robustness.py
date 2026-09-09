"""EXP-E2 -- duration robustness (Phase 73). Emits T21 and G31.

Two stages, like the noise track, so the expensive one can be resumed:

``--stage truncation``
    Clips a stratified PhysioNet sample to 10, 5 and 3 seconds (plus an
    untouched control), re-runs the whole filter / normalize / extract chain on
    each and checkpoints after every batch. Cheaper than the noise sweep --
    ``time_sample_entropy`` is O(N^2), so a 3-second clip costs a fraction of the
    30-second original.

``--stage tables`` (default: both)
    Partitions the stored out-of-fold predictions by the T18.3 bands, measures
    what a sub-5-second window costs, scores the truncation study, and writes
    T21 and G31.

    python scripts/24_duration_robustness.py --stage truncation
    python scripts/24_duration_robustness.py --stage tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python scripts/24_duration_robustness.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("duration_robustness")

SECTION = "outputs/09_robustness_analysis"
RUN_DIR = "EXP-E2"
COMMAND = "python scripts/24_duration_robustness.py"

TRUNCATION_CHECKPOINT = "truncation_features.parquet"

#: The same five stored runs the noise track partitions. Rule 4: never pooled.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="24_duration_robustness")
    parser.add_argument("--stage", default="both", choices=("truncation", "tables", "both"))
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def _sample():
    """Reuse EXP-E1's sample, so the two studies are about the same recordings.

    Sharing the sample means the noise curve and the truncation curve can be read
    against each other: a model that survives 5 dB but not a 5-second clip is
    making a statement about which stressor matters, and that comparison is only
    available if both were measured on the same records.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / SECTION / "EXP-E1" / "awgn_sample.csv"
    if not path.is_file():
        raise FileNotFoundError(
            str(path) + "; run scripts/23_noise_robustness.py --stage sweep first"
        )
    return pd.read_csv(path)


def _holdout_model():
    """The EXP-E1 held-out model: the deployed configuration minus the sample's subjects.

    Rebuilt here rather than imported from ``23_noise_robustness``, whose module
    name starts with a digit and is therefore not importable. Identical inputs
    and seed, so it is the same fitted model.
    """
    import json

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
    data = load_task_data("binary")
    held = set(_sample()["subject_id"].astype(str))
    mask = ~np.isin(data.groups.astype(str), list(held))
    estimator = est.build_estimator(model_id, **dict(manifest["hyperparameters"]))
    built = pl.build_pipeline(estimator, y=data.y[mask], n_features=data.X.shape[1])
    built.fit(data.X[mask], data.y[mask])
    log.info("holdout model %s fitted on %d record(s)", model_id, int(mask.sum()))
    return built, model_id


def run_truncation(args) -> int:
    from src.evaluation.duration import TRUNCATION_SECONDS, truncation_features
    from src.utils.io import save_json

    root = Path(SECTION) / RUN_DIR
    root.mkdir(parents=True, exist_ok=True)
    sample = _sample()
    save_json(
        {
            "sample_source": "outputs/09_robustness_analysis/EXP-E1/awgn_sample.csv",
            "n_sample_records": len(sample),
            "n_sample_subjects": int(sample["subject_id"].nunique()),
            "clip_seconds": [None if not np.isfinite(v) else v for v in TRUNCATION_SECONDS],
            "why_shared_sample": (
                "The same recordings carry both stressors, so the noise curve and "
                "the truncation curve can be read against each other."
            ),
            "window_placement": (
                "Randomly placed, seeded per (record, length). Always clipping the "
                "opening seconds would confound length with position -- the head of "
                "a recording carries the transducer being settled."
            ),
        },
        root / "truncation_design.json",
    )
    table = truncation_features(
        sample, checkpoint=root / TRUNCATION_CHECKPOINT, workers=args.workers
    )
    return 0 if len(table) else 2


def score_truncation(root: Path):
    """Metrics at each clip length, from the same held-out model EXP-E1 used."""
    import pandas as pd

    from src.evaluation import metrics as mt
    from src.feature_extraction.registry import feature_names

    path = root / TRUNCATION_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(str(path) + "; run --stage truncation first")

    truncation = pd.read_parquet(path)
    model, model_id = _holdout_model()
    names = list(feature_names())

    rows = []
    for seconds, block in truncation.groupby("clip_seconds", sort=False):
        ordered = block.sort_values("record_uid")
        X = ordered[names].to_numpy(dtype=float)
        y = ordered["y"].to_numpy(dtype=int)
        row = {
            "model_id": model_id,
            "clip_seconds": float(seconds),
            "realised_seconds_mean": float(ordered["realised_seconds"].mean()),
            "realised_seconds_min": float(ordered["realised_seconds"].min()),
            "n_units": len(ordered),
            "n_nan_features": int(np.isnan(X).sum()),
        }
        row.update(mt.binary_metrics(y, model.predict(X), model.predict_proba(X), labels=[0, 1]))
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("clip_seconds", ascending=False).reset_index(drop=True)
    full = frame[~np.isfinite(frame["clip_seconds"])]
    if len(full):
        reference = full.iloc[0]
        for metric in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc"):
            frame[metric + "_delta_vs_full"] = frame[metric] - float(reference[metric])
    return frame


def run_tables(args) -> int:
    import pandas as pd

    from src.evaluation.duration import short_record_diagnostics, stratify_by_duration
    from src.evaluation.experiment import Experiment
    from src.reporting.duration_report import build_g31, build_t21, summarise_bands
    from src.reporting.graphs import write_graph
    from src.reporting.tables import write_table
    from src.utils.run_manifest import start_run

    section = Path(SECTION)
    root = section / RUN_DIR
    root.mkdir(parents=True, exist_ok=True)
    run = start_run("duration_robustness")
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
        per_fold = stratify_by_duration(
            pd.read_parquet(path),
            labels=labels or experiment.labels,
            class_names=experiment.class_names,
        )
        per_fold.insert(0, "run", run_id)
        per_fold.insert(1, "task", task)
        every.append(per_fold)
        fold_path = Path(directory) / "per_fold_by_duration.csv"
        per_fold.to_csv(fold_path, index=False)
        written["per_fold_by_duration (" + run_id + ")"] = fold_path

    combined = pd.concat(every, ignore_index=True)
    summary = summarise_bands(combined)
    summary_path = root / "per_band_summary.csv"
    summary.to_csv(summary_path, index=False)
    written["band summary"] = summary_path

    diagnostics, shifts = short_record_diagnostics()
    diag_path = root / "short_record_diagnostics.csv"
    diagnostics.to_csv(diag_path, index=False)
    written["short-record diagnostics"] = diag_path
    shift_path = root / "short_vs_long_feature_shift.csv"
    shifts.to_csv(shift_path, index=False)
    written["short-vs-long feature shift"] = shift_path

    truncation = score_truncation(root)
    trunc_path = root / "truncation_metrics.csv"
    truncation.to_csv(trunc_path, index=False)
    written["truncation metrics"] = trunc_path

    sources = (
        str(summary_path).replace("\\", "/"),
        str(trunc_path).replace("\\", "/"),
        str(diag_path).replace("\\", "/"),
    )
    table = build_t21(summary, truncation, diagnostics, sources, command=COMMAND)
    for fmt, path in write_table(table, section).items():
        written["T21 " + fmt] = path

    graph = build_g31(summary, truncation, sources, command=COMMAND)
    for fmt, path in write_graph(graph, formats=("png", "svg")).items():
        written["G31 " + fmt] = path

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    columns = ["clip_seconds", "sensitivity", "specificity", "balanced_accuracy", "roc_auc"]
    print(truncation[columns].round(4).to_string(index=False))
    print()
    for name, path in written.items():
        print(f"{name:38s} -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("truncation", "both"):
        code = run_truncation(args)
        if code:
            return code
    if args.stage in ("tables", "both"):
        return run_tables(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
