"""The PP-09 preprocessing ablation run (T29.3-T29.4).

Phase 29 defined the four filter x normalization arms and then stopped: T29.3
asks for features and a trained model under each arm, and at Phase 29 neither
the locked 138-feature extractor (Phases 31-42) nor the fold-safe CV pipeline
and baseline models (Phases 43-46) existed. Both have existed since Phase 46, so
this module is the deferred half of that task, written against the real
pipeline rather than a throwaway one.

What it does, per arm:

1. Rebuilds the signal from the original wav under that arm's ``Config`` --
   ``use_cache=False``, so the shipped cache is neither read nor polluted by
   three configurations that are not the shipped one.
2. Extracts the same locked 138 features, through the same ``extract_all``.
3. Fits one fast baseline model (M1, untuned) over the **same DA-07 fold map**
   every other binary result in the project uses, through the same
   ``build_pipeline`` and ``run_cv``.

Three things are deliberate:

**Only preprocessing varies.** Same records, same folds, same model, same
hyperparameters, same seed. A delta in this table is attributable to the two
switches and nothing else, which is the only thing that makes it an ablation
rather than four unrelated runs.

**The model is untuned M1, not the finalized binary model.** The finalized model
carries hyperparameters searched under the shipped preprocessing; re-using them
would hand PP-A an advantage the other three arms never had a chance to earn.

**PP-A is re-extracted rather than read from FE-03.** It costs a quarter of the
run and buys a check that the ablation path reproduces the shipped pipeline --
``verify_shipped_arm`` compares the re-extracted PP-A matrix against the
committed FE-03 rows. If those disagree, every delta in the table is measuring
the code path as much as the configuration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.ablation import ABLATION_GRID, AblationArm, arm_config
from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "ABLATION_COLUMNS",
    "BASELINE_MODEL_ID",
    "ABLATION_TASK",
    "REFERENCE_ARM",
    "arm_records",
    "checkpoint_path",
    "extract_arm",
    "score_arm",
    "verify_shipped_arm",
    "build_ablation_table",
    "ablation_path",
    "write_ablation",
    "per_fold_path",
    "write_per_fold",
]

log = get_logger(__name__)

#: The baseline. M1 with its declared defaults -- logistic regression is the
#: fastest of the eight and is also what the finalized binary model turned out
#: to be, so the ablation is measured on the same model family that ships.
BASELINE_MODEL_ID = "M1"

#: D1 / PhysioNet binary. The only track large enough for a 25-fold delta to
#: mean anything, and the one T29.3 names.
ABLATION_TASK = "binary"

#: Deltas are quoted against the shipped configuration, not against the worst
#: arm: the question PP-09 answers is "what do the two shipped switches buy".
REFERENCE_ARM = "PP-A"

ABLATION_COLUMNS: tuple[str, ...] = (
    "arm_id",
    "label",
    "filter_enabled",
    "normalization_enabled",
    "is_shipped_configuration",
    "config_hash",
    "model_id",
    "task",
    "scheme",
    "n_folds",
    "n_records",
    "n_subjects",
    "n_nan_features",
    "sensitivity",
    "sensitivity_sd",
    "specificity",
    "specificity_sd",
    "f1",
    "f1_sd",
    "balanced_accuracy",
    "balanced_accuracy_sd",
    "roc_auc",
    "roc_auc_sd",
    "accuracy",
    "accuracy_sd",
    "sensitivity_delta",
    "specificity_delta",
    "f1_delta",
    "balanced_accuracy_delta",
    "roc_auc_delta",
    "accuracy_delta",
    "extract_seconds",
    "fit_seconds",
)

#: Reported for every arm, and the columns the delta is taken on. Sensitivity
#: and balanced accuracy lead because rule 6 says selection prioritises them.
METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "accuracy",
)


# ---------------------------------------------------------------------------
# the record set
# ---------------------------------------------------------------------------


def arm_records(n_records: int | None = None, *, seed: int = 42) -> Any:
    """The D1 binary records the ablation runs on.

    ``n_records=None`` is every supervised D1 record, which is what makes the
    arms directly comparable to every other binary number in the project: the
    same 3,240 rows and the same DA-07 folds, whole. A smaller ``n_records``
    draws a sample stratified by (subset, class) -- PhysioNet's six
    sub-collections differ enough in class balance that an unstratified draw
    would confound the preprocessing effect with which collection was drawn.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    master = pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv",
        low_memory=False,
    )
    d1 = master[
        (master["dataset_source"] == "D1")
        & (master["use_in_supervised"])
        & (master["binary_label"].notna())
    ].copy()
    d1["y"] = d1["binary_label"].astype(int)

    if n_records is not None and n_records < len(d1):
        share = n_records / len(d1)
        chunks = [
            block.sample(n=min(max(1, round(len(block) * share)), len(block)), random_state=seed)
            for _, block in d1.groupby(["subset", "y"], sort=True)
        ]
        d1 = pd.concat(chunks)

    d1 = d1.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    log.info(
        "ablation record set: %d record(s), %d subject(s), %.1f%% abnormal",
        len(d1),
        d1["subject_id"].nunique(),
        100 * float(d1["y"].mean()),
    )
    return d1[["record_uid", "file_path", "y", "subject_id", "split_group", "subset"]]


# ---------------------------------------------------------------------------
# T29.3 -- features under one arm
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    from src.utils.config import load_config

    base = Path(load_config("paths").require("cache.features"))
    return ensure_dir(base / "pp_ablation")


def checkpoint_path(arm: AblationArm, cfg: Any | None = None) -> Path:
    """Where one arm's feature matrix is checkpointed.

    Keyed by the arm's config hash, not by its id: two arms that somehow
    resolved to the same preprocessing would collide here loudly instead of
    quietly producing two files with identical contents.
    """
    from src.preprocessing.pipeline import config_hash

    digest = config_hash(arm_config(arm, cfg))
    return _cache_root() / (arm.arm_id + "_" + digest + ".parquet")


def _extract_one(record: dict, arm_data: dict, names: list) -> dict[str, Any]:
    """One record under one arm. Runs in a worker process."""
    from src.feature_extraction.extractor import extract_all
    from src.preprocessing.pipeline import preprocess
    from src.utils.config import Config

    uid = str(record["record_uid"])
    cfg = Config("signal", arm_data)
    # use_cache=False in both directions: three of the four arms are not the
    # shipped configuration and must not leave signals in a cache that every
    # other part of the project reads by config hash alone.
    prepared = preprocess(
        Path(record["root"]) / record["file_path"],
        cfg,
        record_uid=uid,
        with_quality=False,
        use_cache=False,
    )
    result = extract_all(prepared.signal, prepared.fs, record_uid=uid)
    row: dict[str, Any] = {
        "record_uid": uid,
        "y": int(record["y"]),
        "split_group": str(record["split_group"]),
        "signal_std": float(np.std(np.asarray(prepared.signal, dtype=float))),
    }
    row.update({name: float(result.values[name]) for name in names})
    return row


def extract_arm(
    arm: AblationArm,
    records: Any,
    *,
    cfg: Any | None = None,
    workers: int = 4,
    batch: int = 32,
    force: bool = False,
) -> tuple[Any, float]:
    """Extract the locked 138 features for every record under one arm.

    Checkpointed after every batch and resumed by ``record_uid``: this is a
    multi-hour job on CPU and the alternative to resuming is starting over.
    Returns the table and the wall seconds actually spent computing (a fully
    resumed arm reports 0.0, which is the honest number for T25).
    """
    import pandas as pd
    from joblib import Parallel, delayed

    from src.feature_extraction.registry import feature_names
    from src.utils.config import load_config

    names = list(feature_names())
    arm_data = arm_config(arm, cfg).as_dict()
    root = str(Path(load_config("paths").require("project_root")))
    target = checkpoint_path(arm, cfg)

    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if target.is_file() and not force:
        existing = pd.read_parquet(target)
        rows = existing.to_dict("records")
        done = set(existing["record_uid"].astype(str))
        log.info("%s: resuming, %d record(s) already extracted", arm.arm_id, len(done))

    wanted = records.to_dict("records")
    pending = [dict(r, root=root) for r in wanted if str(r["record_uid"]) not in done]
    log.info(
        "%s (%s): %d record(s) to extract, %d already done, %d worker(s)",
        arm.arm_id,
        arm.label,
        len(pending),
        len(done),
        workers,
    )

    started = time.perf_counter()
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        produced = Parallel(n_jobs=workers)(
            delayed(_extract_one)(record, arm_data, names) for record in chunk
        )
        rows.extend(produced)
        pd.DataFrame(rows).to_parquet(target, index=False)
        log.info(
            "%s: %d/%d record(s) (%.1f min elapsed)",
            arm.arm_id,
            len(done) + start + len(chunk),
            len(done) + len(pending),
            (time.perf_counter() - started) / 60,
        )
    seconds = time.perf_counter() - started if pending else 0.0

    table = pd.DataFrame(rows)
    keep = set(records["record_uid"].astype(str))
    table = table[table["record_uid"].astype(str).isin(keep)]
    table = table.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    if len(table) != len(records):
        raise RuntimeError(
            arm.arm_id + ": extracted " + str(len(table)) + " row(s) for "
            + str(len(records)) + " record(s)"
        )
    return table, seconds


def verify_shipped_arm(table: Any, *, atol: float = 1e-6) -> dict[str, Any]:
    """Check the re-extracted PP-A against the committed FE-03 matrix.

    PP-A *is* the shipped configuration, so its features must reproduce FE-03
    to within floating-point noise. If they do not, the arms are being compared
    through a code path that differs from the one that produced every other
    number in the project, and the deltas measure that difference too.
    """
    from src.feature_extraction.matrix import load_matrix
    from src.feature_extraction.registry import feature_names

    names = list(feature_names())
    shipped = load_matrix()
    shipped = shipped[shipped["record_uid"].astype(str).isin(set(table["record_uid"].astype(str)))]
    shipped = shipped.sort_values("record_uid", kind="mergesort").reset_index(drop=True)

    ours = table.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    if list(shipped["record_uid"].astype(str)) != list(ours["record_uid"].astype(str)):
        raise RuntimeError("PP-A and FE-03 do not cover the same records")

    left = shipped[names].to_numpy(dtype=float)
    right = ours[names].to_numpy(dtype=float)
    both_nan = np.isnan(left) & np.isnan(right)
    difference = np.abs(left - right)
    difference[both_nan] = 0.0
    scale = np.maximum(np.abs(left), np.abs(right))
    relative = np.where(scale > 1.0, difference / scale, difference)
    worst = int(np.nanargmax(relative)) if relative.size else 0
    report = {
        "n_records": len(ours),
        "n_cells": int(relative.size),
        "max_relative_difference": float(np.nanmax(relative)) if relative.size else 0.0,
        "worst_feature": names[worst % len(names)] if relative.size else "",
        "matches": bool(np.nanmax(relative) <= atol) if relative.size else True,
    }
    log.info(
        "PP-A vs FE-03: max relative difference %.3g on %s (%s)",
        report["max_relative_difference"],
        report["worst_feature"],
        "match" if report["matches"] else "MISMATCH",
    )
    return report


# ---------------------------------------------------------------------------
# T29.3 -- the baseline model under one arm
# ---------------------------------------------------------------------------


def _baseline_factory(y: Any, n_features: int):
    from src.models import estimators as est
    from src.models import pipeline as pl

    def build() -> Any:
        return pl.build_pipeline(
            est.build_estimator(BASELINE_MODEL_ID), y=y, n_features=n_features
        )

    return build


def score_arm(arm: AblationArm, table: Any) -> tuple[dict[str, Any], Any]:
    """Cross-validate the baseline model on one arm's feature matrix.

    The DA-07 fold map is loaded, never re-derived: an ablation that generated
    its own folds would differ from the shipped results in two ways at once.
    ``require_all=False`` only matters when the run is a subsample -- with the
    full record set every fold resolves whole.

    Metrics are computed **per fold and then averaged**, the way every other
    result in this project reports them, not pooled over the out-of-fold
    predictions. Pooling 5 repeats of the same record into one confusion matrix
    would give a number with no companion SD and no basis for a paired test.
    Returns the summary row and the per-fold frame behind it.
    """
    import pandas as pd

    from src.evaluation import cv
    from src.evaluation import metrics as mt
    from src.feature_extraction.registry import feature_names

    names = list(feature_names())
    ordered = table.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    uids = tuple(ordered["record_uid"].astype(str))
    X = ordered[names].to_numpy(dtype=float)
    y = ordered["y"].to_numpy(dtype=int)
    groups = ordered["split_group"].astype(str).to_numpy()

    folds = cv.resolve_folds(cv.load_folds(ABLATION_TASK), uids, require_all=False)
    started = time.perf_counter()
    result = cv.run_cv(
        _baseline_factory(y, X.shape[1]), X, y, groups, folds, task=ABLATION_TASK
    )
    fit_seconds = time.perf_counter() - started

    rows = []
    for fold_result in result.folds:
        proba = None
        if fold_result.y_proba is not None and 1 in fold_result.classes:
            proba = fold_result.y_proba[:, fold_result.classes.index(1)]
        scores = mt.binary_metrics(
            fold_result.y_true, fold_result.y_pred, proba, labels=[0, 1], positive_label=1
        )
        rows.append(
            {
                "arm_id": arm.arm_id,
                "repeat": fold_result.repeat,
                "fold": fold_result.fold,
                "n_train": fold_result.n_train,
                "n_test": len(fold_result.test_uids),
                **{metric: float(scores[metric]) for metric in METRICS},
            }
        )
    per_fold = pd.DataFrame(rows)

    summary: dict[str, Any] = {
        "scheme": result.scheme,
        "n_folds": len(folds),
        "n_records": len(ordered),
        "n_subjects": len(set(groups.tolist())),
        "n_nan_features": int(np.isnan(X).sum()),
        "fit_seconds": round(fit_seconds, 3),
    }
    for metric in METRICS:
        values = per_fold[metric].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary[metric] = float(np.mean(finite)) if finite.size else float("nan")
        summary[metric + "_sd"] = (
            float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        )
    log.info(
        "%s: sensitivity %.4f, balanced accuracy %.4f over %d fold(s) in %.1f s",
        arm.arm_id,
        summary["sensitivity"],
        summary["balanced_accuracy"],
        len(folds),
        fit_seconds,
    )
    return summary, per_fold


# ---------------------------------------------------------------------------
# T29.4 -- PP-09
# ---------------------------------------------------------------------------


def build_ablation_table(rows: list[dict[str, Any]]) -> Any:
    """Add the delta columns and order the table as the 2x2 reads."""
    import pandas as pd

    reference = next((row for row in rows if row["arm_id"] == REFERENCE_ARM), None)
    if reference is None:
        raise ValueError("no " + REFERENCE_ARM + " row to take deltas against")

    for row in rows:
        for metric in METRICS:
            row[metric + "_delta"] = float(row[metric]) - float(reference[metric])

    order = {arm.arm_id: index for index, arm in enumerate(ABLATION_GRID)}
    table = pd.DataFrame(rows)
    table = table.sort_values("arm_id", key=lambda s: s.map(order)).reset_index(drop=True)
    return table[[column for column in ABLATION_COLUMNS if column in table.columns]]


def ablation_path(out_dir: str | Path | None = None) -> Path:
    """PP-09 ``preprocessing_ablation.csv``."""
    from src.preprocessing.ablation import _preprocessing_dir

    return _preprocessing_dir(out_dir) / "preprocessing_ablation.csv"


def write_ablation(table: Any, out_dir: str | Path | None = None) -> Path:
    path = save_csv(table, ablation_path(out_dir))
    log.info("PP-09: %d arm(s) -> %s", len(table), path.name)
    return path


def per_fold_path(out_dir: str | Path | None = None) -> Path:
    """The 25 per-arm fold values PP-09's means are computed from.

    Not a PP artifact -- it is the working behind one, kept beside it so a
    reader can check a mean rather than take it on trust, and so a later paired
    test has the fold values it needs instead of re-running the ablation.
    """
    from src.preprocessing.ablation import _preprocessing_dir

    return _preprocessing_dir(out_dir) / "preprocessing_ablation_per_fold.csv"


def write_per_fold(frames: list[Any], out_dir: str | Path | None = None) -> Path:
    import pandas as pd

    table = pd.concat(frames, ignore_index=True)
    path = save_csv(table, per_fold_path(out_dir))
    log.info("per-fold ablation values: %d row(s) -> %s", len(table), path.name)
    return path
