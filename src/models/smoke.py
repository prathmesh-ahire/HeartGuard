"""Fold-0 smoke runs -- the cheapest honest check that a model works (T46.6, T47.6).

A smoke run is one model, one fold, real data. Not a synthetic matrix: the whole
point is that the 138 real features carry NaNs in named places, wildly different
scales, and seven columns that are constant by construction, and a model that
fits a Gaussian blob happily can still fail on those. Not the full 5x5 either:
this is a "does it run, does it produce usable probabilities, how long does it
take" check, and the numbers it produces are **diagnostic, not results**. The
metrics that go in the thesis come from the full repeated CV in Part VII.

Everything the run needs -- the fold, the matrix, the labels -- comes from
artifacts already on disk: the DA-07 split map, FE-03, and DA-08. Nothing is
re-derived, so a smoke metric is comparable with the full run that follows it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "SMOKE_METRICS_FILENAME",
    "SmokeError",
    "TaskData",
    "SmokeResult",
    "load_task_data",
    "smoke_fold",
    "run_smoke",
    "smoke_frame",
    "write_smoke_metrics",
    "serialized_size",
]

log = get_logger("models.smoke")

SMOKE_METRICS_FILENAME = "baseline_smoke_metrics.csv"


class SmokeError(RuntimeError):
    """The smoke run cannot be assembled from what is on disk."""


@dataclass
class TaskData:
    """One task's feature matrix, labels and groups, in a single row order.

    The three arrays and the uid list share an index by construction. That
    correspondence is the thing a fold resolves against, so it is built once
    here and never re-established downstream by joining again.
    """

    task: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    record_uids: tuple[str, ...]
    feature_names: tuple[str, ...]

    @property
    def n_records(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def classes(self) -> tuple[Any, ...]:
        return tuple(np.unique(self.y).tolist())


@dataclass
class SmokeResult:
    """What one model did on one fold."""

    model_id: str
    task: str
    fold_label: str
    n_train: int
    n_test: int
    n_features: int
    fit_seconds: float
    predict_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    probability: dict[str, Any] = field(default_factory=dict)
    model_bytes: int = -1
    #: The decision rule used, and -- for a thresholded model -- the in-fold
    #: threshold beside its fixed-0.5 counterpart (T50.4).
    decision: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: The fitted pipeline, when the caller asked to keep it (importances need
    #: it). Never written to the CSV -- it is a model, not a measurement.
    pipeline: Any = None

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model_id": self.model_id,
            "task": self.task,
            "fold": self.fold_label,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "fit_seconds": round(self.fit_seconds, 4),
            "predict_seconds": round(self.predict_seconds, 4),
            "model_mb": (
                round(self.model_bytes / 1024**2, 4) if self.model_bytes >= 0 else None
            ),
        }
        row.update(self.metrics)
        row.update(self.probability)
        row.update(self.decision)
        row["notes"] = self.notes
        return row


# ---------------------------------------------------------------------------
# assembling the data
# ---------------------------------------------------------------------------


def load_task_data(task: str, *, matrix: Any = None, master: Any = None) -> TaskData:
    """FE-03 joined to one task's labels, in one fixed row order.

    Rows are ordered by ``record_uid`` rather than by whatever order the parquet
    happened to hold. Fold membership is resolved by uid, so ordering does not
    change any result -- but it does change which physical rows a fitted scaler
    saw, and a run that cannot be reproduced row-for-row is harder to debug than
    one that can.
    """
    from src.data_loader import master as ms
    from src.feature_extraction.matrix import load_matrix
    from src.feature_extraction.registry import feature_names

    frame = load_matrix() if matrix is None else matrix
    reference = ms.load_master() if master is None else master
    labels = ms.task_frame(reference, task)

    if labels.empty:
        raise SmokeError("task " + repr(task) + " has no supervised rows")

    names = feature_names()
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise SmokeError(
            "FE-03 is missing " + str(len(missing)) + " registry feature(s), e.g. "
            + ", ".join(missing[:5])
        )

    wanted = labels[["record_uid", "y", "split_group"]].copy()
    wanted["record_uid"] = wanted["record_uid"].astype(str)
    features = frame[["record_uid", *names]].copy()
    features["record_uid"] = features["record_uid"].astype(str)

    merged = wanted.merge(features, on="record_uid", how="inner", validate="1:1")
    if len(merged) != len(wanted):
        absent = sorted(set(wanted["record_uid"]) - set(merged["record_uid"]))
        raise SmokeError(
            str(len(absent)) + " labelled record(s) of task " + task
            + " have no row in FE-03, e.g. " + ", ".join(absent[:5])
        )

    merged = merged.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    return TaskData(
        task=task,
        X=merged[list(names)].to_numpy(dtype=float),
        y=merged["y"].to_numpy(dtype=int),
        groups=merged["split_group"].astype(str).to_numpy(),
        record_uids=tuple(merged["record_uid"].astype(str)),
        feature_names=tuple(names),
    )


def _predicts_by_argmax(model: Any) -> bool:
    """Whether this model's ``predict`` is the argmax of its ``predict_proba``."""
    final = getattr(model, "named_steps", {}).get("estimator", model)
    declared = getattr(final, "predicts_by_argmax", None)
    return bool(declared) if declared is not None else True


def _decision_rule(model: Any) -> dict[str, Any]:
    """The decision rule the model actually used, for the smoke table.

    Recorded on every row, not only the thresholded ones: a column that is
    present for some models and absent for others is a column nobody can read.
    """
    final = getattr(model, "named_steps", {}).get("estimator", model)
    threshold = getattr(final, "threshold_", None)
    row: dict[str, Any] = {
        "decision_rule": "argmax" if threshold is None else "threshold",
        "threshold": 0.5 if threshold is None else float(threshold),
    }
    choice = getattr(final, "threshold_choice_", None)
    if choice is not None:
        row.update(choice.as_dict())
    report = getattr(final, "fit_report_", None)
    if report is not None:
        row.update(
            {
                key: value
                for key, value in report.as_dict().items()
                if key not in row
            }
        )
    return row


def _scalar_metrics(scores: dict[str, Any]) -> dict[str, float]:
    """Keep the scalar entries. The multiclass report also carries per-class arrays."""
    kept: dict[str, float] = {}
    for key, value in scores.items():
        if np.isscalar(value) and not isinstance(value, (str, bytes)):
            kept[key] = float(value)  # type: ignore[arg-type]
    return kept


def _fold_zero(task: str, data: TaskData) -> Any:
    from src.evaluation import cv

    folds = cv.resolve_folds(cv.load_folds(task), data.record_uids)
    for fold in folds:
        if fold.repeat == 0 and fold.fold == 0:
            return fold
    raise SmokeError("task " + task + " has no repeat 0 / fold 0 in the split map")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def smoke_fold(
    model_id: str,
    estimator_factory: Callable[[], Any],
    data: TaskData,
    fold: Any,
    *,
    pipeline_config: dict[str, Any] | None = None,
    positive_label: int = 1,
    notes: str = "",
    measure_size: bool = True,
    keep_pipeline: bool = False,
) -> SmokeResult:
    """Fit one model on ``fold``'s training rows and score its test rows.

    The estimator is wrapped in the Phase 44 fold-safe pipeline before fitting,
    not fitted bare. A smoke run on a bare estimator would be measuring a
    different object from the one the experiments use -- no imputation, no
    scaling -- and for a distance-based model like M2 that is not a small
    difference.
    """
    from src.evaluation import cv as cv_module
    from src.evaluation import metrics as mt
    from src.models import estimators as est
    from src.models import pipeline as pl

    train_index = np.asarray(fold.train_index, dtype=int)
    test_index = np.asarray(fold.test_index, dtype=int)
    cv_module.assert_group_disjoint(fold)

    shared = set(data.groups[train_index].tolist()) & set(
        data.groups[test_index].tolist()
    )
    if shared:
        raise cv_module.LeakageError(
            fold.label + ": " + str(len(shared)) + " subject group(s) on both sides "
            "of the split after resolving against the matrix"
        )

    built = pl.build_pipeline(
        estimator_factory(),
        config=pipeline_config,
        y=data.y[train_index],
        n_features=data.n_features,
    )

    started = time.perf_counter()
    built.fit(data.X[train_index], data.y[train_index])
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    y_pred = np.asarray(built.predict(data.X[test_index]))
    proba = (
        np.asarray(built.predict_proba(data.X[test_index]))
        if est.has_predict_proba(built)
        else None
    )
    predict_seconds = time.perf_counter() - started

    y_true = data.y[test_index]
    classes = np.asarray(getattr(built, "classes_", np.unique(data.y)))

    if len(classes) == 2:
        scores = mt.binary_metrics(
            y_true, y_pred, proba, labels=tuple(classes.tolist()),
            positive_label=positive_label,
        )
    else:
        scores = mt.multiclass_metrics(
            y_true, y_pred, proba, labels=tuple(classes.tolist())
        )

    probability: dict[str, Any] = {"has_predict_proba": proba is not None}
    if proba is not None:
        # A thresholded classifier predicts by cut-off, not by argmax, so the
        # two are SUPPOSED to disagree -- checking agreement there would fail a
        # working decision rule. Only ask the question of models that do use
        # argmax; for the rest, `threshold_` records what they use instead.
        by_argmax = _predicts_by_argmax(built)
        report = est.probability_report(
            proba,
            n_classes=len(classes),
            y_pred=y_pred if by_argmax else None,
            classes=classes if by_argmax else None,
        )
        probability.update(report.as_dict())
        scores["brier"] = mt.brier_score(y_true, proba, labels=tuple(classes.tolist()))
        scores["ece"] = mt.expected_calibration_error(
            y_true, proba, labels=tuple(classes.tolist())
        )

    result = SmokeResult(
        model_id=model_id,
        task=data.task,
        fold_label=fold.label,
        n_train=int(train_index.size),
        n_test=int(test_index.size),
        n_features=data.n_features,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        metrics=_scalar_metrics(scores),
        probability=probability,
        model_bytes=serialized_size(built) if measure_size else -1,
        decision=_decision_rule(built),
        notes=notes,
        pipeline=built if keep_pipeline else None,
    )
    log.info(
        "%s on %s %s: bal-acc %.4f, sens %.4f, fit %.2f s, %.2f MB",
        model_id,
        data.task,
        fold.label,
        result.metrics.get("balanced_accuracy", float("nan")),
        result.metrics.get("sensitivity", float("nan")),
        fit_seconds,
        max(result.model_bytes, 0) / 1024**2,
    )
    return result


def serialized_size(model: Any) -> int:
    """Bytes the fitted pipeline occupies once joblib has written it out.

    Measured by actually serialising, not by ``sys.getsizeof`` or by counting
    trees. A 500-tree forest's in-memory footprint and its on-disk footprint
    differ by more than an order of magnitude, and the number the complexity
    table (T26) needs is the one a deployment would actually have to ship.

    Written to a temporary file rather than a BytesIO because that is what
    Phase 51 will do, and a size that changes between the measurement and the
    thing it is supposed to describe is not a measurement.
    """
    import tempfile
    from pathlib import Path as _Path

    import joblib

    with tempfile.TemporaryDirectory() as directory:
        target = _Path(directory) / "model.joblib"
        joblib.dump(model, target)
        return int(target.stat().st_size)


def run_smoke(
    model_ids: tuple[str, ...] | list[str],
    *,
    task: str = "binary",
    data: TaskData | None = None,
    factories: dict[str, Callable[[], Any]] | None = None,
    keep_pipelines: bool = False,
) -> tuple[SmokeResult, ...]:
    """Smoke-run several models on the same fold 0 of ``task``."""
    loaded = load_task_data(task) if data is None else data
    fold = _fold_zero(task, loaded)

    results = []
    for model_id in model_ids:
        factory = (factories or {}).get(model_id) or _default_factory(
            model_id, loaded, fold
        )
        results.append(
            smoke_fold(model_id, factory, loaded, fold, keep_pipeline=keep_pipelines)
        )
    return tuple(results)


def _default_factory(model_id: str, data: TaskData, fold: Any) -> Callable[[], Any]:
    """Build one model, handing the ensembles the fold's own subject groups.

    M6 and M7 run an inner CV over the training fold to get the out-of-fold
    probabilities their weights and threshold are chosen on, and that inner CV
    should be subject-aware for the same reason the outer one is. The groups have
    to be **the training fold's, in the training fold's row order** -- which only
    something holding the fold can supply, hence the closure rather than a
    parameter on the factory.
    """
    from src.models import estimators as est

    if model_id in {"M6", "M7"}:
        train_groups = np.asarray(data.groups)[np.asarray(fold.train_index, dtype=int)]

        def build_ensemble() -> Any:
            return est.make_ensemble(model_id, groups=train_groups)

        return build_ensemble

    def build() -> Any:
        return est.build_estimator(model_id)

    return build


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def smoke_frame(results: tuple[SmokeResult, ...] | list[SmokeResult]) -> Any:
    import pandas as pd

    return pd.DataFrame([result.as_row() for result in results])


def write_smoke_metrics(
    results: tuple[SmokeResult, ...] | list[SmokeResult],
    out_dir: str | Path | None = None,
    *,
    filename: str = SMOKE_METRICS_FILENAME,
    append: bool = True,
) -> Path:
    """Write the smoke table to ``outputs/04_models/``.

    Appends by default and de-duplicates on (model_id, task, fold), keeping the
    newest row. Phase 46 writes M1 and M2; Phase 47 adds M3 without erasing
    them, and a re-run of either replaces its own rows rather than doubling
    them.
    """
    import pandas as pd

    from src.utils.config import load_config
    from src.utils.io import ensure_dir, save_csv

    directory = ensure_dir(
        out_dir if out_dir is not None else load_config("paths").require("outputs.models")
    )
    path = Path(directory) / filename

    frame = smoke_frame(results)
    if append and path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["model_id", "task", "fold"], keep="last"
        ).reset_index(drop=True)
    frame = frame.sort_values(["task", "model_id", "fold"], kind="mergesort")
    return save_csv(frame, path)
