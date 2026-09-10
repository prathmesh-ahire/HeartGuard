"""EXP-F3 -- leave-one-sub-collection-out on the binary track.

**The question this answers is the deployment question**, and nothing else in
this project answers it. Phase 76 found that a predictor using no audio at all
-- one that maps a PhysioNet recording's sub-collection to that collection's
most common class -- reaches balanced accuracy 0.719 on the binary task. The
collections were recorded with different equipment in different rooms, so part
of what any model fitted on the pooled corpus learns is the **acquisition
signature** rather than cardiac content. In the field there is no acquisition
signature to lean on, so any performance resting on it evaporates.

The standard DA-07 protocol cannot detect this. Its 25 folds are subject-grouped
but every fold's training set contains records from **all six** collections, so
a model that has learned the six signatures is never tested on an unseen one.
This experiment holds out a whole collection at a time:

    train on 5 collections -> test on the 6th, six times

Subjects never span collections (verified: 0 of 857), so this is subject-disjoint
by construction and rule 3 holds without any extra grouping.

**Read the per-fold rows, not just the mean.** The six collections are wildly
unequal -- ``training-e`` alone is 2,141 of the 3,240 records and is 8.5%
abnormal, while ``training-c`` is 31 records and 77.4% abnormal. Holding out
``training-e`` leaves a training set with a completely different class balance;
holding out ``training-c`` gives a 31-record test fold. Two of the six folds are
therefore noisy in opposite directions, and a single averaged number across them
hides that. The per-fold table is the deliverable; the mean is a convenience.

**What a drop here does and does not mean.** A large drop is evidence that the
model depends on acquisition. It is *not* proof, because holding out a
collection also changes the class balance and the patient population, exactly
the confounds EXP-D1 already documents. The source-only baseline is computed on
the same folds for that reason: it is what pure acquisition-plus-prevalence
scores, and the gap between the model and it is the part that is not explained
by either.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "EXP_ID",
    "TASK",
    "SCHEME",
    "holdout_records",
    "build_folds",
    "run_holdout",
    "compare_with_pooled",
]

log = get_logger("evaluation.source_holdout")

EXP_ID = "EXP-F3"
TASK = "binary"
SCHEME = "leave_one_source_out"

#: The column that names the sub-collection.
SOURCE_COLUMN = "subset"

METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "accuracy",
)


def holdout_records() -> Any:
    """The D1 binary records with their sub-collection."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    master = pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv",
        low_memory=False,
    )
    rows = master[
        (master["dataset_source"] == "D1")
        & (master["use_in_supervised"].astype(bool))
        & (master["binary_label"].notna())
    ].copy()
    rows["y"] = rows["binary_label"].astype(int)
    rows = rows.sort_values("record_uid", kind="mergesort").reset_index(drop=True)

    spanning = rows.groupby("split_group")[SOURCE_COLUMN].nunique()
    if int((spanning > 1).sum()):
        raise RuntimeError(
            str(int((spanning > 1).sum())) + " subject(s) span more than one "
            "sub-collection; leave-one-source-out would leak them"
        )
    return rows[["record_uid", "y", "split_group", SOURCE_COLUMN]]


def build_folds(records: Any = None) -> tuple[Any, ...]:
    """One fold per sub-collection: that collection is the test set."""
    from src.evaluation.cv import Fold

    frame = holdout_records() if records is None else records
    uids = frame["record_uid"].astype(str).to_numpy()
    groups = frame["split_group"].astype(str).to_numpy()
    sources = frame[SOURCE_COLUMN].astype(str).to_numpy()

    folds = []
    for index, source in enumerate(sorted(set(sources.tolist()))):
        held = sources == source
        folds.append(
            Fold(
                task=TASK,
                scheme=SCHEME,
                repeat=0,
                fold=index,
                train_uids=tuple(uids[~held]),
                test_uids=tuple(uids[held]),
                train_groups=tuple(groups[~held]),
                test_groups=tuple(groups[held]),
            )
        )
        overlap = set(groups[held]) & set(groups[~held])
        if overlap:
            raise RuntimeError(source + ": " + str(len(overlap)) + " subject(s) on both sides")
    return tuple(folds)


def run_holdout(
    *, models: tuple[str, ...] | None = None, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Fit each model six times, holding out one sub-collection each time."""
    import numpy as np
    import pandas as pd

    from src.evaluation import cv as cv_module
    from src.evaluation import metrics as mt
    from src.evaluation import source_baseline as sb
    from src.evaluation.experiment import Experiment
    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models import smoke as sm

    chosen = tuple(models) if models else Experiment.load("EXP-A1").models
    data = sm.load_task_data(TASK)
    records = holdout_records()

    order = {uid: index for index, uid in enumerate(data.record_uids)}
    records = records[records["record_uid"].astype(str).isin(order)].copy()
    records = records.sort_values(
        "record_uid", key=lambda s: s.map(order), kind="mergesort"
    ).reset_index(drop=True)

    folds = cv_module.resolve_folds(build_folds(records), data.record_uids)
    source_of = dict(zip(records["record_uid"].astype(str), records[SOURCE_COLUMN], strict=True))

    rows: list[dict[str, Any]] = []
    for model_id in chosen:
        def factory(model_id: str = model_id) -> Any:
            return pl.build_pipeline(
                est.build_estimator(model_id), y=data.y, n_features=data.X.shape[1]
            )

        started = time.perf_counter()
        result = cv_module.run_cv(
            factory, data.X, data.y, data.groups, folds, task=TASK
        )
        elapsed = time.perf_counter() - started

        for fold_result in result.folds:
            held = source_of[str(fold_result.test_uids[0])]
            proba = None
            if fold_result.y_proba is not None and 1 in fold_result.classes:
                proba = fold_result.y_proba[:, fold_result.classes.index(1)]
            scores = mt.binary_metrics(
                fold_result.y_true, fold_result.y_pred, proba,
                labels=[0, 1], positive_label=1,
            )
            rows.append(
                {
                    "exp_id": EXP_ID,
                    "model_id": model_id,
                    "held_out_source": held,
                    "fold": fold_result.fold,
                    "n_train": fold_result.n_train,
                    "n_test": len(fold_result.test_uids),
                    "test_abnormal_rate": float(np.mean(fold_result.y_true)),
                    "train_abnormal_rate": float(
                        np.mean(data.y[np.asarray(folds[fold_result.fold].train_index)])
                    ),
                    **{m: float(scores[m]) for m in METRICS},
                }
            )
        log.info("%s: %d fold(s) in %.1f s", model_id, len(result.folds), elapsed)

    per_fold = pd.DataFrame(rows)
    baseline = sb.source_only_baseline(records, folds, labels=[0, 1])
    baseline["held_out_source"] = [
        source_of[str(folds[int(f)].test_uids[0])] for f in baseline["fold"]
    ]
    return {"per_fold": per_fold, "baseline": baseline, "n_folds": len(folds)}


def compare_with_pooled(per_fold: Any) -> Any:
    """Each model under leave-one-source-out against its pooled 25-fold result.

    The pooled numbers are read from EXP-A1's committed aggregate metrics, never
    recomputed: the whole claim is that the same model, same features and same
    seed score differently only because of how the folds were cut.
    """
    import numpy as np
    import pandas as pd

    from src.evaluation.experiment import Experiment

    pooled = pd.read_csv(
        Experiment.load("EXP-A1").output_dir() / "aggregate_metrics.csv"
    ).set_index("model_id")

    rows = []
    for model_id, block in per_fold.groupby("model_id", sort=False):
        row: dict[str, Any] = {
            "model_id": model_id,
            "n_folds_holdout": len(block),
        }
        for metric in METRICS:
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[metric + "_holdout"] = float(np.mean(finite)) if finite.size else float("nan")
            row[metric + "_holdout_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
            column = metric + "_mean"
            if model_id in pooled.index and column in pooled.columns:
                row[metric + "_pooled"] = float(pooled.loc[model_id, column])
                row[metric + "_drop"] = row[metric + "_pooled"] - row[metric + "_holdout"]
        rows.append(row)
    return pd.DataFrame(rows)
