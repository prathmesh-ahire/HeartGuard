"""How much of a result is available from the RECORDING SOURCE alone.

PhysioNet's six sub-collections behave like six different datasets -- class
balance runs 8.5% to 71.4% abnormal, and all 20 top features by pooled Cohen's d
reverse sign between them. That has a consequence nobody had measured until
Phase 76: a model can score on a PhysioNet task by recognising *which collection
a recording came from* rather than anything clinical.

This module measures exactly that, for any task, with a predictor that sees no
audio whatsoever: a lookup from sub-collection to that collection's most common
class, **fitted inside each training fold** and applied to the test fold. It is
the direct analogue of the "always predict normal scores 0.6941 accuracy"
baseline that every PASCAL B number must be quoted against.

The two tracks it has been run on land in opposite places, which is why both are
reported:

* **binary** -- the source-only predictor reaches balanced accuracy 0.719 but
  sensitivity only 0.493. M1 scores 0.843 / 0.848. The headline binary result
  therefore carries a large amount of information the recording source does not,
  and the baseline is what makes that statement checkable rather than assumed.
* **diagnosis** -- the source-only predictor reaches macro-F1 0.635 and **beats
  every one of the six models** (best 0.620). That track's labels are nearly
  nested inside the sub-collections, so it does not establish audio-based
  diagnosis classification at all.

A baseline this cheap should be run against any new PhysioNet task before its
numbers are written up.
"""

from __future__ import annotations

from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "SOURCE_COLUMN",
    "source_only_baseline",
    "binary_source_baseline",
    "baseline_table",
]

log = get_logger("evaluation.source_baseline")

#: The metadata column that names the sub-collection. Not derived from the uid:
#: the uid format is a convention, the column is the audited fact.
SOURCE_COLUMN = "subset"


def _master() -> Any:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    return pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv",
        low_memory=False,
    )


def source_only_baseline(records: Any, folds: Any, *, labels: Any = None) -> Any:
    """Per-fold scores of the sub-collection lookup.

    ``records`` needs ``record_uid``, ``y`` and :data:`SOURCE_COLUMN`. The lookup
    is rebuilt inside every training fold -- a lookup fitted once on all the data
    would be the same leak this baseline exists to detect.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

    frame = pd.DataFrame(records).reset_index(drop=True)
    for column in ("record_uid", "y", SOURCE_COLUMN):
        if column not in frame.columns:
            raise KeyError("records has no " + column + " column")

    position = {str(uid): index for index, uid in enumerate(frame["record_uid"].astype(str))}
    y = frame["y"].to_numpy(dtype=int)
    classes = sorted(set(y.tolist())) if labels is None else list(labels)
    binary = len(classes) == 2

    rows = []
    for fold in folds:
        train = [position[str(u)] for u in fold.train_uids if str(u) in position]
        test = [position[str(u)] for u in fold.test_uids if str(u) in position]
        if not train or not test:
            continue
        block = frame.iloc[train]
        lookup = (
            block.groupby(SOURCE_COLUMN)["y"]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )
        fallback = block["y"].value_counts().idxmax()
        predicted = np.asarray(
            [lookup.get(s, fallback) for s in frame.iloc[test][SOURCE_COLUMN]], dtype=int
        )
        truth = y[test]

        row: dict[str, Any] = {
            "repeat": int(getattr(fold, "repeat", 0)),
            "fold": int(fold.fold),
            "n_test": len(test),
            "n_sources_seen": int(block[SOURCE_COLUMN].nunique()),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "macro_f1": float(
                f1_score(truth, predicted, average="macro", labels=classes, zero_division=0)
            ),
            "accuracy": float((truth == predicted).mean()),
        }
        if binary:
            row["sensitivity"] = float(
                recall_score(truth, predicted, pos_label=classes[-1], zero_division=0)
            )
            row["specificity"] = float(
                recall_score(truth, predicted, pos_label=classes[0], zero_division=0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def binary_source_baseline() -> Any:
    """The D1 binary track's source-only baseline, over the DA-07 25-fold map."""
    from src.evaluation import cv as cv_module

    master = _master()
    records = master[
        (master["dataset_source"] == "D1")
        & (master["use_in_supervised"].astype(bool))
        & (master["binary_label"].notna())
    ].copy()
    records["y"] = records["binary_label"].astype(int)
    records = records[["record_uid", "y", SOURCE_COLUMN]]

    frame = source_only_baseline(
        records, cv_module.load_folds("binary"), labels=[0, 1]
    )
    log.info(
        "binary source-only baseline: sensitivity %.4f, balanced accuracy %.4f "
        "over %d fold(s)",
        float(frame["sensitivity"].mean()),
        float(frame["balanced_accuracy"].mean()),
        len(frame),
    )
    return frame


def baseline_table() -> Any:
    """One row per track: the source-only score and what the best model scored.

    The comparison column is read from each track's committed aggregate metrics,
    never retyped -- the whole point of this table is that the two numbers sit
    beside each other and can be checked against their sources.
    """
    import pandas as pd

    from src.evaluation.experiment import Experiment

    rows: list[dict[str, Any]] = []

    binary = binary_source_baseline()
    best = pd.read_csv(Experiment.load("EXP-A1").output_dir() / "aggregate_metrics.csv")
    top = best.sort_values("sensitivity_mean", ascending=False).iloc[0]
    rows.append(
        {
            "track": "binary (EXP-A1)",
            "n_folds": len(binary),
            "primary_metric": "sensitivity",
            "source_only": float(binary["sensitivity"].mean()),
            "best_model": str(top["model_id"]),
            "best_model_score": float(top["sensitivity_mean"]),
            "source_only_balanced_accuracy": float(binary["balanced_accuracy"].mean()),
            "best_model_balanced_accuracy": float(top["balanced_accuracy_mean"]),
            "model_beats_source_only": bool(
                float(top["sensitivity_mean"]) > float(binary["sensitivity"].mean())
            ),
        }
    )

    try:
        from src.evaluation import diagnosis_track as dt

        records = dt.diagnosis_records()
        master = _master()
        source_of = dict(
            zip(master["record_uid"].astype(str), master[SOURCE_COLUMN].astype(str), strict=True)
        )
        records = records.copy()
        records[SOURCE_COLUMN] = [source_of[u] for u in records["record_uid"].astype(str)]
        diagnosis = source_only_baseline(records, dt.load_folds())

        aggregate = pd.read_csv(
            dt.split_map_path().parent.parent
            / "07_multiclass_results"
            / "EXP-G1"
            / "aggregate_metrics.csv"
        )
        best_dx = aggregate.sort_values("macro_f1_mean", ascending=False).iloc[0]
        rows.append(
            {
                "track": "diagnosis (EXP-G1)",
                "n_folds": len(diagnosis),
                "primary_metric": "macro_f1",
                "source_only": float(diagnosis["macro_f1"].mean()),
                "best_model": str(best_dx["model_id"]),
                "best_model_score": float(best_dx["macro_f1_mean"]),
                "source_only_balanced_accuracy": float(
                    diagnosis["balanced_accuracy"].mean()
                ),
                "best_model_balanced_accuracy": float(best_dx["balanced_accuracy_mean"]),
                "model_beats_source_only": bool(
                    float(best_dx["macro_f1_mean"]) > float(diagnosis["macro_f1"].mean())
                ),
            }
        )
    except (FileNotFoundError, ImportError, KeyError) as error:
        log.warning("diagnosis track baseline unavailable: %s", error)

    return pd.DataFrame(rows)
