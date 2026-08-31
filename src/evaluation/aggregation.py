"""Recording-to-patient aggregation for the CirCor tracks (T68.4, T69.4).

CirCor labels a **patient**, but the model scores a **recording**: 942 patients
carry 3,163 recordings, taken at up to five auscultation locations. Every metric
in EXP-C1 and EXP-C2 therefore exists at two levels, and they answer different
questions:

* *recording level* -- "given this one recording, is there a murmur in it?"
* *patient level* -- "given everything we recorded from this child, should they
  be referred?"

The second is the clinically meaningful one and the one the 2022 Challenge scores,
so it cannot be left implicit. Three rules are evaluated rather than one, because
each encodes a different screening posture and the choice between them is a
decision about missed cases, not a technical detail:

``max``
    The patient's score is the highest any of their recordings reached. One
    convincing recording refers the patient. Most sensitive.

``mean``
    The patient's score is the average across their recordings. A single noisy
    recording cannot refer a patient on its own, and a murmur audible at only one
    location is diluted by the locations where it is not.

``any_present``
    The patient is positive if **any** recording was classified positive at that
    model's own operating point. This is a rule over decisions, not scores, so it
    inherits whatever threshold the model used.

``max`` and ``any_present`` are close but not identical: ``max`` re-thresholds a
pooled probability, ``any_present`` unions decisions already made. They diverge
exactly when a recording sits on the wrong side of the threshold from its
probability's contribution to the maximum -- which is the case a screening
programme cares about.

**Why no rule is declared the winner here.** All three are reported. Picking one
would be a clinical judgement this project is not positioned to make, and T15
exists precisely to show the reader the size of the difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "AGGREGATION_RULES",
    "AggregationError",
    "patient_frame",
    "aggregate_predictions",
    "evaluate_aggregations",
]

log = get_logger("evaluation.aggregation")

AGGREGATION_RULES: tuple[str, ...] = ("max", "mean", "any_present")


class AggregationError(RuntimeError):
    """The recordings cannot be aggregated to patients as requested."""


def _patient_of(record_uids: Any) -> Any:
    """Map every CirCor record_uid to its patient id, from the DA-07 map.

    Read from the fold map rather than parsed out of the uid string: the patient
    id is native in CirCor and is already the grouping key the folds were built
    on, so taking it from anywhere else risks a second, subtly different
    definition of "same patient" -- which is the leak rule 3 exists to prevent.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "subject_split_map.csv"
    frame = pd.read_csv(path)
    circor = frame[frame["task"].isin(("circor_murmur", "circor_outcome"))]
    lookup = dict(
        zip(
            circor["record_uid"].astype(str),
            circor["split_group"].astype(str),
            strict=True,
        )
    )
    wanted = [str(uid) for uid in record_uids]
    missing = sorted({uid for uid in wanted if uid not in lookup})
    if missing:
        raise AggregationError(
            str(len(missing))
            + " recording(s) have no patient in the DA-07 map, e.g. "
            + ", ".join(missing[:5])
        )
    return [lookup[uid] for uid in wanted]


def patient_frame(predictions: Any) -> Any:
    """Attach the patient id to a recording-level predictions frame."""
    import pandas as pd

    frame = pd.DataFrame(predictions).copy()
    frame["patient_id"] = _patient_of(frame["record_uid"])
    return frame


def aggregate_predictions(
    predictions: Any, *, rule: str, positive_label: int = 1, threshold: float = 0.5
) -> Any:
    """Collapse recording predictions to one row per (model, fold, patient).

    A patient's true label is taken from their recordings and asserted to be
    unanimous -- CirCor labels the patient, so a disagreement would mean the
    propagation in T69.2 had gone wrong rather than that the patient is ambiguous.
    """
    import pandas as pd

    if rule not in AGGREGATION_RULES:
        raise AggregationError(
            "unknown rule " + repr(rule) + "; expected one of " + ", ".join(AGGREGATION_RULES)
        )

    frame = patient_frame(predictions)
    proba_column = "proba_" + str(positive_label)
    has_proba = proba_column in frame.columns
    if rule in ("max", "mean") and not has_proba:
        raise AggregationError(
            "rule " + repr(rule) + " needs " + proba_column + ", which this run did not store"
        )

    rows: list[dict[str, Any]] = []
    for (model_id, fold_label), block in frame.groupby(["model_id", "fold_label"], sort=True):
        for patient, group in block.groupby("patient_id", sort=True):
            truth = {int(v) for v in group["y_true"]}
            if len(truth) != 1:
                raise AggregationError(
                    "patient "
                    + str(patient)
                    + " carries "
                    + str(len(truth))
                    + " different labels across its recordings; the patient-level "
                    "label did not propagate cleanly"
                )
            row: dict[str, Any] = {
                "model_id": model_id,
                "fold_label": fold_label,
                "patient_id": patient,
                "n_recordings": len(group),
                "rule": rule,
                "y_true": int(next(iter(truth))),
            }
            if rule == "max":
                score = float(np.max(group[proba_column].to_numpy(dtype=float)))
                row["score"] = score
                row["y_pred"] = int(score >= threshold)
            elif rule == "mean":
                score = float(np.mean(group[proba_column].to_numpy(dtype=float)))
                row["score"] = score
                row["y_pred"] = int(score >= threshold)
            else:  # any_present -- a union over decisions, not over scores
                positive = (group["y_pred"].to_numpy(dtype=int) == int(positive_label)).any()
                row["y_pred"] = int(bool(positive))
                row["score"] = (
                    float(np.max(group[proba_column].to_numpy(dtype=float)))
                    if has_proba
                    else float("nan")
                )
            rows.append(row)

    table = pd.DataFrame(rows)
    log.info(
        "%s: %d patient row(s) from %d recording row(s)",
        rule,
        len(table),
        len(frame),
    )
    return table


def evaluate_aggregations(
    predictions: Any,
    *,
    labels: Any,
    positive_label: int = 1,
    class_names: Any = None,
    threshold: float = 0.5,
) -> Any:
    """Per-fold patient-level metrics under every rule, plus the recording level.

    Returns one long frame with a ``level`` column (``recording`` or ``patient``)
    and a ``rule`` column, so T13/T14 and the T15 comparison are slices of a
    single table rather than three tables that could drift apart.
    """
    import pandas as pd

    from src.evaluation import metrics as mt

    declared = [int(v) for v in labels]
    binary = len(declared) == 2
    frame = pd.DataFrame(predictions)

    def score(y_true: Any, y_pred: Any, y_proba: Any = None) -> dict[str, float]:
        if binary:
            return mt.binary_metrics(
                y_true, y_pred, y_proba, labels=declared, positive_label=int(positive_label)
            )
        names = tuple(class_names) if class_names else None
        return mt.multiclass_metrics(
            y_true, y_pred, y_proba, labels=declared, class_names=names
        )

    rows: list[dict[str, Any]] = []
    proba_columns = [c for c in frame.columns if c.startswith("proba_")]

    for (model_id, fold_label), block in frame.groupby(["model_id", "fold_label"], sort=True):
        proba = block[proba_columns].to_numpy(dtype=float) if proba_columns else None
        row = {
            "model_id": model_id,
            "fold_label": fold_label,
            "level": "recording",
            "rule": "none",
            "n_units": len(block),
        }
        row.update(score(block["y_true"].to_numpy(int), block["y_pred"].to_numpy(int), proba))
        rows.append(row)

    for rule in AGGREGATION_RULES:
        try:
            patients = aggregate_predictions(
                frame, rule=rule, positive_label=positive_label, threshold=threshold
            )
        except AggregationError as error:
            log.warning("rule %s skipped: %s", rule, error)
            continue
        for (model_id, fold_label), block in patients.groupby(
            ["model_id", "fold_label"], sort=True
        ):
            row = {
                "model_id": model_id,
                "fold_label": fold_label,
                "level": "patient",
                "rule": rule,
                "n_units": len(block),
            }
            # Only the positive-class score survives aggregation, so multiclass
            # probability metrics (OvR AUC) cannot be computed at patient level.
            row.update(
                score(block["y_true"].to_numpy(int), block["y_pred"].to_numpy(int), None)
            )
            rows.append(row)

    return pd.DataFrame(rows)
