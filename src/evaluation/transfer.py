"""EXP-D1 -- adult-to-paediatric cross-dataset transfer (Phase 71).

The finalized binary model is fitted on all 3,240 labelled PhysioNet 2016
recordings and applied, without a single parameter changing, to all 3,163 CirCor
recordings scored against the clinical ``Outcome`` label.

**Read the framing before the number.** This is *not* a like-for-like
generalization test and must never be written up as one:

* PhysioNet 2016 has a **median age of 25** with **0.27% of aged records under
  18** (measured from its own online appendix, 2,199 records carry an age), and
  its abnormal classes are adult degenerative disease -- CAD 287, MVP 134.
* CirCor 2022 is **~98% paediatric**: Child 598, Infant 191, Adolescent 66,
  Neonate 6, against 7 young adults.

A large drop is the **expected consequence of that mismatch**, not evidence that
PV-MEPCG fails to generalize. A second, independent cause is already on the
record: PhysioNet's six sub-collections behave like six different datasets, and
all 20 of its top features by pooled Cohen's d reverse sign between sources (see
the 2026-08-27 entry in ``Docs/note.md``). Describe the result as *"cross-dataset
transfer from an adult cohort to a predominantly paediatric cohort"*. Never as
*"cross-dataset generalization"* unqualified.

**The ordering in this module is deliberate.** The population profile is measured
and written to disk *before* any prediction is made, so the framing cannot be
retrofitted to whatever number comes out. :func:`write_population_metadata`
refuses to write anything that looks like a metric, and T71.7 checks the two
files' timestamps against each other.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "METADATA_FILENAME",
    "TransferError",
    "physionet_population",
    "circor_population",
    "write_population_metadata",
    "transfer_predictions",
    "evaluate_transfer",
    "degradation",
]

log = get_logger("evaluation.transfer")

METADATA_FILENAME = "population_mismatch.json"

#: Any of these appearing as a key in the pre-registered metadata means a metric
#: leaked into the file that is supposed to be written before one exists.
_METRIC_TOKENS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "recall",
    "precision",
    "f1",
    "auc",
    "mcc",
    "brier",
    "ece",
    "score",
    "delta",
    "drop",
    "degradation",
)


class TransferError(RuntimeError):
    """The transfer experiment cannot be run or is not being run honestly."""


# ---------------------------------------------------------------------------
# the populations, measured rather than quoted
# ---------------------------------------------------------------------------


def physionet_population() -> dict[str, Any]:
    """Age and diagnosis profile of the PhysioNet training set, from its appendix.

    ``Online_Appendix_training_set.csv`` covers 3,153 of the 3,240 records and
    carries an age for 2,199 of them. The missing ages are stated rather than
    imputed: an unstated denominator is how "median age 25" becomes a claim about
    a corpus rather than about the two thirds of it that reported an age.
    """
    import pandas as pd

    from src.utils.config import load_config

    root = Path(load_config("paths").require("dataset.d1_physionet.root"))
    path = root / "annotations" / "Online_Appendix_training_set.csv"
    if not path.is_file():
        raise TransferError(str(path) + " is missing; dataset/ must be present")

    frame = pd.read_csv(path)
    ages = pd.to_numeric(frame["Age (year)"], errors="coerce").dropna()
    diagnosis = frame["Diagnosis"].astype(str).str.strip().value_counts().to_dict()
    return {
        "dataset": "D1 PhysioNet 2016",
        "source_file": str(path).replace("\\", "/"),
        "n_records_in_appendix": len(frame),
        "n_records_with_age": len(ages),
        "age_median_years": float(ages.median()),
        "age_mean_years": float(ages.mean()),
        "age_min_years": float(ages.min()),
        "age_max_years": float(ages.max()),
        "n_under_18": int((ages < 18).sum()),
        "share_under_18": float((ages < 18).mean()),
        "diagnosis_counts": {str(k): int(v) for k, v in diagnosis.items()},
        "cohort": "adult",
    }


def circor_population() -> dict[str, Any]:
    """Age-band profile of CirCor, from ``training_data.csv``.

    CirCor records a band (Neonate / Infant / Child / Adolescent / Young Adult)
    rather than a number, so the two corpora cannot be compared on a single age
    scale. That is itself part of the mismatch and is stated, not papered over.
    """
    import pandas as pd

    from src.utils.config import load_config

    path = Path(load_config("paths").require("dataset.d4_circor.demographics_csv"))
    if not path.is_file():
        raise TransferError(str(path) + " is missing; dataset/ must be present")

    frame = pd.read_csv(path)
    bands = frame["Age"].astype(str).where(frame["Age"].notna(), "unrecorded")
    counts = bands.value_counts().to_dict()
    recorded = int(frame["Age"].notna().sum())
    paediatric = sum(
        int(v) for k, v in counts.items() if k in ("Neonate", "Infant", "Child", "Adolescent")
    )
    return {
        "dataset": "D4 CirCor 2022",
        "source_file": str(path).replace("\\", "/"),
        "n_patients": len(frame),
        "n_with_age_band": recorded,
        "age_band_counts": {str(k): int(v) for k, v in counts.items()},
        "n_paediatric": paediatric,
        "share_paediatric_of_recorded": float(paediatric / recorded) if recorded else float("nan"),
        "age_scale": "band, not years -- not comparable to PhysioNet's numeric age",
        "cohort": "predominantly paediatric",
    }


def write_population_metadata(
    directory: str | Path,
    *,
    experiment_id: str = "EXP-D1",
    model_manifest: dict[str, Any] | None = None,
) -> Path:
    """T71.1 -- record the mismatch BEFORE any prediction exists.

    Refuses to write a payload containing anything that looks like a metric. The
    whole value of this file is that it was written first; a metric inside it
    would mean it was not.
    """
    from src.utils.io import save_json

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "written_utc": datetime.now(UTC).isoformat(),
        "written_before_any_metric": True,
        "design": (
            "Adult-to-paediatric cross-dataset transfer. The model is fitted on "
            "PhysioNet 2016 (adult) and applied unchanged to CirCor 2022 "
            "(predominantly paediatric), scored against CirCor's clinical "
            "Outcome label."
        ),
        "framing_rule": (
            "A large drop is the EXPECTED consequence of the population "
            "mismatch recorded here and is NOT evidence that PV-MEPCG fails to "
            "generalize. Describe the result as 'cross-dataset transfer from an "
            "adult cohort to a predominantly paediatric cohort'; never as "
            "'cross-dataset generalization' unqualified."
        ),
        "second_known_cause": (
            "PhysioNet's six sub-collections behave like six different "
            "datasets: class balance runs 8.5% to 71.4% abnormal and all 20 top "
            "features by pooled Cohen's d reverse sign between sources. Any "
            "model fitted on the pooled corpus inherits that heterogeneity, so "
            "the drop has at least two causes and must not be attributed wholly "
            "to age."
        ),
        "retuning_allowed": False,
        "retuning_performed": False,
        "train": physionet_population(),
        "test": circor_population(),
    }
    if model_manifest is not None:
        payload["model"] = {
            "selected_model_id": model_manifest.get("selected_model_id"),
            "estimator_class": model_manifest.get("estimator_class"),
            "hyperparameters": model_manifest.get("hyperparameters"),
            "hyperparameter_source": model_manifest.get("hyperparameter_source"),
            "n_records_fitted": model_manifest.get("n_records_fitted"),
            "fitted_on_task": model_manifest.get("task"),
        }

    offenders = _metric_like_keys(payload)
    if offenders:
        raise TransferError(
            "the pre-registered metadata carries metric-like key(s) "
            + ", ".join(sorted(offenders))
            + "; this file must be written before any metric exists"
        )

    path = save_json(payload, target / METADATA_FILENAME)
    log.info("population mismatch recorded first: %s", path)
    return Path(path)


def _metric_like_keys(payload: Any, prefix: str = "") -> set[str]:
    """Every key anywhere in the payload whose name reads like a metric."""
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = str(key)
            lowered = name.lower()
            # `diagnosis_counts` holds class names, not metrics; and the framing
            # text legitimately contains the word "generalize".
            if isinstance(value, (int, float)) and any(t in lowered for t in _METRIC_TOKENS):
                found.add(prefix + name)
            found |= _metric_like_keys(value, prefix + name + ".")
    elif isinstance(payload, list):
        for item in payload:
            found |= _metric_like_keys(item, prefix)
    return found


# ---------------------------------------------------------------------------
# the transfer itself
# ---------------------------------------------------------------------------


def transfer_predictions(
    *,
    task: str = "circor_outcome",
    model_task: str = "binary",
    model_id: str = "final",
    fold_label: str = "external",
) -> tuple[Any, dict[str, Any]]:
    """Apply the saved PhysioNet model to every CirCor recording, unchanged.

    Returns the predictions frame and the model's manifest. The manifest is
    returned rather than re-read so a caller cannot accidentally check the
    hyperparameters of a different file from the one that produced the numbers.
    """
    import pandas as pd

    from src.feature_extraction.registry import feature_names
    from src.models.registry import load_model
    from src.models.smoke import load_task_data

    names = feature_names()
    model, manifest = load_model(model_task, model_id, feature_names=names)
    if str(manifest.get("task")) != model_task:
        raise TransferError(
            "the saved model reports task "
            + repr(manifest.get("task"))
            + ", not "
            + repr(model_task)
        )

    data = load_task_data(task)
    proba = np.asarray(model.predict_proba(data.X), dtype=float)
    predicted = np.asarray(model.predict(data.X), dtype=int)

    frame = pd.DataFrame(
        {
            "exp_id": "EXP-D1",
            "model_id": str(manifest.get("selected_model_id") or model_id),
            "fold_label": fold_label,
            "record_uid": list(data.record_uids),
            "y_true": data.y.astype(int),
            "y_pred": predicted,
        }
    )
    for index in range(proba.shape[1]):
        frame["proba_" + str(index)] = proba[:, index]

    log.info(
        "transfer: %d CirCor recording(s) scored by %s fitted on %s record(s)",
        len(frame),
        manifest.get("selected_model_id"),
        manifest.get("n_records_fitted"),
    )
    return frame, manifest


def evaluate_transfer(predictions: Any) -> Any:
    """Recording-level and patient-level metrics under every aggregation rule.

    Reuses :mod:`src.evaluation.aggregation` rather than re-deriving the
    patient-level collapse, so EXP-D1 and EXP-C2 answer "what does this patient
    look like" the same way. There is one fold here -- the whole of CirCor is a
    single external test set -- so there is no fold variance to report and none
    is invented.
    """
    from src.evaluation.aggregation import evaluate_aggregations

    scored = evaluate_aggregations(predictions, labels=[0, 1], positive_label=1)
    scored.insert(0, "exp_id", "EXP-D1")
    return scored


def degradation(external: Any, *, in_domain_exp: str = "EXP-A2", model_id: str = "M1") -> Any:
    """T71.5 -- the signed drop against the in-domain nested-CV result.

    The in-domain side is EXP-A2's 25-fold nested cross-validation on PhysioNet,
    which is the only defensible estimate of this model's in-domain performance;
    the refitted final model has no honest score on the rows it was fitted on.
    The comparison is therefore between a 25-fold mean and a single external
    evaluation, and the frame says so in an ``n_folds`` column rather than
    letting a reader assume two like quantities were subtracted.
    """
    import pandas as pd

    from src.evaluation.experiment import load_per_fold_metrics

    in_domain = load_per_fold_metrics(in_domain_exp)
    block = in_domain[in_domain["model_id"] == model_id]
    if block.empty:
        raise TransferError(
            in_domain_exp + " has no rows for model " + model_id + "; cannot quantify the drop"
        )

    metrics = ("sensitivity", "specificity", "balanced_accuracy", "roc_auc", "accuracy", "f1")
    recording = external[external["level"] == "recording"]
    rows: list[dict[str, Any]] = []
    for _, row in external.iterrows():
        for metric in metrics:
            if metric not in block.columns or metric not in external.columns:
                continue
            reference = float(np.mean(np.asarray(block[metric], dtype=float)))
            value = float(row[metric])
            rows.append(
                {
                    "metric": metric,
                    "level": row["level"],
                    "rule": row["rule"],
                    "in_domain_exp": in_domain_exp,
                    "in_domain_mean": reference,
                    "in_domain_n_folds": len(block),
                    "external_value": value,
                    "external_n_folds": 1,
                    "delta": value - reference,
                    "relative_drop": (
                        (reference - value) / reference if reference else float("nan")
                    ),
                }
            )
    table = pd.DataFrame(rows)
    if len(recording):
        worst = table[(table["level"] == "recording") & (table["metric"] == "balanced_accuracy")]
        if len(worst):
            log.info(
                "balanced accuracy %.4f in domain -> %.4f external (%+.4f)",
                float(worst["in_domain_mean"].iloc[0]),
                float(worst["external_value"].iloc[0]),
                float(worst["delta"].iloc[0]),
            )
    return table


def read_metadata(directory: str | Path) -> dict[str, Any]:
    """The pre-registered population metadata, as written."""
    path = Path(directory) / METADATA_FILENAME
    if not path.is_file():
        raise TransferError(str(path) + " is missing; it must be written before any metric")
    return dict(json.loads(path.read_text(encoding="utf-8")))
