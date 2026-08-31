"""Auscultation-location stratification for the CirCor tracks (EXP-C3, Phase 70).

CirCor records each patient at up to five chest positions. Four of them are the
standard valve areas -- aortic (AV), mitral (MV), pulmonary (PV) and tricuspid
(TV) -- and the fifth, ``Phc``, is an "other" position used four times in the
entire corpus.

**Why this analysis exists.** A murmur is a *local* sound. It is generated at one
valve and radiates unevenly, so it can be plainly audible at the pulmonary area
and inaudible at the mitral area of the same child. But CirCor labels the
*patient*, and Phase 69 propagates that label to every one of that patient's
recordings (see ``circor_label_propagation.md``). Every recording of a murmur
patient is therefore labelled ``Present`` whether or not a murmur is audible in
it. If that propagation costs anything, it must cost *unevenly across locations*
-- worst where the murmur is least often audible. Stratifying by location is how
that shows up as a number instead of a caveat.

**Phc is not reported as a result.** Four recordings, spread over five folds, is
one recording or fewer per fold. A sensitivity computed on that is not a weak
estimate, it is arithmetic noise: a single record flips it by up to 1.0. It is
carried through every table with ``reported=False`` and a stated reason, so a
reader can see it was excluded deliberately rather than lost.

The location itself comes from ``metadata_master.recording_location``, not from
the ``record_uid`` string, because a patient recorded twice at one position gets
``..._AV_1`` / ``..._AV_2`` and a naive suffix split reads the location of 43 of
the 3,163 recordings as ``1``, ``2`` or ``3``. The suffix is used only as an
independent cross-check.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "LOCATIONS",
    "REPORTED_LOCATIONS",
    "EXCLUDED_LOCATIONS",
    "MIN_RECORDS_PER_LOCATION",
    "LocationError",
    "location_table",
    "attach_locations",
    "stratified_metrics",
    "most_audible_agreement",
]

log = get_logger("evaluation.location")

#: Every location in the corpus, in the order the tables print them.
LOCATIONS: tuple[str, ...] = ("AV", "MV", "PV", "TV", "Phc")

#: The ones an actual result may be quoted from.
REPORTED_LOCATIONS: tuple[str, ...] = ("AV", "MV", "PV", "TV")

#: Carried, flagged, never reported. See the module docstring.
EXCLUDED_LOCATIONS: tuple[str, ...] = ("Phc",)

#: Below this many recordings a per-location metric is not an estimate. Chosen
#: to match the Phase 76 class-merge floor so the project has one rule, not two.
MIN_RECORDS_PER_LOCATION = 30

#: ``D4_training_data_<patient>_<LOC>`` with an optional ``_<n>`` when a patient
#: was recorded more than once at the same position.
_UID = re.compile(r"^D4_training_data_(?P<patient>\d+)_(?P<location>[A-Za-z]+)(?:_\d+)?$")


class LocationError(RuntimeError):
    """The recordings cannot be stratified by auscultation location."""


def location_table() -> Any:
    """``record_uid -> (patient, location)`` for every CirCor recording.

    The location is read from the audit's ``metadata_master.csv`` and then
    checked against the uid, which encodes it independently. A disagreement
    means one of the two drifted and is raised rather than resolved silently --
    picking a winner here would decide the whole analysis by accident.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv"
    if not path.is_file():
        raise LocationError(str(path) + " is missing; run scripts/01_run_dataset_audit.py")

    frame = pd.read_csv(path, low_memory=False)
    circor = frame[frame["dataset_source"] == "D4"][
        ["record_uid", "subject_id", "recording_location"]
    ].copy()
    if circor.empty:
        raise LocationError("metadata_master.csv holds no D4 rows")

    parsed = circor["record_uid"].astype(str).str.extract(_UID)
    if parsed["location"].isna().any():
        unparsed = circor.loc[parsed["location"].isna(), "record_uid"].head(5).tolist()
        raise LocationError(
            str(int(parsed["location"].isna().sum()))
            + " CirCor uid(s) do not match the expected pattern, e.g. "
            + ", ".join(unparsed)
        )
    disagree = parsed["location"].to_numpy() != circor["recording_location"].to_numpy()
    if disagree.any():
        examples = circor.loc[disagree, "record_uid"].head(5).tolist()
        raise LocationError(
            str(int(disagree.sum()))
            + " recording(s) disagree between metadata_master.recording_location "
            "and the location encoded in the uid, e.g. " + ", ".join(examples)
        )

    circor = circor.rename(columns={"recording_location": "location"})
    circor["patient_id"] = parsed["patient"].to_numpy()
    return circor[["record_uid", "patient_id", "location"]].reset_index(drop=True)


def attach_locations(predictions: Any) -> Any:
    """Add ``location`` and ``patient_id`` to a recording-level predictions frame."""
    import pandas as pd

    frame = pd.DataFrame(predictions).copy()
    table = location_table().set_index("record_uid")
    lookup = table["location"]
    frame["location"] = frame["record_uid"].astype(str).map(lookup)
    frame["patient_id"] = frame["record_uid"].astype(str).map(table["patient_id"])
    missing = frame["location"].isna()
    if missing.any():
        raise LocationError(
            str(int(missing.sum()))
            + " prediction row(s) have no auscultation location, e.g. "
            + ", ".join(frame.loc[missing, "record_uid"].astype(str).head(5))
        )
    return frame


def _reported(location: str, n_records: int) -> tuple[bool, str]:
    """Whether this location's numbers may be quoted, and why not if not."""
    if location in EXCLUDED_LOCATIONS:
        return False, (
            "excluded: "
            + str(n_records)
            + " recording(s) in the whole corpus, under one per fold -- a single "
            "record moves any rate by up to 1.0, so this is arithmetic noise, "
            "not a weak estimate"
        )
    if n_records < MIN_RECORDS_PER_LOCATION:
        return False, (
            "excluded: "
            + str(n_records)
            + " recording(s), below the "
            + str(MIN_RECORDS_PER_LOCATION)
            + "-record floor this project uses for a reportable subgroup"
        )
    return True, ""


def stratified_metrics(
    predictions: Any,
    *,
    labels: Any,
    positive_label: int = 1,
    class_names: Any = None,
) -> Any:
    """Per-fold metrics for every (model, fold, location).

    Folds are kept rather than pooled: the fold map is the only thing keeping a
    patient out of their own training set, and a metric computed over pooled
    out-of-fold predictions has no variance to report and no fold structure for
    Phase 82's paired tests to consume.
    """
    import pandas as pd

    from src.evaluation import metrics as mt

    declared = [int(v) for v in labels]
    binary = len(declared) == 2
    frame = attach_locations(predictions)
    proba_columns = [c for c in frame.columns if c.startswith("proba_")]

    counts = frame.drop_duplicates("record_uid")["location"].value_counts().to_dict()

    def score(y_true: Any, y_pred: Any, y_proba: Any) -> dict[str, float]:
        if binary:
            return mt.binary_metrics(
                y_true, y_pred, y_proba, labels=declared, positive_label=int(positive_label)
            )
        names = tuple(class_names) if class_names else None
        return mt.multiclass_metrics(y_true, y_pred, y_proba, labels=declared, class_names=names)

    rows: list[dict[str, Any]] = []
    for (model_id, fold_label, location), block in frame.groupby(
        ["model_id", "fold_label", "location"], sort=True
    ):
        n_total = int(counts.get(location, 0))
        reported, reason = _reported(str(location), n_total)
        row: dict[str, Any] = {
            "model_id": model_id,
            "fold_label": fold_label,
            "location": location,
            "n_units": len(block),
            "n_records_corpus": n_total,
            "reported": reported,
            "exclusion_reason": reason,
        }
        present = sorted({int(v) for v in block["y_true"]})
        # A fold-location cell can legitimately hold one class only (Phc holds
        # four recordings in total). Probability metrics are undefined there and
        # are left NaN rather than being computed on a degenerate input.
        proba = (
            block[proba_columns].to_numpy(dtype=float)
            if proba_columns and len(present) > 1
            else None
        )
        try:
            row.update(score(block["y_true"].to_numpy(int), block["y_pred"].to_numpy(int), proba))
        except ValueError as error:  # pragma: no cover - degenerate cells only
            log.warning("%s %s %s: %s", model_id, fold_label, location, error)
            continue
        row["n_classes_present"] = len(present)
        rows.append(row)

    table = pd.DataFrame(rows)
    log.info(
        "%d (model, fold, location) cell(s) over %d location(s); reportable: %s",
        len(table),
        table["location"].nunique() if len(table) else 0,
        ", ".join(sorted(set(table.loc[table["reported"], "location"]))) if len(table) else "-",
    )
    return table


def most_audible_agreement(predictions: Any, *, positive_label: int = 1) -> Any:
    """T70.5 -- does the model score highest where the murmur was heard loudest?

    CirCor's ``Most audible location`` field names, for each murmur patient, the
    position at which a human annotator judged the murmur loudest. It is *not* a
    label the model ever saw. So for every murmur-positive patient with that
    field populated, this compares the model's mean ``Present`` probability at
    that position against its mean over that patient's other positions.

    A positive mean difference is evidence the model is responding to the murmur
    where the murmur actually is, rather than to a patient-level confound such as
    body habitus or recording gain -- which is the failure mode that patient-level
    label propagation would otherwise hide.
    """
    import pandas as pd

    from src.utils.config import load_config

    proba_column = "proba_" + str(int(positive_label))
    frame = attach_locations(predictions)
    if proba_column not in frame.columns:
        raise LocationError(
            proba_column + " is absent; this run stored no positive-class probability"
        )

    demographics = load_config("paths").require("dataset.d4_circor.demographics_csv")
    table = pd.read_csv(demographics)
    audible = table[["Patient ID", "Most audible location"]].dropna()
    lookup = {
        str(pid): str(loc)
        for pid, loc in zip(
            audible["Patient ID"], audible["Most audible location"], strict=True
        )
    }

    rows: list[dict[str, Any]] = []
    for (model_id, fold_label, patient), block in frame.groupby(
        ["model_id", "fold_label", "patient_id"], sort=True
    ):
        loudest = lookup.get(str(patient))
        if loudest is None or len(block) < 2:
            continue
        if int(block["y_true"].iloc[0]) != int(positive_label):
            continue
        at = block[block["location"] == loudest][proba_column].to_numpy(dtype=float)
        elsewhere = block[block["location"] != loudest][proba_column].to_numpy(dtype=float)
        if at.size == 0 or elsewhere.size == 0:
            continue
        rows.append(
            {
                "model_id": model_id,
                "fold_label": fold_label,
                "patient_id": patient,
                "most_audible_location": loudest,
                "proba_at_most_audible": float(at.mean()),
                "proba_elsewhere": float(elsewhere.mean()),
                "delta": float(at.mean() - elsewhere.mean()),
                "n_other_locations": int(elsewhere.size),
            }
        )

    result = pd.DataFrame(rows)
    if len(result):
        log.info(
            "most-audible check: %d patient-model pair(s), mean delta %+.4f",
            len(result),
            float(np.mean(result["delta"])),
        )
    return result
