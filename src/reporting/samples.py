"""Built-in sample recordings for the prediction pages (T116.6).

T116.6 asks for sample recordings so every prediction page demonstrates without
an upload. That sounds like "commit a few WAVs", and it is not, for two reasons
that are worth stating because both were decided against.

## No new audio is redistributed

`dataset/archive/LICENSE.txt` is the ODC-By 1.0 grant for the CirCor corpus, and
Phase 113 already used it: `frontend/public/85197_TV.wav` is committed with its
notices in `frontend/public/NOTICE.md`. The PhysioNet 2016 and PASCAL copies in
`dataset/` carry **no** licence file, so the terms under which their audio may be
redistributed cannot be established from the files on disk. Committing them
anyway would be asserting a licence nobody verified, which is the same failure
mode as asserting a metric nobody computed.

So the samples are declared here, resolved against the local corpus at run time,
and served by the API from the operator's own copy — `GET /samples` lists what
this process can actually reach and `POST /predict/sample` scores it. A checkout
without `dataset/` reports the samples as unavailable with that reason rather
than shipping audio of unknown provenance.

## A record has five stored out-of-fold predictions, not one

The fold map is repeated 5x5 grouped CV, so every record is held out once per
repeat and EXP-A2 stores **five** out-of-fold probabilities for it. "The stored
out-of-fold prediction for that record" is therefore five numbers, and the
payload carries all five plus their mean, labelled. Quoting one and calling it
*the* stored value would be picking a fold.

The reference is EXP-A2/M1 because M1 refit on every labelled record is what
`models_saved/binary/final/` holds. The deployed bundle is fitted **in sample**
on all 3,240 records, so its own probability for one of these recordings is not
comparable with an out-of-fold value and the payload says so beside the numbers.

## The samples were selected by query, not by taste

`SampleSpec.selection` records the query each id came out of. Three binary samples:
one confidently abnormal, one confidently normal, and one whose five repeats
disagree — so the low-confidence warning (T116.2) demonstrates on a real
borderline record instead of a state manufactured to show the banner.

## Screening language

Every sample is a **dataset sample**. Never a patient, never a case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.aggregation import AGGREGATION_RULES
from src.inference.predictor import (
    ALLOWED_SUFFIXES,
    DISCLAIMER,
    LOW_CONFIDENCE_MARGIN,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    TASKS,
    task_report,
)
from src.reporting.tables import format_value
from src.utils.logging_setup import get_logger

__all__ = [
    "MASTER_CSV",
    "PATIENT_GROUP",
    "REFERENCE_EXPERIMENT",
    "REFERENCE_MODEL",
    "SAMPLES",
    "TASK_PAGES",
    "SampleSpec",
    "prediction_payload",
    "resolve_sample",
    "sample_locations",
    "sample_records",
    "stored_reference",
]

log = get_logger("reporting.samples")

MASTER_CSV = "outputs/01_dataset_audit/metadata_master.csv"

#: The run whose stored out-of-fold probabilities the binary samples quote, and
#: the model inside it. M1 is what `models_saved/binary/final/` was refit from
#: (`selected_model_id` in its manifest), so any other model here would be
#: quoting a result the deployed page does not use.
REFERENCE_EXPERIMENT = "outputs/06_binary_results/EXP-A2/predictions.parquet"
REFERENCE_MODEL = "M1"

#: Which prediction page offers which task. Three pages, five label spaces, and
#: the spaces are never merged: the multiclass page offers PASCAL A and PASCAL B
#: as two separate choices, not as one seven-class selector.
TASK_PAGES: dict[str, str] = {
    "binary": "/predict/binary/",
    "pascal_a": "/predict/multiclass/",
    "pascal_b": "/predict/multiclass/",
    "murmur": "/predict/murmur/",
    "outcome": "/predict/murmur/",
}


@dataclass(frozen=True)
class SampleSpec:
    """One pinned corpus recording offered as a built-in demo."""

    sample_id: str
    record_uid: str
    #: The label spaces this recording belongs to. A CirCor recording carries
    #: both a murmur annotation and an outcome, so it serves two tasks.
    tasks: tuple[str, ...]
    #: Why this record and not another. Recorded so the choice is auditable.
    selection: str


#: Pinned rather than chosen at export time: a demo that shows a different
#: recording on every build cannot be cross-checked against anything, and the
#: gate for T116.7 quotes one of these ids.
SAMPLES: tuple[SampleSpec, ...] = (
    SampleSpec(
        "binary-abnormal",
        "D1_training-b_b0033",
        ("binary",),
        "shortest PhysioNet record (5.31 s) whose five out-of-fold repeats all "
        "agree on abnormal with a mean probability above 0.85",
    ),
    SampleSpec(
        "binary-normal",
        "D1_training-b_b0272",
        ("binary",),
        "PhysioNet normal record whose five out-of-fold repeats all agree, with "
        "the mean abnormal probability far below the 0.5 operating point",
    ),
    SampleSpec(
        "binary-borderline",
        "D1_training-d_d0015",
        ("binary",),
        "a real borderline record: the five out-of-fold repeats DISAGREE and the "
        "mean sits within 0.01 of the operating point, so the low-confidence "
        "warning demonstrates on a genuine case rather than a contrived one",
    ),
    SampleSpec(
        "pascal-a-normal",
        "D2_set_a_normal__201103140135",
        ("pascal_a",),
        "median-duration PASCAL A normal record",
    ),
    SampleSpec(
        "pascal-a-murmur",
        "D2_set_a_murmur__201108222245",
        ("pascal_a",),
        "median-duration PASCAL A murmur record",
    ),
    SampleSpec(
        "pascal-a-artifact",
        "D2_set_a_artifact__201106070949",
        ("pascal_a",),
        "median-duration PASCAL A artifact record -- a RECORDING-QUALITY label, "
        "included so the page can show what that class actually sounds like",
    ),
    SampleSpec(
        "pascal-b-normal",
        "D3_set_b_normal_noisynormal_198_1308141739338_D",
        ("pascal_b",),
        "median-duration PASCAL B normal record, from the noisy-normal subset",
    ),
    SampleSpec(
        "pascal-b-murmur",
        "D3_set_b_murmur__248_1309201683806_A",
        ("pascal_b",),
        "median-duration PASCAL B murmur record",
    ),
    SampleSpec(
        "pascal-b-extrastole",
        "D3_set_b_extrastole__261_1309353556003_C",
        ("pascal_b",),
        "median-duration PASCAL B extrastole record",
    ),
    SampleSpec(
        "circor-murmur-present",
        "D4_training_data_84853_TV",
        ("murmur", "outcome"),
        "shortest CirCor recording in the 6-14 s band annotated murmur Present "
        "with an Abnormal outcome",
    ),
    SampleSpec(
        "circor-murmur-absent",
        "D4_training_data_50743_AV_2",
        ("murmur", "outcome"),
        "shortest CirCor recording in the 6-14 s band annotated murmur Absent "
        "with a Normal outcome",
    ),
    SampleSpec(
        "circor-murmur-unknown",
        "D4_training_data_68175_PV",
        ("murmur", "outcome"),
        "shortest CirCor recording in the 6-14 s band annotated murmur Unknown "
        "-- the annotator's own third category, not a low-confidence bucket",
    ),
    # One complete CirCor screening: the same subject at all four auscultation
    # locations. T116.4 needs this -- patient-level output is a collapse over
    # several recordings, and it cannot be demonstrated from one recording each
    # of four different subjects. The TV recording above is this subject's, so
    # only the other three are added.
    SampleSpec(
        "circor-84853-av",
        "D4_training_data_84853_AV",
        ("murmur", "outcome"),
        "subject 84853 at the aortic valve; one of four locations for this subject",
    ),
    SampleSpec(
        "circor-84853-mv",
        "D4_training_data_84853_MV",
        ("murmur", "outcome"),
        "subject 84853 at the mitral valve; one of four locations for this subject",
    ),
    SampleSpec(
        "circor-84853-pv",
        "D4_training_data_84853_PV",
        ("murmur", "outcome"),
        "subject 84853 at the pulmonary valve; one of four locations for this subject",
    ),
)

#: The subject whose four recordings demonstrate the recording-to-patient
#: collapse. Declared so a page can group them without parsing sample ids.
PATIENT_GROUP: tuple[str, tuple[str, ...]] = (
    "84853",
    (
        "circor-84853-av",
        "circor-84853-mv",
        "circor-84853-pv",
        "circor-murmur-present",
    ),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sample_records() -> Any:
    """The audited metadata rows for every declared sample, indexed by uid.

    Raises when a declared sample is not in `metadata_master.csv`: a pinned id
    that no longer resolves is a corpus change, and silently dropping it would
    leave a page quietly demonstrating less than it claims.
    """
    import pandas as pd

    path = _project_root() / MASTER_CSV
    if not path.is_file():
        raise FileNotFoundError(
            str(path) + " is missing. It is a committed audit output, so its absence means "
            "outputs/01_dataset_audit/ has been moved rather than that the corpus "
            "is unavailable."
        )
    frame = pd.read_csv(path).set_index("record_uid")
    wanted = [spec.record_uid for spec in SAMPLES]
    missing = [uid for uid in wanted if uid not in frame.index]
    if missing:
        raise KeyError(
            "declared sample recordings are not in "
            + MASTER_CSV
            + ": "
            + ", ".join(missing)
            + ". Re-pin them in SAMPLES rather than dropping them here."
        )
    return frame.loc[wanted]


def resolve_sample(sample_id: str) -> Path | None:
    """The WAV for a declared sample on this machine, or ``None``.

    ``None`` is the fresh-clone and CI answer and is not an error: `dataset/` is
    read-only input that is never committed, so a checkout legitimately has no
    audio to serve.
    """
    spec = next((s for s in SAMPLES if s.sample_id == sample_id), None)
    if spec is None:
        return None
    try:
        rows = sample_records()
    except (FileNotFoundError, KeyError) as error:
        log.warning("sample metadata is unavailable: %s", error)
        return None
    if spec.record_uid not in rows.index:
        return None
    raw = rows.loc[spec.record_uid, "file_path"]
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = _project_root() / candidate
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES or not candidate.is_file():
        return None
    return candidate


def sample_locations() -> dict[str, str | None]:
    """Auscultation location per sample, from the audit rather than from the id.

    Parsed out of `record_uid` this would be wrong twice: a PhysioNet uid ends in
    a record number and a CirCor uid can end in a repeat index
    (`..._AV_2`). The corpus records the location, so it is read.
    """
    try:
        rows = sample_records()
    except (FileNotFoundError, KeyError):
        return {spec.sample_id: None for spec in SAMPLES}

    import pandas as pd

    out: dict[str, str | None] = {}
    for spec in SAMPLES:
        value = rows.loc[spec.record_uid, "recording_location"]
        out[spec.sample_id] = None if pd.isna(value) else str(value)
    return out


def stored_reference() -> dict[str, dict[str, Any]]:
    """Every binary sample's five stored out-of-fold probabilities, by uid.

    Five, not one: each record is held out once per repeat of the 5x5 map. The
    mean travels with them so a page can show a single number honestly labelled,
    and `folds_agree` is carried because it is the difference between a stable
    prediction and one that changed with the training split.
    """
    import pandas as pd

    path = _project_root() / REFERENCE_EXPERIMENT
    if not path.is_file():
        return {}

    binary_uids = {spec.record_uid for spec in SAMPLES if "binary" in spec.tasks}
    frame = pd.read_parquet(path)
    frame = frame[(frame["model_id"] == REFERENCE_MODEL) & (frame["record_uid"].isin(binary_uids))]
    if frame.empty:
        return {}

    classes = TASKS["binary"].classes
    out: dict[str, dict[str, Any]] = {}
    for uid, group in frame.groupby("record_uid"):
        group = group.sort_values(["repeat", "fold"])
        probabilities = [float(v) for v in group["proba_1"]]
        predictions = [int(v) for v in group["y_pred"]]
        mean = sum(probabilities) / len(probabilities)
        out[str(uid)] = {
            "experiment": "EXP-A2",
            "model_id": REFERENCE_MODEL,
            "source": REFERENCE_EXPERIMENT,
            "positive_class": classes[1],
            "n_repeats": len(probabilities),
            "fold_labels": [str(v) for v in group["fold_label"]],
            "probabilities": probabilities,
            "probabilities_display": [format_value(v, "metric") for v in probabilities],
            "mean_probability": mean,
            "mean_probability_display": format_value(mean, "metric"),
            "predicted_classes": [classes[v] for v in predictions],
            "folds_agree": len(set(predictions)) == 1,
            "true_class": classes[int(group["y_true"].iloc[0])],
        }
    return out


def _sample_payload(spec: SampleSpec, row: Any, reference: dict[str, dict[str, Any]]) -> dict:
    def cell(name: str) -> Any:
        import pandas as pd

        if name not in row.index:
            return None
        value = row[name]
        return None if pd.isna(value) else value

    duration = cell("duration_sec")
    labels = {
        "binary": cell("binary_label_name"),
        "pascal_a": cell("multiclass_label_name") if str(cell("dataset_source")) == "D2" else None,
        "pascal_b": cell("multiclass_label_name") if str(cell("dataset_source")) == "D3" else None,
        "murmur": cell("murmur_label_name"),
        "outcome": cell("outcome_label_name"),
    }
    return {
        "sample_id": spec.sample_id,
        "record_uid": spec.record_uid,
        "tasks": list(spec.tasks),
        "selection": spec.selection,
        "dataset_source": str(cell("dataset_source") or ""),
        "dataset_name": str(cell("dataset_name") or ""),
        "subset": str(cell("subset") or ""),
        "subject_id": None if cell("subject_id") is None else str(cell("subject_id")),
        "recording_location": (
            None if cell("recording_location") is None else str(cell("recording_location"))
        ),
        "duration_seconds": None if duration is None else float(duration),
        "duration_display": format_value(duration, "seconds") + " s",
        "original_sample_rate_hz": (
            None if cell("original_fs") is None else int(float(str(cell("original_fs"))))
        ),
        # The reference labels are the corpus's own annotations, shown so a
        # reader can see what the recording IS as well as what the model said.
        "labels": {task: (None if value is None else str(value)) for task, value in labels.items()},
        "reference": reference.get(spec.record_uid),
        # Resolved by the API at run time from the operator's own corpus, never
        # committed. See the module docstring.
        "audio_path": "/samples/" + spec.sample_id + "/audio",
        "predict_path": "/predict/sample",
    }


def prediction_payload() -> dict[str, Any]:
    """Everything the three prediction pages need that is known at build time.

    Deliberately small: the declared label spaces, the pinned samples with their
    corpus labels and their stored out-of-fold references, and the upload bounds.
    Live probabilities are not in here and cannot be — a prediction of a file
    that does not exist yet has no build-time value.
    """
    rows = sample_records()
    reference = stored_reference()

    statuses = {row["task"]: row for row in task_report()}
    tasks = []
    for name, spec in TASKS.items():
        status = statuses.get(name, {})
        tasks.append(
            {
                "task": name,
                "title": spec.title,
                "classes": list(spec.classes),
                "description": spec.description,
                "page": TASK_PAGES[name],
                # Build-time availability. The page re-reads it from `GET /tasks`
                # at run time, because a model saved after this export exists
                # for the API and not for this file.
                "model_available_at_export": bool(status.get("available", False)),
                "reason_at_export": status.get("reason"),
                "model_dir": str(status.get("model_dir", "")),
            }
        )

    samples = [
        _sample_payload(spec, rows.loc[spec.record_uid], reference)
        for spec in SAMPLES
        if spec.record_uid in rows.index
    ]

    return {
        "framework": "PV-MEPCG / PulseVision",
        "disclaimer": DISCLAIMER,
        "tasks": tasks,
        "samples": samples,
        "n_samples_with_reference": sum(1 for s in samples if s["reference"] is not None),
        "patient_group": {
            "subject_id": PATIENT_GROUP[0],
            "sample_ids": list(PATIENT_GROUP[1]),
            "note": (
                "CirCor labels the SUBJECT, not the recording, and screens at four "
                "auscultation locations. A patient-level indication is therefore a "
                "collapse over several recordings, and the rule used to collapse "
                "them changes the answer -- so all three declared rules are shown "
                "rather than one being chosen silently."
            ),
        },
        "aggregation_rules": list(AGGREGATION_RULES),
        "reference": {
            "experiment": "EXP-A2",
            "model_id": REFERENCE_MODEL,
            "source": REFERENCE_EXPERIMENT,
            "note": (
                "Each record is held out once per repeat of the 5x5 grouped map, so "
                "EXP-A2 stores five out-of-fold probabilities for it, not one. All "
                "five are shown. The deployed bundle is a refit of "
                + REFERENCE_MODEL
                + " on every labelled record, so its probability for one of these "
                "recordings is an IN-SAMPLE value and is not comparable with an "
                "out-of-fold one."
            ),
        },
        "upload": {
            "accepted_suffixes": sorted(ALLOWED_SUFFIXES),
            "min_duration_seconds": MIN_DURATION_SECONDS,
            "max_duration_seconds": MAX_DURATION_SECONDS,
            "min_duration_display": format_value(MIN_DURATION_SECONDS, "seconds") + " s",
            "max_duration_display": format_value(MAX_DURATION_SECONDS, "seconds") + " s",
            "duration_note": (
                "Taken from the corpus, not chosen: the shortest labelled recording "
                "is 0.76 s and the longest is 122.00 s. A recording outside that "
                "range is not something these models were fitted on."
            ),
        },
        "low_confidence": {
            "margin": LOW_CONFIDENCE_MARGIN,
            "margin_display": format_value(LOW_CONFIDENCE_MARGIN, "metric"),
            "note": (
                "A prediction is reported low-confidence when the gap between the "
                "top two class probabilities is below this margin. The margin, not "
                "a probability floor: on a three-class task 0.40 is a comfortable "
                "win and on a two-class task it is a coin flip that landed."
            ),
        },
        "operating_point": {
            "threshold": 0.5,
            "threshold_display": format_value(0.5, "metric"),
            "note": (
                "On a TWO-class task the deployed bundle carries no in-fold "
                "threshold, and every stored prediction in the experiment that "
                "backs it is the plain argmax, so inference decides at 0.5 -- "
                "which is what reproduces those results. A multiclass task has no "
                "single operating point at all: the highest probability wins. Each "
                "result below states which rule it used."
            ),
        },
    }
