"""One recording in, one structured prediction out (Phase 106).

This is the only path from a `.wav` on disk to a class, and it is deliberately
thin: it loads a saved pipeline, calls the **same** preprocessing and extraction
functions the training runs called, and hands the resulting 138-wide vector to
`Pipeline.predict_proba`. Nothing in this module transforms a signal or computes
a feature.

## Why "no reimplementation" is the whole design (T106.2)

The failure this module exists to prevent is subtle and quiet: an inference path
that filters at a slightly different cutoff, or normalises after framing instead
of before, produces a *plausible* probability for every upload and is wrong for
all of them. Nothing crashes, and the dashboard shows a confident number.

So `predict_recording` calls `src.preprocessing.pipeline.preprocess` and
`src.feature_extraction.extractor.extract_all` — the functions that built
`all_features_matrix.parquet`. `tests/test_inference.py` closes the loop by
asserting that a corpus record, taken from its raw WAV through this entry point,
reproduces both its stored feature row *bit for bit* and its stored out-of-fold
probability.

## The cache is off for uploads, on purpose

`preprocess(use_cache=True)` keys on a `record_uid`. An upload has none, and two
different uploads would otherwise be liable to collide on a synthesised key.
Corpus records passed by `record_uid` may use the cache; a bare path never does.

## The operating point is 0.5, and the result says so

T50.4 selects a decision threshold inside the training fold for the *experiments*.
The deployed binary bundle carries no such threshold — `models_saved/binary/final/
manifest.json` has no threshold field — and every stored prediction in
`outputs/06_binary_results/` is the plain argmax: `y_pred == (proba_1 >= 0.5)`
holds for all 16,200 M1 rows. Predicting at 0.5 is therefore what reproduces the
experiments, and the result carries `operating_threshold` and
`operating_point_note` so nobody reads a tuned threshold into a number that does
not have one.

## One upload is not a batch, and the last two bits know it

Scoring a single recording is a different BLAS call from scoring a fold: a GEMV
rather than a GEMM, accumulating the dot product in a different order. Measured
over 40 held-out records of EXP-A1/r0f0/M1, a probability from this module sits
within **8 ULP** (4.44e-16) of the same model's batched value, with **zero class
flips**. The fold recomputed as a batch reproduces `predictions.parquet` bit for
bit; one row at a time does not, and cannot.

This is the same effect the Part VII re-run found in `brier` and `ece` -- the two
metrics that sum over thousands of probabilities -- surfacing here because the
deployed path scores one upload at a time. It is far below anything a decision
could notice, and it is stated because a reader comparing an API response against
a stored fold value will otherwise find a disagreement with no explanation.

## Screening language

Every result carries the disclaimer. A prediction here is a screening signal, not
a diagnosis, and `low_confidence` exists so that "the model is unsure" is a
first-class outcome rather than something a reader has to infer from a number
near the middle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "DISCLAIMER",
    "LOW_CONFIDENCE_MARGIN",
    "MAX_DURATION_SECONDS",
    "MIN_DURATION_SECONDS",
    "TASKS",
    "AudioValidationError",
    "ModelBundle",
    "ModelUnavailableError",
    "PredictionResult",
    "TaskSpec",
    "available_tasks",
    "clear_bundle_cache",
    "load_bundle",
    "predict_recording",
    "task_report",
    "validate_recording",
]

log = get_logger("inference.predictor")

DISCLAIMER = (
    "Academic screening and decision-support prototype. Not a diagnostic device "
    "and not a substitute for clinical assessment."
)

#: Below this gap between the top two class probabilities, the prediction is
#: reported as low confidence.
#:
#: 0.20 rather than a probability floor: on a three-class task 0.40 is a
#: comfortable win, on a two-class task it is a coin flip that landed. The
#: margin between the top two is the quantity that means the same thing in both.
LOW_CONFIDENCE_MARGIN = 0.20

#: Duration bounds for an upload, in seconds.
#:
#: Taken from the corpus rather than chosen: the shortest labelled recording is
#: 0.763 s (PASCAL set_b) and the longest is 121.998 s (PhysioNet). A recording
#: outside that range is not something any model here was fitted on, and a
#: confident answer for it would be an extrapolation the caller cannot see.
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 150.0

#: The suffixes soundfile is asked to open. WAV only, deliberately: a lossy
#: codec changes the spectral content every frequency, MFCC and chroma feature
#: is computed from, so an MP3 would be scored against a distribution it does
#: not belong to.
ALLOWED_SUFFIXES = frozenset({".wav"})


class AudioValidationError(ValueError):
    """The upload cannot be scored, with a reason a caller can act on."""


class ModelUnavailableError(RuntimeError):
    """The task is declared but no saved model backs it yet."""


@dataclass(frozen=True)
class TaskSpec:
    """One classification task, its label space, and where its model lives."""

    key: str
    title: str
    #: Class index -> human label, in the label space's own order.
    classes: tuple[str, ...]
    #: Directory under `models_saved/` holding `model.joblib` + `manifest.json`.
    model_dir: str
    #: What the task decides, in screening language.
    description: str


#: The five tasks, never merged (research rule 4). Each has its own target, its
#: own label space and its own saved model; a single entry point routes between
#: them (T106.4) without any of them borrowing another's classes.
TASKS: dict[str, TaskSpec] = {
    "binary": TaskSpec(
        "binary",
        "Binary screening (PhysioNet 2016)",
        ("normal", "abnormal"),
        "binary/final",
        "Normal versus abnormal heart sound.",
    ),
    "pascal_a": TaskSpec(
        "pascal_a",
        "PASCAL A, four classes",
        ("normal", "murmur", "extrahls", "artifact"),
        "pascal_a/final",
        (
            "Four PASCAL Dataset A categories. `artifact` is a RECORDING-QUALITY "
            "label, not a cardiac class, so this is not a four-class cardiac "
            "classifier."
        ),
    ),
    "pascal_b": TaskSpec(
        "pascal_b",
        "PASCAL B, three classes",
        ("normal", "murmur", "extrastole"),
        "pascal_b/final",
        "Three PASCAL Dataset B categories.",
    ),
    "murmur": TaskSpec(
        "murmur",
        "CirCor murmur",
        ("Absent", "Present", "Unknown"),
        "murmur/final",
        (
            "Murmur annotation from the CirCor corpus. `Unknown` is the "
            "annotator's own third category, not a low-confidence bucket."
        ),
    ),
    "outcome": TaskSpec(
        "outcome",
        "CirCor outcome",
        ("Normal", "Abnormal"),
        "outcome/final",
        "Clinical outcome label from the CirCor per-patient text files.",
    ),
}


@dataclass(frozen=True)
class ModelBundle:
    """A loaded pipeline, its feature order, and the manifest that describes it."""

    task: str
    pipeline: Any
    feature_names: tuple[str, ...]
    manifest: dict[str, Any]
    path: Path

    @property
    def model_id(self) -> str:
        return str(self.manifest.get("selected_model_id", self.manifest.get("model_id", "")))

    @property
    def classes(self) -> tuple[str, ...]:
        return TASKS[self.task].classes


@dataclass
class PredictionResult:
    """Everything T106.3 asks for, and the provenance to check it against."""

    task: str
    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float]
    confidence: float
    margin: float
    low_confidence: bool
    low_confidence_margin: float
    operating_threshold: float | None
    operating_point_note: str
    timings_seconds: dict[str, float]
    n_features: int
    n_missing_features: int
    feature_flags: tuple[str, ...]
    quality: dict[str, Any]
    model: dict[str, Any]
    source: str
    disclaimer: str = DISCLAIMER
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready mapping. Non-finite values are the API's problem (T108.3)."""
        return {
            "task": self.task,
            "predicted_class": self.predicted_class,
            "predicted_index": self.predicted_index,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "margin": self.margin,
            "low_confidence": self.low_confidence,
            "low_confidence_margin": self.low_confidence_margin,
            "operating_threshold": self.operating_threshold,
            "operating_point_note": self.operating_point_note,
            "timings_seconds": dict(self.timings_seconds),
            "n_features": self.n_features,
            "n_missing_features": self.n_missing_features,
            "feature_flags": list(self.feature_flags),
            "quality": dict(self.quality),
            "model": dict(self.model),
            "source": self.source,
            "disclaimer": self.disclaimer,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_BUNDLES: dict[str, ModelBundle] = {}


def _models_root() -> Path:
    from src.utils.config import load_config

    paths = load_config("paths")
    try:
        return Path(paths.require("models_saved"))
    except (KeyError, ValueError, RuntimeError):
        return Path(__file__).resolve().parents[2] / "models_saved"


def clear_bundle_cache() -> None:
    """Drop the loaded models. Used by tests and by the API's lifespan hook."""
    _BUNDLES.clear()


def load_bundle(task: str = "binary", *, path: str | Path | None = None) -> ModelBundle:
    """Load one task's saved pipeline, once per process.

    Loading is cached because unpickling a fitted estimator is the expensive part
    of a prediction and an API serving uploads would otherwise pay it per
    request. `path` bypasses both the cache and the task's default location,
    which is how the test loads a refitted fold model through the public entry
    point rather than a private one.
    """
    import json

    import joblib

    if path is None and task in _BUNDLES:
        return _BUNDLES[task]

    spec = TASKS.get(task)
    if spec is None:
        raise ModelUnavailableError(
            "unknown task " + repr(task) + "; declared tasks are " + ", ".join(sorted(TASKS))
        )

    directory = Path(path) if path is not None else _models_root() / spec.model_dir
    model_file = directory / "model.joblib"
    manifest_file = directory / "manifest.json"
    if not model_file.is_file():
        raise ModelUnavailableError(
            "no saved model for task "
            + repr(task)
            + ": "
            + str(model_file)
            + " does not exist. The task is declared, but the experiment that "
            "produces its final model has not been run yet."
        )

    pipeline = joblib.load(model_file)
    manifest: dict[str, Any] = {}
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    names = tuple(str(name) for name in manifest.get("feature_names", ()))
    if not names:
        raise ModelUnavailableError(
            str(manifest_file) + " carries no feature_names, so the column order "
            "the model was fitted on is unknown and a vector cannot be assembled "
            "for it safely"
        )

    bundle = ModelBundle(
        task=task, pipeline=pipeline, feature_names=names, manifest=manifest, path=directory
    )
    if path is None:
        _BUNDLES[task] = bundle
    log.info("loaded %s model from %s (%d features)", task, directory, len(names))
    return bundle


def available_tasks() -> tuple[str, ...]:
    """Tasks with a saved model on disk, in declaration order."""
    root = _models_root()
    return tuple(
        key for key, spec in TASKS.items() if (root / spec.model_dir / "model.joblib").is_file()
    )


def task_report() -> list[dict[str, Any]]:
    """Every declared task and whether it can be served. For `GET /health`.

    A task with no model is reported as unavailable **with the reason**, not
    omitted: a caller that asks for `murmur` deserves "the experiment that
    produces it has not been run" rather than "unknown task".
    """
    root = _models_root()
    rows: list[dict[str, Any]] = []
    for key, spec in TASKS.items():
        directory = root / spec.model_dir
        ready = (directory / "model.joblib").is_file()
        rows.append(
            {
                "task": key,
                "title": spec.title,
                "classes": list(spec.classes),
                "description": spec.description,
                "available": ready,
                "model_dir": spec.model_dir,
                "reason": None
                if ready
                else (
                    "no model at " + spec.model_dir + "; the experiment that produces "
                    "it has not been run yet"
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# T106.5 -- input validation
# ---------------------------------------------------------------------------


def validate_recording(path: str | Path) -> dict[str, Any]:
    """Check one file is scoreable, and return what was read from its header.

    Raises :class:`AudioValidationError` with a sentence naming the problem.
    Every check here is one a caller can act on -- the file is missing, the
    format is wrong, it is too short to frame, it is silent -- and none of them
    is a stack trace from deep inside a decoder.
    """
    candidate = Path(path)
    if not candidate.exists():
        raise AudioValidationError("no such file: " + str(candidate))
    if not candidate.is_file():
        raise AudioValidationError(str(candidate) + " is a directory, not a recording")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise AudioValidationError(
            candidate.name + " is not a WAV file. Only WAV is accepted: a lossy format changes the "
            "spectral content the frequency, MFCC and chroma features are computed "
            "from, so the recording would be scored against a distribution it does "
            "not belong to."
        )
    if candidate.stat().st_size == 0:
        raise AudioValidationError(candidate.name + " is empty (0 bytes)")

    import soundfile as sf

    try:
        info = sf.info(str(candidate))
    except Exception as error:
        raise AudioValidationError(
            candidate.name + " could not be read as audio: " + str(error)
        ) from error

    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    if info.frames == 0 or duration <= 0:
        raise AudioValidationError(candidate.name + " contains no audio frames")
    if duration < MIN_DURATION_SECONDS:
        raise AudioValidationError(
            candidate.name
            + " is "
            + format(duration, ".3f")
            + " s long; the shortest recording any model here was fitted on is "
            + format(MIN_DURATION_SECONDS, ".2f")
            + " s, and a shorter clip cannot fill even one analysis frame"
        )
    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            candidate.name
            + " is "
            + format(duration, ".1f")
            + " s long, beyond the "
            + format(MAX_DURATION_SECONDS, ".0f")
            + " s bound taken from the corpus. Trim it rather than have the "
            "features summarise a recording unlike anything in training."
        )

    return {
        "path": str(candidate),
        "name": candidate.name,
        "bytes": candidate.stat().st_size,
        "duration_seconds": duration,
        "sample_rate_hz": int(info.samplerate),
        # Multi-channel is downmixed by the shared preprocessing path rather
        # than rejected -- `io.load_recording` is where mono conversion lives,
        # and doing it anywhere else would be the reimplementation this module
        # exists to avoid.
        "channels": int(info.channels),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


# ---------------------------------------------------------------------------
# T106.3 / T106.4 -- the prediction
# ---------------------------------------------------------------------------


def predict_recording(
    path: str | Path,
    *,
    task: str = "binary",
    bundle: ModelBundle | None = None,
    record_uid: str | None = None,
    use_cache: bool = False,
) -> PredictionResult:
    """Score one recording. The public entry point for every task.

    `record_uid` is for corpus records only: it lets the preprocessing cache be
    reused when re-scoring something already in the audit. An upload has no uid
    and must not invent one, so `use_cache` defaults to False.
    """
    from src.feature_extraction.extractor import extract_all
    from src.preprocessing.pipeline import preprocess

    started = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []

    mark = time.perf_counter()
    info = validate_recording(path)
    timings["validate"] = time.perf_counter() - mark

    mark = time.perf_counter()
    loaded = bundle if bundle is not None else load_bundle(task)
    timings["load_model"] = time.perf_counter() - mark

    mark = time.perf_counter()
    prepared = preprocess(
        path,
        record_uid=record_uid,
        with_quality=True,
        use_cache=use_cache and record_uid is not None,
    )
    timings["preprocess"] = time.perf_counter() - mark

    mark = time.perf_counter()
    extracted = extract_all(prepared.signal, prepared.fs, record_uid=record_uid)
    timings["extract"] = time.perf_counter() - mark

    vector, missing = _vector_for(extracted.values, loaded.feature_names)
    if missing:
        warnings.append(
            str(len(missing))
            + " of "
            + str(len(loaded.feature_names))
            + " features could not be computed for this recording and were passed "
            "to the model's imputer: "
            + ", ".join(sorted(missing)[:8])
            + ("…" if len(missing) > 8 else "")
        )

    mark = time.perf_counter()
    proba = _predict_proba(loaded, vector)
    timings["predict"] = time.perf_counter() - mark
    timings["total"] = time.perf_counter() - started

    order = np.argsort(proba)[::-1]
    top = int(order[0])
    runner_up = float(proba[int(order[1])]) if proba.size > 1 else 0.0
    margin = float(proba[top]) - runner_up
    classes = loaded.classes
    if len(classes) != proba.size:
        raise ModelUnavailableError(
            "task "
            + loaded.task
            + " declares "
            + str(len(classes))
            + " classes but its model produced "
            + str(proba.size)
            + " probabilities"
        )

    return PredictionResult(
        task=loaded.task,
        predicted_class=classes[top],
        predicted_index=top,
        probabilities={name: float(value) for name, value in zip(classes, proba, strict=True)},
        confidence=float(proba[top]),
        margin=margin,
        low_confidence=bool(margin < LOW_CONFIDENCE_MARGIN),
        low_confidence_margin=LOW_CONFIDENCE_MARGIN,
        operating_threshold=0.5 if len(classes) == 2 else None,
        operating_point_note=(
            "Plain argmax at 0.5. The deployed bundle carries no in-fold selected "
            "threshold, and every stored prediction in outputs/06_binary_results/ "
            "was produced the same way, so this reproduces the experiments rather "
            "than applying an operating point they never used."
        ),
        timings_seconds={key: round(value, 6) for key, value in timings.items()},
        n_features=len(loaded.feature_names),
        n_missing_features=len(missing),
        feature_flags=tuple(extracted.flags),
        quality={
            **{key: _finite(value) for key, value in (prepared.quality or {}).items()},
            "original_sample_rate_hz": info["sample_rate_hz"],
            "channels": info["channels"],
            "duration_seconds": info["duration_seconds"],
            "applied_steps": list(prepared.steps),
        },
        model={
            "task": loaded.task,
            "model_id": loaded.model_id,
            "estimator_class": loaded.manifest.get("estimator_class"),
            "n_features": len(loaded.feature_names),
            "saved_at": loaded.manifest.get("saved_at"),
            "n_records_fitted": loaded.manifest.get("n_records_fitted"),
            "selection_rule": loaded.manifest.get("selection_rule"),
            "package_versions": loaded.manifest.get("package_versions", {}),
            "path": str(loaded.path),
        },
        source=info["name"],
        warnings=warnings,
    )


def _vector_for(
    values: dict[str, float], feature_names: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    """Assemble the model's column order from the extractor's output.

    Ordered by the **model's** `feature_names`, never by whatever order the
    extractor happens to return. A pipeline fitted on a 138-column matrix has no
    way to notice that column 40 and column 41 arrived swapped; it produces a
    confident, wrong answer. This is the one place that ordering is established.
    """
    missing: list[str] = []
    row = np.empty(len(feature_names), dtype=np.float64)
    for index, name in enumerate(feature_names):
        value = values.get(name, np.nan)
        row[index] = value
        if name not in values or not np.isfinite(value):
            missing.append(name)
    return row, missing


def _predict_proba(bundle: ModelBundle, vector: np.ndarray) -> np.ndarray:
    """One row through the fitted pipeline, with BLAS pinned to one thread.

    Pinned for the same reason `extract_all` pins it: a threaded BLAS partitions
    a matrix product differently depending on thread count, so the same upload
    scored on a busy server and an idle one could differ in the last bits.
    Research rule 5 does not have an exception for inference.
    """
    from threadpoolctl import threadpool_limits

    # Matched to how the pipeline was FITTED, not to what is convenient here.
    # The training runs fit on a numpy matrix, so the estimator carries no
    # `feature_names_in_`; handing it a DataFrame makes scikit-learn warn that
    # the names are being ignored -- and a warning that is routinely ignored is
    # how a genuine column-order mismatch gets missed later. The order itself is
    # established in `_vector_for` against the manifest, which is the check that
    # actually protects against a swapped column.
    payload: Any = vector.reshape(1, -1)
    if getattr(bundle.pipeline, "feature_names_in_", None) is not None:
        import pandas as pd

        payload = pd.DataFrame([vector], columns=list(bundle.feature_names))

    with threadpool_limits(limits=1):
        if hasattr(bundle.pipeline, "predict_proba"):
            proba = np.asarray(bundle.pipeline.predict_proba(payload), dtype=np.float64)[0]
        else:
            raise ModelUnavailableError(
                "the saved " + bundle.task + " model does not expose predict_proba, so no class "
                "probabilities or confidence can be reported for it"
            )
    return proba


def _finite(value: Any) -> Any:
    """NaN and inf become None here rather than at the API boundary."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        as_float = float(value)
        return as_float if np.isfinite(as_float) else None
    return value
