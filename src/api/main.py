"""The inference service (Phase 108).

One runtime endpoint does real work — `POST /predict`. Everything else this API
serves is either a status report or a file already on disk.

## The API computes nothing

`src/inference/predictor.py` is the only path from a WAV to a class, and this
module calls it. There is no preprocessing here, no feature extraction, no
thresholding and no metric. That boundary is the reason a number on the
dashboard can be traced to the run that produced it: an API that "helpfully"
re-derived a probability would be a second implementation nobody tests against
the corpus.

The same rule runs the other way for precomputed values. Every metric the
dashboard shows arrives through build-time codegen into
`frontend/lib/generated/`. This API deliberately exposes **no** metrics
endpoint, because the moment one exists a page will fetch from it at runtime and
the codegen boundary is gone.

## Non-finite numbers are coerced once, centrally (T108.3)

Starlette renders JSON with `allow_nan=False`, so a single NaN anywhere in a
response is a 500 with a stack trace rather than a field the client can handle.
That NaN is not hypothetical: a recording too short for a wavelet level, or one
whose envelope is flat, produces features that legitimately have no value, and
`quality` carries per-record measurements that can be undefined.

`to_jsonable` converts numpy scalars to Python scalars and non-finite floats to
`null`, and `SafeRoute` applies it to every endpoint's return value. The route
class is set once on the app's router, so it covers routes added after the app
is built — never per endpoint, which is how one gets forgotten.

It is a **route** class and not merely a response class because a response class
is not app-wide coverage even though it looks like one; see `SafeRoute`.

## CORS is the dev origin only

`next dev` on port 3000 is the only cross-origin caller this project has. In
production the exported site is served by this same process from
`frontend/out/`, so it is same-origin and needs no CORS at all. A wildcard would
let any page on the internet drive a model on the user's machine.

## Screening language

Every prediction response carries the disclaimer, and `low_confidence` is a
first-class field rather than something a client infers from a number near the
middle.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from src.evaluation.aggregation import AGGREGATION_RULES
from src.inference.predictor import (
    DISCLAIMER,
    TASKS,
    AudioValidationError,
    ModelUnavailableError,
    available_tasks,
    clear_bundle_cache,
    load_bundle,
    predict_recording,
    task_report,
)
from src.reporting.pages_10_12 import objective_report_path, reportable_experiments
from src.reporting.samples import SAMPLES, resolve_sample, sample_locations
from src.reporting.tables import format_value
from src.utils.logging_setup import get_logger

__all__ = [
    "API_TITLE",
    "API_VERSION",
    "DEV_ORIGINS",
    "MAX_UPLOAD_BYTES",
    "HealthResponse",
    "ManifestResponse",
    "PatientResponse",
    "PredictResponse",
    "SafeJSONResponse",
    "SampleSummary",
    "SafeRoute",
    "app",
    "create_app",
    "to_jsonable",
]

log = get_logger("api.main")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = PROJECT_ROOT / "frontend" / "out"
GENERATED_MANIFEST = PROJECT_ROOT / "frontend" / "lib" / "generated" / "manifest.json"

API_TITLE = "PV-MEPCG / PulseVision inference API"
API_VERSION = "1.0.0"

#: The only cross-origin callers: `next dev`. The production build is served by
#: this same process from `frontend/out/` and is therefore same-origin.
DEV_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

#: An upload larger than this is refused before it is written to disk. The
#: longest recording in the corpus is 122 s; at 44.1 kHz 16-bit stereo that is
#: about 21 MB, so 32 MB accepts anything the models were fitted to see and
#: refuses a file that is not a recording of a heart.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


# ---------------------------------------------------------------------------
# T108.3 -- the central encoder
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Numpy scalars to Python scalars, non-finite floats to ``None``.

    Applied to whole response bodies rather than to individual fields. A NaN
    that reaches `json.dumps` is a 500, and the client cannot tell that from a
    server that fell over: coercing to `null` says "this recording has no value
    for this" in a form every JSON parser already understands.
    """
    import numpy as np

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class SafeJSONResponse(JSONResponse):
    """The last net: sanitises anything handed straight to a Response."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            to_jsonable(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


class SafeRoute(APIRoute):
    """Sanitise what an endpoint returns, before FastAPI serialises it.

    A response class is **not** app-wide coverage, which is worth stating
    plainly because it looks like it is. FastAPI validates and serialises an
    endpoint's return value through Pydantic *first* and only then hands the
    result to the response class, so a `numpy.int64` in a loosely typed field
    raises `PydanticSerializationError` before `SafeJSONResponse.render` is ever
    called. Measured, not assumed: a probe route returning `np.int64(3)` failed
    with the response class installed.

    Wrapping the endpoint puts the coercion in front of the serializer instead.
    `route_class` is set once on the app's router, so every route gets it --
    including routes added after the app is built, which is the property T108.3
    asks for and the one a per-endpoint call would lose.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        endpoint = kwargs.get("endpoint") or (args[1] if len(args) > 1 else None)
        if endpoint is not None:
            wrapped = _sanitising(endpoint)
            if "endpoint" in kwargs:
                kwargs["endpoint"] = wrapped
            else:
                args = (args[0], wrapped, *args[2:])
        super().__init__(*args, **kwargs)


def _sanitising(endpoint: Callable[..., Any]) -> Callable[..., Any]:
    """`endpoint`, with its return value passed through `to_jsonable`.

    A `Response` the endpoint built itself is left alone: it has already chosen
    its own body, and rewriting that would be the API editing a payload rather
    than passing one through.
    """
    if inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await endpoint(*args, **kwargs)
            return result if isinstance(result, (Response, BaseModel)) else to_jsonable(result)

        return async_wrapper

    @functools.wraps(endpoint)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = endpoint(*args, **kwargs)
        return result if isinstance(result, (Response, BaseModel)) else to_jsonable(result)

    return wrapper


# ---------------------------------------------------------------------------
# T108.2 -- response schemas
# ---------------------------------------------------------------------------


class TaskStatus(BaseModel):
    """One declared task and whether this process can serve it."""

    task: str
    title: str
    classes: list[str]
    description: str
    available: bool
    model_dir: str
    #: Why the task cannot be served, when it cannot. Never omitted, because a
    #: caller asking for `murmur` deserves the reason rather than "unknown task".
    reason: str | None = None
    loaded: bool = False


class HealthResponse(BaseModel):
    status: str
    framework: str = "PV-MEPCG / PulseVision"
    api_version: str = API_VERSION
    disclaimer: str = DISCLAIMER
    tasks: list[TaskStatus]
    n_available: int
    packages: dict[str, str]
    static_site_mounted: bool


class ManifestResponse(BaseModel):
    """What produced the numbers this deployment serves."""

    framework: str
    run_id: str | None = None
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    exported_utc: str | None = None
    exporter: str | None = None
    n_tables: int | None = None
    n_figures: int | None = None
    api_version: str = API_VERSION
    disclaimer: str = DISCLAIMER
    #: False when `frontend/lib/generated/manifest.json` has not been built yet.
    generated: bool = True


class RecordingQuality(BaseModel):
    """Per-recording measurements. Every float here can legitimately be absent.

    `float | None` is explicit on all of them (T108.2): a recording too short for
    a wavelet level, or one whose envelope never rises, has no value for some of
    these, and `0.0` would read as a measurement.
    """

    duration_seconds: float | None = None
    original_sample_rate_hz: int | None = None
    channels: int | None = None
    applied_steps: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class ModelInfo(BaseModel):
    task: str
    model_id: str | None = None
    estimator_class: str | None = None
    n_features: int | None = None
    saved_at: str | None = None
    n_records_fitted: int | None = None
    selection_rule: list[str] | None = None
    note: str | None = None
    package_versions: dict[str, str] = Field(default_factory=dict)
    path: str | None = None


class PredictResponse(BaseModel):
    """The T106.3 structure, unchanged. The API adds nothing to it."""

    task: str
    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float | None]
    confidence: float | None
    margin: float | None
    low_confidence: bool
    low_confidence_margin: float
    #: `None` for a multiclass task, which has no single threshold.
    operating_threshold: float | None = None
    operating_point_note: str
    timings_seconds: dict[str, float | None]
    n_features: int
    n_missing_features: int
    feature_flags: list[str] = Field(default_factory=list)
    quality: RecordingQuality
    model: ModelInfo
    source: str
    disclaimer: str
    warnings: list[str] = Field(default_factory=list)
    #: Every number above, already rounded, as the string a page renders.
    #:
    #: A live probability cannot come through build-time codegen -- the file did
    #: not exist when the export ran -- so the "no client-side rounding" rule is
    #: kept by formatting HERE, in Python, through `tables.format_value`: the
    #: same function and the same T85.6 places as every precomputed table. A
    #: page renders `display.*` and never calls `toFixed`.
    display: dict[str, Any] = Field(default_factory=dict)
    #: T117.2: what drove THIS recording's decision, from the vector that was
    #: scored. Exact for a linear model; for any other estimator `available` is
    #: false and `reason` says why, rather than global importance being
    #: substituted for a per-recording answer.
    explanation: dict[str, Any] | None = None

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    detail: str
    disclaimer: str = DISCLAIMER


class SampleSummary(BaseModel):
    """One built-in sample recording and whether this process can serve it."""

    sample_id: str
    record_uid: str
    tasks: list[str]
    selection: str
    available: bool
    #: Why it cannot be served, when it cannot. `dataset/` is read-only input
    #: that is never committed, so absence is the fresh-clone answer.
    reason: str | None = None
    bytes: int | None = None


class PatientRule(BaseModel):
    """One patient-level collapse of several recordings."""

    rule: str
    predicted_class: str
    score: float | None = None
    score_display: str
    note: str


class PatientResponse(BaseModel):
    """Recording-level results plus every declared patient-level collapse."""

    task: str
    classes: list[str]
    positive_class: str
    n_recordings: int
    recordings: list[PredictResponse]
    rules: list[PatientRule]
    locations: dict[str, str]
    disclaimer: str = DISCLAIMER


def _display_block(payload: dict[str, Any]) -> dict[str, Any]:
    """Round the response once, in Python, through the shared formatter.

    `format_value` is the single rounding authority for the whole project
    (T85.6). Using it here rather than a local `round()` is the point: a live
    probability and a precomputed one are then rendered by the same code to the
    same three places, and a page that shows both is not showing two different
    rounding conventions.
    """
    probabilities = payload.get("probabilities") or {}
    timings = payload.get("timings_seconds") or {}
    quality = payload.get("quality") or {}
    threshold = payload.get("operating_threshold")
    duration = quality.get("duration_seconds")
    return {
        "probabilities": {
            str(name): format_value(value, "metric") for name, value in probabilities.items()
        },
        "probabilities_percent": {
            str(name): (
                format_value(None, "percent")
                if value is None
                else format_value(float(value) * 100.0, "percent") + "%"
            )
            for name, value in probabilities.items()
        },
        "confidence": format_value(payload.get("confidence"), "metric"),
        "margin": format_value(payload.get("margin"), "metric"),
        "low_confidence_margin": format_value(payload.get("low_confidence_margin"), "metric"),
        "operating_threshold": (None if threshold is None else format_value(threshold, "metric")),
        "duration_seconds": (
            None if duration is None else format_value(duration, "seconds") + " s"
        ),
        "timings_seconds": {
            str(name): format_value(value, "seconds") + " s" for name, value in timings.items()
        },
        "n_features": format_value(payload.get("n_features"), "count"),
        "n_missing_features": format_value(payload.get("n_missing_features"), "count"),
    }


# ---------------------------------------------------------------------------
# T108.1 -- the app
# ---------------------------------------------------------------------------


def _preload() -> list[str]:
    """Load every servable task's bundle once. Returns the tasks now resident.

    Unpickling is the expensive part of a prediction (~1 s for the binary
    bundle), and doing it per request would put that on every upload. A task
    that fails to load is logged and reported unavailable rather than taking the
    process down: a broken murmur bundle must not stop the binary task serving.
    """
    loaded: list[str] = []
    for task in available_tasks():
        try:
            load_bundle(task)
        except (ModelUnavailableError, OSError, ValueError) as error:
            log.warning("task %s has a model on disk that would not load: %s", task, error)
            continue
        loaded.append(task)
    log.info("preloaded %d of %d declared tasks: %s", len(loaded), len(TASKS), ", ".join(loaded))
    return loaded


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.loaded_tasks = _preload()
    try:
        yield
    finally:
        clear_bundle_cache()
        application.state.loaded_tasks = []


def create_app(*, static_root: Path | None = None, preload: bool = True) -> FastAPI:
    """Build the application.

    `preload=False` is for tests that must not spend a second unpickling a model
    they never call; it changes nothing about how a request is served, because
    `predict_recording` loads on demand through the same cache.
    """
    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "Screening inference for phonocardiogram recordings. "
            + DISCLAIMER
            + " Precomputed metrics are not served here: they reach the dashboard "
            "through build-time codegen."
        ),
        default_response_class=SafeJSONResponse,
        lifespan=lifespan if preload else None,
    )
    # Set on the router, not per route: every route registered from here on --
    # including any added after this function returns -- gets the coercion.
    application.router.route_class = SafeRoute
    application.state.loaded_tasks = []

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    _register_routes(application)
    _mount_static(application, static_root if static_root is not None else STATIC_ROOT)
    return application


def _register_routes(application: FastAPI) -> None:
    @application.get("/health", response_model=HealthResponse, tags=["status"])
    def health() -> HealthResponse:
        """T108.5: what this process can serve, and what it was built from."""
        resident = set(getattr(application.state, "loaded_tasks", []) or [])
        rows = task_report()
        statuses = [TaskStatus(**row, loaded=row["task"] in resident) for row in rows]
        available = [row for row in rows if row["available"]]
        return HealthResponse(
            status="ok" if available else "degraded",
            tasks=statuses,
            n_available=len(available),
            packages=_package_versions(),
            static_site_mounted=_static_dir(application) is not None,
        )

    @application.get("/manifest", response_model=ManifestResponse, tags=["status"])
    def manifest() -> ManifestResponse:
        """T108.5: the run id, git commit and generation timestamp of the build.

        Read from `frontend/lib/generated/manifest.json`, which the exporter
        writes. Nothing is recomputed here — if the export has not run, this
        says so rather than inventing a run id.
        """
        if not GENERATED_MANIFEST.is_file():
            return ManifestResponse(framework="PV-MEPCG / PulseVision", generated=False)
        try:
            payload = json.loads(GENERATED_MANIFEST.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            log.warning("generated manifest is unreadable: %s", error)
            return ManifestResponse(framework="PV-MEPCG / PulseVision", generated=False)
        fields = set(ManifestResponse.model_fields)
        return ManifestResponse(**{k: v for k, v in payload.items() if k in fields})

    @application.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
        responses={
            400: {"model": ErrorResponse, "description": "The upload is not a usable recording"},
            413: {"model": ErrorResponse, "description": "The upload is too large"},
            503: {"model": ErrorResponse, "description": "No model is available for that task"},
        },
    )
    async def predict(
        file: Annotated[UploadFile, File(description="A mono or multi-channel WAV recording")],
        task: Annotated[str, Form(description="One of the declared label spaces")] = "binary",
    ) -> PredictResponse:
        """T108.4: score one uploaded recording.

        The upload is streamed to a temporary file because `soundfile` and
        `librosa` read paths, and because validation must see the real bytes
        before anything decodes them. The temporary file is removed whatever
        happens — an upload is never kept.
        """
        if task not in TASKS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "unknown task "
                    + repr(task)
                    + ". Declared tasks: "
                    + ", ".join(TASKS)
                    + ". The five label spaces are separate and are never merged."
                ),
            )

        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        handle, temporary = tempfile.mkstemp(prefix="pvmepcg_upload_", suffix=suffix)
        os.close(handle)
        target = Path(temporary)
        try:
            written = await _spool(file, target)
            if written == 0:
                raise HTTPException(status_code=400, detail="the uploaded file is empty (0 bytes)")

            try:
                result, detail = predict_recording(target, task=task, with_detail=True)
            except AudioValidationError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except ModelUnavailableError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error

            return _response_for(result, source=file.filename or None, detail=detail)
        finally:
            with suppress(OSError):
                target.unlink()

    @application.get("/tasks", tags=["status"])
    def tasks() -> list[TaskStatus]:
        """The declared label spaces. Five separate tasks, never merged."""
        resident = set(getattr(application.state, "loaded_tasks", []) or [])
        return [TaskStatus(**row, loaded=row["task"] in resident) for row in task_report()]

    # -- T116.6: the built-in samples -------------------------------------
    #
    # No audio is committed for these. They are pinned corpus records that this
    # process serves from the operator's own read-only `dataset/`, because the
    # PhysioNet and PASCAL copies carry no licence file and redistributing them
    # would be asserting terms nobody verified. See `src/reporting/samples.py`.

    @application.get("/samples", response_model=list[SampleSummary], tags=["inference"])
    def samples() -> list[SampleSummary]:
        """Which built-in samples this process can actually reach.

        An empty list is a legitimate answer, not a failure: a checkout without
        `dataset/` has no audio to offer, and each entry says so individually
        rather than the endpoint disappearing.
        """
        rows: list[SampleSummary] = []
        for spec in SAMPLES:
            path = resolve_sample(spec.sample_id)
            rows.append(
                SampleSummary(
                    sample_id=spec.sample_id,
                    record_uid=spec.record_uid,
                    tasks=list(spec.tasks),
                    selection=spec.selection,
                    available=path is not None,
                    reason=(
                        None
                        if path is not None
                        else (
                            "the corpus recording for "
                            + spec.record_uid
                            + " is not on this machine. dataset/ is read-only input "
                            "and is never committed, so a fresh clone has none of it."
                        )
                    ),
                    bytes=None if path is None else path.stat().st_size,
                )
            )
        return rows

    @application.get(
        "/samples/{sample_id}/audio",
        tags=["inference"],
        responses={404: {"model": ErrorResponse, "description": "No such sample here"}},
    )
    def sample_audio(sample_id: str) -> Response:
        """The WAV for one built-in sample, so a page can draw its waveform.

        Served from the operator's own corpus copy and never cached to disk
        anywhere else. `FileResponse` streams it; `_sanitising` leaves a
        `Response` alone, so the bytes are not passed through the JSON coercer.
        """
        from fastapi.responses import FileResponse

        path = _resolved_sample(sample_id)
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @application.post(
        "/predict/sample",
        response_model=PredictResponse,
        tags=["inference"],
        responses={
            400: {"model": ErrorResponse, "description": "The recording is not usable"},
            404: {"model": ErrorResponse, "description": "No such sample here"},
            503: {"model": ErrorResponse, "description": "No model is available for that task"},
        },
    )
    def predict_sample(
        sample_id: Annotated[str, Form(description="One of the ids from GET /samples")],
        task: Annotated[str, Form(description="One of the declared label spaces")] = "binary",
    ) -> PredictResponse:
        """Score a built-in sample through exactly the same path as an upload.

        `use_cache` is left off inside `predict_recording` for a bare path, so
        this is the upload path with the file already on disk -- not a shortcut
        that could diverge from what a browser gets.
        """
        if task not in TASKS:
            raise HTTPException(status_code=400, detail=_unknown_task(task))
        path = _resolved_sample(sample_id)
        try:
            result, detail = predict_recording(path, task=task, with_detail=True)
        except AudioValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return _response_for(result, source=sample_id, detail=detail)

    # -- T117.3: reports ---------------------------------------------------
    #
    # Documents, not metrics. Each is rendered by the Phase 107 generators from
    # files already on disk (or, for a recording, from the one predict pass), and
    # handed to the browser as a download. The page never reads a number out of
    # one. Every temporary file is removed once the response has been sent.

    @application.get(
        "/report/experiment/{exp_id}",
        tags=["reports"],
        responses={404: {"model": ErrorResponse, "description": "No such experiment run"}},
    )
    def experiment_report(exp_id: str) -> Response:
        """One experiment run summarised from the files it wrote (T107.4)."""
        from src.reporting.sample_report import ReportError, render_experiment_report

        runs = reportable_experiments()
        if exp_id not in runs:
            raise HTTPException(
                status_code=404,
                detail="unknown experiment " + repr(exp_id) + ". Reportable: " + ", ".join(runs),
            )
        workdir = Path(tempfile.mkdtemp(prefix="pvmepcg_report_"))
        target = workdir / (exp_id + "_experiment_report.docx")
        try:
            render_experiment_report(runs[exp_id], target, title=exp_id)
        except ReportError as error:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _document(target, cleanup=workdir)

    @application.get(
        "/report/objectives",
        tags=["reports"],
        responses={404: {"model": ErrorResponse, "description": "T29 has not been generated"}},
    )
    def objectives_report() -> Response:
        """The objective-coverage report: T29's committed DOCX, served as written.

        T29 is generated by `scripts/37_result_tables.py` from the objective map.
        Re-rendering it here would be a second generator for the same document.
        """
        from fastapi.responses import FileResponse

        path = objective_report_path()
        if path is None:
            raise HTTPException(status_code=404, detail="T29 has not been generated")
        return FileResponse(path, media_type=DOCX_MEDIA_TYPE, filename=path.name)

    @application.post(
        "/report/sample",
        tags=["reports"],
        responses={
            400: {"model": ErrorResponse, "description": "The recording is not usable"},
            404: {"model": ErrorResponse, "description": "No such sample here"},
            503: {"model": ErrorResponse, "description": "No model is available for that task"},
        },
    )
    async def sample_report(
        task: Annotated[str, Form(description="One of the declared label spaces")] = "binary",
        sample_id: Annotated[str | None, Form(description="An id from GET /samples")] = None,
        file: Annotated[UploadFile | None, File(description="A WAV recording")] = None,
    ) -> Response:
        """One recording to one DOCX report, through the single predict pass (T107.1)."""
        from src.reporting.sample_report import report_for_recording

        if task not in TASKS:
            raise HTTPException(status_code=400, detail=_unknown_task(task))
        workdir = Path(tempfile.mkdtemp(prefix="pvmepcg_report_"))
        try:
            recording, stem = await _report_source(workdir, sample_id, file)
            target = workdir / (stem + "_" + task + "_report.docx")
            try:
                report_for_recording(recording, target, task=task, figures_dir=workdir)
            except AudioValidationError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except ModelUnavailableError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        return _document(target, cleanup=workdir)

    @application.post(
        "/predict/patient",
        response_model=PatientResponse,
        tags=["inference"],
        responses={
            400: {"model": ErrorResponse, "description": "The request or a recording is unusable"},
            404: {"model": ErrorResponse, "description": "No such sample here"},
            503: {"model": ErrorResponse, "description": "No model is available for that task"},
        },
    )
    def predict_patient(
        sample_ids: Annotated[str, Form(description="Comma-separated ids from GET /samples")],
        task: Annotated[str, Form(description="A CirCor label space")] = "murmur",
    ) -> PatientResponse:
        """T116.4: several recordings of one subject, collapsed three ways.

        CirCor labels the **subject**, not the recording, and screens at four
        auscultation locations. So a patient-level indication is a collapse over
        recordings, and which rule does the collapsing changes the answer --
        `max` re-thresholds a pooled score while `any_present` unions decisions
        already made, and they diverge whenever one recording is confident and
        the rest are not. All three declared rules are returned rather than one
        being chosen here, because choosing one silently is how a reader ends up
        comparing a number against a differently-aggregated one.

        The collapse is computed **here**, in Python, and not in the browser: it
        produces a reported indication, and a client that derived one would be a
        second implementation of the rule.
        """
        if task not in TASKS:
            raise HTTPException(status_code=400, detail=_unknown_task(task))
        ids = [item.strip() for item in sample_ids.split(",") if item.strip()]
        if not ids:
            raise HTTPException(status_code=400, detail="no sample ids were given")

        classes = list(TASKS[task].classes)
        known_locations = sample_locations()
        results: list[PredictResponse] = []
        locations: dict[str, str] = {}
        for sample_id in ids:
            path = _resolved_sample(sample_id)
            try:
                outcome, detail = predict_recording(path, task=task, with_detail=True)
            except AudioValidationError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except ModelUnavailableError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
            response = _response_for(outcome, source=sample_id, detail=detail)
            results.append(response)
            # From the audit, not from the id: a PhysioNet uid ends in a
            # record number and a CirCor one can end in a repeat index.
            locations[sample_id] = known_locations.get(sample_id) or "n/a"

        return PatientResponse(
            task=task,
            classes=classes,
            positive_class=classes[-1],
            n_recordings=len(results),
            recordings=results,
            rules=_patient_rules(results, classes),
            locations=locations,
        )


def _patient_rules(results: list[PredictResponse], classes: list[str]) -> list[PatientRule]:
    """The three declared collapses, applied to live recording-level results.

    The rule NAMES and their meanings come from
    `src.evaluation.aggregation.AGGREGATION_RULES`, which is what the CirCor
    experiments used, so the vocabulary is single-sourced. The arithmetic is
    repeated here rather than reused because `aggregate_predictions` collapses a
    frame that carries `y_true` -- it asserts a patient's recordings agree on
    their label, which is exactly the right check for an experiment and is
    meaningless at inference time, where there is no label. The equivalence is
    pinned by a test that runs both over the same scores.
    """
    positive = classes[-1]
    scores = [
        value
        for value in ((r.probabilities or {}).get(positive) for r in results)
        if value is not None
    ]
    decisions = [r.predicted_class == positive for r in results]

    rules: list[PatientRule] = []
    for rule in AGGREGATION_RULES:
        if rule in ("max", "mean") and not scores:
            rules.append(
                PatientRule(
                    rule=rule,
                    predicted_class="n/a",
                    score=None,
                    score_display=format_value(None, "metric"),
                    note="no probability was available for " + positive,
                )
            )
            continue
        score: float | None
        if rule == "max":
            pooled = max(scores)
            score = pooled
            predicted = positive if pooled >= 0.5 else classes[0]
            note = (
                "the highest "
                + positive
                + " probability across the recordings, re-thresholded at 0.5"
            )
        elif rule == "mean":
            pooled = sum(scores) / len(scores)
            score = pooled
            predicted = positive if pooled >= 0.5 else classes[0]
            note = "the mean " + positive + " probability across the recordings"
        else:  # any_present -- a union over decisions, not over scores
            score = max(scores) if scores else None
            predicted = positive if any(decisions) else classes[0]
            note = (
                "positive if ANY recording was predicted "
                + positive
                + ". A union over decisions already made, not over scores, so it "
                "can disagree with max."
            )
        rules.append(
            PatientRule(
                rule=rule,
                predicted_class=predicted,
                score=score,
                score_display=format_value(score, "metric"),
                note=note,
            )
        )
    return rules


def _unknown_task(task: str) -> str:
    return (
        "unknown task "
        + repr(task)
        + ". Declared tasks: "
        + ", ".join(TASKS)
        + ". The five label spaces are separate and are never merged."
    )


def _resolved_sample(sample_id: str) -> Path:
    """The sample's WAV on this machine, or a 404 that says which of two it is."""
    known = {spec.sample_id for spec in SAMPLES}
    if sample_id not in known:
        raise HTTPException(
            status_code=404,
            detail=(
                "unknown sample "
                + repr(sample_id)
                + ". Declared samples: "
                + ", ".join(sorted(known))
            ),
        )
    path = resolve_sample(sample_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "sample "
                + sample_id
                + " is declared but its recording is not on this machine. The "
                "samples are served from the operator's own read-only dataset/ "
                "copy; no corpus audio is committed to the repository."
            ),
        )
    return path


def _response_for(result: Any, *, source: str | None, detail: Any = None) -> PredictResponse:
    """One result to one response. Every prediction endpoint goes through here.

    The API adds the display strings, the caller's own name for the file and the
    per-recording explanation, and nothing else. Two endpoints building this
    separately is how they drift.
    """
    payload = result.to_dict()
    if source:
        payload["source"] = source
    payload["display"] = _display_block(payload)
    payload["explanation"] = _explanation_block(detail) if detail is not None else None
    return PredictResponse.model_validate(to_jsonable(payload))


#: How many terms the per-recording explanation carries. The same 20 as G19.
EXPLANATION_TOP = 20

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _explanation_block(detail: Any) -> dict[str, Any]:
    """T117.2's per-sample explanation, from the vector that was actually scored.

    `feature_contributions` is the Phase 107 report's own decomposition, so the
    explainability page and the downloaded report show the same terms. Formatted
    here through `format_value` for the same reason `_display_block` is.
    """
    from src.feature_extraction.registry import family_of
    from src.reporting.sample_report import ReportError, feature_contributions

    try:
        contributions, reason = feature_contributions(
            detail.bundle, detail.vector, top=EXPLANATION_TOP
        )
    except ReportError as error:
        contributions, reason = [], str(error)
    if not contributions:
        return {"available": False, "reason": reason, "rows": []}

    def family(name: str) -> str:
        try:
            return family_of(name)
        except (KeyError, ValueError):
            return "unknown"

    return {
        "available": True,
        "reason": None,
        "method": "exact linear decomposition (coefficient x scaled value)",
        "units": "log-odds of the positive class",
        "n_shown": len(contributions),
        "rows": [
            {
                "feature": item.name,
                "family": family(item.name),
                "contribution": item.contribution,
                "contribution_display": format_value(item.contribution, "metric"),
                "value_display": format_value(item.value, "metric"),
                "scaled_display": format_value(item.scaled, "metric"),
                "coefficient_display": format_value(item.coefficient, "metric"),
                "direction": item.direction,
            }
            for item in contributions
        ],
    }


async def _report_source(
    workdir: Path, sample_id: str | None, file: UploadFile | None
) -> tuple[Path, str]:
    """The recording a sample report is built from: a built-in sample or an upload."""
    if sample_id:
        return _resolved_sample(sample_id), sample_id
    if file is None:
        raise HTTPException(status_code=400, detail="send a sample_id or a file")
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    recording = workdir / ("upload" + suffix)
    if await _spool(file, recording) == 0:
        raise HTTPException(status_code=400, detail="the uploaded file is empty (0 bytes)")
    return recording, Path(file.filename or "upload").stem


def _document(path: Path, *, cleanup: Path) -> Response:
    """Stream a generated DOCX, then delete its working directory."""
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask

    return FileResponse(
        path,
        media_type=DOCX_MEDIA_TYPE,
        filename=path.name,
        background=BackgroundTask(shutil.rmtree, cleanup, True),
    )


async def _spool(file: UploadFile, target: Path) -> int:
    """Stream an upload to disk, refusing anything over the size bound."""
    written = 0
    with target.open("wb") as sink:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "the upload exceeds "
                        + str(MAX_UPLOAD_BYTES // (1024 * 1024))
                        + " MB, which is larger than any recording these models were "
                        "fitted on"
                    ),
                )
            sink.write(chunk)
    return written


def _package_versions() -> dict[str, str]:
    """Versions of the packages a prediction actually depends on."""
    from importlib.metadata import PackageNotFoundError, version

    names = ("numpy", "scipy", "scikit-learn", "librosa", "soundfile", "PyWavelets", "fastapi")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - every one is pinned
            versions[name] = "not installed"
    return versions


# ---------------------------------------------------------------------------
# T108.6 -- the exported site, served by this same process
# ---------------------------------------------------------------------------


def _mount_static(application: FastAPI, root: Path) -> None:
    """Serve `frontend/out/` at `/`, if it has been built.

    Mounted last on purpose: FastAPI resolves routes in registration order, so
    a mount at `/` would otherwise shadow `/health` and `/predict`. Absence is
    not an error — the API is useful on its own, and the export is a build
    artifact that a fresh clone will not have until `npm run build` has run.
    """
    if not root.is_dir():
        log.info("no exported site at %s; serving the API only", root)
        return

    from fastapi.staticfiles import StaticFiles

    application.mount("/", StaticFiles(directory=str(root), html=True), name="site")
    application.state.static_root = str(root)
    log.info("serving the exported dashboard from %s", root)


def _static_dir(application: FastAPI) -> str | None:
    return getattr(application.state, "static_root", None)


app = create_app()


def main() -> None:  # pragma: no cover - exercised by running the server
    """`python -m src.api.main` for a local run."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
