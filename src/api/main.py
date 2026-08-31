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
from src.utils.logging_setup import get_logger

__all__ = [
    "API_TITLE",
    "API_VERSION",
    "DEV_ORIGINS",
    "MAX_UPLOAD_BYTES",
    "HealthResponse",
    "ManifestResponse",
    "PredictResponse",
    "SafeJSONResponse",
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

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    detail: str
    disclaimer: str = DISCLAIMER


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
                result = predict_recording(target, task=task)
            except AudioValidationError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            except ModelUnavailableError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error

            payload = result.to_dict()
            payload["source"] = file.filename or payload["source"]
            return PredictResponse.model_validate(to_jsonable(payload))
        finally:
            with suppress(OSError):
                target.unlink()

    @application.get("/tasks", tags=["status"])
    def tasks() -> list[TaskStatus]:
        """The declared label spaces. Five separate tasks, never merged."""
        resident = set(getattr(application.state, "loaded_tasks", []) or [])
        return [TaskStatus(**row, loaded=row["task"] in resident) for row in task_report()]


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
