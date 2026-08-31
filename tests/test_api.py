"""Phase 108's gate: does the service hand back what the predictor produced?

T108.7 asks three things of this file — an upload succeeds, a bad file returns a
clean error, and a record producing NaN features serialises as `null` rather
than breaking JSON. The third is the one worth explaining.

**Why a NaN is a 500 and not a field.** Starlette renders JSON with
`allow_nan=False`, so one non-finite float anywhere in a response body raises
inside the renderer, after the endpoint has already returned successfully. The
client sees a 500 with no way to tell a missing measurement from a server that
fell over.

Where the coercion has to live was found by measurement, not by reading the
docs. A `default_response_class` looks app-wide and is not: FastAPI validates
and serialises an endpoint's return value through Pydantic *before* the response
class sees it, so a probe route returning `np.int64(3)` raised
`PydanticSerializationError` with `SafeJSONResponse` installed. The coercion
therefore lives in `SafeRoute`, a route class set once on the app's router,
which wraps the endpoint itself and runs in front of the serializer.
`test_the_encoder_is_wired_in_app_wide_not_per_endpoint` is the regression test
for that: it adds a **new** route after the app is built and asserts NaN comes
back as `null` without the route doing anything itself.

**The API adds nothing to a prediction.** `test_the_response_body_is_the_
predictor_result` compares the served JSON field by field against
`predict_recording(...).to_dict()`. If the API ever starts rounding, renaming or
"improving" a value, that test fails — which is the point, because a second
implementation of a probability is exactly what this project refuses.

**Bit-level note.** The probability in an API response is computed one row at a
time. Phase 106 measured that a single-record probability sits within 8 ULP of
the same model's batched value; a response and a stored fold value therefore
disagree in the last two bits, always. Nothing here compares the two, and
nothing should without that bound.

Everything needing `dataset/` or the saved bundle skips: both are gitignored, so
CI runs the schema, error-handling and encoder half of this file.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import (
    DEV_ORIGINS,
    MAX_UPLOAD_BYTES,
    SafeJSONResponse,
    create_app,
    to_jsonable,
)
from src.inference.predictor import DISCLAIMER, TASKS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "models_saved" / "binary" / "final"
PHYSIONET = PROJECT_ROOT / "dataset" / "archive (3)" / "training-a"


@pytest.fixture(scope="module")
def client() -> Any:
    """An app with no static mount and no preload.

    `static_root` points somewhere that does not exist so the route table is the
    API's own; the mount is tested separately. `preload=False` keeps the module
    fixture from unpickling a bundle that most tests here never touch --
    `predict_recording` loads through the same cache on demand, so nothing about
    request handling changes.
    """
    app = create_app(static_root=PROJECT_ROOT / "does-not-exist", preload=False)
    with TestClient(app) as running:
        yield running


def _wav(path: Path, seconds: float, fs: int = 2000, channels: int = 1) -> Path:
    """A real WAV file on disk. Silence is fine: this tests the API, not a model."""
    frames = int(seconds * fs)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(fs)
        samples = [int(3000 * math.sin(2 * math.pi * 40 * n / fs)) for n in range(frames)]
        handle.writeframes(b"".join(struct.pack("<h", value) for value in samples * channels))
    return path


# ---------------------------------------------------------------------------
# T108.3 -- the central encoder
# ---------------------------------------------------------------------------


def test_non_finite_floats_become_null_and_numpy_scalars_become_python() -> None:
    payload = {
        "nan": float("nan"),
        "inf": float("inf"),
        "neg_inf": float("-inf"),
        "finite": 0.25,
        "np_float": np.float64("nan"),
        "np_int": np.int64(7),
        "np_bool": np.bool_(True),
        "array": np.array([1.0, np.nan]),
        "nested": {"deep": [float("nan"), 3]},
    }
    cleaned = to_jsonable(payload)

    assert cleaned["nan"] is None
    assert cleaned["inf"] is None
    assert cleaned["neg_inf"] is None
    assert cleaned["finite"] == 0.25
    assert cleaned["np_float"] is None
    assert cleaned["np_int"] == 7 and isinstance(cleaned["np_int"], int)
    assert cleaned["np_bool"] is True
    assert cleaned["array"] == [1.0, None]
    assert cleaned["nested"]["deep"] == [None, 3]

    # The whole point: it now survives a strict dump.
    assert json.loads(json.dumps(cleaned, allow_nan=False)) == cleaned


def test_zero_is_never_confused_with_a_missing_measurement() -> None:
    """`0.0` is a result and `null` is an absence; the encoder must not merge them."""
    cleaned = to_jsonable({"measured": 0.0, "undefined": float("nan")})
    assert cleaned["measured"] == 0.0
    assert cleaned["undefined"] is None


def test_the_encoder_is_wired_in_app_wide_not_per_endpoint() -> None:
    """A route that does nothing about NaN still returns `null`, not a 500."""
    app = create_app(static_root=PROJECT_ROOT / "does-not-exist", preload=False)

    @app.get("/_nan_probe")
    def probe() -> dict[str, Any]:
        return {"value": float("nan"), "ratio": np.float64("inf"), "count": np.int64(3)}

    with TestClient(app) as running:
        response = running.get("/_nan_probe")

    assert response.status_code == 200
    assert response.json() == {"value": None, "ratio": None, "count": 3}
    assert "NaN" not in response.text


def test_the_response_class_refuses_to_emit_a_bare_nan_token() -> None:
    rendered = SafeJSONResponse(content={"x": float("nan")}).render({"x": float("nan")})
    assert b"NaN" not in rendered
    assert json.loads(rendered.decode("utf-8")) == {"x": None}


# ---------------------------------------------------------------------------
# T108.5 -- status endpoints
# ---------------------------------------------------------------------------


def test_health_reports_every_declared_task_including_the_unavailable_ones(
    client: Any,
) -> None:
    payload = client.get("/health").json()
    assert payload["framework"] == "PV-MEPCG / PulseVision"
    assert payload["disclaimer"] == DISCLAIMER

    reported = {row["task"] for row in payload["tasks"]}
    assert reported == set(TASKS), (
        "a task with no model must be reported unavailable with the reason, not "
        "dropped from the list"
    )
    for row in payload["tasks"]:
        assert row["available"] or row["reason"], row["task"] + " is unavailable with no reason"
    assert payload["n_available"] == sum(1 for row in payload["tasks"] if row["available"])
    assert set(payload["packages"]) >= {"numpy", "scikit-learn", "librosa"}


def test_health_never_serves_a_metric(client: Any) -> None:
    """Precomputed numbers reach the dashboard through codegen, never this API."""
    body = client.get("/health").text.lower()
    for forbidden in ("sensitivity", "balanced_accuracy", "roc_auc", "accuracy_mean"):
        assert forbidden not in body, (
            "the API is serving " + forbidden + "; the moment a metrics endpoint "
            "exists a page will fetch it at runtime and the codegen boundary is gone"
        )


def test_manifest_reports_what_produced_this_build(client: Any) -> None:
    payload = client.get("/manifest").json()
    assert payload["framework"]
    generated = PROJECT_ROOT / "frontend" / "lib" / "generated" / "manifest.json"
    if not generated.is_file():
        assert payload["generated"] is False
        return

    source = json.loads(generated.read_text(encoding="utf-8"))
    assert payload["generated"] is True
    assert payload["run_id"] == source["run_id"]
    assert payload["git_commit"] == source["git_commit"]
    assert payload["exported_utc"] == source["exported_utc"]


def test_the_five_label_spaces_are_listed_separately(client: Any) -> None:
    rows = client.get("/tasks").json()
    assert [row["task"] for row in rows] == list(TASKS)
    classes = {row["task"]: row["classes"] for row in rows}
    assert classes["binary"] != classes["pascal_a"]
    assert classes["pascal_a"] != classes["pascal_b"], (
        "PASCAL A and PASCAL B are separate label spaces and are never merged"
    )


# ---------------------------------------------------------------------------
# T108.4 -- bad input, answered cleanly
# ---------------------------------------------------------------------------


def test_an_unknown_task_is_a_400_naming_the_declared_tasks(client: Any, tmp_path: Path) -> None:
    wav = _wav(tmp_path / "clip.wav", 2.0)
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", wav.read_bytes(), "audio/wav")},
        data={"task": "murmur_and_outcome"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "unknown task" in detail
    for task in TASKS:
        assert task in detail


def test_a_non_wav_upload_is_a_400_explaining_why_lossy_audio_is_refused(
    client: Any, tmp_path: Path
) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_bytes(b"this is not audio")
    response = client.post(
        "/predict", files={"file": ("notes.txt", junk.read_bytes(), "text/plain")}
    )
    assert response.status_code == 400
    assert "WAV" in response.json()["detail"]


def test_a_wav_extension_over_bytes_that_are_not_audio_is_a_400(
    client: Any, tmp_path: Path
) -> None:
    """The extension is not the check; `soundfile` reading it is."""
    fake = tmp_path / "pretend.wav"
    fake.write_bytes(b"RIFF" + b"\x00" * 64)
    response = client.post(
        "/predict", files={"file": ("pretend.wav", fake.read_bytes(), "audio/wav")}
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_an_empty_upload_is_a_400_rather_than_a_crash(client: Any) -> None:
    response = client.post("/predict", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_a_recording_shorter_than_anything_in_training_is_refused(
    client: Any, tmp_path: Path
) -> None:
    clip = _wav(tmp_path / "tiny.wav", 0.1)
    response = client.post("/predict", files={"file": ("tiny.wav", clip.read_bytes(), "audio/wav")})
    assert response.status_code == 400
    assert "long" in response.json()["detail"]


def test_a_missing_file_field_is_a_422_from_the_schema(client: Any) -> None:
    response = client.post("/predict", data={"task": "binary"})
    assert response.status_code == 422


def test_an_oversized_upload_is_refused_before_it_is_scored(client: Any) -> None:
    payload = b"RIFF" + b"\x00" * (MAX_UPLOAD_BYTES + 1024)
    response = client.post("/predict", files={"file": ("huge.wav", payload, "audio/wav")})
    assert response.status_code == 413
    assert "MB" in response.json()["detail"]


def test_a_task_with_no_saved_model_is_a_503_not_a_500(client: Any, tmp_path: Path) -> None:
    """`pascal_a` has no bundle; the caller gets a service status, not a stack trace."""
    from src.inference.predictor import available_tasks

    missing = [task for task in TASKS if task not in available_tasks()]
    if not missing:
        pytest.skip("every declared task has a saved model in this checkout")

    wav = _wav(tmp_path / "clip.wav", 3.0, fs=4000)
    response = client.post(
        "/predict",
        files={"file": ("clip.wav", wav.read_bytes(), "audio/wav")},
        data={"task": missing[0]},
    )
    assert response.status_code == 503
    assert missing[0] in response.json()["detail"]


# ---------------------------------------------------------------------------
# CORS, and the mount
# ---------------------------------------------------------------------------


def test_cors_allows_the_dev_origin_and_nothing_else(client: Any) -> None:
    allowed = client.get("/health", headers={"Origin": DEV_ORIGINS[0]})
    assert allowed.headers.get("access-control-allow-origin") == DEV_ORIGINS[0]

    denied = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in denied.headers, (
        "a wildcard would let any page on the internet drive a model on this machine"
    )


def test_the_exported_site_is_served_when_it_exists_and_never_shadows_the_api(
    tmp_path: Path,
) -> None:
    site = tmp_path / "out"
    site.mkdir()
    (site / "index.html").write_text("<h1>PV-MEPCG / PulseVision</h1>", encoding="utf-8")

    app = create_app(static_root=site, preload=False)
    with TestClient(app) as running:
        assert running.get("/health").status_code == 200, (
            "the static mount at / has shadowed the API; it must be registered last"
        )
        page = running.get("/")
        assert page.status_code == 200
        assert "PulseVision" in page.text
        assert running.get("/health").json()["static_site_mounted"] is True


def test_a_missing_export_is_not_an_error(client: Any) -> None:
    assert client.get("/health").json()["static_site_mounted"] is False


# ---------------------------------------------------------------------------
# T108.4 -- the real upload
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recording() -> Path:
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout (gitignored)")
    if not PHYSIONET.is_dir():
        pytest.skip("dataset/ is not present in this checkout")
    recordings = sorted(PHYSIONET.glob("*.wav"))
    if not recordings:
        pytest.skip("no PhysioNet training-a recordings on disk")
    return recordings[0]


def test_an_upload_is_scored_and_the_body_matches_the_schema(client: Any, recording: Path) -> None:
    response = client.post(
        "/predict",
        files={"file": (recording.name, recording.read_bytes(), "audio/wav")},
        data={"task": "binary"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["task"] == "binary"
    assert payload["predicted_class"] in TASKS["binary"].classes
    assert set(payload["probabilities"]) == set(TASKS["binary"].classes)
    assert payload["disclaimer"] == DISCLAIMER
    assert payload["source"] == recording.name
    assert payload["n_features"] == 138
    assert payload["operating_threshold"] == 0.5
    assert "no in-fold selected threshold" in payload["operating_point_note"]
    assert math.isclose(sum(payload["probabilities"].values()), 1.0, rel_tol=1e-9)
    assert "NaN" not in response.text and "Infinity" not in response.text


def test_the_response_body_is_the_predictor_result(client: Any, recording: Path) -> None:
    """The API restates a prediction; it does not produce one.

    Every scalar the endpoint returns is compared against
    `predict_recording(...).to_dict()`. The two calls score the same bytes
    through the same single-row path, so equality here is exact -- there is no
    batch/single split to explain a difference away.
    """
    from src.inference.predictor import predict_recording

    direct = predict_recording(recording, task="binary").to_dict()
    served = client.post(
        "/predict",
        files={"file": (recording.name, recording.read_bytes(), "audio/wav")},
        data={"task": "binary"},
    ).json()

    assert served["predicted_class"] == direct["predicted_class"]
    assert served["predicted_index"] == direct["predicted_index"]
    assert served["confidence"] == direct["confidence"]
    assert served["margin"] == direct["margin"]
    assert served["probabilities"] == direct["probabilities"]
    assert served["low_confidence"] == direct["low_confidence"]
    assert served["n_missing_features"] == direct["n_missing_features"]
    assert served["model"]["model_id"] == direct["model"]["model_id"]
    assert served["quality"]["duration_seconds"] == direct["quality"]["duration_seconds"]


def test_a_stereo_upload_is_downmixed_rather_than_refused(
    client: Any, recording: Path, tmp_path: Path
) -> None:
    stereo = _wav(tmp_path / "stereo.wav", 5.0, fs=4000, channels=2)
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout (gitignored)")
    response = client.post(
        "/predict", files={"file": ("stereo.wav", stereo.read_bytes(), "audio/wav")}
    )
    assert response.status_code == 200, response.text
    assert response.json()["quality"]["channels"] == 2


def test_an_upload_is_not_kept_on_disk(client: Any, recording: Path) -> None:
    """A recording someone uploads is scored and deleted, whatever happens."""
    import tempfile

    spool = Path(tempfile.gettempdir())
    before = set(spool.glob("pvmepcg_upload_*"))
    client.post("/predict", files={"file": (recording.name, recording.read_bytes(), "audio/wav")})
    client.post("/predict", files={"file": ("bad.wav", b"RIFF not audio", "audio/wav")})
    assert set(spool.glob("pvmepcg_upload_*")) == before


def test_a_record_whose_features_cannot_be_computed_serialises_as_null(
    client: Any, tmp_path: Path
) -> None:
    """T108.7's third case, against a recording that really does produce NaN.

    A silent 0.6 s clip is a valid WAV and long enough to accept, but 24 of the
    138 features have no value on it -- chroma over a signal with no pitch,
    shape statistics over zero variance, a crest factor over zero RMS. The
    extractor returns NaN for those and the pipeline's imputer fills them, so a
    probability still comes back.

    Two things must hold and both are asserted. The body must be JSON a strict
    parser reads, with no bare `NaN` token. And the response must **say** the
    features were missing -- `n_missing_features` and a warning -- because a
    confident-looking probability over a quarter-imputed vector, presented
    without that, is the failure mode this whole project exists to avoid.
    """
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout (gitignored)")

    silent = tmp_path / "silence.wav"
    with wave.open(str(silent), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(4000)
        handle.writeframes(b"\x00\x00" * int(0.6 * 4000))

    response = client.post(
        "/predict", files={"file": ("silence.wav", silent.read_bytes(), "audio/wav")}
    )
    assert response.status_code == 200, response.text

    raw = response.text
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in raw, token + " is not JSON and no strict parser will read it"
    payload = json.loads(raw)

    assert payload["n_missing_features"] > 0, (
        "this clip is supposed to defeat the chroma and shape features; if it no "
        "longer does, the test has stopped covering the NaN path"
    )
    assert payload["warnings"], "the caller is not told the vector was imputed"
    assert str(payload["n_missing_features"]) in payload["warnings"][0]
    assert payload["feature_flags"], "the extractor's own flags are dropped on the floor"

    for value in payload["quality"].values():
        assert not isinstance(value, float) or math.isfinite(value)
