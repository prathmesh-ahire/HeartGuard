"""Phase 130: the Analyse page (T130.1-T130.7).

The browser half of the gate is `frontend/e2e/analyse.spec.ts` and
`frontend/e2e/record.spec.ts`, which drive the built page against the real API.
This file holds what needs no browser: the structure rules the page must keep,
and T130.7's server half -- a known record, uploaded the way the page uploads
it, gets the class its stored out-of-fold prediction recorded and lands in
History.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
PAGE = FRONTEND / "app" / "page.tsx"
PREDICT = FRONTEND / "components" / "predict"
AUDIO = FRONTEND / "components" / "audio"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_script(name: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "scripts" / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# T130.1-T130.6: the page's structure
# ---------------------------------------------------------------------------


def test_t130_4_three_checks_each_with_one_plain_description() -> None:
    body = _read(PAGE)
    for label in ("Normal or abnormal", "Sound type", "Murmur and outcome"):
        assert "label: '" + label + "'" in body
    assert body.count("description: '") == 3
    for task in ("binary", "pascal_a", "pascal_b", "murmur", "outcome"):
        assert body.count("task: '" + task + "'") == 1, task + " must be offered exactly once"


def test_t130_4_pascal_a_and_b_are_two_tasks_under_one_check_never_one() -> None:
    body = _read(PAGE)
    sound = body[body.index("label: 'Sound type'") : body.index("label: 'Murmur and outcome'")]
    assert "task: 'pascal_a'" in sound
    assert "task: 'pascal_b'" in sound
    selector = _read(PREDICT / "CheckSelector.tsx")
    # One task is selected and sent: the panel holds a single `task` string.
    assert "onChange: (task: string) => void" in selector


def test_t130_1_and_t130_2_a_recording_takes_the_same_path_as_an_upload() -> None:
    panel = _read(PREDICT / "PredictionPanel.tsx")
    assert re.search(r"<FileUpload\s+onFile=\{acceptFile\}", panel)
    assert re.search(r"<MicRecorder\s+onFile=\{acceptFile\}", panel)
    recorder = _read(AUDIO / "MicRecorder.tsx")
    assert "encodeWav(" in recorder
    assert "new MediaRecorder" not in recorder, "MediaRecorder only writes lossy formats"
    assert "type: 'audio/wav'" in recorder
    # The upload control still validates format and size before the network.
    upload = _read(FRONTEND / "components" / "ui" / "FileUpload.tsx")
    assert "validateRecording(file)" in upload


def test_t130_3_the_player_is_on_analyse_and_wavesurfer_loads_lazily() -> None:
    player = _read(AUDIO / "WaveformPlayer.tsx")
    assert "await import('wavesurfer.js')" in player
    assert not re.search(r"^import .*wavesurfer", player, re.M), "a static import is first-load"
    # Peaks from the page's own PCM parser, so a 2 kHz file never meets Web Audio.
    assert "peaks: [envelope(" in player
    assert "duration: decoded.duration" in player
    # Chromium's <audio> refuses a 2 kHz WAV, so a low-rate file is PLAYED from
    # a resampled copy -- and only played: the panel still sends the original.
    assert "decoded.sampleRate < PLAYBACK_MIN_RATE" in player
    assert "resampleLinear(decoded.samples" in player
    assert "predictFile(file, task)" in _read(PREDICT / "PredictionPanel.tsx")
    assert "<WaveformPlayer" in _read(PREDICT / "PredictionPanel.tsx")
    budget = _load_script("20_check_bundle_budget.py", "bundle_budget_130")
    assert "wavesurfer.js" not in budget.NOT_YET_BUNDLED
    assert "wavesurfer.js" in budget.LAZY_MARKERS


def test_t130_5_the_result_card_shows_server_strings() -> None:
    card = _read(PREDICT / "ResultCard.tsx")
    assert "display={result.display.confidence}" in card
    assert "result.display.probabilities[name]" in card
    assert "result.low_confidence" in card
    gauge = _read(PREDICT / "ConfidenceGauge.tsx")
    assert "{display}" in gauge


def test_t130_6_the_result_links_to_its_history_row_and_offers_the_report() -> None:
    card = _read(PREDICT / "ResultCard.tsx")
    assert "'/history/?record=' + encodeURIComponent(result.history_id" in card
    assert "result.history_note" in card
    panel = _read(PREDICT / "PredictionPanel.tsx")
    assert "sampleReport(task, source)" in panel
    assert "saveDocument(" in panel


def test_the_new_components_pass_the_metric_guard_rail() -> None:
    guard = _load_script("16_check_no_hardcoded_metrics.py", "metric_guard_130")
    findings: list[Any] = []
    suppressed: list[Any] = []
    for path in [PAGE, *sorted(PREDICT.glob("*.tsx")), *sorted(AUDIO.glob("*.tsx"))]:
        found, allowed = guard.scan_file(path)
        findings.extend(found)
        suppressed.extend(allowed)
    assert not findings, "\n".join(str(finding) for finding in findings)
    assert not suppressed, str(suppressed)
    for path in sorted(AUDIO.glob("*.tsx")):
        body = _read(path)
        for forbidden in ("toFixed(", "toPrecision(", "Math.round("):
            assert forbidden not in body, path.name + " formats a number: " + forbidden


# ---------------------------------------------------------------------------
# T130.7: a known record through the upload path, against its stored class
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_bundle() -> object:
    """The binary bundle, or a module-wide skip (`*.joblib` is gitignored)."""
    try:
        from src.inference.predictor import load_bundle

        return load_bundle("binary")
    except Exception as error:  # noqa: BLE001 -- a missing bundle is a skip, not a failure
        pytest.skip("binary model unavailable (" + type(error).__name__ + ")")


@pytest.mark.parametrize("sample_id", ["binary-abnormal", "binary-normal"])
def test_t130_7_an_uploaded_known_record_matches_its_stored_class_and_is_in_history(
    binary_bundle: object, sample_id: str, tmp_path: Path
) -> None:
    """The class, not the probability: the deployed bundle is refit in sample."""
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    from src.reporting.samples import SAMPLES, resolve_sample, stored_reference

    spec = next(item for item in SAMPLES if item.sample_id == sample_id)
    path = resolve_sample(sample_id)
    if path is None or not Path(path).is_file():
        pytest.skip(sample_id + " is not reachable; dataset/ is not in this checkout")
    reference = stored_reference().get(spec.record_uid)
    if reference is None:
        pytest.skip("EXP-A2 predictions are not in this checkout")
    assert reference["folds_agree"]
    stored = reference["predicted_classes"][0]

    app = create_app(
        static_root=tmp_path / "no-site", preload=False, history_path=tmp_path / "history.json"
    )
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": (Path(path).name, Path(path).read_bytes(), "audio/wav")},
            data={"task": "binary"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["predicted_class"] == stored, sample_id + " vs stored out-of-fold " + stored
        assert body["history_saved"] is True
        assert body["history_id"]

        row = client.get("/api/history/" + body["history_id"])
        assert row.status_code == 200
        saved = row.json()
        assert saved["result"] == stored
        assert saved["task"] == "binary"
        assert saved["source_kind"] == "upload"
        assert saved["file_name"] == Path(path).name
