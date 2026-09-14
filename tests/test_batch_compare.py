"""Phase 131: batch analysis and compare (T131.1-T131.7).

The browser half of the gate is `frontend/e2e/batch.spec.ts`, which drives the
built Analyse page against the real API. This file holds the store and route
behaviour a batch rests on, the structure rules the page must keep, and T131.7's
server half: ten uploads under one batch id, one of them corrupt.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import math
import sys
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
PREDICT = FRONTEND / "components" / "predict"
PHYSIONET_A = PROJECT_ROOT / "dataset" / "archive (3)" / "training-a"

BATCH = "0123456789abcdef0123456789abcdef"
OTHER = "fedcba9876543210fedcba9876543210"
CORRUPT = b"this is not a recording, only text with a .wav name. " * 40


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_script(name: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "scripts" / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tone(seconds: float, rate: int = 2000) -> bytes:
    """A real mono 16-bit PCM WAV of a 40 Hz tone."""
    frames = int(seconds * rate)
    samples = bytearray()
    for n in range(frames):
        value = int(3000 * math.sin(2 * math.pi * 40 * n / rate))
        samples += value.to_bytes(2, "little", signed=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(rate)
        sink.writeframes(bytes(samples))
    return buffer.getvalue()


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _binary_payload(confidence: float, *, low: bool = False) -> dict[str, Any]:
    from src.inference.predictor import TASKS

    negative, positive = TASKS["binary"].classes[0], TASKS["binary"].classes[-1]
    return {
        "task": "binary",
        "predicted_class": positive if confidence >= 0.5 else negative,
        "confidence": confidence if confidence >= 0.5 else 1 - confidence,
        "probabilities": {negative: 1 - confidence, positive: confidence},
        "low_confidence": low,
        "scorable": True,
    }


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Any]:
    from src.api.history_store import HistoryStore

    history = HistoryStore(tmp_path / "history.json")
    yield history
    history.close()


# ---------------------------------------------------------------------------
# T131.3: one batch id
# ---------------------------------------------------------------------------


def test_t131_3_a_batch_id_is_32_lowercase_hex_or_nothing() -> None:
    from src.api.history_store import HistoryValidationError, clean_batch_id

    assert clean_batch_id(None) is None
    assert clean_batch_id("   ") is None
    assert clean_batch_id(BATCH) == BATCH
    for bad in (BATCH.upper(), "g" * 32, BATCH + "0", BATCH[:-1], "0123-4567", 42):
        with pytest.raises(HistoryValidationError):
            clean_batch_id(bad)


def test_t131_3_a_batch_reads_back_as_one(store: Any) -> None:
    from src.api.history_store import HistoryValidationError

    first = store.record_prediction(
        _binary_payload(0.9), file_name="a.wav", source_kind="upload", batch_id=BATCH
    )
    store.record_prediction(
        _binary_payload(0.2), file_name="b.wav", source_kind="upload", batch_id=BATCH
    )
    store.record_prediction(
        _binary_payload(0.7), file_name="c.wav", source_kind="upload", batch_id=OTHER
    )
    alone = store.record_prediction(_binary_payload(0.6), file_name="d.wav", source_kind="upload")

    assert first["batch_id"] == BATCH
    assert alone["batch_id"] is None
    listing = store.list_records(batch_id=BATCH, sort="file_name", order="asc")
    assert listing["total"] == 2
    assert [row["file_name"] for row in listing["items"]] == ["a.wav", "b.wav"]
    assert store.list_records()["total"] == 4
    with pytest.raises(HistoryValidationError):
        store.list_records(batch_id="not-a-batch")


def test_a_row_saved_before_phase_131_reads_as_no_batch() -> None:
    from src.api.history_store import HistoryStore

    row = HistoryStore.present(
        {
            "id": "x",
            "created_at": "2026-09-13T00:00:00.000000Z",
            "file_name": "old.wav",
            "task": "binary",
        }
    )
    assert row["batch_id"] is None


# ---------------------------------------------------------------------------
# T131.6: the CSV, written from History
# ---------------------------------------------------------------------------


def test_t131_6_export_reads_scored_rows_from_history_in_the_order_given(store: Any) -> None:
    from src.inference.predictor import TASKS
    from src.reporting.tables import format_value

    classes = TASKS["binary"].classes
    a = store.record_prediction(
        _binary_payload(0.8731), file_name="a.wav", source_kind="upload", batch_id=BATCH
    )
    b = store.record_prediction(
        _binary_payload(0.4512, low=True), file_name="b.wav", source_kind="upload", batch_id=BATCH
    )
    text = store.export_batch_csv(
        batch_id=BATCH,
        task="binary",
        rows=[
            {"file_name": "b.wav", "status": "scored", "history_id": b["id"]},
            {
                "file_name": "corrupt.wav",
                "status": "failed",
                "message": "corrupt.wav could not\nbe read",
            },
            {"file_name": "a.wav", "status": "scored", "history_id": a["id"]},
            {"file_name": "later.wav", "status": "cancelled"},
            {"file_name": "quiet.wav", "status": "not_scored", "message": "too quiet to screen"},
        ],
    )
    assert text.splitlines()[0].split(",") == [
        "file_name",
        "status",
        "task",
        "result",
        "confidence",
        "low_confidence",
        *["probability_" + name for name in classes],
        "message",
        "history_id",
        "created_at",
        "batch_id",
    ]
    rows = _rows(text)
    assert [row["file_name"] for row in rows] == [
        "b.wav",
        "corrupt.wav",
        "a.wav",
        "later.wav",
        "quiet.wav",
    ]
    assert [row["status"] for row in rows] == [
        "scored",
        "failed",
        "scored",
        "cancelled",
        "not_scored",
    ]
    assert {row["batch_id"] for row in rows} == {BATCH}
    for row, saved in ((rows[0], b), (rows[2], a)):
        assert row["result"] == saved["result"]
        assert row["confidence"] == format_value(saved["confidence"], "metric")
        assert row["confidence"] == saved["display"]["confidence"]
        for name in classes:
            assert row["probability_" + name] == format_value(
                saved["probabilities"][name], "metric"
            )
        assert row["history_id"] == saved["id"]
        assert row["created_at"] == saved["created_at"]
        assert row["message"] == ""
    assert rows[0]["low_confidence"] == "true"
    assert rows[2]["low_confidence"] == "false"
    assert rows[1]["message"] == "corrupt.wav could not be read"
    for row in (rows[1], rows[3], rows[4]):
        assert row["result"] == row["confidence"] == row["history_id"] == ""


def test_t131_6_export_writes_formula_looking_text_as_text(store: Any) -> None:
    saved = store.record_prediction(
        _binary_payload(0.9), file_name="=HYPERLINK(1).wav", source_kind="upload", batch_id=BATCH
    )
    rows = _rows(
        store.export_batch_csv(
            batch_id=BATCH,
            task="binary",
            rows=[
                {"file_name": "=HYPERLINK(1).wav", "status": "scored", "history_id": saved["id"]},
                {"file_name": "@x.wav", "status": "failed", "message": "-1 frames"},
            ],
        )
    )
    assert rows[0]["file_name"] == "'=HYPERLINK(1).wav"
    assert rows[1]["file_name"] == "'@x.wav"
    assert rows[1]["message"] == "'-1 frames"


def test_t131_6_export_refuses_what_it_cannot_vouch_for(store: Any) -> None:
    from src.api.history_store import MAX_BATCH_ROWS, HistoryValidationError
    from src.inference.predictor import TASKS

    ours = store.record_prediction(
        _binary_payload(0.9), file_name="a.wav", source_kind="upload", batch_id=BATCH
    )
    theirs = store.record_prediction(
        _binary_payload(0.9), file_name="z.wav", source_kind="upload", batch_id=OTHER
    )
    pascal_b = TASKS["pascal_b"].classes
    other_task = store.record_prediction(
        {
            "task": "pascal_b",
            "predicted_class": pascal_b[0],
            "confidence": 0.7,
            "probabilities": {name: 1 / len(pascal_b) for name in pascal_b},
            "scorable": True,
        },
        file_name="p.wav",
        source_kind="upload",
        batch_id=BATCH,
    )

    cases: list[tuple[dict[str, Any], str]] = [
        (
            {"rows": [{"file_name": "z.wav", "status": "scored", "history_id": theirs["id"]}]},
            "no History entry",
        ),
        (
            {"rows": [{"file_name": "a.wav", "status": "scored", "history_id": "missing"}]},
            "no History entry",
        ),
        (
            {"rows": [{"file_name": "p.wav", "status": "scored", "history_id": other_task["id"]}]},
            "another task",
        ),
        ({"rows": [{"file_name": "x.wav", "status": "failed"}]}, "needs the message"),
        ({"rows": [{"file_name": "x.wav", "status": "done"}]}, "status must be one of"),
        ({"rows": []}, "no rows"),
        (
            {"rows": [{"file_name": "x.wav", "status": "cancelled"}] * (MAX_BATCH_ROWS + 1)},
            "limited to",
        ),
        ({"batch_id": "nope", "rows": [{"file_name": "a.wav", "status": "cancelled"}]}, "batch_id"),
        (
            {"task": "sevenclass", "rows": [{"file_name": "a.wav", "status": "cancelled"}]},
            "unknown task",
        ),
    ]
    for overrides, message in cases:
        arguments = {"batch_id": BATCH, "task": "binary", **overrides}
        with pytest.raises(HistoryValidationError, match=message):
            store.export_batch_csv(**arguments)
    # The good row alone still exports: the refusals above were about the rows.
    assert (
        len(
            _rows(
                store.export_batch_csv(
                    batch_id=BATCH,
                    task="binary",
                    rows=[{"file_name": "a.wav", "status": "scored", "history_id": ours["id"]}],
                )
            )
        )
        == 1
    )


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    app = create_app(
        static_root=tmp_path / "no-site", preload=False, history_path=tmp_path / "history.json"
    )
    with TestClient(app) as test_client:
        yield test_client


def test_t131_3_a_malformed_batch_id_is_refused_before_any_work(client: Any) -> None:
    response = client.post(
        "/predict",
        files={"file": ("a.wav", _tone(3), "audio/wav")},
        data={"task": "binary", "batch_id": "not-a-batch"},
    )
    assert response.status_code == 400
    assert "batch_id" in response.json()["detail"]
    assert client.get("/api/history").json()["total"] == 0


def test_t131_5_refused_files_keep_their_reason_and_are_never_saved(client: Any) -> None:
    """Validation runs before a model loads, so this needs no bundle."""
    cases = [
        ("empty.wav", b"", "empty"),
        ("notes.mp3", _tone(3), "not a WAV file"),
        ("corrupt.wav", CORRUPT, "could not be read as audio"),
        ("long.wav", _tone(151), "beyond the 150 s"),
        ("short.wav", _tone(0.2), "shortest recording"),
    ]
    for name, payload, reason in cases:
        response = client.post(
            "/predict",
            files={"file": (name, payload, "audio/wav")},
            data={"task": "binary", "batch_id": BATCH},
        )
        assert response.status_code == 400, name + ": " + response.text
        assert reason in response.json()["detail"], name
    assert client.get("/api/history", params={"batch_id": BATCH}).json()["total"] == 0


def test_t131_6_the_export_route_answers_a_csv_download(client: Any) -> None:
    saved = client.app.state.history.record_prediction(
        _binary_payload(0.9), file_name="a.wav", source_kind="upload", batch_id=BATCH
    )
    body = {
        "batch_id": BATCH,
        "task": "binary",
        "rows": [
            {"file_name": "a.wav", "status": "scored", "history_id": saved["id"]},
            {
                "file_name": "bad.wav",
                "status": "failed",
                "message": "bad.wav could not be read as audio",
            },
        ],
    }
    response = client.post("/api/history/export", json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        'filename="pv-mepcg-batch-binary-01234567.csv"' in response.headers["content-disposition"]
    )
    rows = _rows(response.content.decode("utf-8"))
    assert [row["status"] for row in rows] == ["scored", "failed"]
    assert rows[0]["confidence"] == saved["display"]["confidence"]

    for broken in (
        {**body, "rows": [{"file_name": "a.wav", "status": "done"}]},
        {**body, "task": "sevenclass"},
        {**body, "batch_id": OTHER},
    ):
        refused = client.post("/api/history/export", json=broken)
        assert refused.status_code == 400, refused.text


# ---------------------------------------------------------------------------
# T131.1, T131.2, T131.4, T131.6: the page's structure
# ---------------------------------------------------------------------------


def test_t131_1_files_go_one_at_a_time_and_cancel_stops_what_has_not_started() -> None:
    panel = _read(PREDICT / "BatchPanel.tsx")
    assert "for (const row of queue)" in panel
    assert "await predictFile(row.file, task, { batchId: id })" in panel
    assert "Promise.all" not in panel, "files are scored sequentially"
    # Cancel never aborts the request in flight: the server would still save it.
    assert "AbortController" not in panel
    assert "cancelRequested.current" in panel
    assert "Cancel batch" in panel
    assert "validateRecording(file)" in panel


def test_t131_2_the_table_sorts_and_renders_server_strings() -> None:
    panel = _read(PREDICT / "BatchPanel.tsx")
    assert "aria-sort=" in panel
    assert "sortRows(rows, sort)" in panel
    assert "row.result?.display.confidence" in panel
    assert "batchSummary(rows, classes)" in panel
    helpers = _read(FRONTEND / "lib" / "batch.ts")
    assert "export function sortRows" in helpers
    assert "export function batchSummary" in helpers


def test_t131_4_the_compare_view_is_reusable_and_subtracts_nothing() -> None:
    view = _read(PREDICT / "CompareView.tsx")
    assert "export interface CompareItem" in view
    assert "<WaveformPlayer" in view
    assert "Audio not kept" in view, "Phase 132 opens it from History, which keeps no audio"
    assert "item.confidenceDisplay" in view
    assert "item.probabilitiesDisplay[name]" in view
    assert "different sets of categories" in view
    assert "<CompareView" in _read(PREDICT / "BatchPanel.tsx")


def test_t131_6_the_page_asks_the_api_for_the_csv() -> None:
    panel = _read(PREDICT / "BatchPanel.tsx")
    assert "exportBatchCsv(batchId, task, exportRows(shown))" in panel
    assert "new Blob" not in panel
    assert "'/api/history/export'" in _read(FRONTEND / "lib" / "api.ts")


def test_batch_mode_is_inside_the_analyse_panel_under_the_chosen_check() -> None:
    panel = _read(PREDICT / "PredictionPanel.tsx")
    assert "<BatchPanel" in panel
    assert "key={task}" in panel
    assert "BatchPanel" not in _read(FRONTEND / "app" / "page.tsx")


def test_the_batch_components_pass_the_metric_guard_rail() -> None:
    guard = _load_script("16_check_no_hardcoded_metrics.py", "metric_guard_131")
    paths = [
        PREDICT / "BatchPanel.tsx",
        PREDICT / "CompareView.tsx",
        PREDICT / "PredictionPanel.tsx",
    ]
    findings: list[Any] = []
    suppressed: list[Any] = []
    for path in paths:
        found, allowed = guard.scan_file(path)
        findings.extend(found)
        suppressed.extend(allowed)
    assert not findings, "\n".join(str(finding) for finding in findings)
    assert not suppressed, str(suppressed)
    for path in [*paths, FRONTEND / "lib" / "batch.ts"]:
        body = _read(path)
        for forbidden in ("toFixed(", "toPrecision(", "Math.round("):
            assert forbidden not in body, path.name + " formats a number: " + forbidden


# ---------------------------------------------------------------------------
# T131.7: ten uploads under one batch id, one of them corrupt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_bundle() -> object:
    """The binary bundle, or a module-wide skip (`*.joblib` is gitignored)."""
    try:
        from src.inference.predictor import load_bundle

        return load_bundle("binary")
    except Exception as error:  # noqa: BLE001 -- a missing bundle is a skip, not a failure
        pytest.skip("binary model unavailable (" + type(error).__name__ + ")")


def test_t131_7_ten_files_one_corrupt_give_nine_results_one_error_nine_rows_and_a_matching_csv(
    binary_bundle: object, client: Any
) -> None:
    real = sorted(PHYSIONET_A.glob("*.wav"), key=str)[:9] if PHYSIONET_A.is_dir() else []
    if len(real) < 9:
        pytest.skip("dataset/ is not in this checkout")
    uploads: list[tuple[str, bytes]] = [(path.name, path.read_bytes()) for path in real]
    uploads.insert(4, ("corrupt.wav", CORRUPT))

    table: list[dict[str, str]] = []
    export: list[dict[str, Any]] = []
    for name, payload in uploads:
        response = client.post(
            "/predict",
            files={"file": (name, payload, "audio/wav")},
            data={"task": "binary", "batch_id": BATCH},
        )
        if response.status_code == 200:
            body = response.json()
            assert body["scorable"] is True, name
            assert body["history_saved"] is True, name
            table.append(
                {
                    "file_name": name,
                    "status": "scored",
                    "result": body["predicted_class"],
                    "confidence": body["display"]["confidence"],
                    "message": "",
                }
            )
            export.append({"file_name": name, "status": "scored", "history_id": body["history_id"]})
        else:
            assert response.status_code == 400, name + ": " + response.text
            detail = response.json()["detail"]
            table.append(
                {
                    "file_name": name,
                    "status": "failed",
                    "result": "",
                    "confidence": "",
                    "message": detail,
                }
            )
            export.append({"file_name": name, "status": "failed", "message": detail})

    assert [row["status"] for row in table].count("scored") == 9
    failed = [row for row in table if row["status"] == "failed"]
    assert [row["file_name"] for row in failed] == ["corrupt.wav"]
    assert "could not be read as audio" in failed[0]["message"]

    listing = client.get("/api/history", params={"batch_id": BATCH, "page_size": 100}).json()
    assert listing["total"] == 9
    assert sorted(row["file_name"] for row in listing["items"]) == sorted(
        path.name for path in real
    )

    response = client.post(
        "/api/history/export", json={"batch_id": BATCH, "task": "binary", "rows": export}
    )
    assert response.status_code == 200, response.text
    rows = _rows(response.content.decode("utf-8"))
    assert len(rows) == 10
    for written, shown in zip(rows, table, strict=True):
        for column in ("file_name", "status", "result", "confidence"):
            assert written[column] == shown[column], (column, shown["file_name"])
        assert written["message"] == " ".join(shown["message"].split())
