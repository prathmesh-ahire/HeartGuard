"""Phase 132: the History page (T132.1-T132.7).

The browser half of the gate is `frontend/e2e/history.spec.ts`, which drives the
built page against the real API and reads the TinyDB file after every change.
This file holds what the page rests on -- the undo of a deletion, the filters as
the page sends them -- checked against the database FILE, a reopened store (a
reload), and the structure rules the page must keep.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
HISTORY = FRONTEND / "components" / "history"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _on_disk(path: Path) -> dict[str, dict[str, Any]]:
    """The rows TinyDB holds in the file, by record id -- not what the store caches."""
    if not path.exists():
        return {}
    table = json.loads(path.read_text(encoding="utf-8")).get("records", {})
    return {row["id"]: row for row in table.values()}


def _load_script(name: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / "scripts" / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(file_name: str, result: str, confidence: float, **extra: Any) -> dict[str, Any]:
    from src.inference.predictor import TASKS

    classes = TASKS["binary"].classes
    other = next(name for name in classes if name != result)
    return {
        "file_name": file_name,
        "task": "binary",
        "result": result,
        "confidence": confidence,
        "probabilities": {result: confidence, other: 1 - confidence},
        **extra,
    }


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Any]:
    from src.api.history_store import HistoryStore

    history = HistoryStore(tmp_path / "history.json")
    yield history
    history.close()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from src.api.main import create_app

    app = create_app(
        static_root=tmp_path / "no-site", preload=False, history_path=tmp_path / "history.json"
    )
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# T132.4: delete one with undo, delete all
# ---------------------------------------------------------------------------


def test_t132_4_undo_puts_back_the_same_entry_in_the_file(store: Any, tmp_path: Path) -> None:
    from src.api.history_store import HistoryStore
    from src.inference.predictor import TASKS

    classes = TASKS["binary"].classes
    saved = store.create(**_row("a.wav", classes[-1], 0.8), notes="keep", tags=["x"])
    store.create(**_row("b.wav", classes[0], 0.7))
    path = tmp_path / "history.json"

    assert store.delete(saved["id"]) is True
    assert saved["id"] not in _on_disk(path), "a deletion reaches the file at once"

    restored = store.restore(saved["id"])
    assert restored is not None
    on_disk = _on_disk(path)[saved["id"]]
    for field in ("id", "created_at", "notes", "tags", "result", "confidence", "probabilities"):
        assert on_disk[field] == saved[field], field
    assert store.restore(saved["id"]) is None, "an undo is used once"

    reopened = HistoryStore(path)
    try:
        assert reopened.get(saved["id"]) is not None, "the restored entry survives a reload"
        assert reopened.restore(saved["id"]) is None, "undo lives in memory, not in the file"
    finally:
        reopened.close()


def test_t132_4_delete_all_ends_every_undo_and_old_undos_expire(store: Any) -> None:
    from src.api.history_store import UNDO_DEPTH
    from src.inference.predictor import TASKS

    positive = TASKS["binary"].classes[-1]
    rows = [store.create(**_row(str(i) + ".wav", positive, 0.9)) for i in range(UNDO_DEPTH + 1)]
    for row in rows:
        store.delete(row["id"])
    assert store.restore(rows[0]["id"]) is None, "the oldest undo was pushed out"
    assert store.restore(rows[-1]["id"]) is not None

    kept = store.create(**_row("k.wav", positive, 0.9))
    store.delete(kept["id"])
    assert store.delete_all() == 1
    assert store.restore(kept["id"]) is None


def test_t132_4_the_restore_route(client: Any) -> None:
    from src.inference.predictor import TASKS

    created = client.post("/api/history", json=_row("a.wav", TASKS["binary"].classes[0], 0.6))
    record_id = created.json()["id"]
    assert client.post("/api/history/" + record_id + "/restore").status_code == 404
    assert client.delete("/api/history/" + record_id).status_code == 200
    restored = client.post("/api/history/" + record_id + "/restore")
    assert restored.status_code == 200, restored.text
    assert restored.json()["id"] == record_id
    assert restored.json()["created_at"] == created.json()["created_at"]
    gone = client.post("/api/history/" + record_id + "/restore")
    assert gone.status_code == 404
    assert "can no longer be undone" in gone.json()["detail"]


# ---------------------------------------------------------------------------
# T132.7, server half: create, filter, edit, delete -- in the file, after a reload
# ---------------------------------------------------------------------------


def test_t132_7_every_change_is_in_tinydb_and_survives_a_reload(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    from src.inference.predictor import TASKS

    negative, positive = TASKS["binary"].classes[0], TASKS["binary"].classes[-1]
    path = tmp_path / "history.json"

    def app() -> Any:
        return create_app(static_root=tmp_path / "no-site", preload=False, history_path=path)

    with TestClient(app()) as api:
        ids = [
            api.post("/api/history", json=_row(name, result, 0.8)).json()["id"]
            for name, result in (
                ("alpha.wav", positive),
                ("beta.wav", negative),
                ("gamma.wav", positive),
            )
        ]
        assert set(_on_disk(path)) == set(ids)

        def names(**params: Any) -> list[str]:
            body = api.get("/api/history", params={"page_size": 100, **params})
            assert body.status_code == 200, body.text
            return sorted(row["file_name"] for row in body.json()["items"])

        assert names(q="beta") == ["beta.wav"]
        assert names(task="binary", result=positive) == ["alpha.wav", "gamma.wav"]
        # The page sends the viewer's day as full ISO times, not a UTC date.
        now = datetime.now(UTC)
        today = {
            "date_from": (now - timedelta(hours=1)).isoformat(),
            "date_to": (now + timedelta(hours=1)).isoformat(),
        }
        assert names(**today) == ["alpha.wav", "beta.wav", "gamma.wav"]
        assert names(date_from="2020-01-01", date_to="2020-01-02") == []

        edited = api.patch(
            "/api/history/" + ids[0], json={"notes": "follow up", "tags": ["clinic-a", "Clinic-A"]}
        )
        assert edited.status_code == 200
        assert _on_disk(path)[ids[0]]["notes"] == "follow up"
        assert _on_disk(path)[ids[0]]["tags"] == ["clinic-a"]
        assert names(tag="CLINIC-A") == ["alpha.wav"]
        assert names(q="follow") == ["alpha.wav"]

        assert api.delete("/api/history/" + ids[2]).status_code == 200
        assert ids[2] not in _on_disk(path)
        assert api.post("/api/history/" + ids[2] + "/restore").status_code == 200
        assert ids[2] in _on_disk(path)
        assert api.delete("/api/history/" + ids[1]).status_code == 200

    with TestClient(app()) as api:  # a reload is a new process on the same file
        listing = api.get("/api/history").json()
        assert sorted(row["file_name"] for row in listing["items"]) == ["alpha.wav", "gamma.wav"]
        assert api.get("/api/history/" + ids[0]).json()["tags"] == ["clinic-a"]
        assert api.delete("/api/history", params={"confirm": "true"}).json()["deleted"] == 2
        assert _on_disk(path) == {}

    with TestClient(app()) as api:
        assert api.get("/api/history").json()["total"] == 0


# ---------------------------------------------------------------------------
# T132.1-T132.6: the page's structure
# ---------------------------------------------------------------------------


def test_the_history_page_is_built_and_no_longer_a_placeholder() -> None:
    page = _read(FRONTEND / "app" / "history" / "page.tsx")
    assert "<HistoryBrowser />" in page
    assert "PagePlaceholder" not in page


def test_t132_1_and_2_the_server_lists_filters_and_pages() -> None:
    browser = _read(HISTORY / "HistoryBrowser.tsx")
    assert "listHistory(listParams(filters))" in browser
    assert "page.display.total" in browser and "page.display.pages" in browser
    assert "record.display.confidence" in browser and "record.display.result" in browser
    helpers = _read(FRONTEND / "lib" / "history.ts")
    for parameter in ("params.q", "params.task", "params.result", "params.tag", "params.batch_id"):
        assert parameter in helpers, parameter
    assert "params.date_from = dayBound(filters.from, false)" in helpers
    assert "params.date_to = dayBound(filters.to, true)" in helpers
    # The links Analyse writes (T130.6, T131.3) are read.
    assert "get('record')" in browser
    assert "'batch'" in helpers


def test_t132_3_the_drawer_edits_notes_and_tags_only() -> None:
    drawer = _read(HISTORY / "RecordDrawer.tsx")
    assert "<Drawer" in drawer
    assert "<WaveformPlayer" in drawer and "Audio not kept" in drawer
    assert "updateHistoryRecord(record.id, { notes, tags })" in drawer
    assert "record.display.probabilities[name]" in drawer
    assert "source_kind === 'sample'" in _read(FRONTEND / "lib" / "history.ts")


def test_t132_4_to_6_undo_confirmation_compare_and_empty_state() -> None:
    browser = _read(HISTORY / "HistoryBrowser.tsx")
    assert "restoreHistoryRecord(undo.id)" in browser
    assert "<Modal" in browser and "deleteAllHistory()" in browser
    assert "<CompareView" in browser and "compareItemFromRecord(" in browser
    assert "<HistoryEmpty />" in browser and "<NoMatches" in browser
    api = _read(FRONTEND / "lib" / "api.ts")
    assert "'/restore'" in api
    assert "'/api/history?confirm=true'" in api


def test_the_history_components_pass_the_metric_guard_rail() -> None:
    guard = _load_script("16_check_no_hardcoded_metrics.py", "metric_guard_132")
    paths = [
        FRONTEND / "app" / "history" / "page.tsx",
        HISTORY / "HistoryBrowser.tsx",
        HISTORY / "RecordDrawer.tsx",
    ]
    findings: list[Any] = []
    suppressed: list[Any] = []
    for path in paths:
        found, allowed = guard.scan_file(path)
        findings.extend(found)
        suppressed.extend(allowed)
    assert not findings, "\n".join(str(finding) for finding in findings)
    assert not suppressed, str(suppressed)
    for path in [*paths, FRONTEND / "lib" / "history.ts"]:
        body = _read(path)
        for forbidden in ("toFixed(", "toPrecision(", "Math.round(", "fetch("):
            assert forbidden not in body, path.name + " uses " + forbidden
