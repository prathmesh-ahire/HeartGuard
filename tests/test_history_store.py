"""Phase 129's gate: the History store and its API.

T129.7 asks for five things, and each has its own section below: CRUD
round-trips, filters, concurrent writes that lose nothing, a corrupt file that
recovers, and aggregates that match a hand-built fixture.

**The Insights fixture is built by hand on purpose.** Its expected numbers are
worked out in the comments next to each row, not produced by calling the code
under test, so a test here cannot agree with a bug in `insights`.

Every store here lives in `tmp_path`. The root conftest also redirects
`cache.history_db` for the whole session, so no test writes into the operator's
real history.

Only the last test needs the dataset and a saved model; it skips without them,
as CI does. Everything else runs on CI.
"""

from __future__ import annotations

import json
import math
import struct
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import history_store as hs
from src.api.history_store import HistoryStore, HistoryValidationError
from src.api.main import DEV_ORIGINS, create_app
from src.inference.predictor import AudioValidationError, ModelUnavailableError, PredictionResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _row(store: HistoryStore, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "file_name": "a0001.wav",
        "task": "binary",
        "result": "normal",
        "confidence": 0.9,
        "probabilities": {"normal": 0.9, "abnormal": 0.1},
    }
    fields.update(overrides)
    return store.create(**fields)


@pytest.fixture()
def store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path / "history" / "history.json")


# ---------------------------------------------------------------------------
# CRUD round-trips
# ---------------------------------------------------------------------------


def test_the_session_never_writes_into_the_real_history() -> None:
    assert "pvmepcg-manifest-" in str(hs.default_history_path())


def test_create_then_get_returns_the_versioned_record(store: HistoryStore) -> None:
    created = _row(store, notes="  first  ", tags=["Follow-up", "follow-up", " clinic  a "])
    fetched = store.get(created["id"])
    assert fetched == created
    for field in (
        "id",
        "created_at",
        "file_name",
        "task",
        "result",
        "confidence",
        "probabilities",
        "notes",
        "tags",
    ):
        assert field in fetched, field
    assert fetched["schema_version"] == hs.SCHEMA_VERSION
    assert fetched["notes"] == "first"
    assert fetched["tags"] == ["Follow-up", "clinic a"], "tags de-duplicate case-insensitively"
    assert fetched["display"]["confidence"] == "0.900"
    assert fetched["display"]["confidence_percent"] == "90.0%"
    assert fetched["display"]["probabilities_percent"] == {"normal": "90.0%", "abnormal": "10.0%"}


def test_rows_persist_across_store_instances(store: HistoryStore) -> None:
    created = _row(store)
    store.close()
    reopened = HistoryStore(store.path)
    assert reopened.get(created["id"]) == created


def test_a_browser_fake_path_is_reduced_to_the_file_name(store: HistoryStore) -> None:
    assert _row(store, file_name="C:\\fakepath\\clip.wav")["file_name"] == "clip.wav"
    assert _row(store, file_name="/home/u/clip2.wav")["file_name"] == "clip2.wav"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": "binary_and_murmur"}, "unknown task"),
        ({"result": "Present"}, "not a class of binary"),
        ({"probabilities": {"normal": 1.5}}, "between 0 and 1"),
        ({"probabilities": {"Present": 0.5}}, "not a class of binary"),
        ({"confidence": -0.1}, "between 0 and 1"),
        ({"notes": "x" * (hs.MAX_NOTES_CHARS + 1)}, "notes are limited"),
        ({"tags": ["t" + str(i) for i in range(hs.MAX_TAGS + 1)]}, "at most"),
        ({"tags": ["x" * (hs.MAX_TAG_CHARS + 1)]}, "a tag is limited"),
        ({"file_name": "  "}, "file_name is required"),
        ({"result": None}, "needs a predicted class"),
    ],
)
def test_invalid_rows_are_refused(
    store: HistoryStore, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(HistoryValidationError, match=message):
        _row(store, **overrides)
    assert store.list_records()["total"] == 0


def test_label_spaces_are_validated_per_task(store: HistoryStore) -> None:
    """A CirCor class is valid for CirCor and invalid for PASCAL: never merged."""
    _row(store, task="murmur", result="Present", probabilities={"Present": 0.7})
    with pytest.raises(HistoryValidationError):
        _row(store, task="pascal_b", result="Present", probabilities={})


def test_update_changes_only_notes_and_tags(
    store: HistoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _row(store)
    later = datetime.now(UTC) + timedelta(minutes=5)
    monkeypatch.setattr(hs, "_utc_now", lambda: later)
    updated = store.update(created["id"], notes="recheck", tags=["a", "b"])
    assert updated is not None
    assert (updated["notes"], updated["tags"]) == ("recheck", ["a", "b"])
    assert updated["updated_at"] > created["updated_at"]
    for frozen in ("created_at", "result", "confidence", "probabilities", "task", "file_name"):
        assert updated[frozen] == created[frozen], frozen
    assert store.update(created["id"], tags=[])["tags"] == []  # type: ignore[index]
    assert store.update(created["id"])["notes"] == "recheck"  # type: ignore[index]
    assert store.update("no-such-id", notes="x") is None


def test_delete_one_and_delete_all(store: HistoryStore) -> None:
    first, second, third = (_row(store, file_name=f"f{i}.wav") for i in range(3))
    assert store.delete(first["id"]) is True
    assert store.delete(first["id"]) is False
    assert store.get(first["id"]) is None
    assert store.get(second["id"]) is not None
    assert store.delete_all() == 2
    assert store.list_records()["total"] == 0
    assert store.get(third["id"]) is None


def test_an_unscored_prediction_is_not_saved(store: HistoryStore) -> None:
    payload = {"task": "binary", "predicted_class": "", "scorable": False, "probabilities": {}}
    assert store.record_prediction(payload, file_name="silence.wav", source_kind="upload") is None
    assert store.list_records()["total"] == 0


# ---------------------------------------------------------------------------
# search, filter, sort, pagination
# ---------------------------------------------------------------------------


@pytest.fixture()
def filled(store: HistoryStore, monkeypatch: pytest.MonkeyPatch) -> HistoryStore:
    base = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
    rows = [
        # (day offset, file, task, result, confidence, low, notes, tags)
        (0, "alpha.wav", "binary", "normal", 0.91, False, "", ["clinic"]),
        (1, "beta.wav", "binary", "abnormal", 0.55, True, "listen again", []),
        (2, "gamma.wav", "pascal_b", "murmur", None, False, "", ["Clinic", "kid"]),
        (3, "delta.wav", "murmur", "Present", 0.62, False, "", []),
        (4, "Epsilon.wav", "binary", "normal", 0.70, False, "", []),
    ]
    for offset, name, task, result, confidence, low, notes, tags in rows:
        moment = base + timedelta(days=offset)
        monkeypatch.setattr(hs, "_utc_now", lambda moment=moment: moment)
        store.create(
            file_name=name,
            task=task,
            result=result,
            confidence=confidence,
            probabilities={},
            low_confidence=low,
            notes=notes,
            tags=tags,
        )
    return store


def _names(listing: dict[str, Any]) -> list[str]:
    return [item["file_name"] for item in listing["items"]]


def test_default_order_is_newest_first(filled: HistoryStore) -> None:
    listing = filled.list_records()
    assert _names(listing) == ["Epsilon.wav", "delta.wav", "gamma.wav", "beta.wav", "alpha.wav"]
    assert (listing["total"], listing["pages"], listing["display"]["total"]) == (5, 1, "5")


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"task": "binary"}, {"alpha.wav", "beta.wav", "Epsilon.wav"}),
        ({"result": "normal"}, {"alpha.wav", "Epsilon.wav"}),
        ({"tag": "CLINIC"}, {"alpha.wav", "gamma.wav"}),
        ({"low_confidence": True}, {"beta.wav"}),
        ({"low_confidence": False}, {"alpha.wav", "gamma.wav", "delta.wav", "Epsilon.wav"}),
        ({"query": "again"}, {"beta.wav"}),
        ({"query": "EPS"}, {"Epsilon.wav"}),
        ({"query": "circor"}, {"delta.wav"}),  # matches the task title
        ({"date_from": "2026-09-12"}, {"gamma.wav", "delta.wav", "Epsilon.wav"}),
        ({"date_to": "2026-09-11"}, {"alpha.wav", "beta.wav"}),
        ({"date_from": "2026-09-11", "date_to": "2026-09-11"}, {"beta.wav"}),
        ({"task": "binary", "result": "normal", "tag": "clinic"}, {"alpha.wav"}),
        ({"query": "nothing-matches"}, set()),
    ],
)
def test_filters_combine_with_and(
    filled: HistoryStore, filters: dict[str, Any], expected: set[str]
) -> None:
    listing = filled.list_records(**filters)
    assert set(_names(listing)) == expected
    assert listing["total"] == len(expected)


def test_sort_by_confidence_puts_missing_values_last_both_ways(filled: HistoryStore) -> None:
    ascending = _names(filled.list_records(sort="confidence", order="asc"))
    descending = _names(filled.list_records(sort="confidence", order="desc"))
    assert ascending == ["beta.wav", "delta.wav", "Epsilon.wav", "alpha.wav", "gamma.wav"]
    assert descending == ["alpha.wav", "Epsilon.wav", "delta.wav", "beta.wav", "gamma.wav"]


def test_sort_by_file_name_ignores_case(filled: HistoryStore) -> None:
    names = _names(filled.list_records(sort="file_name", order="asc"))
    assert names == ["alpha.wav", "beta.wav", "delta.wav", "Epsilon.wav", "gamma.wav"]


def test_pagination_covers_every_row_once(filled: HistoryStore) -> None:
    pages = [filled.list_records(page=n, page_size=2) for n in (1, 2, 3)]
    assert [page["pages"] for page in pages] == [3, 3, 3]
    seen = [name for page in pages for name in _names(page)]
    assert sorted(seen) == sorted(_names(filled.list_records()))
    assert _names(filled.list_records(page=4, page_size=2)) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"sort": "probabilities"},
        {"order": "sideways"},
        {"page": 0},
        {"page_size": hs.MAX_PAGE_SIZE + 1},
        {"date_from": "14/09/2026"},
        {"date_from": "2026-09-12", "date_to": "2026-09-11"},
        {"task": "everything"},
    ],
)
def test_bad_list_parameters_are_refused(filled: HistoryStore, bad: dict[str, Any]) -> None:
    with pytest.raises(HistoryValidationError):
        filled.list_records(**bad)


# ---------------------------------------------------------------------------
# concurrent writes lose nothing; writes are atomic
# ---------------------------------------------------------------------------


def test_concurrent_creates_lose_nothing(store: HistoryStore) -> None:
    def work(worker: int) -> list[str]:
        return [
            _row(store, file_name=f"w{worker}_{i}.wav", tags=[f"w{worker}"])["id"]
            for i in range(25)
        ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = [row_id for batch in pool.map(work, range(8)) for row_id in batch]

    assert len(ids) == len(set(ids)) == 200
    assert store.list_records(page_size=hs.MAX_PAGE_SIZE)["total"] == 200
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))[hs.TABLE_NAME]
    assert {row["id"] for row in on_disk.values()} == set(ids)


def test_concurrent_mixed_writes_lose_nothing(store: HistoryStore) -> None:
    seeds = [_row(store, file_name=f"seed{i}.wav") for i in range(20)]
    barrier = threading.Barrier(3)

    def creator() -> None:
        barrier.wait()
        for i in range(30):
            _row(store, file_name=f"new{i}.wav")

    def tagger() -> None:
        barrier.wait()
        for row in seeds:
            store.update(row["id"], tags=["tagged"])

    def deleter() -> None:
        barrier.wait()
        for row in seeds[:10]:
            store.delete(row["id"])

    threads = [threading.Thread(target=fn) for fn in (creator, tagger, deleter)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.list_records(page_size=100)["total"] == 20 + 30 - 10
    survivors = [store.get(row["id"]) for row in seeds[10:]]
    assert all(row is not None and row["tags"] == ["tagged"] for row in survivors)


def test_concurrent_api_writes_lose_nothing(tmp_path: Path) -> None:
    app = create_app(
        static_root=tmp_path / "no-site", preload=False, history_path=tmp_path / "h.json"
    )

    def work(worker: int) -> int:
        with TestClient(app) as client:
            for i in range(10):
                response = client.post(
                    "/api/history",
                    json={"file_name": f"{worker}-{i}.wav", "task": "binary", "result": "normal"},
                )
                assert response.status_code == 201, response.text
        return worker

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(work, range(6)))
    with TestClient(app) as client:
        assert client.get("/api/history").json()["total"] == 60


def test_a_failed_write_leaves_the_previous_file_intact(
    store: HistoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    kept = _row(store)
    before = store.path.read_bytes()

    def broken_replace(source: Path, target: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(hs, "_replace", broken_replace)
    with pytest.raises(OSError):
        _row(store, file_name="lost.wav")
    monkeypatch.undo()

    assert store.path.read_bytes() == before
    assert [p.name for p in store.path.parent.iterdir()] == [store.path.name], "no temp left"
    assert _names(store.list_records()) == [kept["file_name"]]


def test_stale_temporaries_from_a_killed_process_are_cleared(store: HistoryStore) -> None:
    _row(store)
    stale = store.path.with_name(store.path.name + ".999.dead.tmp")
    stale.write_text("{half", encoding="utf-8")
    store.close()
    assert HistoryStore(store.path).list_records()["total"] == 1
    assert not stale.exists()


# ---------------------------------------------------------------------------
# corrupt or missing files recover
# ---------------------------------------------------------------------------


def test_a_missing_file_is_an_empty_history_and_is_not_created_by_a_read(
    tmp_path: Path,
) -> None:
    store = HistoryStore(tmp_path / "deep" / "er" / "history.json")
    listing = store.list_records()
    assert (listing["total"], listing["notice"]) == (0, None)
    assert not store.path.exists()
    _row(store)
    assert store.path.is_file()


@pytest.mark.parametrize(
    "damage",
    [
        b'{"records": {"1": {"id": "x", "task"',
        b"",
        b"\xff\xfe\x00garbage",
        b"[1, 2, 3]",
        b'{"records": {"1": "not a row"}}',
    ],
    ids=["truncated", "empty", "binary", "wrong-shape", "bad-row"],
)
def test_a_corrupt_file_is_set_aside_and_a_new_history_starts(
    store: HistoryStore, damage: bytes
) -> None:
    store.path.parent.mkdir(parents=True)
    store.path.write_bytes(damage)

    listing = store.list_records()
    assert listing["total"] == 0
    assert listing["notice"] and "damaged copy was kept" in listing["notice"]
    kept = list(store.path.parent.glob("history.json.corrupt-*"))
    assert len(kept) == 1 and kept[0].read_bytes() == damage, "the damaged bytes are preserved"

    created = _row(store)
    assert store.get(created["id"]) == created
    assert json.loads(store.path.read_text(encoding="utf-8"))  # valid again


def test_the_api_reports_a_recovery_and_a_disk_failure_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")
    app = create_app(static_root=tmp_path / "no-site", preload=False, history_path=path)
    with TestClient(app) as client:
        listing = client.get("/api/history")
        assert listing.status_code == 200
        assert listing.json()["notice"]

        def unreadable(self: Any) -> None:
            raise PermissionError("C:/secret/path/history.json is locked")

        monkeypatch.setattr(hs.AtomicJSONStorage, "read", unreadable)
        failed = client.get("/api/history")
        assert failed.status_code == 503
        assert "secret" not in failed.text, "no file path reaches a page"


# ---------------------------------------------------------------------------
# Insights against a hand-built fixture
# ---------------------------------------------------------------------------

#: "Now" for the fixture: 2026-09-14 10:00 UTC, which is 15:30 at UTC+05:30.
NOW = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)


@pytest.fixture()
def insight_store(store: HistoryStore, monkeypatch: pytest.MonkeyPatch) -> HistoryStore:
    rows = [
        # created_at (UTC)          task       result      conf  low
        (NOW - timedelta(hours=1), "binary", "normal", 0.95, False),  # 14th, bin 9
        (NOW - timedelta(hours=2), "binary", "abnormal", 0.55, True),  # 14th, bin 5
        (NOW - timedelta(days=1), "binary", "normal", 0.72, False),  # 13th, bin 7
        (NOW - timedelta(days=3), "pascal_b", "murmur", 0.40, True),  # 11th, bin 4
        (NOW - timedelta(days=40), "murmur", "Present", 0.30, False),  # outside, bin 3
        # 13th 20:00 UTC is 14th 01:30 at UTC+05:30 -- the timezone case.
        (datetime(2026, 9, 13, 20, 0, tzinfo=UTC), "outcome", "Abnormal", None, False),
    ]
    for moment, task, result, confidence, low in rows:
        monkeypatch.setattr(hs, "_utc_now", lambda moment=moment: moment)
        store.create(
            file_name="x.wav",
            task=task,
            result=result,
            confidence=confidence,
            probabilities={},
            low_confidence=low,
        )
    return store


def test_insights_counts_match_the_fixture(insight_store: HistoryStore) -> None:
    got = insight_store.insights(days=7, now=NOW)
    assert (got["total"], got["total_display"]) == (6, "6")
    # 2 of 6 low confidence = 33.3%
    assert got["low_confidence"]["count"] == 2
    assert got["low_confidence"]["percent_display"] == "33.3%"
    # binary 3/6 = 50.0%, pascal_b 1/6 = 16.7%, murmur 1/6, outcome 1/6; declared task order
    assert [(t["task"], t["count"], t["percent_display"]) for t in got["by_task"]] == [
        ("binary", 3, "50.0%"),
        ("pascal_b", 1, "16.7%"),
        ("murmur", 1, "16.7%"),
        ("outcome", 1, "16.7%"),
    ]


def test_insights_result_split_is_per_task_and_never_merged(insight_store: HistoryStore) -> None:
    split = {
        entry["task"]: entry for entry in insight_store.insights(days=7, now=NOW)["result_split"]
    }
    assert set(split) == {"binary", "pascal_b", "murmur", "outcome"}
    binary = {c["class"]: (c["count"], c["percent_display"]) for c in split["binary"]["classes"]}
    # 2 of 3 normal = 66.7%, 1 of 3 abnormal = 33.3%
    assert binary == {"normal": (2, "66.7%"), "abnormal": (1, "33.3%")}
    murmur = {c["class"]: c["count"] for c in split["murmur"]["classes"]}
    assert murmur == {"Absent": 0, "Present": 1, "Unknown": 0}
    assert [c["class"] for c in split["pascal_b"]["classes"]] == ["normal", "murmur", "extrastole"]


def test_insights_trend_by_day_utc(insight_store: HistoryStore) -> None:
    trend = insight_store.insights(days=7, now=NOW)["trend"]
    assert (trend["first_date"], trend["last_date"]) == ("2026-09-08", "2026-09-14")
    counts = {p["date"]: p["count"] for p in trend["points"]}
    # UTC days: 14th two rows; 13th the 0.72 row and the 20:00 outcome row; 11th one
    assert counts == {
        "2026-09-08": 0,
        "2026-09-09": 0,
        "2026-09-10": 0,
        "2026-09-11": 1,
        "2026-09-12": 0,
        "2026-09-13": 2,
        "2026-09-14": 2,
    }
    assert trend["in_window"] == 5  # the 40-day-old row is outside


def test_insights_trend_uses_the_viewers_day(insight_store: HistoryStore) -> None:
    trend = insight_store.insights(days=7, tz_offset_minutes=330, now=NOW)["trend"]
    counts = {p["date"]: p["count"] for p in trend["points"]}
    # At UTC+05:30 the 13th 20:00 UTC row is on the 14th; the 13th keeps only 0.72
    # (13th 10:00 UTC = 15:30 local), the 11th stays the 11th.
    assert (counts["2026-09-14"], counts["2026-09-13"], counts["2026-09-11"]) == (3, 1, 1)


def test_insights_confidence_distribution(insight_store: HistoryStore) -> None:
    dist = insight_store.insights(days=7, now=NOW)["confidence_distribution"]
    assert dist["n"] == 5  # the outcome row has no confidence
    assert [b["count"] for b in dist["bins"]] == [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    assert dist["bins"][0]["label"] == "0.0-0.1"
    assert dist["bins"][9]["label"] == "0.9-1.0"
    assert dist["bins"][9]["percent_display"] == "20.0%"
    assert math.isclose(sum(b["percent"] for b in dist["bins"]), 100.0)


def test_insights_for_one_task_and_for_an_empty_history(
    insight_store: HistoryStore, tmp_path: Path
) -> None:
    only = insight_store.insights(task="binary", days=7, now=NOW)
    assert only["total"] == 3 and [t["task"] for t in only["by_task"]] == ["binary"]
    empty = HistoryStore(tmp_path / "empty.json").insights(days=3, now=NOW)
    assert empty["total"] == 0
    assert empty["low_confidence"]["percent_display"] == "n/a"
    assert [p["count"] for p in empty["trend"]["points"]] == [0, 0, 0]
    assert empty["by_task"] == [] and empty["result_split"] == []


def test_confidence_bin_edges_do_not_drift_on_float_error(store: HistoryStore) -> None:
    for value in (0.3, 0.7, 1.0, 0.0):
        _row(store, confidence=value)
    bins = [b["count"] for b in store.insights()["confidence_distribution"]["bins"]]
    assert bins == [1, 0, 0, 1, 0, 0, 0, 1, 0, 1]


# ---------------------------------------------------------------------------
# the API
# ---------------------------------------------------------------------------


@pytest.fixture()
def api(tmp_path: Path) -> Any:
    app = create_app(
        static_root=tmp_path / "no-site", preload=False, history_path=tmp_path / "h.json"
    )
    with TestClient(app) as client:
        yield client


def test_api_crud_round_trip(api: Any) -> None:
    created = api.post(
        "/api/history",
        json={
            "file_name": "a.wav",
            "task": "pascal_a",
            "result": "murmur",
            "confidence": 0.61,
            "probabilities": {"normal": 0.2, "murmur": 0.61, "extrahls": 0.1, "artifact": 0.09},
            "tags": ["x"],
        },
    )
    assert created.status_code == 201, created.text
    row = created.json()
    assert row["source_kind"] == "manual"
    assert api.get("/api/history/" + row["id"]).json() == row

    patched = api.patch("/api/history/" + row["id"], json={"notes": "n", "tags": ["y"]}).json()
    assert (patched["notes"], patched["tags"]) == ("n", ["y"])
    assert api.get("/api/history", params={"tag": "y"}).json()["total"] == 1

    assert api.delete("/api/history/" + row["id"]).json() == {"deleted": 1, "id": row["id"]}
    for method in ("get", "delete"):
        assert getattr(api, method)("/api/history/" + row["id"]).status_code == 404
    assert api.patch("/api/history/" + row["id"], json={"notes": "z"}).status_code == 404


def test_api_errors_are_400_and_422_not_500(api: Any) -> None:
    bad_task = api.post("/api/history", json={"file_name": "a.wav", "task": "all", "result": "x"})
    assert bad_task.status_code == 400 and "unknown task" in bad_task.json()["detail"]
    assert api.post("/api/history", json={"task": "binary"}).status_code == 422
    assert api.get("/api/history", params={"sort": "nope"}).status_code == 400
    assert api.get("/api/history", params={"page_size": 1000}).status_code == 422
    assert api.get("/api/history/insights", params={"days": 0}).status_code == 422


def test_api_delete_all_needs_confirmation(api: Any) -> None:
    for i in range(3):
        api.post(
            "/api/history", json={"file_name": f"{i}.wav", "task": "binary", "result": "normal"}
        )
    assert api.delete("/api/history").status_code == 400
    assert api.get("/api/history").json()["total"] == 3
    assert api.delete("/api/history", params={"confirm": "true"}).json()["deleted"] == 3
    assert api.get("/api/history").json()["total"] == 0


def test_api_list_parameters_reach_the_store(api: Any) -> None:
    for i, result in enumerate(["normal", "abnormal", "normal"]):
        api.post(
            "/api/history",
            json={
                "file_name": f"rec{i}.wav",
                "task": "binary",
                "result": result,
                "confidence": 0.5 + i / 10,
            },
        )
    body = api.get(
        "/api/history",
        params={"result": "normal", "sort": "confidence", "order": "asc", "page_size": 1},
    ).json()
    assert (body["total"], body["pages"]) == (2, 2)
    assert [item["file_name"] for item in body["items"]] == ["rec0.wav"]


def test_api_insights_is_the_store_aggregate(api: Any) -> None:
    api.post(
        "/api/history",
        json={"file_name": "a.wav", "task": "binary", "result": "abnormal", "confidence": 0.8},
    )
    served = api.get("/api/history/insights", params={"days": 5}).json()
    direct = api.app.state.history.insights(days=5)
    assert served == json.loads(json.dumps(direct))
    assert "sensitivity" not in json.dumps(served).lower(), "counts only, never a model metric"


def test_api_cors_preflight_allows_patch_and_delete_from_the_dev_origin(api: Any) -> None:
    for method in ("PATCH", "DELETE"):
        response = api.options(
            "/api/history/abc",
            headers={"Origin": DEV_ORIGINS[0], "Access-Control-Request-Method": method},
        )
        assert response.status_code == 200
        assert method in response.headers["access-control-allow-methods"]


def test_the_history_page_is_not_shadowed_by_the_api(tmp_path: Path) -> None:
    site = tmp_path / "out"
    (site / "history").mkdir(parents=True)
    (site / "index.html").write_text("<h1>home</h1>", encoding="utf-8")
    (site / "history" / "index.html").write_text("<h1>History page</h1>", encoding="utf-8")
    app = create_app(static_root=site, preload=False, history_path=tmp_path / "h.json")
    with TestClient(app) as client:
        assert "History page" in client.get("/history/").text
        assert client.get("/api/history").headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# T129.5 -- /predict saves successes and nothing else
# ---------------------------------------------------------------------------


def _wav_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(2000)
        handle.writeframes(b"".join(struct.pack("<h", 0) for _ in range(4000)))
    return path.read_bytes()


def _fake_result(scorable: bool = True) -> PredictionResult:
    return PredictionResult(
        task="binary",
        predicted_class="abnormal" if scorable else "",
        predicted_index=1 if scorable else -1,
        probabilities={"normal": 0.3, "abnormal": 0.7}
        if scorable
        else {"normal": None, "abnormal": None},
        confidence=0.7 if scorable else None,
        margin=0.4 if scorable else None,
        low_confidence=False,
        low_confidence_margin=0.1,
        operating_threshold=0.5,
        operating_point_note="fixture",
        timings_seconds={"total": 0.01},
        n_features=138,
        n_missing_features=0 if scorable else 138,
        feature_flags=(),
        quality={"duration_seconds": 2.0},
        model={"task": "binary"},
        source="clip.wav",
        scorable=scorable,
        not_scorable_reason=None if scorable else "silence",
    )


def _post_clip(api: Any, tmp_path: Path, name: str = "C:\\fakepath\\clip.wav") -> Any:
    return api.post(
        "/predict",
        files={"file": (name, _wav_bytes(tmp_path), "audio/wav")},
        data={"task": "binary"},
    )


def test_a_successful_prediction_is_saved(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.main.predict_recording", lambda *a, **k: (_fake_result(), None))
    response = _post_clip(api, tmp_path)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["history_saved"] is True and body["history_id"]

    row = api.get("/api/history/" + body["history_id"]).json()
    assert (row["file_name"], row["task"], row["result"]) == ("clip.wav", "binary", "abnormal")
    assert row["confidence"] == body["confidence"]
    assert row["probabilities"] == body["probabilities"]
    assert row["display"]["confidence"] == body["display"]["confidence"]
    assert row["source_kind"] == "upload"


@pytest.mark.parametrize(
    ("error", "status"),
    [(AudioValidationError("not a WAV"), 400), (ModelUnavailableError("no model"), 503)],
)
def test_a_failed_prediction_is_never_saved(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr("src.api.main.predict_recording", fail)
    assert _post_clip(api, tmp_path).status_code == status
    assert api.post("/predict", files={"file": ("e.wav", b"", "audio/wav")}).status_code == 400
    assert api.get("/api/history").json()["total"] == 0


def test_an_unscorable_recording_is_answered_but_not_saved(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.api.main.predict_recording", lambda *a, **k: (_fake_result(scorable=False), None)
    )
    body = _post_clip(api, tmp_path).json()
    assert body["scorable"] is False
    assert (body["history_saved"], body["history_id"]) == (False, None)
    assert body["history_note"]
    assert api.get("/api/history").json()["total"] == 0


def test_a_store_failure_does_not_cost_the_prediction(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.api.main.predict_recording", lambda *a, **k: (_fake_result(), None))

    def disk_full(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(api.app.state.history, "record_prediction", disk_full)
    response = _post_clip(api, tmp_path)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "abnormal"
    assert (body["history_saved"], body["history_id"]) == (False, None)
    assert "not being saved" in body["history_note"]


BUNDLE = PROJECT_ROOT / "models_saved" / "binary" / "final" / "model.joblib"
PHYSIONET = PROJECT_ROOT / "dataset" / "archive (3)" / "training-a"


def test_a_real_recording_is_saved_exactly_as_it_was_scored(api: Any) -> None:
    """Against the real model and a real PhysioNet file, not a fixture."""
    if not BUNDLE.is_file():
        pytest.skip("no saved binary model in this checkout (gitignored)")
    recordings = sorted(PHYSIONET.glob("*.wav")) if PHYSIONET.is_dir() else []
    if not recordings:
        pytest.skip("dataset/ is not present in this checkout")
    recording = recordings[0]

    body = api.post(
        "/predict",
        files={"file": (recording.name, recording.read_bytes(), "audio/wav")},
        data={"task": "binary"},
    ).json()
    assert body["scorable"] and body["history_saved"], body.get("history_note")
    row = api.get("/api/history/" + body["history_id"]).json()
    assert row["file_name"] == recording.name
    assert row["result"] == body["predicted_class"]
    assert row["probabilities"] == body["probabilities"]
    assert row["confidence"] == body["confidence"]
    assert row["low_confidence"] == body["low_confidence"]
    assert row["display"]["probabilities_percent"] == body["display"]["probabilities_percent"]
