"""The History store (Phase 129).

Every successful screening result is kept so the History and Insights pages can
show it again. This module is the only code that reads or writes that record.

## What is stored, and what is not

One row per result: id, created_at, the file's display name, the task, the
predicted class, confidence, per-class probabilities, notes and tags (T129.2),
and the batch it was scored in, if any (`batch_id`, T131.3).
**The audio is never stored** -- `/predict` still deletes every upload -- and a
file name is reduced to its last path component, so no client-side folder
structure is kept either.

Display strings are NOT stored. `present` formats them on the way out through
`tables.format_value`, the project's single rounding authority, so a row saved
today renders under whatever rounding rule is current rather than freezing one
into the file.

## Why the writes look paranoid (T129.3)

TinyDB's stock `JSONStorage` rewrites the file in place: it seeks to 0, writes
and truncates. A crash or a full disk mid-write leaves half a JSON document and
the whole history is unreadable. `AtomicJSONStorage` writes a temporary file in
the same directory, fsyncs it and `os.replace`s it over the database, which is
atomic on NTFS and POSIX: a reader sees the old file or the new one, never a
mix.

TinyDB objects are not thread-safe, and FastAPI runs sync endpoints on a thread
pool, so one `RLock` guards **every** operation, reads included -- a read that
interleaves with a write's replace could otherwise see a table mid-update.

A file that will not parse is moved aside to `<name>.corrupt-<UTC stamp>` and the
store starts empty. The damaged file is kept, never overwritten: whatever was in
it can still be recovered by hand, and `notice` tells the page it happened
rather than the history silently looking empty.

**Single process only.** The lock is in-process. The API runs as one uvicorn
process (`python -m src.api.main`); running several workers against one file
would need a file lock this module does not take.

## Insights are counts of the user's own results, not model metrics

`insights` counts what the operator has screened. It is not a precomputed
metric, so it does not break the codegen boundary: no row of it exists at build
time. Result splits are **per task** -- the five label spaces are never merged
(research rule 4), so "abnormal" from the binary task and "Present" from CirCor
murmur are never added into one "positive" count.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from tinydb import Query, TinyDB
from tinydb.queries import QueryInstance
from tinydb.storages import Storage

from src.inference.predictor import TASKS
from src.reporting.tables import NA_TEXT, format_value
from src.utils.logging_setup import get_logger

__all__ = [
    "BATCH_STATUSES",
    "CONFIDENCE_BINS",
    "MAX_BATCH_ROWS",
    "MAX_NOTES_CHARS",
    "MAX_PAGE_SIZE",
    "MAX_TAGS",
    "MAX_TAG_CHARS",
    "SCHEMA_VERSION",
    "SORT_FIELDS",
    "AtomicJSONStorage",
    "HistoryStore",
    "HistoryValidationError",
    "clean_batch_id",
    "default_history_path",
]

log = get_logger("api.history_store")

#: Bumped when a stored field changes meaning. `_upgrade` brings older rows up.
SCHEMA_VERSION = 1

TABLE_NAME = "records"
MAX_FILE_NAME_CHARS = 255
MAX_NOTES_CHARS = 2000
MAX_TAGS = 20
MAX_TAG_CHARS = 32
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20
SORT_FIELDS = ("created_at", "file_name", "task", "result", "confidence")
SOURCE_KINDS = ("upload", "sample", "manual")
#: Ten equal-width confidence bins over [0, 1].
CONFIDENCE_BINS = 10
MAX_TREND_DAYS = 366
#: T131.3: a batch is named by 32 lowercase hex characters (a uuid4 without
#: dashes), minted by the page when a batch starts.
_BATCH_ID = re.compile(r"^[0-9a-f]{32}$")
#: T131.6: what one row of an exported batch table can be. Only `scored` rows
#: are in History; the other three never produced a result to save.
BATCH_STATUSES = ("scored", "not_scored", "failed", "cancelled")
#: More rows than any batch the page allows, and a bound on one request body.
MAX_BATCH_ROWS = 500
#: A cell opening with one of these is a formula to a spreadsheet (CSV injection).
_FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")
#: UTC-14 .. UTC+14, the real range of civil time zones.
MAX_TZ_OFFSET_MINUTES = 14 * 60
#: T132.4: how many one-by-one deletions can still be undone. Held in memory
#: only, so a restart (or a delete-all) ends every undo.
UNDO_DEPTH = 50

#: How often `os.replace` is retried. On Windows an antivirus scanner or the
#: search indexer can hold a just-written file open for a few milliseconds, and
#: the replace then fails with PermissionError although nothing is wrong.
_REPLACE_ATTEMPTS = 20
_REPLACE_PAUSE_SECONDS = 0.05


class HistoryValidationError(ValueError):
    """A request the store refuses. The message is safe to show on a page."""


def default_history_path() -> Path:
    """`cache.history_db` from `configs/paths.yaml`, environment overrides applied."""
    from src.utils.config import load_config

    configured = load_config("paths").get("cache.history_db")
    if not configured:
        raise HistoryValidationError("cache.history_db is not configured in paths.yaml")
    return Path(str(configured))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    """One fixed ISO form, so string order is time order."""
    return moment.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


# ---------------------------------------------------------------------------
# T129.3 -- atomic storage with recovery
# ---------------------------------------------------------------------------


class AtomicJSONStorage(Storage):
    """TinyDB storage that never leaves a half-written file behind."""

    def __init__(self, path: str | Path, on_recover: Callable[[str], None] | None = None) -> None:
        self.path = Path(path)
        self._on_recover = on_recover
        self._clear_stale_temporaries()

    def read(self) -> dict[str, dict[str, Any]] | None:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return None
        if not raw.strip():
            # `os.replace` cannot produce an empty file, so one is damage.
            self._quarantine("the file is empty")
            return None
        try:
            # Decoded inside the guard: bytes that are not UTF-8 are damage too.
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            self._quarantine("the file is not valid JSON (" + type(error).__name__ + ")")
            return None
        if not isinstance(data, dict) or not all(
            isinstance(table, dict) and all(isinstance(row, dict) for row in table.values())
            for table in data.values()
        ):
            self._quarantine("the file is JSON but not a TinyDB document")
            return None
        return data

    def write(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            self.path.name + "." + str(os.getpid()) + "." + uuid.uuid4().hex + ".tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as sink:
                json.dump(data, sink, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                sink.flush()
                os.fsync(sink.fileno())
            _replace(temporary, self.path)
        except BaseException:
            with suppress(OSError):
                temporary.unlink()
            raise

    def close(self) -> None:  # nothing is held open between operations
        return None

    def _quarantine(self, reason: str) -> None:
        stamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
        target = self.path.with_name(self.path.name + ".corrupt-" + stamp)
        try:
            _replace(self.path, target)
        except OSError:
            # Could not move it: copy it, so the next write cannot destroy it.
            target.write_bytes(self.path.read_bytes())
        log.warning("history database was unreadable (%s); kept as %s", reason, target.name)
        if self._on_recover is not None:
            self._on_recover(reason)

    def _clear_stale_temporaries(self) -> None:
        """Temporaries from a process killed mid-write. The database is intact."""
        if not self.path.parent.is_dir():
            return
        for leftover in self.path.parent.glob(self.path.name + ".*.tmp"):
            with suppress(OSError):
                leftover.unlink()


def _replace(source: Path, target: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_PAUSE_SECONDS)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _clean_file_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoryValidationError("file_name is required")
    # A browser may send C:\fakepath\x.wav; keep the last component only.
    name = PureWindowsPath(value.strip()).name or value.strip()
    name = Path(name).name or name
    return name[:MAX_FILE_NAME_CHARS]


def _probability(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoryValidationError(field + " must be a number between 0 and 1, or null")
    number = float(value)
    if not math.isfinite(number):
        return None
    if not 0.0 <= number <= 1.0:
        raise HistoryValidationError(field + " must be between 0 and 1")
    return number


def _clean_notes(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise HistoryValidationError("notes must be text")
    if len(value) > MAX_NOTES_CHARS:
        raise HistoryValidationError("notes are limited to " + str(MAX_NOTES_CHARS) + " characters")
    return value.strip()


def _clean_tags(value: Any) -> list[str]:
    """Trimmed, de-duplicated case-insensitively, first spelling kept."""
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise HistoryValidationError("tags must be a list of text values")
    tags: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise HistoryValidationError("tags must be a list of text values")
        tag = " ".join(raw.split())
        if not tag:
            continue
        if len(tag) > MAX_TAG_CHARS:
            raise HistoryValidationError(
                "a tag is limited to " + str(MAX_TAG_CHARS) + " characters"
            )
        if tag.casefold() in seen:
            continue
        seen.add(tag.casefold())
        tags.append(tag)
    if len(tags) > MAX_TAGS:
        raise HistoryValidationError("at most " + str(MAX_TAGS) + " tags per result")
    return tags


def _clean_task(value: Any) -> str:
    if value not in TASKS:
        raise HistoryValidationError(
            "unknown task " + repr(value) + ". Declared tasks: " + ", ".join(TASKS)
        )
    return str(value)


def clean_batch_id(value: Any) -> str | None:
    """T131.3: None or blank for no batch, else 32 lowercase hex characters."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str) or not _BATCH_ID.match(value.strip()):
        raise HistoryValidationError("batch_id must be 32 lowercase hexadecimal characters")
    return value.strip()


def _csv_text(value: str) -> str:
    """Free text a spreadsheet would run as a formula is written as text."""
    return "'" + value if value.startswith(_FORMULA_LEADS) else value


def _day_bound(value: str | None, *, end: bool) -> str | None:
    """`YYYY-MM-DD` (whole UTC day, inclusive) or a full ISO time, as a stored-form string."""
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        if len(text) == 10:
            day = date.fromisoformat(text)
            moment = datetime(day.year, day.month, day.day, tzinfo=UTC)
            if end:
                moment += timedelta(days=1) - timedelta(microseconds=1)
        else:
            moment = _parse_iso(text)
    except ValueError as error:
        raise HistoryValidationError(
            "dates must be YYYY-MM-DD or an ISO 8601 time, got " + repr(text)
        ) from error
    return _iso(moment)


def _percent_text(fraction: float | None) -> str:
    if fraction is None:
        return format_value(None, "percent")
    return format_value(fraction * 100.0, "percent") + "%"


# ---------------------------------------------------------------------------
# T129.2 -- the store
# ---------------------------------------------------------------------------


class HistoryStore:
    """A TinyDB-backed history. Every public method takes the one lock."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._configured_path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._db: TinyDB | None = None
        self._recoveries: list[str] = []
        #: T132.4: rows deleted one by one, newest last, so an undo puts back
        #: the same row -- same id, same date -- rather than a copy.
        self._deleted: dict[str, dict[str, Any]] = {}

    # -- plumbing ---------------------------------------------------------

    @property
    def path(self) -> Path:
        if self._configured_path is None:
            self._configured_path = default_history_path()
        return self._configured_path

    def _table(self) -> Any:
        if self._db is None:
            self._db = TinyDB(
                self.path, storage=AtomicJSONStorage, on_recover=self._recoveries.append
            )
        return self._db.table(TABLE_NAME, cache_size=0)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @property
    def notice(self) -> str | None:
        """Set once a damaged database file has been moved aside in this process."""
        if not self._recoveries:
            return None
        return (
            "The saved history could not be read ("
            + self._recoveries[-1]
            + "), so it was set aside and a new, empty history was started. "
            "The damaged copy was kept next to the database and was not deleted."
        )

    # -- writes -----------------------------------------------------------

    def create(
        self,
        *,
        file_name: str,
        task: str,
        result: str | None,
        confidence: float | None,
        probabilities: Mapping[str, float | None] | None,
        low_confidence: bool = False,
        scorable: bool = True,
        source_kind: str = "manual",
        notes: str | None = None,
        tags: Iterable[str] | None = None,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate and insert one row. Returns the presented row."""
        clean_task = _clean_task(task)
        clean_batch = clean_batch_id(batch_id)
        classes = TASKS[clean_task].classes
        if result is not None and result not in classes:
            raise HistoryValidationError(
                "result "
                + repr(result)
                + " is not a class of "
                + clean_task
                + " ("
                + ", ".join(classes)
                + ")"
            )
        if scorable and result is None:
            raise HistoryValidationError("a scorable result needs a predicted class")
        clean_probabilities: dict[str, float | None] = {}
        for name, value in (probabilities or {}).items():
            if name not in classes:
                raise HistoryValidationError(
                    "probability for " + repr(name) + ", which is not a class of " + clean_task
                )
            clean_probabilities[name] = _probability(value, "probabilities." + name)
        if source_kind not in SOURCE_KINDS:
            raise HistoryValidationError("source_kind must be one of " + ", ".join(SOURCE_KINDS))

        now = _iso(_utc_now())
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "id": uuid.uuid4().hex,
            "created_at": now,
            "updated_at": now,
            "file_name": _clean_file_name(file_name),
            "task": clean_task,
            "source_kind": source_kind,
            "scorable": bool(scorable),
            "result": result,
            "confidence": _probability(confidence, "confidence"),
            "low_confidence": bool(low_confidence),
            "probabilities": clean_probabilities,
            "notes": _clean_notes(notes),
            "tags": _clean_tags(tags),
            "batch_id": clean_batch,
        }
        with self._lock:
            self._table().insert(row)
        return self.present(row)

    def record_prediction(
        self,
        payload: Mapping[str, Any],
        *,
        file_name: str,
        source_kind: str,
        batch_id: str | None = None,
    ) -> dict[str, Any] | None:
        """T129.5: save one `/predict` response. A recording that was not scored is not saved.

        The API calls this only after a 200; an error response never reaches it.
        A `scorable: false` response is a 200 that carries no result at all, so
        it is treated as the failed screen it is and returns None.
        """
        if not payload.get("scorable", True) or not payload.get("predicted_class"):
            return None
        return self.create(
            file_name=file_name,
            task=str(payload.get("task")),
            result=str(payload.get("predicted_class")),
            confidence=payload.get("confidence"),
            probabilities=payload.get("probabilities") or {},
            low_confidence=bool(payload.get("low_confidence", False)),
            scorable=True,
            source_kind=source_kind,
            batch_id=batch_id,
        )

    def update(
        self,
        record_id: str,
        *,
        notes: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        """Change notes and/or tags. Nothing else about a result is editable."""
        changes: dict[str, Any] = {}
        if notes is not None:
            changes["notes"] = _clean_notes(notes)
        if tags is not None:
            changes["tags"] = _clean_tags(tags)
        with self._lock:
            table = self._table()
            found = table.get(Query().id == record_id)
            if found is None:
                return None
            if changes:
                changes["updated_at"] = _iso(_utc_now())
                table.update(changes, Query().id == record_id)
                found = table.get(Query().id == record_id)
        return self.present(dict(found))

    def delete(self, record_id: str) -> bool:
        """Remove one row from the file now, keeping it in memory for `restore`."""
        with self._lock:
            table = self._table()
            found = table.get(Query().id == record_id)
            if found is None:
                return False
            table.remove(Query().id == record_id)
            self._deleted.pop(record_id, None)
            self._deleted[record_id] = dict(found)
            while len(self._deleted) > UNDO_DEPTH:
                self._deleted.pop(next(iter(self._deleted)))
            return True

    def restore(self, record_id: str) -> dict[str, Any] | None:
        """T132.4: undo one deletion. None when it can no longer be undone."""
        with self._lock:
            row = self._deleted.get(record_id)
            if row is None:
                return None
            table = self._table()
            if table.get(Query().id == record_id) is None:
                table.insert(row)
            del self._deleted[record_id]
        return self.present(row)

    def delete_all(self) -> int:
        """Empty the history. Asked for with a confirmation, so it has no undo."""
        with self._lock:
            table = self._table()
            count = len(table)
            table.truncate()
            self._deleted.clear()
            return count

    # -- reads ------------------------------------------------------------

    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            found = self._table().get(Query().id == record_id)
        return None if found is None else self.present(dict(found))

    def _rows(self, condition: QueryInstance | None = None) -> list[dict[str, Any]]:
        with self._lock:
            table = self._table()
            found = table.all() if condition is None else table.search(condition)
        return [_upgrade(dict(row)) for row in found]

    def list_records(
        self,
        *,
        query: str | None = None,
        task: str | None = None,
        result: str | None = None,
        tag: str | None = None,
        low_confidence: bool | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        batch_id: str | None = None,
        sort: str = "created_at",
        order: str = "desc",
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        """T129.4: search, filter, sort and page. Filters combine with AND."""
        if sort not in SORT_FIELDS:
            raise HistoryValidationError("sort must be one of " + ", ".join(SORT_FIELDS))
        if order not in ("asc", "desc"):
            raise HistoryValidationError("order must be asc or desc")
        if page < 1:
            raise HistoryValidationError("page starts at 1")
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise HistoryValidationError("page_size must be between 1 and " + str(MAX_PAGE_SIZE))

        record = Query()
        conditions: list[QueryInstance] = []
        if task:
            conditions.append(record.task == _clean_task(task))
        if result:
            conditions.append(record.result == result)
        if tag:
            wanted = " ".join(tag.split()).casefold()
            conditions.append(
                record.tags.test(lambda tags: wanted in {str(t).casefold() for t in tags or []})
            )
        if low_confidence is not None:
            conditions.append(record.low_confidence == bool(low_confidence))
        wanted_batch = clean_batch_id(batch_id)
        if wanted_batch is not None:
            conditions.append(record.batch_id == wanted_batch)
        lower = _day_bound(date_from, end=False)
        upper = _day_bound(date_to, end=True)
        if lower is not None:
            conditions.append(record.created_at >= lower)
        if upper is not None:
            conditions.append(record.created_at <= upper)
        if lower is not None and upper is not None and lower > upper:
            raise HistoryValidationError("date_from is after date_to")
        needle = " ".join((query or "").split()).casefold()
        if needle:
            conditions.append(_text_match(needle))

        condition: QueryInstance | None = None
        for item in conditions:
            condition = item if condition is None else condition & item
        rows = self._rows(condition)

        rows.sort(key=lambda row: (row["created_at"], row["id"]), reverse=order == "desc")
        if sort != "created_at":
            present = [row for row in rows if row.get(sort) not in (None, "")]
            absent = [row for row in rows if row.get(sort) in (None, "")]
            present.sort(key=lambda row: _sort_key(row[sort]), reverse=order == "desc")
            rows = present + absent  # a missing value sorts last either way

        total = len(rows)
        pages = max(1, math.ceil(total / page_size))
        start = (page - 1) * page_size
        items = [self.present(row) for row in rows[start : start + page_size]]
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "sort": sort,
            "order": order,
            "display": {
                "total": format_value(total, "count"),
                "page": format_value(page, "count"),
                "pages": format_value(pages, "count"),
            },
            "notice": self.notice,
        }

    # -- T131.6 -- batch export ---------------------------------------------

    def export_batch_csv(
        self, *, batch_id: str, task: str, rows: Iterable[Mapping[str, Any]]
    ) -> str:
        """One batch table as CSV text, in the order given.

        A `scored` row is read back from its History entry, which must belong to
        this batch and this task: the file then holds what was saved, formatted
        by `format_value` exactly as `/predict` formatted it for the table. A
        row of any other status has no History entry and carries its message.
        One task per batch -- the five label spaces are never merged.
        """
        clean_batch = clean_batch_id(batch_id)
        if clean_batch is None:
            raise HistoryValidationError("batch_id is required")
        clean_task = _clean_task(task)
        classes = TASKS[clean_task].classes
        items = list(rows)
        if not items:
            raise HistoryValidationError("there are no rows to export")
        if len(items) > MAX_BATCH_ROWS:
            raise HistoryValidationError(
                "a batch export is limited to " + str(MAX_BATCH_ROWS) + " rows"
            )
        stored = {str(row.get("id")): _upgrade(dict(row)) for row in self._rows()}

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow(
            [
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
        )
        for number, item in enumerate(items, start=1):
            if not isinstance(item, Mapping):
                raise HistoryValidationError("row " + str(number) + " is not an object")
            status = item.get("status")
            if status not in BATCH_STATUSES:
                raise HistoryValidationError(
                    "row " + str(number) + ": status must be one of " + ", ".join(BATCH_STATUSES)
                )
            file_name = _clean_file_name(item.get("file_name"))
            if status == "scored":
                record_id = str(item.get("history_id") or "")
                record = stored.get(record_id)
                if record is None or record.get("batch_id") != clean_batch:
                    raise HistoryValidationError(
                        "row "
                        + str(number)
                        + " ("
                        + file_name
                        + ") names no History entry of this batch; it may have been deleted"
                    )
                if record["task"] != clean_task:
                    raise HistoryValidationError(
                        "row " + str(number) + " belongs to another task; a batch is one task"
                    )
                probabilities = record.get("probabilities") or {}
                writer.writerow(
                    [
                        _csv_text(record["file_name"]),
                        status,
                        clean_task,
                        record.get("result") or "",
                        format_value(record.get("confidence"), "metric"),
                        "true" if record.get("low_confidence") else "false",
                        *[format_value(probabilities.get(name), "metric") for name in classes],
                        "",
                        record["id"],
                        record["created_at"],
                        clean_batch,
                    ]
                )
                continue
            message = " ".join(str(item.get("message") or "").split())
            if status in ("failed", "not_scored") and not message:
                raise HistoryValidationError(
                    "row "
                    + str(number)
                    + " ("
                    + file_name
                    + ") needs the message it was refused with"
                )
            writer.writerow(
                [
                    _csv_text(file_name),
                    status,
                    clean_task,
                    "",
                    "",
                    "",
                    *["" for _ in classes],
                    _csv_text(message),
                    "",
                    "",
                    clean_batch,
                ]
            )
        return buffer.getvalue()

    # -- T129.6 -- Insights --------------------------------------------------

    def insights(
        self,
        *,
        task: str | None = None,
        days: int = 30,
        tz_offset_minutes: int = 0,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Counts, per-task result split, daily trend and confidence distribution.

        `tz_offset_minutes` is the viewer's offset from UTC (UTC+05:30 is 330),
        so "today" in the trend is the viewer's day and not London's.
        """
        if not 1 <= days <= MAX_TREND_DAYS:
            raise HistoryValidationError("days must be between 1 and " + str(MAX_TREND_DAYS))
        if not -MAX_TZ_OFFSET_MINUTES <= tz_offset_minutes <= MAX_TZ_OFFSET_MINUTES:
            raise HistoryValidationError("tz_offset_minutes must be within -840 and 840")
        condition = Query().task == _clean_task(task) if task else None
        rows = self._rows(condition)
        zone = timezone(timedelta(minutes=tz_offset_minutes))
        moment = (now or _utc_now()).astimezone(zone)
        total = len(rows)

        def count_entry(count: int, of: int) -> dict[str, Any]:
            fraction = None if of == 0 else count / of
            return {
                "count": count,
                "count_display": format_value(count, "count"),
                "percent": None if fraction is None else fraction * 100.0,
                "percent_display": _percent_text(fraction),
            }

        by_task: list[dict[str, Any]] = []
        result_split: list[dict[str, Any]] = []
        for key, spec in TASKS.items():
            task_rows = [row for row in rows if row["task"] == key]
            if not task_rows:
                continue
            by_task.append({"task": key, "title": spec.title, **count_entry(len(task_rows), total)})
            classes = [
                {
                    "class": name,
                    **count_entry(
                        sum(1 for row in task_rows if row.get("result") == name), len(task_rows)
                    ),
                }
                for name in spec.classes
            ]
            result_split.append(
                {
                    "task": key,
                    "title": spec.title,
                    "total": len(task_rows),
                    "total_display": format_value(len(task_rows), "count"),
                    "classes": classes,
                }
            )

        today = moment.date()
        first_day = today - timedelta(days=days - 1)
        per_day: dict[date, int] = {first_day + timedelta(days=i): 0 for i in range(days)}
        for row in rows:
            local_day = _parse_iso(row["created_at"]).astimezone(zone).date()
            if local_day in per_day:
                per_day[local_day] += 1
        trend = [
            {"date": day.isoformat(), "count": count, "count_display": format_value(count, "count")}
            for day, count in per_day.items()
        ]

        confidences = [row["confidence"] for row in rows if row.get("confidence") is not None]
        counts = [0] * CONFIDENCE_BINS
        for value in confidences:
            counts[min(CONFIDENCE_BINS - 1, math.floor(round(value * CONFIDENCE_BINS, 9)))] += 1
        distribution = []
        for index, count in enumerate(counts):
            lower = index / CONFIDENCE_BINS
            upper = (index + 1) / CONFIDENCE_BINS
            distribution.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "label": format_value(lower, "metric", places=1)
                    + "-"
                    + format_value(upper, "metric", places=1),
                    **count_entry(count, len(confidences)),
                }
            )

        low = sum(1 for row in rows if row.get("low_confidence"))
        in_window = sum(per_day.values())
        return {
            "task": task,
            "total": total,
            "total_display": format_value(total, "count"),
            "low_confidence": count_entry(low, total),
            "by_task": by_task,
            "result_split": result_split,
            "trend": {
                "days": days,
                "tz_offset_minutes": tz_offset_minutes,
                "first_date": first_day.isoformat(),
                "last_date": today.isoformat(),
                "in_window": in_window,
                "in_window_display": format_value(in_window, "count"),
                "points": trend,
            },
            "confidence_distribution": {
                "n": len(confidences),
                "n_display": format_value(len(confidences), "count"),
                "bins": distribution,
            },
            "notice": self.notice,
        }

    # -- presentation -----------------------------------------------------

    @staticmethod
    def present(row: Mapping[str, Any]) -> dict[str, Any]:
        """A stored row plus the display strings a page renders."""
        record = _upgrade(dict(row))
        spec = TASKS.get(record["task"])
        probabilities = record.get("probabilities") or {}
        record["task_title"] = spec.title if spec is not None else record["task"]
        record["display"] = {
            "result": record.get("result") or NA_TEXT,
            "confidence": format_value(record.get("confidence"), "metric"),
            "confidence_percent": _percent_text(record.get("confidence")),
            "probabilities": {
                name: format_value(value, "metric") for name, value in probabilities.items()
            },
            "probabilities_percent": {
                name: _percent_text(value) for name, value in probabilities.items()
            },
        }
        return record


def _upgrade(row: dict[str, Any]) -> dict[str, Any]:
    """Bring a stored row up to `SCHEMA_VERSION`, filling defaults for missing fields.

    Version 1 is the first schema, so today this only guards against a row with
    fields missing. A row from a NEWER schema is passed through unchanged rather
    than rewritten into a shape it did not come from.
    """
    row.setdefault("schema_version", SCHEMA_VERSION)
    row.setdefault("updated_at", row.get("created_at"))
    row.setdefault("source_kind", "manual")
    row.setdefault("scorable", row.get("result") is not None)
    row.setdefault("result", None)
    row.setdefault("confidence", None)
    row.setdefault("low_confidence", False)
    row.setdefault("probabilities", {})
    row.setdefault("notes", "")
    row.setdefault("tags", [])
    row.setdefault("batch_id", None)
    return row


def _text_match(needle: str) -> QueryInstance:
    """Free-text search across file name, task, result, notes and tags."""

    def matches(row: Mapping[str, Any]) -> bool:
        spec = TASKS.get(str(row.get("task")))
        haystack = [
            str(row.get("file_name") or ""),
            str(row.get("task") or ""),
            spec.title if spec is not None else "",
            str(row.get("result") or ""),
            str(row.get("notes") or ""),
            *[str(tag) for tag in row.get("tags") or []],
        ]
        return any(needle in value.casefold() for value in haystack)

    return QueryInstance(matches, ("history_text", needle))


def _sort_key(value: Any) -> Any:
    return value.casefold() if isinstance(value, str) else value
