"""The Reports store (Phase 134, T134.4): recently generated reports, for
re-download.

Every report this module hands out (a per-recording PDF, the model-summary
PDF, or a bulk History CSV export) is also saved here so it can be listed and
re-downloaded later without re-generating it -- a re-download must be
byte-identical to what was first produced, which regenerating cannot promise
once History has changed underneath it.

Built on the same `AtomicJSONStorage` `history_store.py` uses (Phase 129): one
`RLock`, one atomic-write JSON index, the same corrupt-file recovery. The
metadata row lives in the TinyDB file; the report's bytes live next to it as a
plain file named by the row's id, because TinyDB documents are JSON and a PDF
is not text.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinydb import Query, TinyDB

from src.api.history_store import AtomicJSONStorage
from src.utils.logging_setup import get_logger

__all__ = ["ReportStore", "default_reports_db_path", "default_reports_dir"]

log = get_logger("api.report_store")

TABLE_NAME = "reports"
#: Bounds `cache/reports/` growth. Oldest entries (row + file) are pruned past
#: this, the same "cap and prune" shape as History's undo buffer (Phase 132).
MAX_ENTRIES = 30


def default_reports_db_path() -> Path:
    from src.utils.config import load_config

    configured = load_config("paths").get("cache.reports_db")
    if not configured:
        raise ValueError("cache.reports_db is not configured in paths.yaml")
    return Path(str(configured))


def default_reports_dir() -> Path:
    from src.utils.config import load_config

    configured = load_config("paths").get("cache.reports_dir")
    if not configured:
        raise ValueError("cache.reports_dir is not configured in paths.yaml")
    return Path(str(configured))


def _utc_now_iso() -> str:
    """Microsecond precision, like `history_store._iso` -- so string order is
    time order even for reports generated in the same wall-clock second (a
    batch, or several exports in a row), not just usually.
    """
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class ReportStore:
    """Metadata index plus the files it points at. Every method takes one lock."""

    def __init__(
        self, db_path: str | Path | None = None, files_dir: str | Path | None = None
    ) -> None:
        self._configured_db = Path(db_path) if db_path is not None else None
        self._configured_dir = Path(files_dir) if files_dir is not None else None
        self._lock = threading.RLock()
        self._db: TinyDB | None = None
        self._recoveries: list[str] = []

    @property
    def db_path(self) -> Path:
        if self._configured_db is None:
            self._configured_db = default_reports_db_path()
        return self._configured_db

    @property
    def files_dir(self) -> Path:
        if self._configured_dir is None:
            self._configured_dir = default_reports_dir()
        self._configured_dir.mkdir(parents=True, exist_ok=True)
        return self._configured_dir

    def _table(self) -> Any:
        if self._db is None:
            self._db = TinyDB(
                self.db_path, storage=AtomicJSONStorage, on_recover=self._recoveries.append
            )
        return self._db.table(TABLE_NAME, cache_size=0)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @property
    def notice(self) -> str | None:
        if not self._recoveries:
            return None
        return (
            "The saved report list could not be read (" + self._recoveries[-1] + "), so it was "
            "set aside and a new, empty list was started. Nothing already downloaded is affected."
        )

    def save(
        self,
        *,
        kind: str,
        title: str,
        filename: str,
        content_type: str,
        content: bytes,
        record_id: str | None = None,
    ) -> dict[str, Any]:
        """Write the file and its index row; prune past `MAX_ENTRIES`."""
        report_id = uuid.uuid4().hex
        with self._lock:
            path = self.files_dir / report_id
            path.write_bytes(content)
            row = {
                "id": report_id,
                "kind": kind,
                "title": title,
                "filename": filename,
                "content_type": content_type,
                "size_bytes": len(content),
                "record_id": record_id,
                "created_at": _utc_now_iso(),
            }
            self._table().insert(row)
            self._prune_locked()
            return row

    def _prune_locked(self) -> None:
        rows = sorted(self._table().all(), key=lambda r: r["created_at"])
        surplus = len(rows) - MAX_ENTRIES
        if surplus <= 0:
            return
        record = Query()
        for row in rows[:surplus]:
            self._table().remove(record.id == row["id"])
            (self.files_dir / str(row["id"])).unlink(missing_ok=True)

    def list_recent(self, limit: int = MAX_ENTRIES) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(self._table().all(), key=lambda r: r["created_at"], reverse=True)
            return [dict(row) for row in rows[: max(1, min(limit, MAX_ENTRIES))]]

    def get(self, report_id: str) -> tuple[bytes, Mapping[str, Any]] | None:
        with self._lock:
            record = Query()
            row = self._table().get(record.id == report_id)
            if row is None:
                return None
            path = self.files_dir / report_id
            if not path.is_file():
                return None
            return path.read_bytes(), dict(row)
