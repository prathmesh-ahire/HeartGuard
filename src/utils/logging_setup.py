"""Logging configuration (Phase 04, task T04.2).

Dual output: a readable console stream and a rotating file in ``outputs/logs/``.

Two deliberate choices worth knowing:

**Idempotent.** Calling :func:`setup_logging` twice does not attach a second set
of handlers -- the usual cause of every line appearing two or three times in a
long run. Handlers this module owns are tagged and replaced, so a second call
reconfigures rather than accumulates.

**The file always gets DEBUG.** The console level is tunable and defaults to
INFO, but the file handler records DEBUG regardless. Long runs are where the
useful detail lives, and re-running a six-hour extraction because the failing
record id was only ever printed at DEBUG is not an acceptable outcome.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "setup_logging",
    "get_logger",
    "current_log_file",
    "logging_state",
]

_OWNED = "_pvmepcg_handler"
_DEFAULT_LOG_NAME = "pvmepcg"

_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 5               # 50 MB ceiling per log name

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_log_file: Path | None = None
_configured = False


def _resolve_log_dir(log_dir: str | os.PathLike | None) -> Path:
    if log_dir is not None:
        return Path(log_dir)
    try:
        from src.utils.config import load_config

        return Path(load_config("paths").require("outputs.logs"))
    except Exception:  # noqa: BLE001 - logging must never fail on config problems
        return Path(__file__).resolve().parents[2] / "outputs" / "logs"


def setup_logging(
    name: str = _DEFAULT_LOG_NAME,
    *,
    level: int | str = logging.INFO,
    log_dir: str | os.PathLike | None = None,
    filename: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """Configure the root logger with console + rotating file handlers.

    Returns the named logger. Safe to call repeatedly.
    """
    global _log_file, _configured

    directory = _resolve_log_dir(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (filename or (name + ".log"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # handlers do the filtering, not the root

    # Drop only the handlers this module previously installed.
    for handler in list(root.handlers):
        if getattr(handler, _OWNED, False):
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    setattr(file_handler, _OWNED, True)
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(level if isinstance(level, int) else logging.getLevelName(level))
        stream.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
        setattr(stream, _OWNED, True)
        root.addHandler(stream)

    # Third-party noise that would otherwise dominate a long audio run.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    _log_file = path
    _configured = True
    return logging.getLogger(name)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger, configuring logging on first use."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name or _DEFAULT_LOG_NAME)


def current_log_file() -> Path | None:
    """Path of the active log file, or None if logging is not configured."""
    return _log_file


def logging_state() -> dict[str, Any]:
    """The logging block recorded in the run manifest."""
    return {
        "configured": _configured,
        "log_file": str(_log_file) if _log_file else None,
        "max_bytes": _MAX_BYTES,
        "backup_count": _BACKUP_COUNT,
    }
