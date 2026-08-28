"""Evidence index (Phase 04, task T04.6).

Every generated artifact -- table, figure, model, report -- registers one row
here. Phase 102 assembles these rows into ``evidence_index.xlsx`` and asserts
that every row resolves to a real file on disk (T102.3) and that every mandatory
item from both source documents appears (T102.4).

The schema is fixed by T102.2:

    evidence_id, objective, experiment_id, dataset, model, metric_or_asset,
    filename, source_data, command, timestamp, status

Two fields carry most of the weight:

``source_data``
    The CSV or JSON the artifact's numbers came from. A figure whose
    ``source_data`` is blank cannot be traced back to a run, which is the
    definition of an untraceable number under rule 1.

``status``
    ``ok`` when the file exists, ``missing`` when it does not. Registration
    checks the filesystem rather than trusting the caller, so a row can never
    claim an artifact that was never written.

Registration is an **upsert on evidence_id**, not a blind append. Re-running a
phase should refresh its rows, not create duplicates that then disagree with
each other about which file is current.
"""

from __future__ import annotations

import contextlib
import csv
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.io import atomic_path

__all__ = [
    "EVIDENCE_COLUMNS",
    "register_evidence",
    "index_for_artifact",
    "read_evidence",
    "evidence_index_path",
    "is_inside_project",
    "verify_evidence",
]

EVIDENCE_COLUMNS = [
    "evidence_id",
    "objective",
    "experiment_id",
    "dataset",
    "model",
    "metric_or_asset",
    "filename",
    "source_data",
    "command",
    "timestamp",
    "status",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evidence_index_path() -> Path:
    """Location of ``evidence_index.csv``."""
    try:
        from src.utils.config import load_config

        return Path(load_config("paths").require("outputs.evidence_index")) / "evidence_index.csv"
    except Exception:  # noqa: BLE001
        return PROJECT_ROOT / "outputs" / "00_evidence_index" / "evidence_index.csv"


def is_inside_project(path: str | os.PathLike) -> bool:
    """True when ``path`` lives under the project root.

    Public because more than the evidence index needs the rule: a run written
    to a scratch directory must not leave rows in the project's index *or* its
    run manifest, and both call this to decide.
    """
    try:
        Path(path).resolve().relative_to(PROJECT_ROOT)
    except (ValueError, OSError):
        return False
    return True


def _index_for(artifact: str | os.PathLike) -> Path:
    """Which index a given artifact's row belongs in.

    **An artifact outside the project root never registers into the project's
    index.** Its row goes to an index beside the artifact instead.

    This is a structural guard, not a style preference. Any run pointed at a
    different output directory -- a smoke run, a ``--out-dir`` invocation, a
    pytest ``tmp_path`` -- otherwise rewrites the committed rows to paths that
    are deleted minutes later, leaving an index that claims ``status=ok`` for
    files nobody can open. The Phase 30 sweep found exactly that: all nine
    DA rows pointing into ``AppData/Local/Temp/pytest-of-prath/pytest-40/``,
    written there by the T22.7 audit-script test, and no gate had noticed
    because every gate checked the artifacts rather than the index.

    Putting the rule here rather than in each caller means a future phase cannot
    reintroduce the leak by writing one more ``register_evidence`` call.
    """
    if is_inside_project(artifact):
        return evidence_index_path()
    return Path(artifact).resolve().parent / "evidence_index.csv"


def _relative(path: str | os.PathLike) -> str:
    """Store paths relative to the project root so the index stays portable."""
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return candidate.as_posix()


def index_for_artifact(artifact: str | os.PathLike) -> Path:
    """Public form of the routing rule; see :func:`_index_for`."""
    return _index_for(artifact)


def read_evidence(path: str | os.PathLike | None = None) -> list[dict[str, str]]:
    """Every registered row, in file order."""
    target = Path(path) if path else evidence_index_path()
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_rows(rows: list[dict[str, str]], target: Path) -> None:
    with (
        atomic_path(target, suffix=".csv") as tmp,
        tmp.open("w", encoding="utf-8", newline="") as fh,
    ):
        writer = csv.DictWriter(fh, fieldnames=EVIDENCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in EVIDENCE_COLUMNS})


def register_evidence(
    evidence_id: str,
    filename: str | os.PathLike,
    *,
    metric_or_asset: str = "",
    objective: str = "",
    experiment_id: str = "",
    dataset: str = "",
    model: str = "",
    source_data: str | os.PathLike = "",
    command: str = "",
    status: str | None = None,
    index_path: str | os.PathLike | None = None,
) -> dict[str, str]:
    """Register one artifact, returning the row written.

    ``status`` is determined from the filesystem unless explicitly given: an
    artifact that does not exist is recorded as ``missing``, never as ``ok``.
    The row is also attached to the active run manifest.
    """
    artifact = Path(filename)
    target = Path(index_path) if index_path else _index_for(artifact)

    if status is None:
        status = "ok" if artifact.is_file() else "missing"

    row: dict[str, str] = {
        "evidence_id": evidence_id,
        "objective": objective,
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model": model,
        "metric_or_asset": metric_or_asset,
        "filename": _relative(artifact),
        "source_data": _relative(source_data) if source_data else "",
        "command": command,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": status,
    }

    rows = [r for r in read_evidence(target) if r.get("evidence_id") != evidence_id]
    rows.append(row)
    _write_rows(rows, target)

    # Attaching to the manifest is best-effort: a manifest problem must not
    # lose an artifact registration that already succeeded on disk.
    with contextlib.suppress(Exception):
        from src.utils.run_manifest import current_run

        run = current_run()
        if run is not None:
            run.record_artifact(row["filename"])

    return row


def verify_evidence(index_path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Check every row against the filesystem (the T102.3 check, callable early).

    Returns counts plus the rows whose files are missing and the rows with no
    ``source_data`` -- an artifact with no traceable source is the shape of
    problem rule 1 exists to catch.
    """
    rows = read_evidence(index_path)
    missing: list[dict[str, str]] = []
    untraceable: list[dict[str, str]] = []
    for row in rows:
        name = row.get("filename", "")
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if not candidate.is_file():
            missing.append(row)
        if not row.get("source_data"):
            untraceable.append(row)
    return {
        "total": len(rows),
        "ok": len(rows) - len(missing),
        "missing": missing,
        "missing_count": len(missing),
        "no_source_data": untraceable,
        "no_source_data_count": len(untraceable),
    }
