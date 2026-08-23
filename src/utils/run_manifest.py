"""Run manifest (Phase 04, task T04.3).

Rule 5: every run records its seed, fold map, hyperparameters and package
versions. This module is the record. If a number appears in a deliverable, the
manifest is what makes it possible to say *which* run produced it, under which
package versions, from which commit.

``outputs/00_evidence_index/run_manifest.json`` accumulates runs rather than
being overwritten, so a later run never erases the provenance of an earlier one.
Individual experiments embed their own manifest snapshot alongside their results
(T63.3); this file is the index across all of them.

**Git commit when there is no git repository.** Until Phase 07 runs ``git init``
there is no commit to record. The manifest stores an explicit
``{"available": false, "reason": ...}`` rather than ``null`` or a fabricated
placeholder -- "no commit, and here is why" is a fact; a blank field is
ambiguous and a made-up hash is a lie. The same structure is used if ``git`` is
absent from PATH.
"""

from __future__ import annotations

import getpass
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.utils.io import load_json, save_json

__all__ = [
    "RunManifest",
    "SCHEMA_VERSION",
    "start_run",
    "current_run",
    "git_info",
    "package_versions",
    "environment_info",
    "load_manifest",
    "config_snapshot_for",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2: config snapshots deduplicated into a top-level map keyed by content hash,
#    with each run carrying `config_hash` instead of an embedded copy.
SCHEMA_VERSION = 2

# Distributions recorded in every manifest. Sourced from the pinned requirements
# so the manifest reflects what the project declares, not an arbitrary subset.
_REQ_FILES = (
    "requirements.txt",
    "requirements-extra.txt",
    "requirements-api.txt",
    "requirements-report.txt",
)


# ---------------------------------------------------------------------------
# environment capture
# ---------------------------------------------------------------------------


def _pinned_distributions() -> list[str]:
    import re

    pattern = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==")
    names: list[str] = []
    for filename in _REQ_FILES:
        path = PROJECT_ROOT / filename
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            match = pattern.match(line)
            if match:
                names.append(match.group(1))
    return names


def package_versions() -> dict[str, str | None]:
    """Installed version of every pinned distribution, ``None`` if absent."""
    out: dict[str, str | None] = {}
    for dist in _pinned_distributions():
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_info() -> dict[str, Any]:
    """Commit, branch and dirty state -- or an explicit unavailable record."""
    if not (PROJECT_ROOT / ".git").exists():
        return {
            "available": False,
            "reason": (
                "not a git repository -- `git init` is task T07.1, so runs before "
                "Phase 07 have no commit to record"
            ),
            "commit": None,
            "branch": None,
            "dirty": None,
        }

    commit = _run_git(["rev-parse", "HEAD"])
    if commit is None:
        return {
            "available": False,
            "reason": "git is on PATH but `git rev-parse HEAD` failed (no commits yet?)",
            "commit": None,
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": None,
        }

    status = _run_git(["status", "--porcelain"])
    return {
        "available": True,
        "reason": None,
        "commit": commit,
        "commit_short": commit[:8],
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        # A dirty tree means the code that produced this run is not exactly the
        # code at that commit. Recording it is the difference between "this is
        # reproducible" and "this looks reproducible".
        "dirty": bool(status),
        "dirty_files": [line[3:] for line in status.splitlines()] if status else [],
    }


def environment_info() -> dict[str, Any]:
    """Interpreter, OS and hardware facts relevant to reproducing a run."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = None
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "in_venv": sys.prefix != sys.base_prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "user": user,
        # CPU only, by hardware. Recorded so a timing number is never compared
        # against one produced on a GPU box.
        "gpu": "none (CPU-only machine)",
    }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class RunManifest:
    """One run's provenance record."""

    def __init__(
        self,
        *,
        name: str = "run",
        seed: int | None = None,
        config_snapshot: dict[str, Any] | None = None,
        path: str | os.PathLike | None = None,
    ) -> None:
        from src.utils.seed import GLOBAL_SEED, seed_state

        now = datetime.now(timezone.utc)
        self.run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.name = name
        self.started_utc = now.isoformat()
        self.finished_utc: str | None = None
        self.seed = GLOBAL_SEED if seed is None else seed
        self.seed_state = seed_state()
        self.git = git_info()
        self.environment = environment_info()
        self.packages = package_versions()
        self.config_snapshot = config_snapshot if config_snapshot is not None else _snapshot_configs()
        self.timings: dict[str, float] = {}
        self.artifacts: list[str] = []
        self.extra: dict[str, Any] = {}
        self.status = "running"
        self.path = Path(path) if path else _default_manifest_path()

    # -- mutation ----------------------------------------------------------

    def record_timing(self, label: str, seconds: float) -> None:
        """Record a duration. Repeated labels accumulate rather than overwrite."""
        self.timings[label] = round(self.timings.get(label, 0.0) + float(seconds), 6)

    def record_artifact(self, path: str | os.PathLike) -> None:
        value = str(path)
        if value not in self.artifacts:
            self.artifacts.append(value)

    def set(self, key: str, value: Any) -> None:
        """Attach arbitrary run metadata -- fold map, hyperparameters, counts."""
        self.extra[key] = value

    # -- serialization -----------------------------------------------------

    def config_hash(self) -> str:
        """Stable content hash of this run's config snapshot."""
        return _hash_snapshot(self.config_snapshot)

    def to_dict(self) -> dict[str, Any]:
        """The run record.

        Carries ``config_hash`` rather than the snapshot itself. The snapshots
        live once each in the manifest's ``config_snapshots`` map, keyed by that
        hash -- see :meth:`save`.
        """
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "seed": self.seed,
            "seed_state": self.seed_state,
            "git": self.git,
            "environment": self.environment,
            "packages": self.packages,
            "config_hash": self.config_hash(),
            "timings_sec": self.timings,
            "artifacts": self.artifacts,
            "extra": self.extra,
        }

    def save(self) -> Path:
        """Append (or update) this run in the shared manifest file, atomically.

        The config snapshot is **deduplicated**: a full snapshot of all five
        config files is roughly 45 KB, and embedding one per run would push the
        manifest past 20 MB over a few hundred runs while storing the same
        content hundreds of times. Configs change rarely, so snapshots are
        stored once each under their content hash and runs reference it. No
        fidelity is lost -- every run can still be resolved back to the exact
        configuration it ran under, via :func:`config_snapshot_for`.
        """
        from src.utils.logging_setup import logging_state

        self.extra.setdefault("logging", logging_state())

        existing: dict[str, Any] = {"schema": SCHEMA_VERSION, "config_snapshots": {}, "runs": []}
        if self.path.is_file():
            try:
                loaded = load_json(self.path)
                if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                    existing = loaded
                    existing.setdefault("config_snapshots", {})
            except (OSError, ValueError):
                # A corrupt manifest must not take a completed run down with it.
                # Preserve the damaged file rather than overwriting it silently.
                damaged = self.path.with_suffix(".corrupt.json")
                self.path.replace(damaged)

        digest = self.config_hash()
        existing["config_snapshots"].setdefault(digest, self.config_snapshot)

        runs = [r for r in existing["runs"] if r.get("run_id") != self.run_id]
        runs.append(self.to_dict())
        existing["runs"] = runs
        existing["schema"] = SCHEMA_VERSION
        existing["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(existing, self.path)
        return self.path

    def finish(self, status: str = "completed") -> Path:
        self.status = status
        self.finished_utc = datetime.now(timezone.utc).isoformat()
        started = datetime.fromisoformat(self.started_utc)
        finished = datetime.fromisoformat(self.finished_utc)
        self.timings["_total_run"] = round((finished - started).total_seconds(), 6)
        return self.save()

    def __repr__(self) -> str:
        return "RunManifest(" + self.run_id + ", " + self.name + ", " + self.status + ")"


# ---------------------------------------------------------------------------
# module-level active run
# ---------------------------------------------------------------------------

_active: RunManifest | None = None


def _default_manifest_path() -> Path:
    try:
        from src.utils.config import load_config

        return Path(load_config("paths").require("outputs.run_manifest"))
    except Exception:  # noqa: BLE001
        return PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json"


def _snapshot_configs() -> dict[str, Any]:
    """Every config file as loaded, so a run's settings are recoverable later."""
    try:
        from src.utils.config import CONFIG_NAMES, load_config
    except Exception as exc:  # noqa: BLE001
        return {"error": "config snapshot unavailable: " + str(exc)}

    snapshot: dict[str, Any] = {}
    for name in CONFIG_NAMES:
        try:
            cfg = load_config(name)
            snapshot[name] = {"values": cfg.as_dict(), "env_overrides": cfg.overrides}
        except Exception as exc:  # noqa: BLE001
            snapshot[name] = {"error": str(exc)}
    return snapshot


def _hash_snapshot(snapshot: dict[str, Any]) -> str:
    """Stable 16-hex content hash of a config snapshot."""
    import hashlib
    import json as _json

    from src.utils.io import sanitize_for_json

    payload = _json.dumps(sanitize_for_json(snapshot), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_manifest(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Read the manifest file, returning an empty structure if absent."""
    target = Path(path) if path else _default_manifest_path()
    if not target.is_file():
        return {"schema": SCHEMA_VERSION, "config_snapshots": {}, "runs": []}
    return load_json(target)


def config_snapshot_for(
    run_id: str, path: str | os.PathLike | None = None
) -> dict[str, Any] | None:
    """Resolve a run back to the exact configuration it ran under."""
    data = load_manifest(path)
    for record in data.get("runs", []):
        if record.get("run_id") == run_id:
            digest = record.get("config_hash")
            return data.get("config_snapshots", {}).get(digest)
    return None


def start_run(
    name: str = "run",
    *,
    seed: int | None = None,
    apply_seed: bool = True,
    setup_logs: bool = True,
) -> RunManifest:
    """Begin a run: seed, configure logging, and open a manifest."""
    global _active

    from src.utils.seed import GLOBAL_SEED, set_global_seed

    resolved = GLOBAL_SEED if seed is None else seed
    if apply_seed:
        set_global_seed(resolved)
    if setup_logs:
        from src.utils.logging_setup import setup_logging

        setup_logging()

    _active = RunManifest(name=name, seed=resolved)
    _active.save()
    return _active


def current_run() -> RunManifest | None:
    """The active manifest, or None if no run has been started."""
    return _active
