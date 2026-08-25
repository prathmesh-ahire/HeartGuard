"""End-to-end audit-script gate (T22.7).

T22.7 asks for three things from ``scripts/01_run_dataset_audit.py``: it
completes on a cleared output directory, it is idempotent on a second run, and
it records its wall time.

The full run takes just over two minutes cold, so the expensive part is marked
``slow`` and runs into a temporary directory rather than over the real
``outputs/`` tree -- a test that clears the project's own audit outputs and then
fails leaves the repository worse than it found it. The smoke path
(``--limit``) runs unconditionally, because that is the path a developer uses
first and it must not rot.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.needs_data


def _load_script() -> Any:
    """Import the numeric-prefixed script module by path."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "01_run_dataset_audit.py"
    spec = importlib.util.spec_from_file_location("audit_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    return _load_script()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===========================================================================
# argument parsing
# ===========================================================================


def test_parses_its_flags(script: Any) -> None:
    args = script.parse_args([])
    assert args.limit is None and args.force is False

    args = script.parse_args(["--limit", "20", "--force", "--skip-verify"])
    assert args.limit == 20
    assert args.force is True
    assert args.skip_verify is True


# ===========================================================================
# T22.7 -- the smoke path
# ===========================================================================


@pytest.mark.slow
def test_smoke_run_completes_and_writes_no_da_files(
    script: Any, tmp_path: Path
) -> None:
    """--limit must prove the pipeline runs without leaving a partial artifact."""
    result = script.run_audit(
        script.parse_args(["--limit", "20", "--out-dir", str(tmp_path)])
    )
    assert result["mode"] == "smoke"
    assert 0 < result["n_records"] <= 20 * 10  # 7 PhysioNet subsets + 2 PASCAL + D4
    assert result["seconds"] > 0
    assert list(tmp_path.iterdir()) == [], "a smoke run must write no DA files"


@pytest.mark.slow
def test_smoke_run_records_its_wall_time(script: Any, tmp_path: Path) -> None:
    from src.utils.run_manifest import load_manifest

    script.run_audit(script.parse_args(["--limit", "10", "--out-dir", str(tmp_path)]))
    run = load_manifest()["runs"][-1]
    assert run["name"] == "dataset_audit_smoke"
    assert run["status"] == "completed"
    assert run["extra"]["audit_mode"] == "smoke"
    assert float(run["extra"]["audit_wall_seconds"]) > 0


# ===========================================================================
# T22.7 -- the full run, on a cleared directory, twice
# ===========================================================================


@pytest.mark.slow
def test_full_run_on_a_cleared_directory_is_complete_and_idempotent(
    script: Any, tmp_path: Path
) -> None:
    """The gate. Two runs into an empty directory must agree byte for byte.

    ``--skip-verify`` drops only the CirCor SHA-256 pass over 585 MB, which
    Phase 15 already gates; everything that produces a DA artifact still runs.
    """
    from src.data_loader.inventory import AUDIT_ARTIFACTS

    out_dir = tmp_path / "audit"
    out_dir.mkdir()
    assert list(out_dir.iterdir()) == []

    args = script.parse_args(["--out-dir", str(out_dir), "--skip-verify"])
    first = script.run_audit(args)
    assert first["mode"] == "full"
    assert first["n_records"] == 7536
    assert first["n_supervised"] == 6988
    assert first["seconds"] > 0

    # Every DA artifact exists and is non-empty. DA-05's report is written by
    # the integrity phase; the rest by the phase that owns them.
    for evidence_id, filename, _ in AUDIT_ARTIFACTS:
        path = out_dir / filename
        assert path.is_file(), evidence_id + " missing after a full run"
        assert path.stat().st_size > 0, evidence_id + " is empty"

    before = {
        path.name: _digest(path) for path in sorted(out_dir.glob("*.csv"))
    }
    assert len(before) >= 15

    second = script.run_audit(args)
    assert second["n_records"] == first["n_records"]
    assert second["n_assignments"] == first["n_assignments"]

    after = {path.name: _digest(path) for path in sorted(out_dir.glob("*.csv"))}
    assert set(after) == set(before)
    changed = [name for name in before if before[name] != after[name]]
    assert changed == [], "not idempotent: " + ", ".join(changed)


@pytest.mark.slow
def test_full_run_records_wall_time_and_counts_in_the_manifest(
    script: Any, tmp_path: Path
) -> None:
    """T22.5 -- the wall time is a deliverable, not a log line."""
    from src.utils.run_manifest import load_manifest

    script.run_audit(
        script.parse_args(["--out-dir", str(tmp_path), "--skip-verify"])
    )
    run = load_manifest()["runs"][-1]
    assert run["name"] == "dataset_audit"
    assert run["status"] == "completed"

    extra = run["extra"]
    assert float(extra["audit_wall_seconds"]) > 0
    assert extra["audit_records"] == 7536
    assert extra["audit_supervised"] == 6988
    assert extra["audit_files_scanned"] == 8368  # 7,536 records + 832 Heartbeat_Sound
    assert extra["audit_fold_assignments"] == 23607

    timings = run["timings_sec"]
    for phase in (
        "audit:phase_08_11_physionet",
        "audit:phase_12_13_pascal",
        "audit:phase_14_15_circor",
        "audit:phase_16_integrity",
        "audit:phase_17_duplicates",
        "audit:phase_18_summaries",
        "audit:phase_19_master",
        "audit:phase_20_splits",
        "audit:phase_21_inventory",
    ):
        assert phase in timings, phase
        assert timings[phase] > 0, phase


@pytest.mark.slow
def test_main_returns_zero_on_success_and_one_on_failure(script: Any) -> None:
    """A failed audit must exit nonzero, or CI would go green on a broken run."""
    assert script.main(["--limit", "5"]) == 0
    assert script.main(["--limit", "0"]) == 1  # rejected by apply_limit
