"""The batch extraction runner (Phase 39, gate T39.7).

The gate: run in ``--smoke`` mode over 20 records per dataset, and confirm
checkpointing resumes correctly after a deliberate interrupt.

"Resumes correctly" is tested as three separate claims, because two of them are
easy to pass accidentally:

1. after the interrupt, the completed chunk is on disk;
2. the restart *skips* those records rather than redoing them;
3. the final table is identical to what an uninterrupted run produces.

A resume that quietly re-extracted everything would satisfy (1) and (3) and be
useless. A resume that dropped the interrupted chunk would satisfy (1) and (2)
and be wrong. All three are asserted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import batch
from src.feature_extraction.registry import FAMILY_ORDER, FEATURE_NAMES

EXPECTED_TOTAL = 138


class _Interrupt(RuntimeError):
    """Stands in for the closed laptop / Ctrl-C the checkpointing exists for."""


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the feature cache at a throwaway tree for the whole test.

    ``cache/features/`` is a real, expensive artifact; no test writes into it.
    """
    target = tmp_path / "features"
    monkeypatch.setenv("HEARTGUARD__PATHS__CACHE__FEATURES", str(target))

    from src.utils import config as config_module

    config_module.clear_cache()
    yield target
    config_module.clear_cache()


# ---------------------------------------------------------------------------
# cache identity
# ---------------------------------------------------------------------------


def test_the_digest_covers_the_registry_not_only_the_settings():
    """A renamed column changes what every number means, even with identical maths."""
    from src.feature_extraction import registry as reg

    baseline = batch.cache_digest()
    assert len(baseline) == 12
    assert batch.cache_digest() == baseline, "digest must be deterministic"

    original = reg.FEATURE_NAMES
    try:
        reg.FEATURE_NAMES = ("renamed", *original[1:])  # type: ignore[misc]
        assert batch.cache_digest() != baseline
    finally:
        reg.FEATURE_NAMES = original  # type: ignore[misc]


def test_smoke_and_full_runs_never_share_a_directory(isolated_cache: Path):
    """A 20-row shard where a 3,240-row shard belongs reads like a complete one."""
    full = batch.shard_path("D1", smoke=False)
    smoke = batch.shard_path("D1", smoke=True)

    # Compared as path *components*, not as a substring of the whole string:
    # pytest's own tmp_path for this test is named after the test, so a substring
    # check matches the temporary directory rather than the shard layout.
    assert full != smoke
    assert "_smoke" in smoke.parts
    assert "_smoke" not in full.relative_to(full.parents[1]).parts
    assert smoke.parent.parent.name == "_smoke"


def test_shard_paths_live_under_the_configured_feature_cache(isolated_cache: Path):
    path = batch.shard_path("D2")
    assert isolated_cache in path.parents
    assert path.name == "D2_features.parquet"


# ---------------------------------------------------------------------------
# the worker contract
# ---------------------------------------------------------------------------


def test_a_row_carries_the_138_features_plus_its_bookkeeping():
    expected = set(FEATURE_NAMES) | set(batch.META_COLUMNS) | set(batch.TIMING_COLUMNS)
    assert len(batch.TIMING_COLUMNS) == len(FAMILY_ORDER)
    assert len(expected) == EXPECTED_TOTAL + len(batch.META_COLUMNS) + len(FAMILY_ORDER)


def test_an_unreadable_file_becomes_an_error_row_not_an_exception(tmp_path: Path):
    """T39.4 -- one bad file must not kill a 7,536-record run."""
    row = batch._extract_one("bad-uid", "does/not/exist.wav", "D1", str(tmp_path))

    assert row["record_uid"] == "bad-uid"
    assert row["ok"] is False
    assert row["failed_families"] == "preprocess"
    assert row["error"]
    assert row["n_missing"] == EXPECTED_TOTAL
    assert all(np.isnan(row[name]) for name in FEATURE_NAMES)


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        batch.run_extraction(["D9"])


# ---------------------------------------------------------------------------
# error reporting (T39.4)
# ---------------------------------------------------------------------------


def test_errors_are_reported_per_family_not_per_record():
    """"MFCC failed on a short upload" and "the file would not decode" differ."""
    import pandas as pd

    table = pd.DataFrame(
        [
            {
                "record_uid": "R-1",
                "dataset_source": "D3",
                "failed_families": "mfcc;chroma",
                "error": "chroma: ValueError: boom; mfcc: RuntimeError: bang",
                "n_missing": 63,
                "flags": "mfcc:mfcc_signal_too_short",
            },
            {
                "record_uid": "R-2",
                "dataset_source": "D3",
                "failed_families": "",
                "error": "",
                "n_missing": 0,
                "flags": "",
            },
        ]
    )

    rows = batch._error_rows(table)
    assert len(rows) == 2
    assert {row["family"] for row in rows} == {"mfcc", "chroma"}
    assert all(row["record_uid"] == "R-1" for row in rows)

    by_family = {row["family"]: row["error"] for row in rows}
    assert by_family["mfcc"] == "RuntimeError: bang"
    assert by_family["chroma"] == "ValueError: boom"


# ---------------------------------------------------------------------------
# T39.7 -- the gate: smoke mode, and resume after an interrupt
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_smoke_run_extracts_twenty_records_per_dataset(isolated_cache: Path,
                                                       tmp_output_dir: Path):
    """T39.6 -- the fast validation path."""
    summary = batch.run_extraction(
        ["D2", "D3"], n_jobs=2, smoke=True, progress=False, out_dir=tmp_output_dir
    )

    assert summary.smoke is True
    assert summary.datasets == {"D2": 20, "D3": 20}
    assert summary.n_records == 40
    assert summary.n_failed == 0, summary.errors_path

    for dataset in ("D2", "D3"):
        table = batch.load_shard(dataset, smoke=True)
        assert len(table) == 20
        assert set(FEATURE_NAMES).issubset(table.columns)
        assert np.isfinite(table[list(FEATURE_NAMES)].to_numpy()).all(), dataset
        assert bool(table["ok"].all())
        assert table["record_uid"].is_unique

    assert Path(summary.errors_path).is_file()


@pytest.mark.needs_data
def test_an_interrupted_run_resumes_from_its_checkpoint(isolated_cache: Path):
    """The T39.7 assertion, in its three parts. See the module docstring."""
    dataset = "D3"
    digest = batch.cache_digest()

    # --- a clean reference run, uninterrupted -----------------------------
    reference = batch.extract_dataset(
        dataset, n_jobs=2, limit=12, checkpoint_every=4, smoke=True, progress=False
    )
    assert len(reference) == 12
    expected_uids = sorted(reference["record_uid"].astype(str))

    # Start over so the interrupted run has nothing to inherit.
    batch.clear_feature_cache(smoke=True)

    # --- an interrupted run: die right after the first checkpoint ---------
    def die_after_first_chunk(index: int, _written: int) -> None:
        if index == 0:
            raise _Interrupt("simulated interrupt after chunk 0")

    with pytest.raises(_Interrupt):
        batch.extract_dataset(
            dataset,
            n_jobs=2,
            limit=12,
            checkpoint_every=4,
            smoke=True,
            progress=False,
            on_chunk=die_after_first_chunk,
        )

    # (1) the completed chunk survived the interrupt
    done = batch.completed_uids(dataset, digest=digest, smoke=True)
    assert len(done) == 4, "the first checkpoint was not on disk"
    assert not batch.shard_path(dataset, digest=digest, smoke=True).is_file()

    # (2) the restart skips what is already done
    seen: list[str] = []
    original = batch._extract_one

    def spy(uid: str, relative: str, ds: str, root: str) -> dict[str, Any]:
        seen.append(uid)
        return original(uid, relative, ds, root)

    batch._extract_one = spy  # type: ignore[assignment]
    try:
        resumed = batch.extract_dataset(
            dataset, n_jobs=1, limit=12, checkpoint_every=4, smoke=True, progress=False
        )
    finally:
        batch._extract_one = original  # type: ignore[assignment]

    assert len(seen) == 8, "the resume re-extracted records it already had"
    assert not (set(seen) & done), "the resume redid completed records"

    # (3) the result is what an uninterrupted run produces
    assert len(resumed) == 12
    assert sorted(resumed["record_uid"].astype(str)) == expected_uids

    left = reference.set_index("record_uid")[list(FEATURE_NAMES)].sort_index()
    right = resumed.set_index("record_uid")[list(FEATURE_NAMES)].sort_index()
    assert np.array_equal(left.to_numpy(), right.to_numpy()), (
        "resumed values differ from an uninterrupted run"
    )

    # And the checkpoints are cleaned up once the shard is written.
    assert not batch._part_paths(batch.checkpoint_dir(dataset, digest=digest, smoke=True))


@pytest.mark.needs_data
def test_a_second_run_over_a_finished_shard_extracts_nothing(isolated_cache: Path):
    """Resume is also what makes re-running the whole corpus cheap."""
    dataset = "D2"
    batch.extract_dataset(dataset, n_jobs=2, limit=8, smoke=True, progress=False)

    seen: list[str] = []
    original = batch._extract_one

    def spy(uid: str, relative: str, ds: str, root: str) -> dict[str, Any]:
        seen.append(uid)
        return original(uid, relative, ds, root)

    batch._extract_one = spy  # type: ignore[assignment]
    try:
        again = batch.extract_dataset(dataset, n_jobs=1, limit=8, smoke=True, progress=False)
    finally:
        batch._extract_one = original  # type: ignore[assignment]

    assert seen == [], "a completed shard was re-extracted"
    assert len(again) == 8


@pytest.mark.needs_data
def test_force_discards_the_shard_and_re_extracts(isolated_cache: Path):
    dataset = "D2"
    first = batch.extract_dataset(dataset, n_jobs=2, limit=6, smoke=True, progress=False)

    seen: list[str] = []
    original = batch._extract_one

    def spy(uid: str, relative: str, ds: str, root: str) -> dict[str, Any]:
        seen.append(uid)
        return original(uid, relative, ds, root)

    batch._extract_one = spy  # type: ignore[assignment]
    try:
        second = batch.extract_dataset(
            dataset, n_jobs=1, limit=6, smoke=True, progress=False, force=True
        )
    finally:
        batch._extract_one = original  # type: ignore[assignment]

    assert len(seen) == 6, "--force did not re-extract"
    assert np.array_equal(
        first.set_index("record_uid")[list(FEATURE_NAMES)].sort_index().to_numpy(),
        second.set_index("record_uid")[list(FEATURE_NAMES)].sort_index().to_numpy(),
    )


@pytest.mark.needs_data
def test_a_truncated_checkpoint_is_discarded_rather_than_trusted(isolated_cache: Path):
    """A process killed mid-write leaves an unreadable part; it must not be believed."""
    dataset = "D2"
    digest = batch.cache_digest()

    def die_after_first_chunk(index: int, _written: int) -> None:
        if index == 0:
            raise _Interrupt("stop")

    with pytest.raises(_Interrupt):
        batch.extract_dataset(
            dataset,
            n_jobs=2,
            limit=8,
            checkpoint_every=4,
            smoke=True,
            progress=False,
            on_chunk=die_after_first_chunk,
        )

    parts = batch._part_paths(batch.checkpoint_dir(dataset, digest=digest, smoke=True))
    assert len(parts) == 1
    parts[0].write_bytes(b"not a parquet file")

    assert batch.completed_uids(dataset, digest=digest, smoke=True) == set()
    assert not parts[0].exists(), "the unreadable checkpoint was not removed"


# ---------------------------------------------------------------------------
# timing, for the complexity table
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_every_family_reports_a_timing_per_record(isolated_cache: Path):
    table = batch.extract_dataset("D2", n_jobs=2, limit=6, smoke=True, progress=False)

    for column in batch.TIMING_COLUMNS:
        assert column in table.columns
        assert (table[column] >= 0).all()

    summary = batch.summarize_timings(table)
    assert list(summary["family"]) == list(FAMILY_ORDER)
    assert float(summary["share_of_total"].sum()) == pytest.approx(1.0)
    assert (summary["n_records"] == 6).all()


# ---------------------------------------------------------------------------
# the CLI (T39.1)
# ---------------------------------------------------------------------------


def test_the_cli_parses_every_documented_option():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "extract_cli", root / "scripts" / "02_extract_features.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(
        ["--dataset", "D2", "D3", "--limit", "5", "--workers", "2", "--force", "--smoke"]
    )
    assert args.dataset == ["D2", "D3"]
    assert args.limit == 5
    assert args.workers == 2
    assert args.force is True
    assert args.smoke is True

    default = module.parse_args([])
    assert default.dataset == ["all"]
    assert default.workers == -1
    assert default.force is False


def test_the_cli_rejects_an_unknown_dataset():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "extract_cli_reject", root / "scripts" / "02_extract_features.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(SystemExit):
        module.parse_args(["--dataset", "D9"])
