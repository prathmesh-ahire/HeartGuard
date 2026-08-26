"""Preprocessing pipeline gate (T27.6, T27.7).

Rule 5 in its most literal form: two runs of the same command must produce
identical numbers. Not "statistically indistinguishable" -- **bit-identical**,
asserted with ``np.array_equal`` on float32 samples. Everything in the chain is
deterministic (soxr, sosfiltfilt, a z-score), so anything less than exact
equality means something non-deterministic crept in.

The second half of T27.7 is the cache: a second pass must actually shortcut, not
merely agree. A cache that silently misses is invisible in the numbers and only
shows up as a pipeline that takes an hour every time it is asked for something
it already computed.

Every test here writes into a temporary cache directory, never the real
``cache/preprocessed/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.preprocessing import pipeline as pipe

RECORDS_FOR_DETERMINISM = 200


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``cache.preprocessed`` at the config layer, not by patching code.

    Going through the real environment-override mechanism means this fixture
    exercises the same path a user would use to relocate the cache, rather than
    proving something about a monkeypatched function that production never calls.
    """
    from src.utils import config as config_module

    target = tmp_path / "preprocessed"
    monkeypatch.setenv("HEARTGUARD__PATHS__CACHE__PREPROCESSED", str(target))
    config_module.clear_cache()
    yield target
    config_module.clear_cache()


@pytest.fixture
def one_record(master_frame: Any, project_root: Path) -> tuple[str, Path]:
    row = master_frame[master_frame["dataset_source"] == "D3"].iloc[0]
    return str(row["record_uid"]), project_root / str(row["file_path"])


# ===========================================================================
# T27.3 -- the config hash
# ===========================================================================


def test_config_hash_is_stable(signal_config: Any) -> None:
    assert pipe.config_hash(signal_config) == pipe.config_hash(signal_config)
    assert len(pipe.config_hash(signal_config)) == 12


def _cfg_with(signal_config: Any, section: str, **overrides: Any) -> Any:
    import copy

    from src.utils.config import Config

    data = copy.deepcopy(signal_config.as_dict())
    data[section].update(overrides)
    return Config("signal", data)


def test_a_changed_signal_setting_changes_the_hash(signal_config: Any) -> None:
    base = pipe.config_hash(signal_config)
    assert pipe.config_hash(_cfg_with(signal_config, "filter", low_hz=25)) != base
    assert pipe.config_hash(_cfg_with(signal_config, "filter", enabled=False)) != base
    assert pipe.config_hash(_cfg_with(signal_config, "normalization", method="peak")) != base
    assert pipe.config_hash(_cfg_with(signal_config, "resample", target_fs=4000)) != base


def test_an_audit_threshold_does_not_change_the_hash(signal_config: Any) -> None:
    """The documented point of hashing four sections rather than the whole file.

    ``integrity.*`` describes raw files for the Phase 16 audit. The pipeline
    never reads it, so changing it must not invalidate 7,536 cached signals.
    """
    changed = _cfg_with(signal_config, "integrity", near_duplicate_correlation=0.5)
    assert pipe.config_hash(changed) == pipe.config_hash(signal_config)


def test_different_configs_use_different_cache_directories(
    signal_config: Any, tmp_cache: Path
) -> None:
    other = _cfg_with(signal_config, "normalization", method="peak")
    assert pipe.cache_root(signal_config) != pipe.cache_root(other)
    assert pipe.cache_path("D1_x_a0001", signal_config).name == "D1_x_a0001.npz"


# ===========================================================================
# T27.1 / T27.2 -- the chain and its result object
# ===========================================================================


@pytest.mark.needs_data
def test_preprocess_reports_every_step(one_record: tuple[str, Path], tmp_cache: Path) -> None:
    uid, path = one_record
    result = pipe.preprocess(path, record_uid=uid)

    assert result.fs == 2000
    assert result.fs_native == 4000
    assert result.record_uid == uid
    assert result.signal.dtype == np.float32
    assert np.isfinite(result.signal).all()
    assert result.from_cache is False

    kinds = [step.split(":")[0] for step in result.steps]
    assert kinds == ["load", "mono", "resample", "quality", "filter", "normalize"]
    assert any(step.startswith("resample:4000->2000") for step in result.steps)

    # T27.2 -- the structured result carries the metrics, not just the samples.
    assert result.quality["snr_proxy_db"] != 0.0
    assert result.filter_info["filter_applied"] is True
    assert result.normalization_info["norm_applied"] is True
    assert result.normalization_info["std_after_norm"] == pytest.approx(1.0, abs=1e-3)

    row = result.as_dict()
    assert row["record_uid"] == uid
    assert "signal" not in row


@pytest.mark.needs_data
def test_length_survives_the_chain(one_record: tuple[str, Path], tmp_cache: Path) -> None:
    """Filtering and normalization must not trim or pad."""
    from src.preprocessing import io as pio

    uid, path = one_record
    resampled, _ = pio.load_resampled(path)
    result = pipe.preprocess(path, record_uid=uid, use_cache=False)
    assert result.n_samples == resampled.size


# ===========================================================================
# T27.7 -- determinism and the cache
# ===========================================================================


@pytest.mark.needs_data
def test_two_uncached_runs_are_bit_identical(
    one_record: tuple[str, Path], tmp_cache: Path
) -> None:
    uid, path = one_record
    first = pipe.preprocess(path, record_uid=uid, use_cache=False)
    second = pipe.preprocess(path, record_uid=uid, use_cache=False)

    assert np.array_equal(first.signal, second.signal)
    assert first.steps == second.steps
    assert first.quality == second.quality


@pytest.mark.needs_data
def test_the_cache_shortcuts_the_second_pass(
    one_record: tuple[str, Path], tmp_cache: Path
) -> None:
    uid, path = one_record

    cold = pipe.preprocess(path, record_uid=uid)
    assert cold.from_cache is False
    assert pipe.cache_path(uid).is_file()

    warm = pipe.preprocess(path, record_uid=uid)
    assert warm.from_cache is True
    assert np.array_equal(cold.signal, warm.signal)
    assert warm.steps == cold.steps
    assert warm.quality == cold.quality

    stats = pipe.cache_stats()
    assert stats["n_files"] == 1
    assert stats["bytes"] > 0


@pytest.mark.needs_data
def test_force_recomputes_and_overwrites(one_record: tuple[str, Path], tmp_cache: Path) -> None:
    uid, path = one_record
    pipe.preprocess(path, record_uid=uid)
    forced = pipe.preprocess(path, record_uid=uid, force=True)
    assert forced.from_cache is False


@pytest.mark.needs_data
def test_a_corrupt_cache_entry_is_recomputed(
    one_record: tuple[str, Path], tmp_cache: Path
) -> None:
    """An interrupted run leaves half-written entries; they must not be fatal."""
    uid, path = one_record
    good = pipe.preprocess(path, record_uid=uid)

    entry = pipe.cache_path(uid)
    entry.write_bytes(b"not an npz file")

    recovered = pipe.preprocess(path, record_uid=uid)
    assert recovered.from_cache is False
    assert np.array_equal(recovered.signal, good.signal)


@pytest.mark.needs_data
def test_clear_cache_removes_entries(one_record: tuple[str, Path], tmp_cache: Path) -> None:
    uid, path = one_record
    pipe.preprocess(path, record_uid=uid)
    assert pipe.clear_preprocessed_cache() == 1
    assert pipe.cache_stats()["n_files"] == 0


@pytest.mark.needs_data
@pytest.mark.slow
def test_two_hundred_records_preprocess_identically_twice(
    master_frame: Any, project_root: Path, tmp_cache: Path
) -> None:
    """T27.7 -- 200 random records, twice, bit-identical, second pass cached."""
    sample = master_frame.sample(n=RECORDS_FOR_DETERMINISM, random_state=42)

    first: dict[str, np.ndarray] = {}
    for row in sample.itertuples(index=False):
        result = pipe.preprocess(
            project_root / str(row.file_path), record_uid=str(row.record_uid)
        )
        assert result.from_cache is False, row.record_uid
        assert np.isfinite(result.signal).all(), row.record_uid
        first[str(row.record_uid)] = result.signal

    assert pipe.cache_stats()["n_files"] == RECORDS_FOR_DETERMINISM

    for row in sample.itertuples(index=False):
        result = pipe.preprocess(
            project_root / str(row.file_path), record_uid=str(row.record_uid)
        )
        assert result.from_cache is True, row.record_uid
        assert np.array_equal(result.signal, first[str(row.record_uid)]), row.record_uid

    # And bit-identical when recomputed from scratch, not merely read back.
    for row in sample.head(20).itertuples(index=False):
        fresh = pipe.preprocess(
            project_root / str(row.file_path), record_uid=str(row.record_uid), use_cache=False
        )
        assert np.array_equal(fresh.signal, first[str(row.record_uid)]), row.record_uid


@pytest.mark.needs_data
def test_duration_extremes_pass_through_the_pipeline(
    master_frame: Any, project_root: Path, tmp_cache: Path
) -> None:
    """0.76 s and 122 s: neither errors, neither is silently truncated."""
    for index in (master_frame["duration_sec"].idxmin(), master_frame["duration_sec"].idxmax()):
        row = master_frame.loc[index]
        result = pipe.preprocess(
            project_root / str(row["file_path"]), record_uid=str(row["record_uid"])
        )
        assert np.isfinite(result.signal).all(), row["record_uid"]
        assert result.duration_sec == pytest.approx(float(row["duration_sec"]), abs=1e-2)
        assert result.filter_info["filter_too_short"] is False
