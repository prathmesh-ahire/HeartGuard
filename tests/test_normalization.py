"""Normalization gate (T25.6, T25.7).

The headline assertion is T25.7: 50 real records per dataset, preprocessed the
way the pipeline will preprocess them, must come out at mean ~0 and SD ~1. That
runs over 200 real recordings spanning 0.76 s to 122 s and three native sampling
rates, because "the z-score formula is correct" is not the claim being tested --
"it is correct on every record in this corpus" is.

The synthetic half covers what real records happen not to contain: an exactly
constant signal, an exactly silent one, and a large DC offset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.preprocessing import filters as flt
from src.preprocessing import io as pio
from src.preprocessing import normalize as nrm

FS = 2000
SAMPLES_PER_DATASET = 50


# ===========================================================================
# T25.1 -- z-score
# ===========================================================================


def test_zscore_gives_mean_zero_sd_one(synthetic_signal: np.ndarray) -> None:
    result = nrm.normalize(synthetic_signal, method="zscore")

    assert result.applied is True
    assert result.zero_variance is False
    assert result.mean_after == pytest.approx(0.0, abs=1e-5)
    assert result.std_after == pytest.approx(1.0, abs=1e-5)
    assert result.signal.dtype == np.float32
    assert result.signal.shape == synthetic_signal.shape
    assert np.isfinite(result.signal).all()


def test_zscore_is_scale_and_offset_invariant(synthetic_signal: np.ndarray) -> None:
    """The whole purpose: recording gain must not survive this step."""
    quiet = nrm.normalize(synthetic_signal * 0.01, method="zscore").signal
    loud = nrm.normalize(synthetic_signal * 25.0 + 3.0, method="zscore").signal
    assert np.allclose(quiet, loud, atol=1e-4)


def test_zscore_preserves_waveform_shape(synthetic_signal: np.ndarray) -> None:
    """Amplitude changes; the signal does not."""
    normalized = nrm.normalize(synthetic_signal, method="zscore").signal
    assert float(np.corrcoef(synthetic_signal, normalized)[0, 1]) > 0.999999


@pytest.mark.parametrize("kind", ["constant", "silence"])
def test_zero_variance_records_pass_through_flagged(
    edge_case_signals: dict[str, np.ndarray], kind: str
) -> None:
    """T25.1 -- the guard. No inf, no NaN, and the flag says why."""
    x = edge_case_signals[kind]
    result = nrm.normalize(x, method="zscore")

    assert result.zero_variance is True
    assert result.applied is False
    assert np.isfinite(result.signal).all()
    assert np.array_equal(result.signal, np.asarray(x, dtype=np.float32))


def test_a_constant_signal_is_not_silently_divided_by_epsilon() -> None:
    """Returning ``x / 1e-10`` would be 1e10, not an error anyone would notice."""
    constant = np.full(2000, 0.25, dtype=np.float32)
    result = nrm.normalize(constant, method="zscore")
    assert float(np.max(np.abs(result.signal))) == pytest.approx(0.25)


# ===========================================================================
# T25.2 -- peak
# ===========================================================================


def test_peak_normalization_puts_the_loudest_sample_at_one(
    synthetic_signal: np.ndarray,
) -> None:
    result = nrm.normalize(synthetic_signal, method="peak")

    assert result.applied is True
    assert result.peak_after == pytest.approx(1.0, abs=1e-6)
    assert float(np.max(np.abs(result.signal))) <= 1.0 + 1e-6
    assert np.isfinite(result.signal).all()


def test_peak_normalization_of_silence_is_flagged(
    edge_case_signals: dict[str, np.ndarray],
) -> None:
    result = nrm.normalize(edge_case_signals["silence"], method="peak")
    assert result.zero_variance is True
    assert np.isfinite(result.signal).all()


def test_peak_and_zscore_differ(synthetic_signal: np.ndarray) -> None:
    """Rule 4's cousin: two ablation arms that produce the same numbers are one arm."""
    peak = nrm.normalize(synthetic_signal, method="peak").signal
    zscore = nrm.normalize(synthetic_signal, method="zscore").signal
    assert not np.allclose(peak, zscore)


# ===========================================================================
# T25.3 -- DC removal
# ===========================================================================


def test_remove_dc_centres_the_signal(synthetic_signal: np.ndarray) -> None:
    offset = (synthetic_signal + 0.4).astype(np.float32)
    centred = nrm.remove_dc(offset)

    assert centred.dtype == np.float32
    assert float(np.mean(centred.astype(np.float64))) == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(centred, nrm.remove_dc(synthetic_signal), atol=1e-6)


def test_dc_removal_changes_the_peak_normalized_result() -> None:
    """Without it, an offset inflates the divisor and quietly shrinks the signal."""
    t = np.arange(4000, dtype=np.float64) / FS
    offset_tone = (np.sin(2 * np.pi * 100 * t) + 0.5).astype(np.float32)

    with_dc = nrm.normalize(offset_tone, method="peak", remove_dc_offset=False)
    without_dc = nrm.normalize(offset_tone, method="peak", remove_dc_offset=True)

    assert with_dc.dc_removed is False
    assert without_dc.dc_removed is True
    assert with_dc.mean_after == pytest.approx(1 / 3, abs=1e-3)
    assert without_dc.mean_after == pytest.approx(0.0, abs=1e-6)


def test_zscore_centres_even_without_the_dc_flag(synthetic_signal: np.ndarray) -> None:
    """Subtracting the mean is what a z-score is; the flag cannot switch that off."""
    offset = (synthetic_signal + 2.0).astype(np.float32)
    result = nrm.normalize(offset, method="zscore", remove_dc_offset=False)
    assert result.mean_after == pytest.approx(0.0, abs=1e-5)
    assert result.std_after == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# T25.4 -- the switchable no-normalization path
# ===========================================================================


def test_disabled_and_none_both_pass_through(synthetic_signal: np.ndarray) -> None:
    disabled = nrm.normalize(synthetic_signal, enabled=False)
    identity = nrm.normalize(synthetic_signal, method="none")

    for result in (disabled, identity):
        assert result.applied is False
        assert np.allclose(result.signal, synthetic_signal)
    assert disabled.method == "zscore"
    assert identity.method == "none"


def test_unknown_method_is_an_error(synthetic_signal: np.ndarray) -> None:
    with pytest.raises(ValueError, match="unknown normalization method"):
        nrm.normalize(synthetic_signal, method="minmax")


def _signal_cfg_with(signal_config: Any, **overrides: Any) -> Any:
    import copy

    from src.utils.config import Config

    data = copy.deepcopy(signal_config.as_dict())
    data["normalization"].update(overrides)
    return Config("signal", data)


def test_normalize_signal_follows_the_config(
    signal_config: Any, synthetic_signal: np.ndarray
) -> None:
    on = nrm.normalize_signal(synthetic_signal, signal_config)
    assert on.applied is True
    assert on.method == "zscore"
    assert on.std_after == pytest.approx(1.0, abs=1e-5)

    off = nrm.normalize_signal(synthetic_signal, _signal_cfg_with(signal_config, enabled=False))
    assert off.applied is False
    assert np.allclose(off.signal, synthetic_signal)


def test_unknown_zero_variance_policy_is_an_error(
    signal_config: Any, synthetic_signal: np.ndarray
) -> None:
    cfg = _signal_cfg_with(signal_config, zero_variance_policy="clip")
    with pytest.raises(ValueError, match="zero_variance_policy"):
        nrm.normalize_signal(synthetic_signal, cfg)


# ===========================================================================
# contract
# ===========================================================================


def test_empty_signal_is_handled(synthetic_signal: np.ndarray) -> None:
    result = nrm.normalize(np.zeros(0, dtype=np.float32))
    assert result.signal.size == 0
    assert result.zero_variance is True


def test_statistics_are_computed_in_float64() -> None:
    """A float32 accumulator over a 122 s record loses the mean it is measuring."""
    long_signal = (np.sin(np.arange(244_000) * 0.01) + 1000.0).astype(np.float32)
    stats = nrm.normalization_stats(long_signal)
    assert stats["mean"] == pytest.approx(1000.0, abs=1e-2)

    result = nrm.normalize(long_signal, method="zscore")
    assert result.mean_after == pytest.approx(0.0, abs=1e-4)
    assert result.std_after == pytest.approx(1.0, abs=1e-4)


def test_normalization_is_deterministic(synthetic_signal: np.ndarray) -> None:
    first = nrm.normalize(synthetic_signal).signal
    second = nrm.normalize(synthetic_signal).signal
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    "kind", ["shortest_real", "clipped", "constant", "silence", "impulse"]
)
def test_edge_cases_never_produce_nan_or_inf(
    edge_case_signals: dict[str, np.ndarray], kind: str
) -> None:
    if kind not in edge_case_signals:
        pytest.skip("fixture has no " + kind + " case")
    for method in ("zscore", "peak", "none"):
        result = nrm.normalize(edge_case_signals[kind], method=method)
        assert np.isfinite(result.signal).all(), (kind, method)


# ===========================================================================
# T25.5 / T25.7 -- 50 real records per dataset
# ===========================================================================


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset_source", ["D1", "D2", "D3", "D4"])
def test_fifty_real_records_per_dataset_normalize_to_zero_mean_unit_sd(
    sample_records: Any, dataset_source: str, signal_config: Any
) -> None:
    """T25.7 -- the full chain (load, mono, resample, filter, normalize)."""
    records = sample_records(dataset_source, SAMPLES_PER_DATASET)
    assert len(records) == SAMPLES_PER_DATASET, dataset_source

    for uid, path in records:
        signal, _ = pio.load_resampled(path, cfg=signal_config)
        filtered = flt.filter_signal(signal, FS, signal_config)
        result = nrm.normalize_signal(filtered.signal, signal_config)

        assert np.isfinite(result.signal).all(), uid
        assert result.signal.dtype == np.float32, uid
        assert result.signal.size == signal.size, uid
        if result.zero_variance:
            # A dead recording is a data fact, not a normalization failure --
            # but it has to be visible, not averaged into a passing test.
            pytest.fail("zero-variance record reached normalization: " + uid)
        assert result.mean_after == pytest.approx(0.0, abs=1e-3), uid
        assert result.std_after == pytest.approx(1.0, abs=1e-3), uid


@pytest.mark.needs_data
def test_the_quietest_real_record_still_normalizes(
    master_frame: Any, project_root: Path, signal_config: Any
) -> None:
    """set_a's near-silent recording, the one the guard exists for."""
    row = master_frame[master_frame["record_uid"].str.contains("Aunlabelledtest__201106120928")]
    if row.empty:
        pytest.skip("the near-silent set_a record is not in this master table")

    path = project_root / str(row.iloc[0]["file_path"])
    signal, _ = pio.load_resampled(path, cfg=signal_config)
    filtered = flt.filter_signal(signal, FS, signal_config)
    result = nrm.normalize_signal(filtered.signal, signal_config)

    assert np.isfinite(result.signal).all()
    if not result.zero_variance:
        assert result.mean_after == pytest.approx(0.0, abs=1e-3)
        assert result.std_after == pytest.approx(1.0, abs=1e-3)
