"""Bandpass filter gate (T24.6, T24.7).

The three tone tests the task names -- 5 Hz attenuated, 200 Hz passed, 800 Hz
attenuated -- plus the two properties that are easy to lose silently: the filter
must not shift the signal in time (zero phase, because Phase 34 measures S1-S2
intervals on the filtered trace), and it must reproduce the analytic Butterworth
magnitude rather than merely look like a bandpass.

``test_shortest_real_record_filters`` is the T24.7 padlen check on the actual
0.76-second PASCAL B recording, not on a synthetic stand-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.preprocessing import filters as flt
from src.preprocessing import io as pio

FS = 2000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tone(freq_hz: float, seconds: float = 4.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs), dtype=np.float64) / fs
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _rms(x: np.ndarray, *, trim: int = 200) -> float:
    """RMS with the filter's edge transient trimmed off both ends."""
    core = np.asarray(x, dtype=np.float64)[trim:-trim] if x.size > 2 * trim else np.asarray(x)
    return float(np.sqrt(np.mean(core * core))) if core.size else 0.0


def _gain_db(freq_hz: float) -> float:
    tone = _tone(freq_hz)
    out = flt.bandpass_filter(tone, FS).signal
    return 20.0 * np.log10(max(_rms(out) / _rms(tone), 1e-12))


# ===========================================================================
# T24.1 / T24.2 -- design
# ===========================================================================


def test_design_is_four_second_order_sections() -> None:
    """Order 4 prototype -> 8-pole bandpass -> 4 SOS sections."""
    sos = flt.design_bandpass(FS)
    assert sos.shape == (4, 6)
    assert np.isfinite(sos).all()


def test_design_returns_a_copy_of_the_cached_array() -> None:
    """A caller mutating its sections must not poison every later record."""
    first = flt.design_bandpass(FS)
    first[0, 0] = 999.0
    assert flt.design_bandpass(FS)[0, 0] != 999.0


def test_design_rejects_a_passband_at_or_above_nyquist() -> None:
    with pytest.raises(ValueError):
        flt.design_bandpass(FS, 20.0, 1000.0)
    with pytest.raises(ValueError):
        flt.design_bandpass(FS, 400.0, 20.0)


def test_effective_order_records_the_doubling() -> None:
    """T24.2 -- prototype 4, realised 8 poles, doubled to 16 by sosfiltfilt."""
    result = flt.bandpass_filter(_tone(200.0), FS)
    assert result.order == 4
    assert result.effective_order == 16
    assert result.zero_phase is True

    single = flt.bandpass_filter(_tone(200.0), FS, zero_phase=False)
    assert single.effective_order == 8


# ===========================================================================
# T24.6 / T24.7 -- the three tones
# ===========================================================================


def test_5hz_tone_is_attenuated() -> None:
    gain = _gain_db(5.0)
    assert gain < -40.0, "5 Hz gain " + str(gain) + " dB"


def test_200hz_tone_passes() -> None:
    gain = _gain_db(200.0)
    assert gain > -0.5, "200 Hz gain " + str(gain) + " dB"


def test_800hz_tone_is_attenuated() -> None:
    gain = _gain_db(800.0)
    assert gain < -40.0, "800 Hz gain " + str(gain) + " dB"


def test_passband_edges_are_where_the_design_says() -> None:
    """-6.02 dB at 20 and 400 Hz, because the filter runs twice."""
    assert _gain_db(20.0) == pytest.approx(-6.02, abs=0.5)
    assert _gain_db(400.0) == pytest.approx(-6.02, abs=0.5)


def test_baseline_drift_is_removed(synthetic_pcg: Any) -> None:
    """The reason the low cutoff exists, on a PCG rather than a bare tone."""
    t = np.arange(synthetic_pcg.signal.size, dtype=np.float64) / FS
    drift = (0.5 * np.sin(2 * np.pi * 1.5 * t)).astype(np.float32)
    drifted = (synthetic_pcg.signal + drift).astype(np.float32)

    clean = flt.bandpass_filter(synthetic_pcg.signal, FS).signal
    recovered = flt.bandpass_filter(drifted, FS).signal

    trim = 400
    residual = _rms(recovered[trim:-trim] - clean[trim:-trim], trim=0)
    assert residual / _rms(clean) < 0.02, residual


# ===========================================================================
# T24.2 -- zero phase
# ===========================================================================


def test_zero_phase_filtering_introduces_no_delay(synthetic_pcg: Any) -> None:
    """A single forward pass lags; the forward-backward pass must not."""
    x = synthetic_pcg.signal.astype(np.float64)
    trim = 400

    zero_phase = flt.bandpass_filter(x, FS).signal.astype(np.float64)[trim:-trim]
    forward = flt.bandpass_filter(x, FS, zero_phase=False).signal.astype(np.float64)[trim:-trim]
    reference = x[trim:-trim]

    def best_lag(a: np.ndarray, b: np.ndarray, span: int = 100) -> int:
        a = a - a.mean()
        b = b - b.mean()
        lags = np.arange(-span, span + 1)
        scores = [
            np.dot(a[span + lag : a.size - span + lag], b[span : b.size - span])
            for lag in lags
        ]
        return int(lags[int(np.argmax(scores))])

    assert best_lag(zero_phase, reference) == 0
    assert best_lag(forward, reference) != 0


# ===========================================================================
# T24.4 -- validation against the documented equation
# ===========================================================================


def test_implemented_response_matches_the_butterworth_equation() -> None:
    report = flt.validate_response()
    assert report["max_abs_deviation"] < 1e-9, report
    assert report["n_sections"] == 4
    assert report["gain_low_cutoff_db"] == pytest.approx(-3.0103, abs=1e-3)
    assert report["gain_high_cutoff_db"] == pytest.approx(-3.0103, abs=1e-3)
    assert report["gain_centre_db"] == pytest.approx(0.0, abs=1e-6)
    assert report["zero_phase_gain_low_cutoff_db"] == pytest.approx(-6.0206, abs=1e-3)


def test_analytic_magnitude_is_zero_at_dc_and_nyquist() -> None:
    freqs = np.array([0.0, 1000.0, 1200.0], dtype=np.float64)
    magnitude = flt.butterworth_bandpass_magnitude(freqs, FS)
    assert np.isfinite(magnitude).all()
    assert (magnitude == 0.0).all()


def test_transfer_function_plot_is_written(tmp_path: Path) -> None:
    out = flt.plot_transfer_function(tmp_path / "filter_transfer_function.png")
    assert out.is_file()
    assert out.stat().st_size > 5_000


# ===========================================================================
# T24.3 -- the short-signal guard
# ===========================================================================


def test_shortest_synthetic_record_needs_no_reduced_padding() -> None:
    """0.76 s is 1,520 samples; the padlen is 27. The guard does not fire."""
    x = np.asarray(flt.design_bandpass(FS))  # touch the design so padlen is real
    padlen = flt.default_padlen(x)
    assert padlen == 27

    short = _tone(150.0, seconds=0.76)
    assert short.size == 1520
    result = flt.bandpass_filter(short, FS)
    assert result.applied is True
    assert result.padlen_reduced is False
    assert result.too_short is False
    assert np.isfinite(result.signal).all()


def test_padlen_is_reduced_rather_than_raising() -> None:
    result = flt.bandpass_filter(_tone(150.0, seconds=0.01), FS)  # 20 samples
    assert result.applied is True
    assert result.padlen_reduced is True
    assert result.padlen == 19
    assert np.isfinite(result.signal).all()


def test_a_signal_too_short_to_pad_is_flagged_not_filtered() -> None:
    x = _tone(150.0, seconds=0.004)  # 8 samples
    result = flt.bandpass_filter(x, FS)
    assert result.applied is False
    assert result.too_short is True
    assert np.allclose(result.signal, x)

    empty = flt.bandpass_filter(np.zeros(0, dtype=np.float32), FS)
    assert empty.signal.size == 0
    assert empty.too_short is True


# ===========================================================================
# T24.5 -- the config-switchable no-filter path
# ===========================================================================


def _signal_cfg_with(signal_config: Any, **overrides: Any) -> Any:
    """A copy of configs/signal.yaml with ``filter.*`` keys replaced."""
    import copy

    from src.utils.config import Config

    data = copy.deepcopy(signal_config.as_dict())
    data["filter"].update(overrides)
    return Config("signal", data)


def test_disabled_filter_returns_the_input_untouched(synthetic_signal: np.ndarray) -> None:
    result = flt.bandpass_filter(synthetic_signal, FS, enabled=False)
    assert result.applied is False
    assert np.allclose(result.signal, synthetic_signal)


def test_filter_signal_follows_the_config(signal_config: Any, synthetic_signal: np.ndarray) -> None:
    on = flt.filter_signal(synthetic_signal, FS, signal_config)
    assert on.applied is True
    assert (on.low_hz, on.high_hz, on.order) == (20.0, 400.0, 4)

    off = flt.filter_signal(synthetic_signal, FS, _signal_cfg_with(signal_config, enabled=False))
    assert off.applied is False
    assert np.allclose(off.signal, synthetic_signal)


def test_unknown_short_signal_policy_is_an_error(
    signal_config: Any, synthetic_signal: np.ndarray
) -> None:
    cfg = _signal_cfg_with(signal_config, short_signal_policy="truncate")
    with pytest.raises(ValueError, match="short_signal_policy"):
        flt.filter_signal(synthetic_signal, FS, cfg)


# ===========================================================================
# contract
# ===========================================================================


def test_output_is_float32_finite_and_the_same_length(synthetic_signal: np.ndarray) -> None:
    result = flt.bandpass_filter(synthetic_signal, FS)
    assert result.signal.dtype == np.float32
    assert result.signal.shape == synthetic_signal.shape
    assert np.isfinite(result.signal).all()


def test_filtering_is_deterministic(synthetic_signal: np.ndarray) -> None:
    first = flt.bandpass_filter(synthetic_signal, FS).signal
    second = flt.bandpass_filter(synthetic_signal, FS).signal
    assert np.array_equal(first, second)


# ===========================================================================
# real records -- T24.7
# ===========================================================================


@pytest.mark.needs_data
def test_shortest_real_record_filters(master_frame: Any, project_root: Path) -> None:
    """T24.7 -- the actual 0.76 s recording, resampled then filtered."""
    row = master_frame.loc[master_frame["duration_sec"].idxmin()]
    assert float(row["duration_sec"]) < 1.0

    signal, _ = pio.load_resampled(project_root / str(row["file_path"]))
    result = flt.bandpass_filter(signal, FS)

    assert result.applied is True
    assert result.padlen_reduced is False, row["record_uid"]
    assert result.too_short is False, row["record_uid"]
    assert result.signal.size == signal.size
    assert np.isfinite(result.signal).all()


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset_source", ["D1", "D2", "D3", "D4"])
def test_real_records_filter_without_error(sample_records: Any, dataset_source: str) -> None:
    for uid, path in sample_records(dataset_source, 5):
        signal, _ = pio.load_resampled(path)
        result = flt.bandpass_filter(signal, FS)

        assert result.applied is True, uid
        assert result.signal.dtype == np.float32, uid
        assert result.signal.size == signal.size, uid
        assert np.isfinite(result.signal).all(), uid
        # Filtering removes energy; it must never manufacture it.
        assert _rms(result.signal) <= _rms(signal) * 1.05 + 1e-9, uid
