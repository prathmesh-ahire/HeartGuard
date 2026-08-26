"""Audio IO and resampling gate (T23.6, T23.7).

Two halves. The synthetic half checks shape, dtype, finiteness and RMS on
signals whose answer is known in advance. The ``needs_data`` half runs the same
code over real records from all four families, including the duration extremes,
because a resampler that works on a clean 5-second sine says nothing about a
0.76-second PASCAL B record or a 122-second PhysioNet one.

**The aliasing test is the point of this file.** ``test_swept_sine_no_aliasing``
sweeps 20 Hz -> 20 kHz at 44.1 kHz and resamples to 2 kHz. Everything above
1 kHz must disappear; if the anti-alias filter were missing it would fold back
into 20-400 Hz, which is where every diagnostic feature in this project lives.
Naive decimation (``x[::22]``) is run beside it as a control, so the assertion
is calibrated against the failure it exists to catch rather than against a
number chosen by eye.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.preprocessing import io as pio

CHIRP_FS = 44100
CHIRP_SEC = 4.0
CHIRP_F0 = 20.0
CHIRP_F1 = 20000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _swept_sine(fs: int = CHIRP_FS, seconds: float = CHIRP_SEC) -> np.ndarray:
    """Linear chirp 20 Hz -> 20 kHz, unit amplitude, float32."""
    from scipy.signal import chirp

    t = np.arange(round(seconds * fs), dtype=np.float64) / fs
    return chirp(t, f0=CHIRP_F0, f1=CHIRP_F1, t1=seconds, method="linear").astype(np.float32)


def _chirp_time_of(freq_hz: float, seconds: float = CHIRP_SEC) -> float:
    """When the linear sweep passes ``freq_hz``."""
    return seconds * (freq_hz - CHIRP_F0) / (CHIRP_F1 - CHIRP_F0)


# ===========================================================================
# T23.1 / T23.2 -- load and mono
# ===========================================================================


def test_target_rate_agrees_with_config(signal_config: Any) -> None:
    """The module constant and configs/signal.yaml must not drift apart."""
    assert pio.DEFAULT_TARGET_FS == 2000
    assert pio.target_fs(signal_config) == pio.DEFAULT_TARGET_FS
    assert pio.resample_method(signal_config) in pio.RESAMPLE_METHODS


def test_declared_native_rates(signal_config: Any) -> None:
    """T23.5 -- the three native rates the corpus actually holds."""
    assert pio.expected_native_fs("D1", signal_config) == 2000
    assert pio.expected_native_fs("D2", signal_config) == 44100
    assert pio.expected_native_fs("D3", signal_config) == 4000
    assert pio.expected_native_fs("D4", signal_config) == 4000
    assert pio.expected_native_fs("d2_pascal_a", signal_config) == 44100
    assert pio.expected_native_fs("nope", signal_config) is None


def test_load_wav_missing_file_raises() -> None:
    with pytest.raises(pio.AudioLoadError):
        pio.load_wav(Path("does_not_exist_1234.wav"))


def test_to_mono_is_a_noop_on_1d(synthetic_signal: np.ndarray) -> None:
    mono = pio.to_mono(synthetic_signal)
    assert mono is synthetic_signal
    assert mono.ndim == 1
    assert mono.dtype == np.float32


def test_to_mono_averages_channels() -> None:
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    mono = pio.to_mono(stereo)
    assert mono.shape == (100,)
    assert mono.dtype == np.float32
    assert np.allclose(mono, 0.5)

    with pytest.raises(ValueError):
        pio.to_mono(np.zeros((2, 3, 4), dtype=np.float32))


# ===========================================================================
# T23.3 -- resampling contract
# ===========================================================================


def test_resample_ratio_is_exact() -> None:
    """44.1 kHz -> 2 kHz is 20/441, not a factor of 22."""
    assert pio.resample_ratio(44100, 2000) == Fraction(20, 441)
    assert pio.resample_ratio(4000, 2000) == Fraction(1, 2)
    assert pio.resample_ratio(2000, 2000) == Fraction(1, 1)


@pytest.mark.parametrize("fs_in", [44100, 4000, 2000])
@pytest.mark.parametrize("method", ["soxr_hq", "polyphase"])
def test_resample_output_length_dtype_and_finiteness(fs_in: int, method: str) -> None:
    """T23.6 -- the four contract properties, on every native rate."""
    seconds = 3.0
    t = np.arange(int(seconds * fs_in), dtype=np.float64) / fs_in
    x = (0.5 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)

    y = pio.resample_to(x, fs_in, 2000, method=method)

    assert y.dtype == np.float32
    assert y.ndim == 1
    assert np.isfinite(y).all()
    expected = round(x.size * 2000 / fs_in)
    assert abs(y.size - expected) <= 1, (fs_in, method, y.size, expected)


@pytest.mark.parametrize("fs_in", [44100, 4000])
def test_resample_preserves_rms_of_an_in_band_signal(fs_in: int) -> None:
    """T23.6 -- an in-band signal keeps its energy; only out-of-band is removed."""
    seconds = 3.0
    t = np.arange(int(seconds * fs_in), dtype=np.float64) / fs_in
    x = (0.5 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)

    y = pio.resample_to(x, fs_in, 2000)
    assert _rms(y) == pytest.approx(_rms(x), rel=0.02)


def test_resample_noop_path_returns_input_untouched(synthetic_signal: np.ndarray) -> None:
    """PhysioNet's path: 2 kHz in, 2 kHz out, nothing done."""
    y = pio.resample_to(synthetic_signal, 2000, 2000)
    assert y is synthetic_signal


def test_resample_empty_and_invalid_inputs() -> None:
    assert pio.resample_to(np.zeros(0, dtype=np.float32), 44100, 2000).size == 0
    with pytest.raises(ValueError):
        pio.resample_to(np.zeros(10, dtype=np.float32), 0, 2000)
    with pytest.raises(ValueError):
        pio.resample_to(np.zeros(10, dtype=np.float32), 4000, 2000, method="nearest")


def test_soxr_and_polyphase_agree(synthetic_pcg_factory: Any) -> None:
    """The two backends must be interchangeable, not merely both present."""
    pcg = synthetic_pcg_factory(duration_sec=4.0, fs=44100)
    a = pio.resample_to(pcg.signal, 44100, 2000, method="soxr_hq")
    b = pio.resample_to(pcg.signal, 44100, 2000, method="polyphase")

    n = min(a.size, b.size)
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
    assert corr > 0.99, corr


def test_resample_is_deterministic(synthetic_pcg_factory: Any) -> None:
    """Rule 5 -- two runs of the same command produce identical numbers."""
    pcg = synthetic_pcg_factory(duration_sec=2.0, fs=44100)
    first = pio.resample_to(pcg.signal, 44100, 2000)
    second = pio.resample_to(pcg.signal, 44100, 2000)
    assert np.array_equal(first, second)


# ===========================================================================
# T23.4 / T23.7 -- the 44.1 kHz decimation and its aliasing check
# ===========================================================================


def test_swept_sine_no_aliasing() -> None:
    """T23.7 -- nothing survives once the sweep passes the 1 kHz Nyquist.

    The output is split at the moment the input sweep crosses 1 kHz. Before it,
    the chirp is a legitimate in-band signal; after it, every sample is either
    silence (correct) or an alias (a bug). The ratio between the two is the
    measurement.
    """
    x = _swept_sine()
    y = pio.resample_to(x, CHIRP_FS, 2000)

    fs_out = 2000
    in_band = y[int(_chirp_time_of(100.0) * fs_out) : int(_chirp_time_of(800.0) * fs_out)]
    # 0.1 s of guard after the crossing so the resampler's transition band and
    # filter ring-out are not counted as aliasing.
    after = y[int((_chirp_time_of(1000.0) + 0.1) * fs_out) :]

    assert in_band.size > 0 and after.size > 0
    ratio = _rms(after) / _rms(in_band)
    alias_db = 20 * np.log10(max(ratio, 1e-30))
    assert ratio < 1e-3, "alias energy above 1 kHz: " + str(alias_db) + " dB"

    # Control: naive decimation, the thing this module exists to avoid. If this
    # ever stops aliasing, the assertion above has stopped proving anything.
    naive = x[::22]
    naive_in = naive[int(_chirp_time_of(100.0) * fs_out) : int(_chirp_time_of(800.0) * fs_out)]
    naive_after = naive[int((_chirp_time_of(1000.0) + 0.1) * fs_out) :]
    assert _rms(naive_after) / _rms(naive_in) > 0.5


def test_swept_sine_spectrum_has_no_out_of_band_content() -> None:
    """The same check in the frequency domain, over the post-1 kHz portion."""
    from scipy.signal import welch

    y = pio.resample_to(_swept_sine(), CHIRP_FS, 2000)
    tail = y[int((_chirp_time_of(1000.0) + 0.1) * 2000) :]

    nperseg = min(512, tail.size)
    freqs, power = welch(tail.astype(np.float64), fs=2000, nperseg=nperseg)
    band = (freqs >= 20) & (freqs <= 400)
    # The whole tail should be at the noise floor; compare it with the in-band
    # power of the legitimate portion of the same signal.
    head = y[: int(_chirp_time_of(800.0) * 2000)]
    _, head_power = welch(head.astype(np.float64), fs=2000, nperseg=min(512, head.size))

    assert power[band].max() / head_power.max() < 1e-6


# ===========================================================================
# real records -- T23.5
# ===========================================================================


@pytest.mark.needs_data
@pytest.mark.parametrize(
    ("dataset_source", "native_fs"),
    [("D1", 2000), ("D2", 44100), ("D3", 4000), ("D4", 4000)],
)
def test_real_records_load_at_their_declared_rate(
    sample_records: Any, dataset_source: str, native_fs: int
) -> None:
    """T23.5 -- every family's files are at the rate the audit recorded."""
    records = sample_records(dataset_source, 5)
    assert records, dataset_source

    for uid, path in records:
        samples, fs = pio.load_wav(path)
        assert fs == native_fs, uid
        assert samples.dtype == np.float32, uid
        assert samples.ndim == 1, uid
        assert samples.size > 0, uid
        assert np.isfinite(samples).all(), uid


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset_source", ["D1", "D2", "D3", "D4"])
def test_real_records_resample_to_the_target_rate(
    sample_records: Any, dataset_source: str
) -> None:
    for uid, path in sample_records(dataset_source, 5):
        samples, fs_native = pio.load_wav(path)
        out = pio.resample_to(samples, fs_native, 2000)

        assert out.dtype == np.float32, uid
        assert np.isfinite(out).all(), uid
        expected = round(samples.size * 2000 / fs_native)
        assert abs(out.size - expected) <= 1, uid
        # Duration survives the rate change to within one sample.
        assert out.size / 2000 == pytest.approx(samples.size / fs_native, abs=1e-3), uid


@pytest.mark.needs_data
def test_duration_extremes_survive_resampling(master_frame: Any, project_root: Path) -> None:
    """The 0.76 s and 122 s records, by name, not by synthetic proxy."""
    shortest = master_frame.loc[master_frame["duration_sec"].idxmin()]
    longest = master_frame.loc[master_frame["duration_sec"].idxmax()]

    for row in (shortest, longest):
        samples, fs_native = pio.load_wav(project_root / str(row["file_path"]))
        out = pio.resample_to(samples, fs_native, 2000)
        assert out.size > 0, row["record_uid"]
        assert np.isfinite(out).all(), row["record_uid"]
        assert out.size / 2000 == pytest.approx(float(row["duration_sec"]), abs=1e-2)
