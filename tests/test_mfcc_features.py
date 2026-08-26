"""The 39 MFCC features (Phase 34, gate T34.7).

The gate asks for three things: exactly 39 values, ``fmax`` below Nyquist, and a
0.76 s record that degrades rather than raises. Each has a matching test below,
against the real recording rather than a synthetic stand-in for the third.

The known-answer layer here is unusually sharp, and worth reading before
changing anything in ``mfcc.py``. Scaling a signal by a factor ``a`` multiplies
its power by ``a^2``, which shifts every log-mel band by exactly ``20*log10(a)``
dB; under an orthonormal DCT a uniform shift ``d`` across ``N`` bands moves c0 by
exactly ``sqrt(N)*d`` and leaves every other coefficient untouched. That identity
holds only because the dB reference is absolute. If someone reinstates librosa's
default per-record ``ref``/``top_db``, this test fails immediately -- which is
the point, because nothing else about the numbers would look wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import registry as reg
from src.feature_extraction.mfcc import (
    MIN_DELTA_WIDTH,
    MFCCExtractor,
    delta_matrix,
    mfcc_matrix,
    resolve_delta_width,
)

FS = 2000
EXPECTED_COUNT = 39
N_MFCC = 13


@pytest.fixture(scope="module")
def extractor() -> MFCCExtractor:
    return MFCCExtractor()


def _tone(freq_hz: float, seconds: float = 5.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# family shape (T34.3-T34.5)
# ---------------------------------------------------------------------------


def test_family_is_registered_and_holds_39_names(extractor: MFCCExtractor):
    assert extractor.family == "mfcc"
    names = extractor.feature_names()
    assert len(names) == EXPECTED_COUNT
    assert names == reg.feature_names("mfcc")
    assert reg.get_extractor("mfcc").family == "mfcc"


def test_composition_is_13_means_then_13_stds_then_13_delta_means():
    """Blocked by statistic, not interleaved by coefficient (T34.3-T34.5)."""
    names = list(reg.feature_names("mfcc"))
    expected = (
        ["mfcc_" + str(k).zfill(2) + "_mean" for k in range(1, N_MFCC + 1)]
        + ["mfcc_" + str(k).zfill(2) + "_std" for k in range(1, N_MFCC + 1)]
        + ["mfcc_" + str(k).zfill(2) + "_delta_mean" for k in range(1, N_MFCC + 1)]
    )
    assert names == expected
    assert len(names) == EXPECTED_COUNT == 3 * N_MFCC


def test_synthetic_signal_yields_39_finite_values(
    extractor: MFCCExtractor, synthetic_signal: np.ndarray
):
    result = extractor.extract(synthetic_signal, FS, record_uid="synthetic")
    assert result.failed is False, result.error
    assert len(result.values) == EXPECTED_COUNT
    assert np.isfinite(result.vector).all()
    assert result.n_missing == 0


# ---------------------------------------------------------------------------
# T34.1 -- the filterbank, and Nyquist
# ---------------------------------------------------------------------------


def test_settings_come_from_config(extractor: MFCCExtractor, features_config: Any,
                                   signal_config: Any):
    settings = extractor.settings()
    assert settings.n_mfcc == features_config.get("families.mfcc.n_mfcc")
    assert settings.n_mels == features_config.get("families.mfcc.n_mels")
    assert settings.fmin == features_config.get("families.mfcc.fmin")
    assert settings.fmax == features_config.get("families.mfcc.fmax")
    assert settings.n_fft == features_config.get("families.mfcc.n_fft")
    assert settings.hop_length == features_config.get("families.mfcc.hop_length")
    assert settings.htk == features_config.get("families.mfcc.htk")
    assert settings.delta_width == features_config.get("families.mfcc.delta_width")
    assert settings.window == signal_config.get("framing.window")
    assert settings.center == signal_config.get("framing.center")


def test_fmax_is_below_nyquist(extractor: MFCCExtractor, signal_config: Any):
    """The T34.7 assertion, at the project's target rate."""
    target_fs = int(signal_config.get("resample.target_fs"))
    settings = extractor.settings()
    assert settings.fmax < target_fs / 2
    settings.check_nyquist(target_fs)  # must not raise


def test_a_filterbank_reaching_past_nyquist_is_rejected(extractor: MFCCExtractor):
    """Above Nyquist the top filters address bins that do not exist.

    librosa clamps them silently, and the affected coefficients become a
    function of the clamp rather than of the recording -- so this must fail
    rather than produce plausible numbers.
    """
    settings = extractor.settings()
    with pytest.raises(ValueError, match="Nyquist"):
        settings.check_nyquist(600)  # fmax 400 against a 300 Hz Nyquist


def test_the_mel_bank_has_no_empty_filters_at_the_configured_settings(
    extractor: MFCCExtractor,
):
    """An empty filter takes log(0) and injects a floor constant into every record.

    At n_fft=512 and 2 kHz the FFT resolution is 3.906 Hz and 40 bands over
    20-400 Hz each cover 4-5 bins. This is the fact the module docstring rests
    on, checked rather than asserted in prose.
    """
    import librosa

    settings = extractor.settings()
    bank = librosa.filters.mel(
        sr=FS,
        n_fft=settings.n_fft,
        n_mels=settings.n_mels,
        fmin=settings.fmin,
        fmax=settings.fmax,
        htk=settings.htk,
    )
    assert bank.shape[0] == settings.n_mels
    assert np.count_nonzero(bank.sum(axis=1) == 0) == 0
    assert (bank > 0).sum(axis=1).min() >= 2


def test_the_bank_ignores_energy_outside_20_to_400_hz(extractor: MFCCExtractor):
    """A 700 Hz tone is outside the passband; its c0 must sit near the log floor."""
    settings = extractor.settings()
    in_band = float(np.mean(mfcc_matrix(_tone(200.0), FS, settings)[0]))
    out_of_band = float(np.mean(mfcc_matrix(_tone(700.0), FS, settings)[0]))
    assert out_of_band < in_band - 100.0


# ---------------------------------------------------------------------------
# known answers
# ---------------------------------------------------------------------------


def test_amplitude_scaling_moves_only_c0_and_by_exactly_sqrt_n_mels_times_the_db_shift(
    extractor: MFCCExtractor,
):
    """The identity that pins the dB reference as absolute; see the module docstring."""
    settings = extractor.settings()
    factor = 2.0
    quiet = mfcc_matrix(_tone(150.0), FS, settings)
    loud = mfcc_matrix(factor * _tone(150.0), FS, settings)

    expected = np.sqrt(settings.n_mels) * 20.0 * np.log10(factor)
    assert float(np.mean(loud[0] - quiet[0])) == pytest.approx(expected, rel=1e-9)

    higher = np.abs(np.mean(loud[1:] - quiet[1:], axis=1))
    assert float(higher.max()) < 1e-9


def test_a_steady_tone_has_near_zero_delta_in_its_interior(extractor: MFCCExtractor):
    """Nothing changes frame to frame, so the first derivative is ~0 -- inside.

    The *whole-sequence* mean is not ~0, and that is correct rather than a bug:
    ``framing.center`` is true, so the first and last frames are half zero-padded
    and band energy ramps up at the start and down at the end. On a 6 s tone that
    edge ramp moves c0's delta mean to about -1.15 while the interior sits at
    0.001. The physical claim is about the interior, so that is what is asserted;
    the edge behaviour is a property of centred framing, not of the extractor.
    """
    settings = extractor.settings()
    coefficients = mfcc_matrix(_tone(150.0, seconds=6.0), FS, settings)
    deltas = delta_matrix(coefficients, settings.delta_width)

    assert coefficients.shape[1] > 40, "need enough frames to have an interior"
    interior = np.mean(deltas[:, 10:-10], axis=1)
    assert np.abs(interior).max() < 1e-2


def test_delta_of_a_constant_sequence_is_exactly_zero():
    constant = np.tile(np.arange(13.0).reshape(13, 1), (1, 20))
    assert np.abs(delta_matrix(constant, 9)).max() < 1e-12


def test_two_different_tones_produce_different_coefficients(extractor: MFCCExtractor):
    """Guards against an extractor that returns the same 39 numbers for everything."""
    low = extractor.extract(_tone(80.0), FS).vector
    high = extractor.extract(_tone(350.0), FS).vector
    assert not np.allclose(low, high)


def test_extraction_is_deterministic(extractor: MFCCExtractor,
                                     synthetic_signal: np.ndarray):
    first = extractor.extract(synthetic_signal, FS).vector
    second = extractor.extract(synthetic_signal, FS).vector
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# T34.6 -- short-signal degradation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_frames", "expected"),
    [(12, 9), (9, 9), (8, 7), (6, 5), (5, 5), (4, 3), (3, 3), (2, 0), (1, 0)],
)
def test_delta_width_shrinks_to_the_largest_odd_window_that_fits(
    n_frames: int, expected: int
):
    assert resolve_delta_width(9, n_frames) == expected


def test_shrinking_the_delta_width_is_flagged():
    flags: list[str] = []
    assert resolve_delta_width(9, 6, flags) == 5
    assert "mfcc_delta_width_5" in flags

    unflagged: list[str] = []
    assert resolve_delta_width(9, 20, unflagged) == 9
    assert unflagged == []


def test_fewer_than_three_frames_pads_rather_than_raising():
    """The configured ``shrink_delta_width_then_pad`` fallback (T34.6).

    Edge replication means the delta over the padded region is exactly zero,
    which is the honest answer for frames that do not exist -- not a fabricated
    value, and the flag records that it happened.
    """
    flags: list[str] = []
    coefficients = np.random.default_rng(42).standard_normal((13, 2))
    deltas = delta_matrix(coefficients, 9, flags)

    assert deltas.shape == coefficients.shape
    assert np.isfinite(deltas).all()
    assert any(flag.startswith("mfcc_delta_padded_") for flag in flags)
    assert MIN_DELTA_WIDTH == 3


def test_a_signal_shorter_than_n_fft_reduces_the_window_and_the_bank():
    """A coarser spectrum would otherwise leave empty mel filters behind."""
    flags: list[str] = []
    settings = MFCCExtractor().settings()
    coefficients = mfcc_matrix(_tone(150.0, seconds=0.1), FS, settings, flags)

    assert coefficients.shape[0] == N_MFCC
    assert coefficients.shape[1] > 0
    assert np.isfinite(coefficients).all()
    assert any(flag.startswith("mfcc_n_fft_") for flag in flags)
    assert any(flag.startswith("mfcc_n_mels_") for flag in flags)


def test_edge_case_signals_never_raise(extractor: MFCCExtractor,
                                       edge_case_signals: dict[str, np.ndarray]):
    """Any NaN must be explained by a flag; the same rule as the other families."""
    for label, signal in edge_case_signals.items():
        result = extractor.extract(signal, FS, record_uid=label)
        assert len(result.values) == EXPECTED_COUNT, label

        if result.failed:
            assert result.n_missing == EXPECTED_COUNT, label
            continue

        if result.n_missing:
            assert any(
                flag.startswith("mfcc_signal_too_short") for flag in result.flags
            ), label + ": unexplained NaN; flags=" + str(result.flags)


def test_a_single_sample_reports_the_whole_family_as_an_explained_gap(
    extractor: MFCCExtractor,
):
    result = extractor.extract(np.array([0.5]), FS, record_uid="single_sample")
    assert result.failed is False
    assert result.n_missing == EXPECTED_COUNT
    assert "mfcc_signal_too_short" in result.flags


# ---------------------------------------------------------------------------
# real records -- the gate (T34.7)
# ---------------------------------------------------------------------------


def _preprocessed(path: Path, uid: str) -> np.ndarray:
    from src.preprocessing.pipeline import preprocess

    return preprocess(path, record_uid=uid).signal


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_yield_39_finite_values(
    extractor: MFCCExtractor, sample_records: Any, dataset: str
):
    for uid, path in sample_records(dataset, 3):
        result = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid)
        assert result.failed is False, dataset + " " + uid + ": " + str(result.error)
        assert len(result.values) == EXPECTED_COUNT
        bad = [name for name, value in result.values.items() if not np.isfinite(value)]
        assert bad == [], dataset + " " + uid + " non-finite: " + ", ".join(bad)


@pytest.mark.needs_data
def test_the_shortest_real_record_degrades_gracefully(
    extractor: MFCCExtractor, master_frame: Any, project_root: Path
):
    """The 0.76 s PASCAL B recording, verbatim from the gate wording (T34.7).

    1,520 samples at 2 kHz is six analysis frames, below the configured delta
    width of nine. It must shrink the window and say so, not raise and not
    silently return a delta computed over a window it never had.
    """
    shortest = master_frame.loc[master_frame["duration_sec"].idxmin()]
    uid = str(shortest["record_uid"])
    assert float(shortest["duration_sec"]) < 1.0

    signal = _preprocessed(project_root / str(shortest["file_path"]), uid)
    result = extractor.extract(signal, FS, record_uid=uid)

    assert result.failed is False, uid + ": " + str(result.error)
    assert len(result.values) == EXPECTED_COUNT
    assert np.isfinite(result.vector).all(), uid

    settings = extractor.settings()
    n_frames = 1 + signal.size // settings.hop_length
    assert n_frames < settings.delta_width
    assert "mfcc_delta_width_" + str(n_frames if n_frames % 2 else n_frames - 1) in (
        result.flags
    ), uid + " flags=" + str(result.flags)
    # 1,520 samples still clears the 512-sample analysis window, so the
    # filterbank itself was never reduced.
    assert not any(flag.startswith("mfcc_n_fft_") for flag in result.flags), uid
    assert not any(flag.startswith("mfcc_n_mels_") for flag in result.flags), uid


@pytest.mark.needs_data
def test_the_longest_real_record_extracts_without_degradation(
    extractor: MFCCExtractor, master_frame: Any, project_root: Path
):
    """122 s is ~954 frames -- no guard should fire on it."""
    longest = master_frame.loc[master_frame["duration_sec"].idxmax()]
    uid = str(longest["record_uid"])

    signal = _preprocessed(project_root / str(longest["file_path"]), uid)
    result = extractor.extract(signal, FS, record_uid=uid)

    assert result.failed is False, uid + ": " + str(result.error)
    assert np.isfinite(result.vector).all(), uid
    assert result.flags == (), uid + " flags=" + str(result.flags)


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_land_in_plausible_ranges(
    extractor: MFCCExtractor, sample_records: Any, dataset: str
):
    """c0 carries band energy on a z-normalized, bandpassed signal.

    A record whose c0 sat at the log floor would mean the mel bank saw no energy
    in 20-400 Hz at all -- impossible after the preprocessing this project runs,
    and a sign the extractor is reading the wrong signal. Finiteness alone would
    not catch it.

    The check is expressed as the **mean mel-band level in dB** (``c0`` divided
    by ``sqrt(n_mels)``, the orthonormal DCT's c0 gain) rather than as a raw c0
    bound, because the readable quantity is the band level. Note that it is
    routinely *positive*: librosa's STFT is not normalized by window length, so a
    unit-variance signal produces per-band power well above 1. Measured across
    40 real records the level spans about -8 dB to +16 dB, against a log floor of
    -100 dB.
    """
    settings = extractor.settings()
    c0_gain = float(np.sqrt(settings.n_mels))
    floor_db = 10.0 * np.log10(1e-10)

    for uid, path in sample_records(dataset, 3):
        values = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid).values
        band_level_db = values["mfcc_01_mean"] / c0_gain
        assert band_level_db > floor_db + 40.0, uid + " level=" + str(band_level_db)
        assert band_level_db < 60.0, uid + " level=" + str(band_level_db)
        assert values["mfcc_01_std"] >= 0.0, uid
        for k in range(1, N_MFCC + 1):
            key = "mfcc_" + str(k).zfill(2)
            assert values[key + "_std"] >= 0.0, uid + " " + key
            assert abs(values[key + "_delta_mean"]) < 50.0, uid + " " + key
