"""The 22 frequency-domain features (Phase 33, gate T33.7).

The gate asks for two things: exactly 22 finite values, and six band powers that
sum to no more than the total power. Both are necessary and neither is
sufficient -- an extractor that returned the same six numbers for every record
would pass both. So the known-answer layer pins each measure to a signal whose
spectrum is known in closed form: a pure tone has its centroid at the tone, its
dominant frequency at the tone, and essentially all of its power in the one band
that contains it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import registry as reg
from src.feature_extraction.frequency import (
    BANDS_HZ,
    FRAMED_NAMES,
    FrequencyExtractor,
    band_power,
    spectral_entropy,
    spectral_flux,
    welch_psd,
)

FS = 2000
EXPECTED_COUNT = 22
BAND_NAMES = tuple(
    "freq_band_power_" + str(low) + "_" + str(high) for low, high in BANDS_HZ
)


@pytest.fixture(scope="module")
def extractor() -> FrequencyExtractor:
    return FrequencyExtractor()


def _tone(freq_hz: float, seconds: float = 5.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# family shape
# ---------------------------------------------------------------------------


def test_family_is_registered_and_holds_22_names(extractor: FrequencyExtractor):
    assert extractor.family == "frequency"
    names = extractor.feature_names()
    assert len(names) == EXPECTED_COUNT
    assert names == reg.feature_names("frequency")
    assert reg.get_extractor("frequency").family == "frequency"


def test_composition_matches_the_task_breakdown():
    """12 framed + 4 global + 6 band powers = 22."""
    names = list(reg.feature_names("frequency"))
    globals_ = [
        "freq_spectral_entropy",
        "freq_dominant",
        "freq_peak_power",
        "freq_total_power",
    ]
    assert len(FRAMED_NAMES) == 12
    assert len(BAND_NAMES) == 6
    assert names == list(FRAMED_NAMES) + globals_ + list(BAND_NAMES)
    assert len(names) == EXPECTED_COUNT


def test_synthetic_signal_yields_22_finite_values(
    extractor: FrequencyExtractor, synthetic_signal: np.ndarray
):
    result = extractor.extract(synthetic_signal, FS, record_uid="synthetic")
    assert result.failed is False, result.error
    assert len(result.values) == EXPECTED_COUNT
    assert np.isfinite(result.vector).all()


# ---------------------------------------------------------------------------
# T33.6 -- the Welch parameters and band edges are fixed in config
# ---------------------------------------------------------------------------


def test_welch_parameters_come_from_config(extractor: FrequencyExtractor,
                                           features_config: Any):
    settings = extractor._welch_settings()
    assert settings == {
        "nperseg": features_config.get("families.frequency.welch.nperseg"),
        "noverlap": features_config.get("families.frequency.welch.noverlap"),
        "window": features_config.get("families.frequency.welch.window"),
        "detrend": features_config.get("families.frequency.welch.detrend"),
        "scaling": features_config.get("families.frequency.welch.scaling"),
    }
    assert settings["nperseg"] == 512
    assert settings["noverlap"] == 256


def test_band_edges_in_config_match_the_registry_column_names(
    extractor: FrequencyExtractor,
):
    assert extractor.bands() == BANDS_HZ
    for (low, high), name in zip(BANDS_HZ, BAND_NAMES, strict=True):
        assert name in reg.feature_names("frequency")
        assert str(low) in name and str(high) in name


def test_a_config_band_edit_is_rejected_rather_than_silently_relabelling():
    """The column name encodes the band; changing one without the other is a bug.

    A relabelled band is invisible in the numbers -- ``freq_band_power_100_150``
    would simply hold a different band's power -- so it has to fail loudly.
    """

    class _Cfg:
        def get(self, key, default=None):
            if key == "families.frequency.bands_hz":
                return [[20, 60], [60, 100], [100, 150], [150, 250], [250, 350], [350, 400]]
            return default

    with pytest.raises(ValueError, match="bands_hz"):
        FrequencyExtractor(_Cfg()).bands()


def test_the_passband_stays_below_nyquist(signal_config: Any):
    """400 Hz against a 1,000 Hz Nyquist at the 2 kHz target rate."""
    nyquist = signal_config.get("resample.target_fs") / 2
    assert max(high for _low, high in BANDS_HZ) < nyquist


# ---------------------------------------------------------------------------
# known answers
# ---------------------------------------------------------------------------


def test_welch_finds_a_pure_tone(extractor: FrequencyExtractor):
    freqs, psd = welch_psd(_tone(120.0), FS)
    assert freqs[int(np.argmax(psd))] == pytest.approx(120.0, abs=4.0)


def test_a_tone_puts_its_power_in_the_band_that_contains_it(
    extractor: FrequencyExtractor,
):
    """120 Hz belongs to 100-150; that band must dominate the other five."""
    values = extractor.extract(_tone(120.0), FS).values
    assert values["freq_dominant"] == pytest.approx(120.0, abs=4.0)
    assert values["freq_band_power_100_150"] > 0.9
    for name in BAND_NAMES:
        if name != "freq_band_power_100_150":
            assert values[name] < 0.05, name


def test_spectral_centroid_of_a_pure_tone_is_the_tone(extractor: FrequencyExtractor):
    values = extractor.extract(_tone(200.0), FS).values
    assert values["freq_centroid_mean"] == pytest.approx(200.0, rel=0.05)
    # A single spectral line has near-zero spread and near-zero flatness.
    assert values["freq_bandwidth_mean"] < 60.0
    assert values["freq_flatness_mean"] < 0.01


def test_band_power_of_a_flat_psd_is_proportional_to_bandwidth():
    freqs = np.linspace(0.0, 1000.0, 2001)
    psd = np.ones_like(freqs)
    assert band_power(freqs, psd, 20, 50) == pytest.approx(30.0)
    assert band_power(freqs, psd, 150, 250) == pytest.approx(100.0)
    assert band_power(freqs, psd, 400, 20) == 0.0


def test_band_powers_tile_the_passband_without_overlap():
    """The six sub-integrals must add up to the single 20-400 Hz integral.

    This is what the interpolated edges buy: snapping each edge to the nearest
    bin would leave a residue of up to one bin per boundary.
    """
    rng = np.random.default_rng(42)
    freqs = np.linspace(0.0, 1000.0, 257)
    psd = rng.random(freqs.size) + 0.1

    parts = sum(band_power(freqs, psd, low, high) for low, high in BANDS_HZ)
    whole = band_power(freqs, psd, 20, 400)
    assert parts == pytest.approx(whole, rel=1e-12)


def test_spectral_entropy_is_one_for_flat_and_zero_for_a_single_line():
    flat = np.ones(256)
    assert spectral_entropy(flat) == pytest.approx(1.0)

    line = np.zeros(256)
    line[42] = 1.0
    assert spectral_entropy(line) == pytest.approx(0.0)


def test_spectral_flux_is_zero_for_an_unchanging_spectrogram():
    steady = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, 10))
    mean, std = spectral_flux(steady)
    assert mean == pytest.approx(0.0)
    assert std == pytest.approx(0.0)


def test_spectral_flux_needs_two_frames_and_says_so():
    flags: list[str] = []
    mean, std = spectral_flux(np.array([[1.0], [2.0]]), flags)
    assert np.isnan(mean) and np.isnan(std)
    assert "flux_insufficient_frames" in flags


def test_noise_has_higher_flatness_and_entropy_than_a_tone(
    extractor: FrequencyExtractor,
):
    noise = np.random.default_rng(42).standard_normal(FS * 5)
    tone = _tone(150.0)

    noisy = extractor.extract(noise, FS).values
    tonal = extractor.extract(tone, FS).values

    assert noisy["freq_flatness_mean"] > tonal["freq_flatness_mean"]
    assert noisy["freq_spectral_entropy"] > tonal["freq_spectral_entropy"]


def test_extraction_is_deterministic(extractor: FrequencyExtractor,
                                     synthetic_signal: np.ndarray):
    first = extractor.extract(synthetic_signal, FS).vector
    second = extractor.extract(synthetic_signal, FS).vector
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


def test_edge_case_signals_never_raise(extractor: FrequencyExtractor,
                                       edge_case_signals: dict[str, np.ndarray]):
    """Any NaN must be explained by a flag; see the same rule in the time family."""
    explains = dict.fromkeys(FRAMED_NAMES, "stft_")
    explains["freq_flux_mean"] = "flux_"
    explains["freq_flux_std"] = "flux_"

    for label, signal in edge_case_signals.items():
        result = extractor.extract(signal, FS, record_uid=label)
        assert len(result.values) == EXPECTED_COUNT, label

        if result.failed:
            assert result.n_missing == EXPECTED_COUNT, label
            continue

        for name, value in result.values.items():
            if np.isfinite(value):
                continue
            prefix = explains.get(name)
            assert prefix is not None and any(
                flag.startswith(prefix) or flag.startswith("stft_")
                for flag in result.flags
            ), label + ": " + name + " is NaN with no explaining flag; flags=" + str(
                result.flags
            )


def test_a_silent_signal_reports_zero_power_rather_than_dividing_by_it(
    extractor: FrequencyExtractor,
):
    result = extractor.extract(np.zeros(4000), FS, record_uid="silence")
    assert result.failed is False
    assert np.isfinite(result.vector).all()
    assert "psd_zero_power" in result.flags
    assert result.values["freq_total_power"] == 0.0
    assert all(result.values[name] == 0.0 for name in BAND_NAMES)


# ---------------------------------------------------------------------------
# real records -- the gate (T33.7)
# ---------------------------------------------------------------------------


def _preprocessed(path: Path, uid: str) -> np.ndarray:
    from src.preprocessing.pipeline import preprocess

    return preprocess(path, record_uid=uid).signal


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_yield_22_finite_values(
    extractor: FrequencyExtractor, sample_records: Any, dataset: str
):
    for uid, path in sample_records(dataset, 3):
        result = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid)
        assert result.failed is False, dataset + " " + uid + ": " + str(result.error)
        assert len(result.values) == EXPECTED_COUNT
        bad = [name for name, value in result.values.items() if not np.isfinite(value)]
        assert bad == [], dataset + " " + uid + " non-finite: " + ", ".join(bad)


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_band_powers_never_exceed_the_total_power(
    extractor: FrequencyExtractor, sample_records: Any, dataset: str
):
    """The T33.7 assertion, in both the relative and the absolute form.

    Relative: the six fractions sum to at most 1. Absolute: the six integrals
    sum to at most the 0-Nyquist integral. The second is the one that would
    catch a band-edge or normalization slip that happened to keep the fractions
    below 1 by accident.
    """
    for uid, path in sample_records(dataset, 3):
        signal = _preprocessed(path, uid)
        values = extractor.extract(signal, FS, record_uid=uid).values

        fractions = [values[name] for name in BAND_NAMES]
        assert all(value >= 0.0 for value in fractions), uid
        assert sum(fractions) <= 1.0 + 1e-9, uid + " band fractions sum to " + str(
            sum(fractions)
        )

        freqs, psd = welch_psd(signal, FS, **extractor._welch_settings())
        absolute = sum(band_power(freqs, psd, low, high) for low, high in BANDS_HZ)
        total = float(np.trapezoid(psd, freqs))
        assert absolute <= total + 1e-12, uid
        assert values["freq_total_power"] == pytest.approx(total, rel=1e-9), uid


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_land_in_physically_plausible_ranges(
    extractor: FrequencyExtractor, sample_records: Any, dataset: str
):
    """After a 20-400 Hz bandpass, the spectrum must actually live there.

    The centroid of a heart sound that passed this filter cannot sit at 900 Hz.
    If it does, the extractor is reading something other than the preprocessed
    signal -- which a finiteness check would happily pass.
    """
    nyquist = FS / 2
    for uid, path in sample_records(dataset, 3):
        values = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid).values

        assert 0.0 < values["freq_centroid_mean"] < nyquist, uid
        assert 0.0 < values["freq_dominant"] < nyquist, uid
        assert 0.0 <= values["freq_flatness_mean"] <= 1.0, uid
        assert 0.0 <= values["freq_spectral_entropy"] <= 1.0, uid
        assert values["freq_total_power"] > 0.0, uid
        # The bandpass keeps 20-400 Hz; the six bands must hold most of the power.
        assert sum(values[name] for name in BAND_NAMES) > 0.5, uid


@pytest.mark.needs_data
def test_duration_extremes_extract_cleanly(
    extractor: FrequencyExtractor, master_frame: Any, project_root: Path
):
    """0.76 s is 1,520 samples -- three STFT frames and three Welch segments."""
    shortest = master_frame.loc[master_frame["duration_sec"].idxmin()]
    longest = master_frame.loc[master_frame["duration_sec"].idxmax()]

    for row in (shortest, longest):
        uid = str(row["record_uid"])
        signal = _preprocessed(project_root / str(row["file_path"]), uid)
        result = extractor.extract(signal, FS, record_uid=uid)

        assert result.failed is False, uid + ": " + str(result.error)
        assert np.isfinite(result.vector).all(), uid
        assert sum(result.values[name] for name in BAND_NAMES) <= 1.0 + 1e-9, uid
        # 1,520 samples clears both the 512-sample STFT and Welch windows, so
        # neither short-signal guard should have fired on the real extremes.
        assert not any(flag.startswith("stft_") for flag in result.flags), uid
        assert not any(flag.startswith("welch_") for flag in result.flags), uid
