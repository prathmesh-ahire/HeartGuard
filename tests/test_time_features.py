"""The 24 time-domain features (Phase 32, gate T32.7).

Three layers, because each catches something the others cannot:

**Known answers.** Entropy of a uniform histogram, Hjorth mobility of a pure
sine, sample entropy of a constant signal, autocorrelation lag of an exactly
periodic wave. A shape test would pass on an extractor that returns 24 confident
wrong numbers; these do not.

**Real records from all four datasets.** The gate wording. A synthetic 5-second
signal at 2 kHz is nothing like a 0.76 s PASCAL B recording or a 122 s PhysioNet
one, and both exist in this corpus.

**Degenerate input.** Silence, a constant, a single sample. These must come back
flagged and finite, or explicitly failed -- never as an exception that kills a
seven-thousand-record batch run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import registry as reg
from src.feature_extraction.time_domain import (
    TimeDomainExtractor,
    autocorrelation_peak,
    hjorth_parameters,
    sample_entropy,
    shannon_entropy,
    zero_crossing_stats,
)

FS = 2000
EXPECTED_COUNT = 24


@pytest.fixture(scope="module")
def extractor() -> TimeDomainExtractor:
    return TimeDomainExtractor()


# ---------------------------------------------------------------------------
# family shape (T32.6)
# ---------------------------------------------------------------------------


def test_family_is_registered_and_holds_24_names(extractor: TimeDomainExtractor):
    assert extractor.family == "time"
    names = extractor.feature_names()
    assert len(names) == EXPECTED_COUNT
    assert names == reg.feature_names("time")
    assert reg.get_extractor("time").family == "time"


def test_composition_matches_the_task_breakdown():
    """8 basic + 2 shape + 4 energy + 2 zcr + 5 complexity + 3 = 24."""
    names = reg.feature_names("time")
    groups = {
        "basic": ["time_mean", "time_std", "time_var", "time_min", "time_max",
                  "time_range", "time_median", "time_iqr"],
        "shape": ["time_skewness", "time_kurtosis"],
        "energy": ["time_energy", "time_rms", "time_peak_to_peak", "time_crest_factor"],
        "zcr": ["time_zcr_mean", "time_zcr_std"],
        "complexity": ["time_shannon_entropy", "time_sample_entropy",
                       "time_hjorth_activity", "time_hjorth_mobility",
                       "time_hjorth_complexity"],
        "tail": ["time_autocorr_peak_value", "time_autocorr_peak_lag", "time_duration"],
    }
    assert sum(len(group) for group in groups.values()) == EXPECTED_COUNT
    assert [name for group in groups.values() for name in group] == list(names)


def test_synthetic_signal_yields_24_finite_values(
    extractor: TimeDomainExtractor, synthetic_signal: np.ndarray
):
    result = extractor.extract(synthetic_signal, FS, record_uid="synthetic")
    assert result.failed is False, result.error
    assert len(result.values) == EXPECTED_COUNT
    assert np.isfinite(result.vector).all()
    assert result.n_missing == 0


# ---------------------------------------------------------------------------
# known answers
# ---------------------------------------------------------------------------


def test_shannon_entropy_of_a_uniform_histogram_is_log2_bins():
    """A ramp fills all 64 bins equally: entropy is exactly 6 bits."""
    ramp = np.linspace(0.0, 1.0, 64 * 500, endpoint=False)
    assert shannon_entropy(ramp, bins=64) == pytest.approx(6.0, abs=1e-9)
    assert shannon_entropy(np.full(1000, 0.3), bins=64) == pytest.approx(0.0)


def test_sample_entropy_of_a_constant_signal_is_zero():
    """Every template matches every other, so A/B is 1 and -ln(1) is 0."""
    assert sample_entropy(np.full(2000, 0.7)) == pytest.approx(0.0, abs=1e-12)


def test_sample_entropy_orders_a_sine_below_white_noise():
    """The only property sample entropy really promises: regular < random."""
    t = np.arange(4000) / FS
    sine = np.sin(2 * np.pi * 50 * t)
    noise = np.random.default_rng(42).standard_normal(4000)
    assert sample_entropy(sine) < sample_entropy(noise)


def test_sample_entropy_windows_long_signals_and_flags_it():
    flags: list[str] = []
    sample_entropy(np.random.default_rng(0).standard_normal(30000), max_samples=5000,
                   flags=flags)
    assert "sampen_window_5000" in flags


def test_hjorth_mobility_of_a_sine_matches_its_angular_frequency():
    """For x = sin(wt) sampled at fs, mobility -> 2*sin(w/(2*fs)) ~ w/fs.

    The discrete first difference is not the derivative, so the limit is the
    exact discrete form rather than w/fs; both are checked so a sign or
    normalization slip cannot hide.
    """
    freq = 100.0
    t = np.arange(20000) / FS
    sine = np.sin(2 * np.pi * freq * t)
    omega = 2 * np.pi * freq
    activity, mobility, complexity = hjorth_parameters(sine)

    assert activity == pytest.approx(0.5, abs=1e-3)
    assert mobility == pytest.approx(2 * np.sin(omega / (2 * FS)), rel=1e-3)
    # A pure sine's derivative is another sine of the same frequency, so its
    # mobility equals the signal's and complexity is 1.
    assert complexity == pytest.approx(1.0, rel=1e-3)


def test_hjorth_of_a_constant_signal_is_all_zero():
    assert hjorth_parameters(np.full(1000, 2.0)) == (0.0, 0.0, 0.0)


def test_autocorrelation_recovers_a_known_period():
    """A 1.25 Hz square-ish pulse train: the peak lag must be 0.8 s."""
    period_sec = 0.8
    n = FS * 10
    signal = np.zeros(n)
    for start in range(0, n, int(period_sec * FS)):
        signal[start : start + 40] = 1.0

    value, lag = autocorrelation_peak(signal, FS)
    assert lag == pytest.approx(period_sec, abs=1.0 / FS)
    assert 0.5 < value <= 1.0


def test_autocorrelation_of_a_silent_signal_is_zero_and_flagged():
    flags: list[str] = []
    value, lag = autocorrelation_peak(np.zeros(4000), FS, flags=flags)
    assert (value, lag) == (0.0, 0.0)
    assert "autocorr_zero_energy" in flags


def test_synthetic_autocorrelation_lag_matches_the_generated_heart_rate(
    extractor: TimeDomainExtractor, synthetic_pcg_factory
):
    """72 bpm is one beat every 0.833 s; the extractor must find that lag."""
    pcg = synthetic_pcg_factory(duration_sec=12.0, fs=FS, heart_rate_bpm=72.0)
    result = extractor.extract(pcg.signal, FS)
    expected = 60.0 / 72.0
    assert result.values["time_autocorr_peak_lag"] == pytest.approx(expected, abs=0.02)


def test_energy_and_duration_are_arithmetically_exact(extractor: TimeDomainExtractor):
    signal = np.array([1.0, -1.0] * 1000)
    result = extractor.extract(signal, FS)
    assert result.values["time_energy"] == pytest.approx(2000.0)
    assert result.values["time_rms"] == pytest.approx(1.0)
    assert result.values["time_duration"] == pytest.approx(1.0)
    assert result.values["time_peak_to_peak"] == pytest.approx(2.0)
    assert result.values["time_crest_factor"] == pytest.approx(1.0)


def test_zcr_approaches_one_for_a_sample_alternating_signal():
    """Alternating every sample is the maximum possible zero-crossing rate.

    It reaches 1.0 only in the limit, not exactly, and the gap is the framing:
    ``framing.center`` is true, so librosa zero-pads half a frame at each end and
    those two edge frames alternate less than the interior ones. The shortfall is
    therefore ~1/n_frames and shrinks with duration -- 0.924 over 1 s, 0.996 over
    50 s -- which is what this test pins. A value that did NOT move with duration
    would mean the framing is not what the config says.
    """
    short_mean, short_std = zero_crossing_stats(np.array([1.0, -1.0] * 1000), 512, 256, True)
    long_mean, long_std = zero_crossing_stats(np.array([1.0, -1.0] * 50000), 512, 256, True)

    assert long_mean == pytest.approx(1.0, abs=0.005)
    assert long_mean > short_mean
    assert long_std < short_std


def test_extraction_is_deterministic(extractor: TimeDomainExtractor,
                                     synthetic_signal: np.ndarray):
    """Rule 5: two runs of the same command produce identical numbers."""
    first = extractor.extract(synthetic_signal, FS).vector
    second = extractor.extract(synthetic_signal, FS).vector
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


def test_edge_case_signals_never_raise(extractor: TimeDomainExtractor,
                                       edge_case_signals: dict[str, np.ndarray]):
    """Silence, a constant, a single sample: a value or an explained gap, never both.

    The NaN policy is that a non-finite value must be *accounted for* -- either
    the whole family failed, or a flag names the measure that could not be
    computed. An unexplained NaN in a feature matrix is indistinguishable from a
    bug, which is exactly what FE-04 exists to rule out.
    """
    explains = {"time_sample_entropy": "sampen"}

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
                flag.startswith(prefix) for flag in result.flags
            ), label + ": " + name + " is NaN with no explaining flag; flags=" + str(
                result.flags
            )


def test_a_single_sample_reports_sample_entropy_as_an_explained_gap(
    extractor: TimeDomainExtractor,
):
    """One sample is below the m+2 template floor; 23 of the 24 still compute."""
    result = extractor.extract(np.array([0.5]), FS, record_uid="single_sample")

    assert result.failed is False
    assert result.n_missing == 1
    assert np.isnan(result.values["time_sample_entropy"])
    assert "sampen_too_short" in result.flags
    assert result.values["time_duration"] == pytest.approx(1.0 / FS)


def test_silence_and_constant_are_flagged_not_nan(extractor: TimeDomainExtractor):
    for signal in (np.zeros(2000), np.full(2000, 0.5)):
        result = extractor.extract(signal, FS)
        assert result.failed is False
        assert np.isfinite(result.vector).all()
        assert "crest_factor_zero_rms" in result.flags or result.values["time_rms"] > 0
        assert "shape_stats_undefined" in result.flags


# ---------------------------------------------------------------------------
# real records -- the gate (T32.7)
# ---------------------------------------------------------------------------


def _preprocessed(path: Path, uid: str) -> np.ndarray:
    from src.preprocessing.pipeline import preprocess

    return preprocess(path, record_uid=uid).signal


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_one_real_record_from_each_dataset_yields_24_finite_values(
    extractor: TimeDomainExtractor, sample_records: Any, dataset: str
):
    picked = sample_records(dataset, 1)
    assert picked, "no records for " + dataset

    uid, path = picked[0]
    result = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid)

    assert result.failed is False, dataset + " " + uid + ": " + str(result.error)
    assert len(result.values) == EXPECTED_COUNT
    bad = [name for name, value in result.values.items() if not np.isfinite(value)]
    assert bad == [], dataset + " " + uid + " non-finite: " + ", ".join(bad)


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_land_in_physically_plausible_ranges(
    extractor: TimeDomainExtractor, sample_records: Any, dataset: str
):
    """The gate says finite; finite is not the same as sane.

    The signal is z-normalized, so RMS is ~1 by construction and a value far
    from it means the pipeline or the extractor is not seeing what it thinks it
    is. The autocorrelation lag is bounded by the configured 2 s search window.
    """
    for uid, path in sample_records(dataset, 3):
        values = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid).values
        assert values["time_rms"] == pytest.approx(1.0, abs=1e-2), uid
        assert values["time_mean"] == pytest.approx(0.0, abs=1e-2), uid
        assert 0.0 <= values["time_zcr_mean"] <= 1.0, uid
        assert 0.0 <= values["time_shannon_entropy"] <= 6.0, uid
        assert values["time_crest_factor"] >= 1.0, uid
        assert 0.0 <= values["time_autocorr_peak_lag"] <= 2.0, uid
        assert abs(values["time_autocorr_peak_value"]) <= 1.0, uid
        assert values["time_duration"] > 0.0, uid


@pytest.mark.needs_data
def test_duration_extremes_extract_cleanly(
    extractor: TimeDomainExtractor, master_frame: Any, project_root: Path
):
    """The 0.76 s PASCAL B record and the 122 s PhysioNet record (T32.7).

    The short one is 1,520 samples -- fewer than three analysis frames -- and the
    long one is 244,000, above the sample-entropy window cap. Both are the cases
    a synthetic five-second test says nothing about.
    """
    shortest = master_frame.loc[master_frame["duration_sec"].idxmin()]
    longest = master_frame.loc[master_frame["duration_sec"].idxmax()]

    for row in (shortest, longest):
        uid = str(row["record_uid"])
        signal = _preprocessed(project_root / str(row["file_path"]), uid)
        result = extractor.extract(signal, FS, record_uid=uid)

        assert result.failed is False, uid + ": " + str(result.error)
        assert np.isfinite(result.vector).all(), uid
        assert result.values["time_duration"] == pytest.approx(
            float(row["duration_sec"]), abs=0.01
        ), uid

    long_uid = str(longest["record_uid"])
    long_signal = _preprocessed(project_root / str(longest["file_path"]), long_uid)
    long_result = extractor.extract(long_signal, FS, record_uid=long_uid)
    assert any(flag.startswith("sampen_window_") for flag in long_result.flags), (
        "the 122 s record must report that sample entropy used a window, not the "
        "whole recording"
    )
