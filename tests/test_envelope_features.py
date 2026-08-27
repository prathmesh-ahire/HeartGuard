"""The 5 envelope features and FE-09 (Phase 37, gate T37.7).

The gate asks for exactly 5 values and a peak rate in a plausible physiological
range on real recordings. Both are here.

The sharpest test in this file is the synthetic one: a generated PCG at a known
heart rate must yield a peak rate of *twice* that rate in Hz, because a cardiac
cycle produces two envelope peaks (S1 and S2). At 72 bpm the answer is 2.4
sounds per second, exactly. Getting 1.2 would mean the detector is missing S2;
getting 4.8 would mean it is counting the carrier. Neither would be visible in a
range check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import registry as reg
from src.feature_extraction.envelope import (
    EnvelopeExtractor,
    analytic_envelope,
    envelope_peaks,
    smooth_envelope,
)

FS = 2000
EXPECTED_COUNT = 5


@pytest.fixture(scope="module")
def extractor() -> EnvelopeExtractor:
    return EnvelopeExtractor()


# ---------------------------------------------------------------------------
# family shape (T37.4)
# ---------------------------------------------------------------------------


def test_family_is_registered_and_holds_5_names(extractor: EnvelopeExtractor):
    assert extractor.family == "envelope"
    names = extractor.feature_names()
    assert len(names) == EXPECTED_COUNT
    assert names == reg.feature_names("envelope")
    assert reg.get_extractor("envelope").family == "envelope"


def test_composition_is_mean_std_skew_kurtosis_peak_rate():
    assert list(reg.feature_names("envelope")) == [
        "env_mean",
        "env_std",
        "env_skew",
        "env_kurtosis",
        "env_peak_rate",
    ]


def test_synthetic_signal_yields_5_finite_values(
    extractor: EnvelopeExtractor, synthetic_signal: np.ndarray
):
    result = extractor.extract(synthetic_signal, FS, record_uid="synthetic")
    assert result.failed is False, result.error
    assert len(result.values) == EXPECTED_COUNT
    assert np.isfinite(result.vector).all()
    assert result.n_missing == 0


def test_settings_come_from_config(extractor: EnvelopeExtractor, features_config: Any):
    settings = extractor.settings()
    assert settings.method == features_config.get("families.envelope.method") == "hilbert"
    assert settings.lowpass_hz == features_config.get("families.envelope.smooth_lowpass_hz")
    assert settings.min_distance_sec == features_config.get(
        "families.envelope.peak_detection.min_distance_sec"
    )
    assert settings.prominence_factor == features_config.get(
        "families.envelope.peak_detection.prominence_factor"
    )
    low, high = features_config.get("families.envelope.plausible_peak_rate_hz")
    assert (settings.plausible_low_hz, settings.plausible_high_hz) == (low, high)


# ---------------------------------------------------------------------------
# T37.1 / T37.2 -- the analytic envelope and its smoothing
# ---------------------------------------------------------------------------


def test_the_analytic_envelope_of_an_am_tone_recovers_its_modulation():
    """|hilbert(m(t) * cos(wt))| == m(t) for a slowly-varying non-negative m."""
    t = np.arange(20000) / FS
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    carrier = np.cos(2 * np.pi * 200.0 * t)

    envelope = analytic_envelope(modulation * carrier)
    interior = slice(500, -500)  # the Hilbert transform rings at the edges
    assert np.allclose(envelope[interior], modulation[interior], atol=0.02)


def test_smoothing_removes_the_carrier_ripple_the_peak_count_would_otherwise_see(
    extractor: EnvelopeExtractor,
):
    """The low-pass is what makes ``env_peak_rate`` count sounds, not oscillations."""
    t = np.arange(20000) / FS
    bursts = np.zeros_like(t)
    for start in range(0, t.size, FS):  # one burst per second
        bursts[start : start + 200] = 1.0
    signal = bursts * np.sin(2 * np.pi * 150.0 * t)

    raw = analytic_envelope(signal)
    smoothed = smooth_envelope(raw, FS, extractor.settings().lowpass_hz)

    settings = extractor.settings()
    assert envelope_peaks(smoothed, FS, settings).size <= envelope_peaks(raw, FS, settings).size
    assert envelope_peaks(smoothed, FS, settings).size == 10


def test_smoothing_is_skipped_and_flagged_on_a_signal_too_short_to_pad():
    flags: list[str] = []
    short = np.linspace(0.0, 1.0, 12)
    result = smooth_envelope(short, FS, 20.0, flags=flags)
    assert np.array_equal(result, short)
    assert "envelope_lowpass_skipped_short" in flags


def test_smoothing_is_skipped_and_flagged_when_the_cutoff_is_not_below_nyquist():
    flags: list[str] = []
    values = np.linspace(0.0, 1.0, 4000)
    result = smooth_envelope(values, FS, 1500.0, flags=flags)
    assert np.array_equal(result, values)
    assert "envelope_lowpass_skipped_cutoff" in flags


# ---------------------------------------------------------------------------
# T37.4 -- the peak rate, and what it counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heart_rate_bpm", [50.0, 60.0, 72.0, 76.0])
def test_peak_rate_is_twice_the_heart_rate_because_it_counts_s1_and_s2(
    extractor: EnvelopeExtractor, synthetic_pcg_factory, heart_rate_bpm: float
):
    """The known answer this whole family rests on; see the module docstring.

    The rates stop at 76 bpm on purpose, and the boundary was measured rather
    than derived. ``make_synthetic_pcg`` scales its S1->S2 gap linearly with cycle
    length (0.35 s at 60 bpm, so ~21/bpm seconds). The detector resolves both
    sounds at 76 bpm (gap 0.2765 s, rate 2.55) and merges them at 80 bpm (gap
    0.2625 s, rate 1.35), so the effective floor is ~0.27 s -- slightly *above*
    the nominal 0.25 s ``min_distance_sec``, because the 20 Hz low-pass shifts
    envelope maxima away from the burst centres. See the next test, which pins
    the merging behaviour rather than leaving it as a surprise, and note that
    real systole does not shorten the way this generator does: the real-record
    medians are 2.4-2.7 sounds per second across all four datasets.
    """
    pcg = synthetic_pcg_factory(duration_sec=20.0, fs=FS, heart_rate_bpm=heart_rate_bpm)
    result = extractor.extract(pcg.signal, FS)

    expected = 2.0 * heart_rate_bpm / 60.0
    assert result.values["env_peak_rate"] == pytest.approx(expected, abs=0.15), (
        str(heart_rate_bpm) + " bpm -> " + str(result.values["env_peak_rate"])
        + " sounds/s, expected " + str(expected)
    )


def test_sounds_closer_than_the_minimum_distance_are_merged_by_design(
    extractor: EnvelopeExtractor, synthetic_pcg_factory
):
    """``min_distance_sec`` is a floor on S1-S2 separation, and it has a cost.

    At 90 bpm the synthetic generator places S2 just 0.2335 s after S1, inside
    the configured 0.25 s minimum peak distance, so ``find_peaks`` suppresses S2
    and the rate falls to about one sound per beat. That is the documented
    trade-off of the distance guard, not a detector bug -- lowering it would let
    the residual carrier ripple through instead.

    This matters most for CirCor, which is ~98% paediatric: at a genuine 150 bpm
    with a ~0.2 s systole the same merging would occur. It does not appear to
    bite on the real corpus -- D4's median measured 2.39 sounds/s, i.e. both
    sounds resolved -- but a future session reading an unexpectedly low
    ``env_peak_rate`` on a fast record should look here first.
    """
    settings = extractor.settings()
    pcg = synthetic_pcg_factory(duration_sec=20.0, fs=FS, heart_rate_bpm=90.0)

    gap = float(np.median(pcg.s2_times[: pcg.s1_times.size] - pcg.s1_times[: pcg.s2_times.size]))
    assert gap < settings.min_distance_sec, "the premise of this test no longer holds"

    rate = extractor.extract(pcg.signal, FS).values["env_peak_rate"]
    assert rate == pytest.approx(90.0 / 60.0, abs=0.1), "expected one sound per beat"


def test_peak_rate_counts_a_known_burst_train_exactly(extractor: EnvelopeExtractor):
    t = np.arange(10 * FS) / FS
    signal = np.zeros_like(t)
    for start in range(0, t.size, FS // 2):  # two bursts per second
        signal[start : start + 200] = np.sin(2 * np.pi * 150.0 * t[:200])

    result = extractor.extract(signal, FS)
    assert result.values["env_peak_rate"] == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# T37.5 -- the plausibility flag
# ---------------------------------------------------------------------------


def test_an_implausible_peak_rate_is_flagged_and_never_clipped(
    extractor: EnvelopeExtractor,
):
    """The value stands as measured; the flag is what a robustness split groups on."""
    settings = extractor.settings()
    t = np.arange(10 * FS) / FS
    signal = np.zeros_like(t)
    step = int(0.26 * FS)  # ~3.8 sounds/s, just above the configured ceiling
    for start in range(0, t.size - 200, step):
        signal[start : start + 120] = np.sin(2 * np.pi * 150.0 * t[:120])

    result = extractor.extract(signal, FS, record_uid="fast")
    rate = result.values["env_peak_rate"]

    assert rate > settings.plausible_high_hz
    assert "envelope_peak_rate_implausible" in result.flags
    assert np.isfinite(rate), "the flag must not replace or clip the value"


def test_a_plausible_rate_is_not_flagged(extractor: EnvelopeExtractor,
                                         synthetic_pcg_factory):
    pcg = synthetic_pcg_factory(duration_sec=20.0, fs=FS, heart_rate_bpm=72.0)
    result = extractor.extract(pcg.signal, FS)
    assert "envelope_peak_rate_implausible" not in result.flags


def test_extraction_is_deterministic(extractor: EnvelopeExtractor,
                                     synthetic_signal: np.ndarray):
    first = extractor.extract(synthetic_signal, FS).vector
    second = extractor.extract(synthetic_signal, FS).vector
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


def test_edge_case_signals_never_raise(extractor: EnvelopeExtractor,
                                       edge_case_signals: dict[str, np.ndarray]):
    for label, signal in edge_case_signals.items():
        result = extractor.extract(signal, FS, record_uid=label)
        assert len(result.values) == EXPECTED_COUNT, label
        if result.failed:
            assert result.n_missing == EXPECTED_COUNT, label
            continue
        assert np.isfinite(result.vector).all(), (
            label + " produced NaN: "
            + str([n for n, v in result.values.items() if not np.isfinite(v)])
        )


def test_a_dc_only_recording_returns_zero_shape_rather_than_nan(
    extractor: EnvelopeExtractor,
):
    """Regression: scipy's skew/kurtosis return NaN on a near-constant envelope.

    A DC-only signal has an envelope that is constant to within rounding but not
    exactly constant, so an ``env_std == 0`` guard misses it and scipy hits
    catastrophic cancellation. The guard is relative to the float spacing at the
    data's own magnitude. Found by the T37.7 gate; see Docs/note.md, 2026-08-27.
    """
    result = extractor.extract(np.full(4000, -0.8), FS, record_uid="dc_only")

    assert result.failed is False, result.error
    assert np.isfinite(result.vector).all()
    assert result.values["env_skew"] == 0.0
    assert result.values["env_kurtosis"] == 0.0
    assert "envelope_shape_undefined" in result.flags


# ---------------------------------------------------------------------------
# real records -- the gate (T37.7)
# ---------------------------------------------------------------------------


def _preprocessed(path: Path, uid: str) -> np.ndarray:
    from src.preprocessing.pipeline import preprocess

    return preprocess(path, record_uid=uid).signal


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_yield_5_finite_values(
    extractor: EnvelopeExtractor, sample_records: Any, dataset: str
):
    for uid, path in sample_records(dataset, 5):
        result = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid)
        assert result.failed is False, dataset + " " + uid + ": " + str(result.error)
        assert len(result.values) == EXPECTED_COUNT
        bad = [name for name, value in result.values.items() if not np.isfinite(value)]
        assert bad == [], dataset + " " + uid + " non-finite: " + ", ".join(bad)
        assert result.values["env_mean"] > 0.0, uid
        assert result.values["env_std"] > 0.0, uid


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_peak_rate_is_physiologically_plausible_on_real_recordings(
    extractor: EnvelopeExtractor, sample_records: Any, dataset: str
):
    """The T37.7 assertion, checked as a *rate* on the sample, not per record.

    A per-record assertion would be wrong: the flag exists precisely because some
    real recordings are too noisy or too quiet to have a meaningful peak rate, and
    a gate that forbade any such record would be a gate that forbids the data.
    Measured over 60 records per dataset (2026-08-27) the flag fires on 1.7%
    overall, so requiring at least 80% of a 20-record sample inside the range is a
    real constraint with room for the genuine outliers.
    """
    settings = extractor.settings()
    rates = [
        extractor.extract(_preprocessed(path, uid), FS, record_uid=uid).values[
            "env_peak_rate"
        ]
        for uid, path in sample_records(dataset, 20)
    ]
    assert rates

    inside = [
        rate for rate in rates
        if settings.plausible_low_hz <= rate <= settings.plausible_high_hz
    ]
    assert len(inside) / len(rates) >= 0.8, (
        dataset + ": only " + str(len(inside)) + " of " + str(len(rates))
        + " records had a plausible peak rate"
    )
    # And the middle of the distribution must sit where two sounds per beat put
    # it: a median outside 1.5-3.2 sounds/s would mean systematic mis-counting.
    assert 1.5 <= float(np.median(rates)) <= 3.2, dataset + " median " + str(
        float(np.median(rates))
    )


@pytest.mark.needs_data
def test_duration_extremes_extract_cleanly(
    extractor: EnvelopeExtractor, master_frame: Any, project_root: Path
):
    shortest = master_frame.loc[master_frame["duration_sec"].idxmin()]
    longest = master_frame.loc[master_frame["duration_sec"].idxmax()]

    for row in (shortest, longest):
        uid = str(row["record_uid"])
        signal = _preprocessed(project_root / str(row["file_path"]), uid)
        result = extractor.extract(signal, FS, record_uid=uid)
        assert result.failed is False, uid + ": " + str(result.error)
        assert np.isfinite(result.vector).all(), uid
        assert not any(
            flag.startswith("envelope_lowpass_skipped") for flag in result.flags
        ), uid + " flags=" + str(result.flags)


# ---------------------------------------------------------------------------
# T37.6 -- FE-09
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_fe09_signal_envelope_is_emitted(tmp_path: Path):
    from src.feature_extraction.figures import (
        FE_FIGURES,
        features_dir,
        plot_signal_envelope,
    )

    first = plot_signal_envelope(tmp_path / "a.png")
    assert first.is_file()
    assert first.stat().st_size > 10_000

    second = plot_signal_envelope(tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes(), "FE-09 is not reproducible"

    committed = features_dir() / FE_FIGURES["FE-09"]
    assert committed.is_file(), "FE-09 has not been generated into outputs/03_features"


@pytest.mark.needs_data
def test_every_declared_feature_figure_exists():
    """The set is a literal so that adding a figure is a deliberate edit.

    FE-06 joined it in Phase 42; FE-07 through FE-09 arrived in Phases 35-37.
    Equality is kept rather than relaxed to a subset: a figure declared in
    FE_FIGURES and never written is exactly the gap this catches.
    """
    from src.feature_extraction.figures import FE_FIGURES, features_dir

    assert set(FE_FIGURES) == {"FE-06", "FE-07", "FE-08", "FE-09"}
    for evidence_id, filename in FE_FIGURES.items():
        path = features_dir() / filename
        assert path.is_file(), evidence_id + " missing: " + str(path)
        assert path.stat().st_size > 10_000, evidence_id + " is suspiciously small"
