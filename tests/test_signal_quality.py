"""Signal quality analysis gate (T26.7).

The gate itself is the last test: PP-08 must exist for all 7,536 records with no
NaN in any flag, and the noise proxy must still agree with PhysioNet's human SQI
annotation at the level the Phase 26 calibration measured. That last assertion is
the one that matters -- it is what caught the original configuration flagging 33
records and calling it noise detection.

The synthetic tests above it build signals whose correct answer is known by
construction: a pure tone has high in-band SNR, a low-frequency sweep has none, a
clipped signal is clipped. Real recordings cannot prove those, because no
recording comes with its true SNR written on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.preprocessing import quality as q

FS = 2000
CORPUS_SIZE = 7536

# The Phase 26 calibration measured 0.789 for the drift proxy against SQI. The
# floor is set below that with room for a resampler or scipy version to move the
# last decimal -- but far enough above 0.5 that a proxy which has stopped working
# cannot pass. Lowering it to make a run go green is forbidden (CLAUDE.md).
MIN_BALANCED_ACCURACY = 0.70


@pytest.fixture(scope="module")
def thresholds() -> q.QualityThresholds:
    return q.load_thresholds()


@pytest.fixture(scope="module")
def corpus(thresholds: q.QualityThresholds) -> Any:
    """The cached corpus scan, flagged. Built once for the whole module."""
    return q.scan_quality(thresholds=thresholds)


def _tone(freq_hz: float, seconds: float = 5.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * FS), dtype=np.float64) / FS
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


# ===========================================================================
# configuration
# ===========================================================================


def test_thresholds_load(thresholds: q.QualityThresholds) -> None:
    assert thresholds.target_fs == 2000
    assert thresholds.in_band == (20.0, 400.0)
    assert thresholds.out_of_band == (400.0, 1000.0)
    assert thresholds.drift_ratio_db_flag == 5.0


def test_a_passband_mismatch_is_refused(signal_config: Any) -> None:
    """The SNR proxy must measure the band the filter keeps, or it measures nothing."""
    import copy

    from src.utils.config import Config

    data = copy.deepcopy(signal_config.as_dict())
    data["quality"]["snr_proxy"]["in_band"] = [30, 400]
    with pytest.raises(ValueError, match="in_band"):
        q.load_thresholds(Config("signal", data))


# ===========================================================================
# T26.1 - T26.4 -- measurements with a known answer
# ===========================================================================


def test_amplitude_metrics_on_a_known_sine(thresholds: q.QualityThresholds) -> None:
    """A 0.5-amplitude sine has RMS 0.354 and a crest factor of 3.01 dB."""
    row = q.measure_signal(_tone(100.0), FS, thresholds)

    assert row["peak"] == pytest.approx(0.5, abs=1e-3)
    assert row["rms"] == pytest.approx(0.5 / np.sqrt(2), abs=1e-3)
    assert row["crest_factor_db"] == pytest.approx(3.01, abs=0.05)
    assert row["duration_sec"] == pytest.approx(5.0, abs=1e-6)


def test_clipping_ratio_counts_full_scale_samples(thresholds: q.QualityThresholds) -> None:
    clipped = np.clip(_tone(100.0, amplitude=2.0), -1.0, 1.0)
    row = q.measure_signal(clipped, FS, thresholds)
    assert row["clipping_ratio"] > 0.3

    assert q.measure_signal(_tone(100.0), FS, thresholds)["clipping_ratio"] == 0.0


def test_silence_ratio_finds_silence(thresholds: q.QualityThresholds) -> None:
    signal = _tone(100.0, seconds=4.0)
    signal[: 2 * FS] = 0.0  # first half silent

    row = q.measure_signal(signal, FS, thresholds)
    assert row["silence_ratio"] == pytest.approx(0.5, abs=0.05)
    assert q.measure_signal(_tone(100.0), FS, thresholds)["silence_ratio"] == 0.0


def test_snr_proxy_separates_in_band_from_out_of_band(
    thresholds: q.QualityThresholds, rng: np.random.Generator
) -> None:
    """T26.3 -- a 100 Hz tone is all signal; a 700 Hz tone is all noise."""
    in_band = q.measure_signal(_tone(100.0), FS, thresholds)["snr_proxy_db"]
    out_of_band = q.measure_signal(_tone(700.0), FS, thresholds)["snr_proxy_db"]
    white = q.measure_signal(
        rng.normal(0, 0.1, 5 * FS).astype(np.float32), FS, thresholds
    )["snr_proxy_db"]

    assert in_band > 40.0
    assert out_of_band < -20.0
    assert -10.0 < white < 15.0, white


def test_drift_ratio_detects_low_frequency_dominance(
    thresholds: q.QualityThresholds,
) -> None:
    """T26.5's working proxy: energy below 20 Hz against the heart band."""
    clean = _tone(100.0)
    drifty = (clean + _tone(3.0, amplitude=2.0)).astype(np.float32)

    assert q.measure_signal(clean, FS, thresholds)["drift_ratio_db"] < -20.0
    assert q.measure_signal(drifty, FS, thresholds)["drift_ratio_db"] > 5.0


def test_spectral_flatness_ranks_noise_above_tone(
    thresholds: q.QualityThresholds, rng: np.random.Generator
) -> None:
    tone = q.measure_signal(_tone(100.0), FS, thresholds)["spectral_flatness"]
    noise = q.measure_signal(
        rng.normal(0, 0.1, 5 * FS).astype(np.float32), FS, thresholds
    )["spectral_flatness"]
    assert 0.0 <= tone < noise <= 1.0


def test_zero_crossing_rate_tracks_frequency(thresholds: q.QualityThresholds) -> None:
    slow = q.measure_signal(_tone(50.0), FS, thresholds)["zcr_mean"]
    fast = q.measure_signal(_tone(400.0), FS, thresholds)["zcr_mean"]
    # 400 Hz at 2 kHz crosses zero on ~40% of sample pairs; 50 Hz on ~5%.
    assert slow == pytest.approx(0.05, abs=0.02)
    assert fast == pytest.approx(0.40, abs=0.05)


@pytest.mark.parametrize("kind", ["shortest_real", "silence", "constant", "clipped", "impulse"])
def test_degenerate_signals_produce_finite_metrics(
    edge_case_signals: dict[str, np.ndarray], kind: str, thresholds: q.QualityThresholds
) -> None:
    """T26.7's "no NaN" requirement, at the source rather than in the CSV."""
    row = q.measure_signal(edge_case_signals[kind], FS, thresholds)
    for key, value in row.items():
        assert np.isfinite(value), (kind, key, value)


def test_an_empty_signal_is_measured_not_raised(thresholds: q.QualityThresholds) -> None:
    row = q.measure_signal(np.zeros(0, dtype=np.float32), FS, thresholds)
    assert row["n_samples"] == 0
    assert row["silence_ratio"] == 1.0
    assert all(np.isfinite(v) for v in row.values() if isinstance(v, float))


def test_frame_rms_falls_back_for_sub_frame_signals(thresholds: q.QualityThresholds) -> None:
    frames = q.frame_rms(
        np.ones(100, dtype=np.float32), thresholds.frame_length, thresholds.hop_length
    )
    assert frames.shape == (1,)
    assert frames[0] == pytest.approx(1.0)
    assert q.frame_rms(np.zeros(0), 512, 256).size == 0


# ===========================================================================
# T26.5 -- the composite flags
# ===========================================================================


def _table(**columns: Any) -> Any:
    import pandas as pd

    base = {
        "dataset_source": ["D1", "D1", "D2"],
        "duration_sec": [10.0, 10.0, 10.0],
        "clipping_ratio": [0.0, 0.0, 0.0],
        "silence_ratio": [0.0, 0.0, 0.0],
        "snr_proxy_db": [30.0, 30.0, 30.0],
        "spectral_flatness": [0.01, 0.01, 0.01],
        "drift_ratio_db": [-10.0, -10.0, -10.0],
        "zcr_mean": [0.05, 0.05, 0.05],
    }
    base.update(columns)
    return pd.DataFrame(base)


def test_each_flag_fires_on_its_own_metric(thresholds: q.QualityThresholds) -> None:
    flagged = q.apply_flags(
        _table(
            duration_sec=[1.0, 10.0, 10.0],
            clipping_ratio=[0.0, 0.5, 0.0],
            drift_ratio_db=[-10.0, -10.0, 20.0],
        ),
        thresholds,
    )

    assert list(flagged["is_short"]) == [True, False, False]
    assert list(flagged["is_clipped"]) == [False, True, False]
    assert list(flagged["is_noisy"]) == [False, False, True]
    assert list(flagged["is_low_quality"]) == [True, True, True]
    assert list(flagged["quality_reasons"]) == ["short", "clipped", "baseline_drift"]


def test_all_three_noise_proxies_are_live(thresholds: q.QualityThresholds) -> None:
    """Each proxy alone must be able to raise is_noisy."""
    flagged = q.apply_flags(
        _table(
            snr_proxy_db=[1.0, 30.0, 30.0],
            spectral_flatness=[0.01, 0.9, 0.01],
            drift_ratio_db=[-10.0, -10.0, 20.0],
        ),
        thresholds,
    )
    assert list(flagged["is_noisy"]) == [True, True, True]
    assert list(flagged["quality_reasons"]) == ["low_snr", "flat_spectrum", "baseline_drift"]


def test_zcr_anomaly_is_scored_within_each_dataset(thresholds: q.QualityThresholds) -> None:
    """A PASCAL record must not look anomalous merely for not being PhysioNet."""
    flagged = q.apply_flags(
        _table(
            dataset_source=["D1", "D1", "D2"],
            zcr_mean=[0.05, 0.05, 0.40],
        ),
        thresholds,
    )
    assert (flagged["zcr_anomaly"] == 0.0).all()
    assert np.isfinite(flagged["zcr_anomaly"]).all()


def test_flags_are_derived_not_stored(thresholds: q.QualityThresholds) -> None:
    """Re-flagging the same measurements with a different threshold must move."""
    from dataclasses import replace

    table = _table(drift_ratio_db=[-10.0, 0.0, 10.0])
    strict = q.apply_flags(table, replace(thresholds, drift_ratio_db_flag=-20.0))
    loose = q.apply_flags(table, replace(thresholds, drift_ratio_db_flag=50.0))

    assert int(strict["is_noisy"].sum()) == 3
    assert int(loose["is_noisy"].sum()) == 0


# ===========================================================================
# T26.6 / T26.7 -- the gate
# ===========================================================================


@pytest.mark.needs_data
@pytest.mark.slow
def test_pp08_covers_every_record_with_no_nan_flags(corpus: Any) -> None:
    """T26.7 -- 7,536 records, no NaN anywhere."""
    assert len(corpus) == CORPUS_SIZE
    assert corpus["dataset_source"].value_counts().to_dict() == {
        "D1": 3541,
        "D4": 3163,
        "D3": 656,
        "D2": 176,
    }

    for column in (*q.QUALITY_COLUMNS, *q.FLAG_COLUMNS):
        assert column in corpus.columns, column
        assert corpus[column].isna().sum() == 0, column

    numeric = corpus.select_dtypes(include=[float, int])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()

    for column in ("is_short", "is_clipped", "is_silent", "is_noisy", "is_low_quality"):
        assert corpus[column].dtype == bool, column


@pytest.mark.needs_data
@pytest.mark.slow
def test_noisy_flag_agrees_with_physionet_sqi(corpus: Any, thresholds: q.QualityThresholds) -> None:
    """T26.7 -- the sanity check against the human annotation.

    The drift proxy is the one calibrated in T26.5; the composite is what PP-08
    actually writes. Both are asserted, because a composite that scores well
    while its calibrated term has silently stopped working would be a pass for
    the wrong reason.
    """
    calibration = q.calibrate_against_sqi(corpus, thresholds=thresholds)

    configured = calibration[calibration["is_configured"]]
    assert not configured.empty

    drift = configured[configured["metric"] == "drift_ratio_db"].iloc[0]
    assert drift["n_records"] == 3541
    assert drift["n_sqi_poor"] == 404
    assert drift["balanced_accuracy"] >= MIN_BALANCED_ACCURACY, drift.to_dict()
    assert drift["sensitivity_to_sqi_poor"] >= 0.55, drift.to_dict()

    composite = calibration[calibration["metric"].str.startswith("is_noisy")].iloc[0]
    assert composite["balanced_accuracy"] >= MIN_BALANCED_ACCURACY, composite.to_dict()


@pytest.mark.needs_data
@pytest.mark.slow
def test_pp08_is_written_to_disk(corpus: Any, tmp_path: Any) -> None:
    """T26.6 -- the artifact itself, round-tripped."""
    import pandas as pd

    path = q.write_quality_flags(corpus, tmp_path)
    assert path.is_file()

    written = pd.read_csv(path, keep_default_na=False)
    assert len(written) == CORPUS_SIZE
    assert list(written.columns) == [*q.QUALITY_COLUMNS, *q.FLAG_COLUMNS]
