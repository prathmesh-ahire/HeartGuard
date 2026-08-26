"""The 24 chroma features and FE-07 (Phase 35, gate T35.7).

The gate asks for exactly 24 values and stability on a synthetic constant-pitch
signal. "Stable" is made concrete here: on a pure tone the chroma frames must not
move, so every bin's standard deviation over frames sits near zero and the same
bin dominates throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import registry as reg
from src.feature_extraction.chroma import ChromaExtractor, chroma_matrix

FS = 2000
EXPECTED_COUNT = 24
N_CHROMA = 12


@pytest.fixture(scope="module")
def extractor() -> ChromaExtractor:
    return ChromaExtractor()


def _tone(freq_hz: float, seconds: float = 5.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# family shape (T35.2, T35.3)
# ---------------------------------------------------------------------------


def test_family_is_registered_and_holds_24_names(extractor: ChromaExtractor):
    assert extractor.family == "chroma"
    names = extractor.feature_names()
    assert len(names) == EXPECTED_COUNT
    assert names == reg.feature_names("chroma")
    assert reg.get_extractor("chroma").family == "chroma"


def test_composition_is_12_means_then_12_stds():
    names = list(reg.feature_names("chroma"))
    expected = ["chroma_" + str(k).zfill(2) + "_mean" for k in range(1, N_CHROMA + 1)] + [
        "chroma_" + str(k).zfill(2) + "_std" for k in range(1, N_CHROMA + 1)
    ]
    assert names == expected
    assert len(names) == EXPECTED_COUNT == 2 * N_CHROMA


def test_synthetic_signal_yields_24_finite_values(
    extractor: ChromaExtractor, synthetic_signal: np.ndarray
):
    result = extractor.extract(synthetic_signal, FS, record_uid="synthetic")
    assert result.failed is False, result.error
    assert len(result.values) == EXPECTED_COUNT
    assert np.isfinite(result.vector).all()
    assert result.n_missing == 0


def test_settings_come_from_config(extractor: ChromaExtractor, features_config: Any,
                                   signal_config: Any):
    settings = extractor.settings()
    assert settings.n_chroma == features_config.get("families.chroma.n_chroma")
    assert settings.n_fft == features_config.get("families.chroma.n_fft")
    assert settings.hop_length == features_config.get("families.chroma.hop_length")
    assert settings.fmin == features_config.get("families.chroma.fmin")
    assert settings.tuning == features_config.get("families.chroma.tuning")
    assert settings.fmax == signal_config.get("filter.high_hz")
    assert settings.tuning == 0.0, "tuning must be fixed, never estimated"


# ---------------------------------------------------------------------------
# T35.1 -- adapted to the 20-400 Hz band
# ---------------------------------------------------------------------------


def _unmasked_chroma(signal: np.ndarray, settings: Any) -> np.ndarray:
    """Chroma over the full spectrum -- the control this band-limiting is measured against."""
    import librosa

    magnitude = np.abs(
        librosa.stft(
            y=np.asarray(signal, dtype=np.float64),
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            window=settings.window,
            center=settings.center,
        )
    )
    return librosa.feature.chroma_stft(
        S=magnitude,
        sr=FS,
        n_fft=settings.n_fft,
        n_chroma=settings.n_chroma,
        tuning=settings.tuning,
    )


@pytest.mark.parametrize(
    ("label", "contaminant_hz", "amplitude", "min_reduction"),
    [("above the passband", 700.0, 3.0, 50.0), ("below the passband", 3.0, 5.0, 4.0)],
)
def test_band_limiting_suppresses_out_of_band_contamination(
    extractor: ChromaExtractor,
    label: str,
    contaminant_hz: float,
    amplitude: float,
    min_reduction: float,
):
    """Adding an out-of-band tone must barely move the chroma of an in-band one.

    Measured as a *reduction factor* against unmasked chroma rather than as exact
    equality, because exact equality is unachievable and it is worth knowing why:
    a 512-point Hann window has sidelobes, so a 700 Hz tone deposits a little
    energy into bins inside 20-400 Hz before any masking happens. Zeroing
    out-of-band bins cannot remove energy that has already leaked into in-band
    ones. The masking still does the work it is there for -- roughly 300x less
    contamination from above the passband and 8x from below -- and in the real
    pipeline the bandpass has already attenuated both before extraction runs.
    """
    settings = extractor.settings()
    clean = _tone(150.0)
    contaminated = clean + amplitude * _tone(contaminant_hz)

    masked = float(
        np.abs(
            chroma_matrix(contaminated, FS, settings) - chroma_matrix(clean, FS, settings)
        ).mean()
    )
    unmasked = float(
        np.abs(_unmasked_chroma(contaminated, settings) - _unmasked_chroma(clean, settings)).mean()
    )

    assert masked < 0.01, label + ": masked residue " + str(masked)
    assert unmasked > masked * min_reduction, (
        label + ": band-limiting only reduced contamination from "
        + str(unmasked) + " to " + str(masked)
    )


# ---------------------------------------------------------------------------
# T35.5 / T35.7 -- stability on a constant-pitch signal
# ---------------------------------------------------------------------------


def test_a_constant_pitch_signal_gives_stable_chroma(extractor: ChromaExtractor):
    """The gate's stability requirement, made concrete.

    A pure tone does not change from frame to frame, so no bin's value should
    move: every one of the twelve standard deviations must be near zero, and the
    same bin must dominate every frame.
    """
    result = extractor.extract(_tone(150.0, seconds=8.0), FS, record_uid="constant_pitch")
    assert result.failed is False, result.error

    stds = np.array(
        [result.values["chroma_" + str(k).zfill(2) + "_std"] for k in range(1, 13)]
    )
    means = np.array(
        [result.values["chroma_" + str(k).zfill(2) + "_mean"] for k in range(1, 13)]
    )

    assert float(stds.max()) < 0.1, "chroma drifts on a constant-pitch signal"
    assert float(means.max()) == pytest.approx(1.0, abs=1e-6)

    settings = extractor.settings()
    chroma = chroma_matrix(_tone(150.0, seconds=8.0), FS, settings)
    dominant = np.argmax(chroma, axis=0)
    assert len(set(dominant.tolist())) == 1, "the dominant bin moves between frames"


def test_different_pitches_land_in_different_bins(extractor: ChromaExtractor):
    """Guards against a chroma that returns the same 24 numbers for everything."""
    settings = extractor.settings()
    first = int(np.argmax(chroma_matrix(_tone(110.0), FS, settings).mean(axis=1)))
    second = int(np.argmax(chroma_matrix(_tone(155.0), FS, settings).mean(axis=1)))
    assert first != second


def test_extraction_is_deterministic(extractor: ChromaExtractor,
                                     synthetic_signal: np.ndarray):
    first = extractor.extract(synthetic_signal, FS).vector
    second = extractor.extract(synthetic_signal, FS).vector
    assert np.array_equal(first, second)


def test_values_stay_in_the_normalized_range(extractor: ChromaExtractor,
                                             synthetic_signal: np.ndarray):
    values = extractor.extract(synthetic_signal, FS).values
    for k in range(1, N_CHROMA + 1):
        key = "chroma_" + str(k).zfill(2)
        assert 0.0 <= values[key + "_mean"] <= 1.0, key
        assert 0.0 <= values[key + "_std"] <= 1.0, key


# ---------------------------------------------------------------------------
# degenerate input
# ---------------------------------------------------------------------------


def test_edge_case_signals_never_raise(extractor: ChromaExtractor,
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
                flag.startswith("chroma_signal_too_short")
                or flag.startswith("chroma_no_in_band_energy")
                for flag in result.flags
            ), label + ": unexplained NaN; flags=" + str(result.flags)


def test_silence_reports_a_gap_rather_than_twelve_confident_zeros(
    extractor: ChromaExtractor,
):
    """A frame with no in-band energy would normalize 0/0; that is a gap, not a value."""
    result = extractor.extract(np.zeros(4000), FS, record_uid="silence")
    assert result.failed is False
    assert result.n_missing == EXPECTED_COUNT
    assert "chroma_no_in_band_energy" in result.flags


# ---------------------------------------------------------------------------
# real records -- the gate (T35.7)
# ---------------------------------------------------------------------------


def _preprocessed(path: Path, uid: str) -> np.ndarray:
    from src.preprocessing.pipeline import preprocess

    return preprocess(path, record_uid=uid).signal


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_yield_24_finite_values(
    extractor: ChromaExtractor, sample_records: Any, dataset: str
):
    for uid, path in sample_records(dataset, 3):
        result = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid)
        assert result.failed is False, dataset + " " + uid + ": " + str(result.error)
        assert len(result.values) == EXPECTED_COUNT
        bad = [name for name, value in result.values.items() if not np.isfinite(value)]
        assert bad == [], dataset + " " + uid + " non-finite: " + ", ".join(bad)
        for k in range(1, N_CHROMA + 1):
            key = "chroma_" + str(k).zfill(2)
            assert 0.0 <= result.values[key + "_mean"] <= 1.0, uid + " " + key
            assert 0.0 <= result.values[key + "_std"] <= 1.0, uid + " " + key


@pytest.mark.needs_data
def test_duration_extremes_extract_cleanly(
    extractor: ChromaExtractor, master_frame: Any, project_root: Path
):
    shortest = master_frame.loc[master_frame["duration_sec"].idxmin()]
    longest = master_frame.loc[master_frame["duration_sec"].idxmax()]

    for row in (shortest, longest):
        uid = str(row["record_uid"])
        signal = _preprocessed(project_root / str(row["file_path"]), uid)
        result = extractor.extract(signal, FS, record_uid=uid)
        assert result.failed is False, uid + ": " + str(result.error)
        assert np.isfinite(result.vector).all(), uid
        assert result.flags == (), uid + " flags=" + str(result.flags)


# ---------------------------------------------------------------------------
# T35.6 -- FE-07
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_fe07_chroma_heatmap_is_emitted(tmp_path: Path):
    """The figure must exist, be non-trivial, and be byte-identical on a rerun."""
    from src.feature_extraction.figures import FE_FIGURES, generate_all

    first = generate_all(tmp_path / "a")["FE-07"]
    assert first.is_file()
    assert first.name == FE_FIGURES["FE-07"] == "chroma_heatmap.png"
    assert first.stat().st_size > 10_000

    second = generate_all(tmp_path / "b")["FE-07"]
    assert first.read_bytes() == second.read_bytes(), "FE-07 is not reproducible"


@pytest.mark.needs_data
def test_fe07_exists_in_the_features_output_directory():
    from src.feature_extraction.figures import FE_FIGURES, features_dir

    path = features_dir() / FE_FIGURES["FE-07"]
    assert path.is_file(), "FE-07 has not been generated into outputs/03_features"
    assert path.stat().st_size > 10_000
