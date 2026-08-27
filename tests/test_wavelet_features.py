"""The 24 DWT features and FE-08 (Phase 36, gate T36.7).

The gate asks for exactly 24 values and a short signal that reduces the level and
sets a flag instead of failing. Both are here, together with the check that
matters most for a filterbank: that each named sub-band actually holds the
frequency range its name claims. A test that only counted 24 finite numbers would
pass on a decomposition whose sub-bands were shuffled, and every downstream SHAP
plot would then attribute a murmur to the wrong band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.feature_extraction import registry as reg
from src.feature_extraction.wavelet import (
    STATS,
    SUBBANDS,
    WaveletExtractor,
    decompose,
    usable_level,
)

FS = 2000
EXPECTED_COUNT = 24


@pytest.fixture(scope="module")
def extractor() -> WaveletExtractor:
    return WaveletExtractor()


def _tone(freq_hz: float, seconds: float = 10.0, fs: int = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# family shape (T36.2-T36.5)
# ---------------------------------------------------------------------------


def test_family_is_registered_and_holds_24_names(extractor: WaveletExtractor):
    assert extractor.family == "dwt"
    names = extractor.feature_names()
    assert len(names) == EXPECTED_COUNT
    assert names == reg.feature_names("dwt")
    assert reg.get_extractor("dwt").family == "dwt"


def test_composition_is_six_subbands_times_four_statistics():
    """energy(6) + std(6) + entropy(6) + mean_abs(6) = 24, blocked by statistic."""
    names = list(reg.feature_names("dwt"))
    expected = [
        "dwt_" + band + "_" + stat for stat in STATS for band in SUBBANDS
    ]
    assert names == expected
    assert len(SUBBANDS) == 6
    assert len(STATS) == 4
    assert len(names) == EXPECTED_COUNT == 6 * 4


def test_synthetic_signal_yields_24_finite_values(
    extractor: WaveletExtractor, synthetic_signal: np.ndarray
):
    result = extractor.extract(synthetic_signal, FS, record_uid="synthetic")
    assert result.failed is False, result.error
    assert len(result.values) == EXPECTED_COUNT
    assert np.isfinite(result.vector).all()
    assert result.n_missing == 0


def test_settings_come_from_config(extractor: WaveletExtractor, features_config: Any):
    settings = extractor.settings()
    assert settings.wavelet == features_config.get("families.dwt.wavelet") == "db4"
    assert settings.level == features_config.get("families.dwt.level") == 5
    assert settings.mode == features_config.get("families.dwt.mode")
    assert settings.entropy_bins == features_config.get("families.dwt.entropy_bins")
    assert tuple(features_config.get("families.dwt.subbands")) == SUBBANDS
    assert tuple(features_config.get("families.dwt.stats")) == STATS


def test_a_config_subband_edit_is_rejected_rather_than_silently_relabelling():
    class _Cfg:
        def get(self, key, default=None):
            if key == "families.dwt.subbands":
                return ["cA4", "cD4", "cD3", "cD2", "cD1", "cD0"]
            return {"families.dwt.stats": list(STATS)}.get(key, default)

    with pytest.raises(ValueError, match="subbands"):
        WaveletExtractor(_Cfg()).settings()


# ---------------------------------------------------------------------------
# T36.1 -- the decomposition really is the filterbank its names claim
# ---------------------------------------------------------------------------


def test_wavedec_maps_onto_the_six_registry_names(extractor: WaveletExtractor):
    bands = decompose(_tone(100.0), extractor.settings())
    assert list(bands) == list(SUBBANDS)
    for coefficients in bands.values():
        assert coefficients.ndim == 1
        assert coefficients.size > 0


@pytest.mark.parametrize(
    ("freq_hz", "expected_band"),
    [
        (10.0, "cA5"),    # 0-31.25 Hz
        (40.0, "cD5"),    # 31.25-62.5 Hz
        (90.0, "cD4"),    # 62.5-125 Hz
        (180.0, "cD3"),   # 125-250 Hz
        (300.0, "cD2"),   # 250-500 Hz
        (700.0, "cD1"),   # 500-1000 Hz
    ],
)
def test_each_subband_holds_the_frequency_range_its_name_claims(
    extractor: WaveletExtractor, freq_hz: float, expected_band: str
):
    """A tone must land in the sub-band that covers its frequency, and dominate it.

    This is the assertion that a shuffled or mislabelled decomposition fails and a
    finiteness check does not.
    """
    bands = decompose(_tone(freq_hz), extractor.settings())
    energies = {name: float(np.sum(c**2)) for name, c in bands.items()}
    total = sum(energies.values())

    dominant = max(energies, key=lambda name: energies[name])
    assert dominant == expected_band, (
        str(freq_hz) + " Hz landed in " + dominant + ", expected " + expected_band
    )
    assert energies[expected_band] / total > 0.7


def test_the_decomposition_conserves_energy(extractor: WaveletExtractor):
    """db4 is orthogonal, so sub-band energies sum to the signal's, up to the edges.

    The `symmetric` boundary mode extends the signal, so the sum is very slightly
    above the true energy; 1% is loose enough for that and tight enough to catch a
    missing or double-counted sub-band.
    """
    signal = np.random.default_rng(42).standard_normal(20000)
    bands = decompose(signal, extractor.settings())
    subband_energy = sum(float(np.sum(c**2)) for c in bands.values())
    assert subband_energy == pytest.approx(float(np.sum(signal**2)), rel=0.01)


def test_extraction_is_deterministic(extractor: WaveletExtractor,
                                     synthetic_signal: np.ndarray):
    first = extractor.extract(synthetic_signal, FS).vector
    second = extractor.extract(synthetic_signal, FS).vector
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# T36.6 / T36.7 -- short-signal level reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_samples", "expected_level"),
    [(20000, 5), (1520, 5), (224, 5), (200, 4), (100, 3), (30, 2), (14, 1), (10, 0)],
)
def test_level_reduces_to_what_the_signal_supports(
    extractor: WaveletExtractor, n_samples: int, expected_level: int
):
    assert usable_level(n_samples, extractor.settings()) == expected_level


def test_a_short_signal_reduces_the_level_and_flags_it_rather_than_failing(
    extractor: WaveletExtractor,
):
    """The T36.7 assertion, verbatim.

    200 samples supports 4 levels, not 5. The four detail bands that exist keep
    their names and their values; ``cD5`` and ``cA5`` become NaN because there is
    no band with those frequency limits in a 4-level decomposition.
    """
    signal = np.random.default_rng(0).standard_normal(200)
    result = extractor.extract(signal, FS, record_uid="short")

    assert result.failed is False, result.error
    assert len(result.values) == EXPECTED_COUNT
    assert "dwt_level_4" in result.flags

    for stat in STATS:
        for band in ("cD1", "cD2", "cD3", "cD4"):
            assert np.isfinite(result.values["dwt_" + band + "_" + stat]), band + " " + stat
        for band in ("cA5", "cD5"):
            assert np.isnan(result.values["dwt_" + band + "_" + stat]), band + " " + stat


def test_a_reduced_level_never_writes_its_approximation_into_the_cA5_slot(
    extractor: WaveletExtractor,
):
    """cA3 covers 0-125 Hz; cA5 covers 0-31.25 Hz. They are not interchangeable."""
    flags: list[str] = []
    bands = decompose(np.random.default_rng(0).standard_normal(200), extractor.settings(), flags)
    assert "cA5" not in bands
    assert not any(name.startswith("cA") for name in bands)
    assert set(bands) == {"cD1", "cD2", "cD3", "cD4"}


def test_edge_case_signals_never_raise(extractor: WaveletExtractor,
                                       edge_case_signals: dict[str, np.ndarray]):
    """Any NaN must be explained by a flag; the same rule as the other families."""
    for label, signal in edge_case_signals.items():
        result = extractor.extract(signal, FS, record_uid=label)
        assert len(result.values) == EXPECTED_COUNT, label

        if result.failed:
            assert result.n_missing == EXPECTED_COUNT, label
            continue

        if result.n_missing:
            assert any(flag.startswith("dwt_") for flag in result.flags), (
                label + ": unexplained NaN; flags=" + str(result.flags)
            )


def test_a_dc_only_recording_does_not_break_the_entropy_histogram(
    extractor: WaveletExtractor,
):
    """Regression: the T36.7 gate found ``np.histogram`` raising on this record.

    A DC-only signal produces a ``cA5`` band that is constant to within rounding
    but not exactly constant -- 54 coefficients spanning 8.9e-16 around -4.525 --
    and 64 bins over that range are narrower than float64 can represent.
    ``shannon_entropy`` now recognises a degenerate range and returns 0, which is
    the correct entropy of an array carrying no information.
    """
    result = extractor.extract(np.full(1520, -0.8), FS, record_uid="dc_only")

    assert result.failed is False, result.error
    assert np.isfinite(result.vector).all()
    for band in SUBBANDS:
        assert result.values["dwt_" + band + "_entropy"] >= 0.0


# ---------------------------------------------------------------------------
# real records -- the gate (T36.7)
# ---------------------------------------------------------------------------


def _preprocessed(path: Path, uid: str) -> np.ndarray:
    from src.preprocessing.pipeline import preprocess

    return preprocess(path, record_uid=uid).signal


@pytest.mark.needs_data
@pytest.mark.parametrize("dataset", ["D1", "D2", "D3", "D4"])
def test_real_records_yield_24_finite_values(
    extractor: WaveletExtractor, sample_records: Any, dataset: str
):
    for uid, path in sample_records(dataset, 3):
        result = extractor.extract(_preprocessed(path, uid), FS, record_uid=uid)
        assert result.failed is False, dataset + " " + uid + ": " + str(result.error)
        assert len(result.values) == EXPECTED_COUNT
        bad = [name for name, value in result.values.items() if not np.isfinite(value)]
        assert bad == [], dataset + " " + uid + " non-finite: " + ", ".join(bad)
        for band in SUBBANDS:
            assert result.values["dwt_" + band + "_energy"] >= 0.0, uid
            assert result.values["dwt_" + band + "_std"] >= 0.0, uid
            assert result.values["dwt_" + band + "_mean_abs"] >= 0.0, uid


@pytest.mark.needs_data
def test_every_real_record_supports_the_full_five_levels(
    extractor: WaveletExtractor, master_frame: Any
):
    """The shortest recording is 1,520 samples, which supports 7 levels.

    So the level-reduction path never fires on this corpus, and a flagged record
    in a real extraction run would mean the master table is wrong about a
    duration -- worth knowing rather than assuming.
    """
    shortest = int(master_frame["n_samples"].min() * 2000 / master_frame.loc[
        master_frame["n_samples"].idxmin(), "original_fs"
    ])
    assert usable_level(shortest, extractor.settings()) == 5


@pytest.mark.needs_data
def test_duration_extremes_extract_cleanly(
    extractor: WaveletExtractor, master_frame: Any, project_root: Path
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
# T36.6 -- FE-08
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_fe08_wavelet_decomposition_is_emitted(tmp_path: Path):
    from src.feature_extraction.figures import (
        FE_FIGURES,
        features_dir,
        plot_wavelet_decomposition,
    )

    first = plot_wavelet_decomposition(tmp_path / "a.png")
    assert first.is_file()
    assert first.stat().st_size > 10_000

    second = plot_wavelet_decomposition(tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes(), "FE-08 is not reproducible"

    committed = features_dir() / FE_FIGURES["FE-08"]
    assert committed.is_file(), "FE-08 has not been generated into outputs/03_features"
