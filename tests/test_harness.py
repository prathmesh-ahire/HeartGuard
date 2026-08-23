"""Test-harness self-checks (Phase 06).

Verifies the harness itself: that the fixtures produce what they claim, that the
synthetic generator is deterministic, and that the ``slow`` and ``needs_data``
markers exist and behave.

The two marked tests at the bottom are the ones T06.7 observes. Without at least
one test carrying each marker there is nothing to watch skip, and an unexercised
skip path is indistinguishable from a broken one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.make_synthetic_pcg import (
    DEFAULT_FS,
    duration_extremes,
    make_synthetic_pcg,
)


# ---------------------------------------------------------------------------
# synthetic generator (T06.3)
# ---------------------------------------------------------------------------


def test_synthetic_pcg_shape_and_dtype(synthetic_pcg):
    assert synthetic_pcg.fs == DEFAULT_FS
    assert synthetic_pcg.n_samples == 5 * DEFAULT_FS
    assert synthetic_pcg.signal.dtype == np.float32
    assert synthetic_pcg.signal.ndim == 1


def test_synthetic_pcg_is_finite_and_bounded(synthetic_pcg):
    sig = synthetic_pcg.signal
    assert np.isfinite(sig).all()
    assert np.abs(sig).max() <= 1.0
    assert np.abs(sig).max() > 0.5, "signal is suspiciously quiet"


def test_synthetic_pcg_is_deterministic():
    """Bit-identical across calls -- a test comparing two runs compares the code."""
    a = make_synthetic_pcg(3.0, DEFAULT_FS, 72.0)
    b = make_synthetic_pcg(3.0, DEFAULT_FS, 72.0)
    assert np.array_equal(a.signal, b.signal)
    assert np.array_equal(a.s1_times, b.s1_times)


def test_different_seeds_give_different_signals():
    """Guards against the generator ignoring its seed entirely."""
    a = make_synthetic_pcg(3.0, seed=42)
    b = make_synthetic_pcg(3.0, seed=7)
    assert not np.array_equal(a.signal, b.signal)


def test_beat_count_matches_heart_rate():
    pcg = make_synthetic_pcg(duration_sec=10.0, heart_rate_bpm=60.0, noise_level=0.0)
    assert pcg.n_beats == 10
    gaps = np.diff(pcg.s1_times)
    assert np.allclose(gaps, 1.0, atol=1e-3)


def test_systole_is_shorter_than_diastole():
    """The asymmetry envelope-based heart-rate estimators key on."""
    pcg = make_synthetic_pcg(duration_sec=6.0, heart_rate_bpm=60.0, noise_level=0.0)
    systole = pcg.s2_times[0] - pcg.s1_times[0]
    diastole = pcg.s1_times[1] - pcg.s2_times[0]
    assert 0 < systole < diastole


def test_energy_concentrated_in_the_pcg_band():
    """Most energy must sit in 20-400 Hz, or filter tests prove nothing."""
    pcg = make_synthetic_pcg(duration_sec=5.0, noise_level=0.0)
    spectrum = np.abs(np.fft.rfft(pcg.signal.astype(np.float64)))
    freqs = np.fft.rfftfreq(pcg.n_samples, 1 / pcg.fs)
    in_band = spectrum[(freqs >= 20) & (freqs <= 400)].sum()
    assert in_band / spectrum.sum() > 0.8


def test_generator_rejects_invalid_arguments():
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            make_synthetic_pcg(duration_sec=bad)
    with pytest.raises(ValueError):
        make_synthetic_pcg(3.0, fs=0)
    with pytest.raises(ValueError):
        make_synthetic_pcg(3.0, systole_fraction=1.5)


# ---------------------------------------------------------------------------
# edge cases -- the awkward real durations
# ---------------------------------------------------------------------------


def test_edge_case_signals_are_all_present(edge_case_signals):
    expected = {
        "shortest_real", "sub_cycle", "silence", "constant",
        "clipped", "dc_only", "single_sample", "impulse",
    }
    assert set(edge_case_signals) == expected
    for name, sig in edge_case_signals.items():
        assert sig.ndim == 1 and sig.size >= 1, name
        assert np.isfinite(sig).all(), name


def test_shortest_real_matches_the_audited_minimum(edge_case_signals):
    """0.76 s at 2 kHz = 1,520 samples -- below what a 5-level db4 wants."""
    assert edge_case_signals["shortest_real"].size == int(round(0.76 * DEFAULT_FS))
    assert duration_extremes()["min_sec"] == 0.76
    assert duration_extremes()["max_sec"] == 122.0


def test_degenerate_signals_have_the_degeneracy_they_claim(edge_case_signals):
    assert edge_case_signals["silence"].std() == 0
    assert edge_case_signals["constant"].std() == 0
    assert edge_case_signals["dc_only"].mean() != 0
    assert edge_case_signals["single_sample"].size == 1
    assert np.count_nonzero(edge_case_signals["impulse"]) == 1


def test_long_signal_is_buildable_without_blowing_up():
    """122 s at 2 kHz = 244,000 samples. O(n^2) features need a subsample cap."""
    pcg = make_synthetic_pcg(duration_sec=122.0, noise_level=0.0)
    assert pcg.n_samples == 244_000
    assert np.isfinite(pcg.signal).all()


# ---------------------------------------------------------------------------
# fixtures (T06.2)
# ---------------------------------------------------------------------------


def test_tmp_output_dir_mirrors_the_real_layout(tmp_output_dir):
    assert tmp_output_dir.is_dir()
    for section in ("00_evidence_index", "02_preprocessing", "logs", "configs"):
        assert (tmp_output_dir / section).is_dir()


def test_tmp_output_dir_is_not_the_real_outputs(tmp_output_dir, project_root):
    """No test may write into the project's actual deliverables."""
    assert tmp_output_dir != project_root / "outputs"
    assert project_root not in tmp_output_dir.parents


def test_config_fixture_loads_all_five(config):
    assert set(config) == {"paths", "signal", "features", "models", "experiments"}
    assert config["signal"].get("resample.target_fs") == 2000
    assert config["features"].get("expected_total") == 138


def test_seed_fixture_is_autouse_and_resets_between_tests():
    """This value must be identical in the sibling test below."""
    import random

    assert random.random() == pytest.approx(0.6394267984578837, abs=1e-12)


def test_seed_fixture_gives_the_same_value_in_a_different_test():
    import random

    assert random.random() == pytest.approx(0.6394267984578837, abs=1e-12)


def test_rng_fixture_is_seeded(rng):
    assert np.allclose(rng.normal(size=2), np.random.default_rng(42).normal(size=2))


# ---------------------------------------------------------------------------
# marker behaviour -- what T06.7 observes
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_marker_slow_is_skipped_by_default():
    """Skipped unless --runslow. Deliberately trivial: this exists to be watched,
    not to assert anything about the pipeline."""
    assert True


@pytest.mark.needs_data
def test_marker_needs_data_reads_the_real_tree(dataset_root: Path):
    """Skipped when dataset/ is absent (CI) or suppressed (--no-data)."""
    assert dataset_root.is_dir()
    for family in ("archive", "archive (2)", "archive (3)", "Heartbeat_Sound"):
        assert (dataset_root / family).is_dir(), f"missing dataset family: {family}"


@pytest.mark.needs_data
def test_marker_needs_data_second_case(paths_config):
    """A second data test, so the skip count in the gate is unambiguous."""
    circor = Path(paths_config.require("dataset.d4_circor.demographics_csv"))
    assert circor.is_file()
