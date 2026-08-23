"""Deterministic synthetic PCG generator (Phase 06, task T06.3).

Produces a heart-sound-shaped signal: S1 and S2 bursts at a given heart rate,
separated by a systole shorter than the following diastole, plus band-limited
noise. Given the same arguments it returns bit-identical output, so a test that
compares two runs is comparing the code and not the RNG.

**What this is for, and what it is NOT for.**

It is for unit tests of shape, dtype, finiteness and determinism -- "does the
extractor return 138 finite values in a stable order", "does the filter attenuate
5 Hz", "does the pipeline produce identical output twice".

It is **not** evidence that anything works on real recordings. From CLAUDE.md:
*a test proving the extractor returns 138 finite values from a synthetic signal
says nothing about a 0.76-second PASCAL B recording or a 122-second PhysioNet
one.* Both exist in this corpus. Every data-dependent behaviour needs a
``needs_data`` test over the real duration extremes as well.

To make the extremes cheap to reach in tests, :func:`duration_extremes` returns
the real minimum and maximum durations found in the audit, and
:func:`make_edge_case_signals` builds the awkward cases -- too short for a
5-level DWT, silent, constant, clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SyntheticPCG",
    "make_synthetic_pcg",
    "make_edge_case_signals",
    "duration_extremes",
    "DEFAULT_FS",
]

DEFAULT_FS = 2000        # the project target rate
DEFAULT_SEED = 42

# Audited real extremes (Docs/note.md, 2026-08-22). PASCAL B's 0.76 s is below
# what a 5-level db4 decomposition or an MFCC delta window wants; PhysioNet's
# 122 s is long enough that O(n^2) features need a subsample cap.
REAL_MIN_DURATION_SEC = 0.76
REAL_MAX_DURATION_SEC = 122.0


@dataclass(frozen=True, slots=True)
class SyntheticPCG:
    """A generated signal plus the ground truth used to build it."""

    signal: np.ndarray
    fs: int
    duration_sec: float
    heart_rate_bpm: float
    s1_times: np.ndarray      # seconds
    s2_times: np.ndarray      # seconds
    noise_level: float
    seed: int

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    @property
    def n_beats(self) -> int:
        return int(self.s1_times.size)


def _burst(
    n: int,
    fs: int,
    centre_hz: float,
    width_sec: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A Gaussian-windowed tone -- the standard first-order model of S1/S2."""
    t = np.arange(n, dtype=np.float64) / fs
    centre = t[-1] / 2 if n > 1 else 0.0
    envelope = np.exp(-0.5 * ((t - centre) / (width_sec / 4.0)) ** 2)
    phase = rng.uniform(0, 2 * np.pi)
    return envelope * np.sin(2 * np.pi * centre_hz * t + phase)


def make_synthetic_pcg(
    duration_sec: float = 5.0,
    fs: int = DEFAULT_FS,
    heart_rate_bpm: float = 72.0,
    *,
    noise_level: float = 0.05,
    s1_freq_hz: float = 50.0,
    s2_freq_hz: float = 75.0,
    s1_width_sec: float = 0.12,
    s2_width_sec: float = 0.08,
    systole_fraction: float = 0.35,
    seed: int = DEFAULT_SEED,
    dtype: type = np.float32,
) -> SyntheticPCG:
    """Generate a deterministic synthetic PCG.

    S2 is placed at ``systole_fraction`` of the cycle after S1, so systole is
    shorter than diastole -- the asymmetry envelope-based heart-rate estimators
    key on. S1 is lower in frequency and longer than S2, as in real recordings.

    ``duration_sec`` may be shorter than one cycle; the result is simply a
    truncated signal, which is the point when testing 0.76-second records.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive, got " + repr(duration_sec))
    if fs <= 0:
        raise ValueError("fs must be positive, got " + repr(fs))
    if not 0 < systole_fraction < 1:
        raise ValueError("systole_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n = round(duration_sec * fs)
    signal = np.zeros(n, dtype=np.float64)

    cycle_sec = 60.0 / heart_rate_bpm
    s1_n = max(1, round(s1_width_sec * fs))
    s2_n = max(1, round(s2_width_sec * fs))
    s1_kernel = _burst(s1_n, fs, s1_freq_hz, s1_width_sec, rng)
    s2_kernel = _burst(s2_n, fs, s2_freq_hz, s2_width_sec, rng)

    s1_times: list[float] = []
    s2_times: list[float] = []

    beat = 0
    while True:
        start = beat * cycle_sec
        if start * fs >= n:
            break

        for offset, kernel, times, amplitude in (
            (0.0, s1_kernel, s1_times, 1.0),
            (systole_fraction * cycle_sec, s2_kernel, s2_times, 0.7),
        ):
            idx = round((start + offset) * fs)
            if idx >= n:
                continue
            end = min(idx + kernel.size, n)
            signal[idx:end] += amplitude * kernel[: end - idx]
            times.append(idx / fs)

        beat += 1

    if noise_level > 0:
        signal += rng.normal(0.0, noise_level, size=n)

    peak = float(np.max(np.abs(signal))) if n else 0.0
    if peak > 0:
        signal = signal / peak * 0.9

    return SyntheticPCG(
        signal=signal.astype(dtype),
        fs=fs,
        duration_sec=n / fs,
        heart_rate_bpm=heart_rate_bpm,
        s1_times=np.asarray(s1_times, dtype=np.float64),
        s2_times=np.asarray(s2_times, dtype=np.float64),
        noise_level=noise_level,
        seed=seed,
    )


def duration_extremes() -> dict[str, float]:
    """The real duration extremes in this corpus, for edge-case tests."""
    return {
        "min_sec": REAL_MIN_DURATION_SEC,   # PASCAL set_b
        "max_sec": REAL_MAX_DURATION_SEC,   # PhysioNet
        "typical_sec": 20.83,               # PhysioNet median
    }


def make_edge_case_signals(fs: int = DEFAULT_FS) -> dict[str, np.ndarray]:
    """Degenerate and boundary signals every extractor must survive.

    These are the cases that make an extractor raise instead of returning NaN,
    or return a silently wrong value: a signal too short for a 5-level DWT, a
    constant signal whose z-score normalization divides by zero, a clipped
    signal, a pure DC offset.
    """
    n_short = round(REAL_MIN_DURATION_SEC * fs)
    return {
        # 0.76 s -- the shortest real record. Below what a 5-level db4
        # decomposition or an MFCC delta window wants.
        "shortest_real": make_synthetic_pcg(REAL_MIN_DURATION_SEC, fs).signal,
        # Sub-cycle: not even one complete heartbeat.
        "sub_cycle": make_synthetic_pcg(0.3, fs).signal,
        # Silence -- z-score normalization divides by ~0 here.
        "silence": np.zeros(n_short, dtype=np.float32),
        # Constant non-zero -- zero variance with a non-zero mean.
        "constant": np.full(n_short, 0.5, dtype=np.float32),
        # Fully clipped square wave.
        "clipped": np.sign(
            make_synthetic_pcg(REAL_MIN_DURATION_SEC, fs).signal
        ).astype(np.float32),
        # DC offset with no signal.
        "dc_only": np.full(n_short, -0.8, dtype=np.float32),
        # Single sample -- the absolute floor.
        "single_sample": np.array([0.5], dtype=np.float32),
        # Impulse: all energy in one sample.
        "impulse": np.concatenate(
            [np.zeros(n_short // 2, dtype=np.float32),
             np.array([1.0], dtype=np.float32),
             np.zeros(n_short - n_short // 2 - 1, dtype=np.float32)]
        ),
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    pcg = make_synthetic_pcg()
    print("synthetic PCG:", pcg.n_samples, "samples @", pcg.fs, "Hz")
    print("  duration:", round(pcg.duration_sec, 3), "s")
    print("  beats:", pcg.n_beats, "at", pcg.heart_rate_bpm, "bpm")
    print("  range: [", round(float(pcg.signal.min()), 3), ",",
          round(float(pcg.signal.max()), 3), "]")
    print("  edge cases:", ", ".join(make_edge_case_signals()))
