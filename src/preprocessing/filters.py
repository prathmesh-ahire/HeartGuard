"""Butterworth bandpass filtering, 20-400 Hz, zero phase (Phase 24).

The passband is the band a phonocardiogram lives in. Below 20 Hz there is
breathing, body movement and recording-chain baseline drift; above 400 Hz there
is friction, rubbing and electronic noise. S1, S2, murmurs, clicks and gallops
sit between the two.

**Design order 4, effective order 8, doubled again by zero-phase filtering.**
``butter(4, [20, 400], btype="band")`` is a 4th-order *prototype*; the
lowpass-to-bandpass transform doubles it, so the realised filter has 8 poles and
4 second-order sections. :func:`sosfiltfilt` then runs it forwards and backwards,
which squares the magnitude response: -3.01 dB at each cutoff becomes -6.02 dB,
and the roll-off doubles from 80 to 160 dB/decade. Those are the numbers to
quote in the thesis, not "4th-order Butterworth" on its own.

The price of forward-backward filtering is nothing in phase -- exactly zero
group delay, so an S1 peak stays where it was, which matters because Phase 34
measures interval timings on the filtered signal. The price is paid in the
transition width instead, and it is worth it.

**SOS, not b/a.** Transfer-function coefficients for an 8-pole filter with poles
this close to DC (20 Hz at a 2 kHz rate is 0.02 of Nyquist) lose precision badly;
second-order sections do not. This is not hypothetical -- a b/a implementation of
this exact filter is numerically unstable.

**The short-signal guard (T24.3).** ``sosfiltfilt`` needs ``padlen`` samples of
edge padding, 27 for this design. Every record in the corpus clears it easily:
the shortest, at 0.76 s, is 1,520 samples at the 2 kHz target rate. The guard is
kept for the inference API, where an uploaded clip can be any length, and any
record it fires on is flagged rather than silently filtered differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "DEFAULT_LOW_HZ",
    "DEFAULT_HIGH_HZ",
    "DEFAULT_ORDER",
    "DEFAULT_MIN_PADLEN",
    "FilterResult",
    "design_bandpass",
    "default_padlen",
    "bandpass_filter",
    "filter_signal",
    "butterworth_bandpass_magnitude",
    "frequency_response",
    "validate_response",
    "plot_transfer_function",
]

log = get_logger(__name__)

DEFAULT_LOW_HZ = 20.0
DEFAULT_HIGH_HZ = 400.0
DEFAULT_ORDER = 4
DEFAULT_MIN_PADLEN = 12


@dataclass(frozen=True, slots=True)
class FilterResult:
    """A filtered signal and what was done to it.

    ``applied=False`` is the ablation path (PP-09, T24.5) *and* the degenerate
    path -- a signal too short to pad at all comes back untouched with
    ``too_short=True`` rather than raising. The caller can tell the two apart by
    the flags, and Phase 26 records them per record.

    ``order`` is the design prototype order (4). ``effective_order`` is what was
    actually applied: the bandpass transform doubles it to 8 poles and the
    forward-backward pass doubles it again, to 16.
    """

    signal: np.ndarray
    applied: bool
    low_hz: float
    high_hz: float
    order: int
    effective_order: int
    zero_phase: bool
    padlen: int
    padlen_reduced: bool
    too_short: bool

    def as_dict(self) -> dict[str, Any]:
        """The row this result contributes to a per-record report."""
        return {
            "filter_applied": self.applied,
            "filter_low_hz": self.low_hz,
            "filter_high_hz": self.high_hz,
            "filter_order": self.order,
            "filter_effective_order": self.effective_order,
            "filter_zero_phase": self.zero_phase,
            "filter_padlen": self.padlen,
            "filter_padlen_reduced": self.padlen_reduced,
            "filter_too_short": self.too_short,
        }


# ---------------------------------------------------------------------------
# T24.1 -- design
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _design(fs: int, low_hz: float, high_hz: float, order: int) -> Any:
    from scipy.signal import butter

    nyquist = fs / 2.0
    if not 0 < low_hz < high_hz:
        raise ValueError(
            "need 0 < low_hz < high_hz, got " + str(low_hz) + " and " + str(high_hz)
        )
    if high_hz >= nyquist:
        raise ValueError(
            "filter.high_hz (" + str(high_hz) + ") must stay below Nyquist ("
            + str(nyquist) + " Hz at fs " + str(fs) + ")"
        )
    if order < 1:
        raise ValueError("filter order must be at least 1, got " + str(order))

    return butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")


def design_bandpass(
    fs: int,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Second-order sections for the 20-400 Hz Butterworth bandpass.

    Cached: the same four sections are reused for all 7,536 records rather than
    redesigned per record. Returns a copy so a caller cannot mutate the cached
    array.
    """
    return np.array(_design(int(fs), float(low_hz), float(high_hz), int(order)), copy=True)


def default_padlen(sos: np.ndarray) -> int:
    """``sosfiltfilt``'s own default padding length, computed the same way.

    Reproduced here rather than imported because scipy does not export it, and
    the guard in :func:`bandpass_filter` has to know the number it is guarding
    against. 27 samples for the project's design.
    """
    sos = np.asarray(sos)
    n_sections = sos.shape[0]
    ntaps = 2 * n_sections + 1
    ntaps -= int(min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    return 3 * ntaps


# ---------------------------------------------------------------------------
# T24.2 / T24.3 / T24.5 -- application
# ---------------------------------------------------------------------------


def bandpass_filter(
    x: np.ndarray,
    fs: int,
    *,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
    enabled: bool = True,
    zero_phase: bool = True,
    min_padlen: int = DEFAULT_MIN_PADLEN,
) -> FilterResult:
    """Apply the bandpass, zero phase, with the short-signal guard.

    ``enabled=False`` returns the input unchanged with ``applied=False`` -- the
    no-filter arm of the PP-09 ablation (T24.5). A signal shorter than
    ``min_padlen`` edge samples is also returned unchanged, flagged
    ``too_short``: padding a 10-sample clip with 27 reflected samples is
    inventing more signal than it filters.
    """
    from scipy.signal import sosfilt, sosfiltfilt

    samples = np.asarray(x, dtype=np.float64).ravel()
    effective_order = 2 * int(order) * (2 if zero_phase else 1)

    def result(
        signal: np.ndarray,
        *,
        applied: bool,
        padlen: int = 0,
        reduced: bool = False,
        too_short: bool = False,
    ) -> FilterResult:
        return FilterResult(
            signal=np.asarray(signal, dtype=np.float32),
            applied=applied,
            low_hz=float(low_hz),
            high_hz=float(high_hz),
            order=int(order),
            effective_order=effective_order,
            zero_phase=bool(zero_phase),
            padlen=int(padlen),
            padlen_reduced=bool(reduced),
            too_short=bool(too_short),
        )

    if not enabled:
        return result(samples, applied=False)
    if samples.size == 0:
        return result(samples, applied=False, too_short=True)

    sos = design_bandpass(fs, low_hz, high_hz, order)

    if not zero_phase:
        return result(sosfilt(sos, samples), applied=True)

    padlen = default_padlen(sos)
    reduced = False
    if samples.size <= padlen:
        # Reduce rather than refuse: pad with what is available, less one sample.
        padlen = max(0, min(padlen, samples.size - 1))
        reduced = True
        if padlen < min_padlen:
            log.warning(
                "signal of %d samples is too short to filter (padlen %d < %d); "
                "returned unfiltered",
                samples.size,
                padlen,
                min_padlen,
            )
            return result(samples, applied=False, padlen=padlen, reduced=True, too_short=True)

    return result(
        sosfiltfilt(sos, samples, padlen=padlen),
        applied=True,
        padlen=padlen,
        reduced=reduced,
    )


def filter_signal(x: np.ndarray, fs: int, cfg: Any | None = None) -> FilterResult:
    """:func:`bandpass_filter` driven by ``configs/signal.yaml``.

    Honours ``filter.enabled``, which the PP-09 ablation flips (T24.5), and
    ``filter.short_signal_policy``: only ``reduce_padlen`` is implemented, and an
    unknown policy is an error rather than a silent fallback to a different
    filtering behaviour than the config asked for.
    """
    if cfg is None:
        from src.utils.config import load_config

        cfg = load_config("signal")

    policy = str(cfg.get("filter.short_signal_policy", "reduce_padlen"))
    if policy != "reduce_padlen":
        raise ValueError(
            "configs/signal.yaml filter.short_signal_policy " + repr(policy)
            + " is not implemented; only 'reduce_padlen' is"
        )

    return bandpass_filter(
        x,
        fs,
        low_hz=float(cfg.get("filter.low_hz", DEFAULT_LOW_HZ)),
        high_hz=float(cfg.get("filter.high_hz", DEFAULT_HIGH_HZ)),
        order=int(cfg.get("filter.order", DEFAULT_ORDER)),
        enabled=bool(cfg.get("filter.enabled", True)),
        zero_phase=bool(cfg.get("filter.zero_phase", True)),
        min_padlen=int(cfg.get("filter.min_padlen", DEFAULT_MIN_PADLEN)),
    )


# ---------------------------------------------------------------------------
# T24.4 -- response validation and the transfer-function plot
# ---------------------------------------------------------------------------


def butterworth_bandpass_magnitude(
    freqs: np.ndarray,
    fs: int,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """The documented Butterworth magnitude, evaluated analytically.

    The textbook equation is written for a lowpass prototype::

        |H(w)| = 1 / sqrt(1 + (w / wc)^(2N))

    A bandpass substitutes the frequency variable::

        W = (w^2 - w0^2) / (w * BW),   w0 = sqrt(w1 * w2),   BW = w2 - w1

    and the digital filter reaches its cutoffs through the bilinear transform,
    so every frequency is prewarped with ``wa = 2 * fs * tan(pi * f / fs)``
    before the substitution. With the prewarp included this reproduces scipy's
    ``sosfreqz`` to about 1e-13; without it the two disagree by tenths of a dB
    near 400 Hz, which is the sort of mismatch that gets blamed on the filter
    rather than on the comparison. This is the reference T24.4 validates the
    implemented filter against.
    """
    f = np.asarray(freqs, dtype=np.float64)
    nyquist = fs / 2.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        wa = 2.0 * fs * np.tan(np.pi * np.clip(f, 0.0, nyquist) / fs)
        w1 = 2.0 * fs * np.tan(np.pi * low_hz / fs)
        w2 = 2.0 * fs * np.tan(np.pi * high_hz / fs)
        w0_sq = w1 * w2
        bandwidth = w2 - w1
        transformed = (wa * wa - w0_sq) / (wa * bandwidth)
        magnitude = 1.0 / np.sqrt(1.0 + transformed ** (2 * order))

    # DC and Nyquist are exact zeros of a bandpass; the algebra above reaches
    # them as 0/0 and inf/inf respectively.
    magnitude = np.where(np.isfinite(magnitude), magnitude, 0.0)
    magnitude[f <= 0] = 0.0
    magnitude[f >= nyquist] = 0.0
    return magnitude


def frequency_response(
    fs: int = 2000,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
    *,
    n_points: int = 4096,
    zero_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, magnitude)`` of the implemented filter, log-spaced to Nyquist.

    ``zero_phase=True`` squares the magnitude, which is what ``sosfiltfilt``
    actually applies -- reporting the single-pass response of a filter used
    forwards and backwards understates the attenuation by half, in dB.
    """
    from scipy.signal import sosfreqz

    sos = design_bandpass(fs, low_hz, high_hz, order)
    freqs = np.logspace(np.log10(0.5), np.log10(fs / 2.0 - 1e-6), n_points)
    _, response = sosfreqz(sos, worN=freqs, fs=fs)
    magnitude = np.abs(response)
    return freqs, magnitude**2 if zero_phase else magnitude


def validate_response(
    fs: int = 2000,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> dict[str, float]:
    """Compare the implemented filter with the analytic equation (T24.4).

    Returns the maximum absolute deviation over the whole spectrum plus the
    measured gains at the two cutoffs and at the passband centre, single-pass
    and zero-phase. Numbers, not a boolean: the test asserts on them and the
    thesis quotes them.
    """
    from scipy.signal import sosfreqz

    sos = design_bandpass(fs, low_hz, high_hz, order)
    freqs = np.linspace(0.1, fs / 2.0 - 0.1, 8192)
    _, response = sosfreqz(sos, worN=freqs, fs=fs)
    implemented = np.abs(response)
    analytic = butterworth_bandpass_magnitude(freqs, fs, low_hz, high_hz, order)

    centre = float(np.sqrt(low_hz * high_hz))
    edges = np.array([low_hz, centre, high_hz], dtype=np.float64)
    _, edge_response = sosfreqz(sos, worN=edges, fs=fs)
    edge_db = 20.0 * np.log10(np.abs(edge_response))

    return {
        "max_abs_deviation": float(np.max(np.abs(implemented - analytic))),
        "max_db_deviation": float(
            np.max(np.abs(20 * np.log10(implemented + 1e-300) - 20 * np.log10(analytic + 1e-300)))
        ),
        "n_sections": int(sos.shape[0]),
        "padlen": int(default_padlen(sos)),
        "centre_hz": centre,
        "gain_low_cutoff_db": float(edge_db[0]),
        "gain_centre_db": float(edge_db[1]),
        "gain_high_cutoff_db": float(edge_db[2]),
        "zero_phase_gain_low_cutoff_db": float(2 * edge_db[0]),
        "zero_phase_gain_high_cutoff_db": float(2 * edge_db[2]),
    }


def plot_transfer_function(
    path: str | Path | None = None,
    *,
    fs: int = 2000,
    low_hz: float = DEFAULT_LOW_HZ,
    high_hz: float = DEFAULT_HIGH_HZ,
    order: int = DEFAULT_ORDER,
) -> Path:
    """Plot the magnitude response and save it (T24.4).

    Three curves on one axis: the implemented single-pass filter, the analytic
    Butterworth equation over it (they overlie exactly -- that is the point), and
    the zero-phase response that is what actually gets applied.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.io import save_png

    if path is None:
        from src.utils.config import load_config

        path = Path(load_config("paths").require("outputs.preprocessing")) / (
            "filter_transfer_function.png"
        )

    freqs, single = frequency_response(fs, low_hz, high_hz, order, zero_phase=False)
    analytic = butterworth_bandpass_magnitude(freqs, fs, low_hz, high_hz, order)

    def db(values: np.ndarray) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(values, 1e-12))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(freqs, db(single), lw=2.0, label="implemented, single pass (sosfilt)")
    ax.semilogx(freqs, db(analytic), lw=1.0, ls="--", label="Butterworth equation (analytic)")
    ax.semilogx(freqs, db(single**2), lw=1.6, label="zero phase, as applied (sosfiltfilt)")

    for edge in (low_hz, high_hz):
        ax.axvline(edge, color="0.6", lw=0.8, ls=":")
    ax.axhline(-3.0103, color="0.6", lw=0.8, ls=":")
    ax.axhline(-6.0206, color="0.6", lw=0.8, ls=":")

    ax.set_xlim(1, fs / 2)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title(
        "PV-MEPCG bandpass: order "
        + str(order)
        + " Butterworth, "
        + str(int(low_hz))
        + "-"
        + str(int(high_hz))
        + " Hz at "
        + str(fs)
        + " Hz"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower center", fontsize=8)

    return save_png(fig, path)
