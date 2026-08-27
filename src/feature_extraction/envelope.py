"""Envelope features: the 5 of Phase 37.

The Hilbert analytic envelope, smoothed with a 20 Hz low-pass, described by its
mean, standard deviation, skewness, kurtosis and peak rate.

**Why the envelope at all.** Everything else in the 138 describes the *content*
of the signal -- its spectrum, its coefficients, its regularity. The envelope
describes its *shape in time*: the alternation of loud S1/S2 bursts with quiet
systolic and diastolic gaps. A murmur fills one of those gaps, which lifts the
envelope between the bursts and lowers its kurtosis. That is the one place in
this feature set where a cardiac mechanism maps directly onto a number.

**``env_peak_rate`` counts heart *sounds*, not heart *beats* -- read the
threshold accordingly.** A normal cycle produces two envelope peaks, S1 and S2,
and the configured 0.25 s minimum peak distance is short enough to resolve both.
So a 72 bpm recording yields roughly 2.4 peaks per second, not 1.2. The
``plausible_peak_rate_hz`` range in ``features.yaml`` is annotated "36-210 bpm",
which is the reading you would get if this counted beats. It does not. The
flag it drives is a *plausibility* check on the sound rate, not a heart-rate
estimate, and no write-up may convert this feature to bpm by multiplying by 60.
See Docs/note.md, 2026-08-27.

**``min_distance_sec`` is a floor on S1-S2 separation, and it has a cost.** Two
sounds closer than the floor are merged into one peak, so a recording whose
systole is shorter than that measures one sound per beat instead of two. The
*effective* floor is ~0.27 s rather than the nominal 0.25 s, because the 20 Hz
low-pass shifts envelope maxima away from the burst centres: on synthetic PCGs
the detector resolves both sounds at an S1-S2 gap of 0.2765 s and merges them at
0.2625 s. This is the
price of suppressing the residual carrier ripple, and it matters most for
CirCor, which is ~98% paediatric: a genuine 150 bpm with a ~0.2 s systole would
merge. Measured on the real corpus it does not appear to bite -- median rates are
2.50 (D1), 2.44 (D2), 2.74 (D3) and 2.39 (D4) sounds per second, all consistent
with both sounds resolved -- but an unexpectedly low ``env_peak_rate`` on a fast
record is this guard, not a bug. See Docs/note.md, 2026-08-27.

**The low-pass is what makes the peak count mean anything.** A raw Hilbert
envelope of a bandpassed PCG is dense with local maxima from the 20-400 Hz
carrier; peak-picking it counts oscillations, not heart sounds. A zero-phase
20 Hz Butterworth removes that carrier while leaving the burst structure intact,
and zero-phase matters because a lag would move every peak and bias nothing
observable -- it would just be wrong in a way no test would see.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats

__all__ = [
    "EnvelopeExtractor",
    "EnvelopeSettings",
    "extract_envelope_features",
    "analytic_envelope",
    "smooth_envelope",
    "envelope_peaks",
]

from src.feature_extraction.base import BaseFeatureExtractor
from src.feature_extraction.registry import feature_names, register_extractor

FAMILY = "envelope"

#: Butterworth order for the smoothing low-pass. Not in ``features.yaml``, which
#: declares only the cutoff; matched to ``signal.yaml``'s bandpass order so the
#: two filters in this project share one design convention.
SMOOTH_ORDER = 4


class EnvelopeSettings:
    """Envelope parameters resolved from config once."""

    __slots__ = (
        "method",
        "lowpass_hz",
        "min_distance_sec",
        "prominence_factor",
        "plausible_low_hz",
        "plausible_high_hz",
    )

    def __init__(self, features_cfg: Any) -> None:
        self.method = str(features_cfg.get("families.envelope.method", "hilbert"))
        if self.method != "hilbert":
            raise ValueError(
                "features.yaml envelope.method=" + self.method
                + " but only 'hilbert' is implemented (T37.1)"
            )
        self.lowpass_hz = float(features_cfg.get("families.envelope.smooth_lowpass_hz", 20))
        self.min_distance_sec = float(
            features_cfg.get("families.envelope.peak_detection.min_distance_sec", 0.25)
        )
        self.prominence_factor = float(
            features_cfg.get("families.envelope.peak_detection.prominence_factor", 0.3)
        )
        low, high = features_cfg.get("families.envelope.plausible_peak_rate_hz") or (0.6, 3.5)
        self.plausible_low_hz = float(low)
        self.plausible_high_hz = float(high)


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    """``|hilbert(x)|`` -- the instantaneous amplitude envelope (T37.1)."""
    values = np.asarray(signal, dtype=np.float64)
    if values.size < 2:
        return np.abs(values)
    return np.abs(sp_signal.hilbert(values))


def smooth_envelope(
    envelope: np.ndarray,
    fs: int,
    cutoff_hz: float,
    *,
    order: int = SMOOTH_ORDER,
    flags: list[str] | None = None,
) -> np.ndarray:
    """Zero-phase low-pass the envelope (T37.2).

    Returns the input unchanged, flagged, when the signal is too short for the
    filter's edge padding or the cutoff is not below Nyquist -- degrading the
    smoothing is preferable to refusing the record, and the flag says which
    happened.
    """
    values = np.asarray(envelope, dtype=np.float64)
    nyquist = fs / 2.0
    if not 0.0 < cutoff_hz < nyquist:
        if flags is not None:
            flags.append("envelope_lowpass_skipped_cutoff")
        return values

    sos = sp_signal.butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if values.size <= padlen:
        if flags is not None:
            flags.append("envelope_lowpass_skipped_short")
        return values

    return np.asarray(sp_signal.sosfiltfilt(sos, values), dtype=np.float64)


def envelope_peaks(
    envelope: np.ndarray, fs: int, settings: EnvelopeSettings
) -> np.ndarray:
    """Indices of envelope peaks under the configured distance and prominence (T37.4).

    Prominence is scaled to the envelope's own standard deviation rather than
    fixed, because the signal is z-normalized upstream: an absolute prominence
    threshold would mean something different for every recording.
    """
    values = np.asarray(envelope, dtype=np.float64)
    if values.size < 3:
        return np.empty(0, dtype=int)

    spread = float(np.std(values))
    prominence = settings.prominence_factor * spread if spread > 0.0 else None
    distance = max(1, round(settings.min_distance_sec * fs))

    peaks, _properties = sp_signal.find_peaks(
        values, distance=distance, prominence=prominence
    )
    return np.asarray(peaks, dtype=int)


class EnvelopeExtractor(BaseFeatureExtractor):
    """The 5 envelope features (T37.1-T37.5)."""

    family = FAMILY
    name = "envelope"

    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__(cfg)
        self._settings: EnvelopeSettings | None = None

    @property
    def _features(self) -> Any:
        if self._cfg is None:
            from src.utils.config import load_config

            self._cfg = load_config("features")
        return self._cfg

    def settings(self) -> EnvelopeSettings:
        if self._settings is None:
            self._settings = EnvelopeSettings(self._features)
        return self._settings

    def feature_names(self) -> tuple[str, ...]:
        return feature_names(FAMILY)

    def envelope_of(
        self, signal: np.ndarray, fs: int, flags: list[str] | None = None
    ) -> np.ndarray:
        """The smoothed analytic envelope, as the features and FE-09 both see it."""
        settings = self.settings()
        raw = analytic_envelope(np.asarray(signal, dtype=np.float64))
        return smooth_envelope(raw, fs, settings.lowpass_hz, flags=flags)

    def _compute(
        self, signal: np.ndarray, fs: int, flags: list[str]
    ) -> Mapping[str, float]:
        settings = self.settings()
        envelope = self.envelope_of(signal, fs, flags)

        values: dict[str, float] = {}
        values["env_mean"] = float(np.mean(envelope))
        values["env_std"] = float(np.std(envelope))

        # A flat envelope has no shape to describe; 0 is the answer, not NaN.
        #
        # The test is relative, not `std == 0`. A DC-only recording produces an
        # envelope that is constant to within rounding but not exactly constant,
        # and scipy's skew/kurtosis then hit catastrophic cancellation and return
        # NaN with a RuntimeWarning -- an unexplained NaN, which the family gates
        # forbid. Comparing the spread against the float spacing at the data's
        # own magnitude catches that, and is the same guard shape used by
        # `time_domain.shannon_entropy` for the same underlying reason.
        magnitude = float(np.max(np.abs(envelope))) if envelope.size else 0.0
        degenerate = values["env_std"] <= float(np.spacing(magnitude)) * envelope.size
        if envelope.size < 4 or degenerate:
            flags.append("envelope_shape_undefined")
            values["env_skew"] = 0.0
            values["env_kurtosis"] = 0.0
        else:
            values["env_skew"] = float(sp_stats.skew(envelope, bias=False))
            values["env_kurtosis"] = float(
                sp_stats.kurtosis(envelope, fisher=True, bias=False)
            )

        duration = envelope.size / float(fs)
        peaks = envelope_peaks(envelope, fs, settings)
        rate = float(peaks.size) / duration if duration > 0.0 else 0.0
        values["env_peak_rate"] = rate

        # T37.5 -- flagged, never clipped. The value stands as measured; the flag
        # is what a later robustness split can group on.
        if not settings.plausible_low_hz <= rate <= settings.plausible_high_hz:
            flags.append("envelope_peak_rate_implausible")

        return values


def extract_envelope_features(
    signal: np.ndarray, fs: int, *, record_uid: str | None = None, cfg: Any | None = None
):
    """Convenience wrapper returning a :class:`FamilyResult`."""
    return EnvelopeExtractor(cfg).extract(signal, fs, record_uid=record_uid)


register_extractor(EnvelopeExtractor())
