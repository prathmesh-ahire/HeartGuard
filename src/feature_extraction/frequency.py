"""Frequency-domain features: the 22 of Phase 33.

Twelve framed spectral statistics (centroid, bandwidth, two rolloffs, flatness
and flux, each as a mean and an SD over frames), four global statistics from a
Welch PSD, and six relative band powers spanning the 20-400 Hz passband.

**Two spectral estimates, on purpose.** The framed statistics come from an STFT
because they describe how the spectrum *moves*: a murmur is a change in the
spectral content between S1 and S2, and a single spectrum averaged over 30
seconds cannot see it. The global statistics come from a Welch PSD because they
describe the recording as a whole, and Welch's segment averaging gives a far
lower-variance estimate of band power than one STFT frame does. Using the STFT
for both would make the band powers noisy; using Welch for both would delete the
time axis the first six features exist to measure.

**Band powers are relative, and that is what makes them comparable.** The signal
is z-normalized, so absolute power carries no information about the recording --
only about the normalizer. Each band is therefore divided by the PSD's total
power over 0-Nyquist, which makes the six a distribution shape and guarantees
they sum to at most 1 (T33.7). They sum to slightly *less* than 1 rather than
exactly 1, and the missing remainder is real: it is the power outside 20-400 Hz
that the bandpass attenuated by 6 dB at the edges rather than removing.

**Band edges are interpolated, not snapped to bins.** At ``nperseg=512`` and
2 kHz the PSD resolution is 3.906 Hz, so no band edge -- 20, 50, 100, 150, 250,
350, 400 -- lands on a bin. Assigning whole bins to bands would either double
count the shared edge bin or drop it, and the direction of the error would
differ per band. Interpolating the PSD at the exact edge frequencies makes the
six bands tile 20-400 Hz exactly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import signal as sp_signal

from src.feature_extraction.base import BaseFeatureExtractor
from src.feature_extraction.registry import feature_names, register_extractor

__all__ = [
    "FrequencyExtractor",
    "FRAMED_NAMES",
    "BANDS_HZ",
    "extract_frequency_features",
    "welch_psd",
    "band_power",
    "spectral_entropy",
    "spectral_flux",
]

FAMILY = "frequency"

#: The 12 framed statistics, in registry column order. Named explicitly rather
#: than sliced off the registry: a slice would silently follow a reordering, and
#: this list is used to blank exactly those columns when a signal is too short
#: for an STFT.
FRAMED_NAMES: tuple[str, ...] = (
    "freq_centroid_mean",
    "freq_centroid_std",
    "freq_bandwidth_mean",
    "freq_bandwidth_std",
    "freq_rolloff85_mean",
    "freq_rolloff85_std",
    "freq_rolloff95_mean",
    "freq_rolloff95_std",
    "freq_flatness_mean",
    "freq_flatness_std",
    "freq_flux_mean",
    "freq_flux_std",
)

#: Band edges in the registry's column order. Cross-checked against
#: ``features.yaml`` on every extraction so a config edit cannot silently
#: relabel a column.
BANDS_HZ: tuple[tuple[int, int], ...] = (
    (20, 50),
    (50, 100),
    (100, 150),
    (150, 250),
    (250, 350),
    (350, 400),
)


# ---------------------------------------------------------------------------
# spectral estimates
# ---------------------------------------------------------------------------


def welch_psd(
    signal: np.ndarray,
    fs: int,
    *,
    nperseg: int = 512,
    noverlap: int = 256,
    window: str = "hann",
    detrend: str = "constant",
    scaling: str = "density",
    flags: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density, guarded for short recordings.

    scipy silently shortens ``nperseg`` to the signal length and warns. Doing it
    explicitly instead means the reduction becomes a flag on the record rather
    than a warning nobody reads in a 7,536-record batch log.
    """
    values = np.asarray(signal, dtype=np.float64)
    segment = int(nperseg)
    if values.size < segment:
        segment = max(8, int(2 ** np.floor(np.log2(max(values.size, 8)))))
        segment = min(segment, values.size)
        if flags is not None:
            flags.append("welch_nperseg_" + str(segment))

    overlap = min(int(noverlap), segment - 1) if segment > 1 else 0
    freqs, psd = sp_signal.welch(
        values,
        fs=fs,
        window=window,
        nperseg=segment,
        noverlap=overlap,
        detrend=detrend,
        scaling=scaling,
    )
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def band_power(freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    """Integrate the PSD over ``[low_hz, high_hz]``, interpolating both edges.

    The edges are added as interpolated points rather than snapped to the nearest
    bin, so adjacent bands tile the passband exactly instead of sharing or
    dropping the boundary bin.
    """
    if freqs.size < 2:
        return 0.0

    low = max(float(low_hz), float(freqs[0]))
    high = min(float(high_hz), float(freqs[-1]))
    if high <= low:
        return 0.0

    interior = freqs[(freqs > low) & (freqs < high)]
    grid = np.concatenate(([low], interior, [high]))
    values = np.interp(grid, freqs, psd)
    return float(np.trapezoid(values, grid))


def spectral_entropy(psd: np.ndarray) -> float:
    """Shannon entropy of the normalized PSD, scaled to [0, 1].

    Normalized by ``log2(n_bins)`` so the number is comparable across records
    whose PSDs have different lengths -- which happens whenever the short-signal
    guard above reduces ``nperseg``.
    """
    total = float(np.sum(psd))
    if not np.isfinite(total) or total <= 0.0 or psd.size < 2:
        return 0.0
    probabilities = psd / total
    probabilities = probabilities[probabilities > 0]
    if probabilities.size < 2:
        return 0.0
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return entropy / float(np.log2(psd.size))


def spectral_flux(magnitude: np.ndarray, flags: list[str] | None = None) -> tuple[float, float]:
    """Mean and SD of the L2 frame-to-frame change in the L1-normalized spectrum.

    Normalizing each frame to sum 1 first is what makes flux a measure of
    spectral *shape* change rather than of loudness change; without it every
    S1 onset dominates the statistic on amplitude alone.
    """
    if magnitude.ndim != 2 or magnitude.shape[1] < 2:
        if flags is not None:
            flags.append("flux_insufficient_frames")
        return float("nan"), float("nan")

    totals = magnitude.sum(axis=0, keepdims=True)
    totals = np.where(totals > 0.0, totals, 1.0)
    normalized = magnitude / totals
    differences = np.diff(normalized, axis=1)
    flux = np.sqrt(np.sum(differences**2, axis=0))
    return float(np.mean(flux)), float(np.std(flux))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(np.mean(array)), float(np.std(array))


# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------


class FrequencyExtractor(BaseFeatureExtractor):
    """The 22 frequency-domain features (T33.1-T33.6)."""

    family = FAMILY
    name = "frequency"

    def __init__(self, cfg: Any | None = None, signal_cfg: Any | None = None) -> None:
        super().__init__(cfg)
        self._signal_cfg = signal_cfg

    @property
    def _features(self) -> Any:
        if self._cfg is None:
            from src.utils.config import load_config

            self._cfg = load_config("features")
        return self._cfg

    @property
    def _signal(self) -> Any:
        if self._signal_cfg is None:
            from src.utils.config import load_config

            self._signal_cfg = load_config("signal")
        return self._signal_cfg

    def feature_names(self) -> tuple[str, ...]:
        return feature_names(FAMILY)

    # -- config -------------------------------------------------------------

    def bands(self) -> tuple[tuple[int, int], ...]:
        """Band edges from config, checked against the registry's column names.

        T33.6's "verify the parameters are fixed in config" is enforced here
        rather than assumed: a config edit that changed the bands would silently
        make ``freq_band_power_100_150`` hold something else, and no number would
        look wrong.
        """
        configured = self._features.get("families.frequency.bands_hz")
        if not configured:
            return BANDS_HZ
        pairs = tuple((int(low), int(high)) for low, high in configured)
        if pairs != BANDS_HZ:
            raise ValueError(
                "features.yaml bands_hz=" + str([list(p) for p in pairs])
                + " does not match the registry column names "
                + str([list(p) for p in BANDS_HZ])
            )
        return pairs

    def _welch_settings(self) -> dict[str, Any]:
        cfg = self._features
        return {
            "nperseg": int(cfg.get("families.frequency.welch.nperseg", 512)),
            "noverlap": int(cfg.get("families.frequency.welch.noverlap", 256)),
            "window": str(cfg.get("families.frequency.welch.window", "hann")),
            "detrend": str(cfg.get("families.frequency.welch.detrend", "constant")),
            "scaling": str(cfg.get("families.frequency.welch.scaling", "density")),
        }

    def _rolloff_percentiles(self) -> Sequence[float]:
        configured = self._features.get("families.frequency.rolloff_percentiles")
        percentiles = tuple(float(p) for p in (configured or (0.85, 0.95)))
        if percentiles != (0.85, 0.95):
            raise ValueError(
                "features.yaml rolloff_percentiles=" + str(list(percentiles))
                + " but the registry names the 85% and 95% rolloffs"
            )
        return percentiles

    # -- the maths ----------------------------------------------------------

    #: Below this many samples (16 ms at 2 kHz) there is no spectrum worth
    #: estimating, and librosa's own guard would only warn and pad. The shortest
    #: real record here is 1,520 samples, so this path exists for the inference
    #: API, where an upload can be any length.
    MIN_STFT_SAMPLES = 32

    def _stft_magnitude(
        self, signal: np.ndarray, flags: list[str]
    ) -> tuple[np.ndarray | None, int]:
        import librosa

        cfg = self._signal
        n_fft = int(cfg.get("framing.frame_length", 512))
        hop_length = int(cfg.get("framing.hop_length", 256))
        window = str(cfg.get("framing.window", "hann"))
        center = bool(cfg.get("framing.center", True))

        if signal.size < self.MIN_STFT_SAMPLES:
            flags.append("stft_signal_too_short")
            return None, n_fft

        if signal.size < n_fft:
            n_fft = int(2 ** np.floor(np.log2(signal.size)))
            hop_length = max(1, n_fft // 2)
            flags.append("stft_n_fft_" + str(n_fft))

        spectrum = librosa.stft(
            y=signal,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=center,
        )
        return np.abs(spectrum), n_fft

    def _compute(
        self, signal: np.ndarray, fs: int, flags: list[str]
    ) -> Mapping[str, float]:
        import librosa

        values: dict[str, float] = {}
        magnitude, n_fft = self._stft_magnitude(signal, flags)

        # --- framed spectral statistics (12), T33.1-T33.3
        # All four librosa calls reuse the one magnitude spectrogram; recomputing
        # the STFT per statistic would be four times the cost for the same numbers.
        if magnitude is None:
            for name in FRAMED_NAMES:
                values[name] = float("nan")
        else:
            centroid = librosa.feature.spectral_centroid(S=magnitude, sr=fs, n_fft=n_fft)
            bandwidth = librosa.feature.spectral_bandwidth(
                S=magnitude, sr=fs, n_fft=n_fft
            )
            flatness = librosa.feature.spectral_flatness(S=magnitude)

            values["freq_centroid_mean"], values["freq_centroid_std"] = _mean_std(centroid)
            values["freq_bandwidth_mean"], values["freq_bandwidth_std"] = _mean_std(
                bandwidth
            )

            for percent in self._rolloff_percentiles():
                rolloff = librosa.feature.spectral_rolloff(
                    S=magnitude, sr=fs, n_fft=n_fft, roll_percent=percent
                )
                key = "freq_rolloff" + str(round(percent * 100))
                values[key + "_mean"], values[key + "_std"] = _mean_std(rolloff)

            values["freq_flatness_mean"], values["freq_flatness_std"] = _mean_std(flatness)
            values["freq_flux_mean"], values["freq_flux_std"] = spectral_flux(
                magnitude, flags
            )

        # --- global statistics from the Welch PSD (4), T33.4
        freqs, psd = welch_psd(signal, fs, flags=flags, **self._welch_settings())
        total_power = float(np.trapezoid(psd, freqs)) if freqs.size > 1 else 0.0

        values["freq_spectral_entropy"] = spectral_entropy(psd)
        if psd.size and np.any(psd > 0.0):
            peak_bin = int(np.argmax(psd))
            values["freq_dominant"] = float(freqs[peak_bin])
            values["freq_peak_power"] = float(psd[peak_bin])
        else:
            flags.append("psd_zero_power")
            values["freq_dominant"] = 0.0
            values["freq_peak_power"] = 0.0
        values["freq_total_power"] = total_power

        # --- relative band power (6), T33.5
        for low, high in self.bands():
            key = "freq_band_power_" + str(low) + "_" + str(high)
            if total_power > 0.0:
                values[key] = band_power(freqs, psd, low, high) / total_power
            else:
                values[key] = 0.0
        if total_power <= 0.0 and "psd_zero_power" not in flags:
            flags.append("psd_zero_power")

        return values


def extract_frequency_features(
    signal: np.ndarray, fs: int, *, record_uid: str | None = None, cfg: Any | None = None
):
    """Convenience wrapper returning a :class:`FamilyResult`."""
    return FrequencyExtractor(cfg).extract(signal, fs, record_uid=record_uid)


# The framed-statistic list and the registry must not drift apart; checked at
# import so a rename is a hard failure rather than a column of NaN.
if feature_names(FAMILY)[: len(FRAMED_NAMES)] != FRAMED_NAMES:
    raise RuntimeError(
        "FRAMED_NAMES no longer matches the head of the frequency family: registry "
        + str(list(feature_names(FAMILY)[: len(FRAMED_NAMES)]))
    )

register_extractor(FrequencyExtractor())
