"""Chroma features: the 24 of Phase 35.

Twelve pitch-class means and twelve pitch-class standard deviations, blocked by
statistic in registry order.

**The physiological caveat (T35.4), stated once and carried into the write-up.**
Chroma is a *musical* construct. It folds the spectrum onto the twelve
pitch classes of the equal-tempered scale, which exist because of Western
tuning, not because of anything in a heart. A phonocardiogram has no tonic, no
key and no harmonic series in the musical sense, so a chroma bin here does not
mean "this recording contains a C". It is defensible only as a **generic
harmonic-distribution descriptor**: a fixed, log-frequency folding of spectral
energy that happens to be sensitive to the harmonic structure a murmur adds.

Both source documents mandate the family, so it is built, and the locked count
of 138 depends on its 24. What must not happen is the write-up implying a
cardiac rationale for it. Any thesis or paper sentence about these features has
to say what this docstring says.

**Adapted to 20-400 Hz by band-limiting the spectrogram (T35.1).**
``librosa.feature.chroma_stft`` has no ``fmin``/``fmax``: it maps every FFT bin
to a pitch class, including the 400-1000 Hz region the bandpass already
attenuated and the sub-20 Hz region that holds movement and contact artifacts.
Feeding it the full spectrum would fold that residue into the same twelve bins
as the signal. The magnitude spectrogram is therefore zeroed outside the
passband before the chroma mapping runs, so the twelve bins describe only the
band the preprocessing actually produced.

Band-limiting is a strong filter, not a perfect one, and the reason is worth
knowing: a 512-point Hann window has sidelobes, so an out-of-band tone deposits
a little energy into bins *inside* 20-400 Hz before any masking runs, and zeroing
out-of-band bins cannot remove energy that has already leaked into in-band ones.
Measured against unmasked chroma, the masking still cuts contamination from a
700 Hz tone by about 300x and from a 3 Hz drift by about 8x -- and in the real
pipeline the bandpass has already attenuated both before extraction begins, so
this is a second line of defence rather than the only one.

``tuning`` is fixed at 0.0 rather than estimated. librosa's estimator looks for
a stable tonal peak to calibrate against; on a PCG there is none, so the
estimate is noise and would make the same recording map to different bins on
different runs -- which rule 5 forbids outright.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.feature_extraction.base import BaseFeatureExtractor
from src.feature_extraction.registry import feature_names, register_extractor

__all__ = [
    "ChromaExtractor",
    "ChromaSettings",
    "extract_chroma_features",
    "chroma_matrix",
]

FAMILY = "chroma"

#: Below this many samples there is no spectrum worth folding into pitch classes.
MIN_CHROMA_SAMPLES = 32


class ChromaSettings:
    """Chroma and framing parameters resolved from config once."""

    __slots__ = ("n_chroma", "n_fft", "hop_length", "fmin", "fmax", "tuning", "window", "center")

    def __init__(self, features_cfg: Any, signal_cfg: Any) -> None:
        self.n_chroma = int(features_cfg.get("families.chroma.n_chroma", 12))
        self.n_fft = int(features_cfg.get("families.chroma.n_fft", 512))
        self.hop_length = int(features_cfg.get("families.chroma.hop_length", 256))
        self.fmin = float(features_cfg.get("families.chroma.fmin", 20))
        self.tuning = float(features_cfg.get("families.chroma.tuning", 0.0))
        # The chroma section declares fmin but no fmax; the upper edge is the
        # bandpass's own, so the fold covers exactly the band preprocessing kept.
        self.fmax = float(signal_cfg.get("filter.high_hz", 400))
        self.window = str(signal_cfg.get("framing.window", "hann"))
        self.center = bool(signal_cfg.get("framing.center", True))


def chroma_matrix(
    signal: np.ndarray,
    fs: int,
    settings: ChromaSettings,
    flags: list[str] | None = None,
) -> np.ndarray:
    """``(n_chroma, n_frames)`` chroma over a 20-400 Hz band-limited STFT (T35.1)."""
    import librosa

    values = np.asarray(signal, dtype=np.float64)
    if values.size < MIN_CHROMA_SAMPLES:
        if flags is not None:
            flags.append("chroma_signal_too_short")
        return np.empty((settings.n_chroma, 0), dtype=np.float64)

    n_fft = settings.n_fft
    hop_length = settings.hop_length
    if values.size < n_fft:
        n_fft = int(2 ** np.floor(np.log2(values.size)))
        hop_length = max(1, n_fft // 2)
        if flags is not None:
            flags.append("chroma_n_fft_" + str(n_fft))

    magnitude = np.abs(
        librosa.stft(
            y=values,
            n_fft=n_fft,
            hop_length=hop_length,
            window=settings.window,
            center=settings.center,
        )
    )

    frequencies = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    out_of_band = (frequencies < settings.fmin) | (frequencies > settings.fmax)
    magnitude[out_of_band, :] = 0.0

    if not np.any(magnitude > 0.0):
        # Nothing in the passband: every frame would normalize 0/0. Report the
        # gap rather than emitting twelve confident zeros.
        if flags is not None:
            flags.append("chroma_no_in_band_energy")
        return np.empty((settings.n_chroma, 0), dtype=np.float64)

    chroma = librosa.feature.chroma_stft(
        S=magnitude,
        sr=fs,
        n_fft=n_fft,
        n_chroma=settings.n_chroma,
        tuning=settings.tuning,
    )
    return np.asarray(chroma, dtype=np.float64)


class ChromaExtractor(BaseFeatureExtractor):
    """The 24 chroma features (T35.1-T35.3)."""

    family = FAMILY
    name = "chroma"

    def __init__(self, cfg: Any | None = None, signal_cfg: Any | None = None) -> None:
        super().__init__(cfg)
        self._signal_cfg = signal_cfg
        self._settings: ChromaSettings | None = None

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

    def settings(self) -> ChromaSettings:
        if self._settings is None:
            self._settings = ChromaSettings(self._features, self._signal)
        return self._settings

    def feature_names(self) -> tuple[str, ...]:
        return feature_names(FAMILY)

    def _compute(
        self, signal: np.ndarray, fs: int, flags: list[str]
    ) -> Mapping[str, float]:
        settings = self.settings()
        names = self.feature_names()

        chroma = chroma_matrix(signal, fs, settings, flags)
        if chroma.shape[1] == 0:
            return dict.fromkeys(names, float("nan"))

        values: dict[str, float] = {}
        for index in range(settings.n_chroma):
            key = "chroma_" + str(index + 1).zfill(2)
            row = chroma[index]
            values[key + "_mean"] = float(np.mean(row))
            values[key + "_std"] = float(np.std(row)) if row.size > 1 else 0.0
        return values


def extract_chroma_features(
    signal: np.ndarray, fs: int, *, record_uid: str | None = None, cfg: Any | None = None
):
    """Convenience wrapper returning a :class:`FamilyResult`."""
    return ChromaExtractor(cfg).extract(signal, fs, record_uid=record_uid)


register_extractor(ChromaExtractor())
