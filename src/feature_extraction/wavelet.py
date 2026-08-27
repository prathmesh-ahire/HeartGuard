"""Discrete wavelet features: the 24 of Phase 36.

Six sub-bands from a 5-level ``db4`` decomposition -- ``cA5, cD5, cD4, cD3, cD2,
cD1`` -- each described by energy, standard deviation, Shannon entropy and mean
absolute coefficient. Blocked by statistic in registry order, so an
energy-across-sub-bands slice (the physically meaningful one) is contiguous.

**What the six sub-bands actually cover.** At the 2 kHz target rate each level
halves the band, so the decomposition is a dyadic filterbank::

    cD1   500-1000 Hz     cD4   62.5-125 Hz
    cD2   250-500  Hz     cD5   31.25-62.5 Hz
    cD3   125-250  Hz     cA5   0-31.25 Hz

The dyadic split does not line up with the 20-400 Hz passband, and the mismatch
is worth stating precisely rather than approximately. Only ``cD1`` (500-1000 Hz)
lies wholly outside it. ``cD2`` straddles the upper edge -- 250-400 Hz is kept,
400-500 Hz was attenuated -- and ``cA5`` straddles the lower one, holding the
sub-20 Hz drift region *and* the 20-31.25 Hz band where the loudest part of a
heart sound sits.

Measured over 120 real records (30 per dataset, 2026-08-27), the median share of
total decomposition energy runs::

    cD5 27.8%   cD4 26.3%   cD3 16.0%   cA5 6.7%   cD2 3.9%   cD1 0.2%

So the informative bands are ``cD5``, ``cD4`` and ``cD3`` -- 31-250 Hz, which is
where heart sounds live -- and only ``cD1`` is empty enough to expect in Phase
41's near-zero-variance report. ``cA5`` is not: an earlier draft of this
docstring wrote it off as "sub-20 Hz drift", which the numbers above contradict.

**Short recordings reduce the level and report which bands are missing (T36.6).**
``db4`` has filter length 8, so a 5-level decomposition needs roughly
``7 * 2^5 = 224`` samples. Every record in this corpus clears that easily -- the
shortest is 1,520 samples, which supports 7 levels -- so this path exists for the
inference API rather than for the corpus.

When the level *is* reduced to ``L``, the detail bands ``cD1..cDL`` still mean
exactly what their names say: ``cD1`` is fs/4-fs/2 whatever the depth. The
remaining detail slots have no counterpart and become flagged NaN. **``cA5`` also
becomes NaN**, because ``cA3`` is not a shorter ``cA5`` -- it covers 0-125 Hz
rather than 0-31.25 Hz, and writing it into the ``cA5`` column would put a
different frequency band under that name for those records only. A reported gap
is recoverable; a silently mislabelled band is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.feature_extraction.base import BaseFeatureExtractor
from src.feature_extraction.registry import feature_names, register_extractor
from src.feature_extraction.time_domain import shannon_entropy

__all__ = [
    "WaveletExtractor",
    "WaveletSettings",
    "extract_wavelet_features",
    "decompose",
    "usable_level",
]

FAMILY = "dwt"

#: Sub-band names in registry order, coarsest first.
SUBBANDS: tuple[str, ...] = ("cA5", "cD5", "cD4", "cD3", "cD2", "cD1")

#: Statistics in registry order.
STATS: tuple[str, ...] = ("energy", "std", "entropy", "mean_abs")


class WaveletSettings:
    """Decomposition parameters resolved from config once."""

    __slots__ = ("wavelet", "level", "mode", "entropy_bins", "min_level")

    def __init__(self, features_cfg: Any) -> None:
        self.wavelet = str(features_cfg.get("families.dwt.wavelet", "db4"))
        self.level = int(features_cfg.get("families.dwt.level", 5))
        self.mode = str(features_cfg.get("families.dwt.mode", "symmetric"))
        self.entropy_bins = int(features_cfg.get("families.dwt.entropy_bins", 64))
        self.min_level = int(features_cfg.get("families.dwt.min_level", 1))

        configured = tuple(features_cfg.get("families.dwt.subbands") or SUBBANDS)
        if configured != SUBBANDS:
            raise ValueError(
                "features.yaml dwt.subbands=" + str(list(configured))
                + " does not match the registry column names " + str(list(SUBBANDS))
            )
        stats = tuple(features_cfg.get("families.dwt.stats") or STATS)
        if stats != STATS:
            raise ValueError(
                "features.yaml dwt.stats=" + str(list(stats))
                + " does not match the registry column names " + str(list(STATS))
            )


def usable_level(
    n_samples: int, settings: WaveletSettings, flags: list[str] | None = None
) -> int:
    """The deepest decomposition this signal supports, capped at the configured level.

    Returns 0 when even ``min_level`` does not fit, which tells the caller to
    report the whole family as a gap rather than decompose something meaningless.
    """
    import pywt

    maximum = int(pywt.dwt_max_level(int(n_samples), pywt.Wavelet(settings.wavelet).dec_len))
    if maximum >= settings.level:
        return settings.level
    if maximum < settings.min_level:
        if flags is not None:
            flags.append("dwt_signal_too_short")
        return 0
    if flags is not None:
        flags.append("dwt_level_" + str(maximum))
    return maximum


def decompose(
    signal: np.ndarray, settings: WaveletSettings, flags: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Map a wavedec result onto the six registry sub-band names (T36.1).

    Missing names are simply absent from the returned mapping; the caller turns
    those into flagged NaN. See the module docstring for why a reduced-level
    approximation is never written into the ``cA5`` slot.
    """
    import pywt

    values = np.asarray(signal, dtype=np.float64)
    level = usable_level(values.size, settings, flags)
    if level == 0:
        return {}

    coefficients = pywt.wavedec(values, settings.wavelet, mode=settings.mode, level=level)
    # wavedec returns [cA_level, cD_level, cD_{level-1}, ..., cD1].
    bands: dict[str, np.ndarray] = {}
    if level == settings.level:
        bands["cA" + str(level)] = np.asarray(coefficients[0], dtype=np.float64)
    for position, detail in enumerate(coefficients[1:]):
        detail_level = level - position
        bands["cD" + str(detail_level)] = np.asarray(detail, dtype=np.float64)

    return {name: bands[name] for name in SUBBANDS if name in bands}


def _statistics(coefficients: np.ndarray, entropy_bins: int) -> dict[str, float]:
    return {
        "energy": float(np.sum(coefficients**2)),
        "std": float(np.std(coefficients)),
        "entropy": shannon_entropy(coefficients, bins=entropy_bins),
        "mean_abs": float(np.mean(np.abs(coefficients))),
    }


class WaveletExtractor(BaseFeatureExtractor):
    """The 24 DWT features (T36.1-T36.6)."""

    family = FAMILY
    name = "wavelet"

    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__(cfg)
        self._settings: WaveletSettings | None = None

    @property
    def _features(self) -> Any:
        if self._cfg is None:
            from src.utils.config import load_config

            self._cfg = load_config("features")
        return self._cfg

    def settings(self) -> WaveletSettings:
        if self._settings is None:
            self._settings = WaveletSettings(self._features)
        return self._settings

    def feature_names(self) -> tuple[str, ...]:
        return feature_names(FAMILY)

    def _compute(
        self, signal: np.ndarray, fs: int, flags: list[str]
    ) -> Mapping[str, float]:
        settings = self.settings()
        bands = decompose(signal, settings, flags)

        values: dict[str, float] = {}
        computed = {
            name: _statistics(coefficients, settings.entropy_bins)
            for name, coefficients in bands.items()
        }
        for stat in STATS:
            for band in SUBBANDS:
                key = "dwt_" + band + "_" + stat
                values[key] = computed[band][stat] if band in computed else float("nan")
        return values


def extract_wavelet_features(
    signal: np.ndarray, fs: int, *, record_uid: str | None = None, cfg: Any | None = None
):
    """Convenience wrapper returning a :class:`FamilyResult`."""
    return WaveletExtractor(cfg).extract(signal, fs, record_uid=record_uid)


register_extractor(WaveletExtractor())
