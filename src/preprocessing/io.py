"""Audio loading, mono conversion and resampling to 2 kHz (Phase 23).

Every downstream stage -- filtering, normalization, quality analysis, all 138
features -- assumes one sampling rate. This module is where the three native
rates in the corpus (2000, 4000 and 44100 Hz) become that one rate, and it is
the only place in the project allowed to change a sampling rate.

**Why not slice.** The obvious way to go from 44100 Hz to 2000 Hz is
``x[::22]``, and it is wrong. Decimation without an anti-alias filter folds
every component above the new Nyquist back into the band as a false
low-frequency tone -- and a PCG's diagnostic content lives at 20-400 Hz, exactly
where the folded energy lands. A murmur and an aliased 5 kHz artifact are then
indistinguishable to every feature downstream. Worse, 44100/2000 = 22.05 is not
an integer, so plain slicing cannot even hit the target rate: it produces
2004.5 Hz and a length that disagrees with every duration in the master table.

So resampling goes through a band-limited rational resampler (soxr by default,
scipy's polyphase as a documented alternative), and ``tests/test_resample.py``
proves it with a swept sine: a 20 Hz -> 20 kHz chirp resampled from 44.1 kHz
must leave nothing behind once the sweep passes 1 kHz. The same test runs naive
decimation beside it and shows the alias arriving, so the guard is checked
against the failure it exists to prevent rather than asserted.

**Mono.** All 7,536 files are mono (verified in Phase 16), so :func:`to_mono` is
a no-op fast path on real data. It stays because the inference API accepts
uploaded WAVs (Part X), and an uploaded stereo recording must not reach the
feature extractor as a 2-D array.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "DEFAULT_TARGET_FS",
    "RESAMPLE_METHODS",
    "AudioLoadError",
    "load_wav",
    "to_mono",
    "resample_to",
    "resample_ratio",
    "expected_native_fs",
    "target_fs",
    "resample_method",
    "load_resampled",
]

log = get_logger(__name__)

# The project target rate. configs/signal.yaml owns the value; this constant is
# the fallback when a caller does not pass a config, and the two are asserted
# equal in tests/test_resample.py.
DEFAULT_TARGET_FS = 2000

RESAMPLE_METHODS = frozenset({"soxr_vhq", "soxr_hq", "soxr_mq", "polyphase"})

# soxr quality names, keyed by the configs/signal.yaml method string.
_SOXR_QUALITY = {"soxr_vhq": "VHQ", "soxr_hq": "HQ", "soxr_mq": "MQ"}


class AudioLoadError(RuntimeError):
    """Raised when a WAV cannot be decoded."""


# ---------------------------------------------------------------------------
# config access
# ---------------------------------------------------------------------------


def _signal_config(cfg: object | None = None) -> object:
    if cfg is not None:
        return cfg
    from src.utils.config import load_config

    return load_config("signal")


def target_fs(cfg: object | None = None) -> int:
    """The configured target rate (``resample.target_fs``)."""
    return int(_signal_config(cfg).require("resample.target_fs"))  # type: ignore[attr-defined]


def resample_method(cfg: object | None = None) -> str:
    """The configured resampler (``resample.method``)."""
    method = str(_signal_config(cfg).require("resample.method"))  # type: ignore[attr-defined]
    if method not in RESAMPLE_METHODS:
        raise ValueError(
            "configs/signal.yaml resample.method " + repr(method)
            + " is not one of: " + ", ".join(sorted(RESAMPLE_METHODS))
        )
    return method


def expected_native_fs(dataset_source: str, cfg: object | None = None) -> int | None:
    """Native rate declared for a dataset family, or ``None`` if not declared.

    ``dataset_source`` accepts either the short id (``D2``) or the config key
    (``d2_pascal_a``). Used by T23.5 to check that what came off disk is what the
    audit says should be there -- a file at an unexpected rate is a dataset
    problem, not a resampling one, and it should be seen as such.
    """
    table = _signal_config(cfg).get("resample.native_fs", {}) or {}  # type: ignore[attr-defined]
    key = str(dataset_source).strip()
    if key in table:
        return int(table[key])
    lowered = key.lower()
    for name, value in table.items():
        if name.lower() == lowered or name.lower().startswith(lowered + "_"):
            return int(value)
    return None


# ---------------------------------------------------------------------------
# T23.1 -- load
# ---------------------------------------------------------------------------


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode a WAV to float32 samples plus its native sampling rate.

    Samples come back scaled to [-1, 1) -- soundfile divides by the format's
    full scale -- so a 16-bit PASCAL record and a hypothetical 24-bit one are on
    the same amplitude scale before anything else touches them. A mono file
    returns a 1-D array; a multi-channel file returns ``(n_samples, n_channels)``
    and is the caller's problem until :func:`to_mono`.
    """
    import soundfile as sf

    wav_path = Path(path)
    try:
        samples, fs = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises RuntimeError / LibsndfileError
        raise AudioLoadError("cannot read " + wav_path.as_posix() + ": " + str(exc)) from exc

    return np.asarray(samples, dtype=np.float32), int(fs)


# ---------------------------------------------------------------------------
# T23.2 -- mono
# ---------------------------------------------------------------------------


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse channels by mean, returning 1-D float32.

    Fast path: an array that is already 1-D and float32 is returned unchanged,
    without a copy. Every file in the corpus takes that path.
    """
    samples = np.asarray(x)
    if samples.ndim == 1:
        return samples if samples.dtype == np.float32 else samples.astype(np.float32)
    if samples.ndim != 2:
        raise ValueError("expected a 1-D or 2-D array, got " + str(samples.ndim) + " dimensions")
    return samples.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# T23.3 / T23.4 -- resample
# ---------------------------------------------------------------------------


def resample_ratio(fs_in: int, fs_out: int) -> Fraction:
    """The exact rational up/down ratio between two rates.

    44100 -> 2000 reduces to 20/441, not to a decimation factor: the 22.05 in
    the config is the *inverse* of that fraction and is deliberately not an
    integer. Naming it here keeps the polyphase path honest.
    """
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(
            "sampling rates must be positive, got " + str(fs_in) + " -> " + str(fs_out)
        )
    return Fraction(int(fs_out), int(fs_in))


def resample_to(
    x: np.ndarray,
    fs_in: int,
    fs_out: int = DEFAULT_TARGET_FS,
    *,
    method: str = "soxr_hq",
) -> np.ndarray:
    """Band-limited resample of a 1-D signal from ``fs_in`` to ``fs_out``.

    ``fs_in == fs_out`` is a no-op that returns the input unchanged (PhysioNet's
    path -- 3,240 of 7,536 records never get resampled at all). An empty signal
    returns empty rather than raising, so a degenerate record fails in the
    quality report instead of aborting a 7,536-file batch.

    Both methods apply an anti-alias filter before decimation. ``soxr_hq`` has
    roughly 120 dB of stopband rejection, far beyond the 16-bit dynamic range of
    every file in this corpus; ``polyphase`` uses scipy's Kaiser-windowed FIR at
    the same rational ratio and is kept as a dependency-light alternative.
    """
    samples = to_mono(np.asarray(x))
    fs_in, fs_out = int(fs_in), int(fs_out)

    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(
            "sampling rates must be positive, got " + str(fs_in) + " -> " + str(fs_out)
        )
    if method not in RESAMPLE_METHODS:
        raise ValueError(
            "unknown resample method " + repr(method) + "; expected one of: "
            + ", ".join(sorted(RESAMPLE_METHODS))
        )
    if fs_in == fs_out or samples.size == 0:
        return samples

    if method == "polyphase":
        from scipy.signal import resample_poly

        ratio = resample_ratio(fs_in, fs_out)
        out = resample_poly(samples.astype(np.float64), ratio.numerator, ratio.denominator)
    else:
        import soxr

        out = soxr.resample(samples, fs_in, fs_out, quality=_SOXR_QUALITY[method])

    return np.asarray(out, dtype=np.float32)


def load_resampled(
    path: str | Path,
    fs_out: int | None = None,
    *,
    method: str | None = None,
    cfg: object | None = None,
) -> tuple[np.ndarray, int]:
    """Load, collapse to mono and resample in one call.

    Returns ``(signal, fs_native)`` -- the native rate travels with the signal
    because the quality report and the run manifest both record what the file
    actually was, not only what it was converted to.
    """
    signal_cfg = _signal_config(cfg)
    fs_target = int(fs_out) if fs_out is not None else target_fs(signal_cfg)
    how = method or resample_method(signal_cfg)

    samples, fs_native = load_wav(path)
    samples = to_mono(samples)
    return resample_to(samples, fs_native, fs_target, method=how), fs_native
