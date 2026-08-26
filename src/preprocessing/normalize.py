"""Per-record amplitude normalization (Phase 25).

Recording gain is not a property of the heart. Two PASCAL recordings of the same
child through the same stethoscope differ in level because of how hard the
microphone was pressed; PhysioNet's six sources differ by more than that. Every
amplitude-sensitive feature -- RMS, peak, energy, the DWT sub-band energies --
would otherwise learn the recording setup instead of the pathology, and since
source correlates with class in PhysioNet, that is a leakage path with a
plausible-looking accuracy attached to it.

Normalization is therefore **per record**, computed from that record alone.
Nothing here is fitted across records: there is no corpus mean, no dataset scale.
That distinction is what keeps this step outside the fold-safety rule -- a
statistic computed from one record cannot leak anything from another. The
feature-matrix scaler, which *is* fitted across records, lives inside the
training fold (rule 2) and is a different object entirely.

**z-score is the default.** ``(x - mean) / std`` puts every record at mean 0 and
SD 1. Peak normalization (``x / max|x|``) is the alternative arm of the PP-09
ablation: it is more intuitive but is hostage to a single sample, so one click
artifact can scale an entire recording down by 20 dB.

**The zero-variance guard.** A constant or silent record has ``std == 0``, and
dividing by it yields inf or NaN for every sample -- which then propagates
through 138 features and reaches the model as a row of NaN. Such a record is
returned unchanged and flagged ``zero_variance`` instead. The corpus contains one
near-dead recording (set_a's ``Aunlabelledtest__201106120928``, peaking at
-71 dBFS), so this is not a theoretical case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "NORMALIZATION_METHODS",
    "DEFAULT_METHOD",
    "DEFAULT_EPSILON",
    "NormalizationResult",
    "remove_dc",
    "zscore_normalize",
    "peak_normalize",
    "normalize",
    "normalize_signal",
    "normalization_stats",
]

log = get_logger(__name__)

NORMALIZATION_METHODS = frozenset({"zscore", "peak", "none"})
DEFAULT_METHOD = "zscore"
DEFAULT_EPSILON = 1e-10


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """A normalized signal and the statistics either side of the operation.

    The before/after statistics are kept because they are the evidence for
    T25.5: "post-normalization mean is approximately 0 and SD approximately 1"
    is a claim about numbers, and these are the numbers. Phase 26 writes them
    into the per-record quality report.
    """

    signal: np.ndarray
    method: str
    applied: bool
    dc_removed: bool
    zero_variance: bool
    mean_before: float
    std_before: float
    peak_before: float
    mean_after: float
    std_after: float
    peak_after: float

    def as_dict(self) -> dict[str, Any]:
        """The row this result contributes to a per-record report."""
        return {
            "norm_method": self.method,
            "norm_applied": self.applied,
            "norm_dc_removed": self.dc_removed,
            "norm_zero_variance": self.zero_variance,
            "mean_before_norm": self.mean_before,
            "std_before_norm": self.std_before,
            "peak_before_norm": self.peak_before,
            "mean_after_norm": self.mean_after,
            "std_after_norm": self.std_after,
            "peak_after_norm": self.peak_after,
        }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def normalization_stats(x: np.ndarray) -> dict[str, float]:
    """Mean, SD and peak of a signal, in float64.

    Always float64, even for a float32 signal: the naive float32 sum over a
    122-second PhysioNet record (244,000 samples) loses enough precision that
    the mean of an already-centred signal comes back as 1e-3 rather than 1e-9,
    and a test asserting "mean is approximately 0" would then be measuring the
    accumulator, not the signal.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    if samples.size == 0:
        return {"mean": 0.0, "std": 0.0, "peak": 0.0}
    return {
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "peak": float(np.max(np.abs(samples))),
    }


# ---------------------------------------------------------------------------
# T25.3 -- DC offset
# ---------------------------------------------------------------------------


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Subtract the mean.

    A DC offset is a recording-chain artifact, not acoustics. It survives the
    20-400 Hz bandpass only when the filter is disabled for the ablation, which
    is exactly when this step has to be there instead.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    if samples.size == 0:
        return samples.astype(np.float32)
    return (samples - samples.mean()).astype(np.float32)


# ---------------------------------------------------------------------------
# T25.1 / T25.2 -- the two methods
# ---------------------------------------------------------------------------


def zscore_normalize(
    x: np.ndarray, *, epsilon: float = DEFAULT_EPSILON
) -> tuple[np.ndarray, bool]:
    """``(x - mean) / std``. Returns the signal and a zero-variance flag.

    When ``std <= epsilon`` the input is returned unchanged and the flag is
    True: a constant record has no shape to preserve, and dividing by ~0 would
    turn it into inf.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    if samples.size == 0:
        return samples.astype(np.float32), True

    std = float(samples.std())
    if std <= epsilon:
        return samples.astype(np.float32), True
    return ((samples - samples.mean()) / std).astype(np.float32), False


def peak_normalize(
    x: np.ndarray, *, epsilon: float = DEFAULT_EPSILON
) -> tuple[np.ndarray, bool]:
    """``x / max|x|``, so the loudest sample sits at +/-1 (T25.2)."""
    samples = np.asarray(x, dtype=np.float64).ravel()
    if samples.size == 0:
        return samples.astype(np.float32), True

    peak = float(np.max(np.abs(samples)))
    if peak <= epsilon:
        return samples.astype(np.float32), True
    return (samples / peak).astype(np.float32), False


# ---------------------------------------------------------------------------
# T25.4 -- the switchable entry point
# ---------------------------------------------------------------------------


def normalize(
    x: np.ndarray,
    *,
    method: str = DEFAULT_METHOD,
    remove_dc_offset: bool = True,
    epsilon: float = DEFAULT_EPSILON,
    enabled: bool = True,
) -> NormalizationResult:
    """Normalize one record, reporting what was done and the statistics.

    ``enabled=False`` and ``method="none"`` are both the no-normalization arm of
    the PP-09 ablation (T25.4) and both return the input unchanged; they are kept
    distinct because "the config turned it off" and "the config asked for the
    identity method" are different lines in an ablation table.

    With ``method="zscore"`` the mean is removed whether or not
    ``remove_dc_offset`` is set -- subtracting the mean is what a z-score is. The
    flag matters for ``peak``, where a DC offset otherwise inflates the divisor.
    """
    if method not in NORMALIZATION_METHODS:
        raise ValueError(
            "unknown normalization method " + repr(method) + "; expected one of: "
            + ", ".join(sorted(NORMALIZATION_METHODS))
        )

    samples = np.asarray(x, dtype=np.float64).ravel()
    before = normalization_stats(samples)

    def result(
        signal: np.ndarray, *, applied: bool, dc_removed: bool, zero_variance: bool
    ) -> NormalizationResult:
        after = normalization_stats(signal)
        return NormalizationResult(
            signal=np.asarray(signal, dtype=np.float32),
            method=method,
            applied=applied,
            dc_removed=dc_removed,
            zero_variance=zero_variance,
            mean_before=before["mean"],
            std_before=before["std"],
            peak_before=before["peak"],
            mean_after=after["mean"],
            std_after=after["std"],
            peak_after=after["peak"],
        )

    if not enabled or method == "none":
        return result(samples, applied=False, dc_removed=False, zero_variance=False)

    working = samples
    dc_removed = False
    if remove_dc_offset:
        working = np.asarray(remove_dc(working), dtype=np.float64)
        dc_removed = True

    if method == "zscore":
        normalized, zero_variance = zscore_normalize(working, epsilon=epsilon)
    else:
        normalized, zero_variance = peak_normalize(working, epsilon=epsilon)

    if zero_variance:
        log.warning(
            "zero-variance record (std %.3e, peak %.3e): returned unnormalized and flagged",
            before["std"],
            before["peak"],
        )
        # Passthrough means passthrough: the *input*, not the DC-removed copy,
        # so a flagged record is bit-identical to what arrived.
        return result(samples, applied=False, dc_removed=False, zero_variance=True)

    return result(normalized, applied=True, dc_removed=dc_removed, zero_variance=False)


def normalize_signal(x: np.ndarray, cfg: Any | None = None) -> NormalizationResult:
    """:func:`normalize` driven by ``configs/signal.yaml``.

    Honours ``normalization.enabled`` (the PP-09 ablation arm) and refuses an
    unimplemented ``zero_variance_policy`` rather than quietly applying a
    different one.
    """
    if cfg is None:
        from src.utils.config import load_config

        cfg = load_config("signal")

    policy = str(cfg.get("normalization.zero_variance_policy", "passthrough_and_flag"))
    if policy != "passthrough_and_flag":
        raise ValueError(
            "configs/signal.yaml normalization.zero_variance_policy " + repr(policy)
            + " is not implemented; only 'passthrough_and_flag' is"
        )

    return normalize(
        x,
        method=str(cfg.get("normalization.method", DEFAULT_METHOD)),
        remove_dc_offset=bool(cfg.get("normalization.remove_dc", True)),
        epsilon=float(cfg.get("normalization.zero_variance_epsilon", DEFAULT_EPSILON)),
        enabled=bool(cfg.get("normalization.enabled", True)),
    )
