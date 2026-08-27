"""Time-domain features: the 24 of Phase 32.

Eight basic statistics, two shape statistics, four energy measures, two framed
zero-crossing statistics, five complexity measures, and autocorrelation peak
value, peak lag and duration.

**These run on the z-normalized signal, and that decides what several of them
mean.** After normalization every record has mean 0 and SD 1 by construction, so
``time_mean``, ``time_std``, ``time_var`` and ``time_hjorth_activity`` are near
constants across the corpus rather than descriptors of loudness. They are kept
because both source documents list them and because the family count is locked
at 24 -- but Phase 41 will find them in the near-zero-variance report, and that
is the correct place for the fact to surface, not a quiet substitution here.
The informative members of this family are the shape, complexity and
autocorrelation terms, none of which normalization touches.

**Three decisions that are not obvious from the task list:**

*Sample entropy uses a KD-tree, and a centred window.* Sample entropy is O(n^2)
in the naive form. The longest record here is 122 s = 244,000 samples; the naive
form is ~6e10 distance evaluations for one record out of 7,536. A
``cKDTree.count_neighbors`` with the Chebyshev metric answers the same question
in O(n log n). ``max_samples`` (20,000 = 10 s) still applies on top, and the
window is taken from the **centre** of the recording rather than the start:
opening seconds carry stethoscope placement and handling noise far more often
than the middle does.

*The autocorrelation peak is searched after the first zero crossing.* The
normalized autocorrelation is 1 at lag 0 and decays smoothly, so a bare
``argmax`` over positive lags returns lag 1 for every record and measures
nothing. Searching after the first zero crossing is the standard pitch-detection
remedy and needs no additional tuning constant; where no zero crossing exists
inside the 2 s search window the floor falls back to 0.25 s, the 240 bpm ceiling
already configured for envelope peak detection.

*Degenerate signals return real numbers with a flag, never NaN.* A constant
recording has zero crest factor and zero mobility -- those are answers, not
failures. NaN in this project means "this record failed here" (see
``base.py``), so the flag is what carries the caveat.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial import cKDTree

from src.feature_extraction.base import BaseFeatureExtractor
from src.feature_extraction.registry import feature_names, register_extractor

__all__ = [
    "TimeDomainExtractor",
    "extract_time_features",
    "shannon_entropy",
    "sample_entropy",
    "hjorth_parameters",
    "autocorrelation_peak",
    "zero_crossing_stats",
]

FAMILY = "time"


# ---------------------------------------------------------------------------
# individual measures, exposed for unit testing against known answers
# ---------------------------------------------------------------------------


def shannon_entropy(signal: np.ndarray, bins: int = 64) -> float:
    """Shannon entropy in bits of a ``bins``-bin amplitude histogram.

    A histogram rather than a differential entropy: the bin count is fixed in
    config, so two records are comparable, and a constant signal lands in one
    bin and scores exactly 0 instead of diverging.

    **The degenerate-range guard is not defensive padding.** ``np.histogram``
    raises ``ValueError: Too many bins for data range`` when the requested bin
    width falls below the float spacing at the data's magnitude -- which happens
    for an array that is constant to within rounding but not exactly constant.
    A DC-only recording produces exactly that in its ``cA5`` wavelet sub-band:
    54 coefficients spanning 8.88e-16 around -4.525, where 64 bins would each be
    1.4e-17 wide and float64 cannot represent the edges. Such an array carries no
    information, so its entropy is 0 -- returning that is the correct answer, not
    a fallback. Found by the T36.7 gate; see Docs/note.md, 2026-08-27.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0:
        return float("nan")

    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 0.0 or span <= float(np.spacing(max(abs(low), abs(high)))) * int(bins):
        return 0.0

    counts, _edges = np.histogram(values, bins=int(bins))
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def sample_entropy(
    signal: np.ndarray,
    *,
    m: int = 2,
    r_factor: float = 0.2,
    max_samples: int = 20000,
    flags: list[str] | None = None,
) -> float:
    """Sample entropy (Richman & Moorman 2000), KD-tree accelerated.

    ``SampEn = -ln(A/B)`` where ``B`` counts template pairs of length ``m``
    within Chebyshev distance ``r = r_factor * std(x)`` and ``A`` counts the same
    pairs at length ``m+1``. Self-matches are excluded, which is what separates
    sample entropy from approximate entropy and removes its length bias.
    """
    values = np.asarray(signal, dtype=np.float64)
    n_total = values.size

    if max_samples and n_total > max_samples:
        start = (n_total - max_samples) // 2
        values = values[start : start + max_samples]
        if flags is not None:
            flags.append("sampen_window_" + str(max_samples))

    n = values.size
    if n < m + 2:
        if flags is not None:
            flags.append("sampen_too_short")
        return float("nan")

    tolerance = float(r_factor) * float(np.std(values))

    # N - m templates at each of the two lengths, so A and B count over the same
    # index set -- the convention the -ln(A/B) form assumes.
    n_templates = n - m
    strides = values.strides[0]
    templates_m = np.lib.stride_tricks.as_strided(
        values, shape=(n_templates, m), strides=(strides, strides)
    ).copy()
    templates_m1 = np.lib.stride_tricks.as_strided(
        values, shape=(n_templates, m + 1), strides=(strides, strides)
    ).copy()

    # count_neighbors counts ordered pairs including each point with itself;
    # (count - n) / 2 is the number of distinct unordered non-self pairs.
    # `balanced_tree=False, compact_nodes=False` halves the build cost and
    # returns bit-identical counts -- these flags change how the tree is split,
    # not what a radius query finds. Sample entropy is the single most expensive
    # feature of the 138 (measured 2026-08-27: 5.4 s -> 2.8 s per 20,000-sample
    # window), so the factor is worth the two keywords.
    tree_m = cKDTree(templates_m, balanced_tree=False, compact_nodes=False)
    tree_m1 = cKDTree(templates_m1, balanced_tree=False, compact_nodes=False)
    b_count = (tree_m.count_neighbors(tree_m, tolerance, p=np.inf) - n_templates) / 2.0
    a_count = (tree_m1.count_neighbors(tree_m1, tolerance, p=np.inf) - n_templates) / 2.0

    if b_count <= 0 or a_count <= 0:
        # No template match at one of the two lengths: the ratio is undefined.
        # The conventional substitute is the upper bound that would hold if
        # exactly one match existed, which keeps the value finite and ordered
        # correctly against records that did match.
        if flags is not None:
            flags.append("sampen_no_match")
        denominator = (n_templates - 1) * n_templates
        if denominator <= 0:
            return float("nan")
        return float(-np.log(2.0 / denominator))

    return float(-np.log(a_count / b_count))


def hjorth_parameters(signal: np.ndarray) -> tuple[float, float, float]:
    """Hjorth activity, mobility and complexity.

    Activity is the variance, mobility the SD of the first derivative over the SD
    of the signal (a mean-frequency proxy), complexity the mobility of the
    derivative over the mobility of the signal (a bandwidth proxy). A constant
    signal has zero of all three by definition rather than 0/0.
    """
    values = np.asarray(signal, dtype=np.float64)
    var_x = float(np.var(values))
    if values.size < 3 or var_x <= 0.0:
        return var_x, 0.0, 0.0

    d1 = np.diff(values)
    d2 = np.diff(d1)
    var_d1 = float(np.var(d1))
    var_d2 = float(np.var(d2))

    mobility = float(np.sqrt(var_d1 / var_x))
    if var_d1 <= 0.0 or mobility <= 0.0:
        return var_x, mobility, 0.0

    mobility_d1 = float(np.sqrt(var_d2 / var_d1))
    return var_x, mobility, float(mobility_d1 / mobility)


def autocorrelation_peak(
    signal: np.ndarray,
    fs: int,
    *,
    max_lag_sec: float = 2.0,
    fallback_min_lag_sec: float = 0.25,
    flags: list[str] | None = None,
) -> tuple[float, float]:
    """Peak value and lag of the normalized autocorrelation, in (value, seconds).

    Searched from the first zero crossing onward; see the module docstring for
    why a bare argmax over positive lags is meaningless here.
    """
    values = np.asarray(signal, dtype=np.float64)
    n = values.size
    if n < 4:
        if flags is not None:
            flags.append("autocorr_too_short")
        return 0.0, 0.0

    centred = values - values.mean()
    size = int(2 ** np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n]

    zero_lag = float(correlation[0])
    if zero_lag <= 0.0:
        if flags is not None:
            flags.append("autocorr_zero_energy")
        return 0.0, 0.0

    normalized = correlation / zero_lag
    max_lag = min(round(max_lag_sec * fs), n - 1)
    if max_lag < 2:
        if flags is not None:
            flags.append("autocorr_too_short")
        return 0.0, 0.0

    window = normalized[: max_lag + 1]
    negative = np.flatnonzero(window <= 0.0)
    if negative.size:
        start = int(negative[0])
    else:
        start = min(round(fallback_min_lag_sec * fs), max_lag)
        if flags is not None:
            flags.append("autocorr_no_zero_crossing")

    if start >= max_lag:
        start = 1
        if flags is not None:
            flags.append("autocorr_search_window_collapsed")

    segment = window[start : max_lag + 1]
    if segment.size == 0:
        return 0.0, 0.0

    offset = int(np.argmax(segment))
    lag = start + offset
    return float(segment[offset]), float(lag / fs)


# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------


def _framing(cfg: Any) -> tuple[int, int, bool]:
    frame_length = int(cfg.get("framing.frame_length", 512))
    hop_length = int(cfg.get("framing.hop_length", 256))
    center = bool(cfg.get("framing.center", True))
    return frame_length, hop_length, center


def zero_crossing_stats(
    signal: np.ndarray, frame_length: int, hop_length: int, center: bool
) -> tuple[float, float]:
    """Framed zero-crossing rate, mean and SD over frames.

    Framed rather than global: a global ZCR is one number per record and hides
    the alternation between the loud, low-frequency S1/S2 complexes and the
    quiet, higher-frequency systolic gaps, which is exactly the contrast a
    murmur changes.
    """
    import librosa

    rates = librosa.feature.zero_crossing_rate(
        y=signal, frame_length=frame_length, hop_length=hop_length, center=center
    )[0]
    if rates.size == 0:
        return float("nan"), float("nan")
    if rates.size == 1:
        return float(rates[0]), 0.0
    return float(np.mean(rates)), float(np.std(rates))


class TimeDomainExtractor(BaseFeatureExtractor):
    """The 24 time-domain features (T32.1-T32.6)."""

    family = FAMILY
    name = "time_domain"

    def __init__(self, cfg: Any | None = None, signal_cfg: Any | None = None) -> None:
        super().__init__(cfg)
        self._signal_cfg = signal_cfg

    # -- config -------------------------------------------------------------

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

    # -- the maths ----------------------------------------------------------

    def _compute(
        self, signal: np.ndarray, fs: int, flags: list[str]
    ) -> Mapping[str, float]:
        cfg = self._features
        frame_length, hop_length, center = _framing(self._signal)

        values: dict[str, float] = {}

        # --- basic statistics (8), T32.1
        q1, median, q3 = np.percentile(signal, [25, 50, 75])
        minimum = float(np.min(signal))
        maximum = float(np.max(signal))
        values["time_mean"] = float(np.mean(signal))
        values["time_std"] = float(np.std(signal))
        values["time_var"] = float(np.var(signal))
        values["time_min"] = minimum
        values["time_max"] = maximum
        values["time_range"] = maximum - minimum
        values["time_median"] = float(median)
        values["time_iqr"] = float(q3 - q1)

        # --- shape statistics (2), T32.2
        # bias=False is the sample-corrected form; with n in the thousands the
        # correction is negligible, but it is the form quoted in the write-up.
        if signal.size < 4 or values["time_var"] <= 0.0:
            flags.append("shape_stats_undefined")
            values["time_skewness"] = 0.0
            values["time_kurtosis"] = 0.0
        else:
            values["time_skewness"] = float(sp_stats.skew(signal, bias=False))
            values["time_kurtosis"] = float(
                sp_stats.kurtosis(signal, fisher=True, bias=False)
            )

        # --- energy (4), T32.3
        energy = float(np.sum(signal**2))
        rms = float(np.sqrt(energy / signal.size))
        values["time_energy"] = energy
        values["time_rms"] = rms
        values["time_peak_to_peak"] = maximum - minimum
        if rms > 0.0:
            values["time_crest_factor"] = float(np.max(np.abs(signal)) / rms)
        else:
            flags.append("crest_factor_zero_rms")
            values["time_crest_factor"] = 0.0

        # --- zero crossings (2), T32.4
        zcr_mean, zcr_std = zero_crossing_stats(
            signal, frame_length, hop_length, center
        )
        values["time_zcr_mean"] = zcr_mean
        values["time_zcr_std"] = zcr_std

        # --- complexity (5), T32.5
        values["time_shannon_entropy"] = shannon_entropy(
            signal, bins=int(cfg.get("families.time.shannon_entropy_bins", 64))
        )
        values["time_sample_entropy"] = sample_entropy(
            signal,
            m=int(cfg.get("families.time.sample_entropy.m", 2)),
            r_factor=float(cfg.get("families.time.sample_entropy.r_factor", 0.2)),
            max_samples=int(cfg.get("families.time.sample_entropy.max_samples", 20000)),
            flags=flags,
        )
        activity, mobility, complexity = hjorth_parameters(signal)
        values["time_hjorth_activity"] = activity
        values["time_hjorth_mobility"] = mobility
        values["time_hjorth_complexity"] = complexity

        # --- autocorrelation and duration (3), T32.6
        peak_value, peak_lag = autocorrelation_peak(
            signal,
            fs,
            max_lag_sec=float(cfg.get("families.time.autocorr_max_lag_sec", 2.0)),
            fallback_min_lag_sec=float(
                cfg.get("families.envelope.peak_detection.min_distance_sec", 0.25)
            ),
            flags=flags,
        )
        values["time_autocorr_peak_value"] = peak_value
        values["time_autocorr_peak_lag"] = peak_lag
        values["time_duration"] = float(signal.size) / float(fs)

        return values


def extract_time_features(
    signal: np.ndarray, fs: int, *, record_uid: str | None = None, cfg: Any | None = None
):
    """Convenience wrapper returning a :class:`FamilyResult`."""
    return TimeDomainExtractor(cfg).extract(signal, fs, record_uid=record_uid)


register_extractor(TimeDomainExtractor())
