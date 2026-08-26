"""The global feature registry: 138 names, one fixed order (Phase 31).

**Why a registry at all.** A feature matrix is a 2-D array of anonymous floats.
Column 47 is only "MFCC 3 std" because something says so, and if that something
is the order in which six dictionaries happened to merge, then a model trained
today and a model explained tomorrow can disagree about what column 47 was --
silently, with no error and no wrong-looking number anywhere. Every SHAP plot,
every selected-feature list, every ablation and every live prediction in this
project reads its column meanings from here.

**The ordering is a literal, not a computation (T31.3).** The names below are
written out in fixed order rather than derived from a dict, a set, a glob over
modules or an iteration over ``configs/features.yaml``. Any of those would make
the column order depend on import order, hash seeding or a YAML edit. The order
here changes only when someone edits this file, and :func:`registry_fingerprint`
turns the whole list into one hash that a run manifest can record and a later run
can compare against.

**The registry owns the names; the extractors own the maths.** The 138 specs
exist from import, before any family in Phases 32-37 is written. A family
registers its implementation against a name list that already exists and must
match it exactly, so an extractor cannot invent, rename or drop a column. That
inversion is deliberate: the contract is fixed first, and six independent
implementations are then held to it.

Composition, locked (``configs/features.yaml``, Docs/note.md 2026-08-22)::

    time 24 + frequency 22 + mfcc 39 + chroma 24 + dwt 24 + envelope 5 = 138
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

__all__ = [
    "FeatureSpec",
    "RegistryError",
    "ExtractorNotRegistered",
    "FAMILY_ORDER",
    "EXPECTED_FAMILY_COUNTS",
    "EXPECTED_TOTAL",
    "FEATURE_SPECS",
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "FAMILY_MODULES",
    "feature_names",
    "family_of",
    "index_of",
    "spec_for",
    "specs_for_family",
    "family_counts",
    "registry_fingerprint",
    "register_extractor",
    "get_extractor",
    "registered_families",
    "load_all_extractors",
    "validate_against_config",
    "as_records",
]


class RegistryError(Exception):
    """The registry is internally inconsistent, or misused."""


class ExtractorNotRegistered(RegistryError):
    """A family's implementation has not been built or imported yet."""


#: Family order == column order. Never reorder; it renumbers every column.
FAMILY_ORDER: tuple[str, ...] = (
    "time",
    "frequency",
    "mfcc",
    "chroma",
    "dwt",
    "envelope",
)

EXPECTED_FAMILY_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "time": 24,
        "frequency": 22,
        "mfcc": 39,
        "chroma": 24,
        "dwt": 24,
        "envelope": 5,
    }
)

EXPECTED_TOTAL = 138

#: Where each family's implementation lives. Consulted only on demand, so the
#: registry imports cleanly before Phases 32-37 exist.
FAMILY_MODULES: Mapping[str, str] = MappingProxyType(
    {
        "time": "src.feature_extraction.time_domain",
        "frequency": "src.feature_extraction.frequency",
        "mfcc": "src.feature_extraction.mfcc",
        "chroma": "src.feature_extraction.chroma",
        "dwt": "src.feature_extraction.wavelet",
        "envelope": "src.feature_extraction.envelope",
    }
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One column of the feature matrix (T31.2).

    ``equation`` is the reference that ends up in FE-01 ``feature_inventory.csv``
    (T38.3) and in the thesis feature table. It is deliberately terse maths or a
    named library function rather than prose -- an examiner asking "how was
    spectral flatness computed" should be able to answer from this column.

    ``unit`` is the physical unit where one exists. Most values here are unitless
    because they are computed on a z-normalized signal: after normalization the
    amplitude scale is gone by construction, so "amplitude" units would be a
    fiction. Frequencies stay in Hz and durations in seconds.
    """

    index: int
    name: str
    family: str
    extractor: str
    equation: str
    unit: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "family": self.family,
            "extractor": self.extractor,
            "equation": self.equation,
            "unit": self.unit,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# the declarations
#
# (name, equation, unit, description), in column order within each family.
# ---------------------------------------------------------------------------

_TIME: tuple[tuple[str, str, str, str], ...] = (
    # basic statistics (8) -- T32.1
    ("time_mean", "mean(x)", "", "Mean amplitude of the preprocessed signal"),
    ("time_std", "std(x), ddof=0", "", "Standard deviation of amplitude"),
    ("time_var", "var(x), ddof=0", "", "Variance of amplitude"),
    ("time_min", "min(x)", "", "Most negative sample"),
    ("time_max", "max(x)", "", "Most positive sample"),
    ("time_range", "max(x) - min(x)", "", "Full amplitude range"),
    ("time_median", "median(x)", "", "Median amplitude"),
    ("time_iqr", "Q3(x) - Q1(x)", "", "Interquartile range of amplitude"),
    # shape statistics (2) -- T32.2
    (
        "time_skewness",
        "scipy.stats.skew(x), bias=False",
        "",
        "Asymmetry of the amplitude distribution",
    ),
    (
        "time_kurtosis",
        "scipy.stats.kurtosis(x), Fisher, bias=False",
        "",
        "Tailedness of the amplitude distribution; excess kurtosis",
    ),
    # energy (4) -- T32.3
    ("time_energy", "sum(x^2)", "", "Total signal energy"),
    ("time_rms", "sqrt(mean(x^2))", "", "Root-mean-square amplitude"),
    (
        "time_peak_to_peak",
        "max(x) - min(x)",
        "",
        "Peak-to-peak excursion; equals range, kept because the blueprint lists both",
    ),
    ("time_crest_factor", "max(|x|) / rms(x)", "", "Peak-to-RMS ratio; high for impulsive sounds"),
    # zero crossings (2) -- T32.4
    ("time_zcr_mean", "mean over frames of ZCR(frame)", "", "Mean framed zero-crossing rate"),
    (
        "time_zcr_std",
        "std over frames of ZCR(frame)",
        "",
        "Variability of the framed zero-crossing rate",
    ),
    # complexity (5) -- T32.5
    (
        "time_shannon_entropy",
        "-sum(p*log2(p)) over a 64-bin amplitude histogram",
        "bit",
        "Shannon entropy of the amplitude distribution",
    ),
    (
        "time_sample_entropy",
        "-ln(A/B), m=2, r=0.2*std, Richman & Moorman 2000",
        "",
        "Sample entropy; regularity of the waveform",
    ),
    ("time_hjorth_activity", "var(x)", "", "Hjorth activity; signal power"),
    ("time_hjorth_mobility", "sqrt(var(dx)/var(x))", "", "Hjorth mobility; mean frequency proxy"),
    (
        "time_hjorth_complexity",
        "mobility(dx)/mobility(x)",
        "",
        "Hjorth complexity; bandwidth proxy",
    ),
    # autocorrelation and duration (3) -- T32.6
    (
        "time_autocorr_peak_value",
        "max of normalized autocorrelation beyond its first zero crossing",
        "",
        "Strength of the dominant periodicity; a rhythm-regularity proxy",
    ),
    (
        "time_autocorr_peak_lag",
        "argmax lag of the same search, in seconds",
        "s",
        "Period of the dominant periodicity; a cardiac-cycle proxy",
    ),
    ("time_duration", "n_samples / fs", "s", "Recording duration"),
)

_FREQUENCY: tuple[tuple[str, str, str, str], ...] = (
    # framed spectral statistics (12) -- T33.1-T33.3
    (
        "freq_centroid_mean",
        "mean over frames of sum(f*S)/sum(S)",
        "Hz",
        "Mean spectral centre of mass",
    ),
    (
        "freq_centroid_std",
        "std over frames of the same",
        "Hz",
        "Variability of the spectral centroid",
    ),
    (
        "freq_bandwidth_mean",
        "mean over frames of sqrt(sum(S*(f-centroid)^2)/sum(S))",
        "Hz",
        "Mean spectral spread about the centroid",
    ),
    (
        "freq_bandwidth_std",
        "std over frames of the same",
        "Hz",
        "Variability of the spectral bandwidth",
    ),
    (
        "freq_rolloff85_mean",
        "mean over frames of min f with cumsum(S) >= 0.85*sum(S)",
        "Hz",
        "Mean 85% spectral rolloff",
    ),
    ("freq_rolloff85_std", "std over frames of the same", "Hz", "Variability of the 85% rolloff"),
    (
        "freq_rolloff95_mean",
        "mean over frames of min f with cumsum(S) >= 0.95*sum(S)",
        "Hz",
        "Mean 95% spectral rolloff",
    ),
    ("freq_rolloff95_std", "std over frames of the same", "Hz", "Variability of the 95% rolloff"),
    (
        "freq_flatness_mean",
        "mean over frames of geometric_mean(S)/arithmetic_mean(S)",
        "",
        "Mean Wiener entropy; 1 is noise-like, 0 is tonal",
    ),
    ("freq_flatness_std", "std over frames of the same", "", "Variability of spectral flatness"),
    (
        "freq_flux_mean",
        "mean over frame pairs of ||S_t - S_{t-1}||_2, S L1-normalized per frame",
        "",
        "Mean spectral change between consecutive frames",
    ),
    ("freq_flux_std", "std over frame pairs of the same", "", "Variability of spectral flux"),
    # global statistics from the Welch PSD (4) -- T33.4
    (
        "freq_spectral_entropy",
        "-sum(p*log2(p))/log2(n_bins), p = PSD/sum(PSD)",
        "",
        "Normalized spectral entropy of the Welch PSD, in [0, 1]",
    ),
    ("freq_dominant", "f at argmax(PSD)", "Hz", "Frequency carrying the most power"),
    ("freq_peak_power", "max(PSD)", "1/Hz", "Power spectral density at the dominant frequency"),
    ("freq_total_power", "trapezoid(PSD, f) over 0-Nyquist", "", "Total power of the Welch PSD"),
    # relative band power (6) -- T33.5
    (
        "freq_band_power_20_50",
        "trapezoid(PSD, f) over 20-50 Hz / total_power",
        "",
        "Relative power, 20-50 Hz",
    ),
    (
        "freq_band_power_50_100",
        "trapezoid(PSD, f) over 50-100 Hz / total_power",
        "",
        "Relative power, 50-100 Hz",
    ),
    (
        "freq_band_power_100_150",
        "trapezoid(PSD, f) over 100-150 Hz / total_power",
        "",
        "Relative power, 100-150 Hz",
    ),
    (
        "freq_band_power_150_250",
        "trapezoid(PSD, f) over 150-250 Hz / total_power",
        "",
        "Relative power, 150-250 Hz",
    ),
    (
        "freq_band_power_250_350",
        "trapezoid(PSD, f) over 250-350 Hz / total_power",
        "",
        "Relative power, 250-350 Hz",
    ),
    (
        "freq_band_power_350_400",
        "trapezoid(PSD, f) over 350-400 Hz / total_power",
        "",
        "Relative power, 350-400 Hz",
    ),
)

_ENVELOPE: tuple[tuple[str, str, str, str], ...] = (
    ("env_mean", "mean(e), e = smoothed |hilbert(x)|", "", "Mean amplitude envelope"),
    ("env_std", "std(e)", "", "Envelope variability"),
    ("env_skew", "scipy.stats.skew(e), bias=False", "", "Envelope asymmetry"),
    ("env_kurtosis", "scipy.stats.kurtosis(e), Fisher, bias=False", "", "Envelope peakedness"),
    (
        "env_peak_rate",
        "n_peaks(e) / duration",
        "Hz",
        "Envelope peaks per second; a heart-rate proxy",
    ),
)

#: DWT sub-bands, coarsest first. Fixed here, cross-checked against
#: ``features.yaml`` by :func:`validate_against_config`.
_DWT_SUBBANDS: tuple[str, ...] = ("cA5", "cD5", "cD4", "cD3", "cD2", "cD1")
_DWT_STATS: tuple[tuple[str, str, str, str], ...] = (
    ("energy", "sum(c^2)", "", "Sub-band energy"),
    ("std", "std(c), ddof=0", "", "Sub-band standard deviation"),
    ("entropy", "-sum(p*log2(p)) over a 64-bin histogram of c", "bit", "Sub-band Shannon entropy"),
    ("mean_abs", "mean(|c|)", "", "Sub-band mean absolute coefficient"),
)

_N_MFCC = 13
_N_CHROMA = 12


def _mfcc_specs() -> Iterator[tuple[str, str, str, str]]:
    """39 = 13 means, then 13 stds, then 13 delta means. Blocked, not interleaved.

    Blocked so that a contiguous column slice is one statistic across all
    coefficients -- which is what the ablations and the SHAP grouping in Part VII
    actually slice on.
    """
    for stat, equation, description in (
        ("mean", "mean over frames of MFCC_k", "Mean of MFCC coefficient "),
        ("std", "std over frames of MFCC_k", "Standard deviation of MFCC coefficient "),
        (
            "delta_mean",
            "mean over frames of librosa.feature.delta(MFCC_k)",
            "Mean first-order delta of MFCC coefficient ",
        ),
    ):
        for k in range(1, _N_MFCC + 1):
            yield (
                "mfcc_" + str(k).zfill(2) + "_" + stat,
                equation.replace("_k", "_" + str(k)),
                "",
                description + str(k),
            )


def _chroma_specs() -> Iterator[tuple[str, str, str, str]]:
    for stat, equation, description in (
        ("mean", "mean over frames of chroma_k", "Mean of chroma bin "),
        ("std", "std over frames of chroma_k", "Standard deviation of chroma bin "),
    ):
        for k in range(1, _N_CHROMA + 1):
            yield (
                "chroma_" + str(k).zfill(2) + "_" + stat,
                equation.replace("_k", "_" + str(k)),
                "",
                description + str(k) + " (harmonic-distribution descriptor; see T35.4 caveat)",
            )


def _dwt_specs() -> Iterator[tuple[str, str, str, str]]:
    """24 = 6 sub-bands x 4 statistics, blocked by statistic.

    Blocked by statistic rather than by sub-band for the same reason as MFCC: an
    energy-across-sub-bands slice is the physically meaningful one.
    """
    for stat, equation, _unit, description in _DWT_STATS:
        for band in _DWT_SUBBANDS:
            yield (
                "dwt_" + band + "_" + stat,
                equation.replace("(c", "(" + band),
                _unit,
                description + " (" + band + ")",
            )


def _build() -> tuple[FeatureSpec, ...]:
    """Assemble the 138 specs in locked order and self-check them at import."""
    declared: dict[str, Iterable[tuple[str, str, str, str]]] = {
        "time": _TIME,
        "frequency": _FREQUENCY,
        "mfcc": _mfcc_specs(),
        "chroma": _chroma_specs(),
        "dwt": _dwt_specs(),
        "envelope": _ENVELOPE,
    }

    specs: list[FeatureSpec] = []
    seen: set[str] = set()
    for family in FAMILY_ORDER:
        rows = tuple(declared[family])
        expected = EXPECTED_FAMILY_COUNTS[family]
        if len(rows) != expected:
            raise RegistryError(
                "family '" + family + "' declares " + str(len(rows))
                + " features but the locked composition says " + str(expected)
            )
        for name, equation, unit, description in rows:
            if name in seen:
                raise RegistryError("duplicate feature name: " + name)
            seen.add(name)
            specs.append(
                FeatureSpec(
                    index=len(specs),
                    name=name,
                    family=family,
                    extractor=FAMILY_MODULES[family].rsplit(".", 1)[-1],
                    equation=equation,
                    unit=unit,
                    description=description,
                )
            )

    if len(specs) != EXPECTED_TOTAL:
        raise RegistryError(
            "registry holds " + str(len(specs)) + " features, expected "
            + str(EXPECTED_TOTAL)
        )
    return tuple(specs)


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build()
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_INDEX: Mapping[str, int] = MappingProxyType(
    {spec.name: spec.index for spec in FEATURE_SPECS}
)
_BY_FAMILY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        family: tuple(spec.name for spec in FEATURE_SPECS if spec.family == family)
        for family in FAMILY_ORDER
    }
)


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------


def feature_names(family: str | None = None) -> tuple[str, ...]:
    """All 138 names in column order, or one family's names."""
    if family is None:
        return FEATURE_NAMES
    if family not in _BY_FAMILY:
        raise RegistryError("unknown family: " + str(family))
    return _BY_FAMILY[family]


def specs_for_family(family: str) -> tuple[FeatureSpec, ...]:
    if family not in _BY_FAMILY:
        raise RegistryError("unknown family: " + str(family))
    return tuple(spec for spec in FEATURE_SPECS if spec.family == family)


def spec_for(name: str) -> FeatureSpec:
    index = FEATURE_INDEX.get(name)
    if index is None:
        raise RegistryError("unknown feature name: " + str(name))
    return FEATURE_SPECS[index]


def family_of(name: str) -> str:
    return spec_for(name).family


def index_of(name: str) -> int:
    index = FEATURE_INDEX.get(name)
    if index is None:
        raise RegistryError("unknown feature name: " + str(name))
    return index


def family_counts() -> dict[str, int]:
    return {family: len(_BY_FAMILY[family]) for family in FAMILY_ORDER}


def registry_fingerprint() -> str:
    """SHA-256 of the ordered name list.

    Recorded in the run manifest. Two runs that agree on this hash agree on what
    every column of every feature matrix, model and SHAP plot means; two that do
    not are not comparable, however similar their numbers look.
    """
    joined = "\n".join(FEATURE_NAMES).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def as_records() -> list[dict[str, Any]]:
    """The registry as rows -- the source of FE-01 ``feature_inventory.csv``."""
    return [spec.as_dict() for spec in FEATURE_SPECS]


# ---------------------------------------------------------------------------
# extractor binding
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, Any] = {}


def register_extractor(extractor: Any) -> Any:
    """Bind a family implementation, checking its names against the registry.

    Called at import time by each family module. The name check is the point:
    an extractor that returns a renamed, reordered or short name list is
    rejected here rather than producing a matrix whose columns silently mean
    something else.
    """
    family = getattr(extractor, "family", "")
    if family not in _BY_FAMILY:
        raise RegistryError("extractor declares unknown family: " + str(family))

    names = tuple(extractor.feature_names())
    expected = _BY_FAMILY[family]
    if names != expected:
        raise RegistryError(
            "extractor for family '" + family + "' does not match the registry.\n"
            "  registry: " + str(len(expected)) + " names, first="
            + (expected[0] if expected else "-") + ", last="
            + (expected[-1] if expected else "-") + "\n"
            "  extractor: " + str(len(names)) + " names, first="
            + (names[0] if names else "-") + ", last="
            + (names[-1] if names else "-") + "\n"
            "  differing: "
            + ", ".join(sorted(set(names) ^ set(expected))[:8] or ["<order only>"])
        )

    _EXTRACTORS[family] = extractor
    return extractor


def get_extractor(family: str) -> Any:
    """The implementation for a family, importing its module on first use."""
    if family not in _BY_FAMILY:
        raise RegistryError("unknown family: " + str(family))
    if family in _EXTRACTORS:
        return _EXTRACTORS[family]

    module = FAMILY_MODULES[family]
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ExtractorNotRegistered(
            "family '" + family + "' has no implementation yet (" + module
            + " does not exist)"
        ) from exc

    if family not in _EXTRACTORS:
        raise ExtractorNotRegistered(
            module + " imported but registered no extractor for family '"
            + family + "'"
        )
    return _EXTRACTORS[family]


def registered_families() -> tuple[str, ...]:
    """Families whose implementation is currently bound, in column order."""
    return tuple(family for family in FAMILY_ORDER if family in _EXTRACTORS)


def load_all_extractors() -> dict[str, Any]:
    """Import and bind every family. Raises if any is still missing."""
    return {family: get_extractor(family) for family in FAMILY_ORDER}


# ---------------------------------------------------------------------------
# config agreement
# ---------------------------------------------------------------------------


def validate_against_config(cfg: Any | None = None) -> list[str]:
    """Check the registry against ``configs/features.yaml``. Returns problems.

    Two independent statements of the same locked composition -- the literal
    tables above and the YAML counts -- have to agree. They are kept separate on
    purpose: the YAML is what a reader edits, the tables are what the code uses,
    and a disagreement between them is exactly the kind of silent drift this
    project cannot afford.
    """
    if cfg is None:
        from src.utils.config import load_config

        cfg = load_config("features")

    problems: list[str] = []

    total = cfg.get("expected_total")
    if total != EXPECTED_TOTAL:
        problems.append(
            "features.yaml expected_total=" + str(total)
            + " but the registry holds " + str(EXPECTED_TOTAL)
        )

    for family in FAMILY_ORDER:
        declared = cfg.get("families." + family + ".count")
        actual = len(_BY_FAMILY[family])
        if declared != actual:
            problems.append(
                "features.yaml families." + family + ".count=" + str(declared)
                + " but the registry holds " + str(actual)
            )

    n_mfcc = cfg.get("families.mfcc.n_mfcc")
    if n_mfcc != _N_MFCC:
        problems.append(
            "features.yaml n_mfcc=" + str(n_mfcc) + " but the registry names "
            + str(_N_MFCC) + " coefficients"
        )

    n_chroma = cfg.get("families.chroma.n_chroma")
    if n_chroma != _N_CHROMA:
        problems.append(
            "features.yaml n_chroma=" + str(n_chroma) + " but the registry names "
            + str(_N_CHROMA) + " bins"
        )

    subbands = tuple(cfg.get("families.dwt.subbands") or ())
    if subbands != _DWT_SUBBANDS:
        problems.append(
            "features.yaml dwt.subbands=" + str(list(subbands))
            + " but the registry names " + str(list(_DWT_SUBBANDS))
        )

    stats = tuple(cfg.get("families.dwt.stats") or ())
    registry_stats = tuple(stat for stat, _e, _u, _d in _DWT_STATS)
    if stats != registry_stats:
        problems.append(
            "features.yaml dwt.stats=" + str(list(stats))
            + " but the registry names " + str(list(registry_stats))
        )

    return problems
