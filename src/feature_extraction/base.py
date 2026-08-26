"""The extractor contract, the NaN policy and per-family timing (Phase 31).

Six families produce the locked 138 features. They share nothing numerically --
a DWT sub-band entropy and an MFCC delta mean have no common maths -- but they
share three obligations, and this module is where those live so that six
implementations cannot drift into six different answers.

**1. An extractor never raises (T31.4).** A corpus of 7,536 recordings made by
different people on different devices contains records that break things: one is
effectively silent, one is 0.76 s long, one runs 122 s. A family that throws on
any of them kills a six-hour batch run at record 4,000. So
:meth:`BaseFeatureExtractor.extract` catches everything, returns the family's
full name list with ``NaN`` in every slot, and logs the failure against the
record uid. The row survives, the gap is visible in FE-04, and rule 1 holds: a
missing number is reported, never invented and never silently dropped.

The corollary matters as much: **NaN means "this could not be computed", and it
is never left unexplained.** Either the whole family failed (``failed=True``,
every slot NaN, the exception in ``error``) or a *flag* names the measure that
could not run -- a 1-sample signal has no sample entropy, and
``sampen_too_short`` says so. What no extractor may do is return NaN as an
ordinary value for a well-formed signal: a degenerate-but-valid case (a constant
signal has zero crest factor, not an undefined one) returns a real number and
sets a flag. FE-04 (T41.1) reports every NaN in the matrix against its record,
and an unexplained one there is indistinguishable from a bug.

**2. Shape is fixed before the maths runs.** :meth:`extract` asks the subclass
for a mapping, then reindexes it against :meth:`feature_names`. A missing key
becomes NaN and a stray key is an error, so a family cannot quietly return 23
values and shift every column index downstream of it. This is what makes the
registry's fixed ordering (T31.3) enforceable rather than aspirational.

**3. Timing is collected per family, not per call (T31.5).** The complexity
deliverable (T25/T26) needs per-family extraction cost across the corpus.
Writing one manifest row per family per record would be 45,000 rows of noise, so
durations accumulate in a process-local table that the batch runner reads once
at the end.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "FeatureExtractor",
    "BaseFeatureExtractor",
    "FamilyResult",
    "FamilyTiming",
    "timing_table",
    "reset_timings",
    "record_timing",
    "nan_result",
    "as_float",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# the contract (T31.1)
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureExtractor(Protocol):
    """What every family must provide.

    ``name`` identifies the extractor, ``family`` names the registry family it
    fills, ``feature_names()`` is its ordered, fixed name list, and ``extract``
    turns one signal into a :class:`FamilyResult`.
    """

    name: str
    family: str

    def feature_names(self) -> tuple[str, ...]:
        ...

    def extract(
        self, signal: np.ndarray, fs: int, *, record_uid: str | None = None
    ) -> FamilyResult:
        ...


@dataclass(frozen=True, slots=True)
class FamilyResult:
    """One family's contribution to one record's feature vector.

    ``values`` is always the family's full name list in registry order, with NaN
    where a value could not be computed. ``flags`` records non-fatal degradation
    -- a shortened DWT level, a shrunk delta width, a zero-variance signal -- so
    a value that is real but computed under a relaxed setting stays traceable
    rather than indistinguishable from a clean one.
    """

    family: str
    values: dict[str, float]
    seconds: float
    flags: tuple[str, ...] = ()
    failed: bool = False
    error: str | None = None
    record_uid: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.values)

    @property
    def vector(self) -> np.ndarray:
        """The values as a float64 array in name order."""
        return np.asarray(list(self.values.values()), dtype=np.float64)

    @property
    def n_missing(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.vector)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "seconds": self.seconds,
            "flags": list(self.flags),
            "failed": self.failed,
            "error": self.error,
            "record_uid": self.record_uid,
            "n_missing": self.n_missing,
        }


# ---------------------------------------------------------------------------
# per-family timing (T31.5)
# ---------------------------------------------------------------------------


@dataclass
class FamilyTiming:
    """Accumulated wall time for one family across many records."""

    family: str
    calls: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0
    failures: int = 0

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else float("nan")

    def add(self, seconds: float, *, failed: bool = False) -> None:
        self.calls += 1
        self.total_seconds += seconds
        self.min_seconds = min(self.min_seconds, seconds)
        self.max_seconds = max(self.max_seconds, seconds)
        if failed:
            self.failures += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "calls": self.calls,
            "total_seconds": round(self.total_seconds, 6),
            "mean_seconds": round(self.mean_seconds, 6) if self.calls else float("nan"),
            "min_seconds": round(self.min_seconds, 6) if self.calls else float("nan"),
            "max_seconds": round(self.max_seconds, 6),
            "failures": self.failures,
        }


_TIMINGS: dict[str, FamilyTiming] = {}
_TIMING_LOCK = threading.Lock()


def record_timing(family: str, seconds: float, *, failed: bool = False) -> None:
    """Add one measurement to the process-local timing table."""
    with _TIMING_LOCK:
        entry = _TIMINGS.get(family)
        if entry is None:
            entry = FamilyTiming(family=family)
            _TIMINGS[family] = entry
        entry.add(seconds, failed=failed)


def timing_table() -> dict[str, FamilyTiming]:
    """A snapshot of accumulated per-family timings.

    Process-local by design. Under joblib's default loky backend each worker
    keeps its own table, so the batch runner (Phase 39) collects and merges them
    rather than reading a shared global.
    """
    with _TIMING_LOCK:
        return dict(_TIMINGS)


def reset_timings() -> None:
    with _TIMING_LOCK:
        _TIMINGS.clear()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def as_float(value: Any) -> float:
    """Coerce to a plain float, mapping anything non-numeric to NaN.

    numpy scalars, 0-d arrays and Python ints all arrive here from six different
    families; the feature matrix stores float64 and nothing else.
    """
    try:
        return float(np.asarray(value, dtype=np.float64).reshape(()))
    except Exception:  # noqa: BLE001
        # Deliberately blind: this is the last line of the NaN policy. Whatever a
        # family hands back -- a string, a tuple, a masked array -- becomes NaN
        # rather than an exception that ends a 7,536-record run.
        return float("nan")


def nan_result(
    family: str,
    names: Sequence[str],
    *,
    seconds: float = 0.0,
    error: str | None = None,
    record_uid: str | None = None,
    flags: Sequence[str] = (),
) -> FamilyResult:
    """A full-width all-NaN result for a family that could not run."""
    return FamilyResult(
        family=family,
        values={name: float("nan") for name in names},
        seconds=seconds,
        flags=tuple(flags),
        failed=True,
        error=error,
        record_uid=record_uid,
    )


# ---------------------------------------------------------------------------
# the base class every family subclasses
# ---------------------------------------------------------------------------


class BaseFeatureExtractor(ABC):
    """Implements the contract; subclasses implement only ``_compute``.

    A subclass declares ``family`` and ``name``, returns its ordered names from
    :meth:`feature_names`, and computes a mapping in :meth:`_compute`. Everything
    else -- the never-raise policy, the shape check, the timing, the flag
    plumbing -- happens here, once.
    """

    #: registry family this extractor fills
    family: str = ""
    #: identifier for logs and the run manifest
    name: str = ""

    def __init__(self, cfg: Any | None = None) -> None:
        self._cfg = cfg

    # -- subclass interface -------------------------------------------------

    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """The family's ordered names. Must match the registry exactly."""

    @abstractmethod
    def _compute(
        self, signal: np.ndarray, fs: int, flags: list[str]
    ) -> Mapping[str, float]:
        """Compute the family. May raise; :meth:`extract` catches.

        Append to ``flags`` to record non-fatal degradation.
        """

    # -- the public path ----------------------------------------------------

    def extract(
        self, signal: np.ndarray, fs: int, *, record_uid: str | None = None
    ) -> FamilyResult:
        """Run the family under the NaN policy. Never raises."""
        names = self.feature_names()
        flags: list[str] = []
        started = time.perf_counter()

        try:
            prepared = self._prepare(signal)
            raw = self._compute(prepared, int(fs), flags)
            values = self._align(raw, names)
            failed = False
            error = None
        except Exception as exc:  # noqa: BLE001 -- the whole point of the policy
            values = {name: float("nan") for name in names}
            failed = True
            error = type(exc).__name__ + ": " + str(exc)
            log.warning(
                "feature extraction failed: family=%s record_uid=%s error=%s",
                self.family,
                record_uid or "<unknown>",
                error,
            )

        seconds = time.perf_counter() - started
        record_timing(self.family, seconds, failed=failed)

        return FamilyResult(
            family=self.family,
            values=values,
            seconds=seconds,
            flags=tuple(flags),
            failed=failed,
            error=error,
            record_uid=record_uid,
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _prepare(signal: np.ndarray) -> np.ndarray:
        """Validate and normalize the input to a 1-D float64 array.

        float64 rather than the float32 the pipeline emits: sample entropy,
        autocorrelation and spectral flux all accumulate over hundreds of
        thousands of samples, where float32 rounding reaches the reported digits
        and would make rule 5's bit-identical rerun promise depend on
        accumulation order.
        """
        prepared = np.asarray(signal, dtype=np.float64)
        if prepared.ndim == 2 and 1 in prepared.shape:
            prepared = prepared.reshape(-1)
        if prepared.ndim != 1:
            raise ValueError("signal must be 1-D, got shape " + str(prepared.shape))
        if prepared.size == 0:
            raise ValueError("signal is empty")
        if not np.all(np.isfinite(prepared)):
            raise ValueError("signal contains NaN or Inf samples")
        return prepared

    def _align(
        self, raw: Mapping[str, float], names: tuple[str, ...]
    ) -> dict[str, float]:
        """Reindex a computed mapping onto the family's fixed name list."""
        unexpected = set(raw) - set(names)
        if unexpected:
            raise ValueError(
                self.family
                + " returned unregistered feature name(s): "
                + ", ".join(sorted(unexpected))
            )
        missing = [name for name in names if name not in raw]
        if missing:
            log.warning(
                "family=%s did not produce %d of its %d names: %s",
                self.family,
                len(missing),
                len(names),
                ", ".join(missing[:5]),
            )
        return {name: as_float(raw.get(name, float("nan"))) for name in names}
