"""Assembly of the locked 138: one signal in, one ordered vector out (Phase 38).

Six families computed independently in Phases 32-37 become one row here. The
whole value of this module is that it is the *only* place that turns family
results into a vector, so there is exactly one answer to "what is column 47".

**The length and the order are asserted on every call, not once at import
(T38.2).** That looks paranoid until you consider what a silent violation costs:
a 137-column matrix trains a model whose feature importances are all shifted by
one from the names the SHAP plot prints, and nothing anywhere raises. The check
is a set comparison and two integer comparisons per record -- against ~2.4
seconds of sample entropy, it is free.

**A failed family is 24 NaN in the right columns, never a short row.** The base
class already guarantees each family returns its full name list; this module
additionally records *which* families failed and why, so FE-04 (T41.1) can name
the responsible record and the batch runner can write a real error row rather
than a silent gap.

**Timings come back with the values.** Under joblib's loky backend each worker
has its own process-local timing table, so a global accumulator is invisible to
the parent. :class:`ExtractionResult` therefore carries per-family seconds, and
the batch runner aggregates across workers. Those numbers are a deliverable
(T25/T26), not diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.feature_extraction.base import FamilyResult
from src.feature_extraction.registry import (
    EXPECTED_FAMILY_COUNTS,
    EXPECTED_TOTAL,
    FAMILY_ORDER,
    FEATURE_NAMES,
    as_records,
    family_counts,
    get_extractor,
    registry_fingerprint,
)
from src.utils.logging_setup import get_logger

__all__ = [
    "ExtractionResult",
    "AssemblyError",
    "extract_all",
    "feature_inventory",
    "family_summary",
    "benchmark_families",
    "write_feature_artifacts",
    "FE_ARTIFACTS",
]

log = get_logger(__name__)

#: Artifacts this module emits, by evidence id.
FE_ARTIFACTS: dict[str, str] = {
    "FE-01": "feature_inventory.csv",
    "FE-02": "feature_family_summary.csv",
}

#: Supporting file for the complexity table (T38.6). Not an FE id: T25/T26 own it.
TIMING_FILENAME = "feature_extraction_timing.csv"


class AssemblyError(RuntimeError):
    """The assembled vector does not match the registry contract."""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """One record's complete 138-feature row."""

    values: dict[str, float]
    flags: tuple[str, ...] = ()
    failed_families: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    record_uid: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.values)

    @property
    def vector(self) -> np.ndarray:
        """138 float64 values in registry column order."""
        return np.asarray(list(self.values.values()), dtype=np.float64)

    @property
    def n_missing(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.vector)))

    @property
    def seconds(self) -> float:
        return float(sum(self.timings.values()))

    def missing_names(self) -> list[str]:
        return [name for name, value in self.values.items() if not np.isfinite(value)]

    def as_row(self) -> dict[str, Any]:
        """A flat mapping suitable for a DataFrame row."""
        row: dict[str, Any] = {"record_uid": self.record_uid}
        row.update(self.values)
        row["n_missing"] = self.n_missing
        row["flags"] = ";".join(self.flags)
        row["failed_families"] = ";".join(self.failed_families)
        row["extract_seconds"] = round(self.seconds, 6)
        return row


def _validate(values: Mapping[str, float]) -> None:
    """T38.2 -- length and order must match the registry, on every call."""
    if len(values) != EXPECTED_TOTAL:
        raise AssemblyError(
            "assembled vector has " + str(len(values)) + " values, expected "
            + str(EXPECTED_TOTAL)
        )
    names = tuple(values)
    if names != FEATURE_NAMES:
        # Lengths already agree, so a mismatch must have a differing position.
        differing = [
            index
            for index, (got, want) in enumerate(zip(names, FEATURE_NAMES, strict=True))
            if got != want
        ]
        first_bad = differing[0]
        raise AssemblyError(
            "assembled vector does not match the registry order; first difference at "
            "index " + str(first_bad) + ": got " + names[first_bad]
            + ", expected " + FEATURE_NAMES[first_bad]
            + " (" + str(len(differing)) + " positions differ)"
        )


def extract_all(
    signal: np.ndarray,
    fs: int,
    *,
    record_uid: str | None = None,
    families: Sequence[str] | None = None,
) -> ExtractionResult:
    """The 138 features of one preprocessed signal (T38.1).

    ``families`` restricts *which extractors run*, not how many columns come
    back: an omitted family's 24 slots are NaN, flagged ``<family>_not_run``. The
    vector is always 138 wide, because the EXP-F1 ablation subsets columns from a
    complete matrix rather than producing matrices of different shapes.
    """
    requested = tuple(families) if families is not None else FAMILY_ORDER
    unknown = [name for name in requested if name not in FAMILY_ORDER]
    if unknown:
        raise AssemblyError("unknown family/families: " + ", ".join(unknown))

    values: dict[str, float] = {}
    flags: list[str] = []
    failed: list[str] = []
    errors: dict[str, str] = {}
    timings: dict[str, float] = {}

    for family in FAMILY_ORDER:
        if family not in requested:
            names = get_extractor(family).feature_names()
            values.update(dict.fromkeys(names, float("nan")))
            flags.append(family + "_not_run")
            continue

        result: FamilyResult = get_extractor(family).extract(
            signal, fs, record_uid=record_uid
        )
        if len(result.values) != EXPECTED_FAMILY_COUNTS[family]:
            raise AssemblyError(
                "family '" + family + "' returned " + str(len(result.values))
                + " values, expected " + str(EXPECTED_FAMILY_COUNTS[family])
            )

        values.update(result.values)
        flags.extend(family + ":" + flag for flag in result.flags)
        timings[family] = result.seconds
        if result.failed:
            failed.append(family)
            errors[family] = result.error or "unknown error"

    _validate(values)

    if failed:
        log.warning(
            "record %s: %d of 6 families failed (%s)",
            record_uid or "<unknown>",
            len(failed),
            ", ".join(failed),
        )

    return ExtractionResult(
        values=values,
        flags=tuple(flags),
        failed_families=tuple(failed),
        errors=errors,
        timings=timings,
        record_uid=record_uid,
    )


# ---------------------------------------------------------------------------
# FE-01 and FE-02
# ---------------------------------------------------------------------------


def feature_inventory() -> Any:
    """FE-01 -- all 138 features with family, extractor, equation, unit (T38.3)."""
    import pandas as pd

    frame = pd.DataFrame(as_records())
    return frame[
        ["index", "name", "family", "extractor", "equation", "unit", "description"]
    ]


def family_summary() -> Any:
    """FE-02 -- the six families and their locked counts (T38.4).

    ``expected_count`` is the locked constant and ``n_features`` is what the
    registry actually holds. They are separate columns rather than one, so the
    artifact itself shows the check rather than asserting it somewhere else.
    """
    import pandas as pd

    counts = family_counts()
    rows = [
        {
            "family": family,
            "n_features": counts[family],
            "expected_count": EXPECTED_FAMILY_COUNTS[family],
            "matches_expected": counts[family] == EXPECTED_FAMILY_COUNTS[family],
            "first_index": min(
                spec["index"] for spec in as_records() if spec["family"] == family
            ),
            "extractor": next(
                spec["extractor"] for spec in as_records() if spec["family"] == family
            ),
        }
        for family in FAMILY_ORDER
    ]
    frame = pd.DataFrame(rows)
    total = pd.DataFrame(
        [
            {
                "family": "TOTAL",
                "n_features": int(frame["n_features"].sum()),
                "expected_count": EXPECTED_TOTAL,
                "matches_expected": int(frame["n_features"].sum()) == EXPECTED_TOTAL,
                "first_index": 0,
                "extractor": "",
            }
        ]
    )
    return pd.concat([frame, total], ignore_index=True)


# ---------------------------------------------------------------------------
# T38.6 -- per-family benchmark
# ---------------------------------------------------------------------------


def benchmark_families(
    signal: np.ndarray, fs: int, *, repeats: int = 3, record_uid: str | None = None
) -> Any:
    """Time each family over ``repeats`` extractions of one record (T38.6).

    Reports the **minimum** as well as the mean. On a shared desktop the mean is
    inflated by whatever else the machine was doing; the minimum is the closest
    estimate of the cost itself, and the complexity table needs the cost, not the
    contention.
    """
    import pandas as pd

    samples: dict[str, list[float]] = {family: [] for family in FAMILY_ORDER}
    for _ in range(max(1, int(repeats))):
        result = extract_all(signal, fs, record_uid=record_uid)
        for family, seconds in result.timings.items():
            samples[family].append(seconds)

    rows = []
    for family in FAMILY_ORDER:
        times = np.asarray(samples[family], dtype=np.float64)
        rows.append(
            {
                "family": family,
                "n_features": EXPECTED_FAMILY_COUNTS[family],
                "repeats": int(times.size),
                "min_seconds": float(times.min()),
                "mean_seconds": float(times.mean()),
                "max_seconds": float(times.max()),
                "seconds_per_feature": float(times.min() / EXPECTED_FAMILY_COUNTS[family]),
            }
        )

    frame = pd.DataFrame(rows).sort_values("min_seconds", ascending=False)
    frame["share_of_total"] = frame["min_seconds"] / frame["min_seconds"].sum()
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------


def features_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if out_dir is not None:
        return ensure_dir(out_dir)
    return ensure_dir(load_config("paths").require("outputs.features"))


def write_feature_artifacts(
    out_dir: str | Path | None = None,
    *,
    benchmark_signal: np.ndarray | None = None,
    fs: int = 2000,
    repeats: int = 3,
) -> dict[str, Path]:
    """Emit FE-01, FE-02 and, when given a signal, the timing table (T38.3-T38.6)."""
    from src.utils.io import save_csv

    directory = features_dir(out_dir)
    written: dict[str, Path] = {
        "FE-01": save_csv(feature_inventory(), directory / FE_ARTIFACTS["FE-01"]),
        "FE-02": save_csv(family_summary(), directory / FE_ARTIFACTS["FE-02"]),
    }

    if benchmark_signal is not None:
        written["timing"] = save_csv(
            benchmark_families(benchmark_signal, fs, repeats=repeats),
            directory / TIMING_FILENAME,
        )

    log.info(
        "wrote %d feature artifacts to %s (registry %s)",
        len(written),
        directory,
        registry_fingerprint()[:12],
    )
    return written
