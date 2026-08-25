"""Duration, sampling-rate and class-distribution summaries (Phase 18).

Three deliverables, all read off the catalog and the audio scan rather than
recomputed from the files: **DA-03** ``recording_duration_summary.csv``,
**DA-04** ``sampling_rate_summary.csv`` and **DA-02** ``class_distribution.csv``.

Two decisions worth stating, because both change the numbers.

**Summaries describe the supervised corpus by default.** The PhysioNet
``validation/`` duplicates and the PASCAL unlabelled pool are in the catalog on
purpose, but including them in a class distribution would report a corpus this
project never trains or tests on. Every emitted table carries a ``scope`` column
saying which population it counted, and the full-corpus view is emitted beside
the supervised one rather than instead of it.

**Class distributions are per task, never pooled.** Rule 4 again: there is no
row that adds a PASCAL A ``murmur`` count to a PASCAL B ``murmur`` count. Each
row names its task, and the imbalance ratio is computed within a task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "DURATION_BAND_NAMES",
    "assign_duration_bands",
    "duration_summary",
    "sampling_rate_summary",
    "class_distribution",
    "write_duration_summary",
    "write_sampling_rate_summary",
    "write_class_distribution",
    "run_summaries",
]

log = get_logger(__name__)

DURATION_BAND_NAMES: tuple[str, ...] = ("short", "medium", "long")

_PERCENTILES = (("p25", 0.25), ("median", 0.50), ("p75", 0.75))


def _audit_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return ensure_dir(out_dir)
    from src.utils.config import load_config

    return ensure_dir(load_config("paths").require("outputs.dataset_audit"))


def _target_fs() -> int:
    from src.utils.config import load_config

    return int(load_config("signal").require("resample.target_fs"))


# ---------------------------------------------------------------------------
# duration bands (T18.3)
# ---------------------------------------------------------------------------


def assign_duration_bands(
    catalog: Any, short_below: float | None = None, long_above: float | None = None
) -> Any:
    """Add a ``duration_band`` column: short / medium / long (T18.3).

    Bands come from ``configs/signal.yaml`` -- short below 5 s, long above 20 s.
    These are the strata the robustness track (EXP-E) reports against, so every
    record gets one and none is left unassigned.
    """
    import numpy as np

    if short_below is None or long_above is None:
        from src.data_loader.integrity import load_thresholds

        thresholds = load_thresholds()
        short_below = thresholds.short_below if short_below is None else short_below
        long_above = thresholds.long_above if long_above is None else long_above

    banded = catalog.copy()
    banded["duration_band"] = np.select(
        [
            banded["duration_sec"] < short_below,
            banded["duration_sec"] > long_above,
        ],
        ["short", "long"],
        default="medium",
    )
    return banded


def _describe(values: Any, **extra: Any) -> dict[str, Any]:
    """The seven-figure summary T18.1 asks for, plus whatever identifies it."""
    row: dict[str, Any] = dict(extra)
    row["n"] = int(values.size)
    if values.size == 0:
        for key in ("min", *[name for name, _ in _PERCENTILES], "max", "mean", "sd"):
            row[key] = float("nan")
        return row
    row["min"] = round(float(values.min()), 4)
    for name, quantile in _PERCENTILES:
        row[name] = round(float(values.quantile(quantile)), 4)
    row["max"] = round(float(values.max()), 4)
    row["mean"] = round(float(values.mean()), 4)
    # Sample SD (ddof=1). A single-record group has no spread to report, and
    # pandas returns NaN there rather than 0, which is the honest answer.
    row["sd"] = round(float(values.std(ddof=1)), 4) if values.size > 1 else float("nan")
    return row


# ---------------------------------------------------------------------------
# DA-03 (T18.1, T18.2, T18.4)
# ---------------------------------------------------------------------------


def duration_summary(catalog: Any) -> Any:
    """Per-dataset (T18.1) and per-class-within-dataset (T18.2) durations."""
    import pandas as pd

    from src.data_loader.catalog import (
        DATASET_SHORT_NAMES,
        TASK_CLASS_COLUMNS,
        dataset_tasks,
    )

    banded = assign_duration_bands(catalog)
    rows: list[dict[str, Any]] = []

    for scope, population in (
        ("supervised", banded[banded["use_in_supervised"]]),
        ("all_records", banded),
    ):
        for dataset, group in population.groupby("dataset_source", sort=True):
            common = {
                "scope": scope,
                "dataset_source": dataset,
                "dataset_name": DATASET_SHORT_NAMES.get(str(dataset), str(dataset)),
                "task": "",
                "class": "",
            }
            rows.append(_describe(group["duration_sec"], **{**common, "class": "ALL"}))

            for band in DURATION_BAND_NAMES:
                in_band = group[group["duration_band"] == band]
                rows.append(
                    _describe(
                        in_band["duration_sec"],
                        **{**common, "class": "band:" + band},
                    )
                )

            # T18.2 -- per class, within each of the dataset's own tasks.
            for task in dataset_tasks(str(dataset)):
                column = TASK_CLASS_COLUMNS[task]
                labelled = group[group[column].astype("string").fillna("") != ""]
                for class_name, class_group in labelled.groupby(column, sort=True):
                    rows.append(
                        _describe(
                            class_group["duration_sec"],
                            **{**common, "task": task, "class": str(class_name)},
                        )
                    )

    return pd.DataFrame(rows)


def write_duration_summary(catalog: Any, out_dir: str | Path | None = None) -> Path:
    """Write **DA-03** ``recording_duration_summary.csv`` (T18.4)."""
    target = _audit_dir(out_dir) / "recording_duration_summary.csv"
    summary = duration_summary(catalog)
    save_csv(summary, target)
    log.info("wrote %s (%d rows)", target.name, len(summary))
    return target


# ---------------------------------------------------------------------------
# DA-04 (T18.5)
# ---------------------------------------------------------------------------


def sampling_rate_summary(catalog: Any, scan: Any | None = None) -> Any:
    """Original versus converted sampling rate, with counts (T18.5).

    ``converted_fs`` is the single target every dataset is brought to in Phase
    23. The table exists to make the conversion factor explicit per dataset --
    PASCAL A's 22.05x decimation is the one that needs a proper anti-aliased
    resampler, and it is only obvious next to the others.

    When the audio scan is supplied, the rate in each file's header is checked
    against the rate the decoder actually reported. They agree across this
    corpus; a disagreement would mean every duration derived from the header is
    wrong.
    """
    import pandas as pd

    from src.data_loader.catalog import DATASET_SHORT_NAMES

    target = _target_fs()
    rows: list[dict[str, Any]] = []
    for (dataset, original_fs), group in catalog.groupby(
        ["dataset_source", "original_fs"], sort=True
    ):
        original = int(original_fs)
        rows.append(
            {
                "dataset_source": dataset,
                "dataset_name": DATASET_SHORT_NAMES.get(str(dataset), str(dataset)),
                "original_fs": original,
                "converted_fs": target,
                "conversion": (
                    "none"
                    if original == target
                    else ("decimation" if original > target else "upsampling")
                ),
                "factor": round(original / target, 4) if target else float("nan"),
                "n_records": len(group),
                "n_supervised": int(group["use_in_supervised"].sum()),
                "total_hours": round(float(group["duration_sec"].sum()) / 3600.0, 4),
            }
        )

    summary = pd.DataFrame(rows)

    if scan is not None:
        merged = catalog.merge(
            scan[["record_uid", "fs"]], on="record_uid", how="inner"
        )
        disagreeing = merged[merged["original_fs"].astype(int) != merged["fs"].astype(int)]
        if not disagreeing.empty:
            raise ValueError(
                str(len(disagreeing)) + " record(s) whose header sampling rate "
                "disagrees with the decoded rate, first few: "
                + ", ".join(disagreeing["record_uid"].head(5))
            )
    return summary


def write_sampling_rate_summary(
    catalog: Any, scan: Any | None = None, out_dir: str | Path | None = None
) -> Path:
    """Write **DA-04** ``sampling_rate_summary.csv`` (T18.5)."""
    target = _audit_dir(out_dir) / "sampling_rate_summary.csv"
    summary = sampling_rate_summary(catalog, scan)
    save_csv(summary, target)
    log.info("wrote %s (%d rows)", target.name, len(summary))
    return target


# ---------------------------------------------------------------------------
# DA-02 (T18.6)
# ---------------------------------------------------------------------------


def class_distribution(catalog: Any) -> Any:
    """Dataset by class counts, plus the imbalance ratio (T18.6).

    One row per (scope, dataset, task, class). The imbalance ratio is
    ``largest class / this class`` within its own task, so the majority class
    reads 1.0 and a class with a tenth of its records reads 10.0. Reported per
    row rather than once per task because the per-class figure is what a reader
    needs when looking at a single confusion-matrix row later.
    """
    import pandas as pd

    from src.data_loader.catalog import (
        DATASET_SHORT_NAMES,
        TASK_CLASS_COLUMNS,
        dataset_tasks,
    )

    rows: list[dict[str, Any]] = []
    for scope, population in (
        ("supervised", catalog[catalog["use_in_supervised"]]),
        ("all_records", catalog),
    ):
        for dataset, group in population.groupby("dataset_source", sort=True):
            for task in dataset_tasks(str(dataset)):
                column = TASK_CLASS_COLUMNS[task]
                labelled = group[group[column].astype("string").fillna("") != ""]
                counts = labelled[column].value_counts()
                if counts.empty:
                    continue
                largest = int(counts.max())
                total = int(counts.sum())
                for class_name, count in counts.sort_index().items():
                    rows.append(
                        {
                            "scope": scope,
                            "dataset_source": dataset,
                            "dataset_name": DATASET_SHORT_NAMES.get(
                                str(dataset), str(dataset)
                            ),
                            "task": task,
                            "class": str(class_name),
                            "n_records": int(count),
                            "share": round(int(count) / total, 6),
                            "imbalance_ratio": round(largest / int(count), 4),
                            "n_subjects": int(
                                labelled.loc[
                                    labelled[column] == class_name, "subject_id"
                                ].nunique()
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def write_class_distribution(catalog: Any, out_dir: str | Path | None = None) -> Path:
    """Write **DA-02** ``class_distribution.csv`` (T18.6)."""
    target = _audit_dir(out_dir) / "class_distribution.csv"
    summary = class_distribution(catalog)
    save_csv(summary, target)
    log.info("wrote %s (%d rows)", target.name, len(summary))
    return target


def run_summaries(
    catalog: Any,
    scan: Any | None = None,
    *,
    write_outputs: bool = False,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Phase 18 end to end. Returns the three summary frames."""
    summaries = {
        "duration": duration_summary(catalog),
        "sampling_rate": sampling_rate_summary(catalog, scan),
        "class_distribution": class_distribution(catalog),
    }
    if write_outputs:
        write_duration_summary(catalog, out_dir)
        write_sampling_rate_summary(catalog, scan, out_dir)
        write_class_distribution(catalog, out_dir)
    return summaries
