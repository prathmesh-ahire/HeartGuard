"""FE-03: the merged feature matrix (Phase 40).

One row per record, master metadata joined to the locked 138 by ``record_uid``.
This is the table every model in Part V and beyond trains on, so the join is
checked rather than trusted.

**The join is asserted to be exactly one-to-one and total.** A left join that
quietly leaves NaN where a shard was short, or an inner join that quietly drops
records master knows about, both produce a matrix that *looks* fine -- correct
dtypes, plausible values, no warning -- and silently changes what the corpus is.
An extraction that covered 7,100 of 7,536 records would train a model, report a
metric, and never mention the 436 records it never saw. So
:func:`build_matrix` compares the two ``record_uid`` sets in both directions and
raises with the missing identifiers named.

The 138 feature columns appear in **registry order** here exactly as they do in
a shard and in ``extract_all``. Column position is meaning: see
``tests/test_extract_all.py`` for why that is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.feature_extraction.batch import (
    DATASETS,
    TIMING_COLUMNS,
    cache_digest,
    load_shard,
)
from src.feature_extraction.registry import FEATURE_NAMES
from src.utils.logging_setup import get_logger

__all__ = [
    "FE03_FILENAME",
    "WALL_TIME_FILENAME",
    "MatrixReport",
    "MatrixError",
    "build_matrix",
    "write_matrix",
    "load_matrix",
    "matrix_path",
    "wall_time_table",
]

log = get_logger("features.matrix")

FE03_FILENAME = "all_features_matrix.parquet"
WALL_TIME_FILENAME = "extraction_wall_time.csv"

#: Shard bookkeeping carried into FE-03. ``ok``/``error`` stay behind: a matrix
#: row exists only for a record that produced one, and per-family failures are
#: already itemised in ``extraction_errors.csv`` (Phase 39) and FE-04 (Phase 41).
CARRIED_META: tuple[str, ...] = (
    "n_missing",
    "flags",
    "failed_families",
    "extract_seconds",
)


class MatrixError(RuntimeError):
    """The shards and master metadata do not agree."""


@dataclass
class MatrixReport:
    """What the merged matrix contains, for the T40.7 gate and the write-up."""

    n_rows: int = 0
    n_features: int = 0
    n_meta: int = 0
    datasets: dict[str, int] = field(default_factory=dict)
    digest: str = ""
    n_incomplete_rows: int = 0
    total_extract_seconds: float = 0.0

    @property
    def n_columns(self) -> int:
        return self.n_meta + self.n_features


def matrix_path(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if out_dir is not None:
        return ensure_dir(out_dir) / FE03_FILENAME
    return ensure_dir(load_config("paths").require("outputs.features")) / FE03_FILENAME


def build_matrix(
    datasets: tuple[str, ...] | list[str] | None = None,
    *,
    digest: str | None = None,
    master: Any | None = None,
) -> tuple[Any, MatrixReport]:
    """Concatenate the per-dataset shards and join them to master metadata.

    Raises :class:`MatrixError` if the join is not one-to-one and total.
    """
    import pandas as pd

    from src.data_loader import master as ms

    chosen = tuple(datasets) if datasets else DATASETS
    unknown = [name for name in chosen if name not in DATASETS]
    if unknown:
        raise MatrixError("unknown dataset(s): " + ", ".join(unknown))

    resolved_digest = digest or cache_digest()
    if master is None:
        master = ms.load_master()
    master = master[master["dataset_source"].astype(str).isin(chosen)].copy()
    master["record_uid"] = master["record_uid"].astype(str)

    shards = []
    for dataset in chosen:
        shard = load_shard(dataset, digest=resolved_digest)
        shard["record_uid"] = shard["record_uid"].astype(str)
        log.info("%s shard: %d rows", dataset, len(shard))
        shards.append(shard)
    features = pd.concat(shards, ignore_index=True)

    _check_join(master, features)

    # ``dataset_source`` lives in both frames; master's copy is authoritative.
    keep = ["record_uid", *CARRIED_META, *TIMING_COLUMNS, *FEATURE_NAMES]
    keep = [column for column in keep if column in features.columns]
    merged = master.merge(
        features[keep], on="record_uid", how="inner", validate="one_to_one"
    )
    if len(merged) != len(master):
        raise MatrixError(
            "join changed the row count: "
            + str(len(master))
            + " master rows -> "
            + str(len(merged))
        )

    meta_columns = [column for column in merged.columns if column not in FEATURE_NAMES]
    merged = merged[[*meta_columns, *FEATURE_NAMES]]
    merged = merged.sort_values(["dataset_source", "record_uid"]).reset_index(drop=True)

    report = MatrixReport(
        n_rows=len(merged),
        n_features=len(FEATURE_NAMES),
        n_meta=len(meta_columns),
        datasets={
            str(name): int(count)
            for name, count in merged["dataset_source"].value_counts().items()
        },
        digest=resolved_digest,
        n_incomplete_rows=int((merged["n_missing"].astype(float) > 0).sum()),
        total_extract_seconds=float(merged["extract_seconds"].astype(float).sum()),
    )
    return merged, report


def _check_join(master: Any, features: Any) -> None:
    """Both directions, named identifiers, before anything is merged."""
    master_uids = set(master["record_uid"])
    feature_uids = set(features["record_uid"])

    duplicated = len(features) - len(feature_uids)
    if duplicated:
        raise MatrixError(
            "shards contain " + str(duplicated) + " duplicate record_uid rows"
        )

    missing = sorted(master_uids - feature_uids)
    if missing:
        raise MatrixError(
            "extraction is incomplete: "
            + str(len(missing))
            + " master records have no features, e.g. "
            + ", ".join(missing[:5])
        )

    orphan = sorted(feature_uids - master_uids)
    if orphan:
        raise MatrixError(
            str(len(orphan))
            + " extracted records are absent from master, e.g. "
            + ", ".join(orphan[:5])
        )


def write_matrix(
    out_dir: str | Path | None = None,
    *,
    datasets: tuple[str, ...] | list[str] | None = None,
    digest: str | None = None,
    master: Any | None = None,
) -> tuple[Path, MatrixReport]:
    """Build FE-03 and write it, returning the path and the report."""
    merged, report = build_matrix(datasets, digest=digest, master=master)
    path = matrix_path(out_dir)
    merged.to_parquet(path, index=False)
    log.info(
        "FE-03: %d rows x %d columns -> %s", report.n_rows, report.n_columns, path
    )
    return path, report


def load_matrix(out_dir: str | Path | None = None) -> Any:
    """Read FE-03 from disk."""
    import pandas as pd

    path = matrix_path(out_dir)
    if not path.is_file():
        raise FileNotFoundError("FE-03 not found at " + str(path) + "; run Phase 40")
    return pd.read_parquet(path)


def wall_time_table(matrix: Any | None = None) -> Any:
    """Per-dataset extraction cost (T40.1-T40.5), from the matrix itself.

    ``extract_seconds`` is CPU time inside the worker, so the totals here exceed
    the wall clock of a parallel run. The run manifest carries the wall clock;
    this table carries the work done, which is what the complexity deliverables
    (T25/T26) compare across datasets.
    """
    import pandas as pd

    if matrix is None:
        matrix = load_matrix()

    rows = []
    for dataset, group in matrix.groupby("dataset_source", sort=True):
        seconds = group["extract_seconds"].astype(float)
        rows.append(
            {
                "dataset": str(dataset),
                "n_records": len(group),
                "total_audio_sec": float(group["duration_sec"].astype(float).sum()),
                "total_extract_sec": float(seconds.sum()),
                "mean_sec_per_record": float(seconds.mean()),
                "median_sec_per_record": float(seconds.median()),
                "max_sec_per_record": float(seconds.max()),
            }
        )

    frame = pd.DataFrame(rows)
    total_seconds = frame["total_extract_sec"].sum()
    frame["share_of_total"] = frame["total_extract_sec"] / total_seconds
    frame["sec_per_audio_sec"] = frame["total_extract_sec"] / frame["total_audio_sec"]
    return frame
