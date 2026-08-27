"""Batch feature extraction with resumable checkpoints (Phase 39).

Extracting 138 features from 7,536 recordings is the longest single job in this
project -- projected at several CPU-hours, dominated by sample entropy (open
decision 7). Three properties follow from that length, and they are what this
module exists to provide.

**A run that dies must resume, not restart (T39.3).** Records are processed in
chunks of ``extraction.checkpoint_every`` and each completed chunk is written to
its own Parquet part immediately. A restart reads the parts, subtracts the uids
already done, and continues. Losing an hour of work to a closed laptop is the
difference between a job that gets finished and a job that gets abandoned.

**A cache keyed by anything less than the full contract is a trap.** The shard
directory is named by a digest of the **registry fingerprint**, the
**preprocessing config hash** and the **feature settings**. Change any of them
and the extraction lands in a different directory rather than silently mixing
138 columns that mean two different things. This is the same rule
``preprocessing/pipeline.py`` follows, for the same reason: a different setting
is a different path.

**One bad record must not kill the run (T39.4).** A worker catches everything,
returns an error row, and the run continues. Failures are written to
``extraction_errors.csv`` with the record uid, the family and the exception --
per family, not per record, because "MFCC failed on a 0.4 s upload" and "the
file would not decode" need different fixes.

**Smoke runs never write where a full run writes.** ``--smoke`` goes to a
``_smoke`` subtree. A 20-row shard sitting where a 3,240-row shard belongs reads
exactly like a complete one, which is the trap ``scripts/01_run_dataset_audit.py``
documents for the audit and the same trap applies here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.feature_extraction.registry import FAMILY_ORDER, FEATURE_NAMES, registry_fingerprint
from src.utils.logging_setup import get_logger

__all__ = [
    "DATASETS",
    "ERRORS_FILENAME",
    "BatchSummary",
    "cache_digest",
    "features_root",
    "shard_path",
    "checkpoint_dir",
    "completed_uids",
    "extract_dataset",
    "run_extraction",
    "load_shard",
    "clear_feature_cache",
]

log = get_logger(__name__)

#: Dataset ids in the order the runner walks them.
DATASETS: tuple[str, ...] = ("D1", "D2", "D3", "D4")

ERRORS_FILENAME = "extraction_errors.csv"

#: Bump when the *shape* of a shard row changes, so an older shard can never be
#: read as a newer one. Part of the cache digest.
SHARD_VERSION = 1

#: Non-feature columns every shard carries.
META_COLUMNS: tuple[str, ...] = (
    "record_uid",
    "dataset_source",
    "n_missing",
    "flags",
    "failed_families",
    "extract_seconds",
    "ok",
    "error",
)

#: Per-family timing columns, for the complexity table (T25/T26).
TIMING_COLUMNS: tuple[str, ...] = tuple("sec_" + family for family in FAMILY_ORDER)


@dataclass
class BatchSummary:
    """What one extraction run did."""

    datasets: dict[str, int] = field(default_factory=dict)
    n_records: int = 0
    n_failed: int = 0
    n_resumed: int = 0
    seconds: float = 0.0
    digest: str = ""
    smoke: bool = False
    shards: dict[str, str] = field(default_factory=dict)
    errors_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "datasets": dict(self.datasets),
            "n_records": self.n_records,
            "n_failed": self.n_failed,
            "n_resumed": self.n_resumed,
            "seconds": round(self.seconds, 3),
            "digest": self.digest,
            "smoke": self.smoke,
            "shards": dict(self.shards),
            "errors_path": self.errors_path,
        }


# ---------------------------------------------------------------------------
# cache identity
# ---------------------------------------------------------------------------


def cache_digest(cfg: Any | None = None) -> str:
    """12-hex digest of everything that changes what a feature value means.

    Deliberately includes the registry fingerprint: a renamed or reordered
    column changes what every number in the shard refers to, even when the
    arithmetic is untouched.
    """
    from src.preprocessing.pipeline import config_hash
    from src.utils.config import load_config

    features = cfg if cfg is not None else load_config("features")
    payload = {
        "shard_version": SHARD_VERSION,
        "registry": registry_fingerprint(),
        "signal": config_hash(),
        "families": features.get("families"),
        "expected_total": features.get("expected_total"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def features_root(*, digest: str | None = None, smoke: bool = False) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    base = Path(load_config("paths").require("cache.features"))
    if smoke:
        base = base / "_smoke"
    return ensure_dir(base / (digest or cache_digest()))


def shard_path(dataset: str, *, digest: str | None = None, smoke: bool = False) -> Path:
    return features_root(digest=digest, smoke=smoke) / (dataset + "_features.parquet")


def checkpoint_dir(
    dataset: str, *, digest: str | None = None, smoke: bool = False
) -> Path:
    from src.utils.io import ensure_dir

    return ensure_dir(features_root(digest=digest, smoke=smoke) / "_checkpoints" / dataset)


def _part_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("part_*.parquet"))


def completed_uids(
    dataset: str, *, digest: str | None = None, smoke: bool = False
) -> set[str]:
    """Record uids already extracted, from a finished shard or partial checkpoints."""
    import pandas as pd

    done: set[str] = set()

    final = shard_path(dataset, digest=digest, smoke=smoke)
    if final.is_file():
        done.update(pd.read_parquet(final, columns=["record_uid"])["record_uid"].astype(str))

    for part in _part_paths(checkpoint_dir(dataset, digest=digest, smoke=smoke)):
        try:
            done.update(pd.read_parquet(part, columns=["record_uid"])["record_uid"].astype(str))
        except Exception:  # noqa: BLE001
            # A part written by a process killed mid-write is unreadable. Drop it
            # and re-extract those records rather than trusting a truncated file.
            log.warning("discarding unreadable checkpoint %s", part.name)
            part.unlink(missing_ok=True)

    return done


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------


def _extract_one(uid: str, relative_path: str, dataset: str, root: str) -> dict[str, Any]:
    """Worker body. Picklable arguments only; loads its own config."""
    from src.feature_extraction.extractor import extract_all
    from src.preprocessing.pipeline import preprocess

    row: dict[str, Any] = {"record_uid": uid, "dataset_source": dataset}

    try:
        signal = preprocess(Path(root) / relative_path, record_uid=uid).signal
    except Exception as exc:
        log.exception("preprocessing failed for %s", uid)
        row.update(dict.fromkeys(FEATURE_NAMES, float("nan")))
        row.update(
            {
                "n_missing": len(FEATURE_NAMES),
                "flags": "",
                "failed_families": "preprocess",
                "extract_seconds": 0.0,
                "ok": False,
                "error": type(exc).__name__ + ": " + str(exc),
            }
        )
        row.update(dict.fromkeys(TIMING_COLUMNS, 0.0))
        return row

    result = extract_all(signal, 2000, record_uid=uid)
    row.update(result.as_row())
    row["dataset_source"] = dataset
    row["ok"] = not result.failed_families
    row["error"] = "; ".join(
        family + ": " + message for family, message in sorted(result.errors.items())
    )
    for family in FAMILY_ORDER:
        row["sec_" + family] = round(result.timings.get(family, 0.0), 6)
    return row


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------


def extract_dataset(
    dataset: str,
    master: Any | None = None,
    *,
    n_jobs: int = -1,
    force: bool = False,
    limit: int | None = None,
    checkpoint_every: int | None = None,
    smoke: bool = False,
    progress: bool = True,
    digest: str | None = None,
    on_chunk: Callable[[int, int], None] | None = None,
) -> Any:
    """Extract one dataset, resuming from checkpoints unless ``force`` (T39.2, T39.3).

    ``on_chunk(chunk_index, n_written)`` fires after each checkpoint is safely on
    disk. It exists so T39.7 can interrupt a run at exactly the point a real
    interrupt is survivable, rather than at an arbitrary moment a test happens to
    reach.
    """
    import pandas as pd
    from joblib import Parallel, delayed

    from src.data_loader import master as ms
    from src.utils.config import load_config
    from src.utils.timing import timer

    features_cfg = load_config("features")
    if checkpoint_every is None:
        checkpoint_every = int(features_cfg.get("extraction.checkpoint_every", 250))
    resolved_digest = digest or cache_digest(features_cfg)

    if master is None:
        master = ms.load_master()
    subset = master[master["dataset_source"] == dataset]
    if limit is not None:
        subset = subset.head(int(limit))
    if subset.empty:
        raise ValueError("no records for dataset " + dataset)

    parts_dir = checkpoint_dir(dataset, digest=resolved_digest, smoke=smoke)
    final = shard_path(dataset, digest=resolved_digest, smoke=smoke)

    if force:
        shutil.rmtree(parts_dir, ignore_errors=True)
        final.unlink(missing_ok=True)
        parts_dir = checkpoint_dir(dataset, digest=resolved_digest, smoke=smoke)
        done: set[str] = set()
    else:
        done = completed_uids(dataset, digest=resolved_digest, smoke=smoke)

    targets = [
        (str(row.record_uid), str(row.file_path))
        for row in subset.itertuples(index=False)
        if str(row.record_uid) not in done
    ]
    n_resumed = len(subset) - len(targets)
    if n_resumed:
        log.info(
            "%s: resuming -- %d of %d records already extracted",
            dataset,
            n_resumed,
            len(subset),
        )

    root = str(Path(load_config("paths").require("project_root")))
    chunks = [
        targets[start : start + checkpoint_every]
        for start in range(0, len(targets), checkpoint_every)
    ]

    with timer("extract:" + dataset):
        for index, chunk in enumerate(chunks):
            iterator: Any = chunk
            if progress:
                from tqdm import tqdm

                iterator = tqdm(
                    chunk,
                    desc="extract " + dataset + " " + str(index + 1) + "/" + str(len(chunks)),
                    unit="rec",
                )

            rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=8)(
                delayed(_extract_one)(uid, relative, dataset, root)
                for uid, relative in iterator
            )

            part = parts_dir / ("part_" + str(_next_part_index(parts_dir)).zfill(5) + ".parquet")
            _write_parquet(pd.DataFrame(rows), part)
            log.info("%s: checkpoint %s (%d records)", dataset, part.name, len(rows))

            if on_chunk is not None:
                on_chunk(index, len(rows))

    table = _merge_parts(dataset, digest=resolved_digest, smoke=smoke)
    keep = set(subset["record_uid"].astype(str))
    table = table[table["record_uid"].astype(str).isin(keep)].reset_index(drop=True)
    _write_parquet(table, final)
    shutil.rmtree(parts_dir, ignore_errors=True)

    log.info(
        "%s: %d records -> %s (%d failed, %d resumed)",
        dataset,
        len(table),
        final.name,
        int((~table["ok"]).sum()),
        n_resumed,
    )
    return table


def _next_part_index(directory: Path) -> int:
    existing = _part_paths(directory)
    if not existing:
        return 0
    return max(int(path.stem.split("_")[-1]) for path in existing) + 1


def _write_parquet(frame: Any, path: Path) -> Path:
    from src.utils.io import save_parquet

    return save_parquet(frame, path)


def _merge_parts(dataset: str, *, digest: str, smoke: bool) -> Any:
    """Concatenate checkpoints and any prior shard, newest row per uid winning."""
    import pandas as pd

    frames = []
    final = shard_path(dataset, digest=digest, smoke=smoke)
    if final.is_file():
        frames.append(pd.read_parquet(final))
    frames.extend(
        pd.read_parquet(part)
        for part in _part_paths(checkpoint_dir(dataset, digest=digest, smoke=smoke))
    )
    if not frames:
        return pd.DataFrame(columns=["record_uid", *FEATURE_NAMES, *META_COLUMNS])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="record_uid", keep="last")
    return merged.sort_values("record_uid").reset_index(drop=True)


# ---------------------------------------------------------------------------
# the whole run
# ---------------------------------------------------------------------------


def run_extraction(
    datasets: Sequence[str] | None = None,
    *,
    n_jobs: int = -1,
    force: bool = False,
    limit: int | None = None,
    smoke: bool = False,
    progress: bool = True,
    out_dir: str | Path | None = None,
    master: Any | None = None,
) -> BatchSummary:
    """Extract every requested dataset and write the error report (T39.1-T39.6)."""
    import time

    import pandas as pd

    from src.data_loader import master as ms
    from src.utils.config import load_config

    features_cfg = load_config("features")
    if smoke and limit is None:
        limit = int(features_cfg.get("extraction.smoke_limit", 20))

    chosen = tuple(datasets) if datasets else DATASETS
    unknown = [name for name in chosen if name not in DATASETS]
    if unknown:
        raise ValueError("unknown dataset(s): " + ", ".join(unknown))

    if master is None:
        master = ms.load_master()

    digest = cache_digest(features_cfg)
    summary = BatchSummary(digest=digest, smoke=smoke)
    started = time.perf_counter()
    error_rows: list[dict[str, Any]] = []

    for dataset in chosen:
        table = extract_dataset(
            dataset,
            master,
            n_jobs=n_jobs,
            force=force,
            limit=limit,
            smoke=smoke,
            progress=progress,
            digest=digest,
        )
        summary.datasets[dataset] = len(table)
        summary.n_records += len(table)
        summary.shards[dataset] = str(shard_path(dataset, digest=digest, smoke=smoke))
        error_rows.extend(_error_rows(table))

    summary.n_failed = len(error_rows)
    summary.seconds = time.perf_counter() - started

    errors = pd.DataFrame(
        error_rows,
        columns=["record_uid", "dataset_source", "family", "error", "n_missing", "flags"],
    )
    summary.errors_path = str(_write_errors(errors, out_dir, smoke=smoke))

    log.info(
        "extraction complete: %d records across %s, %d family failures, %s",
        summary.n_records,
        ", ".join(chosen),
        summary.n_failed,
        summary.errors_path,
    )
    return summary


def _error_rows(table: Any) -> list[dict[str, Any]]:
    """One row per (record, failed family) -- T39.4 asks for the family, not just the record."""
    rows: list[dict[str, Any]] = []
    failures = table[table["failed_families"].astype(str).str.len() > 0]
    for record in failures.itertuples(index=False):
        messages = _parse_errors(str(getattr(record, "error", "") or ""))
        for family in str(record.failed_families).split(";"):
            if not family:
                continue
            rows.append(
                {
                    "record_uid": str(record.record_uid),
                    "dataset_source": str(record.dataset_source),
                    "family": family,
                    "error": messages.get(family, str(getattr(record, "error", "") or "")),
                    "n_missing": int(record.n_missing),
                    "flags": str(record.flags),
                }
            )
    return rows


def _parse_errors(blob: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for chunk in blob.split("; "):
        family, separator, message = chunk.partition(": ")
        if separator and family in FAMILY_ORDER:
            parsed[family] = message
    return parsed


def _write_errors(errors: Any, out_dir: str | Path | None, *, smoke: bool) -> Path:
    """T39.4 -- the error report, and where a smoke run's copy goes instead.

    A smoke run's report lands beside its shards in the ``_smoke`` cache tree,
    never in ``outputs/03_features/``. ``outputs/`` holds deliverables: an
    ``extraction_errors.csv`` there must describe the run that produced FE-03,
    and a 20-record smoke report sitting next to it is indistinguishable at a
    glance from the real one.
    """
    from src.feature_extraction.extractor import features_dir
    from src.utils.io import save_csv

    if smoke and out_dir is None:
        directory = features_root(smoke=True)
    else:
        directory = features_dir(out_dir)
    name = ("smoke_" if smoke else "") + ERRORS_FILENAME
    return save_csv(errors, directory / name)


# ---------------------------------------------------------------------------
# reading back
# ---------------------------------------------------------------------------


def load_shard(
    dataset: str, *, digest: str | None = None, smoke: bool = False, features_only: bool = False
) -> Any:
    """Read one dataset's shard. ``features_only`` returns uid + the 138 columns."""
    import pandas as pd

    path = shard_path(dataset, digest=digest, smoke=smoke)
    if not path.is_file():
        raise FileNotFoundError("no feature shard for " + dataset + " at " + str(path))

    frame = pd.read_parquet(path)
    if features_only:
        return frame[["record_uid", *FEATURE_NAMES]]
    return frame


def clear_feature_cache(*, digest: str | None = None, smoke: bool = False) -> int:
    """Delete a feature-cache tree. Returns how many files were removed."""
    root = features_root(digest=digest, smoke=smoke)
    if not root.is_dir():
        return 0
    n_files = sum(1 for path in root.rglob("*") if path.is_file())
    shutil.rmtree(root, ignore_errors=True)
    return n_files


def shard_is_complete(dataset: str, master: Any, *, digest: str | None = None) -> bool:
    """True when the shard holds a row for every record of ``dataset`` in master."""
    expected = set(
        master[master["dataset_source"] == dataset]["record_uid"].astype(str)
    )
    if not expected:
        return False
    try:
        got = set(load_shard(dataset, digest=digest)["record_uid"].astype(str))
    except FileNotFoundError:
        return False
    return expected.issubset(got)


def summarize_timings(table: Any) -> Any:
    """Per-family totals from a shard, for the complexity table (T25/T26)."""
    import pandas as pd

    rows = []
    for family in FAMILY_ORDER:
        column = "sec_" + family
        if column not in table:
            continue
        values = np.asarray(table[column], dtype=np.float64)
        rows.append(
            {
                "family": family,
                "n_records": int(values.size),
                "total_seconds": float(values.sum()),
                "mean_seconds": float(values.mean()) if values.size else float("nan"),
                "median_seconds": float(np.median(values)) if values.size else float("nan"),
                "max_seconds": float(values.max()) if values.size else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["share_of_total"] = frame["total_seconds"] / frame["total_seconds"].sum()
    return frame
