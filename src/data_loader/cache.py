"""Loader hardening: deterministic ``--limit`` sampling and metadata caching.

Two small things Part II needs before Part III can rely on it.

``--limit N`` (T22.1)
    A smoke run has to exercise every code path, which means it has to see
    every class. Taking the first N rows of a table sorted by record id gives a
    sample that is all PhysioNet training-a and no CirCor ``Unknown`` -- it
    would run clean and prove nothing. :func:`apply_limit` interleaves classes
    instead, so a 20-record sample of CirCor contains Absent, Present and
    Unknown, and it is deterministic: the same N always yields the same rows.

Metadata caching (T22.2)
    The three loaders take about 25 seconds combined, almost all of it parsing
    7,536 WAV and header files. The parsed tables are cached under
    ``cache/metadata/`` keyed by a hash of the config values the loader reads
    plus the size and mtime of every metadata file it parses, plus the file
    count and total byte size of each audio tree.

    **What that key does not catch:** a WAV edited in place to the same byte
    length, with its mtime restored. That is deliberate -- catching it means
    hashing 1.3 GB on every load. The Phase 16 audio scan does hash content, so
    the integrity layer catches it; this layer trades that for speed and says
    so rather than implying a guarantee it does not make.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "CACHE_VERSION",
    "apply_limit",
    "tree_signature",
    "metadata_cache_key",
    "cached_table",
    "clear_metadata_cache",
]

log = get_logger(__name__)

# Bump when the shape of a cached table changes. A cache written by an older
# loader looks valid by its inputs alone and then loads a table missing columns.
CACHE_VERSION = 1


# --------------------------------------------------------------------------
# T22.1 -- deterministic limiting
# --------------------------------------------------------------------------


def apply_limit(
    table: Any,
    limit: int | None,
    *,
    by: Sequence[str] = ("dataset_source",),
    stratify: Sequence[str] = (),
    order: str = "record_uid",
) -> Any:
    """Keep at most ``limit`` rows per group in ``by``, spread across classes.

    ``stratify`` names the columns whose distinct values must all survive where
    the budget allows. Rows are taken round-robin from each stratum in
    ``order`` order, so the result is deterministic and reproducible under rule
    5 without any RNG at all.
    """
    if limit is None:
        return table
    if limit <= 0:
        raise ValueError("--limit must be a positive integer, got " + str(limit))

    import pandas as pd

    present_by = [c for c in by if c in table.columns]
    if not present_by:
        return _limit_block(table, limit, stratify, order).reset_index(drop=True)

    blocks = [
        _limit_block(block, limit, stratify, order)
        for _, block in table.groupby(list(present_by), sort=True)
    ]
    limited = pd.concat(blocks, ignore_index=True) if blocks else table.head(0).copy()
    log.info(
        "--limit %d: %d of %d row(s) kept across %d group(s)",
        limit,
        len(limited),
        len(table),
        len(blocks),
    )
    return limited


def _limit_block(
    block: Any, limit: int, stratify: Sequence[str], order: str
) -> Any:
    """Round-robin ``limit`` rows out of one group."""
    import pandas as pd

    if len(block) <= limit:
        return block.copy()

    sort_column = order if order in block.columns else block.columns[0]
    block = block.sort_values(sort_column, kind="stable")

    strat_columns = [c for c in stratify if c in block.columns]
    if not strat_columns:
        return block.head(limit).copy()

    key = block[strat_columns].astype(str).agg("|".join, axis=1)
    strata = [block[key == value] for value in sorted(set(key))]

    picked: list[Any] = []
    index = 0
    while sum(len(p) for p in picked) < limit and any(
        index < len(s) for s in strata
    ):
        for stratum in strata:
            if index < len(stratum):
                picked.append(stratum.iloc[[index]])
                if sum(len(p) for p in picked) >= limit:
                    break
        index += 1
    taken = pd.concat(picked, ignore_index=False)
    return taken.sort_values(sort_column, kind="stable").copy()


# --------------------------------------------------------------------------
# T22.2 -- the cache key
# --------------------------------------------------------------------------


def _stat_signature(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError:
        return {"path": path.name, "exists": False}
    return {
        "path": path.name,
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
    }


def tree_signature(root: Path, suffixes: Iterable[str] = (".wav",)) -> dict[str, Any]:
    """File count and total size under ``root``, per suffix.

    Cheap (one ``scandir`` walk, no reads) and catches the failures that matter
    here: a dataset folder moved, truncated, or with files added or removed.
    """
    wanted = {s.lower() for s in suffixes}
    counts: dict[str, int] = dict.fromkeys(wanted, 0)
    sizes: dict[str, int] = dict.fromkeys(wanted, 0)
    if not root.is_dir():
        return {"root": root.as_posix(), "exists": False}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            suffix = Path(name).suffix.lower()
            if suffix not in wanted:
                continue
            counts[suffix] += 1
            try:
                sizes[suffix] += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:  # pragma: no cover -- file vanished mid-walk
                continue
    return {"root": root.as_posix(), "counts": counts, "sizes": sizes}


def metadata_cache_key(
    name: str,
    *,
    metadata_files: Sequence[Path] = (),
    trees: Sequence[Path] = (),
    config: Any = None,
    extra: Any = None,
) -> str:
    """A stable hash over everything a cached table depends on."""
    payload = {
        "cache_version": CACHE_VERSION,
        "name": name,
        "metadata_files": [_stat_signature(Path(p)) for p in metadata_files],
        "trees": [tree_signature(Path(p)) for p in trees],
        "config": config,
        "extra": extra,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_dir() -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    return ensure_dir(load_config("paths").require("cache.metadata"))


def cached_table(
    name: str,
    builder: Callable[[], Any],
    *,
    metadata_files: Sequence[Path] = (),
    trees: Sequence[Path] = (),
    config: Any = None,
    extra: Any = None,
    enabled: bool = True,
    force: bool = False,
) -> Any:
    """Build ``name`` once and reuse it while its inputs are unchanged.

    The key is stored in the Parquet file's own name, so a stale entry is never
    read -- it is simply never looked at again, and a changed input produces a
    new file rather than overwriting a good one.
    """
    if not enabled:
        return builder()

    import pandas as pd

    key = metadata_cache_key(
        name, metadata_files=metadata_files, trees=trees, config=config, extra=extra
    )
    target = _cache_dir() / (name + "_" + key + ".parquet")

    if target.is_file() and not force:
        try:
            table = pd.read_parquet(target)
        # Blind by design: a cache is an optimisation, and no failure to read
        # one may ever break the load it was meant to speed up. pyarrow raises
        # its own exception types for truncated and partially-written files.
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            log.warning(
                "metadata cache %s unreadable (%s); rebuilding", target.name, error
            )
        else:
            log.info("metadata cache: reusing %s (%d rows)", target.name, len(table))
            return table

    table = builder()
    from src.utils.io import save_parquet

    try:
        save_parquet(table, target)
    # Same reasoning: an unserialisable column must degrade to "uncached", not
    # to a failed audit.
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        log.warning(
            "could not cache %s (%s); continuing uncached", name, error
        )
    else:
        log.info("metadata cache: wrote %s (%d rows)", target.name, len(table))
    return table


def clear_metadata_cache(name: str | None = None) -> int:
    """Delete cached metadata tables. Returns how many files were removed."""
    pattern = (name + "_*.parquet") if name else "*.parquet"
    removed = 0
    for path in _cache_dir().glob(pattern):
        path.unlink()
        removed += 1
    if removed:
        log.info("cleared %d cached metadata table(s)", removed)
    return removed
