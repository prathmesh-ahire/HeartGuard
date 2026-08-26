"""The preprocessing pipeline and its signal cache (Phase 27).

One function, :func:`preprocess`, is the only sanctioned way to turn a path on
disk into the signal a feature extractor sees. Everything downstream -- the 138
features, the API's live prediction, every figure in Part III -- goes through it,
so that a recording is preprocessed identically whether it arrives from the
corpus or from a browser upload.

    load -> mono -> resample to 2 kHz -> quality metrics -> bandpass -> normalize

**Quality metrics are taken in the middle, not at the end.** They measure the
resampled but unfiltered, unnormalized signal, for the reasons set out in
``quality.py``: the bandpass deletes the out-of-band term the SNR proxy needs,
and normalization deletes the absolute levels that silence and clipping are
defined against. Running the chain and measuring what falls out of the far end
would produce a table of confident, meaningless numbers.

**The cache is keyed by a hash of the settings that shape the signal** --
``resample``, ``filter``, ``normalization`` and ``framing`` -- and nothing else.
Hashing all of ``signal.yaml`` would make an edit to an audit threshold, which
this pipeline never reads, invalidate 7,536 cached signals. Each configuration
gets its own directory, so the four arms of the PP-09 ablation coexist rather
than overwrite each other, and a stale cache is impossible to hit by accident:
a different setting is a different path.

Cached entries are ``.npz`` holding the signal and its metadata together. A bare
``.npy`` would be smaller, but then the applied-step list and the quality metrics
would have to be recomputed on every cache hit -- which is most of the work the
cache exists to avoid.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import atomic_path, ensure_dir
from src.utils.logging_setup import get_logger

__all__ = [
    "CACHE_VERSION",
    "PIPELINE_STEPS",
    "PreprocessResult",
    "config_hash",
    "cache_root",
    "cache_path",
    "preprocess",
    "preprocess_corpus",
    "cache_stats",
    "clear_preprocessed_cache",
]

log = get_logger(__name__)

# Bump when the *shape* of a cached entry changes (new field, different meaning).
# It is part of the cache key, so an older entry can never be read as a newer one.
CACHE_VERSION = 1

PIPELINE_STEPS: tuple[str, ...] = ("load", "mono", "resample", "quality", "filter", "normalize")

# The sections of configs/signal.yaml that change the samples this module emits.
HASHED_SECTIONS: tuple[str, ...] = ("resample", "filter", "normalization", "framing")

# Individual keys outside those sections that change what a cache entry CONTAINS.
# A cached entry carries its quality metrics, and three quality settings feed
# those measurements: the clipping and silence levels, and the SNR proxy's band
# edges. They are listed one by one rather than by hashing all of `quality`,
# because the rest of that section holds *flag* thresholds -- which are applied
# to a cached table by `quality.apply_flags`, never stored -- and hashing those
# would make recalibrating a flag invalidate 7,536 preprocessed signals.
HASHED_KEYS: tuple[str, ...] = (
    "quality.clipping_threshold",
    "quality.silence_threshold_db",
    "quality.snr_proxy.in_band",
    "quality.snr_proxy.out_of_band",
)


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """A preprocessed signal and everything that was done to produce it (T27.2)."""

    signal: np.ndarray
    fs: int
    fs_native: int
    record_uid: str
    file_path: str
    steps: tuple[str, ...]
    quality: dict[str, Any] = field(default_factory=dict)
    filter_info: dict[str, Any] = field(default_factory=dict)
    normalization_info: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    from_cache: bool = False

    @property
    def n_samples(self) -> int:
        return int(self.signal.size)

    @property
    def duration_sec(self) -> float:
        return float(self.signal.size / self.fs) if self.fs else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Everything except the samples -- the row for a manifest or report."""
        row: dict[str, Any] = {
            "record_uid": self.record_uid,
            "file_path": self.file_path,
            "fs": self.fs,
            "fs_native": self.fs_native,
            "n_samples": self.n_samples,
            "duration_sec": self.duration_sec,
            "steps": ";".join(self.steps),
            "config_hash": self.config_hash,
            "from_cache": self.from_cache,
        }
        row.update(self.filter_info)
        row.update(self.normalization_info)
        row.update({k: v for k, v in self.quality.items() if k not in row})
        return row


# ---------------------------------------------------------------------------
# T27.3 -- the config hash and the cache layout
# ---------------------------------------------------------------------------


def _signal_config(cfg: Any | None = None) -> Any:
    if cfg is not None:
        return cfg
    from src.utils.config import load_config

    return load_config("signal")


def config_hash(cfg: Any | None = None) -> str:
    """A 12-character digest of the settings that shape the output signal.

    Sorted-key JSON, so a reordering of ``signal.yaml`` does not invalidate the
    cache while a changed *value* always does. ``CACHE_VERSION`` is folded in, so
    a change to the entry format cannot be mistaken for a config change.
    """
    signal_cfg = _signal_config(cfg)
    data = signal_cfg.as_dict()
    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        **{section: data.get(section, {}) for section in HASHED_SECTIONS},
    }
    for key in HASHED_KEYS:
        payload[key] = signal_cfg.get(key)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def cache_root(cfg: Any | None = None, *, digest: str | None = None) -> Path:
    """``cache/preprocessed/<config hash>/``."""
    from src.utils.config import load_config

    base = Path(load_config("paths").require("cache.preprocessed"))
    return base / (digest or config_hash(cfg))


def cache_path(
    record_uid: str, cfg: Any | None = None, *, digest: str | None = None
) -> Path:
    """Where one record's preprocessed signal lives.

    The uid is used verbatim -- every uid in this corpus is
    ``{dataset}_{subset}_{record_id}``, already filesystem-safe, and mangling it
    would break the round trip from a cache file back to a record.
    """
    return cache_root(cfg, digest=digest) / (record_uid + ".npz")


def _write_cache(path: Path, result: PreprocessResult) -> None:
    meta = {
        "fs": result.fs,
        "fs_native": result.fs_native,
        "record_uid": result.record_uid,
        "file_path": result.file_path,
        "steps": list(result.steps),
        "quality": result.quality,
        "filter_info": result.filter_info,
        "normalization_info": result.normalization_info,
        "config_hash": result.config_hash,
    }
    ensure_dir(path.parent)
    with atomic_path(path, suffix=".npz") as tmp, tmp.open("wb") as handle:
        np.savez(
            handle,
            signal=result.signal,
            meta=np.asarray(json.dumps(meta, default=str)),
        )


def _read_cache(path: Path) -> PreprocessResult | None:
    """Return a cached result, or ``None`` if the entry is unusable.

    A truncated or half-written entry returns ``None`` rather than raising: an
    interrupted batch run leaves exactly that, and the correct response is to
    recompute the record, not to abort the next run.
    """
    try:
        with np.load(path, allow_pickle=False) as bundle:
            signal = np.asarray(bundle["signal"], dtype=np.float32)
            meta = json.loads(str(bundle["meta"]))
    except Exception as exc:  # noqa: BLE001 - any unreadable entry is just a miss
        log.warning("preprocessed cache entry %s is unreadable (%s); recomputing", path.name, exc)
        return None

    return PreprocessResult(
        signal=signal,
        fs=int(meta["fs"]),
        fs_native=int(meta["fs_native"]),
        record_uid=str(meta["record_uid"]),
        file_path=str(meta["file_path"]),
        steps=tuple(meta["steps"]),
        quality=dict(meta.get("quality", {})),
        filter_info=dict(meta.get("filter_info", {})),
        normalization_info=dict(meta.get("normalization_info", {})),
        config_hash=str(meta.get("config_hash", "")),
        from_cache=True,
    )


# ---------------------------------------------------------------------------
# T27.1 / T27.2 -- the chain
# ---------------------------------------------------------------------------


def preprocess(
    path: str | Path,
    cfg: Any | None = None,
    *,
    record_uid: str | None = None,
    with_quality: bool = True,
    use_cache: bool = True,
    force: bool = False,
) -> PreprocessResult:
    """Load, resample, filter and normalize one recording (T27.1).

    Returns a :class:`PreprocessResult` carrying the signal, both sampling rates,
    the list of steps that actually ran, and the quality metrics measured in the
    pre-filter domain.

    ``use_cache=False`` bypasses the cache in both directions -- used by the
    ablation, which preprocesses the same records under four configurations, and
    by the API, whose uploads have no record uid to key on.
    """
    from src.preprocessing import filters as flt
    from src.preprocessing import io as pio
    from src.preprocessing import normalize as nrm
    from src.preprocessing import quality as qual

    signal_cfg = _signal_config(cfg)
    digest = config_hash(signal_cfg)
    wav_path = Path(path)
    uid = record_uid or wav_path.stem

    entry = cache_path(uid, signal_cfg, digest=digest) if use_cache else None
    if entry is not None and not force and entry.is_file():
        cached = _read_cache(entry)
        if cached is not None:
            return cached

    target = pio.target_fs(signal_cfg)
    method = pio.resample_method(signal_cfg)

    raw, fs_native = pio.load_wav(wav_path)
    mono = pio.to_mono(raw)
    resampled = pio.resample_to(mono, fs_native, target, method=method)

    steps: list[str] = ["load"]
    steps.append("mono:" + ("noop" if raw.ndim == 1 else "channel_mean"))
    steps.append(
        "resample:" + str(fs_native) + "->" + str(target) + ":"
        + ("noop" if fs_native == target else method)
    )

    quality: dict[str, Any] = {}
    if with_quality:
        thresholds = qual.load_thresholds(signal_cfg)
        quality = qual.measure_signal(resampled, target, thresholds)
        steps.append("quality")

    filtered = flt.filter_signal(resampled, target, signal_cfg)
    steps.append(
        "filter:"
        + (
            str(int(filtered.low_hz)) + "-" + str(int(filtered.high_hz))
            + ":order" + str(filtered.order)
            + (":zero_phase" if filtered.zero_phase else ":forward")
            + (":padlen_reduced" if filtered.padlen_reduced else "")
            if filtered.applied
            else "skipped"
        )
    )

    normalized = nrm.normalize_signal(filtered.signal, signal_cfg)
    steps.append(
        "normalize:"
        + (
            normalized.method + (":dc_removed" if normalized.dc_removed else "")
            if normalized.applied
            else "skipped" + (":zero_variance" if normalized.zero_variance else "")
        )
    )

    try:
        relative = wav_path.resolve().relative_to(_project_root()).as_posix()
    except ValueError:
        relative = wav_path.as_posix()

    result = PreprocessResult(
        signal=np.asarray(normalized.signal, dtype=np.float32),
        fs=target,
        fs_native=int(fs_native),
        record_uid=uid,
        file_path=relative,
        steps=tuple(steps),
        quality=quality,
        filter_info=filtered.as_dict(),
        normalization_info=normalized.as_dict(),
        config_hash=digest,
        from_cache=False,
    )

    if entry is not None:
        _write_cache(entry, result)
    return result


def _project_root() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("project_root"))


# ---------------------------------------------------------------------------
# T27.4 / T27.5 -- the batch run
# ---------------------------------------------------------------------------


def _preprocess_one(uid: str, relative_path: str, root: str, force: bool) -> dict[str, Any]:
    """Worker body. Takes only picklable arguments and loads its own config."""
    try:
        result = preprocess(
            Path(root) / relative_path, record_uid=uid, use_cache=True, force=force
        )
    except Exception as exc:  # one bad file must not kill a 7,536-record batch
        log.exception("preprocessing failed for %s", uid)
        return {"record_uid": uid, "file_path": relative_path, "ok": False, "error": str(exc)}

    row = result.as_dict()
    row["ok"] = True
    row["error"] = ""
    return row


def preprocess_corpus(
    master: Any | None = None,
    *,
    n_jobs: int = -1,
    force: bool = False,
    progress: bool = True,
    limit: int | None = None,
) -> Any:
    """Preprocess every record in parallel, filling the cache (T27.4).

    Returns a table with one row per record: the applied steps, the filter and
    normalization flags, the quality metrics, and whether the entry came from
    cache. A record that fails is reported with ``ok=False`` and its error rather
    than aborting the run -- 7,536 records is too many to restart because of one.
    """
    import pandas as pd
    from joblib import Parallel, delayed

    from src.data_loader import master as ms
    from src.utils.timing import timer

    if master is None:
        master = ms.load_master()
    if limit is not None:
        master = master.head(int(limit))

    root = str(_project_root())
    targets = [
        (str(row.record_uid), str(row.file_path)) for row in master.itertuples(index=False)
    ]

    iterator: Any = targets
    if progress:
        from tqdm import tqdm

        iterator = tqdm(targets, desc="preprocess", unit="rec")

    with timer("phase27_preprocess_corpus") as stopwatch:
        rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=32)(
            delayed(_preprocess_one)(uid, relative, root, force) for uid, relative in iterator
        )

    table = pd.DataFrame(rows)
    failures = int((~table["ok"]).sum())
    stats = cache_stats()
    log.info(
        "preprocessed %d records in %s (%d failed, %d from cache) -- cache %d files, %.1f MB",
        len(table),
        stopwatch.seconds if stopwatch.seconds is None else round(stopwatch.seconds, 1),
        failures,
        int(table.get("from_cache", pd.Series(dtype=bool)).sum()),
        stats["n_files"],
        stats["megabytes"],
    )
    return table


# ---------------------------------------------------------------------------
# cache housekeeping
# ---------------------------------------------------------------------------


def cache_stats(cfg: Any | None = None, *, digest: str | None = None) -> dict[str, Any]:
    """File count and size of the cache for one configuration (T27.5)."""
    root = cache_root(cfg, digest=digest)
    if not root.is_dir():
        return {"path": str(root), "n_files": 0, "bytes": 0, "megabytes": 0.0}

    total = 0
    count = 0
    for entry in root.rglob("*.npz"):
        total += entry.stat().st_size
        count += 1
    return {
        "path": str(root),
        "n_files": count,
        "bytes": total,
        "megabytes": round(total / (1024 * 1024), 2),
    }


def clear_preprocessed_cache(cfg: Any | None = None, *, digest: str | None = None) -> int:
    """Delete every cached entry for one configuration. Returns the count."""
    root = cache_root(cfg, digest=digest)
    removed = 0
    if root.is_dir():
        for entry in root.rglob("*.npz"):
            os.remove(entry)
            removed += 1
    log.info("cleared %d cached signals from %s", removed, root)
    return removed
