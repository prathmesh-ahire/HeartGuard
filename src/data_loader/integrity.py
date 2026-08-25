"""Audio integrity scan across D1-D4 (Phase 16), and the shared decode pass.

Every WAV in the corpus is opened once, and everything three phases need is
computed in that single pass: header facts and integrity flags (Phase 16), the
raw-byte and audio-content hashes (T17.1, T17.2), and a fixed-length envelope
(T17.4). Reading 7,536 files takes about a minute; reading them three times
because each phase wanted its own pass would be a minute wasted three ways, and
worse, three chances for the phases to disagree about what they read.

The result is cached to ``cache/metadata/audio_scan.parquet`` (envelopes to a
companion ``.npy``) and reused unless ``force=True``.

**Header facts are re-derived here rather than trusted.** The loaders already
recorded ``original_fs``, ``n_samples`` and ``duration_sec`` from each file's
header. This phase opens the audio and checks the header told the truth: a WAV
whose header claims more frames than the file contains is precisely the
"truncated" case T16.2 exists to catch, and it cannot be detected by reading the
header alone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "AUDIO_SCAN_COLUMNS",
    "FLAG_COLUMNS",
    "SUBTYPE_BITS",
    "IntegrityThresholds",
    "load_thresholds",
    "scan_audio_file",
    "apply_flags",
    "scan_corpus",
    "check_label_coverage",
    "write_missing_corrupt_report",
    "run_integrity_scan",
]

log = get_logger(__name__)

# soundfile reports a subtype string; the audit wants a bit depth.
SUBTYPE_BITS: dict[str, int] = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}

AUDIO_SCAN_COLUMNS: tuple[str, ...] = (
    "record_uid",
    "dataset_source",
    "file_path",
    "readable",
    "error",
    "fs",
    "n_channels",
    "n_frames",
    "n_frames_header",
    "subtype",
    "bit_depth",
    "duration_sec",
    "file_bytes",
    "peak",
    "rms",
    "variance",
    "dc_offset",
    "n_clipped",
    "clipped_fraction",
    "clipping_threshold",
    "raw_sha256",
    "content_sha256",
)


class IntegrityThresholds:
    """The ``integrity`` block of ``configs/signal.yaml``, resolved once."""

    __slots__ = (
        "silence_peak",
        "constant_variance",
        "clipping_threshold",
        "clipping_ratio_flag",
        "min_samples",
        "content_hash_fs",
        "content_hash_decimals",
        "envelope_points",
        "near_duplicate_correlation",
        "short_below",
        "long_above",
    )

    def __init__(self, config: Any) -> None:
        self.silence_peak = float(config["integrity.silence_peak"])
        self.constant_variance = float(config["integrity.constant_variance"])
        self.clipping_threshold = float(config["integrity.clipping_threshold"])
        self.clipping_ratio_flag = float(config["integrity.clipping_ratio_flag"])
        self.min_samples = int(config["integrity.min_samples"])
        self.content_hash_fs = int(config["integrity.content_hash_fs"])
        self.content_hash_decimals = int(config["integrity.content_hash_decimals"])
        self.envelope_points = int(config["integrity.envelope_points"])
        self.near_duplicate_correlation = float(
            config["integrity.near_duplicate_correlation"]
        )
        self.short_below = float(config["integrity.duration_bands.short_below"])
        self.long_above = float(config["integrity.duration_bands.long_above"])


def load_thresholds() -> IntegrityThresholds:
    from src.utils.config import load_config

    return IntegrityThresholds(load_config("signal"))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _audit_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return ensure_dir(out_dir)
    from src.utils.config import load_config

    return ensure_dir(load_config("paths").require("outputs.dataset_audit"))


def _cache_dir() -> Path:
    from src.utils.config import load_config

    return ensure_dir(load_config("paths").require("cache.metadata"))


# ---------------------------------------------------------------------------
# the per-file scan (T16.1 - T16.4, T17.1, T17.2, T17.4)
# ---------------------------------------------------------------------------


def _envelope(signal: Any, points: int) -> Any:
    """A fixed-length amplitude envelope, for near-duplicate comparison (T17.4).

    Bin-averaged ``|x|`` rather than a Hilbert transform: the goal is a cheap
    shape descriptor that survives resampling and level changes, computed 7,536
    times. Fixed length is what makes a 0.76 s PASCAL B record comparable to a
    122 s PhysioNet one at all.

    Z-scored so the later comparison is a plain dot product and so the amplitude
    a recording happened to be captured at cannot make two different signals look
    alike. A constant signal has no shape to compare, and returns zeros.
    """
    import numpy as np

    magnitude = np.abs(np.asarray(signal, dtype=np.float64))
    if magnitude.size == 0:
        return np.zeros(points, dtype=np.float32)

    if magnitude.size < points:
        binned = np.interp(
            np.linspace(0, magnitude.size - 1, points),
            np.arange(magnitude.size),
            magnitude,
        )
    else:
        edges = np.linspace(0, magnitude.size, points + 1).astype(int)
        cumulative = np.concatenate([[0.0], np.cumsum(magnitude)])
        widths = np.maximum(np.diff(edges), 1)
        binned = (cumulative[edges[1:]] - cumulative[edges[:-1]]) / widths

    centered = binned - binned.mean()
    norm = float(np.linalg.norm(centered))
    if norm < 1e-12:
        return np.zeros(points, dtype=np.float32)
    return (centered / norm).astype(np.float32)


def _content_hash(signal: Any, fs: int, thresholds: IntegrityThresholds) -> str:
    """Hash of the decoded audio in a canonical form (T17.2).

    Mono, resampled to one rate, rounded to a fixed number of decimals. The
    rounding is the point: two encodings of the same recording, or the same file
    read by two library versions, agree on the signal but not on its last float
    bits, so an unrounded hash finds nothing a byte hash would not have found
    already.
    """
    import numpy as np
    import soxr

    samples = np.asarray(signal, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if fs != thresholds.content_hash_fs and samples.size:
        samples = soxr.resample(samples, fs, thresholds.content_hash_fs, quality="HQ")
    quantized = np.round(
        np.asarray(samples, dtype=np.float64), thresholds.content_hash_decimals
    )
    # +0.0 and -0.0 hash differently as bytes but are the same sample.
    quantized = quantized + 0.0
    return hashlib.sha256(quantized.astype(np.float32).tobytes()).hexdigest()


def scan_audio_file(
    path: str | Path,
    thresholds: IntegrityThresholds,
    *,
    record_uid: str = "",
    dataset_source: str = "",
    with_hashes: bool = True,
    with_envelope: bool = True,
) -> tuple[dict[str, Any], Any]:
    """Open one WAV and compute every integrity fact about it.

    Returns the scan row and its envelope. An unreadable file returns a row with
    ``readable=False`` and the exception text rather than raising: one corrupt
    file must not abort a scan of 7,536, and "which files failed" is the output
    T16.6 exists to produce.
    """
    import numpy as np
    import soundfile as sf

    wav_path = Path(path)
    row: dict[str, Any] = {
        "record_uid": record_uid or wav_path.stem,
        "dataset_source": dataset_source,
        "file_path": wav_path.as_posix(),
        "readable": False,
        "error": "",
        "fs": 0,
        "n_channels": 0,
        "n_frames": 0,
        "n_frames_header": 0,
        "subtype": "",
        "bit_depth": 0,
        "duration_sec": 0.0,
        "file_bytes": 0,
        "peak": float("nan"),
        "rms": float("nan"),
        "variance": float("nan"),
        "dc_offset": float("nan"),
        "n_clipped": 0,
        "clipped_fraction": float("nan"),
        "clipping_threshold": thresholds.clipping_threshold,
        "raw_sha256": "",
        "content_sha256": "",
    }
    empty_envelope = np.zeros(thresholds.envelope_points, dtype=np.float32)

    try:
        row["file_bytes"] = wav_path.stat().st_size
        info = sf.info(str(wav_path))
        row["n_frames_header"] = int(info.frames)
        row["fs"] = int(info.samplerate)
        row["n_channels"] = int(info.channels)
        row["subtype"] = str(info.subtype)
        row["bit_depth"] = SUBTYPE_BITS.get(str(info.subtype), 0)

        samples, fs = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a finding
        row["error"] = type(exc).__name__ + ": " + str(exc)
        return row, empty_envelope

    row["readable"] = True
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples

    row["n_frames"] = int(mono.shape[0])
    row["duration_sec"] = round(mono.shape[0] / fs, 6) if fs else 0.0

    if mono.size:
        absolute = np.abs(mono)
        row["peak"] = float(absolute.max())
        row["rms"] = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        row["variance"] = float(np.var(mono, dtype=np.float64))
        row["dc_offset"] = float(np.mean(mono, dtype=np.float64))

        # The one measurement that genuinely needs the samples and a threshold
        # at the same time. The threshold used is stored alongside so a later
        # change to it is detectable rather than silently baked into the cache.
        n_clipped = int(np.count_nonzero(absolute >= thresholds.clipping_threshold))
        row["n_clipped"] = n_clipped
        row["clipped_fraction"] = n_clipped / mono.size
    row["clipping_threshold"] = thresholds.clipping_threshold

    if with_hashes:
        digest = hashlib.sha256()
        with wav_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        row["raw_sha256"] = digest.hexdigest()
        row["content_sha256"] = _content_hash(mono, fs, thresholds)

    envelope = (
        _envelope(mono, thresholds.envelope_points) if with_envelope else empty_envelope
    )
    return row, envelope


# ---------------------------------------------------------------------------
# derived flags (T16.2 - T16.4)
# ---------------------------------------------------------------------------

FLAG_COLUMNS: tuple[str, ...] = (
    "is_zero_length",
    "is_truncated",
    "is_silent",
    "is_constant",
    "is_clipped",
)


def apply_flags(scan: Any, thresholds: IntegrityThresholds | None = None) -> Any:
    """Derive the integrity flags from a scan table (T16.2 - T16.4).

    Deliberately **not** computed inside :func:`scan_audio_file`. Every flag here
    is a threshold applied to a measurement that is already in the table, so
    keeping them separate means changing a threshold is instant rather than a
    three-minute re-decode of 8,368 files -- and, more importantly, means a
    cached scan cannot silently carry flags from whatever thresholds happened to
    be configured on the day it was written.

    ``clipped_fraction`` is the exception: counting full-scale samples needs the
    samples and the threshold together. The threshold in force at scan time is
    stored in the table, and a mismatch against the current config raises here
    rather than producing a quietly stale flag.
    """
    thresholds = thresholds or load_thresholds()
    flagged = scan.copy()

    recorded = set(flagged["clipping_threshold"].dropna().unique())
    stale = {t for t in recorded if abs(float(t) - thresholds.clipping_threshold) > 1e-12}
    if stale:
        raise ValueError(
            "the cached audio scan counted full-scale samples at threshold(s) "
            + repr(sorted(stale)) + " but configs/signal.yaml now says "
            + str(thresholds.clipping_threshold)
            + " -- rerun the scan with force=True rather than mixing the two"
        )

    # T16.2 -- zero-length, and truncated: the header promised more audio than
    # the file holds. Only a decode can reveal the second, which is why the
    # loaders' header-only figures are not enough on their own.
    flagged["is_zero_length"] = (flagged["n_frames"] <= 0) | (flagged["file_bytes"] <= 0)
    flagged["is_truncated"] = (
        flagged["n_frames"] < flagged["n_frames_header"]
    ) | ((flagged["n_frames"] > 0) & (flagged["n_frames"] < thresholds.min_samples))

    # T16.3 -- all-silent and constant-value.
    flagged["is_silent"] = flagged["peak"] <= thresholds.silence_peak
    flagged["is_constant"] = flagged["variance"] <= thresholds.constant_variance

    # T16.4 -- clipped.
    flagged["is_clipped"] = flagged["clipped_fraction"] > thresholds.clipping_ratio_flag

    # An unreadable file has no measurements, so every flag on it would be a
    # comparison against NaN. It is reported as unreadable and nothing else.
    unreadable = ~flagged["readable"]
    for column in FLAG_COLUMNS:
        flagged.loc[unreadable, column] = False
        flagged[column] = flagged[column].fillna(False).astype(bool)
    return flagged


# ---------------------------------------------------------------------------
# the corpus scan (T16.1)
# ---------------------------------------------------------------------------


def scan_corpus(
    catalog: Any,
    *,
    thresholds: IntegrityThresholds | None = None,
    force: bool = False,
    with_hashes: bool = True,
    extra_files: dict[str, Path] | None = None,
) -> tuple[Any, Any]:
    """Scan every WAV in ``catalog``, with caching (T16.1).

    ``extra_files`` adds paths that are not records -- specifically the 832
    ``Heartbeat_Sound/`` files, which T17.3 must hash to prove they duplicate
    set_a + set_b, but which are never records in their own right.

    Returns the scan table and the envelope matrix, row-aligned.
    """
    import numpy as np
    import pandas as pd

    thresholds = thresholds or load_thresholds()
    cache = _cache_dir()
    scan_path = cache / "audio_scan.parquet"
    envelope_path = cache / "audio_envelopes.npy"

    wanted = list(catalog["record_uid"]) + list(extra_files or {})
    if not force and scan_path.is_file() and envelope_path.is_file():
        cached = pd.read_parquet(scan_path)
        # The column set is part of the cache key. A scan written before a
        # schema change looks valid by record ids alone and then fails, or worse
        # succeeds with a column missing.
        if (
            list(cached.columns)[: len(AUDIO_SCAN_COLUMNS)] == list(AUDIO_SCAN_COLUMNS)
            and list(cached["record_uid"]) == wanted
        ):
            envelopes = np.load(envelope_path)
            if envelopes.shape[0] == len(cached):
                log.info("audio scan: reusing cache (%d files)", len(cached))
                return apply_flags(cached, thresholds), envelopes
        log.info("audio scan: cache does not match the current corpus; rescanning")

    targets: list[tuple[str, str, Path]] = [
        (row.record_uid, row.dataset_source, _project_root() / row.file_path)
        for row in catalog.itertuples(index=False)
    ]
    targets += [
        (uid, "heartbeat_sound", Path(path)) for uid, path in (extra_files or {}).items()
    ]

    rows: list[dict[str, Any]] = []
    envelopes = np.zeros(
        (len(targets), thresholds.envelope_points), dtype=np.float32
    )
    for index, (uid, dataset, path) in enumerate(targets):
        row, envelope = scan_audio_file(
            path,
            thresholds,
            record_uid=uid,
            dataset_source=dataset,
            with_hashes=with_hashes,
        )
        row["file_path"] = (
            path.resolve().relative_to(_project_root()).as_posix()
            if path.is_absolute() and _project_root() in path.resolve().parents
            else path.as_posix()
        )
        rows.append(row)
        envelopes[index] = envelope
        if (index + 1) % 1000 == 0:
            log.info("audio scan: %d / %d files", index + 1, len(targets))

    table = pd.DataFrame(rows, columns=list(AUDIO_SCAN_COLUMNS))
    table.to_parquet(scan_path, engine="pyarrow", index=False)
    np.save(envelope_path, envelopes)

    table = apply_flags(table, thresholds)
    log.info(
        "audio scan: %d files (%d unreadable, %d zero-length, %d truncated, "
        "%d silent, %d constant, %d clipped)",
        len(table),
        int((~table["readable"]).sum()),
        int(table["is_zero_length"].sum()),
        int(table["is_truncated"].sum()),
        int(table["is_silent"].sum()),
        int(table["is_constant"].sum()),
        int(table["is_clipped"].sum()),
    )
    return table, envelopes


# ---------------------------------------------------------------------------
# label coverage (T16.5)
# ---------------------------------------------------------------------------


def check_label_coverage(catalog: Any, scan: Any | None = None) -> Any:
    """Records on disk with no label, and labels with no file, per dataset (T16.5).

    Both directions, per dataset, because the two failures look nothing alike:
    a file with no label is an extraction that silently trains on nothing, and a
    label with no file is a row that vanishes from a fold without changing any
    reported count.

    The PASCAL unlabelled records are *not* reported here. They are labelled
    ``unlabel`` on purpose, excluded from the supervised tracks by T12.5, and
    counted in their own column -- an expected absence, not a missing label.
    """
    import pandas as pd

    from src.data_loader.catalog import TASK_CLASS_COLUMNS, dataset_tasks

    rows: list[dict[str, Any]] = []
    on_disk = set(scan["record_uid"]) if scan is not None else None

    for dataset, group in catalog.groupby("dataset_source"):
        for task in dataset_tasks(str(dataset)):
            column = TASK_CLASS_COLUMNS[task]
            labelled = group[group[column].astype("string").fillna("") != ""]
            unlabelled = group[
                (group[column].astype("string").fillna("") == "")
                & ~group["is_unlabeled"]
            ]
            for record in unlabelled.itertuples(index=False):
                rows.append(
                    {
                        "record_uid": record.record_uid,
                        "dataset_source": dataset,
                        "task": task,
                        "problem": "file_without_label",
                        "detail": "on disk but carries no " + task + " class",
                        "file_path": record.file_path,
                    }
                )
            if on_disk is not None:
                for record in labelled.itertuples(index=False):
                    if record.record_uid not in on_disk:
                        rows.append(
                            {
                                "record_uid": record.record_uid,
                                "dataset_source": dataset,
                                "task": task,
                                "problem": "label_without_file",
                                "detail": "labelled but absent from the audio scan",
                                "file_path": record.file_path,
                            }
                        )

    return pd.DataFrame(
        rows,
        columns=[
            "record_uid",
            "dataset_source",
            "task",
            "problem",
            "detail",
            "file_path",
        ],
    )


# ---------------------------------------------------------------------------
# DA-05 (T16.6)
# ---------------------------------------------------------------------------


def write_missing_corrupt_report(
    scan: Any,
    coverage: Any,
    out_dir: str | Path | None = None,
) -> Path:
    """Write **DA-05** ``missing_corrupt_files.csv`` (T16.6).

    Written even when empty -- which, on this corpus, is what it should be for
    every category except clipping. A missing file is indistinguishable from a
    scan that never ran; an empty one with a header is a positive statement.
    """
    import pandas as pd

    problems: list[dict[str, Any]] = []
    flags = (
        ("unreadable", ~scan["readable"]),
        ("zero_length", scan["is_zero_length"]),
        ("truncated", scan["is_truncated"]),
        ("all_silent", scan["is_silent"]),
        ("constant_value", scan["is_constant"]),
        ("clipped", scan["is_clipped"]),
    )
    for problem, mask in flags:
        for record in scan[mask].itertuples(index=False):
            problems.append(
                {
                    "record_uid": record.record_uid,
                    "dataset_source": record.dataset_source,
                    "task": "",
                    "problem": problem,
                    "detail": (
                        record.error
                        if problem == "unreadable"
                        else "peak=" + format(record.peak, ".6g")
                        + " var=" + format(record.variance, ".6g")
                        + " clipped_fraction="
                        + format(record.clipped_fraction, ".6g")
                    ),
                    "file_path": record.file_path,
                }
            )

    report = pd.concat(
        [
            pd.DataFrame(
                problems,
                columns=[
                    "record_uid",
                    "dataset_source",
                    "task",
                    "problem",
                    "detail",
                    "file_path",
                ],
            ),
            coverage,
        ],
        ignore_index=True,
    )
    target = _audit_dir(out_dir) / "missing_corrupt_files.csv"
    save_csv(report, target)
    log.info(
        "wrote %s (%d problem row(s): %s)",
        target.name,
        len(report),
        ", ".join(
            key + "=" + str(value)
            for key, value in report["problem"].value_counts().to_dict().items()
        )
        or "none",
    )
    return target


def run_integrity_scan(
    catalog: Any | None = None,
    *,
    force: bool = False,
    write_outputs: bool = False,
    out_dir: str | Path | None = None,
    extra_files: dict[str, Path] | None = None,
) -> tuple[Any, Any, Any]:
    """Phase 16 end to end: scan, coverage check, and DA-05.

    Returns ``(scan, envelopes, coverage)``.
    """
    from src.data_loader.catalog import build_catalog

    catalog = build_catalog() if catalog is None else catalog
    scan, envelopes = scan_corpus(catalog, force=force, extra_files=extra_files)
    record_scan = scan[scan["dataset_source"] != "heartbeat_sound"]
    coverage = check_label_coverage(catalog, record_scan)

    if write_outputs:
        write_missing_corrupt_report(record_scan, coverage, out_dir)
    return scan, envelopes, coverage
