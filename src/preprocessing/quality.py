"""Per-record signal quality analysis and the PP-08 flags (Phase 26).

Every recording in this corpus was made by a different person with a different
device in a different room. Some are clean; some are 40% rubbing noise; one is
effectively silent. Phase 72 partitions test sets into clean / noisy / low-
confidence groups using what this module produces, and the thesis reports
robustness per group, so these flags decide which records end up in which
robustness bucket.

**The measurement domain is the 2 kHz signal *before* filtering and
normalization.** That is not an implementation detail, it is the whole design:

* the SNR proxy compares in-band (20-400 Hz) with out-of-band (400-1000 Hz)
  power, and the bandpass exists precisely to delete the out-of-band term. Run
  after filtering, every record scores +100 dB and the metric measures the
  filter instead of the recording;
* silence is defined in dBFS (-60 dB), an absolute level. After z-scoring, every
  record has SD 1 by construction and "silence" means nothing;
* clipping is |sample| >= 0.99 of full scale, which again only exists before
  normalization.

Phase 16 measured peak, RMS and clipping on the **raw** file at its native rate.
The columns here are their counterparts in the analysis domain and will not match
exactly -- a 44.1 kHz PASCAL A record loses most of its energy on the way to
2 kHz. Both are correct; they answer different questions, and every column here
is named for the domain it was measured in.

**What ``is_noisy`` actually detects, and what it does not.** Three proxies are
OR-ed together, and only one of them earns its place empirically. Scored against
the 404 PhysioNet recordings the 2016 Challenge organisers marked poor quality
(T26.5, all 3,541 D1 records, 2026-08-26):

====================== ============ ============ ==================
proxy                  sensitivity  specificity  balanced accuracy
====================== ============ ============ ==================
``snr_proxy_db < 5``   0.062        0.997        0.530
``flatness > 0.5``     0.000        1.000        0.500
``drift_ratio_db > 5`` 0.634        0.918        0.776
composite ``is_noisy`` 0.663        0.915        **0.789**
====================== ============ ============ ==================

The first two were the original configuration and they are close to blind on
this corpus: heart recordings hold almost no energy above 400 Hz to begin with,
so an in-band/out-of-band ratio is not measuring noise, and nothing in 7,536
records comes near a flatness of 0.5 (the corpus maximum is 0.146). The noise
that actually ruins a PCG is **low frequency** -- rubbing, movement, breathing,
loose stethoscope contact -- and it sits below 20 Hz, where neither of them
looks. ``drift_ratio_db`` does look there, and it tracks the human annotation.

Both weak proxies are kept rather than deleted: they fire on genuinely
out-of-band and white-noise-like records that the drift term misses, almost all
of them in CirCor, which is 4 kHz native and paediatric and therefore carries
real content above 400 Hz. That asymmetry is worth knowing about -- CirCor has
1,500 of the 1,594 flags those two proxies raise corpus-wide, and unlike
PhysioNet it has no human quality reference to check them against.

**These flags are diagnostics and grouping keys. They are never model features.**
``zcr_anomaly`` is computed against a per-dataset median, so it is the one column
here that depends on other records. It is label-independent, which is why it is
allowed at all -- but feeding it into a feature matrix would put a corpus-wide
statistic inside a training fold, and that is rule 2's exact failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "QUALITY_COLUMNS",
    "FLAG_COLUMNS",
    "QualityThresholds",
    "load_thresholds",
    "frame_rms",
    "measure_signal",
    "measure_file",
    "apply_flags",
    "scan_quality",
    "calibrate_against_sqi",
    "quality_flags_path",
    "write_quality_flags",
    "run_quality_scan",
]

log = get_logger(__name__)

EPS = 1e-12
# An SNR proxy with no measurable out-of-band power is reported at this ceiling
# rather than as inf, so the column stays finite and sortable.
SNR_CEILING_DB = 120.0

QUALITY_COLUMNS: tuple[str, ...] = (
    "record_uid",
    "dataset_source",
    "subset",
    "file_path",
    "original_fs",
    "fs",
    "n_samples",
    "duration_sec",
    # T26.1
    "rms",
    "peak",
    "crest_factor_db",
    "dynamic_range_db",
    # T26.2
    "clipping_ratio",
    "silence_ratio",
    # T26.3
    "snr_proxy_db",
    "drift_ratio_db",
    # T26.4
    "spectral_flatness",
    "zcr_mean",
    "zcr_std",
)

FLAG_COLUMNS: tuple[str, ...] = (
    "zcr_anomaly",
    "is_short",
    "is_clipped",
    "is_silent",
    "is_noisy",
    "is_low_quality",
    "quality_reasons",
)


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Everything in ``configs/signal.yaml`` that this module reads."""

    target_fs: int
    clipping_threshold: float
    clipping_ratio_flag: float
    silence_threshold_db: float
    silence_ratio_flag: float
    in_band: tuple[float, float]
    out_of_band: tuple[float, float]
    low_snr_db_flag: float
    spectral_flatness_flag: float
    drift_ratio_db_flag: float
    short_duration_sec: float
    frame_length: int
    hop_length: int
    calibrate_against_sqi: bool


def load_thresholds(cfg: Any | None = None) -> QualityThresholds:
    """Read the quality block, failing loudly on a passband mismatch.

    ``quality.snr_proxy.in_band`` and ``filter.low_hz/high_hz`` describe the same
    band from two directions. If they drift apart, the SNR proxy measures a band
    the pipeline does not keep, and nothing else in the project would notice.
    """
    if cfg is None:
        from src.utils.config import load_config

        cfg = load_config("signal")

    in_band = tuple(float(v) for v in cfg.require("quality.snr_proxy.in_band"))
    out_of_band = tuple(float(v) for v in cfg.require("quality.snr_proxy.out_of_band"))
    low_hz = float(cfg.require("filter.low_hz"))
    high_hz = float(cfg.require("filter.high_hz"))
    if (in_band[0], in_band[1]) != (low_hz, high_hz):
        raise ValueError(
            "configs/signal.yaml quality.snr_proxy.in_band " + str(in_band)
            + " does not match the filter passband (" + str(low_hz) + ", " + str(high_hz)
            + ") -- the SNR proxy must measure the band the pipeline keeps"
        )

    return QualityThresholds(
        target_fs=int(cfg.require("resample.target_fs")),
        clipping_threshold=float(cfg.require("quality.clipping_threshold")),
        clipping_ratio_flag=float(cfg.require("quality.clipping_ratio_flag")),
        silence_threshold_db=float(cfg.require("quality.silence_threshold_db")),
        silence_ratio_flag=float(cfg.require("quality.silence_ratio_flag")),
        in_band=(in_band[0], in_band[1]),
        out_of_band=(out_of_band[0], out_of_band[1]),
        low_snr_db_flag=float(cfg.require("quality.snr_proxy.low_snr_db_flag")),
        spectral_flatness_flag=float(cfg.require("quality.spectral_flatness_flag")),
        drift_ratio_db_flag=float(cfg.require("quality.drift_ratio_db_flag")),
        short_duration_sec=float(cfg.require("quality.short_duration_sec")),
        frame_length=int(cfg.require("framing.frame_length")),
        hop_length=int(cfg.require("framing.hop_length")),
        calibrate_against_sqi=bool(cfg.get("quality.calibrate_against_sqi", True)),
    )


# ---------------------------------------------------------------------------
# T26.1 - T26.4 -- the measurements
# ---------------------------------------------------------------------------


def frame_rms(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Per-frame RMS, in float64, with a short-signal fallback.

    A record shorter than one frame -- none exist at 2 kHz, but an API upload
    can be -- yields a single frame covering the whole signal rather than an
    empty array, which would make every downstream ratio a division by zero.
    """
    samples = np.asarray(x, dtype=np.float64).ravel()
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64)
    if samples.size < frame_length:
        return np.array([np.sqrt(np.mean(samples * samples))], dtype=np.float64)

    n_frames = 1 + (samples.size - frame_length) // hop_length
    strides = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, frame_length),
        strides=(samples.strides[0] * hop_length, samples.strides[0]),
        writeable=False,
    )
    return np.sqrt(np.mean(strides * strides, axis=1))


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    """Integrated PSD between two edges, trapezoidal."""
    band = (freqs >= low) & (freqs < high)
    if not band.any():
        return 0.0
    if band.sum() == 1:
        return float(power[band][0])
    return float(np.trapezoid(power[band], freqs[band]))


def measure_signal(
    x: np.ndarray, fs: int, thresholds: QualityThresholds
) -> dict[str, Any]:
    """Every T26.1-T26.4 measurement for one signal, in the analysis domain.

    The signal must be resampled to the target rate and **not** filtered or
    normalized; see the module docstring for why. Returns finite numbers for
    every input, including empty and constant ones -- a NaN here would reach
    PP-08, and T26.7 forbids that.
    """
    from scipy.signal import welch

    samples = np.asarray(x, dtype=np.float64).ravel()
    n = samples.size

    row: dict[str, Any] = {
        "fs": int(fs),
        "n_samples": int(n),
        "duration_sec": float(n / fs) if fs else 0.0,
    }

    if n == 0:
        row.update(
            {
                "rms": 0.0,
                "peak": 0.0,
                "crest_factor_db": 0.0,
                "dynamic_range_db": 0.0,
                "clipping_ratio": 0.0,
                "silence_ratio": 1.0,
                "snr_proxy_db": 0.0,
                "drift_ratio_db": 0.0,
                "spectral_flatness": 0.0,
                "zcr_mean": 0.0,
                "zcr_std": 0.0,
            }
        )
        return row

    absolute = np.abs(samples)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(absolute.max())

    # -- T26.1 amplitude ---------------------------------------------------
    row["rms"] = rms
    row["peak"] = peak
    row["crest_factor_db"] = 20.0 * np.log10(max(peak, EPS) / max(rms, EPS))

    # Dynamic range as the spread of frame loudness, p95 over p5. The naive
    # "loudest over quietest frame" is a measurement of the quietest frame,
    # which in a recording with one silent gap is the noise floor of that gap.
    frames = frame_rms(samples, thresholds.frame_length, thresholds.hop_length)
    if frames.size:
        loud = float(np.percentile(frames, 95))
        quiet = float(np.percentile(frames, 5))
        row["dynamic_range_db"] = 20.0 * np.log10(max(loud, EPS) / max(quiet, EPS))
    else:
        row["dynamic_range_db"] = 0.0

    # -- T26.2 clipping and silence ---------------------------------------
    row["clipping_ratio"] = float(
        np.count_nonzero(absolute >= thresholds.clipping_threshold) / n
    )
    if frames.size:
        frame_db = 20.0 * np.log10(np.maximum(frames, EPS))
        row["silence_ratio"] = float(
            np.count_nonzero(frame_db < thresholds.silence_threshold_db) / frames.size
        )
    else:
        row["silence_ratio"] = 0.0

    # -- T26.3 SNR proxy ---------------------------------------------------
    nperseg = int(min(thresholds.frame_length, n))
    freqs, power = welch(samples, fs=fs, nperseg=max(nperseg, 8), scaling="density")
    in_power = _band_power(freqs, power, *thresholds.in_band)
    out_power = _band_power(freqs, power, *thresholds.out_of_band)
    drift_power = _band_power(freqs, power, 0.0, thresholds.in_band[0])

    if in_power <= EPS:
        row["snr_proxy_db"] = -SNR_CEILING_DB
    elif out_power <= EPS:
        row["snr_proxy_db"] = SNR_CEILING_DB
    else:
        row["snr_proxy_db"] = float(
            np.clip(10.0 * np.log10(in_power / out_power), -SNR_CEILING_DB, SNR_CEILING_DB)
        )
    row["drift_ratio_db"] = float(
        np.clip(
            10.0 * np.log10(max(drift_power, EPS) / max(in_power, EPS)),
            -SNR_CEILING_DB,
            SNR_CEILING_DB,
        )
    )

    # -- T26.4 spectral flatness and zero crossings ------------------------
    # Flatness is the geometric-to-arithmetic mean ratio of the power spectrum:
    # 1.0 is white noise, near 0 is a clean tonal/impulsive signal. Computed per
    # frame and averaged, so one noisy passage does not disappear into a long
    # clean record.
    row["spectral_flatness"] = _spectral_flatness(samples, thresholds)

    crossings = np.abs(np.diff(np.signbit(samples).astype(np.int8)))
    if crossings.size:
        zcr_frames = frame_rms(  # reuse the framing, on the crossing indicator
            crossings.astype(np.float64), thresholds.frame_length, thresholds.hop_length
        )
        # frame_rms of a 0/1 indicator is sqrt(rate); square it back to a rate.
        rates = zcr_frames**2
        row["zcr_mean"] = float(rates.mean())
        row["zcr_std"] = float(rates.std())
    else:
        row["zcr_mean"] = 0.0
        row["zcr_std"] = 0.0

    for key, value in row.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("quality metric " + key + " is not finite: " + repr(value))
    return row


def _spectral_flatness(samples: np.ndarray, thresholds: QualityThresholds) -> float:
    """Mean per-frame spectral flatness, computed with librosa."""
    import librosa

    if samples.size < 2:
        return 0.0
    n_fft = int(min(thresholds.frame_length, max(samples.size, 2)))
    flatness = librosa.feature.spectral_flatness(
        y=np.asarray(samples, dtype=np.float32),
        n_fft=n_fft,
        hop_length=thresholds.hop_length,
        center=True,
    )
    value = float(np.mean(flatness)) if flatness.size else 0.0
    return value if np.isfinite(value) else 0.0


def measure_file(
    path: str | Path,
    thresholds: QualityThresholds,
    *,
    record_uid: str = "",
    dataset_source: str = "",
    subset: str = "",
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Load, resample and measure one record. No filtering, no normalization."""
    from src.preprocessing.io import load_resampled

    wav_path = Path(path)
    signal, fs_native = load_resampled(wav_path, thresholds.target_fs, cfg=cfg)
    row = measure_signal(signal, thresholds.target_fs, thresholds)
    row.update(
        {
            "record_uid": record_uid or wav_path.stem,
            "dataset_source": dataset_source,
            "subset": subset,
            "file_path": wav_path.as_posix(),
            "original_fs": int(fs_native),
        }
    )
    return row


# ---------------------------------------------------------------------------
# T26.5 -- composite flags
# ---------------------------------------------------------------------------


def apply_flags(table: Any, thresholds: QualityThresholds | None = None) -> Any:
    """Derive ``is_noisy`` / ``is_short`` / ``is_low_quality`` from the metrics.

    Separate from the measurement pass, exactly as in the Phase 16 integrity
    scan: changing a threshold is then a second of arithmetic over a cached
    table rather than a re-decode of 7,536 files, and a cached measurement can
    never carry flags from thresholds that are no longer configured.

    ``zcr_anomaly`` is a robust z-score of the record's zero-crossing rate
    against the **median and MAD of its own dataset**. Comparing a 44.1 kHz-
    sourced PASCAL record with a 2 kHz PhysioNet one on absolute ZCR would say
    more about the microphone than the recording, and the corpus median is not a
    meaningful centre for four populations at once.
    """
    import pandas as pd

    thresholds = thresholds or load_thresholds()
    flagged = table.copy()

    anomaly = pd.Series(0.0, index=flagged.index, dtype="float64")
    for _, group in flagged.groupby("dataset_source"):
        values = group["zcr_mean"].astype("float64")
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = 1.4826 * mad
        if scale <= EPS:
            anomaly.loc[group.index] = 0.0
        else:
            anomaly.loc[group.index] = ((values - median) / scale).abs()
    flagged["zcr_anomaly"] = anomaly.astype("float64")

    flagged["is_short"] = flagged["duration_sec"] < thresholds.short_duration_sec
    flagged["is_clipped"] = flagged["clipping_ratio"] > thresholds.clipping_ratio_flag
    flagged["is_silent"] = flagged["silence_ratio"] > thresholds.silence_ratio_flag
    # Three proxies, OR-ed. The drift term is the one that matches the human
    # SQI annotation (T26.5); the other two catch out-of-band and white-noise-
    # like records it misses. See the module docstring and configs/signal.yaml.
    flagged["is_noisy"] = (
        (flagged["snr_proxy_db"] < thresholds.low_snr_db_flag)
        | (flagged["spectral_flatness"] > thresholds.spectral_flatness_flag)
        | (flagged["drift_ratio_db"] > thresholds.drift_ratio_db_flag)
    )
    flagged["is_low_quality"] = (
        flagged["is_noisy"] | flagged["is_short"] | flagged["is_clipped"] | flagged["is_silent"]
    )

    # A record's flags are useless in a robustness table without the reason;
    # "low quality" covers four different problems with four different meanings.
    reasons = []
    for row in flagged.itertuples(index=False):
        why = []
        if row.snr_proxy_db < thresholds.low_snr_db_flag:
            why.append("low_snr")
        if row.spectral_flatness > thresholds.spectral_flatness_flag:
            why.append("flat_spectrum")
        if row.drift_ratio_db > thresholds.drift_ratio_db_flag:
            why.append("baseline_drift")
        if row.is_short:
            why.append("short")
        if row.is_clipped:
            why.append("clipped")
        if row.is_silent:
            why.append("silent")
        reasons.append(";".join(why))
    flagged["quality_reasons"] = reasons

    for column in ("is_short", "is_clipped", "is_silent", "is_noisy", "is_low_quality"):
        flagged[column] = flagged[column].fillna(False).astype(bool)
    return flagged


# ---------------------------------------------------------------------------
# the corpus scan
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("project_root"))


def _cache_dir() -> Path:
    from src.utils.config import load_config

    return ensure_dir(Path(load_config("paths").require("cache.metadata")))


def scan_quality(
    master: Any | None = None,
    *,
    thresholds: QualityThresholds | None = None,
    force: bool = False,
    cfg: Any | None = None,
) -> Any:
    """Measure every record in the master table, with caching.

    The cache holds **measurements only**, never flags -- see :func:`apply_flags`.
    A cache whose record set does not match the current master table is ignored
    rather than patched: a partially stale quality table is indistinguishable
    from a complete one once it is on disk.
    """
    import pandas as pd

    from src.data_loader import master as ms

    thresholds = thresholds or load_thresholds(cfg)
    if master is None:
        master = ms.load_master()

    cache_path = _cache_dir() / "quality_scan.parquet"
    wanted = list(master["record_uid"])

    if not force and cache_path.is_file():
        cached = pd.read_parquet(cache_path)
        if (
            list(cached.columns)[: len(QUALITY_COLUMNS)] == list(QUALITY_COLUMNS)
            and list(cached["record_uid"]) == wanted
            and int(cached["fs"].iloc[0]) == thresholds.target_fs
        ):
            log.info("quality scan: reusing cache (%d records)", len(cached))
            return apply_flags(cached, thresholds)
        log.info("quality scan: cache does not match the current master table; rescanning")

    root = _project_root()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(master.itertuples(index=False)):
        rows.append(
            measure_file(
                root / str(record.file_path),
                thresholds,
                record_uid=str(record.record_uid),
                dataset_source=str(record.dataset_source),
                subset=str(record.subset),
                cfg=cfg,
            )
        )
        rows[-1]["file_path"] = str(record.file_path)
        if (index + 1) % 1000 == 0:
            log.info("quality scan: %d / %d records", index + 1, len(master))

    table = pd.DataFrame(rows, columns=list(QUALITY_COLUMNS))
    table.to_parquet(cache_path, engine="pyarrow", index=False)
    log.info("quality scan: %d records measured", len(table))
    return apply_flags(table, thresholds)


# ---------------------------------------------------------------------------
# T26.5 -- calibration against PhysioNet REFERENCE-SQI
# ---------------------------------------------------------------------------


def _sqi_labels() -> dict[str, int]:
    """``{record_id: sqi}`` across the six PhysioNet training subsets.

    ``sqi=0`` is the Challenge organisers' "poor quality / unsure" mark on 364
    of the 3,240 training records. The ``validation/`` subset ships no SQI file.
    """
    from src.data_loader.physionet import load_reference_sqi
    from src.utils.config import load_config

    subsets = load_config("paths").require("dataset.d1_physionet.subsets")
    labels: dict[str, int] = {}
    for subset in subsets:
        for record_id, entry in load_reference_sqi(subset).items():
            labels[record_id] = int(entry["sqi"])
    return labels


def calibrate_against_sqi(
    table: Any,
    *,
    thresholds: QualityThresholds | None = None,
    master: Any | None = None,
) -> Any:
    """Score every noise proxy against the human SQI annotation (T26.5).

    One row per (metric, candidate threshold), plus one row for the composite
    ``is_noisy`` rule as configured. The configured threshold of each metric is
    always included and marked, so the table answers "is the configured value
    sane?" and not only "what value maximises agreement?".

    **Balanced accuracy, not agreement, is the column to read.** Only 11.4% of
    PhysioNet records carry ``sqi=0``, so a proxy that flags nothing at all
    scores 88.6% agreement while finding none of the noisy recordings. That is
    exactly what the two originally-configured proxies did, and reading the
    agreement column alone would have called it a pass -- see the 2026-08-26
    Phases 26-27 entry in ``Docs/note.md``.

    The SQI marks are a **sanity reference, not ground truth**: they cover only
    PhysioNet, they are one annotation team's judgement, and "poor quality" there
    means "we are unsure of the label", which is related to but not identical
    with "acoustically noisy".
    """
    import pandas as pd

    from src.data_loader import master as ms

    thresholds = thresholds or load_thresholds()
    if master is None:
        master = ms.load_master()

    sqi = _sqi_labels()
    d1 = table[table["dataset_source"] == "D1"].merge(
        master[["record_uid", "record_id"]], on="record_uid", how="left"
    )
    d1 = d1[d1["record_id"].astype("string").isin(sqi.keys())].copy()
    d1["sqi"] = d1["record_id"].map(sqi).astype("Int64")
    poor = (d1["sqi"] == 0).to_numpy(dtype=bool)
    n_records = len(d1)

    def score(predicted: np.ndarray, metric: str, threshold: float, configured: bool) -> dict:
        tp = int(np.count_nonzero(predicted & poor))
        fp = int(np.count_nonzero(predicted & ~poor))
        fn = int(np.count_nonzero(~predicted & poor))
        tn = int(np.count_nonzero(~predicted & ~poor))
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        return {
            "metric": metric,
            "threshold": round(float(threshold), 4),
            "is_configured": configured,
            "n_records": n_records,
            "n_sqi_poor": int(poor.sum()),
            "n_flagged": int(predicted.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "sensitivity_to_sqi_poor": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "balanced_accuracy": round(0.5 * (sensitivity + specificity), 4),
            "agreement": round((tp + tn) / n_records, 4) if n_records else 0.0,
        }

    # (column, configured threshold, direction) -- "low" means small values are
    # the suspicious ones.
    proxies = (
        ("snr_proxy_db", thresholds.low_snr_db_flag, "low"),
        ("spectral_flatness", thresholds.spectral_flatness_flag, "high"),
        ("drift_ratio_db", thresholds.drift_ratio_db_flag, "high"),
    )

    rows: list[dict] = []
    for column, configured, direction in proxies:
        values = d1[column].to_numpy(dtype=float)
        # Sweep the observed distribution rather than a fixed grid: a grid in dB
        # tells you nothing about a metric whose whole range is 0 to 0.15.
        candidates = sorted(
            {
                *(float(v) for v in np.round(np.quantile(values, np.linspace(0.02, 0.98, 25)), 4)),
                float(configured),
            }
        )
        for candidate in candidates:
            predicted = values < candidate if direction == "low" else values > candidate
            rows.append(
                score(
                    predicted,
                    column,
                    candidate,
                    abs(candidate - configured) < 1e-9,
                )
            )

    composite = (
        (d1["snr_proxy_db"] < thresholds.low_snr_db_flag)
        | (d1["spectral_flatness"] > thresholds.spectral_flatness_flag)
        | (d1["drift_ratio_db"] > thresholds.drift_ratio_db_flag)
    ).to_numpy(dtype=bool)
    rows.append(score(composite, "is_noisy (composite, as configured)", float("nan"), True))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T26.6 -- PP-08
# ---------------------------------------------------------------------------


def quality_flags_path(out_dir: str | Path | None = None) -> Path:
    """PP-08 ``signal_quality_flags.csv``."""
    from src.utils.config import load_config

    directory = (
        ensure_dir(out_dir)
        if out_dir is not None
        else ensure_dir(load_config("paths").require("outputs.preprocessing"))
    )
    return directory / "signal_quality_flags.csv"


def write_quality_flags(table: Any, out_dir: str | Path | None = None) -> Path:
    """Write PP-08 for every record, columns in a fixed order.

    Floats are rounded to six decimals. Not cosmetic: an unrounded float64 repr
    can differ in its last digits between platforms and library builds, so two
    correct runs would produce CSVs that differ byte for byte and a diff would
    stop being evidence of a real change. Six decimals is four orders of
    magnitude finer than the smallest threshold any flag here uses, and the
    flags themselves are computed at full precision before rounding.
    """
    columns = [*QUALITY_COLUMNS, *FLAG_COLUMNS]
    output = table[columns].copy()
    for column in output.select_dtypes(include=["float64", "float32"]).columns:
        output[column] = output[column].round(6)
    return save_csv(output, quality_flags_path(out_dir))


def run_quality_scan(
    *,
    force: bool = False,
    out_dir: str | Path | None = None,
    write_calibration: bool = True,
) -> tuple[Any, Path]:
    """Measure the corpus, flag it, write PP-08 (and the SQI calibration).

    Returns the flagged table and the PP-08 path.
    """
    from src.utils.timing import timer

    thresholds = load_thresholds()
    with timer("phase26_quality_scan"):
        table = scan_quality(thresholds=thresholds, force=force)

    path = write_quality_flags(table, out_dir)
    log.info(
        "PP-08: %d records (%d noisy, %d short, %d clipped, %d silent, %d low quality) -> %s",
        len(table),
        int(table["is_noisy"].sum()),
        int(table["is_short"].sum()),
        int(table["is_clipped"].sum()),
        int(table["is_silent"].sum()),
        int(table["is_low_quality"].sum()),
        path.name,
    )

    if write_calibration and thresholds.calibrate_against_sqi:
        calibration = calibrate_against_sqi(table, thresholds=thresholds)
        save_csv(calibration, quality_flags_path(out_dir).with_name("sqi_calibration.csv"))

    return table, path
