"""EXP-E2 -- duration robustness (Phase 73).

Same two-halves structure as the noise track, for the same reason.

**Observational.** The stored out-of-fold predictions partitioned by the T18.3
duration bands -- short below 5 s, medium 5-20 s, long above 20 s. The bands are
*loaded* from :func:`src.data_loader.summaries.assign_duration_bands`, never
re-cut here, so T18's assignment and EXP-E2's are the same assignment. A gap
between bands is an association: duration is confounded with corpus (every short
record is PASCAL, every record over 60 s is PhysioNet), so a short-band deficit
may be a PASCAL effect wearing a duration label.

**Interventional.** The truncation study: the same PhysioNet recordings clipped
to 10, 5 and 3 seconds and re-extracted, scored by a model refitted without their
subjects. Only the length varies. This is the half that can actually attribute
anything to duration, and it is cheap in a way the noise sweep was not --
``time_sample_entropy`` is O(N^2), so a 3-second clip extracts far faster than
the 30-second original.

**The short-record diagnostic (T73.3).** PASCAL set_b reaches 0.763 s. Measured
over the whole corpus, the extractor returns **zero NaNs at any duration** -- so
"does it break" is answered, and answered no. The interesting question is what
the numbers mean when the window is that small, which is what
:func:`short_record_diagnostics` measures: how much padding the bandpass had to
give up, how many heart sounds the envelope detector could possibly see, and
which feature families sit furthest from their long-record distribution.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "DURATION_BANDS",
    "TRUNCATION_SECONDS",
    "DurationError",
    "duration_bands",
    "stratify_by_duration",
    "short_record_diagnostics",
    "truncate",
    "truncation_features",
]

log = get_logger("evaluation.duration")

#: T18.3, in the order every table prints them.
DURATION_BANDS: tuple[str, ...] = ("short", "medium", "long")

#: Clip lengths in seconds. ``inf`` is the untouched control and must reproduce
#: the stored feature matrix, exactly as in the noise sweep.
TRUNCATION_SECONDS: tuple[float, ...] = (float("inf"), 10.0, 5.0, 3.0)


class DurationError(RuntimeError):
    """The duration analysis cannot be run as requested."""


def duration_bands() -> Any:
    """``record_uid -> (duration_sec, band)`` for every recording.

    The band comes from ``assign_duration_bands``, the same function T18.3 used.
    Re-implementing the cut here would let the two drift, and a robustness result
    reported against different bands from the ones the corpus was described with
    is not comparable to anything.
    """
    import pandas as pd

    from src.data_loader.summaries import assign_duration_bands
    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv"
    if not path.is_file():
        raise DurationError(str(path) + " is missing; run scripts/01_run_dataset_audit.py")

    frame = pd.read_csv(path, low_memory=False)
    banded = assign_duration_bands(frame)
    return banded[
        ["record_uid", "dataset_source", "subset", "duration_sec", "n_samples", "duration_band"]
    ].copy()


def stratify_by_duration(
    predictions: Any,
    *,
    labels: Any,
    positive_label: int = 1,
    class_names: Any = None,
    min_units: int = 30,
) -> Any:
    """Per-fold metrics for every ``(model, fold, duration band)``.

    A band whose fold slice falls below ``min_units`` is carried with
    ``reported=False`` and its size, on the same 30-record floor the location and
    quality analyses use. On PASCAL A the short band is a handful of records and
    would otherwise produce a confident-looking number from three of them.
    """
    import pandas as pd

    from src.evaluation import metrics as mt

    declared = [int(v) for v in labels]
    binary = len(declared) == 2
    frame = pd.DataFrame(predictions).copy()
    lookup = duration_bands().set_index("record_uid")
    frame["duration_band"] = frame["record_uid"].astype(str).map(lookup["duration_band"])
    frame["duration_sec"] = frame["record_uid"].astype(str).map(lookup["duration_sec"])
    missing = frame["duration_band"].isna()
    if missing.any():
        raise DurationError(
            str(int(missing.sum()))
            + " prediction row(s) have no duration band, e.g. "
            + ", ".join(frame.loc[missing, "record_uid"].astype(str).head(5))
        )

    proba_columns = [c for c in frame.columns if c.startswith("proba_")]
    corpus = frame.drop_duplicates("record_uid")["duration_band"].value_counts().to_dict()

    def score(y_true: Any, y_pred: Any, y_proba: Any) -> dict[str, float]:
        if binary:
            return mt.binary_metrics(
                y_true, y_pred, y_proba, labels=declared, positive_label=int(positive_label)
            )
        names = tuple(class_names) if class_names else None
        return mt.multiclass_metrics(y_true, y_pred, y_proba, labels=declared, class_names=names)

    rows: list[dict[str, Any]] = []
    for (model_id, fold_label, band), block in frame.groupby(
        ["model_id", "fold_label", "duration_band"], sort=True
    ):
        present = sorted({int(v) for v in block["y_true"]})
        proba = (
            block[proba_columns].to_numpy(dtype=float)
            if proba_columns and len(present) > 1
            else None
        )
        row: dict[str, Any] = {
            "model_id": model_id,
            "fold_label": fold_label,
            "duration_band": band,
            "n_units": len(block),
            "n_records_corpus": int(corpus.get(band, 0)),
            "reported": len(block) >= int(min_units),
            "median_duration_sec": float(np.median(block["duration_sec"].to_numpy(dtype=float))),
            "min_duration_sec": float(np.min(block["duration_sec"].to_numpy(dtype=float))),
            "n_classes_present": len(present),
        }
        try:
            row.update(score(block["y_true"].to_numpy(int), block["y_pred"].to_numpy(int), proba))
        except ValueError as error:  # pragma: no cover - degenerate slices only
            log.warning("%s %s %s: %s", model_id, fold_label, band, error)
            continue
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T73.3 -- what happens at the bottom of the duration range
# ---------------------------------------------------------------------------


def short_record_diagnostics(*, n_examples: int = 20) -> tuple[Any, Any]:
    """Measure, don't assert, what a sub-5-second window costs.

    Three things are measured per short record, plus a family-level distribution
    shift over the whole corpus:

    ``padlen_reduced`` / ``too_short``
        ``filtfilt`` needs ``3 * max(len(a), len(b))`` samples of padding. Below
        that the project's filter reduces the padding rather than refusing, which
        is the right call but means the shortest records are filtered with less
        edge protection than the rest.

    ``n_envelope_peaks``
        ``env_peak_rate`` counts heart SOUNDS, S1 and S2 -- roughly 2.4/s on this
        corpus. A 0.763 s window can physically contain one or two of them, so a
        rate estimated from it has an enormous relative error even though the
        feature is finite and looks perfectly respectable.

    ``standardized_shift``
        Per feature, ``(mean_short - mean_long) / sd_long``. Names which features
        actually move, rather than assuming the frame-count-dependent ones do.
    """
    import pandas as pd

    from src.feature_extraction.matrix import load_matrix
    from src.feature_extraction.registry import feature_names
    from src.preprocessing import filters as flt
    from src.preprocessing import io as pio
    from src.preprocessing import normalize as nrm

    names = list(feature_names())
    matrix = load_matrix()
    banded = duration_bands().set_index("record_uid")
    matrix = matrix.set_index("record_uid")
    matrix["duration_band"] = banded["duration_band"]

    short = matrix[matrix["duration_band"] == "short"]
    long = matrix[matrix["duration_band"] == "long"]
    if short.empty or long.empty:
        raise DurationError("the corpus has no short or no long records to compare")

    long_mean = long[names].to_numpy(dtype=float).mean(axis=0)
    long_sd = long[names].to_numpy(dtype=float).std(axis=0, ddof=1)
    short_mean = short[names].to_numpy(dtype=float).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(long_sd > 0, (short_mean - long_mean) / long_sd, np.nan)
    shifts = pd.DataFrame(
        {
            "feature": names,
            "family": [name.split("_")[0] for name in names],
            "mean_short": short_mean,
            "mean_long": long_mean,
            "sd_long": long_sd,
            "standardized_shift": shift,
        }
    ).sort_values("standardized_shift", key=lambda s: s.abs(), ascending=False)

    # The n_examples shortest records, filtered individually so the padding
    # behaviour is measured rather than inferred from the duration.
    examples = short.reset_index().nsmallest(n_examples, "duration_sec")
    rows: list[dict[str, Any]] = []
    for _, record in examples.iterrows():
        raw, fs = pio.load_resampled(record["file_path"])
        result = flt.filter_signal(np.asarray(raw, dtype=float), fs)
        prepared = np.asarray(nrm.normalize_signal(result.signal).signal, dtype=float)
        peaks = float(record.get("env_peak_rate", float("nan"))) * float(record["duration_sec"])
        rows.append(
            {
                "record_uid": record["record_uid"],
                "dataset_source": record["dataset_source"],
                "duration_sec": float(record["duration_sec"]),
                "n_samples": len(prepared),
                "fs": int(fs),
                "padlen": int(result.padlen),
                "padlen_reduced": bool(result.padlen_reduced),
                "too_short_to_filter": bool(result.too_short),
                "env_peak_rate": float(record.get("env_peak_rate", float("nan"))),
                "implied_n_heart_sounds": peaks,
                "n_nan_features": int(np.isnan(record[names].to_numpy(dtype=float)).sum()),
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values("duration_sec").reset_index(drop=True)
    log.info(
        "short records: %d of %d had reduced filter padding; "
        "shortest %.3f s implies %.1f heart sound(s)",
        int(diagnostics["padlen_reduced"].sum()),
        len(diagnostics),
        float(diagnostics["duration_sec"].iloc[0]),
        float(diagnostics["implied_n_heart_sounds"].iloc[0]),
    )
    return diagnostics, shifts


# ---------------------------------------------------------------------------
# T73.4 -- the truncation study
# ---------------------------------------------------------------------------


def truncate(signal: Any, fs: int, seconds: float, rng: Any) -> Any:
    """Clip to ``seconds``, from a randomly placed window, not always the start.

    Always taking the opening seconds would confound length with position: the
    first two seconds of a recording carry the transducer being settled and the
    subject being told to hold still, and every corpus here has some of that. The
    window start is drawn from the seeded generator, so it is reproducible.
    """
    x = np.asarray(signal, dtype=float)
    if not np.isfinite(seconds):
        return x.copy()
    want = round(float(seconds) * int(fs))
    if want >= x.size:
        return x.copy()
    start = int(rng.integers(0, x.size - want + 1))
    return x[start : start + want].copy()


def _level_key(seconds: float) -> int:
    value = float(seconds)
    return 2**31 - 1 if not np.isfinite(value) else round(1000 * value)


def _truncate_one(record: dict, lengths: tuple, seed: int, names: list) -> list:
    """Every clip length of one record. Runs in a worker process."""
    from src.feature_extraction.extractor import extract_all
    from src.preprocessing import filters as flt
    from src.preprocessing import io as pio
    from src.preprocessing import normalize as nrm

    def prepare(signal: Any, fs: int) -> Any:
        filtered = flt.filter_signal(np.asarray(signal, dtype=float), fs).signal
        return np.asarray(nrm.normalize_signal(filtered).signal, dtype=float)

    uid = str(record["record_uid"])
    raw, fs = pio.load_resampled(record["file_path"])
    raw = np.asarray(raw, dtype=float)
    out: list[dict[str, Any]] = []
    for seconds in lengths:
        rng = np.random.default_rng(
            [int(seed), int(zlib.crc32(uid.encode("utf-8"))), _level_key(seconds)]
        )
        clipped = truncate(raw, fs, float(seconds), rng)
        # Truncation happens on the RAW waveform, before filtering, because a
        # short recording in the wild is short before anything is done to it.
        result = extract_all(prepare(clipped, fs), fs, record_uid=uid)
        row: dict[str, Any] = {
            "record_uid": uid,
            "clip_seconds": float(seconds),
            "realised_seconds": float(clipped.size / int(fs)),
            "y": int(record["y"]),
            "subject_id": str(record["subject_id"]),
            "original_seconds": float(record["duration_sec"]),
        }
        row.update({name: float(result.values[name]) for name in names})
        out.append(row)
    return out


def truncation_features(
    records: Any,
    *,
    lengths: Any = TRUNCATION_SECONDS,
    seed: int = 42,
    checkpoint: str | Path | None = None,
    workers: int = 4,
    batch: int = 8,
) -> Any:
    """Extract the 138 features of every record at every clip length.

    Checkpointed per batch and seeded per ``(record, length)``, on the same
    contract as the noise sweep: an interrupted run resumes bit for bit and the
    worker count cannot change a value.
    """
    import pandas as pd
    from joblib import Parallel, delayed

    from src.feature_extraction.registry import feature_names

    names = list(feature_names())
    ordered = tuple(float(v) for v in lengths)
    target = Path(checkpoint) if checkpoint else None

    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if target is not None and target.is_file():
        existing = pd.read_parquet(target)
        counts = existing.groupby("record_uid").size()
        done = set(counts[counts == len(ordered)].index.astype(str))
        rows = [r for r in existing.to_dict("records") if str(r["record_uid"]) in done]
        log.info("resuming from %s: %d complete record(s)", target, len(done))

    frame = pd.DataFrame(records)
    pending = [r for r in frame.to_dict("records") if str(r["record_uid"]) not in done]
    log.info(
        "truncation: %d record(s) x %d length(s), %d already done, %d worker(s)",
        len(pending),
        len(ordered),
        len(done),
        workers,
    )

    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        produced = Parallel(n_jobs=workers)(
            delayed(_truncate_one)(record, ordered, seed, names) for record in chunk
        )
        for group in produced:
            rows.extend(group)
        if target is not None:
            pd.DataFrame(rows).to_parquet(target, index=False)
        log.info(
            "checkpoint: %d/%d record(s)",
            len(done) + start + len(chunk),
            len(done) + len(pending),
        )

    table = pd.DataFrame(rows)
    if target is not None:
        table.to_parquet(target, index=False)
    log.info(
        "truncation complete: %d row(s) over %d record(s)",
        len(table),
        table["record_uid"].nunique() if len(table) else 0,
    )
    return table
