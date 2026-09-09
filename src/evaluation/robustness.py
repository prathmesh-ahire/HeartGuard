"""EXP-E1 -- noise robustness (Phase 72).

Two independent questions, kept separate because they are answered by different
evidence and can disagree:

**1. How does the model do on the recordings the corpus already contains that
are noisy?** Answered by partitioning the stored out-of-fold predictions with the
PP-08 quality flags. Observational: no signal is altered, and the groups differ
in more than noise -- a noisy recording may also be from a different site, a
different transducer, or a sicker patient. A gap here is an association.

**2. What happens to a fixed model as noise is added to a fixed recording?**
Answered by the AWGN sweep. Interventional: the same records at 20/15/10/5/0 dB
SNR, so nothing varies except the noise. A gap here is causal for additive white
noise specifically, which is not the noise a stethoscope actually picks up.

Neither alone is enough, and quoting the second as if it answered the first is
the mistake this module is arranged to prevent.

**The sweep model is refitted without the sampled records.** The finalized binary
model was fitted on all 3,240 PhysioNet records, so evaluating it on any of them
is in-sample and optimistic. The sweep therefore refits the same configuration on
the PhysioNet records whose *subjects* are not in the sample -- one extra fit,
and the degradation curve then sits on honest absolute numbers rather than only
being valid as a relative shape.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "QUALITY_GROUPS",
    "SNR_LEVELS",
    "RobustnessError",
    "quality_groups",
    "stratify_by_quality",
    "sqi_crosscheck",
    "add_awgn",
    "measured_snr_db",
    "sweep_features",
]

log = get_logger("evaluation.robustness")

#: The three PP-08 partitions, in the order every table prints them.
QUALITY_GROUPS: tuple[str, ...] = ("clean", "noisy", "low_quality_other")

#: Added-noise levels, in dB SNR relative to the raw recording's own power.
#: ``inf`` is the untouched control and must reproduce the stored feature matrix;
#: it is in the sweep precisely so that a pipeline difference cannot be mistaken
#: for a noise effect.
SNR_LEVELS: tuple[float, ...] = (float("inf"), 20.0, 15.0, 10.0, 5.0, 0.0)

_FLAGS_PATH = ("outputs", "02_preprocessing", "signal_quality_flags.csv")


class RobustnessError(RuntimeError):
    """The robustness analysis cannot be run as requested."""


# ---------------------------------------------------------------------------
# 1. the observational half -- PP-08 quality groups
# ---------------------------------------------------------------------------


def quality_groups() -> Any:
    """``record_uid -> group`` for every recording, from the PP-08 flags.

    ``noisy`` takes precedence over the other low-quality flags: a recording that
    is both short and noisy belongs in the noise analysis, and a record can only
    sit in one group or the group sizes stop summing to the corpus.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT.joinpath(*_FLAGS_PATH)
    if not path.is_file():
        raise RobustnessError(str(path) + " is missing; run the PP-08 quality pass")

    frame = pd.read_csv(path)
    group = np.where(
        frame["is_noisy"].to_numpy(dtype=bool),
        "noisy",
        np.where(
            frame["is_low_quality"].to_numpy(dtype=bool),
            "low_quality_other",
            "clean",
        ),
    )
    out = frame[
        ["record_uid", "dataset_source", "snr_proxy_db", "is_noisy", "is_low_quality"]
    ].copy()
    out["quality_group"] = group
    out["quality_reasons"] = frame["quality_reasons"].fillna("")
    return out


def stratify_by_quality(
    predictions: Any,
    *,
    labels: Any,
    positive_label: int = 1,
    class_names: Any = None,
    min_units: int = 30,
) -> Any:
    """Per-fold metrics for every ``(model, fold, quality group)``.

    ``min_units`` marks a group whose fold slice is too small to quote, using the
    same 30-record floor as the location analysis. Nothing is dropped -- a group
    that vanishes from a table without explanation is indistinguishable from one
    that was never measured.
    """
    import pandas as pd

    from src.evaluation import metrics as mt

    declared = [int(v) for v in labels]
    binary = len(declared) == 2
    frame = pd.DataFrame(predictions).copy()
    lookup = quality_groups().set_index("record_uid")
    frame["quality_group"] = frame["record_uid"].astype(str).map(lookup["quality_group"])
    missing = frame["quality_group"].isna()
    if missing.any():
        raise RobustnessError(
            str(int(missing.sum()))
            + " prediction row(s) have no PP-08 quality flag, e.g. "
            + ", ".join(frame.loc[missing, "record_uid"].astype(str).head(5))
        )
    frame["snr_proxy_db"] = frame["record_uid"].astype(str).map(lookup["snr_proxy_db"])
    proba_columns = [c for c in frame.columns if c.startswith("proba_")]
    corpus = frame.drop_duplicates("record_uid")["quality_group"].value_counts().to_dict()

    def score(y_true: Any, y_pred: Any, y_proba: Any) -> dict[str, float]:
        if binary:
            return mt.binary_metrics(
                y_true, y_pred, y_proba, labels=declared, positive_label=int(positive_label)
            )
        names = tuple(class_names) if class_names else None
        return mt.multiclass_metrics(y_true, y_pred, y_proba, labels=declared, class_names=names)

    rows: list[dict[str, Any]] = []
    for (model_id, fold_label, group), block in frame.groupby(
        ["model_id", "fold_label", "quality_group"], sort=True
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
            "quality_group": group,
            "n_units": len(block),
            "n_records_corpus": int(corpus.get(group, 0)),
            "reported": len(block) >= int(min_units),
            "mean_snr_proxy_db": float(np.nanmean(block["snr_proxy_db"].to_numpy(dtype=float))),
            "n_classes_present": len(present),
        }
        try:
            row.update(score(block["y_true"].to_numpy(int), block["y_pred"].to_numpy(int), proba))
        except ValueError as error:  # pragma: no cover - degenerate slices only
            log.warning("%s %s %s: %s", model_id, fold_label, group, error)
            continue
        rows.append(row)

    return pd.DataFrame(rows)


def sqi_crosscheck(*, subsets: Any = None) -> Any:
    """T72.3 -- the PP-08 grouping against PhysioNet's own SQI annotation.

    ``REFERENCE-SQI.csv`` carries a second column: ``1`` where the challenge
    organisers considered the recording good enough to score, ``0`` where they
    did not. It is an independent human-curated judgement of the same property
    PP-08 estimates from the waveform, and it exists only for D1.

    The agreement is reported as it is, not tuned. It is **poor**, and that is
    the finding: the two are measuring related but different things, so a
    PP-08 group must never be described as "the noisy recordings" without saying
    whose definition of noisy.
    """
    import pandas as pd

    from src.data_loader.physionet import load_reference_sqi
    from src.utils.config import load_config

    declared = subsets or load_config("paths").require("dataset.d1_physionet.subsets")
    rows: list[dict[str, Any]] = []
    for subset in declared:
        try:
            table = load_reference_sqi(subset)
        except (OSError, ValueError) as error:  # pragma: no cover - dataset absent
            raise RobustnessError("cannot read REFERENCE-SQI for " + str(subset)) from error
        for record_id, values in table.items():
            rows.append(
                {
                    "record_uid": "D1_" + str(subset) + "_" + str(record_id),
                    "subset": subset,
                    "sqi_good": int(values.get("sqi", 0)) == 1,
                }
            )

    sqi = pd.DataFrame(rows)
    groups = quality_groups()
    merged = sqi.merge(groups, on="record_uid", how="inner")
    if merged.empty:
        raise RobustnessError("no PhysioNet record joined the SQI table to the PP-08 flags")

    merged["pp08_poor"] = merged["quality_group"] != "clean"
    merged["sqi_poor"] = ~merged["sqi_good"]

    both = int((merged["pp08_poor"] & merged["sqi_poor"]).sum())
    pp08_only = int((merged["pp08_poor"] & ~merged["sqi_poor"]).sum())
    sqi_only = int((~merged["pp08_poor"] & merged["sqi_poor"]).sum())
    neither = int((~merged["pp08_poor"] & ~merged["sqi_poor"]).sum())
    total = len(merged)

    snr = merged.groupby("sqi_poor")["snr_proxy_db"].mean().to_dict()
    summary = {
        "n_records": total,
        "n_sqi_poor": int(merged["sqi_poor"].sum()),
        "n_pp08_poor": int(merged["pp08_poor"].sum()),
        "both_poor": both,
        "pp08_poor_only": pp08_only,
        "sqi_poor_only": sqi_only,
        "both_clean": neither,
        "agreement": (both + neither) / total,
        "recall_of_sqi_poor": both / max(both + sqi_only, 1),
        "precision_on_sqi_poor": both / max(both + pp08_only, 1),
        "mean_snr_proxy_db_sqi_good": float(snr.get(False, float("nan"))),
        "mean_snr_proxy_db_sqi_poor": float(snr.get(True, float("nan"))),
    }
    log.info(
        "PP-08 vs REFERENCE-SQI over %d records: agreement %.4f, recall of SQI-poor %.4f",
        total,
        summary["agreement"],
        summary["recall_of_sqi_poor"],
    )
    return pd.DataFrame([summary]), merged


# ---------------------------------------------------------------------------
# 2. the interventional half -- additive white Gaussian noise
# ---------------------------------------------------------------------------


def add_awgn(signal: Any, snr_db: float, rng: Any) -> Any:
    """Add white Gaussian noise to reach ``snr_db`` against the signal's power.

    SNR is defined on the **raw** recording, before filtering, because that is
    where noise enters a real recording. The bandpass then removes the
    out-of-band share of it, exactly as it would for real noise -- so the
    post-filter SNR is higher than the nominal level, and
    :func:`measured_snr_db` reports what actually survived rather than assuming
    the nominal figure held.
    """
    x = np.asarray(signal, dtype=float)
    if not np.isfinite(snr_db):
        return x.copy()
    power = float(np.mean(x**2))
    if power <= 0:
        raise RobustnessError("cannot set an SNR against a zero-power signal")
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def measured_snr_db(clean: Any, noisy: Any) -> float:
    """The SNR actually realised between two signals, in dB."""
    a = np.asarray(clean, dtype=float)
    b = np.asarray(noisy, dtype=float)
    residual = float(np.mean((b - a) ** 2))
    if residual <= 0:
        return float("inf")
    return float(10.0 * np.log10(float(np.mean(a**2)) / residual))


def _level_key(level: float) -> int:
    """A stable integer per SNR level, including the infinite (clean) one."""
    value = float(level)
    return 2**31 - 1 if not np.isfinite(value) else round(1000 * value)


def _prepare(signal: Any, fs: int) -> Any:
    """Filter and normalize, the same chain ``preprocess`` runs, without a cache."""
    from src.preprocessing import filters as flt
    from src.preprocessing import normalize as nrm

    filtered = flt.filter_signal(np.asarray(signal, dtype=float), fs).signal
    return np.asarray(nrm.normalize_signal(filtered).signal, dtype=float)


def _sweep_one(record: dict, levels: tuple, seed: int, names: list) -> list:
    """Every SNR level of one record. Runs in a worker process."""
    from src.feature_extraction.extractor import extract_all
    from src.preprocessing import io as pio

    uid = str(record["record_uid"])
    raw, fs = pio.load_resampled(record["file_path"])
    raw = np.asarray(raw, dtype=float)
    # The clean signal through the SAME chain, so the realised SNR is measured
    # where the model actually sees it. Measuring it on the raw waveform would
    # just return the nominal level and prove nothing: the bandpass removes the
    # out-of-band share of the noise, so what survives is always higher.
    prepared_clean = _prepare(raw, fs)
    out: list[dict[str, Any]] = []
    for level in levels:
        # zlib.crc32, never the builtin hash(): string hashing is salted per
        # process unless PYTHONHASHSEED is set before the interpreter starts,
        # and a seed that depends on that is not a seed.
        rng = np.random.default_rng(
            [int(seed), int(zlib.crc32(uid.encode("utf-8"))), _level_key(level)]
        )
        noisy = add_awgn(raw, float(level), rng)
        prepared = _prepare(noisy, fs)
        result = extract_all(prepared, fs, record_uid=uid)
        row: dict[str, Any] = {
            "record_uid": uid,
            "snr_db": float(level),
            "realised_snr_db": measured_snr_db(prepared_clean, prepared),
            "raw_snr_db": measured_snr_db(raw, noisy),
            "y": int(record["y"]),
            "subject_id": str(record["subject_id"]),
            "duration_sec": float(record["duration_sec"]),
        }
        row.update({name: float(result.values[name]) for name in names})
        out.append(row)
    return out


def sweep_features(
    records: Any,
    *,
    levels: Any = SNR_LEVELS,
    seed: int = 42,
    checkpoint: str | Path | None = None,
    workers: int = 4,
    batch: int = 8,
) -> Any:
    """Extract the 138 features of every record at every SNR level.

    The noise draw is seeded **per (record, level)** from the global seed and a
    CRC of the record uid, so a resumed run reproduces an interrupted one bit for
    bit, the worker count cannot change any value, and the level order cannot
    change any record's noise. Work is checkpointed after every batch of records,
    because this is the kind of job the laptop gets closed in the middle of.
    """
    import pandas as pd
    from joblib import Parallel, delayed

    from src.feature_extraction.registry import feature_names

    names = list(feature_names())
    ordered = tuple(float(level) for level in levels)
    target = Path(checkpoint) if checkpoint else None

    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if target is not None and target.is_file():
        existing = pd.read_parquet(target)
        rows = existing.to_dict("records")
        counts = existing.groupby("record_uid").size()
        done = set(counts[counts == len(ordered)].index.astype(str))
        rows = [r for r in rows if str(r["record_uid"]) in done]
        log.info("resuming from %s: %d complete record(s)", target, len(done))

    frame = pd.DataFrame(records)
    pending = [r for r in frame.to_dict("records") if str(r["record_uid"]) not in done]
    log.info(
        "sweep: %d record(s) x %d level(s), %d already done, %d worker(s)",
        len(pending),
        len(ordered),
        len(done),
        workers,
    )

    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        produced = Parallel(n_jobs=workers)(
            delayed(_sweep_one)(record, ordered, seed, names) for record in chunk
        )
        for group in produced:
            rows.extend(group)
        if target is not None:
            pd.DataFrame(rows).to_parquet(target, index=False)
        log.info(
            "checkpoint: %d/%d record(s), %d row(s)",
            len(done) + start + len(chunk),
            len(done) + len(pending),
            len(rows),
        )

    table = pd.DataFrame(rows)
    if target is not None:
        table.to_parquet(target, index=False)
    log.info(
        "sweep complete: %d row(s) over %d record(s)",
        len(table),
        table["record_uid"].nunique() if len(table) else 0,
    )
    return table
