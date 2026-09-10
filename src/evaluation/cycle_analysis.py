"""Cardiac-cycle timing from the real segmentations (Phase 84).

Two corpora carry cycle annotation and they carry different *kinds* of it, which
is the first thing this module has to keep straight.

**CirCor** ships a `.tsv` per recording with `start end state` triples, state in
{0 unannotated, 1 S1, 2 systole, 3 S2, 4 diastole}. Every phase is an
**interval**, so S1, systole, S2 and diastole all have real durations.

**PASCAL set_a** ships `set_a_timing.csv`: hand-marked S1 and S2 **instants** at
44.1 kHz, 390 of them over 21 recordings. There are no durations at all. Systole
is derivable as the S1-to-S2 gap and diastole as S2-to-next-S1, but **mean S1
duration and mean S2 duration do not exist for PASCAL and are emitted as NaN**,
not as a number computed from something else. `timing_kind` says which corpus a
row came from and therefore which columns are meaningful.

Nothing else in the project has cycle annotation: PASCAL set_b and PhysioNet
2016 ship none. That is recorded as a row in the coverage summary rather than
left as an absence.

## The systole-to-diastole ratio, and why it is not one number

Computed per cycle and then averaged, never as (total systole / total diastole).
Recordings are annotated in runs with unannotated gaps between them, so the two
totals can cover different stretches of tape and their ratio would be an
artifact of where the annotator stopped. `n_paired_cycles` says how many cycles
had both phases available to pair.

## This is context, not a result (T84.4)

**None of these timings is among the 138 features.** The model has never seen a
segmentation. A difference in systolic duration between murmur-present and
murmur-absent recordings is a description of the corpus, and it cannot explain
or support any model behaviour. Every emitted artifact says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "STATE_NAMES",
    "PHASES",
    "MIN_CYCLES",
    "PASCAL_TIMING_FS",
    "CycleError",
    "CycleRow",
    "circor_cycle_table",
    "pascal_cycle_table",
    "coverage_summary",
    "class_comparison",
    "confidence_correlation",
]

log = get_logger("evaluation.cycle")

#: CirCor's segmentation states.
STATE_NAMES: dict[int, str] = {0: "unannotated", 1: "s1", 2: "systole", 3: "s2", 4: "diastole"}

#: The four annotated phases, in cycle order.
PHASES: tuple[str, ...] = ("s1", "systole", "s2", "diastole")

#: Below this many complete cycles a per-recording mean is an anecdote. Flagged,
#: never dropped.
MIN_CYCLES = 3

#: set_a is recorded at 44.1 kHz and its timing file indexes samples at that rate.
PASCAL_TIMING_FS = 44100


class CycleError(RuntimeError):
    """A cycle table cannot be built as asked."""


@dataclass(frozen=True)
class CycleRow:
    """One recording's cycle statistics. Kept as a dataclass for the field list."""

    record_uid: str
    dataset: str
    timing_kind: str


def _phase_durations(segments: Any) -> dict[str, np.ndarray]:
    """Per-phase arrays of individual segment durations, in file order."""
    out: dict[str, list[float]] = {name: [] for name in PHASES}
    for start, end, state in segments:
        name = STATE_NAMES.get(int(state))
        if name in out:
            out[name].append(float(end) - float(start))
    return {name: np.asarray(values, dtype=float) for name, values in out.items()}


def _paired_ratio(segments: Any) -> tuple[float, float, int]:
    """Mean and SD of the per-cycle systole/diastole ratio, and how many paired.

    A cycle contributes only when a systole and the diastole of the *same* cycle
    are both present. Taking the ratio of the two totals instead would divide a
    sum over one set of cycles by a sum over another.
    """
    ordered = [(float(a), float(b), int(s)) for a, b, s in segments]
    ratios: list[float] = []
    for index, (start, end, state) in enumerate(ordered):
        if state != 2:  # systole
            continue
        systole = end - start
        # The diastole of this cycle is the next state-4 segment before the next
        # systole; anything else belongs to a different cycle.
        for later_start, later_end, later_state in ordered[index + 1 :]:
            if later_state == 2:
                break
            if later_state == 4:
                diastole = later_end - later_start
                if diastole > 0 and systole > 0:
                    ratios.append(systole / diastole)
                break
    values = np.asarray(ratios, dtype=float)
    if not values.size:
        return float("nan"), float("nan"), 0
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        int(values.size),
    )


def circor_cycle_table(*, root: Path | None = None, record_table: Any = None) -> Any:
    """T84.1 -- per-recording cycle statistics from every CirCor ``.tsv``.

    Reads the segmentation files themselves rather than the Phase 15 summary:
    that summary carries per-phase **totals** and a cycle count, and a mean
    computed as total/count would be wrong wherever the two differ -- which is
    every recording whose annotation starts or ends mid-cycle, i.e. most of them.
    """
    import pandas as pd

    from src.data_loader.circor import circor_root, load_patient_files, load_segmentation
    from src.utils.config import load_config

    base = root or circor_root()
    patients = load_patient_files(base)

    audit = Path(load_config("paths").require("outputs.dataset_audit"))
    labels = None
    if (audit / "metadata_master.csv").is_file():
        master = pd.read_csv(audit / "metadata_master.csv", low_memory=False)
        labels = master[master["dataset_source"] == "D4"].set_index("record_uid")

    rows: list[dict[str, Any]] = []
    for patient in patients:
        for recording in patient.recordings:
            record_uid = "D4_training_data_" + recording.record_id
            row: dict[str, Any] = {
                "record_uid": record_uid,
                "dataset": "D4 CirCor 2022",
                "timing_kind": "interval segmentation (CirCor .tsv)",
                "record_id": recording.record_id,
                "patient_id": patient.patient_id,
                "recording_location": recording.location,
                "has_segmentation": bool(recording.tsv_file),
            }
            if not recording.tsv_file:
                row.update(_empty_phase_columns())
                row["segmentation_issues"] = "no .tsv listed in the patient file"
                rows.append(row)
                continue

            parsed = load_segmentation(base / recording.tsv_file)
            durations = _phase_durations(parsed.segments)
            total = float(
                sum(float(end) - float(start) for start, end, _ in parsed.segments)
            )
            annotated = float(
                sum(
                    float(end) - float(start)
                    for start, end, state in parsed.segments
                    if int(state) in (1, 2, 3, 4)
                )
            )
            ratio_mean, ratio_sd, paired = _paired_ratio(parsed.segments)

            row["n_segments"] = len(parsed.segments)
            row["n_cycles"] = int(durations["s1"].size)
            row["segmented_span_sec"] = total
            row["annotated_sec"] = annotated
            row["annotated_fraction"] = annotated / total if total > 0 else float("nan")
            for name in PHASES:
                values = durations[name]
                row["n_" + name] = int(values.size)
                row["mean_" + name + "_sec"] = (
                    float(values.mean()) if values.size else float("nan")
                )
                row["sd_" + name + "_sec"] = (
                    float(values.std(ddof=1)) if values.size > 1 else float("nan")
                )
            row["systole_to_diastole_ratio"] = ratio_mean
            row["systole_to_diastole_ratio_sd"] = ratio_sd
            row["n_paired_cycles"] = paired
            row["below_cycle_floor"] = bool(row["n_cycles"] < MIN_CYCLES)
            row["segmentation_issues"] = "; ".join(parsed.issues)
            rows.append(row)

    frame = pd.DataFrame(rows)
    if labels is not None:
        for column in ("murmur_label_name", "outcome_label_name", "duration_sec"):
            if column in labels.columns:
                frame[column] = frame["record_uid"].map(labels[column])
    log.info(
        "CirCor: %d recording(s), %d with segmentation, %d cycles total",
        len(frame),
        int(frame["has_segmentation"].sum()),
        int(frame.get("n_cycles", pd.Series(dtype=float)).fillna(0).sum()),
    )
    return frame


def _empty_phase_columns() -> dict[str, Any]:
    columns: dict[str, Any] = {
        "n_segments": 0,
        "n_cycles": 0,
        "segmented_span_sec": float("nan"),
        "annotated_sec": float("nan"),
        "annotated_fraction": float("nan"),
        "systole_to_diastole_ratio": float("nan"),
        "systole_to_diastole_ratio_sd": float("nan"),
        "n_paired_cycles": 0,
        "below_cycle_floor": True,
    }
    for name in PHASES:
        columns["n_" + name] = 0
        columns["mean_" + name + "_sec"] = float("nan")
        columns["sd_" + name + "_sec"] = float("nan")
    return columns


def pascal_cycle_table(*, table: Any = None) -> Any:
    """T84.2 -- the same table for PASCAL A, with its missing columns left missing.

    set_a_timing.csv marks S1 and S2 as instants. Systole is the S1-to-S2 gap
    and diastole the S2-to-next-S1 gap, both real intervals. **S1 and S2 have no
    duration in this annotation and their columns stay NaN** -- filling them
    from the CirCor distribution, or from any other recording, would be
    inventing a measurement.
    """
    import pandas as pd

    from src.data_loader.pascal import load_set_a_timing

    timing = load_set_a_timing(table)
    audit_map = _pascal_uid_map()

    rows: list[dict[str, Any]] = []
    for key, block in timing.groupby("canonical_key", sort=True):
        ordered = block.sort_values(["cycle", "time_sec"]).reset_index(drop=True)
        systoles: list[float] = []
        diastoles: list[float] = []
        events = list(zip(ordered["sound"], ordered["time_sec"], strict=True))
        for index, (sound, time) in enumerate(events[:-1]):
            next_sound, next_time = events[index + 1]
            gap = float(next_time) - float(time)
            if gap <= 0:
                continue
            if sound == "S1" and next_sound == "S2":
                systoles.append(gap)
            elif sound == "S2" and next_sound == "S1":
                diastoles.append(gap)

        systole = np.asarray(systoles, dtype=float)
        diastole = np.asarray(diastoles, dtype=float)
        paired = int(min(systole.size, diastole.size))
        ratios = (
            systole[:paired] / diastole[:paired]
            if paired
            else np.asarray([], dtype=float)
        )

        row: dict[str, Any] = {
            "record_uid": audit_map.get(str(key), ""),
            "dataset": "D2 PASCAL set_a",
            "timing_kind": "S1/S2 instants (set_a_timing.csv)",
            "record_id": str(key),
            "patient_id": "",
            "recording_location": "",
            "has_segmentation": True,
            "n_segments": len(ordered),
            "n_cycles": int(ordered["cycle"].nunique()),
            "segmented_span_sec": float(ordered["time_sec"].max() - ordered["time_sec"].min()),
            "annotated_sec": float(systole.sum() + diastole.sum()),
            "annotated_fraction": float("nan"),
            # S1 and S2 are instants here. Not zero, not imputed: absent.
            "n_s1": int((ordered["sound"] == "S1").sum()),
            "mean_s1_sec": float("nan"),
            "sd_s1_sec": float("nan"),
            "n_s2": int((ordered["sound"] == "S2").sum()),
            "mean_s2_sec": float("nan"),
            "sd_s2_sec": float("nan"),
            "n_systole": int(systole.size),
            "mean_systole_sec": float(systole.mean()) if systole.size else float("nan"),
            "sd_systole_sec": (
                float(systole.std(ddof=1)) if systole.size > 1 else float("nan")
            ),
            "n_diastole": int(diastole.size),
            "mean_diastole_sec": float(diastole.mean()) if diastole.size else float("nan"),
            "sd_diastole_sec": (
                float(diastole.std(ddof=1)) if diastole.size > 1 else float("nan")
            ),
            "systole_to_diastole_ratio": float(ratios.mean()) if ratios.size else float("nan"),
            "systole_to_diastole_ratio_sd": (
                float(ratios.std(ddof=1)) if ratios.size > 1 else float("nan")
            ),
            "n_paired_cycles": paired,
            "below_cycle_floor": bool(ordered["cycle"].nunique() < MIN_CYCLES),
            "segmentation_issues": "",
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    log.info("PASCAL set_a: %d annotated recording(s)", len(frame))
    return frame


def _pascal_uid_map() -> dict[str, str]:
    """canonical_key -> record_uid, from the audit's own derivation table."""
    import pandas as pd

    from src.utils.config import load_config

    path = Path(load_config("paths").require("outputs.dataset_audit")) / (
        "pascal_subject_derivation.csv"
    )
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    return dict(
        zip(frame["canonical_key"].astype(str), frame["record_uid"].astype(str), strict=True)
    )


def coverage_summary(circor: Any, pascal: Any) -> Any:
    """T84.2's second half -- which corpora have annotation, and which have none.

    PASCAL set_b and PhysioNet 2016 rows exist with zero counts and a stated
    reason. An absent row would read as an oversight; a zero row is a fact.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = [
        {
            "dataset": "D4 CirCor 2022",
            "timing_kind": "interval segmentation (CirCor .tsv)",
            "n_recordings": len(circor),
            "n_with_annotation": int(circor["has_segmentation"].sum()),
            "share_annotated": float(circor["has_segmentation"].mean()),
            "n_cycles_total": int(circor["n_cycles"].fillna(0).sum()),
            "has_phase_durations": True,
            "note": (
                "start/end/state triples per recording; S1, systole, S2 and "
                "diastole all have real durations."
            ),
        },
        {
            "dataset": "D2 PASCAL set_a",
            "timing_kind": "S1/S2 instants (set_a_timing.csv)",
            "n_recordings": 176,
            "n_with_annotation": len(pascal),
            "share_annotated": float(len(pascal)) / 176.0,
            "n_cycles_total": int(pascal["n_cycles"].fillna(0).sum()) if len(pascal) else 0,
            "has_phase_durations": False,
            "note": (
                "Hand-marked S1 and S2 INSTANTS, not intervals. Systole and "
                "diastole are derivable as the gaps between them; S1 and S2 "
                "durations do not exist and are left NaN. All annotated "
                "recordings are class 'normal'."
            ),
        },
        {
            "dataset": "D3 PASCAL set_b",
            "timing_kind": "none",
            "n_recordings": 656,
            "n_with_annotation": 0,
            "share_annotated": 0.0,
            "n_cycles_total": 0,
            "has_phase_durations": False,
            "note": "No timing annotation ships with set_b. T84.2 records this.",
        },
        {
            "dataset": "D1 PhysioNet 2016",
            "timing_kind": "none",
            "n_recordings": 3240,
            "n_with_annotation": 0,
            "share_annotated": 0.0,
            "n_cycles_total": 0,
            "has_phase_durations": False,
            "note": (
                "No segmentation ships with the 2016 Challenge training set. "
                "T84.2 records this."
            ),
        },
    ]
    return pd.DataFrame(rows)


def class_comparison(circor: Any, pascal: Any) -> Any:
    """T84.4 -- cycle timing by class, descriptively, on the annotated corpora.

    Every row states that these timings are not among the 138 features. The
    model has never seen a segmentation, so no difference here can explain a
    model behaviour or be offered as support for one.
    """
    import pandas as pd

    measures = [
        "mean_s1_sec",
        "mean_systole_sec",
        "mean_s2_sec",
        "mean_diastole_sec",
        "systole_to_diastole_ratio",
        "n_cycles",
        "annotated_fraction",
    ]
    rows: list[dict[str, Any]] = []

    def _add(frame: Any, dataset: str, column: str, label: str) -> None:
        if column not in frame.columns:
            return
        block = frame[frame[column].notna() & frame["has_segmentation"].astype(bool)]
        for level, part in block.groupby(column, sort=True):
            row: dict[str, Any] = {
                "dataset": dataset,
                "grouping": label,
                "level": str(level),
                "n_recordings": len(part),
                "below_reporting_floor": bool(len(part) < 30),
            }
            for measure in measures:
                if measure not in part.columns:
                    continue
                values = np.asarray(part[measure], dtype=float)
                values = values[np.isfinite(values)]
                row[measure + "_mean"] = float(values.mean()) if values.size else float("nan")
                row[measure + "_sd"] = (
                    float(values.std(ddof=1)) if values.size > 1 else float("nan")
                )
                row[measure + "_n"] = int(values.size)
            row["is_a_feature"] = False
            row["caveat"] = (
                "CONTEXT, NOT A RESULT. No cycle timing is among the 138 "
                "features; the model has never seen a segmentation."
            )
            rows.append(row)

    _add(circor, "D4 CirCor 2022", "murmur_label_name", "murmur present vs absent")
    _add(circor, "D4 CirCor 2022", "outcome_label_name", "clinical outcome normal vs abnormal")
    _add(circor, "D4 CirCor 2022", "recording_location", "auscultation location")
    if len(pascal):
        pascal = pascal.copy()
        pascal["pascal_class"] = "normal (every annotated set_a record)"
        _add(pascal, "D2 PASCAL set_a", "pascal_class", "PASCAL A class")

    return pd.DataFrame(rows)


def confidence_correlation(
    circor: Any,
    *,
    runs: tuple[tuple[str, str], ...] = (
        ("EXP-C2", "outputs/08_circor_external_validation/EXP-C2"),
        ("EXP-C1-two_class", "outputs/08_circor_external_validation/EXP-C1-two_class"),
        ("EXP-D1", "outputs/08_circor_external_validation/EXP-D1"),
    ),
    root: Path | None = None,
) -> tuple[Any, Any]:
    """T84.3 -- does annotation quality track confidence, or track the errors?

    Two questions and two answers, because they are different. Spearman is used
    rather than Pearson: annotated fraction is bounded in [0, 1] and strongly
    skewed, and a linear correlation coefficient on it would be reporting the
    skew.

    Returns ``(correlations, per_record)``. The point-biserial equivalent for
    "is this record a failure" is the same Spearman coefficient against a 0/1
    indicator, which is what the ``target='is_error'`` rows are.
    """
    import pandas as pd
    from scipy import stats

    from src.utils.evidence import PROJECT_ROOT

    base = root or PROJECT_ROOT
    annotation = circor.set_index("record_uid")
    rows: list[dict[str, Any]] = []
    joined: list[pd.DataFrame] = []

    for run, directory in runs:
        path = base / directory / "predictions.parquet"
        if not path.is_file():
            log.warning("%s: no predictions.parquet", run)
            continue
        predictions = pd.read_parquet(path)
        if "repeat" in predictions.columns and predictions["repeat"].nunique() > 1:
            predictions = predictions[predictions["repeat"] == 0]

        labels = sorted(
            int(c.split("_")[1]) for c in predictions.columns if c.startswith("proba_")
        )
        matrix = predictions[["proba_" + str(label) for label in labels]].to_numpy(dtype=float)
        block = predictions[["model_id", "record_uid", "y_true", "y_pred"]].copy()
        block["confidence"] = matrix.max(axis=1)
        block["is_error"] = (block["y_true"] != block["y_pred"]).astype(int)
        block["run"] = run
        for column in ("annotated_fraction", "n_cycles", "systole_to_diastole_ratio"):
            if column in annotation.columns:
                block[column] = block["record_uid"].map(annotation[column])
        block = block[block["annotated_fraction"].notna()]
        if not len(block):
            continue
        joined.append(block)

        for model_id, part in block.groupby("model_id", sort=True):
            for predictor in ("annotated_fraction", "n_cycles"):
                if predictor not in part.columns:
                    continue
                x = np.asarray(part[predictor], dtype=float)
                for target in ("confidence", "is_error"):
                    y = np.asarray(part[target], dtype=float)
                    mask = np.isfinite(x) & np.isfinite(y)
                    if mask.sum() < 10 or np.unique(x[mask]).size < 2:
                        continue
                    rho, p_value = stats.spearmanr(x[mask], y[mask])
                    rows.append(
                        {
                            "run": run,
                            "model_id": str(model_id),
                            "predictor": predictor,
                            "target": target,
                            "test": "Spearman rank correlation",
                            "n": int(mask.sum()),
                            "n_description": (
                                "n is the number of recordings with both a "
                                "segmentation and a prediction, each once"
                            ),
                            "rho": float(rho),
                            "p_value": float(p_value),
                            "interpretation_note": (
                                "Spearman, not Pearson: annotated fraction is "
                                "bounded in [0, 1] and heavily skewed, so a "
                                "linear coefficient would largely report the skew."
                            ),
                        }
                    )

    correlations = pd.DataFrame(rows)
    per_record = pd.concat(joined, ignore_index=True) if joined else pd.DataFrame()
    return correlations, per_record
