"""Precomputed preprocessing examples for the dashboard (Phase 114).

T114.4 wants a record picker with raw versus filtered waveforms and quality
indicators; T114.5 wants filter and normalization **toggles driven by
precomputed example pairs**. "Precomputed" is the load-bearing word: the browser
must never run a Butterworth filter, because a second filter implementation
would be a second answer to what the pipeline does, and the one on screen is the
one a reader would believe.

So all four states of the 2x2 grid — filter off/on crossed with normalization
off/on — are computed here, in Python, through
`src.preprocessing.filters.filter_signal` and
`src.preprocessing.normalize.normalize_signal`. Those are the same functions the
corpus was preprocessed with. Nothing in this module implements a filter.

## Why the examples are written to a CSV as well as to the payload

Phase 113 shipped a segmentation overlay that could only be *rebuilt* on a
machine holding the 1.3 GB corpus, so a fresh clone silently exported
"unavailable" while the data sat committed beside it. The same trap applies
here. `export_examples` therefore writes
`outputs/02_preprocessing/preprocessing_examples.csv`, which is committed, and
:func:`examples_payload` reads the corpus when it is present and that CSV when
it is not. The corpus always wins where it exists, so the committed copy cannot
drift unnoticed.

## The point budget, and why decimation is safe here

A 6-second window at 2 kHz is 12,000 samples per series; four states across four
records would be 192,000 numbers. They are decimated to
:data:`POINTS_PER_SERIES` by **striding**, not by averaging: a mean would draw a
curve the pipeline never produced, and this figure exists to show what the
pipeline produced. The stride is recorded in the payload so a reader knows
exactly which samples are on screen, and the full-resolution signal remains in
`outputs/13_figures_diagrams/G05_before_and_after_filtering.csv`.

The window is 6.0 s for the same reason G05 uses one: twenty seconds of PCG in a
browser-width chart is a solid band with no room to see an S1 at all.

Amplitudes are rounded to five decimals and times to four. That is display
precision, chosen against the chart: five decimals is far finer than one pixel
at any width this renders at, and it takes the payload from 86 kB gzipped to
about half that. The full-precision signal is never on this page; it is in the
corpus and in G05's CSV, and nothing here is a metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "EXAMPLES_CSV",
    "EXAMPLE_RECORDS",
    "POINTS_PER_SERIES",
    "WINDOW_SECONDS",
    "ExampleRecord",
    "examples_payload",
    "export_examples",
]

log = get_logger("reporting.signals")

#: Seconds of signal shown. Matches G05 so the page and the 300 dpi figure are
#: looking at the same window.
WINDOW_SECONDS = 6.0

#: Points per series after striding. 1,200 is about two points per pixel at a
#: typical chart width, which is enough to render an S1 transient honestly.
POINTS_PER_SERIES = 1200

EXAMPLES_CSV = "outputs/02_preprocessing/preprocessing_examples.csv"


@dataclass(frozen=True)
class ExampleRecord:
    """One recording the preprocessing page can show.

    Pinned by `record_uid` rather than picked at export time. A viewer that
    shows a different recording on each run cannot be cross-checked against
    anything, and research rule 1 applies to a waveform exactly as it applies to
    a metric.
    """

    key: str
    record_uid: str
    title: str
    #: Why this record is in the set. Rendered, so the choice is not arbitrary.
    note: str


EXAMPLE_RECORDS: tuple[ExampleRecord, ...] = (
    ExampleRecord(
        key="d1_normal",
        record_uid="D1_training-a_a0005",
        title="PhysioNet 2016 — normal",
        note=(
            "The reference record used by the inference gate, so the waveform on "
            "this page and the probability in tests/test_inference.py come from "
            "the same bytes."
        ),
    ),
    ExampleRecord(
        key="d1_abnormal",
        record_uid="D1_training-a_a0001",
        title="PhysioNet 2016 — abnormal",
        note="Adult PhysioNet recording at 2 kHz, the corpus the binary model is fitted on.",
    ),
    ExampleRecord(
        key="d2_pascal_a",
        record_uid="D2_set_a_normal__201101070538",
        title="PASCAL set A — 44.1 kHz",
        note=(
            "Recorded at 44.1 kHz on a phone microphone. Resampling to the working "
            "rate is the largest single change the pipeline makes to this corpus."
        ),
    ),
    ExampleRecord(
        key="d3_pascal_b",
        record_uid="D3_set_b_normal__103_1305031931979_B",
        title="PASCAL set B — 4 kHz",
        note=(
            "PASCAL set B recordings are short; several are under a second, which "
            "is why the duration bound in the inference validator exists."
        ),
    ),
)

#: The 2x2 grid T114.5's toggles select between.
STATES: tuple[tuple[str, bool, bool], ...] = (
    ("raw", False, False),
    ("filtered", True, False),
    ("normalized", False, True),
    ("filtered_normalized", True, True),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stride(values: np.ndarray, budget: int) -> tuple[np.ndarray, int]:
    """Every ``k``-th sample, never an average. Returns ``(kept, stride)``."""
    if values.size <= budget:
        return values, 1
    stride = int(np.ceil(values.size / budget))
    return values[::stride], stride


def _states_for(signal: np.ndarray, fs: int) -> dict[str, np.ndarray]:
    """The four toggle states, each produced by the pipeline's own functions."""
    from src.preprocessing.filters import filter_signal
    from src.preprocessing.normalize import normalize_signal

    filtered = np.asarray(filter_signal(signal, fs).signal, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name, use_filter, use_norm in STATES:
        series = filtered if use_filter else signal
        if use_norm:
            series = np.asarray(normalize_signal(series).signal, dtype=np.float64)
        out[name] = np.asarray(series, dtype=np.float64)
    return out


def export_examples(out_path: str | Path | None = None) -> Path:
    """Recompute the examples from the corpus and write the committed CSV.

    Reads four recordings. This is not the corpus-wide preprocessing run and
    takes about a second; it is deliberately small because the machine's cores
    belong to the experiment runs.
    """
    import pandas as pd

    from src.preprocessing.io import load_resampled, target_fs

    root = _project_root()
    master = pd.read_csv(root / "outputs" / "01_dataset_audit" / "metadata_master.csv")
    indexed = master.set_index("record_uid")

    rows: list[dict[str, Any]] = []
    for record in EXAMPLE_RECORDS:
        if record.record_uid not in indexed.index:
            raise KeyError("example record " + record.record_uid + " is not in metadata_master.csv")
        meta = indexed.loc[record.record_uid]
        path = root / str(meta["file_path"])
        if not path.is_file():
            raise FileNotFoundError("no audio for " + record.record_uid + " at " + str(path))

        # load, mono, resample -- the pipeline's own first three steps. The
        # working rate is what every later stage sees, so it is what the page
        # shows; the native rate travels beside it because "44.1 kHz phone
        # recording" is the interesting fact about a PASCAL A file.
        samples, native_fs = load_resampled(path)
        fs = target_fs()
        signal = np.asarray(samples, dtype=np.float64)[: int(WINDOW_SECONDS * fs)]

        states = _states_for(signal, fs)
        kept, stride = _stride(np.arange(signal.size), POINTS_PER_SERIES)
        seconds = kept.astype(np.float64) / float(fs)

        for index, offset in enumerate(kept):
            row: dict[str, Any] = {
                "record_key": record.key,
                "record_uid": record.record_uid,
                "fs": fs,
                "native_fs": int(native_fs),
                "stride": stride,
                "sample_index": int(offset),
                "time_sec": round(float(seconds[index]), 4),
            }
            for name in states:
                row[name] = round(float(states[name][int(offset)]), 5)
            rows.append(row)

    frame = pd.DataFrame(rows)
    target = Path(out_path) if out_path is not None else root / EXAMPLES_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, lineterminator="\n")
    log.info("wrote %d rows for %d example records -> %s", len(frame), len(EXAMPLE_RECORDS), target)
    return target


def _quality_for(uid: str) -> list[dict[str, Any]]:
    """The record's own quality measurements, formatted in Python.

    Read from `signal_quality_flags.csv`, which the preprocessing run wrote.
    Nothing is recomputed and nothing is judged here -- a threshold turning a
    number into "poor" is a clinical-sounding call the dashboard has no basis
    for making.
    """
    import math

    import pandas as pd

    from src.reporting.tables import format_value

    path = _project_root() / "outputs" / "02_preprocessing" / "signal_quality_flags.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    match = frame[frame["record_uid"] == uid]
    if match.empty:
        return []

    row = match.iloc[0]
    # The kind is declared per field rather than inferred. `infer_kind` is
    # deliberately conservative and falls back to "text", which renders a value
    # unchanged -- so `snr_proxy_db` came out as -19.557766 to fifteen
    # significant figures. A signal-quality measurement printed at that
    # precision claims an accuracy the measurement does not have.
    shown: tuple[tuple[str, str, str], ...] = (
        ("duration_sec", "Duration", "seconds"),
        ("fs", "Working rate (Hz)", "count"),
        ("original_fs", "Native rate (Hz)", "count"),
        ("rms", "RMS", "metric"),
        ("peak", "Peak amplitude", "metric"),
        ("crest_factor_db", "Crest factor (dB)", "metric"),
        ("dynamic_range_db", "Dynamic range (dB)", "metric"),
        ("clipping_ratio", "Clipping ratio", "metric"),
        ("silence_ratio", "Silence ratio", "metric"),
        ("snr_proxy_db", "SNR proxy (dB)", "metric"),
        ("spectral_flatness", "Spectral flatness", "metric"),
        ("zcr_mean", "Zero-crossing rate", "metric"),
    )
    out: list[dict[str, Any]] = []
    for name, label, kind in shown:
        if name not in frame.columns:
            continue
        value = row[name]
        numeric: float | None
        try:
            numeric = float(value)
            if not math.isfinite(numeric):
                numeric = None
        except (TypeError, ValueError):
            numeric = None
        out.append(
            {
                "name": name,
                "label": label,
                "display": format_value(value, kind),
                "value": numeric,
            }
        )
    return out


def examples_payload() -> dict[str, Any]:
    """The four example records and their four states, ready to import.

    The corpus wins when it is present: `export_examples` re-runs and rewrites
    the CSV, so the committed copy cannot silently diverge from the audio. When
    the corpus is absent the committed CSV is read as-is, which is what lets a
    fresh clone build this page at all.
    """
    import pandas as pd

    root = _project_root()
    csv_path = root / EXAMPLES_CSV
    source = "corpus"

    master_path = root / "outputs" / "01_dataset_audit" / "metadata_master.csv"
    can_rebuild = master_path.is_file()
    if can_rebuild:
        try:
            export_examples(csv_path)
        except (FileNotFoundError, KeyError, OSError) as error:
            log.info("cannot rebuild preprocessing examples from the corpus: %s", error)
            can_rebuild = False
    if not can_rebuild:
        source = "committed csv"

    if not csv_path.is_file():
        return {
            "available": False,
            "reason": (
                "neither the corpus nor "
                + EXAMPLES_CSV
                + " is present, so no preprocessing example could be produced"
            ),
            "records": [],
        }

    frame = pd.read_csv(csv_path)
    state_names = [name for name, _f, _n in STATES]

    records: list[dict[str, Any]] = []
    for record in EXAMPLE_RECORDS:
        part = frame[frame["record_key"] == record.key]
        if part.empty:
            continue
        records.append(
            {
                "key": record.key,
                "record_uid": record.record_uid,
                "title": record.title,
                "note": record.note,
                "fs": int(part["fs"].iloc[0]),
                "native_fs": int(part["native_fs"].iloc[0]),
                "stride": int(part["stride"].iloc[0]),
                "n_points": len(part),
                "time_sec": [round(float(v), 4) for v in part["time_sec"].tolist()],
                "series": {
                    name: [round(float(v), 5) for v in part[name].tolist()]
                    for name in state_names
                    if name in part.columns
                },
                "quality": _quality_for(record.record_uid),
            }
        )

    return {
        "available": bool(records),
        "reason": None if records else "the examples CSV holds no rows for the pinned records",
        "source": source,
        "window_seconds": WINDOW_SECONDS,
        "points_per_series": POINTS_PER_SERIES,
        "states": [
            {"key": name, "filter": use_filter, "normalize": use_norm}
            for name, use_filter, use_norm in STATES
        ],
        "note": (
            "All four states are computed in Python by the pipeline's own "
            "filter_signal and normalize_signal, then strided (never averaged) to "
            "the point budget. The browser draws them; it does not produce them."
        ),
        "records": records,
    }
