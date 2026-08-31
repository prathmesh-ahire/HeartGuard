"""The corpus, one row per recording, as a payload a page can filter (Phase 114).

T114.3 asks for "dataset filtering and drill-down to record level". That needs
every record, not a sample: a filter that silently shows a subset is worse than
no filter, because the reader counts what they see.

## Columnar, not a list of objects

All 7,536 records with thirteen fields is 779 kB of JSON as rows-of-objects and
53 kB gzipped as **columns** — the repeated keys are most of the bytes. Columnar
is what makes shipping the whole corpus affordable inside the route budget, so
nothing has to be truncated and no runtime fetch is needed. Every array is the
same length, and `n_records` is the length they must all have.

## What is and is not in here

Identity, provenance and labels: uid, corpus, subset, subject, duration, native
rate, the four label spaces, and the three flags that decide whether a record is
modelled. No features, no probabilities, no metrics. This is the inventory the
audit produced, formatted once in Python.

`duration_display` sits beside `duration_sec` for the reason the exporter states
generally: the string is what a page renders, the number is what a chart or a
filter compares. A page must never format the number itself.

## The four counts that keep colliding

`use_in_supervised` is the modelled subset and is smaller than the corpus:
PhysioNet 3,541 files against 3,240 modelled, PASCAL A 176 against 124, PASCAL B
656 against 461, CirCor 3,163 against 3,163. Anything totalling over this index
has to say which population it counted, which is why `summary()` reports both
and labels them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "INDEX_COLUMNS",
    "MASTER_CSV",
    "dataset_summary_payload",
    "record_index_payload",
]

log = get_logger("reporting.record_index")

MASTER_CSV = "outputs/01_dataset_audit/metadata_master.csv"

#: (column, label, kind). `kind` is the T85.6 rounding class, so a duration and
#: a count are not formatted the same way.
INDEX_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("record_uid", "Record", "text"),
    ("dataset_source", "Corpus", "text"),
    ("dataset_name", "Dataset", "text"),
    ("subset", "Subset", "text"),
    ("subject_id", "Subject", "text"),
    ("subject_derived", "Subject derived", "text"),
    ("duration_sec", "Duration (s)", "seconds"),
    ("original_fs", "Native rate (Hz)", "count"),
    ("n_channels", "Channels", "count"),
    ("binary_label_name", "Binary", "text"),
    ("multiclass_label_name", "Multiclass", "text"),
    ("murmur_label_name", "Murmur", "text"),
    ("outcome_label_name", "Outcome", "text"),
    ("use_in_supervised", "Modelled", "text"),
    ("is_duplicate", "Duplicate", "text"),
    ("is_unlabeled", "Unlabelled", "text"),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cell(value: Any) -> Any:
    """One value as JSON, with non-finite floats and pandas NA becoming ``None``."""
    import pandas as pd

    if value is None:
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def record_index_payload() -> dict[str, Any]:
    """Every audited recording, columnar, with a display string per column.

    Raises when `metadata_master.csv` is absent: that file is committed, so its
    absence means the audit output has been moved or deleted and the page would
    otherwise render an empty corpus as though the corpus were empty.
    """
    import pandas as pd

    from src.reporting.tables import format_value

    path = _project_root() / MASTER_CSV
    if not path.is_file():
        raise FileNotFoundError("the record index needs " + MASTER_CSV)

    frame = pd.read_csv(path)
    missing = [name for name, _label, _kind in INDEX_COLUMNS if name not in frame.columns]
    if missing:
        raise KeyError(MASTER_CSV + " is missing columns: " + ", ".join(missing))

    frame = frame.sort_values("record_uid").reset_index(drop=True)

    columns: list[dict[str, Any]] = []
    for name, label, kind in INDEX_COLUMNS:
        series = frame[name]
        values = [_cell(v) for v in series.tolist()]
        display = [
            format_value(v, kind) if kind != "text" else ("n/a" if v is None else str(v))
            for v in values
        ]
        columns.append(
            {
                "name": name,
                "label": label,
                "kind": kind,
                "display": display,
                # `values` is present only for a column a chart or a numeric
                # filter compares. A text column carries None, per the export
                # contract -- an array of nulls would read as "these were
                # numbers and we lost them".
                "values": values if kind in ("seconds", "count", "metric") else None,
            }
        )

    return {
        "source": MASTER_CSV,
        "n_records": len(frame),
        "n_supervised": int(frame["use_in_supervised"].astype(bool).sum()),
        "columns": columns,
        "facets": _facets(frame),
        "summary": _summary(frame),
        "scope_note": (
            "Every audited file is listed, including unlabelled and duplicate "
            "records. 'Modelled' marks the supervised subset the experiments "
            "actually use, which is smaller than the corpus for three of the four "
            "families. Any total taken over this table has to say which of the two "
            "populations it counted."
        ),
    }


def _facets(frame: Any) -> list[dict[str, Any]]:
    """The categorical columns a filter can offer, with their real value sets."""
    facets: list[dict[str, Any]] = []
    for name, label, kind in INDEX_COLUMNS:
        if kind != "text" or name in ("record_uid", "subject_id"):
            continue
        values = sorted({str(_cell(v)) for v in frame[name].tolist()})
        if len(values) > 24:
            continue
        facets.append({"name": name, "label": label, "values": values})
    return facets


def _summary(frame: Any) -> list[dict[str, Any]]:
    """Per-corpus counts, both populations, formatted here rather than in a page."""
    from src.reporting.tables import format_value

    rows: list[dict[str, Any]] = []
    for source, part in frame.groupby("dataset_source", sort=True):
        modelled = part["use_in_supervised"].astype(bool)
        durations = part.loc[modelled, "duration_sec"].astype(float)
        rows.append(
            {
                "dataset_source": str(source),
                "dataset_name": str(part["dataset_name"].iloc[0]),
                "n_files": len(part),
                "n_files_display": format_value(len(part), "count"),
                "n_modelled": int(modelled.sum()),
                "n_modelled_display": format_value(int(modelled.sum()), "count"),
                "n_subjects": int(part["subject_id"].nunique()),
                "n_subjects_display": format_value(int(part["subject_id"].nunique()), "count"),
                "hours_modelled": float(durations.sum() / 3600.0) if len(durations) else 0.0,
                "hours_modelled_display": format_value(
                    float(durations.sum() / 3600.0) if len(durations) else 0.0, "metric"
                ),
            }
        )
    return rows


def dataset_summary_payload() -> dict[str, Any]:
    """The per-corpus counts alone, without the 7,536-row index.

    A separate payload because it has a separate cost. The home page shows four
    tiles and the dataset page shows the same tiles plus the full index; when
    both imported the index, webpack found a module used by two routes and
    hoisted it into the chunk **every** page loads, putting 41 kB gzipped onto
    twelve pages that show no records at all. A summary and an index are
    different things and are now different files.
    """
    import pandas as pd

    path = _project_root() / MASTER_CSV
    if not path.is_file():
        raise FileNotFoundError("the dataset summary needs " + MASTER_CSV)

    frame = pd.read_csv(path)
    return {
        "source": MASTER_CSV,
        "n_records": len(frame),
        "n_supervised": int(frame["use_in_supervised"].astype(bool).sum()),
        "summary": _summary(frame),
        "scope_note": (
            "Every audited file is counted in 'files'; 'modelled' is the supervised "
            "subset the experiments actually use, which is smaller than the corpus "
            "for three of the four families. Any total taken over these rows has to "
            "say which of the two populations it counted."
        ),
    }
