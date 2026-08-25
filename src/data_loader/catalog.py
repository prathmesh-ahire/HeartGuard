"""One combined view of D1-D4, for the dataset audit (Phases 16-18).

Deliberately minimal. Phases 16, 17 and 18 each need to walk every record in the
corpus, and none of them should each grow its own way of stitching four loaders
together. What they need is the intersection: where the file is, how long it is,
which subject it belongs to, and which class it carries in whichever task owns
it.

**This is not the master metadata table.** T19.1 defines that schema, T19.3 its
uid rules, T19.4 its dtype validation and T19.5 its CSV and Parquet twins. This
module is the audit's working view and nothing more; Phase 19 builds the real
thing on top of the same loaders.

**Rule 4 is enforced by shape.** There is no ``label`` column. Each task gets
its own, and a record carries a value only in the columns for tasks it actually
belongs to -- a PASCAL A record has ``pascal_a_class`` and nothing in
``pascal_b_class``, and a CirCor record carries both of its tasks side by side
without either being folded into the other.
"""

from __future__ import annotations

from typing import Any

from src.utils.constants import (
    TASK_BINARY,
    TASK_CIRCOR_MURMUR,
    TASK_CIRCOR_OUTCOME,
    TASK_PASCAL_A,
    TASK_PASCAL_B,
)
from src.utils.logging_setup import get_logger

__all__ = [
    "CATALOG_COLUMNS",
    "TASK_CLASS_COLUMNS",
    "DATASET_TASKS",
    "DATASET_SHORT_NAMES",
    "build_catalog",
    "dataset_tasks",
]

log = get_logger(__name__)

# The class column each task reads its labels from.
TASK_CLASS_COLUMNS: dict[str, str] = {
    TASK_BINARY: "binary_class",
    TASK_PASCAL_A: "pascal_a_class",
    TASK_PASCAL_B: "pascal_b_class",
    TASK_CIRCOR_MURMUR: "circor_murmur_class",
    TASK_CIRCOR_OUTCOME: "circor_outcome_class",
}

# D4 is the only dataset carrying two tasks.
DATASET_TASKS: dict[str, tuple[str, ...]] = {
    "D1": (TASK_BINARY,),
    "D2": (TASK_PASCAL_A,),
    "D3": (TASK_PASCAL_B,),
    "D4": (TASK_CIRCOR_MURMUR, TASK_CIRCOR_OUTCOME),
}

DATASET_SHORT_NAMES: dict[str, str] = {
    "D1": "PhysioNet 2016",
    "D2": "PASCAL set_a",
    "D3": "PASCAL set_b",
    "D4": "CirCor 2022",
}

CATALOG_COLUMNS: tuple[str, ...] = (
    "record_uid",
    "dataset_source",
    "subset",
    "record_id",
    "file_path",
    "subject_id",
    "subject_derived",
    "recording_location",
    "original_fs",
    "n_samples",
    "duration_sec",
    "is_unlabeled",
    "is_duplicate",
    "duplicate_of",
    "use_in_supervised",
    "binary_class",
    "pascal_a_class",
    "pascal_b_class",
    "circor_murmur_class",
    "circor_outcome_class",
)


def dataset_tasks(dataset_source: str) -> tuple[str, ...]:
    """Tasks a dataset legitimately carries labels for."""
    if dataset_source not in DATASET_TASKS:
        raise KeyError(
            "unknown dataset " + repr(dataset_source) + " -- expected one of: "
            + ", ".join(DATASET_TASKS)
        )
    return DATASET_TASKS[dataset_source]


def _blank_columns(frame: Any) -> Any:
    """Add every catalog column the source table does not have, as empty."""
    import pandas as pd

    for column in CATALOG_COLUMNS:
        if column not in frame.columns:
            if column in ("is_unlabeled", "is_duplicate", "use_in_supervised",
                          "subject_derived"):
                frame[column] = False
            elif column in ("original_fs", "n_samples"):
                frame[column] = pd.NA
            elif column == "duration_sec":
                frame[column] = float("nan")
            else:
                frame[column] = ""
    return frame[list(CATALOG_COLUMNS)]


def build_catalog(
    *,
    physionet: Any | None = None,
    pascal: Any | None = None,
    circor: Any | None = None,
) -> Any:
    """Concatenate the four record tables into the audit's working view.

    Pass an already-built table for any dataset to avoid re-reading it; anything
    omitted is loaded here. All three loaders take tens of seconds, so the audit
    scripts build each once and hand them in.
    """
    import pandas as pd

    if physionet is None:
        from src.data_loader.physionet import load_physionet

        physionet = load_physionet()
    if pascal is None:
        from src.data_loader.pascal import load_pascal

        pascal = load_pascal()
    if circor is None:
        from src.data_loader.circor import load_circor

        circor = load_circor(with_segmentation=False)

    d1 = physionet.copy()
    d1["binary_class"] = d1["binary_label_name"]

    d2_d3 = pascal.copy()
    d2_d3["pascal_a_class"] = d2_d3["multiclass_label_name"].where(
        d2_d3["dataset_source"] == "D2", ""
    )
    d2_d3["pascal_b_class"] = d2_d3["multiclass_label_name"].where(
        d2_d3["dataset_source"] == "D3", ""
    )

    d4 = circor.copy()
    d4["circor_murmur_class"] = d4["murmur"]
    d4["circor_outcome_class"] = d4["outcome"]

    catalog = pd.concat(
        [_blank_columns(d1), _blank_columns(d2_d3), _blank_columns(d4)],
        ignore_index=True,
    )
    catalog = catalog.sort_values(["dataset_source", "record_uid"], kind="stable")
    catalog = catalog.reset_index(drop=True)

    if catalog["record_uid"].duplicated().any():
        raise ValueError("record_uid is not unique across the four record tables")

    # Rule 4, checked rather than assumed: a record must not carry a class in a
    # task its dataset does not own. This is the single place all five label
    # spaces sit in one frame, so it is the place the check belongs.
    for dataset, tasks in DATASET_TASKS.items():
        rows = catalog[catalog["dataset_source"] == dataset]
        allowed = {TASK_CLASS_COLUMNS[t] for t in tasks}
        for task, column in TASK_CLASS_COLUMNS.items():
            if column in allowed:
                continue
            populated = rows[rows[column].astype("string").fillna("") != ""]
            if not populated.empty:
                raise ValueError(
                    str(len(populated)) + " " + dataset + " record(s) carry a "
                    + task + " label, which " + dataset + " does not own -- label "
                    "spaces have been merged"
                )

    log.info(
        "catalog: %d records across %d datasets (%d usable in supervised tracks)",
        len(catalog),
        catalog["dataset_source"].nunique(),
        int(catalog["use_in_supervised"].sum()),
    )
    return catalog
