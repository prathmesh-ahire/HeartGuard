"""DA-08 master metadata table -- one row per recording, all four datasets.

This is the table every later Part loads instead of re-walking the dataset tree.
Phases 16-18 built :mod:`src.data_loader.catalog` as a throwaway working view;
this module builds the real thing on top of the same loaders, plus the audio
scan (Phase 16) for quality flags and the duplicate report (Phase 17) for the
duplicate decision.

Rule 4, and why this table does not break it
--------------------------------------------
T19.1 names a single ``multiclass_label`` column, but PASCAL A (4-class) and
PASCAL B (3-class) are **separate tasks with separate label spaces**, and both
number their ``normal`` class 0. A column holding both would let a downstream
``groupby("multiclass_label")`` silently merge two label spaces -- exactly the
failure rule 4 exists to prevent.

The schema is honoured as written, and the ambiguity is closed by an extra
``multiclass_task`` column naming which label space each value belongs to.
Nothing downstream should read ``multiclass_label`` directly: use
:func:`task_frame`, which selects the rows for one task and returns its labels
in a ``y`` column. :func:`assert_no_cross_task_bleed` is the enforcement, and
T19.7 runs it against the real table.

Where DA-08 is written
----------------------
T19.5 names ``dataset/metadata_master.csv``. It is written to
``outputs/01_dataset_audit/`` instead: ``dataset/`` is read-only input under
CLAUDE.md and is excluded by ``.gitignore``, so the literal path would both
break that rule and hide the table from the repository. Confirmed with the user
2026-08-26; see ``Docs/note.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_loader.catalog import DATASET_SHORT_NAMES, build_catalog
from src.utils.constants import (
    TASK_BINARY,
    TASK_CIRCOR_MURMUR,
    TASK_CIRCOR_OUTCOME,
    TASK_PASCAL_A,
    TASK_PASCAL_B,
)
from src.utils.logging_setup import get_logger

__all__ = [
    "MASTER_SCHEMA",
    "MASTER_DTYPES",
    "EXTENSION_COLUMNS",
    "TASK_LABEL_COLUMNS",
    "TASK_DATASETS",
    "master_csv_path",
    "master_parquet_path",
    "make_record_uid",
    "build_master",
    "validate_master",
    "assert_no_cross_task_bleed",
    "task_frame",
    "write_master",
    "load_master",
    "run_master_assembly",
]

log = get_logger(__name__)

# T19.1, in the order the task lists them. These columns exist, in this order,
# at the front of the table; changing the order is a schema change.
MASTER_SCHEMA: tuple[str, ...] = (
    "record_uid",
    "dataset_source",
    "subset",
    "subject_id",
    "subject_derived",
    "record_id",
    "file_path",
    "original_fs",
    "duration_sec",
    "n_samples",
    "recording_location",
    "binary_label",
    "multiclass_label",
    "murmur_label",
    "outcome_label",
    "diagnosis_class",
    "quality_flags",
    "is_duplicate",
    "is_unlabeled",
    "split_group",
)

# Everything past the T19.1 schema. Each earns its place: without
# ``multiclass_task`` the schema is ambiguous across two label spaces, and
# without ``use_in_supervised`` a consumer has to re-derive the exclusion rules
# for the 301 validation duplicates and the 247 unlabelled PASCAL files.
EXTENSION_COLUMNS: tuple[str, ...] = (
    "dataset_name",
    "multiclass_task",
    "binary_label_name",
    "multiclass_label_name",
    "murmur_label_name",
    "outcome_label_name",
    "use_in_supervised",
    "duplicate_of",
    "n_channels",
    "content_sha256",
)

ALL_COLUMNS: tuple[str, ...] = MASTER_SCHEMA + EXTENSION_COLUMNS

# T19.4 -- checked explicitly rather than trusted. Nullable integer types are
# deliberate: a missing label must be NA, never 0 or -1, or "no label for this
# task" becomes indistinguishable from "class 0".
MASTER_DTYPES: dict[str, str] = {
    "record_uid": "string",
    "dataset_source": "string",
    "subset": "string",
    "subject_id": "string",
    "subject_derived": "bool",
    "record_id": "string",
    "file_path": "string",
    "original_fs": "Int64",
    "duration_sec": "float64",
    "n_samples": "Int64",
    "recording_location": "string",
    "binary_label": "Int64",
    "multiclass_label": "Int64",
    "murmur_label": "Int64",
    "outcome_label": "Int64",
    "diagnosis_class": "string",
    "quality_flags": "string",
    "is_duplicate": "bool",
    "is_unlabeled": "bool",
    "split_group": "string",
    "dataset_name": "string",
    "multiclass_task": "string",
    "binary_label_name": "string",
    "multiclass_label_name": "string",
    "murmur_label_name": "string",
    "outcome_label_name": "string",
    "use_in_supervised": "bool",
    "duplicate_of": "string",
    "n_channels": "Int64",
    "content_sha256": "string",
}

# The label column each task reads, and the datasets allowed to populate it.
TASK_LABEL_COLUMNS: dict[str, str] = {
    TASK_BINARY: "binary_label",
    TASK_PASCAL_A: "multiclass_label",
    TASK_PASCAL_B: "multiclass_label",
    TASK_CIRCOR_MURMUR: "murmur_label",
    TASK_CIRCOR_OUTCOME: "outcome_label",
}

TASK_DATASETS: dict[str, str] = {
    TASK_BINARY: "D1",
    TASK_PASCAL_A: "D2",
    TASK_PASCAL_B: "D3",
    TASK_CIRCOR_MURMUR: "D4",
    TASK_CIRCOR_OUTCOME: "D4",
}

_FLAG_COLUMN_NAMES: tuple[str, ...] = (
    "is_zero_length",
    "is_truncated",
    "is_silent",
    "is_constant",
    "is_clipped",
)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def _project_root() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("project_root"))


def _audit_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if out_dir is not None:
        return ensure_dir(out_dir)
    return ensure_dir(load_config("paths").require("outputs.dataset_audit"))


def master_csv_path(out_dir: str | Path | None = None) -> Path:
    """DA-08 CSV. See the module docstring for why it is not under ``dataset/``."""
    return _audit_dir(out_dir) / "metadata_master.csv"


def master_parquet_path(out_dir: str | Path | None = None) -> Path:
    """The Parquet twin (T19.5). Gitignored -- regenerated from the loaders."""
    return _audit_dir(out_dir) / "metadata_master.parquet"


# --------------------------------------------------------------------------
# T19.3 -- record_uid
# --------------------------------------------------------------------------


def make_record_uid(dataset_source: str, subset: str, record_id: str) -> str:
    """``{dataset}_{subset}_{record_id}`` (T19.3).

    The loaders already build uids this way; this function is the single
    definition of the rule, and T19.6 checks every row against it rather than
    trusting three independent implementations to agree.
    """
    for name, value in (
        ("dataset_source", dataset_source),
        ("subset", subset),
        ("record_id", record_id),
    ):
        if not str(value).strip():
            raise ValueError("record_uid needs a non-empty " + name)
    return str(dataset_source) + "_" + str(subset) + "_" + str(record_id)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------


def _quality_flags(scan: Any) -> Any:
    """Collapse the Phase 16 boolean flag columns into one ``;``-joined string."""
    import pandas as pd

    present = [c for c in _FLAG_COLUMN_NAMES if c in scan.columns]
    if not present:
        return pd.Series([""] * len(scan), index=scan.index, dtype="object")

    def _row(row: Any) -> str:
        return ";".join(name for name in present if bool(row[name]))

    flags = scan[present].apply(_row, axis=1)
    if "readable" in scan.columns:
        flags = flags.mask(~scan["readable"].astype(bool), "unreadable")
    return flags


def _duplicate_report(catalog: Any, scan: Any, envelopes: Any) -> Any:
    """DA-06, preferring the copy already on disk.

    Re-deriving it needs the envelope matrix, which only a fresh
    :func:`scan_corpus` returns; reading the written report avoids forcing a
    rescan on every caller that just wants the keep/drop column.
    """
    import pandas as pd

    report_path = _audit_dir() / "duplicate_report.csv"
    if report_path.is_file():
        return pd.read_csv(report_path, keep_default_na=False)

    from src.data_loader.duplicates import build_duplicate_report
    from src.data_loader.integrity import load_thresholds, scan_corpus

    if envelopes is None:
        scan, envelopes = scan_corpus(catalog)
    return build_duplicate_report(
        scan, envelopes, threshold=load_thresholds().near_duplicate_correlation
    )


def build_master(
    *,
    catalog: Any | None = None,
    scan: Any | None = None,
    duplicates: Any | None = None,
    physionet: Any | None = None,
    pascal: Any | None = None,
    circor: Any | None = None,
) -> Any:
    """Merge the D1-D4 record tables into the master table (T19.2 -- T19.4).

    Every input is optional and rebuilt when omitted, so a caller that already
    holds the tables (the audit script does) pays for them once.
    """
    import pandas as pd

    if catalog is None:
        catalog = build_catalog(physionet=physionet, pascal=pascal, circor=circor)
    if physionet is None or pascal is None or circor is None:
        # The master table needs a handful of per-dataset columns the catalog
        # deliberately drops (diagnosis_class, the numeric label codes). Rather
        # than widen the catalog, re-fetch the source tables -- they are cached
        # by the caller in the audit script, and cheap enough alone.
        if physionet is None:
            from src.data_loader.physionet import load_physionet

            physionet = load_physionet()
        if pascal is None:
            from src.data_loader.pascal import load_pascal

            pascal = load_pascal()
        if circor is None:
            from src.data_loader.circor import load_circor

            circor = load_circor(with_segmentation=False)

    master = catalog.copy()
    master["dataset_name"] = master["dataset_source"].map(DATASET_SHORT_NAMES)

    # ---- T19.2: labels only for the tasks a dataset legitimately owns ------
    # Every label column starts empty. Each dataset then writes into its own
    # columns and no others; nothing is coerced across label spaces.
    for column in ("binary_label", "multiclass_label", "murmur_label", "outcome_label"):
        master[column] = pd.NA
    for column in (
        "binary_label_name",
        "multiclass_label_name",
        "murmur_label_name",
        "outcome_label_name",
        "multiclass_task",
        "diagnosis_class",
    ):
        master[column] = ""

    d1 = physionet.set_index("record_uid")
    d1_rows = master["dataset_source"] == "D1"
    d1_uids = master.loc[d1_rows, "record_uid"]
    master.loc[d1_rows, "binary_label"] = d1.loc[d1_uids, "binary_label"].to_numpy()
    master.loc[d1_rows, "binary_label_name"] = d1.loc[
        d1_uids, "binary_label_name"
    ].to_numpy()
    master.loc[d1_rows, "diagnosis_class"] = (
        d1.loc[d1_uids, "diagnosis_class"].fillna("").astype(str).to_numpy()
    )

    d23 = pascal.set_index("record_uid")
    for dataset, task in (("D2", TASK_PASCAL_A), ("D3", TASK_PASCAL_B)):
        rows = master["dataset_source"] == dataset
        uids = master.loc[rows, "record_uid"]
        labels = d23.loc[uids, "multiclass_label"]
        names = d23.loc[uids, "multiclass_label_name"]
        master.loc[rows, "multiclass_label"] = pd.array(
            [pd.NA if pd.isna(v) else int(v) for v in labels], dtype="Int64"
        )
        master.loc[rows, "multiclass_label_name"] = names.to_numpy()
        # Blank where there is no label: an unlabelled PASCAL file belongs to no
        # task, so tagging it with one would be a lie about its label space.
        master.loc[rows, "multiclass_task"] = [
            "" if pd.isna(v) else task for v in labels
        ]

    d4 = circor.set_index("record_uid")
    d4_rows = master["dataset_source"] == "D4"
    d4_uids = master.loc[d4_rows, "record_uid"]
    master.loc[d4_rows, "murmur_label"] = d4.loc[d4_uids, "murmur_label"].to_numpy()
    master.loc[d4_rows, "murmur_label_name"] = d4.loc[d4_uids, "murmur"].to_numpy()
    master.loc[d4_rows, "outcome_label"] = d4.loc[d4_uids, "outcome_label"].to_numpy()
    master.loc[d4_rows, "outcome_label_name"] = d4.loc[d4_uids, "outcome"].to_numpy()
    master.loc[d4_rows, "n_channels"] = d4.loc[d4_uids, "n_channels"].to_numpy()

    d23_rows = master["dataset_source"].isin(["D2", "D3"])
    master.loc[d23_rows, "n_channels"] = d23.loc[
        master.loc[d23_rows, "record_uid"], "n_channels"
    ].to_numpy()
    master.loc[d1_rows, "n_channels"] = 1  # PhysioNet .hea declares one signal

    # ---- quality flags, from the Phase 16 scan -----------------------------
    envelopes = None
    if scan is None:
        from src.data_loader.integrity import scan_corpus

        scan, envelopes = scan_corpus(catalog)
    flags = pd.Series(
        _quality_flags(scan).to_numpy(), index=scan["record_uid"].to_numpy()
    )
    hashes = pd.Series(
        scan["content_sha256"].to_numpy(), index=scan["record_uid"].to_numpy()
    )
    master["quality_flags"] = (
        master["record_uid"].map(flags).fillna("").astype(str).to_numpy()
    )
    master["content_sha256"] = (
        master["record_uid"].map(hashes).fillna("").astype(str).to_numpy()
    )

    # ---- duplicate decision, from DA-06 ------------------------------------
    if duplicates is None:
        duplicates = _duplicate_report(catalog, scan, envelopes)
    dropped = duplicates[duplicates["decision"] == "drop"]
    drop_map = pd.Series(
        dropped["duplicate_of"].fillna("").to_numpy(),
        index=dropped["record_uid"].to_numpy(),
    )
    master["is_duplicate"] = master["record_uid"].isin(drop_map.index).to_numpy()
    master["duplicate_of"] = (
        master["record_uid"].map(drop_map).fillna("").astype(str).to_numpy()
    )

    # ---- split_group -------------------------------------------------------
    # The key grouped CV groups on. Blank for anything that enters no fold, so
    # "excluded from CV" is visible in the table rather than inferred from three
    # other columns.
    master["split_group"] = master["subject_id"].where(
        master["use_in_supervised"].astype(bool), ""
    )

    master = master.reindex(columns=list(ALL_COLUMNS))
    for column, dtype in MASTER_DTYPES.items():
        if dtype == "Int64":
            master[column] = pd.to_numeric(master[column], errors="coerce").astype(
                "Int64"
            )
        elif dtype == "bool":
            master[column] = master[column].fillna(False).astype(bool)
        elif dtype == "string":
            master[column] = master[column].fillna("").astype("string")
        else:
            master[column] = master[column].astype(dtype)

    master = master.sort_values("record_uid", kind="stable").reset_index(drop=True)
    validate_master(master)

    log.info(
        "master metadata: %d records, %d columns, %d supervised, %d duplicates",
        len(master),
        len(master.columns),
        int(master["use_in_supervised"].sum()),
        int(master["is_duplicate"].sum()),
    )
    return master


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def assert_no_cross_task_bleed(master: Any) -> None:
    """T19.2 / T19.7 -- no record carries a label for a task it does not own.

    Checked in both directions: a dataset must not populate another task's
    label column, and it must not leave its own blank on a supervised record.
    """
    import pandas as pd

    for task, column in TASK_LABEL_COLUMNS.items():
        # multiclass_label is shared by two tasks, so its owners are D2 and D3
        # together; the label-space split is carried by multiclass_task.
        owners = ["D2", "D3"] if column == "multiclass_label" else [TASK_DATASETS[task]]
        foreign = master[~master["dataset_source"].isin(owners)]
        bled = foreign[foreign[column].notna()]
        if not bled.empty:
            raise ValueError(
                str(len(bled)) + " record(s) outside " + "/".join(owners)
                + " carry a " + column + " value -- label spaces have been "
                "merged (rule 4). First: " + str(bled["record_uid"].iloc[0])
            )

    # multiclass_task must agree with the dataset that produced the row.
    expected = master["dataset_source"].map({"D2": TASK_PASCAL_A, "D3": TASK_PASCAL_B})
    tagged = master[master["multiclass_task"].fillna("") != ""]
    mismatch = tagged[tagged["multiclass_task"] != expected.loc[tagged.index]]
    if not mismatch.empty:
        raise ValueError(
            str(len(mismatch)) + " record(s) carry a multiclass_task that does not "
            "match their dataset -- PASCAL A and PASCAL B have been merged"
        )
    # A labelled PASCAL row must say which of the two label spaces it is in;
    # without it, the two 4-class/3-class vocabularies are indistinguishable.
    labelled = master[master["multiclass_label"].notna()]
    untagged = labelled[labelled["multiclass_task"].fillna("") == ""]
    if not untagged.empty:
        raise ValueError(
            str(len(untagged)) + " record(s) carry a multiclass_label with no "
            "multiclass_task -- the label space is ambiguous"
        )

    # Every supervised record must carry its own task's label.
    for task, column in TASK_LABEL_COLUMNS.items():
        owner = TASK_DATASETS[task]
        own = master[
            (master["dataset_source"] == owner)
            & master["use_in_supervised"].astype(bool)
        ]
        missing = own[own[column].isna()]
        if not missing.empty:
            raise ValueError(
                str(len(missing)) + " supervised " + owner + " record(s) have no "
                + column + " -- the " + task + " task is incomplete"
            )
    del pd  # keep the import meaningful only where used


def validate_master(master: Any, *, check_paths: bool = False) -> None:
    """Schema, dtype and uniqueness checks (T19.4).

    ``check_paths`` walks the filesystem and is left off by default so the
    validator stays usable on a synthetic frame; T19.6 turns it on.
    """
    import pandas as pd

    missing = [c for c in MASTER_SCHEMA if c not in master.columns]
    if missing:
        raise ValueError("master table is missing schema columns: " + ", ".join(missing))
    if list(master.columns)[: len(MASTER_SCHEMA)] != list(MASTER_SCHEMA):
        raise ValueError(
            "the first " + str(len(MASTER_SCHEMA)) + " columns must be the T19.1 "
            "schema in order; got " + ", ".join(list(master.columns)[: len(MASTER_SCHEMA)])
        )

    for column, expected in MASTER_DTYPES.items():
        if column not in master.columns:
            continue
        actual = str(master[column].dtype)
        if actual != expected:
            raise TypeError(
                "column " + column + " has dtype " + actual + ", expected " + expected
            )

    if master["record_uid"].duplicated().any():
        dupes = master.loc[master["record_uid"].duplicated(), "record_uid"].tolist()
        raise ValueError(
            str(len(dupes)) + " duplicate record_uid(s): " + ", ".join(dupes[:5])
        )
    if master["record_uid"].isna().any() or (master["record_uid"] == "").any():
        raise ValueError("record_uid must be non-empty on every row")

    rebuilt = [
        make_record_uid(row.dataset_source, row.subset, row.record_id)
        for row in master.itertuples(index=False)
    ]
    mismatched = [
        (a, b) for a, b in zip(master["record_uid"], rebuilt, strict=True) if a != b
    ]
    if mismatched:
        raise ValueError(
            str(len(mismatched)) + " record_uid(s) do not follow the T19.3 rule "
            "{dataset}_{subset}_{record_id}; first: " + repr(mismatched[0])
        )

    if (master["duration_sec"] <= 0).any() or master["duration_sec"].isna().any():
        raise ValueError("every record must have a positive duration_sec")
    if master["original_fs"].isna().any():
        raise ValueError("every record must have an original_fs")

    supervised = master[master["use_in_supervised"].astype(bool)]
    blank_group = supervised[supervised["split_group"].fillna("") == ""]
    if not blank_group.empty:
        raise ValueError(
            str(len(blank_group)) + " supervised record(s) have no split_group"
        )
    excluded = master[~master["use_in_supervised"].astype(bool)]
    leaked_group = excluded[excluded["split_group"].fillna("") != ""]
    if not leaked_group.empty:
        raise ValueError(
            str(len(leaked_group)) + " excluded record(s) still carry a split_group"
        )

    assert_no_cross_task_bleed(master)

    if check_paths:
        root = _project_root()
        missing_files = [
            row.record_uid
            for row in master.itertuples(index=False)
            if not (root / str(row.file_path)).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                str(len(missing_files)) + " file_path(s) do not resolve on disk; "
                "first: " + missing_files[0]
            )
    del pd


def task_frame(master: Any, task: str, *, supervised_only: bool = True) -> Any:
    """The sanctioned way to get one task's rows and labels.

    Returns only the rows belonging to ``task``, with the task's label column
    copied to ``y``. Nothing downstream should read ``multiclass_label``
    directly -- that is how PASCAL A and PASCAL B get merged.
    """
    if task not in TASK_LABEL_COLUMNS:
        raise KeyError(
            "unknown task " + repr(task) + " -- expected one of: "
            + ", ".join(TASK_LABEL_COLUMNS)
        )
    column = TASK_LABEL_COLUMNS[task]
    rows = master[master["dataset_source"] == TASK_DATASETS[task]]
    if column == "multiclass_label":
        rows = master[master["multiclass_task"] == task]
    if supervised_only:
        rows = rows[rows["use_in_supervised"].astype(bool)]
    rows = rows[rows[column].notna()].copy()
    rows["y"] = rows[column].astype("Int64")
    rows["task"] = task
    return rows.reset_index(drop=True)


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def write_master(master: Any, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Emit DA-08 as CSV plus its Parquet twin (T19.5)."""
    from src.utils.io import save_csv, save_parquet

    csv_path = save_csv(master, master_csv_path(out_dir))
    parquet_path = save_parquet(master, master_parquet_path(out_dir))
    log.info("wrote metadata_master.csv (%d rows) and its Parquet twin", len(master))
    return csv_path, parquet_path


def load_master(path: str | Path | None = None) -> Any:
    """Read DA-08 back with its dtypes intact.

    Prefers the Parquet twin, which round-trips ``Int64`` and ``string`` without
    a dtype map; falls back to the CSV.
    """
    import pandas as pd

    if path is not None:
        target = Path(path)
    else:
        target = master_parquet_path()
        if not target.is_file():
            target = master_csv_path()
    if not target.is_file():
        raise FileNotFoundError(
            "DA-08 has not been generated yet -- run scripts/01_run_dataset_audit.py"
        )
    if target.suffix == ".parquet":
        master = pd.read_parquet(target)
    else:
        master = pd.read_csv(target, dtype=MASTER_DTYPES, keep_default_na=False)
        for column, dtype in MASTER_DTYPES.items():
            if dtype == "Int64":
                master[column] = pd.to_numeric(
                    master[column].replace("", None), errors="coerce"
                ).astype("Int64")
            elif dtype == "float64":
                master[column] = pd.to_numeric(master[column], errors="coerce")
            elif dtype == "bool":
                master[column] = master[column].astype(str).isin(["True", "true", "1"])
    return master


def run_master_assembly(
    *,
    catalog: Any | None = None,
    scan: Any | None = None,
    duplicates: Any | None = None,
    physionet: Any | None = None,
    pascal: Any | None = None,
    circor: Any | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Phase 19 end to end: build, validate against disk, write DA-08."""
    from src.utils.evidence import register_evidence

    master = build_master(
        catalog=catalog,
        scan=scan,
        duplicates=duplicates,
        physionet=physionet,
        pascal=pascal,
        circor=circor,
    )
    validate_master(master, check_paths=True)
    csv_path, parquet_path = write_master(master, out_dir)

    register_evidence(
        evidence_id="DA-08",
        objective="Data audit",
        dataset="D1-D4",
        metric_or_asset="Master metadata table (one row per recording)",
        filename=csv_path,
        source_data="src/data_loader/{physionet,pascal,circor}.py",
        command="python scripts/01_run_dataset_audit.py",
    )
    return {
        "master": master,
        "csv_path": csv_path,
        "parquet_path": parquet_path,
        "n_records": len(master),
        "n_supervised": int(master["use_in_supervised"].sum()),
    }
