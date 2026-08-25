"""DA-07 fold maps -- generated once here, loaded everywhere else.

Rule 3 lives in this file. A subject that appears in two folds of the same
repeat invalidates every metric computed from that fold map, and the failure is
silent: nothing crashes, the numbers just come back better than they should. So
grouping is not a parameter a caller can forget -- it is baked into each task's
scheme, and :func:`assert_no_leakage` re-derives the check from the written map
rather than trusting the generator that produced it.

The schemes come from ``configs/experiments.yaml`` (``cv_schemes``); this module
does not invent one. Every experiment in Part VII loads DA-07 instead of
re-splitting, which is what makes EXP-A1 and EXP-A2 comparable fold-for-fold.

Why the map stores test folds only
----------------------------------
A record's row names the fold it is *tested* in, for one repeat. The folds of a
repeat partition the records, so the training set is "everything else in this
repeat" -- storing both halves would multiply DA-07 by the number of folds for
no extra information. :func:`iter_folds` reconstitutes train/test pairs.

sklearn has no RepeatedStratifiedGroupKFold
-------------------------------------------
``RepeatedStratifiedKFold`` exists but ignores groups; ``StratifiedGroupKFold``
honours groups but does not repeat. Repeating is done here by re-running
``StratifiedGroupKFold`` with ``random_state = seed + repeat``, which is what
the sklearn repeated wrappers do internally.

The PhysioNet validation folder
-------------------------------
T20.5 asked for the 301 ``validation/`` records to be held out as an untouched
final test set. They are byte-identical copies of 301 training records whose
205 subjects account for 1,108 of the 3,240 training records, so a
subject-clean holdout would cost 34% of the primary track for a test set
containing no new material. Confirmed with the user 2026-08-26: no D1 external
test set, recorded in ``outputs/missing_outputs_report.txt``. The 301 records
carry ``use_in_supervised=False`` and therefore appear in no fold of any map --
checked, not assumed, by :func:`assert_validation_excluded`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.constants import TASKS
from src.utils.logging_setup import get_logger
from src.utils.seed import GLOBAL_SEED

__all__ = [
    "SPLIT_MAP_COLUMNS",
    "TASK_SCHEMES",
    "SplitScheme",
    "load_scheme",
    "make_split_map",
    "build_split_maps",
    "iter_folds",
    "assert_no_leakage",
    "assert_validation_excluded",
    "split_map_path",
    "write_split_map",
    "load_split_map",
    "fold_summary",
    "run_split_generation",
]

log = get_logger(__name__)

SPLIT_MAP_COLUMNS: tuple[str, ...] = (
    "task",
    "scheme",
    "dataset_source",
    "n_splits",
    "n_repeats",
    "repeat",
    "fold",
    "record_uid",
    "split_group",
    "subject_id",
    "subject_derived",
    "y",
    "class_name",
)

# Which scheme in configs/experiments.yaml each task uses. These match the `cv:`
# key of the experiment that owns the task (EXP-A1, EXP-B1, EXP-B2, EXP-C1,
# EXP-C2), so a scheme change in the config reaches both places at once.
TASK_SCHEMES: dict[str, str] = {
    "binary": "repeated_5x5_grouped",
    "pascal_a": "repeated_5x2_stratified",
    "pascal_b": "grouped_5fold",
    "circor_murmur": "patient_grouped_5fold",
    "circor_outcome": "patient_grouped_5fold",
}


@dataclass(frozen=True)
class SplitScheme:
    """One ``cv_schemes`` entry, resolved."""

    name: str
    kind: str
    n_splits: int
    n_repeats: int
    group_key: str | None
    shuffle: bool
    random_state: int

    @property
    def total_folds(self) -> int:
        return self.n_splits * self.n_repeats


def load_scheme(name: str) -> SplitScheme:
    """Read one CV scheme out of ``configs/experiments.yaml``."""
    from src.utils.config import load_config

    schemes = load_config("experiments").require("cv_schemes")
    if name not in schemes:
        raise KeyError(
            "unknown cv scheme " + repr(name) + " -- configs/experiments.yaml "
            "defines: " + ", ".join(sorted(schemes))
        )
    spec = schemes[name]
    return SplitScheme(
        name=name,
        kind=str(spec["kind"]),
        n_splits=int(spec["n_splits"]),
        n_repeats=int(spec.get("n_repeats", 1)),
        group_key=spec.get("group_key"),
        shuffle=bool(spec.get("shuffle", True)),
        random_state=int(spec.get("random_state", GLOBAL_SEED)),
    )


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def make_split_map(master: Any, task: str, scheme: SplitScheme | None = None) -> Any:
    """Fold assignments for one task (T20.1 -- T20.4).

    Returns one row per (repeat, record): the fold that record is *tested* in.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    from src.data_loader.master import task_frame

    if task not in TASK_SCHEMES:
        raise KeyError(
            "no scheme registered for task " + repr(task) + " -- expected one of: "
            + ", ".join(TASK_SCHEMES)
        )
    scheme = scheme or load_scheme(TASK_SCHEMES[task])

    frame = task_frame(master, task, supervised_only=True)
    if frame.empty:
        raise ValueError("task " + task + " has no supervised records to split")

    y = frame["y"].astype(int).to_numpy()
    # Grouping is not optional. Even PASCAL A, which has no patient IDs, gets a
    # group column: 175 groups over 176 records, the one shared group being the
    # recording Phase 17 found filed under two class labels. Splitting it
    # record-wise would put the same audio in train and test.
    groups = frame["split_group"].astype(str).to_numpy()
    if (groups == "").any():
        raise ValueError(
            "task " + task + " has supervised records with no split_group -- "
            "rule 3 cannot be enforced"
        )

    rows: list[dict[str, Any]] = []
    for repeat in range(scheme.n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=scheme.n_splits,
            shuffle=scheme.shuffle,
            # Every repeat must see a different shuffle, or five repeats produce
            # five identical fold maps and n=25 is a fiction.
            random_state=scheme.random_state + repeat,
        )
        assignment = np.full(len(frame), -1, dtype=int)
        for fold, (_, test_index) in enumerate(
            splitter.split(np.zeros(len(frame)), y, groups)
        ):
            assignment[test_index] = fold
        if (assignment < 0).any():
            raise ValueError(
                "repeat " + str(repeat) + " of " + task + " left "
                + str(int((assignment < 0).sum())) + " record(s) in no fold"
            )
        block = pd.DataFrame(
            {
                "task": task,
                "scheme": scheme.name,
                "dataset_source": frame["dataset_source"].to_numpy(),
                "n_splits": scheme.n_splits,
                "n_repeats": scheme.n_repeats,
                "repeat": repeat,
                "fold": assignment,
                "record_uid": frame["record_uid"].to_numpy(),
                "split_group": groups,
                "subject_id": frame["subject_id"].to_numpy(),
                "subject_derived": frame["subject_derived"].to_numpy(),
                "y": y,
                "class_name": _class_names(frame, task),
            }
        )
        rows.append(block)

    split_map = pd.concat(rows, ignore_index=True)
    return split_map[list(SPLIT_MAP_COLUMNS)]


def _class_names(frame: Any, task: str) -> Any:
    """The human-readable class for each row, from the task's own name column."""
    column = {
        "binary": "binary_label_name",
        "pascal_a": "multiclass_label_name",
        "pascal_b": "multiclass_label_name",
        "circor_murmur": "murmur_label_name",
        "circor_outcome": "outcome_label_name",
    }[task]
    return frame[column].astype(str).to_numpy()


def build_split_maps(master: Any, tasks: list[str] | None = None) -> Any:
    """All five task maps, concatenated into DA-07."""
    import pandas as pd

    tasks = tasks or list(TASK_SCHEMES)
    maps = [make_split_map(master, task) for task in tasks]
    split_map = pd.concat(maps, ignore_index=True)
    for task in tasks:
        scheme = load_scheme(TASK_SCHEMES[task])
        subset = split_map[split_map["task"] == task]
        log.info(
            "%s: %s -- %d folds over %d records, %d group(s)",
            task,
            scheme.name,
            scheme.total_folds,
            int(len(subset) / scheme.n_repeats),
            subset["split_group"].nunique(),
        )
    return split_map


# --------------------------------------------------------------------------
# consumption
# --------------------------------------------------------------------------


def iter_folds(
    split_map: Any, task: str
) -> Iterator[tuple[int, int, list[str], list[str]]]:
    """Yield ``(repeat, fold, train_uids, test_uids)`` for one task.

    The train side is everything in that repeat outside the test fold, which is
    exactly what the stored map means.
    """
    subset = split_map[split_map["task"] == task]
    if subset.empty:
        raise KeyError("task " + repr(task) + " is not in this split map")
    for repeat, block in subset.groupby("repeat", sort=True):
        uids = block["record_uid"].to_numpy()
        folds = block["fold"].to_numpy()
        for fold in sorted(set(folds.tolist())):
            test_mask = folds == fold
            yield (
                int(repeat),
                int(fold),
                uids[~test_mask].tolist(),
                uids[test_mask].tolist(),
            )


# --------------------------------------------------------------------------
# T20.6 / T20.7 -- the leakage checks
# --------------------------------------------------------------------------


def assert_no_leakage(split_map: Any, task: str | None = None) -> None:
    """Zero subject overlap between any two folds of any repeat (rule 3).

    Re-derived from the written map. A generator can be correct and the file it
    wrote still be wrong -- this checks the artifact every experiment loads.
    """
    tasks = [task] if task else sorted(set(split_map["task"]))
    for name in tasks:
        subset = split_map[split_map["task"] == name]
        for repeat, block in subset.groupby("repeat", sort=True):
            # A record must be assigned exactly once per repeat.
            if block["record_uid"].duplicated().any():
                raise ValueError(
                    name + " repeat " + str(repeat) + ": a record is assigned to "
                    "more than one fold"
                )
            per_group = block.groupby("split_group")["fold"].nunique()
            spanning = per_group[per_group > 1]
            if not spanning.empty:
                raise ValueError(
                    str(len(spanning)) + " subject(s) span more than one fold in "
                    + name + " repeat " + str(repeat) + " -- rule 3 violated. "
                    "First: " + str(spanning.index[0])
                )
            # And the same check stated the way the experiments will use it.
            folds = sorted(set(block["fold"].tolist()))
            groups_by_fold = {
                fold: set(block.loc[block["fold"] == fold, "split_group"])
                for fold in folds
            }
            for i, left in enumerate(folds):
                for right in folds[i + 1 :]:
                    shared = groups_by_fold[left] & groups_by_fold[right]
                    if shared:
                        raise ValueError(
                            name + " repeat " + str(repeat) + ": folds "
                            + str(left) + " and " + str(right) + " share "
                            + str(len(shared)) + " subject(s)"
                        )


def assert_validation_excluded(split_map: Any, master: Any) -> None:
    """The 301 PhysioNet ``validation/`` records appear in no fold (T20.5, T20.7)."""
    validation = set(master.loc[master["subset"] == "validation", "record_uid"])
    if not validation:
        raise ValueError("no validation records found in the master table")
    present = validation & set(split_map["record_uid"])
    if present:
        raise ValueError(
            str(len(present)) + " PhysioNet validation record(s) appear in a CV "
            "fold; they are byte-identical duplicates of training records and "
            "must enter no fold. First: " + sorted(present)[0]
        )
    # The twins they duplicate must still be in the binary map -- excluding the
    # copies must not have quietly excluded the originals too.
    twins = set(master.loc[master["subset"] == "validation", "duplicate_of"]) - {""}
    binary = set(split_map.loc[split_map["task"] == "binary", "record_uid"])
    missing = twins - binary
    if missing:
        raise ValueError(
            str(len(missing)) + " training record(s) duplicated by the validation "
            "folder are themselves missing from the binary fold map"
        )


def fold_summary(split_map: Any) -> Any:
    """Per (task, repeat, fold) class counts -- the table T20.7 reads."""
    summary = (
        split_map.groupby(["task", "scheme", "repeat", "fold"])
        .agg(
            n_records=("record_uid", "size"),
            n_groups=("split_group", "nunique"),
            n_classes=("y", "nunique"),
            min_class_count=("y", lambda s: int(s.value_counts().min())),
        )
        .reset_index()
    )
    return summary


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def _audit_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if out_dir is not None:
        return ensure_dir(out_dir)
    return ensure_dir(load_config("paths").require("outputs.dataset_audit"))


def split_map_path(out_dir: str | Path | None = None) -> Path:
    """DA-07 ``subject_split_map.csv``."""
    return _audit_dir(out_dir) / "subject_split_map.csv"


def write_split_map(split_map: Any, out_dir: str | Path | None = None) -> Path:
    from src.utils.io import save_csv

    target = save_csv(split_map, split_map_path(out_dir))
    log.info(
        "wrote %s (%d assignments across %d task(s))",
        target.name,
        len(split_map),
        split_map["task"].nunique(),
    )
    return target


def load_split_map(path: str | Path | None = None) -> Any:
    import pandas as pd

    target = Path(path) if path else split_map_path()
    if not target.is_file():
        raise FileNotFoundError(
            "DA-07 has not been generated yet -- run scripts/01_run_dataset_audit.py"
        )
    return pd.read_csv(target, keep_default_na=False)


def run_split_generation(
    master: Any | None = None, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Phase 20 end to end: build every task map, verify it, write DA-07."""
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv

    if master is None:
        from src.data_loader.master import load_master

        master = load_master()

    split_map = build_split_maps(master)
    assert_no_leakage(split_map)
    assert_validation_excluded(split_map, master)

    for task in TASK_SCHEMES:
        scheme = load_scheme(TASK_SCHEMES[task])
        block = split_map[split_map["task"] == task]
        observed = block.groupby("repeat")["fold"].nunique()
        if set(observed) != {scheme.n_splits}:
            raise ValueError(
                task + " produced " + str(sorted(set(observed))) + " fold(s) per "
                "repeat; the scheme asks for " + str(scheme.n_splits)
            )
        if block["repeat"].nunique() != scheme.n_repeats:
            raise ValueError(
                task + " produced " + str(block["repeat"].nunique()) + " repeat(s); "
                "the scheme asks for " + str(scheme.n_repeats)
            )

    path = write_split_map(split_map, out_dir)
    summary = fold_summary(split_map)
    summary_path = save_csv(summary, _audit_dir(out_dir) / "split_fold_summary.csv")

    register_evidence(
        evidence_id="DA-07",
        objective="Data audit",
        dataset="D1-D4",
        metric_or_asset="Subject-grouped cross-validation fold map (all five tasks)",
        filename=path,
        source_data="outputs/01_dataset_audit/metadata_master.csv",
        command="python scripts/01_run_dataset_audit.py",
    )
    return {
        "split_map": split_map,
        "summary": summary,
        "path": path,
        "summary_path": summary_path,
        "n_assignments": len(split_map),
    }


# Sanity: every task named in the constants module must have a scheme, or a
# later Part asks for a fold map that was never generated.
_missing_schemes = [task for task in TASKS if task not in TASK_SCHEMES]
if _missing_schemes:  # pragma: no cover -- import-time guard
    raise RuntimeError(
        "no CV scheme registered for: " + ", ".join(_missing_schemes)
    )
