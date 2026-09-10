"""EXP-G1 -- the PhysioNet diagnosis multiclass track (Phase 76).

Objective 6 asks for multiclass evidence. PASCAL A is the obvious candidate and
is not good enough: 124 records, 19 in one class, and every model's record-level
confidence interval overlaps every other's. PhysioNet carries a diagnosis
annotation on its abnormal recordings, and that is a far larger sample for the
same question -- which is why ``configs/experiments.yaml`` declares EXP-G1
supplementary rather than optional.

**Scope: the 665 abnormal records, not all 3,240.** The declared ``class_counts``
are the eight abnormal subtypes and sum to exactly the 665 abnormal records of
the binary track. Adding the 2,575 normals would make the largest class 39x the
smallest and turn this into the binary task with extra labels, answering a
question EXP-A1 already answers better.

**The merge policy (T76.2), stated once and applied by code:** a diagnosis class
is retained if it has at least :data:`MIN_CLASS_RECORDS` records; everything
below that threshold merges into a single ``Other pathologic`` class. The
threshold is 50, chosen as 5 folds x 10 records so that every retained class has
about ten test records per fold rather than two. On this corpus that keeps CAD,
MVP, Benign and Pathologic and merges MPC, AD, MR and AS.

``Other pathologic`` is a **statistical convenience, not a diagnosis.** It pools
four unrelated conditions because none of them is individually estimable at this
sample size. No per-class number for it may be read as a claim about any one of
the four, and :func:`merge_report` writes down exactly what went into it.

**Its own fold map, not DA-07's.** DA-07 holds the five primary label spaces and
is asserted against by the mega tests; appending a sixth task to it would change
counts those tests check. This module builds a subject-grouped 5-fold map for
the diagnosis track, runs it through the same leakage assertions, and writes it
beside DA-07 as its own artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "EXP_ID",
    "TASK",
    "SCHEME",
    "MIN_CLASS_RECORDS",
    "OTHER_CLASS",
    "diagnosis_records",
    "merge_report",
    "build_folds",
    "split_map_path",
    "write_split_map",
    "load_folds",
    "run_track",
]

log = get_logger("evaluation.diagnosis_track")

EXP_ID = "EXP-G1"
TASK = "diagnosis_multiclass"
SCHEME = "grouped_5fold"

#: 5 folds x 10 test records. Below this a per-class recall is two or three
#: records wide and its confidence interval spans most of [0, 1].
MIN_CLASS_RECORDS = 50

OTHER_CLASS = "Other pathologic"

N_SPLITS = 5
SEED = 42

# ---------------------------------------------------------------------------
# the within-source variant -- the fix for the confound
# ---------------------------------------------------------------------------

#: The one sub-collection that holds more than one diagnosis class on its own:
#: MVP 134, Benign 118, MPC 23 and AD 17, over 58 subjects. Restricting to it
#: makes the recording source CONSTANT, so the shortcut this track was found to
#: be taking is not merely discouraged, it is arithmetically unavailable. Every
#: other sub-collection holds exactly one class and cannot support a task.
WITHIN_SOURCE_SUBSET = "training-a"

#: A lower bar than :data:`MIN_CLASS_RECORDS` (5 folds x 5 rather than x 10),
#: and the deviation is deliberate. Applying 50 here would collapse the task to
#: Benign-versus-MVP, and a two-class task is much weaker evidence than a
#: three-class one. The cost is that `Other pathologic` carries ~8 test records
#: per fold and its interval is correspondingly wide -- which is reported, not
#: hidden. Stated here because a threshold that changes between runs without a
#: recorded reason is exactly the kind of drift rule 1 exists to stop.
WITHIN_SOURCE_MIN_RECORDS = 25


# ---------------------------------------------------------------------------
# the record set and the merge policy
# ---------------------------------------------------------------------------


def diagnosis_records(master: Any = None, *, within_source: bool = False) -> Any:
    """The abnormal D1 records with their merged diagnosis label.

    ``within_source=True`` returns the de-confounded variant: only
    :data:`WITHIN_SOURCE_SUBSET`, where every recording shares one collection so
    the source carries no information about the class. The global merge is
    applied first and the restriction second, so a class means the same thing in
    both variants.
    """
    import pandas as pd

    if master is None:
        from src.utils.evidence import PROJECT_ROOT

        master = pd.read_csv(
            PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv",
            low_memory=False,
        )

    rows = master[
        (master["dataset_source"] == "D1")
        & (master["use_in_supervised"].astype(bool))
        & (master["binary_label"] == 1)
        & (master["diagnosis_class"].notna())
    ].copy()

    counts = rows["diagnosis_class"].value_counts()
    retained = sorted(counts[counts >= MIN_CLASS_RECORDS].index.astype(str))
    rows["diagnosis_merged"] = [
        name if name in set(retained) else OTHER_CLASS
        for name in rows["diagnosis_class"].astype(str)
    ]

    if within_source:
        rows = rows[rows["subset"].astype(str) == WITHIN_SOURCE_SUBSET].copy()
        kept = rows["diagnosis_merged"].value_counts()
        usable = set(kept[kept >= WITHIN_SOURCE_MIN_RECORDS].index.astype(str))
        dropped = sorted(set(kept.index.astype(str)) - usable)
        if dropped:
            log.info("within-source: dropping %s (below %d records)", dropped,
                     WITHIN_SOURCE_MIN_RECORDS)
        rows = rows[rows["diagnosis_merged"].astype(str).isin(usable)].copy()

    # Label codes are assigned from the SORTED class names, so the mapping is a
    # property of the data and not of the row order the file happened to have.
    classes = sorted(rows["diagnosis_merged"].unique())
    codes = {name: index for index, name in enumerate(classes)}
    rows["y"] = [codes[name] for name in rows["diagnosis_merged"]]

    rows = rows.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    log.info(
        "diagnosis track: %d record(s), %d subject(s), %d class(es) after merge",
        len(rows),
        rows["split_group"].nunique(),
        len(classes),
    )
    return rows[
        [
            "record_uid",
            "subject_id",
            "split_group",
            "subset",
            "diagnosis_class",
            "diagnosis_merged",
            "y",
        ]
    ]


def merge_report(records: Any = None) -> Any:
    """What the policy kept, what it merged, and why (T76.2).

    Written as a deliverable rather than left in a docstring: a merge policy a
    reader cannot audit is indistinguishable from classes that were dropped.
    """
    import pandas as pd

    frame = diagnosis_records() if records is None else records
    original = frame["diagnosis_class"].value_counts()

    rows = []
    for name, count in original.items():
        merged = str(frame[frame["diagnosis_class"] == name]["diagnosis_merged"].iloc[0])
        rows.append(
            {
                "diagnosis_class": str(name),
                "n_records": int(count),
                "n_subjects": int(
                    frame[frame["diagnosis_class"] == name]["split_group"].nunique()
                ),
                "threshold": MIN_CLASS_RECORDS,
                "retained": bool(merged == str(name)),
                "merged_into": merged,
                "reason": (
                    "at or above the " + str(MIN_CLASS_RECORDS) + "-record threshold"
                    if merged == str(name)
                    else "below the " + str(MIN_CLASS_RECORDS) + "-record threshold "
                    "(5 folds x 10 test records); pooled because it is not "
                    "individually estimable at this sample size"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["retained", "n_records"], ascending=[False, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the fold map
# ---------------------------------------------------------------------------


def build_folds(records: Any = None, *, within_source: bool = False) -> Any:
    """A subject-grouped, class-stratified 5-fold map for the diagnosis track.

    ``StratifiedGroupKFold``: the classes are very unequal (287 down to 62), so
    an unstratified grouped split can and does produce a fold missing a class
    entirely, and a per-class recall with no support in a fold is not a number.
    Grouping is by ``split_group``, the same subject key every other track uses,
    so rule 3 holds here exactly as it does elsewhere.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    frame = (
        diagnosis_records(within_source=within_source) if records is None else records
    )
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    assignment = pd.Series(-1, index=frame.index, dtype=int)
    for fold, (_, test_index) in enumerate(
        splitter.split(frame, frame["y"], groups=frame["split_group"])
    ):
        assignment.iloc[test_index] = fold
    if (assignment < 0).any():
        raise RuntimeError("a record was not assigned to any fold")

    table = frame.copy()
    table.insert(0, "task", TASK + ("_within_source" if within_source else ""))
    table.insert(1, "scheme", SCHEME)
    table.insert(2, "dataset_source", "D1")
    table["repeat"] = 0
    table["fold"] = assignment.to_numpy()
    return table[
        [
            "task",
            "scheme",
            "dataset_source",
            "record_uid",
            "subject_id",
            "split_group",
            "subset",
            "diagnosis_class",
            "diagnosis_merged",
            "y",
            "repeat",
            "fold",
        ]
    ]


def split_map_path(
    out_dir: str | Path | None = None, *, within_source: bool = False
) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    root = (
        ensure_dir(out_dir)
        if out_dir is not None
        else ensure_dir(load_config("paths").require("outputs.dataset_audit"))
    )
    stem = "diagnosis_within_source_split_map" if within_source else "diagnosis_split_map"
    return root / (stem + ".csv")


def write_split_map(
    table: Any = None, out_dir: str | Path | None = None, *, within_source: bool = False
) -> Path:
    """Emit the diagnosis fold map, leakage-checked before it is written."""
    from src.utils.io import save_csv

    folds = build_folds(within_source=within_source) if table is None else table
    assert_no_leakage(folds)
    path = save_csv(folds, split_map_path(out_dir, within_source=within_source))
    log.info(
        "diagnosis fold map: %d record(s) over %d fold(s) -> %s",
        len(folds),
        folds["fold"].nunique(),
        path.name,
    )
    return path


def assert_no_leakage(folds: Any) -> None:
    """No subject on both sides of any fold, and no record assigned twice.

    Re-derived from the table rather than trusted from the splitter: the
    artifact every run loads is the thing that has to be right.
    """
    if folds["record_uid"].duplicated().any():
        raise ValueError("a record is assigned to more than one fold")
    for fold in sorted(folds["fold"].unique()):
        test = set(folds[folds["fold"] == fold]["split_group"].astype(str))
        train = set(folds[folds["fold"] != fold]["split_group"].astype(str))
        overlap = test & train
        if overlap:
            raise ValueError(
                "fold " + str(fold) + ": " + str(len(overlap))
                + " subject(s) on both sides, e.g. " + ", ".join(sorted(overlap)[:5])
            )


def load_folds(
    out_dir: str | Path | None = None, *, within_source: bool = False
) -> tuple[Any, ...]:
    """The stored map as ``cv.Fold`` objects, so the shared runner can use it."""
    import pandas as pd

    from src.evaluation.cv import Fold

    path = split_map_path(out_dir, within_source=within_source)
    if not path.is_file():
        raise FileNotFoundError(str(path) + "; run write_split_map() first")
    table = pd.read_csv(path)
    assert_no_leakage(table)

    uids = table["record_uid"].astype(str).to_numpy()
    groups = table["split_group"].astype(str).to_numpy()
    assignment = table["fold"].to_numpy()

    folds = []
    for fold in sorted(set(assignment.tolist())):
        test_mask = assignment == fold
        folds.append(
            Fold(
                task=str(table["task"].iloc[0]),
                scheme=SCHEME,
                repeat=0,
                fold=int(fold),
                train_uids=tuple(uids[~test_mask]),
                test_uids=tuple(uids[test_mask]),
                train_groups=tuple(groups[~test_mask]),
                test_groups=tuple(groups[test_mask]),
            )
        )
    return tuple(folds)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def source_confound_baseline(
    out_dir: str | Path | None = None, *, within_source: bool = False
) -> Any:
    """Macro-F1 of predicting the class from the RECORDING SOURCE alone.

    PhysioNet's diagnosis labels are nearly nested inside its six
    sub-collections: Benign and MVP are 100% ``training-a``, CAD is only in
    ``training-b`` and ``training-e``, Pathologic only in ``training-d`` and
    ``training-f``. Combined with the already-recorded finding that the six
    sub-collections behave like six different datasets, a model can score on
    this task by recognising the recording setup rather than the pathology.

    This is the baseline that measures exactly that: a lookup from sub-collection
    to its most common class, **fitted inside each training fold** and applied to
    the test fold, using no audio at all. It is the direct analogue of the
    "always predict normal" baseline that PASCAL B results must be quoted
    against.

    No EXP-G1 number may be reported without this beside it.
    """
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, f1_score

    records = diagnosis_records(within_source=within_source).copy()

    position = {uid: index for index, uid in enumerate(records["record_uid"].astype(str))}
    y = records["y"].to_numpy()
    labels = sorted(set(y.tolist()))

    rows = []
    for fold in load_folds(out_dir, within_source=within_source):
        train = [position[u] for u in fold.train_uids]
        test = [position[u] for u in fold.test_uids]
        block = records.iloc[train]
        lookup = (
            block.groupby("subset")["y"]
            .agg(lambda s: s.value_counts().idxmax())
            .to_dict()
        )
        fallback = block["y"].value_counts().idxmax()
        predicted = [lookup.get(s, fallback) for s in records.iloc[test]["subset"]]
        rows.append(
            {
                "fold": fold.fold,
                "n_test": len(test),
                "macro_f1": float(
                    f1_score(y[test], predicted, average="macro", labels=labels,
                             zero_division=0)
                ),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
            }
        )

    frame = pd.DataFrame(rows)
    log.warning(
        "SOURCE-ONLY baseline: macro-F1 %.4f, balanced accuracy %.4f -- using no "
        "audio whatsoever. Any model at or below this has not demonstrated "
        "audio-based diagnosis classification.",
        float(frame["macro_f1"].mean()),
        float(frame["balanced_accuracy"].mean()),
    )
    return frame


def class_by_source_table() -> Any:
    """The confound itself, as a table: class against sub-collection."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    records = diagnosis_records()
    master = pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv",
        low_memory=False,
    )
    subset_of = dict(zip(master["record_uid"].astype(str), master["subset"].astype(str)))
    records = records.copy()
    records["subset"] = [subset_of[u] for u in records["record_uid"].astype(str)]

    table = pd.crosstab(records["diagnosis_merged"], records["subset"])
    table["n_records"] = table.sum(axis=1)
    table["n_subsets_present"] = (table.drop(columns=["n_records"]) > 0).sum(axis=1)
    table["largest_subset_share"] = (
        table.drop(columns=["n_records", "n_subsets_present"]).max(axis=1)
        / table["n_records"]
    )
    return table.reset_index()


def _models() -> tuple[str, ...]:
    from src.utils.config import load_config

    spec = load_config("experiments").require("experiments")[EXP_ID]
    return tuple(str(m) for m in spec.get("models", ()))


def run_track(
    *,
    models: tuple[str, ...] | None = None,
    out_dir: str | Path | None = None,
    within_source: bool = False,
) -> dict[str, Any]:
    """Fit every declared model over the diagnosis fold map.

    Class weighting is ``balanced`` as EXP-G1 declares, applied inside each
    training fold. On a 287-versus-62 split an unweighted fit predicts the
    majority class and reports a macro-F1 that looks like a result.
    """
    import time

    import numpy as np
    import pandas as pd

    from src.evaluation import cv as cv_module
    from src.evaluation import metrics as mt
    from src.feature_extraction.matrix import load_matrix
    from src.feature_extraction.registry import feature_names
    from src.models import estimators as est
    from src.models import pipeline as pl

    chosen = tuple(models) if models else _models()
    records = diagnosis_records(within_source=within_source)
    names = list(feature_names())

    matrix = load_matrix()
    matrix = matrix[matrix["record_uid"].astype(str).isin(set(records["record_uid"]))]
    matrix = matrix.sort_values("record_uid", kind="mergesort").reset_index(drop=True)
    joined = records.merge(
        matrix[["record_uid", *names]], on="record_uid", how="inner", validate="1:1"
    )
    if len(joined) != len(records):
        raise RuntimeError(
            str(len(records) - len(joined)) + " diagnosis record(s) have no FE-03 row"
        )

    X = joined[names].to_numpy(dtype=float)
    y = joined["y"].to_numpy(dtype=int)
    groups = joined["split_group"].astype(str).to_numpy()
    classes = (
        joined[["y", "diagnosis_merged"]]
        .drop_duplicates()
        .sort_values("y")["diagnosis_merged"]
        .tolist()
    )
    folds = cv_module.resolve_folds(
        load_folds(out_dir, within_source=within_source),
        tuple(joined["record_uid"].astype(str)),
    )

    per_fold: list[dict[str, Any]] = []
    predictions: list[Any] = []
    for model_id in chosen:
        def factory(model_id: str = model_id) -> Any:
            return pl.build_pipeline(
                est.build_estimator(model_id), y=y, n_features=X.shape[1]
            )

        started = time.perf_counter()
        result = cv_module.run_cv(factory, X, y, groups, folds, task=TASK)
        elapsed = time.perf_counter() - started

        for fold_result in result.folds:
            # class_names is passed so the per-class keys read `recall_CAD`
            # rather than `recall_0`; a per-class column nobody can map back to
            # a diagnosis is not reportable evidence.
            scores = mt.multiclass_metrics(
                fold_result.y_true,
                fold_result.y_pred,
                fold_result.y_proba,
                labels=list(range(len(classes))),
                class_names=classes,
            )
            row: dict[str, Any] = {
                "exp_id": EXP_ID + ("-within_source" if within_source else ""),
                "model_id": model_id,
                "task": TASK + ("_within_source" if within_source else ""),
                "repeat": fold_result.repeat,
                "fold": fold_result.fold,
                "n_train": fold_result.n_train,
                "n_test": len(fold_result.test_uids),
            }
            row.update(scores)
            per_fold.append(row)

        oof = result.oof_frame()
        oof.insert(0, "model_id", model_id)
        predictions.append(oof)
        log.info(
            "%s: macro-F1 %.4f over %d fold(s) in %.1f s",
            model_id,
            float(np.mean([r["macro_f1"] for r in per_fold if r["model_id"] == model_id])),
            len(folds),
            elapsed,
        )

    return {
        "per_fold": pd.DataFrame(per_fold),
        "predictions": pd.concat(predictions, ignore_index=True),
        "classes": tuple(classes),
        "records": joined,
        "n_records": len(joined),
        "n_subjects": len(set(groups.tolist())),
        "within_source": bool(within_source),
    }
