"""Cross-validation driven by the DA-07 split maps (Phase 43).

**Folds are loaded, never re-derived.** ``outputs/01_dataset_audit/
subject_split_map.csv`` already records, for every task, exactly which record
sits in which fold of which repeat, together with the group it belongs to. This
module reads that file. It does not call ``StratifiedGroupKFold`` at runtime,
and that is a deliberate constraint rather than a convenience.

Re-deriving folds would make every result depend on things nobody records in a
paper: the scikit-learn version, the row order the matrix happened to arrive in,
the dtype of the group key. Two runs months apart would disagree, the audit's
published fold sizes would stop matching the folds the models actually used, and
nothing would fail loudly enough to notice. Loading a materialised map makes the
fold assignment a fixed, inspectable input -- diffable, and identical on any
machine.

Three invariants are enforced at runtime, in every fold, on every call:

* train and test **groups** are disjoint (research rule 3 -- no subject leakage);
* train and test **rows** are disjoint;
* a **fresh estimator** is built per fold, so nothing fitted on fold *k* can
  survive into fold *k+1*.

Out-of-fold predictions are assembled **per repeat**. Within one repeat every
record is tested exactly once; across the five repeats of the primary track a
record therefore has five out-of-fold predictions, not one. Collapsing them into
a single OOF vector would silently average five different models' opinions and
destroy the pairing that the repeated-CV statistics depend on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "SPLIT_MAP_FILENAME",
    "SUMMARY_FILENAME",
    "Fold",
    "FoldResult",
    "CVResult",
    "LeakageError",
    "split_map_path",
    "load_split_map",
    "available_tasks",
    "scheme_for",
    "load_folds",
    "resolve_folds",
    "assert_group_disjoint",
    "run_cv",
    "save_cv_result",
    "load_oof_predictions",
]

log = get_logger("evaluation.cv")

SPLIT_MAP_FILENAME = "subject_split_map.csv"
SUMMARY_FILENAME = "split_fold_summary.csv"


class LeakageError(AssertionError):
    """A fold violated an invariant that makes its results meaningless.

    Deliberately an ``AssertionError`` subclass: this is never a condition to
    catch and continue from. If a subject appears in both train and test, every
    number the run produces is inflated and must not be recorded.
    """


@dataclass(frozen=True, eq=False)
class Fold:
    """One train/test split, as recorded by the audit.

    ``train_index``/``test_index`` are positional and are ``None`` until
    :func:`resolve_folds` binds the fold to a specific row ordering. Keeping the
    uid lists as the primary representation means a fold is meaningful on its
    own -- it does not silently depend on whatever order a matrix was loaded in.
    """

    task: str
    scheme: str
    repeat: int
    fold: int
    train_uids: tuple[str, ...]
    test_uids: tuple[str, ...]
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    train_index: np.ndarray | None = None
    test_index: np.ndarray | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.repeat, self.fold)

    @property
    def label(self) -> str:
        return "r" + str(self.repeat) + "f" + str(self.fold)

    @property
    def n_train(self) -> int:
        return len(self.train_uids)

    @property
    def n_test(self) -> int:
        return len(self.test_uids)

    @property
    def is_resolved(self) -> bool:
        return self.train_index is not None and self.test_index is not None


@dataclass
class FoldResult:
    """Predictions from one fold. ``y_proba`` is ``None`` for a hard classifier."""

    repeat: int
    fold: int
    test_uids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    classes: tuple[Any, ...]
    n_train: int
    fit_seconds: float
    predict_seconds: float

    @property
    def label(self) -> str:
        return "r" + str(self.repeat) + "f" + str(self.fold)


@dataclass
class CVResult:
    """Every fold of one CV run, plus the per-repeat out-of-fold predictions."""

    task: str
    scheme: str
    folds: list[FoldResult] = field(default_factory=list)
    classes: tuple[Any, ...] = ()
    n_records: int = 0
    total_seconds: float = 0.0

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def repeats(self) -> tuple[int, ...]:
        return tuple(sorted({result.repeat for result in self.folds}))

    def oof_frame(self) -> Any:
        """One row per (repeat, record): the prediction made while it was held out."""
        import pandas as pd

        rows = []
        for result in self.folds:
            for position, uid in enumerate(result.test_uids):
                row: dict[str, Any] = {
                    "repeat": result.repeat,
                    "fold": result.fold,
                    "record_uid": uid,
                    "y_true": result.y_true[position],
                    "y_pred": result.y_pred[position],
                }
                if result.y_proba is not None:
                    for index, klass in enumerate(result.classes):
                        row["proba_" + str(klass)] = result.y_proba[position, index]
                rows.append(row)
        return pd.DataFrame(rows)

    def fold_frame(self) -> Any:
        """Per-fold bookkeeping: sizes and timings, for the complexity tables."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "repeat": result.repeat,
                    "fold": result.fold,
                    "n_train": result.n_train,
                    "n_test": len(result.test_uids),
                    "fit_seconds": round(result.fit_seconds, 6),
                    "predict_seconds": round(result.predict_seconds, 6),
                }
                for result in self.folds
            ]
        )


# ---------------------------------------------------------------------------
# loading the map (T43.1)
# ---------------------------------------------------------------------------


def split_map_path() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.dataset_audit")) / SPLIT_MAP_FILENAME


def load_split_map(task: str | None = None) -> Any:
    """Read DA-07. Raises if the file is absent -- never re-derives folds."""
    import pandas as pd

    path = split_map_path()
    if not path.is_file():
        raise FileNotFoundError(
            "DA-07 split map not found at "
            + str(path)
            + "; folds are loaded from the audit, never re-derived at runtime"
        )

    frame = pd.read_csv(path, dtype={"record_uid": str, "split_group": str})
    if task is None:
        return frame

    subset = frame[frame["task"].astype(str) == task]
    if subset.empty:
        raise ValueError(
            "no folds for task "
            + task
            + "; available: "
            + ", ".join(sorted(frame["task"].astype(str).unique()))
        )
    return subset


def available_tasks() -> tuple[str, ...]:
    return tuple(sorted(load_split_map()["task"].astype(str).unique()))


def scheme_for(task: str) -> str:
    """The single CV scheme DA-07 recorded for this task."""
    schemes = sorted(load_split_map(task)["scheme"].astype(str).unique())
    if len(schemes) != 1:
        raise ValueError(
            "task " + task + " has " + str(len(schemes)) + " schemes: " + str(schemes)
        )
    return schemes[0]


def load_folds(task: str, *, validate: bool = True) -> tuple[Fold, ...]:
    """Every (repeat, fold) of one task, as train/test uid lists.

    The map stores each record's *test* fold. Train is therefore everything else
    **within the same repeat** -- taking the complement across repeats would put
    a record in its own training set.
    """
    frame = load_split_map(task)
    scheme = scheme_for(task)

    folds: list[Fold] = []
    for repeat, repeat_rows in frame.groupby("repeat", sort=True):
        uids = repeat_rows["record_uid"].astype(str).to_numpy()
        groups = repeat_rows["split_group"].astype(str).to_numpy()
        assignment = repeat_rows["fold"].to_numpy()

        if len(set(uids)) != len(uids):
            raise ValueError(
                "task " + task + " repeat " + str(repeat) + " lists a record twice"
            )

        for fold in sorted(set(assignment.tolist())):
            test_mask = assignment == fold
            folds.append(
                Fold(
                    task=task,
                    scheme=scheme,
                    repeat=int(repeat),
                    fold=int(fold),
                    train_uids=tuple(uids[~test_mask]),
                    test_uids=tuple(uids[test_mask]),
                    train_groups=tuple(groups[~test_mask]),
                    test_groups=tuple(groups[test_mask]),
                )
            )

    if validate:
        for fold in folds:
            assert_group_disjoint(fold)
        _validate_against_config(task, scheme, folds)
    return tuple(folds)


def _validate_against_config(task: str, scheme: str, folds: Sequence[Fold]) -> None:
    """The map must agree with ``configs/experiments.yaml`` on fold counts.

    If the config says 5x5 and the map holds 5x4, one of the two is wrong and
    the run must stop -- not quietly use whichever it found.
    """
    from src.utils.config import load_config

    settings = load_config("experiments").get("cv_schemes." + scheme)
    if not settings:
        # Not a warning. experiments.yaml is the declaration of which schemes
        # exist; a map naming one it has never heard of means the two have
        # drifted apart, and continuing would run folds nothing describes.
        raise ValueError(
            "task "
            + task
            + " uses scheme "
            + scheme
            + ", which experiments.yaml does not declare"
        )

    expected_total = settings.get("total_folds")
    if expected_total is not None and len(folds) != int(expected_total):
        raise ValueError(
            "task "
            + task
            + " scheme "
            + scheme
            + ": DA-07 holds "
            + str(len(folds))
            + " folds, config declares "
            + str(expected_total)
        )

    expected_splits = settings.get("n_splits")
    per_repeat = len({fold.fold for fold in folds})
    if expected_splits is not None and per_repeat != int(expected_splits):
        raise ValueError(
            "task "
            + task
            + ": DA-07 holds "
            + str(per_repeat)
            + " folds per repeat, config declares "
            + str(expected_splits)
        )


# ---------------------------------------------------------------------------
# T43.5 -- the invariant
# ---------------------------------------------------------------------------


def assert_group_disjoint(fold: Fold) -> None:
    """No group may appear on both sides of a split (research rule 3)."""
    shared = set(fold.train_groups) & set(fold.test_groups)
    if shared:
        raise LeakageError(
            fold.task
            + " "
            + fold.label
            + ": "
            + str(len(shared))
            + " group(s) in both train and test, e.g. "
            + ", ".join(sorted(shared)[:5])
        )

    overlap = set(fold.train_uids) & set(fold.test_uids)
    if overlap:
        raise LeakageError(
            fold.task
            + " "
            + fold.label
            + ": "
            + str(len(overlap))
            + " record(s) in both train and test, e.g. "
            + ", ".join(sorted(overlap)[:5])
        )


def resolve_folds(
    folds: Iterable[Fold], record_uids: Sequence[str], *, require_all: bool = True
) -> tuple[Fold, ...]:
    """Bind uid-based folds to positional indices into a specific row ordering.

    ``require_all`` refuses a matrix that is missing records the folds name. A
    fold silently shrinking because three of its records were absent is the kind
    of difference that changes a metric by a point and leaves no trace.
    """
    import dataclasses

    position = {str(uid): index for index, uid in enumerate(record_uids)}
    if len(position) != len(record_uids):
        raise ValueError("record_uids contains duplicates")

    resolved = []
    for fold in folds:
        missing = [uid for uid in (*fold.train_uids, *fold.test_uids) if uid not in position]
        if missing and require_all:
            raise ValueError(
                fold.task
                + " "
                + fold.label
                + ": "
                + str(len(missing))
                + " record(s) named by the fold are absent from the matrix, e.g. "
                + ", ".join(missing[:5])
            )
        train = np.array(
            [position[uid] for uid in fold.train_uids if uid in position], dtype=int
        )
        test = np.array(
            [position[uid] for uid in fold.test_uids if uid in position], dtype=int
        )
        resolved.append(dataclasses.replace(fold, train_index=train, test_index=test))
    return tuple(resolved)


# ---------------------------------------------------------------------------
# T43.2 -- the driver
# ---------------------------------------------------------------------------


def run_cv(
    estimator_factory: Callable[[], Any],
    X: Any,
    y: Any,
    groups: Any,
    folds: Sequence[Fold],
    *,
    task: str = "",
    proba: bool = True,
) -> CVResult:
    """Fit ``estimator_factory()`` on each fold's train rows and predict its test rows.

    ``estimator_factory`` is a callable, not an estimator, on purpose: a single
    estimator instance passed in here would be refitted 25 times and any state
    it carries -- a fitted scaler, a warm start, a random state advanced by the
    previous fold -- would cross fold boundaries. Building a fresh one per fold
    makes that impossible rather than merely unlikely.
    """
    if not folds:
        raise ValueError("no folds supplied")
    if not all(fold.is_resolved for fold in folds):
        raise ValueError("folds must be resolved to positional indices first")

    features = np.asarray(X)
    targets = np.asarray(y)
    group_values = np.asarray(groups, dtype=object)
    if not (len(features) == len(targets) == len(group_values)):
        raise ValueError(
            "X, y and groups disagree on length: "
            + str((len(features), len(targets), len(group_values)))
        )

    result = CVResult(
        task=task or folds[0].task,
        scheme=folds[0].scheme,
        n_records=len(features),
    )
    started = time.perf_counter()

    for fold in folds:
        train_index = np.asarray(fold.train_index, dtype=int)
        test_index = np.asarray(fold.test_index, dtype=int)
        _assert_fold_is_safe(fold, train_index, test_index, group_values)

        estimator = estimator_factory()
        fit_start = time.perf_counter()
        estimator.fit(features[train_index], targets[train_index])
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        y_pred = np.asarray(estimator.predict(features[test_index]))
        y_proba = None
        if proba and hasattr(estimator, "predict_proba"):
            y_proba = np.asarray(estimator.predict_proba(features[test_index]))
        predict_seconds = time.perf_counter() - predict_start

        classes = tuple(getattr(estimator, "classes_", np.unique(targets)).tolist())
        if result.classes and classes != result.classes:
            raise ValueError(
                "fold "
                + fold.label
                + " produced class ordering "
                + str(classes)
                + " but an earlier fold produced "
                + str(result.classes)
                + "; probability columns would not line up"
            )
        result.classes = classes

        result.folds.append(
            FoldResult(
                repeat=fold.repeat,
                fold=fold.fold,
                test_uids=fold.test_uids,
                y_true=targets[test_index],
                y_pred=y_pred,
                y_proba=y_proba,
                classes=classes,
                n_train=len(train_index),
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
            )
        )

    result.total_seconds = time.perf_counter() - started
    _assert_oof_coverage(result, folds)
    log.info(
        "%s: %d folds, %d records, %.2f s",
        result.task,
        result.n_folds,
        result.n_records,
        result.total_seconds,
    )
    return result


def _assert_fold_is_safe(
    fold: Fold, train_index: np.ndarray, test_index: np.ndarray, groups: np.ndarray
) -> None:
    """T43.5 -- checked per fold, at run time, against the arrays actually used.

    ``assert_group_disjoint`` checks the *map*. This checks the **rows being fed
    to the estimator**, which is what a leak would actually travel through: a
    correct map resolved against a mis-ordered matrix still trains on the wrong
    rows, and the map-level check would pass happily.
    """
    if train_index.size == 0 or test_index.size == 0:
        raise LeakageError(fold.label + ": an empty train or test split")

    shared_rows = np.intersect1d(train_index, test_index)
    if shared_rows.size:
        raise LeakageError(
            fold.label
            + ": "
            + str(shared_rows.size)
            + " row(s) appear in both train and test"
        )

    shared_groups = set(groups[train_index].tolist()) & set(groups[test_index].tolist())
    if shared_groups:
        raise LeakageError(
            fold.label
            + ": "
            + str(len(shared_groups))
            + " group(s) appear in both train and test, e.g. "
            + ", ".join(sorted(str(item) for item in shared_groups)[:5])
        )


def _assert_oof_coverage(result: CVResult, folds: Sequence[Fold]) -> None:
    """Within each repeat, every record is tested exactly once (T43.7)."""
    from collections import Counter

    by_repeat: dict[int, Counter[str]] = {}
    for fold_result in result.folds:
        counter = by_repeat.setdefault(fold_result.repeat, Counter())
        counter.update(fold_result.test_uids)

    expected: dict[int, set[str]] = {}
    for fold in folds:
        expected.setdefault(fold.repeat, set()).update(fold.train_uids)
        expected[fold.repeat].update(fold.test_uids)

    for repeat, counter in by_repeat.items():
        repeated = [uid for uid, count in counter.items() if count > 1]
        if repeated:
            raise LeakageError(
                "repeat "
                + str(repeat)
                + ": "
                + str(len(repeated))
                + " record(s) tested more than once, e.g. "
                + ", ".join(sorted(repeated)[:5])
            )
        untested = expected[repeat] - set(counter)
        if untested:
            raise LeakageError(
                "repeat "
                + str(repeat)
                + ": "
                + str(len(untested))
                + " record(s) never appear in any test fold, e.g. "
                + ", ".join(sorted(untested)[:5])
            )


# ---------------------------------------------------------------------------
# T43.3 -- persistence
# ---------------------------------------------------------------------------


def save_cv_result(
    result: CVResult, directory: str | Path, *, folds: Sequence[Fold] | None = None
) -> dict[str, Path]:
    """Persist out-of-fold predictions and per-fold membership.

    The membership table is written as well as the predictions because the
    paired statistical tests in Part VIII compare two models fold by fold. That
    pairing is only valid if both models saw identical folds, and the only way
    to demonstrate it afterwards is to have written down which records were in
    which fold at the time.
    """
    import pandas as pd

    from src.utils.io import ensure_dir

    target = ensure_dir(directory)
    written: dict[str, Path] = {}

    oof_path = target / "oof_predictions.parquet"
    result.oof_frame().to_parquet(oof_path, index=False)
    written["oof"] = oof_path

    fold_path = target / "fold_timings.csv"
    result.fold_frame().to_csv(fold_path, index=False)
    written["timings"] = fold_path

    if folds is not None:
        rows = []
        for fold in folds:
            for uid in fold.train_uids:
                rows.append(
                    {
                        "repeat": fold.repeat,
                        "fold": fold.fold,
                        "record_uid": uid,
                        "split": "train",
                    }
                )
            for uid in fold.test_uids:
                rows.append(
                    {
                        "repeat": fold.repeat,
                        "fold": fold.fold,
                        "record_uid": uid,
                        "split": "test",
                    }
                )
        membership = target / "fold_membership.parquet"
        pd.DataFrame(rows).to_parquet(membership, index=False)
        written["membership"] = membership

    for key, path in written.items():
        log.info("%s -> %s", key, path)
    return written


def load_oof_predictions(directory: str | Path) -> Any:
    import pandas as pd

    path = Path(directory) / "oof_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError("no out-of-fold predictions at " + str(path))
    return pd.read_parquet(path)
