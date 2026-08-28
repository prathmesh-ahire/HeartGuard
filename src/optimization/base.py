"""The common search interface every Part VI optimizer is built on (Phase 54).

Four different search methods are planned -- randomized (SO-01), Bayesian
(SO-02), genetic (SO-03a) and swarm (SO-03b) -- and the only way their results
can be compared is if they differ *only* in how they propose the next point.
Everything else has to be shared: the same fold map, the same inner splits, the
same objective, the same budget accounting, the same trial log. That is what
this module is. A method subclasses :class:`BaseSearch` and implements one
method, ``_propose``; it never touches the data, the folds or the scoring.

Three constraints are enforced here rather than left to each optimizer:

**The outer test fold is never seen.** Outer folds are loaded from DA-07 exactly
as the Phase 43 driver loads them. Inner folds are cut from the outer fold's
*training rows only*, and :class:`RowLedger` records every row index any trial
fitted or scored on, so the claim "the search never touched the test fold" is a
measurement (``tests/test_search_no_leakage.py``) rather than an assertion in a
docstring.

**Inner folds are derived, not loaded -- and this is the one place that is
allowed.** DA-07 materialises the outer folds only; an inner split of an outer
training set does not exist in it. They are cut with ``StratifiedGroupKFold`` on
the training rows under the global seed, so they are deterministic given the
fold map and the matrix row order, and :func:`inner_folds` refuses to build them
from anything but a resolved outer fold. They are also written out beside the
trial log so a reader can see them.

**A budget stops a search; a search never stops itself.** Both a trial count and
a wall-clock ceiling are checked *between* trials, so an exhausted budget ends a
search after a completed trial rather than in the middle of one. A partial
result whose last trial was killed halfway is not a cheaper result -- it is an
unreadable one. The stop reason is recorded next to the trials.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "TRIALS_FILENAME",
    "BEST_PARAMS_FILENAME",
    "INNER_FOLD_FILENAME",
    "DEFAULT_INNER_SPLITS",
    "SearchError",
    "Budget",
    "Objective",
    "InnerFold",
    "RowLedger",
    "Trial",
    "SearchResult",
    "BaseSearch",
    "objective_for",
    "inner_folds",
    "score_params",
    "search_dir",
    "write_trials",
    "write_best_params",
    "write_inner_folds",
]

log = get_logger("optimization.base")

TRIALS_FILENAME = "trials.csv"
BEST_PARAMS_FILENAME = "best_parameters.json"
INNER_FOLD_FILENAME = "inner_fold_map.csv"

#: Inner splits per outer fold. Three, not five: the inner CV is refitted once
#: per trial, so its cost multiplies the whole search. Three keeps the primary
#: track's inner training sets above 1,700 records, which is where the fit cost
#: measured in Phase 47 was taken.
DEFAULT_INNER_SPLITS = 3


class SearchError(RuntimeError):
    """The search cannot be set up, or was asked for something meaningless."""


# ---------------------------------------------------------------------------
# T54.5 -- the budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """How much a single search is allowed to spend.

    Both limits are checked between trials. ``max_seconds`` is a *wall-clock*
    ceiling on the search, not on a trial: a search that has 4 seconds left does
    not start a 16-second trial, because a trial that is cut off produces no
    score and would still have to be logged as something.
    """

    max_trials: int = 50
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_trials < 1:
            raise SearchError("max_trials must be at least 1, got " + str(self.max_trials))
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise SearchError("max_seconds must be positive, got " + str(self.max_seconds))

    def stop_reason(
        self, n_done: int, elapsed: float, *, next_trial_estimate: float = 0.0
    ) -> str | None:
        """Why the search should stop now, or ``None`` to keep going."""
        if n_done >= self.max_trials:
            return "trial budget exhausted (" + str(self.max_trials) + " trials)"
        if self.max_seconds is None:
            return None
        if elapsed >= self.max_seconds:
            return (
                "wall-clock budget exhausted ("
                + str(round(elapsed, 1))
                + " s of "
                + str(self.max_seconds)
                + " s) after "
                + str(n_done)
                + " trial(s)"
            )
        if next_trial_estimate > 0 and elapsed + next_trial_estimate > self.max_seconds:
            return (
                "stopped early after "
                + str(n_done)
                + " trial(s): the next trial is estimated at "
                + str(round(next_trial_estimate, 1))
                + " s and only "
                + str(round(self.max_seconds - elapsed, 1))
                + " s of budget remain"
            )
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"max_trials": self.max_trials, "max_seconds": self.max_seconds}


# ---------------------------------------------------------------------------
# T54.2 -- the objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Objective:
    """What a trial is scored on. Higher is always better.

    The binary objectives are the linear sensitivity/specificity combinations
    already defined for the ensemble's threshold search
    (``src.ensemble.soft_voting.OBJECTIVE_COEFFICIENTS``) rather than a second
    definition of the same thing here -- a search that optimised one notion of
    "balanced accuracy plus sensitivity" while the ensemble optimised another
    would produce two numbers with one name.

    Multiclass is macro-F1: it weights a 19-record class the same as a 320-record
    one, which is the whole point on PASCAL A and B.
    """

    name: str
    kind: str  # "binary" | "multiclass"

    def __post_init__(self) -> None:
        if self.kind not in {"binary", "multiclass"}:
            raise SearchError("objective kind must be binary or multiclass, got " + self.kind)
        if self.kind == "binary":
            from src.ensemble.soft_voting import OBJECTIVES

            if self.name not in OBJECTIVES:
                raise SearchError(
                    "unknown binary objective "
                    + repr(self.name)
                    + "; known: "
                    + ", ".join(sorted(OBJECTIVES))
                )
        elif self.name != "macro_f1":
            raise SearchError(
                "the only multiclass objective is macro_f1, got " + repr(self.name)
            )

    def __call__(self, y_true: Any, y_pred: Any, labels: Sequence[Any]) -> float:
        from src.evaluation import metrics as mt

        if self.kind == "binary":
            from src.ensemble.soft_voting import OBJECTIVES

            scored = mt.binary_metrics(y_true, y_pred, labels=list(labels), positive_label=1)
            return float(OBJECTIVES[self.name](scored["sensitivity"], scored["specificity"]))
        scored = mt.multiclass_metrics(y_true, y_pred, labels=list(labels))
        return float(scored["macro_f1"])

    def components(self, y_true: Any, y_pred: Any, labels: Sequence[Any]) -> dict[str, float]:
        """The reportable metrics behind the score, for the trial log.

        A trials.csv holding only a single objective column cannot answer "did
        that point buy sensitivity or specificity?", which is the first question
        anyone asks of a search on an imbalanced task.
        """
        from src.evaluation import metrics as mt

        keep: tuple[str, ...]
        if self.kind == "binary":
            scored = mt.binary_metrics(y_true, y_pred, labels=list(labels), positive_label=1)
            keep = ("sensitivity", "specificity", "balanced_accuracy", "f1", "accuracy")
        else:
            scored = mt.multiclass_metrics(y_true, y_pred, labels=list(labels))
            keep = ("macro_f1", "balanced_accuracy", "accuracy")
        return {key: float(scored[key]) for key in keep if key in scored}


def objective_for(task: str, *, name: str | None = None) -> Objective:
    """The configured objective for ``task`` (``configs/experiments.yaml``)."""
    from src.utils.config import load_config

    kind = "binary" if task == "binary" else "multiclass"
    if name is not None:
        return Objective(name=name, kind=kind)
    scoring = load_config("experiments").get("defaults.scoring") or {}
    configured = scoring.get(kind)
    if not configured:
        raise SearchError("configs/experiments.yaml declares no defaults.scoring." + kind)
    return Objective(name=str(configured), kind=kind)


# ---------------------------------------------------------------------------
# T54.3 -- nested cross-validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InnerFold:
    """One inner split, in **positions into the full matrix**, not into the fold.

    Holding global positions rather than positions within the outer training
    block is what makes :class:`RowLedger`'s check meaningful: the ledger and the
    outer fold then speak the same coordinate system, so "did any trial touch a
    test row?" is a set intersection rather than an index translation nobody
    would get right twice.
    """

    outer_label: str
    index: int
    train_index: np.ndarray
    val_index: np.ndarray

    @property
    def label(self) -> str:
        return self.outer_label + "-inner" + str(self.index)


def inner_folds(
    fold: Any,
    y: Any,
    groups: Any,
    *,
    n_splits: int = DEFAULT_INNER_SPLITS,
    seed: int | None = None,
) -> tuple[InnerFold, ...]:
    """Cut ``n_splits`` inner folds out of one outer fold's training rows.

    Grouped and stratified, like the outer map: a subject that appears twice in
    the training rows must not straddle the inner split either, or the search
    picks its hyperparameters against an inner score that is inflated for the
    same reason an ungrouped outer score would be.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if getattr(fold, "train_index", None) is None:
        raise SearchError("inner folds need a RESOLVED outer fold; call cv.resolve_folds first")
    if n_splits < 2:
        raise SearchError("n_splits must be at least 2, got " + str(n_splits))

    if seed is None:
        from src.utils.seed import GLOBAL_SEED

        seed = GLOBAL_SEED

    train = np.asarray(fold.train_index, dtype=int)
    targets = np.asarray(y)[train]
    group_values = np.asarray(groups, dtype=object)[train]

    if len(np.unique(targets)) < 2:
        raise SearchError(fold.label + ": the training rows hold a single class")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    built: list[InnerFold] = []
    for index, (inner_train, inner_val) in enumerate(
        splitter.split(np.zeros((len(train), 1)), targets, group_values)
    ):
        global_train = train[inner_train]
        global_val = train[inner_val]
        overlap = set(global_train.tolist()) & set(global_val.tolist())
        if overlap:
            raise SearchError(
                fold.label + " inner" + str(index) + ": train and validation share rows"
            )
        shared = set(group_values[inner_train].tolist()) & set(group_values[inner_val].tolist())
        if shared:
            raise SearchError(
                fold.label
                + " inner"
                + str(index)
                + ": "
                + str(len(shared))
                + " group(s) appear on both sides, e.g. "
                + ", ".join(sorted(str(item) for item in shared)[:5])
            )
        built.append(
            InnerFold(
                outer_label=fold.label,
                index=index,
                train_index=global_train,
                val_index=global_val,
            )
        )
    return tuple(built)


class RowLedger:
    """Every matrix row the search fitted on or scored, as global positions.

    Exists so leakage is *measured*. The Phase 43 driver already refuses a fold
    whose train and test overlap, but a search runs hundreds of fits underneath
    that driver and none of them pass through it. The ledger is written by
    :func:`score_params` on every trial, and the search result carries it, so a
    test can assert set-disjointness against the outer test rows after the fact
    instead of trusting that the inner splitter was called correctly.
    """

    def __init__(self) -> None:
        self._fitted: set[int] = set()
        self._scored: set[int] = set()

    def record(self, train_index: Any, val_index: Any) -> None:
        self._fitted.update(int(value) for value in np.asarray(train_index).ravel())
        self._scored.update(int(value) for value in np.asarray(val_index).ravel())

    @property
    def fitted(self) -> frozenset[int]:
        return frozenset(self._fitted)

    @property
    def scored(self) -> frozenset[int]:
        return frozenset(self._scored)

    @property
    def touched(self) -> frozenset[int]:
        return frozenset(self._fitted | self._scored)

    def assert_disjoint_from(self, forbidden: Any, *, what: str = "outer test fold") -> None:
        blocked = {int(value) for value in np.asarray(forbidden).ravel()}
        fitted = self._fitted & blocked
        scored = self._scored & blocked
        if fitted or scored:
            raise SearchError(
                "search touched the "
                + what
                + ": "
                + str(len(fitted))
                + " row(s) fitted on and "
                + str(len(scored))
                + " row(s) scored; every number this search produced is invalid"
            )


# ---------------------------------------------------------------------------
# T54.4 -- one trial
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """One point in the space, evaluated over every inner fold of one outer fold."""

    index: int
    params: dict[str, Any]
    score: float
    fold_scores: tuple[float, ...] = ()
    seconds: float = 0.0
    status: str = "ok"
    message: str = ""
    components: dict[str, float] = field(default_factory=dict)
    #: The same components, computed **per inner fold** rather than pooled.
    #:
    #: Logged because every objective in this project is a linear function of
    #: sensitivity and specificity, and the score is their mean over folds -- so
    #: with the per-fold components on record, the winner under *any* other
    #: linear objective can be re-derived from the trial log exactly, without
    #: refitting anything. Without them, changing the objective means re-running
    #: the whole search; that cost was paid once, on 2026-08-28, and this field
    #: exists so it is not paid again.
    fold_components: tuple[dict[str, float], ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(np.isfinite(self.score))

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "trial": self.index,
            "score": None if not np.isfinite(self.score) else round(float(self.score), 6),
            "score_std": (
                round(float(np.std(self.fold_scores)), 6) if self.fold_scores else None
            ),
            "n_inner_folds": len(self.fold_scores),
            "seconds": round(float(self.seconds), 4),
            "status": self.status,
        }
        for name, value in sorted(self.components.items()):
            row["inner_" + name] = round(float(value), 6)
        for position, value in enumerate(self.fold_scores):
            row["inner_fold_" + str(position) + "_score"] = round(float(value), 6)
        for position, scored in enumerate(self.fold_components):
            for name, value in sorted(scored.items()):
                row["inner_fold_" + str(position) + "_" + name] = round(float(value), 6)
        for name, value in sorted(self.params.items()):
            row["param_" + name] = value
        row["message"] = self.message
        return row


@dataclass
class SearchResult:
    """Everything one search of one model on one outer fold produced."""

    method: str
    model_id: str
    task: str
    outer_label: str
    repeat: int
    fold: int
    trials: list[Trial] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    stop_reason: str = ""
    seconds: float = 0.0
    n_inner_folds: int = 0
    objective: str = ""
    seed: int = 42
    ledger: RowLedger = field(default_factory=RowLedger)
    outer_test_index: tuple[int, ...] = ()
    outer_train_index: tuple[int, ...] = ()

    @property
    def successful(self) -> list[Trial]:
        return [trial for trial in self.trials if trial.ok]

    @property
    def best(self) -> Trial | None:
        good = self.successful
        if not good:
            return None
        # Ties broken toward the EARLIER trial: with a fixed seed that makes the
        # reported best point a function of the search, not of dictionary order.
        return max(good, key=lambda trial: (trial.score, -trial.index))

    @property
    def best_params(self) -> dict[str, Any]:
        best = self.best
        return dict(best.params) if best is not None else {}

    @property
    def best_score(self) -> float:
        best = self.best
        return float(best.score) if best is not None else float("nan")

    def assert_no_outer_leakage(self) -> None:
        self.ledger.assert_disjoint_from(self.outer_test_index)

    def trials_frame(self) -> Any:
        import pandas as pd

        prefix = {
            "method": self.method,
            "model_id": self.model_id,
            "task": self.task,
            "outer_fold": self.outer_label,
            "repeat": self.repeat,
            "fold": self.fold,
            "objective": self.objective,
        }
        rows = []
        running = -np.inf
        for trial in self.trials:
            if trial.ok:
                running = max(running, trial.score)
            row = dict(prefix)
            row.update(trial.as_row())
            row["best_so_far"] = None if not np.isfinite(running) else round(running, 6)
            rows.append(row)
        return pd.DataFrame(rows)

    def convergence_frame(self) -> Any:
        """Best score versus trial index -- the SO-02 convergence trace (T56.4)."""
        import pandas as pd

        rows = []
        running = -np.inf
        elapsed = 0.0
        for trial in self.trials:
            elapsed += float(trial.seconds)
            if trial.ok:
                running = max(running, trial.score)
            rows.append(
                {
                    "method": self.method,
                    "model_id": self.model_id,
                    "task": self.task,
                    "outer_fold": self.outer_label,
                    "trial": trial.index,
                    "score": None if not trial.ok else round(float(trial.score), 6),
                    "best_so_far": None if not np.isfinite(running) else round(running, 6),
                    "elapsed_seconds": round(elapsed, 4),
                }
            )
        return pd.DataFrame(rows)

    def as_summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model_id": self.model_id,
            "task": self.task,
            "outer_fold": self.outer_label,
            "repeat": self.repeat,
            "fold": self.fold,
            "objective": self.objective,
            "seed": self.seed,
            "n_trials": len(self.trials),
            "n_trials_ok": len(self.successful),
            "n_inner_folds": self.n_inner_folds,
            "best_score": (
                None if not np.isfinite(self.best_score) else round(self.best_score, 6)
            ),
            "best_params": self.best_params,
            "seconds": round(self.seconds, 3),
            "stop_reason": self.stop_reason,
            "budget": self.budget.as_dict(),
            "n_rows_touched": len(self.ledger.touched),
            "n_outer_test_rows": len(self.outer_test_index),
            "outer_test_rows_touched": len(
                self.ledger.touched & frozenset(self.outer_test_index)
            ),
        }


# ---------------------------------------------------------------------------
# evaluating one point
# ---------------------------------------------------------------------------


def score_params(
    model_id: str,
    params: dict[str, Any],
    data: Any,
    splits: Sequence[InnerFold],
    *,
    objective: Objective,
    ledger: RowLedger | None = None,
    pipeline_config: dict[str, Any] | None = None,
) -> Trial:
    """Fit ``model_id(**params)`` on each inner training set and score its validation set.

    A fresh pipeline per inner fold, for the same reason the outer driver builds
    a fresh estimator per outer fold: the imputer, the scaler and any selector
    have to learn from the inner training rows alone. Rule 2 does not stop
    applying one level down.

    A point that raises is recorded as a failed trial rather than propagated. A
    search space is a declaration of what is *plausible*, and a solver that
    refuses one corner of it is information about that corner -- but it is
    counted, named and logged, never silently retried as a different point.
    """
    from sklearn.base import clone

    from src.models import estimators as est
    from src.models.pipeline import build_pipeline

    started = time.perf_counter()
    features = np.asarray(data.X)
    targets = np.asarray(data.y)
    labels = tuple(np.unique(targets).tolist())
    fold_scores: list[float] = []
    fold_components: list[dict[str, float]] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []

    for split in splits:
        if ledger is not None:
            ledger.record(split.train_index, split.val_index)
        x_train = features[split.train_index]
        y_train = targets[split.train_index]
        x_val = features[split.val_index]
        y_val = targets[split.val_index]
        try:
            if model_id in {"M6", "M7"}:
                estimator = est.make_ensemble(
                    model_id,
                    groups=np.asarray(data.groups, dtype=object)[split.train_index],
                    **params,
                )
            else:
                estimator = est.build_estimator(model_id, **params)
            pipeline = build_pipeline(
                clone(estimator),
                config=pipeline_config,
                y=y_train,
                n_features=int(features.shape[1]),
            )
            pipeline.fit(x_train, y_train)
            predicted = np.asarray(pipeline.predict(x_val))
        except Exception as error:  # noqa: BLE001 -- a bad corner of the space is data
            return Trial(
                index=-1,
                params=dict(params),
                score=float("nan"),
                seconds=time.perf_counter() - started,
                status="error",
                message=type(error).__name__ + ": " + str(error)[:200],
            )
        fold_scores.append(objective(y_val, predicted, labels))
        fold_components.append(objective.components(y_val, predicted, labels))
        pooled_true.append(y_val)
        pooled_pred.append(predicted)

    components = objective.components(
        np.concatenate(pooled_true), np.concatenate(pooled_pred), labels
    )
    return Trial(
        index=-1,
        params=dict(params),
        score=float(np.mean(fold_scores)),
        fold_scores=tuple(float(value) for value in fold_scores),
        fold_components=tuple(fold_components),
        seconds=time.perf_counter() - started,
        status="ok",
        components=components,
    )


# ---------------------------------------------------------------------------
# T54.1 -- the interface
# ---------------------------------------------------------------------------


class BaseSearch(ABC):
    """One search method. Subclasses implement ``_propose`` and nothing else.

    The contract is deliberately narrow. ``_propose`` receives the history so far
    and returns the next point to try; it may not fit anything, look at the data,
    or decide when to stop. Keeping the loop here is what makes SO-01 and SO-02
    comparable at equal budget (T56.3/T56.5) -- if each optimizer owned its own
    loop, "equal budget" would mean whatever each one happened to count.
    """

    method: ClassVar[str] = "base"

    def __init__(
        self,
        model_id: str,
        space: Any,
        objective: Objective,
        budget: Budget,
        *,
        seed: int | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ) -> None:
        if seed is None:
            from src.utils.seed import GLOBAL_SEED

            seed = GLOBAL_SEED
        self.model_id = model_id
        self.space = space
        self.objective = objective
        self.budget = budget
        self.seed = int(seed)
        self.pipeline_config = pipeline_config

    # -- subclass hook -----------------------------------------------------

    @abstractmethod
    def _propose(self, history: Sequence[Trial]) -> dict[str, Any]:
        """The next point to evaluate. Must be legal in ``self.space``."""

    def _observe(self, trial: Trial) -> None:
        """Optional hook: tell a model-based optimizer what the trial scored.

        Not abstract. A memoryless method (random search) genuinely has nothing
        to do here, and forcing it to declare an empty override would say the
        opposite of what is true.
        """
        del trial

    # -- the loop ----------------------------------------------------------

    def run(
        self,
        data: Any,
        outer_fold: Any,
        *,
        n_inner_splits: int = DEFAULT_INNER_SPLITS,
        splits: Sequence[InnerFold] | None = None,
        on_trial: Callable[[Trial], None] | None = None,
    ) -> SearchResult:
        """Search ``outer_fold``'s training rows. The test rows are never read."""
        if not self.space.dimensions:
            raise SearchError(
                self.model_id + " declares an empty search space; there is nothing to search"
            )
        if splits is None:
            splits = inner_folds(
                outer_fold, data.y, data.groups, n_splits=n_inner_splits, seed=self.seed
            )
        result = SearchResult(
            method=self.method,
            model_id=self.model_id,
            task=getattr(data, "task", ""),
            outer_label=outer_fold.label,
            repeat=int(outer_fold.repeat),
            fold=int(outer_fold.fold),
            budget=self.budget,
            n_inner_folds=len(splits),
            objective=self.objective.name,
            seed=self.seed,
            outer_test_index=tuple(int(v) for v in np.asarray(outer_fold.test_index)),
            outer_train_index=tuple(int(v) for v in np.asarray(outer_fold.train_index)),
        )

        started = time.perf_counter()
        estimate = 0.0
        while True:
            elapsed = time.perf_counter() - started
            reason = self.budget.stop_reason(
                len(result.trials), elapsed, next_trial_estimate=estimate
            )
            if reason is not None:
                result.stop_reason = reason
                break

            params = self.space.repair(self._propose(result.trials))
            violations = self.space.violations(params)
            if violations:
                trial = Trial(
                    index=len(result.trials),
                    params=dict(params),
                    score=float("nan"),
                    status="invalid",
                    message="; ".join(violations)[:200],
                )
            else:
                trial = score_params(
                    self.model_id,
                    params,
                    data,
                    splits,
                    objective=self.objective,
                    ledger=result.ledger,
                    pipeline_config=self.pipeline_config,
                )
                trial.index = len(result.trials)
            result.trials.append(trial)
            self._observe(trial)
            if on_trial is not None:
                on_trial(trial)
            done = [item.seconds for item in result.trials if item.seconds > 0]
            if done:
                estimate = float(np.median(done))
            log.debug(
                "%s %s %s trial %d: score=%s (%.2f s)",
                self.method,
                self.model_id,
                outer_fold.label,
                trial.index,
                round(trial.score, 5) if trial.ok else trial.status,
                trial.seconds,
            )

        result.seconds = time.perf_counter() - started
        result.assert_no_outer_leakage()
        log.info(
            "%s %s %s: %d trials, best %s, %.1f s (%s)",
            self.method,
            self.model_id,
            outer_fold.label,
            len(result.trials),
            round(result.best_score, 5),
            result.seconds,
            result.stop_reason,
        )
        return result


# ---------------------------------------------------------------------------
# output (T54.4)
# ---------------------------------------------------------------------------


def search_dir(exp: str, out_dir: str | Path | None = None) -> Path:
    """``outputs/05_search_optimization/{exp}/``, created."""
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    root = (
        Path(out_dir)
        if out_dir is not None
        else Path(load_config("paths").require("outputs.search_optimization"))
    )
    return Path(ensure_dir(root / exp))


def write_trials(
    results: Sequence[SearchResult],
    exp: str,
    *,
    out_dir: str | Path | None = None,
    filename: str = TRIALS_FILENAME,
) -> Path:
    """Every trial of every search in one table (T54.4)."""
    import pandas as pd

    from src.utils.io import save_csv

    if not results:
        raise SearchError("no search results to write")
    frame = pd.concat([result.trials_frame() for result in results], ignore_index=True)
    return save_csv(frame, search_dir(exp, out_dir) / filename)


def write_best_params(
    results: Sequence[SearchResult],
    exp: str,
    *,
    out_dir: str | Path | None = None,
    filename: str = BEST_PARAMS_FILENAME,
    extra: dict[str, Any] | None = None,
) -> Path:
    """The best point per (model, outer fold), plus the budget it was found under."""
    from src.utils.io import save_json

    payload: dict[str, Any] = {
        "experiment": exp,
        "searches": [result.as_summary() for result in results],
    }
    if extra:
        payload.update(extra)
    return save_json(payload, search_dir(exp, out_dir) / filename)


def write_inner_folds(
    splits_by_fold: dict[str, Sequence[InnerFold]],
    exp: str,
    record_uids: Sequence[str],
    *,
    out_dir: str | Path | None = None,
) -> Path:
    """The derived inner splits, materialised so they can be inspected like DA-07."""
    import pandas as pd

    from src.utils.io import save_csv

    uids = list(record_uids)
    rows = []
    for outer_label, splits in sorted(splits_by_fold.items()):
        for split in splits:
            for role, index in (("train", split.train_index), ("val", split.val_index)):
                for position in np.asarray(index, dtype=int):
                    rows.append(
                        {
                            "outer_fold": outer_label,
                            "inner_fold": split.index,
                            "role": role,
                            "row": int(position),
                            "record_uid": uids[int(position)],
                        }
                    )
    return save_csv(pd.DataFrame(rows), search_dir(exp, out_dir) / INNER_FOLD_FILENAME)
