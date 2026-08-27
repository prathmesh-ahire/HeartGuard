"""Running a search method over outer folds, and scoring what it chose (Phase 54-56).

:mod:`src.optimization.base` defines what one search of one model on one outer
fold is. This module is the layer above: pick the outer folds, run a method on
each of them, refit the winning point on the whole outer training set and score
it on the outer test fold that the search never saw. That last step is what makes
the number a *nested* CV estimate rather than a search score -- a best inner
score is an optimistically biased estimate of anything, and quoting one as a
result is one of the commonest ways a tuned pipeline reports a metric it cannot
reproduce.

The nested evaluation is deliberately cheap here: one refit per (model, outer
fold). The expensive part is always the search itself, and it is bounded by the
budget rather than by how many folds are evaluated afterwards.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.optimization.base import (
    Budget,
    InnerFold,
    Objective,
    SearchError,
    SearchResult,
    inner_folds,
    objective_for,
)
from src.utils.logging_setup import get_logger

__all__ = [
    "METHODS",
    "NestedOutcome",
    "SearchRun",
    "build_search",
    "outer_folds_for",
    "run_search",
    "evaluate_best_on_outer",
]

log = get_logger("optimization.driver")


def _methods() -> dict[str, Any]:
    from src.optimization.bayesian import BayesianSearch
    from src.optimization.randomized import RandomizedSearch

    return {RandomizedSearch.method: RandomizedSearch, BayesianSearch.method: BayesianSearch}


#: Method name -> experiment id. SO-03a/SO-03b arrive in Phases 58-59.
METHODS: dict[str, str] = {"random": "SO-01", "bayes": "SO-02"}


@dataclass
class NestedOutcome:
    """The best point of one search, scored on the outer test fold it never saw."""

    method: str
    model_id: str
    task: str
    outer_label: str
    repeat: int
    fold: int
    best_params: dict[str, Any]
    inner_score: float
    outer_score: float
    outer_metrics: dict[str, float] = field(default_factory=dict)
    refit_seconds: float = 0.0
    n_train: int = 0
    n_test: int = 0

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": self.method,
            "model_id": self.model_id,
            "task": self.task,
            "outer_fold": self.outer_label,
            "repeat": self.repeat,
            "fold": self.fold,
            "inner_best_score": round(float(self.inner_score), 6),
            "outer_score": round(float(self.outer_score), 6),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "refit_seconds": round(float(self.refit_seconds), 4),
        }
        for name, value in sorted(self.outer_metrics.items()):
            row["outer_" + name] = round(float(value), 6)
        for name, value in sorted(self.best_params.items()):
            row["best_" + name] = value
        return row


@dataclass
class SearchRun:
    """Every search of one method over one task: the results and their nested scores."""

    method: str
    exp: str
    task: str
    objective: str
    budget: Budget
    results: list[SearchResult] = field(default_factory=list)
    outcomes: list[NestedOutcome] = field(default_factory=list)
    splits_by_fold: dict[str, tuple[InnerFold, ...]] = field(default_factory=dict)
    seconds: float = 0.0

    def outcome_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame([outcome.as_row() for outcome in self.outcomes])

    def convergence_frame(self) -> Any:
        import pandas as pd

        if not self.results:
            raise SearchError("no search results")
        return pd.concat(
            [result.convergence_frame() for result in self.results], ignore_index=True
        )


def build_search(
    method: str,
    model_id: str,
    *,
    objective: Objective,
    budget: Budget,
    seed: int | None = None,
    **kwargs: Any,
) -> Any:
    """Construct one search method by name, reading the model's declared space."""
    from src.models import spaces

    available = _methods()
    if method not in available:
        raise SearchError(
            "unknown search method " + repr(method) + "; known: " + ", ".join(sorted(available))
        )
    space = spaces.load_space(model_id)
    if not space.dimensions:
        raise SearchError(
            model_id
            + " declares an empty search space in configs/models.yaml; it is a fixed"
            + " configuration, not a searchable model"
        )
    return available[method](
        model_id, space, objective, budget, seed=seed, **kwargs
    )


def outer_folds_for(
    task: str,
    data: Any,
    *,
    repeats: Sequence[int] | None = None,
    folds: Sequence[int] | None = None,
) -> tuple[Any, ...]:
    """The DA-07 outer folds to search, resolved against ``data``'s row order.

    Selecting a subset of repeats is a **budget** decision and is recorded as
    one: a search run over repeat 0 only is not a cheaper version of the full
    25-fold run, it is a smaller experiment, and the fold labels in every output
    say which folds were actually searched.
    """
    from src.evaluation import cv

    resolved = cv.resolve_folds(cv.load_folds(task), data.record_uids)
    chosen = [
        fold
        for fold in resolved
        if (repeats is None or fold.repeat in set(repeats))
        and (folds is None or fold.fold in set(folds))
    ]
    if not chosen:
        raise SearchError(
            task + ": no outer fold matches repeats=" + str(repeats) + " folds=" + str(folds)
        )
    return tuple(chosen)


def evaluate_best_on_outer(
    result: SearchResult,
    data: Any,
    outer_fold: Any,
    objective: Objective,
    *,
    pipeline_config: dict[str, Any] | None = None,
) -> NestedOutcome:
    """Refit the search's best point on the outer training rows; score the test rows.

    The refit uses the whole outer training block -- every row the search's inner
    folds were cut from -- which is the standard nested-CV protocol: the search
    chose the point on inner validation data, and the point is then given all the
    training data it is entitled to before being judged once.
    """
    from sklearn.base import clone

    from src.models import estimators as est
    from src.models.pipeline import build_pipeline

    best = result.best
    if best is None:
        raise SearchError(
            result.model_id + " " + result.outer_label + ": every trial failed; no best point"
        )

    features = np.asarray(data.X)
    targets = np.asarray(data.y)
    train = np.asarray(outer_fold.train_index, dtype=int)
    test = np.asarray(outer_fold.test_index, dtype=int)
    labels = tuple(np.unique(targets).tolist())

    if result.model_id in {"M6", "M7"}:
        estimator = est.make_ensemble(
            result.model_id,
            groups=np.asarray(data.groups, dtype=object)[train],
            **best.params,
        )
    else:
        estimator = est.build_estimator(result.model_id, **best.params)
    pipeline = build_pipeline(
        clone(estimator),
        config=pipeline_config,
        y=targets[train],
        n_features=int(features.shape[1]),
    )
    started = time.perf_counter()
    pipeline.fit(features[train], targets[train])
    refit_seconds = time.perf_counter() - started
    predicted = np.asarray(pipeline.predict(features[test]))

    return NestedOutcome(
        method=result.method,
        model_id=result.model_id,
        task=result.task,
        outer_label=result.outer_label,
        repeat=result.repeat,
        fold=result.fold,
        best_params=dict(best.params),
        inner_score=float(best.score),
        outer_score=objective(targets[test], predicted, labels),
        outer_metrics=objective.components(targets[test], predicted, labels),
        refit_seconds=refit_seconds,
        n_train=int(train.size),
        n_test=int(test.size),
    )


def run_search(
    method: str,
    model_ids: Sequence[str],
    data: Any,
    outer: Sequence[Any],
    *,
    budget: Budget | dict[str, Budget],
    objective: Objective | None = None,
    n_inner_splits: int | None = None,
    seed: int | None = None,
    pipeline_config: dict[str, Any] | None = None,
    evaluate_outer: bool = True,
    **search_kwargs: Any,
) -> SearchRun:
    """Run ``method`` for every model over every outer fold, then score the winners.

    Inner folds are cut **once per outer fold** and reused by every model and
    every method. That is not only a saving: SO-01 and SO-02 are compared at
    equal budget, and a comparison in which the two methods scored their trials
    against differently-cut validation sets would confound the method with the
    split.

    ``budget`` may be a single :class:`Budget` or a per-model mapping. Per-model
    budgets exist because trial cost differs by two orders of magnitude across
    this stack -- M3 costs ~4 s a trial on D1 and M5 ~210 s -- so one number for
    all of them either starves the cheap models or makes the run overnight-long.
    What must stay equal is the budget a given model gets under SO-01 and under
    SO-02, since that is the comparison T56.3 makes; across models it never was
    equal and was never meant to be.
    """
    from src.optimization.base import DEFAULT_INNER_SPLITS

    if objective is None:
        objective = objective_for(data.task)
    splits = int(n_inner_splits or DEFAULT_INNER_SPLITS)
    budgets = budget if isinstance(budget, dict) else {}
    default_budget = budget if isinstance(budget, Budget) else None
    if default_budget is None:
        missing = [model_id for model_id in model_ids if model_id not in budgets]
        if missing:
            raise SearchError(
                "no budget given for " + ", ".join(missing) + "; a per-model mapping must "
                + "name every model it is asked to search"
            )

    run = SearchRun(
        method=method,
        exp=METHODS.get(method, method),
        task=data.task,
        objective=objective.name,
        budget=default_budget or budgets[next(iter(model_ids))],
    )
    started = time.perf_counter()

    for fold in outer:
        run.splits_by_fold[fold.label] = inner_folds(
            fold, data.y, data.groups, n_splits=splits, seed=seed
        )

    for model_id in model_ids:
        model_budget = budgets.get(model_id, default_budget)
        for fold in outer:
            search = build_search(
                method,
                model_id,
                objective=objective,
                budget=model_budget,
                seed=seed,
                pipeline_config=pipeline_config,
                **search_kwargs,
            )
            result = search.run(data, fold, splits=run.splits_by_fold[fold.label])
            run.results.append(result)
            if evaluate_outer and result.best is not None:
                run.outcomes.append(
                    evaluate_best_on_outer(
                        result, data, fold, objective, pipeline_config=pipeline_config
                    )
                )
            elif evaluate_outer:
                log.warning(
                    "%s %s %s: every trial failed; no outer evaluation",
                    method,
                    model_id,
                    fold.label,
                )

    run.seconds = time.perf_counter() - started
    return run
