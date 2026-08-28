"""Fitness for a 138-bit feature mask, shared by the GA (SO-03a) and PSO (SO-03b).

Both population searches ask the same question of a candidate -- "how good is
this subset?" -- and T59.5 compares them at identical folds, budget and seed. If
each owned its own evaluator, "identical" would mean whatever each one happened
to do, so the evaluator lives here and both are handed the same instance.

THE CACHE IS NOT AN OPTIMISATION DETAIL
---------------------------------------
Elitism copies the best individuals forward unchanged, crossover of two similar
parents often reproduces a parent, and a PSO particle that has converged stops
moving. Re-fitting those is pure waste -- and worse, it makes "500 evaluations"
mean two different amounts of work under two algorithms whose comparison is the
whole point of Phase 59. Every distinct mask is fitted once and remembered; the
trace records both the evaluation count and the number of distinct masks, so the
budget can be read either way.

The cache is keyed on the mask bits alone. That is only sound because everything
else a fitness depends on -- the fold, the inner splits, the estimator, the seed,
the J weights -- is fixed for the lifetime of one evaluator and cannot be passed
per call.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.optimization import multi_objective as mo

__all__ = [
    "MaskError",
    "MaskFitness",
    "MaskEvaluator",
    "repair_mask",
    "mask_to_columns",
    "columns_to_mask",
]


class MaskError(RuntimeError):
    """A mask cannot be evaluated as given."""


def mask_to_columns(mask: Any) -> np.ndarray:
    """The column positions a boolean mask switches on."""
    return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(int)


def columns_to_mask(columns: Sequence[int], n_features: int) -> np.ndarray:
    """A boolean mask of width ``n_features`` with ``columns`` switched on."""
    mask = np.zeros(int(n_features), dtype=bool)
    mask[np.asarray(columns, dtype=int)] = True
    return mask


def repair_mask(
    mask: Any, *, min_features: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Switch random bits on until at least ``min_features`` are set.

    Repaired rather than discarded, and repaired rather than penalised. A
    discarded child would make the population smaller than the number the config
    reports, and a penalty would leave illegal individuals in the population
    consuming evaluations to be told again that they are illegal. Repair keeps
    the budget meaning what it says.
    """
    bits = np.asarray(mask, dtype=bool).copy()
    need = int(min_features) - int(bits.sum())
    if need <= 0:
        return bits
    off = np.flatnonzero(~bits)
    if off.size == 0:
        return bits
    generator = rng if rng is not None else np.random.default_rng(0)
    chosen = generator.choice(off, size=min(need, off.size), replace=False)
    bits[chosen] = True
    return bits


@dataclass(frozen=True)
class MaskFitness:
    """One mask's J, the metrics behind it, and what it cost to find out."""

    fitness: float
    macro_f1: float
    n_selected: int
    metrics: dict[str, float] = field(default_factory=dict)
    j_terms: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    cached: bool = False

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "fitness": float(self.fitness),
            "macro_f1": float(self.macro_f1),
            "n_selected": int(self.n_selected),
            "seconds": float(self.seconds),
            "cached": bool(self.cached),
        }
        row.update({key: float(value) for key, value in self.metrics.items()})
        return row


class MaskEvaluator:
    """Scores feature masks on the inner folds of one outer fold. Minimises J.

    Exposes a :class:`~src.optimization.base.RowLedger`, for the same reason the
    hyperparameter search does: leakage is measured here, not assumed. A
    population search runs hundreds of fits that no fold driver ever sees, and
    the ledger is what lets a test assert afterwards that not one of them touched
    an outer test row.
    """

    def __init__(
        self,
        data: Any,
        fold: Any,
        *,
        model_id: str = "M3",
        n_inner_splits: int = 3,
        seed: int | None = None,
        min_features: int = 10,
        weights: mo.JWeights | None = None,
        cost_model: mo.FamilyCostModel | None = None,
        pipeline_config: dict[str, Any] | None = None,
    ) -> None:
        from src.optimization.base import RowLedger, inner_folds

        if seed is None:
            from src.utils.seed import GLOBAL_SEED

            seed = GLOBAL_SEED

        self.data = data
        self.fold = fold
        self.model_id = str(model_id)
        self.seed = int(seed)
        self.min_features = int(min_features)
        self.weights = weights or mo.load_weights()
        self.cost_model = cost_model or mo.load_cost_model()
        self.pipeline_config = pipeline_config

        self.features = np.asarray(data.X, dtype=float)
        self.targets = np.asarray(data.y)
        self.feature_names = tuple(str(name) for name in data.feature_names)
        self.n_features = int(self.features.shape[1])
        self.splits = inner_folds(
            fold, data.y, data.groups, n_splits=int(n_inner_splits), seed=self.seed
        )
        self.ledger = RowLedger()
        for split in self.splits:
            self.ledger.record(split.train_index, split.val_index)

        self._cache: dict[bytes, MaskFitness] = {}
        self.n_calls = 0
        self.n_fitted = 0
        self.seconds = 0.0

    # -- the score ---------------------------------------------------------

    @property
    def n_distinct(self) -> int:
        return len(self._cache)

    def columns_of(self, mask: Any) -> np.ndarray:
        return mask_to_columns(mask)

    def names_of(self, mask: Any) -> tuple[str, ...]:
        return tuple(self.feature_names[int(c)] for c in mask_to_columns(mask))

    def fitness(self, mask: Any) -> MaskFitness:
        """J for ``mask``, averaged over the inner folds. Lower is better."""
        bits = np.asarray(mask, dtype=bool)
        if bits.shape != (self.n_features,):
            raise MaskError(
                "mask has width " + str(bits.shape) + ", matrix has "
                + str(self.n_features) + " columns"
            )
        if int(bits.sum()) < 1:
            raise MaskError("an all-zero mask selects no features and cannot be scored")

        self.n_calls += 1
        key = np.packbits(bits).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return MaskFitness(
                fitness=hit.fitness,
                macro_f1=hit.macro_f1,
                n_selected=hit.n_selected,
                metrics=hit.metrics,
                j_terms=hit.j_terms,
                seconds=0.0,
                cached=True,
            )

        from src.feature_selection.sweep import score_subset

        columns = mask_to_columns(bits)
        started = time.perf_counter()
        fold_macro: list[float] = []
        fold_metrics: list[dict[str, float]] = []
        for split in self.splits:
            metrics, _, _ = score_subset(
                columns,
                self.model_id,
                self.features[split.train_index],
                self.targets[split.train_index],
                self.features[split.val_index],
                self.targets[split.val_index],
                feature_names=self.feature_names,
                weights=self.weights,
                cost_model=self.cost_model,
                pipeline_config=self.pipeline_config,
            )
            fold_macro.append(float(metrics["macro_f1"]))
            fold_metrics.append(metrics)
        seconds = time.perf_counter() - started

        macro = float(np.mean(fold_macro))
        j = mo.score_j(
            macro,
            [self.feature_names[int(c)] for c in columns],
            weights=self.weights,
            cost_model=self.cost_model,
        )
        averaged = {
            key_: float(np.mean([m.get(key_, np.nan) for m in fold_metrics]))
            for key_ in fold_metrics[0]
        }
        scored = MaskFitness(
            fitness=float(j.value),
            macro_f1=macro,
            n_selected=int(columns.size),
            metrics=averaged,
            j_terms=j.as_dict(),
            seconds=float(seconds),
            cached=False,
        )
        self._cache[key] = scored
        self.n_fitted += 1
        self.seconds += seconds
        return scored

    def assert_no_outer_leakage(self) -> None:
        """No inner split ever reached a row of the outer test fold."""
        self.ledger.assert_disjoint_from(self.fold.test_index)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "outer_fold": str(self.fold.label),
            "n_inner_splits": len(self.splits),
            "n_features": self.n_features,
            "min_features": self.min_features,
            "seed": self.seed,
            "evaluations": int(self.n_calls),
            "distinct_masks_fitted": int(self.n_fitted),
            "fit_seconds": float(self.seconds),
            "j_weights": self.weights.as_dict(),
            "outer_test_rows_touched": len(
                self.ledger.touched & frozenset(np.asarray(self.fold.test_index).tolist())
            ),
        }
