"""SO-03a: a genetic algorithm over the 138-bit feature mask (Phase 58).

A compact custom implementation rather than DEAP. T58.1 allows either, and the
algorithm is ~150 lines of tournament selection, uniform crossover, per-bit
mutation and elitism; DEAP would add a dependency that has to be pinned, wheel-
checked on Windows/Python 3.11 and proved deterministic under seed 42 -- all the
work of writing it, plus a dependency.

Everything stochastic goes through ONE ``numpy.random.Generator`` seeded from the
run's seed, so two runs of the same command produce the same population, the same
crossovers and the same mutations (research rule 5). Nothing here calls the
global numpy random state.

FITNESS IS J, AND J IS MINIMISED
--------------------------------
T58.2 names the fitness as the multi-objective score. J is a cost, so the GA
minimises: "best" is the lowest value everywhere in this module, and the
convergence trace records best and mean per generation (T58.4) in that direction.
A reader used to maximised fitness will find the trace descending -- that is the
score being defined as a cost, not the search failing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.optimization.masks import MaskEvaluator, repair_mask

__all__ = [
    "GAError",
    "GAConfig",
    "Generation",
    "GAResult",
    "load_ga_config",
    "run_ga",
    "run_joint_ga",
]


class GAError(RuntimeError):
    """The GA cannot be configured or run as asked."""


@dataclass(frozen=True)
class GAConfig:
    """Population, generations, operators -- every one of them from config (T58.3)."""

    population_size: int = 30
    generations: int = 25
    crossover_rate: float = 0.8
    mutation_rate: float = 0.02
    tournament_size: int = 3
    elitism: int = 2
    min_features: int = 10
    init_density: float = 0.5

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise GAError("population_size must be at least 2")
        if self.generations < 1:
            raise GAError("generations must be at least 1")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise GAError("crossover_rate must lie in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise GAError("mutation_rate must lie in [0, 1]")
        if not 1 <= self.tournament_size <= self.population_size:
            raise GAError("tournament_size must lie in [1, population_size]")
        if not 0 <= self.elitism < self.population_size:
            raise GAError("elitism must lie in [0, population_size)")
        if not 0.0 < self.init_density <= 1.0:
            raise GAError("init_density must lie in (0, 1]")

    @property
    def max_evaluations(self) -> int:
        """Fitness calls if nothing were cached: the budget PSO has to match."""
        return int(self.population_size * self.generations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "population_size": int(self.population_size),
            "generations": int(self.generations),
            "crossover_rate": float(self.crossover_rate),
            "mutation_rate": float(self.mutation_rate),
            "tournament_size": int(self.tournament_size),
            "elitism": int(self.elitism),
            "min_features": int(self.min_features),
            "init_density": float(self.init_density),
            "max_evaluations": self.max_evaluations,
        }


def load_ga_config(config: dict[str, Any] | None = None) -> GAConfig:
    """`optimization.genetic_algorithm` from configs/models.yaml."""
    if config is None:
        from src.utils.config import load_config

        config = load_config("models").get("optimization.genetic_algorithm") or {}
    settings = dict(config)
    return GAConfig(
        population_size=int(settings.get("population_size", 30)),
        generations=int(settings.get("generations", 25)),
        crossover_rate=float(settings.get("crossover_rate", 0.8)),
        mutation_rate=float(settings.get("mutation_rate", 0.02)),
        tournament_size=int(settings.get("tournament_size", 3)),
        elitism=int(settings.get("elitism", 2)),
        min_features=int(settings.get("min_features", 10)),
        init_density=float(settings.get("init_density", 0.5)),
    )


# ---------------------------------------------------------------------------
# the trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """One generation's fitness summary (T58.4)."""

    index: int
    best: float
    #: The best fitness seen since generation 0, which is NOT the same column.
    #: A GA with elitism cannot worsen, so its two columns coincide; a swarm has
    #: no elitism and its per-iteration best genuinely goes up and down. A
    #: convergence plot that overlaid the two methods on `best` alone would show
    #: PSO "diverging" when it is only reporting a different quantity, so the
    #: monotone column is carried explicitly and is the one G20 draws.
    best_so_far: float
    mean: float
    worst: float
    std: float
    best_n_selected: int
    best_macro_f1: float
    mean_n_selected: float
    evaluations: int
    distinct_fitted: int
    seconds: float

    def as_row(self) -> dict[str, Any]:
        return {
            "generation": int(self.index),
            "best_fitness": float(self.best),
            "best_so_far": float(self.best_so_far),
            "mean_fitness": float(self.mean),
            "worst_fitness": float(self.worst),
            "std_fitness": float(self.std),
            "best_n_selected": int(self.best_n_selected),
            "best_macro_f1": float(self.best_macro_f1),
            "mean_n_selected": float(self.mean_n_selected),
            "evaluations": int(self.evaluations),
            "distinct_fitted": int(self.distinct_fitted),
            "seconds": float(self.seconds),
        }


@dataclass
class GAResult:
    """The best mask found, the per-generation trace, and the ledger behind them."""

    method: str
    model_id: str
    outer_label: str
    task: str
    best_mask: np.ndarray
    best_fitness: float
    best_macro_f1: float
    best_metrics: dict[str, float]
    generations: list[Generation] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    evaluator: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    best_weights: tuple[float, ...] = ()

    @property
    def n_selected(self) -> int:
        return int(np.asarray(self.best_mask, dtype=bool).sum())

    def trace_frame(self) -> Any:
        import pandas as pd

        frame = pd.DataFrame([generation.as_row() for generation in self.generations])
        frame.insert(0, "outer_fold", self.outer_label)
        frame.insert(0, "model_id", self.model_id)
        frame.insert(0, "method", self.method)
        return frame

    def as_summary(self, feature_names: Sequence[str]) -> dict[str, Any]:
        names = [
            str(feature_names[int(c)])
            for c in np.flatnonzero(np.asarray(self.best_mask, dtype=bool))
        ]
        summary: dict[str, Any] = {
            "method": self.method,
            "model_id": self.model_id,
            "task": self.task,
            "outer_fold": self.outer_label,
            "best_fitness": float(self.best_fitness),
            "best_macro_f1": float(self.best_macro_f1),
            "n_selected": self.n_selected,
            "features": names,
            "generations": len(self.generations),
            "seconds": float(self.seconds),
            "config": dict(self.config),
            "evaluator": dict(self.evaluator),
            "best_metrics": {k: float(v) for k, v in self.best_metrics.items()},
        }
        if self.best_weights:
            summary["best_weights"] = [float(w) for w in self.best_weights]
        return summary


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------


def _initial_population(
    size: int, n_features: int, density: float, min_features: int, rng: np.random.Generator
) -> np.ndarray:
    population = rng.random((size, n_features)) < float(density)
    return np.asarray(
        [repair_mask(row, min_features=min_features, rng=rng) for row in population],
        dtype=bool,
    )


def _tournament(
    fitness: np.ndarray, size: int, rng: np.random.Generator
) -> int:
    """Index of the best of ``size`` randomly drawn individuals. Lower is better."""
    contenders = rng.choice(fitness.size, size=int(size), replace=False)
    return int(contenders[int(np.argmin(fitness[contenders]))])


def _crossover(
    left: np.ndarray, right: np.ndarray, rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover.

    Uniform rather than single- or two-point: a feature mask has no meaningful
    ordering along the chromosome -- column 3 and column 4 are two acoustic
    features that happen to sit next to each other in the registry -- so a
    positional crossover would preserve an adjacency that means nothing.
    """
    if rng.random() >= float(rate):
        return left.copy(), right.copy()
    swap = rng.random(left.size) < 0.5
    child_a = np.where(swap, right, left)
    child_b = np.where(swap, left, right)
    return child_a, child_b


def _mutate(mask: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(mask.size) < float(rate)
    return np.logical_xor(mask, flips)


# ---------------------------------------------------------------------------
# T58.1-T58.4 -- the mask GA
# ---------------------------------------------------------------------------


def run_ga(
    evaluator: MaskEvaluator,
    config: GAConfig | None = None,
    *,
    seed: int | None = None,
    method: str = "ga",
    progress: Any = None,
) -> GAResult:
    """Evolve a feature mask against ``evaluator``. Minimises J.

    The evaluator owns the fold and the inner splits, so this function cannot
    reach the outer test rows even by accident -- it never sees the matrix.
    """
    config = config or load_ga_config()
    if seed is None:
        seed = evaluator.seed
    rng = np.random.default_rng(int(seed))

    n_features = evaluator.n_features
    population = _initial_population(
        config.population_size, n_features, config.init_density, config.min_features, rng
    )
    started = time.perf_counter()
    result_generations: list[Generation] = []

    best_mask = population[0].copy()
    best_fitness = float("inf")
    best_macro = float("nan")
    best_metrics: dict[str, float] = {}

    for index in range(config.generations):
        generation_started = time.perf_counter()
        scored = [evaluator.fitness(individual) for individual in population]
        fitness = np.asarray([item.fitness for item in scored], dtype=float)
        sizes = np.asarray([item.n_selected for item in scored], dtype=float)

        champion = int(np.argmin(fitness))
        if fitness[champion] < best_fitness:
            best_fitness = float(fitness[champion])
            best_mask = population[champion].copy()
            best_macro = float(scored[champion].macro_f1)
            best_metrics = dict(scored[champion].metrics)

        record = Generation(
            index=index,
            best=float(np.min(fitness)),
            best_so_far=float(best_fitness),
            mean=float(np.mean(fitness)),
            worst=float(np.max(fitness)),
            std=float(np.std(fitness)),
            best_n_selected=int(scored[champion].n_selected),
            best_macro_f1=float(scored[champion].macro_f1),
            mean_n_selected=float(np.mean(sizes)),
            evaluations=int(evaluator.n_calls),
            distinct_fitted=int(evaluator.n_fitted),
            seconds=time.perf_counter() - generation_started,
        )
        result_generations.append(record)
        if progress is not None:
            progress(record)

        if index == config.generations - 1:
            break

        # Elites carried forward unchanged. Without them a generation can be
        # strictly worse than the one before it, which turns the convergence
        # trace T58.4 asks for into a random walk.
        order = np.argsort(fitness, kind="stable")
        children = [population[position].copy() for position in order[: config.elitism]]
        while len(children) < config.population_size:
            mother = population[_tournament(fitness, config.tournament_size, rng)]
            father = population[_tournament(fitness, config.tournament_size, rng)]
            child_a, child_b = _crossover(mother, father, config.crossover_rate, rng)
            for child in (child_a, child_b):
                if len(children) >= config.population_size:
                    break
                mutated = _mutate(child, config.mutation_rate, rng)
                children.append(
                    repair_mask(mutated, min_features=config.min_features, rng=rng)
                )
        population = np.asarray(children, dtype=bool)

    evaluator.assert_no_outer_leakage()
    return GAResult(
        method=method,
        model_id=evaluator.model_id,
        outer_label=str(evaluator.fold.label),
        task=str(evaluator.data.task),
        best_mask=best_mask,
        best_fitness=best_fitness,
        best_macro_f1=best_macro,
        best_metrics=best_metrics,
        generations=result_generations,
        config=config.as_dict(),
        evaluator=evaluator.as_dict(),
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# T58.5 -- the joint chromosome: mask bits AND ensemble weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointGAConfig:
    """The reduced budget and the member set the joint GA runs under."""

    members: tuple[str, ...] = ("M1", "M3", "M4")
    population_size: int = 16
    generations: int = 10
    crossover_rate: float = 0.8
    mutation_rate: float = 0.02
    tournament_size: int = 3
    elitism: int = 2
    min_features: int = 10
    init_density: float = 0.5
    weight_mutation_sigma: float = 0.15

    def as_dict(self) -> dict[str, Any]:
        return {
            "members": list(self.members),
            "population_size": int(self.population_size),
            "generations": int(self.generations),
            "crossover_rate": float(self.crossover_rate),
            "mutation_rate": float(self.mutation_rate),
            "tournament_size": int(self.tournament_size),
            "elitism": int(self.elitism),
            "min_features": int(self.min_features),
            "weight_mutation_sigma": float(self.weight_mutation_sigma),
            "max_evaluations": int(self.population_size * self.generations),
        }


def load_joint_config(config: dict[str, Any] | None = None) -> JointGAConfig:
    """`optimization.genetic_algorithm.joint_weights`, with the GA's own operators."""
    if config is None:
        from src.utils.config import load_config

        loaded = load_config("models").get("optimization.genetic_algorithm") or {}
    else:
        loaded = dict(config)
    joint = dict(loaded.get("joint_weights") or {})
    return JointGAConfig(
        members=tuple(joint.get("members", ["M1", "M3", "M4"])),
        population_size=int(joint.get("population_size", 16)),
        generations=int(joint.get("generations", 10)),
        crossover_rate=float(loaded.get("crossover_rate", 0.8)),
        mutation_rate=float(loaded.get("mutation_rate", 0.02)),
        tournament_size=int(loaded.get("tournament_size", 3)),
        elitism=int(loaded.get("elitism", 2)),
        min_features=int(loaded.get("min_features", 10)),
        init_density=float(loaded.get("init_density", 0.5)),
        weight_mutation_sigma=float(joint.get("weight_mutation_sigma", 0.15)),
    )


def run_joint_ga(
    data: Any,
    fold: Any,
    config: JointGAConfig | None = None,
    *,
    n_inner_splits: int = 3,
    seed: int | None = None,
    weights: Any = None,
    cost_model: Any = None,
    progress: Any = None,
) -> GAResult:
    """Evolve the feature mask and the ensemble weights in one chromosome (T58.5).

    The chromosome is 138 mask bits followed by one non-negative real gene per
    member; the genes are normalised to the simplex before use, so the search
    space is the simplex without a constraint handler.

    WHY THIS RUNS AT A SMALLER BUDGET, AND WITH A DIFFERENT MEMBER SET
    ------------------------------------------------------------------
    Nothing about the joint chromosome can be cached across masks: a new mask
    means every member has to be refitted before any weight can be scored. The
    M7 member set (M3, M4, M5) costs 347 s per evaluation measured on this fold,
    which is 72 hours at the mask GA's 750 evaluations. The members here are
    M1, M3 and M4 -- the same three model families, minus the gradient boosting
    that accounts for almost all of that -- at 16 x 10 = 160 evaluations.

    That makes this a demonstration that the joint encoding works, not a result
    that can be compared to M7. Every write-up must say so.
    """
    from src.ensemble.soft_voting import fuse_probabilities
    from src.optimization import multi_objective as mo
    from src.optimization.base import RowLedger, inner_folds

    config = config or load_joint_config()
    if seed is None:
        from src.utils.seed import GLOBAL_SEED

        seed = GLOBAL_SEED
    rng = np.random.default_rng(int(seed))
    weights = weights or mo.load_weights()
    cost_model = cost_model or mo.load_cost_model()

    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    names = tuple(str(name) for name in data.feature_names)
    n_features = int(features.shape[1])
    n_members = len(config.members)

    splits = inner_folds(fold, data.y, data.groups, n_splits=int(n_inner_splits), seed=seed)
    ledger = RowLedger()
    for split in splits:
        ledger.record(split.train_index, split.val_index)

    labels = tuple(np.unique(targets).tolist())

    def evaluate(mask: np.ndarray, gene: np.ndarray) -> tuple[float, float, dict[str, float]]:
        from sklearn.base import clone

        from src.models import estimators as est
        from src.models.pipeline import build_pipeline

        columns = np.flatnonzero(mask)
        vector = np.clip(np.asarray(gene, dtype=float), 0.0, None)
        if float(vector.sum()) <= 0:
            vector = np.ones(n_members, dtype=float)
        fold_macro: list[float] = []
        for split in splits:
            x_train = features[np.ix_(split.train_index, columns)]
            y_train = targets[split.train_index]
            x_val = features[np.ix_(split.val_index, columns)]
            y_val = targets[split.val_index]
            stack = []
            for member in config.members:
                pipeline = build_pipeline(
                    clone(est.build_estimator(member)),
                    config=None,
                    y=y_train,
                    n_features=int(columns.size),
                )
                pipeline.fit(x_train, y_train)
                stack.append(np.asarray(pipeline.predict_proba(x_val), dtype=float))
            fused = fuse_probabilities(np.stack(stack), vector)
            predicted = np.asarray(
                [labels[int(position)] for position in np.argmax(fused, axis=1)]
            )
            fold_macro.append(mo.macro_f1(y_val, predicted, labels=list(labels)))
        macro = float(np.mean(fold_macro))
        j = mo.score_j(
            macro,
            [names[int(c)] for c in columns],
            weights=weights,
            cost_model=cost_model,
        )
        return float(j.value), macro, {"macro_f1": macro}

    masks = _initial_population(
        config.population_size, n_features, config.init_density, config.min_features, rng
    )
    genes = rng.random((config.population_size, n_members)) + 0.1

    started = time.perf_counter()
    generations: list[Generation] = []
    best_fitness = float("inf")
    best_mask = masks[0].copy()
    best_gene = genes[0].copy()
    best_macro = float("nan")
    best_metrics: dict[str, float] = {}
    evaluations = 0

    for index in range(config.generations):
        generation_started = time.perf_counter()
        scored = [evaluate(masks[i], genes[i]) for i in range(config.population_size)]
        evaluations += config.population_size
        fitness = np.asarray([item[0] for item in scored], dtype=float)
        champion = int(np.argmin(fitness))
        if fitness[champion] < best_fitness:
            best_fitness = float(fitness[champion])
            best_mask = masks[champion].copy()
            best_gene = genes[champion].copy()
            best_macro = float(scored[champion][1])
            best_metrics = dict(scored[champion][2])

        record = Generation(
            index=index,
            best=float(np.min(fitness)),
            best_so_far=float(best_fitness),
            mean=float(np.mean(fitness)),
            worst=float(np.max(fitness)),
            std=float(np.std(fitness)),
            best_n_selected=int(masks[champion].sum()),
            best_macro_f1=float(scored[champion][1]),
            mean_n_selected=float(np.mean(masks.sum(axis=1))),
            evaluations=evaluations,
            distinct_fitted=evaluations,
            seconds=time.perf_counter() - generation_started,
        )
        generations.append(record)
        if progress is not None:
            progress(record)
        if index == config.generations - 1:
            break

        order = np.argsort(fitness, kind="stable")
        next_masks = [masks[position].copy() for position in order[: config.elitism]]
        next_genes = [genes[position].copy() for position in order[: config.elitism]]
        while len(next_masks) < config.population_size:
            first = _tournament(fitness, config.tournament_size, rng)
            second = _tournament(fitness, config.tournament_size, rng)
            child_mask, _ = _crossover(
                masks[first], masks[second], config.crossover_rate, rng
            )
            # Arithmetic (blend) crossover on the weight genes: a uniform swap
            # would only ever recombine weights the population already holds,
            # and a simplex is continuous.
            blend = rng.random()
            child_gene = blend * genes[first] + (1.0 - blend) * genes[second]
            child_gene = np.clip(
                child_gene + rng.normal(0.0, config.weight_mutation_sigma, n_members),
                1e-6,
                None,
            )
            next_masks.append(
                repair_mask(
                    _mutate(child_mask, config.mutation_rate, rng),
                    min_features=config.min_features,
                    rng=rng,
                )
            )
            next_genes.append(child_gene)
        masks = np.asarray(next_masks, dtype=bool)
        genes = np.asarray(next_genes, dtype=float)

    ledger.assert_disjoint_from(fold.test_index)
    normalised = np.asarray(best_gene, dtype=float)
    normalised = normalised / float(normalised.sum())
    return GAResult(
        method="ga_joint",
        model_id="+".join(config.members),
        outer_label=str(fold.label),
        task=str(data.task),
        best_mask=best_mask,
        best_fitness=best_fitness,
        best_macro_f1=best_macro,
        best_metrics=best_metrics,
        generations=generations,
        config=config.as_dict(),
        evaluator={
            "members": list(config.members),
            "outer_fold": str(fold.label),
            "n_inner_splits": len(splits),
            "evaluations": evaluations,
            "seed": int(seed),
            "outer_test_rows_touched": len(
                ledger.touched & frozenset(np.asarray(fold.test_index).tolist())
            ),
        },
        seconds=time.perf_counter() - started,
        best_weights=tuple(float(value) for value in normalised),
    )
