"""SO-03b: particle swarm optimization (Phase 59).

Two swarms, because T59.1 and T59.2 ask two different questions:

* **Binary PSO over the 138-bit feature mask** (T59.2) -- the alternative to the
  GA, and the arm T59.5 compares against it at identical folds, budget and seed.
  Kennedy-Eberhart binary PSO: velocity is continuous, position is a Bernoulli
  draw with p = sigmoid(velocity).
* **Continuous PSO over the ensemble-weight simplex** (T59.1) -- a different
  search space entirely, over cached out-of-fold member probabilities.

WHY THE WEIGHT SWARM IS CHEAP AND THE MASK SWARM IS NOT
--------------------------------------------------------
Changing a feature mask changes what every member has to be refitted on, so a
mask evaluation costs a full inner-CV fit. Changing a WEIGHT does not: with the
mask fixed, the members' out-of-fold probabilities are fixed too, and every
weight vector in the swarm is a different average of the same cached arrays.
So the members are fitted once, up front, and the entire swarm runs as numpy
arithmetic afterwards -- 600 particle evaluations for the cost of one ensemble
fit. That is also why the weight swarm can afford the real M7 member set (M3,
M4, M5) while the joint GA in Phase 58 could not.

The same seeded-Generator discipline as the GA: one generator, no global state.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.optimization.genetic import GAResult, Generation
from src.optimization.masks import MaskEvaluator, repair_mask

__all__ = [
    "PSOError",
    "PSOConfig",
    "WeightSwarmConfig",
    "WeightSwarmResult",
    "load_pso_config",
    "load_weight_config",
    "run_binary_pso",
    "run_weight_pso",
    "member_oof_probabilities",
]


class PSOError(RuntimeError):
    """The swarm cannot be configured or run as asked."""


@dataclass(frozen=True)
class PSOConfig:
    """Swarm size, iterations and the three coefficients (T59.3)."""

    swarm_size: int = 30
    iterations: int = 25
    inertia: float = 0.729
    cognitive: float = 1.49445
    social: float = 1.49445
    velocity_clamp: float = 4.0
    min_features: int = 10

    def __post_init__(self) -> None:
        if self.swarm_size < 2:
            raise PSOError("swarm_size must be at least 2")
        if self.iterations < 1:
            raise PSOError("iterations must be at least 1")
        if self.velocity_clamp <= 0:
            raise PSOError("velocity_clamp must be positive")
        for name in ("inertia", "cognitive", "social"):
            if float(getattr(self, name)) < 0:
                raise PSOError(name + " must be non-negative")

    @property
    def max_evaluations(self) -> int:
        return int(self.swarm_size * self.iterations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "swarm_size": int(self.swarm_size),
            "iterations": int(self.iterations),
            "inertia": float(self.inertia),
            "cognitive": float(self.cognitive),
            "social": float(self.social),
            "velocity_clamp": float(self.velocity_clamp),
            "min_features": int(self.min_features),
            "max_evaluations": self.max_evaluations,
        }


def load_pso_config(config: dict[str, Any] | None = None) -> PSOConfig:
    """`optimization.particle_swarm` from configs/models.yaml."""
    if config is None:
        from src.utils.config import load_config

        config = load_config("models").get("optimization.particle_swarm") or {}
    settings = dict(config)
    return PSOConfig(
        swarm_size=int(settings.get("swarm_size", 30)),
        iterations=int(settings.get("iterations", 25)),
        inertia=float(settings.get("inertia", 0.729)),
        cognitive=float(settings.get("cognitive", 1.49445)),
        social=float(settings.get("social", 1.49445)),
        velocity_clamp=float(settings.get("velocity_clamp", 4.0)),
        min_features=int(settings.get("min_features", 10)),
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    # Clipped before the exponential, not after: exp(710) overflows to inf and
    # numpy warns, and the clipped value is numerically identical to sigmoid's
    # limit anyway.
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


# ---------------------------------------------------------------------------
# T59.2 -- binary PSO over the feature mask
# ---------------------------------------------------------------------------


def run_binary_pso(
    evaluator: MaskEvaluator,
    config: PSOConfig | None = None,
    *,
    seed: int | None = None,
    method: str = "pso",
    progress: Any = None,
) -> GAResult:
    """Kennedy-Eberhart binary PSO over the feature mask. Minimises J.

    Returns the same :class:`~src.optimization.genetic.GAResult` the GA returns,
    with iterations recorded in the ``generation`` column. That is deliberate:
    T59.5 compares the two traces, and a comparison whose two arms carried
    different result types would need a translation layer that could quietly
    disagree with itself.
    """
    config = config or load_pso_config()
    if seed is None:
        seed = evaluator.seed
    rng = np.random.default_rng(int(seed))

    n_features = evaluator.n_features
    size = config.swarm_size
    clamp = float(config.velocity_clamp)

    # Velocities start small and symmetric so the first positions are near-fair
    # coin flips; a large initial velocity would pin particles at all-on or
    # all-off before the swarm has scored anything.
    velocity = rng.uniform(-1.0, 1.0, size=(size, n_features))
    position = np.asarray(
        [
            repair_mask(
                rng.random(n_features) < _sigmoid(velocity[index]),
                min_features=config.min_features,
                rng=rng,
            )
            for index in range(size)
        ],
        dtype=bool,
    )

    personal_best = position.copy()
    personal_fitness = np.full(size, np.inf)
    global_best = position[0].copy()
    global_fitness = float("inf")
    global_macro = float("nan")
    global_metrics: dict[str, float] = {}

    started = time.perf_counter()
    iterations: list[Generation] = []

    for index in range(config.iterations):
        iteration_started = time.perf_counter()
        scored = [evaluator.fitness(individual) for individual in position]
        fitness = np.asarray([item.fitness for item in scored], dtype=float)

        improved = fitness < personal_fitness
        personal_fitness = np.where(improved, fitness, personal_fitness)
        personal_best[improved] = position[improved]

        champion = int(np.argmin(fitness))
        if fitness[champion] < global_fitness:
            global_fitness = float(fitness[champion])
            global_best = position[champion].copy()
            global_macro = float(scored[champion].macro_f1)
            global_metrics = dict(scored[champion].metrics)

        record = Generation(
            index=index,
            best=float(np.min(fitness)),
            best_so_far=float(global_fitness),
            mean=float(np.mean(fitness)),
            worst=float(np.max(fitness)),
            std=float(np.std(fitness)),
            best_n_selected=int(scored[champion].n_selected),
            best_macro_f1=float(scored[champion].macro_f1),
            mean_n_selected=float(np.mean(position.sum(axis=1))),
            evaluations=int(evaluator.n_calls),
            distinct_fitted=int(evaluator.n_fitted),
            seconds=time.perf_counter() - iteration_started,
        )
        iterations.append(record)
        if progress is not None:
            progress(record)

        if index == config.iterations - 1:
            break

        r_cognitive = rng.random((size, n_features))
        r_social = rng.random((size, n_features))
        velocity = (
            config.inertia * velocity
            + config.cognitive * r_cognitive * (personal_best.astype(float) - position)
            + config.social * r_social * (global_best.astype(float) - position)
        )
        # The clamp is what keeps binary PSO exploring: sigmoid(4) is 0.982, so
        # an unclamped velocity drives the flip probability to 0 or 1 and the
        # particle stops moving for the rest of the run.
        velocity = np.clip(velocity, -clamp, clamp)
        drawn = rng.random((size, n_features)) < _sigmoid(velocity)
        position = np.asarray(
            [
                repair_mask(drawn[row], min_features=config.min_features, rng=rng)
                for row in range(size)
            ],
            dtype=bool,
        )

    evaluator.assert_no_outer_leakage()
    return GAResult(
        method=method,
        model_id=evaluator.model_id,
        outer_label=str(evaluator.fold.label),
        task=str(evaluator.data.task),
        best_mask=global_best,
        best_fitness=global_fitness,
        best_macro_f1=global_macro,
        best_metrics=global_metrics,
        generations=iterations,
        config=config.as_dict(),
        evaluator=evaluator.as_dict(),
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# T59.1 -- continuous PSO over the ensemble-weight simplex
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightSwarmConfig:
    """The weight swarm's own budget and member set."""

    members: tuple[str, ...] = ("M3", "M4", "M5")
    swarm_size: int = 20
    iterations: int = 30
    inertia: float = 0.729
    cognitive: float = 1.49445
    social: float = 1.49445

    def as_dict(self) -> dict[str, Any]:
        return {
            "members": list(self.members),
            "swarm_size": int(self.swarm_size),
            "iterations": int(self.iterations),
            "inertia": float(self.inertia),
            "cognitive": float(self.cognitive),
            "social": float(self.social),
            "max_evaluations": int(self.swarm_size * self.iterations),
        }


def load_weight_config(config: dict[str, Any] | None = None) -> WeightSwarmConfig:
    """`optimization.particle_swarm.weight_simplex`, with the swarm's coefficients."""
    if config is None:
        from src.utils.config import load_config

        loaded = load_config("models").get("optimization.particle_swarm") or {}
    else:
        loaded = dict(config)
    simplex = dict(loaded.get("weight_simplex") or {})
    return WeightSwarmConfig(
        members=tuple(simplex.get("members", ["M3", "M4", "M5"])),
        swarm_size=int(simplex.get("swarm_size", 20)),
        iterations=int(simplex.get("iterations", 30)),
        inertia=float(loaded.get("inertia", 0.729)),
        cognitive=float(loaded.get("cognitive", 1.49445)),
        social=float(loaded.get("social", 1.49445)),
    )


def member_oof_probabilities(
    data: Any,
    fold: Any,
    members: Sequence[str],
    *,
    n_inner_splits: int = 3,
    seed: int | None = None,
    columns: Sequence[int] | None = None,
    pipeline_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any]]:
    """Fit every member on every inner training fold; return out-of-fold probabilities.

    Returns ``(stack, y_out_of_fold, ledger, detail)`` where ``stack`` is
    ``(n_members, n_out_of_fold_rows, n_classes)``. Every row is predicted by a
    model that was not trained on it, which is what makes the weights that come
    out of the swarm usable rather than fitted to their own training scores.
    """
    from sklearn.base import clone

    from src.models import estimators as est
    from src.models.pipeline import build_pipeline
    from src.optimization.base import RowLedger, inner_folds

    if seed is None:
        from src.utils.seed import GLOBAL_SEED

        seed = GLOBAL_SEED

    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    picked = (
        np.arange(features.shape[1], dtype=int)
        if columns is None
        else np.sort(np.asarray(columns, dtype=int))
    )
    splits = inner_folds(fold, data.y, data.groups, n_splits=int(n_inner_splits), seed=seed)
    ledger = RowLedger()

    per_member: list[list[np.ndarray]] = [[] for _ in members]
    truth: list[np.ndarray] = []
    started = time.perf_counter()
    for split in splits:
        ledger.record(split.train_index, split.val_index)
        x_train = features[np.ix_(split.train_index, picked)]
        y_train = targets[split.train_index]
        x_val = features[np.ix_(split.val_index, picked)]
        truth.append(targets[split.val_index])
        for position, member in enumerate(members):
            pipeline = build_pipeline(
                clone(est.build_estimator(member)),
                config=pipeline_config,
                y=y_train,
                n_features=int(picked.size),
            )
            pipeline.fit(x_train, y_train)
            per_member[position].append(
                np.asarray(pipeline.predict_proba(x_val), dtype=float)
            )
    seconds = time.perf_counter() - started

    stack = np.stack([np.concatenate(rows, axis=0) for rows in per_member])
    y_oof = np.concatenate(truth)
    detail = {
        "members": list(members),
        "outer_fold": str(fold.label),
        "n_inner_splits": len(splits),
        "n_out_of_fold_rows": int(y_oof.size),
        "n_features": int(picked.size),
        "fit_seconds": float(seconds),
        "seed": int(seed),
    }
    return stack, y_oof, ledger, detail


@dataclass
class WeightSwarmResult:
    """The weight vector the swarm converged on, and its per-iteration trace."""

    members: tuple[str, ...]
    outer_label: str
    task: str
    best_weights: tuple[float, ...]
    best_score: float
    objective: str
    metrics: dict[str, float] = field(default_factory=dict)
    iterations: list[Generation] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def trace_frame(self) -> Any:
        import pandas as pd

        frame = pd.DataFrame([item.as_row() for item in self.iterations])
        frame.insert(0, "outer_fold", self.outer_label)
        frame.insert(0, "members", "+".join(self.members))
        frame.insert(0, "method", "pso_weights")
        return frame

    def as_summary(self) -> dict[str, Any]:
        return {
            "method": "pso_weights",
            "members": list(self.members),
            "task": self.task,
            "outer_fold": self.outer_label,
            "objective": self.objective,
            "best_weights": [float(value) for value in self.best_weights],
            "weights_sum": float(sum(self.best_weights)),
            "best_score": float(self.best_score),
            "metrics": {key: float(value) for key, value in self.metrics.items()},
            "config": dict(self.config),
            "detail": dict(self.detail),
            "seconds": float(self.seconds),
        }


def run_weight_pso(
    data: Any,
    fold: Any,
    config: WeightSwarmConfig | None = None,
    *,
    n_inner_splits: int = 3,
    seed: int | None = None,
    objective: str | None = None,
    columns: Sequence[int] | None = None,
    progress: Any = None,
) -> WeightSwarmResult:
    """PSO over the weight simplex, on cached out-of-fold member probabilities.

    MAXIMISES the configured classification objective -- unlike the mask
    searches, which minimise J. A weight vector changes neither the feature
    count nor the extraction cost, so two of J's three terms are constant across
    the whole search space and including them would only shift every score by
    the same amount.

    Weights are kept on the simplex by clipping to non-negative and renormalising
    after every move (T60.3 states the same constraint for SO-05). Clipping
    rather than a penalty: a negative weight is not a worse ensemble, it is not
    an ensemble at all.
    """
    from src.ensemble.soft_voting import fuse_probabilities
    from src.evaluation import metrics as mt
    from src.optimization.base import objective_for

    config = config or load_weight_config()
    if seed is None:
        from src.utils.seed import GLOBAL_SEED

        seed = GLOBAL_SEED
    rng = np.random.default_rng(int(seed))
    scorer = objective_for(str(data.task), name=objective)

    stack, y_oof, ledger, detail = member_oof_probabilities(
        data, fold, config.members,
        n_inner_splits=n_inner_splits, seed=seed, columns=columns,
    )
    labels = tuple(np.unique(np.asarray(data.y)).tolist())
    n_members = len(config.members)

    def score(vector: np.ndarray) -> float:
        fused = fuse_probabilities(stack, vector)
        predicted = np.asarray([labels[int(p)] for p in np.argmax(fused, axis=1)])
        return float(scorer(y_oof, predicted, labels))

    def normalise(matrix: np.ndarray) -> np.ndarray:
        clipped = np.clip(matrix, 0.0, None)
        totals = clipped.sum(axis=1, keepdims=True)
        # A particle that lands on all-zeros has no ensemble to describe; equal
        # weights is the neutral point to send it back to, not a random one.
        clipped = np.where(totals > 0, clipped, np.ones_like(clipped))
        return clipped / clipped.sum(axis=1, keepdims=True)

    position = normalise(rng.random((config.swarm_size, n_members)))
    velocity = rng.uniform(-0.1, 0.1, size=(config.swarm_size, n_members))
    personal_best = position.copy()
    personal_score = np.full(config.swarm_size, -np.inf)
    global_best = position[0].copy()
    global_score = -np.inf

    started = time.perf_counter()
    trace: list[Generation] = []
    for index in range(config.iterations):
        iteration_started = time.perf_counter()
        scores = np.asarray([score(row) for row in position], dtype=float)

        improved = scores > personal_score
        personal_score = np.where(improved, scores, personal_score)
        personal_best[improved] = position[improved]

        champion = int(np.argmax(scores))
        if scores[champion] > global_score:
            global_score = float(scores[champion])
            global_best = position[champion].copy()

        trace.append(
            Generation(
                index=index,
                best=float(np.max(scores)),
                best_so_far=float(global_score),
                mean=float(np.mean(scores)),
                worst=float(np.min(scores)),
                std=float(np.std(scores)),
                best_n_selected=n_members,
                best_macro_f1=float("nan"),
                mean_n_selected=float(n_members),
                evaluations=int((index + 1) * config.swarm_size),
                distinct_fitted=0,
                seconds=time.perf_counter() - iteration_started,
            )
        )
        if progress is not None:
            progress(trace[-1])
        if index == config.iterations - 1:
            break

        r_cognitive = rng.random((config.swarm_size, n_members))
        r_social = rng.random((config.swarm_size, n_members))
        velocity = (
            config.inertia * velocity
            + config.cognitive * r_cognitive * (personal_best - position)
            + config.social * r_social * (global_best - position)
        )
        position = normalise(position + velocity)

    ledger.assert_disjoint_from(fold.test_index)
    fused = fuse_probabilities(stack, global_best)
    predicted = np.asarray([labels[int(p)] for p in np.argmax(fused, axis=1)])
    report = (
        mt.binary_metrics(y_oof, predicted, labels=list(labels), positive_label=labels[-1])
        if len(labels) == 2
        else mt.multiclass_metrics(y_oof, predicted, labels=list(labels))
    )
    return WeightSwarmResult(
        members=tuple(config.members),
        outer_label=str(fold.label),
        task=str(data.task),
        best_weights=tuple(float(value) for value in global_best),
        best_score=float(global_score),
        objective=scorer.name,
        metrics={
            key: float(value)
            for key, value in report.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        iterations=trace,
        config=config.as_dict(),
        detail={
            **detail,
            "outer_test_rows_touched": len(
                ledger.touched & frozenset(np.asarray(fold.test_index).tolist())
            ),
        },
        seconds=time.perf_counter() - started,
    )
