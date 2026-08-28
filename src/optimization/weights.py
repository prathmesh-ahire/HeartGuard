"""SO-05: ensemble weight optimization on out-of-fold probabilities (Phase 60).

Three optimizers over the same three-member simplex, scored on the same
out-of-fold probabilities, so the comparison is about the optimizer and nothing
else:

* **`grid`** (T60.1) -- an exhaustive lattice over the simplex. Deterministic on
  every machine, and the baseline the other two have to beat.
* **`slsqp`** (T60.2) -- SciPy constrained minimization of the objective itself.
* **`slsqp_logloss`** (T60.2) -- SciPy on a *smooth surrogate*, because the
  objective itself is not differentiable. See below.

WHY THERE ARE TWO SciPy VARIANTS
--------------------------------
Balanced accuracy is a step function of the weights: moving a weight changes the
fused probabilities continuously, but the *score* only changes when some record
crosses the threshold. Between crossings the gradient is exactly zero, so a
gradient method evaluates a numerical derivative of 0 in every direction and
reports convergence without having moved. `configs/models.yaml` already records
this as the reason M7's weights are grid-searched rather than optimized.

Running SLSQP on it anyway is worth doing once, because "it stalls" should be a
measurement in this repository rather than a claim in a comment -- `slsqp`
exists to produce that number. `slsqp_logloss` is the honest use of a gradient
method: log loss over the fused probabilities IS differentiable in the weights,
so SLSQP can actually descend it, and the threshold is tuned afterwards. It
optimizes a different quantity than it is scored on, which is a real limitation
and is reported as one.

FOLD SAFETY (T60.4)
-------------------
Every weight here is chosen on **out-of-fold probabilities from the inner splits
of one outer fold's training rows**. No member is ever asked to predict a row it
was trained on, and no outer test row is touched -- measured by the row ledger
that `member_oof_probabilities` returns, not assumed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.ensemble.soft_voting import (
    OBJECTIVES,
    fuse_probabilities,
    objective_standard_error,
    select_threshold,
    simplex_grid,
)

__all__ = [
    "WeightSearchError",
    "WeightScore",
    "WeightResult",
    "METHODS",
    "score_weights",
    "equal_weights",
    "normalize_weights",
    "candidate_grid",
    "optimize_grid",
    "optimize_slsqp",
    "optimize_fold",
    "stability_frame",
]

#: method name -> human description, for the emitted tables.
METHODS: dict[str, str] = {
    "equal": "equal weights (M6 baseline, not optimized)",
    "grid": "exhaustive simplex lattice (T60.1)",
    "slsqp": "SciPy SLSQP on the objective itself (T60.2)",
    "slsqp_logloss": "SciPy SLSQP on a differentiable log-loss surrogate (T60.2)",
}


class WeightSearchError(RuntimeError):
    """The weight search cannot be run or would not mean what it says."""


# ---------------------------------------------------------------------------
# T60.3 -- the constraint
# ---------------------------------------------------------------------------


def normalize_weights(weights: Any) -> np.ndarray:
    """Project onto the simplex: clip negatives to zero, then divide by the sum.

    **The documented normalization (T60.3).** Clipping rather than penalising,
    because a negative weight is not a worse ensemble -- it is a member voting
    against itself, which is not an object this study is about. An all-zero
    vector is sent to equal weights rather than raising: it is the neutral point,
    and it is the only choice that does not silently privilege one member.
    """
    vector = np.clip(np.asarray(weights, dtype=float).ravel(), 0.0, None)
    total = float(vector.sum())
    if total <= 0:
        vector = np.ones_like(vector)
        total = float(vector.sum())
    return vector / total


def equal_weights(n_members: int) -> np.ndarray:
    """The M6 baseline: every member weighted the same."""
    if n_members < 1:
        raise WeightSearchError("an ensemble needs at least one member")
    return np.full(int(n_members), 1.0 / float(n_members), dtype=float)


# ---------------------------------------------------------------------------
# scoring one weight vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightScore:
    """One weight vector's score, and the threshold it was scored at."""

    score: float
    threshold: float
    sensitivity: float
    specificity: float


def score_weights(
    oof: np.ndarray,
    y_true: Any,
    weights: Any,
    *,
    objective: str = "balanced_accuracy",
    positive_label: Any = 1,
) -> WeightScore:
    """Fuse under ``weights`` and score at the best threshold for those weights.

    Jointly with the threshold, for the reason Phase 50 established: the best
    weights at a fixed 0.5 are not the best weights at the threshold the model
    will actually use, and on an imbalanced task the two differ.
    """
    if objective not in OBJECTIVES:
        raise WeightSearchError(
            "unknown objective " + repr(objective) + "; expected one of "
            + ", ".join(sorted(OBJECTIVES))
        )
    stack = np.asarray(oof, dtype=float)
    truth = np.asarray(y_true)
    labels = tuple(np.unique(truth).tolist())
    if len(labels) != 2:
        raise WeightSearchError(
            "SO-05 as specified is a binary weight search; got " + str(len(labels)) + " classes"
        )
    column = int(list(labels).index(positive_label))

    fused = fuse_probabilities(stack, normalize_weights(weights))[:, column]
    choice = select_threshold(
        truth, fused, objective=objective, positive_label=positive_label
    )
    return WeightScore(
        score=float(OBJECTIVES[objective](choice.sensitivity, choice.specificity)),
        threshold=float(choice.threshold),
        sensitivity=float(choice.sensitivity),
        specificity=float(choice.specificity),
    )


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------


@dataclass
class WeightResult:
    """One optimizer's answer on one outer fold."""

    method: str
    members: tuple[str, ...]
    outer_label: str
    weights: tuple[float, ...]
    score: float
    threshold: float
    sensitivity: float
    specificity: float
    objective: str
    n_candidates: int = 0
    n_iterations: int = 0
    seconds: float = 0.0
    moved_from_start: float = float("nan")
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def weights_sum(self) -> float:
        return float(sum(self.weights))

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": self.method,
            "outer_fold": self.outer_label,
            "objective": self.objective,
            "score": float(self.score),
            "threshold": float(self.threshold),
            "sensitivity": float(self.sensitivity),
            "specificity": float(self.specificity),
            "weights_sum": self.weights_sum,
            "n_candidates": int(self.n_candidates),
            "n_iterations": int(self.n_iterations),
            "moved_from_start": float(self.moved_from_start),
            "seconds": float(self.seconds),
        }
        for member, weight in zip(self.members, self.weights, strict=True):
            row["w_" + member] = float(weight)
        return row


# ---------------------------------------------------------------------------
# T60.1 -- the deterministic grid
# ---------------------------------------------------------------------------


def candidate_grid(n_members: int, resolution: float = 0.05) -> np.ndarray:
    """The simplex lattice, with the exact equal-weight vector guaranteed present.

    **A shrinkage rule needs its shrinkage target inside the candidate set**, and
    the lattice does not contain it. With three members equal weights is
    (1/3, 1/3, 1/3), and 1/3 is not a multiple of any decimal resolution: at 0.05
    the nearest lattice point is (0.30, 0.35, 0.35), 0.0408 away, and refining to
    0.01 only closes it to 0.0082. It is never exactly representable, at any
    resolution, for any member count that does not divide 1 evenly.

    Left alone, a one-standard-error rule that claims to "take the candidate
    closest to equal weights" can never actually return equal weights -- it
    returns a neighbour that is a different ensemble and, measured on D1 fold 0,
    scores marginally *worse* than the baseline it was shrinking toward. So the
    exact vector is prepended here. It costs one extra candidate.
    """
    lattice = simplex_grid(int(n_members), float(resolution))
    centre = equal_weights(int(n_members))
    if np.isclose(np.linalg.norm(lattice - centre, axis=1), 0.0).any():
        return lattice
    return np.vstack([centre[None, :], lattice])


def optimize_grid(
    oof: np.ndarray,
    y_true: Any,
    members: Sequence[str],
    *,
    objective: str = "balanced_accuracy",
    positive_label: Any = 1,
    resolution: float = 0.05,
    n_standard_errors: float | None = None,
    outer_label: str = "",
) -> WeightResult:
    """Exhaustive lattice over the simplex, each point at its own best threshold.

    An integer lattice, not a float sweep, so the candidate set is byte-identical
    on every machine (research rule 5 for free). ``n_standard_errors`` applies the
    same one-standard-error rule M7 uses: among candidates within one SE of the
    best, take the one closest to equal weights, so the ensemble departs from its
    own baseline only where the evidence exceeds the noise.
    """
    started = time.perf_counter()
    grid = candidate_grid(len(members), resolution)
    scores = np.empty(grid.shape[0], dtype=float)
    sensitivities = np.empty(grid.shape[0], dtype=float)
    specificities = np.empty(grid.shape[0], dtype=float)
    thresholds = np.empty(grid.shape[0], dtype=float)

    for index, candidate in enumerate(grid):
        scored = score_weights(
            oof, y_true, candidate, objective=objective, positive_label=positive_label
        )
        scores[index] = scored.score
        sensitivities[index] = scored.sensitivity
        specificities[index] = scored.specificity
        thresholds[index] = scored.threshold

    best_index = int(np.nanargmax(scores))
    chosen_index = best_index
    detail: dict[str, Any] = {
        "resolution": float(resolution),
        "argmax_weights": np.round(grid[best_index], 4).tolist(),
        "argmax_score": float(scores[best_index]),
    }

    if n_standard_errors is not None:
        truth = np.asarray(y_true)
        n_positive = int((truth == positive_label).sum())
        n_negative = int(truth.size - n_positive)
        standard_error = objective_standard_error(
            objective,
            float(sensitivities[best_index]),
            float(specificities[best_index]),
            n_positive,
            n_negative,
        )
        margin = float(n_standard_errors) * standard_error
        within = np.flatnonzero(scores >= scores[best_index] - margin)
        # Closest to equal weights among the statistically indistinguishable.
        centre = equal_weights(len(members))
        chosen_index = int(within[np.argmin(np.linalg.norm(grid[within] - centre, axis=1))])
        detail.update(
            {
                "standard_error": float(standard_error),
                "margin": float(margin),
                "n_within_margin": int(within.size),
                "selection_rule": "one_standard_error",
            }
        )

    return WeightResult(
        method="grid",
        members=tuple(str(m) for m in members),
        outer_label=outer_label,
        weights=tuple(float(value) for value in grid[chosen_index]),
        score=float(scores[chosen_index]),
        threshold=float(thresholds[chosen_index]),
        sensitivity=float(sensitivities[chosen_index]),
        specificity=float(specificities[chosen_index]),
        objective=objective,
        n_candidates=int(grid.shape[0]),
        seconds=time.perf_counter() - started,
        moved_from_start=float(
            np.linalg.norm(grid[chosen_index] - equal_weights(len(members)))
        ),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# T60.2 -- SciPy constrained minimization
# ---------------------------------------------------------------------------


def optimize_slsqp(
    oof: np.ndarray,
    y_true: Any,
    members: Sequence[str],
    *,
    objective: str = "balanced_accuracy",
    positive_label: Any = 1,
    surrogate: str = "log_loss",
    start: Any = None,
    outer_label: str = "",
    max_iter: int = 200,
) -> WeightResult:
    """SLSQP over the simplex: non-negative weights summing to 1 (T60.2, T60.3).

    The constraint is expressed to the optimizer rather than imposed afterwards:
    ``bounds=(0, 1)`` per weight plus a linear equality ``sum(w) = 1``. That is
    what makes the search space the simplex itself, so every point SLSQP
    evaluates is a legal ensemble and the answer needs no repair.

    ``surrogate="log_loss"`` minimises cross-entropy of the fused probabilities,
    which is differentiable in the weights; the threshold is then tuned on the
    result and the reported score is the real objective. ``surrogate="objective"``
    hands SLSQP the step function directly, which is the configuration that
    demonstrates why the grid exists -- it is expected to return its starting
    point, and `moved_from_start` is how far it actually got.
    """
    from scipy.optimize import minimize

    started = time.perf_counter()
    stack = np.asarray(oof, dtype=float)
    truth = np.asarray(y_true)
    labels = tuple(np.unique(truth).tolist())
    column = int(list(labels).index(positive_label))
    n_members = len(members)
    initial = equal_weights(n_members) if start is None else normalize_weights(start)

    if surrogate == "log_loss":
        positive = (truth == positive_label).astype(float)

        def cost(vector: np.ndarray) -> float:
            fused = fuse_probabilities(stack, normalize_weights(vector))[:, column]
            clipped = np.clip(fused, 1e-12, 1.0 - 1e-12)
            return float(
                -np.mean(positive * np.log(clipped) + (1.0 - positive) * np.log(1.0 - clipped))
            )
    elif surrogate == "objective":

        def cost(vector: np.ndarray) -> float:
            return -float(
                score_weights(
                    stack, truth, vector,
                    objective=objective, positive_label=positive_label,
                ).score
            )
    else:
        raise WeightSearchError(
            "unknown surrogate " + repr(surrogate) + "; expected 'log_loss' or 'objective'"
        )

    outcome = minimize(
        cost,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_members,
        constraints=[{"type": "eq", "fun": lambda vector: float(np.sum(vector) - 1.0)}],
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    weights = normalize_weights(outcome.x)
    scored = score_weights(
        stack, truth, weights, objective=objective, positive_label=positive_label
    )
    return WeightResult(
        method="slsqp_logloss" if surrogate == "log_loss" else "slsqp",
        members=tuple(str(m) for m in members),
        outer_label=outer_label,
        weights=tuple(float(value) for value in weights),
        score=float(scored.score),
        threshold=float(scored.threshold),
        sensitivity=float(scored.sensitivity),
        specificity=float(scored.specificity),
        objective=objective,
        n_candidates=int(getattr(outcome, "nfev", 0)),
        n_iterations=int(getattr(outcome, "nit", 0)),
        seconds=time.perf_counter() - started,
        moved_from_start=float(np.linalg.norm(weights - initial)),
        detail={
            "surrogate": surrogate,
            "success": bool(outcome.success),
            "status": int(outcome.status),
            "message": str(outcome.message),
            "final_cost": float(outcome.fun),
            "start": [float(v) for v in initial],
        },
    )


# ---------------------------------------------------------------------------
# one outer fold, every method
# ---------------------------------------------------------------------------


def optimize_fold(
    data: Any,
    fold: Any,
    members: Sequence[str],
    *,
    objective: str = "balanced_accuracy",
    n_inner_splits: int = 3,
    seed: int | None = None,
    resolution: float = 0.05,
    n_standard_errors: float | None = 1.0,
    columns: Sequence[int] | None = None,
) -> tuple[list[WeightResult], dict[str, Any]]:
    """Fit the members once, then run every optimizer over the cached probabilities.

    The members are fitted **once per outer fold** and every optimizer scores the
    same arrays. That is the Phase 59 pattern and it is what makes SO-05 cheap:
    the fitting is the whole cost, the searching is arithmetic, and the three
    methods cannot differ because one of them saw different probabilities.
    """
    from src.optimization.swarm import member_oof_probabilities

    stack, y_oof, ledger, detail = member_oof_probabilities(
        data, fold, members,
        n_inner_splits=n_inner_splits, seed=seed, columns=columns,
    )
    ledger.assert_disjoint_from(fold.test_index)

    labels = tuple(np.unique(np.asarray(data.y)).tolist())
    positive = labels[-1]
    label = str(fold.label)

    baseline = score_weights(
        stack, y_oof, equal_weights(len(members)),
        objective=objective, positive_label=positive,
    )
    results = [
        WeightResult(
            method="equal",
            members=tuple(str(m) for m in members),
            outer_label=label,
            weights=tuple(float(v) for v in equal_weights(len(members))),
            score=baseline.score,
            threshold=baseline.threshold,
            sensitivity=baseline.sensitivity,
            specificity=baseline.specificity,
            objective=objective,
            n_candidates=1,
            moved_from_start=0.0,
        ),
        optimize_grid(
            stack, y_oof, members,
            objective=objective, positive_label=positive,
            resolution=resolution, n_standard_errors=n_standard_errors,
            outer_label=label,
        ),
        optimize_slsqp(
            stack, y_oof, members,
            objective=objective, positive_label=positive,
            surrogate="objective", outer_label=label,
        ),
        optimize_slsqp(
            stack, y_oof, members,
            objective=objective, positive_label=positive,
            surrogate="log_loss", outer_label=label,
        ),
    ]
    detail["outer_test_rows_touched"] = len(
        ledger.touched & frozenset(np.asarray(fold.test_index).tolist())
    )
    return results, detail


# ---------------------------------------------------------------------------
# T60.6 -- the stability check
# ---------------------------------------------------------------------------


def stability_frame(results: Sequence[WeightResult]) -> Any:
    """Per-member mean, standard deviation and range of the weights across folds.

    The variance is the point of the table, not a decoration. A weight vector
    that swings from (0.05, 0.90, 0.05) on one fold to (0.60, 0.10, 0.30) on the
    next is a search fitting fold noise, and its mean is a number no fold
    actually chose. This is what T60.6 asks to be reported, and it is what a
    reader should look at before believing any single optimized weight vector.
    """
    import pandas as pd

    if not results:
        raise WeightSearchError("no weight results to summarise")

    rows = []
    for method in sorted({item.method for item in results}):
        block = [item for item in results if item.method == method]
        members = block[0].members
        matrix = np.asarray([item.weights for item in block], dtype=float)
        scores = np.asarray([item.score for item in block], dtype=float)
        for position, member in enumerate(members):
            column = matrix[:, position]
            rows.append(
                {
                    "method": method,
                    "member": member,
                    "n_folds": len(block),
                    "mean_weight": float(np.mean(column)),
                    "std_weight": float(np.std(column, ddof=1)) if len(block) > 1 else 0.0,
                    "min_weight": float(np.min(column)),
                    "max_weight": float(np.max(column)),
                    "range_weight": float(np.max(column) - np.min(column)),
                    "mean_score": float(np.mean(scores)),
                }
            )
    return pd.DataFrame(rows)
