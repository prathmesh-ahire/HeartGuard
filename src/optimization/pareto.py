"""SO-06: the performance-versus-complexity Pareto front (Phase 61).

J collapses three quantities into one number, and the weighting that does the
collapsing is a judgement. This module takes the judgement out of the picture in
two complementary ways:

* **The Pareto front** (T61.4) -- the configurations that no other configuration
  beats on *every* axis at once. It needs no alpha, beta or gamma at all, so it
  is the part of SO-06 that is a measurement rather than a preference.
* **The weighting sweep** (T61.4) -- every (alpha, beta, gamma) on a simplex
  lattice, recording which configuration each one would select. That turns "the
  weights are a judgement" into a map showing exactly which judgements lead
  where, and how large a region of weight space picks the shipped subset.

WHAT THE OPERATING POINT IS, AND WHY IT IS NOT RE-DERIVED HERE
---------------------------------------------------------------
T61.5 asks for the final operating point on the front. It is the subset SO-04
already ships (T57.5) -- not a fresh choice. Two selections of "the final feature
subset" that could disagree would be one selection too many, and the SO-04 choice
is the one FE-12, the A8 ablation and Phase 65 all point at. SO-06's job is to
show *where that point sits* on the front and how much of weight space agrees
with it, not to relitigate it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.optimization import multi_objective as mo

__all__ = [
    "ParetoError",
    "OBJECTIVE_SENSE",
    "configuration_frame",
    "is_dominated",
    "pareto_front",
    "weighting_grid",
    "sweep_weightings",
    "operating_point",
]


class ParetoError(RuntimeError):
    """The front cannot be traced from what was given."""


#: column -> +1 if larger is better, -1 if smaller is better.
OBJECTIVE_SENSE: dict[str, int] = {
    "macro_f1": +1,
    "n_selected": -1,
    "normalized_inference_time": -1,
}


def configuration_frame(sweep: Any) -> Any:
    """Collapse the per-fold sweep to one row per (ranker, k).

    The front is a statement about configurations, not about fold evaluations, so
    the fold axis is averaged out first. The standard error comes along because
    a front drawn through points whose spread exceeds their separation is a front
    through noise, and a reader has to be able to see that.
    """
    required = {"ranker", "k", "macro_f1", "n_selected", "families"}
    missing = required - set(sweep.columns)
    if missing:
        raise ParetoError(
            "the sweep table is missing " + ", ".join(sorted(missing))
        )

    grouped = sweep.groupby(["ranker", "k"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        sensitivity=("sensitivity", "mean"),
        specificity=("specificity", "mean"),
        n_selected=("n_selected", "mean"),
        n_evaluations=("macro_f1", "size"),
    )
    grouped["macro_f1_se"] = grouped["macro_f1_std"] / np.sqrt(
        grouped["n_evaluations"].clip(lower=1)
    )

    # The families a configuration needs are a property of the subset, and the
    # subset is re-derived per fold, so a configuration can need different
    # families on different folds. The UNION is the honest charge: a deployed
    # extractor has to be able to serve whichever subset the fold map produced.
    families = (
        sweep.groupby(["ranker", "k"])["families"]
        .apply(lambda values: ";".join(sorted({f for v in values for f in str(v).split(";")})))
        .reset_index(name="families")
    )
    grouped = grouped.merge(families, on=["ranker", "k"], how="left")

    cost_model = mo.load_cost_model()
    grouped["normalized_inference_time"] = grouped["families"].map(
        lambda value: cost_model.normalized(str(value).split(";"))
    )
    grouped["normalized_features"] = grouped["n_selected"] / float(
        mo.load_weights().n_features_total
    )
    return grouped


# ---------------------------------------------------------------------------
# T61.4 -- the front
# ---------------------------------------------------------------------------


def is_dominated(
    candidate: Sequence[float], others: Any, sense: Sequence[int]
) -> bool:
    """Whether some row of ``others`` is at least as good everywhere and better somewhere."""
    point = np.asarray(candidate, dtype=float) * np.asarray(sense, dtype=float)
    matrix = np.asarray(others, dtype=float) * np.asarray(sense, dtype=float)
    at_least_as_good = np.all(matrix >= point, axis=1)
    strictly_better = np.any(matrix > point, axis=1)
    return bool(np.any(at_least_as_good & strictly_better))


def pareto_front(frame: Any, objectives: Sequence[str] | None = None) -> Any:
    """Mark the non-dominated configurations. Needs no weighting at all.

    Returns the frame with an ``on_front`` boolean column added, so the dominated
    points stay in the table -- a front plotted without the cloud it was drawn
    from is a picture of a conclusion rather than of the evidence.
    """
    columns = list(objectives or OBJECTIVE_SENSE)
    unknown = [name for name in columns if name not in OBJECTIVE_SENSE]
    if unknown:
        raise ParetoError(
            "no optimisation direction declared for " + ", ".join(unknown)
        )
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ParetoError("the frame has no " + ", ".join(missing) + " column")

    sense = [OBJECTIVE_SENSE[name] for name in columns]
    matrix = frame[columns].to_numpy(dtype=float)
    on_front = [
        not is_dominated(matrix[index], np.delete(matrix, index, axis=0), sense)
        for index in range(matrix.shape[0])
    ]
    out = frame.copy()
    out["on_front"] = on_front
    return out


# ---------------------------------------------------------------------------
# T61.4 -- the weighting sweep
# ---------------------------------------------------------------------------


def weighting_grid(resolution: float = 0.05) -> np.ndarray:
    """Every (alpha, beta, gamma) on a lattice with alpha + beta + gamma = 1.

    Normalised to sum to 1 so the sweep is over the *ratio* between the three
    terms, which is the only thing that changes which configuration wins: scaling
    all three by a constant scales J by that constant and reorders nothing.
    """
    from src.ensemble.soft_voting import simplex_grid

    return simplex_grid(3, float(resolution))


def sweep_weightings(
    frame: Any,
    grid: np.ndarray | None = None,
    *,
    n_standard_errors: float | None = None,
) -> Any:
    """For every weighting, which configuration would J select?

    ``n_standard_errors`` applies the same performance guard T57.5 ships under, so
    the sweep can be read both ways: what raw J does with each weighting, and what
    J-under-the-guard does. The difference between the two maps is the clearest
    statement of what the guard is actually buying.
    """
    import pandas as pd

    grid = weighting_grid() if grid is None else np.asarray(grid, dtype=float)
    performance = frame["macro_f1"].to_numpy(dtype=float)
    features = frame["normalized_features"].to_numpy(dtype=float)
    inference = frame["normalized_inference_time"].to_numpy(dtype=float)

    eligible = np.ones(performance.shape, dtype=bool)
    guard_value = float("nan")
    if n_standard_errors is not None:
        best = int(np.argmax(performance))
        guard_value = float(
            performance[best] - float(n_standard_errors) * float(frame["macro_f1_se"].iloc[best])
        )
        eligible = performance >= guard_value

    rows = []
    for alpha, beta, gamma in grid:
        j = alpha * (1.0 - performance) + beta * features + gamma * inference
        masked = np.where(eligible, j, np.inf)
        winner = int(np.argmin(masked))
        rows.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "selected_ranker": str(frame["ranker"].iloc[winner]),
                "selected_k": int(frame["k"].iloc[winner]),
                "selected_macro_f1": float(performance[winner]),
                "selected_n_selected": float(frame["n_selected"].iloc[winner]),
                "selected_normalized_inference_time": float(inference[winner]),
                "j": float(masked[winner]),
                "guarded": n_standard_errors is not None,
                "performance_floor": guard_value,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T61.5 -- the operating point
# ---------------------------------------------------------------------------


def operating_point(
    frame: Any,
    ranker: str,
    k: int,
    *,
    sweep: Any | None = None,
) -> dict[str, Any]:
    """Where the shipped subset sits on the front, and how much of weight space picks it.

    ``share_of_weight_space`` is the fraction of sampled (alpha, beta, gamma)
    that select this configuration. It is the honest way to say how load-bearing
    the weighting is: a point chosen by 3% of weight space is a point chosen by
    the weights, and a point chosen by 60% of it is a point chosen by the data.
    """
    match = frame[(frame["ranker"] == ranker) & (frame["k"] == int(k))]
    if match.empty:
        raise ParetoError(
            "the shipped configuration " + str(ranker) + " k=" + str(k)
            + " is not in the sweep; SO-06 cannot place a point it has no evidence for"
        )
    row = match.iloc[0]
    detail: dict[str, Any] = {
        "ranker": str(ranker),
        "k": int(k),
        "macro_f1": float(row["macro_f1"]),
        "macro_f1_se": float(row["macro_f1_se"]),
        "n_selected": float(row["n_selected"]),
        "normalized_inference_time": float(row["normalized_inference_time"]),
        "families": str(row["families"]),
        "on_pareto_front": bool(row["on_front"]) if "on_front" in row else None,
        "source": "SO-04 T57.5; SO-06 places it, it does not re-choose it",
    }
    if sweep is not None and len(sweep):
        picked = (sweep["selected_ranker"] == ranker) & (sweep["selected_k"] == int(k))
        detail["share_of_weight_space"] = float(picked.mean())
        detail["n_weightings_sampled"] = len(sweep)
    return detail
