"""Hyperparameter search spaces, read from config rather than written in code.

A search space is data, not a literal buried in whichever script happens to run
the search. ``configs/models.yaml`` declares one per model and this module turns
it into something samplable, checkable and printable. Two consequences matter:
the space that Part VI searches is the space the write-up quotes, and a space
can be inspected without running anything.

**Not every point in a product space is a legal estimator.** Two of the four
dimensions declared for M1 interact -- ``lbfgs`` accepts only an L2 penalty, so
``solver="lbfgs"`` with ``l1_ratio=1.0`` raises rather than degrades -- and one
of M2's is conditional: ``p`` is read only when ``metric="minkowski"`` and is
silently ignored otherwise, which turns half of that space into duplicates of
the other half. Both are handled here as named constraints attached to the model
in config, not as a ``try/except`` around the fit. A search that discovers its
own space by catching exceptions reports a trial count that does not mean what
it says.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "DIMENSION_KINDS",
    "SpaceError",
    "Dimension",
    "Constraint",
    "SearchSpace",
    "CONSTRAINTS",
    "load_space",
    "describe_space",
]

log = get_logger("models.spaces")

DIMENSION_KINDS: tuple[str, ...] = (
    "uniform",
    "log_uniform",
    "int_uniform",
    "categorical",
)


class SpaceError(ValueError):
    """The declared space is malformed, or a point in it is not legal."""


# ---------------------------------------------------------------------------
# one dimension
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dimension:
    """One tunable parameter: its name, its kind, and its bounds or choices."""

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in DIMENSION_KINDS:
            raise SpaceError(
                self.name + ": unknown dimension kind " + repr(self.kind)
                + "; expected one of " + str(DIMENSION_KINDS)
            )
        if self.kind == "categorical":
            if not self.choices:
                raise SpaceError(self.name + ": categorical needs a non-empty choices list")
            return
        if self.low is None or self.high is None:
            raise SpaceError(self.name + ": " + self.kind + " needs both low and high")
        if not self.low < self.high:
            raise SpaceError(
                self.name + ": low must be below high, got "
                + str(self.low) + " and " + str(self.high)
            )
        if self.kind == "log_uniform" and self.low <= 0:
            raise SpaceError(self.name + ": log_uniform needs a strictly positive low")

    @property
    def _bounds(self) -> tuple[float, float]:
        """``low`` and ``high`` as plain floats. Only ever called off categorical.

        ``__post_init__`` has already refused a non-categorical dimension with a
        missing bound, so the assertion here is for the type checker rather than
        for a case that can occur at run time.
        """
        assert self.low is not None and self.high is not None  # noqa: S101
        return float(self.low), float(self.high)

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value. ``log_uniform`` is uniform in log space, not in value."""
        if self.kind == "categorical":
            # rng.choice would coerce a mixed list (None, 5, "sqrt") to strings.
            return self.choices[int(rng.integers(len(self.choices)))]
        low, high = self._bounds
        if self.kind == "int_uniform":
            return int(rng.integers(int(low), int(high) + 1))
        if self.kind == "uniform":
            return float(rng.uniform(low, high))
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))

    def contains(self, value: Any) -> bool:
        if self.kind == "categorical":
            return any(value == choice for choice in self.choices)
        if value is None or isinstance(value, bool):
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if self.kind == "int_uniform" and numeric != int(numeric):
            return False
        low, high = self._bounds
        return low <= numeric <= high

    def describe(self) -> str:
        if self.kind == "categorical":
            return "categorical" + str(list(self.choices))
        return self.kind + "[" + str(self.low) + ", " + str(self.high) + "]"


def _dimension_from_config(name: str, settings: Any) -> Dimension:
    if not isinstance(settings, dict):
        raise SpaceError(name + ": expected a mapping, got " + type(settings).__name__)
    kind = str(settings.get("type", "")).strip()
    choices = settings.get("choices")
    return Dimension(
        name=name,
        kind=kind,
        low=settings.get("low"),
        high=settings.get("high"),
        choices=tuple(choices) if choices is not None else (),
    )


# ---------------------------------------------------------------------------
# constraints between dimensions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    """A rule that some combinations of otherwise-valid values are not allowed.

    ``repair`` is optional and exists for the *conditional* case rather than the
    *illegal* case. M2's ``p`` off a minkowski metric is not an error -- sklearn
    accepts it and ignores it -- so rejecting the draw would be wrong; pinning it
    to its canonical value collapses the duplicates instead. M1's illegal solver
    pairing has no repair: the draw is rejected and re-drawn.
    """

    name: str
    explain: str
    check: Callable[[dict[str, Any]], bool]
    repair: Callable[[dict[str, Any]], dict[str, Any]] | None = None


def _m1_solver_penalty(params: dict[str, Any]) -> bool:
    """lbfgs is L2-only; liblinear does L1 or L2 but not elasticnet; saga does all.

    Expressed against ``l1_ratio`` because ``penalty`` was deprecated in
    scikit-learn 1.8 and is removed in 1.10. 0.0 is the old ``"l2"``, 1.0 the old
    ``"l1"``, anything between is elasticnet.
    """
    solver = params.get("solver", "lbfgs")
    ratio = params.get("l1_ratio", 0.0)
    if ratio is None:
        return True
    ratio = float(ratio)
    if solver in {"lbfgs", "newton-cg", "newton-cholesky", "sag"}:
        return ratio == 0.0
    if solver == "liblinear":
        return ratio in (0.0, 1.0)
    return 0.0 <= ratio <= 1.0


def _m2_p_only_for_minkowski(params: dict[str, Any]) -> bool:
    return params.get("metric", "minkowski") == "minkowski" or params.get("p", 2) == 2


def _m2_pin_p(params: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(params)
    if repaired.get("metric", "minkowski") != "minkowski":
        repaired["p"] = 2
    return repaired


#: Named in ``configs/models.yaml`` under each model's ``constraints`` list.
CONSTRAINTS: dict[str, Constraint] = {
    "M1_SOLVER_PENALTY": Constraint(
        name="M1_SOLVER_PENALTY",
        explain=(
            "lbfgs/newton-cg/sag accept l1_ratio 0.0 only; liblinear accepts "
            "0.0 or 1.0; saga accepts any value in [0, 1]"
        ),
        check=_m1_solver_penalty,
    ),
    "M2_P_ONLY_FOR_MINKOWSKI": Constraint(
        name="M2_P_ONLY_FOR_MINKOWSKI",
        explain=(
            "p is read only when metric is 'minkowski'; off it, p is pinned to 2 "
            "so the space does not hold duplicate configurations"
        ),
        check=_m2_p_only_for_minkowski,
        repair=_m2_pin_p,
    ),
}


# ---------------------------------------------------------------------------
# the space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSpace:
    """Every tunable parameter of one model, plus the rules linking them."""

    model_id: str
    dimensions: tuple[Dimension, ...]
    constraints: tuple[Constraint, ...] = ()

    def __len__(self) -> int:
        return len(self.dimensions)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    def dimension(self, name: str) -> Dimension:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        raise SpaceError(self.model_id + " has no dimension " + repr(name))

    # -- validity ----------------------------------------------------------

    def violations(self, params: dict[str, Any]) -> tuple[str, ...]:
        """Every reason ``params`` is not a legal point, named. Empty means legal."""
        problems: list[str] = []
        for name, value in params.items():
            if name not in self.names:
                continue
            if not self.dimension(name).contains(value):
                problems.append(
                    name + "=" + repr(value) + " outside "
                    + self.dimension(name).describe()
                )
        for constraint in self.constraints:
            if not constraint.check(params):
                problems.append(constraint.name + ": " + constraint.explain)
        return tuple(problems)

    def is_valid(self, params: dict[str, Any]) -> bool:
        return not self.violations(params)

    def repair(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply every constraint that has a canonical fix; leave the rest alone."""
        repaired = dict(params)
        for constraint in self.constraints:
            if constraint.repair is not None:
                repaired = constraint.repair(repaired)
        return repaired

    # -- sampling ----------------------------------------------------------

    def sample(
        self, rng: np.random.Generator | int | None = None, *, max_tries: int = 200
    ) -> dict[str, Any]:
        """One legal point. Repairs what can be repaired, re-draws what cannot.

        Rejection sampling with a bounded budget rather than an unbounded loop:
        a constraint that rejects everything is a bug in the space, and it should
        surface as a clear error at the first sample instead of a hang partway
        through a 200-trial search.
        """
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        for _ in range(max_tries):
            draw = {dim.name: dim.sample(generator) for dim in self.dimensions}
            draw = self.repair(draw)
            if self.is_valid(draw):
                return draw
        raise SpaceError(
            self.model_id + ": no legal point found in " + str(max_tries)
            + " draws; the constraints reject (nearly) the whole space"
        )

    def sample_many(
        self, n: int, rng: np.random.Generator | int | None = None
    ) -> tuple[dict[str, Any], ...]:
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        return tuple(self.sample(generator) for _ in range(n))

    # -- interop -----------------------------------------------------------

    def to_distributions(self) -> dict[str, Any]:
        """A ``param_distributions`` mapping for ``RandomizedSearchCV``.

        Constraints are **not** representable here -- sklearn's randomized search
        samples each dimension independently and has no notion of a rejected
        combination. Part VI's search wrapper filters with :meth:`is_valid`
        before dispatching; a caller using this mapping directly must do the
        same or it will hit a solver error mid-search.
        """
        from scipy import stats

        distributions: dict[str, Any] = {}
        for dimension in self.dimensions:
            if dimension.kind == "categorical":
                distributions[dimension.name] = list(dimension.choices)
                continue
            low, high = dimension._bounds
            if dimension.kind == "int_uniform":
                distributions[dimension.name] = stats.randint(int(low), int(high) + 1)
            elif dimension.kind == "uniform":
                distributions[dimension.name] = stats.uniform(low, high - low)
            else:
                distributions[dimension.name] = stats.loguniform(low, high)
        return distributions


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_space(model_id: str, config: dict[str, Any] | None = None) -> SearchSpace:
    """Read one model's declared space out of ``configs/models.yaml``."""
    spec = config
    if spec is None:
        from src.utils.config import load_config

        spec = load_config("models").get("models." + model_id)
    if spec is None:
        raise SpaceError("no model " + repr(model_id) + " in configs/models.yaml")

    declared = spec.get("search_space") or {}
    dimensions = tuple(
        _dimension_from_config(name, settings) for name, settings in declared.items()
    )

    names = [str(name) for name in (spec.get("constraints") or [])]
    unknown = [name for name in names if name not in CONSTRAINTS]
    if unknown:
        raise SpaceError(
            model_id + " names unknown constraint(s): " + ", ".join(unknown)
            + "; known: " + ", ".join(sorted(CONSTRAINTS))
        )
    constraints = tuple(CONSTRAINTS[name] for name in names)
    return SearchSpace(model_id=model_id, dimensions=dimensions, constraints=constraints)


def describe_space(space: SearchSpace) -> Any:
    """A one-row-per-dimension table, for the search-configuration deliverable."""
    import pandas as pd

    rows = [
        {
            "model_id": space.model_id,
            "parameter": dimension.name,
            "kind": dimension.kind,
            "low": dimension.low,
            "high": dimension.high,
            "choices": (
                "|".join(str(choice) for choice in dimension.choices)
                if dimension.choices
                else ""
            ),
        }
        for dimension in space.dimensions
    ]
    frame = pd.DataFrame(rows)
    frame.attrs["constraints"] = [
        constraint.name + ": " + constraint.explain for constraint in space.constraints
    ]
    return frame
