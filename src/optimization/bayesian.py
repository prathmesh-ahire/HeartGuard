"""SO-02 -- Bayesian optimization over the declared spaces (Phase 56).

The advanced search method this project commits to. Where SO-01 draws every
point independently, this one fits a Gaussian-process surrogate to the trials so
far and asks it where to look next, so the budget concentrates on the promising
region instead of being spread uniformly over a space whose interesting part is
a corner of it. Whether that actually pays here at equal budget is a measured
question, not an assumed one -- that is T56.3 and the user's decision at T56.5.

**Availability is checked, never assumed** (T56.1). ``scikit-optimize`` imports
cleanly against several scikit-learn versions it then fails on at ``.fit()``, so
:func:`skopt_available` runs the import *and* an ask/tell round trip before
declaring the method usable. If it is unusable the caller records the reason in
``outputs/missing_outputs_report.txt`` (T56.6) rather than quietly emitting
nothing.

**Why ``Optimizer`` and not ``BayesSearchCV``.** ``BayesSearchCV`` is a thin
wrapper that owns a CV loop; ``Optimizer`` is the ask/tell engine inside it. The
three reasons SO-01 does not use ``RandomizedSearchCV`` apply here unchanged --
constrained spaces, a loaded fold map, and a per-trial log with a wall-clock
budget -- and one more applies only here: an equal-budget comparison against
SO-01 is only meaningful if both methods count a trial the same way, which they
do because both run inside the *same* :class:`~src.optimization.base.BaseSearch`
loop over the *same* inner folds. A ``BayesSearchCV`` counting its own iterations
against our trial counter would be comparing two different units.

**Categoricals are encoded as index strings.** Two of the searched spaces mix
types inside one categorical dimension -- M4's ``max_features`` holds
``"sqrt"``, ``"log2"``, ``0.3``, ``0.5`` and its ``max_depth`` holds ``None``
beside integers. Handing those to ``skopt.space.Categorical`` puts numpy in
charge of finding a common dtype, which turns ``0.3`` into ``"0.3"`` somewhere
inside the transformer and hands the estimator a string. Encoding each choice as
its index (``"0"``, ``"1"``, ...) and decoding on the way out keeps the surrogate's
one-hot treatment of an unordered dimension while making the round trip exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from src.optimization.base import BaseSearch, Objective, Trial
from src.utils.logging_setup import get_logger

__all__ = [
    "SkoptCapability",
    "skopt_available",
    "to_skopt_dimensions",
    "encode_params",
    "decode_point",
    "BayesianSearch",
]

log = get_logger("optimization.bayesian")

#: Acquisition function, pinned. ``Optimizer``'s default is ``gp_hedge``, which
#: picks among three acquisitions by an internal bandit -- reproducible under a
#: fixed seed, but it makes "why did it look there?" unanswerable from the trial
#: log alone. Expected improvement is the one the write-up can describe.
ACQUISITION = "EI"


@dataclass(frozen=True)
class SkoptCapability:
    """Whether Bayesian optimization can actually run here, and why not if it cannot."""

    available: bool
    version: str | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"available": self.available, "version": self.version, "reason": self.reason}


def skopt_available() -> SkoptCapability:
    """Import scikit-optimize and exercise it. Importing alone proves nothing.

    ``skopt`` pins nothing about scikit-learn and reaches into private sklearn
    internals; the failure mode observed elsewhere in this stack is a clean
    import followed by an exception inside the first ``fit``. So this runs a real
    ask/tell round trip against a mixed space before answering yes.
    """
    try:
        import skopt
        from skopt import Optimizer
        from skopt.space import Categorical, Integer, Real
    except Exception as error:  # noqa: BLE001 -- any import failure is a "no"
        return SkoptCapability(
            available=False,
            reason="scikit-optimize could not be imported: "
            + type(error).__name__
            + ": "
            + str(error)[:200],
        )

    version = getattr(skopt, "__version__", None)
    try:
        probe = Optimizer(
            [Real(1e-3, 1e3, prior="log-uniform"), Integer(1, 10), Categorical(["0", "1"])],
            base_estimator="GP",
            acq_func=ACQUISITION,
            n_initial_points=2,
            random_state=42,
        )
        for value in (0.4, 0.6, 0.5):
            probe.tell(probe.ask(), value)
    except Exception as error:  # noqa: BLE001 -- a runtime failure is also a "no"
        return SkoptCapability(
            available=False,
            version=version,
            reason="scikit-optimize "
            + str(version)
            + " imports but fails at run time: "
            + type(error).__name__
            + ": "
            + str(error)[:200],
        )
    return SkoptCapability(available=True, version=version)


# ---------------------------------------------------------------------------
# T56.2 -- space conversion
# ---------------------------------------------------------------------------


def to_skopt_dimensions(space: Any) -> tuple[list[Any], tuple[str, ...]]:
    """One skopt dimension per declared dimension, in the declared order.

    Returns the dimensions and the parameter names they correspond to. Order is
    the contract: ``ask()`` returns a list, and nothing else identifies which
    value belongs to which parameter.
    """
    from skopt.space import Categorical, Integer, Real

    dimensions: list[Any] = []
    for dimension in space.dimensions:
        if dimension.kind == "categorical":
            dimensions.append(
                Categorical(
                    [str(index) for index in range(len(dimension.choices))],
                    name=dimension.name,
                )
            )
        elif dimension.kind == "int_uniform":
            dimensions.append(
                Integer(int(dimension.low), int(dimension.high), name=dimension.name)
            )
        elif dimension.kind == "log_uniform":
            dimensions.append(
                Real(
                    float(dimension.low),
                    float(dimension.high),
                    prior="log-uniform",
                    name=dimension.name,
                )
            )
        else:
            dimensions.append(
                Real(
                    float(dimension.low),
                    float(dimension.high),
                    prior="uniform",
                    name=dimension.name,
                )
            )
    return dimensions, space.names


def decode_point(space: Any, point: Sequence[Any]) -> dict[str, Any]:
    """A skopt point back into real parameter values."""
    params: dict[str, Any] = {}
    for dimension, value in zip(space.dimensions, point, strict=True):
        if dimension.kind == "categorical":
            params[dimension.name] = dimension.choices[int(value)]
        elif dimension.kind == "int_uniform":
            params[dimension.name] = int(value)
        else:
            params[dimension.name] = float(value)
    return params


def encode_params(space: Any, params: dict[str, Any]) -> list[Any]:
    """Real parameter values back into a skopt point, for ``tell``.

    Needed because the point that was *evaluated* is not always the point that
    was *asked for*: constraint repair can pin a dimension (M2's ``p`` off
    minkowski). Telling the surrogate the asked-for point would teach it a score
    that belongs to a different configuration.
    """
    point: list[Any] = []
    for dimension in space.dimensions:
        value = params[dimension.name]
        if dimension.kind == "categorical":
            matches = [
                index
                for index, choice in enumerate(dimension.choices)
                if choice == value or (choice is None and value is None)
            ]
            if not matches:
                raise ValueError(
                    dimension.name + "=" + repr(value) + " is not one of its declared choices"
                )
            point.append(str(matches[0]))
        elif dimension.kind == "int_uniform":
            point.append(int(value))
        else:
            point.append(float(value))
    return point


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


def _objective_floor(objective: Objective) -> float:
    """The worst score the objective can return -- what a failed trial is told.

    A trial that raised has no score, but the surrogate needs *some* observation
    or it will keep proposing the same broken corner. The floor is computed from
    the objective's own definition (sensitivity 0, specificity 0) rather than
    picked as a round number, so it is genuinely the worst attainable value and
    not an arbitrary penalty that distorts the surrogate's scale.
    """
    if objective.kind == "binary":
        from src.ensemble.soft_voting import OBJECTIVES

        return float(OBJECTIVES[objective.name](0.0, 0.0))
    return 0.0


class BayesianSearch(BaseSearch):
    """Gaussian-process guided search, on the same loop and folds as SO-01."""

    method: ClassVar[str] = "bayes"

    def __init__(
        self,
        *args: Any,
        n_initial_points: int | None = None,
        base_estimator: str = "GP",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        capability = skopt_available()
        if not capability.available:
            raise RuntimeError(capability.reason)

        from skopt import Optimizer

        dimensions, _ = to_skopt_dimensions(self.space)
        if n_initial_points is None:
            # skopt's default is 10. Under a small budget that would spend most
            # of it on random draws and the "Bayesian" run would be a randomized
            # run with extra steps, so it scales down with the budget -- but
            # never below 3, which is the fewest points a GP can be fitted to
            # without the surrogate being meaningless.
            n_initial_points = max(3, min(10, self.budget.max_trials // 4))
        self.n_initial_points = int(n_initial_points)
        self._floor = _objective_floor(self.objective)
        self._optimizer = Optimizer(
            dimensions,
            base_estimator=base_estimator,
            acq_func=ACQUISITION,
            n_initial_points=self.n_initial_points,
            random_state=self.seed,
        )

    def _propose(self, history: Sequence[Trial]) -> dict[str, Any]:
        del history  # the surrogate holds it; see _observe
        return decode_point(self.space, self._optimizer.ask())

    def _observe(self, trial: Trial) -> None:
        value = trial.score if trial.ok else self._floor
        try:
            point = encode_params(self.space, trial.params)
        except ValueError as error:
            # A repaired point that fell outside the declared choices cannot be
            # told to the surrogate. Skipping the observation is wrong (the
            # optimizer would re-ask it forever) and silently guessing a
            # substitute is worse, so it surfaces.
            raise RuntimeError(
                self.model_id + ": cannot encode trial " + str(trial.index) + " for skopt: "
                + str(error)
            ) from error
        # skopt MINIMISES; every objective in this project is higher-is-better.
        self._optimizer.tell(point, float(-value))

    @property
    def surrogate_ready(self) -> bool:
        """Whether enough points have been told for the GP to actually be fitted."""
        return len(self._optimizer.yi) >= self.n_initial_points
