"""Soft-voting ensembles M6 and M7, and the decision rule they predict with.

**M6** averages its members' probability vectors with equal weights -- the
mandatory baseline. **M7** learns the weights -- the proposed model. Both fuse
probabilities rather than votes, which is the only reason the ensemble can be
better than its best member: a hard vote throws away every member's confidence,
so two models that are wrong-but-unsure cannot be outvoted by one that is
right-and-certain.

Three things here are decided inside the training fold, and every one of them
would inflate the reported metric if it were not:

* the **members' own fits** -- refitted per fold by the pipeline above;
* the **weights**, optimised on out-of-fold probabilities from an inner CV over
  the training rows (T50.5);
* the **decision threshold**, chosen on those same out-of-fold probabilities
  (T50.4).

The inner CV is the mechanism that makes all three legitimate. Optimising
weights on the members' *training* predictions would pick whichever member
overfits hardest, because on training rows that member looks best. So the
members are fitted k times over inner splits of the training fold, their
held-out probabilities are collected, and the weights and threshold are chosen
against those. Only then are the members refitted on the whole training fold for
actual prediction.

**Why the threshold is not 0.5.** Phase 47 measured, and Phases 48-49 confirmed
on three more models, that calibrating a class-weighted model maps its scores
back onto the observed class prior -- so ``argmax`` at 0.5 re-imposes a
majority-favouring rule and gives back most of the sensitivity the weighting
bought. On D1 fold 0 that was 20 points for M3. Since M6/M7 fuse calibrated
probabilities and this project selects on sensitivity and balanced accuracy
(research rule 6), a fixed 0.5 is the wrong rule here. The chosen threshold is
recorded per fold **beside** the fixed-0.5 metrics, so the difference is visible
rather than assumed.

**Why a simplex grid rather than an optimiser.** Balanced accuracy is a step
function of the weights -- it changes only when a record crosses the threshold --
so it has zero gradient almost everywhere and SLSQP or L-BFGS stall at their
starting point. A grid over the simplex is exhaustive at its resolution,
deterministic, needs no seed, and for three members at 0.05 resolution is 231
points evaluated in vectorised numpy. Reproducibility (research rule 5) comes
free; a stochastic optimiser would have to be seeded and would still be a worse
answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from src.utils.logging_setup import get_logger

__all__ = [
    "OBJECTIVES",
    "OBJECTIVE_COEFFICIENTS",
    "objective_standard_error",
    "EnsembleError",
    "SoftVotingEnsemble",
    "ThresholdChoice",
    "fuse_probabilities",
    "simplex_grid",
    "weight_candidates",
    "select_threshold",
    "ensemble_members",
]

log = get_logger("ensemble.soft_voting")


class EnsembleError(ValueError):
    """The ensemble cannot be built, fitted or asked to predict as specified."""


# ---------------------------------------------------------------------------
# fusion (T50.1)
# ---------------------------------------------------------------------------


def fuse_probabilities(probabilities: Any, weights: Any = None) -> np.ndarray:
    """Weighted average of member probability matrices.

    ``probabilities`` is ``(n_members, n_samples, n_classes)``. Weights are
    normalised to sum to 1 before use, so a caller may pass raw scores; passing
    weights that already sum to 1 changes nothing. The result is a probability
    matrix in its own right -- a convex combination of rows that each sum to 1
    sums to 1 -- which is what lets the fused output feed a calibration metric
    or another ensemble without further treatment.
    """
    stack = np.asarray(probabilities, dtype=float)
    if stack.ndim != 3:
        raise EnsembleError(
            "expected (n_members, n_samples, n_classes), got shape " + str(stack.shape)
        )

    if weights is None:
        vector = np.ones(stack.shape[0], dtype=float)
    else:
        vector = np.asarray(weights, dtype=float)
    if vector.shape != (stack.shape[0],):
        raise EnsembleError(
            "expected " + str(stack.shape[0]) + " weights, got " + str(vector.shape)
        )
    if (vector < 0).any():
        raise EnsembleError("weights must be non-negative, got " + str(vector))

    total = float(vector.sum())
    if total <= 0:
        raise EnsembleError("weights sum to zero; there is nothing to average")
    return np.tensordot(vector / total, stack, axes=(0, 0))


def simplex_grid(n_members: int, resolution: float = 0.05) -> np.ndarray:
    """Every non-negative weight vector on the simplex at ``resolution``.

    Enumerated on an integer lattice rather than by rounding floats: the lattice
    is exact, so the grid is byte-identical on every machine and every run
    (research rule 5). The final division by ``steps`` is ordinary float
    arithmetic, so a row sums to 1 only to within representation error --
    ``0.05 + 0.10 + 0.85`` is not exactly 1.0 in binary. That is harmless here
    because :func:`fuse_probabilities` normalises by the sum anyway, but it is
    worth stating rather than claiming an exactness the floats do not have.
    """
    if n_members < 1:
        raise EnsembleError("need at least one member")
    steps = round(1.0 / resolution)
    if steps < 1:
        raise EnsembleError("resolution must be in (0, 1], got " + str(resolution))

    rows = [
        combination
        for combination in product(range(steps + 1), repeat=n_members)
        if sum(combination) == steps
    ]
    if not rows:
        raise EnsembleError("empty simplex grid")
    return np.asarray(rows, dtype=float) / steps


# ---------------------------------------------------------------------------
# the objective and the threshold (T50.4)
# ---------------------------------------------------------------------------


def weight_candidates(n_members: int, resolution: float = 0.05) -> np.ndarray:
    """``simplex_grid`` with the exact equal-weight vector guaranteed present.

    FIXED 2026-08-28, on measurement. The one-standard-error rule below shrinks
    toward "the candidate closest to equal weights, which is M6" -- but equal
    weights was not in the candidate set. For three members it is
    (1/3, 1/3, 1/3), and 1/3 is not a multiple of any decimal resolution: at
    0.05 the nearest lattice point is (0.30, 0.35, 0.35), 0.0408 away, and
    refining to 0.01 only closes it to 0.0082. It is never exactly representable.

    Measured consequence on D1 fold 0, where 181 of 231 candidates sat inside the
    margin -- so the rule was firmly in fall-back mode: M7 returned
    (0.30, 0.35, 0.35) scoring 0.859130, while equal weights scores 0.859583.
    The shrinkage landed BELOW the baseline it was shrinking toward, and the
    published weight vector was an artifact of lattice spacing rather than of
    evidence. With the exact point present, M7 returns equal weights whenever the
    evidence does not exceed the noise -- which is what the rule always claimed
    to do. See the 2026-08-28 Phases 60-62 entry in Docs/note.md.
    """
    lattice = simplex_grid(int(n_members), float(resolution))
    centre = np.full(int(n_members), 1.0 / float(n_members), dtype=float)
    if np.isclose(np.linalg.norm(lattice - centre, axis=1), 0.0).any():
        return lattice
    return np.vstack([centre[None, :], lattice])


def _sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: Any
) -> tuple[float, float]:
    positive = y_true == positive_label
    negative = ~positive
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    sensitivity = (
        float((y_pred[positive] == positive_label).sum() / n_positive)
        if n_positive
        else float("nan")
    )
    specificity = (
        float((y_pred[negative] != positive_label).sum() / n_negative)
        if n_negative
        else float("nan")
    )
    return sensitivity, specificity


def _balanced_accuracy(sensitivity: float, specificity: float) -> float:
    return float(np.nanmean([sensitivity, specificity]))


#: Objectives the weight search and threshold search may be scored against, each
#: as the coefficients ``(a, b)`` of ``a*sensitivity + b*specificity``.
#:
#: Every one is built from sensitivity and specificity rather than accuracy,
#: because on a 79/21 task accuracy is maximised by a model that rarely says
#: "abnormal" -- research rule 6. And every one is **linear** in the two, which
#: is not a coincidence worth losing: sensitivity and specificity are binomial
#: proportions with known denominators, so a linear objective has an exact
#: standard error and :func:`objective_standard_error` can compute how much of a
#: score difference is noise without bootstrapping anything.
OBJECTIVE_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "balanced_accuracy": (0.5, 0.5),
    # Weights sensitivity at 3x specificity once expanded (1.5*sens + 0.5*spec).
    # Measured on D1 fold 0 to drive the threshold to a saturating 1.000
    # sensitivity; kept available, not the shipped default. See Docs/note.md.
    "balanced_accuracy_plus_sensitivity": (1.5, 0.5),
    "sensitivity": (1.0, 0.0),
    # Youden's J is 2*balanced_accuracy - 1 -- the same ranking under a different
    # name, so it always picks the same threshold and the same weights.
    "youden": (1.0, 1.0),
}

_OBJECTIVE_OFFSET: dict[str, float] = {"youden": -1.0}


def _linear_objective(name: str) -> Any:
    a, b = OBJECTIVE_COEFFICIENTS[name]
    offset = _OBJECTIVE_OFFSET.get(name, 0.0)

    def score(sensitivity: float, specificity: float) -> float:
        return float(np.nansum([a * sensitivity, b * specificity])) + offset

    return score


OBJECTIVES: dict[str, Any] = {
    name: _linear_objective(name) for name in OBJECTIVE_COEFFICIENTS
}


def objective_standard_error(
    objective: str,
    sensitivity: float,
    specificity: float,
    n_positive: int,
    n_negative: int,
) -> float:
    """How much of this objective's value is sampling noise, exactly.

    Sensitivity is a proportion over ``n_positive`` records and specificity over
    ``n_negative``; both are binomial, so ``var = p(1-p)/n``. Every objective
    here is ``a*sens + b*spec``, and the two proportions are computed on disjoint
    record sets, so the variances add with no covariance term:

        SE = sqrt(a^2 * sens(1-sens)/n_pos + b^2 * spec(1-spec)/n_neg)

    This is what makes the one-standard-error rule usable here without
    bootstrapping 231 candidates. It is exact for the objective as defined, and
    it is deliberately **not** an estimate of how the score will transfer to the
    outer fold -- that is a larger and different quantity. It measures only the
    noise in the in-fold estimate, which is the thing the rule needs.
    """
    if objective not in OBJECTIVE_COEFFICIENTS:
        raise EnsembleError("unknown objective " + repr(objective))
    a, b = OBJECTIVE_COEFFICIENTS[objective]

    variance = 0.0
    if n_positive > 0 and np.isfinite(sensitivity):
        variance += (a**2) * sensitivity * (1.0 - sensitivity) / n_positive
    if n_negative > 0 and np.isfinite(specificity):
        variance += (b**2) * specificity * (1.0 - specificity) / n_negative
    return float(np.sqrt(max(variance, 0.0)))


@dataclass
class ThresholdChoice:
    """The decision threshold picked in-fold, and what it was picked over."""

    threshold: float
    objective: str
    score: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    n_candidates: int
    n_scored_rows: int
    #: The same measurements at a fixed 0.5, so the difference is visible in
    #: every output rather than being something a reader has to take on trust.
    fixed_half_sensitivity: float = float("nan")
    fixed_half_specificity: float = float("nan")
    fixed_half_balanced_accuracy: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "threshold_objective": self.objective,
            "threshold_score": self.score,
            "threshold_sensitivity": self.sensitivity,
            "threshold_specificity": self.specificity,
            "threshold_balanced_accuracy": self.balanced_accuracy,
            "threshold_n_candidates": self.n_candidates,
            "threshold_n_scored_rows": self.n_scored_rows,
            "fixed_half_sensitivity": self.fixed_half_sensitivity,
            "fixed_half_specificity": self.fixed_half_specificity,
            "fixed_half_balanced_accuracy": self.fixed_half_balanced_accuracy,
            "sensitivity_gained": self.sensitivity - self.fixed_half_sensitivity,
        }


def select_threshold(
    y_true: Any,
    positive_scores: Any,
    *,
    objective: str = "balanced_accuracy_plus_sensitivity",
    positive_label: Any = 1,
    max_candidates: int = 200,
) -> ThresholdChoice:
    """Choose the probability cut-off that maximises ``objective``.

    Candidates are the **midpoints between observed scores**, not a fixed grid
    of round numbers. Only a value that sits between two adjacent scores can
    change any prediction, so this enumerates every distinct classifier the
    scores admit -- a linspace over [0, 1] both misses achievable operating
    points and wastes evaluations on identical ones.

    Ties go to the threshold nearest 0.5. A tie means two cut-offs classify the
    scored rows identically, and the more central one generalises better to rows
    whose scores fall between them.
    """
    if objective not in OBJECTIVES:
        raise EnsembleError(
            "unknown objective " + repr(objective) + "; expected one of "
            + ", ".join(sorted(OBJECTIVES))
        )

    truth = np.asarray(y_true)
    scores = np.asarray(positive_scores, dtype=float)
    if truth.shape[0] != scores.shape[0]:
        raise EnsembleError(
            "y_true and scores disagree on length: "
            + str((truth.shape[0], scores.shape[0]))
        )
    if truth.size == 0:
        raise EnsembleError("no rows to choose a threshold on")

    unique = np.unique(scores)
    if unique.size > max_candidates:
        # Quantiles rather than a uniform subsample: the interesting region is
        # where the scores actually are, which on an imbalanced task is not the
        # middle of [0, 1].
        unique = np.unique(
            np.quantile(scores, np.linspace(0.0, 1.0, max_candidates))
        )
    candidates = (
        (unique[:-1] + unique[1:]) / 2.0 if unique.size > 1 else np.asarray([0.5])
    )
    candidates = np.unique(np.concatenate([candidates, [0.5]]))

    score_function = OBJECTIVES[objective]
    best: tuple[float, float, float, float, float] | None = None
    for threshold in candidates:
        predicted = np.where(scores >= threshold, positive_label, _other(truth, positive_label))
        sensitivity, specificity = _sensitivity_specificity(
            truth, predicted, positive_label
        )
        value = float(score_function(sensitivity, specificity))
        key = (value, -abs(float(threshold) - 0.5))
        if best is None or key > (best[0], -abs(best[1] - 0.5)):
            best = (value, float(threshold), sensitivity, specificity,
                    _balanced_accuracy(sensitivity, specificity))

    assert best is not None  # noqa: S101 - candidates is never empty
    value, threshold, sensitivity, specificity, balanced = best

    half = np.where(scores >= 0.5, positive_label, _other(truth, positive_label))
    half_sensitivity, half_specificity = _sensitivity_specificity(
        truth, half, positive_label
    )

    return ThresholdChoice(
        threshold=threshold,
        objective=objective,
        score=value,
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=balanced,
        n_candidates=int(candidates.size),
        n_scored_rows=int(truth.size),
        fixed_half_sensitivity=half_sensitivity,
        fixed_half_specificity=half_specificity,
        fixed_half_balanced_accuracy=_balanced_accuracy(
            half_sensitivity, half_specificity
        ),
    )


def _other(y_true: np.ndarray, positive_label: Any) -> Any:
    labels = [value for value in np.unique(y_true).tolist() if value != positive_label]
    if len(labels) != 1:
        raise EnsembleError(
            "threshold selection is a binary operation; found labels "
            + str(np.unique(y_true).tolist())
        )
    return labels[0]


def _pick_among_ties(
    grid: np.ndarray, scores: np.ndarray, *, margin: float = 0.0
) -> tuple[np.ndarray, int]:
    """The simplest weight vector whose score is within ``margin`` of the best.

    "Simplest" means closest to **equal weights**, in L2, which is minimised only
    by the uniform vector itself -- so the choice is unique and deterministic
    rather than a function of enumeration order.

    Two separate problems are solved by the same mechanism, and they need
    distinguishing because only one of them is about noise.

    **Exact ties (margin 0).** The objective is a step function of the weights,
    so many vectors score *identically*. Measured: a member whose out-of-fold
    probabilities were a constant 0.5 could take 95% of the weight without
    changing the score at all, because shrinking every fused probability toward
    0.5 is a monotone transform the tuned threshold undoes. Taking the first
    maximum in grid order reported that 0.95 as if it had been learned. M7's
    weights are a published result; they must not be an artefact of iteration
    order.

    **Near-ties (margin > 0) -- the one-standard-error rule.** Picking the single
    best of 231 candidates scored on a few thousand rows captures some genuine
    signal and some luck, and the more candidates there are the more of the
    winning margin is luck. The standard remedy in model selection is to prefer
    the simplest candidate within one standard error of the best, and here
    "simplest" has an obvious meaning: equal weights, which is M6. So M7 departs
    from its own baseline only where the evidence exceeds the noise.

    This is a **selection rule, not a tuning knob.** It was adopted because
    selecting the argmax of many noisy estimates is known to overfit the
    selection data, and it would be the right rule whether M7 were winning or
    losing. It must not be adjusted in response to a score.
    """
    if grid.shape[0] != scores.shape[0]:
        raise EnsembleError(
            "grid and scores disagree: " + str((grid.shape[0], scores.shape[0]))
        )
    if margin < 0:
        raise EnsembleError("margin must be non-negative, got " + str(margin))

    best = float(np.nanmax(scores))
    # 1e-12 absorbs float noise even at margin 0; it is not a tolerance choice.
    within = np.flatnonzero(scores >= best - margin - 1e-12)
    uniform = np.full(grid.shape[1], 1.0 / grid.shape[1])
    distances = np.linalg.norm(grid[within] - uniform, axis=1)
    chosen = within[int(np.argmin(distances))]
    return np.asarray(grid[chosen], dtype=float), int(within.size)


# ---------------------------------------------------------------------------
# the ensemble (T50.2, T50.3, T50.5)
# ---------------------------------------------------------------------------


@dataclass
class EnsembleFitReport:
    """What the ensemble decided while it was being fitted, and on what."""

    weights: dict[str, float] = field(default_factory=dict)
    weight_strategy: str = "equal"
    weight_objective: str = ""
    weight_candidates: int = 0
    #: How the weights were selected among near-equal candidates (the
    #: one-standard-error rule): margin, how many candidates fell inside it, and
    #: what a plain argmax would have chosen instead.
    weight_selection: dict[str, Any] = field(default_factory=dict)
    inner_folds: int = 0
    n_oof_rows: int = 0
    grouped_inner_cv: bool = False
    threshold: ThresholdChoice | None = None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "weight_strategy": self.weight_strategy,
            "weight_objective": self.weight_objective,
            "weight_candidates": self.weight_candidates,
            "inner_folds": self.inner_folds,
            "n_oof_rows": self.n_oof_rows,
            "grouped_inner_cv": self.grouped_inner_cv,
        }
        row.update({"weight_" + name: value for name, value in self.weights.items()})
        row.update(
            {"selection_" + name: value for name, value in self.weight_selection.items()}
        )
        if self.threshold is not None:
            row.update(self.threshold.as_dict())
        return row


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted probability averaging over base models, with an in-fold decision rule.

    ``estimators`` is a list of ``(name, estimator)`` pairs, unfitted. They are
    cloned, so the objects handed in are never fitted and cannot carry state
    between folds.

    ``weights`` is ``"equal"`` (M6) or ``"optimized"`` (M7), or an explicit
    sequence. ``groups`` carries the training fold's subject ids so the inner CV
    can be subject-aware; it must be supplied per fold, in the row order of the
    ``X`` passed to ``fit``, and passing groups from the full matrix is a leak.
    """

    def __init__(
        self,
        estimators: list[tuple[str, Any]] | None = None,
        *,
        weights: str | list[float] | np.ndarray = "equal",
        voting: str = "soft",
        inner_cv: int = 3,
        groups: Any = None,
        objective: str = "balanced_accuracy",
        weight_resolution: float = 0.05,
        selection_standard_errors: float = 1.0,
        tune_threshold: bool = True,
        positive_label: Any = 1,
        random_state: int = 42,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.voting = voting
        self.inner_cv = inner_cv
        self.groups = groups
        self.objective = objective
        self.weight_resolution = weight_resolution
        self.selection_standard_errors = selection_standard_errors
        self.tune_threshold = tune_threshold
        self.positive_label = positive_label
        self.random_state = random_state

    # -- fitting -----------------------------------------------------------

    def fit(self, X: Any, y: Any) -> SoftVotingEnsemble:
        if not self.estimators:
            raise EnsembleError("SoftVotingEnsemble needs at least one member")
        if self.voting != "soft":
            raise EnsembleError(
                "only soft voting is implemented; hard voting discards the "
                "confidences that make fusion worth doing"
            )

        features = np.asarray(X)
        targets = np.asarray(y)
        self.classes_ = np.unique(targets)
        self.member_names_ = [str(name) for name, _ in self.estimators]
        report = EnsembleFitReport(weight_strategy=str(self.weights))

        needs_oof = isinstance(self.weights, str) and self.weights == "optimized"
        needs_oof = needs_oof or (self.tune_threshold and self.classes_.size == 2)

        oof: np.ndarray | None = None
        if needs_oof:
            oof, splits, grouped = self._out_of_fold_probabilities(features, targets)
            report.inner_folds = len(splits)
            report.n_oof_rows = int(features.shape[0])
            report.grouped_inner_cv = grouped

        # -- weights (T50.3) ------------------------------------------------
        if isinstance(self.weights, str):
            if self.weights == "equal":
                vector = np.ones(len(self.estimators), dtype=float)
                vector /= vector.sum()
            elif self.weights == "optimized":
                assert oof is not None  # noqa: S101 - needs_oof was true
                vector, candidates = self._optimize_weights(oof, targets)
                report.weight_objective = self.objective
                report.weight_candidates = candidates
                report.weight_selection = dict(getattr(self, "weight_selection_", {}))
            else:
                raise EnsembleError(
                    "weights must be 'equal', 'optimized', or a sequence; got "
                    + repr(self.weights)
                )
        else:
            vector = np.asarray(self.weights, dtype=float)
            if vector.shape != (len(self.estimators),):
                raise EnsembleError(
                    "expected " + str(len(self.estimators)) + " weights, got "
                    + str(vector.shape)
                )
            if (vector < 0).any() or vector.sum() <= 0:
                raise EnsembleError("weights must be non-negative and not all zero")
            vector = vector / vector.sum()

        self.weights_ = vector
        report.weights = dict(
            zip(self.member_names_, vector.tolist(), strict=True)
        )

        # -- threshold (T50.4) ----------------------------------------------
        self.threshold_ = 0.5
        self.threshold_choice_ = None
        # Kept so the in-fold decision can be re-examined after the fact -- a
        # different threshold objective can be scored against the very rows the
        # shipped one was chosen on, without refitting anything. Two float
        # columns per training row; negligible next to the members themselves.
        self.oof_fused_ = None
        self.oof_targets_ = None
        if oof is not None:
            self.oof_fused_ = fuse_probabilities(oof, vector)
            self.oof_targets_ = targets.copy()
        if self.tune_threshold and self.classes_.size == 2:
            assert self.oof_fused_ is not None  # noqa: S101 - needs_oof was true
            fused = self.oof_fused_
            choice = select_threshold(
                targets,
                fused[:, self._positive_column()],
                objective=self.objective,
                positive_label=self.positive_label,
            )
            self.threshold_ = choice.threshold
            self.threshold_choice_ = choice
            report.threshold = choice
            log.info(
                "threshold %.4f chosen in-fold: sensitivity %.4f vs %.4f at 0.5",
                choice.threshold,
                choice.sensitivity,
                choice.fixed_half_sensitivity,
            )

        # -- refit the members on the whole training fold --------------------
        self.fitted_ = []
        for name, estimator in self.estimators:
            member = clone(estimator)
            member.fit(features, targets)
            self._check_member_classes(name, member)
            self.fitted_.append((name, member))

        self.n_features_in_ = features.shape[1]
        self.fit_report_ = report
        return self

    def _declared_members(self) -> list[tuple[str, Any]]:
        """The constructor's member list, refusing an empty one."""
        if not self.estimators:
            raise EnsembleError("SoftVotingEnsemble needs at least one member")
        return list(self.estimators)

    def _positive_column(self) -> int:
        classes = np.asarray(self.classes_).tolist()
        if self.positive_label not in classes:
            raise EnsembleError(
                "positive_label " + repr(self.positive_label)
                + " is not among the classes " + str(classes)
            )
        return classes.index(self.positive_label)

    def _check_member_classes(self, name: str, member: Any) -> None:
        member_classes = np.asarray(getattr(member, "classes_", self.classes_))
        if not np.array_equal(member_classes, np.asarray(self.classes_)):
            raise EnsembleError(
                "member " + name + " orders its classes as " + str(member_classes.tolist())
                + " but the ensemble uses " + str(np.asarray(self.classes_).tolist())
                + "; averaging those columns would mix classes"
            )

    def _inner_splits(
        self, features: np.ndarray, targets: np.ndarray
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], bool]:
        """Splits of the TRAINING fold only, subject-grouped where groups are given."""
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

        n_splits = int(self.inner_cv)
        if n_splits < 2:
            raise EnsembleError("inner_cv must be at least 2, got " + str(n_splits))

        if self.groups is not None:
            groups = np.asarray(self.groups)
            if groups.shape[0] != targets.shape[0]:
                raise EnsembleError(
                    "groups has " + str(groups.shape[0]) + " entries for "
                    + str(targets.shape[0]) + " training rows; they must be the "
                    "training fold's own groups, in its own row order"
                )
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            return list(splitter.split(features, targets, groups=groups)), True

        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        return list(splitter.split(features, targets)), False

    def _out_of_fold_probabilities(
        self, features: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], bool]:
        """Each member's held-out probabilities over an inner CV of the training fold.

        This is the step that makes weight and threshold selection honest. Every
        probability used to choose them was produced by a copy of the member that
        had not seen that row -- so a member which memorises its training data
        looks exactly as good here as it will on the outer test fold, which is
        the whole point.
        """
        members = self._declared_members()
        splits, grouped = self._inner_splits(features, targets)
        n_members = len(members)
        oof = np.full((n_members, features.shape[0], self.classes_.size), np.nan)

        for index, (name, estimator) in enumerate(members):
            for train_index, held_index in splits:
                member = clone(estimator)
                member.fit(features[train_index], targets[train_index])
                proba = np.asarray(member.predict_proba(features[held_index]))
                member_classes = np.asarray(getattr(member, "classes_", self.classes_))
                oof[index, held_index] = _align_columns(
                    proba, member_classes, self.classes_, name
                )

        missing = np.isnan(oof).any(axis=(0, 2))
        if missing.any():
            raise EnsembleError(
                str(int(missing.sum())) + " training row(s) got no out-of-fold "
                "probability; the inner CV did not cover the training fold"
            )
        return oof, splits, grouped

    def _optimize_weights(
        self, oof: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Grid-search the simplex, scoring each weight vector at its own best threshold.

        Weights and threshold are chosen **jointly**. Scoring every weight vector
        at a fixed 0.5 and then tuning the threshold afterwards optimises the
        wrong thing: the best weights at 0.5 are not the best weights at the
        threshold the model will actually use, and on an imbalanced task the two
        can differ substantially.
        """
        grid = weight_candidates(len(self._declared_members()), self.weight_resolution)
        score_function = OBJECTIVES[self.objective]

        if self.classes_.size != 2:
            # Multiclass has no single threshold to co-optimise; the decision is
            # argmax, so the objective is scored directly on the fused argmax.
            return self._optimize_weights_multiclass(oof, targets, grid), int(grid.shape[0])

        column = self._positive_column()
        scores = np.empty(grid.shape[0], dtype=float)
        sensitivities = np.empty(grid.shape[0], dtype=float)
        specificities = np.empty(grid.shape[0], dtype=float)
        for index, candidate in enumerate(grid):
            fused = fuse_probabilities(oof, candidate)[:, column]
            choice = select_threshold(
                targets, fused, objective=self.objective,
                positive_label=self.positive_label,
            )
            scores[index] = float(score_function(choice.sensitivity, choice.specificity))
            sensitivities[index] = choice.sensitivity
            specificities[index] = choice.specificity

        # The one-standard-error rule. The margin is the noise in the BEST
        # candidate's own estimate, so it adapts to the fold: a small or
        # badly-balanced training fold gives a wide margin and M7 stays close to
        # equal weights, a large one gives a narrow margin and M7 is free to
        # depart. Nothing here is tuned to a score.
        best_index = int(np.nanargmax(scores))
        n_positive = int((targets == self.positive_label).sum())
        n_negative = int(targets.size - n_positive)
        standard_error = objective_standard_error(
            self.objective,
            float(sensitivities[best_index]),
            float(specificities[best_index]),
            n_positive,
            n_negative,
        )
        margin = float(self.selection_standard_errors) * standard_error

        best_weights, n_within = _pick_among_ties(grid, scores, margin=margin)
        chosen_index = int(
            np.argmin(np.linalg.norm(grid - best_weights, axis=1))
        )
        self.weight_selection_ = {
            "n_candidates": int(grid.shape[0]),
            "n_within_margin": int(n_within),
            "best_score": float(scores[best_index]),
            "chosen_score": float(scores[chosen_index]),
            "standard_error": round(standard_error, 6),
            "selection_standard_errors": float(self.selection_standard_errors),
            "margin": round(margin, 6),
            "argmax_weights": np.round(grid[best_index], 4).tolist(),
        }
        log.info(
            "weights %s from %d simplex points (%d within %.4f of the best; "
            "argmax was %s), objective %s = %.4f (best %.4f)",
            np.round(best_weights, 3).tolist(),
            grid.shape[0],
            n_within,
            margin,
            np.round(grid[best_index], 3).tolist(),
            self.objective,
            float(scores[chosen_index]),
            float(scores[best_index]),
        )
        return best_weights, int(grid.shape[0])

    def _optimize_weights_multiclass(
        self, oof: np.ndarray, targets: np.ndarray, grid: np.ndarray
    ) -> np.ndarray:
        from sklearn.metrics import balanced_accuracy_score, f1_score

        classes = np.asarray(self.classes_)
        scores = np.empty(grid.shape[0], dtype=float)
        for index, candidate in enumerate(grid):
            predicted = classes[np.argmax(fuse_probabilities(oof, candidate), axis=1)]
            # Macro-F1 for multiclass, per research rule 6 and T54.2.
            scores[index] = float(
                f1_score(targets, predicted, average="macro", zero_division=0)
                + balanced_accuracy_score(targets, predicted)
            )
        # Multiclass keeps exact-tie breaking only. The objective there is
        # macro-F1 plus balanced accuracy, which is not linear in a pair of
        # binomial proportions, so it has no exact standard error to build a
        # margin from -- and inventing an approximate one would make the rule
        # look principled while resting on a guess. Revisit if the multiclass
        # tracks show the same selection overfitting.
        return _pick_among_ties(grid, scores)[0]

    # -- prediction (T50.4) ------------------------------------------------

    def _fitted(self) -> list[tuple[str, Any]]:
        members = getattr(self, "fitted_", None)
        if not members:
            raise EnsembleError("SoftVotingEnsemble is not fitted yet; call fit first")
        return members

    def member_probabilities(self, X: Any) -> np.ndarray:
        """Each member's probability matrix, class columns aligned. ``(m, n, k)``."""
        features = np.asarray(X)
        stack = []
        for name, member in self._fitted():
            proba = np.asarray(member.predict_proba(features))
            member_classes = np.asarray(getattr(member, "classes_", self.classes_))
            stack.append(_align_columns(proba, member_classes, self.classes_, name))
        return np.asarray(stack, dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        return fuse_probabilities(self.member_probabilities(X), self.weights_)

    def predict(self, X: Any) -> np.ndarray:
        """Argmax over the fused vector for multiclass; the in-fold threshold for binary.

        The two rules agree exactly when the threshold is 0.5, so the binary path
        is a generalisation of the documented argmax rather than a departure
        from it.
        """
        proba = self.predict_proba(X)
        classes = np.asarray(self.classes_)
        if classes.size != 2:
            return classes[np.argmax(proba, axis=1)]

        column = self._positive_column()
        negative = classes[1 - column]
        return np.where(
            proba[:, column] >= self.threshold_, self.positive_label, negative
        )

    @property
    def predicts_by_argmax(self) -> bool:
        """False once a non-0.5 threshold is in play.

        Anything checking that ``predict`` agrees with ``argmax(predict_proba)``
        has to ask this first. For a thresholded binary classifier the two
        *should* disagree -- that is the entire point of the threshold -- so an
        unconditional agreement check turns a working decision rule into a
        failing gate.
        """
        return bool(
            np.asarray(self.classes_).size != 2
            or float(getattr(self, "threshold_", 0.5)) == 0.5
        )

    def __sklearn_tags__(self) -> Any:  # pragma: no cover - sklearn plumbing
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags


def _align_columns(
    proba: np.ndarray, member_classes: np.ndarray, classes: Any, name: str
) -> np.ndarray:
    """Reorder a member's columns to the ensemble's class order.

    Members are refitted per inner split, and a split that happens to contain
    only one class produces a narrower matrix. Averaging by position without
    checking would add one member's P(normal) to another's P(abnormal) -- a
    catastrophic error that produces perfectly well-formed output.
    """
    target = np.asarray(classes)
    if np.array_equal(member_classes, target):
        return proba
    missing = [value for value in target.tolist() if value not in member_classes.tolist()]
    if missing:
        raise EnsembleError(
            "member " + name + " never saw class(es) " + str(missing)
            + "; its probabilities cannot be fused with the others'"
        )
    order = [member_classes.tolist().index(value) for value in target.tolist()]
    return proba[:, order]


# ---------------------------------------------------------------------------
# building M6 and M7 from config
# ---------------------------------------------------------------------------


def ensemble_members(model_id: str = "M6") -> list[tuple[str, Any]]:
    """The member estimators declared for M6/M7, calibrated per the Phase 48-49 verdicts.

    The members do **not** all get the same treatment, and that is deliberate.
    Phase 47 built M3 with its own explicit calibration because an SVM has no
    probabilities at all. Phase 48 measured that calibrating the random forest
    makes its expected calibration error *worse* (0.027 -> 0.039), and Phase 49
    measured that it improves gradient boosting's substantially (0.088 -> 0.029).
    Applying one policy to all three would knowingly degrade one member to keep
    the code symmetrical.
    """
    from src.models import estimators as est
    from src.utils.config import load_config

    spec = load_config("models").get("models." + model_id) or {}
    names = [str(name) for name in (spec.get("members") or [])]
    if not names:
        raise EnsembleError(model_id + " declares no members in configs/models.yaml")

    to_calibrate = {str(name) for name in (spec.get("calibrate_members") or [])}
    members: list[tuple[str, Any]] = []
    for name in names:
        estimator = est.build_estimator(name)
        if name in to_calibrate:
            from src.models.calibration import CalibratedSVM, calibration_settings

            settings = calibration_settings()
            estimator = CalibratedSVM(
                estimator=estimator,
                method=settings["method"],
                cv=settings["cv"],
                class_weight=None,
                ensemble=settings["ensemble"],
            )
        members.append((name, estimator))
    return members
