"""The multi-objective score J, and the inference-time term underneath it.

    J = alpha * (1 - MacroF1) + beta * (SelectedFeatures / 138)
        + gamma * NormalizedInferenceTime

Minimised. Every term is bounded in [0, 1] by construction, so J is too, and a
J of 0 would be a perfect classifier using no features in no time -- which is
the right shape for a score nobody should ever reach.

Built here in Phase 57 because T57.5 selects the final subset by it and T58.2
uses it as the GA's fitness. Phase 61 (SO-06) sweeps alpha/beta/gamma across the
Pareto front and picks the operating point; this module only computes the score
at one weighting, and takes that weighting from `configs/models.yaml`.

THE INFERENCE-TIME TERM IS EXTRACTION TIME, NOT PREDICT TIME
------------------------------------------------------------
The obvious reading of "inference time" is how long the fitted model takes to
predict. For this project that would be measuring noise. Extraction of all 138
features costs ~2.14 s per record and a Random Forest's predict on one row costs
tens of microseconds -- four to five orders of magnitude apart. What a feature
subset actually buys back is the *extraction* it no longer has to run.

And extraction is per FAMILY, not per feature: computing one MFCC coefficient
means computing the MFCC stack. So a subset pays a family's full measured cost
if it keeps any feature from that family and nothing if it keeps none. On this
corpus that makes the term almost binary -- the `time` family alone is 98% of
extraction -- which is a real property of the pipeline, not an artifact of the
formula. See the 2026-08-28 Phases 57-59 entry in Docs/note.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MultiObjectiveError",
    "JWeights",
    "FamilyCostModel",
    "JScore",
    "load_weights",
    "load_cost_model",
    "family_of",
    "families_needed",
    "score_j",
    "macro_f1",
]


class MultiObjectiveError(ValueError):
    """J cannot be computed as asked, or would not mean what it says."""


# ---------------------------------------------------------------------------
# T61.2 -- the weighting, from config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JWeights:
    """alpha, beta, gamma and the total feature count they are defined against."""

    alpha: float = 0.70
    beta: float = 0.20
    gamma: float = 0.10
    n_features_total: int = 138
    performance_metric: str = "macro_f1"

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise MultiObjectiveError(name + " must be finite and >= 0, got " + str(value))
        if self.n_features_total < 1:
            raise MultiObjectiveError("n_features_total must be >= 1")
        if float(self.alpha + self.beta + self.gamma) <= 0:
            raise MultiObjectiveError("alpha + beta + gamma must be > 0")

    @property
    def total(self) -> float:
        return float(self.alpha + self.beta + self.gamma)

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "n_features_total": int(self.n_features_total),
            "performance_metric": self.performance_metric,
        }


def load_weights(config: dict[str, Any] | None = None) -> JWeights:
    """The configured weighting, or the module defaults if config is absent."""
    if config is None:
        from src.utils.config import load_config

        config = load_config("models").get("optimization.multi_objective") or {}
    settings = dict(config)
    return JWeights(
        alpha=float(settings.get("alpha", 0.70)),
        beta=float(settings.get("beta", 0.20)),
        gamma=float(settings.get("gamma", 0.10)),
        n_features_total=int(settings.get("n_features_total", 138)),
        performance_metric=str(settings.get("performance_metric", "macro_f1")),
    )


# ---------------------------------------------------------------------------
# T61.3 -- the inference-time term
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyCostModel:
    """Per-family extraction seconds, measured, plus the all-families total.

    ``seconds`` maps family name -> measured seconds per record. ``total`` is the
    cost of the slowest configuration (every family), which is what the term is
    normalised against, so a subset needing every family scores exactly 1.0 and
    one needing none scores 0.0.
    """

    seconds: dict[str, float]
    source: str = ""

    def __post_init__(self) -> None:
        if not self.seconds:
            raise MultiObjectiveError("the cost model is empty; no family timings were loaded")
        for family, value in self.seconds.items():
            if not np.isfinite(value) or value < 0:
                raise MultiObjectiveError(
                    "family " + str(family) + " has a non-finite or negative cost: " + str(value)
                )

    @property
    def total(self) -> float:
        return float(sum(self.seconds.values()))

    def cost_of(self, families: Sequence[str]) -> float:
        """Seconds per record for a subset needing exactly ``families``."""
        wanted = set(families)
        unknown = wanted - set(self.seconds)
        if unknown:
            raise MultiObjectiveError(
                "no measured cost for family/families " + ", ".join(sorted(unknown))
            )
        return float(sum(self.seconds[name] for name in wanted))

    def normalized(self, families: Sequence[str]) -> float:
        """``cost_of`` divided by the all-families total. Bounded in [0, 1]."""
        total = self.total
        if total <= 0:
            raise MultiObjectiveError("the all-families total cost is zero; cannot normalise")
        value = self.cost_of(families) / total
        return float(min(1.0, max(0.0, value)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "total_seconds": self.total,
            "seconds": {name: float(value) for name, value in sorted(self.seconds.items())},
        }


def load_cost_model(
    path: str | Path | None = None, *, statistic: str = "min_seconds"
) -> FamilyCostModel:
    """Per-family extraction seconds from the Phase 39 timing table.

    ``min_seconds``, following the reasoning T38.6 already wrote into
    ``benchmark_families``: over three repeats on a shared desktop the mean is
    inflated by whatever else the machine was doing, and the minimum is the
    closest estimate of the cost itself. It makes almost no difference here --
    the term is a ratio, and the `time` family is 97.9% of the total under the
    minimum against 98.3% under the mean -- but the two tables in this project
    that talk about extraction cost should not disagree about which column that
    cost lives in.

    ONE FEATURE, NOT ONE FAMILY, IS THE COST
    ----------------------------------------
    Measured on 2026-08-28 over a 30 s record at 2 kHz: the whole time family
    costs ~1.2 s and ``time_sample_entropy`` alone costs ~1.4 s of it (the
    excess is run-to-run variance). Sample entropy is O(N^2) in the record
    length; every other time feature is a few milliseconds. So the entire
    inference-time term of J is, in practice, a proxy for "does this subset
    still need sample entropy".

    The model stays family-wise anyway, because that is how extraction actually
    runs: ``extract_all`` computes a family as a unit and the live ``/predict``
    path calls it. But no write-up may present the time term as a general
    argument about time-domain features, and the real deployment saving is to
    drop that one feature -- a Phase 79/80 inference optimisation, not a feature
    selection result. See the 2026-08-28 Phases 57-59 entry in Docs/note.md.
    """
    import pandas as pd

    from src.utils.config import load_config

    if path is None:
        root = Path(load_config("paths").require("outputs.features"))
        path = root / "feature_extraction_timing.csv"
    target = Path(path)
    if not target.is_file():
        raise MultiObjectiveError(
            "the inference-time term needs " + str(target) + ", which does not exist; "
            "run scripts/03_feature_reports.py first"
        )
    frame = pd.read_csv(target)
    missing = {"family", statistic} - set(frame.columns)
    if missing:
        raise MultiObjectiveError(
            str(target) + " has no " + ", ".join(sorted(missing)) + " column"
        )
    seconds = {
        str(row["family"]): float(row[statistic])
        for _, row in frame.iterrows()
        if np.isfinite(float(row[statistic]))
    }
    return FamilyCostModel(seconds=seconds, source=str(target) + "#" + statistic)


def family_of(feature_name: str) -> str:
    """The feature family a column belongs to, from the locked registry.

    Goes through the registry rather than splitting on the first underscore:
    the names are stable but the prefix convention is not something a scoring
    function should be re-deriving.
    """
    from src.feature_extraction import registry

    return str(registry.family_of(feature_name))


def families_needed(feature_names: Sequence[str]) -> tuple[str, ...]:
    """The distinct families a subset still forces the extractor to compute."""
    return tuple(sorted({family_of(str(name)) for name in feature_names}))


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------


def macro_f1(y_true: Any, y_pred: Any, labels: Sequence[Any] | None = None) -> float:
    """Macro-averaged F1, the performance term the blueprint documents for J.

    Defined for the binary task too, where it is the unweighted mean of the F1
    of "abnormal" and the F1 of "normal" -- not the same thing as the binary F1
    the rest of the project reports, which is the positive class alone.
    """
    from sklearn.metrics import f1_score

    kwargs: dict[str, Any] = {"average": "macro", "zero_division": 0}
    if labels is not None:
        kwargs["labels"] = list(labels)
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred), **kwargs))


@dataclass(frozen=True)
class JScore:
    """One evaluation of J, with every term kept so the total can be audited."""

    value: float
    performance: float
    n_selected: int
    normalized_features: float
    normalized_time: float
    families: tuple[str, ...] = ()
    weights: JWeights = field(default_factory=JWeights)

    def as_dict(self) -> dict[str, Any]:
        return {
            "j": float(self.value),
            "macro_f1": float(self.performance),
            "n_selected": int(self.n_selected),
            "term_performance": float(self.weights.alpha * (1.0 - self.performance)),
            "term_features": float(self.weights.beta * self.normalized_features),
            "term_time": float(self.weights.gamma * self.normalized_time),
            "normalized_features": float(self.normalized_features),
            "normalized_inference_time": float(self.normalized_time),
            "families": ";".join(self.families),
        }


def score_j(
    performance: float,
    feature_names: Sequence[str],
    *,
    weights: JWeights | None = None,
    cost_model: FamilyCostModel | None = None,
) -> JScore:
    """Compute J for a subset that scored ``performance`` (macro-F1).

    ``feature_names`` are the columns the subset keeps, by name -- not a count.
    The names are what decide the inference-time term, because they decide which
    families still have to be extracted; two subsets of the same size can differ
    by 98% of the extraction cost.
    """
    weights = weights or load_weights()
    if not np.isfinite(float(performance)):
        raise MultiObjectiveError("performance must be finite, got " + str(performance))
    names = [str(name) for name in feature_names]
    if not names:
        raise MultiObjectiveError("an empty subset has no J; it cannot be scored at all")

    n_selected = len(names)
    if n_selected > weights.n_features_total:
        raise MultiObjectiveError(
            "subset of " + str(n_selected) + " exceeds the declared total of "
            + str(weights.n_features_total)
        )
    families = families_needed(names)
    cost_model = cost_model or load_cost_model()
    normalized_time = cost_model.normalized(families)
    normalized_features = float(n_selected) / float(weights.n_features_total)

    value = (
        weights.alpha * (1.0 - float(performance))
        + weights.beta * normalized_features
        + weights.gamma * normalized_time
    )
    return JScore(
        value=float(value),
        performance=float(performance),
        n_selected=int(n_selected),
        normalized_features=normalized_features,
        normalized_time=float(normalized_time),
        families=families,
        weights=weights,
    )
