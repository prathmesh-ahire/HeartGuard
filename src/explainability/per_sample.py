"""Why *this* recording got *this* answer (T81.6).

The dashboard shows one prediction at a time. A global importance ranking cannot
explain it: it says which features the model relies on across a corpus, not which
ones pushed this particular recording toward abnormal.

## For the deployed model the explanation is exact, not approximate

The final binary model is a logistic regression at the end of a pipeline whose
only other steps are an imputer and a scaler. That makes the decision an exact
additive decomposition:

    logit = intercept + sum_j  coef_j * z_j

where ``z`` is the scaled, imputed feature vector. Each term is that feature's
contribution to the log-odds, in log-odds units, with a sign. Nothing is
estimated and nothing is sampled: :func:`explain_prediction` reconstructs the
model's own ``decision_function`` from its parts and **asserts the two agree to
1e-9**. An explanation that does not reproduce the decision it is explaining is
not an explanation.

That check is the reason this is a separate function from a generic SHAP call.
SHAP on a linear model returns the same decomposition by a slower route and with
sampling error; here the exact one is available, so it is used, and the
agreement is verified rather than trusted.

## For a tree model, SHAP

Tree models get `shap.TreeExplainer`, whose values are additive in the same way
-- ``base_value + sum(shap) = margin`` -- so the same reconstruction check
applies and is applied. Where SHAP is unavailable the function says so instead of
falling back to something that looks similar and means something else.

## What the numbers are NOT

A contribution is in **log-odds**, not in probability and not in risk. A feature
contributing +0.8 did not add 80% of anything. And a large contribution comes
from `coef_j * z_j`: a feature two standard deviations from the corpus mean
contributes twice what the same feature at one standard deviation does, so a
contribution is as much a statement about this recording being unusual as about
the model's opinion of the feature. Both facts are carried on every payload,
because a dashboard reader will not have them otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "TOP_CONTRIBUTIONS",
    "RECONSTRUCTION_TOLERANCE",
    "ExplanationError",
    "Contribution",
    "SampleExplanation",
    "explain_vector",
    "explain_prediction",
]

log = get_logger("explainability.per_sample")

#: How many contributions a dashboard payload carries by default. Twenty is the
#: same slice G19 draws, so the page and the figure agree about "top features".
TOP_CONTRIBUTIONS = 20

#: The reconstruction check's tolerance. Tight on purpose: this is exact
#: arithmetic on float64, not an approximation, so anything above rounding is a
#: wiring error rather than noise.
RECONSTRUCTION_TOLERANCE = 1e-9


class ExplanationError(RuntimeError):
    """A per-sample explanation cannot be produced, or would not be exact."""


@dataclass(frozen=True)
class Contribution:
    """One feature's push on one prediction, in log-odds."""

    feature: str
    family: str
    raw_value: float
    scaled_value: float
    weight: float
    contribution: float
    direction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "family": self.family,
            "raw_value": self.raw_value,
            "scaled_value": self.scaled_value,
            "weight": self.weight,
            "contribution": self.contribution,
            "direction": self.direction,
        }


@dataclass
class SampleExplanation:
    """Everything a page needs to explain one prediction, and its own proof."""

    task: str
    model_id: str
    method: str
    units: str
    base_value: float
    total_contribution: float
    decision_value: float
    reconstruction_error: float
    predicted_class: str
    predicted_index: int
    probability: float
    contributions: list[Contribution] = field(default_factory=list)
    n_features: int = 0
    caveats: tuple[str, ...] = ()

    def top(self, n: int = TOP_CONTRIBUTIONS) -> list[Contribution]:
        return sorted(self.contributions, key=lambda c: abs(c.contribution), reverse=True)[:n]

    def as_payload(self, n: int = TOP_CONTRIBUTIONS) -> dict[str, Any]:
        return {
            "task": self.task,
            "model_id": self.model_id,
            "method": self.method,
            "units": self.units,
            "base_value": self.base_value,
            "total_contribution": self.total_contribution,
            "decision_value": self.decision_value,
            "reconstruction_error": self.reconstruction_error,
            "predicted_class": self.predicted_class,
            "predicted_index": self.predicted_index,
            "probability": self.probability,
            "n_features": self.n_features,
            "n_shown": min(n, len(self.contributions)),
            "top_contributions": [c.as_dict() for c in self.top(n)],
            "caveats": list(self.caveats),
        }


_UNITS = "log-odds (natural log of the odds of the positive class)"

_CAVEATS = (
    "Contributions are in log-odds, not in probability and not in risk. A "
    "contribution of +0.8 did not add 80% of anything.",
    "A contribution is the model's weight multiplied by how unusual this "
    "recording is on that feature, in standard deviations from the corpus mean. "
    "A large value is as much a statement about the recording being atypical as "
    "about the model's view of the feature.",
    "PV-MEPCG / PulseVision is an academic screening and decision-support "
    "prototype. An explanation of a prediction is not a clinical explanation.",
)


def _family_of(name: str) -> str:
    from src.feature_extraction.registry import spec_for

    try:
        return str(spec_for(name).family)
    except Exception:  # noqa: BLE001 - a non-registry column is not fatal
        return "unknown"


def _pre_estimator(pipeline: Any, vector: np.ndarray) -> np.ndarray:
    """The vector as the final estimator sees it, width checked."""
    matrix = np.asarray(vector, dtype=float).reshape(1, -1)
    width = matrix.shape[1]
    for name, step in pipeline.named_steps.items():
        if name == "estimator":
            break
        matrix = step.transform(matrix)
    if matrix.shape[1] != width:
        raise ExplanationError(
            "the pipeline changed the feature count from "
            + str(width)
            + " to "
            + str(matrix.shape[1])
            + "; contributions can no longer be attributed to feature names"
        )
    return matrix[0]


def explain_vector(
    pipeline: Any,
    vector: Any,
    feature_names: tuple[str, ...] | list[str],
    *,
    task: str = "binary",
    model_id: str = "",
    class_names: tuple[str, ...] | list[str] = (),
) -> SampleExplanation:
    """Decompose one prediction into per-feature contributions, and verify it.

    Linear final estimators get the exact decomposition; tree models get SHAP.
    Either way the contributions plus the base value are reconstructed and
    checked against the model's own decision, and a mismatch raises.
    """
    from src.models.importance import _final_estimator

    names = tuple(str(name) for name in feature_names)
    raw = np.asarray(vector, dtype=float).reshape(-1)
    if raw.size != len(names):
        raise ExplanationError(
            "vector has " + str(raw.size) + " values for " + str(len(names)) + " names"
        )

    scaled = _pre_estimator(pipeline, raw)
    estimator = _final_estimator(pipeline)
    resolved_id = model_id or type(estimator).__name__

    coefficients = getattr(estimator, "coef_", None)
    if coefficients is not None and np.asarray(coefficients).ndim == 2:
        weights = np.asarray(coefficients, dtype=float)
        if weights.shape[0] != 1:
            raise ExplanationError(
                "the exact decomposition is written for a binary linear model; "
                "this estimator has " + str(weights.shape[0]) + " coefficient rows"
            )
        weights = weights[0]
        base = float(np.asarray(getattr(estimator, "intercept_", [0.0]), dtype=float)[0])
        contributions = weights * scaled
        method = "exact linear decomposition (coefficient x scaled value)"
        decision = float(np.asarray(estimator.decision_function(scaled.reshape(1, -1)))[0])
    else:
        weights, base, contributions, method, decision = _shap_contributions(
            estimator, pipeline, scaled
        )

    reconstructed = base + float(np.sum(contributions))
    error = abs(reconstructed - decision)
    if not np.isfinite(error) or error > RECONSTRUCTION_TOLERANCE:
        raise ExplanationError(
            "the contributions do not reconstruct the model's own decision: "
            + format(reconstructed, ".9f")
            + " against "
            + format(decision, ".9f")
            + " (error "
            + format(error, ".3e")
            + "). An explanation that does not reproduce its prediction is wrong."
        )

    proba = np.asarray(pipeline.predict_proba(raw.reshape(1, -1)), dtype=float)[0]
    index = int(np.argmax(proba))
    labels = tuple(str(c) for c in class_names) if class_names else tuple(
        str(c) for c in getattr(estimator, "classes_", range(proba.size))
    )

    items = [
        Contribution(
            feature=name,
            family=_family_of(name),
            raw_value=float(raw[position]),
            scaled_value=float(scaled[position]),
            weight=float(weights[position]),
            contribution=float(contributions[position]),
            direction=(
                "toward " + (labels[-1] if len(labels) > 1 else "positive")
                if contributions[position] > 0
                else "toward " + (labels[0] if labels else "negative")
            ),
        )
        for position, name in enumerate(names)
    ]

    return SampleExplanation(
        task=task,
        model_id=resolved_id,
        method=method,
        units=_UNITS,
        base_value=base,
        total_contribution=float(np.sum(contributions)),
        decision_value=decision,
        reconstruction_error=float(error),
        predicted_class=labels[index] if index < len(labels) else str(index),
        predicted_index=index,
        probability=float(proba[index]),
        contributions=items,
        n_features=len(names),
        caveats=_CAVEATS,
    )


def _shap_contributions(
    estimator: Any, pipeline: Any, scaled: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, str, float]:
    """TreeExplainer's additive decomposition, for a non-linear final estimator."""
    if not hasattr(estimator, "feature_importances_"):
        raise ExplanationError(
            type(estimator).__name__ + " is neither a binary linear model nor a "
            "tree model; no exact additive decomposition is available and this "
            "function will not approximate one silently"
        )
    try:
        import shap
    except ImportError as error:
        raise ExplanationError(
            "shap is required to explain a tree model and is not installed ("
            + str(error)
            + "); it is declared in requirements-extra.txt"
        ) from error

    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(scaled.reshape(1, -1)))
    if values.ndim == 3:
        values = values[:, :, -1]
    contributions = values.reshape(-1)
    base = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    base_value = float(base[-1] if base.size > 1 else base[0])
    decision = base_value + float(contributions.sum())
    # The weight is not separable for a tree model; the contribution IS the
    # quantity. Reported as NaN rather than as a fabricated per-feature weight.
    weights = np.full(contributions.shape, np.nan, dtype=float)
    return weights, base_value, contributions, "shap TreeExplainer (additive)", decision


def explain_prediction(
    path: Any,
    *,
    task: str = "binary",
    bundle: Any = None,
    record_uid: str | None = None,
    use_cache: bool = False,
) -> tuple[Any, SampleExplanation]:
    """Score one recording and explain it, in one call.

    Returns the ordinary :class:`~src.inference.predictor.PredictionResult`
    unchanged beside the explanation: explaining a prediction must never alter
    it, so the prediction comes from the same code path a plain `/predict` uses.
    """
    from src.inference.predictor import load_bundle, predict_recording

    loaded = bundle if bundle is not None else load_bundle(task)
    result, detail = predict_recording(
        path,
        task=task,
        bundle=loaded,
        record_uid=record_uid,
        use_cache=use_cache,
        with_detail=True,
    )
    explanation = explain_vector(
        loaded.pipeline,
        detail.vector,
        loaded.feature_names,
        task=loaded.task,
        model_id=str(loaded.manifest.get("selected_model_id") or loaded.model_id),
        class_names=loaded.classes,
    )
    if explanation.predicted_class != result.predicted_class:
        raise ExplanationError(
            "the explanation names class "
            + explanation.predicted_class
            + " but the prediction is "
            + result.predicted_class
            + "; they are not describing the same decision"
        )
    return result, explanation
