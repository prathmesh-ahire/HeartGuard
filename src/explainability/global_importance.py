"""Which features the models actually use, measured over several folds (Phase 81).

`src/models/importance.py` (Phase 48) already defines the two measures and
enforces the rule that matters -- permutation importance is computed only on
rows the model never saw. It does it for **one** fold. Phase 81 needs the
reported ranking, so it needs a spread, and this module supplies it.

## Which folds, and why not all twenty-five

Repeat 0's five folds. That is a **complete partition of the corpus**: every
record is held out exactly once across the five, so the five measurements
together score all 3,240 records with no record counted twice. Running all 25
would score every record five times for a 2.2x narrower interval on a quantity
whose fold-to-fold spread is the thing being reported, and would cost five times
the wall clock. `FOLD_REPEAT` is a constant so a longer run is one edit.

## Permutation is the primary measure. Impurity is reported beside it, as a foil

Impurity importance is free, measured on the training rows, and biased toward
features with many split points. On this matrix that bias has a target: the 39
MFCC and 24 wavelet columns are strongly correlated within their families, and
impurity splits one contribution arbitrarily among the members. Permutation
importance asks what the fitted model actually *uses*, on rows it never saw, and
can go negative -- a negative value means shuffling the feature improved the
score, so the model was being misled by it.

The two disagree, and the disagreement is a finding rather than an error. Both
are emitted with a `kind` column; nothing averages them together.

## Which models

Permutation runs for every model that is cheap enough to re-score 1,381 times
per fold. **M6 and M7 are excluded and the reason is measured, not assumed**:
each fit costs ~228 s and each batch prediction ~0.58 s on a fold's test rows, so
one fold is ~17 minutes and the pair is ~3 hours -- for a quantity that is a
weighted blend of three members whose importances are already reported
individually. `--models` overrides it; the exclusion travels in the coverage CSV
rather than being silent.

## SHAP (T81.3)

`shap.TreeExplainer` on the tree models, computed on the held-out rows of the
first fold. SHAP values are additive per prediction, so they carry information
neither of the other two measures does -- direction and per-sample attribution --
and :mod:`src.explainability.per_sample` uses that. If the package is missing the
skip reason is recorded rather than the section quietly disappearing.

**SHAP sees the transformed matrix.** The pipeline is imputer -> scaler ->
estimator with 138 columns in and 138 out, so a column index means the same
feature at both ends; :func:`_transformed` asserts that rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "FOLD_REPEAT",
    "PERMUTATION_MODELS",
    "IMPURITY_MODELS",
    "SHAP_MODELS",
    "EXCLUDED_MODELS",
    "N_REPEATS",
    "TOP_N",
    "ExplainError",
    "ModelFit",
    "fit_models_for_fold",
    "fold_importances",
    "aggregate_importance",
    "aggregate_family_importance",
    "shap_importance",
    "top_features",
]

log = get_logger("explainability.global")


class ExplainError(RuntimeError):
    """An explanation cannot be produced as asked."""


#: The repeat whose five folds partition the corpus exactly once.
FOLD_REPEAT = 0

#: Permutation importance is computed for these. M1 is the deployed model.
PERMUTATION_MODELS: tuple[str, ...] = ("M1", "M3", "M4", "M5", "M8")

#: Impurity importance exists only where the estimator exposes
#: ``feature_importances_``: the two tree models T81.1 names, plus XGBoost.
IMPURITY_MODELS: tuple[str, ...] = ("M4", "M5", "M8")

#: TreeExplainer applies to the same three.
SHAP_MODELS: tuple[str, ...] = ("M4", "M5", "M8")

#: Excluded, with the reason that goes into the coverage table.
EXCLUDED_MODELS: dict[str, str] = {
    "M2": (
        "KNN has no feature_importances_ and is not in the reported model set; "
        "permutation importance would be measurable but the model is not one "
        "any table reports."
    ),
    "M6": (
        "soft-voting ensemble: ~228 s to fit and ~0.58 s per batch prediction, "
        "so 138 features x 10 repeats is ~17 minutes per fold and ~85 minutes "
        "over the five. Its importance is a weighted blend of M3, M4 and M5, "
        "each reported individually here. Run with --models M6 to include it."
    ),
    "M7": (
        "same cost as M6, and M7 differs from it only in its weights. See M6."
    ),
}

#: Permutation repeats per feature per fold. Matches Phase 48 so the fold-0
#: numbers stay comparable with the ones already in feature_importance.csv.
N_REPEATS = 10

#: How many features G19 and FE-11's headline slice carry.
TOP_N = 20


@dataclass
class ModelFit:
    """One model fitted on one fold, with the indices that made it."""

    model_id: str
    fold_label: str
    pipeline: Any
    train_index: np.ndarray
    test_index: np.ndarray


def _folds(task: str, data: Any, repeat: int = FOLD_REPEAT) -> list[Any]:
    from src.evaluation.cv import load_folds, resolve_folds

    folds = resolve_folds(load_folds(task), data.record_uids)
    chosen = [fold for fold in folds if fold.repeat == repeat]
    if not chosen:
        raise ExplainError(
            "no fold with repeat=" + str(repeat) + " in the " + task + " map"
        )
    covered = sum(len(fold.test_uids) for fold in chosen)
    if covered != len(data.record_uids):
        raise ExplainError(
            "repeat " + str(repeat) + " covers " + str(covered) + " of "
            + str(len(data.record_uids)) + " records; it is not a partition"
        )
    return chosen


def fit_models_for_fold(
    fold: Any, data: Any, model_ids: tuple[str, ...]
) -> list[ModelFit]:
    """Fit each model on one fold's training rows, under config defaults.

    Config defaults, not a searched configuration: the ranking is meant to be a
    property of the feature set and the model family, and a per-fold search would
    make each fold's ranking a statement about a different model.
    """
    from src.models import estimators as est
    from src.models import pipeline as pl

    train = np.asarray(fold.train_index, dtype=int)
    test = np.asarray(fold.test_index, dtype=int)
    fits: list[ModelFit] = []
    for model_id in model_ids:
        estimator = est.build_estimator(model_id)
        built = pl.build_pipeline(estimator, y=data.y[train], n_features=data.X.shape[1])
        built.fit(data.X[train], data.y[train])
        fits.append(ModelFit(model_id, fold.label, built, train, test))
        log.info("%s %s: fitted on %d row(s)", model_id, fold.label, train.size)
    return fits


def fold_importances(
    *,
    task: str = "binary",
    data: Any = None,
    permutation_models: tuple[str, ...] = PERMUTATION_MODELS,
    impurity_models: tuple[str, ...] = IMPURITY_MODELS,
    n_repeats: int = N_REPEATS,
    repeat: int = FOLD_REPEAT,
) -> Any:
    """Per (model, fold, feature) importance, both kinds, across the partition."""
    import pandas as pd

    from src.models import importance as imp
    from src.models.smoke import load_task_data

    resolved = data if data is not None else load_task_data(task)
    names = tuple(resolved.feature_names)
    wanted = tuple(dict.fromkeys((*permutation_models, *impurity_models)))

    blocks: list[pd.DataFrame] = []
    for fold in _folds(task, resolved, repeat):
        for fit in fit_models_for_fold(fold, resolved, wanted):
            results = []
            if fit.model_id in impurity_models and imp.supports_impurity_importance(
                fit.pipeline
            ):
                results.append(
                    imp.impurity_importance(fit.pipeline, names, model_id=fit.model_id)
                )
            if fit.model_id in permutation_models:
                results.append(
                    imp.permutation_importance(
                        fit.pipeline,
                        resolved.X,
                        resolved.y,
                        held_out_index=fit.test_index,
                        train_index=fit.train_index,
                        feature_names=names,
                        model_id=fit.model_id,
                        n_repeats=n_repeats,
                    )
                )
            if not results:
                continue
            frame = imp.importance_frame(results)
            frame.insert(0, "task", task)
            frame.insert(1, "fold_label", fit.fold_label)
            blocks.append(frame)
            log.info(
                "%s %s: %s",
                fit.model_id,
                fit.fold_label,
                ", ".join(sorted({str(k) for k in frame["kind"]})),
            )

    if not blocks:
        raise ExplainError("no importance was computed for any model")
    return pd.concat(blocks, ignore_index=True)


def aggregate_importance(per_fold: Any) -> Any:
    """mean +/- SD across folds per (model, kind, feature), ranked within each.

    The rank is taken on the **mean**, and `rank_sd` reports how much the
    per-fold ranks themselves moved. A feature ranked 3rd on average with a rank
    SD of 30 is not a third-place feature; it is an unstable one, and correlated
    families produce exactly that.
    """

    keys = ["task", "model_id", "kind", "feature"]
    grouped = per_fold.groupby(keys, sort=False)
    frame = grouped.agg(
        importance_mean=("importance", "mean"),
        importance_sd=("importance", "std"),
        importance_min=("importance", "min"),
        importance_max=("importance", "max"),
        rank_mean=("rank", "mean"),
        rank_sd=("rank", "std"),
        rank_best=("rank", "min"),
        rank_worst=("rank", "max"),
        n_folds=("importance", "size"),
    ).reset_index()
    frame["n_folds_positive"] = (
        per_fold.assign(positive=per_fold["importance"] > 0)
        .groupby(keys, sort=False)["positive"]
        .sum()
        .to_numpy()
    )
    frame["rank"] = (
        frame.groupby(["task", "model_id", "kind"])["importance_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return frame.sort_values(["model_id", "kind", "rank"]).reset_index(drop=True)


def aggregate_family_importance(per_fold: Any) -> Any:
    """T81.4 -- roll every measurement up to the six declared feature families.

    The family total is the stable quantity. Individual importances of correlated
    features are close to arbitrary between members; which of the 39 MFCC columns
    wins tells nobody anything, and their sum does.

    ``share_of_positive`` divides by the positive part only. Permutation
    importance can be negative, so a signed grand total can be near zero or
    change sign and turn every share into nonsense.
    """
    import pandas as pd

    from src.feature_extraction.registry import spec_for

    frame = per_fold.copy()
    families = []
    for name in frame["feature"]:
        try:
            families.append(spec_for(str(name)).family)
        except Exception:  # noqa: BLE001 - a non-registry column is not fatal
            families.append("unknown")
    frame["family"] = families

    rows: list[dict[str, Any]] = []
    keys = ["task", "model_id", "kind", "fold_label"]
    for values, block in frame.groupby(keys, sort=False):
        positive_total = float(block.loc[block["importance"] > 0, "importance"].sum())
        for family, part in block.groupby("family", sort=True):
            total = float(part["importance"].sum())
            positive = float(part.loc[part["importance"] > 0, "importance"].sum())
            rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "family": str(family),
                    "n_features": len(part),
                    "total_importance": total,
                    "positive_importance": positive,
                    "mean_importance": float(part["importance"].mean()),
                    "max_importance": float(part["importance"].max()),
                    "n_negative": int((part["importance"] < 0).sum()),
                    "share_of_positive": (
                        positive / positive_total if positive_total > 0 else float("nan")
                    ),
                    "net_is_negative": bool(total < 0),
                }
            )
    per_fold_family = pd.DataFrame(rows)

    summary_keys = ["task", "model_id", "kind", "family"]
    summary = (
        per_fold_family.groupby(summary_keys, sort=False)
        .agg(
            n_features=("n_features", "first"),
            total_importance_mean=("total_importance", "mean"),
            total_importance_sd=("total_importance", "std"),
            share_of_positive_mean=("share_of_positive", "mean"),
            share_of_positive_sd=("share_of_positive", "std"),
            n_folds=("total_importance", "size"),
            n_folds_net_negative=("net_is_negative", "sum"),
        )
        .reset_index()
    )
    summary["rank"] = (
        summary.groupby(["task", "model_id", "kind"])["total_importance_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return per_fold_family, summary.sort_values(
        ["model_id", "kind", "rank"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# T81.3 -- SHAP
# ---------------------------------------------------------------------------


def _transformed(pipeline: Any, X: Any) -> np.ndarray:
    """The matrix the final estimator actually sees, with its width checked.

    SHAP indexes columns, so a pipeline step that dropped or reordered features
    would silently attach every value to the wrong name. The width is asserted
    rather than assumed; if a selector is ever added to the pipeline this raises
    instead of mislabelling a published figure.
    """
    matrix = np.asarray(X, dtype=float)
    for name, step in pipeline.named_steps.items():
        if name == "estimator":
            break
        matrix = step.transform(matrix)
    if matrix.shape[1] != np.asarray(X).shape[1]:
        raise ExplainError(
            "the pipeline changed the feature count from "
            + str(np.asarray(X).shape[1])
            + " to "
            + str(matrix.shape[1])
            + "; column indices no longer map to feature names"
        )
    return matrix


def shap_importance(
    *,
    task: str = "binary",
    data: Any = None,
    models: tuple[str, ...] = SHAP_MODELS,
    repeat: int = FOLD_REPEAT,
    max_rows: int = 400,
) -> tuple[Any, dict[str, Any]]:
    """Mean |SHAP| per feature for the tree models, on held-out rows.

    Returns ``(frame, report)``. ``report`` always says what happened -- the
    package version used, or the exact reason the section is absent. A missing
    optional dependency has to leave a record, not a gap.
    """
    import pandas as pd

    from src.models.smoke import load_task_data

    report: dict[str, Any] = {"requested_models": list(models)}
    try:
        import shap
    except ImportError as error:
        report["status"] = "skipped"
        report["reason"] = (
            "shap is not installed in this environment (" + str(error) + "). It is "
            "declared in requirements-extra.txt; install it and re-run to produce "
            "the SHAP section."
        )
        log.warning("SHAP unavailable: %s", report["reason"])
        return pd.DataFrame(), report

    report["shap_version"] = getattr(shap, "__version__", "unknown")
    resolved = data if data is not None else load_task_data(task)
    names = tuple(resolved.feature_names)
    fold = _folds(task, resolved, repeat)[0]
    report["fold_label"] = fold.label

    rows: list[dict[str, Any]] = []
    computed: list[str] = []
    failed: dict[str, str] = {}
    for fit in fit_models_for_fold(fold, resolved, models):
        try:
            estimator = _final_tree(fit.pipeline)
            held = np.asarray(fit.test_index, dtype=int)[:max_rows]
            matrix = _transformed(fit.pipeline, resolved.X[held])
            explainer = shap.TreeExplainer(estimator)
            values = np.asarray(explainer.shap_values(matrix))
            # Binary tree models return either (n, f) or (n, f, 2). Take the
            # positive class where a class axis exists, so the sign means
            # "pushes toward abnormal" in every case.
            if values.ndim == 3:
                values = values[:, :, -1]
            magnitude = np.abs(values).mean(axis=0)
            signed = values.mean(axis=0)
        except Exception as error:  # noqa: BLE001 - a SHAP failure is recorded, not fatal
            failed[fit.model_id] = type(error).__name__ + ": " + str(error)
            log.warning("SHAP failed for %s: %s", fit.model_id, error)
            continue

        computed.append(fit.model_id)
        for index, name in enumerate(names):
            rows.append(
                {
                    "task": task,
                    "model_id": fit.model_id,
                    "kind": "shap",
                    "fold_label": fit.fold_label,
                    "feature": name,
                    "mean_abs_shap": float(magnitude[index]),
                    "mean_signed_shap": float(signed[index]),
                    "n_rows": int(matrix.shape[0]),
                }
            )

    frame = pd.DataFrame(rows)
    if len(frame):
        frame["rank"] = (
            frame.groupby("model_id")["mean_abs_shap"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
    report["status"] = "computed" if computed else "failed"
    report["models_computed"] = computed
    report["models_failed"] = failed
    report["n_rows_per_model"] = max_rows
    report["note"] = (
        "TreeExplainer on the transformed matrix (imputer -> scaler), on the "
        "held-out rows of one fold. Signed values are for the positive class, so "
        "a positive mean pushes toward 'abnormal'. Unlike permutation "
        "importance, SHAP is computed on the model's own structure rather than "
        "by re-scoring, so it measures attribution rather than reliance."
    )
    return frame, report


def _final_tree(pipeline: Any) -> Any:
    from src.models.importance import _final_estimator

    estimator = _final_estimator(pipeline)
    if not hasattr(estimator, "feature_importances_"):
        raise ExplainError(
            type(estimator).__name__ + " is not a tree model; TreeExplainer does "
            "not apply"
        )
    return estimator


# ---------------------------------------------------------------------------
# FE-11
# ---------------------------------------------------------------------------


def top_features(aggregated: Any, *, kind: str = "permutation", n: int = TOP_N) -> Any:
    """FE-11's frame: the top ``n`` features per model under one measure.

    Permutation by default, because T81.2 makes it the primary reported measure
    and a "top features" table that led with the training-set-measured impurity
    ranking would be reporting the foil as the finding.
    """
    block = aggregated[aggregated["kind"] == kind]
    if not len(block):
        raise ExplainError("no rows of kind " + repr(kind) + " to rank")
    return (
        block[block["rank"] <= n]
        .sort_values(["model_id", "rank"])
        .reset_index(drop=True)
    )
