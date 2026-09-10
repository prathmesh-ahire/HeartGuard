"""EXP-F2 -- the optimization ablation, A9 and A10 (Phase 75).

**Nothing here fits a model.** Every number is read from per-fold metrics that
EXP-A1 and EXP-A2 already wrote, over the same DA-07 repeated 5x5 grouped fold
map with the same seed. That is not a shortcut, it is the only way the
comparison can honour ``identical_folds_as: EXP-A1``: re-fitting the stages here
would produce a second set of numbers for runs that already exist, and any
disagreement between them would be unresolvable.

The stages, in the pipeline's own order:

===  ====================================  ==========  =========================
#    Stage                                 Source      What it adds
===  ====================================  ==========  =========================
1    best individual model, defaults       EXP-A1      the baseline
2    equal-weight ensemble, defaults       EXP-A1      ensembling
3    best individual model, tuned          EXP-A2      hyperparameter search
4    equal-weight ensemble, tuned          EXP-A2      search + ensembling
5    optimized-weight ensemble, tuned      EXP-A2      + weight optimization
===  ====================================  ==========  =========================

**A9** is stage 1 against stage 2 (and 3 against 4): individual versus ensemble.
**A10** is stage 4 against stage 5: equal weights versus optimized weights.

**EXP-A1 declares no M7, and that is deliberate, not a gap.** M7 *is* an
optimization -- it searches ensemble weights -- so it has no place in an arm
whose definition is "default parameters, nothing optimized". A10 is therefore
measured at the tuned stage, where both ensembles exist on identical folds.

**"Best individual model" is chosen by the project's selection rule**
(sensitivity, then balanced accuracy), not by whichever model happens to make
the ablation look best, and the chosen id is written into the table so a reader
can see which model each stage is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "EXP_ID",
    "STAGES",
    "INDIVIDUAL_MODELS",
    "EQUAL_WEIGHT_ENSEMBLE",
    "OPTIMIZED_WEIGHT_ENSEMBLE",
    "load_stage_folds",
    "best_individual",
    "build_stages",
    "pairwise_comparison",
    "ensemble_weight_frame",
]

log = get_logger("evaluation.optimization_ablation")

EXP_ID = "EXP-F2"

#: The individual models. M6/M7 are ensembles and are handled separately.
INDIVIDUAL_MODELS: tuple[str, ...] = ("M1", "M3", "M4", "M5", "M8")
EQUAL_WEIGHT_ENSEMBLE = "M6"
OPTIMIZED_WEIGHT_ENSEMBLE = "M7"

#: (order, stage id, human label, source run, what selects the model).
#: ``kind`` is how the row's model is resolved: ``best`` runs the selection rule
#: over the individual models, ``fixed`` names one id.
STAGES: tuple[tuple[int, str, str, str, str, str], ...] = (
    (1, "default_single", "Best individual model, default parameters",
     "EXP-A1", "best", ""),
    (2, "default_ensemble", "Equal-weight ensemble, default parameters",
     "EXP-A1", "fixed", EQUAL_WEIGHT_ENSEMBLE),
    (3, "tuned_single", "Best individual model, tuned parameters",
     "EXP-A2", "best", ""),
    (4, "tuned_ensemble_equal", "Equal-weight ensemble, tuned members",
     "EXP-A2", "fixed", EQUAL_WEIGHT_ENSEMBLE),
    (5, "tuned_ensemble_optimized", "Optimized-weight ensemble, tuned members",
     "EXP-A2", "fixed", OPTIMIZED_WEIGHT_ENSEMBLE),
)

METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "accuracy",
)


def load_stage_folds(exp_id: str) -> Any:
    """One run's stored per-fold metrics. Read, never recomputed."""
    import pandas as pd

    from src.evaluation.experiment import Experiment

    path = Experiment.load(exp_id).output_dir() / "per_fold_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            str(path) + " is missing; EXP-F2 reads stored runs and cannot fit them"
        )
    frame = pd.read_csv(path)
    frame["exp_id"] = exp_id
    return frame


def best_individual(per_fold: Any) -> str:
    """The best individual model under the project's rule: sensitivity, then balanced.

    The same lexicographic rule the final binary model was selected by, applied
    to fold means. Using accuracy here would pick a different model -- on this
    corpus the two crown models 13 sensitivity points apart -- and would quietly
    make the ablation's baseline a different experiment's answer.
    """
    block = per_fold[per_fold["model_id"].isin(INDIVIDUAL_MODELS)]
    if block.empty:
        raise ValueError("no individual model rows to choose a baseline from")
    ranked = (
        block.groupby("model_id")[["sensitivity", "balanced_accuracy"]]
        .mean()
        .sort_values(["sensitivity", "balanced_accuracy"], ascending=False)
    )
    chosen = str(ranked.index[0])
    log.info(
        "best individual model: %s (sensitivity %.4f, balanced %.4f)",
        chosen,
        float(ranked.iloc[0]["sensitivity"]),
        float(ranked.iloc[0]["balanced_accuracy"]),
    )
    return chosen


def _fold_key(frame: Any) -> Any:
    return list(zip(frame["repeat"].astype(int), frame["fold"].astype(int), strict=True))


def build_stages() -> tuple[Any, Any]:
    """The five stages with incremental and cumulative deltas (T75.3, T75.4).

    Incremental deltas are computed on **shared folds only** and are paired: the
    stages ran the same fold map, so the difference is taken fold by fold and
    then averaged, not as a difference of two independently-averaged means. The
    two agree when both stages cover every fold, and only the paired form is
    honest when one does not.

    Returns the stage table and the per-fold rows behind it.
    """
    import numpy as np
    import pandas as pd

    loaded = {exp_id: load_stage_folds(exp_id) for exp_id in ("EXP-A1", "EXP-A2")}
    chosen: dict[str, str] = {}

    rows: list[dict[str, Any]] = []
    kept: list[Any] = []
    for order, stage, label, source, kind, fixed in STAGES:
        source_folds = loaded[source]
        if kind == "best":
            if source not in chosen:
                chosen[source] = best_individual(source_folds)
            model_id = chosen[source]
        else:
            model_id = fixed

        block = source_folds[source_folds["model_id"] == model_id].copy()
        if block.empty:
            log.warning("%s: %s has no %s rows; stage skipped", stage, source, model_id)
            continue
        block.insert(0, "stage_order", order)
        block.insert(1, "stage", stage)
        kept.append(block)

        row: dict[str, Any] = {
            "stage_order": order,
            "stage": stage,
            "comparison": label,
            "source_run": source,
            "model_id": model_id,
            "n_folds": len(block),
        }
        for metric in METRICS:
            if metric not in block.columns:
                continue
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[metric] = float(np.mean(finite)) if finite.size else float("nan")
            row[metric + "_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
        rows.append(row)

    per_fold = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    stages = pd.DataFrame(rows).sort_values("stage_order").reset_index(drop=True)

    baseline = str(stages.iloc[0]["stage"]) if len(stages) else ""
    for position, row in enumerate(stages.to_dict("records")):
        previous = str(stages.iloc[position - 1]["stage"]) if position else ""
        for metric in METRICS:
            if metric not in stages.columns:
                continue
            incremental = (
                _paired_mean_difference(per_fold, row["stage"], previous, metric)
                if previous
                else float("nan")
            )
            cumulative = (
                _paired_mean_difference(per_fold, row["stage"], baseline, metric)
                if position
                else 0.0
            )
            stages.loc[position, metric + "_incremental"] = incremental
            stages.loc[position, metric + "_cumulative"] = cumulative

    return stages, per_fold


def _paired_mean_difference(per_fold: Any, stage: str, against: str, metric: str) -> float:
    """Mean of (stage - against) over the folds both ran."""
    import numpy as np

    if not against or metric not in per_fold.columns:
        return float("nan")
    left = per_fold[per_fold["stage"] == stage].set_index(["repeat", "fold"])[metric]
    right = per_fold[per_fold["stage"] == against].set_index(["repeat", "fold"])[metric]
    shared = left.index.intersection(right.index)
    if not len(shared):
        return float("nan")
    difference = left.loc[shared].to_numpy(dtype=float) - right.loc[shared].to_numpy(dtype=float)
    finite = difference[np.isfinite(difference)]
    return float(np.mean(finite)) if finite.size else float("nan")


def pairwise_comparison(per_fold: Any, left_stage: str, right_stage: str) -> Any:
    """One named comparison (A9 or A10) as a per-fold paired frame."""
    import numpy as np
    import pandas as pd

    left = per_fold[per_fold["stage"] == left_stage].set_index(["repeat", "fold"])
    right = per_fold[per_fold["stage"] == right_stage].set_index(["repeat", "fold"])
    shared = sorted(left.index.intersection(right.index))
    if not shared:
        raise ValueError("no shared folds between " + left_stage + " and " + right_stage)

    rows = []
    for key in shared:
        row: dict[str, Any] = {
            "repeat": int(key[0]),
            "fold": int(key[1]),
            "left_stage": left_stage,
            "right_stage": right_stage,
            "left_model": str(left.loc[key, "model_id"]),
            "right_model": str(right.loc[key, "model_id"]),
        }
        for metric in METRICS:
            if metric not in left.columns:
                continue
            a = float(left.loc[key, metric])
            b = float(right.loc[key, metric])
            row[metric + "_left"] = a
            row[metric + "_right"] = b
            row[metric + "_delta"] = b - a
        rows.append(row)

    frame = pd.DataFrame(rows)
    for metric in METRICS:
        column = metric + "_delta"
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=float)
            frame.attrs[metric + "_wins"] = int(np.sum(values > 0))
            frame.attrs[metric + "_ties"] = int(np.sum(values == 0))
    return frame


def ensemble_weight_frame() -> Any:
    """SO-05's per-member ensemble weights, for G23.

    Read from the weight-stability table the optimizer wrote, not re-derived:
    the weights are a *result* of Phase 50/60-62 and re-deriving them here would
    be a second implementation of the shrinkage rule.
    """
    import pandas as pd

    from src.utils.config import load_config

    path = (
        Path(load_config("paths").require("outputs.search_optimization"))
        / "SO-05"
        / "weight_stability.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(str(path) + " is missing; G23 has no weights to plot")
    return pd.read_csv(path)

