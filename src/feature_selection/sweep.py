"""SO-04: the feature-count sweep, the final subset, and the 138-vs-subset comparison.

Three questions, in the order they have to be answered:

1. **How does performance move with feature count?** (T57.4) Every ranker is cut
   at every k in the configured grid and scored on inner validation folds. That
   is the trade-off curve, and G22 is drawn from it.
2. **Which subset ships?** (T57.5) The (ranker, k) pair with the lowest mean J,
   the multi-objective score -- performance, compactness and extraction cost in
   one number. FE-12 is written from it.
3. **Is it better than using all 138?** (T57.6) All 138 against the subset, on
   the same outer folds under the same seed.

WHERE THE SELECTION HAPPENS, AND WHY IT IS IN TWO PLACES
--------------------------------------------------------
The k is chosen on INNER validation folds. The chosen configuration is then
re-selected from scratch inside each outer fold's training rows and scored once
on that fold's test rows. Nothing that decides a feature ever sees a test row.

Which leaves the question FE-12 actually asks: what subset does a deployed model
use? The tempting answer -- rank on the whole matrix, take the top k -- would be
fitted on every test row in the project, and the number beside it in T57.6 would
then be describing a different subset from the one shipped. So FE-12 is instead
the CONSENSUS of the five per-fold selections: how often each feature survived,
cut at the same k. It is reproducible from the per-fold table, it never used a
row outside the fold that selected it, and the selection frequency it carries is
a stability measure rather than a claim.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.feature_selection import ranking as fr
from src.optimization import multi_objective as mo

__all__ = [
    "SweepError",
    "SubsetScore",
    "SweepResult",
    "FinalSubset",
    "load_settings",
    "score_subset",
    "sweep_feature_counts",
    "curve_frame",
    "recompute_j",
    "choose_configuration",
    "select_per_fold",
    "consensus_subset",
    "compare_all_versus_selected",
]

SUBSET_FILENAME = "selected_feature_subset.csv"
CURVE_FILENAME = "feature_count_sweep.csv"
COMPARISON_FILENAME = "all_features_vs_selected.csv"
PER_FOLD_FILENAME = "per_fold_selection.csv"
SETTINGS_FILENAME = "so04_settings.json"


class SweepError(RuntimeError):
    """The sweep cannot be assembled or run as configured."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def load_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """`optimization.feature_selection` from configs/models.yaml, with defaults."""
    if config is None:
        from src.utils.config import load_config

        config = load_config("models").get("optimization.feature_selection") or {}
    settings = dict(config)
    settings.setdefault("task", "binary")
    settings.setdefault("evaluation_model", "M4")
    settings.setdefault("rankers", ["mutual_info", "anova_f", "rf_importance", "gb_importance"])
    settings.setdefault("k_grid", [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 138])
    settings.setdefault("repeats", [0])
    settings.setdefault("inner_splits", 3)
    settings.setdefault(
        "rfecv",
        {"estimator": "logistic_regression", "step": 5,
         "min_features_to_select": 10, "inner_cv": 3},
    )
    return settings


# ---------------------------------------------------------------------------
# scoring one subset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetScore:
    """One (ranker, k) evaluated on one validation block."""

    ranker: str
    k: int
    outer_label: str
    inner_index: int
    columns: tuple[int, ...]
    metrics: dict[str, float]
    j: float
    j_terms: dict[str, float]
    seconds: float

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "ranker": self.ranker,
            "k": int(self.k),
            "outer_fold": self.outer_label,
            "inner_fold": int(self.inner_index),
            "n_selected": len(self.columns),
            "j": float(self.j),
            "seconds": float(self.seconds),
        }
        row.update({key: float(value) for key, value in self.metrics.items()})
        row.update(dict(self.j_terms))
        return row


def score_subset(
    columns: Sequence[int] | np.ndarray,
    model_id: str,
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    *,
    feature_names: Sequence[str],
    weights: mo.JWeights | None = None,
    cost_model: mo.FamilyCostModel | None = None,
    pipeline_config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], mo.JScore, float]:
    """Fit ``model_id`` on ``columns`` of the training block; score the validation block.

    Returns the full metric report, the J it earns, and the seconds it took. The
    metric report is the whole report, not the one number J needs -- research
    rule 6 does not stop applying because the caller only asked for macro-F1.
    """
    from sklearn.base import clone

    from src.evaluation import metrics as mt
    from src.models import estimators as est
    from src.models.pipeline import build_pipeline

    picked = np.sort(np.asarray(columns, dtype=int))
    if picked.size == 0:
        raise SweepError("an empty subset cannot be scored")

    features_train = np.asarray(x_train, dtype=float)[:, picked]
    features_val = np.asarray(x_val, dtype=float)[:, picked]
    targets_train = np.asarray(y_train)
    targets_val = np.asarray(y_val)
    labels = tuple(np.unique(np.concatenate([targets_train, targets_val])).tolist())

    started = time.perf_counter()
    estimator = est.build_estimator(model_id)
    pipeline = build_pipeline(
        clone(estimator),
        config=pipeline_config,
        y=targets_train,
        n_features=int(picked.size),
    )
    pipeline.fit(features_train, targets_train)
    predicted = np.asarray(pipeline.predict(features_val))
    seconds = time.perf_counter() - started

    if len(labels) == 2:
        report = mt.binary_metrics(
            targets_val, predicted, labels=list(labels), positive_label=labels[-1]
        )
    else:
        report = mt.multiclass_metrics(targets_val, predicted, labels=list(labels))
    scalar = {
        key: float(value)
        for key, value in report.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    scalar["macro_f1"] = mo.macro_f1(targets_val, predicted, labels=list(labels))

    names = [str(feature_names[int(c)]) for c in picked]
    j = mo.score_j(scalar["macro_f1"], names, weights=weights, cost_model=cost_model)
    return scalar, j, float(seconds)


# ---------------------------------------------------------------------------
# T57.4 -- the sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Every (ranker, k) evaluation, plus the settings that produced them."""

    task: str
    evaluation_model: str
    scores: list[SubsetScore] = field(default_factory=list)
    rfecv: list[dict[str, Any]] = field(default_factory=list)
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    settings: dict[str, Any] = field(default_factory=dict)

    def frame(self) -> Any:
        import pandas as pd

        if not self.scores:
            raise SweepError("the sweep produced no scores")
        return pd.DataFrame([score.as_row() for score in self.scores])


def sweep_feature_counts(
    data: Any,
    outer: Sequence[Any],
    *,
    settings: dict[str, Any] | None = None,
    seed: int | None = None,
    weights: mo.JWeights | None = None,
    cost_model: mo.FamilyCostModel | None = None,
    pipeline_config: dict[str, Any] | None = None,
    progress: Any = None,
) -> SweepResult:
    """Rank inside every inner training fold, cut at every k, score every cut.

    The ranker is fitted **once per (outer fold, inner split, ranker)** and cut
    at all eleven k values from that one fitting. Re-ranking per k would cost
    eleven times as much and, because a top-40 is a prefix of a top-60 under the
    same ranking, would return the identical subsets.
    """
    from src.optimization.base import inner_folds

    settings = load_settings(settings)
    if seed is None:
        from src.utils.seed import GLOBAL_SEED

        seed = GLOBAL_SEED
    weights = weights or mo.load_weights()
    cost_model = cost_model or mo.load_cost_model()

    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    names = list(data.feature_names)
    n_features = int(features.shape[1])
    if len(names) != n_features:
        raise SweepError(
            "the matrix has " + str(n_features) + " columns and "
            + str(len(names)) + " feature names"
        )

    grid = sorted({min(int(k), n_features) for k in settings["k_grid"]})
    rankers = list(settings["rankers"])
    model_id = str(settings["evaluation_model"])
    result = SweepResult(
        task=str(data.task),
        evaluation_model=model_id,
        settings={**settings, "k_grid": grid, "seed": int(seed), "n_features": n_features},
    )
    started = time.perf_counter()

    for fold in outer:
        splits = inner_folds(
            fold, data.y, data.groups, n_splits=int(settings["inner_splits"]), seed=seed
        )
        for split in splits:
            x_train = features[split.train_index]
            y_train = targets[split.train_index]
            x_val = features[split.val_index]
            y_val = targets[split.val_index]
            for kind in rankers:
                ranked = fr.rank_features(kind, x_train, y_train, seed=seed)
                for k in grid:
                    columns = ranked.top(k)
                    metrics, j, seconds = score_subset(
                        columns,
                        model_id,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        feature_names=names,
                        weights=weights,
                        cost_model=cost_model,
                        pipeline_config=pipeline_config,
                    )
                    result.scores.append(
                        SubsetScore(
                            ranker=kind,
                            k=int(k),
                            outer_label=str(fold.label),
                            inner_index=int(split.index),
                            columns=tuple(int(c) for c in columns),
                            metrics=metrics,
                            j=float(j.value),
                            j_terms=j.as_dict(),
                            seconds=seconds,
                        )
                    )
                    if progress is not None:
                        progress(result.scores[-1])

    result.seconds = time.perf_counter() - started
    return result


def curve_frame(result: SweepResult | Any) -> Any:
    """The trade-off curve: one row per (ranker, k), averaged over every fold.

    Standard deviation is carried beside every mean because the whole point of
    the curve is to read where it flattens, and a difference smaller than the
    fold-to-fold spread is not a place where anything flattened.
    """
    frame: Any = result.frame() if isinstance(result, SweepResult) else result
    keep = [
        column
        for column in ("macro_f1", "balanced_accuracy", "sensitivity", "specificity",
                       "accuracy", "f1", "j", "seconds")
        if column in frame.columns
    ]
    grouped = frame.groupby(["ranker", "k"], sort=True)
    summary = grouped[keep].agg(["mean", "std"])
    summary.columns = [left + "_" + right for left, right in summary.columns.to_flat_index()]
    summary["n_evaluations"] = grouped.size()
    return summary.reset_index().sort_values(["ranker", "k"]).reset_index(drop=True)


def recompute_j(
    frame: Any,
    *,
    weights: mo.JWeights | None = None,
    cost_model: mo.FamilyCostModel | None = None,
) -> Any:
    """Recompute the ``j`` column of a sweep table under a different weighting.

    The sweep is ~40 minutes of fitting; J is arithmetic over three numbers the
    table already carries -- the macro-F1 it measured, the subset size, and the
    families that subset still needs extracted. So a change to alpha, beta,
    gamma or the cost model is applied to the measurement rather than paying for
    the measurement again.

    This is the same lesson as `Trial.fold_components` in Phase 54: record enough
    of the decomposition that the score can be recomputed, and an objective
    change costs seconds instead of hours.
    """
    weights = weights or mo.load_weights()
    cost_model = cost_model or mo.load_cost_model()
    for column in ("macro_f1", "n_selected", "families"):
        if column not in frame.columns:
            raise SweepError(
                "cannot recompute J: the sweep table has no " + column + " column"
            )

    out = frame.copy()
    normalized_features = out["n_selected"].astype(float) / float(weights.n_features_total)
    normalized_time = out["families"].map(
        lambda value: cost_model.normalized(str(value).split(";"))
    )
    out["normalized_features"] = normalized_features
    out["normalized_inference_time"] = normalized_time
    out["term_performance"] = weights.alpha * (1.0 - out["macro_f1"].astype(float))
    out["term_features"] = weights.beta * normalized_features
    out["term_time"] = weights.gamma * normalized_time
    out["j"] = out["term_performance"] + out["term_features"] + out["term_time"]
    return out


# ---------------------------------------------------------------------------
# T57.5 -- choosing the configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalSubset:
    """The shipped subset, and everything needed to defend the choice of it."""

    ranker: str
    k: int
    columns: tuple[int, ...]
    names: tuple[str, ...]
    frequency: dict[str, int]
    n_folds: int
    j: float
    macro_f1: float
    best_performance_ranker: str = ""
    best_performance_k: int = 0
    best_performance_macro_f1: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ranker": self.ranker,
            "k": int(self.k),
            "n_selected": len(self.columns),
            "mean_inner_j": float(self.j),
            "mean_inner_macro_f1": float(self.macro_f1),
            "n_folds": int(self.n_folds),
            "best_performance_ranker": self.best_performance_ranker,
            "best_performance_k": int(self.best_performance_k),
            "best_performance_macro_f1": float(self.best_performance_macro_f1),
            "macro_f1_given_up": float(self.best_performance_macro_f1 - self.macro_f1),
        }


def choose_configuration(
    result: SweepResult | Any, *, n_standard_errors: float | None = None
) -> tuple[str, int, dict[str, Any]]:
    """The (ranker, k) with the lowest mean J over every inner evaluation.

    Ties -- exact ties do happen, since two rankers can agree on a small k --
    are broken toward the SMALLER k and then the alphabetically first ranker, so
    the choice is deterministic and, where J genuinely cannot separate two
    options, lands on the more compact one.

    ``n_standard_errors`` adds a PERFORMANCE GUARD, and is the same
    one-standard-error rule Phase 50 applies to M7's weights. J's weighting fixes
    an exchange rate between macro-F1 and compactness, and at any weighting that
    rate is a judgement, not a measurement -- so the guard restricts J's choice
    to the configurations whose macro-F1 is statistically indistinguishable from
    the best, and lets J pick the cheapest of those. Below the guard, a subset is
    not "cheaper", it is worse. ``None`` disables it and J chooses unrestricted.
    """
    # Accepts a SweepResult or the sweep table read back from disk, so a change
    # to the SELECTION RULE can be applied to an existing sweep in seconds
    # instead of re-running the ~40 minutes of fitting that produced it.
    frame: Any = result.frame() if isinstance(result, SweepResult) else result
    grouped = (
        frame.groupby(["ranker", "k"], as_index=False)
        .agg(
            mean_j=("j", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            n=("macro_f1", "size"),
        )
    )
    grouped["se_macro_f1"] = grouped["std_macro_f1"] / np.sqrt(grouped["n"].clip(lower=1))

    by_performance = grouped.sort_values(
        ["mean_macro_f1", "k", "ranker"], ascending=[False, True, True]
    ).iloc[0]

    eligible = grouped
    guard = float("nan")
    if n_standard_errors is not None:
        guard = float(
            by_performance.mean_macro_f1
            - float(n_standard_errors) * float(by_performance.se_macro_f1)
        )
        eligible = grouped[grouped["mean_macro_f1"] >= guard]
        if eligible.empty:  # pragma: no cover - the best point always qualifies
            eligible = grouped

    best = eligible.sort_values(
        ["mean_j", "k", "ranker"], ascending=[True, True, True]
    ).iloc[0]
    detail: dict[str, Any] = {
        "mean_j": float(best.mean_j),
        "mean_macro_f1": float(best.mean_macro_f1),
        "se_macro_f1": float(best.se_macro_f1),
        "selection_rule": (
            "min_j" if n_standard_errors is None
            else "min_j_within_" + str(n_standard_errors) + "_se_of_best_macro_f1"
        ),
        "performance_guard": guard,
        "n_eligible": len(eligible),
        "n_configurations": len(grouped),
        "best_performance_ranker": str(by_performance.ranker),
        "best_performance_k": int(by_performance.k),
        "best_performance_macro_f1": float(by_performance.mean_macro_f1),
        "best_performance_se": float(by_performance.se_macro_f1),
        "unguarded_j_ranker": str(
            grouped.sort_values(["mean_j", "k", "ranker"]).iloc[0].ranker
        ),
        "unguarded_j_k": int(grouped.sort_values(["mean_j", "k", "ranker"]).iloc[0].k),
        "unguarded_j_macro_f1": float(
            grouped.sort_values(["mean_j", "k", "ranker"]).iloc[0].mean_macro_f1
        ),
    }
    return str(best.ranker), int(best.k), detail


def select_per_fold(
    data: Any,
    outer: Sequence[Any],
    ranker: str,
    k: int,
    *,
    seed: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Re-rank inside each outer fold's TRAINING rows and cut at ``k``.

    Not the inner-fold rankings reused: the point selected on inner folds is a
    configuration (which ranker, how many features), and the subset it implies
    is then derived from all the training data that fold is entitled to -- the
    same nested protocol the hyperparameter search refits under.
    """
    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    picked: dict[str, tuple[int, ...]] = {}
    for fold in outer:
        train = np.asarray(fold.train_index, dtype=int)
        ranked = fr.rank_features(ranker, features[train], targets[train], seed=seed)
        picked[str(fold.label)] = tuple(int(c) for c in ranked.top(int(k)))
    return picked


def consensus_subset(
    per_fold: dict[str, tuple[int, ...]],
    k: int,
    feature_names: Sequence[str],
) -> tuple[tuple[int, ...], dict[str, int]]:
    """The ``k`` features chosen by the most folds; ties broken by column position.

    This is FE-12. It is a vote over the per-fold selections rather than a fresh
    ranking over every row, because a ranking fitted on every row would have been
    fitted on every test row too, and the T57.6 numbers would then be describing
    a subset nobody evaluated.
    """
    names = list(feature_names)
    counts: dict[int, int] = {}
    for columns in per_fold.values():
        for column in columns:
            counts[int(column)] = counts.get(int(column), 0) + 1
    if not counts:
        raise SweepError("no per-fold selections to take a consensus of")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    chosen = tuple(sorted(column for column, _ in ordered[: int(k)]))
    frequency = {names[column]: int(counts[column]) for column in chosen}
    return chosen, frequency


# ---------------------------------------------------------------------------
# T57.6 -- all 138 versus the selected subset
# ---------------------------------------------------------------------------


def compare_all_versus_selected(
    data: Any,
    outer: Sequence[Any],
    per_fold: dict[str, tuple[int, ...]],
    *,
    model_id: str,
    seed: int | None = None,
    weights: mo.JWeights | None = None,
    cost_model: mo.FamilyCostModel | None = None,
    pipeline_config: dict[str, Any] | None = None,
    extra_configurations: dict[str, dict[str, tuple[int, ...]]] | None = None,
) -> Any:
    """Score every configuration on every outer test fold. Same folds, same seed.

    ``per_fold`` maps outer-fold label to the columns selected inside THAT fold's
    training rows, which is what makes this comparison honest: the "selected"
    arm never sees a subset chosen with help from the rows it is scored on.
    """
    import pandas as pd

    del seed
    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    names = list(data.feature_names)
    all_columns = tuple(range(int(features.shape[1])))

    configurations: dict[str, dict[str, tuple[int, ...]]] = {
        "all_features": {str(fold.label): all_columns for fold in outer},
        "selected_subset": per_fold,
    }
    configurations.update(extra_configurations or {})

    rows: list[dict[str, Any]] = []
    for configuration, mapping in configurations.items():
        for fold in outer:
            label = str(fold.label)
            if label not in mapping:
                raise SweepError(
                    configuration + " has no columns for outer fold " + label
                )
            columns = mapping[label]
            train = np.asarray(fold.train_index, dtype=int)
            test = np.asarray(fold.test_index, dtype=int)
            metrics, j, seconds = score_subset(
                columns,
                model_id,
                features[train],
                targets[train],
                features[test],
                targets[test],
                feature_names=names,
                weights=weights,
                cost_model=cost_model,
                pipeline_config=pipeline_config,
            )
            row: dict[str, Any] = {
                "configuration": configuration,
                "outer_fold": label,
                "repeat": int(fold.repeat),
                "fold": int(fold.fold),
                "model_id": model_id,
                "n_features": len(columns),
                "j": float(j.value),
                "fit_predict_seconds": float(seconds),
                "n_train": int(train.size),
                "n_test": int(test.size),
            }
            row.update({key: float(value) for key, value in metrics.items()})
            row.update(j.as_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def output_dir(out_dir: str | Path | None = None) -> Path:
    """``outputs/05_search_optimization/SO-04/``, created."""
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    root = (
        Path(out_dir)
        if out_dir is not None
        else Path(load_config("paths").require("outputs.search_optimization"))
    )
    return Path(ensure_dir(root / "SO-04"))
