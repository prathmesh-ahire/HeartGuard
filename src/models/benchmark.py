"""SVM cost and calibration benchmarks (T47.4, T47.5).

Two questions Phase 47 has to answer with numbers rather than intuition.

**How much does one M3 fit cost?** An RBF SVM is the one model here whose cost
grows worse than linearly in the number of records -- roughly between quadratic
and cubic -- so a search budget that is fine for the 124-record PASCAL A track
can be hours on the 3,240-record PhysioNet one. The answer decides whether the
search has to subsample and whether ``cache_size`` needs raising, and it is
measured on the real matrix at real fold sizes, not extrapolated.

**Which calibrator, and fitted how?** ``sigmoid`` versus ``isotonic``, and
``ensemble=True`` (average the per-calibration-fold models) versus
``ensemble=False`` (calibrate one model refitted on everything). Four
combinations, judged on Brier score and expected calibration error, because a
soft-voting ensemble is only as good as the probabilities it averages -- a
member with well-ranked but badly-scaled scores drags the fusion without
showing up in its own AUC.

Everything here fits on **fold 0's training rows and scores fold 0's test rows**,
through the Phase 44 pipeline. The other 24 folds are untouched: this is a
budgeting measurement, and spending the real folds on it would make the eventual
result a re-use of data already looked at.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "FIT_TIME_FILENAME",
    "CALIBRATION_FILENAME",
    "fit_time_benchmark",
    "calibration_benchmark",
    "search_budget_estimate",
]

log = get_logger("models.benchmark")

FIT_TIME_FILENAME = "svm_fit_time_benchmark.csv"
CALIBRATION_FILENAME = "svm_calibration_benchmark.csv"

#: The corners of M3's declared search space that cost the most. A benchmark at
#: default hyperparameters understates a search: `gamma` well above "scale"
#: makes almost every training row a support vector.
_SEARCH_CORNERS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("default (C=1, gamma=scale)", {"C": 1.0, "gamma": "scale"}),
    ("high C (C=1000, gamma=scale)", {"C": 1000.0, "gamma": "scale"}),
    ("high gamma (C=1, gamma=10)", {"C": 1.0, "gamma": 10.0}),
    ("both high (C=1000, gamma=10)", {"C": 1000.0, "gamma": 10.0}),
    ("low both (C=0.01, gamma=1e-4)", {"C": 0.01, "gamma": 1.0e-4}),
)


def _subsample(n: int, total: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(total)[: min(n, total)]


def _timed_fit(estimator: Any, X: Any, y: Any) -> tuple[Any, float]:
    from src.models import pipeline as pl

    built = pl.build_pipeline(estimator, y=y, apply_class_weight=False)
    started = time.perf_counter()
    built.fit(X, y)
    return built, time.perf_counter() - started


# ---------------------------------------------------------------------------
# T47.4 -- fit time
# ---------------------------------------------------------------------------


def fit_time_benchmark(
    data: Any,
    *,
    sizes: tuple[int, ...] = (500, 1000, 2000, 3240),
    cache_sizes: tuple[int, ...] = (200, 500, 1000),
    cache_repeats: int = 5,
    seed: int = 42,
) -> Any:
    """Measure the bare and calibrated SVM fit cost against n and ``cache_size``.

    ``data`` is a :class:`~src.models.smoke.TaskData`. The n-and-hyperparameter
    arms are single runs: the differences that matter there are factors, not
    percents, and averaging them on a laptop under a browser is not more truthful
    -- just slower. The ``cache_size`` arm is repeated and reported as a median,
    because there the question is whether an effect exists at all, and a single
    timing cannot answer it.
    """
    import pandas as pd
    from sklearn.svm import SVC

    from src.models.calibration import CalibratedSVM

    rows: list[dict[str, Any]] = []
    total = data.n_records

    for n in sizes:
        if n > total:
            continue
        index = _subsample(n, total, seed)
        X, y = data.X[index], data.y[index]

        for label, params in _SEARCH_CORNERS:
            svc = SVC(
                kernel="rbf", class_weight="balanced", cache_size=500,
                random_state=seed, **params,
            )
            fitted, seconds = _timed_fit(svc, X, y)
            rows.append(
                {
                    "variant": "bare SVC",
                    "hyperparameters": label,
                    "n_train": int(n),
                    "cache_size_mb": 500,
                    "fit_seconds": round(seconds, 4),
                    "n_support_vectors": int(
                        fitted.named_steps["estimator"].n_support_.sum()
                    ),
                }
            )

    # The cache_size arm is the one place a repeat is worth its cost. It has to
    # separate a real effect from run-to-run jitter, and the effect it is looking
    # for -- if there is one at all -- is smaller than the jitter of a single
    # timing on a laptop. Everything else in this table is measuring factors.
    for cache in cache_sizes:
        index = _subsample(total, total, seed)
        X, y = data.X[index], data.y[index]
        measured: list[float] = []
        support = 0
        for _ in range(cache_repeats):
            svc = SVC(
                kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                cache_size=cache, random_state=seed,
            )
            fitted, seconds = _timed_fit(svc, X, y)
            measured.append(seconds)
            support = int(fitted.named_steps["estimator"].n_support_.sum())
        rows.append(
            {
                "variant": "bare SVC",
                "hyperparameters": "default (C=1, gamma=scale)",
                "n_train": int(total),
                "cache_size_mb": int(cache),
                "fit_seconds": round(float(np.median(measured)), 4),
                "fit_seconds_min": round(min(measured), 4),
                "fit_seconds_max": round(max(measured), 4),
                "n_repeats": int(cache_repeats),
                "n_support_vectors": support,
            }
        )

    for n in sizes:
        if n > total:
            continue
        index = _subsample(n, total, seed)
        X, y = data.X[index], data.y[index]
        for method in ("sigmoid", "isotonic"):
            wrapped = CalibratedSVM(
                estimator=SVC(
                    kernel="rbf", C=1.0, gamma="scale", cache_size=500,
                    random_state=seed,
                ),
                method=method,
                cv=3,
                class_weight="balanced",
            )
            fitted, seconds = _timed_fit(wrapped, X, y)
            rows.append(
                {
                    "variant": "CalibratedSVM (" + method + ", cv=3)",
                    "hyperparameters": "default (C=1, gamma=scale)",
                    "n_train": int(n),
                    "cache_size_mb": 500,
                    "fit_seconds": round(seconds, 4),
                    "n_support_vectors": -1,
                }
            )

    frame = pd.DataFrame(rows)
    frame["kernel_matrix_mb"] = (frame["n_train"] ** 2 * 8 / 1024**2).round(1)
    return frame


def search_budget_estimate(
    benchmark: Any, *, n_trials: int = 200, inner_folds: int = 3, outer_folds: int = 25
) -> dict[str, float]:
    """What a full M3 search would cost, from the measured worst corner.

    Deliberately built on the **slowest** hyperparameter corner rather than the
    mean. A budget sized on the average trial is exceeded by the first expensive
    region the search wanders into, which is precisely where a search spends its
    time once it starts converging.

    Two nested CVs are in play and they are not the same thing. ``inner_folds``
    is the *search's* inner cross-validation -- how many times one candidate is
    scored. The calibrator's own 3-way split is already inside
    ``calibration_overhead_x``, measured rather than assumed. Multiplying the two
    is correct; conflating them undercounts the budget threefold.
    """
    calibrated = benchmark[benchmark["variant"].str.startswith("CalibratedSVM")]
    bare = benchmark[benchmark["variant"] == "bare SVC"]

    largest = int(bare["n_train"].max())
    worst = float(bare[bare["n_train"] == largest]["fit_seconds"].max())
    typical_calibrated = float(
        calibrated[calibrated["n_train"] == calibrated["n_train"].max()][
            "fit_seconds"
        ].mean()
    )
    overhead = typical_calibrated / max(
        float(bare[bare["n_train"] == largest]["fit_seconds"].min()), 1e-9
    )

    per_trial = worst * overhead * inner_folds
    return {
        "worst_bare_fit_seconds": round(worst, 4),
        "calibration_overhead_x": round(overhead, 2),
        "seconds_per_trial": round(per_trial, 2),
        "search_minutes_one_outer_fold": round(per_trial * n_trials / 60.0, 1),
        "search_hours_all_outer_folds": round(
            per_trial * n_trials * outer_folds / 3600.0, 2
        ),
    }


# ---------------------------------------------------------------------------
# T47.5 -- calibration quality
# ---------------------------------------------------------------------------


def calibration_benchmark(data: Any, fold: Any, *, seed: int = 42) -> Any:
    """Score all four calibrator variants, plus a grouped-split arm, on fold 0.

    The grouped arm exists because the calibrator's internal CV is not
    subject-aware by default. It is a within-training-fold concern only -- the
    outer test fold is untouched in every arm -- so the difference between the
    grouped and ungrouped rows measures how much the default arrangement
    flatters its own sigmoid, not how much the reported metric is inflated.
    """
    import pandas as pd
    from sklearn.svm import SVC

    from src.evaluation import metrics as mt
    from src.models import estimators as est
    from src.models.calibration import CalibratedSVM, grouped_calibration_cv

    train_index = np.asarray(fold.train_index, dtype=int)
    test_index = np.asarray(fold.test_index, dtype=int)
    X_train, y_train = data.X[train_index], data.y[train_index]
    X_test, y_test = data.X[test_index], data.y[test_index]
    labels = tuple(np.unique(data.y).tolist())

    grouped = grouped_calibration_cv(
        data.groups[train_index], y_train, n_splits=3, seed=seed
    )

    arms: list[tuple[str, str, bool, Any]] = []
    for method in ("sigmoid", "isotonic"):
        for ensemble in (True, False):
            arms.append((method, "ungrouped 3-fold", ensemble, 3))
        arms.append((method, "subject-grouped 3-fold", True, grouped))

    rows: list[dict[str, Any]] = []
    for method, split_kind, ensemble, cv in arms:
        wrapped = CalibratedSVM(
            estimator=SVC(
                kernel="rbf", C=1.0, gamma="scale", cache_size=500, random_state=seed
            ),
            method=method,
            cv=cv,
            class_weight="balanced",
            ensemble=ensemble,
        )
        fitted, seconds = _timed_fit(wrapped, X_train, y_train)
        y_pred = np.asarray(fitted.predict(X_test))
        proba = np.asarray(fitted.predict_proba(X_test))
        classes = np.asarray(fitted.named_steps["estimator"].classes_)

        report = est.probability_report(
            proba, n_classes=len(labels), y_pred=y_pred, classes=classes
        )
        scores = mt.binary_metrics(y_test, y_pred, proba, labels=labels, positive_label=1)
        rows.append(
            {
                "method": method,
                "calibration_split": split_kind,
                "ensemble": ensemble,
                "fit_seconds": round(seconds, 4),
                "balanced_accuracy": round(scores["balanced_accuracy"], 4),
                "sensitivity": round(scores["sensitivity"], 4),
                "specificity": round(scores["specificity"], 4),
                "roc_auc": round(scores["roc_auc"], 4),
                "brier": round(mt.brier_score(y_test, proba, labels=labels), 4),
                "ece": round(
                    mt.expected_calibration_error(y_test, proba, labels=labels), 4
                ),
                "proba_min": report.min_value,
                "proba_max": report.max_value,
                "max_row_sum_error": report.max_row_sum_error,
                "n_nan": report.n_nan,
                "agrees_with_predict": report.agrees_with_predict,
                "well_formed": report.is_well_formed,
            }
        )
        log.info(
            "M3 %s / %s / ensemble=%s: brier %.4f, ece %.4f",
            method,
            split_kind,
            ensemble,
            rows[-1]["brier"],
            rows[-1]["ece"],
        )

    return pd.DataFrame(rows)
