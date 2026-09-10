"""Calibration analysis over the stored out-of-fold probabilities (Phase 78).

Every experiment in this project already writes `predictions.parquet` -- one row
per (model, fold, record) with the full probability vector. Phase 78 asks what
those probabilities are *worth*: a model that says 0.9 should be right about 90%
of the time, and the four headline metrics say nothing about whether it is.

## Why this reads the stored predictions rather than refitting

The probabilities were produced inside the training fold under the Phase 43
driver, by the fitted pipeline that Phase 63-70 recorded. Refitting to measure
calibration would measure a *different* model -- and a calibration number that
does not belong to the model the tables report is worse than no number. The one
place a fit happens here is :func:`svm_calibration_comparison`, where comparing
two calibrators IS the question (T78.4), and there the fold map and the seed are
the experiment's own.

## Two reliability definitions, and why both exist

`src/reporting/curves.py` already writes a reliability curve: predicted
probability of the **positive class** against the observed positive rate. That is
the right curve for a binary screening operating point and the dashboard uses it.

It does not generalise. PASCAL A has four classes and no positive class, so
Phase 78 measures reliability the way :func:`metrics.expected_calibration_error`
does -- **confidence of the predicted class against the accuracy of that
prediction**. The two agree on nothing numerically and must never share an axis:
the confidence curve is bounded below by 1/n_classes, the positive-class curve by
0. `kind` on every emitted frame names which one produced the row.

## Bin counts (T78.3)

`RELIABILITY_BINS` is the reported grid; `BIN_SWEEP` is the sensitivity check.
ECE is not bin-count invariant -- more bins always finds more miscalibration,
because a bin with one record in it has zero within-bin averaging to hide behind
-- so reporting a single ECE without its bin count is reporting half a number.
Every emitted row carries `n_bins`.

## Per-fold, then averaged. Never pooled.

The binary map is repeated 5x5 grouped CV, so every record appears once per
repeat and pooling all 25 folds would count the corpus five times -- the same
trap `curves.py` documents. Brier and ECE are computed inside each fold and
averaged, with the SD kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "RELIABILITY_BINS",
    "BIN_SWEEP",
    "CONFIDENCE_BINS",
    "RUNS",
    "RunSpec",
    "CalibrationError",
    "run_specs",
    "load_predictions",
    "proba_matrix",
    "per_fold_calibration",
    "summarise_calibration",
    "reliability_frame",
    "confidence_histogram",
    "svm_calibration_comparison",
    "final_model_calibration",
]

log = get_logger("evaluation.calibration")

#: The reported reliability / ECE grid. Ten equal-width bins is what
#: `metrics.ECE_BINS` and `curves.CALIBRATION_BINS` both use, so the three
#: modules cannot disagree about what "the" ECE of a model is.
RELIABILITY_BINS = 10

#: The T78.3 sensitivity check: the same predictions re-binned. A model whose
#: ECE triples between 5 and 20 bins is not well calibrated at 5 bins, it is
#: under-resolved at 5 bins.
BIN_SWEEP: tuple[int, ...] = (5, 10, 15, 20)

#: G33's histogram grid. Finer than the reliability grid because a histogram has
#: no within-bin averaging to protect and the shape is the point.
CONFIDENCE_BINS = 20


class CalibrationError(RuntimeError):
    """A calibration analysis cannot be run as asked."""


@dataclass(frozen=True)
class RunSpec:
    """One stored run: where its predictions are and which label space they use.

    ``labels`` and ``class_names`` are declared here rather than only derived,
    and :func:`run_specs` cross-checks the declaration against the experiment
    config. Rule 4 is the reason: five label spaces that must never merge are
    safer written down twice and compared than inferred once.
    """

    run: str
    directory: str
    task: str
    exp_id: str
    variant: str | None
    labels: tuple[int, ...]
    class_names: tuple[str, ...]
    note: str = ""


#: Every run that holds out-of-fold probabilities, covering all five label
#: spaces. EXP-A1 and EXP-A2 are both the binary task and are reported
#: separately: they are the untuned and tuned arms, and tuning moves calibration.
RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "EXP-A1",
        "outputs/06_binary_results/EXP-A1",
        "binary",
        "EXP-A1",
        None,
        (0, 1),
        ("normal", "abnormal"),
        "config defaults, 25 folds",
    ),
    RunSpec(
        "EXP-A2",
        "outputs/06_binary_results/EXP-A2",
        "binary",
        "EXP-A2",
        None,
        (0, 1),
        ("normal", "abnormal"),
        "nested Bayesian search, 25 folds -- the headline binary run",
    ),
    RunSpec(
        "EXP-B1",
        "outputs/07_multiclass_results/EXP-B1",
        "pascal_a",
        "EXP-B1",
        None,
        (0, 1, 2, 3),
        ("normal", "murmur", "extrahls", "artifact"),
        "PASCAL A, n=124 records; 'artifact' is a recording-quality class",
    ),
    RunSpec(
        "EXP-B2",
        "outputs/07_multiclass_results/EXP-B2",
        "pascal_b",
        "EXP-B2",
        None,
        (0, 1, 2),
        ("normal", "murmur", "extrastole"),
        "PASCAL B, n=461 records, 5 folds",
    ),
    RunSpec(
        "EXP-C1-three_class",
        "outputs/08_circor_external_validation/EXP-C1-three_class",
        "circor_murmur",
        "EXP-C1",
        "three_class",
        (0, 1, 2),
        ("Absent", "Present", "Unknown"),
        "CirCor murmur, 3-class (matches the 2022 Challenge)",
    ),
    RunSpec(
        "EXP-C1-two_class",
        "outputs/08_circor_external_validation/EXP-C1-two_class",
        "circor_murmur",
        "EXP-C1",
        "two_class",
        (0, 1),
        ("Absent", "Present"),
        "CirCor murmur, 2-class over the 874 known-murmur patients",
    ),
    RunSpec(
        "EXP-C2",
        "outputs/08_circor_external_validation/EXP-C2",
        "circor_outcome",
        "EXP-C2",
        None,
        (0, 1),
        ("Normal", "Abnormal"),
        "CirCor clinical outcome, 5 patient-grouped folds",
    ),
    RunSpec(
        "EXP-D1",
        "outputs/08_circor_external_validation/EXP-D1",
        "circor_outcome",
        "EXP-D1",
        None,
        (0, 1),
        ("Normal", "Abnormal"),
        (
            "ONE external evaluation, not cross-validation: n_folds=1, so every "
            "SD is undefined. Adult-to-paediatric transfer -- the population "
            "mismatch, not the method, dominates this row."
        ),
    ),
)


def run_specs(*, verify: bool = True) -> tuple[RunSpec, ...]:
    """:data:`RUNS`, with each label space checked against the experiment config.

    EXP-D1 declares no ``task`` of its own (it trains on binary and tests on
    ``circor_outcome``), so it is checked against EXP-C2, which owns that label
    space. Skipping the check would leave the one run whose label space is
    genuinely ambiguous as the only unverified one.
    """
    if not verify:
        return RUNS

    from src.evaluation.experiment import Experiment

    for spec in RUNS:
        owner = "EXP-C2" if spec.exp_id == "EXP-D1" else spec.exp_id
        experiment = Experiment.load(owner)
        if spec.variant:
            experiment = experiment.for_variant(spec.variant)
        if tuple(experiment.labels) != spec.labels:
            raise CalibrationError(
                spec.run
                + ": declared labels "
                + str(spec.labels)
                + " disagree with "
                + owner
                + "'s "
                + str(tuple(experiment.labels))
            )
        if tuple(experiment.class_names) != spec.class_names:
            raise CalibrationError(
                spec.run
                + ": declared class names "
                + str(spec.class_names)
                + " disagree with "
                + owner
                + "'s "
                + str(tuple(experiment.class_names))
            )
    return RUNS


def load_predictions(spec: RunSpec, *, root: Path | None = None) -> Any:
    """One run's ``predictions.parquet``, or ``None`` if it has not been run."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    base = root or PROJECT_ROOT
    path = base / spec.directory / "predictions.parquet"
    if not path.is_file():
        log.warning("%s: no predictions.parquet, skipping", spec.run)
        return None
    frame = pd.read_parquet(path)
    expected = ["proba_" + str(label) for label in spec.labels]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise CalibrationError(
            str(path) + " is missing probability column(s) " + ", ".join(missing)
        )
    return frame


def proba_matrix(frame: Any, labels: tuple[int, ...]) -> np.ndarray:
    """The (n_records, n_labels) probability matrix in declared label order."""
    columns = ["proba_" + str(label) for label in labels]
    return frame[columns].to_numpy(dtype=float)


def _confidence_and_correct(
    frame: Any, labels: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence of the argmax class, whether it was right, and the truth.

    The argmax class, deliberately, and **not** ``y_pred``. M6/M7 predict by a
    tuned threshold rather than by argmax, so their stored ``y_pred`` can differ
    from the argmax of their own probability vector -- that is the documented
    design, not a bug. Reliability is a statement about the probability vector,
    so it has to be measured against the vector's own argmax; mixing in a
    thresholded label would produce a curve that is neither the model's
    calibration nor its decision rule.
    """
    matrix = proba_matrix(frame, labels)
    order = np.asarray(labels)
    confidence = matrix.max(axis=1)
    argmax_label = order[matrix.argmax(axis=1)]
    truth = frame["y_true"].to_numpy()
    return confidence, (argmax_label == truth).astype(float), truth


def per_fold_calibration(
    frame: Any, labels: tuple[int, ...], *, bins: tuple[int, ...] = BIN_SWEEP
) -> Any:
    """Brier and ECE for every ``(model, fold)``, at each requested bin count.

    One row per (model, fold, n_bins). Brier does not depend on the binning and
    is repeated across a model's rows unchanged, which is the point: it is the
    bin-free companion to a bin-dependent ECE.
    """
    import pandas as pd

    from src.evaluation import metrics as mt

    rows: list[dict[str, Any]] = []
    for (model_id, fold_label), block in frame.groupby(["model_id", "fold_label"], sort=True):
        matrix = proba_matrix(block, labels)
        truth = block["y_true"].to_numpy()
        confidence, correct, _ = _confidence_and_correct(block, labels)
        brier = mt.brier_score(truth, matrix, labels=list(labels))
        for n_bins in bins:
            rows.append(
                {
                    "model_id": str(model_id),
                    "fold_label": str(fold_label),
                    "n_bins": int(n_bins),
                    "n_records": len(block),
                    "brier": float(brier),
                    "ece": float(
                        mt.expected_calibration_error(
                            truth, matrix, labels=list(labels), n_bins=n_bins
                        )
                    ),
                    "mean_confidence": float(confidence.mean()),
                    "argmax_accuracy": float(correct.mean()),
                    # Positive when the model is over-confident: it claims more
                    # certainty than its own hit rate justifies. Signed, because
                    # ECE throws the sign away and under-confidence on a
                    # screening task is a different problem from over-confidence.
                    "confidence_gap": float(confidence.mean() - correct.mean()),
                }
            )
    return pd.DataFrame(rows)


def summarise_calibration(per_fold: Any) -> Any:
    """mean +/- SD across folds for each ``(run, task, model, n_bins)``."""
    import pandas as pd

    keys = [k for k in ("run", "task", "model_id", "n_bins") if k in per_fold.columns]
    measures = ("brier", "ece", "mean_confidence", "argmax_accuracy", "confidence_gap")
    rows: list[dict[str, Any]] = []
    for values, block in per_fold.groupby(keys, sort=True):
        row = dict(zip(keys, values, strict=True))
        row["n_folds"] = len(block)
        row["n_records_total"] = int(block["n_records"].sum())
        for measure in measures:
            series = np.asarray(block[measure], dtype=float)
            finite = series[np.isfinite(series)]
            row[measure + "_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[measure + "_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_frame(
    frame: Any, labels: tuple[int, ...], *, n_bins: int = RELIABILITY_BINS
) -> Any:
    """Reliability points per model: mean confidence against observed accuracy.

    Computed inside each fold and averaged over the folds that had records in
    that bin, exactly as ``curves.calibration_frame`` does. An empty bin
    contributes nothing -- filling it with zero would assert that nothing in that
    confidence band was correct, which is a claim and not a measurement.
    """
    import pandas as pd

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []

    for model_id, part in frame.groupby("model_id", sort=True):
        per_bin: dict[int, list[tuple[float, float, int]]] = {b: [] for b in range(n_bins)}
        for _fold, block in part.groupby("fold_label", sort=True):
            confidence, correct, _ = _confidence_and_correct(block, labels)
            index = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
            for b in range(n_bins):
                mask = index == b
                count = int(mask.sum())
                if count == 0:
                    continue
                per_bin[b].append(
                    (float(confidence[mask].mean()), float(correct[mask].mean()), count)
                )

        for b in range(n_bins):
            observations = per_bin[b]
            if not observations:
                continue
            predicted = np.array([item[0] for item in observations])
            observed = np.array([item[1] for item in observations])
            rows.append(
                {
                    "model_id": str(model_id),
                    "kind": "confidence_vs_accuracy",
                    "n_bins": int(n_bins),
                    "bin": b,
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "confidence_mean": float(predicted.mean()),
                    "accuracy_mean": float(observed.mean()),
                    "accuracy_sd": (
                        float(observed.std(ddof=1)) if observed.size > 1 else float("nan")
                    ),
                    "n_folds": len(observations),
                    "n_records": int(sum(item[2] for item in observations)),
                }
            )
    return pd.DataFrame(rows)


def confidence_histogram(
    frame: Any, labels: tuple[int, ...], *, n_bins: int = CONFIDENCE_BINS
) -> Any:
    """G33's source: how confidence is distributed, split by right and wrong.

    Pooled over folds on purpose, unlike every metric here, and the column
    ``n_folds_pooled`` says so. A histogram is a shape rather than an estimate;
    each record contributing once per repeat inflates the counts uniformly and
    changes nothing about the shape, while per-fold histograms of a 124-record
    corpus are too sparse to read.
    """
    import pandas as pd

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []

    for model_id, part in frame.groupby("model_id", sort=True):
        confidence, correct, _ = _confidence_and_correct(part, labels)
        index = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
        n_folds = int(part["fold_label"].nunique())
        for b in range(n_bins):
            mask = index == b
            total = int(mask.sum())
            n_correct = int(correct[mask].sum()) if total else 0
            rows.append(
                {
                    "model_id": str(model_id),
                    "n_bins": int(n_bins),
                    "bin": b,
                    "bin_low": float(edges[b]),
                    "bin_high": float(edges[b + 1]),
                    "n_predictions": total,
                    "n_correct": n_correct,
                    "n_incorrect": total - n_correct,
                    "accuracy_in_bin": (float(n_correct) / total if total else float("nan")),
                    "share_of_predictions": float(total) / float(len(part)) if len(part) else 0.0,
                    "n_folds_pooled": n_folds,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T78.4 -- sigmoid against isotonic, for the SVM
# ---------------------------------------------------------------------------


def svm_calibration_comparison(
    *,
    model_id: str = "M3",
    task: str = "binary",
    methods: tuple[str, ...] = ("sigmoid", "isotonic"),
    data: Any = None,
    folds: Any = None,
) -> Any:
    """Fit M3 under each calibrator on the **same** folds and score both.

    Everything except the calibration method is held fixed: the config's M3
    hyperparameters, the Phase 44 pipeline, the loaded fold map, seed 42. The
    `calibration_cv` is left at the config's plain 3-fold rather than the
    subject-grouped split, because the question here is sigmoid versus isotonic
    and changing two things at once answers neither.

    Returns one row per (method, fold) -- the paired structure Phase 82's tests
    need. ``outputs/04_models/svm_calibration_benchmark.csv`` already compared
    the two on fold 0 alone; n=1 cannot separate a 0.002 Brier difference from
    noise, which is why this runs the whole map.
    """
    import pandas as pd

    from src.evaluation import metrics as mt
    from src.evaluation.cv import load_folds, resolve_folds
    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models.smoke import load_task_data

    resolved_data = data if data is not None else load_task_data(task)
    if folds is None:
        folds = resolve_folds(load_folds(task), resolved_data.record_uids)

    labels = sorted(np.unique(resolved_data.y).tolist())
    rows: list[dict[str, Any]] = []

    for method in methods:
        for fold in folds:
            train, test = fold.train_index, fold.test_index
            estimator = est.build_estimator(model_id, calibration_method=method)
            built = pl.build_pipeline(
                estimator, y=resolved_data.y[train], n_features=resolved_data.X.shape[1]
            )
            import time

            started = time.perf_counter()
            built.fit(resolved_data.X[train], resolved_data.y[train])
            fit_seconds = time.perf_counter() - started

            proba = built.predict_proba(resolved_data.X[test])
            predicted = built.predict(resolved_data.X[test])
            truth = resolved_data.y[test]
            row: dict[str, Any] = {
                "model_id": model_id,
                "method": method,
                "fold_label": fold.label,
                "repeat": fold.repeat,
                "fold": fold.fold,
                "n_train": int(train.size),
                "n_test": int(test.size),
                "fit_seconds": float(fit_seconds),
                "isotonic_below_min_samples": bool(
                    method == "isotonic" and train.size < _isotonic_min()
                ),
            }
            row.update(mt.binary_metrics(truth, predicted, proba, labels=labels))
            row["brier"] = mt.brier_score(truth, proba, labels=labels)
            row["ece"] = mt.expected_calibration_error(
                truth, proba, labels=labels, n_bins=RELIABILITY_BINS
            )
            rows.append(row)
            log.info(
                "%s %s %s: brier %.4f ece %.4f (%.1f s)",
                model_id,
                method,
                fold.label,
                row["brier"],
                row["ece"],
                fit_seconds,
            )

    return pd.DataFrame(rows)


def _isotonic_min() -> int:
    from src.models.calibration import ISOTONIC_MIN_SAMPLES

    return int(ISOTONIC_MIN_SAMPLES)


def final_model_calibration(*, comparison: Any = None) -> dict[str, Any]:
    """What the deployed model actually does about calibration, recorded (T78.4).

    The honest answer has two halves and neither is "the SVM's calibrator",
    which is why this is a dict and not a single string:

    * The **final binary model is M1**, a regularised logistic regression. It
      contains no SVM and no post-hoc calibrator at all -- logistic regression
      is already fitted by maximising a proper scoring rule, so its outputs are
      probabilities by construction rather than by a second stage.
    * M3 -- which *is* the SVM, and which sits inside the M6/M7 ensembles every
      comparison table reports -- is calibrated with **sigmoid** (Platt) scaling
      under `configs/models.yaml`, at `cv: 3` with `ensemble: true`.

    So "which calibration is used in the final model" is answered by naming the
    selected model, and the sigmoid-versus-isotonic comparison is evidence about
    the ensemble members rather than about the deployed estimator. Writing it
    the other way round -- quoting M3's calibrator as the final model's -- would
    describe a model the project does not ship.
    """
    import json

    from src.models.calibration import ISOTONIC_MIN_SAMPLES, calibration_settings
    from src.utils.evidence import PROJECT_ROOT

    manifest_path = PROJECT_ROOT / "models_saved" / "binary" / "final" / "manifest.json"
    if not manifest_path.is_file():
        raise CalibrationError(
            str(manifest_path) + " is missing; run scripts/13_finalize_binary_model.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = calibration_settings()

    record: dict[str, Any] = {
        "final_model_id": manifest.get("selected_model_id"),
        "final_model_estimator": manifest.get("estimator_class"),
        "final_model_has_post_hoc_calibrator": False,
        "final_model_calibration": (
            "none -- M1 is a logistic regression, fitted by maximising a proper "
            "scoring rule, so it emits probabilities without a second stage"
        ),
        "svm_member_model_id": "M3",
        "svm_calibration_method": settings["method"],
        "svm_calibration_cv": settings["cv"],
        "svm_calibration_ensemble": settings["ensemble"],
        "svm_calibration_source": "configs/models.yaml calibration block",
        "isotonic_min_samples": int(ISOTONIC_MIN_SAMPLES),
        "ensembles_carrying_the_svm": ["M6", "M7"],
        "why": (
            "The selected final model carries no SVM, so the sigmoid-versus-"
            "isotonic comparison is evidence about the ensemble members (M6/M7), "
            "not about the deployed estimator."
        ),
    }

    if comparison is not None and len(comparison):
        summary = (
            comparison.groupby("method")[["brier", "ece", "sensitivity", "balanced_accuracy"]]
            .mean()
            .round(6)
        )
        record["comparison_n_folds"] = int(comparison["fold_label"].nunique())
        record["comparison_mean"] = {
            str(method): {k: float(v) for k, v in row.items()}
            for method, row in summary.iterrows()
        }
        # Lower is better for both calibration measures. Named explicitly so a
        # reader does not have to remember which way each one points.
        record["better_brier"] = str(summary["brier"].idxmin())
        record["better_ece"] = str(summary["ece"].idxmin())
        wins = _paired_wins(comparison, "brier")
        record["brier_fold_wins"] = wins
        record["configured_method_matches_better_brier"] = bool(
            record["better_brier"] == settings["method"]
        )
    return record


def _paired_wins(comparison: Any, measure: str) -> dict[str, int]:
    """How many folds each method wins on ``measure`` (lower is better)."""
    wide = comparison.pivot_table(index="fold_label", columns="method", values=measure)
    methods = list(wide.columns)
    if len(methods) != 2:
        return {}
    first, second = methods
    return {
        str(first): int((wide[first] < wide[second]).sum()),
        str(second): int((wide[second] < wide[first]).sum()),
        "tied": int((wide[first] == wide[second]).sum()),
    }
