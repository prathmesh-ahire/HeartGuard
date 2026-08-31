"""ROC, precision-recall and calibration curves, computed once in Python (Phase 115).

T115.4 wants ROC, PR and confusion-matrix viewers driven by a model selector.
The confusion matrices are already committed in each experiment's
`confusion_matrices.json`. The curves are not, and they cannot be: they come
from `predictions.parquet`, which is gitignored.

## Why the curve points are written to a committed CSV

Phase 113 shipped an overlay that could only be *rebuilt* on a machine holding
the corpus, so a fresh clone exported "unavailable" while the data sat committed
beside it. The same trap applies to anything derived from a parquet. So this
module writes `roc_pr_curve_points.csv` and `calibration_points.csv` into the
experiment's own directory, those are committed, and the exporter reads them.
The parquet always wins where it exists, so the committed copy cannot drift.

## How the curve is aggregated, and why not by pooling

The fold map is repeated 5x5 grouped CV: **every record appears once per
repeat**, so pooling all 25 folds into one curve would count each record five
times. The pooled curve would look smooth, would be reported as "the" ROC, and
its support would be five times the corpus.

Instead a curve is computed **inside each fold**, on that fold's own held-out
records, and the 25 curves are interpolated onto a fixed grid and averaged. That
is the standard cross-validated curve and it carries a spread, which a pooled
curve destroys. `sd` travels with `mean` for exactly that reason: a band that is
wide somewhere is a fact about the model, not noise to be smoothed away.

Two consequences worth stating on the page, and both are in the payload:

* The mean curve's own area is **not** the reported AUC. The reported AUC is the
  mean of the 25 per-fold AUCs, which is what `aggregate_metrics.csv` holds and
  what every table shows. Averaging curves and averaging areas are different
  operations and they do not commute.
* PR is interpolated on recall, not on precision. Precision is not monotone in
  the threshold, so interpolating it the other way is meaningless.

## Calibration

Ten equal-width probability bins, per fold, then averaged across folds. A bin
with no predictions in a fold contributes nothing rather than a zero -- an empty
bin is an absence of evidence and a zero is a claim that nothing in that band was
positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "CALIBRATION_BINS",
    "CALIBRATION_CSV",
    "CURVE_CSV",
    "GRID_POINTS",
    "CurveSet",
    "calibration_frame",
    "curve_frame",
    "curves_payload",
    "export_curves",
]

log = get_logger("reporting.curves")

CURVE_CSV = "roc_pr_curve_points.csv"
CALIBRATION_CSV = "calibration_points.csv"

#: Points on the common grid each fold's curve is interpolated onto. 101 gives a
#: step of 0.01, which is finer than the chart can resolve and small enough that
#: six models across three experiments stay a few tens of kilobytes.
GRID_POINTS = 101

CALIBRATION_BINS = 10


@dataclass(frozen=True)
class CurveSet:
    """One model's mean curve and its spread across folds."""

    model_id: str
    kind: str
    x: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    n_folds: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _grid() -> np.ndarray:
    return np.linspace(0.0, 1.0, GRID_POINTS)


def curve_frame(predictions: Any) -> Any:
    """Per-model mean ROC and PR across folds, as a tidy frame.

    `predictions` is the experiment's `predictions.parquet` frame: one row per
    (model, fold, record) with `y_true` and `proba_1`.
    """
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, roc_curve

    grid = _grid()
    rows: list[dict[str, Any]] = []

    for model_id, part in predictions.groupby("model_id", sort=True):
        roc: list[np.ndarray] = []
        pr: list[np.ndarray] = []
        for _fold, fold_rows in part.groupby("fold_label", sort=True):
            truth = fold_rows["y_true"].to_numpy()
            score = fold_rows["proba_1"].to_numpy()
            if len(np.unique(truth)) < 2:
                # A fold with one class has no ROC. Skipped rather than filled
                # with zeros, and the surviving fold count is reported.
                continue

            fpr, tpr, _ = roc_curve(truth, score)
            roc.append(np.interp(grid, fpr, tpr))

            precision, recall, _ = precision_recall_curve(truth, score)
            # `precision_recall_curve` returns recall descending; np.interp
            # needs it ascending.
            order = np.argsort(recall)
            pr.append(np.interp(grid, recall[order], precision[order]))

        for kind, stack in (("roc", roc), ("pr", pr)):
            if not stack:
                continue
            block = np.vstack(stack)
            mean = block.mean(axis=0)
            sd = block.std(axis=0, ddof=1) if block.shape[0] > 1 else np.zeros_like(mean)
            for index, position in enumerate(grid):
                rows.append(
                    {
                        "model_id": str(model_id),
                        "kind": kind,
                        "x": round(float(position), 4),
                        "mean": round(float(mean[index]), 6),
                        "sd": round(float(sd[index]), 6),
                        "n_folds": int(block.shape[0]),
                    }
                )

    return pd.DataFrame(rows)


def calibration_frame(predictions: Any) -> Any:
    """Per-model reliability points: mean predicted against observed, per bin."""
    import pandas as pd

    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    rows: list[dict[str, Any]] = []

    for model_id, part in predictions.groupby("model_id", sort=True):
        per_bin: dict[int, list[tuple[float, float, int]]] = {
            b: [] for b in range(CALIBRATION_BINS)
        }
        for _fold, fold_rows in part.groupby("fold_label", sort=True):
            truth = fold_rows["y_true"].to_numpy(dtype=float)
            score = fold_rows["proba_1"].to_numpy(dtype=float)
            index = np.clip(np.digitize(score, edges[1:-1]), 0, CALIBRATION_BINS - 1)
            for b in range(CALIBRATION_BINS):
                mask = index == b
                count = int(mask.sum())
                if count == 0:
                    # An empty bin contributes nothing. Filling it with 0.0
                    # would assert that nothing in that probability band was
                    # positive, which is a claim and not a measurement.
                    continue
                per_bin[b].append((float(score[mask].mean()), float(truth[mask].mean()), count))

        for b in range(CALIBRATION_BINS):
            observations = per_bin[b]
            if not observations:
                continue
            predicted = np.array([item[0] for item in observations])
            observed = np.array([item[1] for item in observations])
            rows.append(
                {
                    "model_id": str(model_id),
                    "bin": b,
                    "bin_low": round(float(edges[b]), 3),
                    "bin_high": round(float(edges[b + 1]), 3),
                    "predicted_mean": round(float(predicted.mean()), 6),
                    "observed_mean": round(float(observed.mean()), 6),
                    "observed_sd": round(
                        float(observed.std(ddof=1)) if observed.size > 1 else 0.0, 6
                    ),
                    "n_folds": len(observations),
                    "n_records": int(sum(item[2] for item in observations)),
                }
            )

    return pd.DataFrame(rows)


def export_curves(exp_dir: str | Path) -> dict[str, Path]:
    """Recompute both frames from `predictions.parquet` and write the CSVs."""
    import pandas as pd

    directory = Path(exp_dir)
    parquet = directory / "predictions.parquet"
    if not parquet.is_file():
        raise FileNotFoundError("no predictions.parquet in " + str(directory))

    predictions = pd.read_parquet(parquet)
    if "proba_1" not in predictions.columns:
        raise KeyError(
            str(parquet) + " has no proba_1 column; curves are defined for the "
            "binary task only, and a multiclass experiment needs its own "
            "one-vs-rest treatment"
        )

    written: dict[str, Path] = {}
    for name, frame in (
        (CURVE_CSV, curve_frame(predictions)),
        (CALIBRATION_CSV, calibration_frame(predictions)),
    ):
        target = directory / name
        frame.to_csv(target, index=False, lineterminator="\n")
        written[name] = target
        log.info("wrote %d rows -> %s", len(frame), target)
    return written


def curves_payload(exp_dir: str | Path) -> dict[str, Any]:
    """Curves for one experiment, from the parquet where it exists, else the CSV."""
    import pandas as pd

    directory = Path(exp_dir)
    source = "predictions.parquet"
    if (directory / "predictions.parquet").is_file():
        try:
            export_curves(directory)
        except (KeyError, OSError, ValueError) as error:
            log.info("cannot rebuild curves for %s: %s", directory.name, error)
            source = "committed csv"
    else:
        source = "committed csv"

    curve_path = directory / CURVE_CSV
    calibration_path = directory / CALIBRATION_CSV
    if not curve_path.is_file():
        return {
            "available": False,
            "reason": (
                "neither predictions.parquet nor " + CURVE_CSV + " is present for " + directory.name
            ),
            "models": [],
        }

    curves = pd.read_csv(curve_path)
    calibration = pd.read_csv(calibration_path) if calibration_path.is_file() else pd.DataFrame()

    models: list[dict[str, Any]] = []
    for model_id in sorted(curves["model_id"].astype(str).unique()):
        part = curves[curves["model_id"].astype(str) == model_id]
        entry: dict[str, Any] = {"model_id": model_id}
        for kind in ("roc", "pr"):
            block = part[part["kind"] == kind].sort_values("x")
            if block.empty:
                continue
            entry[kind] = {
                "x": [float(v) for v in block["x"]],
                # The axis tick labels, formatted HERE. A chart that called
                # toFixed on the grid would be formatting a number in the
                # browser, which is the rule the whole codegen boundary exists
                # to hold -- and Phase 113's chart gate fails the build on it.
                "x_display": [format(float(v), ".2f") for v in block["x"]],
                "mean": [float(v) for v in block["mean"]],
                "sd": [float(v) for v in block["sd"]],
                "n_folds": int(block["n_folds"].iloc[0]),
            }
        if not calibration.empty:
            bins = calibration[calibration["model_id"].astype(str) == model_id].sort_values("bin")
            if not bins.empty:
                entry["calibration"] = {
                    "predicted": [float(v) for v in bins["predicted_mean"]],
                    "observed": [float(v) for v in bins["observed_mean"]],
                    "observed_sd": [float(v) for v in bins["observed_sd"]],
                    "n_records": [int(v) for v in bins["n_records"]],
                }
        models.append(entry)

    return {
        "available": bool(models),
        "reason": None if models else "the curve CSV holds no rows",
        "source": source,
        "grid_points": GRID_POINTS,
        "calibration_bins": CALIBRATION_BINS,
        "aggregation_note": (
            "Each curve is computed inside a fold, on that fold's own held-out "
            "records, and the folds are interpolated onto a common grid and "
            "averaged; the shaded band is one standard deviation across folds. "
            "The folds are NOT pooled: the map is repeated 5x5 CV, so every "
            "record appears once per repeat and pooling would count each record "
            "five times. One consequence: the area under the mean curve is not "
            "the reported AUC. The reported AUC is the mean of the per-fold "
            "areas, which is what the metric tables show."
        ),
        "models": models,
    }
