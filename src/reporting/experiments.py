"""Experiment results as an exportable payload (Phase 115).

T115.3 wants a sortable metric table across all models with error bars from the
fold-wise results, and T115.4 wants ROC, PR and confusion-matrix viewers driven
by a model selector. Both need the contents of `outputs/06_binary_results/` and
`outputs/07_multiclass_results/`, which the table engine has not turned into
`T##` deliverables yet -- that is Phases 87-89, in Part IX, which has not run.

So this module reads the experiment directories directly and formats everything
in Python. It computes **no metric**: every value is a cell of
`aggregate_metrics.csv` or `per_fold_metrics.csv`, or a count from
`confusion_matrices.json`. The one derived quantity anywhere near here is the
curve interpolation in `curves.py`, which says so in its own docstring.

## Rule 6 decides the column order, not convenience

`aggregate_metrics.csv` is 71 columns wide and accuracy sorts near the front
alphabetically. The reported set and its order are declared in
:data:`REPORTED_METRICS`: sensitivity, specificity, balanced accuracy, F1 and
the AUCs first, with accuracy well down the list, because final model selection
prioritises sensitivity and balanced accuracy and a table that leads with
accuracy invites the opposite reading.

## Every experiment says what it is

An experiment id on its own does not tell a reader whether they are looking at a
baseline over default hyperparameters or a nested search. `EXPERIMENTS`
therefore carries a title, a description and the `tuned` flag from the run's own
config snapshot -- so EXP-A1 cannot be read as an optimized result, and EXP-A2's
nested estimate cannot be read as the deployed model's own score.

## The confusion matrix's support is not the corpus

`confusion_matrices.json` sums element-wise over 25 folds of a repeated 5x5 map,
so each record is counted once per repeat and the total support is five times the
number of records. That note travels in the payload and is rendered, because a
reader adding up the four cells and comparing them against the dataset page
would otherwise find a factor of five with no explanation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "EXPERIMENTS",
    "REPORTED_METRICS",
    "ExperimentSpec",
    "experiment_payload",
    "experiments_payload",
]

log = get_logger("reporting.experiments")

#: (column stem, label, "higher is better"). The order is research rule 6's.
REPORTED_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("sensitivity", "Sensitivity", True),
    ("specificity", "Specificity", True),
    ("balanced_accuracy", "Balanced accuracy", True),
    ("f1", "F1", True),
    ("precision", "Precision", True),
    ("roc_auc", "ROC AUC", True),
    ("pr_auc", "PR AUC", True),
    ("mcc", "MCC", True),
    ("accuracy", "Accuracy", True),
    ("brier", "Brier score", False),
    ("ece", "Calibration error", False),
    ("macro_f1", "Macro F1", True),
    ("macro_recall", "Macro recall", True),
    ("macro_precision", "Macro precision", True),
    ("weighted_f1", "Weighted F1", True),
    ("ovr_auc_macro", "One-vs-rest AUC (macro)", True),
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment directory, and what a reader has to know to read it."""

    exp_id: str
    task: str
    directory: str
    title: str
    description: str
    #: What this run is NOT evidence for. Rendered beside the table.
    caveat: str | None = None


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        exp_id="EXP-A1",
        task="binary",
        directory="outputs/06_binary_results/EXP-A1",
        title="PhysioNet binary baseline",
        description=(
            "Every model at its configured defaults over the 25-fold repeated "
            "subject-grouped map. No search runs inside it."
        ),
        caveat=(
            "A baseline, not an optimized result. Comparing it against a tuned run "
            "measures what the search bought, and nothing here should be quoted as "
            "the framework's performance."
        ),
    ),
    ExperimentSpec(
        exp_id="EXP-A2",
        task="binary",
        directory="outputs/06_binary_results/EXP-A2",
        title="PhysioNet binary, nested search",
        description=(
            "Hyperparameters chosen inside each training fold by nested "
            "cross-validation. The outer fold is never seen by the search."
        ),
        caveat=(
            "This is the honest performance estimate. The deployed model is refitted "
            "on every labelled record afterwards; its estimate is this run, not "
            "anything measured on the rows it was fitted to."
        ),
    ),
    ExperimentSpec(
        exp_id="EXP-A2-so04_subset",
        task="binary",
        directory="outputs/06_binary_results/EXP-A2-so04_subset",
        title="PhysioNet binary, nested search on the selected subset",
        description=(
            "The same nested protocol restricted to the feature subset SO-04 "
            "selected, so the cost of dropping features is measurable."
        ),
    ),
    ExperimentSpec(
        exp_id="EXP-B1",
        task="pascal_a",
        directory="outputs/07_multiclass_results/EXP-B1",
        title="PASCAL A, four classes",
        description="Four-class classification over PASCAL set A.",
        caveat=(
            "124 records with 19 in one class. PASCAL's 'artifact' label is a "
            "recording-quality category, not a cardiac class, so this is never a "
            "four-class cardiac classifier."
        ),
    ),
    ExperimentSpec(
        exp_id="EXP-B2",
        task="pascal_b",
        directory="outputs/07_multiclass_results/EXP-B2",
        title="PASCAL B, three classes",
        description="Three-class classification over PASCAL set B.",
        caveat=(
            "PASCAL B and PASCAL A are separate label spaces and are never merged. "
            "461 labelled records."
        ),
    ),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "run_manifest.json"
    if not path.is_file():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}


def _tuned(directory: Path) -> bool | None:
    """Whether the run searched, read from its own config snapshot."""
    path = directory / "config_snapshot.yaml"
    if not path.is_file():
        return None
    import yaml

    try:
        snapshot = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    for block in (snapshot, snapshot.get("experiment", {})):
        if isinstance(block, dict) and "tuned" in block:
            return bool(block["tuned"])
    return None


def _metric_rows(aggregate: Any, class_names: list[str]) -> list[dict[str, Any]]:
    """One row per model: display strings for text, numbers for chart geometry."""
    from src.reporting.tables import format_value

    rows: list[dict[str, Any]] = []
    for position in range(len(aggregate)):
        record = aggregate.iloc[position]
        metrics: dict[str, Any] = {}
        for column, _label, _higher in REPORTED_METRICS:
            mean_column = column + "_mean"
            if mean_column not in aggregate.columns:
                continue
            mean = record[mean_column]
            sd = record.get(column + "_sd")
            metrics[column] = {
                "mean": _finite(mean),
                "sd": _finite(sd),
                "mean_display": format_value(mean, "metric"),
                "sd_display": format_value(sd, "metric"),
                "display": format_value(mean, "metric")
                + (" +/- " + format_value(sd, "metric") if sd is not None else ""),
            }
        rows.append(
            {
                "model_id": str(record["model_id"]),
                "n_folds": int(record["n_folds"]) if "n_folds" in aggregate.columns else None,
                "n_folds_display": format_value(record.get("n_folds"), "count"),
                "metrics": metrics,
                "per_class": _per_class(aggregate, record, class_names),
            }
        )
    return rows


def _per_class(aggregate: Any, record: Any, class_names: list[str]) -> list[dict[str, Any]]:
    """Recall and F1 per class, which research rule 6 requires for multiclass.

    A macro average over four classes says nothing about the class with 19
    samples in it, and that class is the whole reason the multiclass track is
    caveated. Support travels with each row so a reader can see how thin the
    evidence for any one class is.
    """
    from src.reporting.tables import format_value

    rows: list[dict[str, Any]] = []
    for name in class_names:
        entry: dict[str, Any] = {"class": name}
        found = False
        for stem, label in (("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")):
            column = stem + "_" + name + "_mean"
            if column not in aggregate.columns:
                continue
            found = True
            sd = record.get(stem + "_" + name + "_sd")
            entry[stem] = {
                "label": label,
                "mean": _finite(record[column]),
                "sd": _finite(sd),
                "display": format_value(record[column], "metric")
                + (" +/- " + format_value(sd, "metric") if sd is not None else ""),
            }
        support_column = "support_" + name + "_mean"
        if support_column in aggregate.columns:
            entry["support"] = _finite(record[support_column])
            entry["support_display"] = format_value(record[support_column], "metric")
        if found:
            rows.append(entry)
    return rows


def _finite(value: Any) -> float | None:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fold_rows(per_fold: Any) -> list[dict[str, Any]]:
    """Per-fold values per model, for the error bars T115.3 asks for."""
    rows: list[dict[str, Any]] = []
    columns = [name for name, _label, _higher in REPORTED_METRICS if name in per_fold.columns]
    for model_id, part in per_fold.groupby("model_id", sort=True):
        ordered = part.sort_values("fold_label")
        rows.append(
            {
                "model_id": str(model_id),
                "fold_labels": [str(v) for v in ordered["fold_label"]],
                "metrics": {name: [_finite(v) for v in ordered[name]] for name in columns},
            }
        )
    return rows


def _confusion(directory: Path) -> dict[str, Any]:
    path = directory / "confusion_matrices.json"
    if not path.is_file():
        return {"available": False, "reason": "no confusion_matrices.json in this run"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "reason": str(error)}

    models = payload.get("models") or {}
    return {
        "available": bool(models),
        "reason": None if models else "the file records no models",
        "class_names": list(payload.get("class_names") or []),
        "labels": list(payload.get("labels") or []),
        "note": str(payload.get("note") or ""),
        "models": {
            str(model_id): {"total": block.get("total")} for model_id, block in models.items()
        },
    }


def experiment_payload(spec: ExperimentSpec) -> dict[str, Any]:
    """One experiment, or a stated absence."""
    import pandas as pd

    from src.reporting.curves import curves_payload

    directory = _project_root() / spec.directory
    aggregate_path = directory / "aggregate_metrics.csv"
    if not aggregate_path.is_file():
        return {
            "exp_id": spec.exp_id,
            "task": spec.task,
            "title": spec.title,
            "available": False,
            "reason": (
                spec.directory
                + "/aggregate_metrics.csv is not present; this experiment has not run"
            ),
        }

    aggregate = pd.read_csv(aggregate_path)
    per_fold_path = directory / "per_fold_metrics.csv"
    per_fold = pd.read_csv(per_fold_path) if per_fold_path.is_file() else pd.DataFrame()
    manifest = _read_manifest(directory)
    git = manifest.get("git") or {}

    curves: dict[str, Any] = {
        "available": False,
        "reason": (
            "ROC and PR curves here are defined for the binary task. A multiclass "
            "run needs a one-vs-rest treatment, which is G17's job in Phase 93."
        ),
    }
    if spec.task == "binary":
        curves = curves_payload(directory)

    confusion = _confusion(directory)
    class_names = [str(name) for name in confusion.get("class_names") or []]

    return {
        "exp_id": spec.exp_id,
        "task": spec.task,
        "title": spec.title,
        "description": spec.description,
        "caveat": spec.caveat,
        "available": True,
        "reason": None,
        "directory": spec.directory,
        "tuned": _tuned(directory),
        "cv": str(aggregate["cv"].iloc[0]) if "cv" in aggregate.columns else None,
        "n_models": len(aggregate),
        "run_id": manifest.get("run_id"),
        "git_commit": git.get("commit"),
        "seed": manifest.get("seed"),
        "metrics": [
            {"name": name, "label": label, "higher_is_better": higher}
            for name, label, higher in REPORTED_METRICS
            if name + "_mean" in aggregate.columns
        ],
        "models": _metric_rows(aggregate, class_names),
        "folds": _fold_rows(per_fold) if not per_fold.empty else [],
        "confusion": confusion,
        "curves": curves,
    }


def experiments_payload() -> dict[str, Any]:
    """Every declared experiment, present or not.

    An experiment that has not run is reported with the reason rather than
    dropped: a page listing four results where five were declared says nothing
    about the fifth, and a reader counts what they see.
    """
    payloads = [experiment_payload(spec) for spec in EXPERIMENTS]
    return {
        "n_declared": len(EXPERIMENTS),
        "n_available": sum(1 for item in payloads if item["available"]),
        "selection_note": (
            "Final model selection prioritises sensitivity and balanced accuracy, "
            "not accuracy. The metric order in every table here follows that, and "
            "accuracy is reported because it must be, never as the headline."
        ),
        "label_space_note": (
            "Binary, PASCAL A, PASCAL B, CirCor murmur and CirCor outcome are five "
            "separate tasks with five separate targets. They are never merged, and "
            "no number is comparable across them."
        ),
        "experiments": payloads,
    }
