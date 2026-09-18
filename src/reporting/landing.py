"""The `/` landing sections, above the Analyse tool (Phase 140, T140.1).

**No metric is computed here.** Every number is a selection from a payload
another module already built (`experiments_payload`, `features_payload`,
`dataset_summary_payload`, `objectives_payload`) or a cell already sitting in
one of the tables `frontend_export.export_all` discovers under `outputs/`
(T16, T20, T23, T25, T28) -- the same tables the About pages render. Picking
"the deployed model's row" or "the row where group == 'clean'" is a
selection, not a computation; the number itself is never touched.

## Which model is "the" model

Every table here holds several models. The row this module shows is always
the one `models_saved/<task>/final/manifest.json` actually deploys
(`selected_model_id`) -- binary is M1, PASCAL A is M4, PASCAL B is M1. A
headline that quietly picked whichever model scored highest would not be the
model a visitor's own recording is scored by.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.reporting.experiments import experiments_payload
from src.reporting.method import features_payload
from src.reporting.objectives import objectives_payload
from src.reporting.record_index import MASTER_CSV, dataset_summary_payload
from src.reporting.tables import format_value

__all__ = ["landing_payload"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deployed_model_id(task: str) -> str | None:
    """The model `models_saved/<task>/final/manifest.json` actually deploys."""
    path = _project_root() / "models_saved" / task / "final" / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model_id = manifest.get("selected_model_id")
    return str(model_id) if model_id else None


def _table_rows(table: dict[str, Any] | None) -> list[dict[str, Any]]:
    """A `_export_table` payload, as row dicts: `{column: display}` plus `_values`."""
    if table is None:
        return []
    columns = table["columns"]
    rows: list[dict[str, Any]] = []
    for i in range(table["n_rows"]):
        row: dict[str, Any] = {c["name"]: c["display"][i] for c in columns}
        row["_values"] = {c["name"]: (c["values"][i] if c["values"] else None) for c in columns}
        rows.append(row)
    return rows


def _find_row(rows: list[dict[str, Any]], **filters: str) -> dict[str, Any] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            return row
    return None


#: (metric key, label). Binary headline order follows research rule 6.
_BINARY_METRICS: tuple[tuple[str, str], ...] = (
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("roc_auc", "ROC AUC"),
)
#: Multiclass has no ROC AUC in the binary sense; `ovr_auc_macro` is its analogue.
_MULTICLASS_METRICS: tuple[tuple[str, str], ...] = (
    ("macro_f1", "Macro F1"),
    ("macro_recall", "Macro recall"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("ovr_auc_macro", "AUC (one-vs-rest)"),
)
#: The six tiles of the metrics strip (T140.3), all off the champion binary model.
_STRIP_METRICS: tuple[tuple[str, str], ...] = (
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("f1", "F1"),
    ("roc_auc", "ROC AUC"),
)


#: The raw `cv` config codes each experiment's `aggregate_metrics.csv` carries,
#: relabelled for a reader rather than shown as a config key. A code not in
#: here is shown verbatim rather than guessed at.
_CV_LABELS: dict[str, str] = {
    "repeated_5x5_grouped": "25-fold repeated subject-grouped (5x5)",
    "repeated_5x2_stratified": "10-fold repeated stratified (5x2)",
    "grouped_5fold": "5-fold subject-grouped",
}


def _cv_label(code: str) -> str:
    return _CV_LABELS.get(code, code)


def _model_row(models: list[dict[str, Any]], model_id: str | None) -> dict[str, Any] | None:
    if not models:
        return None
    if model_id is not None:
        found = next((m for m in models if m["model_id"] == model_id), None)
        if found is not None:
            return found
    return models[0]


def _metric_entries(
    row: dict[str, Any] | None, wanted: tuple[tuple[str, str], ...]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if row is None:
        return entries
    for key, label in wanted:
        metric = row["metrics"].get(key)
        if metric is None:
            continue
        entries.append(
            {
                "key": key,
                "label": label,
                "value": metric["mean"],
                "display": metric["mean_display"],
            }
        )
    return entries


def _experiment_card(experiment: dict[str, Any], deployed: dict[str, str | None]) -> dict[str, Any]:
    exp_id = experiment["exp_id"]
    if not experiment["available"]:
        return {
            "exp_id": exp_id,
            "task": experiment["task"],
            "title": experiment["title"],
            "description": experiment.get("description"),
            "available": False,
            "reason": experiment["reason"],
            "model_id": None,
            "metrics": [],
            "finding": None,
            "link": "/about/models/",
        }

    wanted = _BINARY_METRICS if experiment["task"] == "binary" else _MULTICLASS_METRICS
    model_id = deployed.get(experiment["task"])
    row = _model_row(experiment["models"], model_id)
    metrics = _metric_entries(row, wanted)
    model_used = row["model_id"] if row is not None else None

    finding = None
    if row is not None and metrics:
        parts = [entry["label"].lower() + " " + entry["display"] for entry in metrics[:2]]
        cv_code = experiment.get("cv")
        cv_text = _cv_label(str(cv_code)) if cv_code else str(row.get("n_folds_display") or "") + " folds"
        finding = "Model " + model_used + ": " + " and ".join(parts) + " over " + cv_text + "."

    return {
        "exp_id": exp_id,
        "task": experiment["task"],
        "title": experiment["title"],
        "description": experiment.get("description"),
        "available": True,
        "reason": None,
        "model_id": model_used,
        "metrics": metrics,
        "finding": finding,
        "link": "/about/models/",
    }


def _dataset_cards() -> list[dict[str, Any]]:
    import pandas as pd

    path = _project_root() / MASTER_CSV
    frame = pd.read_csv(path)
    summary_by_source = {row["dataset_source"]: row for row in dataset_summary_payload()["summary"]}

    label_columns = (
        "binary_label_name",
        "multiclass_label_name",
        "murmur_label_name",
        "outcome_label_name",
    )
    cards: list[dict[str, Any]] = []
    for source, part in frame.groupby("dataset_source", sort=True):
        summary = summary_by_source.get(str(source))
        if summary is None:
            continue
        classes: list[str] = []
        for column in label_columns:
            if column not in part.columns:
                continue
            values = sorted({str(v) for v in part[column].dropna().tolist()})
            if values:
                classes = values
                break
        rates = sorted({int(v) for v in part["original_fs"].dropna().tolist()})
        cards.append(
            {
                "dataset_source": str(source),
                "dataset_name": summary["dataset_name"],
                "n_files_display": summary["n_files_display"],
                "n_modelled_display": summary["n_modelled_display"],
                "n_subjects_display": summary["n_subjects_display"],
                "hours_modelled_display": summary["hours_modelled_display"],
                "classes": classes,
                "sample_rate_display": " / ".join(format_value(rate, "count") for rate in rates)
                + " Hz",
            }
        )
    return cards


def landing_payload(tables: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Everything `/`'s landing sections show, above the existing Analyse tool."""
    experiments = experiments_payload()
    features = features_payload()
    objectives = objectives_payload()
    dataset_summary = dataset_summary_payload()

    deployed = {
        "binary": _deployed_model_id("binary"),
        "pascal_a": _deployed_model_id("pascal_a"),
        "pascal_b": _deployed_model_id("pascal_b"),
        "murmur": _deployed_model_id("murmur"),
        "outcome": _deployed_model_id("outcome"),
    }

    exp_a2 = next((e for e in experiments["experiments"] if e["exp_id"] == "EXP-A2"), None)
    champion = _model_row(exp_a2["models"], deployed["binary"]) if exp_a2 is not None else None

    n_features_display = format_value(features["n_features"], "count")

    hero_headline = _metric_entries(champion, (("sensitivity", "Sensitivity"), ("balanced_accuracy", "Balanced accuracy")))
    if champion is not None:
        roc = champion["metrics"].get("roc_auc")
        if roc is not None:
            hero_headline.append({"key": "roc_auc", "label": "ROC AUC", "value": roc["mean"], "display": roc["mean_display"]})
    hero_headline.append(
        {
            "key": "records_supervised",
            "label": "Recordings analysed",
            "value": dataset_summary["n_supervised"],
            "display": format_value(dataset_summary["n_supervised"], "count"),
        }
    )

    metrics_strip = _metric_entries(champion, _STRIP_METRICS)
    metrics_strip.append(
        {
            "key": "n_features",
            "label": "Engineered features",
            "value": features["n_features"],
            "display": n_features_display,
        }
    )

    trust_line: list[dict[str, str]] = []
    if exp_a2 is not None:
        if exp_a2.get("cv"):
            trust_line.append({"label": "Cross-validation", "detail": _cv_label(str(exp_a2["cv"]))})
        if exp_a2.get("seed") is not None:
            trust_line.append({"label": "Seed", "detail": "Fixed at " + str(exp_a2["seed"]) + ", every run"})
    trust_line.append(
        {
            "label": "Fold safety",
            "detail": "Scaler, imputer and feature selector are fit inside the training fold only.",
        }
    )
    trust_line.append({"label": "Label spaces", "detail": experiments["label_space_note"]})

    circor_rows = _table_rows(tables.get("T16"))
    circor_row = _find_row(
        circor_rows, metric="balanced_accuracy", level="recording", rule="none"
    )
    if circor_row is not None:
        trust_line.append(
            {
                "label": "CirCor external transfer",
                "detail": (
                    "Balanced accuracy "
                    + circor_row["in_domain_mean"]
                    + " in-domain to "
                    + circor_row["external_value"]
                    + " external ("
                    + circor_row["relative_drop"]
                    + " relative drop) -- an adult-to-paediatric population "
                    "effect (EXP-D1), not a method failure."
                ),
            }
        )

    significance_rows = _table_rows(tables.get("T28"))
    significance_row = _find_row(
        significance_rows,
        run="EXP-A2",
        metric="sensitivity",
        model_a="M1",
        model_b="M6",
    )
    if significance_row is not None:
        trust_line.append(
            {
                "label": "Statistical significance",
                "detail": significance_row["interpretation"],
            }
        )

    experiment_cards = [_experiment_card(item, deployed) for item in experiments["experiments"]]

    calibration_rows = _table_rows(tables.get("T23"))
    calibration_row = _find_row(calibration_rows, run="EXP-A2", task="binary", model_id="M1")

    robustness_rows = _table_rows(tables.get("T20"))
    robustness_clean = _find_row(robustness_rows, run="EXP-A2", model_id="M1", group="clean")
    robustness_noisy = _find_row(robustness_rows, run="EXP-A2", model_id="M1", group="noisy")

    inference_rows = _table_rows(tables.get("T25"))
    inference_row = _find_row(inference_rows, run="inference bench", model_id="total")

    evidence_highlights: list[dict[str, Any]] = []
    if calibration_row is not None:
        evidence_highlights.append(
            {
                "label": "Calibration (Brier / ECE)",
                "display": calibration_row["brier_mean"] + " / " + calibration_row["ece_mean"],
                "source": "T23",
            }
        )
    if robustness_clean is not None and robustness_noisy is not None:
        evidence_highlights.append(
            {
                "label": "Noise robustness (SNR-proxy grouping)",
                "display": (
                    "clean " + robustness_clean["sensitivity_mean"] + " vs noisy "
                    + robustness_noisy["sensitivity_mean"] + " sensitivity"
                ),
                "source": "T20",
            }
        )
    if inference_row is not None:
        evidence_highlights.append(
            {
                "label": "Inference time (bench, per recording)",
                "display": inference_row["mean_seconds"] + " s",
                "source": "T25",
            }
        )

    version = "0.1.0"
    package_json = _project_root() / "frontend" / "package.json"
    if package_json.is_file():
        try:
            version = str(json.loads(package_json.read_text(encoding="utf-8")).get("version", version))
        except (OSError, json.JSONDecodeError):
            pass

    results_summary = (
        "PV-MEPCG screens "
        + format_value(dataset_summary["n_supervised"], "count")
        + " labelled recordings across "
        + str(len(dataset_summary["summary"]))
        + " public corpora, over "
        + str(objectives["n_objectives"])
        + " locked objectives and "
        + str(experiments["n_available"])
        + " of "
        + str(experiments["n_declared"])
        + " declared experiments. Final model selection prioritises sensitivity "
        "and balanced accuracy, never accuracy alone."
    )

    return {
        "champion_model_id": deployed["binary"],
        "hero": {
            "headline": hero_headline,
            "exp_id": "EXP-A2",
        },
        "metrics_strip": metrics_strip,
        "trust_line": trust_line,
        "evidence_highlights": evidence_highlights,
        "experiments": experiment_cards,
        "datasets": _dataset_cards(),
        "footer": {
            "brand": "PV-MEPCG / PulseVision",
            "tagline": "Phonocardiogram heart-sound analysis -- academic research prototype.",
            "results_summary": results_summary,
            "version": version,
            "run_id": manifest.get("run_id"),
            "git_commit": manifest.get("git_commit"),
        },
        "disclaimer": (
            "Academic screening and decision-support prototype. Not a diagnostic "
            "device and not a substitute for clinical assessment."
        ),
    }
