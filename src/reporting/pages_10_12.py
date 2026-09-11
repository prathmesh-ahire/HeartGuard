"""Payloads for dashboard pages 10-12 and the limitations page (Phase 117).

Robustness Analytics renders the committed T16/T20/T21/T22/T-S5 tables through
the ordinary table export, so it needs nothing here. The three pages that do
are built from files that are not tables:

``explainability_payload``
    T117.2. Global permutation importance, the family-level share and the stored
    per-sample decomposition, read from ``outputs/04_models/explainability/``.
``limitations_payload``
    T117.5. The three caveats the task names -- EXP-D1's population mismatch,
    PASCAL's sample sizes, CirCor's public subset -- plus EXP-F3's source
    hold-out, which `note.md`'s Known Limitation 1 makes mandatory on any page
    that frames a result. Every number is a cell read from a committed file.
``reports_payload``
    T117.3. Which reports the running API can produce, and from which files.

Every number leaves this module as a ``*_display`` string formatted by
``tables.format_value`` -- the one rounding authority -- beside the raw value
for chart geometry. The browser never formats either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.reporting.tables import NA_TEXT, content_digest, format_value

__all__ = [
    "EXPLAINABILITY_DIR",
    "OBJECTIVE_REPORT",
    "REPORT_PARENTS",
    "TOP_FEATURES",
    "explainability_payload",
    "limitations_payload",
    "objective_report_path",
    "payload_evidence",
    "reportable_experiments",
    "reports_payload",
]

EXPLAINABILITY_DIR = "outputs/04_models/explainability"

#: How many features the global ranking shows per model. The same 20 as G19.
TOP_FEATURES = 20

#: `outputs/` keys whose immediate subdirectories are experiment runs. A run is
#: reportable when it holds `aggregate_metrics.csv`, which is the file
#: `render_experiment_report` reads.
REPORT_PARENTS: tuple[str, ...] = (
    "binary_results",
    "multiclass_results",
    "circor_external_validation",
    "ablation",
)

#: T29 is the objective-to-evidence map; its DOCX is the objective-coverage report.
OBJECTIVE_REPORT = "outputs/00_evidence_index/T29_objective_to_evidence_mapping.docx"

_POPULATION = "outputs/08_circor_external_validation/EXP-D1/population_mismatch.json"
_T02 = "outputs/01_dataset_audit/T02_class_distribution_and_imbalance_ratio.csv"
_T16 = "outputs/08_circor_external_validation/T16_cross_dataset_generalization.csv"
_TS5 = "outputs/09_ablation/T-S5_leave_one_source_out_generalization.csv"


def _root() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("project_root"))


def _outputs(key: str) -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs." + key))


def _csv(relative: str) -> Any | None:
    import pandas as pd

    path = _root() / relative
    return pd.read_csv(path) if path.is_file() else None


def _json(relative: str) -> dict[str, Any] | None:
    path = _root() / relative
    if not path.is_file():
        return None
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _number(value: Any) -> float | None:
    """A strict-JSON number: NaN and Inf become ``None``."""
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent(share: Any) -> str:
    """A 0-1 share as ``"20.5%"``, rounded once by the shared formatter."""
    number = _number(share)
    if number is None:
        return NA_TEXT
    return format_value(number * 100.0, "percent") + "%"


def _source(relative: str) -> dict[str, Any]:
    path = _root() / relative
    if not path.is_file():
        return {"path": relative, "sha256": None, "available": False}
    return {"path": relative, "sha256": content_digest(path)[0], "available": True}


def _family(name: str) -> str:
    from src.feature_extraction.registry import family_of

    try:
        return family_of(name)
    except (KeyError, ValueError):
        return "unknown"


def _absent(reason: str, sources: list[str]) -> dict[str, Any]:
    return {"available": False, "reason": reason, "sources": [_source(s) for s in sources]}


# ---------------------------------------------------------------------------
# T117.2 -- explainability
# ---------------------------------------------------------------------------


def _importance_groups(frame: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for (task, model_id, kind), rows in frame.groupby(["task", "model_id", "kind"], sort=False):
        top = rows.sort_values("rank").head(TOP_FEATURES)
        n_folds = int(top["n_folds"].max()) if len(top) else 0
        groups.append(
            {
                "task": str(task),
                "model_id": str(model_id),
                "kind": str(kind),
                "n_features_ranked": len(rows),
                "n_features_ranked_display": format_value(len(rows), "count"),
                "n_folds": n_folds,
                "rows": [
                    {
                        "rank": int(row["rank"]),
                        "feature": str(row["feature"]),
                        "family": _family(str(row["feature"])),
                        "importance": _number(row["importance_mean"]),
                        "importance_display": format_value(row["importance_mean"], "metric"),
                        "importance_sd_display": format_value(row["importance_sd"], "metric"),
                        "folds_positive_display": (
                            format_value(row["n_folds_positive"], "count")
                            + " of "
                            + format_value(row["n_folds"], "count")
                        ),
                    }
                    for _, row in top.iterrows()
                ],
            }
        )
    return groups


def _family_groups(frame: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for (task, model_id, kind), rows in frame.groupby(["task", "model_id", "kind"], sort=False):
        ordered = rows.sort_values("rank")
        groups.append(
            {
                "task": str(task),
                "model_id": str(model_id),
                "kind": str(kind),
                "rows": [
                    {
                        "rank": int(row["rank"]),
                        "family": str(row["family"]),
                        "n_features_display": format_value(row["n_features"], "count"),
                        "share": _number(row["share_of_positive_mean"]),
                        "share_display": format_value(row["share_of_positive_mean"], "metric"),
                        "share_sd_display": format_value(row["share_of_positive_sd"], "metric"),
                        "total_display": format_value(row["total_importance_mean"], "metric"),
                        "net_negative_display": (
                            format_value(row["n_folds_net_negative"], "count")
                            + " of "
                            + format_value(row["n_folds"], "count")
                        ),
                    }
                    for _, row in ordered.iterrows()
                ],
            }
        )
    return groups


def _stored_example(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        "record_uid": str(payload.get("record_uid", "")),
        "selection_rule": str(payload.get("selection_rule", "")),
        "task": str(payload.get("task", "")),
        "model_id": str(payload.get("model_id", "")),
        "method": str(payload.get("method", "")),
        "units": str(payload.get("units", "")),
        "true_class": str(payload.get("true_class", "")),
        "predicted_class": str(payload.get("predicted_class", "")),
        "probability_display": format_value(payload.get("probability"), "metric"),
        "base_value_display": format_value(payload.get("base_value"), "metric"),
        "decision_value_display": format_value(payload.get("decision_value"), "metric"),
        "n_shown_display": format_value(payload.get("n_shown"), "count"),
        "n_features_display": format_value(payload.get("n_features"), "count"),
        "caveats": [str(item) for item in payload.get("caveats", [])],
        "rows": [
            {
                "feature": str(item["feature"]),
                "family": str(item.get("family", _family(str(item["feature"])))),
                "contribution": _number(item.get("contribution")),
                "contribution_display": format_value(item.get("contribution"), "metric"),
                "raw_value_display": format_value(item.get("raw_value"), "metric"),
                "scaled_value_display": format_value(item.get("scaled_value"), "metric"),
                "weight_display": format_value(item.get("weight"), "metric"),
                "direction": str(item.get("direction", "")),
            }
            for item in payload.get("top_contributions", [])
        ],
    }


def explainability_payload() -> dict[str, Any]:
    """T117.2: what the models lean on, globally and for one stored record."""
    summary_path = EXPLAINABILITY_DIR + "/importance_summary.csv"
    family_path = EXPLAINABILITY_DIR + "/feature_family_importance.csv"
    coverage_path = EXPLAINABILITY_DIR + "/explainability_coverage.csv"
    example_path = EXPLAINABILITY_DIR + "/per_sample_explanation.json"
    sources = [summary_path, family_path, coverage_path, example_path]

    summary = _csv(summary_path)
    families = _csv(family_path)
    if summary is None or families is None:
        return {
            **_absent(
                "the Phase 81 explainability outputs are not in this checkout; run "
                "scripts/33_explainability.py",
                sources,
            ),
            "importance": [],
            "families": [],
            "coverage": [],
            "example": {"available": False},
            "notes": [],
        }

    coverage = _csv(coverage_path)
    example = _json(example_path)
    return {
        "available": True,
        "reason": None,
        "sources": [_source(s) for s in sources],
        "top_n": TOP_FEATURES,
        "importance": _importance_groups(summary),
        "families": _family_groups(families),
        "coverage": (
            []
            if coverage is None
            else [
                {
                    "model_id": str(row["model_id"]),
                    "methods": [
                        name
                        for name in ("permutation", "impurity", "shap")
                        if bool(row.get(name, False))
                    ],
                    "excluded_reason": (
                        None
                        if not isinstance(row.get("excluded_reason"), str)
                        else str(row["excluded_reason"])
                    ),
                }
                for _, row in coverage.iterrows()
            ]
        ),
        "example": _stored_example(example) if example is not None else {"available": False},
        "notes": [
            "Permutation importance is the drop in the fold's held-out score when "
            "one feature's column is shuffled, measured on the outer test rows of "
            "each fold. It says what the fitted model uses, not what is true about "
            "the heart.",
            "PhysioNet 2016's six sub-collections behave like six datasets and the "
            "top features reverse sign between them. A pooled ranking is a "
            "statement about this corpus, never a cardiac finding.",
            "The family share is each family's portion of the positive importance "
            "mass, averaged over folds. Families are different sizes (MFCC has 39 "
            "features, envelope 5), so a share is not a per-feature rate.",
        ],
    }


# ---------------------------------------------------------------------------
# T117.5 -- limitations
# ---------------------------------------------------------------------------


def _population() -> dict[str, Any]:
    payload = _json(_POPULATION)
    if payload is None:
        return _absent("EXP-D1's population_mismatch.json has not been written", [_POPULATION])
    train = payload.get("train", {})
    test = payload.get("test", {})
    transfer: list[dict[str, str]] = []
    t16 = _csv(_T16)
    if t16 is not None:
        # One row per level: the recording-level score and the patient-level
        # `any_present` collapse, which is the rule the CirCor tables headline.
        chosen = t16[
            (t16["metric"] == "balanced_accuracy")
            & (
                ((t16["level"] == "recording") & (t16["rule"] == "none"))
                | ((t16["level"] == "patient") & (t16["rule"] == "any_present"))
            )
        ]
        for _, row in chosen.iterrows():
            transfer.append(
                {
                    "level": str(row["level"]),
                    "rule": str(row["rule"]),
                    "n_units_display": format_value(row["n_units"], "count"),
                    "in_domain_display": format_value(row["in_domain_mean"], "metric"),
                    "external_display": format_value(row["external_value"], "metric"),
                }
            )
    return {
        "available": True,
        "sources": [_source(_POPULATION), _source(_T16)],
        "design": str(payload.get("design", "")),
        "framing_rule": str(payload.get("framing_rule", "")),
        "second_known_cause": str(payload.get("second_known_cause", "")),
        "train_dataset": str(train.get("dataset", "")),
        "train_n_with_age_display": format_value(train.get("n_records_with_age"), "count"),
        "train_age_median_display": format_value(train.get("age_median_years"), "mean_count"),
        "train_n_under_18_display": format_value(train.get("n_under_18"), "count"),
        "train_share_under_18_display": _percent(train.get("share_under_18")),
        "test_dataset": str(test.get("dataset", "")),
        "test_n_patients_display": format_value(test.get("n_patients"), "count"),
        "test_n_with_age_band_display": format_value(test.get("n_with_age_band"), "count"),
        "test_n_paediatric_display": format_value(test.get("n_paediatric"), "count"),
        "test_share_paediatric_display": _percent(test.get("share_paediatric_of_recorded")),
        "test_age_scale": str(test.get("age_scale", "")),
        "transfer": transfer,
    }


def _pascal() -> dict[str, Any]:
    frame = _csv(_T02)
    if frame is None:
        return _absent("T02 has not been generated", [_T02])
    tracks: list[dict[str, Any]] = []
    tasks = (("pascal_a", "PASCAL set_a (4-class)"), ("pascal_b", "PASCAL set_b (3-class)"))
    for task, title in tasks:
        rows = frame[frame["task"] == task].sort_values("n_records")
        if rows.empty:
            continue
        smallest = rows.iloc[0]
        tracks.append(
            {
                "task": task,
                "title": title,
                "n_records_display": format_value(int(rows["n_records"].sum()), "count"),
                "smallest_class": str(smallest["class"]),
                "smallest_n_display": format_value(smallest["n_records"], "count"),
                "classes": [
                    {
                        "class": str(row["class"]),
                        "n_records_display": format_value(row["n_records"], "count"),
                        "n_subjects_display": format_value(row["n_subjects"], "count"),
                    }
                    for _, row in rows.sort_values("class").iterrows()
                ],
            }
        )
    return {"available": True, "sources": [_source(_T02)], "tracks": tracks}


def _circor() -> dict[str, Any]:
    frame = _csv(_T02)
    if frame is None:
        return _absent("T02 has not been generated", [_T02])
    outcome = frame[frame["task"] == "circor_outcome"]
    murmur = frame[frame["task"] == "circor_murmur"]
    unknown = murmur[murmur["class"] == "Unknown"]
    return {
        "available": True,
        "sources": [_source(_T02)],
        "n_recordings_display": format_value(int(outcome["n_records"].sum()), "count"),
        "n_patients_display": format_value(int(outcome["n_subjects"].sum()), "count"),
        "unknown_murmur_patients_display": (
            format_value(unknown["n_subjects"].iloc[0], "count") if len(unknown) else NA_TEXT
        ),
    }


def _source_holdout() -> dict[str, Any]:
    frame = _csv(_TS5)
    if frame is None:
        return _absent("EXP-F3 (T-S5) has not been generated", [_TS5])
    rows = frame[frame["model_id"] == "M1"]
    if rows.empty:
        return _absent("T-S5 holds no row for the deployed model M1", [_TS5])
    row = rows.iloc[0]
    return {
        "available": True,
        "sources": [_source(_TS5)],
        "model_id": "M1",
        "n_folds_display": format_value(row["n_folds_holdout"], "count"),
        "auc_pooled_display": format_value(row["roc_auc_pooled"], "metric"),
        "auc_holdout_display": format_value(row["roc_auc_holdout"], "metric"),
        "balanced_accuracy_pooled_display": format_value(row["balanced_accuracy_pooled"], "metric"),
        "balanced_accuracy_holdout_display": format_value(
            row["balanced_accuracy_holdout"], "metric"
        ),
    }


def limitations_payload() -> dict[str, Any]:
    """T117.5: the caveats a reader must see before any number is quoted."""
    return {
        "population": _population(),
        "pascal": _pascal(),
        "circor": _circor(),
        "source_holdout": _source_holdout(),
    }


# ---------------------------------------------------------------------------
# T117.3 -- reports
# ---------------------------------------------------------------------------


def reportable_experiments() -> dict[str, Path]:
    """``{exp_id: directory}`` for every run `render_experiment_report` can read.

    Discovered from the four results parents rather than typed, and keyed by the
    directory NAME only -- so the API resolves an id by lookup in this mapping
    and a request can never name a path.
    """
    found: dict[str, Path] = {}
    for key in REPORT_PARENTS:
        parent = _outputs(key)
        if not parent.is_dir():
            continue
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "aggregate_metrics.csv").is_file():
                found.setdefault(child.name, child)
    return found


def objective_report_path() -> Path | None:
    path = _root() / OBJECTIVE_REPORT
    return path if path.is_file() else None


def reports_payload() -> dict[str, Any]:
    """T117.3: which reports exist, and what each one is built from."""
    from src.inference.predictor import TASKS
    from src.reporting.experiments import EXPERIMENTS

    titles = {Path(spec.directory).name: spec.title for spec in EXPERIMENTS}
    experiments = []
    for exp_id, directory in reportable_experiments().items():
        relative = directory.resolve().relative_to(_root().resolve()).as_posix()
        manifest = directory / "run_manifest.json"
        run_id = None
        if manifest.is_file():
            try:
                run_id = json.loads(manifest.read_text(encoding="utf-8")).get("run_id")
            except (OSError, ValueError):
                run_id = None
        experiments.append(
            {
                "exp_id": exp_id,
                "title": titles.get(exp_id, exp_id),
                "directory": relative,
                "run_id": run_id,
                "download_path": "/report/experiment/" + exp_id,
            }
        )

    objective = objective_report_path()
    return {
        "note": (
            "Reports are produced by the running inference API, from the same files "
            "this page was exported from. The browser requests a document and saves "
            "it; nothing in a report is computed by the page."
        ),
        "sample": {
            "download_path": "/report/sample",
            "tasks": [{"task": spec.key, "title": spec.title} for spec in TASKS.values()],
            "contents": (
                "The recording's waveform and spectrogram, the prediction, its "
                "confidence and low-confidence flag, the features that drove it where "
                "the model is linear, the screening disclaimer and the model version."
            ),
        },
        "experiments": experiments,
        "objective": {
            "available": objective is not None,
            "path": OBJECTIVE_REPORT,
            "table_id": "T29",
            "download_path": "/report/objectives",
            "reason": (
                None if objective is not None else OBJECTIVE_REPORT + " has not been generated"
            ),
        },
    }


# ---------------------------------------------------------------------------
# evidence rows for the three payloads
# ---------------------------------------------------------------------------


#: The script that writes each payload source that has no `.meta.json` of its
#: own. A source in neither place fails `test_every_evidence_entry_resolves_...`,
#: which is the point: an evidence row with no upstream is a dead end.
PRODUCERS: dict[str, str] = {
    EXPLAINABILITY_DIR + "/": "scripts/33_explainability.py",
    "outputs/08_circor_external_validation/EXP-D1/": "scripts/22_cross_dataset.py",
}


def _upstream(relative: str) -> list[str]:
    """What a payload source was built from: its meta's sources, else its producer."""
    path = _root() / relative
    meta = path.with_name(path.stem + ".meta.json")
    if meta.is_file():
        try:
            declared = json.loads(meta.read_text(encoding="utf-8")).get("sources", [])
        except (OSError, ValueError):
            declared = []
        found = [
            str(item["path"]) for item in declared if isinstance(item, dict) and item.get("path")
        ]
        if found:
            return found
    return [script for prefix, script in PRODUCERS.items() if relative.startswith(prefix)]


def payload_evidence(name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One evidence row per source file a payload read, for `evidence.json`."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "path" in node and "sha256" in node and node.get("sha256"):
                path = str(node["path"])
                if path not in seen:
                    seen.add(path)
                    rows.append(
                        {
                            "key": name + "." + Path(path).stem,
                            "kind": "page_payload",
                            "artifact": name,
                            "generated_from": path,
                            "generated_from_sha256": str(node["sha256"]),
                            "upstream_sources": _upstream(path),
                        }
                    )
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(payload)
    return rows
