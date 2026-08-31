"""The feature registry and the search results, as exportable payloads (Phase 115).

Two pages' worth of data, kept in one module because they answer one question
between them: what the 138 features are, and which of them the search kept.

## The registry order is load-bearing and is never sorted

`feature_inventory.csv` is emitted in the locked column order of the 138-vector
-- time, frequency, MFCC, chroma, DWT, envelope -- which is a literal in
`src/feature_extraction/registry.py` and is fingerprinted. note.md records T05
coming out alphabetical once: every count was right and the order told a reader
the order was arbitrary. It is not. `index` is exported and the payload is
sorted by it, never by name.

## The per-record feature vector comes from the matrix, and is committed

T115.1 asks for a per-record feature vector view. The values live in
`all_features_matrix.parquet`, which is gitignored, so one pinned record's 138
values are written to `outputs/03_features/example_feature_vector.csv` and the
payload reads the parquet where it exists and that CSV where it does not -- the
same arrangement Phase 113's segmentation and Phase 115's curves use, and for
the same reason.

## What "selected" means, and what it does not

`selected_feature_subset.csv` carries `selected_in_folds` out of `n_folds`. A
feature chosen in 3 of 25 folds and one chosen in 25 are both "selected" by the
column name and are not the same claim, so the count travels with every row and
the payload states it. Feature selection ran **inside** the training fold, which
is why there is a per-fold count at all rather than one global subset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "EXAMPLE_VECTOR_CSV",
    "EXAMPLE_RECORD_UID",
    "SEARCH_RUNS",
    "export_example_vector",
    "features_payload",
    "optimization_payload",
]

log = get_logger("reporting.method")

#: The record whose feature vector the page shows. Pinned, not picked: the same
#: recording the inference gate uses, so a reader can follow one record from its
#: waveform on the preprocessing page through its 138 features to its stored
#: out-of-fold probability.
EXAMPLE_RECORD_UID = "D1_training-a_a0005"
EXAMPLE_VECTOR_CSV = "outputs/03_features/example_feature_vector.csv"

#: The search runs, with what each one searched over. An `SO-` id alone does not
#: tell a reader whether they are looking at a hyperparameter search, a feature
#: search or a weight search.
SEARCH_RUNS: tuple[tuple[str, str, str], ...] = (
    ("SO-01", "Randomized hyperparameter search", "Random sampling of the model search space."),
    (
        "SO-02",
        "Bayesian hyperparameter search",
        "Sequential model-based search over the same space.",
    ),
    (
        "SO-03a",
        "Genetic feature search",
        "A GA over feature masks, with the weights searched jointly.",
    ),
    ("SO-03b", "Particle-swarm feature search", "A PSO over the same mask space."),
    ("SO-04", "Feature-count sweep", "How performance moves as the subset size changes."),
    ("SO-05", "Ensemble weight search", "Voting weights for the heterogeneous ensemble."),
    ("SO-06", "Multi-objective front", "Performance against complexity, as a Pareto front."),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read(relative: str) -> Any:
    """A committed CSV, or ``None`` when it has not been produced."""
    import pandas as pd

    path = _project_root() / relative
    if not path.is_file():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# T115.1 / T115.2 -- the feature registry
# ---------------------------------------------------------------------------


def export_example_vector(out_path: str | Path | None = None) -> Path:
    """Write one record's 138 feature values from the matrix to a committed CSV."""
    import pandas as pd

    root = _project_root()
    matrix = root / "outputs" / "03_features" / "all_features_matrix.parquet"
    if not matrix.is_file():
        raise FileNotFoundError("no all_features_matrix.parquet to read")

    frame = pd.read_parquet(matrix)
    row = frame[frame["record_uid"] == EXAMPLE_RECORD_UID]
    if row.empty:
        raise KeyError(EXAMPLE_RECORD_UID + " is not in all_features_matrix.parquet")

    inventory = _read("outputs/03_features/feature_inventory.csv")
    if inventory is None:
        raise FileNotFoundError("no feature_inventory.csv to order the vector by")

    record = row.iloc[0]
    rows = [
        {
            "record_uid": EXAMPLE_RECORD_UID,
            "index": int(item["index"]),
            "name": str(item["name"]),
            "family": str(item["family"]),
            "value": float(record[str(item["name"])])
            if str(item["name"]) in frame.columns
            else float("nan"),
        }
        for _, item in inventory.sort_values("index").iterrows()
    ]

    target = Path(out_path) if out_path is not None else root / EXAMPLE_VECTOR_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target, index=False, lineterminator="\n")
    log.info("wrote %d feature values for %s -> %s", len(rows), EXAMPLE_RECORD_UID, target)
    return target


def _example_vector() -> dict[str, Any]:
    from src.reporting.tables import format_value

    root = _project_root()
    csv_path = root / EXAMPLE_VECTOR_CSV
    source = "all_features_matrix.parquet"
    if (root / "outputs" / "03_features" / "all_features_matrix.parquet").is_file():
        try:
            export_example_vector(csv_path)
        except (FileNotFoundError, KeyError, OSError) as error:
            log.info("cannot rebuild the example feature vector: %s", error)
            source = "committed csv"
    else:
        source = "committed csv"

    frame = _read(EXAMPLE_VECTOR_CSV)
    if frame is None:
        return {
            "available": False,
            "reason": (
                "neither all_features_matrix.parquet nor " + EXAMPLE_VECTOR_CSV + " is present"
            ),
            "values": [],
        }

    ordered = frame.sort_values("index")
    return {
        "available": True,
        "reason": None,
        "source": source,
        "record_uid": EXAMPLE_RECORD_UID,
        "n_features": len(ordered),
        "note": (
            "One recording's full 138-vector, in the locked registry order. This is "
            "the same record the inference gate scores, so its waveform, its features "
            "and its stored out-of-fold probability are all the same bytes."
        ),
        "values": [
            {
                "index": int(item["index"]),
                "name": str(item["name"]),
                "family": str(item["family"]),
                "value": _finite(item["value"]),
                "display": format_value(item["value"], "metric"),
            }
            for _, item in ordered.iterrows()
        ],
    }


def features_payload() -> dict[str, Any]:
    """The 138-feature registry, the family composition, and the selected subset."""
    from src.reporting.tables import format_value

    inventory = _read("outputs/03_features/feature_inventory.csv")
    if inventory is None:
        raise FileNotFoundError("the features page needs outputs/03_features/feature_inventory.csv")

    families = _read("outputs/03_features/feature_family_summary.csv")
    selected = _read("outputs/03_features/selected_feature_subset.csv")
    separation = _read("outputs/03_features/feature_class_separation.csv")

    ordered = inventory.sort_values("index")
    if list(ordered["index"]) != list(range(len(ordered))):
        raise ValueError(
            "feature_inventory.csv is not a contiguous 0..n-1 index; the registry "
            "order is load-bearing and cannot be reconstructed by sorting"
        )

    separation_by_name: dict[str, float | None] = {}
    if separation is not None:
        separation_by_name = {
            str(row["feature"]): _finite(row["abs_cohens_d"]) for _, row in separation.iterrows()
        }

    return {
        "n_features": len(ordered),
        "registry_note": (
            "The column order of the 138-vector is a literal in "
            "src/feature_extraction/registry.py and is fingerprinted; two runs that "
            "disagree on that fingerprint are not comparable. This list is in that "
            "order and is never sorted by name."
        ),
        "features": [
            {
                "index": int(row["index"]),
                "name": str(row["name"]),
                "family": str(row["family"]),
                "extractor": str(row["extractor"]),
                "unit": None if row.get("unit") is None else str(row.get("unit")),
                "description": str(row.get("description") or ""),
                "abs_cohens_d": separation_by_name.get(str(row["name"])),
                "abs_cohens_d_display": format_value(
                    separation_by_name.get(str(row["name"])), "metric"
                ),
            }
            for _, row in ordered.iterrows()
        ],
        "families": _families(families, ordered),
        "selected": _selected(selected),
        "example_vector": _example_vector(),
    }


def _families(families: Any, inventory: Any) -> list[dict[str, Any]]:
    """Per-family counts, recomputed from the inventory and checked against FE-02."""
    from src.reporting.tables import format_value

    counted = inventory.groupby("family", sort=False).size().to_dict()
    order: list[str] = []
    for name in inventory["family"]:
        if str(name) not in order:
            order.append(str(name))

    if families is not None and "family" in families.columns:
        declared = {str(row["family"]): row for _, row in families.iterrows()}
        for name in order:
            row = declared.get(name)
            if row is None:
                continue
            for column in ("n_features", "counted_features", "count"):
                if column in families.columns:
                    if int(row[column]) != int(counted[name]):
                        raise ValueError(
                            "family "
                            + name
                            + " counts "
                            + str(counted[name])
                            + " features in the inventory but "
                            + str(int(row[column]))
                            + " in feature_family_summary.csv"
                        )
                    break

    return [
        {
            "family": name,
            "n_features": int(counted[name]),
            "n_features_display": format_value(int(counted[name]), "count"),
            "first_index": int(inventory[inventory["family"] == name]["index"].min()),
        }
        for name in order
    ]


def _selected(selected: Any) -> dict[str, Any]:
    from src.reporting.tables import format_value

    if selected is None:
        return {
            "available": False,
            "reason": "outputs/03_features/selected_feature_subset.csv has not been produced",
            "features": [],
        }
    return {
        "available": True,
        "reason": None,
        "n_selected": len(selected),
        "stability_note": (
            "Feature selection runs inside the training fold, so a feature is "
            "selected in some number of folds rather than globally. A feature kept "
            "in 3 of 25 folds and one kept in 25 are both listed here and are not "
            "the same claim, which is why the count is on every row."
        ),
        "features": [
            {
                "rank": int(row["rank"]),
                "feature": str(row["feature"]),
                "family": str(row["family"]),
                "ranker": str(row["ranker"]),
                "selected_in_folds": int(row["selected_in_folds"]),
                "n_folds": int(row["n_folds"]),
                "share": _finite(row["selected_in_folds"] / row["n_folds"]),
                "share_display": format_value(row["selected_in_folds"] / row["n_folds"], "metric"),
            }
            for _, row in selected.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# T115.5 -- the search runs
# ---------------------------------------------------------------------------


def _json(relative: str) -> Any:
    path = _project_root() / relative
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _convergence(run_id: str) -> dict[str, Any]:
    """Best-so-far against trial, per model and outer fold."""
    frame = _read("outputs/05_search_optimization/" + run_id + "/convergence.csv")
    if frame is None or frame.empty:
        return {"available": False, "reason": run_id + " wrote no convergence.csv", "series": []}

    key_columns = [c for c in ("method", "model_id", "outer_fold") if c in frame.columns]
    value_column = "best_so_far" if "best_so_far" in frame.columns else "score"
    x_column = "trial" if "trial" in frame.columns else "generation"
    if x_column not in frame.columns or value_column not in frame.columns:
        return {
            "available": False,
            "reason": run_id + "'s convergence.csv has no trial/score columns",
            "series": [],
        }

    series: list[dict[str, Any]] = []
    for key, part in frame.groupby(key_columns or [x_column], sort=True):
        values = key if isinstance(key, tuple) else (key,)
        ordered = part.sort_values(x_column)
        series.append(
            {
                "label": " · ".join(str(v) for v in values),
                "x": [_finite(v) for v in ordered[x_column]],
                "y": [_finite(v) for v in ordered[value_column]],
            }
        )
    return {
        "available": True,
        "reason": None,
        "x_label": x_column,
        "y_label": value_column,
        "n_series": len(series),
        "series": series,
    }


def optimization_payload() -> dict[str, Any]:
    """Convergence, search space, selected parameters, features and the front."""
    from src.reporting.tables import format_value

    runs: list[dict[str, Any]] = []
    for run_id, title, description in SEARCH_RUNS:
        directory = _project_root() / "outputs" / "05_search_optimization" / run_id
        runs.append(
            {
                "run_id": run_id,
                "title": title,
                "description": description,
                "available": directory.is_dir(),
                "reason": None if directory.is_dir() else run_id + " has not run",
                "convergence": _convergence(run_id) if directory.is_dir() else None,
                "best_parameters": _json(
                    "outputs/05_search_optimization/" + run_id + "/best_parameters.json"
                ),
            }
        )

    front = _read("outputs/05_search_optimization/SO-06/pareto_front.csv")
    weights = _read("outputs/05_search_optimization/SO-05/weight_stability.csv")
    equal = _read("outputs/05_search_optimization/SO-05/equal_vs_optimized.csv")
    sweep = _read("outputs/05_search_optimization/SO-04/feature_count_curve.csv")
    methods = _read("outputs/05_search_optimization/search_method_comparison.csv")

    return {
        "n_runs": len(SEARCH_RUNS),
        "fold_safety_note": (
            "Every search runs inside the training fold. The outer test fold is "
            "never seen by a search, never scored during one, and never used to fit "
            "a scaler or a selector. That is why each run reports a curve per outer "
            "fold rather than one curve."
        ),
        "runs": runs,
        "pareto": _frame_payload(front, "SO-06 pareto_front.csv"),
        "weight_stability": _frame_payload(weights, "SO-05 weight_stability.csv"),
        "equal_vs_optimized": _frame_payload(equal, "SO-05 equal_vs_optimized.csv"),
        "feature_count_curve": _frame_payload(sweep, "SO-04 feature_count_curve.csv"),
        "method_comparison": _frame_payload(methods, "search_method_comparison.csv"),
        "final_weights": _json("outputs/05_search_optimization/SO-05/final_weights.json"),
        "operating_point": _json("outputs/05_search_optimization/SO-06/operating_point.json"),
        "n_display": format_value(len(SEARCH_RUNS), "count"),
    }


def _frame_payload(frame: Any, source: str) -> dict[str, Any]:
    """A CSV as columns, formatted in Python, with its source named."""
    from src.reporting.tables import format_value, infer_kind

    if frame is None or frame.empty:
        return {
            "available": False,
            "reason": source + " has not been produced",
            "source": source,
            "columns": [],
            "n_rows": 0,
        }

    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        kind = infer_kind(str(name), frame[name])
        values = [_finite(v) for v in frame[name]] if kind != "text" else None
        columns.append(
            {
                "name": str(name),
                "kind": kind,
                "display": [format_value(v, kind) for v in frame[name]],
                "values": values,
            }
        )
    return {
        "available": True,
        "reason": None,
        "source": source,
        "n_rows": len(frame),
        "columns": columns,
    }
