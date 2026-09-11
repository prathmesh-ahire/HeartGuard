"""Phase 62 -- the Part VI deliverables: T07 and the figures G20, G21, G22.

Everything here READS what the searches emitted and draws it. Nothing is
recomputed, nothing is refitted, and no number is formatted that did not come out
of a CSV in ``outputs/05_search_optimization/`` -- research rule 1 applied to
figures, which are just numbers with axes on them.

G20 IS NOT A SINGLE-AXIS PLOT, AND THAT IS THE WHOLE DIFFICULTY
----------------------------------------------------------------
T62.2 asks for one figure overlaying Randomized, Bayesian, GA and PSO. Those four
do not share a y-axis:

    SO-01, SO-02   best_so_far is balanced accuracy   RISING    0.79 - 0.85
    SO-03a, SO-03b best_so_far is J, a cost           FALLING   0.27 - 0.32

The two ranges overlap, so a naive overlay produces a figure that looks entirely
reasonable and states that the GA converged far below random search -- comparing
a cost against an accuracy. This module therefore draws two panels sharing an
x-axis, each labelled with its own objective and direction, and stamps the
distinction into the caption. A single shared axis is never produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SearchReportError",
    "T07_FILENAME",
    "NOT_SEARCHED",
    "FIGURES",
    "search_section",
    "figures_dir",
    "build_t07",
    "read_t07",
]

T07_FILENAME = "search_space_and_best_parameters.csv"

#: What a cell says when no search ever visited that parameter.
NOT_SEARCHED = "not searched"

#: figure id -> filename. The figures are drawn by ``result_graphs.py`` through
#: the graph engine since Phase 93, which names them from their titles; the
#: Phase 62 names (``search_convergence_plot.png`` ...) had no source CSV beside
#: them and no registry number, and are retired. See note.md, 2026-09-11.
FIGURES: dict[str, str] = {
    "G20": "G20_search_convergence_plot.png",
    "G21": "G21_all_features_versus_selected_features.png",
    "G22": "G22_f1_and_accuracy_versus_feature_count.png",
}


class SearchReportError(RuntimeError):
    """A Part VI deliverable cannot be built from what is on disk."""


def search_section(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config

    return (
        Path(out_dir)
        if out_dir is not None
        else Path(load_config("paths").require("outputs.search_optimization"))
    )


def figures_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    root = (
        Path(out_dir)
        if out_dir is not None
        else Path(load_config("paths").require("outputs.figures_diagrams"))
    )
    return Path(ensure_dir(root))


def _require(path: Path) -> Path:
    if not path.is_file():
        raise SearchReportError(
            str(path) + " does not exist; the Part VI report cannot be built without it"
        )
    return path


# ---------------------------------------------------------------------------
# T62.1 -- T07
# ---------------------------------------------------------------------------


def _format_range(row: Any) -> str:
    """The declared range or choice set, as one readable cell."""
    if isinstance(row.get("choices"), str) and row["choices"]:
        return "{" + row["choices"].replace("|", ", ") + "}"
    low, high = row.get("low"), row.get("high")
    if low is None or high is None or (isinstance(low, float) and np.isnan(low)):
        return "fixed"
    return "[" + str(low) + ", " + str(high) + "]"


def _render(value: Any) -> str:
    """A selected hyperparameter value, as a cell a reader cannot misread.

    `None` is a legitimate value for several parameters -- M3's `class_weight`
    chose it, meaning "no class weighting" -- and Python's `None` writes to CSV
    as an empty cell that `read_csv` returns as NaN. In a deliverable that is
    indistinguishable from "we did not record this", so a real decision would
    read as a hole. It is written out as the literal string instead.
    """
    if value is None:
        return "None"
    return str(value)


def read_t07(path: str | Path | None = None) -> Any:
    """Read T07 back without pandas turning real values into NaN.

    `read_csv` treats the strings "None", "NA", "null" and "nan" as missing by
    default, and `class_weight=None` is a value several models legitimately
    chose. Reading T07 naively therefore erases a real decision and shows a hole
    where the table says "None". Every programmatic reader of this file must use
    `keep_default_na=False`; this function is the one that does.
    """
    import pandas as pd

    from src.utils.config import load_config

    target = (
        Path(path)
        if path is not None
        else search_section() / T07_FILENAME
    )
    del load_config
    return pd.read_csv(target, keep_default_na=False)


def build_t07(section: Path | None = None, spaces: Any | None = None) -> Any:
    """T07: variable, range, distribution and final selected value, per model.

    The "final selected value" is taken from **SO-02**, the method the user
    adopted on 2026-08-28, with SO-01's value carried alongside. Both are shown
    rather than only the winner, because the two methods disagreeing on a
    hyperparameter is information about how flat that dimension is -- and a table
    that showed only one would hide it.
    """
    import json

    import pandas as pd

    from src.utils.config import load_config

    section = search_section() if section is None else Path(section)
    if spaces is None:
        spaces_path = _require(
            Path(load_config("paths").require("outputs.models")) / "model_search_spaces.csv"
        )
        spaces = pd.read_csv(spaces_path)

    best: dict[str, dict[str, Any]] = {}
    for exp in ("SO-01", "SO-02"):
        path = section / exp / "best_parameters.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("searches", []):
            best.setdefault(exp, {})[str(entry["model_id"])] = entry

    if not best:
        raise SearchReportError(
            "no best_parameters.json under " + str(section) + "; run the searches first"
        )

    rows = []
    for record in spaces.to_dict("records"):
        model_id = str(record["model_id"])
        parameter = str(record["parameter"])
        row: dict[str, Any] = {
            "model_id": model_id,
            "parameter": parameter,
            "distribution": str(record.get("kind", "")),
            "range_or_choices": _format_range(record),
        }
        for exp in ("SO-01", "SO-02"):
            entry = best.get(exp, {}).get(model_id)
            chosen = (entry or {}).get("best_params", {})
            row[exp.replace("-", "_").lower() + "_selected"] = (
                _render(chosen.get(parameter, NOT_SEARCHED)) if entry else NOT_SEARCHED
            )
            row[exp.replace("-", "_").lower() + "_score"] = (
                float(entry["best_score"]) if entry else float("nan")
            )
            row[exp.replace("-", "_").lower() + "_searched"] = bool(entry)
        # NOT_SEARCHED rather than an empty cell. An empty string survives the
        # DataFrame but comes back from `read_csv` as NaN, which renders in a
        # deliverable as the literal text "nan" -- a blank a reader has to guess
        # at. M2 is declared with a full search space and was never searched;
        # the table has to say that rather than leave a hole.
        if row.get("so_02_searched"):
            row["final_selected"] = row.get("so_02_selected", NOT_SEARCHED)
            row["final_source"] = "SO-02"
        elif row.get("so_01_searched"):
            row["final_selected"] = row.get("so_01_selected", NOT_SEARCHED)
            row["final_source"] = "SO-01"
        else:
            row["final_selected"] = NOT_SEARCHED
            row["final_source"] = NOT_SEARCHED
        for exp in ("so_01", "so_02"):
            if not row[exp + "_searched"]:
                row[exp + "_selected"] = NOT_SEARCHED
        rows.append(row)
    return pd.DataFrame(rows)
