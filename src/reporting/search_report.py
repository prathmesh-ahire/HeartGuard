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
    "plot_convergence",
    "plot_all_versus_selected",
    "plot_feature_count_curve",
]

T07_FILENAME = "search_space_and_best_parameters.csv"

#: What a cell says when no search ever visited that parameter.
NOT_SEARCHED = "not searched"

#: figure id -> filename, as named in todo.md.
FIGURES: dict[str, str] = {
    "G20": "search_convergence_plot.png",
    "G21": "all_features_vs_selected_features.png",
    "G22": "f1_accuracy_vs_feature_count.png",
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


# ---------------------------------------------------------------------------
# T62.2 -- G20
# ---------------------------------------------------------------------------


def plot_convergence(path: str | Path, section: Path | None = None) -> Path:
    """G20: all four searches, on TWO panels because they minimise different things."""
    import pandas as pd

    from src.reporting.plot_style import class_color, figure, save_figure

    section = search_section() if section is None else Path(section)
    hyper = []
    for exp, label in (("SO-01", "SO-01 Randomized"), ("SO-02", "SO-02 Bayesian")):
        path_ = section / exp / "convergence.csv"
        if path_.is_file():
            frame = pd.read_csv(path_)
            frame["label"] = label
            hyper.append(frame)
    mask = []
    for exp, label in (("SO-03a", "SO-03a Genetic"), ("SO-03b", "SO-03b PSO")):
        path_ = section / exp / "convergence.csv"
        if path_.is_file():
            frame = pd.read_csv(path_)
            frame["label"] = label
            mask.append(frame)
    if not hyper and not mask:
        raise SearchReportError("no convergence traces under " + str(section))

    # "tall", not "double": two stacked panels in a 3.2-inch canvas squash each
    # plot to about an inch, and the two y-axis labels then overlap each other.
    fig, axes = figure("tall", nrows=2, sharex=False)
    top, bottom = axes

    colour = 0
    for frame in hyper:
        for model_id, block in frame.groupby("model_id"):
            block = block.sort_values("trial")
            top.plot(
                block["trial"] + 1, block["best_so_far"],
                color=class_color(colour), linewidth=1.2,
                label=str(frame["label"].iloc[0]) + " - " + str(model_id),
            )
            colour += 1
    top.set_ylabel("best balanced\naccuracy so far", fontsize=8)
    top.set_xlabel("trial", fontsize=8)
    top.set_title(
        "Hyperparameter search - MAXIMISING balanced accuracy (higher is better)",
        loc="left", fontsize=9,
    )
    top.legend(fontsize=5.5, ncol=3, frameon=False, loc="lower right")
    top.grid(True, alpha=0.3)

    colour = 0
    for frame in mask:
        for label, block in frame.groupby("label"):
            block = block.sort_values("generation")
            bottom.plot(
                block["generation"] + 1, block["best_so_far"],
                color=class_color(colour + 4), linewidth=1.4, marker="o", markersize=2.5,
                label=str(label),
            )
            colour += 1
    bottom.set_ylabel("best J so far", fontsize=8)
    bottom.set_xlabel("generation / iteration", fontsize=8)
    bottom.set_title(
        "Feature-mask search - MINIMISING the multi-objective score J (lower is better)",
        loc="left", fontsize=9,
    )
    bottom.legend(fontsize=7, frameon=False, loc="upper right")
    bottom.grid(True, alpha=0.3)
    for axis in (top, bottom):
        axis.tick_params(labelsize=7)

    return save_figure(
        fig, path,
        source=(
            "G20 (T62.2). Sources: SO-01/SO-02/SO-03a/SO-03b convergence.csv.\n"
            "TWO PANELS, NOT ONE AXIS: the hyperparameter searches maximise balanced "
            "accuracy; the mask searches minimise J, a cost. The two\n"
            "ranges overlap numerically, so a shared axis would read as the mask "
            "searches converging 'below' the others. They are different quantities.\n"
            "Every curve is best-so-far: PSO keeps no elite, so its per-iteration "
            "best is not monotone and only the running best is comparable."
        ),
    )


# ---------------------------------------------------------------------------
# T62.3 -- G21
# ---------------------------------------------------------------------------


def plot_all_versus_selected(path: str | Path, section: Path | None = None) -> Path:
    """G21: every configuration's held-out metrics, all 138 against the subsets."""
    import pandas as pd

    from src.reporting.plot_style import class_color, figure, save_figure

    section = search_section() if section is None else Path(section)
    frame = pd.read_csv(_require(section / "SO-04" / "all_features_vs_selected.csv"))

    metrics = ["macro_f1", "balanced_accuracy", "sensitivity", "specificity"]
    summary = (
        frame.groupby("configuration")
        .agg({**dict.fromkeys(metrics, "mean"), "n_features": "mean"})
        .sort_values("balanced_accuracy")
    )
    errors = frame.groupby("configuration")[metrics].sem()

    fig, ax = figure("double")
    positions = np.arange(len(summary))
    width = 0.2
    for index, metric in enumerate(metrics):
        ax.barh(
            positions + (index - 1.5) * width,
            summary[metric],
            height=width,
            xerr=errors.loc[summary.index, metric],
            color=class_color(index),
            label=metric.replace("_", " "),
            error_kw={"elinewidth": 0.6, "capsize": 1.5},
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [
            str(name).replace("_", " ")
            + "  (" + str(round(summary.loc[name, "n_features"])) + " features)"
            for name in summary.index
        ],
        fontsize=7,
    )
    ax.set_xlabel("score on the held-out outer folds (mean +/- SE)", fontsize=8)
    ax.set_xlim(0.6, 1.0)
    ax.tick_params(labelsize=7)
    # Above the axes, not inside them: at loc="lower right" the legend box sits
    # squarely on top of the bottom group's bars, which on this chart is the
    # all-138 baseline every other row is meant to be read against.
    ax.legend(
        fontsize=7, ncol=4, frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, 1.01),
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("All 138 features versus the selected subsets", loc="left", pad=22)

    return save_figure(
        fig, path,
        source=(
            "G21 (T62.3). Source: SO-04/all_features_vs_selected.csv, "
            + str(frame["outer_fold"].nunique()) + " held-out outer folds, seed 42.\n"
            "Every arm was selected inside its own fold's training rows and scored "
            "once on that fold's test rows. Screening prototype - not a diagnostic tool."
        ),
    )


# ---------------------------------------------------------------------------
# T62.4 -- G22
# ---------------------------------------------------------------------------


def plot_feature_count_curve(path: str | Path, section: Path | None = None) -> Path:
    """G22: performance against feature count, one curve per ranker."""
    import json

    import pandas as pd

    from src.reporting.plot_style import class_color, figure, save_figure

    section = search_section() if section is None else Path(section)
    curve = pd.read_csv(_require(section / "SO-04" / "feature_count_curve.csv"))
    settings_path = section / "SO-04" / "so04_settings.json"
    chosen = (
        json.loads(settings_path.read_text(encoding="utf-8")).get("chosen", {})
        if settings_path.is_file()
        else {}
    )

    fig, axes = figure("double", nrows=2, sharex=True)
    for index, (metric, label) in enumerate(
        (("macro_f1", "macro-F1"), ("balanced_accuracy", "balanced accuracy"))
    ):
        ax = axes[index]
        for position, (ranker, block) in enumerate(curve.groupby("ranker")):
            block = block.sort_values("k")
            ax.errorbar(
                block["k"], block[metric + "_mean"],
                yerr=block[metric + "_std"] / np.sqrt(block["n_evaluations"]),
                color=class_color(position), marker="o", markersize=3,
                linewidth=1.2, capsize=2, label=str(ranker),
            )
        full = curve[curve["k"] == curve["k"].max()][metric + "_mean"].mean()
        ax.axhline(full, color="#888888", linestyle="--", linewidth=0.9)
        ax.annotate(
            "all " + str(int(curve["k"].max())) + " features",
            xy=(curve["k"].max(), full), xytext=(-4, 4),
            textcoords="offset points", ha="right", fontsize=6, color="#555555",
        )
        if chosen:
            ax.axvline(int(chosen["k"]), color="#B00020", linestyle=":", linewidth=1.1)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if index == 0:
            ax.legend(fontsize=6.5, ncol=4, frameon=False, loc="lower right")
            ax.set_title(
                "Performance versus feature count (inner folds, mean +/- SE)",
                loc="left", fontsize=9,
            )
    axes[-1].set_xlabel("features kept (k)", fontsize=8)

    marker = ""
    if chosen:
        marker = (
            "Red line: the shipped subset, " + str(chosen.get("ranker", "")) + " at k="
            + str(chosen.get("k", "")) + ", chosen by J under a "
            + str(chosen.get("selection_rule", "")) + " performance guard.\n"
        )
    return save_figure(
        fig, path,
        source=(
            "G22 (T62.4). Source: SO-04/feature_count_curve.csv - inner validation "
            "folds only, so no point on this curve saw a test row.\n" + marker
            + "Screening prototype - not a diagnostic tool."
        ),
    )
