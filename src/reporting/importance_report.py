"""FE-11, G18 and G19 -- what the models use (Phase 81).

**FE-11** `top_feature_importance.csv` is the top features per model under the
**permutation** measure, because T81.2 makes that the primary reported one. A
"top features" table headed by the impurity ranking would be publishing the foil
as the finding.

**G18** is the importance plot: the family-level rollup beside the two measures'
disagreement. The family total is the stable quantity -- individual importances
of correlated features are close to arbitrary between members, and which of the
39 MFCC columns wins tells nobody anything.

**G19** is the top-20 chart for the reported model, with the fold spread drawn.
A feature whose importance interval crosses zero is not a top feature; it is a
feature the folds could not agree about, and the bar has to show that.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.reporting.graphs import Graph, GraphSpec, subplots
from src.reporting.plot_style import class_color
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "PRIMARY_KIND",
    "FAMILY_ORDER",
    "build_fe11",
    "build_g18",
    "build_g19",
]

log = get_logger("reporting.importance")

#: T81.2: permutation importance is the primary reported measure.
PRIMARY_KIND = "permutation"

#: Registry order, not alphabetical -- the same order G10 and the feature vector
#: use, so two figures about the six families cannot disagree about their order.
FAMILY_ORDER: tuple[str, ...] = (
    "time",
    "frequency",
    "mfcc",
    "chroma",
    "dwt",
    "envelope",
)

_DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support "
    "prototype. A feature the model relies on is not a clinical marker."
)

_MEASURE_NOTE = (
    "PERMUTATION IS THE REPORTED MEASURE AND IMPURITY IS THE FOIL. Impurity "
    "importance is measured on the TRAINING rows and is biased toward features "
    "with many split points, which on this matrix means the 39 MFCC and 24 "
    "wavelet columns. Permutation importance is measured on rows the model never "
    "saw and asks what the fitted model actually uses. Where they disagree, "
    "prefer permutation; nothing here averages the two."
)

_NEGATIVE_NOTE = (
    "PERMUTATION IMPORTANCE CAN BE NEGATIVE and a negative value is not a small "
    "positive one: it means shuffling the feature IMPROVED the held-out score, "
    "so the model was being misled by it on those folds. Family shares are "
    "therefore taken over the positive part only -- a signed grand total can sit "
    "near zero and turn every share into nonsense."
)


def build_fe11(
    top: Any, family: Any, sources: tuple[str, ...], command: str = ""
) -> Table:
    """FE-11 -- the reported top features per model, with their fold spread."""
    frame = top.sort_values(["model_id", "rank"]).reset_index(drop=True)

    columns = (
        Column("model_id", "Model"),
        Column("kind", "Measure"),
        Column("rank", "Rank", kind="integer"),
        Column("feature", "Feature"),
        Column("importance_mean", "Importance mean", kind="metric", places=5),
        Column("importance_sd", "Importance SD", kind="metric", places=5),
        Column("importance_min", "Min across folds", kind="metric", places=5),
        Column("importance_max", "Max across folds", kind="metric", places=5),
        Column("rank_mean", "Mean per-fold rank", kind="metric", places=1),
        Column("rank_sd", "Rank SD", kind="metric", places=1),
        Column("n_folds", "Folds", kind="count"),
        Column("n_folds_positive", "Folds positive", kind="count"),
    )

    leaders = (
        family[family["kind"] == PRIMARY_KIND]
        .sort_values("total_importance_mean", ascending=False)
        .groupby("model_id")
        .head(1)
    )
    family_note = (
        "T81.4 -- family-level rollup, which is the stable quantity. Leading "
        "family per model under "
        + PRIMARY_KIND
        + ": "
        + "; ".join(
            str(row["model_id"])
            + " -> "
            + str(row["family"])
            + " ("
            + format(float(row["share_of_positive_mean"]) * 100, ".1f")
            + "% of positive importance)"
            for _, row in leaders.iterrows()
        )
        + ". Full rollup in feature_family_importance.csv."
        if len(leaders)
        else "No family rollup was produced."
    )

    spec = TableSpec(
        table_id="FE-11",
        title="Top Feature Importance",
        caption=(
            "The highest-ranked features per model under permutation importance, "
            "computed on held-out rows across the five folds of one repeat -- a "
            "complete partition of the corpus, so every record is scored exactly "
            "once. The rank is taken on the mean; 'Rank SD' says how much the "
            "per-fold ranks themselves moved, which is the number that separates "
            "a genuinely top feature from an unstable one."
        ),
        sources=sources,
        columns=columns,
        exp_id="n/a (fitted for this analysis, config defaults)",
        objective="O2 (feature contribution), O3 (model behaviour)",
        dataset="D1 PhysioNet 2016",
        notes=(
            _MEASURE_NOTE,
            _NEGATIVE_NOTE,
            "'Folds positive' counts how many of the folds gave the feature a "
            "positive importance. A feature ranked highly on the mean but "
            "positive in only three of five folds is carried by one fold.",
            "Models are fitted under configs/models.yaml defaults, not a searched "
            "configuration: the ranking is meant to be a property of the feature "
            "set and the model family, and a per-fold search would make each "
            "fold's ranking a statement about a different model.",
            family_note,
            _DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)


def build_g18(
    family: Any, aggregated: Any, sources: tuple[str, ...], command: str = ""
) -> Graph:
    """G18 -- the family rollup, and where the two measures disagree."""

    def draw(data: Any) -> Any:
        fig, axes = subplots((10.0, 3.8), ncols=2, nrows=1)
        left, right = np.atleast_1d(axes)[0], np.atleast_1d(axes)[1]

        primary = data[data["kind"] == PRIMARY_KIND]
        models = sorted(set(primary["model_id"]))
        families = [f for f in FAMILY_ORDER if f in set(primary["family"])]
        width = 0.8 / max(len(models), 1)

        for index, model in enumerate(models):
            block = primary[primary["model_id"] == model].set_index("family")
            values, errors = [], []
            for name in families:
                if name in block.index:
                    values.append(float(block.loc[name, "share_of_positive_mean"]))
                    errors.append(float(block.loc[name, "share_of_positive_sd"] or 0.0))
                else:
                    values.append(np.nan)
                    errors.append(np.nan)
            positions = np.arange(len(families)) + index * width - 0.4 + width / 2
            left.bar(
                positions,
                values,
                width=width,
                yerr=errors,
                capsize=1.5,
                color=class_color(index),
                edgecolor="black",
                linewidth=0.3,
                label=model,
                error_kw={"elinewidth": 0.6},
            )
        left.set_xticks(np.arange(len(families)))
        left.set_xticklabels(families, fontsize=6)
        left.set_ylabel("share of positive permutation importance", fontsize=7)
        left.set_title("Which family carries the signal (T81.4)", fontsize=8)
        left.legend(fontsize=5, ncol=3)

        # The disagreement panel: impurity rank against permutation rank, for
        # the models that have both.
        both = aggregated.pivot_table(
            index=["model_id", "feature"], columns="kind", values="rank"
        ).dropna()
        if "impurity" in both.columns and PRIMARY_KIND in both.columns:
            for index, model in enumerate(sorted({m for m, _ in both.index})):
                part = both.loc[model]
                right.scatter(
                    part["impurity"],
                    part[PRIMARY_KIND],
                    s=8,
                    alpha=0.65,
                    color=class_color(index),
                    edgecolor="none",
                    label=model,
                )
            limit = float(max(both["impurity"].max(), both[PRIMARY_KIND].max()))
            right.plot([1, limit], [1, limit], linestyle="--", color="grey", linewidth=0.7)
            right.set_xlabel("impurity rank (measured on training rows)", fontsize=7)
            right.set_ylabel("permutation rank (held-out)", fontsize=7)
            right.set_title("The two measures disagree, and that is the point", fontsize=8)
            right.legend(fontsize=5)
        else:  # pragma: no cover - only when one measure is absent
            right.axis("off")

        fig.subplots_adjust(wspace=0.28, top=0.84, bottom=0.16)
        fig.suptitle("Feature importance: families, and the two measures", y=0.97)
        return fig

    spec = GraphSpec(
        figure_id="G18",
        title="Feature Importance Plot",
        caption=(
            "Left: each feature family's share of the positive permutation "
            "importance, per model, averaged over the five folds of a complete "
            "partition of the corpus with SD whiskers. Families are in registry "
            "order, the same order the feature vector uses. The family total is "
            "the stable quantity: individual importances of correlated features "
            "-- 39 MFCC columns, 24 wavelet columns -- are close to arbitrary "
            "between members. Right: every feature's impurity rank against its "
            "permutation rank. Points off the diagonal are features the training "
            "rows and the held-out rows disagree about, which is exactly what "
            "impurity importance is reported here to expose."
        ),
        sources=sources,
        exp_id="n/a (fitted for this analysis, config defaults)",
        objective="O2 (feature contribution)",
        dataset="D1 PhysioNet 2016",
        command=command,
        size="wide",
    )
    return Graph(spec=spec, frame=family, draw=draw)


def build_g19(
    top: Any, sources: tuple[str, ...], command: str = "", model_id: str = ""
) -> Graph:
    """G19 -- the top-20 chart for one model, with the across-fold spread."""
    focus = model_id or str(top["model_id"].iloc[0])

    def draw(data: Any) -> Any:
        block = data[data["model_id"] == focus].sort_values("rank")
        ordered = block.iloc[::-1]
        positions = np.arange(len(ordered))
        means = ordered["importance_mean"].to_numpy(dtype=float)
        lows = ordered["importance_min"].to_numpy(dtype=float)
        highs = ordered["importance_max"].to_numpy(dtype=float)
        crosses_zero = lows <= 0

        fig, axis = subplots((7.2, 4.6))
        axis.barh(
            positions,
            means,
            xerr=np.vstack(
                [np.clip(means - lows, 0, None), np.clip(highs - means, 0, None)]
            ),
            color=[
                class_color(1) if flag else class_color(0) for flag in crosses_zero
            ],
            edgecolor="black",
            linewidth=0.3,
            error_kw={"elinewidth": 0.6, "capsize": 1.5},
        )
        axis.axvline(0.0, color="black", linewidth=0.6)
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [
                str(row["feature"]) + "  (" + str(int(row["rank"])) + ")"
                for _, row in ordered.iterrows()
            ],
            fontsize=5.5,
        )
        axis.set_xlabel("permutation importance -- drop in balanced accuracy when shuffled")
        axis.tick_params(labelsize=6)
        axis.set_title(
            focus + ": top " + str(len(ordered)) + " features. Whiskers span the "
            "observed range across the five folds; an amber bar's range reaches "
            "zero or below.",
            fontsize=6.5,
        )
        fig.subplots_adjust(left=0.34, top=0.90, bottom=0.12)
        return fig

    spec = GraphSpec(
        figure_id="G19",
        title="Top 20 Features Chart",
        caption=(
            "The 20 highest-ranked features for "
            + focus
            + " under permutation importance on held-out rows, ranked by the mean "
            "over the five folds of a complete partition of the corpus. The "
            "whisker spans the observed minimum and maximum across those folds "
            "rather than a standard deviation, because the quantity is not "
            "symmetric. A bar drawn in amber has a range that reaches zero or "
            "below: at least one fold found the feature useless or actively "
            "misleading, so its rank is not something to build a claim on. The "
            "number in brackets is the rank."
        ),
        sources=sources,
        exp_id="n/a (fitted for this analysis, config defaults)",
        objective="O2 (feature contribution)",
        dataset="D1 PhysioNet 2016",
        command=command,
    )
    return Graph(spec=spec, frame=top, draw=draw)
