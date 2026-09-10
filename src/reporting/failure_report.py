"""T27, G35 and the narrative failure report (Phase 80).

Three artifacts over one question: **which recordings does the model get wrong,
and what do they have in common?**

``T27`` is the categorised table -- error, false-positive and false-negative
rates within every level of every grouping, for both the in-domain and the
external error set, with the denominator beside every rate.

``G35`` draws the same numbers as six panels, because a table of forty rates is
not something anyone reads.

``failure_report.md`` is T80.6's narrative, and it **names records**. Every
number in it is formatted from the frames rather than typed, and the records are
chosen by a stated rule (:func:`~src.evaluation.failure_analysis.example_records`)
so the document regenerates identically.

## Rates, and the two denominators that are not interchangeable

`error_rate` is over all predictions in the group. `false_negative_rate` is over
that group's **positive** records only, `false_positive_rate` over its negatives.
The three move independently and quoting the wrong one is the usual way a
subgroup finding turns out to be a class-balance finding: a group that is 90%
normal will have a low error rate however badly it misses abnormals.

For a screening prototype the false-negative rate is the one that matters, so it
leads in every panel and is stated first in the narrative.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.failure_analysis import MIN_GROUP
from src.reporting.graphs import Graph, GraphSpec, subplots
from src.reporting.plot_style import class_color
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "PANEL_CATEGORIES",
    "build_t27",
    "build_g35",
    "write_failure_report",
]

log = get_logger("reporting.failure")

#: The five groupings G35 draws, in panel order. The sixth panel is the
#: multiclass confused-pair ranking.
PANEL_CATEGORIES: tuple[str, ...] = (
    "duration_band",
    "noise_flag",
    "subset",
    "confidence_band",
    "diagnosis_class",
)

_DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support "
    "prototype, not a diagnostic tool. A named record here is a recording the "
    "model classified incorrectly, not a clinical finding about a person."
)


def build_t27(
    categorised: Any, pairs: Any, sources: tuple[str, ...], command: str = ""
) -> Table:
    """T27 -- false positives and false negatives, categorised, with denominators."""
    frame = categorised.sort_values(["source", "category", "level"]).reset_index(drop=True)

    columns = (
        Column("source", "Error set"),
        Column("category", "Grouping"),
        Column("level", "Level"),
        Column("n_predictions", "Predictions", kind="count"),
        Column("n_records", "Records", kind="count"),
        Column("n_positive", "Abnormal predictions", kind="count"),
        Column("n_negative", "Normal predictions", kind="count"),
        Column("n_positive_records", "Abnormal records", kind="count"),
        Column("n_negative_records", "Normal records", kind="count"),
        Column("n_fp", "FP", kind="count"),
        Column("n_fn", "FN", kind="count"),
        Column("n_errors", "Errors", kind="count"),
        Column("error_rate", "Error rate", kind="metric"),
        Column("false_negative_rate", "FN rate (of abnormal)", kind="metric"),
        Column("false_positive_rate", "FP rate (of normal)", kind="metric"),
        Column("mean_confidence_when_wrong", "Confidence when wrong", kind="metric"),
        Column("below_reporting_floor", "Group below floor"),
        Column("fn_below_reporting_floor", "FN rate below floor"),
        Column("fp_below_reporting_floor", "FP rate below floor"),
    )

    top = pairs.nlargest(1, "n_confused") if pairs is not None and len(pairs) else None
    pair_note = (
        "T80.4 -- the most-confused pair across every stored multiclass "
        "confusion matrix is "
        + str(top["pair"].iloc[0])
        + " on "
        + str(top["run"].iloc[0])
        + " "
        + str(top["model_id"].iloc[0])
        + " ("
        + format(float(top["n_confused"].iloc[0]), ".0f")
        + " summed over folds, "
        + format(float(top["share_of_true_class"].iloc[0]) * 100, ".1f")
        + "% of that class). Every pair is in confused_class_pairs.csv. Read "
        "share_of_true_class, not the count: 40 confusions out of 320 records "
        "and 40 out of 46 are different findings."
        if top is not None
        else "No multiclass confusion matrix was available for the pair analysis."
    )

    spec = TableSpec(
        table_id="T27",
        title="False Positive and False Negative Analysis",
        caption=(
            "Every error of the final binary model, grouped four ways plus by "
            "PhysioNet diagnosis. Two error sets, never pooled: the in-domain set "
            "is EXP-A2's out-of-fold predictions over 25 repeated grouped folds, "
            "and the external set is the deployed model scoring CirCor once. "
            "Rates carry their denominators because they use different ones -- "
            "error rate is over all predictions in the group, FN rate over that "
            "group's abnormal records only, FP rate over its normal ones."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-A2, EXP-D1",
        objective="O4 (robustness), O5 (error characterisation)",
        dataset="D1 PhysioNet 2016, D4 CirCor 2022",
        notes=(
            "THE FALSE-NEGATIVE RATE IS THE ONE THAT MATTERS HERE. A missed "
            "abnormal recording is the failure a screening tool is judged on, and "
            "a group that is mostly normal can post a low error rate while "
            "missing most of its abnormals.",
            "THE FLOOR IS APPLIED PER RATE, NOT PER GROUP, AND IT HAS TO BE. "
            "Each rate has its own denominator: the FN rate's is the group's "
            "abnormal records and the FP rate's is its normal ones, so a group "
            "can clear the "
            + str(MIN_GROUP)
            + "-record floor overall while one of its rates rests on a handful. "
            "PhysioNet training-c is the case that forced this: 31 records, above "
            "the floor, of which SEVEN are normal -- and its false-positive rate "
            "of 1.000 is a statement about those seven. Read 'FN rate below "
            "floor' and 'FP rate below floor' per rate; 'Group below floor' "
            "covers the error rate. Flagged rows are reported rather than "
            "dropped, because a suppressed group reads as an absence of "
            "failures.",
            "IN-DOMAIN COUNTS ARE OVER PREDICTIONS, NOT RECORDS. The repeated 5x5 "
            "map holds every record out once per repeat, so each is scored five "
            "times and the prediction counts are five times the corpus. "
            "record_failures.csv carries the per-record view, where a record "
            "wrong in 5 of 5 (five independently fitted models agreeing) is "
            "distinguished from one wrong in 1 of 5.",
            "THE TWO ERROR SETS ARE NOT COMPARABLE. EXP-D1 is adult-to-paediatric "
            "transfer whose population mismatch is documented and expected; its "
            "error rate is not evidence about the method.",
            "The diagnosis grouping is a DESCRIPTION, not a subgroup result. "
            "PhysioNet's diagnosis field is very unevenly filled and most of its "
            "categories fall below the reporting floor.",
            pair_note,
            _DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)


def build_g35(
    categorised: Any, pairs: Any, sources: tuple[str, ...], command: str = "", source: str = ""
) -> Graph:
    """G35 -- the failure distribution: five groupings plus the confused pairs."""
    focus = source or str(categorised["source"].iloc[0])

    def draw(data: Any) -> Any:
        block = data[data["source"] == focus]
        fig, axes = subplots((10.0, 5.6), ncols=3, nrows=2, squeeze=False)

        for position, category in enumerate(PANEL_CATEGORIES):
            axis = axes[position // 3][position % 3]
            part = block[block["category"] == category].copy()
            if not len(part):
                axis.axis("off")
                continue
            # Ordered by how much is at stake, not alphabetically, except for
            # the two groupings whose levels have a natural order.
            if category in ("duration_band", "confidence_band"):
                part = part.sort_values("level")
            else:
                part = part.sort_values("n_records", ascending=False).head(8)

            positions = np.arange(len(part))
            # Each bar is hatched against ITS OWN denominator's floor. A shared
            # flag would have left PhysioNet training-c's FP rate of 1.000 --
            # seven normal records -- looking like a measured result.
            axis.bar(
                positions - 0.2,
                part["false_negative_rate"].to_numpy(dtype=float),
                width=0.4,
                color=class_color(3),
                edgecolor="black",
                linewidth=0.3,
                hatch=["//" if flag else "" for flag in part["fn_below_reporting_floor"]],
                label="FN rate (of abnormal)",
            )
            axis.bar(
                positions + 0.2,
                part["false_positive_rate"].to_numpy(dtype=float),
                width=0.4,
                color=class_color(0),
                edgecolor="black",
                linewidth=0.3,
                hatch=["//" if flag else "" for flag in part["fp_below_reporting_floor"]],
                label="FP rate (of normal)",
            )
            axis.set_xticks(positions)
            axis.set_xticklabels(part["level"], rotation=35, ha="right", fontsize=5)
            axis.set_ylim(0, 1)
            axis.tick_params(labelsize=5)
            axis.set_title(category.replace("_", " "), fontsize=7)
            if position == 0:
                from matplotlib.patches import Patch

                axis.set_ylabel("rate", fontsize=7)
                # The hatch is explained in the legend rather than in a caption
                # line across the top of the figure: at this panel count that
                # line runs straight through the second and third panel titles.
                handles, labels = axis.get_legend_handles_labels()
                handles.append(
                    Patch(facecolor="white", edgecolor="black", hatch="//", linewidth=0.3)
                )
                labels.append("denominator below " + str(MIN_GROUP) + " records")
                axis.legend(handles, labels, fontsize=5)

        axis = axes[1][2]
        if pairs is not None and len(pairs):
            top = pairs.nlargest(8, "share_of_true_class").iloc[::-1]
            labels = [
                str(row["run"]).replace("EXP-", "") + " " + str(row["model_id"]) + ": "
                + str(row["pair"])
                for _, row in top.iterrows()
            ]
            axis.barh(
                np.arange(len(top)),
                top["share_of_true_class"].to_numpy(dtype=float),
                color=class_color(2),
                edgecolor="black",
                linewidth=0.3,
            )
            axis.set_yticks(np.arange(len(top)))
            axis.set_yticklabels(labels, fontsize=4.5)
            axis.set_xlim(0, 1)
            axis.tick_params(labelsize=5)
            axis.set_title("multiclass: worst confused pairs", fontsize=7)
            axis.set_xlabel("share of the true class", fontsize=6)
        else:  # pragma: no cover - only when no multiclass run exists
            axis.axis("off")

        fig.subplots_adjust(hspace=0.62, wspace=0.3, top=0.88, bottom=0.14)
        fig.suptitle(
            "Where the binary model fails: "
            + focus
            + " -- the two rates have different denominators",
            y=0.975,
        )
        return fig

    spec = GraphSpec(
        figure_id="G35",
        title="False Positive And False Negative Distribution",
        caption=(
            "Failures of the final binary model on "
            + focus
            + ", by duration band, PP-08 noise flag, PhysioNet sub-collection, "
            "predicted confidence and diagnosis, plus the worst confused class "
            "pairs from every stored multiclass confusion matrix. The two bars in "
            "each pair have DIFFERENT denominators: the false-negative rate is "
            "over that group's abnormal records and the false-positive rate over "
            "its normal ones, so they do not sum to the error rate. A bar is "
            "hatched when its own denominator holds fewer than "
            + str(MIN_GROUP)
            + " records -- PhysioNet training-c posts a false-positive rate of "
            "1.000 over seven normal recordings -- and hatched bars are shown for "
            "completeness rather than as results."
        ),
        sources=sources,
        exp_id="EXP-A2, EXP-D1",
        objective="O4 (robustness), O5 (error characterisation)",
        dataset="D1 PhysioNet 2016, D4 CirCor 2022",
        command=command,
        size="tall",
    )
    return Graph(spec=spec, frame=categorised, draw=draw)


def _rate_clause(rate: float, n_records: int, noun: str, wording: str, flagged: bool) -> str:
    """One rate in prose, or the reason it does not exist.

    A group with no records of a class has no rate for that class, and printing
    "nan%" there says the number failed to compute when in fact it was never
    defined. PhysioNet's diagnosis categories are all abnormal, so every one of
    them hits this.
    """
    if not n_records:
        return "it holds no " + noun + " recordings, so that rate is undefined"
    if not np.isfinite(rate):  # pragma: no cover - guarded by the branch above
        return "its " + wording + " rate could not be computed"
    return (
        format(rate * 100, ".1f")
        + "% "
        + wording
        + " (over "
        + format(n_records, ",d")
        + " records"
        + ("; below the reporting floor" if flagged else "")
        + ")"
    )


def _rate_line(row: Any) -> str:
    return (
        "`"
        + str(row["level"])
        + "` -- "
        + _rate_clause(
            float(row["false_negative_rate"]),
            int(row["n_positive_records"]),
            "abnormal",
            "of its abnormal recordings missed",
            bool(row["fn_below_reporting_floor"]),
        )
        + "; "
        + _rate_clause(
            float(row["false_positive_rate"]),
            int(row["n_negative_records"]),
            "normal",
            "of its normal recordings flagged",
            bool(row["fp_below_reporting_floor"]),
        )
    )


def write_failure_report(
    path: Any,
    *,
    categorised: Any,
    records: Any,
    examples: Any,
    pairs: Any,
    sources: tuple[str, ...],
    command: str = "",
) -> Any:
    """T80.6 -- the narrative, with every number formatted from the frames.

    Nothing in this document is typed. The record ids come from
    :func:`example_records`' stated selection rule, and the rates are read out
    of the frames T27 was built from, so re-running the analysis rewrites the
    prose to match rather than leaving it describing an older run.
    """
    from pathlib import Path

    from src.utils.io import ensure_dir

    target = Path(path)
    ensure_dir(target.parent)
    lines: list[str] = []

    lines.append("# Failure analysis -- where the binary model gets it wrong")
    lines.append("")
    lines.append(
        "Generated by `" + command + "`. Every number below is read from the "
        "frames named at the end; none is typed. " + _DISCLAIMER
    )
    lines.append("")

    for source in sorted(set(categorised["source"])):
        block = categorised[categorised["source"] == source]
        source_records = records[records["source"] == source]
        lines.append("## " + source)
        lines.append("")

        n_records = len(source_records)
        n_wrong = int((source_records["n_wrong"] > 0).sum())
        n_consistent = int(source_records["consistently_wrong"].sum())
        n_evals = int(source_records["n_evaluations"].max())
        lines.append(
            "Of **" + format(n_records, ",d") + " records**, "
            + format(n_wrong, ",d")
            + " ("
            + format(100.0 * n_wrong / n_records, ".1f")
            + "%) were misclassified at least once and **"
            + format(n_consistent, ",d")
            + " ("
            + format(100.0 * n_consistent / n_records, ".1f")
            + "%) were misclassified on every one of their "
            + str(n_evals)
            + " evaluation(s)**."
            + (
                " A record wrong on all of them was wrong under "
                + str(n_evals)
                + " separately fitted models, which makes it a property of the "
                "recording rather than of one fold's fit."
                if n_evals > 1
                else ""
            )
        )
        lines.append("")

        for category in PANEL_CATEGORIES:
            part = block[block["category"] == category]
            if not len(part):
                continue
            reportable = part[~part["fn_below_reporting_floor"]]
            pool = reportable if len(reportable) >= 2 else part
            worst = pool.loc[pool["false_negative_rate"].idxmax()]
            best = pool.loc[pool["false_negative_rate"].idxmin()]
            lines.append("**By " + category.replace("_", " ") + ".** ")
            lines.append("Worst: " + _rate_line(worst) + ".")
            lines.append("")
            lines.append("Best: " + _rate_line(best) + ".")
            lines.append("")
            ratio = float(worst["false_negative_rate"]) / max(
                float(best["false_negative_rate"]), 1e-9
            )
            if np.isfinite(ratio) and ratio > 1.5:
                lines.append(
                    "That is a **"
                    + format(ratio, ".1f")
                    + "x** difference in missed abnormals between the worst and "
                    "best level of this grouping."
                )
                lines.append("")

        picks = examples[examples["source"] == source]
        if len(picks):
            lines.append("### Named examples")
            lines.append("")
            lines.append(
                "Chosen by rule, not by eye: among the records this model got "
                "wrong on every evaluation, the ones it was most confident about. "
                "Confidently and repeatably wrong is the worst case a screening "
                "prototype has."
            )
            lines.append("")
            lines.append(
                "| record | true class | error | wrong / evaluations | mean confidence when wrong "
                "| duration (s) | quality | diagnosis |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
            for _, row in picks.iterrows():
                lines.append(
                    "| `"
                    + str(row["record_uid"])
                    + "` | "
                    + str(row.get("binary_label_name", ""))
                    + " | "
                    + str(row["error_type"])
                    + " | "
                    + str(int(row["n_wrong"]))
                    + " / "
                    + str(int(row["n_evaluations"]))
                    + " | "
                    + format(float(row["mean_confidence_when_wrong"]), ".3f")
                    + " | "
                    + format(float(row["duration_sec"]), ".1f")
                    + " | "
                    + str(row.get("noise_flag", ""))
                    + " | "
                    + (
                        str(row.get("diagnosis_class"))
                        if str(row.get("diagnosis_class")) not in ("nan", "None")
                        else "-"
                    )
                    + " |"
                )
            lines.append("")

    if pairs is not None and len(pairs):
        lines.append("## Most-confused class pairs (T80.4)")
        lines.append("")
        lines.append(
            "Off-diagonal cells of every stored multiclass confusion matrix, "
            "ranked by the share of the true class rather than by the raw count. "
            "A class whose recordings go almost entirely to another class is a "
            "model that has stopped predicting it."
        )
        lines.append("")
        lines.append("| run | model | pair | confusions | share of the true class |")
        lines.append("|---|---|---|---|---|")
        for _, row in pairs.nlargest(12, "share_of_true_class").iterrows():
            lines.append(
                "| "
                + str(row["run"])
                + " | "
                + str(row["model_id"])
                + " | "
                + str(row["pair"])
                + " | "
                + format(float(row["n_confused"]), ".0f")
                + " | "
                + format(float(row["share_of_true_class"]) * 100, ".1f")
                + "% |"
            )
        lines.append("")
        lines.append(
            "A share at or near 100% means the model never predicted that class "
            "at all on that run. Counts are summed over folds, so on a repeated "
            "map their support is the corpus times the number of repeats."
        )
        lines.append("")

    lines.append("## Sources")
    lines.append("")
    for item in sources:
        lines.append("- `" + str(item) + "`")
    lines.append("")

    target.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    log.info("wrote %d line(s) -> %s", len(lines), target)
    return target
