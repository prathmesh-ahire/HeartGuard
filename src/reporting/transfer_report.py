"""T16 and G29 -- adult-to-paediatric transfer, reported honestly (Phase 71).

Both deliverables here carry the same obligation: **the population mismatch is
stated on the artifact itself, not left in a paragraph somewhere else.** The
caption of T16 and the subtitle of G29 both say what was transferred to what,
because a table of numbers travels into a slide deck without its surrounding
prose and a drop of this size is trivially misread as a method failure.

The comparison is deliberately asymmetric and labelled as such: the in-domain
side is EXP-A2's 25-fold nested cross-validation, the external side is one
evaluation over the whole of CirCor. Subtracting them is meaningful; pretending
they carry the same uncertainty is not, so the SD column exists on one side only
and the ``n_folds`` columns are printed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.reporting.graphs import Graph, GraphSpec, subplots
from src.reporting.plot_style import class_color
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "TRANSFER_HEADLINE",
    "build_t16",
    "build_g29",
    "build_transfer_frame",
]

log = get_logger("reporting.transfer")

#: The one sentence that must accompany every EXP-D1 number, anywhere.
TRANSFER_HEADLINE = (
    "Cross-dataset transfer from an adult cohort (PhysioNet 2016, median age 25, "
    "0.27% of aged records under 18) to a predominantly paediatric cohort "
    "(CirCor 2022, 98.2% of age-banded patients paediatric). A large drop is the "
    "expected consequence of that mismatch and is NOT evidence that PV-MEPCG "
    "fails to generalize."
)

_METRICS = ("sensitivity", "specificity", "balanced_accuracy", "roc_auc", "accuracy", "f1")


def build_transfer_frame(external: Any, drops: Any) -> Any:
    """One row per (level, rule, metric): external value, in-domain mean, delta."""
    import pandas as pd

    merged = drops.copy()
    counts = external.set_index(["level", "rule"])["n_units"].to_dict()
    merged["n_units"] = [
        int(counts.get((row["level"], row["rule"]), 0)) for _, row in merged.iterrows()
    ]
    order = {metric: index for index, metric in enumerate(_METRICS)}
    merged["_order"] = merged["metric"].map(order).fillna(len(order))
    merged = merged.sort_values(["level", "rule", "_order"]).drop(columns="_order")
    return pd.DataFrame(merged).reset_index(drop=True)


def build_t16(frame: Any, sources: tuple[str, ...], command: str = "") -> Table:
    """T16 -- cross-dataset generalization, with the framing in the caption."""
    columns = (
        Column("metric", "Metric"),
        Column("level", "Level"),
        Column("rule", "Aggregation"),
        Column("n_units", "Units", kind="count"),
        Column("in_domain_exp", "In-domain run"),
        Column("in_domain_mean", "In-domain (PhysioNet)", kind="metric"),
        Column("in_domain_n_folds", "In-domain folds", kind="count"),
        Column("external_value", "External (CirCor)", kind="metric"),
        Column("external_n_folds", "External folds", kind="count"),
        Column("delta", "Delta", kind="metric"),
        Column("relative_drop", "Relative drop", kind="metric"),
    )
    spec = TableSpec(
        table_id="T16",
        title="Cross Dataset Generalization",
        caption=(
            TRANSFER_HEADLINE
            + " The model is the finalized binary model (M1), fitted on all 3,240 "
            "labelled PhysioNet recordings and applied to all 3,163 CirCor "
            "recordings with NO retuning of any kind. In-domain figures are "
            "EXP-A2's 25-fold nested cross-validation mean; the external figure "
            "is a single evaluation over the whole external corpus, so the two "
            "carry different uncertainty and the fold counts are printed."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-D1",
        objective="O5 (external validation)",
        dataset="D1 PhysioNet 2016 -> D4 CirCor 2022",
        notes=(
            "The population mismatch was recorded in population_mismatch.json "
            "BEFORE any prediction was made, so the framing cannot have been "
            "retrofitted to the result (T71.1, T71.7).",
            "A second, independent cause of the drop is on the record: "
            "PhysioNet's six sub-collections behave like six different datasets "
            "and all 20 of its top features by pooled Cohen's d reverse sign "
            "between sources. The drop must not be attributed wholly to age.",
            "CirCor Outcome is a clinical decision label, not an acoustic one. A "
            "model trained to hear an abnormal heart sound is being asked a "
            "different question here, which is a third reason the two numbers "
            "are not like for like.",
            "PV-MEPCG / PulseVision is an academic screening prototype, not a "
            "diagnostic tool.",
        ),
        command=command,
    )
    return build_table(spec, frame)


def build_g29(frame: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G29 -- the drop, drawn as paired bars with the framing on the figure."""
    plotted = frame[frame["level"] == "recording"].copy()
    plotted = plotted[plotted["metric"].isin(_METRICS)].reset_index(drop=True)

    def draw(data: Any) -> Any:
        metrics = [m for m in _METRICS if m in set(data["metric"])]
        indexed = data.set_index("metric")
        fig, axes = subplots("double", ncols=1, nrows=1)
        axis = np.atleast_1d(axes)[0]
        positions = np.arange(len(metrics))
        width = 0.38

        in_domain = [float(indexed.loc[m, "in_domain_mean"]) for m in metrics]
        external = [float(indexed.loc[m, "external_value"]) for m in metrics]
        axis.bar(
            positions - width / 2,
            in_domain,
            width=width,
            color=class_color(0),
            edgecolor="black",
            linewidth=0.4,
            label="In domain: PhysioNet, EXP-A2 25-fold nested CV",
        )
        axis.bar(
            positions + width / 2,
            external,
            width=width,
            color=class_color(1),
            edgecolor="black",
            linewidth=0.4,
            hatch="//",
            label="External: CirCor, single evaluation, no retuning",
        )
        for index, (before, after) in enumerate(zip(in_domain, external, strict=True)):
            axis.text(
                index,
                max(before, after) + 0.03,
                format(after - before, "+.3f"),
                ha="center",
                fontsize=7,
            )
        axis.axhline(0.5, color="grey", linewidth=0.6, linestyle="--")
        axis.set_xticks(positions)
        axis.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=20, ha="right")
        axis.set_ylim(0, 1.28)
        axis.set_ylabel("score")
        axis.legend(fontsize=6, loc="upper right", frameon=True)
        fig.suptitle("Adult-to-paediatric transfer: PhysioNet 2016 -> CirCor 2022")
        fig.text(
            0.5,
            0.90,
            "A drop of this size is the expected population effect, NOT a failure "
            "to generalize. The dashed line is chance.",
            fontsize=6,
            ha="center",
        )
        return fig

    spec = GraphSpec(
        figure_id="G29",
        title="Cross Dataset Performance Drop",
        caption=(
            TRANSFER_HEADLINE
            + " Recording-level metrics. The in-domain bar is EXP-A2's 25-fold "
            "nested cross-validation mean for the same model; the external bar "
            "is one evaluation over all 3,163 CirCor recordings."
        ),
        sources=sources,
        exp_id="EXP-D1",
        objective="O5 (external validation)",
        dataset="D1 PhysioNet 2016 -> D4 CirCor 2022",
        notes=(
            "Patient-level aggregation is deliberately not drawn here: it moves "
            "the operating point rather than recovering performance, and putting "
            "it beside the in-domain bar would invite reading it as a fix. It is "
            "in T16.",
        ),
        command=command,
    )
    return Graph(spec=spec, frame=plotted, draw=draw)
