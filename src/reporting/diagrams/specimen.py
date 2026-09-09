"""A specimen sheet exercising the whole visual language (T97.4, T97.5, T97.7).

This is the pipeline's own test subject, not one of the twenty deliverables. It
places every node kind and draws every edge kind on one canvas, so that the gate
for Phase 97 can render a real diagram end to end -- both formats, the registry
row, the legibility measurement -- before any of F01-F20 exists.

It is also the reference sheet to look at when adding a diagram: what a
``forbidden`` edge looks like beside a ``flow`` one is a design decision that is
easier to check by eye on a single page than by reading ``language.py``.

It is written to a temporary directory by the test and by
``--specimen``; it never lands in ``outputs/``, because the thesis has twenty
diagrams in it and this is not one of them.
"""

from __future__ import annotations

import textwrap

from src.reporting.diagrams.canvas import DiagramCanvas
from src.reporting.diagrams.language import EDGE_STYLES, NODE_STYLES
from src.reporting.diagrams.render import DiagramSpec

__all__ = ["SPECIMEN_SPEC", "build_specimen"]

SPECIMEN_SPEC = DiagramSpec(
    "F00",
    "Diagram language specimen",
    "Every node shape and every arrow semantic in the PV-MEPCG diagram "
    "vocabulary, on one sheet. Not a thesis figure: the pipeline's own test "
    "subject.",
    task="T97.4",
    notes=(
        "Renders every kind declared in language.py, so a kind added without a "
        "shape or a legend meaning fails here first.",
    ),
)


def build_specimen() -> DiagramCanvas:
    """Draw one box of every kind, then one arrow of every kind between them."""
    canvas = DiagramCanvas(
        columns=12.0,
        rows=6.6,
        size=(9.0, 5.8),
        title="PV-MEPCG diagram language",
        subtitle="Shape carries the meaning; colour repeats it. Every arrow style is a "
        "different relationship.",
        legend_rows=4,
    )

    # Row 1: one box per node kind, in the order they are declared.
    drawable = [kind for kind, style in NODE_STYLES.items() if style.shape != "plain"]
    span = 12.0 / len(drawable)
    for index, kind in enumerate(drawable):
        canvas.node(
            kind,
            NODE_STYLES[kind].meaning,
            kind=kind,
            col=index * span + 0.15,
            row=0.35,
            width=span - 0.3,
            height=1.7,
            sublabel=kind,
        )

    # Row 2: one arrow per edge kind, each between its own pair of anchors, so
    # the dash patterns can be compared side by side at printed size.
    for index, (kind, style) in enumerate(EDGE_STYLES.items()):
        left = index * (12.0 / len(EDGE_STYLES)) + 0.2
        width = 12.0 / len(EDGE_STYLES) - 1.4
        canvas.node(
            "a_" + kind,
            "A",
            kind="process",
            col=left,
            row=2.6,
            width=width / 2.0,
            height=0.8,
        )
        canvas.node(
            "b_" + kind,
            "B",
            kind="process",
            col=left + width / 2.0 + 0.9,
            row=2.6,
            width=width / 2.0,
            height=0.8,
        )
        canvas.edge("a_" + kind, "b_" + kind, kind=kind, label=kind)
        canvas.text(
            "\n".join(textwrap.wrap(style.meaning, width=16)),
            col=left + width / 2.0 + 0.45,
            row=3.65,
            ha="center",
            va="top",
        )

    # A lane, a feedback loop back across it, and a free annotation: the three
    # remaining primitives a real diagram uses.
    canvas.lane("a stage boundary (dashed container)", col=0.15, row=4.7, width=11.7, height=1.55)
    canvas.node("loop_a", "search trial", kind="process", col=0.6, row=5.15, width=2.4, height=0.75)
    canvas.node("loop_b", "inner score", kind="process", col=4.0, row=5.15, width=2.4, height=0.75)
    canvas.node("loop_c", "chosen point", kind="store", col=7.4, row=5.15, width=2.4, height=0.75)
    canvas.edge("loop_a", "loop_b", kind="flow")
    canvas.edge("loop_b", "loop_c", kind="flow")
    canvas.edge("loop_b", "loop_a", kind="feedback", rad=0.45)
    canvas.text("annotation text", col=10.1, row=5.52, ha="left")
    return canvas
