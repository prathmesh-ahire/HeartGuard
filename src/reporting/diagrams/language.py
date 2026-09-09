"""The one visual language every PV-MEPCG diagram is drawn in (T97.4, T97.5).

Twenty diagrams end up interleaved with thirty-five graphs in a thesis. If a
"process" is a blue rounded box in F02 and a green rectangle in F11, the reader
spends their attention decoding the drawing instead of reading it. So the
vocabulary is declared here, once, and the canvas can only draw what is in it.

**Meaning is never carried by colour alone.** Every node kind has its own
*shape*, and every edge kind its own *dash pattern*, before either has a colour.
This is not a preference: the thesis is printed, examiners photocopy pages, and
about one man in twelve cannot separate the red-green pair that most pipeline
diagrams lean on. Colour here is a second, redundant channel over a shape
distinction that already works in black and white.

**Greyscale legibility is a computation, not an opinion (T97.5).** WCAG relative
luminance is a weighted sum of the three channels with no hue term in it, so a
contrast ratio computed from it *is* the greyscale contrast: a text/fill pair
that clears 4.5:1 here clears it on a monochrome laser printer by construction.
:func:`palette_report` runs that check over every kind in the vocabulary, and a
test fails the build if any pair drops below the threshold.

**Legibility is checked at printed size, not at screen size.** A 9 pt label on a
9-inch-wide figure squeezed into a 6-inch thesis column prints at 6 pt. The
canvas therefore records the point size of every string it draws, and
:func:`effective_point_size` converts it to what the page will actually show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.reporting.plot_style import OKABE_ITO

__all__ = [
    "THESIS_WIDTH_INCHES",
    "MIN_EFFECTIVE_PT",
    "MIN_TEXT_CONTRAST",
    "MIN_KIND_SEPARATION",
    "NodeStyle",
    "EdgeStyle",
    "NODE_STYLES",
    "EDGE_STYLES",
    "INK",
    "PAPER",
    "MUTED",
    "tint",
    "relative_luminance",
    "contrast_ratio",
    "readable_text_color",
    "effective_point_size",
    "palette_report",
]

#: Text width of the thesis page. Diagrams are drawn at or above this width and
#: scaled down to it, never up, so nothing is checked at a size it never prints.
THESIS_WIDTH_INCHES = 6.0

#: Smallest point size any diagram string may print at. Below roughly 7 pt a
#: serif face loses its thin strokes on a 300 dpi laser print.
MIN_EFFECTIVE_PT = 7.0

#: WCAG AA for normal text. Applied to label-on-fill for every node kind.
MIN_TEXT_CONTRAST = 4.5

#: Two node fills this close in luminance are the same grey on paper. Shape
#: already separates the kinds, so this is a warning threshold in general --
#: except between kinds sharing a shape, where it is the only separation left.
MIN_KIND_SEPARATION = 1.15

#: Line and text colour. Pure black on white; every diagram here is line art.
INK = "#000000"
PAPER = "#FFFFFF"
MUTED = "#555555"


def relative_luminance(color: Any) -> float:
    """WCAG 2.x relative luminance of a matplotlib colour, in [0, 1].

    This is also the greyscale value the colour prints as, which is why the
    contrast check below needs no separate monochrome pass.
    """
    from matplotlib.colors import to_rgb

    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = to_rgb(color)
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def contrast_ratio(first: Any, second: Any) -> float:
    """WCAG contrast ratio between two colours, from 1.0 (identical) to 21.0."""
    a, b = relative_luminance(first), relative_luminance(second)
    lighter, darker = (a, b) if a >= b else (b, a)
    return (lighter + 0.05) / (darker + 0.05)


def tint(color: Any, amount: float) -> str:
    """Mix ``color`` with white. ``amount`` is how much of the colour survives.

    Fills are tints rather than the saturated palette entries so that black
    label text stays readable on them; the saturated colour is kept for the
    node border, where it still identifies the kind at a glance.
    """
    from matplotlib.colors import to_hex, to_rgb

    red, green, blue = to_rgb(color)
    mixed = (
        red * amount + (1.0 - amount),
        green * amount + (1.0 - amount),
        blue * amount + (1.0 - amount),
    )
    return str(to_hex(mixed))


def readable_text_color(background: Any) -> str:
    """Black on the fill unless black fails AA there, in which case white."""
    return INK if contrast_ratio(INK, background) >= MIN_TEXT_CONTRAST else PAPER


@dataclass(frozen=True)
class NodeStyle:
    """One box kind: what it means, what shape says so, and how it is filled."""

    kind: str
    #: What this kind stands for. Rendered into the legend, so it is the caption
    #: the reader actually sees -- keep it to a few words.
    meaning: str
    #: One of ``rect``, ``round``, ``parallelogram``, ``diamond``, ``folded``,
    #: ``stadium``, ``double``, ``plain``. The primary, colour-free channel.
    shape: str
    facecolor: str
    edgecolor: str
    linewidth: float = 1.0
    linestyle: str = "-"

    @property
    def textcolor(self) -> str:
        return readable_text_color(self.facecolor)


@dataclass(frozen=True)
class EdgeStyle:
    """One arrow kind. The dash pattern carries the meaning; colour repeats it."""

    kind: str
    meaning: str
    linestyle: Any
    color: str = INK
    linewidth: float = 1.1
    #: matplotlib arrowstyle. A filled head is a real transfer, an open head a
    #: derivation, and a headless line a relationship that does not flow.
    arrowstyle: str = "-|>"


NODE_STYLES: dict[str, NodeStyle] = {
    # A corpus or a file read from disk. Skewed, the way input has been drawn in
    # flowcharts since ISO 5807, so it needs no legend lookup.
    "input": NodeStyle(
        "input", "dataset / file read", "parallelogram", tint(OKABE_ITO[5], 0.30), OKABE_ITO[0]
    ),
    # Computation. The default, and deliberately the plainest thing on the page.
    "process": NodeStyle("process", "processing step", "round", PAPER, INK),
    # A fitted estimator. Doubled border: it holds learned state, unlike a step.
    "model": NodeStyle(
        "model", "trained model", "double", tint(OKABE_ITO[4], 0.30), OKABE_ITO[4], linewidth=1.2
    ),
    # A branch.
    "decision": NodeStyle(
        "decision", "decision / branch", "diamond", tint(OKABE_ITO[3], 0.25), OKABE_ITO[3]
    ),
    # A deliverable written under outputs/. Folded corner = a document on disk.
    "artifact": NodeStyle(
        "artifact", "artifact written to outputs/", "folded", tint(OKABE_ITO[2], 0.28), OKABE_ITO[2]
    ),
    # Persisted state that is read back later: cache/, models_saved/, generated/.
    "store": NodeStyle("store", "persisted store", "stadium", tint(OKABE_ITO[6], 0.45), MUTED),
    # A caption on the drawing itself. No border, so it never reads as a step.
    "note": NodeStyle("note", "annotation", "plain", PAPER, PAPER, linewidth=0.0),
}

EDGE_STYLES: dict[str, EdgeStyle] = {
    # Something moves: a signal, a matrix, a request.
    "flow": EdgeStyle("flow", "data or control flow", "-"),
    # B is generated from A ahead of time. This dashed line is the build-time
    # codegen boundary F16 has to make visible.
    "derive": EdgeStyle("derive", "generated from (build time)", (0, (5, 2)), MUTED, 1.0, "-|>"),
    # A loop: search, refit, iterate.
    "feedback": EdgeStyle(
        "feedback", "iteration / feedback", (0, (1, 2)), OKABE_ITO[0], 1.0, "-|>"
    ),
    # A path that deliberately does not exist. Drawn because "the test fold never
    # reaches the scaler" is a claim about an absence, and an absence cannot be
    # shown by leaving the arrow out.
    "forbidden": EdgeStyle(
        "forbidden", "never crosses (fold safety)", (0, (4, 2)), OKABE_ITO[1], 1.3, "-"
    ),
    # Structural containment or reference, not a transfer.
    "link": EdgeStyle("link", "references", (0, (1, 3)), MUTED, 0.9, "-"),
}


def effective_point_size(points: float, figure_width_inches: float) -> float:
    """The size ``points`` prints at once the figure is scaled to the page.

    Scaling is one-directional: a figure narrower than the text block is placed
    at its own size rather than stretched, so it never gains legibility here.
    """
    scale = min(1.0, THESIS_WIDTH_INCHES / float(figure_width_inches))
    return float(points) * scale


def palette_report() -> dict[str, Any]:
    """Contrast measurements for the whole vocabulary, with the failures named.

    Returned rather than asserted so the numbers can be printed by the renderer
    and pinned by a test, instead of living only inside a passing assertion.
    """
    text_rows: list[dict[str, Any]] = []
    for style in NODE_STYLES.values():
        if style.shape == "plain":  # no fill, so no fill contrast to measure
            continue
        text_rows.append(
            {
                "kind": style.kind,
                "fill": style.facecolor,
                "text": style.textcolor,
                "text_contrast": round(contrast_ratio(style.textcolor, style.facecolor), 3),
                "border_contrast": round(contrast_ratio(style.edgecolor, style.facecolor), 3),
                "fill_luminance": round(relative_luminance(style.facecolor), 4),
            }
        )

    kinds = [s for s in NODE_STYLES.values() if s.shape != "plain"]
    pairs: list[dict[str, Any]] = []
    for index, first in enumerate(kinds):
        for second in kinds[index + 1 :]:
            pairs.append(
                {
                    "pair": first.kind + " / " + second.kind,
                    "same_shape": first.shape == second.shape,
                    "fill_contrast": round(contrast_ratio(first.facecolor, second.facecolor), 3),
                }
            )

    failures = [
        str(row["kind"]) for row in text_rows if float(row["text_contrast"]) < MIN_TEXT_CONTRAST
    ]
    failures += [
        str(pair["pair"])
        for pair in pairs
        if pair["same_shape"] and float(pair["fill_contrast"]) < MIN_KIND_SEPARATION
    ]
    return {
        "min_text_contrast": MIN_TEXT_CONTRAST,
        "min_kind_separation": MIN_KIND_SEPARATION,
        "text": text_rows,
        "pairs": pairs,
        "shapes_unique": len({s.shape for s in kinds}) == len(kinds),
        "edge_linestyles_unique": (
            len({str(e.linestyle) for e in EDGE_STYLES.values()}) == len(EDGE_STYLES)
        ),
        "failures": failures,
    }
