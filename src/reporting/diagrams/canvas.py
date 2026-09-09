"""A box-and-arrow canvas over matplotlib (T97.1, T97.2, T97.3).

**Why matplotlib and not Graphviz or Mermaid.** Graphviz needs a ``dot`` binary
on PATH that this machine does not have and that a grader's machine will not
have either; Mermaid's CLI pulls a headless Chromium, the same dependency that
already left ``kaleido`` unusable here (see ``requirements/report.txt``).
matplotlib is pinned in ``requirements/base.txt``, already draws all thirty-five
graphs, renders SVG and 300 dpi PNG natively, and runs offline. The cost is that
layout is manual instead of automatic -- which for twenty diagrams that each
need a deliberate reading order is arguably the right trade anyway.

**The layout is a grid, and the grid is in the source.** A diagram places nodes
at ``(col, row)`` in grid units, and the grid is the only positioning primitive.
That keeps the twenty diagrams visually consistent without a layout engine, and
it keeps every diagram a small, diffable, version-controlled Python function
(T97.2) rather than a binary nobody can regenerate.

**Nothing is placed by ``tight_layout``.** The axes rectangle is computed in
inches and set once. A diagram whose canvas silently resizes to fit its content
prints at a different scale from the one beside it, and the point-size check in
:meth:`DiagramCanvas.legibility_report` would then be measuring a figure that no
longer exists.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

from src.reporting.diagrams.language import (
    EDGE_STYLES,
    INK,
    MIN_EFFECTIVE_PT,
    MUTED,
    NODE_STYLES,
    THESIS_WIDTH_INCHES,
    EdgeStyle,
    NodeStyle,
    effective_point_size,
)

__all__ = ["DIAGRAM_SIZES", "Node", "DiagramCanvas"]

#: Named page sizes, in inches. ``page`` is the thesis text block; ``wide`` is a
#: landscape figure that will be scaled down to it, which is why the label point
#: size below is derived from the width rather than fixed.
DIAGRAM_SIZES: dict[str, tuple[float, float]] = {
    "column": (3.4, 3.4),
    "page": (6.5, 4.2),
    "page_tall": (6.5, 7.8),
    "wide": (9.0, 5.2),
    "wide_tall": (9.0, 7.6),
}

#: Point size a node label prints at *on the page*, after any down-scaling.
#: Everything else on the canvas is a fixed multiple of it, so one number
#: controls the whole type scale and the legibility check has a single knob.
PRINTED_LABEL_PT = 8.5
_SUB_FACTOR = 0.88
_TITLE_FACTOR = 1.30

#: Reserved strips, in inches. Fixed rather than measured: see the module note
#: about tight_layout.
_TITLE_STRIP = 0.34
_SUBTITLE_STRIP = 0.20
_LEGEND_ROW = 0.22
_STAMP_STRIP = 0.34
_MARGIN = 0.06


@dataclass
class Node:
    """One placed box. Coordinates are grid units, y increasing downward."""

    key: str
    label: str
    kind: str
    col: float
    row: float
    width: float
    height: float
    sublabel: str = ""
    style: NodeStyle = field(default_factory=lambda: NODE_STYLES["process"])

    @property
    def center(self) -> tuple[float, float]:
        return (self.col + self.width / 2.0, self.row + self.height / 2.0)

    def anchor(self, side: str) -> tuple[float, float]:
        cx, cy = self.center
        if side == "left":
            return (self.col, cy)
        if side == "right":
            return (self.col + self.width, cy)
        if side == "top":
            return (cx, self.row)
        if side == "bottom":
            return (cx, self.row + self.height)
        raise ValueError("unknown anchor side: " + side)


class DiagramCanvas:
    """A grid of boxes and semantic arrows that renders to one matplotlib figure.

    Construct it inside the renderer's style context, place nodes and edges, then
    call :meth:`finish`. The figure exists from construction so that grid units
    can be converted to physical inches -- a rounded corner or an arrowhead has
    to be the same size in x and y, and the grid almost never is square.
    """

    def __init__(
        self,
        *,
        columns: float,
        rows: float,
        size: str | tuple[float, float] = "page",
        title: str = "",
        subtitle: str = "",
        legend_rows: int = 1,
    ) -> None:
        import matplotlib.pyplot as plt

        self.width_inches, self.height_inches = (
            DIAGRAM_SIZES[size] if isinstance(size, str) else size
        )
        self.columns = float(columns)
        self.rows = float(rows)
        self.title = title
        self.subtitle = subtitle
        self._legend_rows = legend_rows

        #: One knob for the whole type scale. Held constant *on the page*, so a
        #: wide figure that gets scaled down starts larger to compensate.
        self.label_pt = PRINTED_LABEL_PT * max(1.0, self.width_inches / THESIS_WIDTH_INCHES)
        self.sub_pt = self.label_pt * _SUB_FACTOR
        self.title_pt = self.label_pt * _TITLE_FACTOR

        self.figure = plt.figure(figsize=(self.width_inches, self.height_inches))
        self.figure.patch.set_facecolor("white")

        # A subtitle is a sentence, and a sentence does not fit one line at every
        # figure width. Wrapping it here rather than at draw time is what lets the
        # strip above the axes be reserved for the number of lines it will
        # actually take: reserving one line and drawing two lands the second on
        # top of the first row of boxes, and no clipping check catches that,
        # because it is inside the figure.
        self.title = self._wrap_to_width(title, self.title_pt)
        self.subtitle = self._wrap_to_width(subtitle, self.sub_pt)
        self._title_lines = len(self.title.splitlines()) if title else 0
        self._subtitle_lines = len(self.subtitle.splitlines()) if subtitle else 0

        top = _TITLE_STRIP * self._title_lines if title else _MARGIN
        top += _SUBTITLE_STRIP * self._subtitle_lines
        bottom = _STAMP_STRIP + (_LEGEND_ROW * legend_rows if legend_rows else 0.0)
        axes_height = self.height_inches - top - bottom
        if axes_height <= 0.5:
            raise ValueError("figure is too short for its reserved strips")

        self.axes = self.figure.add_axes(
            (
                _MARGIN / self.width_inches,
                bottom / self.height_inches,
                1.0 - 2.0 * _MARGIN / self.width_inches,
                axes_height / self.height_inches,
            )
        )
        self.axes.set_xlim(0.0, self.columns)
        self.axes.set_ylim(self.rows, 0.0)  # row 0 at the top, like a page
        self.axes.set_axis_off()

        axes_width_inches = self.width_inches - 2.0 * _MARGIN
        self._x_per_inch = self.columns / axes_width_inches
        self._y_per_inch = self.rows / axes_height

        self.nodes: dict[str, Node] = {}
        self._used_node_kinds: list[str] = []
        self._used_edge_kinds: list[str] = []
        #: ``(role, points)`` for every string drawn, for the legibility report.
        self._texts: list[tuple[str, float]] = []
        #: Node keys whose wrapped label needs more vertical room than the box
        #: gives it. Text that spills out of its shape is the single most common
        #: way a hand-laid-out diagram goes wrong, and it is invisible in the
        #: source, so it is measured here instead of noticed in the thesis.
        self._overflowing: list[str] = []
        #: ``(name, artist)`` for the things that can run off the page: the
        #: legend, the title, the stamp. Measured, not eyeballed -- see
        #: :meth:`legibility_report`.
        self._measured: list[tuple[str, Any]] = []

    # -- unit helpers ----------------------------------------------------

    def _wrap_to_width(self, text: str, points: float) -> str:
        """Wrap a heading to the drawable width at its own point size.

        0.58 em per character, not the 0.52 the node labels use: a heading is a
        prose sentence in mixed case, where DejaVu Serif runs wider than the
        short mostly-lowercase phrases inside a box. Measured, after 0.50 let an
        F02 subtitle run off a 9-inch figure. The real check is still the
        clipping test in :meth:`legibility_report`, which fails the render.
        """
        if not text:
            return text
        usable = self.width_inches - 2.0 * _MARGIN
        chars = max(20, int(usable * 72.0 / (points * 0.58)))
        return "\n".join(textwrap.wrap(text, width=chars)) or text

    def x_units(self, inches: float) -> float:
        """Convert a physical length to grid units along x."""
        return inches * self._x_per_inch

    def y_units(self, inches: float) -> float:
        """Convert a physical length to grid units along y."""
        return inches * self._y_per_inch

    # -- content ---------------------------------------------------------

    def lane(
        self,
        label: str,
        *,
        col: float,
        row: float,
        width: float,
        height: float,
        color: str = MUTED,
    ) -> None:
        """A dashed container grouping nodes -- a fold, a stage, a process boundary.

        Drawn at the lowest z-order so nodes always sit on top of it, and
        labelled inside its own top-left corner rather than above it, so two
        adjacent lanes cannot have their captions confused.
        """
        from matplotlib.patches import FancyBboxPatch

        patch = FancyBboxPatch(
            (col, row),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=" + str(self.x_units(0.06)),
            mutation_aspect=self._y_per_inch / self._x_per_inch,
            facecolor="#F7F7F7",
            edgecolor=color,
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        self.axes.add_patch(patch)
        if label:
            self.axes.text(
                col + self.x_units(0.07),
                row + self.y_units(0.17),
                label,
                fontsize=self.sub_pt,
                color=color,
                ha="left",
                va="center",
                style="italic",
                zorder=2,
            )
            self._texts.append(("lane", self.sub_pt))

    def node(
        self,
        key: str,
        label: str,
        *,
        kind: str = "process",
        col: float,
        row: float,
        width: float = 2.0,
        height: float | None = None,
        sublabel: str = "",
    ) -> Node:
        """Place one box and return it. ``key`` is what :meth:`edge` refers to.

        Leave ``height`` out and the box takes the height its own text needs
        (:meth:`fit_height`). Pass one only where a row of boxes has to share a
        height, or where the box is deliberately larger than its label.
        """
        if kind not in NODE_STYLES:
            raise ValueError(
                "unknown node kind "
                + kind
                + "; the visual language declares "
                + ", ".join(NODE_STYLES)
            )
        if key in self.nodes:
            raise ValueError("duplicate node key: " + key)

        style = NODE_STYLES[kind]
        if height is None:
            height = self.fit_height(label, width=width, sublabel=sublabel, kind=kind)
        placed = Node(key, label, kind, col, row, width, height, sublabel, style)
        self.nodes[key] = placed
        if kind not in self._used_node_kinds:
            self._used_node_kinds.append(kind)

        self._draw_shape(placed)
        self._draw_label(placed)
        return placed

    def text(
        self,
        content: str,
        *,
        col: float,
        row: float,
        ha: str = "left",
        va: str = "center",
        color: str = MUTED,
        italic: bool = True,
        width: float | None = None,
    ) -> None:
        """A free annotation on the drawing. Not a step, so it gets no box.

        Wrapped to the room actually available from where it is anchored --
        ``width`` in grid columns, or the distance to the edge it runs toward.
        An unwrapped annotation is the easiest thing on a canvas to run off the
        page, because unlike a box it has no border to make the overflow
        obvious in the source.
        """
        if width is None:
            width = {
                "left": self.columns - col,
                "right": col,
                "center": 2.0 * min(col, self.columns - col),
            }.get(ha, self.columns - col)
        inches = max(0.5, width) / self._x_per_inch
        chars = max(20, int(inches * 72.0 / (self.sub_pt * 0.58)))
        wrapped = "\n".join(
            line for block in content.split("\n") for line in (textwrap.wrap(block, chars) or [""])
        )
        annotation = self.axes.text(
            col,
            row,
            wrapped,
            fontsize=self.sub_pt,
            color=color,
            ha=ha,
            va=va,
            style="italic" if italic else "normal",
            zorder=4,
            linespacing=1.3,
        )
        self._measured.append(("annotation", annotation))
        self._texts.append(("annotation", self.sub_pt))

    def edge(
        self,
        source: str,
        target: str,
        *,
        kind: str = "flow",
        label: str = "",
        source_side: str = "auto",
        target_side: str = "auto",
        rad: float = 0.0,
    ) -> None:
        """Connect two placed nodes with one of the declared arrow kinds."""
        if kind not in EDGE_STYLES:
            raise ValueError(
                "unknown edge kind "
                + kind
                + "; the visual language declares "
                + ", ".join(EDGE_STYLES)
            )
        style = EDGE_STYLES[kind]
        src, dst = self.nodes[source], self.nodes[target]
        start_side, end_side = self._sides(src, dst, source_side, target_side)
        start, end = src.anchor(start_side), dst.anchor(end_side)

        from matplotlib.patches import FancyArrowPatch

        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle=style.arrowstyle,
            connectionstyle="arc3,rad=" + str(rad),
            mutation_scale=self.label_pt * 1.1,
            shrinkA=1.5,
            shrinkB=1.5,
            color=style.color,
            linewidth=style.linewidth,
            linestyle=style.linestyle,
            zorder=2,
        )
        self.axes.add_patch(arrow)
        if kind not in self._used_edge_kinds:
            self._used_edge_kinds.append(kind)

        mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        if kind == "forbidden":
            # An absence needs a mark. A dashed line on its own reads as "some
            # other kind of flow"; the cross says the path does not exist.
            self.axes.plot(
                [mid[0]],
                [mid[1]],
                marker="x",
                markersize=self.label_pt * 0.8,
                markeredgewidth=1.6,
                color=style.color,
                zorder=3,
            )
        if label:
            self.axes.text(
                mid[0],
                mid[1] - self.y_units(0.07),
                label,
                fontsize=self.sub_pt,
                color=style.color,
                ha="center",
                va="bottom",
                zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
            )
            self._texts.append(("edge_label", self.sub_pt))

    # -- drawing internals -----------------------------------------------

    @staticmethod
    def _sides(src: Node, dst: Node, source_side: str, target_side: str) -> tuple[str, str]:
        """Pick anchors from the dominant axis unless the caller named them."""
        sx, sy = src.center
        dx, dy = dst.center
        horizontal = abs(dx - sx) >= abs(dy - sy)
        if source_side == "auto":
            source_side = (
                ("right" if dx >= sx else "left")
                if horizontal
                else ("bottom" if dy >= sy else "top")
            )
        if target_side == "auto":
            target_side = (
                ("left" if dx >= sx else "right")
                if horizontal
                else ("top" if dy >= sy else "bottom")
            )
        return source_side, target_side

    def _draw_shape(self, node: Node) -> None:
        from matplotlib.patches import FancyBboxPatch, Polygon

        style = node.style
        x, y, w, h = node.col, node.row, node.width, node.height
        aspect = self._y_per_inch / self._x_per_inch
        common: dict[str, Any] = {
            "facecolor": style.facecolor,
            "edgecolor": style.edgecolor,
            "linewidth": style.linewidth,
            "linestyle": style.linestyle,
            "zorder": 3,
        }

        if style.shape == "plain":
            return

        if style.shape in {"round", "double", "stadium", "rect"}:
            rounding = {
                "round": self.x_units(0.07),
                "double": self.x_units(0.07),
                "stadium": self.x_units(min(0.20, h / self._y_per_inch / 2.0)),
                "rect": 0.0,
            }[style.shape]
            boxstyle = (
                "square,pad=0" if rounding == 0.0 else "round,pad=0,rounding_size=" + str(rounding)
            )
            self.axes.add_patch(
                FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, mutation_aspect=aspect, **common)
            )
            if style.shape == "double":
                inset_x, inset_y = self.x_units(0.045), self.y_units(0.045)
                self.axes.add_patch(
                    FancyBboxPatch(
                        (x + inset_x, y + inset_y),
                        w - 2 * inset_x,
                        h - 2 * inset_y,
                        boxstyle="round,pad=0,rounding_size=" + str(self.x_units(0.05)),
                        mutation_aspect=aspect,
                        facecolor="none",
                        edgecolor=style.edgecolor,
                        linewidth=0.7,
                        zorder=3,
                    )
                )
            return

        if style.shape == "parallelogram":
            skew = self.x_units(0.12)
            points = [(x + skew, y), (x + w, y), (x + w - skew, y + h), (x, y + h)]
        elif style.shape == "diamond":
            points = [
                (x + w / 2.0, y),
                (x + w, y + h / 2.0),
                (x + w / 2.0, y + h),
                (x, y + h / 2.0),
            ]
        elif style.shape == "folded":
            fold_x, fold_y = self.x_units(0.13), self.y_units(0.13)
            points = [
                (x, y),
                (x + w - fold_x, y),
                (x + w, y + fold_y),
                (x + w, y + h),
                (x, y + h),
            ]
        else:  # pragma: no cover - guarded by NODE_STYLES
            raise ValueError("unknown shape " + style.shape)

        self.axes.add_patch(Polygon(points, closed=True, **common))
        if style.shape == "folded":
            fold_x, fold_y = self.x_units(0.13), self.y_units(0.13)
            self.axes.add_patch(
                Polygon(
                    [(x + w - fold_x, y), (x + w, y + fold_y), (x + w - fold_x, y + fold_y)],
                    closed=True,
                    facecolor="white",
                    edgecolor=style.edgecolor,
                    linewidth=style.linewidth * 0.8,
                    zorder=4,
                )
            )

    def _label_metrics(
        self, label: str, sublabel: str, width: float, shape: str = "round"
    ) -> tuple[str, str, float]:
        """Wrap a box's text to its width and say how tall it needs to be.

        One function, used by both the drawing and the fit check, so a box can
        never be measured against a different wrap from the one it is drawn
        with. The line heights are the ``linespacing`` values used below, and
        the 0.07 inch allowance keeps a descender off the border.
        """
        inches = width / self._x_per_inch
        # A shape that is not a rectangle holds less text than its bounding box:
        # a diamond is half its width at the vertical midline, and a
        # parallelogram loses a skew's worth at the top and bottom. Wrapping to
        # the full width hangs the label out over the slanted edges.
        inches *= {"diamond": 0.62, "parallelogram": 0.84, "folded": 0.92}.get(shape, 1.0)
        # 0.56 em per character for DejaVu Serif at these sizes; 72 pt per inch.
        # Measured up from 0.52 after an F02 label ran to both borders of its
        # box: the estimate has to be pessimistic, because being wrong the other
        # way is invisible until someone looks at the render.
        chars = max(8, int(inches * 72.0 / (self.label_pt * 0.56)))
        wrapped = "\n".join(textwrap.wrap(label, width=chars)) or label

        wrapped_sub = ""
        sub_lines = 0
        if sublabel:
            sub_chars = max(10, int(inches * 72.0 / (self.sub_pt * 0.56)))
            lines = textwrap.wrap(sublabel, width=sub_chars) or [sublabel]
            wrapped_sub = "\n".join(lines)
            sub_lines = len(lines)

        needed = (
            len(wrapped.splitlines()) * self.label_pt * 1.25 / 72.0
            + sub_lines * self.sub_pt * 1.20 / 72.0
            + 0.07
        )
        return wrapped, wrapped_sub, needed

    def fit_height(
        self,
        label: str,
        *,
        width: float,
        sublabel: str = "",
        kind: str = "process",
        minimum_inches: float = 0.5,
        pad_inches: float = 0.12,
    ) -> float:
        """The height, in grid rows, this box needs for its own text.

        Sizing boxes by hand is how a diagram ends up with its caption sitting
        across its own border: the wrap depends on the box width, the figure
        width and the point size, and guessing that right twenty times is not
        realistic. Ask instead, and the overflow check never has to fire.
        """
        _, _, needed = self._label_metrics(label, sublabel, width, NODE_STYLES[kind].shape)
        return self.y_units(max(minimum_inches, needed + pad_inches))

    def _draw_label(self, node: Node) -> None:
        """Wrap the label to the box, then write it centred.

        The wrap width is derived from the box's physical width and the label's
        point size rather than a fixed character count, because the same box
        holds far fewer characters on a 3.4-inch column figure than on a
        9-inch landscape one.
        """
        cx, cy = node.center
        wrapped, wrapped_sub, needed = self._label_metrics(
            node.label, node.sublabel, node.width, node.style.shape
        )

        # A sublabel is anchored to the bottom of the box and the main label is
        # centred in what is left, rather than both being offset from the middle:
        # a two- or three-line label offset upward from centre climbs out of the
        # box while the sublabel stays put, and the two overlap.
        #
        # "What is left" is measured from the sublabel's own wrapped height. A
        # fixed band was right for a one-line caption and put a three-line one
        # straight through the label above it.
        sub_lines = len(wrapped_sub.splitlines()) if node.sublabel else 0
        sub_band = self.y_units(sub_lines * self.sub_pt * 1.20 / 72.0 + 0.06) if sub_lines else 0.0
        self.axes.text(
            cx,
            cy - sub_band / 2.0,
            wrapped,
            fontsize=self.label_pt,
            color=node.style.textcolor,
            ha="center",
            va="center",
            zorder=5,
            linespacing=1.25,
        )
        self._texts.append(("node_label", self.label_pt))

        if node.sublabel:
            self.axes.text(
                cx,
                node.row + node.height - self.y_units(0.06),
                wrapped_sub,
                fontsize=self.sub_pt,
                color=MUTED,
                ha="center",
                va="bottom",
                zorder=5,
                linespacing=1.2,
            )
            self._texts.append(("node_sublabel", self.sub_pt))

        if needed > node.height / self._y_per_inch:
            self._overflowing.append(node.key)

    # -- finishing -------------------------------------------------------

    def finish(self) -> Any:
        """Title, legend and return the figure. Called once, by the renderer."""
        if self.title:
            self._measured.append(
                (
                    "title",
                    self.figure.text(
                        _MARGIN / self.width_inches,
                        1.0 - (_TITLE_STRIP * self._title_lines * 0.62) / self.height_inches,
                        self.title,
                        fontsize=self.title_pt,
                        fontweight="bold",
                        color=INK,
                        ha="left",
                        va="center",
                    ),
                )
            )
            self._texts.append(("title", self.title_pt))
        if self.subtitle:
            self._measured.append(
                (
                    "subtitle",
                    self.figure.text(
                        _MARGIN / self.width_inches,
                        1.0
                        - (
                            _TITLE_STRIP * self._title_lines
                            + _SUBTITLE_STRIP * self._subtitle_lines * 0.55
                        )
                        / self.height_inches,
                        self.subtitle,
                        fontsize=self.sub_pt,
                        color=MUTED,
                        ha="left",
                        va="center",
                    ),
                )
            )
            self._texts.append(("subtitle", self.sub_pt))
        if self._legend_rows:
            self._draw_legend()
        return self.figure

    def _draw_legend(self) -> None:
        """Legend the kinds this diagram actually used, and only those.

        A fixed legend listing all seven node kinds on a diagram that uses three
        of them teaches the reader that four of the entries are noise.
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles: list[Any] = []
        labels: list[str] = []
        for kind in self._used_node_kinds:
            style: NodeStyle = NODE_STYLES[kind]
            if style.shape == "plain":
                continue
            handles.append(
                Patch(facecolor=style.facecolor, edgecolor=style.edgecolor, linewidth=0.8)
            )
            labels.append(style.meaning)
        for kind in self._used_edge_kinds:
            edge: EdgeStyle = EDGE_STYLES[kind]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=edge.color,
                    linestyle=edge.linestyle,
                    linewidth=edge.linewidth,
                )
            )
            labels.append(edge.meaning)
        if not handles:
            return

        columns = max(1, -(-len(handles) // max(1, self._legend_rows)))
        legend = self.figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(
                0.5,
                (_STAMP_STRIP - 0.04) / self.height_inches,
            ),
            ncol=columns,
            frameon=False,
            fontsize=self.sub_pt,
            handlelength=1.8,
            columnspacing=1.4,
            borderpad=0.0,
        )
        self._measured.append(("legend", legend))
        self._texts.extend(("legend", self.sub_pt) for _ in labels)

    def stamp(self, text: str) -> None:
        """Write the provenance footer into the strip reserved for it.

        Excluded from the legibility check on purpose: it is a traceability
        stamp for whoever regenerates the figure, not content the reader is
        expected to read at arm's length.
        """
        stamped = self.figure.text(
            _MARGIN / self.width_inches,
            0.05 / self.height_inches,
            text,
            fontsize=6.0,
            color=MUTED,
            ha="left",
            va="bottom",
            linespacing=1.35,
        )
        self._measured.append(("stamp", stamped))

    # -- checks -----------------------------------------------------------

    def legibility_report(self) -> dict[str, Any]:
        """What every string on this canvas prints at, once scaled to the page.

        This is T97.5 made mechanical. ``violations`` lists any role that falls
        below :data:`language.MIN_EFFECTIVE_PT`, and the renderer refuses to
        write a diagram with a non-empty list.

        It also measures what actually fits. The figure is saved at its declared
        size rather than with ``bbox_inches="tight"``, precisely so that every
        diagram in the thesis is the same scale -- which means a legend or a
        title wider than the page is *clipped* rather than silently growing the
        canvas. Off the page is therefore a build error, and the only way to
        know is to draw once and ask the artists where they landed.
        """
        scale = min(1.0, THESIS_WIDTH_INCHES / self.width_inches)
        roles: dict[str, float] = {}
        for role, points in self._texts:
            printed = effective_point_size(points, self.width_inches)
            roles[role] = min(roles.get(role, printed), printed)
        return {
            "clipped": self._clipped(),
            "figure_inches": [round(self.width_inches, 3), round(self.height_inches, 3)],
            "print_scale": round(scale, 4),
            "min_effective_pt": round(min(roles.values()), 3) if roles else None,
            "threshold_pt": MIN_EFFECTIVE_PT,
            "roles": {role: round(value, 3) for role, value in sorted(roles.items())},
            "strings": len(self._texts),
            "overflowing_nodes": sorted(self._overflowing),
            "violations": sorted(role for role, value in roles.items() if value < MIN_EFFECTIVE_PT),
        }

    def _clipped(self) -> list[str]:
        """Names of page furniture that does not fit inside the figure."""
        self.figure.canvas.draw()
        # Only the Agg canvas exposes get_renderer(), and the base class it is
        # typed as does not; every diagram is drawn on Agg (plot_style forces
        # it), so this is a typing accommodation rather than a real branch.
        get_renderer = getattr(self.figure.canvas, "get_renderer", None)
        if get_renderer is None:  # pragma: no cover - not reachable on Agg
            return []
        renderer = get_renderer()
        outside: list[str] = []
        for name, artist in self._measured:
            # An annotation is checked against the drawing area, not the page:
            # text that leaves the axes is still inside the figure, so a
            # figure-level check would pass it -- and it lands on the legend or
            # the stamp, which is exactly how it goes wrong.
            page = self.axes.bbox if name == "annotation" else self.figure.bbox
            try:
                box = artist.get_window_extent(renderer)
            except (AttributeError, RuntimeError):  # pragma: no cover - defensive
                continue
            # One pixel of slack: an artist placed exactly on the margin rounds
            # a fraction of a pixel over it and is not a layout problem.
            if (
                box.x0 < page.x0 - 1.0
                or box.y0 < page.y0 - 1.0
                or box.x1 > page.x1 + 1.0
                or box.y1 > page.y1 + 1.0
            ):
                outside.append(name)
        return sorted(outside)
