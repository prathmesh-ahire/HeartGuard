"""The graph engine -- a PNG is never published without the CSV behind it (Phase 90).

``src/reporting/plot_style.py`` already settles what a figure *looks* like: 300
dpi, DejaVu Serif, the Okabe-Ito colourblind-safe palette in a fixed order,
viridis for continuous data. This module settles what a figure *is*: a
:class:`GraphSpec`, the exact data frame that was plotted, and a draw function
that turns the second into pixels.

Why the frame is a first-class argument (T90.2)
-----------------------------------------------
Rule 1 says every number traces to its source CSV. A chart is a few hundred
numbers with axes on them, and the usual way of writing one -- load a file,
slice it, aggregate it, plot the result -- leaves the plotted values existing
nowhere but inside the PNG. Nobody can check them, and a bug in the slicing
looks exactly like a real finding.

So :func:`write_graph` takes the frame and the draw function separately, writes
the frame to ``<slug>.csv`` at full precision, and calls ``draw(frame)``. **The
PNG is drawn from the file that was written**, not from something adjacent to
it. The correspondence is structural rather than promised, which is why the
T90.7 gate can assert it by reading both back.

``sources`` on the spec is the layer above: the upstream CSVs the frame was
derived FROM (``class_distribution.csv``), while ``<slug>.csv`` is what was
actually plotted. Both are recorded, with sha256 for the former, so a figure
built from a superseded audit file is detectable rather than merely suspicious.

Stable figure numbers (T90.3)
-----------------------------
``figure_registry.csv`` maps ``figure_id`` (G01) to a printed figure number.
The number is assigned **once**, on first registration, and never reassigned:
a thesis that says "see Figure 7" must still mean the same figure after the
figures are regenerated in a different order, or after a new one is inserted.
Within one batch, ids are registered in sorted order so that a first run is
itself order-independent.

Profiles (T90.5)
----------------
``screen`` is the project default. ``print`` is for the thesis: a hard white
ground with no transparency, heavier lines and rules that survive a laser
printer, and Type 42 fonts so the text stays selectable rather than being
outlined. Both are the same figure and the same numbers -- only the ink changes.
"""

from __future__ import annotations

import contextlib
import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.reporting.plot_style import (
    DPI,
    FIGSIZE,
    OKABE_ITO,
    RC_PARAMS,
    SEQUENTIAL_CMAP,
    class_color,
    styled,
)
from src.utils.io import ensure_dir, save_csv, save_json
from src.utils.logging_setup import get_logger

__all__ = [
    "GraphSpec",
    "Graph",
    "PROFILES",
    "PRINT_PROFILE",
    "FORMATS",
    "REGISTRY_FILENAME",
    "REGISTRY_COLUMNS",
    "DPI",
    "FIGSIZE",
    "OKABE_ITO",
    "SEQUENTIAL_CMAP",
    "class_color",
    "subplots",
    "figures_dir",
    "registry_path",
    "read_registry",
    "figure_number_for",
    "write_graph",
    "write_graphs",
    "source_fingerprint",
    "content_digest",
]

log = get_logger("reporting.graphs")

FORMATS: tuple[str, ...] = ("png", "svg")

REGISTRY_FILENAME = "figure_registry.csv"
REGISTRY_COLUMNS = [
    "figure_number",
    "figure_id",
    "title",
    "caption",
    "filename",
    "source_csv",
    "upstream_sources",
    "first_registered_utc",
    "last_written_utc",
]

#: T90.5 -- the thesis print profile. Overlaid on RC_PARAMS, never replacing it,
#: so a style decision made once in plot_style.py still holds here.
PRINT_PROFILE: dict[str, Any] = {
    # A transparent ground picks up whatever the page behind it is; a printed
    # figure needs its own white.
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.facecolor": "white",
    # Hairlines that read fine at 150 dpi on a screen disappear at 300 dpi on
    # paper. Grid and spines go up together so the ratio between them holds.
    "lines.linewidth": 1.2,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "patch.linewidth": 0.6,
    # Type 42 keeps text as text in the PDF the thesis is compiled to: it stays
    # selectable, searchable, and does not thicken when the figure is scaled.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

PROFILES: dict[str, dict[str, Any]] = {"screen": {}, "print": PRINT_PROFILE}


def subplots(size: str | tuple[float, float] = "double", **kwargs: Any) -> Any:
    """``plt.subplots`` at a named page size, inside whatever profile is active.

    Deliberately not :func:`plot_style.figure`, which calls ``apply_style()``
    and would overwrite the profile the caller is drawing under.
    """
    import matplotlib.pyplot as plt

    figsize = FIGSIZE[size] if isinstance(size, str) else size
    return plt.subplots(figsize=figsize, **kwargs)


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSpec:
    """Everything about a figure that is not its pixels."""

    figure_id: str
    title: str
    caption: str
    sources: tuple[str, ...]
    exp_id: str = ""
    objective: str = ""
    dataset: str = ""
    notes: tuple[str, ...] = ()
    command: str = ""
    size: str = "double"

    def slug(self) -> str:
        import re

        cleaned = re.sub(r"[^a-z0-9]+", "_", self.title.lower()).strip("_")
        return self.figure_id + "_" + cleaned

    def experiment_id(self) -> str:
        from src.reporting.tables import NOT_EXPERIMENT_BOUND

        return self.exp_id or NOT_EXPERIMENT_BOUND


@dataclass
class Graph:
    """A spec, the exact frame that was plotted, and how to plot it."""

    spec: GraphSpec
    frame: Any
    draw: Callable[[Any], Any]
    csv_kwargs: dict[str, Any] = field(default_factory=dict)


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    """Path, size and content digest of one upstream source. See ``tables.py``."""
    from src.reporting.tables import source_fingerprint as fingerprint

    return fingerprint(path)


def content_digest(path: str | Path) -> tuple[str, str]:
    """``(sha256, method)`` for one file, stable across platforms.

    Re-exported from ``tables.py`` so a figure and a table cannot end up
    using two different definitions of 'the same file'.
    """
    from src.reporting.tables import content_digest as digest

    return digest(path)


# ---------------------------------------------------------------------------
# T90.3 -- the figure-number registry
# ---------------------------------------------------------------------------


def figures_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.figures_diagrams"))


def registry_path(out_dir: str | Path | None = None) -> Path:
    return figures_dir(out_dir) / REGISTRY_FILENAME


def read_registry(path: str | Path | None = None) -> list[dict[str, str]]:
    target = Path(path) if path else registry_path()
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def figure_number_for(figure_id: str, path: str | Path | None = None) -> int | None:
    """The printed number this figure already holds, or ``None`` if it is new."""
    for row in read_registry(path):
        if row.get("figure_id") == figure_id:
            return int(row["figure_number"])
    return None


def _write_registry(rows: list[dict[str, str]], target: Path) -> Path:
    from src.utils.io import atomic_path

    ordered = sorted(rows, key=lambda row: int(row["figure_number"]))
    ensure_dir(target.parent)
    with (
        atomic_path(target, suffix=".csv") as tmp,
        tmp.open("w", encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({c: row.get(c, "") for c in REGISTRY_COLUMNS})
    return target


def _warn_if_partial_batch(figure_id: str, number: int) -> None:
    """Flag a new number that lands BELOW the id's own index.

    A numbered series registered as a partial batch numbers wrongly and then
    stays that way, because T90.3's whole point is that a number, once given, is
    never reassigned. Running ``--skip-audio`` first put G10 at figure 5, and
    G05-G09 would then have taken 6-10: stable, and nonsense to read.

    Only the below case is worth a warning. A later series legitimately starts
    high -- F01 following G35 becomes figure 36 -- so a number above the index
    is normal and says nothing.
    """
    import re

    match = re.fullmatch(r"[A-Za-z]+0*(\d+)", figure_id)
    if match is not None and number < int(match.group(1)):
        log.warning(
            "%s was assigned figure number %d, below its own index %s. This looks "
            "like a partial batch: earlier ids in the series were not registered, "
            "and the number will NOT be reassigned later. Delete %s and regenerate "
            "the whole series if the numbering matters.",
            figure_id,
            number,
            match.group(1),
            REGISTRY_FILENAME,
        )


def _register_figure(spec: GraphSpec, png_name: str, csv_name: str, target: Path) -> int:
    """Upsert one figure, preserving a number it already holds (T90.3)."""
    rows: list[dict[str, str]] = [dict(row) for row in read_registry(target)]
    now = datetime.now(UTC).isoformat()

    existing = next((r for r in rows if r.get("figure_id") == spec.figure_id), None)
    if existing is not None:
        number = int(existing["figure_number"])
        first_seen = existing.get("first_registered_utc") or now
        rows = [r for r in rows if r.get("figure_id") != spec.figure_id]
    else:
        taken = [int(r["figure_number"]) for r in rows]
        number = max(taken) + 1 if taken else 1
        first_seen = now
        _warn_if_partial_batch(spec.figure_id, number)

    rows.append(
        {
            "figure_number": str(number),
            "figure_id": spec.figure_id,
            "title": spec.title,
            "caption": spec.caption,
            "filename": png_name,
            "source_csv": csv_name,
            "upstream_sources": "; ".join(spec.sources),
            "first_registered_utc": first_seen,
            "last_written_utc": now,
        }
    )
    _write_registry(rows, target)
    return number


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def _source_stamp(spec: GraphSpec, csv_name: str) -> str:
    """Short lines, one source per line.

    Never one long line. ``savefig(bbox_inches="tight")`` expands the *canvas*
    to fit a wide text box, so a 130-character stamp silently turns a 7.2-inch
    figure into a 10-inch one and changes the aspect ratio of the plot it is
    annotating. Learned in Phase 28; it applies here for the same reason.
    """
    lines = [spec.figure_id + "  |  " + spec.title + "  |  plotted from " + csv_name]
    if spec.sources:
        lines.append("derived from:")
        lines.extend("    " + source for source in spec.sources)
    else:
        lines.append("derived from: (none recorded)")
    return "\n".join(lines)


#: Vertical space one 6 pt stamp line needs, in inches, plus a bottom margin.
_STAMP_LINE_INCHES = 0.13
_STAMP_MARGIN_INCHES = 0.10


def _stamp(figure: Any, text: str) -> None:
    """Reserve footer space for the stamp, then write it.

    Two things :func:`plot_style.annotate_source` does not handle, both found by
    looking at the output rather than by a test:

    **Constrained layout.** ``annotate_source`` calls ``tight_layout(rect=...)``,
    which is right for a plain figure and wrong for one using a layout engine:
    it does not raise, it just relocates the axes and leaves a colorbar sitting
    on top of the panel it belongs to. G06 was drawn that way once.

    **The reserve is a fraction, and figures are not all one height.** 0.045 of
    the figure per line is about 0.14 inches on a 3.2-inch ``double`` figure and
    0.27 inches on a 6-inch ``tall`` one, so the same four-line stamp left an
    inch of blank paper under G07. Reserving in inches and converting keeps the
    footer the same physical size whatever the figure is.
    """
    lines = len(text.splitlines())
    height = float(figure.get_size_inches()[1]) or 1.0
    reserved = (_STAMP_LINE_INCHES * lines + _STAMP_MARGIN_INCHES) / height

    engine = getattr(figure, "get_layout_engine", lambda: None)()
    if engine is not None:
        with contextlib.suppress(Exception):
            engine.set(rect=(0.0, reserved, 1.0, 1.0))
    else:
        try:
            figure.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
        except (ValueError, AttributeError):  # some layouts refuse; not fatal
            figure.subplots_adjust(bottom=max(figure.subplotpars.bottom, reserved + 0.08))

    figure.text(
        0.005,
        _STAMP_MARGIN_INCHES / height / 2.0,
        text,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#555555",
        linespacing=1.4,
    )


def write_graph(
    graph: Graph,
    out_dir: str | Path | None = None,
    *,
    formats: tuple[str, ...] = ("png",),
    profile: str = "screen",
    evidence_index: str | Path | None = None,
    registry: str | Path | None = None,
    stamp: bool = True,
) -> dict[str, Path]:
    """Write the plotted CSV, then draw the figure FROM it, then register both.

    The order is the point (T90.2): the frame is written first, the draw call
    receives that same frame, and the registry records the pair. There is no
    path through this function that produces a PNG without the CSV behind it.
    """
    import matplotlib.pyplot as plt

    from src.utils.evidence import register_evidence
    from src.utils.io import save_png

    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError("unknown graph format(s): " + ", ".join(unknown))
    if profile not in PROFILES:
        raise ValueError("unknown profile " + profile + "; expected one of " + ", ".join(PROFILES))
    if graph.frame is None or len(graph.frame) == 0:
        raise ValueError(graph.spec.figure_id + ": refusing to draw a figure from an empty frame")

    target_dir = ensure_dir(figures_dir(out_dir))
    slug = graph.spec.slug()

    written: dict[str, Path] = {}
    written["csv"] = save_csv(graph.frame, target_dir / (slug + ".csv"), **graph.csv_kwargs)

    with styled(**PROFILES[profile]):
        figure = graph.draw(graph.frame)
        if figure is None:
            raise ValueError(graph.spec.figure_id + ": draw() returned no figure")
        if stamp:
            _stamp(figure, _source_stamp(graph.spec, written["csv"].name))
        for fmt in formats:
            path = target_dir / (slug + "." + fmt)
            if fmt == "png":
                written["png"] = save_png(figure, path, dpi=DPI, close=False)
            else:
                figure.savefig(path, format=fmt, dpi=DPI, bbox_inches="tight")
                written[fmt] = path
        plt.close(figure)

    number = _register_figure(
        graph.spec,
        written.get("png", written["csv"]).name,
        written["csv"].name,
        Path(registry) if registry else registry_path(out_dir),
    )

    written["meta"] = save_json(
        {
            "figure_id": graph.spec.figure_id,
            "figure_number": number,
            "title": graph.spec.title,
            "caption": graph.spec.caption,
            "exp_id": graph.spec.experiment_id(),
            "objective": graph.spec.objective or None,
            "dataset": graph.spec.dataset or None,
            "framework": "PV-MEPCG / PulseVision",
            "profile": profile,
            "dpi": DPI,
            "palette": "Okabe-Ito (colourblind-safe), fixed order",
            "generated_utc": datetime.now(UTC).isoformat(),
            "command": graph.spec.command or None,
            "plotted_rows": len(graph.frame),
            "plotted_csv": written["csv"].name,
            # Newline-normalized, not a raw byte hash: the same committed CSV
            # is CRLF in a Windows working tree and LF in the repository, so a
            # raw digest reports "stale" on CI and clean locally. See
            # tables.content_digest.
            "plotted_csv_sha256": content_digest(written["csv"])[0],
            "plotted_csv_digest_method": content_digest(written["csv"])[1],
            "sources": [source_fingerprint(s) for s in graph.spec.sources],
            "written": {k: str(v).replace("\\", "/") for k, v in written.items()},
            "notes": list(graph.spec.notes),
        },
        target_dir / (slug + ".meta.json"),
    )

    register_evidence(
        graph.spec.figure_id,
        written.get("png", written["csv"]),
        metric_or_asset=graph.spec.title,
        objective=graph.spec.objective,
        experiment_id=graph.spec.exp_id,
        dataset=graph.spec.dataset,
        source_data=str(written["csv"]),
        command=graph.spec.command,
        index_path=evidence_index,
    )
    log.info(
        "%s (figure %d): %s",
        graph.spec.figure_id,
        number,
        ", ".join(p.name for p in written.values()),
    )
    return written


def write_graphs(
    graphs: list[Graph],
    out_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Path]]:
    """Write several figures, registering ids in sorted order.

    Sorting matters for T90.3: a first run over an empty registry must assign
    the same numbers whatever order the caller happened to build the list in.
    """
    ordered = sorted(graphs, key=lambda graph: graph.spec.figure_id)
    return {graph.spec.figure_id: write_graph(graph, out_dir, **kwargs) for graph in ordered}


def profile_rc(profile: str) -> dict[str, Any]:
    """The full rcParams a profile draws under, for inspection and testing."""
    return {**RC_PARAMS, **PROFILES[profile]}
