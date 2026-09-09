"""Writing a diagram: both formats, the registry row, the meta, the evidence.

This is the diagram counterpart of ``graphs.write_graph``, and it deliberately
differs from it in two places.

**A diagram has no plotted CSV, so its provenance is its source code.** A graph
is written CSV-first and the figure is drawn from that file, because a graph
asserts numbers. A diagram asserts *structure*: what F16 claims is that the
browser reads ``generated/`` and calls exactly one endpoint, and the artifact
that claim has to stay true to is the repository, not a table. So the meta
records the module and function that drew it, plus the content digest of that
module -- change the drawing and the digest moves (T97.2).

**Diagrams are numbered from their own id, not allocated sequentially.** The
G-series grew figure by figure, so ``graphs.py`` hands out the next free number
and never reassigns one. The F-series is a closed set of twenty declared up
front, so F07 is diagram 7 by arithmetic. That also keeps the two series out of
each other's way: this window renders F01-F20 while another session is still
registering G11-G35 into ``figure_registry.csv``, and a shared read-modify-write
of one CSV from two processes is a way to lose rows.

Both formats are always written (T97.3): SVG because the thesis template scales
it without resampling and an examiner can open it in Inkscape, PNG at 300 dpi
because Word and the journal submission system take raster.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.reporting.diagrams.canvas import DiagramCanvas
from src.reporting.graphs import PRINT_PROFILE
from src.reporting.plot_style import DPI, styled
from src.utils.io import ensure_dir, save_json
from src.utils.logging_setup import get_logger

__all__ = [
    "DIAGRAM_REGISTRY_FILENAME",
    "DIAGRAM_REGISTRY_COLUMNS",
    "FORMATS",
    "DiagramSpec",
    "DiagramBuilder",
    "diagrams_dir",
    "registry_path",
    "read_registry",
    "diagram_number",
    "write_diagram",
    "write_diagrams",
]

log = get_logger("reporting.diagrams")

FORMATS: tuple[str, ...] = ("png", "svg")

DIAGRAM_REGISTRY_FILENAME = "diagram_registry.csv"
DIAGRAM_REGISTRY_COLUMNS = [
    "diagram_number",
    "diagram_id",
    "title",
    "caption",
    "png",
    "svg",
    "source_module",
    "source_function",
    "source_sha256",
    "min_effective_pt",
    "first_registered_utc",
    "last_written_utc",
]

#: A builder takes nothing and returns a canvas with its content already placed.
#: It is called *inside* the print style context, so it must not create the
#: figure until it is invoked.
DiagramBuilder = Callable[[], DiagramCanvas]


@dataclass(frozen=True)
class DiagramSpec:
    """Everything about a diagram that is not its geometry."""

    diagram_id: str
    title: str
    caption: str
    #: The todo.md task this diagram discharges, e.g. ``T95.1``. Carried into the
    #: meta so a deliverable can be traced back to the line that asked for it.
    task: str = ""
    objective: str = ""
    notes: tuple[str, ...] = ()
    sources: tuple[str, ...] = field(default_factory=tuple)

    def slug(self) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", self.title.lower()).strip("_")
        return self.diagram_id + "_" + cleaned


def diagrams_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.figures_diagrams"))


def registry_path(out_dir: str | Path | None = None) -> Path:
    return diagrams_dir(out_dir) / DIAGRAM_REGISTRY_FILENAME


def read_registry(path: str | Path | None = None) -> list[dict[str, str]]:
    target = Path(path) if path else registry_path()
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def diagram_number(diagram_id: str) -> int:
    """``F07`` -> 7. Deterministic, so a partial run still numbers correctly."""
    match = re.fullmatch(r"[A-Za-z]+0*(\d+)", diagram_id)
    if match is None:
        raise ValueError("diagram id " + diagram_id + " has no numeric part")
    return int(match.group(1))


def _write_registry(rows: list[dict[str, str]], target: Path) -> Path:
    from src.utils.io import atomic_path

    ordered = sorted(rows, key=lambda row: int(row["diagram_number"]))
    ensure_dir(target.parent)
    with (
        atomic_path(target, suffix=".csv") as tmp,
        tmp.open("w", encoding="utf-8", newline="") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=DIAGRAM_REGISTRY_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({column: row.get(column, "") for column in DIAGRAM_REGISTRY_COLUMNS})
    return target


def _register(spec: DiagramSpec, row: dict[str, str], target: Path) -> int:
    rows = [dict(existing) for existing in read_registry(target)]
    now = datetime.now(UTC).isoformat()
    previous = next((r for r in rows if r.get("diagram_id") == spec.diagram_id), None)
    first_seen = (previous or {}).get("first_registered_utc") or now
    rows = [r for r in rows if r.get("diagram_id") != spec.diagram_id]

    number = diagram_number(spec.diagram_id)
    rows.append(
        {
            **row,
            "diagram_number": str(number),
            "diagram_id": spec.diagram_id,
            "title": spec.title,
            "caption": spec.caption,
            "first_registered_utc": first_seen,
            "last_written_utc": now,
        }
    )
    _write_registry(rows, target)
    return number


def _source_of(builder: DiagramBuilder) -> tuple[str, str, str]:
    """``(module path, function name, content digest)`` for the drawing code."""
    import inspect

    from src.reporting.tables import content_digest

    function = getattr(builder, "__name__", "<builder>")
    try:
        source_file = inspect.getsourcefile(builder)
    except TypeError:  # pragma: no cover - builtins never appear here
        source_file = None
    if source_file is None:
        return ("<unknown>", function, "")

    path = Path(source_file).resolve()
    try:
        relative = path.relative_to(Path(__file__).resolve().parents[3])
    except ValueError:
        relative = path
    return (str(relative).replace("\\", "/"), function, content_digest(path)[0])


def _stamp_text(spec: DiagramSpec, module: str, function: str) -> str:
    """Two lines, never more: the strip reserved on the canvas holds two."""
    return "\n".join(
        [
            spec.diagram_id + "  |  " + spec.title + "  |  drawn by " + module + ":" + function,
            "PV-MEPCG / PulseVision  |  regenerate: python scripts/05_render_diagrams.py --only "
            + spec.diagram_id,
        ]
    )


def write_diagram(
    spec: DiagramSpec,
    builder: DiagramBuilder,
    out_dir: str | Path | None = None,
    *,
    formats: tuple[str, ...] = FORMATS,
    registry: str | Path | None = None,
    evidence_index: str | Path | None = None,
    stamp: bool = True,
) -> dict[str, Path]:
    """Draw, check legibility, write both formats, register, return the paths.

    The legibility check runs *before* anything is written: a diagram whose
    labels would print below the threshold is a bug in the diagram, and letting
    it land in ``outputs/`` means it reaches the thesis and is discovered by an
    examiner instead of by the build.
    """
    import matplotlib.pyplot as plt

    from src.utils.evidence import register_evidence
    from src.utils.io import save_png

    unknown = [fmt for fmt in formats if fmt not in FORMATS]
    if unknown:
        raise ValueError("unknown diagram format(s): " + ", ".join(unknown))

    module, function, digest = _source_of(builder)
    target_dir = ensure_dir(diagrams_dir(out_dir))
    slug = spec.slug()
    written: dict[str, Path] = {}

    # savefig.bbox is "tight" for every graph in this project, and must not be
    # for a diagram: tight *expands the canvas* to fit whatever overflows, so a
    # 9-inch figure with a wide legend silently becomes a 13.7-inch one and
    # every point size the legibility check just verified is wrong. Passing
    # bbox_inches=None is not enough -- matplotlib reads the rcParam when the
    # argument is None -- so the rcParam itself is cleared for the save.
    with styled(**PRINT_PROFILE, **{"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        canvas = builder()
        if not isinstance(canvas, DiagramCanvas):
            raise TypeError(spec.diagram_id + ": builder did not return a DiagramCanvas")
        if not canvas.nodes:
            raise ValueError(spec.diagram_id + ": refusing to write a diagram with no nodes")
        figure = canvas.finish()
        if stamp:
            canvas.stamp(_stamp_text(spec, module, function))

        report = canvas.legibility_report()
        if report["violations"]:
            plt.close(figure)
            raise ValueError(
                spec.diagram_id
                + ": these text roles would print below "
                + str(report["threshold_pt"])
                + " pt at thesis width: "
                + ", ".join(report["violations"])
            )
        if report["clipped"]:
            plt.close(figure)
            raise ValueError(
                spec.diagram_id
                + ": this page furniture runs off the figure and would be clipped: "
                + ", ".join(report["clipped"])
                + " -- widen the figure, add a legend row, or shorten the text"
            )
        if report["overflowing_nodes"]:
            plt.close(figure)
            raise ValueError(
                spec.diagram_id
                + ": these boxes are too short for their own labels: "
                + ", ".join(report["overflowing_nodes"])
                + " -- give them more height or shorter text"
            )

        for fmt in formats:
            path = target_dir / (slug + "." + fmt)
            if fmt == "png":
                # bbox_inches=None, not "tight": the canvas reserved its own
                # margins in inches and a tight box would silently change the
                # figure width the legibility check was computed against.
                written["png"] = save_png(figure, path, dpi=DPI, close=False, bbox_inches=None)
            else:
                figure.savefig(path, format=fmt, dpi=DPI)
                written[fmt] = path
        plt.close(figure)

    number = _register(
        spec,
        {
            "png": written.get("png", Path("")).name,
            "svg": written.get("svg", Path("")).name,
            "source_module": module,
            "source_function": function,
            "source_sha256": digest,
            "min_effective_pt": str(report["min_effective_pt"]),
        },
        Path(registry) if registry else registry_path(out_dir),
    )

    written["meta"] = save_json(
        {
            "diagram_id": spec.diagram_id,
            "diagram_number": number,
            "title": spec.title,
            "caption": spec.caption,
            "task": spec.task or None,
            "objective": spec.objective or None,
            "framework": "PV-MEPCG / PulseVision",
            "toolchain": "matplotlib (src/reporting/diagrams)",
            "dpi": DPI,
            "formats": list(formats),
            "palette": "Okabe-Ito (colourblind-safe); shape carries meaning, colour repeats it",
            "source_module": module,
            "source_function": function,
            "source_sha256": digest,
            "legibility": report,
            "generated_utc": datetime.now(UTC).isoformat(),
            "command": "python scripts/05_render_diagrams.py --only " + spec.diagram_id,
            "sources": list(spec.sources),
            "written": {key: str(value).replace("\\", "/") for key, value in written.items()},
            "notes": list(spec.notes),
        },
        target_dir / (slug + ".meta.json"),
    )

    register_evidence(
        spec.diagram_id,
        written.get("png", written["meta"]),
        metric_or_asset=spec.title,
        objective=spec.objective,
        source_data=module,
        command="python scripts/05_render_diagrams.py --only " + spec.diagram_id,
        index_path=evidence_index,
    )
    log.info(
        "%s (diagram %d): %s",
        spec.diagram_id,
        number,
        ", ".join(path.name for path in written.values()),
    )
    return written


def write_diagrams(
    items: list[tuple[DiagramSpec, DiagramBuilder]],
    out_dir: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Path]]:
    """Write several diagrams in id order."""
    ordered = sorted(items, key=lambda item: item[0].diagram_id)
    return {
        spec.diagram_id: write_diagram(spec, builder, out_dir, **kwargs)
        for spec, builder in ordered
    }
