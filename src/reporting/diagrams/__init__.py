"""Diagram production for PV-MEPCG / PulseVision (Phase 97).

Twenty architecture and pipeline diagrams, drawn in matplotlib from
version-controlled Python, in one visual language, to SVG and 300 dpi PNG.

* ``language`` -- the vocabulary: node shapes, arrow semantics, contrast rules.
* ``canvas``   -- the grid a diagram is placed on.
* ``render``   -- writing one: both formats, the registry row, the meta.
* ``catalogue``-- the closed list of twenty and what draws each of them.
* ``specimen`` -- a sheet of every kind, used to test the pipeline itself.

Regenerate everything with ``python scripts/05_render_diagrams.py``.
"""

from __future__ import annotations

from src.reporting.diagrams.canvas import DIAGRAM_SIZES, DiagramCanvas, Node
from src.reporting.diagrams.catalogue import (
    CATALOGUE,
    EXPECTED_COUNT,
    catalogue_report,
    diagram,
    implemented,
    load_builders,
    pending,
    spec_for,
)
from src.reporting.diagrams.language import (
    EDGE_STYLES,
    NODE_STYLES,
    contrast_ratio,
    palette_report,
)
from src.reporting.diagrams.render import (
    DiagramSpec,
    diagram_number,
    diagrams_dir,
    read_registry,
    registry_path,
    write_diagram,
    write_diagrams,
)

__all__ = [
    "CATALOGUE",
    "DIAGRAM_SIZES",
    "EDGE_STYLES",
    "EXPECTED_COUNT",
    "NODE_STYLES",
    "DiagramCanvas",
    "DiagramSpec",
    "Node",
    "catalogue_report",
    "contrast_ratio",
    "diagram",
    "diagram_number",
    "diagrams_dir",
    "implemented",
    "load_builders",
    "palette_report",
    "pending",
    "read_registry",
    "registry_path",
    "spec_for",
    "write_diagram",
    "write_diagrams",
]
