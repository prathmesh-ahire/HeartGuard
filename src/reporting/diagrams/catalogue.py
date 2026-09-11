"""The twenty diagrams, declared once, and the registry of what draws them.

The catalogue is separate from the drawings on purpose. It is the closed list
the renderer counts against, so ``scripts/05_render_diagrams.py --expect 20``
can fail on a *missing* diagram rather than quietly rendering nineteen and
reporting success -- the same reason the project writes an explicit
``missing_outputs_report.txt`` instead of letting a gap look like a decision.

Builders attach themselves with the :func:`diagram` decorator from whichever
module implements them (F01-F10 in ``architecture_figures``, F11-F20 in
``pipeline_figures``). A builder for an id that is not in this catalogue is
refused at import time, and a catalogue entry with no builder is reported as
pending rather than skipped.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from src.reporting.diagrams.render import DiagramBuilder, DiagramSpec

__all__ = [
    "CATALOGUE",
    "EXPECTED_COUNT",
    "BUILDER_MODULES",
    "diagram",
    "load_builders",
    "spec_for",
    "implemented",
    "pending",
]

#: Modules that register builders. Imported by :func:`load_builders`; absent
#: ones are tolerated so the pipeline is testable before Phases 95 and 96 fill
#: them in, and are then reported as pending rather than passing silently.
BUILDER_MODULES: tuple[str, ...] = (
    "src.reporting.diagrams.architecture_figures",
    "src.reporting.diagrams.pipeline_figures",
)

#: ``objective`` names the blueprint objectives (1-6, ``objectives.OBJECTIVES``)
#: a diagram supports. Until Phase 96 F11-F20 cited O7-O9, which do not exist:
#: the blueprint has six objectives. Pinned by tests/test_pipeline_figures.py.
CATALOGUE: tuple[DiagramSpec, ...] = (
    DiagramSpec(
        "F01",
        "Overall PV-MEPCG proposed architecture",
        "The PV-MEPCG / PulseVision framework end to end: corpora, harmonization, "
        "fold-safe preprocessing, 138-feature extraction, search-optimized "
        "heterogeneous ensemble, and the evaluation that reports it.",
        task="T95.1",
        objective="O1",
    ),
    DiagramSpec(
        "F02",
        "End-to-end PCG classification workflow",
        "The twelve documented architecture steps, each naming the module that "
        "implements it and the outputs/ directory that evidences it.",
        task="T95.2",
        objective="O1",
    ),
    DiagramSpec(
        "F03",
        "Three-dataset experimental track diagram",
        "PhysioNet 2016, PASCAL A and B, and CirCor 2022 as separate tracks with "
        "five label spaces that are never merged.",
        task="T95.3",
        objective="O6",
    ),
    DiagramSpec(
        "F04",
        "Dataset harmonization and metadata workflow",
        "How four corpora at three native sampling rates become one master "
        "metadata table without losing provenance or subject identity.",
        task="T95.3",
        objective="O6",
    ),
    DiagramSpec(
        "F05",
        "Subject-wise data splitting",
        "Grouped cross-validation: a subject appears in exactly one side of a "
        "split, and the outer test fold fits nothing.",
        task="T95.4",
        objective="O1",
    ),
    DiagramSpec(
        "F06",
        "Signal preprocessing architecture",
        "Resampling, band-pass filtering, normalization and quality flagging, in "
        "the order the pipeline applies them.",
        task="T95.4",
        objective="O4",
    ),
    DiagramSpec(
        "F07",
        "138-feature extraction architecture",
        "The five feature families and the per-family counts that sum to the 138 "
        "engineered features.",
        task="T95.5",
        objective="O4",
    ),
    DiagramSpec(
        "F08",
        "Feature family fusion",
        "How the family blocks are concatenated into one matrix, and where the "
        "fold-safe scaler and imputer sit relative to that join.",
        task="T95.5",
        objective="O4",
    ),
    DiagramSpec(
        "F09",
        "Search-based feature selection workflow",
        "Selection driven by search inside the training fold only, with the outer "
        "test fold never scored during the search.",
        task="T95.6",
        objective="O5",
    ),
    DiagramSpec(
        "F10",
        "SVM RF GB optimized soft voting architecture",
        "The heterogeneous ensemble: three tuned base learners, per-class "
        "probability averaging, and the searched weight vector.",
        task="T95.6",
        objective="O3",
    ),
    DiagramSpec(
        "F11",
        "Binary classification pipeline",
        "The normal versus abnormal task end to end, from PhysioNet recordings to "
        "the reported sensitivity and balanced accuracy.",
        task="T96.1",
        objective="O1",
    ),
    DiagramSpec(
        "F12",
        "PASCAL multiclass pipeline",
        "PASCAL A (4-class) and PASCAL B (3-class) as two separate targets, with "
        "artifact shown as the recording-quality label it is.",
        task="T96.1",
        objective="O6",
    ),
    DiagramSpec(
        "F13",
        "CirCor recording to subject aggregation",
        "Several auscultation locations per patient collapsed to one patient-level "
        "indication under the max, mean and any-present rules.",
        task="T96.2",
        objective="O6",
    ),
    DiagramSpec(
        "F14",
        "Cross-dataset external validation workflow",
        "EXP-D1: trained on adult PhysioNet, tested on paediatric CirCor, with the "
        "population mismatch recorded before the metric.",
        task="T96.3",
        objective="O1",
    ),
    DiagramSpec(
        "F15",
        "Noise and duration robustness framework",
        "Quality-flag partitions, synthetic AWGN levels and the duration bands the "
        "robustness experiments report over.",
        task="T96.3",
        objective="O4",
    ),
    DiagramSpec(
        "F16",
        "Dashboard system architecture",
        "The build-time codegen boundary from outputs/ to frontend/lib/generated/, "
        "and the single live /predict endpoint that crosses the wire at runtime.",
        task="T96.4",
        objective="O1",
    ),
    DiagramSpec(
        "F17",
        "Prediction and report generation flow",
        "One uploaded recording through preprocessing, extraction, the deployed "
        "bundle and the generated screening report.",
        task="T96.4",
        objective="O1",
    ),
    DiagramSpec(
        "F18",
        "Objective to module traceability",
        "Each research objective mapped to the modules that implement it and the "
        "artifacts that evidence it.",
        task="T96.5",
        objective="O1-O6",
    ),
    DiagramSpec(
        "F19",
        "Novelty contribution diagram",
        "What this framework adds over the baseline it is compared against, "
        "component by component.",
        task="T96.6",
        objective="O3, O5",
    ),
    DiagramSpec(
        "F20",
        "Final research contribution summary",
        "The contributions of PV-MEPCG / PulseVision, each tied to the experiment "
        "that supports it.",
        task="T96.6",
        objective="O1-O6",
    ),
)

#: What a complete run renders. Asserted by the renderer, not assumed.
EXPECTED_COUNT = 20

_SPECS: dict[str, DiagramSpec] = {spec.diagram_id: spec for spec in CATALOGUE}
_BUILDERS: dict[str, DiagramBuilder] = {}
_LOADED = False


def diagram(diagram_id: str) -> Callable[[DiagramBuilder], DiagramBuilder]:
    """Register the decorated function as the builder for ``diagram_id``."""

    def decorate(builder: DiagramBuilder) -> DiagramBuilder:
        if diagram_id not in _SPECS:
            raise KeyError(diagram_id + " is not in the diagram catalogue; add a DiagramSpec first")
        if diagram_id in _BUILDERS and _BUILDERS[diagram_id] is not builder:
            raise KeyError(diagram_id + " already has a builder")
        _BUILDERS[diagram_id] = builder
        return builder

    return decorate


def load_builders(*, reload: bool = False) -> dict[str, DiagramBuilder]:
    """Import the builder modules and return ``{diagram_id: builder}``."""
    global _LOADED
    if _LOADED and not reload:
        return dict(_BUILDERS)
    for name in BUILDER_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:  # a real missing dependency, not a pending module
                raise
            continue
        if reload:
            importlib.reload(module)
    _LOADED = True
    return dict(_BUILDERS)


def spec_for(diagram_id: str) -> DiagramSpec:
    return _SPECS[diagram_id]


def implemented() -> list[str]:
    """Catalogue ids that currently have a builder, in id order."""
    builders = load_builders()
    return [spec.diagram_id for spec in CATALOGUE if spec.diagram_id in builders]


def pending() -> list[str]:
    """Catalogue ids with no builder yet. Reported, never silently dropped."""
    builders = load_builders()
    return [spec.diagram_id for spec in CATALOGUE if spec.diagram_id not in builders]


def catalogue_report() -> dict[str, Any]:
    """A one-glance summary for the renderer's console output."""
    done, todo = implemented(), pending()
    return {
        "expected": EXPECTED_COUNT,
        "declared": len(CATALOGUE),
        "implemented": done,
        "pending": todo,
        "complete": len(done) == EXPECTED_COUNT and not todo,
    }
