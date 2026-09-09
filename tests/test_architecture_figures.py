"""Phase 95 gate: F01-F10 render, and say only what the repository says.

T95.7 is a ``[TEST/MANUAL]`` check that a reviewer confirms the ten figures
render in both formats and stay legible at thesis width. The reviewer is not
available, so this is the automated substitute the work order calls for, and it
checks two things a human eye is worse at anyway:

* **Both formats, at the declared size, above the print threshold.** The writer
  refuses a diagram whose labels would print below 7 pt, whose boxes are too
  small for their own text, or whose legend would be clipped; these tests assert
  the recorded measurement rather than trusting that the refusal fired.
* **Nothing in a figure is typed by hand.** The step titles in F01 and F02 come
  from ``ARCHITECTURE_STEPS``, the corpus counts in F03 and F04 from the dataset
  audit, the family counts in F07 and F08 from the feature registry, and the
  ensemble weights in F10 from SO-05 -- each compared here against the payload
  it claims to read. A diagram is a deliverable, and rule 1 applies to it.

Greyscale legibility itself is covered by ``test_diagram_pipeline.py``: the
palette check is luminance-based, so it holds for every diagram drawn in the
language rather than needing a per-figure eye.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.reporting.architecture import ensemble_payload, pipeline_payload
from src.reporting.diagrams.canvas import DiagramCanvas
from src.reporting.diagrams.catalogue import CATALOGUE, load_builders, spec_for
from src.reporting.diagrams.language import MIN_EFFECTIVE_PT
from src.reporting.diagrams.render import write_diagram
from src.reporting.method import features_payload
from src.reporting.record_index import dataset_summary_payload

PHASE_95 = ["F0" + str(index) for index in range(1, 10)] + ["F10"]


@pytest.fixture(scope="module")
def builders() -> dict[str, Any]:
    return load_builders()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Path]]:
    """Render all ten once, into a temporary directory."""
    from src.reporting.diagrams.catalogue import load_builders as load

    target = tmp_path_factory.mktemp("f01_f10")
    available = load()
    written = {}
    for diagram_id in PHASE_95:
        written[diagram_id] = write_diagram(
            spec_for(diagram_id),
            available[diagram_id],
            target,
            registry=target / "diagram_registry.csv",
            evidence_index=target / "evidence_index.csv",
        )
    return written


def _drawn_text(build: Any) -> str:
    """Every string a builder puts on its canvas, joined.

    Read off the figure rather than off the source, so a label that is computed
    is checked as the reader sees it.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    canvas = build()
    assert isinstance(canvas, DiagramCanvas)
    canvas.finish()
    pieces = [text.get_text() for text in canvas.axes.texts]
    pieces += [text.get_text() for text in canvas.figure.texts]
    legend = canvas.figure.legends[0] if canvas.figure.legends else None
    if legend is not None:
        pieces += [entry.get_text() for entry in legend.get_texts()]
    plt.close(canvas.figure)
    # Wrapping inserts newlines mid-sentence; join them back for substring tests.
    return " ".join(" ".join(piece.split()) for piece in pieces)


# ---------------------------------------------------------------------------
# they all render, in both formats, above the print threshold
# ---------------------------------------------------------------------------


def test_all_ten_have_builders(builders: dict[str, Any]) -> None:
    missing = [diagram_id for diagram_id in PHASE_95 if diagram_id not in builders]
    assert missing == [], "no builder for " + ", ".join(missing)


@pytest.mark.parametrize("diagram_id", PHASE_95)
def test_each_renders_to_svg_and_300_dpi_png(
    diagram_id: str, rendered: dict[str, dict[str, Path]]
) -> None:
    paths = rendered[diagram_id]
    assert paths["png"].is_file() and paths["svg"].is_file()
    assert paths["png"].stat().st_size > 20_000
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["dpi"] == 300
    assert set(meta["formats"]) == {"png", "svg"}
    assert meta["framework"] == "PV-MEPCG / PulseVision"


@pytest.mark.parametrize("diagram_id", PHASE_95)
def test_each_stays_legible_at_thesis_width(
    diagram_id: str, rendered: dict[str, dict[str, Path]]
) -> None:
    legibility = json.loads(rendered[diagram_id]["meta"].read_text(encoding="utf-8"))["legibility"]
    assert legibility["violations"] == []
    assert legibility["overflowing_nodes"] == []
    assert legibility["clipped"] == []
    assert legibility["min_effective_pt"] >= MIN_EFFECTIVE_PT


def test_the_ten_share_one_registry_and_their_own_numbers(
    rendered: dict[str, dict[str, Path]],
) -> None:
    from src.reporting.diagrams.render import read_registry

    rows = {
        row["diagram_id"]: row
        for row in read_registry(rendered["F01"]["png"].parent / "diagram_registry.csv")
    }
    assert set(rows) == set(PHASE_95)
    for diagram_id, row in rows.items():
        assert int(row["diagram_number"]) == int(diagram_id[1:])
        assert row["source_module"].endswith("architecture_figures.py")
        assert row["source_sha256"]


# ---------------------------------------------------------------------------
# nothing is typed by hand
# ---------------------------------------------------------------------------


def test_f01_and_f02_read_the_documented_steps(builders: dict[str, Any]) -> None:
    """The twelve steps are read, not restated (T95.2, and the work order).

    Both figures are checked against ``pipeline_payload`` -- the same list the
    dashboard walkthrough renders, and the one that refuses to emit a step whose
    module or evidence directory is missing from the repository.
    """
    steps = pipeline_payload()["steps"]
    f02 = _drawn_text(builders["F02"])
    for step in steps:
        title = str(step["title"]).split(",")[0].strip()
        assert title in f02, "F02 does not carry step " + str(step["index"])
        assert str(step["module"]) in f02, "F02 does not name " + str(step["module"])

    f01 = _drawn_text(builders["F01"])
    for step in steps:
        assert str(step["title"]).split(",")[0].strip() in f01
    for directory in {str(step["evidence_dir"]) for step in steps}:
        assert directory in f01, "F01 does not name " + directory


def test_f02_marks_exactly_the_steps_that_declare_a_rule(builders: dict[str, Any]) -> None:
    guarded = [step for step in pipeline_payload()["steps"] if step["rule"]]
    text = _drawn_text(builders["F02"])
    for step in guarded:
        assert str(step["index"]) + ". " + str(step["rule"]).split(":")[0] in text
    assert len(guarded) >= 1


@pytest.mark.parametrize("diagram_id", ["F03", "F04"])
def test_corpus_counts_are_the_audited_display_strings(
    diagram_id: str, builders: dict[str, Any]
) -> None:
    """The counts in the figure are the strings the audit already formatted."""
    text = _drawn_text(builders[diagram_id])
    for row in dataset_summary_payload()["summary"]:
        assert str(row["dataset_name"]) in text
    if diagram_id == "F03":
        for row in dataset_summary_payload()["summary"]:
            assert str(row["n_modelled_display"]) in text
            assert str(row["n_subjects_display"]) in text


def test_f03_keeps_the_five_label_spaces_apart(builders: dict[str, Any]) -> None:
    """Rule 4 drawn: five targets, and the merge that never happens."""
    from src.reporting.diagrams.architecture_figures import _TRACKS

    canvas = builders["F03"]()
    canvas.finish()
    assert len({label for _, label, _ in _TRACKS}) == 5
    assert "forbidden" in canvas._used_edge_kinds
    text = _drawn_text(builders["F03"])
    assert "never merged" in text


@pytest.mark.parametrize("diagram_id", ["F05", "F08", "F09"])
def test_the_fold_safety_figures_draw_the_path_that_does_not_exist(
    diagram_id: str, builders: dict[str, Any]
) -> None:
    """An absence cannot be shown by leaving the arrow out."""
    canvas = builders[diagram_id]()
    canvas.finish()
    assert "forbidden" in canvas._used_edge_kinds, diagram_id + " states no fold-safety boundary"


@pytest.mark.parametrize("diagram_id", ["F07", "F08"])
def test_family_counts_come_from_the_feature_registry(
    diagram_id: str, builders: dict[str, Any]
) -> None:
    payload = features_payload()
    text = _drawn_text(builders[diagram_id])
    for family in payload["families"]:
        assert str(family["family"]) in text
    assert str(payload["n_features"]) in text
    assert sum(int(f["n_features"]) for f in payload["families"]) == int(payload["n_features"])


def test_f10_shows_the_searched_weights_without_re_rounding(builders: dict[str, Any]) -> None:
    """The weight strings are SO-05's own display values, formatted in Python."""
    payload = ensemble_payload()
    if not payload.get("available"):
        pytest.skip("SO-05 weights are not available in this checkout")
    text = _drawn_text(builders["F10"])
    for member in payload["members"]:
        assert str(member["name"]) in text
        assert str(member["weight_display"]) in text
        assert str(member["weight_std_display"]) in text
    # The finding that matters more than the weights themselves.
    assert str(payload["folds_identical_display"]) in text
    assert str(payload["equal_weight_display"]) in text


def test_f06_reads_the_preprocessing_steps_and_their_configured_parameters(
    builders: dict[str, Any],
) -> None:
    from src.preprocessing.pipeline import PIPELINE_STEPS
    from src.utils.config import load_config

    signal = load_config("signal")
    text = _drawn_text(builders["F06"])
    for step in PIPELINE_STEPS:
        assert step in text
    assert str(signal.require("resample.target_fs")) in text
    assert str(signal.require("filter.low_hz")) in text
    assert str(signal.require("filter.high_hz")) in text


def test_f05_reads_the_declared_cross_validation_scheme(builders: dict[str, Any]) -> None:
    from src.utils.config import load_config

    scheme = load_config("experiments").require("cv_schemes.repeated_5x5_grouped")
    text = _drawn_text(builders["F05"])
    assert str(scheme["group_key"]) in text
    assert str(int(scheme["n_splits"]) * int(scheme["n_repeats"])) in text


def test_every_phase_95_spec_names_its_task(builders: dict[str, Any]) -> None:
    for spec in CATALOGUE:
        if spec.diagram_id in PHASE_95:
            assert spec.task.startswith("T95."), spec.diagram_id
            assert spec.objective, spec.diagram_id
