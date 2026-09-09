"""Phase 97 gate: the diagram pipeline regenerates everything from source.

T97.7 asks for one command, from clean, with no manual step. What that has to
mean *before* F01-F20 exist (Phases 95 and 96 draw them; the pipeline has to
exist first) is checked here on the specimen sheet, which exercises every node
kind and every arrow kind the language declares:

* both formats land, at the declared size, at 300 dpi;
* the PNG is byte-identical across two runs, so "regenerate" cannot mean
  "produce something slightly different";
* the registry, the meta and the source digest are written;
* the checks that keep a diagram legible actually fail when they should.

The full twenty-diagram form of the gate -- ``--expect 20`` returning zero --
is run at the end of Phase 96, when there are twenty builders to run it over.
Until then :func:`test_a_partial_run_is_an_error_not_a_shorter_table` pins the
behaviour that matters: a run that renders fewer than it was told to expect
fails loudly instead of reporting success.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.reporting.diagrams.canvas import DiagramCanvas
from src.reporting.diagrams.catalogue import (
    CATALOGUE,
    EXPECTED_COUNT,
    catalogue_report,
    diagram,
    load_builders,
)
from src.reporting.diagrams.language import (
    EDGE_STYLES,
    MIN_EFFECTIVE_PT,
    MIN_TEXT_CONTRAST,
    NODE_STYLES,
    PAPER,
    contrast_ratio,
    effective_point_size,
    palette_report,
)
from src.reporting.diagrams.render import (
    DiagramSpec,
    diagram_number,
    read_registry,
    write_diagram,
)
from src.reporting.diagrams.specimen import SPECIMEN_SPEC, build_specimen

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "05_render_diagrams.py"


# ---------------------------------------------------------------------------
# rendering the specimen
# ---------------------------------------------------------------------------


def _render(target: Path, **kwargs: Any) -> dict[str, Path]:
    return write_diagram(
        SPECIMEN_SPEC,
        build_specimen,
        target,
        registry=target / "diagram_registry.csv",
        evidence_index=target / "evidence_index.csv",
        **kwargs,
    )


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Render the specimen once for the whole module."""
    target = tmp_path_factory.mktemp("diagrams")
    return _render(target)


def test_both_formats_are_written(rendered: dict[str, Path]) -> None:
    assert rendered["png"].is_file()
    assert rendered["svg"].is_file()
    # A file that exists but holds nothing is not a rendered diagram.
    assert rendered["png"].stat().st_size > 20_000
    assert rendered["svg"].stat().st_size > 5_000


def test_the_png_is_exactly_the_declared_size_at_300_dpi(rendered: dict[str, Path]) -> None:
    """The saved raster must match figsize x 300, not whatever fitted.

    This is the check that catches ``savefig.bbox="tight"`` leaking in from the
    project style: tight grows the canvas to fit an overflowing legend, and the
    printed point size of every label then differs from the one the legibility
    check verified. It was doing exactly that until the rcParam was cleared.
    """
    import matplotlib.image as mpimg

    pixels = mpimg.imread(rendered["png"])
    meta = json.loads(rendered["meta"].read_text(encoding="utf-8"))
    width, height = meta["legibility"]["figure_inches"]
    assert meta["dpi"] == 300
    assert pixels.shape[1] == round(width * 300)
    assert pixels.shape[0] == round(height * 300)


def test_the_svg_keeps_its_text_as_text(rendered: dict[str, Path]) -> None:
    """``svg.fonttype: none`` in the print profile, so the title is searchable."""
    content = rendered["svg"].read_text(encoding="utf-8", errors="replace")
    assert "PV-MEPCG diagram language" in content
    assert "<svg" in content


def test_rendering_twice_produces_the_same_png(tmp_path: Path) -> None:
    """Regeneration is deterministic, like every other output in this project.

    Only the PNG is compared: matplotlib stamps a creation date into the SVG
    metadata, so its bytes legitimately differ between two runs of identical
    code.
    """
    first = _render(tmp_path / "one")
    second = _render(tmp_path / "two")
    assert first["png"].read_bytes() == second["png"].read_bytes()


def test_the_registry_row_traces_back_to_the_source_module(rendered: dict[str, Path]) -> None:
    from src.reporting.tables import content_digest

    rows = read_registry(rendered["png"].parent / "diagram_registry.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["diagram_id"] == "F00"
    assert row["source_module"].endswith("src/reporting/diagrams/specimen.py")
    assert row["source_function"] == "build_specimen"
    assert row["source_sha256"] == content_digest(ROOT / "src/reporting/diagrams/specimen.py")[0]
    assert float(row["min_effective_pt"]) >= MIN_EFFECTIVE_PT


def test_the_meta_records_what_it_takes_to_redraw_it(rendered: dict[str, Path]) -> None:
    meta = json.loads(rendered["meta"].read_text(encoding="utf-8"))
    assert meta["framework"] == "PV-MEPCG / PulseVision"
    assert meta["toolchain"].startswith("matplotlib")
    assert set(meta["formats"]) == {"png", "svg"}
    assert meta["command"].startswith("python scripts/05_render_diagrams.py")
    assert meta["source_function"] == "build_specimen"
    assert meta["legibility"]["violations"] == []
    assert meta["legibility"]["overflowing_nodes"] == []
    assert meta["legibility"]["clipped"] == []


def test_re_registering_keeps_the_first_registration_time(tmp_path: Path) -> None:
    _render(tmp_path)
    first = read_registry(tmp_path / "diagram_registry.csv")[0]
    _render(tmp_path)
    second = read_registry(tmp_path / "diagram_registry.csv")[0]
    assert second["first_registered_utc"] == first["first_registered_utc"]
    assert second["last_written_utc"] >= first["last_written_utc"]
    assert second["diagram_number"] == first["diagram_number"]


# ---------------------------------------------------------------------------
# the visual language (T97.4, T97.5)
# ---------------------------------------------------------------------------


def test_every_node_kind_is_told_apart_by_shape_before_colour() -> None:
    report = palette_report()
    assert report["shapes_unique"], "two node kinds share a shape; greyscale loses one of them"
    assert report["edge_linestyles_unique"], "two arrow kinds share a dash pattern"


def test_every_label_clears_wcag_aa_on_its_own_fill() -> None:
    """Luminance-based, so passing here is passing in greyscale."""
    report = palette_report()
    assert report["failures"] == []
    for row in report["text"]:
        assert row["text_contrast"] >= MIN_TEXT_CONTRAST, row


def test_every_shape_is_visible_against_white_paper() -> None:
    """Either the fill or the border has to separate from the page."""
    for style in NODE_STYLES.values():
        if style.shape == "plain":
            continue
        separation = max(
            contrast_ratio(style.facecolor, PAPER), contrast_ratio(style.edgecolor, PAPER)
        )
        assert separation >= 1.25, style.kind + " is invisible on white"


def test_the_specimen_uses_every_kind_the_language_declares() -> None:
    """A kind added without a legend meaning or a shape fails here first."""
    canvas = build_specimen()
    drawable = {kind for kind, style in NODE_STYLES.items() if style.shape != "plain"}
    assert drawable.issubset({node.kind for node in canvas.nodes.values()})
    assert set(EDGE_STYLES).issubset(set(canvas._used_edge_kinds))


def test_nothing_on_the_specimen_prints_below_seven_point(rendered: dict[str, Path]) -> None:
    meta = json.loads(rendered["meta"].read_text(encoding="utf-8"))
    legibility = meta["legibility"]
    assert legibility["min_effective_pt"] >= MIN_EFFECTIVE_PT
    for role, points in legibility["roles"].items():
        assert points >= MIN_EFFECTIVE_PT, role


def test_a_wide_figure_is_measured_at_the_size_it_prints_at() -> None:
    """9 inches into a 6-inch text block is a two-thirds reduction, not a wash."""
    assert effective_point_size(12.0, 9.0) == pytest.approx(8.0)
    assert effective_point_size(9.0, 6.0) == pytest.approx(9.0)
    # Narrower than the page is placed, never stretched up.
    assert effective_point_size(9.0, 3.4) == pytest.approx(9.0)


def test_a_greyscale_print_of_the_specimen_still_has_ink_on_it(
    rendered: dict[str, Path],
) -> None:
    """Convert the render to luminance and check it is a drawing, not a wash.

    A diagram whose fills are all pale and whose lines are all hairlines passes
    every colour check and prints as a blank page. This looks at the pixels.
    """
    import matplotlib.image as mpimg
    import numpy as np

    pixels = mpimg.imread(rendered["png"])[:, :, :3]
    grey = 0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]
    ink = float((grey < 0.5).mean())
    assert 0.003 < ink < 0.40, "ink coverage " + str(ink) + " is a blank or a smudge"
    assert len(np.unique(np.round(grey, 2))) >= 16


# ---------------------------------------------------------------------------
# the checks refuse what they are meant to refuse
# ---------------------------------------------------------------------------


def _tiny_box_canvas() -> DiagramCanvas:
    canvas = DiagramCanvas(columns=4.0, rows=2.0, size="page", legend_rows=0)
    canvas.node(
        "cramped",
        "a label far longer than this box can possibly hold at nine point",
        col=0.2,
        row=0.2,
        width=0.6,
        height=0.25,
    )
    return canvas


def test_a_box_too_small_for_its_label_is_refused(tmp_path: Path) -> None:
    spec = DiagramSpec("F00", "Cramped", "a box that cannot hold its own text")
    with pytest.raises(ValueError, match="too short for their own labels"):
        write_diagram(
            spec,
            _tiny_box_canvas,
            tmp_path,
            registry=tmp_path / "diagram_registry.csv",
            evidence_index=tmp_path / "evidence_index.csv",
        )


def test_an_empty_diagram_is_refused(tmp_path: Path) -> None:
    def nothing() -> DiagramCanvas:
        return DiagramCanvas(columns=4.0, rows=2.0, size="page", legend_rows=0)

    spec = DiagramSpec("F00", "Empty", "no nodes at all")
    with pytest.raises(ValueError, match="no nodes"):
        write_diagram(
            spec,
            nothing,
            tmp_path,
            registry=tmp_path / "diagram_registry.csv",
            evidence_index=tmp_path / "evidence_index.csv",
        )


def test_a_kind_outside_the_language_is_refused() -> None:
    canvas = DiagramCanvas(columns=4.0, rows=2.0, size="page", legend_rows=0)
    with pytest.raises(ValueError, match="unknown node kind"):
        canvas.node("x", "x", kind="cloud", col=0.2, row=0.2, width=1.0, height=0.5)
    canvas.node("a", "a", col=0.2, row=0.2, width=1.0, height=0.5)
    canvas.node("b", "b", col=2.0, row=0.2, width=1.0, height=0.5)
    with pytest.raises(ValueError, match="unknown edge kind"):
        canvas.edge("a", "b", kind="squiggle")


def test_a_builder_for_an_unknown_id_is_refused() -> None:
    with pytest.raises(KeyError):
        diagram("F99")(lambda: DiagramCanvas(columns=1.0, rows=1.0))


# ---------------------------------------------------------------------------
# the catalogue and the one command (T97.6, T97.7)
# ---------------------------------------------------------------------------


def test_the_catalogue_declares_exactly_the_twenty_diagrams() -> None:
    ids = [spec.diagram_id for spec in CATALOGUE]
    assert len(ids) == EXPECTED_COUNT == 20
    assert ids == sorted(ids)
    assert ids == ["F" + str(index).zfill(2) for index in range(1, 21)]
    assert len(set(ids)) == len(ids)


def test_every_declared_diagram_names_the_task_that_asked_for_it() -> None:
    for spec in CATALOGUE:
        assert spec.task.startswith("T9"), spec.diagram_id
        assert len(spec.caption) > 40, spec.diagram_id
        assert spec.slug().startswith(spec.diagram_id + "_")


def test_a_diagram_is_numbered_from_its_own_id() -> None:
    """Not allocated sequentially, so a partial run still numbers correctly."""
    assert diagram_number("F01") == 1
    assert diagram_number("F16") == 16
    with pytest.raises(ValueError):
        diagram_number("F")


def test_pending_diagrams_are_reported_rather_than_skipped() -> None:
    report = catalogue_report()
    builders = load_builders()
    assert set(report["implemented"]) == set(builders) & {s.diagram_id for s in CATALOGUE}
    assert len(report["implemented"]) + len(report["pending"]) == EXPECTED_COUNT
    assert report["complete"] == (len(report["implemented"]) == EXPECTED_COUNT)


def _script_module() -> Any:
    spec = importlib.util.spec_from_file_location("render_diagrams_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses and typing resolve against it
    spec.loader.exec_module(module)
    return module


def test_a_partial_run_is_an_error_not_a_shorter_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--expect N`` is the whole point of the command being one command.

    Rendering nineteen of twenty and exiting zero is the failure mode this
    project treats as fabrication, so it is pinned rather than assumed.
    """

    class _Run:
        def set(self, *args: Any, **kwargs: Any) -> None: ...
        def record_artifact(self, *args: Any, **kwargs: Any) -> None: ...
        def finish(self, *args: Any, **kwargs: Any) -> None: ...

    monkeypatch.setattr("src.utils.run_manifest.start_run", lambda *a, **k: _Run())
    module = _script_module()
    code = module.main(["--only", "F01", "--expect", "20", "--out-dir", str(tmp_path)])
    assert code == 1


def test_the_command_lists_the_catalogue_from_a_clean_process() -> None:
    """Run it for real: a subprocess, no imports already warm in this one."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for spec in CATALOGUE:
        assert spec.diagram_id in result.stdout
    assert "of 20 implemented" in result.stdout


def test_the_command_renders_the_specimen_end_to_end(tmp_path: Path) -> None:
    """T97.7 from clean: one command, no manual step, both formats out."""
    target = tmp_path / "specimen"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--specimen", str(target)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    written = sorted(path.name for path in target.glob("F00_*"))
    assert written == [
        "F00_diagram_language_specimen.meta.json",
        "F00_diagram_language_specimen.png",
        "F00_diagram_language_specimen.svg",
    ]


def test_the_diagram_registry_is_not_the_graph_registry() -> None:
    """Two series, two files, so two processes cannot lose each other's rows.

    Another session registers G11-G35 into ``figure_registry.csv`` while this
    one renders F01-F20; a shared read-modify-write of one CSV drops rows.
    """
    from src.reporting.diagrams.render import DIAGRAM_REGISTRY_FILENAME
    from src.reporting.graphs import REGISTRY_FILENAME

    assert DIAGRAM_REGISTRY_FILENAME != REGISTRY_FILENAME
