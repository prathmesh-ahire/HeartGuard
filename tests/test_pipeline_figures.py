"""Phase 96 gate: F11-F20 render, say only what the repository says, and all 20 regenerate.

T96.7 is ``[TEST/MANUAL]``: a reviewer confirms the ten render in both formats
and that all twenty share one visual language. This is the automated part of
that check, and the one T97.7 needs:

* **Both formats, at 300 dpi, above the print threshold, with nothing
  overlapping.** Every refusal the writer can raise is asserted here as a
  recorded measurement -- including the three Phase 96 added after looking at
  the first render: boxes drawn over boxes, boxes off the grid, and free text
  drawn over a box.
* **Nothing is typed.** Each figure's drawn text is compared with the payload,
  config, constant or table it claims to read.
* **All twenty from one command.** ``scripts/05_render_diagrams.py --expect 20``
  is run in a clean subprocess into an empty directory -- the full form of the
  T97.7 gate that could not run until this phase built F11-F20.

"One visual language" is structural rather than a matter of taste here: every
diagram is drawn by the same canvas from the same seven node kinds and five
edge kinds (``language.py``), and the test below asserts no builder reaches
past that vocabulary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.reporting.diagrams.canvas import DiagramCanvas
from src.reporting.diagrams.catalogue import CATALOGUE, load_builders, spec_for
from src.reporting.diagrams.language import EDGE_STYLES, MIN_EFFECTIVE_PT, NODE_STYLES
from src.reporting.diagrams.render import write_diagram

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "05_render_diagrams.py"
PHASE_96 = ["F" + str(index) for index in range(11, 21)]


@pytest.fixture(scope="module")
def builders() -> dict[str, Any]:
    return load_builders()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Path]]:
    target = tmp_path_factory.mktemp("f11_f20")
    available = load_builders()
    return {
        diagram_id: write_diagram(
            spec_for(diagram_id),
            available[diagram_id],
            target,
            registry=target / "diagram_registry.csv",
            evidence_index=target / "evidence_index.csv",
        )
        for diagram_id in PHASE_96
    }


def _drawn(build: Any) -> tuple[str, DiagramCanvas]:
    """Every string on a builder's canvas, joined, and the canvas itself."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    canvas = build()
    assert isinstance(canvas, DiagramCanvas)
    canvas.finish()
    pieces = [text.get_text() for text in canvas.axes.texts]
    pieces += [text.get_text() for text in canvas.figure.texts]
    plt.close(canvas.figure)
    return " ".join(" ".join(piece.split()) for piece in pieces), canvas


def _text(build: Any) -> str:
    return _drawn(build)[0]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_all_ten_have_builders(builders: dict[str, Any]) -> None:
    assert [d for d in PHASE_96 if d not in builders] == []


@pytest.mark.parametrize("diagram_id", PHASE_96)
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
    assert meta["source_module"].endswith("pipeline_figures.py")


@pytest.mark.parametrize("diagram_id", PHASE_96)
def test_each_is_legible_and_nothing_overlaps(
    diagram_id: str, rendered: dict[str, dict[str, Path]]
) -> None:
    legibility = json.loads(rendered[diagram_id]["meta"].read_text(encoding="utf-8"))["legibility"]
    assert legibility["min_effective_pt"] >= MIN_EFFECTIVE_PT
    for key in (
        "violations",
        "clipped",
        "overflowing_nodes",
        "overlapping_nodes",
        "nodes_off_canvas",
        "text_over_nodes",
    ):
        assert legibility[key] == [], diagram_id + " " + key + ": " + str(legibility[key])


def test_every_diagram_speaks_only_the_declared_vocabulary(builders: dict[str, Any]) -> None:
    """One visual language across all twenty: no kind outside language.py."""
    for spec in CATALOGUE:
        _, canvas = _drawn(builders[spec.diagram_id])
        assert set(canvas._used_node_kinds) <= set(NODE_STYLES), spec.diagram_id
        assert set(canvas._used_edge_kinds) <= set(EDGE_STYLES), spec.diagram_id
        assert canvas.label_pt == pytest.approx(
            DiagramCanvas(columns=1.0, rows=1.0, size=(9.0, 5.0)).label_pt
        ), spec.diagram_id


def test_catalogue_objectives_name_real_blueprint_objectives() -> None:
    """The blueprint has six objectives; F11-F20 used to cite O7-O9."""
    import re

    from src.reporting.objectives import OBJECTIVES

    numbers = {objective.number for objective in OBJECTIVES}
    for spec in CATALOGUE:
        cited = {int(n) for n in re.findall(r"O(\d+)", spec.objective)}
        assert cited, spec.diagram_id
        assert cited <= numbers, spec.diagram_id + " cites " + spec.objective


# ---------------------------------------------------------------------------
# the canvas refuses what the first render got wrong
# ---------------------------------------------------------------------------


def _write(build: Any, tmp_path: Path) -> None:
    spec = spec_for("F11")
    write_diagram(
        spec,
        build,
        tmp_path,
        registry=tmp_path / "r.csv",
        evidence_index=tmp_path / "e.csv",
    )


def test_overlapping_boxes_are_refused(tmp_path: Path) -> None:
    def build() -> DiagramCanvas:
        canvas = DiagramCanvas(columns=12.0, rows=6.0, size="wide", legend_rows=1)
        canvas.node("a", "one", col=1.0, row=1.0, width=4.0)
        canvas.node("b", "two", col=3.0, row=1.2, width=4.0)
        return canvas

    with pytest.raises(ValueError, match="on top of each other"):
        _write(build, tmp_path)


def test_a_box_off_the_grid_is_refused(tmp_path: Path) -> None:
    def build() -> DiagramCanvas:
        canvas = DiagramCanvas(columns=12.0, rows=6.0, size="wide", legend_rows=1)
        canvas.node("a", "one", col=1.0, row=5.5, width=4.0)
        return canvas

    with pytest.raises(ValueError, match="outside the grid"):
        _write(build, tmp_path)


def test_text_over_a_box_is_refused(tmp_path: Path) -> None:
    def build() -> DiagramCanvas:
        canvas = DiagramCanvas(columns=12.0, rows=6.0, size="wide", legend_rows=1)
        canvas.node("a", "one", col=1.0, row=1.0, width=4.0)
        canvas.text("a note written straight across the box", col=1.2, row=1.3)
        return canvas

    with pytest.raises(ValueError, match="over a box"):
        _write(build, tmp_path)


def test_a_diamond_is_sized_for_its_narrowing_corners() -> None:
    """A diamond needs more height than a rectangle for the same text."""
    canvas = DiagramCanvas(columns=12.0, rows=6.0, size="wide")
    label, sub = "Validate the upload", "bounds from the module"
    diamond = canvas.fit_height(label, width=3.5, sublabel=sub, kind="decision")
    box = canvas.fit_height(label, width=3.5, sublabel=sub, kind="process")
    assert diamond > 1.8 * box


# ---------------------------------------------------------------------------
# nothing is typed
# ---------------------------------------------------------------------------


def test_f11_reads_protocol_counts_budget_and_final_model(builders: dict[str, Any]) -> None:
    from src.evaluation.tuned import DEFAULT_TRIALS
    from src.reporting.diagrams.pipeline_figures import class_counts, final_binary_model
    from src.reporting.record_index import dataset_summary_payload
    from src.utils.config import load_config

    text = _text(builders["F11"])
    d1 = next(r for r in dataset_summary_payload()["summary"] if r["dataset_source"] == "D1")
    assert str(d1["n_modelled_display"]) in text and str(d1["n_subjects_display"]) in text
    for name, n in class_counts("binary"):
        assert name + " " + format(n, ",") in text
    assert str(DEFAULT_TRIALS) + " trials per model" in text
    assert "Final model " + str(final_binary_model()["selected_model_id"]) in text
    rule = load_config("experiments").require("defaults.selection_rule")
    assert ", then ".join(rule) in text
    for model_id in load_config("experiments").require("experiments.EXP-A2.models"):
        assert model_id in text
    assert "within the PhysioNet 2016 corpus" in " ".join(text.split())


def test_f12_keeps_pascal_a_and_b_apart_with_audited_counts(builders: dict[str, Any]) -> None:
    from src.reporting.diagrams.pipeline_figures import class_counts

    text, canvas = _drawn(builders["F12"])
    for task in ("pascal_a", "pascal_b"):
        for name, n in class_counts(task):
            assert name + " " + str(n) in text
    assert "forbidden" in canvas._used_edge_kinds and "never merged" in text
    assert "not a cardiac class" in text


def test_f13_draws_every_rule_the_aggregator_implements(builders: dict[str, Any]) -> None:
    from src.evaluation.aggregation import AGGREGATION_RULES

    text = _text(builders["F13"])
    for rule in AGGREGATION_RULES:
        assert rule in text
    assert "excluded: Phc" in text


def test_f13_rule_descriptions_match_the_aggregator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two recordings, 0.7 (called positive) and 0.2: max and any say yes, mean says no.

    That is what the three captions claim -- a maximum and a mean of the
    probabilities, thresholded, and a union over decisions -- run through the
    real function with the patient lookup stubbed.
    """
    import pandas as pd

    from src.evaluation import aggregation

    monkeypatch.setattr(aggregation, "_patient_of", lambda uids: ["p1"] * len(list(uids)))
    frame = pd.DataFrame(
        {
            "record_uid": ["r1", "r2"],
            "model_id": ["M"] * 2,
            "fold_label": ["f"] * 2,
            "y_true": [1, 1],
            "y_pred": [1, 0],
            "proba_1": [0.7, 0.2],
        }
    )
    decided = {
        rule: int(aggregation.aggregate_predictions(frame, rule=rule)["y_pred"].iloc[0])
        for rule in aggregation.AGGREGATION_RULES
    }
    assert decided == {"max": 1, "mean": 0, "any_present": 1}


def test_f14_reads_the_population_mismatch_written_before_the_metric(
    builders: dict[str, Any],
) -> None:
    from src.reporting.diagrams.pipeline_figures import population_mismatch

    meta = population_mismatch()
    text = _text(builders["F14"])
    assert "median age " + format(float(meta["train"]["age_median_years"]), ".0f") in text
    assert format(100.0 * float(meta["train"]["share_under_18"]), ".1f") + "% under 18" in text
    assert format(100.0 * float(meta["test"]["share_paediatric_of_recorded"]), ".1f") in text
    assert "retuning allowed: false" in text
    assert meta["written_before_any_metric"] is True
    assert "written before any metric: true" in text


def test_f15_levels_bands_and_groups_are_the_module_constants(builders: dict[str, Any]) -> None:
    from src.evaluation.duration import TRUNCATION_SECONDS
    from src.evaluation.robustness import QUALITY_GROUPS, SNR_LEVELS
    from src.utils.config import load_config

    text = _text(builders["F15"])
    for group in QUALITY_GROUPS:
        assert group in text
    for level in SNR_LEVELS:
        assert ("clean" if level == float("inf") else format(level, "g") + " dB") in text
    for seconds in TRUNCATION_SECONDS:
        assert ("full length" if seconds == float("inf") else format(seconds, "g") + " s") in text
    bands = load_config("signal").require("integrity.duration_bands")
    assert "under " + format(bands["short_below"], "g") + " s" in text
    assert "over " + format(bands["long_above"], "g") + " s" in text


def test_f16_routes_are_the_applications_and_only_predict_computes(
    builders: dict[str, Any],
) -> None:
    """T96.4: the codegen boundary, and the live /predict endpoint.

    The app has three POST routes, all under /predict (upload, stored sample,
    patient); every other route is a read-only status or asset route.
    """
    from src.reporting.diagrams.pipeline_figures import api_routes
    from src.reporting.frontend_export import GENERATED_FILES

    routes = api_routes()
    text, canvas = _drawn(builders["F16"])
    posts = [path for method, path in routes if method == "POST"]
    # Phase 117 added POST /report/sample, which scores the recording through
    # the same `predict_recording` pass before rendering its DOCX. The invariant
    # is "every POST route runs the predictor and nothing else computes", so it
    # is named explicitly rather than widened to any path.
    assert posts and all(path.startswith("/predict") or path == "/report/sample" for path in posts)
    for _, path in routes:
        assert path in text
    assert str(len(GENERATED_FILES)) + " generated files" in text
    assert "frontend/lib/generated/" in text and "outputs/" in text
    assert "scripts/16_check_no_hardcoded_metrics.py" in text
    assert {"derive", "forbidden"} <= set(canvas._used_edge_kinds)


def test_f17_bounds_are_the_predictor_constants(builders: dict[str, Any]) -> None:
    from src.inference.predictor import (
        LOW_CONFIDENCE_MARGIN,
        MAX_DURATION_SECONDS,
        MIN_DURATION_SECONDS,
        TASKS,
    )

    text = _text(builders["F17"])
    assert format(MIN_DURATION_SECONDS, "g") + "-" + format(MAX_DURATION_SECONDS, "g") in text
    assert format(LOW_CONFIDENCE_MARGIN, ".2f") in text
    for task in TASKS:
        assert task in text
    assert "disclaimer" in text


def test_f18_traces_every_objective_to_its_modules_and_tables(builders: dict[str, Any]) -> None:
    from src.reporting.conclusion_tables import OBJECTIVE_EVIDENCE
    from src.reporting.objectives import OBJECTIVES

    text = _text(builders["F18"])
    for objective in OBJECTIVES:
        assert objective.handle in text
        for module in objective.modules:
            assert module.rsplit("/", 1)[-1] in text
    for row in OBJECTIVE_EVIDENCE:
        for table in row.tables:
            assert table in text


def test_f19_deltas_are_t19s_to_three_places(builders: dict[str, Any]) -> None:
    from src.reporting.diagrams.pipeline_figures import t19_stages

    stages = t19_stages()
    text = _text(builders["F19"])
    for stage in stages:
        assert str(stage["comparison"]) in text
    for stage in stages[1:]:
        assert format(float(stage["sensitivity_incremental"]), "+.3f") in text
        assert format(float(stage["balanced_accuracy_incremental"]), "+.3f") in text
    best = max(stages[1:], key=lambda s: float(s["sensitivity_incremental"]))
    assert "'" + str(best["comparison"]) + "'" in text


def test_f20_verdicts_are_t30s_own_first_clauses(builders: dict[str, Any]) -> None:
    from src.reporting.diagrams.pipeline_figures import t30_rows, verdict_headline

    text = _text(builders["F20"])
    for row in t30_rows():
        assert str(row["handle"]) in text
        assert verdict_headline(str(row["verdict"])) in text


# ---------------------------------------------------------------------------
# T97.7, the full form: all twenty from one command, from clean
# ---------------------------------------------------------------------------


def test_all_twenty_regenerate_from_one_command(tmp_path: Path) -> None:
    from src.reporting.diagrams.render import read_registry

    target = tmp_path / "clean"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--expect",
            "20",
            "--out-dir",
            str(target),
            "--evidence-index",
            str(tmp_path / "evidence_index.csv"),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "rendered 20 of 20" in result.stdout
    rows = read_registry(target / "diagram_registry.csv")
    assert [int(r["diagram_number"]) for r in rows] == list(range(1, 21))
    assert sorted(r["diagram_id"] for r in rows) == [s.diagram_id for s in CATALOGUE]
    for spec in CATALOGUE:
        for suffix in (".png", ".svg", ".meta.json"):
            assert (target / (spec.slug() + suffix)).is_file(), spec.diagram_id + suffix


def test_no_committed_diagram_meta_records_a_machine_path() -> None:
    """The F-series equivalent of the Phase 94 G-series check.

    All twenty F metas recorded ``D:/Projects/...`` under ``written`` until Phase
    96: a path that resolves on one machine and fails any re-check on CI.
    """
    from src.reporting.diagrams.render import diagrams_dir

    metas = sorted(diagrams_dir().glob("F[0-9][0-9]_*.meta.json"))
    assert len(metas) == len(CATALOGUE)
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for value in (meta.get("written") or {}).values():
            assert ":" not in str(value) and not Path(str(value)).is_absolute(), (
                meta_path.name + " records " + str(value)
            )
