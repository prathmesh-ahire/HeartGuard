"""The T90.6 / T90.7 gate: no PNG without its CSV, and numbers that stay put.

Two claims, both structural rather than stylistic.

**A figure and its data are written together (T90.2).** Every generator must
emit a non-empty PNG *and* the exact frame that produced it. The test proves the
correspondence by reading the written CSV back and comparing it to the frame the
graph was built from -- not by checking that a file merely exists.

**Figure numbers survive a regeneration (T90.3).** A thesis that says "see
Figure 7" has to still mean the same figure after the figures are rebuilt, in a
different order, with a new one inserted in the middle. That is asserted here by
doing exactly those three things and re-reading the registry.

Everything runs on synthetic frames in ``tmp_path``, so the gate needs no
dataset, no pipeline output and no committed artifact -- it exercises the engine
itself. ``tests/test_data_graphs.py`` covers the real G01-G10.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import graphs as gr


def _frame(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [f"class_{i}" for i in range(rows)],
            "n_records": np.arange(10, 10 + rows) * 7,
            "share": np.linspace(0.1, 0.9, rows),
        }
    )


def _spec(figure_id: str = "X01", title: str | None = None) -> gr.GraphSpec:
    return gr.GraphSpec(
        figure_id=figure_id,
        title=title or ("Engine Self Check " + figure_id),
        caption="Synthetic figure used by the T90.7 gate. Not a result.",
        sources=("outputs/01_dataset_audit/class_distribution.csv",),
        objective="O0 (engine self-check)",
        command="pytest tests/test_graph_engine.py",
    )


def _bar_graph(figure_id: str = "X01", rows: int = 5) -> gr.Graph:
    def draw(data: pd.DataFrame):
        fig, axis = gr.subplots("double")
        axis.bar(
            range(len(data)), data["n_records"], color=[gr.class_color(i) for i in range(len(data))]
        )
        axis.set_xticks(range(len(data)))
        axis.set_xticklabels(list(data["label"]))
        return fig

    return gr.Graph(spec=_spec(figure_id), frame=_frame(rows), draw=draw)


# ---------------------------------------------------------------------------
# T90.2 / T90.6 -- a non-empty PNG and the exact CSV behind it
# ---------------------------------------------------------------------------


def test_every_write_emits_a_non_empty_png_and_its_source_csv(tmp_path: Path) -> None:
    graph = _bar_graph()
    written = gr.write_graph(graph, tmp_path, evidence_index=tmp_path / "evidence_index.csv")

    assert written["png"].is_file() and written["png"].stat().st_size > 0
    assert written["csv"].is_file() and written["csv"].stat().st_size > 0
    assert written["meta"].is_file()
    # A real 300 dpi PNG, not a stub: the smallest figure here is tens of KB.
    assert written["png"].stat().st_size > 5_000
    assert written["png"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_written_csv_is_the_frame_that_was_drawn(tmp_path: Path) -> None:
    graph = _bar_graph()
    written = gr.write_graph(graph, tmp_path, evidence_index=tmp_path / "evidence_index.csv")
    back = pd.read_csv(written["csv"])
    pd.testing.assert_frame_equal(back, graph.frame, check_dtype=False)


def test_the_draw_function_receives_the_frame_that_was_written(tmp_path: Path) -> None:
    """Not 'a frame like it' -- the same object. That is what makes the CSV proof."""
    seen: list[pd.DataFrame] = []

    def draw(data: pd.DataFrame):
        seen.append(data)
        fig, axis = gr.subplots("double")
        axis.plot(data["n_records"])
        return fig

    graph = gr.Graph(spec=_spec(), frame=_frame(), draw=draw)
    gr.write_graph(graph, tmp_path, evidence_index=tmp_path / "evidence_index.csv")
    assert len(seen) == 1
    assert seen[0] is graph.frame


def test_an_empty_frame_is_refused_rather_than_drawn(tmp_path: Path) -> None:
    graph = gr.Graph(
        spec=_spec(),
        frame=pd.DataFrame({"label": [], "n_records": []}),
        draw=lambda _: None,
    )
    with pytest.raises(ValueError, match="empty frame"):
        gr.write_graph(graph, tmp_path, evidence_index=tmp_path / "evidence_index.csv")


def test_a_draw_that_returns_nothing_is_an_error_not_a_blank_png(
    tmp_path: Path,
) -> None:
    graph = gr.Graph(spec=_spec(), frame=_frame(), draw=lambda _: None)
    with pytest.raises(ValueError, match="returned no figure"):
        gr.write_graph(graph, tmp_path, evidence_index=tmp_path / "evidence_index.csv")


def test_an_unknown_format_or_profile_is_refused(tmp_path: Path) -> None:
    graph = _bar_graph()
    with pytest.raises(ValueError, match="unknown graph format"):
        gr.write_graph(graph, tmp_path, formats=("png", "gif"))
    with pytest.raises(ValueError, match="unknown profile"):
        gr.write_graph(graph, tmp_path, profile="poster")


# ---------------------------------------------------------------------------
# T90.4 -- the SVG export path
# ---------------------------------------------------------------------------


def test_svg_export_produces_a_real_editable_svg(tmp_path: Path) -> None:
    written = gr.write_graph(
        _bar_graph(),
        tmp_path,
        formats=("png", "svg"),
        evidence_index=tmp_path / "evidence_index.csv",
    )
    assert written["svg"].is_file() and written["svg"].stat().st_size > 0
    body = written["svg"].read_text(encoding="utf-8", errors="ignore")
    assert "<svg" in body
    # Editable means vector, not one embedded raster.
    assert body.count("<path") > 5


# ---------------------------------------------------------------------------
# T90.5 -- the print profile
# ---------------------------------------------------------------------------


def test_print_profile_is_an_overlay_not_a_replacement() -> None:
    screen = gr.profile_rc("screen")
    printed = gr.profile_rc("print")
    # The style decisions made once in plot_style.py still hold under print.
    assert printed["font.family"] == screen["font.family"]
    assert printed["savefig.dpi"] == screen["savefig.dpi"] == gr.DPI
    # And the print-specific ones are applied.
    assert printed["savefig.facecolor"] == "white"
    assert printed["savefig.transparent"] is False
    assert printed["pdf.fonttype"] == 42
    assert printed["lines.linewidth"] > screen["lines.linewidth"]


def test_print_profile_renders_the_same_numbers_on_a_light_ground(
    tmp_path: Path,
) -> None:
    screen_dir, print_dir = tmp_path / "screen", tmp_path / "print"
    index = tmp_path / "evidence_index.csv"
    on_screen = gr.write_graph(_bar_graph(), screen_dir, evidence_index=index)
    in_print = gr.write_graph(_bar_graph(), print_dir, profile="print", evidence_index=index)

    # Same data, different ink.
    pd.testing.assert_frame_equal(pd.read_csv(on_screen["csv"]), pd.read_csv(in_print["csv"]))
    assert in_print["png"].stat().st_size > 5_000
    assert json.loads(in_print["meta"].read_text(encoding="utf-8"))["profile"] == "print"


# ---------------------------------------------------------------------------
# T90.3 / T90.7 -- numbering stays stable across a regeneration
# ---------------------------------------------------------------------------


def test_a_first_batch_numbers_in_id_order_whatever_order_it_is_given(
    tmp_path: Path,
) -> None:
    graphs = [_bar_graph(fid) for fid in ("X03", "X01", "X02")]
    gr.write_graphs(graphs, tmp_path, evidence_index=tmp_path / "evidence_index.csv")
    registry = {
        r["figure_id"]: int(r["figure_number"])
        for r in gr.read_registry(gr.registry_path(tmp_path))
    }
    assert registry == {"X01": 1, "X02": 2, "X03": 3}


def test_numbers_survive_a_regeneration_in_a_different_order(tmp_path: Path) -> None:
    index = tmp_path / "evidence_index.csv"
    gr.write_graphs([_bar_graph(f) for f in ("X01", "X02", "X03")], tmp_path, evidence_index=index)
    before = {
        r["figure_id"]: r["figure_number"] for r in gr.read_registry(gr.registry_path(tmp_path))
    }

    # Rebuild, reversed, with different data in each frame.
    gr.write_graphs(
        [_bar_graph(f, rows=8) for f in ("X03", "X02", "X01")], tmp_path, evidence_index=index
    )
    after = {
        r["figure_id"]: r["figure_number"] for r in gr.read_registry(gr.registry_path(tmp_path))
    }
    assert after == before, "regenerating renumbered the figures"


def test_inserting_a_new_figure_does_not_renumber_the_existing_ones(
    tmp_path: Path,
) -> None:
    index = tmp_path / "evidence_index.csv"
    gr.write_graphs([_bar_graph(f) for f in ("X01", "X02", "X03")], tmp_path, evidence_index=index)
    # X02b sorts between X02 and X03 but must NOT take number 3.
    gr.write_graph(_bar_graph("X02b"), tmp_path, evidence_index=index)

    registry = {
        r["figure_id"]: int(r["figure_number"])
        for r in gr.read_registry(gr.registry_path(tmp_path))
    }
    assert registry["X01"] == 1
    assert registry["X02"] == 2
    assert registry["X03"] == 3
    assert registry["X02b"] == 4


def test_a_regenerated_figure_keeps_its_first_registration_timestamp(
    tmp_path: Path,
) -> None:
    index = tmp_path / "evidence_index.csv"
    gr.write_graph(_bar_graph(), tmp_path, evidence_index=index)
    first = gr.read_registry(gr.registry_path(tmp_path))[0]
    gr.write_graph(_bar_graph(rows=9), tmp_path, evidence_index=index)
    second = gr.read_registry(gr.registry_path(tmp_path))[0]

    assert second["first_registered_utc"] == first["first_registered_utc"]
    assert second["last_written_utc"] >= first["last_written_utc"]


def test_figure_number_for_reports_none_before_registration(tmp_path: Path) -> None:
    registry = gr.registry_path(tmp_path)
    assert gr.figure_number_for("X01", registry) is None
    gr.write_graph(_bar_graph(), tmp_path, evidence_index=tmp_path / "e.csv")
    assert gr.figure_number_for("X01", registry) == 1


def test_the_registry_records_the_caption_and_both_source_layers(
    tmp_path: Path,
) -> None:
    written = gr.write_graph(_bar_graph(), tmp_path, evidence_index=tmp_path / "evidence_index.csv")
    row = gr.read_registry(gr.registry_path(tmp_path))[0]
    assert row["caption"].startswith("Synthetic figure")
    assert row["filename"] == written["png"].name
    assert row["source_csv"] == written["csv"].name
    assert "class_distribution.csv" in row["upstream_sources"]


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_meta_fingerprints_the_plotted_csv_and_the_upstream_sources(
    tmp_path: Path,
) -> None:
    written = gr.write_graph(_bar_graph(), tmp_path, evidence_index=tmp_path / "evidence_index.csv")
    meta = json.loads(written["meta"].read_text(encoding="utf-8"))

    assert meta["figure_id"] == "X01"
    assert meta["figure_number"] == 1
    assert meta["dpi"] == gr.DPI
    assert meta["framework"] == "PV-MEPCG / PulseVision"
    assert meta["exp_id"], "a blank experiment id reads the same as a missing one"
    assert meta["plotted_rows"] == 5
    # The digest is newline-normalized, not a raw byte hash: the same committed
    # CSV is CRLF in a Windows working tree and LF in the repository. See
    # tables.content_digest and the CRLF regression tests in test_table_engine.
    digest, method = gr.content_digest(written["csv"])
    assert meta["plotted_csv_sha256"] == digest
    assert meta["plotted_csv_digest_method"] == method
    assert len(meta["sources"]) == 1


def test_the_source_stamp_names_the_plotted_csv_and_never_runs_as_one_long_line(
    tmp_path: Path,
) -> None:
    """A wide text box expands the CANVAS under bbox_inches='tight' (Phase 28)."""
    spec = gr.GraphSpec(
        figure_id="X09",
        title="Many Sources",
        caption="Checks the stamp wraps.",
        sources=tuple(
            f"outputs/01_dataset_audit/a_very_long_source_name_{i}.csv" for i in range(4)
        ),
    )
    stamp = gr._source_stamp(spec, "X09_many_sources.csv")
    lines = stamp.splitlines()
    assert len(lines) == 6  # title line + "derived from:" + four sources
    assert max(len(line) for line in lines) < 80
    assert "X09_many_sources.csv" in lines[0]


def test_evidence_registration_lands_in_the_given_index(tmp_path: Path) -> None:
    index = tmp_path / "evidence_index.csv"
    written = gr.write_graph(_bar_graph(), tmp_path, evidence_index=index)
    rows = pd.read_csv(index)
    row = rows[rows["evidence_id"] == "X01"].iloc[0]
    assert row["status"] == "ok"
    assert Path(str(row["source_data"])).name == written["csv"].name
