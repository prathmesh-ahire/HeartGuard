"""Phase 113's checkable half: the codegen boundary, the equations, the overlay.

T113.7 asks three things. Two of them are checkable here without a browser, and
the third -- that all fifteen equations render -- is checked by the build itself,
because `Equations.tsx` calls KaTeX with `throwOnError`, so a formula that does
not render fails `next build` rather than printing red text on a page.

What this file asserts:

* **every chart consumes only `generated/` data.** No chart component accepts a
  loose number, none imports a payload JSON directly, and none formats a number
  for display (T113.1/T113.2);
* **the cardiac-cycle overlay uses real CirCor segmentation.** The exported
  segments equal the TSV in `dataset/` row for row, the copied audio is byte
  identical to the corpus file, and label 0 is carried as "unannotated" rather
  than folded into a cardiac phase (T113.6);
* **the fifteen equations cross-reference the code that implements them**, and
  the two that depart from the blueprint's typography say so (T113.5);
* the ODC-By notices exist in all three places the licence requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.equations import EQUATIONS, equations_payload
from src.reporting.segmentation import (
    CIRCOR_LICENCE_NAME,
    CIRCOR_ROOT,
    SAMPLE_RECORD_ID,
    SEGMENT_LABELS,
    attribution_line,
    load_sample,
    read_segmentation,
    resolve_sources,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
COMPONENTS = FRONTEND / "components"
APP = FRONTEND / "app"
CHARTS = COMPONENTS / "charts"


@pytest.fixture(scope="module")
def scaffolded() -> None:
    if not (FRONTEND / "package.json").is_file():
        pytest.skip("frontend/ is not scaffolded in this checkout")


def is_client_component(path: Path) -> bool:
    """True when the file carries the `'use client'` directive.

    An exact line match, not a substring search: these modules explain in prose
    why they are NOT client components, and a substring test would read that
    explanation as the thing it warns about.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() in ("'use client';", '"use client";'):
            return True
    return False


@pytest.fixture(scope="module")
def corpus() -> None:
    if not (PROJECT_ROOT / CIRCOR_ROOT).is_dir():
        pytest.skip("dataset/ is not present in this checkout")


# ---------------------------------------------------------------------------
# T113.5 -- fifteen equations, each answerable to the code
# ---------------------------------------------------------------------------


def test_all_fifteen_equations_are_declared() -> None:
    payload = equations_payload()
    assert payload["n_equations"] == 15, "blueprint section 11 lists fifteen formulas"
    assert [item["number"] for item in payload["equations"]] == list(range(1, 16))
    assert len({item["key"] for item in payload["equations"]}) == 15


def test_every_equation_names_the_module_and_symbol_that_implement_it() -> None:
    for equation in EQUATIONS:
        module = PROJECT_ROOT / equation.implemented_in
        assert module.is_file(), (
            "equation " + str(equation.number) + " names a module that is not there"
        )
        body = module.read_text(encoding="utf-8", errors="replace")
        assert equation.implements in body, (
            "equation "
            + str(equation.number)
            + " says "
            + equation.implemented_in
            + " implements it via "
            + equation.implements
            + ", which is not in that file"
        )


def test_an_equation_claiming_an_absent_symbol_refuses_to_export(monkeypatch) -> None:
    """The cross-reference has to actually fail, or it is decoration."""
    from src.reporting import equations as module

    broken = module.Equation(
        1,
        "broken",
        "Claims a symbol that is not there",
        "x = 1",
        "test",
        "src/evaluation/metrics.py",
        "a_function_that_does_not_exist_anywhere",
        (("x", "a symbol"),),
    )
    monkeypatch.setattr(module, "EQUATIONS", (broken,))
    with pytest.raises(ValueError, match="does_not_exist"):
        module.equations_payload()


def test_every_symbol_in_an_equation_is_defined() -> None:
    """A formula with an undefined symbol is a formula the reader cannot use."""
    for equation in EQUATIONS:
        assert equation.symbols, equation.key + " defines no symbols"
        for symbol, meaning in equation.symbols:
            assert symbol.strip(), equation.key + " has an empty symbol"
            assert len(meaning.strip()) > 3, equation.key + " has an empty meaning"


def test_the_two_transcription_departures_are_recorded() -> None:
    """Silently 'fixing' a source document is how a discrepancy hides."""
    noted = {equation.key for equation in EQUATIONS if equation.transcription_note is not None}
    assert "rms" in noted, "the misplaced radical in section 11 is not recorded"
    assert "spectral_centroid" in noted, "the unparenthesised centroid is not recorded"


def test_the_binary_threshold_caveat_travels_with_the_prediction_rule() -> None:
    """T50.4 does not use argmax at 0.5 for binary, and the page must not imply it."""
    prediction = next(equation for equation in EQUATIONS if equation.key == "prediction")
    assert prediction.transcription_note is not None
    assert "threshold" in prediction.transcription_note.lower()


def test_equations_render_on_the_server_not_in_the_browser(scaffolded: None) -> None:
    """KaTeX is 74 kB. It rendered into HTML at build time or it did not run."""
    module = COMPONENTS / "equations" / "Equations.tsx"
    assert not is_client_component(module), (
        "a client boundary here drags KaTeX into the browser; that is what the "
        "server/client split of /design exists to prevent"
    )
    component = module.read_text(encoding="utf-8")
    assert "renderToString" in component
    assert "throwOnError: true" in component, (
        "a malformed formula must fail the build, not render as red text"
    )
    page = APP / "design" / "page.tsx"
    assert not is_client_component(page), (
        "the page rendering EquationList must stay a server component"
    )
    assert "EquationList" in page.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# T113.6 -- the overlay is the dataset's own segmentation
# ---------------------------------------------------------------------------


def test_the_exported_segments_equal_the_tsv_row_for_row(corpus: None) -> None:
    """Not resampled, not smoothed, not inferred. The file's rows."""
    sample = load_sample()
    raw = read_segmentation(sample.tsv_source)
    assert sample.segments == raw
    with sample.tsv_source.open(encoding="utf-8") as handle:
        n_lines = sum(1 for line in handle if line.strip())
    assert len(raw) == n_lines, "rows were dropped or merged on the way out"


def test_the_segment_boundaries_are_contiguous_and_ordered(corpus: None) -> None:
    sample = load_sample()
    previous_end = 0.0
    for segment in sample.segments:
        assert segment["start"] <= segment["end"]
        assert segment["start"] == pytest.approx(previous_end, abs=1e-6), (
            "a gap or overlap appeared between segments, which the TSV does not contain"
        )
        previous_end = segment["end"]
    assert previous_end <= sample.duration_seconds + 0.05


def test_label_zero_is_unannotated_and_not_a_cardiac_phase() -> None:
    """The segmentation declining to label a span is information."""
    assert SEGMENT_LABELS[0]["key"] == "unannotated"
    assert "not a cardiac phase" in SEGMENT_LABELS[0]["description"].lower()
    for label, expected in ((1, "s1"), (2, "systole"), (3, "s2"), (4, "diastole")):
        assert SEGMENT_LABELS[label]["key"] == expected


def test_the_committed_audio_is_byte_identical_to_the_corpus_file(
    scaffolded: None, corpus: None
) -> None:
    copied = FRONTEND / "public" / (SAMPLE_RECORD_ID + ".wav")
    if not copied.is_file():
        pytest.skip("the sample has not been exported")
    original = PROJECT_ROOT / CIRCOR_ROOT / (SAMPLE_RECORD_ID + ".wav")
    assert copied.read_bytes() == original.read_bytes()


def test_the_exported_payload_matches_the_corpus(scaffolded: None, corpus: None) -> None:
    path = FRONTEND / "lib" / "generated" / "segmentation.json"
    if not path.is_file():
        pytest.skip("segmentation.json has not been exported")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = load_sample()
    assert payload["record_id"] == SAMPLE_RECORD_ID
    assert payload["n_segments"] == len(sample.segments)
    assert payload["segments"] == sample.segments
    assert payload["sample_rate_hz"] == sample.sample_rate


def test_the_overlay_rebuilds_on_a_machine_with_no_corpus(scaffolded: None) -> None:
    """A fresh clone and every CI checkout have no `dataset/`.

    Both the audio and its TSV are committed so the export falls back to them.
    Without that, `npm run build` would write a payload saying the sample is
    unavailable while the audio sat in `public/` beside it, and a grader's
    dashboard would silently lose the one figure T113.6 exists to produce.
    """
    from unittest import mock

    from src.reporting import segmentation as module

    if not (FRONTEND / "public" / (SAMPLE_RECORD_ID + ".tsv")).is_file():
        pytest.skip("the sample has not been exported")

    with mock.patch.object(module, "CIRCOR_ROOT", "dataset/deliberately_absent"):
        _wav, _tsv, origin = resolve_sources()
        assert origin == "committed"
        sample = load_sample()
    assert len(sample.segments) > 0
    assert sample.sample_rate == 4000


def test_the_corpus_wins_over_the_committed_copy_when_present(corpus: None) -> None:
    """So the committed copies can never drift from the dataset unnoticed."""
    _wav, _tsv, origin = resolve_sources()
    assert origin == "corpus"


# ---------------------------------------------------------------------------
# the ODC-By notices
# ---------------------------------------------------------------------------


def test_the_licence_notice_sits_beside_the_audio(scaffolded: None) -> None:
    """ODC-By 4.2(d): a WAV cannot carry a notice, so the directory does."""
    notice = FRONTEND / "public" / "NOTICE.md"
    if not notice.is_file():
        pytest.skip("the sample has not been exported")
    body = notice.read_text(encoding="utf-8")
    assert SAMPLE_RECORD_ID in body, "the notice does not name the record it describes"
    assert "ODC-By" in body
    assert "opendatacommons.org" in body
    assert "physionet.org" in body
    assert "sha256" in body.lower()
    assert "dataset sample" in body.lower(), "screening language: not a patient, not a case"


def test_the_readme_carries_the_licence(scaffolded: None) -> None:
    """ODC-By 4.2(b): the licence URI belongs in the documentation too."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "opendatacommons.org/licenses/by/1-0/" in readme
    assert "CirCor DigiScope" in readme
    assert SAMPLE_RECORD_ID in readme


def test_the_attribution_is_rendered_in_the_interface(scaffolded: None) -> None:
    """ODC-By 4.3: the notice travels with the Produced Work, which is the UI."""
    viewer = (COMPONENTS / "audio" / "RecordingViewer.tsx").read_text(encoding="utf-8")
    assert "Contains information from" in viewer
    assert "provenance.licence_uri" in viewer
    assert "provenance.dataset" in viewer
    assert "record " in viewer


def test_the_sample_is_called_a_dataset_sample_never_a_patient(scaffolded: None) -> None:
    """Rule 7, and a privacy line: de-identified screening data is not a case."""
    assert attribution_line().startswith("Contains information from")
    assert CIRCOR_LICENCE_NAME in attribution_line()
    for path in (COMPONENTS / "audio" / "RecordingViewer.tsx", FRONTEND / "public" / "NOTICE.md"):
        if not path.is_file():
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        # "does not diagnose" is the disclaimer and must stay; what is forbidden
        # is a diagnostic CLAIM and any framing of the recording as a person.
        for forbidden in (
            "the patient",
            "this patient",
            "case study",
            "diagnosis of",
            "diagnoses ",
            "is diagnostic",
        ):
            assert forbidden not in lowered, path.name + " uses clinical framing: " + forbidden
        assert "dataset sample" in lowered, path.name + " does not call it a dataset sample"


# ---------------------------------------------------------------------------
# T113.1 / T113.2 -- the charts consume generated data and nothing else
# ---------------------------------------------------------------------------


def test_all_six_chart_types_exist(scaffolded: None) -> None:
    body = (CHARTS / "Charts.tsx").read_text(encoding="utf-8")
    for name in (
        "RocCurve",
        "PrCurve",
        "ConfusionMatrix",
        "GroupedBars",
        "ScatterPlot",
        "CalibrationCurve",
    ):
        assert "export function " + name in body, name + " is missing"


def test_a_chart_takes_a_generated_source_not_loose_numbers(scaffolded: None) -> None:
    """There is no parameter a literal could be passed through."""
    types = (CHARTS / "types.ts").read_text(encoding="utf-8")
    assert "GeneratedTable" in types and "GeneratedFigure" in types
    body = (CHARTS / "Charts.tsx").read_text(encoding="utf-8")
    assert "source?: Source" in body
    assert "numericColumn(" in body, "marks are positioned from the values array"


def test_no_chart_formats_a_number_for_display(scaffolded: None) -> None:
    """Rounding happens once, in Python. `display` is what a page renders."""
    for path in sorted(CHARTS.glob("*.ts*")):
        body = path.read_text(encoding="utf-8")
        for forbidden in ("toFixed(", "toPrecision(", "Math.round("):
            assert forbidden not in body, path.name + " formats a number client-side: " + forbidden


def test_a_missing_value_is_dropped_rather_than_zeroed(scaffolded: None) -> None:
    """A null plotted at the origin is indistinguishable from a real zero."""
    types = (CHARTS / "types.ts").read_text(encoding="utf-8")
    assert "typeof x === 'number' && typeof y === 'number'" in types


def test_a_chart_with_no_data_says_so_rather_than_drawing_an_empty_axis(
    scaffolded: None,
) -> None:
    body = (CHARTS / "Charts.tsx").read_text(encoding="utf-8")
    assert "EmptyState" in body
    assert body.count("<Absent") >= 6, "not every chart handles the absent case"
    assert "NO_RESULTS" in body


def test_the_chart_layer_applies_the_contrast_stroke_rule(scaffolded: None) -> None:
    """theme.json says which fills vanish into the page. Phase 111 required this."""
    body = (CHARTS / "Charts.tsx").read_text(encoding="utf-8")
    assert "needsOutlineOn(" in body
    assert "borderWidth" in body


def test_echarts_is_only_ever_imported_dynamically(scaffolded: None) -> None:
    for path in list(APP.rglob("*.tsx")) + list(COMPONENTS.rglob("*.tsx")):
        body = path.read_text(encoding="utf-8")
        assert "from 'echarts'" not in body, (
            path.name + " imports echarts statically; it is ~300 kB and must stay lazy"
        )


def test_the_figure_download_serves_the_matplotlib_png(scaffolded: None) -> None:
    """T113.4: the canonical 300 dpi file, not a canvas screenshot."""
    body = (CHARTS / "FigureDownload.tsx").read_text(encoding="utf-8")
    assert "figure.png" in body
    assert "figure.dpi" in body
    assert "toDataURL" not in body, "that would export the browser canvas, not the print figure"
    assert "getDataURL" not in body


# ---------------------------------------------------------------------------
# T113.3 -- the results table
# ---------------------------------------------------------------------------


def test_the_table_renders_display_strings_and_sorts_on_values(scaffolded: None) -> None:
    body = (COMPONENTS / "table" / "ResultsTable.tsx").read_text(encoding="utf-8")
    assert "column.display[index]" in body, "cells must come from the formatted strings"
    assert "__sort__" in body, "the numeric array is a sort key only"
    assert "@tanstack/react-table" in body


def test_the_csv_download_writes_what_is_on_screen(scaffolded: None) -> None:
    """A re-serialisation could differ in the last decimal from the table."""
    body = (COMPONENTS / "table" / "ResultsTable.tsx").read_text(encoding="utf-8")
    assert "as_displayed" in body
    assert "toFixed" not in body


def test_the_table_is_sortable_filterable_and_announced(scaffolded: None) -> None:
    body = (COMPONENTS / "table" / "ResultsTable.tsx").read_text(encoding="utf-8")
    assert "getSortedRowModel" in body
    assert "getFilteredRowModel" in body
    assert "aria-sort" in body, "a sortable column must announce its direction"


# ---------------------------------------------------------------------------
# the budget check knows about the new libraries
# ---------------------------------------------------------------------------


def test_the_budget_check_enforces_the_new_libraries() -> None:
    import importlib.util
    import sys

    path = PROJECT_ROOT / "scripts" / "20_check_bundle_budget.py"
    spec = importlib.util.spec_from_file_location("bundle_budget_113", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bundle_budget_113"] = module
    spec.loader.exec_module(module)

    assert "echarts" in module.LAZY_MARKERS
    assert "wavesurfer.js" in module.LAZY_MARKERS
    # KaTeX is a different rule: absent from every chunk, and its output present.
    assert "katex" in module.SERVER_ONLY
    assert module.SERVER_ONLY["katex"]["html_marker"] == "katex-html"
