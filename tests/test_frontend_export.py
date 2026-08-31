"""The T109.7 gate: strict JSON, no NaN token, and a manifest that matches the run.

This is the gate on the correctness boundary, so it checks the boundary rather
than the plumbing:

* every emitted file parses as **strict** JSON and carries no bare ``NaN`` or
  ``Infinity`` token -- Python's parser accepts all three, a browser's does not,
  so parsing alone would pass a file the frontend cannot read;
* the manifest names the commit, the run and the exact sources that produced
  what is on disk, verified by re-hashing them;
* every displayed value is the Python-formatted one, re-derived here from the
  source CSV rather than compared against a literal;
* a NaN reaches the browser as ``n/a`` and ``null``, never as ``0``.

The metric guard rail (`scripts/16_check_no_hardcoded_metrics.py`) is tested
here too, against fixture files written in ``tmp_path``: a guard that has never
been shown to fail on a real violation is not a guard.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.reporting import frontend_export as fe
from src.reporting.tables import content_digest, format_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GUARD = PROJECT_ROOT / "scripts" / "16_check_no_hardcoded_metrics.py"


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> fe.ExportResult:
    """A real export into a throwaway directory, or a skip on a bare checkout."""
    target = tmp_path_factory.mktemp("generated")
    public = tmp_path_factory.mktemp("public")
    try:
        return fe.export_all(
            out_dir=target, public_dir=public, command="pytest tests/test_frontend_export.py"
        )
    except (ValueError, FileNotFoundError) as error:
        pytest.skip("nothing to export in this checkout (" + str(error)[:80] + ")")


# ---------------------------------------------------------------------------
# T109.7 -- strict JSON, no NaN token
# ---------------------------------------------------------------------------


def test_every_declared_file_is_emitted_and_non_empty(exported: fe.ExportResult) -> None:
    for name in fe.GENERATED_FILES:
        path = exported.generated / name
        assert path.is_file(), name + " was not emitted"
        assert path.stat().st_size > 0, name + " is empty"


def test_every_json_file_parses_strictly(exported: fe.ExportResult) -> None:
    for name in fe.GENERATED_FILES:
        if name.endswith(".json"):
            fe.verify_strict_json(exported.generated / name)


def test_no_emitted_json_contains_a_bare_nan_or_infinity_token(
    exported: fe.ExportResult,
) -> None:
    """``json.loads`` accepts these; ``JSON.parse`` in a browser does not."""
    for name in fe.GENERATED_FILES:
        if not name.endswith(".json"):
            continue
        text = (exported.generated / name).read_text(encoding="utf-8")
        for token in ("NaN", "Infinity", "-Infinity"):
            assert not re.search(r"(?<![\"\w])" + re.escape(token) + r"(?![\"\w])", text), (
                name + " contains a bare " + token
            )


def test_verify_strict_json_rejects_a_file_the_browser_would_reject(
    tmp_path: Path,
) -> None:
    """The verifier itself must fail on the thing it exists to catch."""
    bad = tmp_path / "bad.json"
    bad.write_text('{"sensitivity": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"non-finite|bare token"):
        fe.verify_strict_json(bad)

    good = tmp_path / "good.json"
    good.write_text('{"sensitivity": null}', encoding="utf-8")
    assert fe.verify_strict_json(good)["bytes"] > 0


def test_a_nan_becomes_n_a_for_display_and_null_for_values_never_zero() -> None:
    payload = fe.column_payload("sensitivity", pd.Series([0.856789, np.nan, np.inf]))
    assert payload["kind"] == "metric"
    assert payload["display"] == ["0.857", "n/a", "inf"]
    assert payload["values"] == [0.856789, None, None]
    assert 0 not in [v for v in payload["values"] if v is not None]


def test_a_text_column_has_null_values_rather_than_an_empty_list() -> None:
    """`null` says 'not a number'; `[]` says 'no data'. They are different."""
    payload = fe.column_payload("model_name", pd.Series(["M1", "M3"]))
    assert payload["values"] is None
    assert payload["display"] == ["M1", "M3"]


# ---------------------------------------------------------------------------
# T109.4 -- the manifest matches the run that produced the artifacts
# ---------------------------------------------------------------------------


def test_manifest_names_the_commit_the_run_and_the_framework(
    exported: fe.ExportResult,
) -> None:
    from src.utils.run_manifest import git_info

    manifest = json.loads((exported.generated / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["framework"] == "PV-MEPCG / PulseVision"
    assert manifest["git_commit"] == git_info().get("commit")
    assert manifest["git_branch"] == git_info().get("branch")
    # Parses as a real timestamp, not a formatted string nobody can compare.
    assert datetime.fromisoformat(manifest["exported_utc"])
    assert manifest["n_tables"] == len(exported.tables)
    assert manifest["n_figures"] == len(exported.figures)


def test_manifest_sources_still_hash_to_what_was_recorded(
    exported: fe.ExportResult,
) -> None:
    manifest = json.loads((exported.generated / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"], "the export recorded no sources"
    for source in manifest["sources"]:
        path = PROJECT_ROOT / source["path"]
        assert path.is_file(), source["path"] + " is named but absent"
        digest, method = content_digest(path)
        assert source["sha256"] == digest, source["path"] + " changed after the export"
        assert source["digest_method"] == method


def test_the_exporter_never_reads_a_gitignored_artifact(
    exported: fe.ExportResult,
) -> None:
    """Only committed types, so a CI build reads exactly what a local one does."""
    manifest = json.loads((exported.generated / "manifest.json").read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        suffix = Path(source["path"]).suffix.lower()
        assert suffix in fe.READABLE_SUFFIXES, source["path"] + " is not a readable type"
        assert suffix != ".parquet"


def test_the_results_directories_are_excluded_with_a_stated_reason(
    exported: fe.ExportResult,
) -> None:
    manifest = json.loads((exported.generated / "manifest.json").read_text(encoding="utf-8"))
    excluded = {entry["dir"]: entry["reason"] for entry in manifest["excluded_dirs"]}
    assert set(excluded) == set(fe.RESULT_DIRS)
    for reason in excluded.values():
        assert len(reason) > 40, "an exclusion without a reason is a silent omission"
    for key in fe.RESULT_DIRS:
        assert key not in manifest["source_dirs"]


# ---------------------------------------------------------------------------
# T109.2 -- the formatting is Python's, and it matches the source CSV
# ---------------------------------------------------------------------------


def test_every_displayed_value_is_the_python_formatted_source_value(
    exported: fe.ExportResult,
) -> None:
    payload = json.loads((exported.generated / "tables.json").read_text(encoding="utf-8"))
    assert payload, "no tables were exported"

    for table_id, table in payload.items():
        source = pd.read_csv(PROJECT_ROOT / table["source_csv"])
        assert table["n_rows"] == len(source)
        for column in table["columns"]:
            expected = [format_value(v, column["kind"]) for v in source[column["name"]]]
            assert column["display"] == expected, (
                table_id + "." + column["name"] + " is not the formatted source value"
            )


def test_a_metric_column_is_rounded_to_three_places_and_a_count_is_separated(
    exported: fe.ExportResult,
) -> None:
    """The T85.6 rules, asserted on real exported output rather than a fixture."""
    payload = json.loads((exported.generated / "tables.json").read_text(encoding="utf-8"))
    checked = 0
    for table in payload.values():
        for column in table["columns"]:
            if column["kind"] == "metric":
                assert column["places"] == 3
                for shown in column["display"]:
                    assert shown == "n/a" or re.fullmatch(r"-?\d+\.\d{3}", shown), shown
                checked += 1
            if column["kind"] == "count":
                assert column["places"] == 0
                for shown in column["display"]:
                    assert shown == "n/a" or re.fullmatch(r"-?[\d,]+", shown), shown
                checked += 1
    assert checked > 0, "no metric or count column was exported to check"


def test_values_carries_full_precision_so_charts_do_not_re_round(
    exported: fe.ExportResult,
) -> None:
    payload = json.loads((exported.generated / "tables.json").read_text(encoding="utf-8"))
    for table in payload.values():
        source = pd.read_csv(PROJECT_ROOT / table["source_csv"])
        for column in table["columns"]:
            if column["values"] is None:
                continue
            for shipped, original in zip(column["values"], source[column["name"]], strict=True):
                if shipped is None:
                    assert not np.isfinite(float(original)), (
                        "a finite source value was shipped as null"
                    )
                else:
                    assert shipped == pytest.approx(float(original), abs=0.0)


# ---------------------------------------------------------------------------
# T109.5 -- the evidence index maps a displayed value back to its CSV
# ---------------------------------------------------------------------------


def test_every_exported_column_and_figure_has_an_evidence_entry(
    exported: fe.ExportResult,
) -> None:
    evidence = json.loads((exported.generated / "evidence.json").read_text(encoding="utf-8"))
    keys = {entry["key"] for entry in evidence}

    tables = json.loads((exported.generated / "tables.json").read_text(encoding="utf-8"))
    for table_id, table in tables.items():
        for column in table["columns"]:
            assert "tables." + table_id + "." + column["name"] in keys

    figures = json.loads((exported.generated / "figures.json").read_text(encoding="utf-8"))
    for figure_id in figures:
        assert "figures." + figure_id in keys


def test_every_evidence_entry_resolves_to_a_file_with_the_recorded_digest(
    exported: fe.ExportResult,
) -> None:
    evidence = json.loads((exported.generated / "evidence.json").read_text(encoding="utf-8"))
    assert evidence
    for entry in evidence:
        path = PROJECT_ROOT / entry["generated_from"]
        assert path.is_file(), entry["key"] + " points at a missing file"
        assert content_digest(path)[0] == entry["generated_from_sha256"]
        assert entry["upstream_sources"], entry["key"] + " has no upstream source"


# ---------------------------------------------------------------------------
# figures: the PNG is served, and an omission is stated
# ---------------------------------------------------------------------------


def test_each_figure_serves_its_canonical_png_from_public(
    exported: fe.ExportResult,
) -> None:
    figures = json.loads((exported.generated / "figures.json").read_text(encoding="utf-8"))
    assert figures
    for figure_id, figure in figures.items():
        assert figure["png"], figure_id + " has no PNG to download (T113.4)"
        assert figure["png"].startswith("/figures/")
        assert figure["dpi"] == 300


def test_an_omitted_frame_says_so_and_says_why(exported: fe.ExportResult) -> None:
    """A silent omission is indistinguishable from data that never existed."""
    figures = json.loads((exported.generated / "figures.json").read_text(encoding="utf-8"))
    for figure_id, figure in figures.items():
        if figure["data_omitted"]:
            assert figure["columns"] == []
            assert figure["n_rows"] > 0, figure_id + " omitted its data and its row count"
            assert "exceeds" in (figure["data_omitted_reason"] or "")
            assert figure["source_csv"] in (figure["data_omitted_reason"] or "")
        else:
            assert figure["data_omitted_reason"] is None
            assert figure["columns"], figure_id + " inlined nothing but claims it did"


def test_the_inline_budgets_are_actually_applied(exported: fe.ExportResult) -> None:
    figures = json.loads((exported.generated / "figures.json").read_text(encoding="utf-8"))
    for figure_id, figure in figures.items():
        cells = figure["n_rows"] * max(len(figure["columns"]), 1)
        too_big = figure["n_rows"] > fe.MAX_INLINE_ROWS or cells > fe.MAX_INLINE_CELLS
        if not figure["data_omitted"]:
            assert not too_big, figure_id + " was inlined past the budget"


# ---------------------------------------------------------------------------
# T109.6 -- TypeScript that breaks the build rather than the page
# ---------------------------------------------------------------------------


def test_typescript_declares_every_payload_and_index_assigns_without_a_cast(
    exported: fe.ExportResult,
) -> None:
    types = (exported.generated / "types.ts").read_text(encoding="utf-8")
    for name in (
        "GeneratedColumn",
        "GeneratedTable",
        "GeneratedFigure",
        "GeneratedManifest",
        "GeneratedEvidenceEntry",
    ):
        assert "export interface " + name in types, name + " is not declared"

    # Every emitted module, not only the barrel. Phase 114 moved the four large
    # payloads out of `index.ts` into their own modules -- `tables.json` and
    # `figures.json` among them -- because the barrel is a single module
    # imported by the root layout, so anything in it is bundled into all fifteen
    # routes. The property this test protects is unchanged and is NOT weakened:
    # every payload is assigned to its declared type WITHOUT a cast. It is now
    # checked across the modules rather than in one file, because that is where
    # the assignments are.
    modules = {
        path.name: path.read_text(encoding="utf-8")
        for path in exported.generated.glob("*.ts")
        if path.name != "types.ts"
    }
    assert "index.ts" in modules
    joined = "\n".join(modules.values())

    # A cast would defeat the whole point: tsc must structurally check the JSON.
    for forbidden in (" as GeneratedTable", " as GeneratedFigure", " as unknown", " as any"):
        assert forbidden not in joined, "a cast in the generated TypeScript: " + forbidden

    assert "export const manifest: GeneratedManifest = manifestJson;" in modules["index.ts"]
    assert "export const tables: Record<string, GeneratedTable> = payload;" in modules["tables.ts"]
    assert (
        "export const figures: Record<string, GeneratedFigure> = payload;" in modules["figures.ts"]
    )
    assert "export const records: GeneratedRecordIndex = payload;" in modules["records.ts"]

    # Every JSON the exporter wrote is imported by exactly one module, so no
    # payload is silently duplicated into two chunks.
    for json_file in exported.generated.glob("*.json"):
        importers = [name for name, text in modules.items() if "'./" + json_file.name + "'" in text]
        assert len(importers) == 1, (
            json_file.name
            + " is imported by "
            + str(len(importers))
            + " modules: "
            + ", ".join(importers)
        )


def test_the_typescript_states_the_display_versus_values_rule(
    exported: fe.ExportResult,
) -> None:
    """The one rule a grep-based guard cannot enforce must at least be written down."""
    types = (exported.generated / "types.ts").read_text(encoding="utf-8")
    assert "Never render a `values` entry as text" in types
    assert "chart geometry only" in types


# ---------------------------------------------------------------------------
# the metric guard rail itself
# ---------------------------------------------------------------------------


def _guard() -> Any:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("metric_guard", GUARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules["metric_guard"] = module
    spec.loader.exec_module(module)
    return module


def test_the_metric_guard_script_exists_where_ci_invokes_it() -> None:
    """.github/workflows/ci.yml hardcodes this path and fails loudly without it."""
    assert GUARD.is_file()
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/16_check_no_hardcoded_metrics.py" in workflow


def test_the_guard_catches_the_three_ways_a_metric_gets_typed_in(tmp_path: Path) -> None:
    guard = _guard()
    page = tmp_path / "app" / "page.tsx"
    page.parent.mkdir(parents=True)
    page.write_text(
        "\n".join(
            [
                "export default function Page() {",
                "  const accuracy = 0.9;",
                "  const sens = 0.8588;",
                "  return <p>reached 95.82% on PhysioNet</p>;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    findings, suppressed = guard.scan_file(page)
    rules = {finding.rule for finding in findings}
    assert len(rules) == 3, rules
    assert not suppressed
    assert guard.main(["--frontend", str(tmp_path)]) == 1


def test_the_guard_passes_ordinary_ui_numbers(tmp_path: Path) -> None:
    """A guard people route around is not a guard. Layout numbers must pass."""
    guard = _guard()
    component = tmp_path / "components" / "Card.tsx"
    component.parent.mkdir(parents=True)
    component.write_text(
        "\n".join(
            [
                "import { tables } from '@/lib/generated';",
                "export function Card() {",
                "  return <div style={{ opacity: 0.75, width: '50%' }}",
                "    className='h-[42px] translate-x-1/2'>",
                "    {tables.T02.columns[4].display[0]}",
                "  </div>;",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    findings, _ = guard.scan_file(component)
    assert findings == [], [f.rule for f in findings]
    assert guard.main(["--frontend", str(tmp_path)]) == 0


def test_a_suppression_is_reported_even_when_the_run_passes(tmp_path: Path) -> None:
    guard = _guard()
    page = tmp_path / "app" / "page.tsx"
    page.parent.mkdir(parents=True)
    page.write_text("  const accuracy = 0.9; // metric-guard: allow -- fixture\n", encoding="utf-8")
    findings, suppressed = guard.scan_file(page)
    assert findings == []
    assert len(suppressed) == 1
    assert guard.main(["--frontend", str(tmp_path)]) == 0


def test_the_guard_reports_cleanly_when_there_are_no_pages_yet(tmp_path: Path) -> None:
    assert _guard().main(["--frontend", str(tmp_path)]) == 0
