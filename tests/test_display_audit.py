"""Phase 119's gates: the displayed-value audit, the footer manifest and the screenshot gate.

Each check in `src/reporting/display_audit.py` is proven twice: it passes on the
real build, and it FAILS on a copy of that build with one thing broken -- a
fabricated percentage in a page, a table cell edited in `generated/`, a footer
naming a run that never happened, a site rebuilt after its audit. A check that
has only ever been seen passing has not been shown to check anything.

Skips when `frontend/out/` is absent (the Python CI job); the frontend CI job
builds the site and the `postbuild` hook runs the audit itself.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.reporting import display_audit as da

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "frontend" / "out"
GENERATED = PROJECT_ROOT / "frontend" / "lib" / "generated"

pytestmark = pytest.mark.skipif(
    not (OUT / "index.html").is_file(),
    reason="frontend/out/ is not built in this checkout; run npm run build",
)


#: The committed run manifest. `conftest.py` points `outputs.run_manifest` at an
#: empty temporary file so no test can write the real one; an audit of the REAL
#: build must be checked against the real record, so it is named here.
REAL_MANIFEST = PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json"


@pytest.fixture(autouse=True)
def _audit_against_the_committed_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(da, "default_run_manifest", lambda: REAL_MANIFEST)


@pytest.fixture(scope="module")
def real() -> da.AuditReport:
    return da.run_audit(run_manifest=REAL_MANIFEST)


@pytest.fixture()
def small_site(tmp_path: Path) -> Path:
    """Two real pages of the build, enough for every check that reads HTML."""
    site = tmp_path / "out"
    (site / "limitations").mkdir(parents=True)
    shutil.copyfile(OUT / "index.html", site / "index.html")
    shutil.copyfile(OUT / "limitations" / "index.html", site / "limitations" / "index.html")
    return site


def _copy_generated(tmp_path: Path) -> Path:
    target = tmp_path / "generated"
    shutil.copytree(GENERATED, target)
    return target


# ---------------------------------------------------------------------------
# T119.3 -- the real build passes, and every check can fail
# ---------------------------------------------------------------------------


def test_the_real_build_passes_every_check(real: da.AuditReport) -> None:
    assert real.passed, json.dumps(real.to_dict()["findings"], indent=2)
    assert real.checked["pages"] >= 16, "the crawl missed pages"
    assert real.checked["rendered_metric_tokens"] > 500, "the crawl saw implausibly few values"
    assert real.checked["table_and_figure_cells"] > 10_000


def test_a_fabricated_percentage_on_a_page_is_caught(small_site: Path) -> None:
    page = small_site / "limitations" / "index.html"
    page.write_text(
        page.read_text(encoding="utf-8").replace("</h1>", " 95.82% accuracy</h1>", 1),
        encoding="utf-8",
    )
    report = da.run_audit(out_dir=small_site)
    assert any("95.82%" in item for item in report.findings["screen_to_source"])
    assert not report.passed


def test_a_generated_cell_that_disagrees_with_its_csv_is_caught(
    small_site: Path, tmp_path: Path
) -> None:
    generated = _copy_generated(tmp_path)
    tables = json.loads((generated / "tables.json").read_text(encoding="utf-8"))
    column = next(c for c in tables["T08"]["columns"] if c["kind"] == "metric")
    column["display"][0] = "0.999"
    (generated / "tables.json").write_text(json.dumps(tables), encoding="utf-8")

    report = da.run_audit(out_dir=small_site, generated_dir=generated)
    assert any(
        item.startswith("T08." + column["name"] + "[0]")
        for item in report.findings["source_to_generated"]
    )


def test_a_stale_evidence_digest_is_caught(small_site: Path, tmp_path: Path) -> None:
    generated = _copy_generated(tmp_path)
    evidence = json.loads((generated / "evidence.json").read_text(encoding="utf-8"))
    evidence[0]["generated_from_sha256"] = "0" * 64
    (generated / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")

    report = da.run_audit(out_dir=small_site, generated_dir=generated)
    assert any("changed since the export" in item for item in report.findings["evidence_current"])


# ---------------------------------------------------------------------------
# T119.5 -- the footer is the run
# ---------------------------------------------------------------------------


def test_every_page_footer_shows_the_run_that_produced_the_artifacts(real: da.AuditReport) -> None:
    manifest: dict[str, Any] = json.loads((GENERATED / "manifest.json").read_text(encoding="utf-8"))
    runs = json.loads(
        (PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    run = next(item for item in runs["runs"] if item["run_id"] == manifest["run_id"])
    assert run["name"] == "export_frontend_data"
    assert real.findings["footer_is_the_run"] == []
    assert real.run_id == manifest["run_id"]


def test_a_footer_naming_an_unrecorded_run_is_caught(small_site: Path, tmp_path: Path) -> None:
    generated = _copy_generated(tmp_path)
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = "20990101T000000Z-deadbeef"
    (generated / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    findings = da.run_audit(out_dir=small_site, generated_dir=generated).findings[
        "footer_is_the_run"
    ]
    assert any("is not in the project's run manifest" in item for item in findings)
    assert any("the footer does not show 20990101T000000Z-deadbeef" in item for item in findings)


# ---------------------------------------------------------------------------
# T119.4 -- the screenshot gate
# ---------------------------------------------------------------------------


def test_the_gate_is_closed_without_an_audit(small_site: Path, tmp_path: Path) -> None:
    with pytest.raises(da.AuditGateError, match="no displayed-value audit"):
        da.require_passed_audit(out_dir=small_site, report_path=tmp_path / "none.json")


def test_the_gate_is_closed_after_a_failed_audit(small_site: Path, tmp_path: Path) -> None:
    page = small_site / "index.html"
    page.write_text(
        page.read_text(encoding="utf-8").replace("</h1>", " 12.345</h1>", 1), encoding="utf-8"
    )
    stamp = da.write_report(da.run_audit(out_dir=small_site), tmp_path / "stamp.json")
    with pytest.raises(da.AuditGateError, match="FAILED"):
        da.require_passed_audit(out_dir=small_site, report_path=stamp)


def test_the_gate_opens_only_for_the_exact_build_that_passed(
    small_site: Path, tmp_path: Path
) -> None:
    report = da.run_audit(out_dir=small_site)
    assert report.passed, report.to_dict()["findings"]
    stamp = da.write_report(report, tmp_path / "stamp.json")
    assert da.require_passed_audit(out_dir=small_site, report_path=stamp)["status"] == "pass"

    # Rebuilt afterwards -- even with no new number -- and the gate closes.
    page = small_site / "index.html"
    page.write_text(page.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    page.write_text(
        page.read_text(encoding="utf-8").replace("</body>", "<p>rebuilt</p></body>"),
        encoding="utf-8",
    )
    with pytest.raises(da.AuditGateError, match="rebuilt after it was audited"):
        da.require_passed_audit(out_dir=small_site, report_path=stamp)
