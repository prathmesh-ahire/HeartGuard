"""🔴 MEGA TEST 6 — the product UI (Phase 138).

Covers Part XII (Phases 127-137): the five-page redesign (Analyse, History,
Insights, Reports, About the Model), its new backend (`/api/history`,
`/api/reports`) and everything Part XII changed in the guard rail. Nothing
after this file is green.

MEGA TEST 5 (Phase 121) proved the *old* twelve-route dashboard reproduced
end to end. Part XII replaced that site with a different one, so its display-
audit and screenshot checks are re-run here against the CURRENT build rather
than trusted from a site that no longer exists.

* T138.1 -- one continuous browser walkthrough, not repeated here: pytest does
  not drive a browser on this project, Playwright does. See
  `frontend/e2e/end-to-end.spec.ts`. This file only confirms that file exists
  and actually chains through all four pages, so a spec quietly gutted to a
  no-op would still be caught here.
* T138.2 -- `display_audit.run_audit` (T119's checks) crawls the built site
  unchanged by Part XII; re-run fresh below.
* T138.3 -- the thirteen screenshots, same shape as T121.7, now against the
  routes T127.3 moved them to.
* T138.4 -- README.md and CLAUDE.md name no route T127.4 retired.
"""

from __future__ import annotations

import csv
import json
import re

import pytest

from src.reporting import display_audit as da
from src.reporting import screenshots as ss
from src.utils.evidence import PROJECT_ROOT, read_evidence

OUT = PROJECT_ROOT / "frontend" / "out"
SCREENSHOTS = PROJECT_ROOT / "outputs" / "15_dashboard_screenshots"
REAL_MANIFEST = PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json"
END_TO_END_SPEC = PROJECT_ROOT / "frontend" / "e2e" / "end-to-end.spec.ts"

built = pytest.mark.skipif(
    not (OUT / "index.html").is_file(),
    reason="frontend/out/ is not built in this checkout; run npm run build",
)

# Every route T127.4 retired in favour of a redirect. A name reappearing in the
# README or CLAUDE.md as a live path (not inside the retirement table itself)
# means a doc was written against the site Part XII replaced.
RETIRED_ROUTES = (
    "/dataset/",
    "/preprocessing/",
    "/features/",
    "/models/",
    "/optimization/",
    "/robustness/",
    "/explainability/",
    "/limitations/",
    "/predict/binary/",
    "/predict/multiclass/",
    "/predict/murmur/",
)


# ---------------------------------------------------------------------------
# T138.1 -- the cross-page walkthrough exists and is not a no-op
# ---------------------------------------------------------------------------


def test_t138_1_the_end_to_end_spec_chains_through_all_four_pages() -> None:
    assert END_TO_END_SPEC.is_file(), "frontend/e2e/end-to-end.spec.ts is missing"
    body = END_TO_END_SPEC.read_text(encoding="utf-8")
    for needle in (
        "goto('/'",
        "View in History",
        "/insights/",
        "/reports/",
        "Download PDF",
    ):
        assert needle in body, "the walkthrough no longer touches " + needle


# ---------------------------------------------------------------------------
# T138.2 -- the built site, crawled fresh
# ---------------------------------------------------------------------------


@built
def test_t138_2_every_rendered_value_on_the_five_page_site_matches_its_source() -> None:
    report = da.run_audit(run_manifest=REAL_MANIFEST)
    assert report.passed, json.dumps(report.to_dict(), indent=2)[:4000]
    # The old twelve-route site cleared 12; the five pages plus About's six
    # tabs plus eleven legacy redirects plus the print view is well past it.
    # A drop below this floor means pages silently stopped being built.
    assert report.checked.get("pages", 0) >= 20


@built
def test_t138_2_the_screenshot_gate_is_open_on_the_site_on_disk() -> None:
    if not da.default_report_path().is_file():
        pytest.skip("no audit stamp in this checkout; run npm run build")
    da.require_passed_audit()


# ---------------------------------------------------------------------------
# T138.3 -- the thirteen screenshots are of the current (five-page) site
# ---------------------------------------------------------------------------


def test_t138_3_the_thirteen_screenshots_exist_and_are_registered() -> None:
    """Read-only: a test must not rewrite the deliverable it is checking."""
    index = SCREENSHOTS / "screenshot_index.csv"
    if not index.is_file():
        pytest.skip("the gated capture has not been run in this checkout")

    with index.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["screenshot_id"]: row for row in csv.DictReader(handle)}
    assert len(rows) == 13

    for shot in ss.CAPTURE_PLAN:
        row = rows.get(shot.shot_id)
        assert row is not None, shot.shot_id + " is not in the index"
        assert row["filename"] == shot.filename
        png = SCREENSHOTS / shot.filename
        assert png.is_file(), shot.filename
        assert png.stat().st_size >= ss.MIN_BYTES, shot.shot_id

    registered = {row["evidence_id"]: row for row in read_evidence()}
    for shot in ss.CAPTURE_PLAN:
        row = registered.get(shot.shot_id)
        assert row is not None, shot.shot_id + " is not in the evidence index"
        assert row["status"] == "ok", shot.shot_id


def test_t138_3_no_shot_still_points_at_a_route_t127_retired() -> None:
    for shot in ss.CAPTURE_PLAN:
        assert shot.route not in RETIRED_ROUTES, (
            shot.shot_id + " captures " + shot.route + ", which T127.4 retired"
        )


# ---------------------------------------------------------------------------
# T138.4 -- the docs name no retired route
# ---------------------------------------------------------------------------


def test_t138_4_readme_and_claude_md_name_no_retired_route() -> None:
    for doc in (PROJECT_ROOT / "README.md", PROJECT_ROOT / "CLAUDE.md"):
        text = doc.read_text(encoding="utf-8")
        for route in RETIRED_ROUTES:
            # Word-boundary-ish: `/models/` inside `/about/models/` must not match.
            assert re.search(r"(?<![\w/-])" + re.escape(route), text) is None, (
                doc.name + " still names the retired route " + route
            )
