"""Phase 120: the capture plan, and what it refuses to caption.

The screenshots themselves are taken by a browser, so what is checkable here is
the plan behind them -- that it names the thirteen screenshots the extraction
spec asks for, at routes that exist, with the empty-state assertions T120.7
depends on -- and that `finalize` refuses a missing or blank capture instead of
writing a caption under it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting import screenshots as ss
from src.reporting.evidence_pack import SPEC_SCREENSHOTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTES_TS = PROJECT_ROOT / "frontend" / "lib" / "routes.ts"


def test_there_are_thirteen_shots_numbered_one_to_thirteen() -> None:
    assert len(ss.CAPTURE_PLAN) == 13
    assert [shot.number for shot in ss.CAPTURE_PLAN] == list(range(1, 14))
    assert [shot.shot_id for shot in ss.CAPTURE_PLAN] == [
        "SS-" + str(n).zfill(2) for n in range(1, 14)
    ]


def test_every_shot_is_the_spec_screenshot_of_that_number() -> None:
    """The titles are the extraction spec's section 14, in its order."""
    assert tuple(shot.title for shot in ss.CAPTURE_PLAN) == SPEC_SCREENSHOTS


def test_every_capture_route_is_a_declared_route() -> None:
    """A screenshot of a route the navbar does not know about is a 404 waiting."""
    declared = ROUTES_TS.read_text(encoding="utf-8")
    for shot in ss.CAPTURE_PLAN:
        if shot.route == "/":
            continue
        assert "href: '" + shot.route + "'" in declared, shot.shot_id + " " + shot.route


def test_filenames_are_unique_and_carry_their_shot_id() -> None:
    names = [shot.filename for shot in ss.CAPTURE_PLAN]
    assert len(set(names)) == len(names)
    for shot in ss.CAPTURE_PLAN:
        assert shot.filename.startswith(shot.shot_id + "_")
        assert shot.filename.endswith(".png")


def test_every_shot_refuses_an_unbuilt_route_and_a_failed_chart() -> None:
    """T120.7's "no placeholder, no empty chart", as a pre-shutter assertion."""
    for shot in ss.CAPTURE_PLAN:
        assert "Route scaffolded; content not built yet." in shot.must_not_contain
        assert "This chart could not be drawn." in shot.must_not_contain
        assert "No model is deployed for this task yet" in shot.must_not_contain


def test_every_prediction_capture_waits_for_a_result_before_the_shutter() -> None:
    """A prediction page photographed mid-request shows a spinner, not output."""
    for shot in ss.CAPTURE_PLAN:
        runs_model = any(
            step.get("action") == "click" and step.get("name") == "2. Run the screening model"
            for step in shot.steps
        )
        if not runs_model:
            continue
        assert any(step.get("action") == "await_result" for step in shot.steps), shot.shot_id
        assert "Nothing scored yet" in shot.must_not_contain, shot.shot_id


def test_the_explainability_shot_scores_a_recording_first() -> None:
    """Its per-sample half reads `sessionStorage`; cold, it is an empty state."""
    shot = next(item for item in ss.CAPTURE_PLAN if item.number == 12)
    actions = [step.get("action") for step in shot.steps]
    assert actions.index("await_result") < actions.index("goto")
    assert "No prediction has been made in this browser tab yet" in shot.must_not_contain


def test_the_plan_payload_is_json_and_carries_every_field_the_spec_reads() -> None:
    payload = json.loads(json.dumps(ss.plan_payload()))
    assert payload["disclaimer"] == ss.DISCLAIMER_MARK
    assert len(payload["shots"]) == 13
    for shot in payload["shots"]:
        for key in (
            "number",
            "shot_id",
            "title",
            "route",
            "filename",
            "caption",
            "steps",
            "must_contain",
            "must_not_contain",
            "full_page",
            "height",
        ):
            assert key in shot, key


def test_finalize_refuses_a_missing_capture(tmp_path: Path) -> None:
    with pytest.raises(ss.ScreenshotError) as raised:
        ss.finalize(tmp_path, register=False, strict=True)
    assert "13" in str(raised.value)


def test_finalize_refuses_a_blank_capture(tmp_path: Path) -> None:
    """A 200-byte PNG is a blank frame; a caption under one is a fabricated asset."""
    for shot in ss.CAPTURE_PLAN:
        (tmp_path / shot.filename).write_bytes(b"\x89PNG" + b"\x00" * 200)
    with pytest.raises(ss.ScreenshotError) as raised:
        ss.finalize(tmp_path, register=False, strict=True)
    assert "bytes" in str(raised.value)


def test_finalize_captions_and_indexes_real_sized_captures(tmp_path: Path) -> None:
    for shot in ss.CAPTURE_PLAN:
        (tmp_path / shot.filename).write_bytes(b"\x89PNG" + b"\x00" * ss.MIN_BYTES)
    result = ss.finalize(tmp_path, register=False, strict=True)
    assert len(result.rows) == 13
    assert result.missing == []

    captions = result.captions.read_text(encoding="utf-8")
    assert "Not a diagnostic device" in captions
    for shot in ss.CAPTURE_PLAN:
        assert shot.shot_id in captions
        assert shot.caption in captions

    index = result.index.read_text(encoding="utf-8")
    assert "screenshot_id,number,title,route,filename,bytes,caption" in index
    assert len([line for line in index.splitlines() if line.startswith("SS-")]) == 13


def test_the_capture_spec_reads_the_plan_rather_than_declaring_its_own() -> None:
    """No route or sample id may be hard-coded in the TypeScript."""
    spec = (PROJECT_ROOT / "frontend" / "screenshots" / "dashboard.spec.ts").read_text(
        encoding="utf-8"
    )
    assert "capture_plan.json" in spec
    for shot in ss.CAPTURE_PLAN:
        assert "'" + shot.route + "'" not in spec, shot.route + " is hard-coded in the spec"


def test_the_capture_run_is_gated_on_the_displayed_value_audit() -> None:
    """T119.4: no screenshot of an unaudited page, enforced by globalSetup."""
    config = (PROJECT_ROOT / "frontend" / "playwright.screenshots.config.ts").read_text(
        encoding="utf-8"
    )
    assert "globalSetup: './e2e/screenshot-gate.ts'" in config
    assert "testDir: './screenshots'" in config
    # The capture must drive the real API, or six of the thirteen have nothing
    # to photograph.
    assert "src.api.main:app" in config
