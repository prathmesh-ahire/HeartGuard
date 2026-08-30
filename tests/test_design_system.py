"""Phase 111's checkable half: palette provenance, contrast, and the states.

T111.7 is a **[TEST/MANUAL]** gate -- somebody has to open `/design` and look at
it, and no assertion replaces that. What is checkable without eyes is checked
here:

* the dashboard palette IS the matplotlib palette, byte for byte, rather than a
  copy of it that can drift (T111.1);
* the contrast figures shown on `/design` are computed, correct, and identify
  the colours that genuinely fail (T111.5);
* the error state is structurally distinguishable from the empty state, because
  "the request failed" and "there is no data" must never look alike (T111.4);
* no component hardcodes a colour that the palette is supposed to own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.reporting.frontend_export import (
    DARK_GROUND,
    LIGHT_GROUND,
    NON_TEXT_CONTRAST,
    contrast_ratio,
    palette_contrast,
    theme_payload,
)
from src.reporting.plot_style import OKABE_ITO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
UI = FRONTEND / "components" / "ui"


@pytest.fixture(scope="module")
def scaffolded() -> None:
    if not (FRONTEND / "package.json").is_file():
        pytest.skip("frontend/ is not scaffolded in this checkout")


@pytest.fixture(scope="module")
def exported_theme() -> dict:
    path = FRONTEND / "lib" / "generated" / "theme.json"
    if not path.is_file():
        pytest.skip("theme.json has not been exported")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# T111.1 -- one palette, not two
# ---------------------------------------------------------------------------


def test_the_exported_palette_is_the_matplotlib_palette(exported_theme: dict) -> None:
    """A chart in the browser and its 300 dpi PNG must colour series alike."""
    assert exported_theme["palette"]["series"] == list(OKABE_ITO)
    assert exported_theme["palette"]["order_is_meaningful"] is True
    assert exported_theme["palette"]["colourblind_safe"] is True


def test_the_semantic_roles_point_into_the_same_palette(exported_theme: dict) -> None:
    roles = exported_theme["palette"]["roles"]
    assert roles["normal"] == OKABE_ITO[0]
    assert roles["abnormal"] == OKABE_ITO[1]
    for colour in roles.values():
        assert colour in OKABE_ITO


def test_the_tokens_module_reads_the_palette_rather_than_restating_it(
    scaffolded: None,
) -> None:
    tokens = (FRONTEND / "lib" / "tokens.ts").read_text(encoding="utf-8")
    assert "from '@/lib/generated'" in tokens
    assert "theme.palette.series" in tokens
    # One fallback hex is allowed, in seriesColor's `?? '#0072B2'` guard.
    hexes = re.findall(r"#[0-9a-fA-F]{6}", tokens)
    assert len(hexes) <= 1, "the palette is being restated in TypeScript: " + str(hexes)


def test_no_ui_component_hardcodes_a_series_colour(scaffolded: None) -> None:
    for path in sorted(UI.glob("*.tsx")):
        body = path.read_text(encoding="utf-8")
        for colour in OKABE_ITO:
            assert colour not in body, path.name + " hardcodes " + colour


# ---------------------------------------------------------------------------
# T111.5 -- contrast, measured
# ---------------------------------------------------------------------------


def test_contrast_ratio_matches_the_wcag_reference_values() -> None:
    """Black on white is exactly 21:1; a colour against itself is exactly 1:1."""
    assert contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0, abs=1e-6)
    assert contrast_ratio("#FFFFFF", "#000000") == pytest.approx(21.0, abs=1e-6)
    assert contrast_ratio("#777777", "#777777") == pytest.approx(1.0, abs=1e-6)
    # Symmetric: the order of the arguments must not change the answer.
    assert contrast_ratio("#0072B2", "#FFFFFF") == pytest.approx(
        contrast_ratio("#FFFFFF", "#0072B2"), abs=1e-9
    )


def test_every_series_colour_is_measured_on_both_grounds() -> None:
    rows = palette_contrast()
    assert len(rows) == len(OKABE_ITO)
    for row, colour in zip(rows, OKABE_ITO, strict=True):
        assert row["colour"] == colour
        assert row["on_light"] == pytest.approx(contrast_ratio(colour, LIGHT_GROUND), abs=0.001)
        assert row["on_dark"] == pytest.approx(contrast_ratio(colour, DARK_GROUND), abs=0.001)


def test_the_colours_that_fail_are_flagged_rather_than_hidden() -> None:
    """Four of the eight fail 3:1 on one ground. The design must know which."""
    rows = {row["colour"]: row for row in palette_contrast()}

    for colour in ("#E69F00", "#56B4E9", "#F0E442"):
        row = rows[colour]
        assert row["on_light"] < NON_TEXT_CONTRAST
        assert "light" in row["needs_outline_on"], colour + " fails on light but is unflagged"

    black = rows["#000000"]
    assert black["on_dark"] < NON_TEXT_CONTRAST
    assert "dark" in black["needs_outline_on"]

    # And a colour that passes on both is not flagged for either.
    blue = rows["#0072B2"]
    assert blue["on_light"] >= NON_TEXT_CONTRAST
    assert blue["on_dark"] >= NON_TEXT_CONTRAST
    assert blue["needs_outline_on"] == []


def test_a_flag_means_the_measurement_that_produced_it() -> None:
    """The flag must be derived, not asserted separately from the number."""
    for row in palette_contrast():
        expected = [
            ground
            for ground, ratio in (("light", row["on_light"]), ("dark", row["on_dark"]))
            if ratio < NON_TEXT_CONTRAST
        ]
        assert row["needs_outline_on"] == expected


def test_the_theme_payload_carries_the_standard_it_was_measured_against() -> None:
    contrast = theme_payload()["contrast"]
    assert "1.4.11" in contrast["standard"]
    assert contrast["threshold"] == NON_TEXT_CONTRAST
    assert contrast["light_ground"] == LIGHT_GROUND
    assert contrast["dark_ground"] == DARK_GROUND


def test_the_frontend_exposes_the_outline_rule_the_charts_must_apply(
    scaffolded: None,
) -> None:
    tokens = (FRONTEND / "lib" / "tokens.ts").read_text(encoding="utf-8")
    assert "needsOutlineOn" in tokens
    assert "needs_outline_on" in tokens


# ---------------------------------------------------------------------------
# T111.2 / T111.4 -- the primitives and the states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "component",
    [
        "GlassCard.tsx",
        "SectionHeader.tsx",
        "StatTile.tsx",
        "AnimatedCounter.tsx",
        "Badge.tsx",
        "Tabs.tsx",
        "Tooltip.tsx",
        "States.tsx",
        "FileUpload.tsx",
    ],
)
def test_every_declared_primitive_exists(scaffolded: None, component: str) -> None:
    assert (UI / component).is_file(), component + " is missing"


def test_tabs_and_tooltip_are_built_on_radix(scaffolded: None) -> None:
    """T111.2 names Radix: keyboard and ARIA behaviour correct, not approximated."""
    assert "@radix-ui/react-tabs" in (UI / "Tabs.tsx").read_text(encoding="utf-8")
    assert "@radix-ui/react-tooltip" in (UI / "Tooltip.tsx").read_text(encoding="utf-8")


def test_the_error_state_is_announced_and_says_it_is_not_a_zero(
    scaffolded: None,
) -> None:
    """A failed request must render a visible error, never an empty chart."""
    states = (UI / "States.tsx").read_text(encoding="utf-8")
    assert 'role="alert"' in states, "the error state is not announced to assistive tech"
    assert "this is a failure, not a" in states
    # Empty and error must be separate components with separate wording.
    assert "export function EmptyState" in states
    assert "export function ErrorState" in states
    assert "export function LoadingState" in states


def test_the_loading_state_is_announced_as_busy(scaffolded: None) -> None:
    states = (UI / "States.tsx").read_text(encoding="utf-8")
    assert 'role="status"' in states
    assert 'aria-busy="true"' in states


def test_the_upload_rejects_a_non_wav_file_with_a_reason(scaffolded: None) -> None:
    """A control that silently ignores a dropped file reads as broken."""
    upload = (UI / "FileUpload.tsx").read_text(encoding="utf-8")
    assert "export function validateRecording" in upload
    assert "is not a WAV file" in upload
    # The reason is split across two source lines by the line length limit, so
    # each half is checked rather than the joined sentence.
    assert "a lossy format changes the" in upload
    assert "spectral content the features are computed from" in upload
    assert "MAX_BYTES" in upload


def test_the_animated_counter_settles_on_the_python_formatted_string(
    scaffolded: None,
) -> None:
    """Intermediate frames are a transition; the value at rest is Python's."""
    counter = (UI / "AnimatedCounter.tsx").read_text(encoding="utf-8")
    assert "display: string" in counter
    assert "setFrame(null)" in counter, "the animation never returns to the display string"
    assert "prefers-reduced-motion" in counter
    assert "{frame ?? display}" in counter


def test_the_stat_tile_requires_a_display_string_not_a_number(scaffolded: None) -> None:
    tile = (UI / "StatTile.tsx").read_text(encoding="utf-8")
    assert "display: string;" in tile
    assert "value?: number | null;" in tile, "value must be optional; display is not"


# ---------------------------------------------------------------------------
# T111.6 -- the reference page
# ---------------------------------------------------------------------------


def test_the_design_page_exists_and_is_a_declared_route(scaffolded: None) -> None:
    assert (FRONTEND / "app" / "design" / "page.tsx").is_file()
    routes = (FRONTEND / "lib" / "routes.ts").read_text(encoding="utf-8")
    assert "UTILITY_ROUTES" in routes
    assert "'/design/'" in routes


def test_the_design_page_reads_real_payloads_rather_than_placeholder_numbers(
    scaffolded: None,
) -> None:
    """A placeholder literal is how a fabricated result reaches a page."""
    page = (FRONTEND / "app" / "design" / "page.tsx").read_text(encoding="utf-8")
    assert "from '@/lib/generated'" in page
    assert "display={" in page
    assert "PALETTE_CONTRAST" in page


def test_the_design_page_renders_all_three_request_states(scaffolded: None) -> None:
    page = (FRONTEND / "app" / "design" / "page.tsx").read_text(encoding="utf-8")
    for state in ("<LoadingState", "<EmptyState", "<ErrorState", "<FileUpload"):
        assert state in page, state + " is not on the design reference page"
