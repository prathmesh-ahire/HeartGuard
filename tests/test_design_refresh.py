"""Phase 128 (T128.7): the design system refresh, checked without a browser.

The browser half -- every primitive rendering in both themes, and reduced motion
actually switching the animation off -- is `frontend/e2e-design/design.spec.ts`,
run against `next dev` because `/design` is kept out of the production build.
What is checkable from the source is checked here:

* the brand is the user's four colours: exact in the light theme, and every
  other brand token in both themes a lightness step of one of them (T128.1);
* every text colour is at least 4.5:1 on every ground it can sit on, and the
  focus ring at least 3:1, in both themes (T128.1);
* no page or component reaches past the tokens with a stock Tailwind palette
  class or a hex literal (T128.1);
* the layout scale, shell, overlays, motion and states exist with their rules
  (T128.2-T128.6), and the design reference renders every one of them.
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

import pytest

from src.reporting.frontend_export import contrast_ratio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
COMPONENTS = FRONTEND / "components"
UI = COMPONENTS / "ui"

#: The user's scheme (2026-09-13), and the light-theme token each one IS.
SCHEME = {
    "surface": "#F2E0D2",
    "accent": "#9E182B",
    "line": "#F2AFBC",
    "accent-soft": "#F9CBD6",
}

#: Tokens that are brand colour. Status tokens (danger, good, warn) are
#: meanings, deliberately off the scheme, and are not listed.
BRAND_TOKENS = (
    "surface", "panel", "sunken", "raised", "line", "line-strong", "ink", "ink-2",
    "ink-3", "accent", "accent-strong", "accent-deep", "accent-soft", "accent-line",
    "on-accent", "scrim", "grid",
)  # fmt: skip

GROUNDS = ("surface", "panel", "sunken", "raised", "accent-soft")

#: Every text colour, and every ground the components put it on.
TEXT_ON_GROUND = {
    "ink": GROUNDS,
    "ink-2": GROUNDS,
    "ink-3": GROUNDS,
    "accent": GROUNDS,
    "accent-strong": GROUNDS,
    "accent-deep": GROUNDS,
    "on-accent": ("accent", "accent-strong"),
    "danger": ("danger-soft", "panel"),
    "good": ("good-soft", "panel"),
    "warn": ("warn-soft", "panel"),
}

TEXT_CONTRAST = 4.5
FOCUS_CONTRAST = 3.0
#: How far, in degrees of hue, a derived brand token may sit from a scheme hue.
HUE_TOLERANCE = 12.0
#: Below this chroma (0-255) a colour has no meaningful hue: white, black.
NEUTRAL_CHROMA = 8


@pytest.fixture(scope="module")
def themes() -> dict[str, dict[str, str]]:
    path = FRONTEND / "app" / "globals.css"
    if not path.is_file():
        pytest.skip("frontend/ is not scaffolded in this checkout")
    css = path.read_text(encoding="utf-8")
    blocks = {
        "light": re.search(r"^:root,\s*\.light\s*\{(.*?)^\}", css, re.M | re.S),
        "dark": re.search(r"^\.dark\s*\{(.*?)^\}", css, re.M | re.S),
    }
    parsed = {}
    for mode, match in blocks.items():
        assert match is not None, "no " + mode + " token block in globals.css"
        parsed[mode] = {
            name: f"#{int(r):02X}{int(g):02X}{int(b):02X}"
            for name, r, g, b in re.findall(r"--([\w-]+):\s*(\d+)\s+(\d+)\s+(\d+);", match.group(1))
        }
    return parsed


def _hue_and_chroma(colour: str) -> tuple[float, int]:
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    hue, _, _ = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return hue * 360.0, max(r, g, b) - min(r, g, b)


def _hue_distance(a: float, b: float) -> float:
    difference = abs(a - b) % 360.0
    return min(difference, 360.0 - difference)


# ---------------------------------------------------------------------------
# T128.1 -- the scheme
# ---------------------------------------------------------------------------


def test_the_light_theme_is_the_four_scheme_colours_exactly(themes: dict) -> None:
    for token, colour in SCHEME.items():
        assert themes["light"][token] == colour, token + " is not " + colour
    # Rules and the accent's own rule are the same pink.
    assert themes["light"]["accent-line"] == SCHEME["line"]


def test_the_dark_theme_is_the_same_four_colours_inverted(themes: dict) -> None:
    dark = themes["dark"]
    assert dark["ink"] == SCHEME["surface"], "the cream should become the text"
    assert dark["accent"] == SCHEME["line"], "the pink should become the accent"
    assert dark["accent-line"] == SCHEME["accent"], "the wine should become the rule"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_brand_token_is_a_lightness_step_of_a_scheme_colour(themes: dict, mode: str) -> None:
    scheme_hues = [_hue_and_chroma(colour)[0] for colour in SCHEME.values()]
    off_scheme = []
    for token in BRAND_TOKENS:
        colour = themes[mode][token]
        hue, chroma = _hue_and_chroma(colour)
        if chroma <= NEUTRAL_CHROMA:
            continue
        nearest = min(_hue_distance(hue, scheme) for scheme in scheme_hues)
        if nearest > HUE_TOLERANCE:
            off_scheme.append(token + " " + colour + " is " + format(nearest, ".1f") + " deg off")
    assert off_scheme == [], mode + ": " + "; ".join(off_scheme)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_every_text_colour_passes_4_5_to_1_on_every_ground(themes: dict, mode: str) -> None:
    tokens = themes[mode]
    failures = []
    for text, grounds in TEXT_ON_GROUND.items():
        for ground in grounds:
            ratio = contrast_ratio(tokens[text], tokens[ground])
            if ratio < TEXT_CONTRAST:
                failures.append(text + " on " + ground + " = " + format(ratio, ".2f"))
    assert failures == [], mode + ": " + "; ".join(failures)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_focus_ring_passes_3_to_1_on_every_ground(themes: dict, mode: str) -> None:
    css = (FRONTEND / "app" / "globals.css").read_text(encoding="utf-8")
    assert "outline-accent" in css, "the focus ring is no longer the accent"
    tokens = themes[mode]
    for ground in GROUNDS:
        ratio = contrast_ratio(tokens["accent"], tokens[ground])
        assert ratio >= FOCUS_CONTRAST, mode + " focus ring on " + ground + " = " + str(ratio)


def test_a_known_failing_pair_is_caught() -> None:
    """The check has to be able to fail: the blush fill as text on cream does."""
    assert contrast_ratio(SCHEME["accent-soft"], SCHEME["surface"]) < TEXT_CONTRAST


_PALETTE_CLASS = re.compile(
    r"\b(?:bg|text|border|ring|fill|stroke|outline|from|to|via|divide)-"
    r"(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|"
    r"cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose|white|black)\b"
)
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def test_no_page_or_component_bypasses_the_tokens() -> None:
    """A stock palette class or a hex literal does not follow the theme or the scheme.

    `lib/tokens.ts` keeps its one Okabe-Ito fallback hex (see
    `test_design_system.py`); `lib/generated/` is the exported chart palette.
    """
    if not FRONTEND.is_dir():
        pytest.skip("frontend/ is not scaffolded in this checkout")
    offenders = []
    for directory in ("app", "components", "lib"):
        for path in sorted((FRONTEND / directory).rglob("*.ts*")):
            relative = path.relative_to(FRONTEND).as_posix()
            if relative.startswith("lib/generated/") or relative == "lib/tokens.ts":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for match in _PALETTE_CLASS.findall(line) + _HEX.findall(line):
                    offenders.append(relative + ":" + str(number) + " " + match)
    assert offenders == [], "; ".join(offenders)


# ---------------------------------------------------------------------------
# T128.2 / T128.4 -- layout scale and shell
# ---------------------------------------------------------------------------


def test_the_layout_scale_is_declared_once() -> None:
    config = (FRONTEND / "tailwind.config.ts").read_text(encoding="utf-8")
    assert "content: '76rem'" in config
    assert "reading: '42rem'" in config
    for size in ("lede:", "title:", "display:"):
        assert size in config, size + " is missing from the type scale"
    tokens = (FRONTEND / "lib" / "tokens.ts").read_text(encoding="utf-8")
    assert "export const LAYOUT" in tokens
    assert "px-4 sm:px-6 lg:px-10" in tokens, "the Stitch page margins are gone"


def test_the_shell_uses_the_layout_and_a_drawer_on_narrow_screens() -> None:
    shell = (COMPONENTS / "AppShell.tsx").read_text(encoding="utf-8")
    assert "LAYOUT.page" in shell
    assert 'href="#main"' in shell and 'id="main"' in shell, "no skip link"
    nav = (COMPONENTS / "SideNav.tsx").read_text(encoding="utf-8")
    assert "<Drawer" in nav and 'side="left"' in nav
    assert nav.count("<PrimaryNav") == 2, "the rail and the drawer must share one list"
    bar = (COMPONENTS / "TopBar.tsx").read_text(encoding="utf-8")
    assert "aria-expanded={menuOpen}" in bar


def test_the_title_area_has_exactly_one_primary_action_slot() -> None:
    header = (UI / "PageHeader.tsx").read_text(encoding="utf-8")
    assert "primaryAction?: ReactNode;" in header
    assert header.count("{primaryAction}") == 1
    assert header.index("{actions}") < header.index("{primaryAction}")


# ---------------------------------------------------------------------------
# T128.3 / T128.5 -- overlays and motion
# ---------------------------------------------------------------------------


def test_the_overlays_are_native_dialogs_with_a_reduced_motion_path() -> None:
    body = (UI / "Overlay.tsx").read_text(encoding="utf-8")
    for export in ("export function Modal", "export function Drawer", "export function Sheet"):
        assert export in body
    assert "<dialog" in body and "showModal()" in body
    assert "returnTo?.focus()" in body, "focus does not return to the trigger"
    assert "event.key !== 'Escape'" in body
    animations = re.findall(r"[\w:-]*animate-[\w-]+", body)
    assert animations, "the overlays have no entrance at all"
    for animation in animations:
        assert animation.startswith("motion-safe:"), animation + " ignores reduced motion"


def test_every_new_animation_is_behind_motion_safe() -> None:
    for name in ("Skeleton.tsx", "States.tsx", "PageStates.tsx", "Badge.tsx"):
        body = (UI / name).read_text(encoding="utf-8")
        for animation in re.findall(r"[\w:-]*animate-[\w-]+", body):
            assert animation.startswith("motion-safe:"), name + ": " + animation


def test_the_stagger_honours_reduced_motion() -> None:
    reveal = (COMPONENTS / "motion" / "Reveal.tsx").read_text(encoding="utf-8")
    assert "export function Stagger" in reveal and "export function StaggerItem" in reveal
    assert "staggerChildren: reduced ? 0 : step" in reveal
    assert "transitionFor(reduced," in reveal
    motion = (FRONTEND / "lib" / "motion.ts").read_text(encoding="utf-8")
    assert "reduced ? { duration: 0, delay: 0 } : transition" in motion


# ---------------------------------------------------------------------------
# T128.6 -- states and the design reference
# ---------------------------------------------------------------------------


def test_the_page_error_state_is_an_error_not_an_empty_panel() -> None:
    states = (UI / "PageStates.tsx").read_text(encoding="utf-8")
    service = states[states.index("export function ServiceUnavailable") :]
    service = service[: service.index("\n}\n")]
    assert "<ErrorState" in service and "<EmptyState" not in service
    for loading in ("HistoryLoading", "InsightsLoading"):
        block = states[states.index("export function " + loading) :]
        assert "<LoadingRegion" in block[: block.index("\n}\n")]


def test_the_design_reference_renders_every_new_primitive_in_both_themes() -> None:
    client = (FRONTEND / "app" / "design" / "DesignClient.tsx").read_text(encoding="utf-8")
    assert "<RefreshPreview" in client
    preview = (FRONTEND / "app" / "design" / "RefreshPreview.tsx").read_text(encoding="utf-8")
    for element in (
        "<Modal", "<Drawer", "<Sheet", "<Skeleton ", "<SkeletonText", "<SkeletonTiles",
        "<SkeletonList", "<SkeletonChart", "<HistoryEmpty", "<InsightsEmpty", "<NoMatches",
        "<HistoryLoading", "<InsightsLoading", "<ServiceUnavailable", "<PageHeader",
        "<Stagger", "<StaggerItem", "<PageTransition", "<AnimatedCounter",
    ):  # fmt: skip
        assert element in preview, element + " is not on the design reference"
    assert '<Primitives mode="light" />' in preview
    assert '<Primitives mode="dark" />' in preview
    css = (FRONTEND / "app" / "globals.css").read_text(encoding="utf-8")
    assert re.search(r"^:root,\s*\.light\s*\{", css, re.M), "no .light token scope for the preview"
