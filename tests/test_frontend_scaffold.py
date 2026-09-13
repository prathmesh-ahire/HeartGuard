"""The T110.7 gate's repeatable half: the scaffold invariants (Phase 110).

`npm run build` is the gate itself and it runs in two places: by hand when a
phase closes, and in CI's frontend job, which installs Node and Python and then
runs exactly that command. Neither is reproducible from the Python test job,
which has no Node.

So this file asserts the invariants the build depends on, statically, from the
files on disk:

* `npm run build` cannot run without the exporter running first (T110.5);
* the config is a static export with TypeScript strict mode on (T110.1/T110.6);
* the screening notice, navigation and footer live in the ROOT layout, so no
  route can render without them (T110.3), and the notice is switched by one
  flag (T127.2);
* the navigation is the five product pages, every declared route and every
  retired route has a page, and every page is declared (T110.4, T127.3/4);
* the design reference is not a production page (T127.5);
* no page imports a generated JSON payload directly, bypassing the typed index.

A change that breaks any of these breaks the build too -- but it breaks it in
CI, minutes later, in a job people skim. It breaks here immediately, with a
sentence saying what the rule was.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
APP = FRONTEND / "app"

EXPORTER = "scripts/17_export_frontend_data.py"


@pytest.fixture(scope="module")
def package_json() -> dict:
    path = FRONTEND / "package.json"
    if not path.is_file():
        pytest.skip("frontend/ is not scaffolded in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scaffolded() -> None:
    if not (FRONTEND / "package.json").is_file():
        pytest.skip("frontend/ is not scaffolded in this checkout")


# ---------------------------------------------------------------------------
# T110.5 -- the build cannot run against stale data
# ---------------------------------------------------------------------------


def test_build_runs_the_exporter_before_next_build(package_json: dict) -> None:
    """The site must never build against a `generated/` nobody refreshed."""
    scripts = package_json["scripts"]
    # Phase 119 (T119.1) put the metric guard in front of the exporter, so a
    # hand-typed metric fails `npm run build` itself and not only CI. The
    # property this test protects is unchanged -- the exporter is the LAST step
    # before `next build` -- and is now asserted on the chain, not a literal.
    steps = [step.strip() for step in scripts["prebuild"].split("&&")]
    assert steps[-1] == "npm run export-data", (
        "npm runs `prebuild` automatically before `build`; that hook is what makes "
        "the ordering structural rather than a habit"
    )
    assert steps[0] == "npm run check-metrics", "the metric guard must run before anything else"
    assert EXPORTER in scripts["export-data"]
    assert scripts["build"] == "next build"


def test_the_exporter_the_build_calls_actually_exists(package_json: dict) -> None:
    assert (PROJECT_ROOT / EXPORTER).is_file()
    assert (PROJECT_ROOT / "scripts" / "16_check_no_hardcoded_metrics.py").is_file()
    assert "16_check_no_hardcoded_metrics.py" in package_json["scripts"]["check-metrics"]


def test_the_locked_stack_is_pinned_exactly(package_json: dict) -> None:
    """Pinned, not ranged: two installs of the same commit must resolve alike."""
    dependencies = {**package_json["dependencies"], **package_json["devDependencies"]}
    for name, version in dependencies.items():
        assert version[0].isdigit(), name + " is not pinned to an exact version"
    assert dependencies["next"].startswith("14."), "Next.js 14 is the locked major"
    assert dependencies["react"].startswith("18.")
    assert "typescript" in dependencies
    assert "tailwindcss" in dependencies
    assert "next-themes" in dependencies


def test_the_lockfile_is_committed(scaffolded: None) -> None:
    assert (FRONTEND / "package-lock.json").is_file(), (
        "package-lock.json must be committed so a reproduction installs the same tree"
    )


# ---------------------------------------------------------------------------
# T110.1 / T110.6 -- static export, strict TypeScript
# ---------------------------------------------------------------------------


def test_next_is_configured_as_a_fully_static_export(scaffolded: None) -> None:
    config = (FRONTEND / "next.config.mjs").read_text(encoding="utf-8")
    assert "output: 'export'" in config
    # The default image optimizer needs a Node server, which a static export has
    # no way to provide.
    assert "unoptimized: true" in config


def test_typescript_strict_mode_is_on(scaffolded: None) -> None:
    tsconfig = json.loads(
        "\n".join(
            line
            for line in (FRONTEND / "tsconfig.json").read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("//")
        )
    )
    options = tsconfig["compilerOptions"]
    assert options["strict"] is True
    assert options["noEmit"] is True
    assert options["resolveJsonModule"] is True, "the generated payloads are JSON imports"
    assert options["paths"]["@/*"] == ["./*"]


def test_tailwind_follows_the_theme_class_not_the_os_setting(scaffolded: None) -> None:
    config = (FRONTEND / "tailwind.config.ts").read_text(encoding="utf-8")
    assert "darkMode: 'class'" in config, "next-themes toggles a class; Tailwind must follow it"
    for directory in ("./app/", "./components/"):
        assert directory in config


# ---------------------------------------------------------------------------
# T110.3 -- the disclaimer is in the layout, not on each page
# ---------------------------------------------------------------------------


def test_the_root_layout_renders_the_disclaimer_navigation_and_footer(
    scaffolded: None,
) -> None:
    """The layout owns all three, so no route can render without them.

    Until 2026-09-13 the navigation was a `<Navbar />` rendered by the layout.
    It is now a rail and a bar inside `AppShell`, which the layout renders and
    which receives the disclaimer and footer as slots. The guarantee is
    unchanged: every element is placed by the layout, never by a page.
    """
    layout = (APP / "layout.tsx").read_text(encoding="utf-8")
    for element in ("<DisclaimerBanner />", "<AppShell", "<Footer />"):
        assert element in layout, element + " is not in the root layout"
    # T127.2: hidden for the presentation, but still wired here behind ONE flag,
    # so T138.6 restores it everywhere by flipping that flag.
    assert "SHOW_SCREENING_NOTICE ? <DisclaimerBanner /> : null" in layout

    shell = (FRONTEND / "components" / "AppShell.tsx").read_text(encoding="utf-8")
    for element in ("<SideNav", "<TopBar", "{disclaimer}", "{footer}"):
        assert element in shell, element + " is not rendered by the app shell"


def test_the_screening_notice_has_exactly_one_switch(scaffolded: None) -> None:
    """Every place that renders the notice reads the same flag, and nothing else hides it."""
    import re

    flags = (FRONTEND / "lib" / "flags.ts").read_text(encoding="utf-8")
    assert re.search(r"export const SHOW_SCREENING_NOTICE = (true|false);", flags)
    for path in list(APP.rglob("*.tsx")) + list((FRONTEND / "components").rglob("*.tsx")):
        body = path.read_text(encoding="utf-8")
        if ".disclaimer}" in body:
            assert "SHOW_SCREENING_NOTICE ?" in body, (
                path.name + " renders the disclaimer without the T127.2 flag"
            )


def test_no_page_renders_its_own_disclaimer(scaffolded: None) -> None:
    """Per-page placement is how a disclaimer goes missing on the page added next."""
    for page in APP.rglob("page.tsx"):
        body = page.read_text(encoding="utf-8")
        assert "DisclaimerBanner" not in body, (
            page.name + " renders its own disclaimer; it belongs in the layout only"
        )
        for chrome in ("AppShell", "SideNav", "TopBar"):
            assert chrome not in body, (
                page.name + " renders " + chrome + "; it belongs in the layout only"
            )


def test_the_disclaimer_uses_screening_language_and_no_diagnostic_claim(
    scaffolded: None,
) -> None:
    """Research rule 7."""
    body = (FRONTEND / "components" / "Disclaimer.tsx").read_text(encoding="utf-8")
    lowered = body.lower()
    assert "not a medical device" in lowered
    assert "does not diagnose" in lowered
    for forbidden in ("diagnoses ", "replaces a doctor", "treatment plan", "prescri"):
        assert forbidden not in lowered, "diagnostic language in the disclaimer: " + forbidden


def test_no_chrome_shows_the_run_manifest(scaffolded: None) -> None:
    """T127.1 inverted the old footer check: the run is verified, never displayed.

    `scripts/45_audit_displayed_values.py` still proves the site was built from a
    recorded export run, and fails the build if a commit, run id or path reaches
    any page.
    """
    for name in ("Footer.tsx", "SideNav.tsx", "TopBar.tsx", "AppShell.tsx"):
        body = (FRONTEND / "components" / name).read_text(encoding="utf-8")
        for field in ("git_commit", "git_branch", "exported_utc", "run_id", "excluded_dirs"):
            assert field not in body, name + " surfaces manifest." + field


def test_the_ui_is_named_pv_mepcg_never_the_repository(scaffolded: None) -> None:
    """`HeartGuard` is the repository. `PV-MEPCG / PulseVision` is the framework."""
    for path in list(APP.rglob("*.tsx")) + list((FRONTEND / "components").rglob("*.tsx")):
        body = path.read_text(encoding="utf-8")
        assert "HeartGuard" not in body, path.name + " calls the framework by the repo name"


# ---------------------------------------------------------------------------
# T110.4 -- the route tree
# ---------------------------------------------------------------------------


def _route_block(name: str) -> str:
    """The body of one `export const <name>` array literal, and only that one.

    Slicing to the end of the file was correct while ROUTES was the last
    declaration in routes.ts. Phase 111 added UTILITY_ROUTES below it, and the
    open-ended slice quietly counted `/design/` as a twelfth document page --
    a helper that broadens itself whenever the file grows.
    """
    source = (FRONTEND / "lib" / "routes.ts").read_text(encoding="utf-8")
    start = source.index("export const " + name)
    return source[start : source.index("\n];", start)]


def _declared_routes() -> list[str]:
    """The five product pages, in navigation order."""
    import re

    return re.findall(r"href:\s*'([^']+)'", _route_block("ROUTES"))


def _about_tabs() -> list[str]:
    import re

    return re.findall(r"href:\s*'([^']+)'", _route_block("ABOUT_TABS"))


def _legacy_redirects() -> dict[str, str]:
    """Every retired route and its new home (T127.4)."""
    import re

    source = (FRONTEND / "lib" / "routes.ts").read_text(encoding="utf-8")
    start = source.index("export const LEGACY_REDIRECTS")
    block = source[start : source.index("\n};", start)]
    return dict(re.findall(r"'(/[^']*)':\s*'(/[^']*)'", block))


def _page_routes() -> list[str]:
    routes = []
    for page in APP.rglob("page.tsx"):
        relative = page.parent.relative_to(APP).as_posix()
        routes.append("/" if relative == "." else "/" + relative + "/")
    return sorted(routes)


def test_the_navigation_is_exactly_the_five_product_pages(scaffolded: None) -> None:
    declared = _declared_routes()
    assert declared == ["/", "/history/", "/insights/", "/reports/", "/about/"], declared
    nav = (FRONTEND / "components" / "SideNav.tsx").read_text(encoding="utf-8")
    assert "ROUTES.map" in nav, "the rail must be built from the route table"


def test_every_declared_route_has_a_page_and_every_page_is_declared(
    scaffolded: None,
) -> None:
    declared = set(_declared_routes()) | set(_about_tabs()) | set(_legacy_redirects())
    built = set(_page_routes())
    assert declared - built == set(), "declared but not built: " + str(sorted(declared - built))
    assert built - declared == set(), "built but not declared: " + str(sorted(built - declared))


def test_every_retired_route_redirects_to_a_live_page(scaffolded: None) -> None:
    redirects = _legacy_redirects()
    retired = {
        "/dataset/", "/preprocessing/", "/features/", "/models/", "/optimization/",
        "/robustness/", "/explainability/", "/limitations/",
        "/predict/binary/", "/predict/multiclass/", "/predict/murmur/",
    }  # fmt: skip
    assert set(redirects) == retired
    live = set(_declared_routes()) | set(_about_tabs())
    for source, target in redirects.items():
        assert target in live, source + " redirects to " + target + ", which is not a page"
        body = (APP / source.strip("/") / "page.tsx").read_text(encoding="utf-8")
        assert "<Redirect" in body and "LEGACY_REDIRECTS['" + source + "']" in body, source


def test_every_route_carries_a_summary_used_as_its_description(scaffolded: None) -> None:
    # Counted inside each literal only: `summary:` also appears once in the
    # RouteDefinition interface.
    assert _route_block("ROUTES").count("summary:") == len(_declared_routes())
    assert _route_block("ABOUT_TABS").count("summary:") == len(_about_tabs())


def test_the_design_reference_is_not_a_production_page(scaffolded: None) -> None:
    """T127.5: `page.design.tsx` is a page only under `next dev` or PV_DESIGN_PAGE=1."""
    assert (APP / "design" / "page.design.tsx").is_file()
    assert not (APP / "design" / "page.tsx").exists()
    config = (FRONTEND / "next.config.mjs").read_text(encoding="utf-8")
    assert "PHASE_DEVELOPMENT_SERVER" in config
    assert "pageExtensions: includeDesign ? ['design.tsx', 'tsx', 'ts'] : ['tsx', 'ts']" in config


def test_a_scaffold_page_says_it_is_a_scaffold(scaffolded: None) -> None:
    """A placeholder that looks finished is worse than one that looks empty."""
    placeholder = (FRONTEND / "components" / "PagePlaceholder.tsx").read_text(encoding="utf-8")
    assert "not built yet" in placeholder
    assert "Nothing on it is a result" in placeholder


# ---------------------------------------------------------------------------
# the codegen boundary
# ---------------------------------------------------------------------------


def test_no_page_or_component_imports_a_generated_json_directly(
    scaffolded: None,
) -> None:
    """Bypassing `lib/generated/index.ts` loses the declared types (T109.6)."""
    for path in list(APP.rglob("*.tsx")) + list((FRONTEND / "components").rglob("*.tsx")):
        body = path.read_text(encoding="utf-8")
        for payload in ("tables.json", "figures.json", "manifest.json", "evidence.json"):
            assert payload not in body, (
                path.name + " imports " + payload + " directly; import from "
                "'@/lib/generated' so a schema change is a compile error"
            )


def test_eslint_blocks_the_same_bypass(scaffolded: None) -> None:
    config = json.loads((FRONTEND / ".eslintrc.json").read_text(encoding="utf-8"))
    rules = json.dumps(config.get("rules", {}))
    assert "generated/*.json" in rules, "eslint does not block direct JSON payload imports"


def test_the_generated_directory_is_present_and_typed(scaffolded: None) -> None:
    generated = FRONTEND / "lib" / "generated"
    for name in (
        "manifest.json",
        "tables.json",
        "figures.json",
        "evidence.json",
        "types.ts",
        "index.ts",
    ):
        assert (generated / name).is_file(), name + " has not been exported"
    index = (generated / "index.ts").read_text(encoding="utf-8")
    assert " as GeneratedTable" not in index, "a cast would defeat the type check"
