"""Phase 112's checkable half: the 3D boundary, the steps, and the weights.

T112.7 is partly something a person has to look at -- a scene either renders or
it does not, and no assertion replaces opening `/design` on the machine this
project is marked on. What *is* checkable without a browser is checked here, and
it is the half that fails silently:

* the twelve architecture steps name modules and output directories that really
  exist, so a step on the page is a verified claim rather than a caption
  (T112.3);
* three.js is imported **only** through the `ssr: false` dynamic boundary, since
  a direct import still builds, still works locally, and quietly puts several
  hundred kilobytes on every route (T112.1/T112.6);
* every animated component has a `prefers-reduced-motion` path, and the scroll
  narrative's path is *no animation at all* rather than a faster one (T112.2,
  T112.3, T112.5);
* the ensemble view reads SO-05's real weights and carries the sentence that
  stops the picture overstating them (T112.4).

The bundle sizes themselves are measured by `scripts/20_check_bundle_budget.py`,
which needs a build; it runs as the frontend's `postbuild`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.architecture import (
    ARCHITECTURE_STEPS,
    ENSEMBLE_SOURCE,
    ensemble_payload,
    pipeline_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
COMPONENTS = FRONTEND / "components"
APP = FRONTEND / "app"


@pytest.fixture(scope="module")
def scaffolded() -> None:
    if not (FRONTEND / "package.json").is_file():
        pytest.skip("frontend/ is not scaffolded in this checkout")


# ---------------------------------------------------------------------------
# T112.3 -- the twelve steps are verified, not captioned
# ---------------------------------------------------------------------------


def test_there_are_twelve_steps_numbered_in_order() -> None:
    payload = pipeline_payload()
    assert payload["n_steps"] == 12, "T112.3 names twelve documented steps"
    assert [step["index"] for step in payload["steps"]] == list(range(1, 13))
    keys = [step["key"] for step in payload["steps"]]
    assert len(set(keys)) == 12, "two steps share a key"


def test_every_step_names_a_module_that_exists() -> None:
    """A step that cannot point at its implementation is a caption."""
    for step in ARCHITECTURE_STEPS:
        assert (PROJECT_ROOT / step.module).is_file(), (
            "step " + str(step.index) + " names " + step.module + ", which is not there"
        )


def test_every_step_names_an_outputs_directory_that_exists() -> None:
    for step in ARCHITECTURE_STEPS:
        assert (PROJECT_ROOT / "outputs" / step.evidence_dir).is_dir(), (
            "step " + str(step.index) + " claims evidence in outputs/" + step.evidence_dir
        )


def test_a_step_pointing_at_a_missing_module_refuses_to_export(monkeypatch) -> None:
    """The verification has to actually fail, or it is decoration."""
    from src.reporting import architecture

    broken = architecture.ArchitectureStep(
        1,
        "broken",
        "A step naming a module that is not there",
        "...",
        "src/does_not_exist/nowhere.py",
        "01_dataset_audit",
    )
    monkeypatch.setattr(architecture, "ARCHITECTURE_STEPS", (broken,))
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        architecture.pipeline_payload()


def test_the_fold_safety_steps_say_so() -> None:
    """The three fold-safety steps are where the project's credibility sits."""
    by_key = {step.key: step for step in ARCHITECTURE_STEPS}
    for key in ("fold_local", "selection", "search"):
        rule = by_key[key].rule
        assert rule is not None and "fold" in rule.lower(), key + " does not state the fold rule"
    assert by_key["grouping"].rule is not None
    assert "subject" in by_key["grouping"].rule.lower()


def test_the_pipeline_payload_is_exported(scaffolded: None) -> None:
    path = FRONTEND / "lib" / "generated" / "pipeline.json"
    if not path.is_file():
        pytest.skip("pipeline.json has not been exported")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_steps"] == 12
    assert len(payload["steps"]) == 12


# ---------------------------------------------------------------------------
# T112.4 -- the weights, and the sentence that keeps them honest
# ---------------------------------------------------------------------------


def test_the_ensemble_weights_come_from_so05() -> None:
    payload = ensemble_payload()
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))
    assert payload["source"] == ENSEMBLE_SOURCE
    ids = [member["model_id"] for member in payload["members"]]
    assert ids == ["M3", "M4", "M5"], "T112.4 names SVM, RF and GB"
    names = [member["name"] for member in payload["members"]]
    assert names[0].startswith("SVM")
    assert names[1] == "Random Forest"
    assert names[2] == "Gradient Boosting"


def test_the_weights_sum_to_one_and_are_non_negative() -> None:
    payload = ensemble_payload()
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))
    weights = [member["weight"] for member in payload["members"]]
    assert all(weight >= 0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0, abs=1e-9)


def test_every_weight_arrives_pre_formatted() -> None:
    """The client renders `weight_display`; `weight` only positions a bar."""
    payload = ensemble_payload()
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))
    for member in payload["members"]:
        assert member["weight_display"] == format(member["weight"], ".3f")
        assert member["weight_std_display"] == format(member["weight_std"], ".3f")


def test_the_payload_states_how_close_to_equal_the_search_landed() -> None:
    """21 of 25 folds chose equal weights. A viewer must be told, not left
    to infer a large reweighting from three nearly identical bars."""
    payload = ensemble_payload()
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))
    assert payload["folds_identical_to_equal"] > 0
    assert payload["folds_identical_to_equal"] <= payload["n_folds"]
    interpretation = str(payload["interpretation"]).lower()
    assert "equal" in interpretation
    assert str(payload["folds_identical_to_equal"]) in interpretation


def test_the_demonstration_vote_is_computed_in_python() -> None:
    """Otherwise the browser would be doing arithmetic it then displays."""
    payload = ensemble_payload()
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))
    demo = payload["demonstration"]
    assert demo is not None
    expected = sum(
        member["weight"] * item["probability"]
        for member, item in zip(payload["members"], demo["inputs"], strict=True)
    )
    assert demo["vote"] == pytest.approx(expected, abs=1e-12)
    assert demo["vote_display"] == format(demo["vote"], ".3f")
    assert "not predictions" in str(demo["note"]).lower()


def test_the_ensemble_component_renders_the_interpretation(scaffolded: None) -> None:
    body = (COMPONENTS / "ensemble" / "EnsembleVote.tsx").read_text(encoding="utf-8")
    assert "ensemble.interpretation" in body, "the caveat is exported but not shown"
    assert "equal_weight_display" in body, "the equal-weight reference is not drawn"
    assert "vote_display" in body
    assert "toFixed" not in body, "the client is formatting a number"


# ---------------------------------------------------------------------------
# T112.1 / T112.6 -- the 3D boundary
# ---------------------------------------------------------------------------


def test_three_js_is_imported_only_inside_the_dynamic_boundary(scaffolded: None) -> None:
    """A direct import from a page still builds, and costs every route."""
    allowed = {COMPONENTS / "three" / "HeartScene.tsx"}
    offenders = []
    for path in list(APP.rglob("*.tsx")) + list(COMPONENTS.rglob("*.tsx")):
        if path in allowed:
            continue
        body = path.read_text(encoding="utf-8")
        for token in ("from 'three'", "@react-three/fiber", "@react-three/drei"):
            if token in body:
                offenders.append(path.relative_to(FRONTEND).as_posix() + " imports " + token)
    assert offenders == [], (
        "three.js must be reached only through components/three/Hero3D.tsx's "
        "dynamic import: " + "; ".join(offenders)
    )


def test_the_wrapper_disables_server_rendering(scaffolded: None) -> None:
    body = (COMPONENTS / "three" / "Hero3D.tsx").read_text(encoding="utf-8")
    assert "next/dynamic" in body
    assert "ssr: false" in body, (
        "output: 'export' pre-renders every page in Node, where a WebGL context "
        "does not exist; without ssr: false the build fails rather than degrading"
    )


def test_the_scene_is_not_referenced_by_any_page(scaffolded: None) -> None:
    for path in APP.rglob("*.tsx"):
        body = path.read_text(encoding="utf-8")
        assert "HeartScene" not in body, (
            path.name + " imports the scene directly, bypassing the lazy boundary"
        )


def test_no_webgl_is_a_designed_fallback_not_an_error(scaffolded: None) -> None:
    """An absent GPU is an ordinary outcome. Nothing failed."""
    body = (COMPONENTS / "three" / "Hero3D.tsx").read_text(encoding="utf-8")
    assert "HeroFallback" in body
    # `<ErrorState` rather than the bare word: the module's own docstring
    # explains why an absent GPU is not an error, and prose is not a render.
    assert "<ErrorState" not in body, "a missing WebGL context is not an error state"
    assert "WebGL is unavailable" in body


def test_the_capability_probe_releases_its_context(scaffolded: None) -> None:
    """A browser allows a limited number of live contexts per page."""
    body = (FRONTEND / "lib" / "capability.ts").read_text(encoding="utf-8")
    assert "WEBGL_lose_context" in body
    assert "loseContext()" in body


def test_capability_is_false_until_the_client_has_looked(scaffolded: None) -> None:
    """Server render and first client render must produce identical markup."""
    body = (FRONTEND / "lib" / "capability.ts").read_text(encoding="utf-8")
    assert "ready: false" in body
    assert "useEffect" in body


# ---------------------------------------------------------------------------
# T112.2 / T112.3 / T112.5 -- reduced motion, everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "three/Hero3D.tsx",
        "three/HeartScene.tsx",
        "motion/SmoothScroll.tsx",
        "motion/Reveal.tsx",
        "pipeline/PipelineWalkthrough.tsx",
        "ensemble/EnsembleVote.tsx",
    ],
)
def test_every_animated_component_has_a_reduced_motion_path(
    scaffolded: None, relative: str
) -> None:
    body = (COMPONENTS / relative).read_text(encoding="utf-8")
    markers = ("useReducedMotion", "reducedMotion", "animate", "prefers-reduced-motion")
    assert any(marker in body for marker in markers), relative + " ignores the motion setting"


def test_smooth_scroll_is_not_started_at_all_under_reduced_motion(scaffolded: None) -> None:
    """Not started gently. Not started with a shorter duration. Not started."""
    body = (COMPONENTS / "motion" / "SmoothScroll.tsx").read_text(encoding="utf-8")
    assert "if (reduced) return;" in body
    assert "new Lenis(" in body
    assert body.index("if (reduced) return;") < body.index("new Lenis(")


def test_the_walkthrough_loads_no_gsap_under_reduced_motion(scaffolded: None) -> None:
    body = (COMPONENTS / "pipeline" / "PipelineWalkthrough.tsx").read_text(encoding="utf-8")
    assert "if (reduced) return;" in body
    assert body.index("if (reduced) return;") < body.index("import('gsap')")


def test_gsap_and_lenis_are_dynamically_imported(scaffolded: None) -> None:
    walkthrough = (COMPONENTS / "pipeline" / "PipelineWalkthrough.tsx").read_text(encoding="utf-8")
    assert "await Promise.all([" in walkthrough
    assert "import('gsap')" in walkthrough
    assert "import('gsap/ScrollTrigger')" in walkthrough
    assert "from 'gsap'" not in walkthrough, "a static gsap import defeats the split"


def test_framer_motion_is_code_split_behind_lazy_motion(scaffolded: None) -> None:
    """`motion.div` in a template is 34 kB gzipped on every single route."""
    body = (COMPONENTS / "motion" / "Reveal.tsx").read_text(encoding="utf-8")
    assert "LazyMotion" in body
    assert "<m.div" in body
    assert "<motion." not in body, "the eager component is back"
    assert "strict" in body, "without strict, `motion` silently re-bundles the features"
    assert (COMPONENTS / "motion" / "features.ts").is_file()


def test_the_transition_is_a_template_not_a_layout(scaffolded: None) -> None:
    """A layout does not remount, so a transition inside it never plays."""
    template = APP / "template.tsx"
    assert template.is_file()
    assert "PageTransition" in template.read_text(encoding="utf-8")
    layout = (APP / "layout.tsx").read_text(encoding="utf-8")
    assert "PageTransition" not in layout


def test_the_heart_animation_is_labelled_as_decoration(scaffolded: None) -> None:
    """Nothing on the page may read the beat as a measurement."""
    scene = (COMPONENTS / "three" / "HeartScene.tsx").read_text(encoding="utf-8")
    assert "no data behind it" in scene
    # Phase 113 split /design: the server half renders the equations, the client
    # half everything with state. The hero copy lives in the client half.
    page = "".join(
        (APP / "design" / name).read_text(encoding="utf-8")
        for name in ("page.tsx", "DesignClient.tsx")
        if (APP / "design" / name).is_file()
    )
    assert "decoration with no " in page


# ---------------------------------------------------------------------------
# the budget check itself
# ---------------------------------------------------------------------------


def test_the_bundle_budget_runs_after_every_build(scaffolded: None) -> None:
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    scripts = package["scripts"]
    assert scripts["postbuild"] == "npm run check-bundle"
    assert "20_check_bundle_budget.py" in scripts["check-bundle"]
    assert (PROJECT_ROOT / "scripts" / "20_check_bundle_budget.py").is_file()


def test_the_budget_marker_for_gsap_is_not_the_import_specifier() -> None:
    """`ScrollTrigger` appears in the chunk that correctly DEFERS ScrollTrigger.

    Using it as the marker would report a leak every time the split worked.
    """
    import importlib.util
    import sys

    path = PROJECT_ROOT / "scripts" / "20_check_bundle_budget.py"
    spec = importlib.util.spec_from_file_location("bundle_budget", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bundle_budget"] = module
    spec.loader.exec_module(module)

    assert module.LAZY_MARKERS["gsap ScrollTrigger"] == "scrollerProxy"
    assert module.LAZY_MARKERS["three.js"] == "WebGLRenderer"
