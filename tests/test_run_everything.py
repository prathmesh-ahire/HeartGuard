"""Phase 122: the one-command reproduction, and what it must not leave out.

The value of `scripts/00_run_everything.py` is entirely in its completeness: a
stage list that quietly omits a script is worse than no runner at all, because
it turns "one command rebuilds the project" into a claim nobody can rely on.
So the load-bearing test here is the drift guard -- every script the evidence
index names as an artifact's reproduction command must appear in the stage list.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from src.pipeline import run_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_CSV = PROJECT_ROOT / "outputs" / "00_evidence_index" / "evidence_index.csv"

#: Scripts that are checks or wrappers rather than producers, so no evidence row
#: names them and no stage needs to carry them separately.
NOT_A_STAGE = {
    "00_run_everything.py",
    "12_run_all_experiments.py",
    "16_check_no_hardcoded_metrics.py",  # runs inside `npm run build` (prebuild)
    "17_export_frontend_data.py",  # runs inside `npm run build` (prebuild)
    "20_check_bundle_budget.py",  # runs inside `npm run build` (postbuild)
    "45_audit_displayed_values.py",  # runs inside `npm run build` (postbuild)
}


def _scripts_in_stages() -> set[str]:
    found: set[str] = set()
    for stage in run_all.STAGES:
        for token in stage.argv:
            if token.startswith("scripts/"):
                found.add(token.split("/", 1)[1])
    return found


def test_stage_ids_are_unique() -> None:
    ids = [stage.stage_id for stage in run_all.STAGES]
    assert len(set(ids)) == len(ids)


def test_every_stage_command_is_runnable_as_declared() -> None:
    """A stage naming a script that does not exist fails at hour nine, not hour one."""
    for stage in run_all.STAGES:
        if stage.npm:
            continue
        head = stage.argv[0]
        if head == "-c":
            assert len(stage.argv) == 2, stage.stage_id
            continue
        assert (PROJECT_ROOT / head).is_file(), stage.stage_id + " -> " + head


def test_every_declared_output_path_is_relative_to_the_project_root() -> None:
    for stage in run_all.STAGES:
        for path in stage.produces:
            assert not Path(path).is_absolute(), stage.stage_id + " " + path
            assert not path.startswith(".."), stage.stage_id + " " + path


@pytest.mark.skipif(not EVIDENCE_CSV.is_file(), reason="evidence_index.csv not generated here")
def test_every_script_the_evidence_index_names_is_a_stage() -> None:
    """The drift guard: a producer that is not in the runner is not reproducible."""
    with EVIDENCE_CSV.open("r", encoding="utf-8", newline="") as handle:
        commands = {row["command"].strip() for row in csv.DictReader(handle)}
    named: set[str] = set()
    for command in commands:
        for match in re.finditer(r"scripts/([0-9A-Za-z_]+\.py)", command):
            named.add(match.group(1))
    missing = sorted(named - _scripts_in_stages() - NOT_A_STAGE)
    assert missing == [], "not in the one-command reproduction: " + ", ".join(missing)


def test_the_frontend_chain_is_four_contiguous_stages_and_is_skippable() -> None:
    """T122.2: export, npm ci, npm run build, guard rail, then the screenshots.

    This asserted the chain was the **last four** stages, which it was until
    Phase 126 added documentation, QA, compliance and delivery after it. Those
    four are deliberately NOT marked `frontend`: `--skip-frontend` means "do not
    build the dashboard", not "skip the QA sweep", and a sweep that vanished
    with the build would be missing exactly when a fresh machine most needs it.
    The invariant that still matters -- the chain is these four stages, in this
    order, contiguously, and removable -- is asserted instead. Changed
    deliberately; see the Phases 123-126 entry in Docs/note.md.
    """
    ids = [stage.stage_id for stage in run_all.STAGES]
    frontend = [stage.stage_id for stage in run_all.STAGES if stage.frontend]
    assert frontend == ["npm_ci", "frontend_build", "screenshots", "evidence_index_final"]

    first = ids.index("npm_ci")
    assert ids[first : first + 4] == frontend, "the frontend chain is not contiguous"

    without = run_all.select(skip_frontend=True)
    assert all(not stage.frontend for stage in without)
    assert {"final_qa", "compliance_review", "delivery", "project_docs"} <= {
        stage.stage_id for stage in without
    }, "--skip-frontend must not skip the Part XI stages"


def test_the_build_stage_is_npm_run_build_so_the_guard_rail_cannot_be_skipped() -> None:
    """`prebuild` and `postbuild` carry the four checks; calling `next build` skips them."""
    build = next(stage for stage in run_all.STAGES if stage.stage_id == "frontend_build")
    assert build.npm is True
    assert build.argv == ("run", "build")
    package = (PROJECT_ROOT / "frontend" / "package.json").read_text(encoding="utf-8")
    assert '"prebuild": "npm run check-metrics && npm run export-data"' in package
    assert '"postbuild": "npm run check-bundle && npm run audit-display"' in package


def test_selection_by_start_and_by_id() -> None:
    ids = [stage.stage_id for stage in run_all.STAGES]
    assert [stage.stage_id for stage in run_all.select(start="exp_a1")] == ids[
        ids.index("exp_a1") :
    ]
    assert [stage.stage_id for stage in run_all.select(only=["features", "exp_a2"])] == [
        "features",
        "exp_a2",
    ]
    with pytest.raises(KeyError):
        run_all.select(only=["no_such_stage"])


def test_a_dry_run_touches_nothing_and_reports_every_stage() -> None:
    payload = run_all.run_stages(list(run_all.STAGES), dry_run=True)
    assert payload["n_stages"] == len(run_all.STAGES)
    assert all(entry["status"] == "dry-run" for entry in payload["stages"])
    assert "manifest" not in payload


def test_resume_skips_a_stage_whose_outputs_exist(tmp_path: Path) -> None:
    present = run_all.Stage("present", "already done", ("-c", "pass"), produces=("README.md",))
    absent = run_all.Stage(
        "absent", "not done", ("-c", "pass"), produces=("outputs/does_not_exist.csv",)
    )
    payload = run_all.run_stages([present, absent], resume=True, out_root=tmp_path)
    statuses = {entry["stage_id"]: entry["status"] for entry in payload["stages"]}
    assert statuses["present"] == "skipped-present"
    assert statuses["absent"] == "ok"


def test_a_failing_stage_stops_the_run_rather_than_continuing(tmp_path: Path) -> None:
    """A pipeline that carries on past a failure produces assets from stale inputs."""
    payload = run_all.run_stages(
        [
            run_all.Stage("boom", "fails", ("-c", "raise SystemExit(3)")),
            run_all.Stage("after", "must not run", ("-c", "pass")),
        ],
        out_root=tmp_path,
    )
    assert [entry["stage_id"] for entry in payload["stages"]] == ["boom"]
    assert payload["n_failed"] == 1
    assert (tmp_path / run_all.MANIFEST_FILENAME).is_file()


def test_smoke_runs_the_reduced_form_and_says_so_when_there_is_none(tmp_path: Path) -> None:
    payload = run_all.run_stages(
        [
            run_all.Stage(
                "has_smoke", "ok", ("-c", "raise SystemExit(1)"), smoke_argv=("-c", "pass")
            ),
            run_all.Stage("no_smoke", "skipped", ("-c", "raise SystemExit(1)")),
        ],
        smoke=True,
        out_root=tmp_path,
    )
    statuses = {entry["stage_id"]: entry["status"] for entry in payload["stages"]}
    assert statuses["has_smoke"] == "ok"
    assert statuses["no_smoke"] == "skipped-no-smoke-form"


@pytest.mark.skipif(
    not (PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json").is_file(),
    reason="run_manifest.json not present in this checkout",
)
def test_the_wall_time_estimate_is_measured_rather_than_typed() -> None:
    """T122.3's total is summed from recorded run start/finish times, never asserted."""
    measured = run_all.estimate_from_manifest()
    assert measured["n_runs_counted"] > 0
    assert measured["total_hours"] > 0
    assert "experiment_EXP-A2" in measured["by_stage_hours"]
