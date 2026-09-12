"""The whole pipeline as one ordered list of commands (T122.1, T122.2).

``scripts/00_run_everything.py`` is the command; this module is the plan behind
it. Every stage is declared here in dependency order with the exact argv the
evidence index records as that artifact's reproduction command, so "how do I
rebuild this project" and "what does this artifact's `command` column say" are
the same answer rather than two lists that drift apart.

Three things are deliberate.

**Stages, not a DAG.** The real dependency graph is a straight line with a few
independent branches, and a scheduler would add machinery nobody needs to run a
pipeline whose slowest stage is fourteen hours on its own. The order below is
the dependency order; ``--from`` and ``--only`` are the escape hatches.

**Resume is by artifact, not by bookkeeping.** ``--resume`` skips a stage whose
declared outputs are already on disk. It is the same rule a human uses, it
survives a crash, and it cannot claim a stage ran when its output is missing.
Long stages have their own internal checkpointing (Phase 37, Phase 63) on top of
this.

**Smoke is a real path, not a mock.** ``--smoke`` runs the same commands with
the reduced arguments the scripts already declare (``--smoke``, ``--limit``),
which is what CLAUDE.md requires before any long run. A stage with no smoke form
is skipped and says so, rather than pretending.

### The wall-clock reality

A complete run from an empty ``outputs/`` is **not** an afternoon. The committed
run manifest records ~67 h of CPU across 296 recorded runs on this machine, of
which EXP-A2 alone is ~23.6 h and EXP-C1 ~16.4 h. That number is reported by
``scripts/00_run_everything.py --estimate``, read from the manifest rather than
guessed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "MANIFEST_FILENAME",
    "STAGES",
    "Stage",
    "StageResult",
    "estimate_from_manifest",
    "run_stages",
    "select",
]

log = get_logger("pipeline.run_all")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FILENAME = "reproduction_manifest.json"


@dataclass(frozen=True)
class Stage:
    """One command in the reproduction, with what it writes and how to smoke it."""

    stage_id: str
    title: str
    #: Argv AFTER the interpreter. ``["scripts/01_run_dataset_audit.py"]`` runs
    #: as ``python scripts/01_run_dataset_audit.py``; a stage whose first element
    #: is ``"-c"`` is a module call, exactly as the evidence index records it.
    argv: tuple[str, ...]
    #: Paths, relative to the project root, that prove this stage has run.
    #: ``--resume`` skips the stage when every one of them exists.
    produces: tuple[str, ...] = ()
    #: Argv for ``--smoke``. ``None`` means "no reduced form": the stage is
    #: skipped in a smoke run and reported as skipped.
    smoke_argv: tuple[str, ...] | None = None
    #: Part of the frontend chain (T122.2), skippable with ``--skip-frontend``.
    frontend: bool = False
    #: Run from ``frontend/`` rather than the project root.
    cwd: str = "."
    #: An npm command rather than a Python one.
    npm: bool = False


def _py(*argv: str) -> tuple[str, ...]:
    return argv


#: The pipeline, in dependency order. Part numbers refer to `Docs/todo.md`.
STAGES: tuple[Stage, ...] = (
    # -- Part II: data ------------------------------------------------------
    Stage(
        "env",
        "Verify the interpreter, the packages and the dataset roots",
        _py("scripts/verify_env.py"),
        smoke_argv=_py("scripts/verify_env.py"),
    ),
    Stage(
        "dataset_audit",
        "Audit all four corpora and build the master metadata table",
        _py("scripts/01_run_dataset_audit.py"),
        produces=("outputs/01_dataset_audit/metadata_master.csv",),
        smoke_argv=_py("scripts/01_run_dataset_audit.py"),
    ),
    # -- Part III: preprocessing -------------------------------------------
    Stage(
        "preprocessing_quality",
        "Signal-quality scan over every recording",
        _py("-c", "from src.preprocessing.quality import run_quality_scan; run_quality_scan()"),
        produces=("outputs/02_preprocessing/signal_quality_scan.csv",),
    ),
    Stage(
        "preprocessing_settings",
        "Preprocessing settings table and ablation grid",
        _py(
            "-c",
            "from src.preprocessing.ablation import write_settings, write_grid; "
            "write_settings(); write_grid()",
        ),
        produces=("outputs/02_preprocessing/preprocessing_settings.csv",),
        smoke_argv=_py(
            "-c",
            "from src.preprocessing.ablation import write_settings, write_grid; "
            "write_settings(); write_grid()",
        ),
    ),
    Stage(
        "preprocessing_figures",
        "Before/after figures and the filter transfer function",
        _py(
            "-c",
            "from src.preprocessing.figures import generate_all; generate_all(); "
            "from src.preprocessing.filters import plot_transfer_function; "
            "plot_transfer_function()",
        ),
        produces=("outputs/02_preprocessing",),
    ),
    # -- Part IV: features --------------------------------------------------
    Stage(
        "features",
        "Extract the locked 138 features over all 7,536 recordings",
        _py("scripts/02_extract_features.py", "--dataset", "all"),
        produces=("outputs/03_features/all_features_matrix.parquet",),
        smoke_argv=_py("scripts/02_extract_features.py", "--dataset", "all", "--smoke"),
    ),
    Stage(
        "feature_artifacts",
        "FE-03 artifacts and the feature figures",
        _py(
            "-c",
            "from src.feature_extraction.extractor import write_feature_artifacts; "
            "write_feature_artifacts(); "
            "from src.feature_extraction.figures import generate_all; generate_all()",
        ),
        produces=("outputs/03_features",),
    ),
    Stage(
        "feature_reports",
        "Feature reports and distribution tables",
        _py("scripts/03_feature_reports.py"),
        produces=("outputs/03_features",),
    ),
    # -- Part V: models -----------------------------------------------------
    Stage(
        "model_smoke",
        "Fit every declared model once and record its smoke table",
        _py("scripts/04_model_smoke.py"),
        produces=("outputs/04_models",),
        smoke_argv=_py("scripts/04_model_smoke.py", "--smoke"),
    ),
    # -- Part VI: search ----------------------------------------------------
    Stage(
        "search_random",
        "SO-01 random search",
        _py("scripts/05_run_search.py", "--method", "random"),
        produces=("outputs/05_search_optimization",),
        smoke_argv=_py("scripts/05_run_search.py", "--method", "random", "--smoke"),
    ),
    Stage(
        "search_bayes",
        "SO-02 Bayesian search",
        _py("scripts/05_run_search.py", "--method", "bayes"),
        produces=("outputs/05_search_optimization",),
        smoke_argv=_py("scripts/05_run_search.py", "--method", "bayes", "--smoke"),
    ),
    Stage(
        "feature_selection",
        "SO-04 feature selection inside the training folds",
        _py("scripts/06_run_feature_selection.py"),
        produces=("outputs/05_search_optimization",),
    ),
    Stage(
        "population_ga",
        "SO-03 genetic algorithm",
        _py("scripts/07_run_population_search.py", "--method", "ga"),
        produces=("outputs/05_search_optimization",),
    ),
    Stage(
        "population_pso",
        "SO-03 particle swarm",
        _py("scripts/07_run_population_search.py", "--method", "pso"),
        produces=("outputs/05_search_optimization",),
    ),
    Stage(
        "weight_optimization",
        "SO-05 ensemble weight optimization",
        _py("scripts/08_run_weight_optimization.py"),
        produces=("outputs/05_search_optimization",),
        smoke_argv=_py("scripts/08_run_weight_optimization.py", "--smoke"),
    ),
    Stage(
        "multi_objective",
        "SO-06 Pareto front",
        _py("scripts/09_run_multi_objective.py"),
        produces=("outputs/05_search_optimization",),
    ),
    Stage(
        "search_reports",
        "T05-T07 and the search figures",
        _py("scripts/10_search_reports.py"),
        produces=("outputs/05_search_optimization",),
    ),
    # -- Part VII: experiments ---------------------------------------------
    Stage(
        "exp_a1",
        "EXP-A1 untuned baseline over the 25-fold map",
        _py("scripts/11_run_experiment.py", "--exp", "EXP-A1"),
        produces=("outputs/06_binary_results/EXP-A1/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_a2",
        "EXP-A2 nested search over the 25-fold map (the longest stage)",
        _py("scripts/11_run_experiment.py", "--exp", "EXP-A2", "--planner", "nested"),
        produces=("outputs/06_binary_results/EXP-A2/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_a2_subset",
        "EXP-A2 against the SO-04 20-feature subset",
        _py(
            "scripts/11_run_experiment.py",
            "--exp",
            "EXP-A2",
            "--planner",
            "nested_subset",
            "--variant",
            "so04_subset",
        ),
        produces=("outputs/06_binary_results/EXP-A2-so04_subset/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_b1",
        "EXP-B1 PASCAL A four-class, tuned and untuned",
        _py("scripts/11_run_experiment.py", "--exp", "EXP-B1", "--planner", "nested"),
        produces=("outputs/07_multiclass_results/EXP-B1/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_b1_defaults",
        "EXP-B1 defaults variant",
        _py("scripts/11_run_experiment.py", "--exp", "EXP-B1", "--variant", "defaults"),
        produces=("outputs/07_multiclass_results/EXP-B1-defaults/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_b2",
        "EXP-B2 PASCAL B three-class",
        _py("scripts/11_run_experiment.py", "--exp", "EXP-B2", "--planner", "nested"),
        produces=("outputs/07_multiclass_results/EXP-B2/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_b2_defaults",
        "EXP-B2 defaults variant",
        _py("scripts/11_run_experiment.py", "--exp", "EXP-B2", "--variant", "defaults"),
        produces=("outputs/07_multiclass_results/EXP-B2-defaults/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_c1_three",
        "EXP-C1 CirCor murmur, three-class (headline)",
        _py(
            "scripts/11_run_experiment.py",
            "--exp",
            "EXP-C1",
            "--planner",
            "nested",
            "--variant",
            "three_class",
        ),
        produces=(
            "outputs/08_circor_external_validation/EXP-C1-three_class/aggregate_metrics.csv",
        ),
    ),
    Stage(
        "exp_c1_two",
        "EXP-C1 CirCor murmur, two-class on the 874 known patients",
        _py(
            "scripts/11_run_experiment.py",
            "--exp",
            "EXP-C1",
            "--planner",
            "nested",
            "--variant",
            "two_class",
        ),
        produces=("outputs/08_circor_external_validation/EXP-C1-two_class/aggregate_metrics.csv",),
    ),
    Stage(
        "exp_c2",
        "EXP-C2 CirCor clinical outcome",
        _py("scripts/11_run_experiment.py", "--exp", "EXP-C2", "--planner", "nested"),
        produces=("outputs/08_circor_external_validation/EXP-C2/aggregate_metrics.csv",),
    ),
    Stage(
        "finalize_binary",
        "T09, the selection rule and models_saved/binary/final/",
        _py("scripts/13_finalize_binary_model.py"),
        produces=("models_saved/binary/final/manifest.json",),
    ),
    Stage(
        "finalize_tasks",
        "models_saved/{pascal_a,pascal_b,murmur,outcome}/final/",
        _py("scripts/46_finalize_task_models.py"),
        produces=(
            "models_saved/pascal_a/final/manifest.json",
            "models_saved/pascal_b/final/manifest.json",
            "models_saved/murmur/final/manifest.json",
            "models_saved/outcome/final/manifest.json",
        ),
    ),
    # -- Part VIII: analyses ------------------------------------------------
    Stage(
        "location_analysis",
        "T22 auscultation-location analysis",
        _py("scripts/21_location_analysis.py"),
        produces=("outputs/08_circor_external_validation/T22_auscultation_location_analysis.csv",),
    ),
    Stage(
        "cross_dataset",
        "EXP-D1 adult-to-paediatric transfer and T16",
        _py("scripts/22_cross_dataset.py"),
        produces=("outputs/08_circor_external_validation/T16_cross_dataset_generalization.csv",),
    ),
    Stage(
        "noise_robustness",
        "EXP-E1 noise robustness",
        _py("scripts/23_noise_robustness.py"),
        produces=("outputs/09_robustness_analysis/EXP-E1",),
    ),
    Stage(
        "duration_robustness",
        "EXP-E2 duration robustness",
        _py("scripts/24_duration_robustness.py"),
        produces=("outputs/09_robustness_analysis/EXP-E2",),
    ),
    Stage(
        "preprocessing_ablation",
        "PP-A preprocessing ablation",
        _py("scripts/25_preprocessing_ablation.py"),
        produces=("outputs/09_ablation",),
    ),
    Stage(
        "feature_ablation",
        "EXP-F1 feature-family ablation",
        _py("scripts/26_feature_ablation.py"),
        produces=("outputs/09_ablation",),
    ),
    Stage(
        "optimization_ablation",
        "EXP-F2 optimization-stage ablation",
        _py("scripts/27_optimization_ablation.py"),
        produces=("outputs/09_ablation",),
    ),
    Stage(
        "diagnosis_track",
        "EXP-G1 PhysioNet diagnosis multiclass track",
        _py("scripts/28_diagnosis_track.py"),
        produces=("outputs/07_multiclass_results/EXP-G1",),
    ),
    Stage(
        "source_holdout",
        "EXP-F3 leave-one-sub-collection-out",
        _py("scripts/29_source_holdout.py"),
        produces=("outputs/10_robustness",),
    ),
    Stage(
        "calibration",
        "Calibration and confidence analysis",
        _py("scripts/30_calibration_analysis.py"),
        produces=("outputs/06_binary_results",),
    ),
    Stage(
        "complexity",
        "T24-T26 complexity, model size and timing",
        _py("scripts/31_complexity_analysis.py"),
        produces=("outputs/11_complexity",),
    ),
    Stage(
        "failure_analysis",
        "T27 false-positive and false-negative analysis",
        _py("scripts/32_failure_analysis.py"),
        produces=("outputs/06_binary_results",),
    ),
    Stage(
        "explainability",
        "T23 global importance, family contribution and SHAP",
        _py("scripts/33_explainability.py"),
        produces=("outputs/11_complexity",),
    ),
    Stage(
        "statistics",
        "T28 statistical validation and its reporting",
        _py("scripts/34_statistical_validation.py"),
        produces=("outputs/12_statistics",),
    ),
    Stage(
        "statistics_reports",
        "Statistical summary tables and figures",
        _py("scripts/35_statistical_reporting.py"),
        produces=("outputs/12_statistics",),
    ),
    Stage(
        "cycle_analysis",
        "SEG-01 cardiac-cycle analysis",
        _py("scripts/36_cycle_analysis.py"),
        produces=("outputs/02_preprocessing",),
    ),
    # -- Part IX: deliverables ---------------------------------------------
    Stage(
        "tables_setup",
        "T01-T07 setup tables",
        _py(
            "scripts/18_setup_tables.py",
            "--tables",
            "T01", "T02", "T03", "T04", "T05", "T06", "T07",
        ),
        produces=("outputs/01_dataset_audit",),
    ),
    Stage(
        "tables_binary",
        "T08 and T10 binary result tables",
        _py("scripts/14_binary_tables.py"),
        produces=("outputs/06_binary_results",),
    ),
    Stage(
        "tables_multiclass_a",
        "T11 PASCAL A results",
        _py("scripts/15_multiclass_tables.py", "--exp", "EXP-B1"),
        produces=("outputs/07_multiclass_results/T11_pascal_a_results.csv",),
    ),
    Stage(
        "tables_multiclass_b",
        "T12 PASCAL B results",
        _py("scripts/15_multiclass_tables.py", "--exp", "EXP-B2"),
        produces=("outputs/07_multiclass_results/T12_pascal_b_results.csv",),
    ),
    Stage(
        "tables_circor",
        "T13-T15 CirCor tables",
        _py("scripts/18_circor_tables.py"),
        produces=("outputs/08_circor_external_validation/T13_circor_murmur_results.csv",),
    ),
    Stage(
        "tables_results",
        "T08-T15 rendered result tables",
        _py(
            "scripts/37_result_tables.py",
            "--tables",
            "T08", "T09", "T10", "T11", "T12", "T13", "T14", "T15",
        ),
        produces=("outputs/06_binary_results",),
    ),
    Stage(
        "graphs_data",
        "G01-G10 data graphs",
        _py(
            "scripts/19_data_graphs.py",
            "--figures",
            "G01", "G02", "G03", "G04", "G05", "G06", "G07", "G08", "G09", "G10",
        ),
        produces=("outputs/13_figures_diagrams",),
    ),
    Stage(
        "graphs_results",
        "G11-G17 and G28 result graphs",
        _py(
            "scripts/38_result_graphs.py",
            "--figures",
            "G11", "G12", "G13", "G14", "G15", "G16", "G17", "G28",
        ),
        produces=("outputs/13_figures_diagrams",),
    ),
    Stage(
        "diagrams",
        "F01-F20 method diagrams",
        _py("scripts/05_render_diagrams.py"),
        produces=("outputs/13_figures_diagrams",),
    ),
    Stage(
        "algorithms",
        "ALG-01 to ALG-20",
        _py("scripts/39_export_algorithms.py"),
        produces=("outputs/14_algorithms",),
    ),
    Stage(
        "equations",
        "The equations reference",
        _py("scripts/40_export_equations.py"),
        produces=("outputs/14_algorithms",),
    ),
    Stage(
        "literature",
        "LIT-01 and LIT-02 literature review",
        _py("scripts/41_literature_review.py"),
        produces=("outputs/16_literature_review",),
    ),
    Stage(
        "matrices",
        "T29 objective coverage and T30 final conclusion matrix",
        _py("scripts/37_result_tables.py", "--tables", "T29", "T30"),
        produces=(
            "outputs/00_evidence_index/T29_objective_to_evidence_mapping.csv",
            "outputs/00_evidence_index/T30_final_conclusion_matrix.csv",
        ),
    ),
    Stage(
        "q1_assets",
        "The Q1 / IEEE paper asset pack",
        _py("scripts/43_q1_assets.py"),
        produces=("outputs/Q1_PAPER_ASSETS",),
    ),
    Stage(
        "thesis_assets",
        "The thesis asset pack",
        _py("scripts/44_thesis_assets.py"),
        produces=("outputs/THESIS_ASSETS",),
    ),
    Stage(
        "evidence_index",
        "Assemble evidence_index.xlsx and the completeness audit",
        _py("scripts/42_evidence_index.py"),
        produces=("outputs/00_evidence_index/evidence_index.xlsx",),
        smoke_argv=_py("scripts/42_evidence_index.py"),
    ),
    # -- Part X: the frontend chain (T122.2) -------------------------------
    Stage(
        "npm_ci",
        "Install the pinned frontend dependencies",
        ("ci",),
        produces=("frontend/node_modules",),
        smoke_argv=("ci",),
        frontend=True,
        cwd="frontend",
        npm=True,
    ),
    Stage(
        "frontend_build",
        "Guard rail, export, next build, bundle budget and displayed-value audit",
        ("run", "build"),
        produces=("frontend/out/index.html", "frontend/.display-audit.json"),
        smoke_argv=("run", "build"),
        frontend=True,
        cwd="frontend",
        npm=True,
    ),
    Stage(
        "screenshots",
        "The thirteen gated dashboard screenshots",
        _py("scripts/47_dashboard_screenshots.py"),
        produces=("outputs/15_dashboard_screenshots/screenshot_index.csv",),
        frontend=True,
    ),
    Stage(
        "evidence_index_final",
        "Re-assemble the evidence index with the screenshots registered",
        _py("scripts/42_evidence_index.py"),
        produces=(),
        smoke_argv=_py("scripts/42_evidence_index.py"),
        frontend=True,
    ),
)


@dataclass
class StageResult:
    """What one stage did."""

    stage_id: str
    status: str
    seconds: float
    returncode: int | None = None
    command: str = ""
    detail: str = ""


def select(
    *,
    only: list[str] | None = None,
    start: str | None = None,
    skip_frontend: bool = False,
) -> list[Stage]:
    """The stages this invocation will run, in order."""
    stages = list(STAGES)
    known = {stage.stage_id for stage in stages}
    for name in list(only or []) + ([start] if start else []):
        if name not in known:
            raise KeyError(
                "unknown stage " + repr(name) + "; run --list to see the " + str(len(stages))
                + " declared stages"
            )
    if start is not None:
        index = next(i for i, stage in enumerate(stages) if stage.stage_id == start)
        stages = stages[index:]
    if only:
        wanted = set(only)
        stages = [stage for stage in stages if stage.stage_id in wanted]
    if skip_frontend:
        stages = [stage for stage in stages if not stage.frontend]
    return stages


def _satisfied(stage: Stage) -> bool:
    return bool(stage.produces) and all(
        (PROJECT_ROOT / path).exists() for path in stage.produces
    )


def _command(stage: Stage, *, smoke: bool, npm_executable: str) -> list[str] | None:
    argv = stage.smoke_argv if smoke else stage.argv
    if argv is None:
        return None
    if stage.npm:
        return [npm_executable, *argv]
    return [sys.executable, *argv]


def run_stages(
    stages: list[Stage],
    *,
    smoke: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    out_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the selected stages in order, timing each and writing the manifest."""
    import shutil

    npm_executable = shutil.which("npm") or shutil.which("npm.cmd") or "npm"
    results: list[StageResult] = []
    started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()

    for stage in stages:
        command = _command(stage, smoke=smoke, npm_executable=npm_executable)
        printable = " ".join(command) if command else ""
        if command is None:
            log.info("skip (no smoke form): %s", stage.stage_id)
            results.append(
                StageResult(stage.stage_id, "skipped-no-smoke-form", 0.0, detail=stage.title)
            )
            continue
        if resume and _satisfied(stage):
            log.info("skip (outputs present): %s", stage.stage_id)
            results.append(
                StageResult(stage.stage_id, "skipped-present", 0.0, command=printable)
            )
            continue
        if dry_run:
            results.append(StageResult(stage.stage_id, "dry-run", 0.0, command=printable))
            continue

        log.info("stage %s: %s", stage.stage_id, stage.title)
        clock = time.perf_counter()
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT / stage.cwd, check=False
        )
        seconds = time.perf_counter() - clock
        status = "ok" if completed.returncode == 0 else "failed"
        results.append(
            StageResult(
                stage.stage_id,
                status,
                round(seconds, 3),
                returncode=completed.returncode,
                command=printable,
            )
        )
        log.info("stage %s: %s in %.1f s", stage.stage_id, status, seconds)
        if status == "failed":
            break

    total = time.perf_counter() - started
    payload: dict[str, Any] = {
        "mode": "smoke" if smoke else "full",
        "resume": resume,
        "started_utc": started_utc,
        "finished_utc": datetime.now(UTC).isoformat(),
        "total_seconds": round(total, 3),
        "total_hours": round(total / 3600.0, 4),
        "n_stages": len(results),
        "n_ok": sum(1 for item in results if item.status == "ok"),
        "n_failed": sum(1 for item in results if item.status == "failed"),
        "n_skipped": sum(1 for item in results if item.status.startswith("skipped")),
        "stages": [vars(item) for item in results],
    }
    if not dry_run:
        target = _manifest_path(out_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        payload["manifest"] = str(target)
    return payload


def _manifest_path(out_root: str | Path | None = None) -> Path:
    if out_root is not None:
        return Path(out_root) / MANIFEST_FILENAME
    try:
        from src.utils.config import load_config

        return Path(load_config("paths").require("outputs.evidence_index")) / MANIFEST_FILENAME
    except Exception:  # noqa: BLE001 -- a missing config must not lose the timings
        return PROJECT_ROOT / "outputs" / "00_evidence_index" / MANIFEST_FILENAME


def estimate_from_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Measured wall time of a complete pipeline, from the committed run manifest.

    Every stage that has ever run recorded its start and finish in
    ``run_manifest.json``. Summing the distinct run durations is a *measurement*
    of what a clean run costs on this machine, not an estimate -- which is why
    this is reported rather than a number anybody typed.
    """

    target = (
        Path(path)
        if path is not None
        else PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    by_name: dict[str, float] = {}
    counted = 0
    for run in payload.get("runs", []):
        start, finish = run.get("started_utc"), run.get("finished_utc")
        if not start or not finish:
            continue
        seconds = (
            datetime.fromisoformat(finish) - datetime.fromisoformat(start)
        ).total_seconds()
        if seconds <= 0:
            continue
        name = str(run.get("name", "?"))
        by_name[name] = by_name.get(name, 0.0) + seconds
        counted += 1
    total = sum(by_name.values())
    return {
        "source": str(target),
        "n_runs_counted": counted,
        "total_seconds": round(total, 1),
        "total_hours": round(total / 3600.0, 2),
        "by_stage_hours": {
            name: round(seconds / 3600.0, 3)
            for name, seconds in sorted(by_name.items(), key=lambda kv: -kv[1])
        },
    }
