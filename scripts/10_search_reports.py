"""Part VI reporting: T07 and the figures G20, G21, G22 (Phase 62).

Emits:

    outputs/05_search_optimization/search_space_and_best_parameters.csv   T07
    outputs/13_figures_diagrams/search_convergence_plot.png               G20
    outputs/13_figures_diagrams/all_features_vs_selected_features.png     G21
    outputs/13_figures_diagrams/f1_accuracy_vs_feature_count.png          G22

and registers every SO artifact that exists in the evidence index (T62.5).

Reads what the searches wrote and draws it. It fits nothing and computes no
metric, so every number on every axis is traceable to a CSV under
``outputs/05_search_optimization/``.

Usage
-----
    python scripts/10_search_reports.py
    python scripts/10_search_reports.py --check   # report coverage, write nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/10_search_reports.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("search_reports")

#: T62.5 -- every artifact Part VI produced, and the evidence id it registers under.
#: A file that is absent is reported, never silently skipped.
SO_ARTIFACTS: list[tuple[str, str, str]] = [
    ("SO-01", "SO-01/trials.csv", "SO-01 trial history (random search)"),
    ("SO-01-BEST", "SO-01/best_parameters.json", "SO-01 best hyperparameters per model and fold"),
    ("SO-01-CONV", "SO-01/convergence.csv", "SO-01 best-so-far versus trial index"),
    ("SO-02", "SO-02/trials.csv", "SO-02 trial history (bayes search)"),
    ("SO-02-BEST", "SO-02/best_parameters.json", "SO-02 best hyperparameters per model and fold"),
    ("SO-02-CONV", "SO-02/convergence.csv", "SO-02 best-so-far versus trial index"),
    ("SO-COMPARE", "search_method_comparison.csv", "SO-01 vs SO-02 at equal budget"),
    ("SO-03a", "SO-03a/convergence.csv", "SO-03a genetic algorithm convergence trace"),
    ("SO-03a-BEST", "SO-03a/best_subset.json", "SO-03a selected feature mask and budget"),
    ("SO-03a-OUTER", "SO-03a/nested_outcomes.csv", "SO-03a winning mask on the held-out fold"),
    ("SO-03a-JOINT", "SO-03a/joint_weights.json", "SO-03a joint mask+weights chromosome (T58.5)"),
    ("SO-03b", "SO-03b/convergence.csv", "SO-03b particle swarm convergence trace"),
    ("SO-03b-BEST", "SO-03b/best_subset.json", "SO-03b selected feature mask and budget"),
    ("SO-03b-OUTER", "SO-03b/nested_outcomes.csv", "SO-03b winning mask on the held-out fold"),
    ("SO-03b-WEIGHTS", "SO-03b/weight_swarm.json", "SO-03b PSO over the ensemble-weight simplex"),
    ("SO-03-COMPARE", "ga_vs_pso_comparison.csv", "SO-03a vs SO-03b at identical folds and budget"),
    ("SO-04", "SO-04/feature_count_curve.csv", "SO-04 performance versus feature count"),
    ("SO-04-SWEEP", "SO-04/feature_count_sweep.csv",
     "SO-04 every (ranker, k) on every inner fold"),
    ("SO-04-COMPARE", "SO-04/all_features_vs_selected.csv", "SO-04 all 138 versus the subset"),
    ("SO-04-RFECV", "SO-04/rfecv_selection.csv", "SO-04 RFECV stopping point per outer fold"),
    ("SO-04-THRESH", "SO-04/embedded_thresholds.csv",
     "SO-04 RF/GB importance-threshold subsets"),
    ("SO-05", "SO-05/equal_vs_optimized.csv", "SO-05 equal weights versus optimized, per fold"),
    ("SO-05-STABILITY", "SO-05/weight_stability.csv",
     "SO-05 per-member weight spread across folds"),
    ("SO-05-FINAL", "SO-05/final_weights.json", "SO-05 final weight vector and its constraint"),
    ("SO-06", "SO-06/pareto_front.csv", "SO-06 Pareto front over the three objectives"),
    ("SO-06-SWEEP", "SO-06/weighting_sweep.csv",
     "SO-06 which configuration each weighting selects"),
    ("SO-06-POINT", "SO-06/operating_point.json", "SO-06 final operating point on the front"),
    ("SO-06-J", "SO-06/j_definition.json", "SO-06 multi-objective score J as implemented"),
    ("T07", "search_space_and_best_parameters.csv",
     "T07 search space, distribution and final selected value per model"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="10_search_reports",
        description="Build the Part VI deliverables: T07, G20, G21, G22.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report which SO artifacts exist and exit; write nothing",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.reporting import search_report as sr
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv
    from src.utils.run_manifest import start_run
    from src.utils.seed import set_global_seed

    args = parse_args(argv)
    set_global_seed()
    section = sr.search_section(args.out_dir)
    command = "python scripts/10_search_reports.py"

    def scan() -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
        here = [(eid, rel, lab) for eid, rel, lab in SO_ARTIFACTS if (section / rel).is_file()]
        gone = [(eid, rel) for eid, rel, _ in SO_ARTIFACTS if not (section / rel).is_file()]
        return here, gone

    if args.check:
        present, absent = scan()
        log.info("%d of %d SO artifacts present", len(present), len(SO_ARTIFACTS))
        for eid, rel in absent:
            log.warning("  absent: %s -> %s", eid, rel)
        return 0 if not absent else 1

    run = start_run("search_reports")
    run.set("exp", "PART-VI")
    started = time.perf_counter()

    # --- T62.1: T07 ----------------------------------------------------------
    t07 = sr.build_t07(section)
    t07_path = save_csv(t07, section / sr.T07_FILENAME)
    log.info("T07: %d parameter rows over %d models", len(t07), t07["model_id"].nunique())

    # --- T62.2 / T62.3 / T62.4: the figures ---------------------------------
    figures = sr.figures_dir()
    written: dict[str, Path] = {}
    for figure_id, builder in (
        ("G20", sr.plot_convergence),
        ("G21", sr.plot_all_versus_selected),
        ("G22", sr.plot_feature_count_curve),
    ):
        target = figures / sr.FIGURES[figure_id]
        written[figure_id] = builder(target, section)
        log.info("%s -> %s", figure_id, written[figure_id].name)

    # --- T62.5: every SO artifact in the evidence index ---------------------
    #
    # Scanned AFTER T07 is written, not before: T07 is one of the artifacts on
    # the list and this script is what creates it, so a scan taken at the top
    # reports the file this run is about to produce as missing.
    present, absent = scan()
    registered = 0
    for evidence_id, relative, label in present:
        register_evidence(
            evidence_id, section / relative,
            metric_or_asset=label, dataset="D1", command=command,
        )
        registered += 1
    for figure_id, path in written.items():
        register_evidence(
            figure_id, path,
            metric_or_asset={
                "G20": "G20 search convergence, hyperparameter and mask searches on separate axes",
                "G21": "G21 all 138 features versus the selected subsets on held-out folds",
                "G22": "G22 performance versus feature count, per ranker, on inner folds",
            }[figure_id],
            dataset="D1",
            source_data="outputs/05_search_optimization/",
            command=command,
        )
        registered += 1
    register_evidence(
        "T07", t07_path,
        metric_or_asset="T07 search space, distribution and final selected value per model",
        dataset="D1", command=command,
    )

    for path in [t07_path, *written.values()]:
        run.record_artifact(path)
    run.record_timing("part_vi_report", time.perf_counter() - started)

    log.info(
        "registered %d Part VI artifacts; %d declared artifact(s) absent",
        registered, len(absent),
    )
    for evidence_id, relative in absent:
        log.warning("  absent, not registered: %s -> %s", evidence_id, relative)
    print(t07.head(12).to_string(index=False))
    run.finish(status="ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
