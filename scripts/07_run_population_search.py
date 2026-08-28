"""SO-03a genetic algorithm and SO-03b particle swarm over the feature mask (Phases 58-59).

Emits, into ``outputs/05_search_optimization/{SO-03a,SO-03b}/``:

    convergence.csv          per-generation / per-iteration best, mean and worst fitness
    best_subset.json         the winning mask, its features and the budget it was found under
    selected_features.csv    the winning mask as a feature list
    joint_weights.json       SO-03a only: the joint mask+weights chromosome (T58.5)
    weight_swarm.json        SO-03b only: PSO over the ensemble-weight simplex (T59.1)
    weight_convergence.csv   SO-03b only: that swarm's per-iteration trace

and, at the section root, ``ga_vs_pso_comparison.csv`` -- the two mask searches on
identical folds, budget and seed (T59.5).

Both searches minimise the multi-objective score J (T58.2), on inner folds cut
from one outer fold's training rows. The outer test fold is never scored; that is
enforced by ``src.optimization.masks.MaskEvaluator``'s row ledger and asserted
before either result is returned.

Usage
-----
    python scripts/07_run_population_search.py --method ga --smoke
    python scripts/07_run_population_search.py --method ga
    python scripts/07_run_population_search.py --method pso
    python scripts/07_run_population_search.py --compare
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/07_run_population_search.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("run_population_search")

METHODS = {"ga": "SO-03a", "pso": "SO-03b"}
COMPARISON_FILENAME = "ga_vs_pso_comparison.csv"
CONVERGENCE_FILENAME = "convergence.csv"
BEST_FILENAME = "best_subset.json"
FEATURES_FILENAME = "selected_features.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="07_run_population_search",
        description="Run SO-03a (GA) / SO-03b (PSO) feature-mask search.",
    )
    parser.add_argument("--method", default="ga", choices=sorted(METHODS))
    parser.add_argument("--task", default="binary")
    parser.add_argument("--model", default=None, metavar="ID",
                        help="fitness estimator (default: from config)")
    parser.add_argument("--population", type=int, default=None,
                        help="population / swarm size")
    parser.add_argument("--generations", type=int, default=None,
                        help="generations / iterations")
    parser.add_argument("--repeats", nargs="+", type=int, default=None, metavar="R")
    parser.add_argument("--folds", nargs="+", type=int, default=None, metavar="F")
    parser.add_argument("--inner-splits", type=int, default=None)
    parser.add_argument("--skip-joint", action="store_true",
                        help="SO-03a: skip the joint mask+weights chromosome (T58.5)")
    parser.add_argument("--skip-weight-swarm", action="store_true",
                        help="SO-03b: skip the ensemble-weight simplex swarm (T59.1)")
    parser.add_argument(
        "--rescore-only", action="store_true",
        help=(
            "re-score the masks already in best_subset.json on their outer test "
            "folds and rewrite nested_outcomes.csv. Costs a handful of fits; does "
            "not re-run the search"
        ),
    )
    parser.add_argument("--compare", action="store_true",
                        help="write ga_vs_pso_comparison.csv from what is on disk (T59.5)")
    parser.add_argument("--smoke", action="store_true",
                        help="a tiny population over a cheap estimator, to prove the path")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def _section_root(out_dir: str | None) -> Path:
    from src.utils.config import load_config

    return (
        Path(out_dir)
        if out_dir
        else Path(load_config("paths").require("outputs.search_optimization"))
    )


def _method_dir(method: str, out_dir: str | None) -> Path:
    from src.utils.io import ensure_dir

    return Path(ensure_dir(_section_root(out_dir) / METHODS[method]))


# ---------------------------------------------------------------------------
# T59.5 -- the comparison
# ---------------------------------------------------------------------------


def _comparison(out_dir: str | None):
    """GA against PSO, matched on outer fold. Refuses to compare unequal budgets."""
    import pandas as pd

    from src.utils.io import load_json

    root = _section_root(out_dir)
    loaded: dict[str, dict] = {}
    for method, exp in METHODS.items():
        path = root / exp / BEST_FILENAME
        if not path.is_file():
            raise SystemExit(
                "cannot compare: " + str(path) + " does not exist. Run "
                "`python scripts/07_run_population_search.py --method " + method + "` first."
            )
        loaded[method] = load_json(path)

    rows = []
    for fold_label in sorted(loaded["ga"]["by_fold"]):
        if fold_label not in loaded["pso"]["by_fold"]:
            continue
        ga = loaded["ga"]["by_fold"][fold_label]
        pso = loaded["pso"]["by_fold"][fold_label]
        ga_budget = int(ga["config"]["max_evaluations"])
        pso_budget = int(pso["config"]["max_evaluations"])
        if ga_budget != pso_budget:
            raise SystemExit(
                "refusing to write the comparison: on " + fold_label + " the GA had "
                + str(ga_budget) + " evaluations and PSO " + str(pso_budget)
                + ". T59.5 requires identical budgets; re-run one of them."
            )
        if ga["evaluator"]["seed"] != pso["evaluator"]["seed"]:
            raise SystemExit(
                "refusing to write the comparison: the two runs used different seeds "
                + str(ga["evaluator"]["seed"]) + " and " + str(pso["evaluator"]["seed"])
            )
        rows.append(
            {
                "outer_fold": fold_label,
                "budget_evaluations": ga_budget,
                "seed": int(ga["evaluator"]["seed"]),
                "fitness_model": ga["model_id"],
                "inner_splits": int(ga["evaluator"]["n_inner_splits"]),
                "ga_fitness": float(ga["best_fitness"]),
                "pso_fitness": float(pso["best_fitness"]),
                "fitness_delta_pso_minus_ga": (
                    float(pso["best_fitness"]) - float(ga["best_fitness"])
                ),
                "ga_macro_f1": float(ga["best_macro_f1"]),
                "pso_macro_f1": float(pso["best_macro_f1"]),
                "macro_f1_delta_pso_minus_ga": (
                    float(pso["best_macro_f1"]) - float(ga["best_macro_f1"])
                ),
                "ga_n_selected": int(ga["n_selected"]),
                "pso_n_selected": int(pso["n_selected"]),
                "ga_distinct_fitted": int(ga["evaluator"]["distinct_masks_fitted"]),
                "pso_distinct_fitted": int(pso["evaluator"]["distinct_masks_fitted"]),
                "ga_seconds": float(ga["seconds"]),
                "pso_seconds": float(pso["seconds"]),
                "winner": (
                    "ga" if float(ga["best_fitness"]) <= float(pso["best_fitness"]) else "pso"
                ),
            }
        )
    if not rows:
        raise SystemExit("no outer fold was searched by both methods; nothing to compare")
    return pd.DataFrame(rows)


@dataclass
class _StoredResult:
    """Just enough of a search result to re-score its mask (--rescore-only)."""

    best_mask: Any
    model_id: str
    outer_label: str
    best_fitness: float
    best_macro_f1: float
    n_selected: int


def _stored_results(exp: str, out_dir: str | None, feature_names) -> list[_StoredResult]:
    import numpy as np

    from src.utils.io import load_json

    path = _section_root(out_dir) / exp / BEST_FILENAME
    if not path.is_file():
        raise SystemExit("--rescore-only needs " + str(path) + ", which does not exist")
    payload = load_json(path)
    position_of = {str(name): index for index, name in enumerate(feature_names)}
    stored = []
    for label, summary in sorted(payload["by_fold"].items()):
        mask = np.zeros(len(position_of), dtype=bool)
        mask[[position_of[name] for name in summary["features"]]] = True
        stored.append(
            _StoredResult(
                best_mask=mask,
                model_id=str(summary["model_id"]),
                outer_label=str(label),
                best_fitness=float(summary["best_fitness"]),
                best_macro_f1=float(summary["best_macro_f1"]),
                n_selected=int(summary["n_selected"]),
            )
        )
    return stored


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import pandas as pd

    from src.feature_selection import sweep as fs
    from src.models import smoke as sm
    from src.optimization import driver as od
    from src.optimization import genetic as ga_mod
    from src.optimization import multi_objective as mo
    from src.optimization import swarm as pso_mod
    from src.optimization.masks import MaskEvaluator
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run
    from src.utils.seed import GLOBAL_SEED, set_global_seed

    args = parse_args(argv)
    set_global_seed()
    seed = GLOBAL_SEED

    if args.compare:
        frame = _comparison(args.out_dir)
        path = save_csv(frame, _section_root(args.out_dir) / COMPARISON_FILENAME)
        log.info("wrote %s (%d matched fold(s))", path, len(frame))
        register_evidence(
            "SO-03-COMPARE", path,
            metric_or_asset="SO-03a GA vs SO-03b PSO at identical folds, budget and seed",
            dataset="D1",
            command="python scripts/07_run_population_search.py --compare",
        )
        print(frame.to_string(index=False))
        return 0

    from src.utils.config import load_config

    config = load_config("models")
    method = args.method
    exp = METHODS[method]
    command = "python scripts/07_run_population_search.py --method " + method

    if method == "ga":
        settings = dict(config.get("optimization.genetic_algorithm") or {})
        search_config: object = ga_mod.load_ga_config(settings)
    else:
        settings = dict(config.get("optimization.particle_swarm") or {})
        search_config = pso_mod.load_pso_config(settings)

    model_id = args.model or str(settings.get("fitness_model", "M3"))
    inner_splits = int(args.inner_splits or settings.get("inner_splits", 3))
    repeats = args.repeats if args.repeats is not None else list(settings.get("repeats", [0]))
    folds = args.folds if args.folds is not None else list(settings.get("folds", [0]))
    min_features = int(settings.get("min_features", 10))

    if args.smoke:
        model_id = "M1"
        repeats, folds = [0], [0]
        if method == "ga":
            search_config = ga_mod.GAConfig(
                population_size=6, generations=3, elitism=1, tournament_size=2,
                min_features=min_features,
            )
        else:
            search_config = pso_mod.PSOConfig(
                swarm_size=6, iterations=3, min_features=min_features
            )
    else:
        if args.population or args.generations:
            if method == "ga":
                assert isinstance(search_config, ga_mod.GAConfig)
                search_config = ga_mod.GAConfig(
                    population_size=int(args.population or search_config.population_size),
                    generations=int(args.generations or search_config.generations),
                    crossover_rate=search_config.crossover_rate,
                    mutation_rate=search_config.mutation_rate,
                    tournament_size=search_config.tournament_size,
                    elitism=search_config.elitism,
                    min_features=search_config.min_features,
                    init_density=search_config.init_density,
                )
            else:
                assert isinstance(search_config, pso_mod.PSOConfig)
                search_config = pso_mod.PSOConfig(
                    swarm_size=int(args.population or search_config.swarm_size),
                    iterations=int(args.generations or search_config.iterations),
                    inertia=search_config.inertia,
                    cognitive=search_config.cognitive,
                    social=search_config.social,
                    velocity_clamp=search_config.velocity_clamp,
                    min_features=search_config.min_features,
                )

    run = start_run("population_search_" + method)
    run.set("exp", exp)
    run.set("task", args.task)
    run.set("fitness_model", model_id)
    run.set("config", search_config.as_dict())  # type: ignore[attr-defined]

    data = sm.load_task_data(args.task)
    outer = od.outer_folds_for(args.task, data, repeats=repeats, folds=folds)
    weights = mo.load_weights()
    cost_model = mo.load_cost_model()

    log.info(
        "%s on %s: %s, fitness %s, budget %d evaluations",
        exp, args.task, [f.label for f in outer], model_id,
        search_config.max_evaluations,  # type: ignore[attr-defined]
    )

    started = time.perf_counter()
    directory = _method_dir(method, args.out_dir)
    artifacts: list[Path] = []

    if args.rescore_only:
        # Re-scoring reads the masks the search already found and touches nothing
        # else, so the convergence trace and best_subset.json on disk stay exactly
        # as the search wrote them.
        results = _stored_results(exp, args.out_dir, data.feature_names)
        log.info("re-scoring %d stored mask(s) from %s", len(results), exp)
        keep = {item.outer_label for item in results}
        outer = tuple(fold for fold in outer if str(fold.label) in keep)
    else:
        results = []
        traces = []
        for fold in outer:
            evaluator = MaskEvaluator(
                data, fold,
                model_id=model_id,
                n_inner_splits=inner_splits,
                seed=seed,
                min_features=min_features,
                weights=weights,
                cost_model=cost_model,
            )

            def progress(record, label=str(fold.label)) -> None:
                log.info(
                    "  %s gen %02d: best J %.5f (macro-F1 %.4f, %d features), mean %.5f, "
                    "%d evals / %d fitted, %.0fs",
                    label, record.index, record.best, record.best_macro_f1,
                    record.best_n_selected, record.mean, record.evaluations,
                    record.distinct_fitted, record.seconds,
                )

            if method == "ga":
                result = ga_mod.run_ga(
                    evaluator, search_config, seed=seed, progress=progress  # type: ignore[arg-type]
                )
            else:
                result = pso_mod.run_binary_pso(
                    evaluator, search_config, seed=seed, progress=progress  # type: ignore[arg-type]
                )
            results.append(result)
            traces.append(result.trace_frame())

        convergence_path = save_csv(
            pd.concat(traces, ignore_index=True), directory / CONVERGENCE_FILENAME
        )
        payload = {
            "experiment": exp,
            "task": args.task,
            "j_weights": weights.as_dict(),
            "by_fold": {
                item.outer_label: item.as_summary(data.feature_names) for item in results
            },
        }
        best_path = save_json(payload, directory / BEST_FILENAME)

        rows = []
        for item in results:
            for position, column in enumerate(
                np.flatnonzero(np.asarray(item.best_mask, dtype=bool)), start=1
            ):
                name = str(data.feature_names[int(column)])
                rows.append(
                    {
                        "outer_fold": item.outer_label,
                        "rank": position,
                        "column": int(column),
                        "feature": name,
                        "family": mo.family_of(name),
                    }
                )
        features_path = save_csv(pd.DataFrame(rows), directory / FEATURES_FILENAME)
        artifacts.extend([convergence_path, best_path, features_path])

    # --- the winning mask, scored on the outer test fold it never saw --------
    #
    # Under the SO-04 evaluation model, not the fitness model. The GA and PSO
    # search with M3 because 750 evaluations at M4's 10.5 s a piece is 2.2 hours
    # per method; SO-04 scores its subsets with M4. A J from one and a J from the
    # other are therefore NOT the same quantity and must never share a column.
    # Re-scoring the winner under SO-04's estimator costs one fit per fold and
    # makes the two phases comparable on a number that is held out from both.
    outcome_rows = []
    evaluation_model = str(
        (config.get("optimization.feature_selection") or {}).get("evaluation_model", "M4")
    )
    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    for result, fold in zip(results, outer, strict=True):
        columns = np.flatnonzero(np.asarray(result.best_mask, dtype=bool))
        train = np.asarray(fold.train_index, dtype=int)
        test = np.asarray(fold.test_index, dtype=int)
        # BOTH estimators, because otherwise the drop from inner to outer is
        # confounded: the search scored with the fitness model on inner folds and
        # the comparison against SO-04 needs the evaluation model on the outer
        # fold, so a single number moves two things at once and neither can be
        # blamed. The fitness model on the outer fold separates them.
        for role, scoring_model in (
            ("fitness_model", str(result.model_id)),
            ("evaluation_model", evaluation_model),
        ):
            metrics, j, seconds = fs.score_subset(
                columns, scoring_model,
                features[train], targets[train], features[test], targets[test],
                feature_names=data.feature_names, weights=weights, cost_model=cost_model,
            )
            outcome_rows.append(
                {
                    "method": exp,
                    "outer_fold": result.outer_label,
                    "scored_with": scoring_model,
                    "role": role,
                    "fitness_model": str(result.model_id),
                    "n_selected": result.n_selected,
                    "inner_fitness_j": float(result.best_fitness),
                    "inner_macro_f1": float(result.best_macro_f1),
                    "outer_j": float(j.value),
                    "refit_seconds": float(seconds),
                    "n_train": int(train.size),
                    "n_test": int(test.size),
                    **{key: float(value) for key, value in metrics.items()},
                }
            )
            log.info(
                "%s %s: winner rescored with %s (%s) on the held-out fold -- "
                "macro-F1 %.4f, balanced accuracy %.4f",
                exp, result.outer_label, scoring_model, role,
                metrics["macro_f1"], metrics.get("balanced_accuracy", float("nan")),
            )
    outcome_path = save_csv(pd.DataFrame(outcome_rows), directory / "nested_outcomes.csv")

    artifacts.append(outcome_path)

    # --- T58.5 / T59.1: the second search each phase owns --------------------
    if method == "ga" and not args.skip_joint and not args.smoke and not args.rescore_only:
        joint_config = ga_mod.load_joint_config(settings)
        log.info(
            "T58.5 joint mask+weights GA: members %s, %d x %d = %d evaluations",
            joint_config.members, joint_config.population_size,
            joint_config.generations,
            joint_config.population_size * joint_config.generations,
        )
        joint_results = []
        for fold in outer:
            joint = ga_mod.run_joint_ga(
                data, fold, joint_config,
                n_inner_splits=inner_splits, seed=seed,
                weights=weights, cost_model=cost_model,
                progress=lambda record, label=str(fold.label): log.info(
                    "  %s joint gen %02d: best J %.5f (%d features), %.0fs",
                    label, record.index, record.best, record.best_n_selected, record.seconds,
                ),
            )
            joint_results.append(joint)
        artifacts.append(
            save_json(
                {
                    "experiment": exp,
                    "task": args.task,
                    "note": (
                        "T58.5. Reduced member set and budget: nothing caches across "
                        "masks, and the M7 member set costs 347 s per evaluation on "
                        "this fold. Demonstrates the joint encoding; NOT comparable "
                        "to M7."
                    ),
                    "by_fold": {
                        item.outer_label: item.as_summary(data.feature_names)
                        for item in joint_results
                    },
                },
                directory / "joint_weights.json",
            )
        )
        artifacts.append(
            save_csv(
                pd.concat([item.trace_frame() for item in joint_results], ignore_index=True),
                directory / "joint_convergence.csv",
            )
        )

    if (
        method == "pso" and not args.skip_weight_swarm and not args.smoke
        and not args.rescore_only
    ):
        weight_config = pso_mod.load_weight_config(settings)
        log.info(
            "T59.1 weight-simplex PSO: members %s, %d x %d evaluations on cached OOF "
            "probabilities",
            weight_config.members, weight_config.swarm_size, weight_config.iterations,
        )
        weight_results = []
        for fold in outer:
            weight_results.append(
                pso_mod.run_weight_pso(
                    data, fold, weight_config,
                    n_inner_splits=inner_splits, seed=seed,
                    progress=lambda record, label=str(fold.label): log.info(
                        "  %s weight iter %02d: best %.5f, mean %.5f",
                        label, record.index, record.best, record.mean,
                    ),
                )
            )
        artifacts.append(
            save_json(
                {
                    "experiment": exp,
                    "task": args.task,
                    "by_fold": {item.outer_label: item.as_summary() for item in weight_results},
                },
                directory / "weight_swarm.json",
            )
        )
        artifacts.append(
            save_csv(
                pd.concat([item.trace_frame() for item in weight_results], ignore_index=True),
                directory / "weight_convergence.csv",
            )
        )

    elapsed = time.perf_counter() - started
    for path in artifacts:
        run.record_artifact(path)
    run.record_timing(exp + "_total", elapsed)

    if not args.rescore_only:
        register_evidence(
            exp, convergence_path,
            metric_or_asset=exp + " convergence trace (per-generation best and mean fitness)",
            dataset="D1" if args.task == "binary" else args.task,
            source_data="outputs/03_features/all_features_matrix.parquet",
            command=command,
        )
        register_evidence(
            exp + "-BEST", best_path,
            metric_or_asset=exp + " selected feature mask and the budget it was found under",
            dataset="D1" if args.task == "binary" else args.task,
            command=command,
        )
    register_evidence(
        exp + "-OUTER", outcome_path,
        metric_or_asset=(
            exp + " winning mask refit on the outer training rows and scored once on "
            "the held-out fold, under both the fitness and the evaluation estimator"
        ),
        dataset="D1" if args.task == "binary" else args.task,
        command=command,
    )

    for result in results:
        if args.rescore_only:
            log.info(
                "%s %s: %d features, inner J %.5f, macro-F1 %.4f (from the stored search)",
                exp, result.outer_label, result.n_selected,
                result.best_fitness, result.best_macro_f1,
            )
            continue
        log.info(
            "%s %s: best J %.5f, macro-F1 %.4f, %d features, %d/%d distinct masks "
            "fitted, %.1f min",
            exp, result.outer_label, result.best_fitness, result.best_macro_f1,
            result.n_selected, result.evaluator["distinct_masks_fitted"],
            result.evaluator["evaluations"], result.seconds / 60.0,
        )
    log.info("%s finished in %.1f min", exp, elapsed / 60.0)
    run.finish(status="ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
