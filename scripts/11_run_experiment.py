"""Run one declared experiment end to end (T63.5).

The single entry point for every run in Part VII. Which models, which folds,
which label space and where the output goes are all declared in
``configs/experiments.yaml`` -- this script chooses none of them, it only
selects an experiment id and, optionally, a subset of it.

Script numbering note: ``todo.md`` calls this ``scripts/03_run_experiment.py``,
but 01-10 were taken by the audit, extraction, feature reports, model smoke,
search, feature selection, population search, weight optimization,
multi-objective and search-report runners. The drift is deliberate and recorded
in ``Docs/note.md``; the file name is what matters, not the number.

Usage
-----
    python scripts/11_run_experiment.py --exp EXP-A1 --smoke
    python scripts/11_run_experiment.py --exp EXP-A1
    python scripts/11_run_experiment.py --exp EXP-A1 --models M1 M3 --repeats 0
    python scripts/11_run_experiment.py --exp EXP-A2 --planner nested
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/11_run_experiment.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("run_experiment")

PLANNERS = ("default", "tuned", "nested", "nested_subset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="11_run_experiment",
        description="Run one experiment from configs/experiments.yaml over its fold map.",
    )
    parser.add_argument("--exp", required=True, metavar="EXP-ID")
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="ID",
        help="subset of the experiment's declared models (default: all of them)",
    )
    parser.add_argument(
        "--repeats", nargs="+", type=int, default=None, metavar="R",
        help="which repeats of the fold map to run (default: all)",
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None, metavar="F",
        help="which folds within those repeats (default: all)",
    )
    parser.add_argument(
        "--planner", default="default", choices=list(PLANNERS),
        help=(
            "default = config defaults (EXP-A1); tuned = the T07 selected point; "
            "nested = a search inside every outer training fold (EXP-A2); "
            "nested_subset = nested, restricted to the SO-04 feature subset"
        ),
    )
    parser.add_argument(
        "--search-method", default="bayes", choices=["bayes", "random"],
        help="nested planners only: which search runs inside each outer fold",
    )
    parser.add_argument(
        "--trials", type=int, default=12,
        help="nested planners only: trial budget per model per outer fold",
    )
    parser.add_argument(
        "--model-trials", nargs="+", default=[], metavar="ID=N",
        help="nested planners only: per-model override, e.g. M5=5",
    )
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--out-dir", default=None, help="override the output root")
    parser.add_argument(
        "--variant", default=None, metavar="NAME",
        help=(
            "write to outputs/<section>/<EXP-ID>-<NAME>/ instead. Used by T65.3, "
            "whose SO-04 subset run is a variant of EXP-A2 and must not overwrite it"
        ),
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="recompute every (model, fold) even if a matching checkpoint exists",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="first declared model, repeat 0 only -- proves the path, not a result",
    )
    return parser.parse_args(argv)


def _model_trials(pairs: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError("--model-trials expects ID=N, got " + repr(pair))
        model_id, _, count = pair.partition("=")
        parsed[model_id.strip()] = int(count)
    return parsed


def build_planner(args: argparse.Namespace, exp: object) -> object:
    from src.evaluation.experiment import DefaultPlanner

    if args.planner == "default":
        return DefaultPlanner()

    from src.evaluation.tuned import NestedSearchPlanner, SubsetPlanner, TunedPlanner

    if args.planner == "tuned":
        return TunedPlanner()

    kind = SubsetPlanner if args.planner == "nested_subset" else NestedSearchPlanner
    return kind(
        method=args.search_method,
        trials=args.trials,
        model_trials=_model_trials(args.model_trials),
        inner_splits=args.inner_splits,
    )


def main(argv: list[str] | None = None) -> int:
    from src.evaluation import experiment as ex
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    exp = ex.Experiment.load(args.exp)

    models = list(args.models) if args.models else list(exp.models)
    repeats = args.repeats
    if args.smoke:
        models = models[:1]
        repeats = [0]

    planner = build_planner(args, exp)
    command = "python scripts/11_run_experiment.py --exp " + args.exp
    if args.planner != "default":
        command += " --planner " + args.planner

    if args.variant:
        # A variant keeps its own contract-complete folder rather than adding
        # columns to the headline run: T09 compares two runs, and a table that
        # silently mixed 138-feature and 20-feature rows would be unreadable.
        exp = dataclasses.replace(exp, variant=args.variant)
        command += " --variant " + args.variant
    out_dir = args.out_dir

    run = start_run("experiment_" + args.exp)
    run.set("exp", args.exp)
    run.set("title", exp.title)
    run.set("task", exp.task)
    run.set("cv", exp.cv)
    run.set("models", models)
    run.set("planner", getattr(planner, "name", args.planner))
    run.set("repeats", repeats)
    run.set("folds", args.folds)
    run.set("resume", not args.no_resume)
    run.set("smoke", bool(args.smoke))
    run.set("variant", args.variant)

    result = ex.run_experiment(
        exp,
        models=models,
        repeats=repeats,
        folds=args.folds,
        planner=planner,  # type: ignore[arg-type]
        pipeline_config=getattr(planner, "pipeline_config", None),
        out_dir=out_dir,
        resume=not args.no_resume,
    )
    written = ex.write_outputs(result, out_dir=out_dir, run=run, command=command)

    aggregate = result.aggregate_frame()
    headline = "balanced_accuracy" if exp.is_binary else "macro_f1"
    columns = ["model_id", "n_folds"]
    for metric in (headline, "sensitivity", "specificity", "f1", "roc_auc"):
        for suffix in ("_mean", "_sd"):
            name = metric + suffix
            if name in aggregate.columns:
                columns.append(name)
    print()
    print(aggregate[columns].round(4).to_string(index=False))
    print()
    print("contract written to " + str(written["per_fold_metrics.csv"].parent))
    log.info(
        "%s finished: %d computed, %d resumed, %.1f s",
        args.exp,
        result.n_computed,
        result.n_resumed,
        result.seconds,
    )
    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
