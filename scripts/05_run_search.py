"""Hyperparameter search runs: SO-01 randomized and SO-02 Bayesian (Phases 54-56).

Emits, into ``outputs/05_search_optimization/{SO-01,SO-02}/``:

    trials.csv               every trial: params, score, per-inner-fold scores, seconds (T54.4)
    best_parameters.json     best point per (model, outer fold), plus the budget (T55.6, T56.6)
    inner_fold_map.csv       the derived inner splits, materialised so they can be read
    nested_outcomes.csv      each best point refit on the outer train, scored on the outer test
    convergence.csv          best-so-far versus trial index (T56.4)
    search_capability.json   SO-02 only: whether scikit-optimize actually runs here (T56.1)

and, at the section root, ``search_method_comparison.csv`` -- SO-01 against SO-02
at equal budget on the same folds (T56.5).

The outer test fold is never scored during a search. That is enforced at run
time by ``src.optimization.base.RowLedger`` (every result asserts it before it
is returned) and tested in ``tests/test_search_no_leakage.py``.

Usage
-----
    python scripts/05_run_search.py --method random --smoke
    python scripts/05_run_search.py --method random --models M1 M3 M4 M5 M8 --trials 60
    python scripts/05_run_search.py --method bayes  --models M1 M3 M4 --trials 60
    python scripts/05_run_search.py --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/05_run_search.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("run_search")

#: T55.2-T55.5: SVM-RBF, Random Forest, Gradient Boosting, Logistic Regression
#: and the external baseline. M2 is an optional baseline; M6/M7 declare no
#: hyperparameters (their weights are searched by SO-05, not here).
DEFAULT_MODELS = ("M1", "M3", "M4", "M5", "M8")
COMPARISON_FILENAME = "search_method_comparison.csv"
OUTCOME_FILENAME = "nested_outcomes.csv"
CONVERGENCE_FILENAME = "convergence.csv"
CAPABILITY_FILENAME = "search_capability.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="05_run_search",
        description="Run SO-01 / SO-02 hyperparameter search on the loaded fold map.",
    )
    parser.add_argument("--method", default="random", choices=["random", "bayes"])
    parser.add_argument("--task", default="binary")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), metavar="ID")
    parser.add_argument("--trials", type=int, default=40, help="trial budget per search")
    parser.add_argument(
        "--model-trials",
        nargs="+",
        default=[],
        metavar="ID=N",
        help=(
            "per-model trial budget, e.g. M5=15. Trial cost spans two orders of "
            "magnitude across this stack; what must stay equal is the budget a "
            "model gets under SO-01 and under SO-02, not the budget across models"
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="wall-clock ceiling per search; stops gracefully between trials",
    )
    parser.add_argument(
        "--repeats", nargs="+", type=int, default=[0], metavar="R",
        help="which repeats of the outer map to search (default: repeat 0 only)",
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None, metavar="F",
        help="which outer folds within those repeats (default: all)",
    )
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="6 trials on one outer fold -- proves the path, produces no result",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="emit the SO-01 vs SO-02 equal-budget comparison from what is on disk (T56.5)",
    )
    return parser.parse_args(argv)


def _model_trials(pairs: list[str]) -> dict[str, int]:
    """``["M5=15"]`` -> ``{"M5": 15}``. Malformed input fails loudly, never silently."""
    parsed: dict[str, int] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError("--model-trials expects ID=N, got " + repr(pair))
        model_id, _, count = pair.partition("=")
        parsed[model_id.strip()] = int(count)
    return parsed


def _capability_note(capability: object) -> str:
    from src.optimization.bayesian import ACQUISITION

    info = capability.as_dict()  # type: ignore[attr-defined]
    if info["available"]:
        return (
            "scikit-optimize "
            + str(info["version"])
            + " imports and completes an ask/tell round trip on this machine; "
            + "SO-02 runs with a Gaussian-process surrogate and the "
            + ACQUISITION
            + " acquisition function."
        )
    return "SO-02 SKIPPED -- " + str(info["reason"])


def _comparison(out_dir: str | None) -> object:
    """SO-01 against SO-02, matched on (model, outer fold) -- T56.5.

    Matched pairs, not two independent averages: the two methods ran the same
    models over the same outer folds against the same inner splits, so the
    honest comparison is a per-pair difference. Averaging each method's scores
    separately would let a model that only one method searched move the result.
    """
    import pandas as pd

    from src.optimization.base import search_dir

    frames = {}
    for exp in ("SO-01", "SO-02"):
        path = search_dir(exp, out_dir) / OUTCOME_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                "cannot compare: " + str(path) + " does not exist; run that method first"
            )
        frames[exp] = pd.read_csv(path)

    keys = ["model_id", "task", "outer_fold"]
    merged = frames["SO-01"].merge(
        frames["SO-02"], on=keys, suffixes=("_random", "_bayes"), validate="1:1"
    )
    if merged.empty:
        raise ValueError("SO-01 and SO-02 share no (model, outer fold) pair to compare")

    merged["inner_delta_bayes_minus_random"] = (
        merged["inner_best_score_bayes"] - merged["inner_best_score_random"]
    ).round(6)
    merged["outer_delta_bayes_minus_random"] = (
        merged["outer_score_bayes"] - merged["outer_score_random"]
    ).round(6)
    columns = [
        *keys,
        "inner_best_score_random",
        "inner_best_score_bayes",
        "inner_delta_bayes_minus_random",
        "outer_score_random",
        "outer_score_bayes",
        "outer_delta_bayes_minus_random",
    ]
    return merged[columns].sort_values(["model_id", "outer_fold"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.models import smoke as sm
    from src.optimization import base as ob
    from src.optimization import driver as od
    from src.utils.config import load_config
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run
    from src.utils.seed import set_global_seed

    args = parse_args(argv)
    set_global_seed()

    if args.compare:
        frame = _comparison(args.out_dir)
        root = (
            Path(args.out_dir)
            if args.out_dir
            else Path(load_config("paths").require("outputs.search_optimization"))
        )
        path = save_csv(frame, root / COMPARISON_FILENAME)
        log.info("wrote %s (%d matched pairs)", path, len(frame))
        register_evidence(
            "SO-COMPARE",
            path,
            metric_or_asset="SO-01 vs SO-02 at equal budget, matched by model and outer fold",
            dataset="D1",
            command="python scripts/05_run_search.py --compare",
        )
        print(frame.to_string(index=False))
        return 0

    trials = 6 if args.smoke else args.trials
    folds = [0] if args.smoke else args.folds
    exp = od.METHODS[args.method]
    overrides = {} if args.smoke else _model_trials(args.model_trials)
    budgets = {
        model_id: ob.Budget(
            max_trials=int(overrides.get(model_id, trials)), max_seconds=args.max_seconds
        )
        for model_id in args.models
    }
    command = "python scripts/05_run_search.py --method " + args.method

    run = start_run("search_" + args.method)
    run.set("exp", exp)
    run.set("task", args.task)
    run.set("models", list(args.models))
    run.set("budget", {key: value.as_dict() for key, value in budgets.items()})
    run.set("repeats", list(args.repeats))
    run.set("inner_splits", args.inner_splits)
    run.set("smoke", bool(args.smoke))

    capability = None
    if args.method == "bayes":
        from src.optimization.bayesian import skopt_available

        capability = skopt_available()
        run.set("skopt", capability.as_dict())
        if not capability.available:
            # T56.6 -- a missing method is recorded with its technical reason,
            # never silently omitted.
            from src.utils.io import ensure_dir

            directory = Path(ensure_dir(ob.search_dir(exp, args.out_dir)))
            save_json(capability.as_dict(), directory / CAPABILITY_FILENAME)
            report = Path(load_config("paths").require("outputs.missing_outputs_report"))
            with report.open("a", encoding="utf-8") as handle:
                handle.write("\nSO-02 (Bayesian optimization) -- NOT PRODUCED\n")
                handle.write("  " + capability.reason + "\n")
            log.error("SO-02 unavailable: %s", capability.reason)
            run.finish(status="skipped")
            return 1

    data = sm.load_task_data(args.task)
    run.set("n_records", data.n_records)
    run.set("n_features", data.n_features)
    outer = od.outer_folds_for(
        args.task, data, repeats=args.repeats, folds=folds
    )
    log.info(
        "%s %s: %d model(s) x %d outer fold(s), %d trials each, %d inner splits",
        exp,
        args.method,
        len(args.models),
        len(outer),
        trials,
        args.inner_splits,
    )

    result = od.run_search(
        args.method,
        tuple(args.models),
        data,
        outer,
        budget=budgets,
        n_inner_splits=args.inner_splits,
    )

    directory = ob.search_dir(exp, args.out_dir)
    trials_path = ob.write_trials(result.results, exp, out_dir=args.out_dir)
    best_path = ob.write_best_params(
        result.results,
        exp,
        out_dir=args.out_dir,
        extra={
            "method": args.method,
            "task": args.task,
            "objective": result.objective,
            "budget": {key: value.as_dict() for key, value in budgets.items()},
            "n_inner_splits": args.inner_splits,
            "outer_folds": [fold.label for fold in outer],
            "smoke": bool(args.smoke),
            "total_seconds": round(result.seconds, 2),
            "skopt": None if capability is None else capability.as_dict(),
        },
    )
    fold_path = ob.write_inner_folds(
        dict(result.splits_by_fold), exp, data.record_uids, out_dir=args.out_dir
    )
    outcome_path = save_csv(result.outcome_frame(), directory / OUTCOME_FILENAME)
    convergence_path = save_csv(result.convergence_frame(), directory / CONVERGENCE_FILENAME)
    if capability is not None:
        save_json(
            {**capability.as_dict(), "note": _capability_note(capability)},
            directory / CAPABILITY_FILENAME,
        )

    for path in (trials_path, best_path, fold_path, outcome_path, convergence_path):
        run.record_artifact(path)
    run.record_timing("search_total", result.seconds)
    register_evidence(
        exp,
        trials_path,
        metric_or_asset=exp + " trial history (" + args.method + " search)",
        dataset="D1" if args.task == "binary" else args.task,
        source_data="outputs/03_features/all_features_matrix.parquet",
        command=command,
    )
    register_evidence(
        exp + "-BEST",
        best_path,
        metric_or_asset=exp + " best hyperparameters per model and outer fold",
        dataset="D1" if args.task == "binary" else args.task,
        command=command,
    )

    frame = pd.read_csv(trials_path)
    per_model = (
        frame.groupby("model_id")
        .agg(
            trials=("trial", "count"),
            failed=("status", lambda column: int((column != "ok").sum())),
            best=("score", "max"),
            median_trial_seconds=("seconds", "median"),
            total_seconds=("seconds", "sum"),
        )
        .round(4)
    )
    print(per_model.to_string())
    print()
    print(result.outcome_frame().to_string(index=False))
    log.info("%s finished in %.1f s", exp, result.seconds)
    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
