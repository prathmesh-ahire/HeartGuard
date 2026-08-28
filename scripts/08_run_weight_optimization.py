"""SO-05: ensemble weight optimization on out-of-fold probabilities (Phase 60).

Emits, into ``outputs/05_search_optimization/SO-05/``:

    weight_search.csv          every optimizer on every outer fold (T60.1, T60.2)
    equal_vs_optimized.csv     the per-fold delta against equal weights (T60.5)
    weight_stability.csv       per-member mean, sd and range across folds (T60.6)
    final_weights.json         the shipped weight vector and how it was chosen
    so05_settings.json         members, objective, folds, constraint, wall clock

Every weight is chosen on out-of-fold probabilities from the inner splits of one
outer fold's training rows (T60.4). The members are fitted once per fold and all
four methods score the same cached arrays, so the comparison is about the
optimizer and nothing else.

Usage
-----
    python scripts/08_run_weight_optimization.py --smoke
    python scripts/08_run_weight_optimization.py
    python scripts/08_run_weight_optimization.py --repeats 0 --members M3 M4 M5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/08_run_weight_optimization.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("run_weight_optimization")

EXP = "SO-05"
SEARCH_FILENAME = "weight_search.csv"
DELTA_FILENAME = "equal_vs_optimized.csv"
STABILITY_FILENAME = "weight_stability.csv"
FINAL_FILENAME = "final_weights.json"
SETTINGS_FILENAME = "so05_settings.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="08_run_weight_optimization",
        description="Run SO-05 ensemble weight optimization on the loaded fold map.",
    )
    parser.add_argument("--task", default="binary")
    parser.add_argument("--members", nargs="+", default=None, metavar="ID")
    parser.add_argument("--objective", default=None)
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument(
        "--n-standard-errors", type=float, default=None,
        help="the shrinkage margin for the grid; omit to take the raw argmax",
    )
    parser.add_argument(
        "--repeats", nargs="+", type=int, default=None, metavar="R",
        help="which repeats of the outer map (default: all, for the stability check)",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=None, metavar="F")
    parser.add_argument("--inner-splits", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="one fold, cheap members")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def _output_dir(out_dir: str | None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    root = (
        Path(out_dir)
        if out_dir
        else Path(load_config("paths").require("outputs.search_optimization"))
    )
    return Path(ensure_dir(root / EXP))


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import pandas as pd

    from src.models import smoke as sm
    from src.optimization import driver as od
    from src.optimization import weights as wt
    from src.utils.config import load_config
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run
    from src.utils.seed import GLOBAL_SEED, set_global_seed

    args = parse_args(argv)
    set_global_seed()
    seed = GLOBAL_SEED

    config = load_config("models")
    m7 = dict(config.get("models.M7") or {})
    search = dict(m7.get("weight_search") or {})

    members = tuple(args.members or m7.get("members", ["M3", "M4", "M5"]))
    objective = str(args.objective or search.get("objective", "balanced_accuracy"))
    resolution = float(args.resolution or search.get("resolution", 0.05))
    inner_splits = int(args.inner_splits or m7.get("defaults", {}).get("inner_cv", 3))
    guard = args.n_standard_errors
    if guard is None and str(search.get("selection_rule", "")) == "one_standard_error":
        guard = float(search.get("n_standard_errors", 1.0))

    if args.smoke:
        members, inner_splits = ("M1", "M3"), 3
        repeats, folds = [0], [0]
    else:
        repeats, folds = args.repeats, args.folds

    command = "python scripts/08_run_weight_optimization.py"
    run = start_run("weight_optimization")
    run.set("exp", EXP)
    run.set("task", args.task)
    run.set("members", list(members))
    run.set("objective", objective)

    data = sm.load_task_data(args.task)
    outer = od.outer_folds_for(args.task, data, repeats=repeats, folds=folds)
    log.info(
        "%s on %s: members %s, objective %s, %d outer fold(s), grid resolution %.2f, "
        "shrinkage %s",
        EXP, args.task, list(members), objective, len(outer), resolution,
        "off" if guard is None else str(guard) + " SE",
    )

    started = time.perf_counter()
    results: list[wt.WeightResult] = []
    details: list[dict] = []
    for index, fold in enumerate(outer, start=1):
        fold_results, detail = wt.optimize_fold(
            data, fold, members,
            objective=objective,
            n_inner_splits=inner_splits,
            seed=seed,
            resolution=resolution,
            n_standard_errors=guard,
        )
        results.extend(fold_results)
        details.append({"outer_fold": str(fold.label), **detail})
        by_method = {item.method: item for item in fold_results}
        log.info(
            "  %s (%d/%d): equal %.5f | grid %.5f %s | slsqp %.5f moved %.4f | "
            "slsqp_logloss %.5f moved %.4f | %.0fs",
            fold.label, index, len(outer),
            by_method["equal"].score,
            by_method["grid"].score, np.round(by_method["grid"].weights, 3).tolist(),
            by_method["slsqp"].score, by_method["slsqp"].moved_from_start,
            by_method["slsqp_logloss"].score, by_method["slsqp_logloss"].moved_from_start,
            time.perf_counter() - started,
        )

    directory = _output_dir(args.out_dir)
    frame = pd.DataFrame([item.as_row() for item in results])
    search_path = save_csv(frame, directory / SEARCH_FILENAME)

    # --- T60.5: equal weights against every optimizer, per fold --------------
    baseline = frame[frame.method == "equal"].set_index("outer_fold")["score"]
    delta_rows = []
    for method in sorted(set(frame.method) - {"equal"}):
        block = frame[frame.method == method].set_index("outer_fold")
        for label in block.index:
            delta_rows.append(
                {
                    "outer_fold": label,
                    "method": method,
                    "equal_score": float(baseline[label]),
                    "optimized_score": float(block.loc[label, "score"]),
                    "delta": float(block.loc[label, "score"] - baseline[label]),
                    "sensitivity_delta": float(
                        block.loc[label, "sensitivity"]
                        - frame[(frame.method == "equal") & (frame.outer_fold == label)]
                        ["sensitivity"].iloc[0]
                    ),
                    "moved_from_equal": float(block.loc[label, "moved_from_start"]),
                }
            )
    delta = pd.DataFrame(delta_rows)
    delta_path = save_csv(delta, directory / DELTA_FILENAME)

    # --- T60.6: the stability check -----------------------------------------
    stability = wt.stability_frame(results)
    stability_path = save_csv(stability, directory / STABILITY_FILENAME)

    # --- the shipped vector --------------------------------------------------
    grid_results = [item for item in results if item.method == "grid"]
    matrix = np.asarray([item.weights for item in grid_results], dtype=float)
    mean_weights = wt.normalize_weights(matrix.mean(axis=0))
    final = {
        "experiment": EXP,
        "task": args.task,
        "members": list(members),
        "objective": objective,
        "selection_rule": "one_standard_error" if guard is not None else "argmax",
        "n_standard_errors": guard,
        "grid_resolution": resolution,
        "n_folds": len(grid_results),
        "mean_weights": [float(v) for v in mean_weights],
        "mean_weights_sum": float(mean_weights.sum()),
        "per_fold_weights": {
            item.outer_label: [float(v) for v in item.weights] for item in grid_results
        },
        "per_member_std": {
            member: float(np.std(matrix[:, position], ddof=1)) if len(grid_results) > 1 else 0.0
            for position, member in enumerate(members)
        },
        "folds_identical_to_equal": int(
            sum(
                1 for item in grid_results
                if np.allclose(item.weights, wt.equal_weights(len(members)))
            )
        ),
        "constraint": (
            "non-negative and summing to 1; enforced as SLSQP bounds plus a linear "
            "equality for the continuous methods, and by construction for the "
            "simplex lattice. See src/optimization/weights.normalize_weights."
        ),
        "fold_detail": details,
        "seconds": time.perf_counter() - started,
        "seed": int(seed),
    }
    final_path = save_json(final, directory / FINAL_FILENAME)
    settings_path = save_json(
        {
            "experiment": EXP,
            "methods": wt.METHODS,
            "members": list(members),
            "objective": objective,
            "inner_splits": inner_splits,
            "outer_folds": [str(fold.label) for fold in outer],
            "grid_resolution": resolution,
            "n_standard_errors": guard,
            "seed": int(seed),
        },
        directory / SETTINGS_FILENAME,
    )

    for path in (search_path, delta_path, stability_path, final_path, settings_path):
        run.record_artifact(path)
    run.record_timing("so05_total", final["seconds"])

    register_evidence(
        EXP, delta_path,
        metric_or_asset="SO-05 equal weights versus optimized, per outer fold",
        dataset="D1" if args.task == "binary" else args.task,
        source_data="outputs/03_features/all_features_matrix.parquet",
        command=command,
    )
    register_evidence(
        EXP + "-STABILITY", stability_path,
        metric_or_asset="SO-05 per-member weight mean, standard deviation and range across folds",
        dataset="D1" if args.task == "binary" else args.task,
        command=command,
    )
    register_evidence(
        EXP + "-FINAL", final_path,
        metric_or_asset="SO-05 final weight vector, its per-fold spread and the constraint",
        dataset="D1" if args.task == "binary" else args.task,
        command=command,
    )

    print(
        frame.groupby("method")
        .agg(
            score=("score", "mean"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            moved=("moved_from_start", "mean"),
            seconds=("seconds", "mean"),
        )
        .sort_values("score", ascending=False)
        .to_string()
    )
    print()
    print(stability.to_string(index=False))
    log.info(
        "%s finished in %.1f min; %d of %d folds chose exactly equal weights",
        EXP, final["seconds"] / 60.0, final["folds_identical_to_equal"], len(grid_results),
    )
    run.finish(status="ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
