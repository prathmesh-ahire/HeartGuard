"""SO-04 feature selection: filter, embedded, RFECV, the k sweep and FE-12 (Phase 57).

Emits, into ``outputs/05_search_optimization/SO-04/``:

    feature_count_sweep.csv        every (ranker, k) on every inner fold (T57.4)
    feature_count_curve.csv        the same, averaged per (ranker, k) -- G22's source
    rfecv_selection.csv            RFECV's own stopping point per outer fold (T57.3)
    embedded_thresholds.csv        RF/GB importance-threshold subsets per fold (T57.2)
    per_fold_selection.csv         the columns each outer fold selected, for FE-12
    all_features_vs_selected.csv   138 against the subset, same folds and seed (T57.6)
    so04_settings.json             the configuration, the J weights and the wall clock

and, into ``outputs/03_features/``:

    selected_feature_subset.csv    FE-12 (T57.5)

Nothing that decides a feature ever sees a test row: rankers are fitted on inner
training blocks for the sweep and on outer training blocks for the per-fold
selection, and the outer test fold is scored exactly once, at the end.

Usage
-----
    python scripts/06_run_feature_selection.py --smoke
    python scripts/06_run_feature_selection.py
    python scripts/06_run_feature_selection.py --folds 0 1 2 3 4 --eval-model M4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/06_run_feature_selection.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("run_feature_selection")

EXP = "SO-04"
SUBSET_FILENAME = "selected_feature_subset.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="06_run_feature_selection",
        description="Run SO-04 feature selection on the loaded DA-07 fold map.",
    )
    parser.add_argument("--task", default="binary")
    parser.add_argument(
        "--eval-model", default=None, metavar="ID",
        help="the estimator every subset is scored with (default: from config)",
    )
    parser.add_argument(
        "--rankers", nargs="+", default=None, metavar="KIND",
        help="which rankers to sweep (default: from config)",
    )
    parser.add_argument(
        "--k-grid", nargs="+", type=int, default=None, metavar="K",
        help="feature counts to evaluate (default: from config)",
    )
    parser.add_argument("--repeats", nargs="+", type=int, default=None, metavar="R")
    parser.add_argument("--folds", nargs="+", type=int, default=None, metavar="F")
    parser.add_argument("--inner-splits", type=int, default=None)
    parser.add_argument(
        "--n-standard-errors", type=float, default=None, metavar="N",
        help=(
            "restrict J's choice to configurations within N standard errors of "
            "the best mean macro-F1, then take the lowest J among those (the "
            "Phase 50 one-standard-error rule). Default: from config"
        ),
    )
    parser.add_argument(
        "--from-sweep", action="store_true",
        help=(
            "reuse feature_count_sweep.csv instead of re-running the sweep. The "
            "sweep is ~40 minutes of fitting and the SELECTION RULE applied to it "
            "costs seconds, so a change of rule does not have to pay for the "
            "measurement again"
        ),
    )
    parser.add_argument(
        "--skip-rfecv", action="store_true",
        help="skip T57.3; the skip is written to missing_outputs_report.txt",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="one outer fold, one ranker, two feature counts, a cheap estimator",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def _settings_from(args: argparse.Namespace) -> dict:
    from src.feature_selection import sweep as fs

    settings = fs.load_settings()
    if args.smoke:
        settings = {
            **settings,
            "evaluation_model": "M1",
            "rankers": ["anova_f"],
            "k_grid": [20, 138],
            "inner_splits": 3,
        }
    if args.eval_model:
        settings["evaluation_model"] = args.eval_model
    if args.rankers:
        settings["rankers"] = list(args.rankers)
    if args.k_grid:
        settings["k_grid"] = list(args.k_grid)
    if args.inner_splits:
        settings["inner_splits"] = int(args.inner_splits)
    return settings


def _rfecv_table(data, outer, settings, seed):
    """T57.3 -- RFECV inside each outer fold's training rows."""
    import numpy as np
    import pandas as pd

    from src.feature_selection import ranking as fr

    options = dict(settings.get("rfecv") or {})
    rows = []
    columns_by_fold: dict[str, tuple[int, ...]] = {}
    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    groups = np.asarray(data.groups, dtype=object)
    for fold in outer:
        train = np.asarray(fold.train_index, dtype=int)
        started = time.perf_counter()
        columns, detail = fr.rfecv_select(
            features[train],
            targets[train],
            groups[train],
            step=int(options.get("step", 5)),
            min_features_to_select=int(options.get("min_features_to_select", 10)),
            inner_cv=int(options.get("inner_cv", 3)),
            seed=seed,
        )
        columns_by_fold[str(fold.label)] = tuple(int(c) for c in columns)
        rows.append(
            {
                "outer_fold": str(fold.label),
                "seconds": time.perf_counter() - started,
                **detail,
                "features": ";".join(
                    str(data.feature_names[int(c)]) for c in columns
                ),
            }
        )
        log.info("RFECV %s selected %d features", fold.label, len(columns))
    return pd.DataFrame(rows), columns_by_fold


def _threshold_table(data, outer, seed):
    """T57.2 -- embedded selection at an importance threshold, per outer fold."""
    import numpy as np
    import pandas as pd

    from src.feature_selection import ranking as fr

    features = np.asarray(data.X, dtype=float)
    targets = np.asarray(data.y)
    rows = []
    by_name: dict[str, dict[str, tuple[int, ...]]] = {}
    for name in sorted(fr.THRESHOLD_SELECTORS):
        by_name[name] = {}
        for fold in outer:
            train = np.asarray(fold.train_index, dtype=int)
            started = time.perf_counter()
            columns, threshold = fr.threshold_select(
                name, features[train], targets[train], seed=seed
            )
            by_name[name][str(fold.label)] = tuple(int(c) for c in columns)
            rows.append(
                {
                    "selector": name,
                    "outer_fold": str(fold.label),
                    "threshold": float(threshold),
                    "n_selected": int(columns.size),
                    "seconds": time.perf_counter() - started,
                    "features": ";".join(
                        str(data.feature_names[int(c)]) for c in columns
                    ),
                }
            )
    return pd.DataFrame(rows), by_name


def _columns_from_table(frame, feature_names, *, key: str):
    """Read a selector table's `features` column back into column positions."""
    index_of = {str(name): position for position, name in enumerate(feature_names)}
    out: dict[str, dict[str, tuple[int, ...]]] = {}
    for _, row in frame.iterrows():
        out.setdefault(str(row[key]), {})[str(row["outer_fold"])] = tuple(
            int(index_of[name]) for name in str(row["features"]).split(";")
        )
    return out


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.feature_extraction import registry
    from src.feature_selection import sweep as fs
    from src.models import smoke as sm
    from src.optimization import driver as od
    from src.optimization import multi_objective as mo
    from src.utils.config import load_config
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run
    from src.utils.seed import GLOBAL_SEED, set_global_seed

    args = parse_args(argv)
    set_global_seed()
    seed = GLOBAL_SEED

    settings = _settings_from(args)
    repeats = args.repeats if args.repeats is not None else list(settings.get("repeats", [0]))
    folds = [0] if args.smoke else args.folds
    command = "python scripts/06_run_feature_selection.py"

    run = start_run("feature_selection")
    run.set("exp", EXP)
    run.set("task", args.task)
    run.set("settings", settings)

    data = sm.load_task_data(args.task)
    outer = od.outer_folds_for(args.task, data, repeats=repeats, folds=folds)
    log.info(
        "SO-04 on %s: %d outer fold(s) %s, %d rankers, k in %s, evaluated with %s",
        args.task, len(outer), [f.label for f in outer],
        len(settings["rankers"]), settings["k_grid"], settings["evaluation_model"],
    )

    weights = mo.load_weights()
    cost_model = mo.load_cost_model()
    started = time.perf_counter()

    # --- T57.4: the sweep ---------------------------------------------------
    done = [0]
    total = (
        len(outer) * int(settings["inner_splits"])
        * len(settings["rankers"]) * len(set(settings["k_grid"]))
    )

    def progress(score) -> None:
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == total:
            log.info("  %d/%d evaluations (%.0fs elapsed)", done[0], total,
                     time.perf_counter() - started)

    directory = fs.output_dir(args.out_dir)
    if args.from_sweep:
        sweep_path = directory / fs.CURVE_FILENAME
        if not sweep_path.is_file():
            raise SystemExit(
                "--from-sweep needs " + str(sweep_path) + ", which does not exist; "
                "run without it once to produce the sweep"
            )
        stored = pd.read_csv(sweep_path)
        # Recomputed, not trusted: the stored `j` was produced under whatever
        # weighting was configured when the sweep ran, and the point of reusing
        # the sweep is usually that the weighting has changed since.
        result = fs.recompute_j(stored, weights=weights, cost_model=cost_model)
        moved = float((result["j"] - stored["j"]).abs().max())
        log.info(
            "reusing the sweep on disk: %d evaluations from %s; J recomputed under the "
            "current weighting (max change %.6f)",
            len(result), sweep_path, moved,
        )
    else:
        result = fs.sweep_feature_counts(
            data, outer, settings=settings, seed=seed,
            weights=weights, cost_model=cost_model, progress=progress,
        )
        sweep_path = save_csv(result.frame(), directory / fs.CURVE_FILENAME)
    curve = fs.curve_frame(result)
    curve_path = save_csv(curve, directory / "feature_count_curve.csv")

    # --- T57.5: the configuration, and what pure performance would have picked
    guard = (
        args.n_standard_errors
        if args.n_standard_errors is not None
        else settings.get("n_standard_errors")
    )
    ranker, k, detail = fs.choose_configuration(
        result, n_standard_errors=None if guard is None else float(guard)
    )
    log.info(
        "%s picks %s at k=%d (mean J %.4f, macro-F1 %.4f); best macro-F1 is %s at k=%d "
        "(%.4f); unguarded J would take %s at k=%d (macro-F1 %.4f)",
        detail["selection_rule"], ranker, k, detail["mean_j"], detail["mean_macro_f1"],
        detail["best_performance_ranker"], detail["best_performance_k"],
        detail["best_performance_macro_f1"],
        detail["unguarded_j_ranker"], detail["unguarded_j_k"],
        detail["unguarded_j_macro_f1"],
    )

    per_fold = fs.select_per_fold(data, outer, ranker, k, seed=seed)
    columns, frequency = fs.consensus_subset(per_fold, k, data.feature_names)
    names = [str(data.feature_names[int(c)]) for c in columns]

    # --- T57.2 / T57.3: the two selectors that choose their own size ---------
    #
    # Neither depends on the k the sweep chose -- both decide their own size from
    # the data -- so under --from-sweep they are read back rather than refitted.
    # That is what keeps a change of SELECTION RULE to a few minutes: the parts
    # of the run that the rule cannot affect are not paid for twice.
    extra: dict[str, dict[str, tuple[int, ...]]] = {}
    threshold_path = directory / "embedded_thresholds.csv"
    reused_selectors = args.from_sweep and threshold_path.is_file()
    if reused_selectors:
        threshold_columns = _columns_from_table(
            pd.read_csv(threshold_path), data.feature_names, key="selector"
        )
        log.info("reusing %s", threshold_path)
    else:
        threshold_frame, threshold_columns = _threshold_table(data, outer, seed)
        threshold_path = save_csv(threshold_frame, directory / "embedded_thresholds.csv")
    extra.update(threshold_columns)

    rfecv_path: Path | None = directory / "rfecv_selection.csv"
    if args.skip_rfecv:
        log.warning("RFECV skipped by --skip-rfecv; recording the skip")
        rfecv_path = None
    elif args.from_sweep and rfecv_path.is_file():
        frame = pd.read_csv(rfecv_path)
        extra["rfecv"] = {
            str(row["outer_fold"]): tuple(
                int(index_of[name]) for name in str(row["features"]).split(";")
            )
            for _, row in frame.iterrows()
            for index_of in [{n: i for i, n in enumerate(data.feature_names)}]
        }
        log.info("reusing %s", rfecv_path)
    else:
        rfecv_frame, rfecv_columns = _rfecv_table(data, outer, settings, seed)
        rfecv_path = save_csv(rfecv_frame, directory / "rfecv_selection.csv")
        extra["rfecv"] = rfecv_columns

    # --- T57.6: all 138 against the subset, same folds, same seed ------------
    comparison = fs.compare_all_versus_selected(
        data, outer, per_fold,
        model_id=str(settings["evaluation_model"]),
        weights=weights, cost_model=cost_model,
        extra_configurations=extra,
    )
    comparison_path = save_csv(comparison, directory / fs.COMPARISON_FILENAME)

    per_fold_frame = pd.DataFrame(
        [
            {
                "outer_fold": label,
                "ranker": ranker,
                "k": int(k),
                "rank": position,
                "column": int(column),
                "feature": str(data.feature_names[int(column)]),
            }
            for label, cols in sorted(per_fold.items())
            for position, column in enumerate(cols)
        ]
    )
    per_fold_path = save_csv(per_fold_frame, directory / fs.PER_FOLD_FILENAME)

    # --- FE-12 --------------------------------------------------------------
    subset_frame = pd.DataFrame(
        {
            "rank": range(1, len(names) + 1),
            "feature": names,
            "column": [int(c) for c in columns],
            "family": [mo.family_of(name) for name in names],
            "selected_in_folds": [int(frequency[name]) for name in names],
            "n_folds": len(per_fold),
            "ranker": ranker,
        }
    )
    features_root = Path(load_config("paths").require("outputs.features"))
    subset_path = save_csv(subset_frame, features_root / SUBSET_FILENAME)

    elapsed = time.perf_counter() - started
    families = mo.families_needed(names)
    settings_payload = {
        "experiment": EXP,
        "task": args.task,
        "settings": (
            settings if args.from_sweep
            else {**result.settings, "reused_sweep": False}
        ),
        "reused_sweep": bool(args.from_sweep),
        "outer_folds": [str(fold.label) for fold in outer],
        "j_weights": weights.as_dict(),
        "cost_model": cost_model.as_dict(),
        "chosen": {
            "ranker": ranker,
            "k": int(k),
            "n_selected": len(names),
            "families_needed": list(families),
            "normalized_inference_time": cost_model.normalized(families),
            **detail,
        },
        "selection_frequency": {name: int(frequency[name]) for name in names},
        "seconds": float(elapsed),
        "seed": int(seed),
        "n_features_total": len(registry.feature_names()),
    }
    settings_path = save_json(settings_payload, directory / fs.SETTINGS_FILENAME)

    for path in (sweep_path, curve_path, threshold_path, comparison_path,
                 per_fold_path, subset_path, settings_path):
        run.record_artifact(path)
    if rfecv_path is not None:
        run.record_artifact(rfecv_path)
    run.record_timing("so04_total", elapsed)

    register_evidence(
        "FE-12", subset_path,
        metric_or_asset="SO-04 selected feature subset (" + ranker + ", k=" + str(k) + ")",
        dataset="D1" if args.task == "binary" else args.task,
        source_data="outputs/03_features/all_features_matrix.parquet",
        command=command,
    )
    register_evidence(
        EXP, curve_path,
        metric_or_asset="SO-04 performance versus feature count, per ranker",
        dataset="D1" if args.task == "binary" else args.task,
        command=command,
    )
    register_evidence(
        EXP + "-COMPARE", comparison_path,
        metric_or_asset="SO-04 all 138 features versus the selected subset, same folds and seed",
        dataset="D1" if args.task == "binary" else args.task,
        command=command,
    )
    if rfecv_path is not None:
        register_evidence(
            EXP + "-RFECV", rfecv_path,
            metric_or_asset="SO-04 RFECV stopping point per outer fold",
            dataset="D1" if args.task == "binary" else args.task,
            command=command,
        )

    summary = (
        comparison.groupby("configuration")
        .agg(
            n_features=("n_features", "mean"),
            macro_f1=("macro_f1", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            j=("j", "mean"),
        )
        .sort_values("macro_f1", ascending=False)
    )
    print(summary.to_string())
    log.info("SO-04 finished in %.1f min", elapsed / 60.0)
    run.finish(status="ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
