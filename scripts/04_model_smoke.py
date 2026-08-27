"""Baseline model smoke runs and the SVM cost benchmark (Phases 46-47).

Emits, into ``outputs/04_models/``:

    baseline_smoke_metrics.csv     M1/M2/M3 on D1 fold 0 (T46.6, T47.6)
    model_search_spaces.csv        every declared dimension, per model (T46.2/4, T47.3)
    svm_fit_time_benchmark.csv     fit cost vs n and cache_size (T47.4)
    svm_calibration_benchmark.csv  sigmoid/isotonic x ensemble x split (T47.5)

These are **diagnostic** numbers, not results. One fold of one repeat says
whether a model runs, produces usable probabilities and costs what was expected;
it does not say how well the model works. The reported metrics come from the
full repeated CV in Part VII, and nothing here may be quoted as a result.

Usage
-----
    python scripts/04_model_smoke.py                  # everything
    python scripts/04_model_smoke.py --models M1 M2   # smoke only, chosen models
    python scripts/04_model_smoke.py --skip-benchmark # no SVM timing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/04_model_smoke.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("model_smoke")

DEFAULT_MODELS = ("M1", "M2", "M3")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="04_model_smoke",
        description="Smoke-run the baseline models on D1 fold 0 and benchmark M3.",
    )
    parser.add_argument("--task", default="binary", help="which task's fold 0 to use")
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS), metavar="ID"
    )
    parser.add_argument(
        "--out-dir", default=None, help="write elsewhere than outputs/04_models"
    )
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="overwrite the smoke table instead of merging into it",
    )
    return parser.parse_args(argv)


def _write(frame: object, directory: Path, name: str) -> Path:
    from src.utils.io import save_csv

    path = save_csv(frame, directory / name)
    log.info("wrote %s (%d rows)", path, len(frame))  # type: ignore[arg-type]
    return path


def _budget_note(budget: dict[str, float], timings: object) -> str:
    """The T47.4 decision, written next to the numbers it was taken from."""
    import pandas as pd

    frame = pd.DataFrame(timings)
    largest = int(frame["n_train"].max())
    cache = frame[
        (frame["variant"] == "bare SVC")
        & (frame["n_train"] == largest)
        & (frame["hyperparameters"] == "default (C=1, gamma=scale)")
    ]
    spread = round(float(cache["fit_seconds"].max() - cache["fit_seconds"].min()), 4)
    jitter = round(
        float((cache["fit_seconds_max"] - cache["fit_seconds_min"]).max()), 4
    )
    repeats = int(cache["n_repeats"].max())
    kernel_mb = round(float(frame["kernel_matrix_mb"].max()), 1)

    lines = [
        "M3 (SVM-RBF) fit cost and search budget -- T47.4",
        "=" * 52,
        "",
        "Measured on the real D1 matrix (n=" + str(largest) + " x 138), CPU only.",
        "",
        "Measured numbers",
        "----------------",
    ]
    lines += [key + ": " + str(value) for key, value in budget.items()]
    lines += [
        "full-matrix kernel size (MB): " + str(kernel_mb),
        "cache_size spread over 200/500/1000 MB, medians of "
        + str(repeats)
        + " (s): "
        + str(spread),
        "widest within-setting jitter across those repeats (s): " + str(jitter),
        "",
        "Decision: cache_size stays at 500 MB, uncapped.",
        "  The kernel matrix at the full n is "
        + str(kernel_mb)
        + " MB -- below even the",
        "  200 MB setting -- so the cache never thrashes. Across "
        + str(repeats)
        + " repeats the three",
        "  settings' medians differ by " + str(spread) + " s while a single setting varies",
        "  by up to " + str(jitter) + " s on its own: the spread is jitter, not an effect.",
        "  Raising cache_size would reserve memory that is never touched.",
        "",
        "Decision: no subsampling per fit.",
        "  The slowest corner of the declared space costs "
        + str(budget["worst_bare_fit_seconds"])
        + " s at full n.",
        "  Degrading every fit would change the model being searched in order to",
        "  fix a problem that is not per-fit.",
        "",
        "Open for Part VI: the search TOTAL is the cost, not the fit.",
        "  "
        + str(budget["search_hours_all_outer_folds"])
        + " h for 200 trials across all 25 outer folds.",
        "  The lever is the trial count and how many outer folds are searched",
        "  (one repeat = 5 outer folds cuts it fivefold), not the size of each",
        "  fit. Decided at Phase 54/62 against the rest of the search timings.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.models import benchmark as bm
    from src.models import smoke as sm
    from src.models import spaces
    from src.utils.config import load_config
    from src.utils.evidence import register_evidence
    from src.utils.io import ensure_dir
    from src.utils.run_manifest import start_run
    from src.utils.seed import set_global_seed

    args = parse_args(argv)
    set_global_seed()

    directory = Path(
        ensure_dir(
            args.out_dir
            if args.out_dir is not None
            else load_config("paths").require("outputs.models")
        )
    )
    command = "python scripts/04_model_smoke.py"
    run = start_run("model_smoke")
    run.set("task", args.task)
    run.set("models", list(args.models))

    try:
        # -- the declared search spaces (T46.2, T46.4, T47.3) ----------------
        described = []
        constraint_notes: list[str] = []
        for model_id in args.models:
            space = spaces.load_space(model_id)
            frame = spaces.describe_space(space)
            described.append(frame)
            constraint_notes.extend(
                model_id + " -- " + line for line in frame.attrs["constraints"]
            )
        space_table = pd.concat(described, ignore_index=True)
        space_path = _write(space_table, directory, "model_search_spaces.csv")
        if constraint_notes:
            (directory / "model_search_space_constraints.txt").write_text(
                "\n".join(constraint_notes) + "\n", encoding="utf-8"
            )
        run.record_artifact(space_path)

        # -- the smoke runs (T46.6, T47.6) -----------------------------------
        data = sm.load_task_data(args.task)
        run.set("n_records", data.n_records)
        run.set("n_features", data.n_features)
        log.info(
            "%s: %d records x %d features, classes %s",
            args.task,
            data.n_records,
            data.n_features,
            data.classes,
        )

        results = sm.run_smoke(tuple(args.models), task=args.task, data=data)
        smoke_path = sm.write_smoke_metrics(
            results, directory, append=not args.fresh
        )
        run.record_artifact(smoke_path)
        for result in results:
            run.record_timing(result.model_id + "_fit", result.fit_seconds)
        register_evidence(
            "MD-SMOKE",
            smoke_path,
            metric_or_asset="baseline smoke metrics, D1 fold 0 (diagnostic)",
            dataset="D1",
            source_data="outputs/03_features/all_features_matrix.parquet",
            command=command,
        )

        # -- the SVM benchmark (T47.4, T47.5) --------------------------------
        if not args.skip_benchmark and "M3" in args.models:
            fold = sm._fold_zero(args.task, data)

            timings = bm.fit_time_benchmark(data)
            timing_path = _write(timings, directory, bm.FIT_TIME_FILENAME)
            budget = bm.search_budget_estimate(timings)
            for key, value in budget.items():
                run.set("m3_" + key, value)
            log.info("M3 search budget estimate: %s", budget)
            (directory / "svm_search_budget.txt").write_text(
                _budget_note(budget, timings), encoding="utf-8"
            )

            calibration = bm.calibration_benchmark(data, fold)
            calibration_path = _write(calibration, directory, bm.CALIBRATION_FILENAME)

            run.record_artifact(timing_path)
            run.record_artifact(calibration_path)
            register_evidence(
                "MD-SVMTIME",
                timing_path,
                metric_or_asset="M3 fit time vs n and cache_size",
                dataset="D1",
                command=command,
            )
            register_evidence(
                "MD-SVMCAL",
                calibration_path,
                metric_or_asset="M3 calibration variants: Brier, ECE, well-formedness",
                dataset="D1",
                command=command,
            )
    except BaseException:
        run.finish("failed")
        raise

    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
