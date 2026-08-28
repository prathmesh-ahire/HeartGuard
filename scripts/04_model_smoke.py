"""Baseline model smoke runs and the SVM cost benchmark (Phases 46-47).

Emits, into ``outputs/04_models/``:

    baseline_smoke_metrics.csv        M1-M5, M8 on D1 fold 0 (T46.6, T47.6, T48.6, T49.6)
    model_search_spaces.csv           every declared dimension, per model (T46.2/4, T47.3,
                                      T48.2, T49.2, T49.4)
    svm_fit_time_benchmark.csv        fit cost vs n and cache_size (T47.4)
    svm_calibration_benchmark.csv     sigmoid/isotonic x ensemble x split (T47.5)
    feature_importance.csv            M4 impurity + permutation (T48.3, T48.4)
    feature_importance_by_family.csv  the same, rolled up to the six families
    tree_calibration_assessment.csv   M4/M5/M8 raw vs calibrated (T48.5, T49.5)
    gradient_boosting_choice.csv      classic vs histogram boosting (T49.1)
    external_model_capability.csv     whether M8 is usable, and why not (T49.3)

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

DEFAULT_MODELS = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")


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
        "--permutation-repeats",
        type=int,
        default=10,
        metavar="N",
        help="shuffles per feature for permutation importance (T48.4)",
    )
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


def _threshold_objectives(result: object, data: object, sv: object) -> object:
    """Score every threshold objective on the SAME in-fold probabilities.

    One fit, several objectives. The ensemble keeps the out-of-fold fused
    probabilities it chose its threshold on, so a different objective can be
    evaluated against exactly the rows the shipped one saw -- no refitting, and
    no risk of the comparison being run on different data from the decision.
    """
    import numpy as np
    import pandas as pd

    from src.evaluation import metrics as mt

    ensemble = result.pipeline.named_steps["estimator"]  # type: ignore[attr-defined]
    if getattr(ensemble, "oof_fused_", None) is None:
        return None

    fold = data.record_uids  # type: ignore[attr-defined]
    del fold
    oof = ensemble.oof_fused_[:, 1]
    oof_targets = ensemble.oof_targets_

    from src.models import smoke as sm

    test_fold = sm._fold_zero(data.task, data)  # type: ignore[attr-defined]
    test_index = np.asarray(test_fold.test_index, dtype=int)
    test_proba = result.pipeline.predict_proba(data.X[test_index])[:, 1]  # type: ignore[attr-defined]
    y_test = data.y[test_index]  # type: ignore[attr-defined]

    rows = []
    for name in sorted(sv.OBJECTIVES):  # type: ignore[attr-defined]
        choice = sv.select_threshold(oof_targets, oof, objective=name)  # type: ignore[attr-defined]
        predicted = (test_proba >= choice.threshold).astype(int)
        scored = mt.binary_metrics(y_test, predicted, labels=(0, 1), positive_label=1)
        rows.append(
            {
                "model_id": result.model_id,  # type: ignore[attr-defined]
                "objective": name,
                "threshold": round(choice.threshold, 4),
                "infold_sensitivity": round(choice.sensitivity, 4),
                "infold_specificity": round(choice.specificity, 4),
                "infold_balanced_accuracy": round(choice.balanced_accuracy, 4),
                "test_sensitivity": round(scored["sensitivity"], 4),
                "test_specificity": round(scored["specificity"], 4),
                "test_balanced_accuracy": round(scored["balanced_accuracy"], 4),
                "shipped": name == ensemble.objective,
            }
        )

    predicted = (test_proba >= 0.5).astype(int)
    scored = mt.binary_metrics(y_test, predicted, labels=(0, 1), positive_label=1)
    rows.append(
        {
            "model_id": result.model_id,  # type: ignore[attr-defined]
            "objective": "(fixed 0.5, no tuning)",
            "threshold": 0.5,
            "infold_sensitivity": None,
            "infold_specificity": None,
            "infold_balanced_accuracy": None,
            "test_sensitivity": round(scored["sensitivity"], 4),
            "test_specificity": round(scored["specificity"], 4),
            "test_balanced_accuracy": round(scored["balanced_accuracy"], 4),
            "shipped": False,
        }
    )
    return pd.DataFrame(rows)



def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.ensemble import soft_voting as sv
    from src.models import benchmark as bm
    from src.models import estimators as est
    from src.models import importance as imp
    from src.models import registry as reg
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
        #
        # Built from EVERY declared model, not from `--models`. This table is a
        # registry of what the config declares, not a record of what this run
        # fitted, and T07 (Phase 62) reads it. Building it from the subset meant
        # that re-running two models to fix one of them silently truncated it to
        # those two -- which is exactly what happened on 2026-08-28 when M6 and M7
        # were re-run alone and the table went from 28 rows to 0, because those
        # two declare no hyperparameters at all.
        from src.models import registry as model_registry

        described = []
        constraint_notes: list[str] = []
        for model_id in model_registry.model_ids():
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

        results = sm.run_smoke(
            tuple(args.models), task=args.task, data=data, keep_pipelines=True
        )
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

        # -- feature importance, from M4 (T48.3, T48.4) ----------------------
        forest = next((r for r in results if r.model_id == "M4"), None)
        if forest is not None and forest.pipeline is not None:
            fold = sm._fold_zero(args.task, data)
            measured = [
                imp.impurity_importance(
                    forest.pipeline, data.feature_names, model_id="M4"
                ),
                imp.permutation_importance(
                    forest.pipeline,
                    data.X,
                    data.y,
                    held_out_index=fold.test_index,
                    train_index=fold.train_index,
                    feature_names=data.feature_names,
                    model_id="M4",
                    n_repeats=args.permutation_repeats,
                    n_jobs=-1,
                ),
            ]
            importance_path = _write(
                imp.importance_frame(measured), directory, imp.IMPORTANCE_FILENAME
            )
            family_path = _write(
                pd.concat([imp.family_importance(m) for m in measured], ignore_index=True),
                directory,
                "feature_importance_by_family.csv",
            )
            run.record_artifact(importance_path)
            run.record_artifact(family_path)
            for measure in measured:
                log.info(
                    "M4 %s importance, top 5: %s",
                    measure.kind,
                    ", ".join(name for name, _ in measure.top(5)),
                )
            register_evidence(
                "MD-IMPORTANCE",
                importance_path,
                metric_or_asset=(
                    "M4 impurity (train) and permutation (held-out fold) importance"
                ),
                dataset="D1",
                model="M4",
                command=command,
            )

        # -- do the tree models need calibrating? (T48.5, T49.5) -------------
        tree_ids = tuple(m for m in args.models if m in ("M4", "M5", "M8"))
        if not args.skip_benchmark and tree_ids:
            fold = sm._fold_zero(args.task, data)
            assessment = bm.tree_calibration_assessment(data, fold, model_ids=tree_ids)
            if not assessment.empty:
                assessment_path = _write(
                    assessment, directory, bm.TREE_CALIBRATION_FILENAME
                )
                run.record_artifact(assessment_path)
                register_evidence(
                    "MD-TREECAL",
                    assessment_path,
                    metric_or_asset="M4/M5/M8 raw vs calibrated: Brier, ECE, verdict",
                    dataset="D1",
                    command=command,
                )

            # -- classic gradient boosting or histogram? (T49.1) -------------
            if "M5" in args.models:
                choice = bm.gradient_boosting_choice(data, fold)
                choice_path = _write(choice, directory, bm.GB_CHOICE_FILENAME)
                run.record_artifact(choice_path)
                register_evidence(
                    "MD-GBCHOICE",
                    choice_path,
                    metric_or_asset=(
                        "M5 classic vs histogram gradient boosting, weighted and not"
                    ),
                    dataset="D1",
                    command=command,
                )

        # -- the model registry and persistence (Phase 51) -------------------
        registry_path = _write(
            reg.registry_frame(), directory, "model_registry.csv"
        )
        run.record_artifact(registry_path)
        register_evidence(
            "MD-REGISTRY",
            registry_path,
            metric_or_asset="model inventory M1-M9: availability, members, search size",
            command=command,
        )

        persisted = []
        for result in results:
            if result.pipeline is None:
                continue
            saved = reg.save_model(
                result.pipeline,
                model_id=result.model_id,
                task=args.task,
                feature_names=data.feature_names,
                fit_seconds=result.fit_seconds,
                X_sample=data.X[sm._fold_zero(args.task, data).test_index],
                fold=result.fold_label,
                extra={"smoke_metrics": result.metrics},
            )
            persisted.append(
                {
                    "model_id": saved.model_id,
                    "task": saved.task,
                    "path": str(saved.path.relative_to(Path.cwd()))
                    if saved.path.is_relative_to(Path.cwd())
                    else str(saved.path),
                    "model_mb": saved.manifest["model_mb"],
                    "fit_seconds": saved.manifest["fit_seconds"],
                    "inference_seconds_per_record": saved.manifest.get(
                        "inference_seconds_per_record"
                    ),
                    "inference_seconds_per_record_batched": saved.manifest.get(
                        "inference_seconds_per_record_batched"
                    ),
                    "n_features": saved.manifest["n_features"],
                }
            )
        if persisted:
            complexity_path = _write(
                pd.DataFrame(persisted), directory, "model_complexity.csv"
            )
            run.record_artifact(complexity_path)
            register_evidence(
                "MD-COMPLEXITY",
                complexity_path,
                metric_or_asset="per-model size on disk, training time, inference time",
                dataset="D1",
                command=command,
            )

        # -- the threshold objective comparison (T50.4 evidence) -------------
        ensemble = next(
            (r for r in results if r.model_id in ("M6", "M7") and r.pipeline is not None),
            None,
        )
        if ensemble is not None:
            comparison = _threshold_objectives(ensemble, data, sv)
            if comparison is not None:
                objective_path = _write(
                    comparison, directory, "threshold_objective_comparison.csv"
                )
                run.record_artifact(objective_path)
                register_evidence(
                    "MD-THRESHOLD",
                    objective_path,
                    metric_or_asset=(
                        "in-fold threshold objectives vs the fixed 0.5 rule (T50.4)"
                    ),
                    dataset="D1",
                    command=command,
                )

        # -- M8's capability check, recorded either way (T49.3) --------------
        capability = est.m8_capability()
        capability_path = _write(
            pd.DataFrame([capability.as_row()]), directory, "external_model_capability.csv"
        )
        run.record_artifact(capability_path)
        run.set("m8_available", capability.available)
        run.set("m8_backend", capability.backend)
        if not capability.available:
            log.warning("M8 unavailable: %s", capability.reason)
    except BaseException:
        run.finish("failed")
        raise

    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
