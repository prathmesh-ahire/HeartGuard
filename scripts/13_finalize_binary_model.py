"""T09, the selection rule, and the persisted binary model (T65.4 - T65.6).

Reads what EXP-A1 and EXP-A2 already wrote and produces three things:

``T09_ensemble_weight_comparison.csv``
    Equal-weight (M6) against optimized-weight (M7) soft voting, **paired
    fold-for-fold** over the same 25 folds. Paired, not two independent means:
    the two ensembles differ only in their weights and share every fold, member
    and threshold rule, so the per-fold difference is the whole signal and
    averaging each separately would throw the pairing away.

``final_model_selection.csv`` / ``.json``
    The ranking under the documented rule -- **sensitivity, then balanced
    accuracy**, never raw accuracy (research rule 6, T65.4) -- together with
    which model raw accuracy would have chosen, so a disagreement between the
    two is visible rather than implied.

``models_saved/binary/final/``
    The selected model, refitted on every labelled D1 record, with its feature
    list and manifest (T65.6).

**Which hyperparameters the deployed model uses.** The nested run produces 25
points, one per outer fold; they estimate *performance*, they are not a single
model. The deployed model takes the T07 selected point -- the project's declared
"final selected value" table -- which is the standard split between a nested
estimate and a deployed fit. Recorded in the manifest so the two are never
confused.

Usage
-----
    python scripts/13_finalize_binary_model.py
    python scripts/13_finalize_binary_model.py --baseline EXP-A1 --optimized EXP-A2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/13_finalize_binary_model.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("finalize_binary")

T09_FILENAME = "T09_ensemble_weight_comparison.csv"
SELECTION_CSV = "final_model_selection.csv"
SELECTION_JSON = "final_model_selection.json"
PAIRED_METRICS = (
    "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc", "accuracy",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="13_finalize_binary_model",
        description="Emit T09, apply the selection rule, and persist the final binary model.",
    )
    parser.add_argument("--baseline", default="EXP-A1")
    parser.add_argument("--optimized", default="EXP-A2")
    parser.add_argument(
        "--skip-persist", action="store_true",
        help="emit the tables but do not refit or write models_saved/binary/final/",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def build_t09(baseline: str, optimized: str, out_dir: str | None = None) -> object:
    """M6 against M7, fold by fold, in both the baseline and the optimized run."""
    import pandas as pd

    from src.evaluation.experiment import load_per_fold_metrics

    rows = []
    for exp_id in (baseline, optimized):
        try:
            frame = load_per_fold_metrics(exp_id, out_dir=out_dir)
        except FileNotFoundError as error:
            log.warning("%s: %s", exp_id, error)
            continue
        available = set(frame["model_id"])
        if not {"M6", "M7"} <= available:
            # EXP-A1 declares no M7 -- equal weights is the baseline, and there
            # is nothing to pair it against there. Recorded, not silently
            # dropped, so the shape of the table matches the shape of the runs.
            log.info(
                "%s has %s -- no M6/M7 pair to compare",
                exp_id,
                ", ".join(sorted(available & {"M6", "M7"})) or "neither",
            )
            continue
        equal = frame[frame["model_id"] == "M6"].set_index("fold_label")
        optimised = frame[frame["model_id"] == "M7"].set_index("fold_label")
        shared = sorted(set(equal.index) & set(optimised.index))
        if not shared:
            continue
        for label in shared:
            row = {
                "exp_id": exp_id,
                "fold_label": label,
                "repeat": int(equal.loc[label, "repeat"]),
                "fold": int(equal.loc[label, "fold"]),
            }
            for metric in PAIRED_METRICS:
                if metric not in equal.columns:
                    continue
                m6 = float(equal.loc[label, metric])
                m7 = float(optimised.loc[label, metric])
                row["M6_" + metric] = m6
                row["M7_" + metric] = m7
                row["delta_" + metric] = m7 - m6
            rows.append(row)
    return pd.DataFrame(rows)


def t09_summary(detail: object) -> object:
    """Mean, SD and the count of folds where the weights actually differed."""
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(detail)
    if frame.empty:
        return frame
    rows = []
    for exp_id, block in frame.groupby("exp_id"):
        row: dict[str, object] = {"exp_id": exp_id, "n_folds": len(block)}
        identical = 0
        for metric in PAIRED_METRICS:
            column = "delta_" + metric
            if column not in block.columns:
                continue
            values = block[column].to_numpy(dtype=float)
            row["M6_" + metric + "_mean"] = float(block["M6_" + metric].mean())
            row["M7_" + metric + "_mean"] = float(block["M7_" + metric].mean())
            row["delta_" + metric + "_mean"] = float(np.mean(values))
            row["delta_" + metric + "_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
            row["n_folds_M7_better_" + metric] = int((values > 0).sum())
            row["n_folds_M7_worse_" + metric] = int((values < 0).sum())
            if metric == "balanced_accuracy":
                identical = int((values == 0).sum())
        row["n_folds_identical_on_balanced_accuracy"] = identical
        rows.append(row)
    return pd.DataFrame(rows)


def persist_final_model(model_id: str, *, selection: dict) -> Path:
    """Refit ``model_id`` on every labelled D1 record and save it with its features.

    Fitted through the same Phase 44 pipeline the folds used -- imputer, scaler,
    estimator -- so what is saved is the whole thing a prediction needs, not a
    bare estimator that would silently receive unscaled features at inference.
    """
    import time

    import numpy as np

    from src.evaluation.tuned import ENSEMBLE_MEMBERS, selected_parameters
    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models import registry as reg
    from src.models import smoke as sm

    data = sm.load_task_data("binary")
    points = selected_parameters()

    if model_id in {"M6", "M7"}:
        member_params = {
            member: points[member] for member in ENSEMBLE_MEMBERS if member in points
        }
        estimator = est.make_ensemble(
            model_id,
            groups=np.asarray(data.groups, dtype=object),
            member_params=member_params,
        )
        params: dict = {"members": member_params}
    else:
        params = dict(points.get(model_id, {}))
        estimator = est.build_estimator(model_id, **params)

    built = pl.build_pipeline(estimator, y=data.y, n_features=data.n_features)
    started = time.perf_counter()
    built.fit(data.X, data.y)
    fit_seconds = time.perf_counter() - started

    saved = reg.save_model(
        built,
        model_id="final",
        task="binary",
        feature_names=data.feature_names,
        fit_seconds=fit_seconds,
        X_sample=data.X[:64],
        fold="all-records",
        extra={
            "selected_model_id": model_id,
            "selection_rule": selection["rule"],
            "selection_ranking": selection["ranking"],
            "accuracy_would_have_chosen": selection["accuracy_would_have_chosen"],
            "hyperparameters": params,
            "hyperparameter_source": "T07 search_space_and_best_parameters.csv",
            "n_records_fitted": int(data.n_records),
            "note": (
                "Refitted on every labelled D1 record. Its performance estimate "
                "is EXP-A2's nested cross-validation, NOT anything measured on "
                "these rows. The 25 nested folds each chose their own "
                "hyperparameters; the deployed point is T07's."
            ),
            "disclaimer": (
                "Academic screening and decision-support prototype. Not a "
                "diagnostic device and not a substitute for clinical assessment."
            ),
        },
    )
    log.info("final model: %s -> %s (%.2f MB)", model_id, saved.path, saved.size_mb)
    return saved.path


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.evaluation.experiment import Experiment
    from src.evaluation.tuned import select_final_model
    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    exp = Experiment.load(args.optimized)
    section = exp.output_dir(args.out_dir).parent
    command = "python scripts/13_finalize_binary_model.py"

    run = start_run("finalize_binary")
    run.set("baseline", args.baseline)
    run.set("optimized", args.optimized)

    # -- T09 ---------------------------------------------------------------
    detail = build_t09(args.baseline, args.optimized, args.out_dir)
    if len(detail) == 0:  # type: ignore[arg-type]
        log.error("no M6/M7 pair found in either run; T09 cannot be produced")
        run.finish(status="failed")
        return 1
    summary = t09_summary(detail)
    t09_path = save_csv(detail, section / T09_FILENAME)
    summary_path = save_csv(summary, section / "T09_ensemble_weight_summary.csv")
    run.record_artifact(t09_path)
    run.record_artifact(summary_path)
    register_evidence(
        "T09",
        t09_path,
        metric_or_asset="Equal-weight (M6) vs optimized-weight (M7) soft voting, paired by fold",
        experiment_id=args.optimized,
        dataset="D1",
        model="M6, M7",
        command=command,
    )

    # -- T65.4: the selection rule ----------------------------------------
    aggregate = pd.read_csv(exp.output_dir(args.out_dir) / "aggregate_metrics.csv")
    rule = exp.selection_rule or ("sensitivity", "balanced_accuracy")
    selection = select_final_model(aggregate, rule=rule)
    selection["exp_id"] = args.optimized
    selection["n_folds"] = int(aggregate["n_folds"].max())
    selection["note"] = (
        "Research rule 6: selection prioritises sensitivity and balanced "
        "accuracy, never raw accuracy. `accuracy_would_have_chosen` records "
        "what a raw-accuracy rule would have picked instead."
    )
    ranking_path = save_csv(pd.DataFrame(selection["ranking"]), section / SELECTION_CSV)
    selection_path = save_json(selection, section / SELECTION_JSON)
    run.record_artifact(ranking_path)
    run.record_artifact(selection_path)
    register_evidence(
        "T65-SELECTION",
        selection_path,
        metric_or_asset="Final binary model chosen by sensitivity then balanced accuracy",
        experiment_id=args.optimized,
        dataset="D1",
        model=selection["model_id"],
        command=command,
    )

    print()
    print(summary.round(5).to_string(index=False))
    print()
    print(pd.DataFrame(selection["ranking"]).round(5).to_string(index=False))
    print()
    print("selected by the rule: " + selection["model_id"])
    print("raw accuracy would have chosen: " + (selection["accuracy_would_have_chosen"] or "n/a"))

    # -- T65.6: persist ----------------------------------------------------
    if not args.skip_persist:
        path = persist_final_model(selection["model_id"], selection=selection)
        run.record_artifact(path.parent / "manifest.json")
        register_evidence(
            "T65-FINAL-MODEL",
            path.parent / "manifest.json",
            metric_or_asset="Final binary pipeline, refitted on all labelled D1 records",
            experiment_id=args.optimized,
            dataset="D1",
            model=selection["model_id"],
            command=command,
        )
        print("final model persisted to " + str(path.parent))

    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
