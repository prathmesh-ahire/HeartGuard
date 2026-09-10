"""Explainability (Phase 81). Emits FE-11, G18, G19 and the per-sample demo.

Fits each model once per fold on the five folds of one repeat -- a complete
partition of the corpus -- and measures impurity importance on the training rows,
permutation importance on the held-out rows, and SHAP on the tree models. Then
rolls everything up to the six feature families and demonstrates the per-sample
explanation on a real recording.

The permutation stage re-scores each model 1,381 times per fold, so it takes
roughly an hour on the default model set and wants a quiet machine.

    python scripts/33_explainability.py
    python scripts/33_explainability.py --models M1 M4      # a shorter run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/33_explainability.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("explainability")

SECTION = "outputs/04_models"
RUN_DIR = "explainability"
FEATURES_SECTION = "outputs/03_features"
COMMAND = "python scripts/33_explainability.py"

PER_FOLD_CSV = "importance_per_fold.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.explainability.global_importance import N_REPEATS, PERMUTATION_MODELS

    parser = argparse.ArgumentParser(prog="33_explainability")
    parser.add_argument("--models", nargs="*", default=list(PERMUTATION_MODELS))
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="recompute even if the per-fold CSV exists"
    )
    return parser.parse_args(argv)


def _root() -> Path:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION / RUN_DIR


def _demo_record():
    """One real abnormal PhysioNet recording, chosen deterministically.

    The first abnormal D1 record in record_uid order. A fixed rule rather than a
    hand-picked example, so the demo payload regenerates identically.
    """
    import pandas as pd

    from src.utils.config import load_config

    master = pd.read_csv(
        Path(load_config("paths").require("outputs.dataset_audit")) / "metadata_master.csv",
        low_memory=False,
    )
    candidates = master[
        (master["dataset_source"] == "D1")
        & (master["binary_label"] == 1)
        & master["use_in_supervised"].astype(bool)
    ].sort_values("record_uid")
    if not len(candidates):
        raise RuntimeError("no supervised abnormal D1 record to demonstrate on")
    return candidates.iloc[0]


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.explainability.global_importance import (
        EXCLUDED_MODELS,
        IMPURITY_MODELS,
        SHAP_MODELS,
        TOP_N,
        aggregate_family_importance,
        aggregate_importance,
        fold_importances,
        shap_importance,
        top_features,
    )
    from src.explainability.per_sample import explain_prediction
    from src.reporting.graphs import write_graph
    from src.reporting.importance_report import build_fe11, build_g18, build_g19
    from src.reporting.tables import write_table
    from src.utils.evidence import PROJECT_ROOT
    from src.utils.io import save_csv, save_json
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    run = start_run("explainability")
    written: dict[str, Path] = {}

    permutation_models = tuple(args.models)
    impurity_models = tuple(m for m in IMPURITY_MODELS if m in permutation_models) or tuple(
        m for m in IMPURITY_MODELS if m in ("M4", "M5", "M8")
    )

    per_fold_path = root / PER_FOLD_CSV
    if per_fold_path.is_file() and not args.force:
        log.info("%s exists; reusing it (pass --force to recompute)", per_fold_path)
        per_fold = pd.read_csv(per_fold_path)
    else:
        per_fold = fold_importances(
            permutation_models=permutation_models,
            impurity_models=impurity_models,
            n_repeats=args.n_repeats,
        )
        written["per-fold importance"] = save_csv(per_fold, per_fold_path)

    aggregated = aggregate_importance(per_fold)
    per_fold_family, family = aggregate_family_importance(per_fold)
    written["aggregated importance"] = save_csv(aggregated, root / "importance_summary.csv")
    written["per-fold family"] = save_csv(
        per_fold_family, root / "feature_family_importance_per_fold.csv"
    )
    written["family importance"] = save_csv(family, root / "feature_family_importance.csv")

    if args.no_shap:
        shap_frame, shap_report = pd.DataFrame(), {
            "status": "skipped",
            "reason": "--no-shap was passed on the command line",
        }
    else:
        shap_frame, shap_report = shap_importance(
            models=tuple(m for m in SHAP_MODELS if m in permutation_models) or SHAP_MODELS
        )
    if len(shap_frame):
        written["shap importance"] = save_csv(shap_frame, root / "shap_importance.csv")
    written["shap report"] = save_json(shap_report, root / "shap_report.json")

    coverage = pd.DataFrame(
        [
            {
                "model_id": model,
                "permutation": model in permutation_models,
                "impurity": model in impurity_models,
                "shap": model in set(shap_frame["model_id"]) if len(shap_frame) else False,
                "excluded_reason": EXCLUDED_MODELS.get(model, ""),
            }
            for model in sorted(set(permutation_models) | set(EXCLUDED_MODELS))
        ]
    )
    written["coverage"] = save_csv(coverage, root / "explainability_coverage.csv")

    top = top_features(aggregated, n=TOP_N)

    record = _demo_record()
    _, explanation = explain_prediction(
        PROJECT_ROOT / str(record["file_path"]), record_uid=str(record["record_uid"])
    )
    payload = explanation.as_payload()
    payload["record_uid"] = str(record["record_uid"])
    payload["true_class"] = str(record["binary_label_name"])
    payload["selection_rule"] = "first abnormal D1 record in record_uid order"
    written["per-sample demo"] = save_json(payload, root / "per_sample_explanation.json")

    sources = tuple(
        str(p).replace("\\", "/")
        for p in (
            root / PER_FOLD_CSV,
            written["aggregated importance"],
            written["family importance"],
            root / "shap_importance.csv",
        )
    )

    table = build_fe11(top, family, sources, command=COMMAND)
    for fmt, path in write_table(table, PROJECT_ROOT / FEATURES_SECTION).items():
        written["FE-11 " + fmt] = path

    for builder, frame, label in (
        (build_g18, family, "G18"),
        (build_g19, top, "G19"),
    ):
        graph = (
            builder(frame, aggregated, sources, command=COMMAND)
            if label == "G18"
            else builder(frame, sources, command=COMMAND)
        )
        for fmt, path in write_graph(graph, formats=("png", "svg")).items():
            written[label + " " + fmt] = path

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    print(
        family[family["kind"] == "permutation"][
            ["model_id", "family", "total_importance_mean", "share_of_positive_mean", "rank"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()
    print("per-sample demo on " + payload["record_uid"] + " (" + payload["true_class"] + "):")
    print(
        "  predicted "
        + payload["predicted_class"]
        + " at p="
        + format(payload["probability"], ".4f")
        + ", reconstruction error "
        + format(payload["reconstruction_error"], ".2e")
    )
    for item in payload["top_contributions"][:5]:
        print(
            "  {feature:28s} {contribution:+8.4f}  {direction}".format(**item)
        )
    print()
    for name, path in written.items():
        print(f"{name:26s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
