"""Deploy the four non-binary task models (Phase 120 prerequisite).

T65.6 persisted ``models_saved/binary/final/``. Nothing ever persisted PASCAL A,
PASCAL B, CirCor murmur or CirCor outcome, so four of the five prediction pages
could only render "No model is deployed for this task yet" -- which is what
Phase 120's screenshots 9 and 10 are supposed to show output for. The rule this
applies is stated in ``src/models/deploy.py``; it is T65.6's, generalized, not a
new one.

Usage
-----
    python scripts/46_finalize_task_models.py
    python scripts/46_finalize_task_models.py --task pascal_a --task pascal_b
    python scripts/46_finalize_task_models.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/46_finalize_task_models.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.deploy import DEPLOYMENTS, deploy_task, rank_models
from src.utils.logging_setup import get_logger

log = get_logger("finalize_task_models")

SUMMARY_JSON = "deployed_task_models.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="46_finalize_task_models",
        description="Rank, search and persist models_saved/<task>/final for the four "
        "non-binary tasks.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=[spec.task for spec in DEPLOYMENTS],
        help="deploy only this task; repeatable (default: all four)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the ranking and the model the rule selects, and stop",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.utils.config import load_config
    from src.utils.evidence import register_evidence
    from src.utils.io import ensure_dir, save_json
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    wanted = set(args.task or [spec.task for spec in DEPLOYMENTS])
    specs = [spec for spec in DEPLOYMENTS if spec.task in wanted]

    if args.dry_run:
        for spec in specs:
            selection = rank_models(spec)
            print(spec.task + ": " + selection["selected_model_id"] + " by " + "/".join(
                selection["rule"]
            ))
            for row in selection["ranking"]:
                print("    " + json.dumps(row))
            print("    accuracy would have chosen: "
                  + str(selection["accuracy_would_have_chosen"]))
        return 0

    results = []
    run = start_run("deploy_task_models")
    try:
        for spec in specs:
            outcome = deploy_task(spec)
            results.append(outcome)
            print(
                spec.task
                + ": "
                + outcome["selected_model_id"]
                + " refit on "
                + str(outcome["n_records_fitted"])
                + " records -> "
                + outcome["path"]
            )
            run.record_artifact(outcome["path"])
    except Exception:
        run.finish(status="failed")
        raise
    run.finish(status="ok")

    root = Path(load_config("paths").require("outputs.models"))
    target = save_json({"deployments": results}, ensure_dir(root) / SUMMARY_JSON)
    register_evidence(
        "MODEL-DEPLOY",
        target,
        metric_or_asset="Deployed model per task: selection rule, ranking and "
        "hyperparameter source",
        objective="Deployment",
        source_data=",".join(spec.run_dir + "/aggregate_metrics.csv" for spec in specs),
        command="python scripts/46_finalize_task_models.py",
    )
    print("summary: " + str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
