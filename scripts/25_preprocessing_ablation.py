"""PP-09 -- the preprocessing ablation (T29.3, T29.4).

Runs the four filter x normalization arms defined in Phase 29 end to end: for
each arm, rebuild every D1 signal under that arm's configuration, extract the
locked 138 features, and cross-validate one fast baseline model (M1, untuned)
over the DA-07 5x5 grouped fold map. Emits PP-09 with the metric delta of each
arm against PP-A, the shipped configuration.

Deferred at Phase 29 because it needs features and a model that Parts IV and V
build; both have existed since Phase 46.

    python scripts/25_preprocessing_ablation.py --smoke        # 24 records, no PP-09
    python scripts/25_preprocessing_ablation.py                # full D1, writes PP-09
    python scripts/25_preprocessing_ablation.py --n-records 800

Feature extraction dominates the wall clock (~5 s of CPU per record per arm)
and is the reason for the checkpoint: each arm's matrix is written to
``cache/features/pp_ablation/`` after every batch and a restart resumes from
there rather than beginning again. ``--stage score`` re-scores what is already
extracted without touching the audio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/25_preprocessing_ablation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger
from src.utils.timing import format_duration

log = get_logger("pp_ablation")

COMMAND = "python scripts/25_preprocessing_ablation.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="25_preprocessing_ablation")
    parser.add_argument(
        "--stage",
        default="both",
        choices=("extract", "score", "both"),
        help="extract features, score what is extracted, or both (default: both)",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=None,
        metavar="N",
        help="stratified subsample of D1; default is every supervised record",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--force", action="store_true", help="discard checkpoints and re-extract"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="24 records, into a scratch out-dir; validates the path without writing PP-09",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write PP-09 somewhere other than outputs/02_preprocessing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import time

    from src.preprocessing import ablation_run as ar
    from src.preprocessing.ablation import ABLATION_GRID, arm_rows
    from src.utils.io import save_json
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    n_records = 24 if args.smoke else args.n_records
    out_dir = args.out_dir
    if args.smoke and out_dir is None:
        # A 24-record PP-09 sitting where a 3,240-record one belongs reads
        # exactly like the real thing. Same rule as the feature extractor's
        # _smoke subtree.
        out_dir = str(Path("outputs") / "02_preprocessing" / "_smoke")

    run = start_run("preprocessing_ablation" + ("_smoke" if args.smoke else ""))
    run.set("ablation_stage", args.stage)
    run.set("ablation_n_records", n_records)
    run.set("ablation_model_id", ar.BASELINE_MODEL_ID)
    run.set("ablation_workers", args.workers)

    try:
        records = ar.arm_records(n_records)
        grid = {row["arm_id"]: row for row in arm_rows()}
        summaries: list[dict] = []
        per_fold: list = []
        started = time.perf_counter()

        for arm in ABLATION_GRID:
            table, extract_seconds = ar.extract_arm(
                arm,
                records,
                workers=args.workers,
                batch=args.batch,
                force=args.force,
            )
            if arm.arm_id == ar.REFERENCE_ARM and not args.smoke:
                report = ar.verify_shipped_arm(table)
                run.set("ablation_ppa_vs_fe03", report)
                if not report["matches"]:
                    log.warning(
                        "PP-A does not reproduce FE-03 (max relative difference %.3g on %s)"
                        " -- the deltas below include a code-path difference",
                        report["max_relative_difference"],
                        report["worst_feature"],
                    )
            if args.stage == "extract":
                continue

            scores, folds = ar.score_arm(arm, table)
            per_fold.append(folds)
            summaries.append(
                {
                    "arm_id": arm.arm_id,
                    "label": arm.label,
                    "filter_enabled": arm.filter_enabled,
                    "normalization_enabled": arm.normalization_enabled,
                    "is_shipped_configuration": grid[arm.arm_id]["is_shipped_configuration"],
                    "config_hash": grid[arm.arm_id]["config_hash"],
                    "model_id": ar.BASELINE_MODEL_ID,
                    "task": ar.ABLATION_TASK,
                    "extract_seconds": round(extract_seconds, 3),
                    **scores,
                }
            )

        wall = time.perf_counter() - started
        if args.stage == "extract":
            log.info("extraction only: %s", format_duration(wall))
            run.set("ablation_wall_seconds", round(wall, 3))
            run.finish("ok")
            return 0

        table = ar.build_ablation_table(summaries)
        path = ar.write_ablation(table, out_dir)
        ar.write_per_fold(per_fold, out_dir)
        save_json(
            {
                "model_id": ar.BASELINE_MODEL_ID,
                "task": ar.ABLATION_TASK,
                "reference_arm": ar.REFERENCE_ARM,
                "n_records": len(records),
                "n_subjects": int(records["subject_id"].nunique()),
                "smoke": bool(args.smoke),
                "wall_seconds": round(wall, 3),
                "command": COMMAND,
            },
            Path(path).with_name("preprocessing_ablation_run.json"),
        )

        for row in table.to_dict("records"):
            log.info(
                "%s  %-38s sensitivity %.4f (%+.4f)  balanced %.4f (%+.4f)",
                row["arm_id"],
                row["label"],
                row["sensitivity"],
                row["sensitivity_delta"],
                row["balanced_accuracy"],
                row["balanced_accuracy_delta"],
            )
        log.info("PP-09 written in %s", format_duration(wall))

        if not args.smoke:
            from src.preprocessing.artifacts import register_preprocessing_artifacts

            register_preprocessing_artifacts()

        run.set("ablation_wall_seconds", round(wall, 3))
    except BaseException:
        run.finish("failed")
        raise

    run.finish("ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
