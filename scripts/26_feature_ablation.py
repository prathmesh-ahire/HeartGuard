"""EXP-F1 -- the feature-family ablation (Phase 74). Emits T17 and T18.

Runs configurations A1-A8 over the DA-07 5x5 grouped fold map, each one a
column subset of the single FE-03 matrix, then emits T17 (the family ablation)
and T18 (all 138 versus the SO-04 selected subset).

    python scripts/26_feature_ablation.py --smoke              # one repeat, M1 only
    python scripts/26_feature_ablation.py --stage run          # the eight arms
    python scripts/26_feature_ablation.py --stage tables       # T17 and T18
    python scripts/26_feature_ablation.py                      # both

Every arm resumes: ``run_experiment`` checkpoints each (model, fold) unit, so a
killed run recomputes only what it had not finished. A7 is checked against the
committed EXP-A1 numbers before any table is written -- it is the same
configuration, so a mismatch means the runner drifted and no delta is
trustworthy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/26_feature_ablation.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("feature_ablation")

COMMAND = "python scripts/26_feature_ablation.py"
SECTION = "outputs/09_ablation"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="26_feature_ablation")
    parser.add_argument("--stage", default="both", choices=("run", "tables", "both"))
    parser.add_argument("--configurations", nargs="+", default=None, metavar="ID")
    parser.add_argument("--models", nargs="+", default=None, metavar="ID")
    parser.add_argument(
        "--smoke", action="store_true", help="M1 only, one repeat, into a scratch out-dir"
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def run_stage(args) -> dict:
    from src.evaluation import feature_ablation as fa
    from src.models import smoke as sm

    models = tuple(args.models) if args.models else (("M1",) if args.smoke else fa.ABLATION_MODELS)
    configurations = tuple(args.configurations) if args.configurations else fa.configuration_ids()
    data = sm.load_task_data("binary")

    results = {}
    for config_id in configurations:
        results[config_id] = fa.run_configuration(
            config_id,
            data=data,
            models=models,
            out_dir=args.out_dir,
            resume=not args.no_resume,
        )
    return results


def tables_stage(args) -> int:

    from src.evaluation import feature_ablation as fa
    from src.reporting.ablation_report import build_t17, build_t18
    from src.reporting.tables import write_table
    from src.utils.io import save_csv

    section = Path(args.out_dir) if args.out_dir else Path(SECTION)
    section.mkdir(parents=True, exist_ok=True)

    per_fold = fa.collect_per_fold(out_dir=args.out_dir)
    summary = fa.summarise(per_fold)
    save_csv(per_fold, section / "feature_ablation_per_fold.csv")
    save_csv(summary, section / "feature_ablation_summary.csv")

    # The zeroth arm: remove the audio entirely. A1-A8 ask what each feature
    # family is worth; this asks what is available from the recording's
    # sub-collection alone, which on the diagnosis track turns out to be more
    # than any model achieves. Cheap, and it belongs beside the family ablation.
    from src.evaluation import source_baseline as sb

    save_csv(sb.baseline_table(), section / "source_only_baselines.csv")

    # The A7 control, before any table is written. A drifting control makes
    # every other row meaningless, so it is a hard stop rather than a warning.
    control = fa.assert_a7_reproduces_exp_a1(out_dir=args.out_dir)
    save_csv(control, section / "a7_vs_exp_a1_control.csv")

    sources = (
        "outputs/09_ablation/feature_ablation_per_fold.csv",
        "outputs/03_features/all_features_matrix.parquet",
        "outputs/01_dataset_audit/subject_split_map.csv",
        "outputs/03_features/selected_feature_subset.csv",
    )
    written = {}
    for builder in (build_t17, build_t18):
        table = builder(summary, per_fold, sources=sources, command=COMMAND)
        written[table.spec.table_id] = write_table(table, section)
        log.info("%s -> %s", table.spec.table_id, written[table.spec.table_id]["csv"].name)

    for row in summary[summary["model_id"] == "M1"].to_dict("records"):
        log.info(
            "%-3s %-45s n=%3d  sensitivity %.4f (%+.4f vs A7)",
            row["config_id"],
            row["families"],
            row["n_features"],
            row["sensitivity"],
            row["sensitivity_delta_vs_A7"],
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    if args.smoke and args.out_dir is None:
        args.out_dir = str(Path(SECTION) / "_smoke")

    run = start_run("feature_ablation" + ("_smoke" if args.smoke else ""))
    run.set("ablation_stage", args.stage)
    run.set("ablation_smoke", bool(args.smoke))

    try:
        if args.stage in ("run", "both"):
            results = run_stage(args)
            run.set("ablation_configurations", sorted(results))
        if args.stage in ("tables", "both") and not args.smoke:
            tables_stage(args)
    except BaseException:
        run.finish("failed")
        raise

    run.finish("ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
