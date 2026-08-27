"""Build FE-03 and every feature report and figure over it (Phases 40-42).

Reads the shards written by ``02_extract_features.py`` and emits, in order:

    FE-03  all_features_matrix.parquet      the merged matrix (T40.6)
           extraction_wall_time.csv         per-dataset cost (T40.1-T40.5)
    FE-04  feature_missing_values.csv       every NaN and Inf, with records (T41.1)
           feature_variance_report.csv      constant / near-zero variance (T41.2)
           feature_outlier_report.csv       tails and the clipping policy (T41.3)
           feature_domain_shift.csv         per-dataset means, for EXP-D1 (T41.5)
    FE-10  feature_correlation_matrix.png   + its source CSV (T41.6)
    FE-05  feature_distribution_plots/      top features by class separation (T42.1)
    FE-06  mfcc_heatmap.png                 representative record (T42.2)
    G10    feature_family_count_chart.png   24/22/39/24/24/5 (T42.3)
           class_conditional_top10_D1.png   overlays (T42.4)

Usage
-----
    python scripts/03_feature_reports.py                # everything
    python scripts/03_feature_reports.py --skip-figures # tables only
    python scripts/03_feature_reports.py --repro 50     # also run the T41.4 check

The matrix step refuses to run on an incomplete extraction: a matrix missing
records is not a smaller matrix, it is a different corpus, and it would train a
model that silently never saw them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/03_feature_reports.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("feature_reports")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="03_feature_reports",
        description="Merge the feature shards into FE-03 and emit the FE reports.",
    )
    parser.add_argument(
        "--out-dir", default=None, help="write elsewhere than outputs/03_features"
    )
    parser.add_argument(
        "--skip-figures", action="store_true", help="tables only, no PNGs"
    )
    parser.add_argument(
        "--repro",
        type=int,
        default=0,
        metavar="N",
        help="re-extract N random records and assert bit-identical values (T41.4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.feature_extraction import distributions, figures, matrix, quality
    from src.utils.evidence import register_evidence
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    run = start_run("feature_reports")

    try:
        # -- FE-03 ----------------------------------------------------------
        path, report = matrix.write_matrix(args.out_dir)
        log.info(
            "FE-03: %d rows, %d meta + %d feature columns, digest %s",
            report.n_rows,
            report.n_meta,
            report.n_features,
            report.digest,
        )
        for dataset, count in sorted(report.datasets.items()):
            log.info("  %-3s %5d", dataset, count)

        table = matrix.load_matrix(args.out_dir)
        directory = path.parent
        wall = matrix.wall_time_table(table)
        wall_path = directory / matrix.WALL_TIME_FILENAME
        wall.to_csv(wall_path, index=False)
        log.info("wall time -> %s", wall_path)

        run.set("matrix_rows", report.n_rows)
        run.set("matrix_columns", report.n_columns)
        run.set("matrix_incomplete_rows", report.n_incomplete_rows)
        run.set("matrix_digest", report.digest)

        register_evidence(
            "FE-03",
            path,
            metric_or_asset="merged feature matrix (" + str(report.n_rows) + " x 138)",
            dataset="D1+D2+D3+D4",
            source_data="cache/features/" + report.digest,
            command="python scripts/03_feature_reports.py",
        )

        # -- FE-04, FE-10 and the supporting tables --------------------------
        written = quality.write_quality_artifacts(table, args.out_dir)
        register_evidence(
            "FE-04",
            written["FE-04"],
            metric_or_asset="per-feature NaN/Inf with named records",
            dataset="D1+D2+D3+D4",
            source_data=path,
            command="python scripts/03_feature_reports.py",
        )
        register_evidence(
            "FE-10",
            written["FE-10"],
            metric_or_asset="138x138 feature correlation heatmap",
            dataset="D1+D2+D3+D4",
            source_data=written["correlation_csv"],
            command="python scripts/03_feature_reports.py",
        )

        # -- figures ---------------------------------------------------------
        if not args.skip_figures:
            distribution = distributions.write_distribution_artifacts(
                table, args.out_dir
            )
            panels = sorted(distribution["FE-05-dir"].glob("*.png"))
            # FE-05's registered filename is the directory's manifest, not the
            # directory: the evidence index verifies with is_file(), and a
            # directory row would pass that check while being empty.
            register_evidence(
                "FE-05",
                distribution["FE-05"],
                metric_or_asset=(
                    str(len(panels))
                    + " class-conditional panels in "
                    + distribution["FE-05-dir"].name
                    + "/"
                ),
                dataset="D1",
                source_data=path,
                command="python scripts/03_feature_reports.py",
            )
            fe06 = figures.plot_mfcc_heatmap(
                figures.features_dir(args.out_dir) / figures.FE_FIGURES["FE-06"]
            )
            register_evidence(
                "FE-06",
                fe06,
                metric_or_asset="MFCC heatmap, representative record",
                dataset="D1",
                command="python scripts/03_feature_reports.py",
            )
            g10 = figures.plot_family_counts(
                figures.diagrams_dir() / figures.G10_FILENAME
            )
            register_evidence(
                "G10",
                g10,
                metric_or_asset="feature-family composition 24/22/39/24/24/5",
                source_data="feature registry",
                command="python scripts/03_feature_reports.py",
            )

        # -- T41.4 -----------------------------------------------------------
        if args.repro:
            result = quality.reproducibility_check(table, args.repro)
            log.info(
                "reproducibility: %d records x 138 = %d values, %d mismatches",
                result["n_checked"],
                result["n_values"],
                result["n_mismatches"],
            )
            run.set("repro_checked", result["n_checked"])
            run.set("repro_mismatches", result["n_mismatches"])
            if not result["identical"]:
                for entry in result["mismatches"][:10]:
                    log.error("  %s", entry)
                run.finish("failed")
                return 1
    except BaseException:
        run.finish("failed")
        raise

    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
