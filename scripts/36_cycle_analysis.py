"""Cardiac-cycle and segmentation analysis (Phase 84).

Builds the per-recording cycle table from CirCor's TSV segmentations and from
PASCAL set_a's S1/S2 timing file, records which corpora have no annotation at
all, tests whether annotation quality tracks prediction confidence or the
failure cases, compares cycle timing between classes descriptively, and draws the
annotated-waveform figure the dashboard's cycle viewer consumes.

Nothing is fitted and nothing is re-detected: every band is the annotator's.

    python scripts/36_cycle_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/36_cycle_analysis.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("cycle_analysis")

SECTION = "outputs/10_robustness"
COMMAND = "python scripts/36_cycle_analysis.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.reporting.segmentation import SAMPLE_RECORD_ID

    parser = argparse.ArgumentParser(prog="36_cycle_analysis")
    parser.add_argument(
        "--record",
        default=SAMPLE_RECORD_ID,
        help="the CirCor record the annotated-waveform figure is drawn from",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.evaluation import cycle_analysis as ca
    from src.reporting.cycle_report import build_cycle_table, draw_annotated_waveform
    from src.reporting.tables import write_table
    from src.utils.evidence import PROJECT_ROOT, register_evidence
    from src.utils.io import save_csv
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    section = PROJECT_ROOT / SECTION
    section.mkdir(parents=True, exist_ok=True)
    run = start_run("cycle_analysis")
    written: dict[str, Path] = {}

    circor = ca.circor_cycle_table()
    pascal = ca.pascal_cycle_table()
    cycles = pd.concat([circor, pascal], ignore_index=True)
    written["cycle statistics"] = save_csv(
        cycles, section / "segmentation_cycle_statistics.csv"
    )

    coverage = ca.coverage_summary(circor, pascal)
    written["coverage"] = save_csv(coverage, section / "segmentation_coverage_summary.csv")

    comparison = ca.class_comparison(circor, pascal)
    written["class comparison"] = save_csv(
        comparison, section / "segmentation_timing_by_class.csv"
    )

    correlations, per_record = ca.confidence_correlation(circor)
    written["correlations"] = save_csv(
        correlations, section / "segmentation_confidence_correlation.csv"
    )
    if len(per_record):
        written["per-record join"] = save_csv(
            per_record, section / "segmentation_confidence_per_record.csv"
        )

    sources = tuple(
        str(p).replace("\\", "/")
        for p in (
            written["cycle statistics"],
            written["coverage"],
            written["class comparison"],
            written["correlations"],
        )
    )

    table = build_cycle_table(cycles, coverage, comparison, correlations, sources, command=COMMAND)
    for fmt, path in write_table(table, section).items():
        written["SEG-01 " + fmt] = path

    figure = draw_annotated_waveform(section, args.record)
    for name, path in figure.items():
        written["waveform " + name] = path
    register_evidence(
        "SEG-02",
        figure["png"],
        metric_or_asset="Annotated waveform: S1 / systole / S2 / diastole on a real recording",
        objective="O1 (dataset characterisation)",
        experiment_id="n/a (corpus annotation, not an experiment)",
        dataset="D4 CirCor 2022",
        source_data="; ".join(
            str(figure[key]).replace("\\", "/") for key in ("samples", "bands")
        ),
        command=COMMAND,
    )

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    print(coverage.to_string(index=False))
    print()
    if len(correlations):
        print(
            correlations[
                ["run", "model_id", "predictor", "target", "n", "rho", "p_value"]
            ]
            .round(4)
            .to_string(index=False)
        )
    print()
    for name, path in written.items():
        print(f"{name:26s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
