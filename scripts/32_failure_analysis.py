"""Failure analysis (Phase 80). Emits T27, G35 and the narrative report.

Reads the stored out-of-fold predictions of the final binary model and goes back
to the records: every false positive and false negative, categorised by duration
band, PP-08 noise flag, sub-collection, predicted confidence and PhysioNet
diagnosis; the most-confused class pairs from every multiclass confusion matrix;
and a narrative that names concrete records.

Nothing here fits a model, so it needs no idle machine and runs in seconds.

    python scripts/32_failure_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/32_failure_analysis.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("failure_analysis")

SECTION = "outputs/10_robustness"
RUN_DIR = "failure_analysis"
COMMAND = "python scripts/32_failure_analysis.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="32_failure_analysis")
    parser.add_argument(
        "--figure-source",
        default="",
        help="which error set G35 draws; defaults to the first declared source",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from src.evaluation.failure_analysis import (
        CATEGORIES,
        SOURCES,
        categorise,
        confusion_pairs,
        example_records,
        prediction_failures,
        record_failures,
    )
    from src.reporting.failure_report import build_g35, build_t27, write_failure_report
    from src.reporting.graphs import write_graph
    from src.reporting.tables import write_table
    from src.utils.evidence import PROJECT_ROOT
    from src.utils.io import save_csv
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    section = PROJECT_ROOT / SECTION
    root = section / RUN_DIR
    root.mkdir(parents=True, exist_ok=True)
    run = start_run("failure_analysis")
    written: dict[str, Path] = {}

    all_failures: list[pd.DataFrame] = []
    all_categorised: list[pd.DataFrame] = []
    for source in SOURCES:
        try:
            failures = prediction_failures(source)
        except FileNotFoundError as error:
            log.warning("%s: %s", source.name, error)
            continue
        all_failures.append(failures)
        # The diagnosis grouping only exists where the corpus carries the field.
        groupings = [*CATEGORIES, "diagnosis_class"]
        for category in groupings:
            if category not in failures.columns:
                continue
            block = failures[failures[category].notna()]
            if not len(block):
                continue
            all_categorised.append(categorise(block, category))
        log.info(
            "%s: %d prediction(s), %d error(s)",
            source.name,
            len(failures),
            int(failures["is_error"].sum()),
        )

    if not all_failures:
        log.error("no binary run had stored predictions to analyse")
        return 2

    failures = pd.concat(all_failures, ignore_index=True)
    categorised = pd.concat(all_categorised, ignore_index=True)
    records = record_failures(failures)
    errors_only = failures[failures["is_error"]]
    examples = example_records(records)
    pairs = confusion_pairs()

    written["all predictions"] = save_csv(failures, root / "binary_predictions_labelled.csv")
    written["errors only"] = save_csv(errors_only, root / "false_positives_and_negatives.csv")
    written["per record"] = save_csv(records, root / "record_failures.csv")
    written["categorised"] = save_csv(categorised, root / "failures_by_category.csv")
    written["named examples"] = save_csv(examples, root / "named_example_records.csv")
    written["confused pairs"] = save_csv(pairs, root / "confused_class_pairs.csv")

    sources = tuple(
        str(p).replace("\\", "/")
        for p in (
            written["categorised"],
            written["errors only"],
            written["per record"],
            written["confused pairs"],
        )
    )

    table = build_t27(categorised, pairs, sources, command=COMMAND)
    for fmt, path in write_table(table, section).items():
        written["T27 " + fmt] = path

    graph = build_g35(categorised, pairs, sources, command=COMMAND, source=args.figure_source)
    for fmt, path in write_graph(graph, formats=("png", "svg")).items():
        written["G35 " + fmt] = path

    written["narrative report"] = write_failure_report(
        root / "failure_report.md",
        categorised=categorised,
        records=records,
        examples=examples,
        pairs=pairs,
        sources=sources,
        command=COMMAND,
    )

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    columns = ["source", "category", "level", "n_records", "error_rate", "false_negative_rate"]
    print(categorised[columns].round(4).to_string(index=False))
    print()
    for name, path in written.items():
        print(f"{name:24s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
