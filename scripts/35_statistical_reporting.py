"""Statistical reporting (Phase 83). Emits T28, the matrix, the DOCX and the findings.

Reads what `scripts/34_statistical_validation.py` computed and renders it four
ways: the model-by-model p-value matrix, the summary DOCX, the numbered T28
table, and the plain-statement findings document that says where the proposed
ensemble is NOT significantly better and where a test was underpowered rather
than negative.

    python scripts/35_statistical_reporting.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/35_statistical_reporting.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("statistical_reporting")

SECTION = "outputs/12_statistics"
COMMAND = "python scripts/35_statistical_reporting.py"

REQUIRED = (
    "paired_fold_tests.csv",
    "friedman_omnibus.csv",
    "foldwise_mean_sd.csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="35_statistical_reporting")
    parser.add_argument(
        "--metric",
        default="sensitivity",
        help="the metric T28 and the findings document lead on (rule 6: sensitivity)",
    )
    return parser.parse_args(argv)


def _read(root: Path, name: str):
    import pandas as pd

    path = root / name
    return pd.read_csv(path) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    from src.reporting.statistics_report import (
        build_t28,
        interpret,
        significance_matrix,
        write_findings,
        write_summary_docx,
    )
    from src.reporting.tables import write_table
    from src.utils.evidence import PROJECT_ROOT, register_evidence
    from src.utils.io import save_csv
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    root = PROJECT_ROOT / SECTION
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        log.error(
            "missing %s; run scripts/34_statistical_validation.py first",
            ", ".join(missing),
        )
        return 2

    run = start_run("statistical_reporting")
    written: dict[str, Path] = {}

    paired = _read(root, "paired_fold_tests.csv")
    omnibus = _read(root, "friedman_omnibus.csv")
    posthoc = _read(root, "nemenyi_posthoc.csv")
    mcnemar = _read(root, "mcnemar_paired_predictions.csv")
    bootstrap = _read(root, "bootstrap_auc_ci.csv")

    paired = paired.copy()
    paired["interpretation"] = [interpret(row) for _, row in paired.iterrows()]
    written["annotated paired tests"] = save_csv(
        paired, root / "paired_fold_tests_interpreted.csv"
    )

    matrix, matrix_tests = significance_matrix(paired, value="p_holm")
    written["significance matrix"] = save_csv(
        matrix, root / "statistical_significance_matrix.csv"
    )
    written["significance matrix (test per cell)"] = save_csv(
        matrix_tests, root / "statistical_significance_matrix_tests.csv"
    )
    raw_matrix, _ = significance_matrix(paired, value="p_value")
    written["significance matrix (raw p)"] = save_csv(
        raw_matrix, root / "statistical_significance_matrix_raw.csv"
    )
    if mcnemar is not None and len(mcnemar):
        mcnemar_matrix, _ = significance_matrix(mcnemar, value="p_holm")
        written["mcnemar matrix"] = save_csv(
            mcnemar_matrix, root / "mcnemar_significance_matrix.csv"
        )

    written["summary docx"] = write_summary_docx(
        root / "statistical_summary_table.docx",
        paired=paired,
        omnibus=omnibus,
        mcnemar=mcnemar,
        bootstrap=bootstrap,
        command=COMMAND,
    )

    sources = tuple(
        str(root / name).replace("\\", "/")
        for name in (
            "paired_fold_tests.csv",
            "friedman_omnibus.csv",
            "mcnemar_paired_predictions.csv",
            "bootstrap_auc_ci.csv",
            "foldwise_mean_sd.csv",
        )
    )

    table = build_t28(paired, sources, command=COMMAND, metric=args.metric)
    for fmt, path in write_table(table, root).items():
        written["T28 " + fmt] = path

    written["findings"] = write_findings(
        root / "statistical_findings.md",
        paired=paired,
        omnibus=omnibus,
        posthoc=posthoc,
        sources=sources,
        command=COMMAND,
        metric=args.metric,
    )

    # T83.6 -- everything in this section is registered, not only the numbered
    # table the table engine registers on its own.
    registrations = {
        "STAT-01": ("significance matrix", "Model-by-model Holm-corrected p-value matrix"),
        "STAT-02": ("summary docx", "Statistical summary table (test, statistic, p, effect)"),
        "STAT-03": ("findings", "Plain statements: non-significant and underpowered results"),
        "STAT-04": ("annotated paired tests", "Paired fold-level tests with interpretations"),
    }
    for evidence_id, (key, description) in registrations.items():
        if key not in written:
            continue
        register_evidence(
            evidence_id,
            written[key],
            metric_or_asset=description,
            objective="O3 (model comparison)",
            experiment_id="EXP-A1, EXP-A2, EXP-B1, EXP-B2, EXP-C1, EXP-C2",
            dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
            source_data="; ".join(sources),
            command=COMMAND,
        )

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    print((root / "statistical_findings.md").read_text(encoding="utf-8")[:2400])
    print()
    for name, path in written.items():
        print(f"{name:28s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
