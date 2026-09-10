"""Complexity analysis (Phase 79). Emits T24, T25, T26 and G25-G27.

Three stages, because two of them are *measurements* and must not be taken while
anything else is using the CPU. One measured episode of three concurrent jobs on
this 4-core machine cost M8 a factor of 14 in fit time -- see the 2026-09-10
contention entry in Docs/note.md. Both benches refuse to start if another
project job is running.

``--stage bench``
    T79.4 -- fits every model once on binary fold r0f0 under configs/models.yaml
    defaults, recording serialized size, fit time and peak resident set. ~7
    minutes, dominated by the two ensembles.

``--stage inference``
    T79.2 / T79.3 -- scores a duration-stratified sample of real recordings
    end to end, repeatedly, and decomposes the time by pipeline stage.

``--stage tables`` (default: all three)
    T79.1 plus the assembly: aggregates every stored run's per-fold fit and
    predict times and writes T24, T25, T26, G25, G26 and G27.

    python scripts/31_complexity_analysis.py --stage bench
    python scripts/31_complexity_analysis.py --stage inference
    python scripts/31_complexity_analysis.py --stage tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/31_complexity_analysis.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("complexity_analysis")

SECTION = "outputs/11_complexity"
COMMAND = "python scripts/31_complexity_analysis.py"

FOOTPRINT_CSV = "model_footprint.csv"
INFERENCE_CSV = "inference_timing.csv"

#: Scripts whose presence in the process table invalidates a timing measurement.
#: Named rather than "any python": this script is itself python, and so is the
#: editor's language server.
BUSY_MARKERS = (
    "scripts/01_",
    "scripts/02_",
    "scripts/11_",
    "scripts/12_",
    "scripts/23_",
    "scripts/24_",
    "scripts/25_",
    "scripts/26_",
    "scripts/27_",
    "scripts/28_",
    "scripts/29_",
    "scripts/30_",
    "joblib.externals.loky",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="31_complexity_analysis")
    parser.add_argument("--stage", default="all", choices=("bench", "inference", "tables", "all"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing bench CSV")
    parser.add_argument(
        "--ignore-busy",
        action="store_true",
        help="take a timing anyway while another job is running (the number will be wrong)",
    )
    return parser.parse_args(argv)


def _root() -> Path:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def busy_processes() -> list[str]:
    """Other project jobs currently holding CPU, by command line.

    Windows-only detection via WMI; on anything else this returns an empty list
    and the caller says the check could not run rather than claiming the machine
    is idle.
    """
    import os
    import subprocess

    if os.name != "nt":
        return []
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" "
                "| ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return []

    mine = str(os.getpid())
    busy: list[str] = []
    for line in completed.stdout.splitlines():
        text = line.strip().replace("\\", "/")
        if not text or text.split(" ", 1)[0] == mine:
            continue
        if any(marker in text for marker in BUSY_MARKERS):
            busy.append(text[:160])
    return busy


def _require_idle(args, what: str) -> bool:
    busy = busy_processes()
    if not busy:
        return True
    log.error(
        "%s is a TIMING measurement and %d other project job(s) are running:\n  %s",
        what,
        len(busy),
        "\n  ".join(busy),
    )
    if args.ignore_busy:
        log.warning("--ignore-busy given; the numbers will be wrong by up to an order of magnitude")
        return True
    log.error("wait for them to finish, or pass --ignore-busy to measure anyway")
    return False


def run_bench(args) -> int:
    from src.evaluation.complexity import model_footprint
    from src.utils.io import save_csv

    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / FOOTPRINT_CSV
    if target.is_file() and not args.force:
        log.info("%s already exists; pass --force to re-measure", target)
        return 0
    if not _require_idle(args, "the model footprint bench"):
        return 3

    frame = model_footprint()
    save_csv(frame, target)
    print(
        frame[
            [
                "model_id",
                "fit_seconds",
                "model_mb",
                "peak_rss_delta_mb",
                "n_samples",
                "memory_measurement_reliable",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    return 0 if len(frame) else 2


def run_inference(args) -> int:
    from src.evaluation.complexity import inference_sample, inference_timing, summarise_inference
    from src.utils.io import save_csv

    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / INFERENCE_CSV
    if target.is_file() and not args.force:
        log.info("%s already exists; pass --force to re-measure", target)
        return 0
    if not _require_idle(args, "the end-to-end inference bench"):
        return 3

    sample = inference_sample()
    save_csv(sample, root / "inference_sample.csv")
    frame = inference_timing(sample)
    save_csv(frame, target)
    stages = summarise_inference(frame)
    save_csv(stages, root / "inference_stage_summary.csv")
    print(stages.round(4).to_string(index=False))
    return 0 if len(frame) else 2


def _headline_quality() -> object:
    """The headline run's per-model fold means, for T24's benefit column."""
    import pandas as pd

    from src.reporting.complexity_report import HEADLINE_RUN
    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs/06_binary_results" / HEADLINE_RUN / "per_fold_metrics.csv"
    if not path.is_file():
        return pd.DataFrame({"model_id": []})
    frame = pd.read_csv(path)
    wanted = [
        m
        for m in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc", "f1")
        if m in frame.columns
    ]
    return frame.groupby("model_id", as_index=False)[wanted].mean()


def run_tables(args) -> int:
    import pandas as pd

    from src.evaluation.complexity import (
        summarise_inference,
        summarise_training_time,
        training_time_frame,
    )
    from src.reporting.complexity_report import (
        build_g25,
        build_g26,
        build_g27,
        build_overview,
        build_t24,
        build_t25,
        build_t26,
    )
    from src.reporting.graphs import write_graph
    from src.reporting.tables import write_table
    from src.utils.io import save_csv
    from src.utils.run_manifest import start_run

    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    footprint_path = root / FOOTPRINT_CSV
    inference_path = root / INFERENCE_CSV
    for path, stage in ((footprint_path, "bench"), (inference_path, "inference")):
        if not path.is_file():
            log.error("%s is missing; run --stage %s first", path, stage)
            return 2

    run = start_run("complexity_analysis")
    written: dict[str, Path] = {}

    per_fold = training_time_frame()
    summary = summarise_training_time(per_fold)
    written["per-fold training time"] = save_csv(per_fold, root / "training_time_per_fold.csv")
    written["training time summary"] = save_csv(summary, root / "training_time_summary.csv")

    footprint = pd.read_csv(footprint_path)
    timings = pd.read_csv(inference_path)
    stages = summarise_inference(timings)
    written["inference stage summary"] = save_csv(stages, root / "inference_stage_summary.csv")

    overview = build_overview(footprint, summary, stages, _headline_quality())
    written["complexity overview"] = save_csv(overview, root / "complexity_overview.csv")

    sources = tuple(
        str(p).replace("\\", "/")
        for p in (
            written["training time summary"],
            written["inference stage summary"],
            footprint_path,
            written["complexity overview"],
        )
    )

    from src.utils.evidence import PROJECT_ROOT

    section = PROJECT_ROOT / SECTION
    for builder, args_for, label in (
        (build_t24, (overview,), "T24"),
        (build_t25, (summary, stages), "T25"),
        (build_t26, (footprint,), "T26"),
    ):
        table = builder(*args_for, sources, command=COMMAND)
        for fmt, path in write_table(table, section).items():
            written[label + " " + fmt] = path

    for builder, frame, label in (
        (build_g25, overview, "G25"),
        (build_g26, summary, "G26"),
        (build_g27, footprint, "G27"),
    ):
        graph = builder(frame, sources, command=COMMAND)
        for fmt, path in write_graph(graph, formats=("png", "svg")).items():
            written[label + " " + fmt] = path

    for path in written.values():
        run.record_artifact(path)
    run.finish(status="ok")

    print()
    print(stages.round(4).to_string(index=False))
    print()
    for name, path in written.items():
        print(f"{name:34s} -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage in ("bench", "all"):
        code = run_bench(args)
        if code:
            return code
    if args.stage in ("inference", "all"):
        code = run_inference(args)
        if code:
            return code
    if args.stage in ("tables", "all"):
        return run_tables(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
