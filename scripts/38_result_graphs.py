"""Generate the result figures G11-G17, G20-G22 and G28 (Phases 92-93).

Each figure is written into ``outputs/13_figures_diagrams/`` as a 300 dpi PNG,
the **exact CSV that produced it** (T90.2), a ``.meta.json`` provenance record
and a row in ``figure_registry.csv`` holding its printed figure number (T90.3).

The other G-figures are drawn by the analysis that owns their numbers and are
not duplicated here: G18/G19 by ``33_explainability.py``, G23/G24 by
``27_optimization_ablation.py``, G25-G27 by ``31_complexity_analysis.py``,
G29-G35 by scripts 21-24, 30 and 32.

G17 reads per-class probabilities from the gitignored ``predictions.parquet``;
``--skip-parquet`` builds the rest on a checkout without it.

``--normalize-registry`` rewrites every registry row and meta to repo-relative
provenance and renumbers the G-series into index order (see
``graphs.normalize_registry`` for why that is allowed exactly once).

Usage
-----
    python scripts/38_result_graphs.py
    python scripts/38_result_graphs.py --figures G12 G17
    python scripts/38_result_graphs.py --normalize-registry --figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/38_result_graphs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.graphs import FORMATS, PROFILES
from src.reporting.result_graphs import NEEDS_PARQUET, RESULT_GRAPH_IDS
from src.utils.logging_setup import get_logger

log = get_logger("result_graphs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="38_result_graphs",
        description="Generate G11-G17, G20-G22 and G28 with their source CSVs.",
    )
    parser.add_argument("--figures", nargs="*", default=list(RESULT_GRAPH_IDS), metavar="ID")
    parser.add_argument(
        "--formats", nargs="+", default=["png"], choices=list(FORMATS), metavar="FMT"
    )
    parser.add_argument("--profile", default="screen", choices=list(PROFILES))
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="skip the figures that need a gitignored predictions.parquet (G17)",
    )
    parser.add_argument(
        "--normalize-registry",
        action="store_true",
        help="after writing, make registry provenance portable and number G<nn> as nn",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.reporting.graphs import normalize_registry, read_registry, registry_path, write_graphs
    from src.reporting.result_graphs import build_result_graphs
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    wanted = [f for f in args.figures if not (args.skip_parquet and f in NEEDS_PARQUET)]
    command = "python scripts/38_result_graphs.py --figures " + " ".join(wanted)

    run = start_run("result_graphs")
    run.set("figures", wanted)
    run.set("formats", list(args.formats))
    run.set("profile", args.profile)

    graphs = build_result_graphs(tuple(wanted), command=command)
    written = write_graphs(
        graphs,
        args.out_dir,
        formats=tuple(args.formats),
        profile=args.profile,
        evidence_index=args.evidence_index,
    )
    for paths in written.values():
        for path in paths.values():
            run.record_artifact(path)

    if args.normalize_registry:
        normalize_registry(args.out_dir, renumber_prefix="G")
        run.set("registry_normalized", True)

    registry = {row["figure_id"]: row for row in read_registry(registry_path(args.out_dir))}
    print()
    print(f"{'ID':<5} {'#':>3}  {'Figure':<42} {'Rows':>7}")
    print("-" * 64)
    for figure_id in sorted(registry):
        row = registry[figure_id]
        rows = next((len(g.frame) for g in graphs if g.spec.figure_id == figure_id), None)
        shown = f"{rows:>7,}" if rows is not None else "      -"
        print(f"{figure_id:<5} {row['figure_number']:>3}  {row['title']:<42} {shown}")
    print()
    print("registry: " + str(registry_path(args.out_dir)).replace("\\", "/"))

    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
