"""Generate G01-G10, the data and signal figures (Phase 91).

Every figure is written into ``outputs/13_figures_diagrams/`` as four files: the
PNG at 300 dpi, the **exact CSV that produced it** (T90.2), a ``.meta.json``
provenance record, and a row in ``figure_registry.csv`` holding the stable
printed figure number (T90.3).

G05-G09 read audio from ``dataset/`` (read-only) and run the preprocessing
pipeline over a handful of records. They cannot be built on a checkout without
the corpus; ``--skip-audio`` builds the six that need only committed CSVs.

Usage
-----
    python scripts/19_data_graphs.py
    python scripts/19_data_graphs.py --figures G01 G10 --formats png svg
    python scripts/19_data_graphs.py --profile print --skip-audio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/19_data_graphs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.data_graphs import DATA_GRAPH_IDS, NEEDS_AUDIO
from src.reporting.graphs import FORMATS, PROFILES
from src.utils.logging_setup import get_logger

log = get_logger("data_graphs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="19_data_graphs",
        description="Generate G01-G10 with their source CSVs and figure registry.",
    )
    parser.add_argument("--figures", nargs="+", default=list(DATA_GRAPH_IDS), metavar="ID")
    parser.add_argument(
        "--formats", nargs="+", default=["png"], choices=list(FORMATS), metavar="FMT"
    )
    parser.add_argument("--profile", default="screen", choices=list(PROFILES))
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="build only the figures that need no audio (G01-G04, G10)",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.reporting.data_graphs import build_data_graphs
    from src.reporting.graphs import write_graphs
    from src.utils.run_manifest import start_run

    args = parse_args(argv)
    wanted = [f for f in args.figures if not (args.skip_audio and f in NEEDS_AUDIO)]
    command = "python scripts/19_data_graphs.py --figures " + " ".join(wanted)

    run = start_run("data_graphs")
    run.set("figures", wanted)
    run.set("formats", list(args.formats))
    run.set("profile", args.profile)

    graphs = build_data_graphs(tuple(wanted), command=command)
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

    from src.reporting.graphs import read_registry, registry_path

    registry = {row["figure_id"]: row for row in read_registry(registry_path(args.out_dir))}
    print()
    print(f"{'ID':<5} {'#':>3}  {'Figure':<38} {'Rows':>7}  Files")
    print("-" * 92)
    for figure_id, paths in written.items():
        number = registry.get(figure_id, {}).get("figure_number", "?")
        title = registry.get(figure_id, {}).get("title", "")
        rows = len(next(g for g in graphs if g.spec.figure_id == figure_id).frame)
        names = ", ".join(sorted({p.suffix.lstrip(".") for p in paths.values()}))
        print(f"{figure_id:<5} {number:>3}  {title:<38} {rows:>7,}  {names}")
    print()
    print("registry: " + str(registry_path(args.out_dir)).replace("\\", "/"))

    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
