"""Regenerate every PV-MEPCG diagram from version-controlled source (T97.6).

One command, no manual step: no Graphviz binary, no headless browser, no hand
edit in Inkscape between the source and the deliverable. Each diagram is a
Python function in ``src/reporting/diagrams/``; this script draws all of them to
SVG and 300 dpi PNG, writes a provenance ``.meta.json`` beside each, and keeps
``diagram_registry.csv`` in step.

The catalogue is a closed set of twenty. ``--expect 20`` makes a missing diagram
an error instead of a shorter table, which is the check T97.7 actually needs:
rendering nineteen and exiting zero is exactly the failure mode this project
treats as fabrication.

Usage
-----
    python scripts/05_render_diagrams.py
    python scripts/05_render_diagrams.py --only F01 F16
    python scripts/05_render_diagrams.py --expect 20
    python scripts/05_render_diagrams.py --list
    python scripts/05_render_diagrams.py --specimen build/specimen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/05_render_diagrams.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.diagrams.catalogue import (
    CATALOGUE,
    EXPECTED_COUNT,
    catalogue_report,
    load_builders,
    spec_for,
)
from src.reporting.diagrams.render import FORMATS, registry_path, write_diagram
from src.utils.logging_setup import get_logger

log = get_logger("render_diagrams")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="05_render_diagrams",
        description="Render F01-F20 to SVG and 300 dpi PNG from version-controlled source.",
    )
    parser.add_argument("--only", nargs="+", default=None, metavar="ID")
    parser.add_argument(
        "--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS), metavar="FMT"
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="fail unless exactly this many diagrams render (use 20 for a full run)",
    )
    parser.add_argument("--list", action="store_true", help="show the catalogue and exit")
    parser.add_argument(
        "--specimen",
        default=None,
        metavar="DIR",
        help="render the visual-language specimen sheet into DIR and exit",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--evidence-index", default=None)
    parser.add_argument("--no-stamp", action="store_true")
    return parser.parse_args(argv)


def _print_catalogue() -> None:
    report = catalogue_report()
    builders = load_builders()
    print()
    print(f"{'ID':<5} {'#':>3}  {'Title':<46} {'Task':<7} Builder")
    print("-" * 96)
    for spec in CATALOGUE:
        builder = builders.get(spec.diagram_id)
        where = getattr(builder, "__name__", "-- pending --") if builder else "-- pending --"
        print(
            f"{spec.diagram_id:<5} {spec.diagram_id[1:]:>3}  "
            f"{spec.title[:46]:<46} {spec.task:<7} {where}"
        )
    print()
    print(
        f"{len(report['implemented'])} of {report['expected']} implemented"
        + ("" if report["complete"] else "; pending: " + ", ".join(report["pending"]))
    )


def main(argv: list[str] | None = None) -> int:
    from src.utils.run_manifest import start_run

    args = parse_args(argv)

    if args.list:
        _print_catalogue()
        return 0

    if args.specimen is not None:
        from src.reporting.diagrams.specimen import SPECIMEN_SPEC, build_specimen

        specimen_paths = write_diagram(
            SPECIMEN_SPEC,
            build_specimen,
            args.specimen,
            formats=tuple(args.formats),
            registry=Path(args.specimen) / "diagram_registry.csv",
            evidence_index=Path(args.specimen) / "evidence_index.csv",
            stamp=not args.no_stamp,
        )
        for path in specimen_paths.values():
            print(str(path).replace("\\", "/"))
        return 0

    builders = load_builders()
    wanted = [spec.diagram_id for spec in CATALOGUE]
    if args.only:
        unknown = [d for d in args.only if d not in wanted]
        if unknown:
            print("unknown diagram id(s): " + ", ".join(unknown), file=sys.stderr)
            return 2
        wanted = [d for d in wanted if d in set(args.only)]

    missing = [d for d in wanted if d not in builders]
    renderable = [d for d in wanted if d in builders]

    run = start_run("render_diagrams")
    run.set("diagrams", renderable)
    run.set("pending", missing)
    run.set("formats", list(args.formats))

    written: dict[str, dict[str, Path]] = {}
    for diagram_id in renderable:
        written[diagram_id] = write_diagram(
            spec_for(diagram_id),
            builders[diagram_id],
            args.out_dir,
            formats=tuple(args.formats),
            evidence_index=args.evidence_index,
            stamp=not args.no_stamp,
        )
        for path in written[diagram_id].values():
            run.record_artifact(path)

    print()
    print(f"{'ID':<5} {'#':>3}  {'Title':<46} {'pt':>5}  Files")
    print("-" * 92)
    for diagram_id, paths in written.items():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        formats = ", ".join(sorted({p.suffix.lstrip(".") for p in paths.values()}))
        print(
            f"{diagram_id:<5} {meta['diagram_number']:>3}  {meta['title'][:46]:<46} "
            f"{meta['legibility']['min_effective_pt']:>5}  {formats}"
        )
    print()
    print(f"rendered {len(written)} of {EXPECTED_COUNT} declared diagrams")
    if missing:
        print("pending (no builder yet): " + ", ".join(missing))
    print("registry: " + str(registry_path(args.out_dir)).replace("\\", "/"))

    if args.expect is not None and len(written) != args.expect:
        print(
            "FAILED: expected " + str(args.expect) + " diagrams, rendered " + str(len(written)),
            file=sys.stderr,
        )
        run.finish(status="failed")
        return 1

    run.finish(status="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
