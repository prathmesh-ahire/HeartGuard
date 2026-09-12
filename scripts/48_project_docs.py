"""Regenerate ARCHITECTURE.md and CONFIGURATION.md from the repository (T123.2, T123.3).

Both documents are read out of the code and the YAML at generation time -- the
module map from every ``__init__`` and module docstring, the configuration
reference from every key in ``configs/*.yaml`` -- so neither can drift from what
it describes without a regeneration failing or a test noticing.

Usage
-----
    python scripts/48_project_docs.py            # write both
    python scripts/48_project_docs.py --check    # exit 1 if either is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/48_project_docs.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="48_project_docs",
        description="Generate the module map and the configuration reference.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if either document differs from what the repo would produce",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.reporting.project_docs import (
        PROJECT_ROOT,
        DocsError,
        architecture_markdown,
        configuration_markdown,
    )
    from src.utils.evidence import register_evidence

    try:
        wanted = {
            PROJECT_ROOT / "ARCHITECTURE.md": architecture_markdown(),
            PROJECT_ROOT / "CONFIGURATION.md": configuration_markdown(),
        }
    except DocsError as error:
        print("FAILED: " + str(error))
        return 1

    if args.check:
        stale = [
            path.name
            for path, text in wanted.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        if stale:
            print("stale, regenerate with `python scripts/48_project_docs.py`: " + ", ".join(stale))
            return 1
        print("ARCHITECTURE.md and CONFIGURATION.md are current")
        return 0

    for path, text in wanted.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.name} ({len(text.splitlines())} lines)")

    register_evidence(
        "DOC-ARCH",
        PROJECT_ROOT / "ARCHITECTURE.md",
        metric_or_asset="project structure and the role of each module (T123.2)",
        command="python scripts/48_project_docs.py",
    )
    register_evidence(
        "DOC-CONFIG",
        PROJECT_ROOT / "CONFIGURATION.md",
        metric_or_asset="every configuration option and its default (T123.3)",
        source_data=PROJECT_ROOT / "configs",
        command="python scripts/48_project_docs.py",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
