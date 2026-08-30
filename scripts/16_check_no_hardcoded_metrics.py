"""Fail the build on any hand-typed metric in the frontend (the guard rail).

The rule this enforces is the one that matters more than any other in this
project: **the client never computes, and never declares, a metric.** Every
precomputed number reaches the browser through
`scripts/17_export_frontend_data.py` -> `frontend/lib/generated/`, formatted in
Python. Pages import from `generated/` and nothing else.

It exists because a parallel implementation of this same brief displayed a
95.82% PhysioNet result its pipeline never produced, and a CirCor validation it
never ran. Neither number was a lie anyone told on purpose; both were literals
someone typed into a page while the real pipeline was still being written, and
nothing ever removed them.

`todo.md` and CLAUDE.md name this `scripts/07_check_no_hardcoded_metrics.py`.
The `07` slot was taken by `07_run_population_search.py` in Phase 58, so it
lands as `16_`, and `.github/workflows/ci.yml` invokes it under that name.

What it flags
-------------
Three patterns, chosen because each one is a way the 95.82% bug actually
happens rather than a general dislike of numbers:

1. **A metric-named key assigned a literal** -- ``accuracy: 0.9``,
   ``sensitivity = .86``, ``rocAuc={0.91}``. The highest-signal rule: it catches
   a rounded value that rules 2 and 3 would miss.
2. **A literal with three or more decimal places** -- ``0.8588``. Three is the
   thesis rounding rule for metrics (T85.6), so a number written to that
   precision in a page is a metric by construction. Ordinary UI numbers
   (opacity 0.8, a 0.25s delay, 1.5rem) do not reach three places.
3. **A percentage literal in text** -- ``95.82%``, ``86%``. How a metric is
   usually written when it is written by hand.

What it does not flag
---------------------
Whole numbers and one- or two-place decimals outside a metric-named key: those
are durations, opacities, breakpoints, spring constants and z-indices, and
flagging them would make the guard something people route around. The gap is
covered from the other side by T119.3's displayed-value audit, which compares
what a page *renders* against the source CSV.

An escape hatch exists -- a ``metric-guard: allow`` comment on the line or the
one above -- and every use is REPORTED even when the run passes, so a
suppression cannot be quiet. There should be zero.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/16_check_no_hardcoded_metrics.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Directories scanned, relative to the frontend root.
SCANNED = ("app", "components")

SOURCE_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs"})

ALLOW_MARKER = "metric-guard: allow"

METRIC_WORDS = (
    "accuracy",
    "balanced_?accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "fbeta",
    "auc",
    "auroc",
    "auprc",
    "roc_?auc",
    "pr_?auc",
    "mcc",
    "brier",
    "ece",
    "kappa",
    "macro_?f1",
    "imbalance_?ratio",
)

#: 1 -- a metric-named key or variable assigned a numeric literal.
METRIC_ASSIGNMENT = re.compile(
    r"\b(" + "|".join(METRIC_WORDS) + r")\b\s*[:=]\s*\{?\s*-?\d*\.?\d+",
    re.IGNORECASE,
)

#: 2 -- a literal at metric precision (T85.6 rounds metrics to 3 places).
PRECISE_LITERAL = re.compile(r"(?<![\w.])-?\d*\.\d{3,}(?![\w.])")

#: 3 -- a percentage written out in text.
PERCENT_LITERAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*%")

#: Lines that legitimately carry numbers which look like the above.
_EXEMPT_LINE = re.compile(
    r"^\s*(?:import\b|export\s+\*|//|/\*|\*)"  # imports and comments
    r"|\bfrom\s+['\"]"  # module specifiers
    r"|\bversion['\"]?\s*[:=]"  # package versions
)

#: Percentages inside a CSS-ish context are layout, not results.
_CSS_PERCENT = re.compile(
    r"(?:width|height|top|left|right|bottom|translate[XY]?|inset|basis|flex|size|"
    r"stop-?color|offset|scale|gradient|calc)\s*[:(=]|"
    r"\b(?:w|h|top|left|right|bottom|translate-[xy])-\[\d",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line_number: int
    rule: str
    text: str

    def render(self, root: Path) -> str:
        try:
            shown = self.path.relative_to(root).as_posix()
        except ValueError:
            shown = self.path.as_posix()
        return (
            shown
            + ":"
            + str(self.line_number)
            + "  ["
            + self.rule
            + "]  "
            + self.text.strip()[:120]
        )


def frontend_root(override: str | None = None) -> Path:
    if override:
        return Path(override)
    try:
        from src.utils.config import load_config

        return Path(load_config("paths").require("frontend.root"))
    except Exception:  # noqa: BLE001 - the guard must run without the config too
        return Path(__file__).resolve().parents[1] / "frontend"


def source_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for name in SCANNED:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                found.append(path)
    return found


def _allowed(lines: list[str], index: int) -> bool:
    if ALLOW_MARKER in lines[index]:
        return True
    return index > 0 and ALLOW_MARKER in lines[index - 1]


def scan_file(path: Path) -> tuple[list[Finding], list[Finding]]:
    """``(findings, suppressed)`` for one source file."""
    findings: list[Finding] = []
    suppressed: list[Finding] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for index, line in enumerate(lines):
        if _EXEMPT_LINE.search(line):
            continue
        hits: list[str] = []
        if METRIC_ASSIGNMENT.search(line):
            hits.append("metric-named key assigned a literal")
        if PRECISE_LITERAL.search(line):
            hits.append("literal at metric precision (>=3 decimals)")
        if PERCENT_LITERAL.search(line) and not _CSS_PERCENT.search(line):
            hits.append("percentage literal in text")
        for rule in hits:
            finding = Finding(path, index + 1, rule, line)
            (suppressed if _allowed(lines, index) else findings).append(finding)
    return findings, suppressed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="16_check_no_hardcoded_metrics",
        description="Fail the build on a hand-typed metric in frontend/app or components.",
    )
    parser.add_argument("--frontend", default=None, help="frontend root to scan")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="exit 0 when there is nothing to scan (the default; kept explicit)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = frontend_root(args.frontend)

    files = source_files(root)
    if not files:
        print(
            "metric guard: nothing to scan under "
            + root.as_posix()
            + "/{"
            + ",".join(SCANNED)
            + "} -- the pages do not exist yet."
        )
        return 0

    findings: list[Finding] = []
    suppressed: list[Finding] = []
    for path in files:
        found, allowed = scan_file(path)
        findings.extend(found)
        suppressed.extend(allowed)

    print("metric guard: scanned " + str(len(files)) + " source files under " + root.as_posix())

    if suppressed:
        print()
        print(
            "SUPPRESSED (" + str(len(suppressed)) + ") -- every one of these is a "
            "hand-typed number somebody decided was acceptable. There should be none:"
        )
        for finding in suppressed:
            print("  " + finding.render(root))

    if findings:
        print()
        print("HARD-CODED METRICS (" + str(len(findings)) + "):")
        for finding in findings:
            print("  " + finding.render(root))
        print()
        print(
            "The client never computes and never declares a metric. Move the value "
            "into scripts/17_export_frontend_data.py so it is formatted in Python, "
            "and import it from frontend/lib/generated/."
        )
        return 1

    print("clean: no hand-typed metric literals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
