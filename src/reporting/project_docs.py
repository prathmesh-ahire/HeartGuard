"""Generated project documentation: the module map and the configuration reference.

T123.2 asks for the project structure and the role of each module; T123.3 asks
for "every configuration option and its default". Both are facts about files
that change, so both are **read out of the repository** rather than typed into
a markdown file that would start drifting the day after it was written.

Two documents come out of here, and ``scripts/48_project_docs.py`` writes them:

``ARCHITECTURE.md``
    Every package under ``src/`` with the role from its ``__init__`` docstring,
    every module inside it with its own summary line, every runnable script with
    the first line of its docstring, and the build-time codegen boundary stated
    once with the checks that enforce it named beside it.

``CONFIGURATION.md``
    Every leaf key in every ``configs/*.yaml``, its dotted path, its declared
    default, its type, and the comment the YAML carries for it.

Both live at the repository root because ``Docs/`` is gitignored (the user's
2026-08-23 decision) -- a reference a grader cannot open is not a deliverable.

## Why the YAML is parsed twice

``yaml.safe_load`` gives the values and throws the comments away, and the
comments are where this project keeps the *reason* for a setting -- the doubled
CirCor directory, the signed YAML 1.1 exponent, the pinned ``n_jobs``. So the
raw text is walked a second time for the comment attached to each key.

The values come from ``yaml.safe_load`` on the **raw file**, never from
``src.utils.config.load_config``: the loader resolves every path to an absolute
one and applies ``HEARTGUARD__*`` environment overrides. Both are correct at
runtime and both would put this machine's ``D:/Projects/HeartGuard`` into a
committed document. A test asserts neither document contains a drive letter.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging_setup import get_logger

__all__ = [
    "PROJECT_ROOT",
    "ConfigOption",
    "ModuleEntry",
    "PackageEntry",
    "architecture_markdown",
    "config_options",
    "configuration_markdown",
    "module_inventory",
    "script_inventory",
    "write_architecture",
    "write_configuration",
]

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _by_name(paths: Any) -> list[Path]:
    """Sort paths by name, the same way on every platform.

    `sorted(directory.iterdir())` compares `PurePath` objects, and on Windows
    that comparison is **case-folded** while on Linux it is case-sensitive. So
    `outputs/Q1_PAPER_ASSETS` sorts after `outputs/configs` here and before it
    on CI, and a document generated on one machine is "stale" on the other --
    which is exactly how CI went red on 50d6c88. Sorting by `name` is an
    ordinary string sort on both.
    """
    return sorted(paths, key=lambda item: item.name)

#: Packages are listed in pipeline order, not alphabetically -- the order a
#: recording travels through them. A package on disk that is missing from this
#: tuple is appended at the end and flagged by the test, so adding one is
#: noticed rather than silently sorted into the middle.
PACKAGE_ORDER: tuple[str, ...] = (
    "utils",
    "data_loader",
    "preprocessing",
    "feature_extraction",
    "feature_selection",
    "models",
    "ensemble",
    "optimization",
    "evaluation",
    "explainability",
    "reporting",
    "inference",
    "api",
    "pipeline",
)

#: What each ``outputs/`` directory holds. The directories themselves are found
#: on disk; this maps the ones we know to a description, and an unknown
#: directory is listed with an empty description rather than omitted.
OUTPUT_MAP: dict[str, str] = {
    "00_evidence_index": (
        "evidence index (CSV + XLSX), run manifest, QA and reproducibility reports"
    ),
    "01_dataset_audit": "DA-01..DA-09: integrity scan, label maps, class counts, split maps",
    "02_preprocessing": (
        "PP-01..PP-09: filter responses, segmentation, SQI calibration, the ablation"
    ),
    "03_features": (
        "FE-01..FE-12: the feature matrix, distributions, correlations, the selected subset"
    ),
    "04_models": "baseline smoke metrics, model complexity, search spaces, deployed task models",
    "05_search_optimization": (
        "SO-01..SO-06: random, Bayesian, GA/PSO, weight and multi-objective searches"
    ),
    "06_binary_results": "EXP-A1/EXP-A2 and the binary result tables (T08-T10)",
    "07_multiclass_results": "EXP-B1/EXP-B2 and the PASCAL tables (T11, T12)",
    "08_circor_external_validation": "EXP-C1/EXP-C2/EXP-D1 and the CirCor tables (T13-T15)",
    "09_ablation": "preprocessing, feature and optimization ablations",
    "09_robustness_analysis": "EXP-E1 noise sweep and EXP-E2 duration truncation",
    "10_robustness": "robustness tables and the cross-dataset transfer report",
    "11_complexity": "T24-T26: timing, memory and inference-cost analysis",
    "12_statistics": "T29-T33: Wilcoxon, Friedman, effect sizes, the significance matrix",
    "13_figures_diagrams": "all 35 G-figures and 20 F-diagrams under one figure registry",
    "14_algorithms": "ALG-01..ALG-20, exported from the code",
    "15_dashboard_screenshots": (
        "SS-01..SS-13, captured from the built dashboard behind the audit gate"
    ),
    "16_literature_review": "LIT-01 review and LIT-02 indicative comparison (objective 2)",
    "Q1_PAPER_ASSETS": "the Q1 paper pack, rebuilt from the sources it quotes",
    "THESIS_ASSETS": "the thesis pack, copies checked by newline-normalized digest",
    "configs": "the frozen environment: pip freeze, resolved configs, package versions",
    "logs": "rotating run logs (gitignored; the manifest is not)",
}

#: The codegen boundary, stated once. Each row is (what, where, what enforces
#: it). ``scripts/48_project_docs.py`` checks every path named here exists
#: before writing, so a renamed guard rail fails the doc build instead of
#: leaving a false claim in a committed file.
CODEGEN_BOUNDARY: tuple[tuple[str, str, str], ...] = (
    (
        "Numbers are formatted in Python, at build time",
        "scripts/17_export_frontend_data.py",
        "prebuild: runs last before `next build`, so the site cannot be built against stale data",
    ),
    (
        "Pages import from generated/ and nothing else",
        "frontend/lib/generated/",
        "scripts/16_check_no_hardcoded_metrics.py (prebuild)",
    ),
    (
        "No metric literal is typed into a page",
        "frontend/app/, frontend/components/",
        "scripts/16_check_no_hardcoded_metrics.py (prebuild)",
    ),
    (
        "No payload leaks into every route",
        "frontend/out/",
        "scripts/20_check_bundle_budget.py (postbuild)",
    ),
    (
        "Every value on screen re-formats from its source file",
        "frontend/out/",
        "scripts/45_audit_displayed_values.py (postbuild)",
    ),
    (
        "No screenshot of an unaudited page",
        "outputs/15_dashboard_screenshots/",
        "scripts/45_audit_displayed_values.py --gate",
    ),
    (
        "The only runtime call is live inference",
        "frontend/lib/api.ts",
        "`POST /predict` in `src/api/main.py`; nothing else crosses the wire",
    ),
)

#: Paths that :data:`CODEGEN_BOUNDARY` claims exist. Checked before the document
#: is written.
BOUNDARY_PATHS: tuple[str, ...] = (
    "scripts/16_check_no_hardcoded_metrics.py",
    "scripts/17_export_frontend_data.py",
    "scripts/20_check_bundle_budget.py",
    "scripts/45_audit_displayed_values.py",
    "frontend/lib/generated",
    "frontend/lib/api.ts",
    "src/api/main.py",
)


class DocsError(RuntimeError):
    """A document could not be written because a path it claims is not there."""


@dataclass(frozen=True)
class ModuleEntry:
    """One ``.py`` file under ``src/`` or ``scripts/``."""

    path: str
    summary: str


@dataclass(frozen=True)
class PackageEntry:
    """One package under ``src/``."""

    name: str
    role: str
    modules: tuple[ModuleEntry, ...]


@dataclass(frozen=True)
class ConfigOption:
    """One key in one ``configs/*.yaml``."""

    file: str
    key: str
    default: str
    type_name: str
    comment: str
    is_section: bool


def _summary(path: Path) -> str:
    """The first line of a file's module docstring, or ``""``.

    Parsed with :mod:`ast` rather than imported: importing 157 modules to read
    their docstrings would run every import side effect in the project, and
    several of them read ``outputs/``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    doc = ast.get_docstring(tree) or ""
    return doc.strip().split("\n", 1)[0].strip()


def module_inventory(root: Path | None = None) -> tuple[PackageEntry, ...]:
    """Every package under ``src/``, in :data:`PACKAGE_ORDER`, with its modules."""
    base = (root or PROJECT_ROOT) / "src"
    found = {p.name for p in base.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    ordered = [name for name in PACKAGE_ORDER if name in found]
    ordered += sorted(found - set(PACKAGE_ORDER))

    packages: list[PackageEntry] = []
    for name in ordered:
        package = base / name
        modules = tuple(
            ModuleEntry(path=f"src/{name}/{item.name}", summary=_summary(item))
            for item in _by_name(package.glob("*.py"))
            if item.name != "__init__.py"
        )
        packages.append(
            PackageEntry(name=name, role=_summary(package / "__init__.py"), modules=modules)
        )
    return tuple(packages)


def script_inventory(root: Path | None = None) -> tuple[ModuleEntry, ...]:
    """Every runnable entry point in ``scripts/``, sorted by its numeric prefix."""
    base = (root or PROJECT_ROOT) / "scripts"

    def sort_key(path: Path) -> tuple[int, str]:
        match = re.match(r"^(\d+)_", path.name)
        return (int(match.group(1)) if match else 999, path.name)

    return tuple(
        ModuleEntry(path=f"scripts/{item.name}", summary=_summary(item))
        for item in sorted(base.glob("*.py"), key=sort_key)
    )


_KEY = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][\w.-]*):(?P<rest>\s.*|)$")


def _inline_comment(rest: str) -> str:
    """The ``#`` comment on a key's own line, if the ``#`` is not inside a string."""
    in_single = in_double = False
    for index, char in enumerate(rest):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or rest[index - 1].isspace():
                return rest[index + 1 :].strip()
            continue
    return ""


def _leading_comment(lines: list[str], start: int, indent: int, limit: int = 4) -> str:
    """The comment block directly above a key, at the same indentation.

    Capped at ``limit`` lines: several keys in ``paths.yaml`` and ``models.yaml``
    sit under a twenty-line rationale that belongs in the YAML, not in a
    reference table. The cap keeps the lines nearest the key, which are the ones
    that describe it.
    """
    block: list[str] = []
    index = start - 1
    while index >= 0:
        line = lines[index]
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        if len(line) - len(line.lstrip()) != indent:
            break
        block.append(stripped.lstrip("#").strip())
        index -= 1
    block.reverse()
    if len(block) > limit:
        block = block[-limit:]
    return " ".join(part for part in block if part)


def _render_default(value: Any) -> str:
    """A one-cell rendering of a declared default."""
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (int, float, str)):
        text = str(value)
        return "`" + (text if len(text) <= 70 else text[:67] + "...") + "`"
    if isinstance(value, list):
        if not value or all(isinstance(item, (str, int, float, bool)) for item in value):
            text = ", ".join(str(item) for item in value)
            return "`[" + (text if len(text) <= 70 else text[:67] + "...") + "]`"
        return f"_{len(value)} entries_"
    if isinstance(value, dict):
        return f"_{len(value)} keys_"
    return "`" + str(value) + "`"


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "section"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _dig(data: Any, dotted: str) -> Any:
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def config_options(root: Path | None = None) -> tuple[ConfigOption, ...]:
    """Every key in every ``configs/*.yaml`` with its declared default.

    Keys nested inside a **list item** are not addressable by a dotted path and
    are therefore not emitted. Keys inside a **flow mapping** are a different
    case and are: ``A10: {comparison: "..."}`` is one line, so the line walk
    never sees ``comparison``, yet ``experiments.EXP-F2...A10.comparison`` is a
    real dotted key that ``cfg[...]`` resolves. :func:`_expand_flow` puts those
    back, directly under the parent they belong to. A test asserts every
    dotted-addressable path in every parsed tree has a row here.
    """
    base = (root or PROJECT_ROOT) / "configs"
    options: list[ConfigOption] = []

    for path in _by_name(base.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        lines = text.splitlines()
        stack: list[tuple[int, str]] = []

        for number, line in enumerate(lines):
            match = _KEY.match(line)
            if match is None:
                continue
            indent = len(match.group("indent"))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, match.group("key")))
            dotted = ".".join(part for _, part in stack)
            value = _dig(data, dotted)
            comment = _inline_comment(match.group("rest")) or _leading_comment(
                lines, number, indent
            )
            options.append(
                ConfigOption(
                    file=path.name,
                    key=dotted,
                    default=_render_default(value),
                    type_name=_type_name(value),
                    comment=comment,
                    is_section=isinstance(value, dict),
                )
            )
        options = _expand_flow(path.name, data, options)
    return tuple(options)


def _expand_flow(
    filename: str, data: dict[str, Any], options: list[ConfigOption]
) -> list[ConfigOption]:
    """Insert keys that live inside a flow mapping, under their parent.

    ``A10: {comparison: "x"}`` gives the line walk one key where the config
    loader resolves two. The children are inserted immediately after the parent
    row, in declaration order, and carry no comment -- a flow mapping has
    nowhere to put one.
    """
    present = {option.key for option in options if option.file == filename}
    others = [option for option in options if option.file != filename]
    mine = [option for option in options if option.file == filename]

    def children(prefix: str) -> list[ConfigOption]:
        node = _dig(data, prefix)
        if not isinstance(node, dict):
            return []
        rows: list[ConfigOption] = []
        for key, value in node.items():
            dotted = f"{prefix}.{key}"
            if dotted in present:
                continue
            present.add(dotted)
            rows.append(
                ConfigOption(
                    file=filename,
                    key=dotted,
                    default=_render_default(value),
                    type_name=_type_name(value),
                    comment="",
                    is_section=isinstance(value, dict),
                )
            )
            rows.extend(children(dotted))
        return rows

    expanded: list[ConfigOption] = []
    for option in mine:
        expanded.append(option)
        if option.is_section:
            expanded.extend(children(option.key))
    return others + expanded


def _escape(text: str) -> str:
    """Make a comment safe inside a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


_GENERATED = (
    "<!-- GENERATED by scripts/48_project_docs.py -- do not edit by hand. -->\n"
    "<!-- Every row below is read out of the repository at generation time. -->\n"
)


#: Deliberately NOT a date. These documents are compared byte for byte against
#: what the repository would produce right now (`--check`, and a test), so any
#: value that changes on its own makes them stale at midnight UTC and turns CI
#: red for no reason. Git already records when the file changed, and the
#: `GENERATED` header says what to re-run.
_PROVENANCE = "Generated by `python scripts/48_project_docs.py` -- do not edit by hand."


def _check_boundary_paths(base: Path) -> None:
    missing = [name for name in BOUNDARY_PATHS if not (base / name).exists()]
    if missing:
        raise DocsError(
            "ARCHITECTURE.md claims these exist and they do not: " + ", ".join(missing)
        )


def _file_blurb(name: str) -> str:
    return {
        "experiments.yaml": (
            "experiment declarations, cross-validation, search budgets and selection rules"
        ),
        "features.yaml": "the 138-feature registry's parameters, per family",
        "models.yaml": "the eight models, their search spaces, calibration and class weighting",
        "paths.yaml": "every path in the project, resolved against the repository root",
        "signal.yaml": "resampling, filtering, normalization, segmentation and signal quality",
    }.get(name, "configuration")


def architecture_markdown(root: Path | None = None) -> str:
    """``ARCHITECTURE.md`` (T123.2)."""
    base = root or PROJECT_ROOT
    _check_boundary_paths(base)
    packages = module_inventory(base)
    scripts = script_inventory(base)

    out: list[str] = [
        _GENERATED,
        "# PV-MEPCG / PulseVision -- project structure",
        "",
        _PROVENANCE,
        "",
        "Every configuration key and its default is in [CONFIGURATION.md](CONFIGURATION.md).",
        "The scope boundary, install steps and run instructions are in [README.md](README.md).",
        "",
        "## The shape of the thing",
        "",
        "```",
        "dataset/  read-only input  ->  src/data_loader",
        "                           ->  src/preprocessing       -> cache/signals",
        "                           ->  src/feature_extraction  -> cache/features",
        "                           ->  src/feature_selection",
        "                           ->  src/models, src/ensemble, src/optimization",
        "                           ->  src/evaluation          -> outputs/",
        "                           ->  src/reporting           -> outputs/ tables, figures, packs",
        "                           ->  scripts/17_export_frontend_data.py",
        "                                                       -> frontend/lib/generated/",
        "                           ->  frontend (static export) + src/api (live inference)",
        "```",
        "",
        "Nothing downstream reaches back. `src/reporting` reads `outputs/`, never the",
        "dataset; the frontend reads `frontend/lib/generated/`, never `outputs/`; and the",
        "browser reads neither.",
        "",
        "## The codegen boundary",
        "",
        "**The client never computes, and never declares, a metric.** This is the rule",
        "that matters more than any other in this repository. It is enforced by checks",
        "that run inside `npm run build` -- the build fails if any of them fails --",
        "rather than trusted:",
        "",
        "| What | Where | Enforced by |",
        "|---|---|---|",
    ]
    for what, where, enforced in CODEGEN_BOUNDARY:
        out.append(f"| {_escape(what)} | `{where}` | {_escape(enforced)} |")

    out += [
        "",
        "The rule exists because a parallel implementation of this same brief displayed a",
        "95.82% PhysioNet result its pipeline never produced, and a CirCor validation it",
        "never ran.",
        "",
        "## Packages",
        "",
        "Listed in the order a recording travels through them.",
        "",
        "| Package | Role | Modules |",
        "|---|---|---|",
    ]
    for package in packages:
        out.append(
            f"| [`src/{package.name}/`](src/{package.name}/) | {_escape(package.role)} "
            f"| {len(package.modules)} |"
        )

    for package in packages:
        out += [
            "",
            f"### `src/{package.name}/` -- {_escape(package.role)}",
            "",
            "| Module | What it does |",
            "|---|---|",
        ]
        for module in package.modules:
            name = module.path.rsplit("/", 1)[-1]
            out.append(f"| [`{name}`]({module.path}) | {_escape(module.summary)} |")

    out += [
        "",
        "## Entry points",
        "",
        "Every script is runnable as `python scripts/<name>.py`; all of them are also",
        "stages of `scripts/00_run_everything.py`, which is what reproduces the project.",
        "",
        "| Script | What it does |",
        "|---|---|",
    ]
    for script in scripts:
        name = script.path.rsplit("/", 1)[-1]
        out.append(f"| [`{name}`]({script.path}) | {_escape(script.summary)} |")

    out += [
        "",
        "## Output map",
        "",
        "| Directory | Holds |",
        "|---|---|",
    ]
    outputs_dir = base / "outputs"
    if outputs_dir.is_dir():
        for item in _by_name(outputs_dir.iterdir()):
            if not item.is_dir():
                continue
            out.append(f"| `outputs/{item.name}/` | {_escape(OUTPUT_MAP.get(item.name, ''))} |")

    out += [
        "",
        "Other top-level directories:",
        "",
        "| Directory | Holds |",
        "|---|---|",
        "| `configs/` | the five YAML files -- see [CONFIGURATION.md](CONFIGURATION.md) |",
        "| `models_saved/` | one deployed model per task plus its manifest (binaries gitignored, "
        "manifests committed) |",
        "| `cache/` | preprocessed signals and feature shards keyed by config digest -- "
        "regenerable, gitignored |",
        "| `frontend/` | the Next.js dashboard; `frontend/out/` is the committed static export |",
        "| `tests/` | the pytest suite, including the five MEGA TEST modules |",
        "| `dataset/` | read-only input, 1.3 GB, gitignored -- never written to |",
        "",
    ]
    return "\n".join(out)


def configuration_markdown(root: Path | None = None) -> str:
    """``CONFIGURATION.md`` (T123.3)."""
    options = config_options(root)
    files = sorted({option.file for option in options})

    out: list[str] = [
        _GENERATED,
        "# PV-MEPCG / PulseVision -- configuration reference",
        "",
        _PROVENANCE,
        "",
        "Every key in every file under `configs/`, with the default as **declared in the",
        "YAML** -- not as resolved at runtime. `paths.yaml` values are shown relative,",
        "because `src/utils/config.py` resolves them against the repository root at load",
        "time and printing this machine's absolute paths into a committed document would",
        "be wrong on every other machine.",
        "",
        "## How a value is read, and how to override one",
        "",
        "```python",
        "from src.utils.config import load_config",
        "cfg = load_config('signal')",
        "cfg['filter.low_hz']            # dotted access; a missing level names the full path",
        "cfg.get('filter.low_hz', 20)    # with a default",
        "```",
        "",
        "Any leaf can be overridden without editing a file, using",
        "`HEARTGUARD__<FILE>__<KEY__PATH>` (double underscores between levels):",
        "",
        "```powershell",
        '$env:HEARTGUARD__SIGNAL__RESAMPLE__TARGET_FS = "4000"   # PowerShell',
        "```",
        "",
        "```bash",
        "export HEARTGUARD__MODELS__GLOBAL__N_JOBS=4              # bash",
        "```",
        "",
        "Values are parsed as YAML scalars, so `4000` arrives as an `int` and `true` as a",
        "`bool`. **Every override that fires is recorded in the run manifest** -- an",
        "override that changes a result and leaves no trace would break research rule 5.",
        "",
        "Four things are validated at load and are errors, not warnings: the feature",
        "counts must still sum to 138, the filter passband must stay below Nyquist at the",
        "target sampling rate, band-power edges must stay inside the passband, and an",
        "unknown top-level key is rejected rather than silently ignored.",
        "",
        "> **Write scientific notation with a signed exponent.** PyYAML is YAML 1.1:",
        '> `1.0e3` parses as the *string* `"1.0e3"`, `1.0e+3` as a float.',
        "",
        "## Contents",
        "",
    ]
    for name in files:
        anchor = "configs" + name.replace(".", "")
        out.append(f"- [`configs/{name}`](#{anchor}) -- {_file_blurb(name)}")

    for name in files:
        out += [
            "",
            f"## `configs/{name}`",
            "",
            _file_blurb(name)[0].upper() + _file_blurb(name)[1:] + ".",
            "",
            "| Key | Default | Type | Notes |",
            "|---|---|---|---|",
        ]
        for option in options:
            if option.file != name:
                continue
            default = "" if option.is_section else option.default
            out.append(
                f"| `{option.key}` | {default} | {option.type_name} | {_escape(option.comment)} |"
            )

    out.append("")
    return "\n".join(out)


def write_architecture(path: Path | None = None, *, root: Path | None = None) -> Path:
    """Write ``ARCHITECTURE.md``."""
    base = root or PROJECT_ROOT
    target = Path(path) if path else base / "ARCHITECTURE.md"
    target.write_text(architecture_markdown(base), encoding="utf-8")
    LOGGER.info("wrote %s", target)
    return target


def write_configuration(path: Path | None = None, *, root: Path | None = None) -> Path:
    """Write ``CONFIGURATION.md``."""
    base = root or PROJECT_ROOT
    target = Path(path) if path else base / "CONFIGURATION.md"
    target.write_text(configuration_markdown(base), encoding="utf-8")
    LOGGER.info("wrote %s", target)
    return target
