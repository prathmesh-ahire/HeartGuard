"""Build-time codegen: `outputs/` in, `frontend/lib/generated/` out (Phase 109).

This module is **the correctness boundary of the dashboard**. Everything the
browser displays that is not live inference comes through here, formatted in
Python, and the frontend imports the result and nothing else. No page fetches a
CSV at runtime, no page rounds a number, and no page contains a metric literal.

The rule exists because a parallel implementation of this same brief displayed a
95.82% PhysioNet result its pipeline never produced, and a CirCor validation it
never ran. A number that reaches a screen without passing through a function
like this one cannot be traced back to a run, and "we would have noticed" is not
a control.

Two arrays per column, and the line between them
------------------------------------------------
Every numeric column is emitted twice:

``display``
    Pre-formatted **strings** under the T85.6 rounding rules -- 3 decimals for
    metrics, 1 for percentages, thousands-separated integers for counts, ``n/a``
    for a NaN. This is what a page renders. Anywhere a number is *stated* -- a
    stat tile, a table cell, a tooltip, a chart data label -- it comes from here.
``values``
    The same column as **numbers**, for chart geometry only. A bar needs a
    height and an axis needs a scale, and neither can be computed from a string.

The line is: `values` may position a mark; it may never be rendered as text.
That is not enforceable by grep, so it is stated here, restated in the emitted
TypeScript, and checked by the T119.3 displayed-value audit against the source
CSVs. `scripts/16_check_no_hardcoded_metrics.py` catches the other half -- a
literal typed into a page -- which grep *can* see.

What is deliberately not exported
----------------------------------
**Only CSV, JSON and YAML are read.** Everything the exporter needs is committed
(105 CSVs and 29 JSONs under `outputs/`), so it runs identically on CI and
locally. `*.parquet` and working state are gitignored and must never be read: a
build that succeeds locally and fails on CI because an input is absent is the
failure mode T110.5 exists to prevent.

**Large plotted frames are not inlined.** G05's waveform is 12,000 rows and
G09's wavelet decomposition 12,032; shipping them as JSON would be megabytes for
a figure the reader sees as a 300 dpi PNG anyway. Frames above
:data:`MAX_INLINE_ROWS` export their metadata with ``data_omitted`` set and the
row count stated. Nothing is silently truncated -- an omission the page cannot
see is indistinguishable from data that was never there.

**Results directories are opt-in.** ``06_binary_results`` and
``07_multiclass_results`` are excluded unless ``--include-results`` is passed.
On 2026-08-30 a second session was rewriting them, and `pandas.read_csv` on a
file mid-write returns a truncated frame rather than an error -- short row count,
real-looking numbers, nothing logged. The exclusion, and the reason, are recorded
in the manifest so a `generated/` without headline metrics is self-explaining.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.reporting.architecture import ensemble_payload, pipeline_payload
from src.reporting.plot_style import (
    DIVERGING_CMAP,
    DPI,
    FIGSIZE,
    OKABE_ITO,
    SEQUENTIAL_CMAP,
)
from src.reporting.tables import (
    PLACES,
    content_digest,
    format_value,
    infer_kind,
)
from src.utils.io import ensure_dir, save_json
from src.utils.logging_setup import get_logger

__all__ = [
    "GENERATED_FILES",
    "MAX_INLINE_ROWS",
    "MAX_INLINE_CELLS",
    "DEFAULT_SOURCE_DIRS",
    "RESULT_DIRS",
    "READABLE_SUFFIXES",
    "ExportResult",
    "generated_dir",
    "public_figures_dir",
    "export_all",
    "column_payload",
    "verify_strict_json",
    "theme_payload",
    "contrast_ratio",
    "palette_contrast",
    "ensemble_payload",
    "pipeline_payload",
]

log = get_logger("reporting.frontend_export")

#: Frames larger than either budget export metadata only. See the module
#: docstring. Two budgets rather than one because they catch different shapes:
#: G05's waveform is 12,000 rows by 3 columns, while G06's spectrogram is only
#: 514 rows but 47 columns wide -- 24,000 cells, and 1.3 MB of JSON on its own.
MAX_INLINE_ROWS = 2000
MAX_INLINE_CELLS = 20_000

#: `outputs/` subdirectories read by default, as `configs/paths.yaml` keys.
DEFAULT_SOURCE_DIRS: tuple[str, ...] = (
    "dataset_audit",
    "preprocessing",
    "features",
    "models",
    "search_optimization",
    "figures_diagrams",
)

#: Read only with ``--include-results``. See the module docstring.
RESULT_DIRS: tuple[str, ...] = ("binary_results", "multiclass_results")

#: The only suffixes the exporter will open. Never ``.parquet``.
READABLE_SUFFIXES: frozenset[str] = frozenset({".csv", ".json", ".yaml", ".yml"})

GENERATED_FILES: tuple[str, ...] = (
    "manifest.json",
    "theme.json",
    "evidence.json",
    "tables.json",
    "figures.json",
    "pipeline.json",
    "ensemble.json",
    "types.ts",
    "index.ts",
)

_TABLE_ID = re.compile(r"^T\d{2}$")
_FIGURE_ID = re.compile(r"^G\d{2}$")


@dataclass
class ExportResult:
    """What one export run produced, for the CLI and the gate."""

    generated: Path
    written: list[Path] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def _paths() -> Any:
    from src.utils.config import load_config

    return load_config("paths")


def generated_dir(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return Path(_paths().require("frontend.generated"))


def public_figures_dir(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return Path(_paths().require("frontend.root")) / "public" / "figures"


def _outputs_dir(key: str) -> Path:
    return Path(_paths().require("outputs." + key))


def _project_root() -> Path:
    return Path(_paths().require("project_root"))


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_project_root()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


# ---------------------------------------------------------------------------
# T109.2 / T109.3 -- formatting and NaN policy, both in Python
# ---------------------------------------------------------------------------


def _json_number(value: Any) -> float | int | None:
    """A number safe for strict JSON: NaN and +/-Inf become ``null`` (T109.3)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    if isinstance(value, bool):
        return None
    return number


def column_payload(name: str, series: Any, kind: str | None = None) -> dict[str, Any]:
    """One column as ``{kind, display, values}``.

    ``display`` is what a page renders; ``values`` is for chart geometry only,
    and is ``null`` for non-numeric columns rather than an empty list, so a
    caller cannot mistake "not a number" for "no data".
    """
    import pandas as pd

    resolved = kind or infer_kind(name, series)
    raw = list(series)
    display = [format_value(v, resolved) for v in raw]

    numeric: list[float | int | None] | None = None
    series_view = pd.Series(raw)
    # Booleans satisfy is_numeric_dtype, and T05's `matches_expected` column is
    # a bool. Shipping it as `values` produced an array of nulls sitting under a
    # text column -- which reads as "we had numbers and lost them" rather than
    # "this was never numeric". A bool is a category; it belongs in `display`.
    if pd.api.types.is_numeric_dtype(series_view) and not pd.api.types.is_bool_dtype(series_view):
        numeric = [_json_number(v) for v in raw]

    return {
        "name": name,
        "kind": resolved,
        "places": PLACES.get(resolved),
        "display": display,
        "values": numeric,
    }


def _frame_payload(frame: Any, kinds: dict[str, str] | None = None) -> dict[str, Any]:
    kinds = kinds or {}
    return {
        "n_rows": len(frame),
        "columns": [column_payload(str(c), frame[c], kinds.get(str(c))) for c in frame.columns],
    }


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _read_meta(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("unreadable meta: %s", path)
        return None


def _discover(directories: tuple[str, ...]) -> tuple[list[Path], list[Path]]:
    """``(table metas, figure metas)`` across the requested output directories."""
    tables: list[Path] = []
    figures: list[Path] = []
    for key in directories:
        directory = _outputs_dir(key)
        if not directory.is_dir():
            continue
        for meta in sorted(directory.glob("*.meta.json")):
            payload = _read_meta(meta)
            if payload is None:
                continue
            if _TABLE_ID.match(str(payload.get("table_id", ""))):
                tables.append(meta)
            elif _FIGURE_ID.match(str(payload.get("figure_id", ""))):
                figures.append(meta)
    return tables, figures


# ---------------------------------------------------------------------------
# tables and figures
# ---------------------------------------------------------------------------


def _export_table(meta_path: Path, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    meta = _read_meta(meta_path) or {}
    csv_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".csv"))
    frame = pd.read_csv(csv_path)
    payload = _frame_payload(frame, meta.get("column_kinds", {}))

    headers = {c["name"]: c["name"] for c in payload["columns"]}
    for column in payload["columns"]:
        column["header"] = headers.get(column["name"], column["name"])

    table_id = str(meta["table_id"])
    sources = [str(s["path"]) for s in meta.get("sources", []) if s.get("path")]
    for column in payload["columns"]:
        evidence.append(
            {
                "key": "tables." + table_id + "." + column["name"],
                "kind": "table_column",
                "artifact": table_id,
                "generated_from": _relative(csv_path),
                "generated_from_sha256": content_digest(csv_path)[0],
                "upstream_sources": sources,
            }
        )
    return {
        "id": table_id,
        "title": meta.get("title", table_id),
        "caption": meta.get("caption", ""),
        "exp_id": meta.get("exp_id", ""),
        "objective": meta.get("objective"),
        "dataset": meta.get("dataset"),
        "notes": meta.get("notes", []),
        "sources": sources,
        "source_csv": _relative(csv_path),
        "generated_utc": meta.get("generated_utc"),
        **payload,
    }


def _export_figure(
    meta_path: Path,
    evidence: list[dict[str, Any]],
    public_dir: Path,
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    import shutil

    import pandas as pd

    meta = _read_meta(meta_path) or {}
    figure_id = str(meta["figure_id"])
    csv_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".csv"))
    png_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".png"))

    frame = pd.read_csv(csv_path)
    cells = len(frame) * len(frame.columns)
    inlined = len(frame) <= MAX_INLINE_ROWS and cells <= MAX_INLINE_CELLS
    if inlined:
        payload: dict[str, Any] = _frame_payload(frame)
        payload["data_omitted"] = False
        payload["data_omitted_reason"] = None
    else:
        # Stated, never silent: an omission the page cannot see is
        # indistinguishable from data that was never there.
        limit = (
            str(MAX_INLINE_ROWS) + "-row"
            if len(frame) > MAX_INLINE_ROWS
            else str(MAX_INLINE_CELLS) + "-cell"
        )
        reason = (
            str(len(frame))
            + " rows x "
            + str(len(frame.columns))
            + " columns exceeds the "
            + limit
            + " inline budget; this figure is served as its canonical 300 dpi "
            "PNG and its full data is at " + _relative(csv_path)
        )
        payload = {
            "n_rows": len(frame),
            "columns": [],
            "data_omitted": True,
            "data_omitted_reason": reason,
        }
        skipped.append({"artifact": figure_id, "reason": reason})

    # T113.4 wants a "download the 300 dpi figure" action beside every chart,
    # serving the canonical matplotlib PNG. A static export can only serve what
    # is under public/, so the PNG is copied at build time rather than linked.
    public_png: str | None = None
    if png_path.is_file():
        ensure_dir(public_dir)
        shutil.copyfile(png_path, public_dir / png_path.name)
        public_png = "/figures/" + png_path.name

    evidence.append(
        {
            "key": "figures." + figure_id,
            "kind": "figure",
            "artifact": figure_id,
            "generated_from": _relative(csv_path),
            "generated_from_sha256": content_digest(csv_path)[0],
            "upstream_sources": [str(s["path"]) for s in meta.get("sources", []) if s.get("path")],
        }
    )
    return {
        "id": figure_id,
        "number": meta.get("figure_number"),
        "title": meta.get("title", figure_id),
        "caption": meta.get("caption", ""),
        "exp_id": meta.get("exp_id", ""),
        "objective": meta.get("objective"),
        "dataset": meta.get("dataset"),
        "notes": meta.get("notes", []),
        "sources": [str(s["path"]) for s in meta.get("sources", []) if s.get("path")],
        "source_csv": _relative(csv_path),
        "png": public_png,
        "dpi": meta.get("dpi"),
        **payload,
    }


# ---------------------------------------------------------------------------
# T111.1 -- the figure palette, exported rather than retyped
# ---------------------------------------------------------------------------


#: The two page grounds the dashboard renders on: Tailwind `white` and
#: `slate-950`, which `app/globals.css` sets on <body> in each theme.
LIGHT_GROUND = "#FFFFFF"
DARK_GROUND = "#020617"

#: WCAG 2.2 SC 1.4.11 (non-text contrast). A chart mark, a border or an icon
#: needs 3:1 against its background to be distinguishable. Text needs 4.5:1
#: (SC 1.4.3), which is checked separately for the UI chrome.
NON_TEXT_CONTRAST = 3.0


def _relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance of an ``#RRGGBB`` colour."""
    value = hex_colour.lstrip("#")
    channels = [int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio, 1.0 to 21.0."""
    a = _relative_luminance(foreground)
    b = _relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def palette_contrast() -> list[dict[str, Any]]:
    """Measured contrast for every series colour on both grounds (T111.5).

    Reported rather than asserted away. Okabe-Ito is designed for colourblind
    *discriminability* -- the eight hues stay apart under protanopia,
    deuteranopia and tritanopia -- which is a different property from luminance
    contrast against a page. The yellow in particular is nearly as bright as
    white, so a yellow mark on a white ground is real data the reader cannot
    see.

    The fix is not to drop the colour, which would break the fixed series order
    the figures depend on. It is to give marks a stroke on the ground where
    their fill is too close to it, and `needs_outline_on` is what tells the
    chart layer where.
    """
    rows: list[dict[str, Any]] = []
    for index, colour in enumerate(OKABE_ITO):
        on_light = contrast_ratio(colour, LIGHT_GROUND)
        on_dark = contrast_ratio(colour, DARK_GROUND)
        needs = []
        if on_light < NON_TEXT_CONTRAST:
            needs.append("light")
        if on_dark < NON_TEXT_CONTRAST:
            needs.append("dark")
        rows.append(
            {
                "index": index,
                "colour": colour,
                "on_light": round(on_light, 3),
                "on_dark": round(on_dark, 3),
                "needs_outline_on": needs,
            }
        )
    return rows


def theme_payload() -> dict[str, Any]:
    """The matplotlib style, as data the browser can read.

    T111.1 asks the dashboard's design tokens to match the figure palette "so
    charts and pages agree". The obvious way to do that is to copy eight hex
    strings into a TypeScript file, and the obvious way for it to go wrong is
    for somebody to change one of them in one place. So the palette is exported
    from ``src/reporting/plot_style.py`` -- the module every figure already
    draws through -- and the frontend imports it like any other generated value.

    Okabe & Ito's eight colours stay distinguishable under protanopia,
    deuteranopia and tritanopia (about one man in twelve has one of them), and
    the ORDER is meaningful: index 0 is the first series in every figure, so a
    bar chart in the browser and the 300 dpi PNG beside it colour the same
    series the same way.
    """
    return {
        "palette": {
            "name": "Okabe-Ito (Color Universal Design, 2008)",
            "colourblind_safe": True,
            "order_is_meaningful": True,
            "series": list(OKABE_ITO),
            "roles": {
                "series_1": OKABE_ITO[0],
                "series_2": OKABE_ITO[1],
                "series_3": OKABE_ITO[2],
                "normal": OKABE_ITO[0],
                "abnormal": OKABE_ITO[1],
            },
        },
        "contrast": {
            "standard": "WCAG 2.2 SC 1.4.11 (non-text), 3:1",
            "light_ground": LIGHT_GROUND,
            "dark_ground": DARK_GROUND,
            "threshold": NON_TEXT_CONTRAST,
            "series": palette_contrast(),
        },
        "colormaps": {
            "sequential": SEQUENTIAL_CMAP,
            "diverging": DIVERGING_CMAP,
        },
        "figure": {
            "dpi": DPI,
            "sizes_inches": {name: list(size) for name, size in FIGSIZE.items()},
        },
        "typography": {
            "family": "serif",
            "note": (
                "Figures are set in DejaVu Serif because the thesis and paper are "
                "serif documents. The dashboard uses its own UI face; only the "
                "colours are shared."
            ),
        },
    }


# ---------------------------------------------------------------------------
# T109.6 -- TypeScript declarations
# ---------------------------------------------------------------------------

_TYPES_TS = """// GENERATED by scripts/17_export_frontend_data.py -- do not edit.
//
// These declarations exist so a schema change breaks the BUILD rather than a
// page (T109.6). `index.ts` assigns the imported JSON to these types without a
// cast, so `tsc` structurally checks the emitted data against them.
//
// `display` is pre-formatted text under the thesis rounding rules and is what
// a page renders. `values` is numeric and exists ONLY so a chart can position a
// mark. Never render a `values` entry as text: the client neither computes nor
// declares a metric.

export type ColumnKind =
{kind_union};

export interface GeneratedColumn {{
  name: string;
  header?: string;
  kind: string;
  places: number | null;
  /** Pre-formatted for display. Render this. */
  display: string[];
  /** Numeric, for chart geometry only. Never rendered as text. */
  values: (number | null)[] | null;
}}

export interface GeneratedTable {{
  id: string;
  title: string;
  caption: string;
  exp_id: string;
  objective: string | null;
  dataset: string | null;
  notes: string[];
  sources: string[];
  source_csv: string;
  generated_utc: string | null;
  n_rows: number;
  columns: GeneratedColumn[];
}}

export interface GeneratedFigure {{
  id: string;
  number: number | null;
  title: string;
  caption: string;
  exp_id: string;
  objective: string | null;
  dataset: string | null;
  notes: string[];
  sources: string[];
  source_csv: string;
  /** Path under /public to the canonical 300 dpi matplotlib PNG. */
  png: string | null;
  dpi: number | null;
  n_rows: number;
  columns: GeneratedColumn[];
  /** True when the frame was too large to inline; the PNG carries the figure. */
  data_omitted: boolean;
  data_omitted_reason: string | null;
}}

export interface GeneratedTheme {{
  palette: {{
    name: string;
    colourblind_safe: boolean;
    order_is_meaningful: boolean;
    /** Okabe-Ito, in the fixed order every figure uses. Index 0 is series 1. */
    series: string[];
    roles: Record<string, string>;
  }};
  contrast: {{
    standard: string;
    light_ground: string;
    dark_ground: string;
    threshold: number;
    /** Measured WCAG ratio per series colour on each page ground. */
    series: {{
      index: number;
      colour: string;
      on_light: number;
      on_dark: number;
      /** Grounds where this fill is too close to the page and needs a stroke. */
      needs_outline_on: string[];
    }}[];
  }};
  colormaps: {{ sequential: string; diverging: string }};
  figure: {{ dpi: number; sizes_inches: Record<string, number[]> }};
  typography: {{ family: string; note: string }};
}}

export interface GeneratedSource {{
  path: string;
  sha256: string;
  digest_method: string;
  bytes: number;
}}

export interface GeneratedManifest {{
  framework: string;
  run_id: string;
  git_commit: string | null;
  git_branch: string | null;
  git_dirty: boolean | null;
  exported_utc: string;
  exporter: string;
  source_dirs: string[];
  excluded_dirs: {{ dir: string; reason: string }}[];
  n_tables: number;
  n_figures: number;
  sources: GeneratedSource[];
  omissions: {{ artifact: string; reason: string }}[];
}}

export interface GeneratedPipelineStep {{
  index: number;
  key: string;
  title: string;
  summary: string;
  /** Repository-relative module implementing this step. Verified at export. */
  module: string;
  /** `outputs/` directory holding this step's evidence. Verified at export. */
  evidence_dir: string;
  /** The research rule this step is where we keep, if it is one of them. */
  rule: string | null;
}}

export interface GeneratedPipeline {{
  n_steps: number;
  note: string;
  steps: GeneratedPipelineStep[];
}}

export interface GeneratedEnsembleMember {{
  model_id: string;
  name: string;
  short_name: string;
  /** For bar geometry only. Render `weight_display`. */
  weight: number;
  weight_display: string;
  weight_std: number;
  weight_std_display: string;
}}

export interface GeneratedEnsemble {{
  available: boolean;
  reason?: string;
  source: string;
  sha256?: string;
  digest_method?: string;
  experiment?: string;
  task?: string;
  objective?: string;
  selection_rule?: string;
  n_folds?: number;
  seed?: number | null;
  equal_weight?: number;
  equal_weight_display?: string;
  folds_identical_to_equal?: number;
  folds_identical_display?: string;
  constraint?: string;
  members: GeneratedEnsembleMember[];
  /** A fixed illustration. Not a prediction, not a result. */
  demonstration: {{
    note: string;
    inputs: {{
      model_id: string;
      short_name: string;
      probability: number;
      probability_display: string;
    }}[];
    vote: number;
    vote_display: string;
  }} | null;
  /** Why the bars look nearly equal. Render it beside them, not instead. */
  interpretation?: string;
}}

export interface GeneratedEvidenceEntry {{
  key: string;
  kind: string;
  artifact: string;
  generated_from: string;
  generated_from_sha256: string;
  upstream_sources: string[];
}}
"""

_INDEX_TS = """// GENERATED by scripts/17_export_frontend_data.py -- do not edit.
//
// The only module a page imports precomputed data from. Each JSON payload is
// assigned to its declared type WITHOUT a cast, so a shape change emitted by
// the exporter is a compile error here rather than a blank chart in the browser.

import ensembleJson from './ensemble.json';
import evidenceJson from './evidence.json';
import figuresJson from './figures.json';
import manifestJson from './manifest.json';
import pipelineJson from './pipeline.json';
import tablesJson from './tables.json';
import themeJson from './theme.json';

import type {
  GeneratedEnsemble,
  GeneratedEvidenceEntry,
  GeneratedFigure,
  GeneratedManifest,
  GeneratedPipeline,
  GeneratedTable,
  GeneratedTheme,
} from './types';

export const manifest: GeneratedManifest = manifestJson;
export const tables: Record<string, GeneratedTable> = tablesJson;
export const figures: Record<string, GeneratedFigure> = figuresJson;
export const evidence: GeneratedEvidenceEntry[] = evidenceJson;
/** The matplotlib palette, so a browser chart and its 300 dpi PNG agree. */
export const theme: GeneratedTheme = themeJson;
/** The twelve architecture steps, each verified against the repository. */
export const pipeline: GeneratedPipeline = pipelineJson;
/** SO-05's searched voting weights, or a stated absence. */
export const ensemble: GeneratedEnsemble = ensembleJson;

export type {
  ColumnKind,
  GeneratedColumn,
  GeneratedEnsemble,
  GeneratedEnsembleMember,
  GeneratedEvidenceEntry,
  GeneratedFigure,
  GeneratedManifest,
  GeneratedPipeline,
  GeneratedPipelineStep,
  GeneratedSource,
  GeneratedTable,
  GeneratedTheme,
} from './types';

/** One table by id, or undefined. Pages should handle undefined explicitly. */
export function table(id: string): GeneratedTable | undefined {
  return tables[id];
}

/** One figure by id, or undefined. */
export function figure(id: string): GeneratedFigure | undefined {
  return figures[id];
}
"""


def _write_typescript(target: Path) -> list[Path]:
    kinds = "\n".join("  | '" + kind + "'" for kind in sorted(PLACES)) + ";"
    types = target / "types.ts"
    types.write_text(_TYPES_TS.format(kind_union=kinds), encoding="utf-8")
    index = target / "index.ts"
    index.write_text(_INDEX_TS, encoding="utf-8")
    return [types, index]


# ---------------------------------------------------------------------------
# T109.3 -- strict JSON verification
# ---------------------------------------------------------------------------


def verify_strict_json(path: Path) -> dict[str, Any]:
    """Parse strictly and refuse the tokens ``json.dumps`` emits for NaN/Inf.

    ``json.loads`` ACCEPTS ``NaN``, ``Infinity`` and ``-Infinity`` by default --
    Python's parser is more permissive than the JSON spec and more permissive
    than ``JSON.parse`` in a browser. Parsing alone would therefore pass a file
    the frontend cannot read, so the raw text is scanned as well.
    """
    text = path.read_text(encoding="utf-8")
    json.loads(text, parse_constant=_reject_constant)
    for token in ("NaN", "Infinity", "-Infinity"):
        if re.search(r"(?<![\"\w])" + re.escape(token) + r"(?![\"\w])", text):
            raise ValueError(
                str(path) + " contains the bare token " + token + ", which is not "
                "valid JSON and which JSON.parse rejects in the browser"
            )
    return {"path": path, "bytes": len(text.encode("utf-8"))}


def _reject_constant(name: str) -> Any:
    raise ValueError("non-finite JSON constant: " + name)


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------


def export_all(
    *,
    out_dir: str | Path | None = None,
    public_dir: str | Path | None = None,
    include_results: bool = False,
    source_dirs: tuple[str, ...] = DEFAULT_SOURCE_DIRS,
    command: str = "",
) -> ExportResult:
    """Read `outputs/`, emit `frontend/lib/generated/`. The whole boundary."""
    from src.utils.run_manifest import current_run, git_info

    target = ensure_dir(generated_dir(out_dir))
    figures_public = public_figures_dir(public_dir)

    directories = tuple(source_dirs)
    excluded = [
        {
            "dir": key,
            "reason": (
                "results directories are opt-in: a CSV being rewritten by another "
                "process reads as a valid CSV with a truncated row count, and no "
                "gate would see it. Re-run with --include-results once the "
                "experiments are settled."
            ),
        }
        for key in RESULT_DIRS
        if not include_results
    ]
    if include_results:
        directories = directories + RESULT_DIRS

    evidence: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    table_metas, figure_metas = _discover(directories)

    tables = {}
    for meta in table_metas:
        payload = _export_table(meta, evidence)
        tables[payload["id"]] = payload
    figures = {}
    for meta in figure_metas:
        payload = _export_figure(meta, evidence, figures_public, skipped)
        figures[payload["id"]] = payload

    if not tables and not figures:
        raise ValueError(
            "nothing to export: no T##/G## .meta.json found under "
            + ", ".join(directories)
            + ". Run scripts/18_setup_tables.py and scripts/19_data_graphs.py first."
        )

    sources: dict[str, dict[str, Any]] = {}
    for payload in list(tables.values()) + list(figures.values()):
        for source in [*payload["sources"], payload["source_csv"]]:
            candidate = _project_root() / source
            if source in sources or not candidate.is_file():
                continue
            if candidate.suffix.lower() not in READABLE_SUFFIXES:
                continue
            digest, method = content_digest(candidate)
            sources[source] = {
                "path": source,
                "sha256": digest,
                "digest_method": method,
                "bytes": candidate.stat().st_size,
            }

    run = current_run()
    git = git_info()
    manifest = {
        "framework": "PV-MEPCG / PulseVision",
        "run_id": run.run_id if run is not None else "",
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "exported_utc": datetime.now(UTC).isoformat(),
        "exporter": command or "scripts/17_export_frontend_data.py",
        "source_dirs": list(directories),
        "excluded_dirs": excluded,
        "n_tables": len(tables),
        "n_figures": len(figures),
        "sources": sorted(sources.values(), key=lambda s: str(s["path"])),
        "omissions": skipped,
    }

    written = [
        save_json(manifest, target / "manifest.json"),
        save_json(theme_payload(), target / "theme.json"),
        save_json(evidence, target / "evidence.json"),
        save_json(tables, target / "tables.json"),
        save_json(figures, target / "figures.json"),
        # T112.3/T112.4: the twelve architecture steps, verified against the
        # repository at export time, and SO-05's searched voting weights.
        save_json(pipeline_payload(), target / "pipeline.json"),
        save_json(ensemble_payload(), target / "ensemble.json"),
    ]
    written += _write_typescript(target)

    for path in written:
        if path.suffix == ".json":
            verify_strict_json(path)

    log.info(
        "exported %d tables and %d figures to %s",
        len(tables),
        len(figures),
        target,
    )
    return ExportResult(
        generated=target,
        written=written,
        tables=tables,
        figures=figures,
        manifest=manifest,
        evidence=evidence,
        skipped=skipped,
    )
