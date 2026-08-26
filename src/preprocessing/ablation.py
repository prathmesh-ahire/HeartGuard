"""Preprocessing settings table and ablation grid (Phase 29, T29.1-T29.2).

Two deliverables live here, and one deliberate gap.

**PP-07** (:func:`write_settings`) is the preprocessing configuration as a table:
every setting that shapes a signal, its value, its unit and the config key it
came from. Phase 86 renders thesis table T04 straight out of it, so it is
generated from ``configs/signal.yaml`` at run time rather than transcribed --
a hand-typed settings table is a hand-typed number under rule 1, and it goes
stale the first time a cutoff changes.

**The ablation grid** (:data:`ABLATION_GRID`) is the 2x2 T29.2 asks for: the
bandpass on or off, crossed with normalization on or off. Each arm materializes
a real ``Config`` and therefore a real cache directory, so the four arms coexist
on disk instead of overwriting one another (see ``pipeline.config_hash``).

**The gap: PP-09 is not produced here.** T29.3 asks for features and a trained
model under each arm, and neither exists yet -- feature extraction is Phases
31-42 and the fold-safe pipeline and baseline models are Phases 43-46. What this
module does now is define the four arms, prove they actually differ (a 2x2 whose
cells produce identical signals would be four copies of one result), and leave
the metric column to be filled in once there is a model to measure it with. The
gap is recorded in ``outputs/missing_outputs_report.txt``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "ABLATION_GRID",
    "SETTINGS_COLUMNS",
    "GRID_COLUMNS",
    "AblationArm",
    "arm_config",
    "arm_rows",
    "settings_rows",
    "settings_path",
    "grid_path",
    "write_settings",
    "write_grid",
]

log = get_logger(__name__)

SETTINGS_COLUMNS: tuple[str, ...] = (
    "stage",
    "setting",
    "value",
    "unit",
    "config_key",
    "notes",
)

GRID_COLUMNS: tuple[str, ...] = (
    "arm_id",
    "label",
    "filter_enabled",
    "normalization_enabled",
    "config_hash",
    "cache_dir",
    "is_shipped_configuration",
)


@dataclass(frozen=True, slots=True)
class AblationArm:
    """One cell of the 2x2 preprocessing ablation (T29.2)."""

    arm_id: str
    filter_enabled: bool
    normalization_enabled: bool

    @property
    def label(self) -> str:
        return (
            ("filter on" if self.filter_enabled else "filter off")
            + ", "
            + ("normalization on" if self.normalization_enabled else "normalization off")
        )


# PP-A is the shipped configuration; the other three are the comparisons that
# justify it. Ordered so the table reads as a 2x2.
ABLATION_GRID: tuple[AblationArm, ...] = (
    AblationArm("PP-A", filter_enabled=True, normalization_enabled=True),
    AblationArm("PP-B", filter_enabled=True, normalization_enabled=False),
    AblationArm("PP-C", filter_enabled=False, normalization_enabled=True),
    AblationArm("PP-D", filter_enabled=False, normalization_enabled=False),
)


# ---------------------------------------------------------------------------
# T29.1 -- PP-07
# ---------------------------------------------------------------------------


def _signal_config(cfg: Any | None = None) -> Any:
    if cfg is not None:
        return cfg
    from src.utils.config import load_config

    return load_config("signal")


def settings_rows(cfg: Any | None = None) -> list[dict[str, Any]]:
    """Every preprocessing setting as a row, read from the live config.

    Setting names are unique across the whole table, not merely within a stage.
    Phase 86 renders this as thesis table T04, where a reader meets the rows
    without the surrounding code: two rows both called "method" (one resampling,
    one normalization) and two called "enabled" are ambiguous on the page, and
    the ambiguity is invisible while the ``stage`` column is in front of you.
    """
    signal_cfg = _signal_config(cfg)

    def value(key: str) -> Any:
        return signal_cfg.require(key)

    order = int(value("filter.order"))
    rows: list[dict[str, Any]] = [
        {
            "stage": "resample",
            "setting": "target sampling rate",
            "value": value("resample.target_fs"),
            "unit": "Hz",
            "config_key": "resample.target_fs",
            "notes": "all four datasets are brought to this rate",
        },
        {
            "stage": "resample",
            "setting": "resampling method",
            "value": value("resample.method"),
            "unit": "",
            "config_key": "resample.method",
            "notes": "band-limited rational resampler; naive decimation aliases at 44.1 kHz",
        },
        {
            "stage": "resample",
            "setting": "native rates encountered",
            "value": "2000; 4000; 44100",
            "unit": "Hz",
            "config_key": "resample.native_fs",
            "notes": "D1 2 kHz (no-op), D3/D4 4 kHz, D2 44.1 kHz",
        },
        {
            "stage": "resample",
            "setting": "channel handling",
            "value": value("resample.mono"),
            "unit": "",
            "config_key": "resample.mono",
            "notes": "all 7,536 corpus files are already mono; no-op fast path",
        },
        {
            "stage": "filter",
            "setting": "type",
            "value": value("filter.type"),
            "unit": "",
            "config_key": "filter.type",
            "notes": "designed in second-order sections for numerical stability",
        },
        {
            "stage": "filter",
            "setting": "design order",
            "value": order,
            "unit": "",
            "config_key": "filter.order",
            "notes": "prototype order",
        },
        {
            "stage": "filter",
            "setting": "effective order applied",
            "value": 4 * order,
            "unit": "",
            "config_key": "filter.order + filter.zero_phase",
            "notes": (
                "bandpass transform doubles the prototype to "
                + str(2 * order)
                + " poles; sosfiltfilt doubles it again"
            ),
        },
        {
            "stage": "filter",
            "setting": "low cutoff",
            "value": value("filter.low_hz"),
            "unit": "Hz",
            "config_key": "filter.low_hz",
            "notes": "-6.02 dB here, not -3.01, because the filter runs forwards and backwards",
        },
        {
            "stage": "filter",
            "setting": "high cutoff",
            "value": value("filter.high_hz"),
            "unit": "Hz",
            "config_key": "filter.high_hz",
            "notes": "-6.02 dB here; well below the 1000 Hz Nyquist at the target rate",
        },
        {
            "stage": "filter",
            "setting": "phase",
            "value": "zero" if value("filter.zero_phase") else "forward only",
            "unit": "",
            "config_key": "filter.zero_phase",
            "notes": "zero group delay; S1/S2 timings survive filtering",
        },
        {
            "stage": "filter",
            "setting": "filter enabled",
            "value": value("filter.enabled"),
            "unit": "",
            "config_key": "filter.enabled",
            "notes": "ablation arms PP-C and PP-D set this false",
        },
        {
            "stage": "normalization",
            "setting": "normalization method",
            "value": value("normalization.method"),
            "unit": "",
            "config_key": "normalization.method",
            "notes": "per record; no statistic is shared between records",
        },
        {
            "stage": "normalization",
            "setting": "DC offset removed",
            "value": value("normalization.remove_dc"),
            "unit": "",
            "config_key": "normalization.remove_dc",
            "notes": "applied before normalization",
        },
        {
            "stage": "normalization",
            "setting": "zero-variance policy",
            "value": value("normalization.zero_variance_policy"),
            "unit": "",
            "config_key": "normalization.zero_variance_policy",
            "notes": "a constant record passes through unchanged and is flagged",
        },
        {
            "stage": "normalization",
            "setting": "normalization enabled",
            "value": value("normalization.enabled"),
            "unit": "",
            "config_key": "normalization.enabled",
            "notes": "ablation arms PP-B and PP-D set this false",
        },
        {
            "stage": "framing",
            "setting": "frame length",
            "value": value("framing.frame_length"),
            "unit": "samples",
            "config_key": "framing.frame_length",
            "notes": "256 ms at 2 kHz; shared by quality analysis and the spectral features",
        },
        {
            "stage": "framing",
            "setting": "hop length",
            "value": value("framing.hop_length"),
            "unit": "samples",
            "config_key": "framing.hop_length",
            "notes": "128 ms, 50% overlap",
        },
        {
            "stage": "framing",
            "setting": "window",
            "value": value("framing.window"),
            "unit": "",
            "config_key": "framing.window",
            "notes": "",
        },
    ]
    return rows


def _preprocessing_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config

    if out_dir is not None:
        return ensure_dir(out_dir)
    return ensure_dir(load_config("paths").require("outputs.preprocessing"))


def settings_path(out_dir: str | Path | None = None) -> Path:
    """PP-07 ``preprocessing_settings.csv``."""
    return _preprocessing_dir(out_dir) / "preprocessing_settings.csv"


def write_settings(out_dir: str | Path | None = None, cfg: Any | None = None) -> Path:
    """Emit PP-07 from the live configuration (T29.1)."""
    import pandas as pd

    table = pd.DataFrame(settings_rows(cfg), columns=list(SETTINGS_COLUMNS))
    path = save_csv(table, settings_path(out_dir))
    log.info("PP-07: %d settings -> %s", len(table), path.name)
    return path


# ---------------------------------------------------------------------------
# T29.2 -- the ablation grid
# ---------------------------------------------------------------------------


def arm_config(arm: AblationArm, cfg: Any | None = None) -> Any:
    """A ``Config`` for one arm: the shipped settings with two switches flipped."""
    from src.utils.config import Config

    data = copy.deepcopy(_signal_config(cfg).as_dict())
    data["filter"]["enabled"] = bool(arm.filter_enabled)
    data["normalization"]["enabled"] = bool(arm.normalization_enabled)
    return Config("signal", data)


def arm_rows(cfg: Any | None = None) -> list[dict[str, Any]]:
    """One row per arm, carrying the cache directory each would write to."""
    from src.preprocessing.pipeline import cache_root, config_hash
    from src.utils.config import load_config

    root = Path(load_config("paths").require("project_root"))
    shipped = config_hash(_signal_config(cfg))
    rows = []
    for arm in ABLATION_GRID:
        arm_cfg = arm_config(arm, cfg)
        digest = config_hash(arm_cfg)
        directory = cache_root(digest=digest)
        # Relative to the project root: an absolute D:\... path would make this
        # deliverable disagree with itself on the next machine that opens it.
        try:
            printable = directory.resolve().relative_to(root).as_posix()
        except ValueError:
            printable = directory.as_posix()
        rows.append(
            {
                "arm_id": arm.arm_id,
                "label": arm.label,
                "filter_enabled": arm.filter_enabled,
                "normalization_enabled": arm.normalization_enabled,
                "config_hash": digest,
                "cache_dir": printable,
                "is_shipped_configuration": digest == shipped,
            }
        )
    return rows


def grid_path(out_dir: str | Path | None = None) -> Path:
    """The T29.2 grid definition. Not PP-09 -- PP-09 carries the metrics."""
    return _preprocessing_dir(out_dir) / "preprocessing_ablation_grid.csv"


def write_grid(out_dir: str | Path | None = None, cfg: Any | None = None) -> Path:
    """Emit the four-arm grid definition (T29.2).

    Distinct config hashes are asserted, not assumed. Four arms that resolve to
    one hash would silently be one arm run four times, and the resulting PP-09
    would show a preprocessing ablation in which preprocessing made no
    difference -- a conclusion, not a bug, to anyone reading the table later.
    """
    import pandas as pd

    rows = arm_rows(cfg)
    hashes = {row["config_hash"] for row in rows}
    if len(hashes) != len(ABLATION_GRID):
        raise ValueError(
            "the " + str(len(ABLATION_GRID)) + " ablation arms produced only "
            + str(len(hashes)) + " distinct config hashes: " + repr(sorted(hashes))
        )

    table = pd.DataFrame(rows, columns=list(GRID_COLUMNS))
    path = save_csv(table, grid_path(out_dir))
    log.info("T29.2: %d ablation arms defined -> %s", len(table), path.name)
    return path
