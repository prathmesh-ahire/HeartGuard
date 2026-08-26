"""The PP-01 .. PP-09 artifact manifest and its evidence registration (T29.5).

Mirrors ``src/data_loader/inventory.AUDIT_ARTIFACTS``: one list, in one place,
naming every artifact Part III is supposed to produce, what it is, and what it
was generated from. Phase 102 assembles the evidence index from these rows and
asserts each one resolves to a real file.

**A missing artifact is registered as ``missing``, never skipped.** PP-09 does
not exist yet -- it needs features and a trained model, which are Parts IV and V
-- so it registers with ``status=missing`` and appears in the index as a known,
dated gap with its reason in ``outputs/missing_outputs_report.txt``. An artifact
quietly absent from the index is indistinguishable from one nobody ever planned
to produce, and by submission nobody would remember which it was.

The three supporting files are registered under ``PP-S*`` ids. They are not in
the source document's PP list, and they are not padding: the transfer function
is the evidence behind the filter's stated -6 dB cutoffs, the SQI calibration is
the evidence behind PP-08's noise flag, and the grid defines the four arms PP-09
will compare. Each would otherwise be an unexplained file in an output folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "PP_ARTIFACTS",
    "SUPPORTING_ARTIFACTS",
    "preprocessing_dir",
    "register_preprocessing_artifacts",
    "verify_preprocessing_artifacts",
]

log = get_logger(__name__)

# (evidence id, filename, description, source it was generated from)
PP_ARTIFACTS: tuple[tuple[str, str, str, str], ...] = (
    ("PP-01", "original_waveform_normal.png",
     "Raw waveform, representative PhysioNet normal recording",
     "outputs/01_dataset_audit/metadata_master.csv"),
    ("PP-02", "original_waveform_abnormal.png",
     "Raw waveform, representative PhysioNet abnormal recording",
     "outputs/01_dataset_audit/metadata_master.csv"),
    ("PP-03", "before_after_filtering.png",
     "Raw versus 20-400 Hz Butterworth bandpass, one record, shared axes",
     "outputs/02_preprocessing/signal_quality_flags.csv"),
    ("PP-04", "normalization_comparison.png",
     "Before and after per-record z-score normalization",
     "outputs/01_dataset_audit/metadata_master.csv"),
    ("PP-05", "normal_spectrogram.png",
     "Spectrogram of a normal recording (colour scale shared with PP-06)",
     "outputs/01_dataset_audit/metadata_master.csv"),
    ("PP-06", "abnormal_spectrogram.png",
     "Spectrogram of an abnormal recording (colour scale shared with PP-05)",
     "outputs/01_dataset_audit/metadata_master.csv"),
    ("PP-07", "preprocessing_settings.csv",
     "Preprocessing configuration: resampling, filter, normalization, framing",
     "configs/signal.yaml"),
    ("PP-08", "signal_quality_flags.csv",
     "Per-record quality metrics and flags for all 7,536 recordings",
     "outputs/01_dataset_audit/metadata_master.csv"),
    ("PP-09", "preprocessing_ablation.csv",
     "Metric delta across the four filter x normalization configurations",
     "outputs/02_preprocessing/preprocessing_ablation_grid.csv"),
)

SUPPORTING_ARTIFACTS: tuple[tuple[str, str, str, str], ...] = (
    ("PP-S1", "filter_transfer_function.png",
     "Bandpass magnitude response: implemented, analytic and zero-phase (T24.4)",
     "configs/signal.yaml"),
    ("PP-S2", "sqi_calibration.csv",
     "Noise proxies scored against PhysioNet REFERENCE-SQI (T26.5)",
     "outputs/02_preprocessing/signal_quality_flags.csv"),
    ("PP-S3", "preprocessing_ablation_grid.csv",
     "The four filter x normalization arms and their config hashes (T29.2)",
     "configs/signal.yaml"),
)

def _cmd(module: str, function: str) -> str:
    return (
        "python -c \"from src.preprocessing." + module + " import " + function
        + "; " + function + "()\""
    )


FIGURE_COMMAND = _cmd("figures", "generate_all")

_COMMANDS: dict[str, str] = {
    "PP-01": FIGURE_COMMAND,
    "PP-07": _cmd("ablation", "write_settings"),
    "PP-08": _cmd("quality", "run_quality_scan"),
    "PP-09": "not yet generated -- see outputs/missing_outputs_report.txt",
    "PP-S1": _cmd("filters", "plot_transfer_function"),
    "PP-S3": _cmd("ablation", "write_grid"),
}


def preprocessing_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if out_dir is not None:
        return ensure_dir(out_dir)
    return ensure_dir(load_config("paths").require("outputs.preprocessing"))


def register_preprocessing_artifacts(
    out_dir: str | Path | None = None, *, index_path: str | Path | None = None
) -> list[dict[str, str]]:
    """Register PP-01 .. PP-09 and the supporting files (T29.5).

    Evidence rows follow their artifacts: an ``out_dir`` outside the configured
    preprocessing directory registers into an index inside it, never into the
    project's real one. See ``inventory.register_audit_artifacts`` for the
    incident that made this rule explicit.
    """
    from src.utils.evidence import evidence_index_path, register_evidence

    directory = preprocessing_dir(out_dir)
    if index_path is None and directory.resolve() != preprocessing_dir().resolve():
        index_path = directory / "evidence_index.csv"
    target = Path(index_path) if index_path is not None else evidence_index_path()
    rows: list[dict[str, str]] = []

    for evidence_id, filename, description, source in (*PP_ARTIFACTS, *SUPPORTING_ARTIFACTS):
        rows.append(
            register_evidence(
                evidence_id=evidence_id,
                objective="Signal preprocessing",
                dataset="D1-D4",
                metric_or_asset=description,
                filename=directory / filename,
                source_data=source,
                command=_COMMANDS.get(evidence_id, FIGURE_COMMAND),
                index_path=target,
            )
        )

    missing = [r["evidence_id"] for r in rows if r.get("status") != "ok"]
    log.info(
        "registered %d preprocessing artifacts (%d missing%s)",
        len(rows),
        len(missing),
        ": " + ", ".join(missing) if missing else "",
    )
    return rows


def verify_preprocessing_artifacts(out_dir: str | Path | None = None) -> dict[str, Any]:
    """Which PP artifacts are on disk and which are not (T29.7)."""
    directory = preprocessing_dir(out_dir)
    present: list[str] = []
    absent: list[str] = []
    for evidence_id, filename, _, _ in (*PP_ARTIFACTS, *SUPPORTING_ARTIFACTS):
        path = directory / filename
        target = present if path.is_file() and path.stat().st_size > 0 else absent
        target.append(evidence_id)
    return {"present": present, "missing": absent, "directory": str(directory)}
