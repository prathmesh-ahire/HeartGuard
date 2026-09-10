"""The annotated-waveform figure and the cycle summary (Phase 84).

T84.6 asks for a figure overlaying S1 / systole / S2 / diastole on a **real**
recording -- the asset the dashboard's cardiac-cycle viewer consumes. This module
draws it from the corpus WAV and its own `.tsv`, with the record id on the
figure, so a reader can get from the coloured band on screen back to the row in
the segmentation file that put it there.

## Why this figure is not a numbered G-figure

The thirty-five G-figures are a fixed, counted set that Phase 105's gate
reconciles (`35 graphs`). T84.6 asks for "an annotated-waveform figure" without a
G number, so registering one would make the count 36 and fail a gate that is
right to be strict. It is written beside its source CSV like every G-figure, and
registered in the evidence index under `SEG-01`, but it is deliberately kept out
of `figure_registry.csv`.

## The overlay is drawn from the TSV rows, not from a detector

Nothing here runs S1/S2 detection. The bands are the annotator's, read from the
file, and the figure's own CSV holds one row per band with its start, end and
state -- so the picture and the numbers behind it are the same object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.reporting.plot_style import DPI, class_color, styled
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.io import ensure_dir, save_csv, save_json, save_png
from src.utils.logging_setup import get_logger

__all__ = [
    "PHASE_COLOURS",
    "WINDOW_SECONDS",
    "build_cycle_table",
    "waveform_frame",
    "draw_annotated_waveform",
]

log = get_logger("reporting.cycle")

#: One colour per annotated phase, from the project's colourblind-safe palette,
#: in cycle order. Fixed here so the figure and the dashboard viewer agree.
#: Keyed by the `key` field of `segmentation.SEGMENT_LABELS`, not by its display
#: `name`. The first version keyed on the names and silently drew only S1 and S2,
#: because the display names are "Systole" and "Diastole" with capitals -- a
#: lookup miss on a colour map is a missing band, not an error.
PHASE_COLOURS: dict[str, str] = {
    "s1": class_color(0),
    "systole": class_color(1),
    "s2": class_color(2),
    "diastole": class_color(3),
}

#: How much of the recording the figure shows. A whole 20-second recording drawn
#: at page width makes every band a hairline; four seconds shows three or four
#: complete cycles at a width where the phases are separable.
WINDOW_SECONDS = 4.0

_CONTEXT_NOTE = (
    "CONTEXT, NOT A RESULT. None of these timings is among the 138 features -- "
    "the model has never seen a segmentation -- so no difference here explains or "
    "supports any model behaviour."
)

_DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support "
    "prototype, not a diagnostic tool. The recording shown is a de-identified "
    "dataset sample."
)


def build_cycle_table(
    cycles: Any,
    coverage: Any,
    comparison: Any,
    correlations: Any,
    sources: tuple[str, ...],
    command: str = "",
) -> Table:
    """The per-recording cycle statistics, as a numbered-table artifact.

    Given a table id of ``SEG-01`` rather than a T number: the thirty numbered
    tables are a fixed set the Phase 105 gate counts, and T84 asks for a CSV by
    filename rather than for T31.
    """
    frame = cycles.sort_values(["dataset", "record_uid"]).reset_index(drop=True)

    columns = (
        Column("record_uid", "Record"),
        Column("dataset", "Dataset"),
        Column("timing_kind", "Annotation kind"),
        Column("recording_location", "Location"),
        Column("has_segmentation", "Annotated"),
        Column("n_cycles", "Cycles", kind="count"),
        Column("mean_s1_sec", "S1 mean (s)", kind="metric"),
        Column("mean_systole_sec", "Systole mean (s)", kind="metric"),
        Column("mean_s2_sec", "S2 mean (s)", kind="metric"),
        Column("mean_diastole_sec", "Diastole mean (s)", kind="metric"),
        Column("systole_to_diastole_ratio", "Systole / diastole", kind="metric"),
        Column("n_paired_cycles", "Paired cycles", kind="count"),
        Column("annotated_fraction", "Annotated fraction", kind="metric"),
        Column("below_cycle_floor", "Below cycle floor"),
    )
    present = tuple(c for c in columns if c.name in frame.columns)

    correlation_note = _correlation_note(correlations)
    coverage_note = (
        "COVERAGE: "
        + "; ".join(
            str(row["dataset"])
            + " "
            + format(int(row["n_with_annotation"]), ",d")
            + " of "
            + format(int(row["n_recordings"]), ",d")
            + " annotated"
            for _, row in coverage.iterrows()
        )
        + ". PASCAL set_b and PhysioNet 2016 ship no timing annotation at all."
    )

    spec = TableSpec(
        table_id="SEG-01",
        title="Segmentation Cycle Statistics",
        caption=(
            "Per-recording cardiac-cycle timing from the annotations the corpora "
            "ship. CirCor's are interval segmentations, so S1, systole, S2 and "
            "diastole all have durations. PASCAL set_a's are hand-marked S1 and "
            "S2 INSTANTS, so systole and diastole are the gaps between them and "
            "the S1/S2 duration columns are legitimately empty. Means are over "
            "individual segments, never a phase total divided by a cycle count -- "
            "those differ on every recording whose annotation starts or ends "
            "mid-cycle."
        ),
        sources=sources,
        columns=present,
        exp_id="n/a (corpus annotation, not an experiment)",
        objective="O1 (dataset characterisation)",
        dataset="D4 CirCor 2022, D2 PASCAL set_a",
        notes=(
            _CONTEXT_NOTE,
            "THE SYSTOLE-TO-DIASTOLE RATIO IS A PER-CYCLE MEAN, not the ratio of "
            "the two totals. Recordings are annotated in runs with gaps between "
            "them, so the totals can cover different stretches of tape and their "
            "ratio would be an artifact of where the annotator stopped. 'Paired "
            "cycles' counts the cycles that had both phases available.",
            "S1 AND S2 DURATIONS DO NOT EXIST FOR PASCAL and are left empty "
            "rather than imputed. Every annotated set_a recording is also class "
            "'normal', so PASCAL contributes no between-class comparison.",
            coverage_note,
            correlation_note,
            _DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)


def _correlation_note(correlations: Any) -> str:
    """T84.3's answer, written from the correlation frame."""
    if correlations is None or not len(correlations):
        return "T84.3: no correlation could be computed; see the coverage file."
    parts = []
    for target in ("confidence", "is_error"):
        block = correlations[
            (correlations["target"] == target)
            & (correlations["predictor"] == "annotated_fraction")
        ]
        if not len(block):
            continue
        strongest = block.loc[block["rho"].abs().idxmax()]
        significant = int((block["p_value"] < 0.05).sum())
        parts.append(
            target
            + ": strongest |rho| "
            + format(float(strongest["rho"]), ".3f")
            + " ("
            + str(strongest["run"])
            + " "
            + str(strongest["model_id"])
            + ", n="
            + str(int(strongest["n"]))
            + ", p="
            + format(float(strongest["p_value"]), ".3g")
            + "), "
            + str(significant)
            + " of "
            + str(len(block))
            + " model-run pairs significant at 0.05"
        )
    return (
        "T84.3 -- Spearman rank correlation of annotated fraction against "
        + " | ".join(parts)
        + ". Spearman rather than Pearson because annotated fraction is bounded "
        "in [0, 1] and heavily skewed. A weak correlation here is the expected "
        "result: the model never sees the segmentation, so annotation quality "
        "can only relate to its behaviour through whatever made a recording hard "
        "to annotate also making it hard to classify."
    )


def waveform_frame(record_id: str, *, window: float = WINDOW_SECONDS) -> tuple[Any, Any, dict]:
    """The plotted samples and the annotation bands, as two frames plus metadata.

    Returns ``(samples, bands, meta)``. The samples frame is what the line is
    drawn from and the bands frame is what the shading is drawn from, so the
    figure's CSVs and its pixels cannot disagree.
    """
    import pandas as pd
    import soundfile as sf

    from src.reporting.segmentation import read_segmentation, resolve_sources

    wav_path, tsv_path, origin = resolve_sources(record_id)
    signal, fs = sf.read(str(wav_path), dtype="float64", always_2d=False)
    if np.asarray(signal).ndim > 1:  # pragma: no cover - the corpus is mono
        signal = np.asarray(signal)[:, 0]
    signal = np.asarray(signal, dtype=float)

    segments = read_segmentation(tsv_path)
    annotated = [s for s in segments if s["label"] in (1, 2, 3, 4)]
    if not annotated:
        raise ValueError(str(tsv_path) + " holds no annotated segment to draw")

    # Start the window at the first annotated segment: the head of a recording is
    # often unannotated, and a window drawn from t=0 can be entirely empty.
    start = float(annotated[0]["start"])
    end = min(start + window, len(signal) / float(fs))
    low, high = int(start * fs), int(end * fs)

    times = np.arange(low, high, dtype=float) / float(fs)
    samples = pd.DataFrame({"time_sec": times, "amplitude": signal[low:high]})

    bands = pd.DataFrame(
        [
            {
                "record_id": record_id,
                "state": item["label"],
                "phase": item["name"],
                "phase_key": item["key"],
                "start_sec": float(item["start"]),
                "end_sec": float(item["end"]),
                "duration_sec": float(item["end"]) - float(item["start"]),
                "in_window": bool(item["end"] > start and item["start"] < end),
            }
            for item in segments
        ]
    )
    meta = {
        "record_id": record_id,
        "source_origin": origin,
        "wav_path": str(wav_path).replace("\\", "/"),
        "tsv_path": str(tsv_path).replace("\\", "/"),
        "sample_rate_hz": int(fs),
        "duration_sec": float(len(signal) / float(fs)),
        "window_start_sec": start,
        "window_end_sec": end,
        "n_segments_total": len(segments),
        "n_segments_in_window": int(bands["in_window"].sum()),
        "window_rule": (
            "starts at the first annotated segment and runs "
            + format(window, ".1f")
            + " s; the head of a recording is often unannotated"
        ),
    }
    return samples, bands, meta


def draw_annotated_waveform(
    out_dir: str | Path,
    record_id: str,
    *,
    stem: str = "segmentation_annotated_waveform",
    window: float = WINDOW_SECONDS,
) -> dict[str, Path]:
    """T84.6 -- the overlay figure, its source CSVs and its metadata."""
    import matplotlib.pyplot as plt

    samples, bands, meta = waveform_frame(record_id, window=window)
    target = ensure_dir(Path(out_dir))
    written: dict[str, Path] = {
        "samples": save_csv(samples, target / (stem + "_samples.csv")),
        "bands": save_csv(bands, target / (stem + "_segments.csv")),
        "meta": save_json(meta, target / (stem + ".meta.json")),
    }

    with styled():
        fig, axis = plt.subplots(figsize=(10.0, 3.2))
        window_bands = bands[bands["in_window"]]
        seen: set[str] = set()
        for _, band in window_bands.iterrows():
            key = str(band["phase_key"])
            phase = str(band["phase"])
            colour = PHASE_COLOURS.get(key)
            if colour is None:  # unannotated stretches stay unshaded
                continue
            axis.axvspan(
                float(band["start_sec"]),
                float(band["end_sec"]),
                color=colour,
                alpha=0.30,
                linewidth=0,
                label=phase if phase not in seen else None,
            )
            seen.add(phase)

        axis.plot(
            samples["time_sec"].to_numpy(dtype=float),
            samples["amplitude"].to_numpy(dtype=float),
            color="black",
            linewidth=0.6,
        )
        axis.set_xlim(meta["window_start_sec"], meta["window_end_sec"])
        axis.set_xlabel("time (s)")
        axis.set_ylabel("amplitude")
        axis.set_title(
            "Cardiac-cycle segmentation on dataset sample "
            + str(meta["record_id"])
            + " (CirCor, "
            + str(meta["sample_rate_hz"])
            + " Hz). Bands are the corpus annotation, not a detector.",
            fontsize=8,
        )
        axis.legend(fontsize=6, ncol=4, loc="upper right")

        stamp = (
            "Annotated waveform | drawn from "
            + stem
            + "_samples.csv and "
            + stem
            + "_segments.csv\n"
            + "audio: "
            + str(meta["wav_path"])
            + "\nsegmentation: "
            + str(meta["tsv_path"])
            + "\n"
            + _DISCLAIMER
        )
        from src.reporting.plot_style import annotate_source

        annotate_source(fig, stamp)
        written["png"] = save_png(fig, target / (stem + ".png"), dpi=DPI, close=False)
        fig.savefig(target / (stem + ".svg"), format="svg", dpi=DPI, bbox_inches="tight")
        written["svg"] = target / (stem + ".svg")
        plt.close(fig)

    log.info("wrote the annotated waveform for %s -> %s", meta["record_id"], target)
    return written
