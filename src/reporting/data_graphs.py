"""G01-G10, the data and signal figures (Phase 91).

Ten figures covering the corpus (G01-G04), the signal path (G05-G06) and the
feature representations (G07-G10). Every one is built as a
:class:`~src.reporting.graphs.Graph`: a spec, **the exact frame that gets
plotted**, and a draw function. The frame is written to CSV before the figure is
drawn from it, so the numbers behind every bar, bin and pixel are on disk and
checkable (T90.2).

===  ==========================================  ===========================
G01  Dataset Recording Counts                    DA-01
G02  Class Distribution                          DA-02
G03  Class Distribution Shares                   DA-02
G04  Recording Duration Histogram                DA-08 (metadata_master)
G05  Before and After Filtering                  audio + preprocessing pipeline
G06  Normal Versus Abnormal Spectrogram          audio + preprocessing pipeline
G07  MFCC Heatmap                                audio + MFCC extractor
G08  Chroma Heatmap                              audio + chroma extractor
G09  Wavelet Decomposition                       audio + DWT extractor
G10  Feature Family Counts                       FE-02
===  ==========================================  ===========================

The six-second window, and why the signal figures use one
---------------------------------------------------------
G05-G09 plot a **6-second window** from the start of the preprocessed signal
rather than the whole recording. Two reasons, one editorial and one practical.

Editorially, the representative PhysioNet records run 8-25 s. Twenty seconds of
PCG drawn into a seven-inch figure is a solid band: S1 and S2 are 100 ms events
and there is no width left to see them in. Six seconds is five to seven cardiac
cycles -- enough to show periodicity, wide enough to show morphology.

Practically, the frame that produces the figure is committed to the repository.
A full 25-second waveform at 2 kHz is 50,000 rows, and the wavelet
decomposition of it is more; the same figures over a 6-second window are a few
hundred kilobytes and lose nothing a reader can see. The window is stated in
every caption, and the phase-28/42 figures (PP-03, FE-06..FE-09) still show the
full record, so nothing is hidden -- the two sets of figures answer different
questions.

Record selection is NOT made here
---------------------------------
G05-G09 use ``src.preprocessing.figures.select_records``, the deterministic rule
Phase 28 already established: the D1 record nearest the median duration within
its class, ties broken by ``record_uid``. Re-deriving a "representative record"
here would mean two different records called representative in one thesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.reporting.graphs import (
    Graph,
    GraphSpec,
    class_color,
    subplots,
)
from src.utils.logging_setup import get_logger

__all__ = [
    "DATA_GRAPH_IDS",
    "WINDOW_SECONDS",
    "SUPERVISED_SCOPE",
    "build_g01",
    "build_g02",
    "build_g03",
    "build_g04",
    "build_g05",
    "build_g06",
    "build_g07",
    "build_g08",
    "build_g09",
    "build_g10",
    "build_data_graphs",
]

log = get_logger("reporting.data_graphs")

DATA_GRAPH_IDS: tuple[str, ...] = (
    "G01",
    "G02",
    "G03",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
)

#: The window G05-G09 plot, in seconds. See the module docstring.
WINDOW_SECONDS = 6.0

SUPERVISED_SCOPE = "supervised"

#: The registry order of the six feature families -- the column order of the
#: locked 138-vector, not alphabetical. See the Phase 86 entry in note_window2.
FAMILY_ORDER = ("time", "frequency", "mfcc", "chroma", "dwt", "envelope")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _outputs(key: str) -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs." + key))


def _relative(path: Path) -> str:
    from src.utils.config import load_config

    root = Path(load_config("paths").require("project_root"))
    try:
        return path.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def _read(path: Path) -> Any:
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(
            "source not found: "
            + str(path)
            + " -- the phase that writes it has not run in this checkout"
        )
    return pd.read_csv(path)


def _supervised(frame: Any) -> Any:
    return frame[frame["scope"] == SUPERVISED_SCOPE] if "scope" in frame else frame


def _windowed(signal: np.ndarray, fs: int) -> np.ndarray:
    """The first :data:`WINDOW_SECONDS` of a signal, or all of it if shorter."""
    limit = round(WINDOW_SECONDS * fs)
    return np.asarray(signal[:limit], dtype=np.float64)


def _records() -> dict[str, Any]:
    from src.preprocessing.figures import select_records

    return select_records()


def _preprocessed(record: Any) -> tuple[np.ndarray, int]:
    from src.preprocessing.pipeline import preprocess
    from src.utils.config import load_config

    root = Path(load_config("paths").require("project_root"))
    result = preprocess(root / str(record["file_path"]), record_uid=str(record["record_uid"]))
    return np.asarray(result.signal, dtype=np.float64), int(result.fs)


def _record_label(record: Any) -> str:
    return (
        str(record["record_uid"])
        + " ("
        + str(record["dataset_name"])
        + ", "
        + format(float(record["duration_sec"]), ".1f")
        + " s)"
    )


def _matrix_frame(matrix: np.ndarray, row_name: str, rows: Any, times: np.ndarray) -> Any:
    """A (rows x frames) matrix as a wide CSV: one row per bin, one column per frame.

    Wide rather than long on purpose. A 257-bin spectrogram over 46 frames is
    11,822 values; in long form that is 11,822 CSV rows carrying the frequency
    and time labels again on every one. Wide keeps the file to 257 rows and
    keeps the shape of the thing visible in the file itself.
    """
    import pandas as pd

    frame = pd.DataFrame(
        matrix,
        columns=["t_" + format(float(t), ".3f") for t in times],
    )
    frame.insert(0, row_name, list(rows))
    return frame


def _times(n_frames: int, duration: float) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.linspace(0.0, duration, n_frames, endpoint=False)


# ---------------------------------------------------------------------------
# G01 -- dataset-wise recording counts (T91.1)
# ---------------------------------------------------------------------------


def build_g01(command: str = "") -> Graph:
    source = _outputs("dataset_audit") / "dataset_inventory.csv"
    inventory = _read(source)
    frame = inventory[["dataset_source", "dataset_name", "total_files", "usable_files"]].copy()
    frame["unlabelled_files"] = frame["total_files"] - frame["usable_files"]

    def draw(data: Any) -> Any:
        fig, axis = subplots("double")
        positions = np.arange(len(data))
        width = 0.38
        axis.bar(
            positions - width / 2,
            data["total_files"],
            width,
            label="all files",
            color=class_color(0),
        )
        axis.bar(
            positions + width / 2,
            data["usable_files"],
            width,
            label="labeled (modelled)",
            color=class_color(1),
        )
        for index, row in enumerate(data.itertuples(index=False)):
            axis.text(
                index - width / 2,
                row.total_files,
                f"{row.total_files:,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            axis.text(
                index + width / 2,
                row.usable_files,
                f"{row.usable_files:,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [f"{r.dataset_source}\n{r.dataset_name}" for r in data.itertuples(index=False)]
        )
        axis.set_ylabel("recordings")
        axis.set_title("Recordings per corpus: on disk against labeled")
        axis.legend(loc="upper right")
        axis.margins(y=0.15)
        return fig

    spec = GraphSpec(
        figure_id="G01",
        title="Dataset Recording Counts",
        caption=(
            "Recordings in each corpus. 'All files' is everything in the folder; "
            "'labeled' is the subset with a usable label, which is what enters "
            "modelling. The gap is unlabelled material: 301 PhysioNet validation "
            "twins, 52 PASCAL A and 195 PASCAL B unlabelled records."
        ),
        sources=(_relative(source),),
        objective="O1 (corpus definition)",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G02 / G03 -- class distribution (T91.2)
# ---------------------------------------------------------------------------


def _class_distribution() -> tuple[Path, Any]:
    source = _outputs("dataset_audit") / "class_distribution.csv"
    frame = _supervised(_read(source))
    frame = frame[["dataset_source", "dataset_name", "task", "class", "n_records", "share"]].copy()
    return source, frame.reset_index(drop=True)


def _task_panels(data: Any) -> list[tuple[str, Any]]:
    """One panel per label space, in a fixed order. Rule 4: never merged."""
    order = ("binary", "pascal_a", "pascal_b", "circor_murmur", "circor_outcome")
    present = [task for task in order if task in set(data["task"])]
    present += [task for task in sorted(set(data["task"])) if task not in order]
    return [(task, data[data["task"] == task]) for task in present]


def build_g02(command: str = "") -> Graph:
    source, frame = _class_distribution()

    def draw(data: Any) -> Any:
        panels = _task_panels(data)
        fig, axes = subplots(("wide"), ncols=len(panels), nrows=1)
        axes = np.atleast_1d(axes)
        for axis, (task, block) in zip(axes, panels, strict=True):
            colors = [class_color(i) for i in range(len(block))]
            axis.bar(range(len(block)), block["n_records"], color=colors)
            for index, value in enumerate(block["n_records"]):
                axis.text(index, value, f"{int(value):,}", ha="center", va="bottom", fontsize=7)
            axis.set_xticks(range(len(block)))
            axis.set_xticklabels(list(block["class"]), rotation=30, ha="right", fontsize=7)
            axis.set_title(task + "\n" + str(block["dataset_source"].iloc[0]), fontsize=8)
            axis.margins(y=0.18)
        axes[0].set_ylabel("recordings")
        fig.suptitle("Class distribution, five separate label spaces")
        return fig

    spec = GraphSpec(
        figure_id="G02",
        title="Class Distribution",
        caption=(
            "Class counts for each of the five label spaces, over the supervised "
            "records. The panels are separate because the tasks are: binary, "
            "PASCAL A, PASCAL B, CirCor murmur and CirCor outcome have five "
            "different targets and are never merged. PASCAL's 'artifact' is a "
            "recording-quality label, not a cardiac class."
        ),
        sources=(_relative(source),),
        objective="O1 (corpus definition)",
        size="wide",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


def build_g03(command: str = "") -> Graph:
    source, frame = _class_distribution()

    def draw(data: Any) -> Any:
        panels = _task_panels(data)
        fig, axes = subplots(("wide"), ncols=len(panels), nrows=1)
        axes = np.atleast_1d(axes)
        for axis, (task, block) in zip(axes, panels, strict=True):
            colors = [class_color(i) for i in range(len(block))]
            axis.pie(
                block["n_records"],
                labels=list(block["class"]),
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
                textprops={"fontsize": 7},
                wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            )
            axis.set_title(task + "\n" + f"n = {int(block['n_records'].sum()):,}", fontsize=8)
        fig.suptitle("Class share within each label space")
        return fig

    spec = GraphSpec(
        figure_id="G03",
        title="Class Distribution Shares",
        caption=(
            "The same counts as G02 as within-task shares. Percentages are of "
            "the task's own supervised total, never of the whole corpus: the "
            "five label spaces cover different record sets and a share across "
            "them would be meaningless."
        ),
        sources=(_relative(source),),
        objective="O1 (corpus definition)",
        size="wide",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G04 -- recording duration histogram (T91.3)
# ---------------------------------------------------------------------------

#: Fixed bin edges, in seconds. Declared rather than derived from the data so
#: that regenerating after new records arrive does not silently rebin every
#: published histogram. 0-30 s at 2 s, then coarser out to the 122 s maximum.
DURATION_BIN_EDGES: tuple[float, ...] = (
    0.0,
    2.0,
    4.0,
    6.0,
    8.0,
    10.0,
    12.0,
    14.0,
    16.0,
    18.0,
    20.0,
    22.0,
    24.0,
    26.0,
    28.0,
    30.0,
    40.0,
    50.0,
    60.0,
    80.0,
    125.0,
)


def build_g04(command: str = "") -> Graph:
    import pandas as pd

    source = _outputs("dataset_audit") / "metadata_master.csv"
    master = _read(source)
    supervised = master[master["use_in_supervised"].astype(bool)]
    if supervised.empty:
        raise ValueError("DA-08 has no supervised records; cannot draw G04")

    edges = np.asarray(DURATION_BIN_EDGES, dtype=np.float64)
    rows = []
    for dataset, block in supervised.groupby("dataset_source", sort=True):
        counts, _ = np.histogram(block["duration_sec"].to_numpy(float), bins=edges)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "dataset_source": dataset,
                    "dataset_name": str(block["dataset_name"].iloc[0]),
                    "bin_low_sec": float(edges[index]),
                    "bin_high_sec": float(edges[index + 1]),
                    "n_records": int(count),
                }
            )
    frame = pd.DataFrame(rows)
    if int(frame["n_records"].sum()) != len(supervised):
        raise ValueError(
            "G04: the bins hold "
            + str(int(frame["n_records"].sum()))
            + " records but DA-08 has "
            + str(len(supervised))
            + " supervised -- a recording falls outside the declared bin edges"
        )

    def draw(data: Any) -> Any:
        datasets = list(dict.fromkeys(data["dataset_source"]))
        fig, axes = subplots("tall", nrows=len(datasets), sharex=True)
        axes = np.atleast_1d(axes)
        for index, (axis, dataset) in enumerate(zip(axes, datasets, strict=True)):
            block = data[data["dataset_source"] == dataset]
            widths = block["bin_high_sec"] - block["bin_low_sec"]
            axis.bar(
                block["bin_low_sec"],
                block["n_records"],
                width=widths,
                align="edge",
                color=class_color(index),
                edgecolor="white",
                linewidth=0.4,
            )
            axis.set_ylabel(
                dataset + "\n" + str(block["dataset_name"].iloc[0]),
                rotation=0,
                ha="right",
                va="center",
                fontsize=7,
            )
            axis.locator_params(axis="y", nbins=3)
        axes[-1].set_xlabel("recording duration (s)")
        axes[0].set_title("Supervised recording durations, unequal bins beyond 30 s")
        fig.align_ylabels(axes)
        return fig

    spec = GraphSpec(
        figure_id="G04",
        title="Recording Duration Histogram",
        caption=(
            "Duration of every supervised recording, per corpus, on fixed bin "
            "edges: 2 s bins to 30 s, then coarser out to the 122 s maximum. Bin "
            "edges are declared in code rather than derived from the data, so a "
            "regeneration cannot silently rebin a published histogram. The "
            "extremes are real -- PASCAL set_b reaches 0.76 s and PhysioNet "
            "reaches 122.0 s -- and both are handled by the extractor."
        ),
        sources=(_relative(source),),
        objective="O1 (corpus definition)",
        size="tall",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G05 -- before and after filtering (T91.4)
# ---------------------------------------------------------------------------


def build_g05(command: str = "") -> Graph:
    import pandas as pd

    from src.preprocessing.filters import filter_signal
    from src.preprocessing.io import load_resampled
    from src.utils.config import load_config

    record = _records()["drifty"]
    root = Path(load_config("paths").require("project_root"))
    raw_full, fs = load_resampled(root / str(record["file_path"]))
    raw = _windowed(np.asarray(raw_full, dtype=np.float64), int(fs))
    filtered = _windowed(filter_signal(np.asarray(raw_full), int(fs)).signal, int(fs))

    frame = pd.DataFrame(
        {
            "time_sec": np.arange(raw.size, dtype=np.float64) / float(fs),
            "raw_amplitude": raw,
            "filtered_amplitude": filtered[: raw.size],
        }
    )
    label = _record_label(record)
    drift = float(record["drift_ratio_db"])

    def draw(data: Any) -> Any:
        fig, axes = subplots("tall", nrows=2, sharex=True, sharey=True)
        limit = float(np.max(np.abs(data["raw_amplitude"]))) * 1.05 or 1.0
        axes[0].plot(data["time_sec"], data["raw_amplitude"], linewidth=0.6, color=class_color(4))
        axes[0].set_title("Before - raw signal, baseline drift present")
        axes[0].set_ylabel("amplitude")
        axes[1].plot(
            data["time_sec"], data["filtered_amplitude"], linewidth=0.6, color=class_color(0)
        )
        axes[1].set_title("After - 4th-order Butterworth bandpass, 20-400 Hz, zero phase")
        axes[1].set_ylabel("amplitude")
        axes[1].set_xlabel("time (s)")
        axes[0].set_ylim(-limit, limit)
        axes[0].set_xlim(0.0, float(data["time_sec"].iloc[-1]))
        axes[0].text(
            0.99,
            0.94,
            "sub-20 Hz power " + format(drift, "+.1f") + " dB relative to the 20-400 Hz band",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#444444",
        )
        fig.align_ylabels(axes)
        return fig

    spec = GraphSpec(
        figure_id="G05",
        title="Before And After Filtering",
        caption=(
            "The first "
            + format(WINDOW_SECONDS, ".0f")
            + " seconds of "
            + label
            + ", raw against bandpass-filtered. The y-axes are shared, so the "
            "amplitude reduction is the filter removing baseline wander rather "
            "than a change of scale. The record is the D1 recording with the "
            "largest sub-20 Hz power relative to the passband, selected by the "
            "deterministic Phase 28 rule."
        ),
        sources=(
            "dataset/ (read-only) via src/preprocessing/pipeline.py",
            _relative(_outputs("dataset_audit") / "metadata_master.csv"),
            _relative(_outputs("preprocessing") / "signal_quality_flags.csv"),
        ),
        objective="O3 (preprocessing pipeline)",
        dataset="D1",
        size="tall",
        notes=(
            "Raw here is resampled to the 2 kHz working rate but otherwise "
            "untouched: no filter, no normalization.",
        ),
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G06 -- normal versus abnormal spectrogram (T91.5)
# ---------------------------------------------------------------------------


def build_g06(command: str = "") -> Graph:
    import pandas as pd

    from src.preprocessing.figures import _spectrogram

    chosen = _records()
    blocks = []
    labels: dict[str, str] = {}
    for role in ("normal", "abnormal"):
        record = chosen[role]
        signal, fs = _preprocessed(record)
        window = _windowed(signal, fs)
        freqs, times, power_db = _spectrogram(window, fs)
        block = _matrix_frame(power_db, "frequency_hz", freqs, times)
        block.insert(0, "role", role)
        blocks.append(block)
        labels[role] = _record_label(record)
    frame = pd.concat(blocks, ignore_index=True)

    def draw(data: Any) -> Any:
        # Constrained layout, not tight_layout: one colorbar shared by two axes
        # is exactly the case tight_layout mishandles, and it does so silently --
        # the first draft of this figure had the bar sitting on top of the
        # abnormal panel with its x-label cut off. graphs._stamp knows how to
        # reserve the footer space under a layout engine.
        fig, axes = subplots("double", ncols=2, sharey=True, layout="constrained")
        time_columns = [c for c in data.columns if str(c).startswith("t_")]
        times = np.asarray([float(str(c)[2:]) for c in time_columns])
        values = data[time_columns].to_numpy(float)
        vmin, vmax = float(np.percentile(values, 2)), float(np.percentile(values, 99))
        image = None
        for axis, role in zip(axes, ("normal", "abnormal"), strict=True):
            block = data[data["role"] == role]
            freqs = block["frequency_hz"].to_numpy(float)
            image = axis.pcolormesh(
                times,
                freqs,
                block[time_columns].to_numpy(float),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
            axis.set_title(role + "\n" + labels[role], fontsize=8)
            axis.set_xlabel("time (s)")
            axis.set_ylim(0.0, 400.0)
            axis.grid(visible=False)
        axes[0].set_ylabel("frequency (Hz)")
        bar = fig.colorbar(image, ax=list(axes))
        bar.set_label("power (dB re 1)")
        return fig

    spec = GraphSpec(
        figure_id="G06",
        title="Normal Versus Abnormal Spectrogram",
        caption=(
            "Spectrograms of one normal and one abnormal PhysioNet recording "
            "over the same " + format(WINDOW_SECONDS, ".0f") + "-second window, "
            "on a shared colour scale so the two panels are comparable. Both "
            "records are the ones nearest their class median duration under the "
            "Phase 28 selection rule. The y-axis is limited to the 20-400 Hz "
            "passband the preprocessing keeps."
        ),
        sources=(
            "dataset/ (read-only) via src/preprocessing/pipeline.py",
            _relative(_outputs("dataset_audit") / "metadata_master.csv"),
        ),
        objective="O3 (preprocessing pipeline)",
        dataset="D1",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G07 / G08 / G09 -- the feature representations (T91.6)
# ---------------------------------------------------------------------------


def build_g07(command: str = "") -> Graph:
    from src.feature_extraction.mfcc import MFCCExtractor, mfcc_matrix

    record = _records()["normal"]
    signal, fs = _preprocessed(record)
    window = _windowed(signal, fs)
    settings = MFCCExtractor().settings()
    matrix = mfcc_matrix(window, fs, settings)
    if matrix.size == 0:
        raise ValueError("MFCC matrix is empty; cannot draw G07")

    duration = window.size / fs
    times = _times(matrix.shape[1], duration)
    # Zero-based, matching librosa and FE-06: c0 is the log-energy coefficient.
    frame = _matrix_frame(matrix, "coefficient", range(matrix.shape[0]), times)
    label = _record_label(record)

    def draw(data: Any) -> Any:
        # c0 and c1 are drawn as LINES, not as heatmap rows. Their ranges do not
        # overlap the rest -- c0 is frame log-energy and c1 the overall spectral
        # slope, and on this record they sit tens of dB away from c2-c12. Putting
        # either into the shared colour scale paints its row saturated and
        # flattens every other coefficient into one uniform block: arithmetically
        # correct, completely unreadable. Phase 42 found this on FE-06 and fixed
        # it the same way; the first draft of G07 reproduced the bug.
        #
        # Dropping them would be worse. The extractor computes all 13 and the
        # feature vector uses all 13, so a figure showing 11 would misrepresent
        # what the family is. They are shown, on an axis where they are legible.
        time_columns = [c for c in data.columns if str(c).startswith("t_")]
        values = data[time_columns].to_numpy(float)
        times_axis = np.asarray([float(str(c)[2:]) for c in time_columns])

        fig, axes = subplots("tall", nrows=2, height_ratios=[1, 3])
        axes[0].plot(
            times_axis, values[0], linewidth=0.9, color=class_color(1), label="c0 (log-energy)"
        )
        if values.shape[0] > 1:
            axes[0].plot(
                times_axis,
                values[1],
                linewidth=0.9,
                color=class_color(0),
                label="c1 (spectral slope)",
            )
        low = float(values[:2].min())
        high = float(values[:2].max())
        axes[0].set_ylim(low - 0.05 * (high - low), high + 0.42 * (high - low))
        axes[0].set_xlim(0.0, duration)
        axes[0].set_ylabel("value", fontsize=7)
        axes[0].tick_params(labelbottom=False, labelsize=7)
        axes[0].legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.9)
        axes[0].set_title("MFCC representation, " + str(record["record_uid"]))

        rest = values[2:]
        image = axes[1].imshow(
            rest,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=(0.0, duration, 1.5, rest.shape[0] + 1.5),
        )
        axes[1].set_yticks(range(2, rest.shape[0] + 2))
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("coefficient (c2 - c" + str(rest.shape[0] + 1) + ")")
        axes[1].grid(visible=False)
        bar = fig.colorbar(image, ax=axes[1])
        bar.set_label("coefficient value (dB reference 1.0)")
        return fig

    spec = GraphSpec(
        figure_id="G07",
        title="MFCC Heatmap",
        caption=(
            "The " + str(settings.n_mfcc) + " mel-frequency cepstral coefficients "
            "over the first "
            + format(WINDOW_SECONDS, ".0f")
            + " seconds of "
            + label
            + ". c0 (frame log-energy) and c1 (spectral slope) are drawn "
            "as lines rather than heatmap rows: their ranges sit tens of dB away "
            "from c2-c12, and either one inside the shared colour scale flattens "
            "every other coefficient into a uniform block. All 13 are shown "
            "because the feature vector uses all 13. "
            "The mel spectrogram uses an ABSOLUTE dB reference "
            "(ref=1.0, top_db=None), not librosa's default, which clips 80 dB "
            "below each record's own maximum and would normalise every record "
            "against itself inside a feature meant to be compared across records."
        ),
        sources=(
            "dataset/ (read-only) via src/preprocessing/pipeline.py",
            _relative(_outputs("features") / "feature_inventory.csv"),
        ),
        objective="O4 (feature engineering)",
        dataset="D1",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


def build_g08(command: str = "") -> Graph:
    from src.feature_extraction.chroma import ChromaExtractor, chroma_matrix

    record = _records()["normal"]
    signal, fs = _preprocessed(record)
    window = _windowed(signal, fs)
    settings = ChromaExtractor().settings()
    matrix = chroma_matrix(window, fs, settings)
    if matrix.size == 0:
        raise ValueError("chroma matrix is empty; cannot draw G08")

    duration = window.size / fs
    times = _times(matrix.shape[1], duration)
    frame = _matrix_frame(matrix, "chroma_bin", range(1, matrix.shape[0] + 1), times)
    label = _record_label(record)

    def draw(data: Any) -> Any:
        time_columns = [c for c in data.columns if str(c).startswith("t_")]
        values = data[time_columns].to_numpy(float)
        low, high = float(values.min()), float(values.max())
        fig, axis = subplots("double")
        # The scale spans the DATA, not a nominal 0-1: chroma_stft normalises
        # each frame by its own maximum and a PCG spreads energy nearly evenly
        # across the twelve bins, so forcing vmin=0 draws a uniform bright
        # rectangle -- arithmetically right and completely unreadable.
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=low,
            vmax=high,
            extent=(0.0, duration, 0.5, values.shape[0] + 0.5),
        )
        axis.set_yticks(range(1, values.shape[0] + 1))
        axis.set_xlabel("time (s)")
        axis.set_ylabel("chroma bin")
        axis.set_title("Chroma representation, " + str(record["record_uid"]))
        axis.grid(visible=False)
        bar = fig.colorbar(image, ax=axis)
        bar.set_label("normalized chroma energy")
        return fig

    spec = GraphSpec(
        figure_id="G08",
        title="Chroma Heatmap",
        caption=(
            "The "
            + str(settings.n_chroma)
            + " chroma bins over the first "
            + format(WINDOW_SECONDS, ".0f")
            + " seconds of "
            + label
            + ". Chroma is a musical-pitch construct with no physiological "
            "meaning for heart sounds; it is used here purely as a generic "
            "harmonic-distribution descriptor. The colour scale spans the actual "
            "range rather than a nominal 0-1, because the real dynamic range on "
            "this corpus is narrow and that flatness is itself the point."
        ),
        sources=(
            "dataset/ (read-only) via src/preprocessing/pipeline.py",
            _relative(_outputs("features") / "feature_inventory.csv"),
        ),
        objective="O4 (feature engineering)",
        dataset="D1",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


def build_g09(command: str = "") -> Graph:
    import pandas as pd

    from src.feature_extraction.wavelet import SUBBANDS, WaveletExtractor, decompose

    record = _records()["normal"]
    signal, fs = _preprocessed(record)
    window = _windowed(signal, fs)
    settings = WaveletExtractor().settings()
    bands = decompose(window, settings)
    if not bands:
        raise ValueError("no DWT sub-bands; cannot draw G09")

    duration = window.size / fs
    present = [name for name in SUBBANDS if name in bands]
    rows = []
    for name in present:
        coefficients = np.asarray(bands[name], dtype=np.float64)
        times = np.linspace(0.0, duration, coefficients.size, endpoint=False)
        rows.append(
            pd.DataFrame(
                {
                    "subband": name,
                    "index": np.arange(coefficients.size),
                    "time_sec": times,
                    "coefficient": coefficients,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    label = _record_label(record)

    def draw(data: Any) -> Any:
        names = list(dict.fromkeys(data["subband"]))
        fig, axes = subplots("tall", nrows=len(names), sharex=True)
        axes = np.atleast_1d(axes)
        nyquist = fs / 2.0
        for axis, name in zip(axes, names, strict=True):
            block = data[data["subband"] == name]
            axis.plot(block["time_sec"], block["coefficient"], linewidth=0.5, color=class_color(0))
            if name.startswith("cD"):
                level = int(name[2:])
                band = (
                    format(nyquist / 2**level, ".4g")
                    + "-"
                    + format(nyquist / 2 ** (level - 1), ".4g")
                    + " Hz"
                )
            else:
                band = "0-" + format(nyquist / 2**settings.level, ".4g") + " Hz"
            axis.set_ylabel(
                name + "\n" + band + "\n" + str(len(block)) + " coef.",
                rotation=0,
                ha="right",
                va="center",
                fontsize=7,
            )
            axis.locator_params(axis="y", nbins=3)
            axis.tick_params(labelsize=7)
        axes[-1].set_xlabel("time (s)")
        axes[0].set_title(
            str(settings.level)
            + "-level "
            + settings.wavelet
            + " decomposition, "
            + str(record["record_uid"])
        )
        fig.align_ylabels(axes)
        return fig

    spec = GraphSpec(
        figure_id="G09",
        title="Wavelet Decomposition",
        caption=(
            "The "
            + str(settings.level)
            + "-level "
            + settings.wavelet
            + " sub-bands of the first "
            + format(WINDOW_SECONDS, ".0f")
            + " seconds of "
            + label
            + ". The y-axes are independent: sub-band "
            "amplitudes span three orders of magnitude and a shared scale would "
            "draw most of them as flat lines. Only cD1 lies wholly outside the "
            "20-400 Hz passband, and it carries a median 0.2% of the "
            "decomposition energy across the corpus."
        ),
        sources=(
            "dataset/ (read-only) via src/preprocessing/pipeline.py",
            _relative(_outputs("features") / "feature_inventory.csv"),
        ),
        objective="O4 (feature engineering)",
        dataset="D1",
        size="tall",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# G10 -- feature family counts (T91.6)
# ---------------------------------------------------------------------------


def build_g10(command: str = "") -> Graph:
    source = _outputs("features") / "feature_family_summary.csv"
    families = _read(source)
    frame = families[families["family"] != "TOTAL"][
        ["family", "extractor", "n_features", "first_index"]
    ].copy()
    frame = frame.sort_values("first_index").reset_index(drop=True)

    order = list(frame["family"])
    if order != list(FAMILY_ORDER):
        raise ValueError(
            "G10: registry order is " + ", ".join(order) + " but the locked order "
            "is " + ", ".join(FAMILY_ORDER)
        )
    total = int(frame["n_features"].sum())
    if total != 138:
        raise ValueError("G10: families sum to " + str(total) + ", not the locked 138")

    def draw(data: Any) -> Any:
        fig, axis = subplots("double")
        colors = [class_color(i) for i in range(len(data))]
        bars = axis.barh(range(len(data)), data["n_features"], color=colors)
        for bar, value in zip(bars, data["n_features"], strict=True):
            axis.text(
                bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                str(int(value)),
                va="center",
                fontsize=8,
            )
        axis.set_yticks(range(len(data)))
        axis.set_yticklabels(list(data["family"]))
        axis.invert_yaxis()  # registry order reads top to bottom
        axis.set_xlabel("features")
        axis.set_title("The locked 138-feature registry by family (registry order)")
        axis.margins(x=0.12)
        return fig

    spec = GraphSpec(
        figure_id="G10",
        title="Feature Family Counts",
        caption=(
            "The 138 engineered features by family, in registry order rather "
            "than alphabetically: the order is the fixed column order of the "
            "feature vector, declared as a literal in "
            "src/feature_extraction/registry.py and fingerprinted, and two runs "
            "that disagree on it are not comparable. "
            "138 = time 24 + frequency 22 + MFCC 39 + chroma 24 + DWT 24 + "
            "envelope 5."
        ),
        sources=(_relative(source),),
        objective="O4 (feature engineering)",
        command=command,
    )
    return Graph(spec=spec, frame=frame, draw=draw)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_BUILDERS = {
    "G01": build_g01,
    "G02": build_g02,
    "G03": build_g03,
    "G04": build_g04,
    "G05": build_g05,
    "G06": build_g06,
    "G07": build_g07,
    "G08": build_g08,
    "G09": build_g09,
    "G10": build_g10,
}

#: G05-G09 read audio from ``dataset/`` and run the preprocessing pipeline.
#: They cannot be built on a checkout without the corpus.
NEEDS_AUDIO: tuple[str, ...] = ("G05", "G06", "G07", "G08", "G09")


def build_data_graphs(
    figure_ids: tuple[str, ...] = DATA_GRAPH_IDS, *, command: str = ""
) -> list[Graph]:
    """Build the requested figures without writing anything."""
    built: list[Graph] = []
    for figure_id in figure_ids:
        if figure_id not in _BUILDERS:
            raise KeyError("unknown data graph: " + figure_id)
        graph = _BUILDERS[figure_id](command)
        log.info("%s built (%d plotted rows)", figure_id, len(graph.frame))
        built.append(graph)
    return built
