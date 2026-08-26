"""The six preprocessing figures, PP-01 to PP-06 (Phase 28).

Every record shown here is **selected by a rule, not by eye**. A figure captioned
"a representative normal recording" that was actually chosen because it looked
good is a quiet form of cherry-picking, and it is not reproducible: nobody can
regenerate it, and nobody can check it. :func:`select_records` states each rule
in code, resolves it against the master table and PP-08, and breaks ties on
``record_uid`` so the same six records come back on every machine.

The rules:

``PP-01`` / ``PP-02``
    The PhysioNet normal / abnormal recording whose duration is closest to the
    median of its class, among records that are labelled, non-duplicate and not
    flagged low quality. Median duration because a 3-second and a 122-second
    recording are both real but neither is representative.

``PP-03``
    The D1 record, 8-20 seconds long, with the **most baseline drift that the
    filter can remove while leaving a legible signal behind** -- the highest
    ``drift_ratio_db`` at or below +8 dB.

    The cap is the whole point, and it was added after the first version of this
    figure was drawn without one. Maximising drift outright selects
    ``D1_training-e_e01797`` at +36.6 dB, where the sub-20 Hz artifact carries
    99.98% of the energy: on the shared y-axis T28.3 requires, the filtered panel
    is a flat line at zero. That figure is arithmetically correct and completely
    useless -- it shows the bandpass deleting a recording rather than cleaning
    one. Since the fraction of amplitude surviving the filter is
    ``1 / sqrt(1 + 10^(drift/10))``, capping drift at +8 dB keeps at least ~37%
    of it, which is what makes both panels readable at one scale.

``PP-04``
    The PP-01 record, deliberately: a clean recording isolates what z-scoring
    changes (the amplitude scale) from what it does not (the shape).

``PP-05`` / ``PP-06``
    The PP-01 and PP-02 records again, so the reader compares the same two
    recordings in the time and frequency domains.

**The shared colour scale (T28.5) is computed jointly and stamped on both
figures.** PP-05 and PP-06 are drawn from the *preprocessed* signal -- filtered
and z-scored -- because two recordings made at different gains cannot share a
colour scale meaningfully until they share an amplitude scale. Both figures then
print the same dB range in their colourbar label, which is what makes the T28.7
manual check checkable rather than a matter of impression.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.reporting.plot_style import (
    FIGSIZE,
    OKABE_ITO,
    SEQUENTIAL_CMAP,
    apply_style,
    save_figure,
)
from src.utils.io import ensure_dir
from src.utils.logging_setup import get_logger

__all__ = [
    "PP_FIGURES",
    "SELECTION_RULES",
    "SpectrogramScale",
    "select_records",
    "plot_original_waveform",
    "plot_before_after_filtering",
    "plot_normalization_comparison",
    "plot_spectrogram_pair",
    "generate_all",
]

log = get_logger(__name__)

PP_FIGURES: dict[str, str] = {
    "PP-01": "original_waveform_normal.png",
    "PP-02": "original_waveform_abnormal.png",
    "PP-03": "before_after_filtering.png",
    "PP-04": "normalization_comparison.png",
    "PP-05": "normal_spectrogram.png",
    "PP-06": "abnormal_spectrogram.png",
}

_MEDIAN_RULE = (
    "D1, binary={cls}, labelled, non-duplicate, not low quality, "
    "duration nearest the class median"
)

SELECTION_RULES: dict[str, str] = {
    "normal": _MEDIAN_RULE.format(cls="normal"),
    "abnormal": _MEDIAN_RULE.format(cls="abnormal"),
    "drifty": "D1, duration 8-20 s, highest drift_ratio_db at or below the +8 dB legibility cap",
}

# PP-03's legibility cap, in dB. See the module docstring: above this, the
# filtered panel is a flat line on the shared axis T28.3 asks for.
PP03_DRIFT_CAP_DB = 8.0

# Spectrograms are drawn down to this frequency; above it a 20-400 Hz filtered
# signal holds only the filter's stopband.
SPECTROGRAM_MAX_HZ = 500.0
# Floor of the shared colour scale, in dB below the joint maximum.
SPECTROGRAM_DYNAMIC_RANGE_DB = 60.0

NORMAL_COLOR, ABNORMAL_COLOR, RAW_COLOR = OKABE_ITO[0], OKABE_ITO[1], "#888888"


@dataclass(frozen=True, slots=True)
class SpectrogramScale:
    """The colour scale shared by PP-05 and PP-06 (T28.5)."""

    vmin_db: float
    vmax_db: float

    def label(self) -> str:
        return (
            "power (dB, shared scale "
            + format(self.vmin_db, ".1f")
            + " to "
            + format(self.vmax_db, ".1f")
            + " dB)"
        )


# ---------------------------------------------------------------------------
# record selection
# ---------------------------------------------------------------------------


def _joined(master: Any | None = None, quality: Any | None = None) -> Any:
    """Master metadata joined to PP-08, with the duplicated columns dropped."""
    from src.data_loader import master as ms
    from src.preprocessing import quality as qual

    master = ms.load_master() if master is None else master
    quality = qual.scan_quality() if quality is None else quality

    overlap = ["dataset_source", "subset", "file_path", "duration_sec", "n_samples", "original_fs"]
    return master.merge(quality.drop(columns=overlap), on="record_uid", how="inner")


def select_records(master: Any | None = None, quality: Any | None = None) -> dict[str, Any]:
    """Resolve the three selection rules to three concrete records.

    Returns ``{"normal": row, "abnormal": row, "drifty": row}``. Raises if a rule
    matches nothing -- an empty selection must stop the figure run, not produce a
    figure of whatever happened to be first.
    """
    table = _joined(master, quality)
    d1 = table[
        (table["dataset_source"] == "D1")
        & (~table["is_duplicate"])
        & (table["use_in_supervised"])
    ]

    def nearest_median(label: int, name: str) -> Any:
        group = d1[
            (d1["binary_label"] == label)
            & (~d1["is_low_quality"])
            & d1["duration_sec"].between(8.0, 25.0)
        ]
        if group.empty:
            raise ValueError("no D1 record matches the " + name + " selection rule")
        median = float(group["duration_sec"].median())
        ordered = group.assign(_gap=(group["duration_sec"] - median).abs()).sort_values(
            ["_gap", "record_uid"]  # uid breaks ties, so the choice is deterministic
        )
        return ordered.iloc[0]

    drifty_pool = d1[
        d1["duration_sec"].between(8.0, 20.0)
        & (d1["drift_ratio_db"] <= PP03_DRIFT_CAP_DB)
    ]
    if drifty_pool.empty:
        raise ValueError("no D1 record matches the drifty selection rule")
    drifty = drifty_pool.sort_values(
        ["drift_ratio_db", "record_uid"], ascending=[False, True]
    ).iloc[0]

    return {
        "normal": nearest_median(0, "normal"),
        "abnormal": nearest_median(1, "abnormal"),
        "drifty": drifty,
    }


def _load(record: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """``(resampled_raw, preprocessed, fs)`` for one record.

    The "raw" here is resampled to 2 kHz but otherwise untouched -- no filter, no
    normalization. Plotting the true native-rate signal for PASCAL A would draw
    218,000 points into a 7-inch figure, and the reader would see the resampled
    version anyway once it reached a model.
    """
    from src.preprocessing.io import load_resampled
    from src.preprocessing.pipeline import preprocess
    from src.utils.config import load_config

    root = Path(load_config("paths").require("project_root"))
    path = root / str(record["file_path"])

    raw, _ = load_resampled(path)
    result = preprocess(path, record_uid=str(record["record_uid"]))
    return raw, result.signal, result.fs


def _source_stamp(record: Any, extra: str = "") -> str:
    """Two short lines, not one long one -- see :func:`annotate_source`."""
    first = (
        str(record["record_uid"])
        + "  |  "
        + str(record["dataset_name"])
        + "  |  "
        + format(float(record["duration_sec"]), ".2f")
        + " s at "
        + str(int(record["original_fs"]))
        + " Hz"
    )
    if extra:
        first += "  |  " + extra
    second = "source: metadata_master.csv (DA-08) + signal_quality_flags.csv (PP-08)"
    return "\n".join([first, second])


# ---------------------------------------------------------------------------
# T28.1 / T28.2 -- PP-01, PP-02
# ---------------------------------------------------------------------------


def plot_original_waveform(record: Any, path: str | Path, *, color: str = NORMAL_COLOR) -> Path:
    """One recording's waveform, as it arrives (PP-01 / PP-02)."""
    apply_style()

    raw, _, fs = _load(record)
    time = np.arange(raw.size, dtype=np.float64) / fs

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.plot(time, raw, color=color, linewidth=0.6)
    ax.set_xlim(0, time[-1] if time.size else 1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude (full scale)")
    ax.set_title(
        str(record["binary_label_name"]).capitalize()
        + " phonocardiogram - "
        + str(record["record_uid"])
    )
    peak = float(np.max(np.abs(raw))) if raw.size else 0.0
    ax.text(
        0.99,
        0.96,
        "peak " + format(peak, ".3f") + "   RMS " + format(float(record["rms"]), ".4f"),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )
    return save_figure(fig, path, source=_source_stamp(record, "unfiltered, unnormalized, 2 kHz"))


# ---------------------------------------------------------------------------
# T28.3 -- PP-03
# ---------------------------------------------------------------------------


def plot_before_after_filtering(record: Any, path: str | Path) -> Path:
    """Raw against 20-400 Hz filtered, same record, shared axes (PP-03)."""
    apply_style()

    from src.preprocessing.filters import filter_signal

    raw, _, fs = _load(record)
    filtered = filter_signal(raw, fs).signal
    time = np.arange(raw.size, dtype=np.float64) / fs

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=FIGSIZE["tall"], sharex=True, sharey=True  # shared, per T28.3
    )
    limit = float(np.max(np.abs(raw))) * 1.05 if raw.size else 1.0

    axes[0].plot(time, raw, color=RAW_COLOR, linewidth=0.6)
    axes[0].set_title("Before - raw signal, baseline drift present")
    axes[0].set_ylabel("amplitude")

    axes[1].plot(time, filtered, color=NORMAL_COLOR, linewidth=0.6)
    axes[1].set_title("After - 4th-order Butterworth bandpass, 20-400 Hz, zero phase")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("time (s)")

    axes[0].set_ylim(-limit, limit)
    axes[0].set_xlim(0, time[-1] if time.size else 1.0)

    drift = float(record["drift_ratio_db"])
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
    return save_figure(
        fig, path, source=_source_stamp(record, "y-axes shared between panels")
    )


# ---------------------------------------------------------------------------
# T28.4 -- PP-04
# ---------------------------------------------------------------------------


def plot_normalization_comparison(record: Any, path: str | Path) -> Path:
    """Before and after per-record z-score normalization (PP-04).

    The two panels deliberately do **not** share a y-axis: the entire effect
    being shown is a change of amplitude scale, and forcing one scale would
    either flatten the normalized panel or clip the raw one. Each axis is
    labelled in its own units and the statistics are printed on the figure.
    """
    apply_style()

    from src.preprocessing.filters import filter_signal
    from src.preprocessing.normalize import normalize_signal

    raw, _, fs = _load(record)
    filtered = filter_signal(raw, fs)
    normalized = normalize_signal(filtered.signal, None)
    time = np.arange(raw.size, dtype=np.float64) / fs

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE["tall"], sharex=True)

    axes[0].plot(time, filtered.signal, color=RAW_COLOR, linewidth=0.6)
    axes[0].set_title("Before - filtered, original amplitude scale")
    axes[0].set_ylabel("amplitude (full scale)")
    axes[0].text(
        0.99,
        0.94,
        "mean "
        + format(normalized.mean_before, ".2e")
        + "    SD "
        + format(normalized.std_before, ".4f"),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )

    axes[1].plot(time, normalized.signal, color=NORMAL_COLOR, linewidth=0.6)
    axes[1].set_title("After - per-record z-score, (x - mean) / SD")
    axes[1].set_ylabel("amplitude (SD units)")
    axes[1].set_xlabel("time (s)")
    axes[1].axhline(0.0, color="#444444", linewidth=0.5, linestyle=":")
    axes[1].text(
        0.99,
        0.94,
        "mean "
        + format(normalized.mean_after, ".2e")
        + "    SD "
        + format(normalized.std_after, ".4f"),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )

    axes[0].set_xlim(0, time[-1] if time.size else 1.0)
    fig.align_ylabels(axes)
    return save_figure(
        fig, path, source=_source_stamp(record, "amplitude scale differs by design")
    )


# ---------------------------------------------------------------------------
# T28.5 -- PP-05, PP-06
# ---------------------------------------------------------------------------


def _spectrogram(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(freqs, times, power_db)`` with the project's framing settings."""
    from scipy.signal import spectrogram as scipy_spectrogram

    from src.utils.config import load_config

    cfg = load_config("signal")
    frame = int(cfg.require("framing.frame_length"))
    hop = int(cfg.require("framing.hop_length"))

    freqs, times, power = scipy_spectrogram(
        np.asarray(signal, dtype=np.float64),
        fs=fs,
        nperseg=min(frame, max(signal.size, 8)),
        noverlap=min(frame - hop, max(signal.size - 1, 0)),
        scaling="density",
        mode="psd",
    )
    return freqs, times, 10.0 * np.log10(np.maximum(power, 1e-12))


def plot_spectrogram_pair(
    normal_record: Any,
    abnormal_record: Any,
    normal_path: str | Path,
    abnormal_path: str | Path,
) -> tuple[Path, Path, SpectrogramScale]:
    """PP-05 and PP-06 on one shared colour scale (T28.5).

    Both spectrograms are computed first, the scale is derived from the pair, and
    only then is either figure drawn. Drawing one and then the other would give
    each its own autoscaled colour range, and the two images would be
    incomparable while looking exactly as if they were not.
    """
    apply_style()

    panels = []
    for record in (normal_record, abnormal_record):
        _, preprocessed, fs = _load(record)
        panels.append((record, *_spectrogram(preprocessed, fs)))

    vmax = max(float(np.max(power)) for _, _, _, power in panels)
    scale = SpectrogramScale(
        vmin_db=round(vmax - SPECTROGRAM_DYNAMIC_RANGE_DB, 4), vmax_db=round(vmax, 4)
    )

    import matplotlib.pyplot as plt

    written = []
    for (record, freqs, times, power), path in zip(
        panels, (normal_path, abnormal_path), strict=True
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE["double"])
        mesh = ax.pcolormesh(
            times,
            freqs,
            power,
            cmap=SEQUENTIAL_CMAP,
            vmin=scale.vmin_db,
            vmax=scale.vmax_db,
            shading="auto",
            rasterized=True,
        )
        ax.set_ylim(0, SPECTROGRAM_MAX_HZ)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (Hz)")
        ax.grid(False)
        ax.set_title(
            str(record["binary_label_name"]).capitalize()
            + " spectrogram - "
            + str(record["record_uid"])
        )
        fig.colorbar(mesh, ax=ax, pad=0.02).set_label(scale.label(), fontsize=8)
        written.append(
            save_figure(
                fig,
                path,
                source=_source_stamp(
                    record, "preprocessed signal; colour scale shared with its pair"
                ),
            )
        )

    log.info(
        "PP-05/PP-06 share one colour scale: %.1f to %.1f dB", scale.vmin_db, scale.vmax_db
    )
    return written[0], written[1], scale


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def generate_all(out_dir: str | Path | None = None) -> dict[str, Path]:
    """Emit PP-01 through PP-06. Returns ``{artifact id: path}``."""
    from src.utils.config import load_config

    directory = (
        ensure_dir(out_dir)
        if out_dir is not None
        else ensure_dir(load_config("paths").require("outputs.preprocessing"))
    )
    chosen = select_records()
    log.info(
        "PP figures use normal=%s abnormal=%s drifty=%s",
        chosen["normal"]["record_uid"],
        chosen["abnormal"]["record_uid"],
        chosen["drifty"]["record_uid"],
    )

    paths: dict[str, Path] = {
        "PP-01": plot_original_waveform(
            chosen["normal"], directory / PP_FIGURES["PP-01"], color=NORMAL_COLOR
        ),
        "PP-02": plot_original_waveform(
            chosen["abnormal"], directory / PP_FIGURES["PP-02"], color=ABNORMAL_COLOR
        ),
        "PP-03": plot_before_after_filtering(chosen["drifty"], directory / PP_FIGURES["PP-03"]),
        "PP-04": plot_normalization_comparison(chosen["normal"], directory / PP_FIGURES["PP-04"]),
    }
    normal_path, abnormal_path, _ = plot_spectrogram_pair(
        chosen["normal"],
        chosen["abnormal"],
        directory / PP_FIGURES["PP-05"],
        directory / PP_FIGURES["PP-06"],
    )
    paths["PP-05"] = normal_path
    paths["PP-06"] = abnormal_path
    return paths
