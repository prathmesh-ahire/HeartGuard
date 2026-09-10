"""T24, T25, T26 and G25-G27 -- computational cost (Phase 79).

The extraction spec asks for three tables over one body of data, and CLAUDE.md
says to generate all three rather than treat the redundancy as a misreading. They
are three *views*, and each answers a different question:

``T24`` **Complexity Analysis** -- one row per model, cost beside benefit. What
a reader wants when choosing a model: how much does it cost, and what does the
cost buy.

``T25`` **Training and Inference Time** -- the distribution. Every (run, arm,
model) with mean, SD, median, min, max and the max/min spread, because on the
nested map a single fit time is meaningless for the ensembles.

``T26`` **Model Size and Memory** -- the deployment footprint, measured under one
declared configuration so the models are comparable with each other.

## The one number nobody should quote alone

**End-to-end inference is dominated by feature extraction, and extraction is
dominated by one feature.** `time_sample_entropy` is O(N^2) and is 97.9% of the
corpus-wide extraction cost. So the difference between the cheapest and the
dearest model is invisible at the pipeline level: M1 predicts ~190x faster than
M6, and a user waiting for a result would not be able to tell. G25 plots the
model-level cost because that is the part a model choice controls, and its
caption says what the pipeline-level cost actually is.

## Why T25 and T26 do not agree, and must not

T25 aggregates `fit_seconds` recorded *by the experiments themselves*, over
months, on a machine that was sometimes running other jobs -- one measured
episode cost M8 a factor of 14. T26 is a controlled bench: one fold, one
declared configuration, an idle machine. T25 answers "what did this project's
runs cost"; T26 answers "what does this model cost". Neither substitutes for the
other and both say which they are.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.reporting.graphs import Graph, GraphSpec, subplots
from src.reporting.plot_style import class_color
from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "HEADLINE_RUN",
    "build_overview",
    "build_t24",
    "build_t25",
    "build_t26",
    "build_g25",
    "build_g26",
    "build_g27",
]

log = get_logger("reporting.complexity")

#: The run whose accuracy is set against the cost. EXP-A2 is the headline binary
#: run and the one the model-selection rule was applied to.
HEADLINE_RUN = "EXP-A2"

_DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support "
    "prototype, not a diagnostic tool."
)

_CPU_NOTE = (
    "Measured on CPU only -- this machine has no CUDA GPU, and no number here "
    "may be compared against a GPU-trained model. M9 (1D-CNN) is out of scope "
    "for that reason and is not fitted anywhere in this project."
)


def build_overview(footprint: Any, training: Any, inference: Any, quality: Any) -> Any:
    """T24's frame: one row per model, cost joined to what the cost buys.

    ``quality`` is the headline run's per-model metric frame. The join is a left
    join from the footprint, so a model that was benched but never scored still
    appears with an empty metric rather than vanishing -- a missing row in a
    cost table reads as "this model is free".
    """
    import pandas as pd

    frame = footprint.copy()
    headline = training[training["run"] == HEADLINE_RUN]
    columns = [
        "model_id",
        "fit_seconds_mean",
        "fit_seconds_min",
        "fit_seconds_max",
        "fit_seconds_spread",
        "n_folds",
    ]
    available = [c for c in columns if c in headline.columns]
    frame = frame.merge(
        headline[available].rename(
            columns={
                "fit_seconds_mean": "run_fit_seconds_mean",
                "fit_seconds_min": "run_fit_seconds_min",
                "fit_seconds_max": "run_fit_seconds_max",
                "fit_seconds_spread": "run_fit_seconds_spread",
                "n_folds": "run_n_folds",
            }
        ),
        on="model_id",
        how="left",
    )
    frame = frame.merge(quality, on="model_id", how="left")

    # The pipeline-level latency every model shares. Stated per row so a reader
    # comparing two `single_predict_seconds` values sees immediately that the
    # difference is a rounding error against what the pipeline costs anyway.
    total = inference[inference["stage"] == "total"]
    extract = inference[inference["stage"] == "extract"]
    frame["pipeline_seconds_per_recording"] = (
        float(total["mean_seconds"].iloc[0]) if len(total) else float("nan")
    )
    frame["extraction_share_of_pipeline"] = (
        float(extract["share_of_total"].iloc[0]) if len(extract) else float("nan")
    )
    frame["headline_run"] = HEADLINE_RUN
    return pd.DataFrame(frame).sort_values("model_id").reset_index(drop=True)


def build_t24(
    overview: Any, sources: tuple[str, ...], command: str = "", metric: str = "balanced_accuracy"
) -> Table:
    """T24 -- complexity analysis: cost beside benefit, one row per model."""
    columns = [
        Column("model_id", "Model"),
        Column("task", "Task"),
        Column("configuration", "Benched configuration"),
        Column("n_features", "Features", kind="count"),
        Column("fit_seconds", "Bench fit (s)", kind="seconds"),
        Column("run_fit_seconds_mean", "Run fit mean (s)", kind="seconds"),
        Column("run_fit_seconds_spread", "Run fit max/min", kind="metric", places=1),
        Column("single_predict_seconds", "Predict, 1 record (s)", kind="metric", places=6),
        Column(
            "batch_predict_seconds_per_record",
            "Predict, batched (s/record)",
            kind="metric",
            places=6,
        ),
        Column("model_mb", "On disk (MB)", kind="metric"),
        Column("peak_rss_delta_mb", "Peak RSS above baseline (MB)", kind="metric", places=1),
        Column(
            "pipeline_seconds_per_recording",
            "Whole pipeline (s/recording)",
            kind="metric",
            places=3,
        ),
    ]
    if metric in overview.columns:
        columns.append(Column(metric, metric.replace("_", " "), kind="metric"))
    if "sensitivity" in overview.columns:
        columns.append(Column("sensitivity", "sensitivity", kind="metric"))
    present = tuple(c for c in columns if c.name in overview.columns)

    spec = TableSpec(
        table_id="T24",
        title="Complexity Analysis",
        caption=(
            "Computational cost beside the performance it buys, one row per "
            "model. 'Bench fit' and every size and memory column come from one "
            "controlled fit on binary fold r0f0 under configs/models.yaml "
            "defaults, on an otherwise idle machine. 'Run fit mean' and 'Run fit "
            "max/min' come from the "
            + HEADLINE_RUN
            + " run's own instrumentation across its folds and are therefore a "
            "different quantity: a searched configuration, measured while the "
            "project was running. Metrics are that run's fold means."
        ),
        sources=sources,
        columns=present,
        exp_id=HEADLINE_RUN,
        objective="O5 (complexity and deployability)",
        dataset="D1 PhysioNet 2016",
        notes=(
            "THE MODEL IS NOT THE LATENCY. End-to-end inference on a real "
            "recording is dominated by feature extraction, which is identical "
            "for every model; the 'Whole pipeline' column is the same value in "
            "every row for that reason. A 190x difference in predict time is "
            "invisible to anyone waiting for a result.",
            "'Peak RSS above baseline' is the process resident set sampled every "
            "20 ms during the fit, minus the reading taken immediately before "
            "it. It is a peak DURING the fit, not a peak CAUSED by it, and a fit "
            "shorter than a few sample intervals carries too few samples to mean "
            "anything -- model_footprint.csv reports the sample count and a "
            "memory_measurement_reliable flag per row.",
            "The two fit-time columns are not comparable with each other and "
            "neither is an error. See T25.",
            _CPU_NOTE,
            _DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, overview)


def build_t25(summary: Any, inference: Any, sources: tuple[str, ...], command: str = "") -> Table:
    """T25 -- training and inference time, as distributions rather than points."""
    import pandas as pd

    training = summary.copy()
    training.insert(0, "measurement", "training (stored run instrumentation)")
    training = training.rename(
        columns={
            "fit_seconds_mean": "mean_seconds",
            "fit_seconds_sd": "sd_seconds",
            "fit_seconds_median": "median_seconds",
            "fit_seconds_min": "min_seconds",
            "fit_seconds_max": "max_seconds",
            "fit_seconds_spread": "spread_max_over_min",
        }
    )
    training["unit"] = "one fold fit"

    stages = inference.copy()
    stages.insert(0, "measurement", "inference (controlled bench, per recording)")
    stages["run"] = "inference bench"
    stages["task"] = "binary"
    stages["arm"] = "deployed bundle, warm"
    stages["model_id"] = stages["stage"]
    stages["n_folds"] = stages["n_measurements"]
    stages["unit"] = "one recording, one stage"
    stages["spread_max_over_min"] = np.where(
        stages["min_seconds"] > 0, stages["max_seconds"] / stages["min_seconds"], np.nan
    )

    keep = [
        "measurement",
        "run",
        "task",
        "arm",
        "model_id",
        "unit",
        "n_folds",
        "mean_seconds",
        "sd_seconds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "spread_max_over_min",
    ]
    frame = pd.concat(
        [training[[c for c in keep if c in training.columns]], stages[keep]], ignore_index=True
    )

    columns = (
        Column("measurement", "Measurement"),
        Column("run", "Run"),
        Column("task", "Task"),
        Column("arm", "Arm"),
        Column("model_id", "Model / stage"),
        Column("unit", "Unit"),
        Column("n_folds", "Observations", kind="count"),
        Column("mean_seconds", "Mean (s)", kind="seconds"),
        Column("sd_seconds", "SD (s)", kind="seconds"),
        Column("median_seconds", "Median (s)", kind="seconds"),
        Column("min_seconds", "Min (s)", kind="seconds"),
        Column("max_seconds", "Max (s)", kind="seconds"),
        Column("spread_max_over_min", "Max / min", kind="metric", places=1),
    )

    spec = TableSpec(
        table_id="T25",
        title="Training and Inference Time",
        caption=(
            "Training time per fold, from every stored run's own instrumentation, "
            "and end-to-end inference time per recording decomposed by pipeline "
            "stage. Training rows are distributions, not points: min, max and the "
            "max/min ratio travel with the mean. Inference rows are the warm "
            "repeats of a controlled bench over a duration-stratified sample of "
            "real PhysioNet recordings."
        ),
        sources=sources,
        columns=columns,
        exp_id="EXP-A1, EXP-A2, EXP-B1, EXP-B2, EXP-C1, EXP-C2",
        objective="O5 (complexity and deployability)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        notes=(
            "A SINGLE ENSEMBLE FIT TIME IS MEANINGLESS. M6 and M7 fit each member "
            "four times, and a nested fold whose member search picked a large "
            "gradient-boosting configuration costs an order of magnitude more "
            "than one that picked a small one. Read the min/max, and never quote "
            "an ensemble's mean fit time without the spread beside it.",
            "TRAINING TIMES WERE NOT MEASURED ON AN IDLE MACHINE. They are what "
            "the project's own runs recorded, over months, sometimes with other "
            "jobs running -- one measured episode of three concurrent jobs cost "
            "M8 a factor of 14. They are honest records of what the runs cost and "
            "they are NOT a like-for-like model comparison; T26's controlled "
            "bench is. A large max/min on a 'config defaults' arm is more likely "
            "contention than a model property.",
            "The instrumented inference stages sum to slightly less than 'total': "
            "assembling the feature vector between extraction and prediction is "
            "not separately timed, and the remainder is reported as the "
            "'unattributed' stage rather than absorbed into a neighbour.",
            "The first scoring of a process pays for librosa's numba JIT and the "
            "bundle's first load. That cost is real exactly once per process, so "
            "the reported rows are the warm repeats; the cold first repeat is "
            "kept in inference_timing.csv.",
            _CPU_NOTE,
            _DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, frame)


def build_t26(footprint: Any, sources: tuple[str, ...], command: str = "") -> Table:
    """T26 -- on-disk size and peak memory, under one declared configuration."""
    columns = (
        Column("model_id", "Model"),
        Column("task", "Task"),
        Column("fold_label", "Fold"),
        Column("configuration", "Configuration"),
        Column("n_train", "Training rows", kind="count"),
        Column("n_features", "Features", kind="count"),
        Column("model_bytes", "On disk (bytes)", kind="count"),
        Column("model_mb", "On disk (MB)", kind="metric"),
        Column("fit_seconds", "Fit (s)", kind="seconds"),
        Column("baseline_rss_mb", "RSS before fit (MB)", kind="metric", places=1),
        Column("peak_rss_mb", "Peak RSS (MB)", kind="metric", places=1),
        Column("peak_rss_delta_mb", "Peak above baseline (MB)", kind="metric", places=1),
        Column("tracemalloc_peak_mb", "tracemalloc peak (MB)", kind="metric", places=1),
        Column("n_samples", "RSS samples", kind="count"),
        Column("memory_measurement_reliable", "Reliable"),
    )

    spec = TableSpec(
        table_id="T26",
        title="Model Size and Memory",
        caption=(
            "Serialized size and peak memory for every model, each fitted once on "
            "the training rows of binary fold r0f0 under configs/models.yaml "
            "defaults. One declared configuration for all of them, so the sizes "
            "are comparable with each other; deliberately not a searched "
            "configuration, whose cost is an accident of one fold's search."
        ),
        sources=sources,
        columns=columns,
        exp_id="n/a (controlled bench, not an experiment)",
        objective="O5 (complexity and deployability)",
        dataset="D1 PhysioNet 2016, fold r0f0 training rows",
        notes=(
            "TWO MEMORY NUMBERS, AND THEY MEASURE DIFFERENT THINGS. tracemalloc "
            "counts allocations made through CPython's allocator, which on this "
            "stack includes numpy's array buffers but NOT memory taken inside "
            "native library code that calls malloc directly -- a BLAS workspace "
            "or XGBoost's own allocator. The resident-set peak sees all of it but "
            "is process-wide, so it is a peak DURING the fit rather than a peak "
            "CAUSED by it. Where the two disagree, the gap is native allocation "
            "and page-touching behaviour.",
            "'Reliable' is false where the resident-set sampler collected fewer "
            "than five samples -- a fit shorter than about 100 ms has been "
            "glanced at, not measured, and its memory column must not be quoted.",
            "On-disk size is the joblib serialization of the WHOLE pipeline "
            "(imputer, scaler, estimator), which is what a deployment loads, not "
            "the bare estimator.",
            _CPU_NOTE,
            _DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, footprint)


# ---------------------------------------------------------------------------
# G25 / G26 / G27
# ---------------------------------------------------------------------------


def build_g25(
    overview: Any, sources: tuple[str, ...], command: str = "", metric: str = "balanced_accuracy"
) -> Graph:
    """G25 -- inference time against performance, with the pipeline floor drawn."""

    def draw(data: Any) -> Any:
        frame = data[data[metric].notna()] if metric in data.columns else data
        fig, axes = subplots("double", ncols=2, nrows=1)
        left, right = np.atleast_1d(axes)[0], np.atleast_1d(axes)[1]

        # Labels are staggered above and below alternately. Several models sit
        # within a few percent of each other on both axes -- M6 and M7 are the
        # same ensemble with different weights -- so a fixed offset overplots
        # them into an unreadable blob.
        ordered = frame.sort_values("single_predict_seconds").reset_index(drop=True)
        for index, row in ordered.iterrows():
            x = float(row["single_predict_seconds"])
            y = float(row[metric])
            left.scatter(
                x, y, s=42, color=class_color(index), zorder=3, edgecolor="black", linewidth=0.4
            )
            offsets = ((7, 5), (7, -11), (-19, 5), (-19, -11))
            left.annotate(
                str(row["model_id"]),
                (x, y),
                textcoords="offset points",
                xytext=offsets[index % len(offsets)],
                fontsize=6,
            )
        left.set_xscale("log")
        left.set_xmargin(0.25)
        left.set_ymargin(0.18)
        left.set_xlabel("predict time for one record (s, log scale)")
        left.set_ylabel(metric.replace("_", " ") + " (" + HEADLINE_RUN + " fold mean)")
        left.set_title("What a model choice controls", fontsize=8)

        pipeline = float(frame["pipeline_seconds_per_recording"].iloc[0])
        share = float(frame["extraction_share_of_pipeline"].iloc[0])
        slowest = float(frame["single_predict_seconds"].max())
        labels = ["whole pipeline", "of which extraction", "slowest model's predict"]
        values = [pipeline, pipeline * share, slowest]
        # Linear, not log. The point of this panel is that one bar is a sliver
        # of another, and a log axis is precisely the transform that hides that.
        right.barh(
            labels,
            values,
            color=[class_color(0), class_color(1), class_color(2)],
            edgecolor="black",
            linewidth=0.4,
        )
        for position, value in enumerate(values):
            right.text(
                value + pipeline * 0.02,
                position,
                format(value, ".3f") + " s",
                va="center",
                fontsize=6,
            )
        right.set_xlim(0, pipeline * 1.25)
        right.set_xlabel("seconds per recording")
        right.set_title("What the user actually waits for", fontsize=8)
        right.tick_params(labelsize=6)

        fig.suptitle("Inference cost against performance")
        return fig

    spec = GraphSpec(
        figure_id="G25",
        title="Inference Time Versus Performance",
        caption=(
            "Left: each model's predict time for a single prepared feature vector "
            "against its "
            + metric.replace("_", " ")
            + " on "
            + HEADLINE_RUN
            + ", both from the controlled bench and the stored run. Right: the "
            "same numbers against what an end-to-end inference on a real "
            "recording costs. Feature extraction is the same work for every "
            "model, so the spread on the left is invisible to anyone waiting for "
            "a result -- a model choice buys accuracy here, not latency. The left "
            "axis is logarithmic because the models span two orders of magnitude; "
            "the right one is deliberately linear, because a log axis is exactly "
            "the transform that would hide how small the model's share is."
        ),
        sources=sources,
        exp_id=HEADLINE_RUN,
        objective="O5 (complexity and deployability)",
        dataset="D1 PhysioNet 2016",
        command=command,
    )
    return Graph(spec=spec, frame=overview, draw=draw)


def build_g26(summary: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G26 -- training time per model, per run, on a log scale with the spread."""

    def draw(data: Any) -> Any:
        fig, axis = subplots("wide")
        runs = sorted(set(data["run"]))
        models = sorted(set(data["model_id"]))
        width = 0.8 / max(len(runs), 1)

        for index, run in enumerate(runs):
            block = data[data["run"] == run].set_index("model_id")
            means, lows, highs = [], [], []
            for model in models:
                if model in block.index:
                    row = block.loc[model]
                    means.append(float(row["fit_seconds_mean"]))
                    lows.append(float(row["fit_seconds_min"]))
                    highs.append(float(row["fit_seconds_max"]))
                else:
                    means.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)
            positions = np.arange(len(models)) + index * width - 0.4 + width / 2
            mean_array = np.asarray(means, dtype=float)
            errors = np.vstack(
                [
                    np.clip(mean_array - np.asarray(lows, dtype=float), 0, None),
                    np.clip(np.asarray(highs, dtype=float) - mean_array, 0, None),
                ]
            )
            # The Okabe-Ito palette has eight colours and there are ten runs, so
            # the ninth and tenth would silently reuse the first two. Hatching
            # the wrapped ones keeps every run distinguishable without inventing
            # a colour outside the project's declared palette.
            axis.bar(
                positions,
                mean_array,
                width=width,
                yerr=errors,
                capsize=1.5,
                color=class_color(index),
                edgecolor="black",
                linewidth=0.3,
                hatch="///" if index >= 8 else "",
                label=run,
                error_kw={"elinewidth": 0.6},
            )

        axis.set_yscale("log")
        axis.set_xticks(np.arange(len(models)))
        axis.set_xticklabels(models)
        axis.set_xlim(-0.6, len(models) - 0.4)
        axis.set_ylabel("seconds to fit one fold (log scale)")
        axis.legend(fontsize=5, ncol=3)
        axis.set_title(
            "Training time per fold. Whiskers are the observed min and max, not a "
            "standard deviation.",
            fontsize=7,
        )
        fig.suptitle("Training time comparison")
        return fig

    spec = GraphSpec(
        figure_id="G26",
        title="Training Time Comparison",
        caption=(
            "Mean seconds to fit one fold, per model and per run, on a log scale. "
            "Whiskers span the observed minimum and maximum across that run's "
            "folds rather than a standard deviation, because the quantity is not "
            "symmetric: an ensemble fold whose member search picked a large "
            "gradient-boosting configuration costs an order of magnitude more "
            "than one that picked a small one. These are the times the project's "
            "own runs recorded on a machine that was sometimes running other "
            "jobs; T26's controlled bench is the like-for-like comparison. The "
            "colourblind-safe palette holds eight colours and there are ten runs, "
            "so the last two are hatched rather than given a ninth colour."
        ),
        sources=sources,
        exp_id="EXP-A1, EXP-A2, EXP-B1, EXP-B2, EXP-C1, EXP-C2",
        objective="O5 (complexity and deployability)",
        dataset="D1 PhysioNet 2016, D2/D3 PASCAL, D4 CirCor 2022",
        command=command,
        size="wide",
    )
    return Graph(spec=spec, frame=summary, draw=draw)


def build_g27(footprint: Any, sources: tuple[str, ...], command: str = "") -> Graph:
    """G27 -- serialized size beside the peak memory the fit needed."""

    def draw(data: Any) -> Any:
        frame = data.sort_values("model_id").reset_index(drop=True)
        fig, axes = subplots("double", ncols=2, nrows=1)
        left, right = np.atleast_1d(axes)[0], np.atleast_1d(axes)[1]
        positions = np.arange(len(frame))

        left.bar(
            positions,
            frame["model_mb"].to_numpy(dtype=float),
            color=[class_color(i) for i in range(len(frame))],
            edgecolor="black",
            linewidth=0.4,
        )
        left.set_yscale("log")
        left.set_xticks(positions)
        left.set_xticklabels(frame["model_id"], fontsize=6)
        left.set_title("On disk", fontsize=8)

        reliable = frame["memory_measurement_reliable"].astype(bool).to_numpy()
        right.bar(
            positions,
            frame["peak_rss_delta_mb"].to_numpy(dtype=float),
            color=[class_color(i) for i in range(len(frame))],
            edgecolor="black",
            linewidth=0.4,
            hatch=["" if ok else "//" for ok in reliable],
        )
        right.set_xticks(positions)
        right.set_xticklabels(frame["model_id"], fontsize=6)
        right.set_ylabel("peak resident set above baseline (MB)", fontsize=7)
        right.set_title("Peak memory during the fit", fontsize=8)
        left.set_ylabel("serialized pipeline (MB, log scale)", fontsize=7)

        # The right panel's axis label is long enough to reach the suptitle at
        # this figure size; widen the gutter and lower the panels rather than
        # abbreviating a label that has to say which memory this is.
        fig.subplots_adjust(wspace=0.32, top=0.78)
        fig.suptitle("Model size and fit memory, one declared configuration", y=0.97)
        fig.text(
            0.5,
            0.90,
            "Hatched bars had fewer than five resident-set samples -- the fit was "
            "too short to measure and the value must not be quoted.",
            fontsize=6,
            ha="center",
        )
        return fig

    spec = GraphSpec(
        figure_id="G27",
        title="Model Size Comparison",
        caption=(
            "Left: the joblib size of the whole fitted pipeline -- imputer, "
            "scaler and estimator, which is what a deployment loads -- on a log "
            "scale, spanning three orders of magnitude. Right: peak process "
            "resident set above the pre-fit baseline, sampled every 20 ms. Every "
            "model was fitted once on the training rows of binary fold r0f0 under "
            "configs/models.yaml defaults, on an otherwise idle machine, so the "
            "bars are comparable with each other. Hatched bars collected fewer "
            "than five samples: the fit finished too quickly to measure."
        ),
        sources=sources,
        exp_id="n/a (controlled bench, not an experiment)",
        objective="O5 (complexity and deployability)",
        dataset="D1 PhysioNet 2016, fold r0f0 training rows",
        command=command,
    )
    return Graph(spec=spec, frame=footprint, draw=draw)
