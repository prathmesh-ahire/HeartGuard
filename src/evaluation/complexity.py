"""Computational cost: training time, inference time, model size, peak memory (Phase 79).

Four measurements, and each one is a different kind of number. Keeping them apart
is most of the work here, because the obvious mistake is to quote one of them as
if it were another.

## Training time is a DISTRIBUTION, not a constant

`per_fold_metrics.csv` already records `fit_seconds` for every (model, fold) of
every run, so T79.1 is an aggregation rather than a measurement. It has to be
read as a spread: an M6/M7 fit on the nested binary map ranges from ~117 s to
~2,235 s depending on what that fold's member search selected -- the ensemble
fits each member four times, and a search landing on `M5 max_depth=7, 658 trees`
instead of `max_depth=2, 532 trees` costs 37 minutes on that fold against 2.
Reporting a mean without the min and max would describe a model nobody fitted,
which is why :func:`summarise_training_time` carries five order statistics and
the count.

## Inference time is measured END TO END, on real audio

T79.2 is explicit that it covers load, preprocess, extract and predict. The
number that matters to a deployment is not `predict_proba` on a prepared feature
vector -- that is microseconds -- it is what happens when a `.wav` arrives.
`src.inference.predictor.predict_recording` already instruments exactly those
stages, so this module drives it over a stratified sample of real recordings and
repeats each one.

The decomposition (T79.3) is the finding: **extraction dominates**, and within
extraction `time_sample_entropy` is 97.9% of the corpus-wide extraction cost
because it is O(N^2). A latency budget is therefore a statement about one feature.

## Peak memory needs a sampler, and the sampler's limits are reported

`tracemalloc` sees allocations made through CPython's allocator. That includes
numpy's array buffers on this stack -- measured, not assumed: a 128 MB array
shows up as a 128.03 MB tracemalloc peak. What it does **not** see is memory
taken inside third-party native code that calls `malloc` directly, which is
where a BLAS workspace or XGBoost's own allocator lives, and those are exactly
the models whose fits are expensive.

So :func:`peak_memory_during` also samples the process resident set from the OS
on a background thread and reports the peak above the pre-fit baseline -- with
the sample count beside it, because a fit shorter than a few sample intervals
has not been measured, it has been glanced at. Both numbers are recorded: RSS
sees everything and is process-wide, tracemalloc sees only Python's allocator
and is exact for what it sees. Where they disagree, the gap is native
allocation and page-touching behaviour.

Resident set is process-wide, so this is a peak **during** the fit, not a peak
**caused by** it. Nothing else runs in the measured window, and the baseline is
taken immediately before, but a garbage collection from earlier work can still
land inside it. The column is named `peak_rss_delta_mb` rather than
`model_memory_mb` for that reason.
"""

from __future__ import annotations

import gc
import os
import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "RUNS",
    "TIMED_MODELS",
    "INFERENCE_RECORDS",
    "INFERENCE_REPEATS",
    "STAGE_ORDER",
    "MEMORY_SAMPLE_SECONDS",
    "ComplexityError",
    "RunSpec",
    "training_time_frame",
    "summarise_training_time",
    "process_rss_bytes",
    "peak_memory_during",
    "model_footprint",
    "inference_sample",
    "inference_timing",
    "summarise_inference",
]

log = get_logger("evaluation.complexity")


class ComplexityError(RuntimeError):
    """A complexity measurement cannot be made as asked."""


@dataclass(frozen=True)
class RunSpec:
    """One stored run whose per-fold timings are aggregated.

    ``exp_id`` and ``variant`` exist so the arm label can be *derived* from
    `configs/experiments.yaml` rather than written down here. The first draft
    hard-coded the arm and labelled EXP-C1 and EXP-C2 "config defaults"; both
    declare ``tuned: true``, so the table would have described a nested search
    as an untuned baseline -- which is the one distinction the column exists to
    make. Anything a config already knows is read from the config.
    """

    run: str
    directory: str
    task: str
    exp_id: str
    variant: str | None = None
    note: str = ""


#: Every run that recorded per-fold fit and predict times.
RUNS: tuple[RunSpec, ...] = (
    RunSpec("EXP-A1", "outputs/06_binary_results/EXP-A1", "binary", "EXP-A1"),
    RunSpec("EXP-A2", "outputs/06_binary_results/EXP-A2", "binary", "EXP-A2"),
    RunSpec(
        "EXP-A2-so04_subset",
        "outputs/06_binary_results/EXP-A2-so04_subset",
        "binary",
        "EXP-A2",
        note="on the SO-04 20-feature subset rather than all 138",
    ),
    RunSpec("EXP-B1", "outputs/07_multiclass_results/EXP-B1", "pascal_a", "EXP-B1"),
    RunSpec(
        "EXP-B1-defaults",
        "outputs/07_multiclass_results/EXP-B1-defaults",
        "pascal_a",
        "EXP-B1",
        note="the defaults arm shipped beside EXP-B1",
    ),
    RunSpec("EXP-B2", "outputs/07_multiclass_results/EXP-B2", "pascal_b", "EXP-B2"),
    RunSpec(
        "EXP-B2-defaults",
        "outputs/07_multiclass_results/EXP-B2-defaults",
        "pascal_b",
        "EXP-B2",
        note="the defaults arm shipped beside EXP-B2",
    ),
    RunSpec(
        "EXP-C1-three_class",
        "outputs/08_circor_external_validation/EXP-C1-three_class",
        "circor_murmur",
        "EXP-C1",
        "three_class",
    ),
    RunSpec(
        "EXP-C1-two_class",
        "outputs/08_circor_external_validation/EXP-C1-two_class",
        "circor_murmur",
        "EXP-C1",
        "two_class",
    ),
    RunSpec(
        "EXP-C2",
        "outputs/08_circor_external_validation/EXP-C2",
        "circor_outcome",
        "EXP-C2",
    ),
)

#: How the ``-defaults`` sibling runs are recognised. They are separate output
#: directories of the same declared experiment, so the config's ``tuned`` flag
#: describes the tuned arm and would mislabel the other one.
DEFAULTS_SUFFIX = "-defaults"


def arm_label(spec: RunSpec) -> str:
    """"tuned" or "config defaults" for one run, read from its declaration.

    The distinction is load-bearing: the same model id costs 0.21 s untuned and
    2.97 s under a 12-trial nested search on the same corpus, and averaging the
    two would describe neither.
    """
    from src.evaluation.experiment import Experiment

    if spec.run.endswith(DEFAULTS_SUFFIX):
        label = "config defaults"
    else:
        experiment = Experiment.load(spec.exp_id)
        label = "nested search" if experiment.tuned else "config defaults"
    return label + ((" (" + spec.note + ")") if spec.note else "")

#: The models T26 fits under one declared configuration. M2 (KNN) is included
#: because it is in the registry and `model_complexity.csv` already times it;
#: M9 (1D-CNN) is out of scope on CPU-only hardware and is not fitted anywhere.
TIMED_MODELS: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")

#: How many real recordings the end-to-end inference timing uses, and how many
#: times each is scored. 20 x 5 is 100 measured pipelines; the spread across
#: recordings is dominated by duration, so the sample is stratified by it.
INFERENCE_RECORDS = 20
INFERENCE_REPEATS = 5

#: The stages `predict_recording` instruments, in pipeline order.
STAGE_ORDER: tuple[str, ...] = ("validate", "load_model", "preprocess", "extract", "predict")

#: Resident-set sampling interval. Finer than this and the sampler's own cost
#: starts showing up in the measurement it is taking.
MEMORY_SAMPLE_SECONDS = 0.02


# ---------------------------------------------------------------------------
# T79.1 -- training time, from the stored instrumentation
# ---------------------------------------------------------------------------


def training_time_frame(*, runs: tuple[RunSpec, ...] = RUNS, root: Path | None = None) -> Any:
    """Per ``(run, model, fold)`` fit and predict seconds, from every stored run."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    base = root or PROJECT_ROOT
    blocks: list[pd.DataFrame] = []
    for spec in runs:
        path = base / spec.directory / "per_fold_metrics.csv"
        if not path.is_file():
            log.warning("%s: no per_fold_metrics.csv, skipping", spec.run)
            continue
        frame = pd.read_csv(path)
        keep = [
            c
            for c in (
                "model_id",
                "fold_label",
                "repeat",
                "fold",
                "n_train",
                "n_test",
                "n_features",
                "fit_seconds",
                "predict_seconds",
                "params",
            )
            if c in frame.columns
        ]
        block = frame[keep].copy()
        block.insert(0, "run", spec.run)
        block.insert(1, "task", spec.task)
        block.insert(2, "arm", arm_label(spec))
        blocks.append(block)

    if not blocks:
        raise ComplexityError("no run had a per_fold_metrics.csv to read timings from")
    return pd.concat(blocks, ignore_index=True)


def summarise_training_time(per_fold: Any) -> Any:
    """Five order statistics per ``(run, task, arm, model)``, not just a mean.

    ``fit_seconds_spread`` is max/min. On the nested binary map M6/M7 come out
    around 19x, which is the number that makes a single quoted fit time
    meaningless for those two models.
    """
    import pandas as pd

    keys = ["run", "task", "arm", "model_id"]
    rows: list[dict[str, Any]] = []
    for values, block in per_fold.groupby(keys, sort=True):
        row = dict(zip(keys, values, strict=True))
        row["n_folds"] = len(block)
        row["n_train_mean"] = float(np.mean(block["n_train"])) if "n_train" in block else float(
            "nan"
        )
        for measure in ("fit_seconds", "predict_seconds"):
            if measure not in block.columns:
                continue
            series = np.asarray(block[measure], dtype=float)
            finite = series[np.isfinite(series)]
            if not finite.size:
                continue
            row[measure + "_mean"] = float(finite.mean())
            row[measure + "_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
            row[measure + "_median"] = float(np.median(finite))
            row[measure + "_min"] = float(finite.min())
            row[measure + "_max"] = float(finite.max())
            row[measure + "_total"] = float(finite.sum())
            row[measure + "_spread"] = (
                float(finite.max() / finite.min()) if finite.min() > 0 else float("inf")
            )
        if "predict_seconds_mean" in row and "n_test" in block.columns:
            per_record = np.asarray(block["predict_seconds"], dtype=float) / np.asarray(
                block["n_test"], dtype=float
            )
            row["predict_seconds_per_record_mean"] = float(np.mean(per_record))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T79.4 -- peak memory and on-disk size
# ---------------------------------------------------------------------------


def process_rss_bytes() -> int:
    """Resident set size of this process, in bytes, without a third-party dep.

    Windows goes through ``GetProcessMemoryInfo``; Linux reads
    ``/proc/self/statm``. Anything else returns 0, and the caller reports that
    the sampler was unavailable rather than reporting a zero peak as a result.
    """
    # `os.name`, not `sys.platform`: mypy narrows `sys.platform` at check time
    # and would declare the other branch unreachable on whichever OS is running
    # the type checker, which is exactly the branch that needs checking.
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        # Reached through getattr because the two platform-specific names in
        # this function -- ctypes.windll and resource.getpagesize -- exist in
        # only one platform's typeshed stubs each. A `type: ignore` for one is
        # an *unused* ignore on the other OS, so a checked-both-ways build can
        # never be clean with a direct attribute access here.
        windll = getattr(ctypes, "windll")  # noqa: B009 -- see above

        # `restype` is not optional here and getting it wrong fails silently.
        # GetCurrentProcess returns the pseudo-handle (HANDLE)-1; ctypes defaults
        # an unannotated return to a 32-bit int, so on 64-bit Windows the value
        # is truncated and GetProcessMemoryInfo rejects it with
        # ERROR_INVALID_HANDLE (6) -- returning 0 rather than raising. The first
        # version of this function did exactly that and produced a whole column
        # of NaN peak memory that looked like "the fits were too fast to sample".
        current = windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        current.argtypes = []
        info = windll.psapi.GetProcessMemoryInfo
        info.restype = wintypes.BOOL
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]

        ok = info(current(), ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else 0

    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            fields = handle.read().split()
        import resource

        page = getattr(resource, "getpagesize")  # noqa: B009 -- stubs, as above
        return int(fields[1]) * int(page())
    except (OSError, IndexError, ImportError):
        return 0


def peak_memory_during(work: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Run ``work``, sampling resident set on a thread; return its result and the peak.

    The returned dict reports ``n_samples``. A fit that completed in less than a
    few sample intervals has not been measured and the caller must say so rather
    than print the number it happens to hold.
    """
    samples: list[int] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            value = process_rss_bytes()
            if value:
                samples.append(value)
            stop.wait(MEMORY_SAMPLE_SECONDS)

    gc.collect()
    baseline = process_rss_bytes()
    available = baseline > 0
    tracemalloc.start()
    thread = threading.Thread(target=sampler, daemon=True)
    if available:
        thread.start()
    started = time.perf_counter()
    try:
        result = work()
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        if available:
            thread.join(timeout=1.0)
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    peak = max(samples) if samples else 0
    report = {
        "seconds": float(elapsed),
        "rss_sampler_available": bool(available),
        "n_samples": len(samples),
        "sample_interval_seconds": MEMORY_SAMPLE_SECONDS,
        "baseline_rss_mb": round(baseline / 1e6, 3) if available else float("nan"),
        "peak_rss_mb": round(peak / 1e6, 3) if peak else float("nan"),
        "peak_rss_delta_mb": (
            round(max(peak - baseline, 0) / 1e6, 3) if peak and available else float("nan")
        ),
        "tracemalloc_peak_mb": round(traced_peak / 1e6, 3),
        "tracemalloc_retained_mb": round(traced_current / 1e6, 3),
    }
    return result, report


def model_footprint(
    *,
    models: tuple[str, ...] = TIMED_MODELS,
    task: str = "binary",
    fold_label: str = "r0f0",
    data: Any = None,
) -> Any:
    """Fit each model on one declared fold and record size, time and peak memory.

    **One declared configuration, stated in the output.** Every model is built
    from `configs/models.yaml` defaults on the training rows of ``fold_label``,
    so the row is reproducible and the models are comparable with each other. It
    is deliberately NOT the searched configuration: a nested fold's M6 fit
    depends on what its member search picked and varies 19x across folds, so a
    single searched measurement would be an accident of one fold rather than a
    property of the model. The searched spread is carried by
    :func:`summarise_training_time` instead, and T25 reports both.
    """
    import pandas as pd

    from src.evaluation.cv import load_folds, resolve_folds
    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models.smoke import load_task_data, serialized_size

    resolved = data if data is not None else load_task_data(task)
    folds = resolve_folds(load_folds(task), resolved.record_uids)
    matching = [f for f in folds if f.label == fold_label]
    if not matching:
        raise ComplexityError(
            "no fold labelled " + fold_label + " in the " + task + " map; have "
            + ", ".join(f.label for f in folds[:5]) + " ..."
        )
    fold = matching[0]
    if not fold.is_resolved:  # pragma: no cover - resolve_folds always binds
        raise ComplexityError(fold.label + " was not bound to the matrix row order")
    train = np.asarray(fold.train_index, dtype=int)
    test = np.asarray(fold.test_index, dtype=int)

    rows: list[dict[str, Any]] = []
    for model_id in models:
        estimator = est.build_estimator(model_id)
        built = pl.build_pipeline(
            estimator, y=resolved.y[train], n_features=resolved.X.shape[1]
        )

        def fit(pipeline: Any = built) -> Any:
            # Bound as a default argument, not captured: the closure is invoked
            # inside this iteration, but a late-binding closure over a loop
            # variable is the shape of a bug even when it currently is not one.
            pipeline.fit(resolved.X[train], resolved.y[train])
            return pipeline

        _, memory = peak_memory_during(fit)

        started = time.perf_counter()
        built.predict(resolved.X[test])
        batch_seconds = time.perf_counter() - started

        single = resolved.X[test][:1]
        started = time.perf_counter()
        for _ in range(3):
            built.predict(single)
        single_seconds = (time.perf_counter() - started) / 3.0

        row: dict[str, Any] = {
            "model_id": model_id,
            "task": task,
            "fold_label": fold_label,
            "configuration": "configs/models.yaml defaults",
            "n_train": int(train.size),
            "n_test": int(test.size),
            "n_features": int(resolved.X.shape[1]),
            "estimator_class": type(built).__name__,
            "model_bytes": int(serialized_size(built)),
            "model_mb": round(serialized_size(built) / 1e6, 6),
            "fit_seconds": memory["seconds"],
            "batch_predict_seconds": float(batch_seconds),
            "batch_predict_seconds_per_record": float(batch_seconds) / float(test.size),
            "single_predict_seconds": float(single_seconds),
        }
        row.update({k: v for k, v in memory.items() if k != "seconds"})
        row["memory_measurement_reliable"] = bool(
            memory["rss_sampler_available"] and memory["n_samples"] >= 5
        )
        rows.append(row)
        log.info(
            "%s: %.1f s fit, %.3f MB on disk, peak RSS delta %.1f MB over %d sample(s)",
            model_id,
            row["fit_seconds"],
            row["model_mb"],
            row["peak_rss_delta_mb"],
            row["n_samples"],
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T79.2 / T79.3 -- end-to-end inference, decomposed
# ---------------------------------------------------------------------------


def inference_sample(n_records: int = INFERENCE_RECORDS, *, seed: int = 42) -> Any:
    """A duration-stratified sample of real D1 recordings to time.

    Stratified by the T18.3 duration bands rather than drawn uniformly: the
    extraction stage is superlinear in length, so a uniform sample of a corpus
    whose durations span 0.76 s to 122 s would report a mean latency that
    describes no recording in particular.
    """
    import pandas as pd

    from src.evaluation.duration import duration_bands
    from src.utils.config import load_config

    master = pd.read_csv(
        Path(load_config("paths").require("outputs.dataset_audit")) / "metadata_master.csv",
        low_memory=False,
    )
    bands = duration_bands().set_index("record_uid")["duration_band"]
    frame = master[
        (master["dataset_source"] == "D1") & master["use_in_supervised"].astype(bool)
    ].copy()
    frame["duration_band"] = frame["record_uid"].map(bands)

    rng = np.random.default_rng(seed)
    picks: list[Any] = []
    present = [b for b in ("short", "medium", "long") if (frame["duration_band"] == b).any()]
    per_band = max(1, n_records // max(len(present), 1))
    for band in present:
        block = frame[frame["duration_band"] == band].sort_values("record_uid")
        take = min(per_band, len(block))
        picks.append(block.iloc[rng.choice(len(block), size=take, replace=False)])
    sample = pd.concat(picks, ignore_index=True)
    if len(sample) > n_records:
        sample = sample.iloc[:n_records]
    return sample[
        ["record_uid", "file_path", "duration_sec", "duration_band", "binary_label"]
    ].reset_index(drop=True)


def inference_timing(
    sample: Any | None = None,
    *,
    repeats: int = INFERENCE_REPEATS,
    task: str = "binary",
) -> Any:
    """One row per (record, repeat): every stage of the real inference pipeline.

    `use_cache=False` throughout. The preprocessing cache exists to make a
    corpus-wide extraction resumable; a timing that read it would measure the
    cache, not the pipeline a deployment runs.
    """
    import pandas as pd

    from src.inference.predictor import load_bundle, predict_recording
    from src.utils.evidence import PROJECT_ROOT

    records = inference_sample() if sample is None else sample
    bundle = load_bundle(task)

    rows: list[dict[str, Any]] = []
    for _, record in records.iterrows():
        path = PROJECT_ROOT / str(record["file_path"])
        if not path.is_file():
            raise ComplexityError("sampled recording is missing: " + str(path))
        for repeat in range(repeats):
            result = predict_recording(path, task=task, bundle=bundle, use_cache=False)
            timings = dict(result.timings_seconds)
            row: dict[str, Any] = {
                "record_uid": str(record["record_uid"]),
                "duration_sec": float(record["duration_sec"]),
                "duration_band": str(record["duration_band"]),
                "repeat": repeat,
                "task": task,
                "model_id": result.model.get("model_id"),
                "is_warm": repeat > 0,
            }
            for stage in STAGE_ORDER:
                row[stage + "_seconds"] = float(timings.get(stage, float("nan")))
            row["total_seconds"] = float(timings.get("total", float("nan")))
            # The stages instrumented sum to slightly less than `total`: the
            # feature vector is assembled between `extract` and `predict` and is
            # not separately timed. Reported rather than absorbed into a stage.
            row["unattributed_seconds"] = row["total_seconds"] - sum(
                row[stage + "_seconds"] for stage in STAGE_ORDER
            )
            rows.append(row)
        log.info(
            "%s (%.1f s, %s): %.3f s per inference",
            record["record_uid"],
            record["duration_sec"],
            record["duration_band"],
            float(np.mean([r["total_seconds"] for r in rows[-repeats:]])),
        )

    return pd.DataFrame(rows)


def summarise_inference(timings: Any, *, warm_only: bool = True) -> Any:
    """Per-stage mean, SD, median and share of total (T79.3).

    ``warm_only`` drops the first repeat of each recording. The first call pays
    for librosa's numba JIT and the bundle's first touch, which is a real cost
    exactly once per process and would otherwise be averaged into a per-record
    latency as if every recording paid it.
    """
    import pandas as pd

    frame = timings[timings["is_warm"]] if warm_only else timings
    if not len(frame):
        raise ComplexityError("no warm repeats to summarise; run with repeats >= 2")

    total_mean = float(np.mean(frame["total_seconds"]))
    rows: list[dict[str, Any]] = []
    for stage in (*STAGE_ORDER, "unattributed", "total"):
        column = stage + "_seconds"
        series = np.asarray(frame[column], dtype=float)
        finite = series[np.isfinite(series)]
        rows.append(
            {
                "stage": stage,
                "n_measurements": int(finite.size),
                "mean_seconds": float(finite.mean()) if finite.size else float("nan"),
                "sd_seconds": (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
                ),
                "median_seconds": float(np.median(finite)) if finite.size else float("nan"),
                "min_seconds": float(finite.min()) if finite.size else float("nan"),
                "max_seconds": float(finite.max()) if finite.size else float("nan"),
                "share_of_total": (
                    float(finite.mean()) / total_mean if total_mean > 0 else float("nan")
                ),
                "warm_only": bool(warm_only),
            }
        )
    return pd.DataFrame(rows)
