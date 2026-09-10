"""T79.7 -- the complexity analysis gate.

The gate: **training time, per-stage inference time, model size and peak memory
are recorded for every model, and T24 through T26 exist.**

"Every model" is checked against the model registry rather than a list written
here, so a model added later cannot go unmeasured while the gate still passes.

Artifact tests skip rather than fail when their input is absent -- the bench
outputs are produced by a script that must run on an idle machine, and a fresh
clone must report "skipped", never a false pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/11_complexity"


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(name: str) -> Any:
    import pandas as pd

    path = _root() / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/31_complexity_analysis.py")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# the measurement machinery itself
# ---------------------------------------------------------------------------


def test_the_resident_set_reader_returns_a_real_number() -> None:
    """The first version returned 0 on Windows and produced a NaN column.

    `GetCurrentProcess` returns the pseudo-handle (HANDLE)-1; without an
    explicit `restype` ctypes truncated it to 32 bits and every call failed with
    ERROR_INVALID_HANDLE, silently. A zero here is not "unsupported platform",
    it is that bug coming back.
    """
    import os

    from src.evaluation.complexity import process_rss_bytes

    value = process_rss_bytes()
    if os.name not in ("nt", "posix"):  # pragma: no cover - platform dependent
        pytest.skip("no resident-set reader for this platform")
    assert value > 1_000_000, "a live CPython process cannot have a 1 MB resident set"


def test_peak_memory_sees_an_allocation_it_is_given() -> None:
    """Measured against a known allocation, not asserted about a model."""
    from src.evaluation.complexity import peak_memory_during

    size_mb = 128.0

    def work() -> float:
        block = np.zeros((4000, 4000))  # 128 MB of float64
        return float(block[0, 0])

    _, report = peak_memory_during(work)
    if not report["rss_sampler_available"]:  # pragma: no cover - platform dependent
        pytest.skip("resident-set sampler unavailable on this platform")
    assert report["tracemalloc_peak_mb"] >= size_mb * 0.9
    assert report["n_samples"] >= 1


def test_the_arm_label_is_read_from_the_config_not_written_down() -> None:
    """EXP-C1 and EXP-C2 declare tuned:true and must not be labelled defaults.

    The first draft hard-coded the arm and got exactly these two wrong, which
    would have printed a nested search as an untuned baseline in T25.
    """
    from src.evaluation.complexity import RUNS, arm_label

    labels = {spec.run: arm_label(spec) for spec in RUNS}
    assert labels["EXP-A1"].startswith("config defaults")
    assert labels["EXP-A2"].startswith("nested search")
    assert labels["EXP-C1-two_class"].startswith("nested search")
    assert labels["EXP-C2"].startswith("nested search")
    assert labels["EXP-B1-defaults"].startswith("config defaults")


# ---------------------------------------------------------------------------
# clause 1 -- training time for every model
# ---------------------------------------------------------------------------


def test_training_time_is_recorded_for_every_model_of_every_run() -> None:
    import pandas as pd

    from src.evaluation.complexity import RUNS
    from src.utils.evidence import PROJECT_ROOT

    summary = _csv("training_time_summary.csv")
    checked = 0
    for spec in RUNS:
        path = PROJECT_ROOT / spec.directory / "per_fold_metrics.csv"
        if not path.is_file():
            continue
        expected = set(pd.read_csv(path)["model_id"].astype(str))
        block = summary[summary["run"] == spec.run]
        assert set(block["model_id"].astype(str)) == expected, spec.run
        assert block["fit_seconds_mean"].notna().all(), spec.run
        assert (block["fit_seconds_mean"] > 0).all(), spec.run
        checked += 1
    assert checked >= 5


def test_training_time_is_reported_as_a_distribution_not_a_point() -> None:
    """A single ensemble fit time is meaningless; min, max and spread must be there."""
    summary = _csv("training_time_summary.csv")
    for column in ("fit_seconds_min", "fit_seconds_max", "fit_seconds_spread"):
        assert column in summary.columns
    assert (summary["fit_seconds_min"] <= summary["fit_seconds_mean"] + 1e-9).all()
    assert (summary["fit_seconds_max"] >= summary["fit_seconds_mean"] - 1e-9).all()

    # The finding the column exists for: the nested ensembles vary by an order
    # of magnitude across folds of the same run.
    nested = summary[(summary["run"] == "EXP-A2") & (summary["model_id"].isin(["M6", "M7"]))]
    if len(nested):
        assert (nested["fit_seconds_spread"] > 5).all(), nested.to_string()


# ---------------------------------------------------------------------------
# clause 2 -- per-stage inference time
# ---------------------------------------------------------------------------


def test_every_pipeline_stage_is_timed_and_the_stages_reconcile() -> None:
    from src.evaluation.complexity import STAGE_ORDER

    stages = _csv("inference_stage_summary.csv")
    present = set(stages["stage"])
    assert set(STAGE_ORDER) <= present
    assert {"total", "unattributed"} <= present

    parts = stages[stages["stage"].isin([*STAGE_ORDER, "unattributed"])]["mean_seconds"].sum()
    total = float(stages[stages["stage"] == "total"]["mean_seconds"].iloc[0])
    assert abs(parts - total) < 1e-6, "the stages do not sum to the measured total"
    assert total > 0


def test_the_inference_measurement_is_end_to_end_on_real_recordings() -> None:
    """T79.2 says load, preprocess, extract, predict -- not predict alone."""
    timings = _csv("inference_timing.csv")
    sample = _csv("inference_sample.csv")

    assert timings["record_uid"].nunique() >= 10
    assert set(timings["record_uid"]) <= set(sample["record_uid"])
    assert (timings["preprocess_seconds"] > 0).all()
    assert (timings["extract_seconds"] > 0).all()
    assert (timings["predict_seconds"] > 0).all()
    assert timings["is_warm"].any() and (~timings["is_warm"]).any()
    # Real recordings, not synthetic input: the sample must span the corpus's
    # real duration spread rather than one convenient length.
    assert sample["duration_sec"].max() / sample["duration_sec"].min() > 1.5


def test_extraction_dominates_the_pipeline_and_the_table_says_so() -> None:
    """The T79.3 finding. Asserted because it drives every latency claim made."""
    stages = _csv("inference_stage_summary.csv")
    share = float(stages[stages["stage"] == "extract"]["share_of_total"].iloc[0])
    predict = float(stages[stages["stage"] == "predict"]["share_of_total"].iloc[0])
    assert share > 0.5, "extraction was expected to dominate; it did not"
    assert predict < share


# ---------------------------------------------------------------------------
# clause 3 -- size and peak memory for every model
# ---------------------------------------------------------------------------


def test_size_and_memory_are_recorded_for_every_implemented_model() -> None:
    from src.evaluation.complexity import TIMED_MODELS

    footprint = _csv("model_footprint.csv")
    assert set(footprint["model_id"]) == set(TIMED_MODELS)
    assert (footprint["model_bytes"] > 0).all()
    assert (footprint["fit_seconds"] > 0).all()
    assert footprint["configuration"].nunique() == 1, "the bench must hold one configuration"
    assert footprint["fold_label"].nunique() == 1


def test_the_memory_column_is_measured_and_its_failures_are_flagged() -> None:
    """An all-NaN column is not a measurement -- that bug shipped once already."""
    footprint = _csv("model_footprint.csv")
    assert "memory_measurement_reliable" in footprint.columns
    reliable = footprint[footprint["memory_measurement_reliable"].astype(bool)]
    assert len(reliable) >= len(footprint) - 2, "most models must be measurable"
    assert reliable["peak_rss_delta_mb"].notna().all()
    assert (reliable["peak_rss_delta_mb"] > 0).all()
    assert (reliable["n_samples"] >= 5).all()
    # An unreliable row keeps its number and its flag; it is not silently blanked.
    assert footprint["n_samples"].notna().all()


def test_every_implemented_model_in_the_registry_was_benched() -> None:
    """The list is checked against the registry, not against itself."""
    from src.evaluation.complexity import TIMED_MODELS
    from src.models.estimators import IMPLEMENTED_MODELS

    assert set(TIMED_MODELS) == set(IMPLEMENTED_MODELS), (
        "a model was added to the registry without being added to the bench"
    )


# ---------------------------------------------------------------------------
# clause 4 -- T24 through T26 exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table_id", "slug"),
    [
        ("T24", "T24_complexity_analysis"),
        ("T25", "T25_training_and_inference_time"),
        ("T26", "T26_model_size_and_memory"),
    ],
)
def test_the_three_tables_exist_in_every_format(table_id: str, slug: str) -> None:
    import pandas as pd

    root = _root()
    csv = root / (slug + ".csv")
    if not csv.is_file():
        pytest.skip(str(csv) + " does not exist; run scripts/31_complexity_analysis.py")
    frame = pd.read_csv(csv)
    assert len(frame), table_id + " is empty"
    for suffix in (".docx", ".tex", ".md", ".meta.json"):
        assert (root / (slug + suffix)).is_file(), table_id + " has no " + suffix


def test_t24_carries_both_fit_time_columns_and_says_they_differ() -> None:
    """The bench fit and the run fit are different quantities and must not merge."""
    import json

    table = _csv("T24_complexity_analysis.csv")
    assert {"fit_seconds", "run_fit_seconds_mean"} <= set(table.columns)

    meta = json.loads((_root() / "T24_complexity_analysis.meta.json").read_text(encoding="utf-8"))
    notes = " ".join(meta.get("notes", []))
    assert "not comparable" in notes.lower()
    assert "CPU only" in notes
    assert "screening" in notes.lower()


def test_t26_states_the_limits_of_its_memory_measurement() -> None:
    import json

    meta_path = _root() / "T26_model_size_and_memory.meta.json"
    if not meta_path.is_file():
        pytest.skip(str(meta_path) + " does not exist; run scripts/31_complexity_analysis.py")
    notes = " ".join(json.loads(meta_path.read_text(encoding="utf-8")).get("notes", []))
    assert "tracemalloc" in notes
    assert "DURING the fit" in notes
    assert "Reliable" in notes


@pytest.mark.parametrize(
    ("figure_id", "slug"),
    [
        ("G25", "G25_inference_time_versus_performance"),
        ("G26", "G26_training_time_comparison"),
        ("G27", "G27_model_size_comparison"),
    ],
)
def test_the_three_figures_exist_with_the_csv_they_were_drawn_from(
    figure_id: str, slug: str
) -> None:
    import pandas as pd

    from src.reporting.graphs import figures_dir

    directory = figures_dir()
    png = directory / (slug + ".png")
    csv = directory / (slug + ".csv")
    if not png.is_file():
        pytest.skip(str(png) + " does not exist; run scripts/31_complexity_analysis.py")
    assert csv.is_file(), figure_id + " has a PNG with no source CSV"
    frame = pd.read_csv(csv)
    assert len(frame)
    assert not frame.isna().all().all()
    assert png.stat().st_size > 10_000
