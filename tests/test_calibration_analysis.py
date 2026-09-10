"""T78.7 -- the calibration analysis gate.

The gate: **Brier and ECE are computed for every model, sigmoid versus isotonic
was compared for the SVM, and the chosen calibration is recorded.**

"Every model" is checked against each run's own `predictions.parquet` rather
than against a hard-coded list, so a model added to an experiment later cannot
quietly go unmeasured while the gate still passes.

Every artifact test skips rather than fails when its input is absent: the
outputs of `scripts/30_calibration_analysis.py` are not committed in full and a
fresh clone must report "skipped", never a false pass. See the standing rule in
Docs/note.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/10_robustness"
RUN_DIR = "calibration"


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION / RUN_DIR


def _csv(name: str) -> Any:
    import pandas as pd

    path = _root() / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/30_calibration_analysis.py")
    return pd.read_csv(path)


def _section_csv(name: str) -> Any:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / SECTION / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/30_calibration_analysis.py")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# the module itself, on real predictions
# ---------------------------------------------------------------------------


def test_declared_label_spaces_match_the_experiment_configs() -> None:
    """Rule 4: five label spaces, and the declaration is compared, not trusted."""
    from src.evaluation.calibration_analysis import run_specs

    specs = run_specs(verify=True)
    assert len(specs) >= 7
    tasks = {spec.task for spec in specs}
    assert tasks == {"binary", "pascal_a", "pascal_b", "circor_murmur", "circor_outcome"}
    for spec in specs:
        assert len(spec.labels) == len(spec.class_names)


def test_recomputed_brier_and_ece_reproduce_the_stored_per_fold_values() -> None:
    """The analysis must be reading the same probabilities the experiments scored.

    `per_fold_metrics.csv` already holds Brier and ECE at the default 10 bins.
    Recomputing them from `predictions.parquet` and finding a different number
    would mean one of the two files describes a model the other does not.
    """
    import pandas as pd

    from src.evaluation.calibration_analysis import (
        RELIABILITY_BINS,
        load_predictions,
        per_fold_calibration,
        run_specs,
    )
    from src.utils.evidence import PROJECT_ROOT

    spec = next(s for s in run_specs() if s.run == "EXP-A1")
    frame = load_predictions(spec)
    if frame is None:
        pytest.skip("EXP-A1 predictions.parquet unavailable")
    stored_path = PROJECT_ROOT / spec.directory / "per_fold_metrics.csv"
    if not stored_path.is_file():
        pytest.skip("EXP-A1 per_fold_metrics.csv unavailable")

    recomputed = per_fold_calibration(frame, spec.labels, bins=(RELIABILITY_BINS,))
    stored = pd.read_csv(stored_path)[["model_id", "fold_label", "brier", "ece"]]
    merged = recomputed.merge(stored, on=["model_id", "fold_label"], suffixes=("_new", "_old"))
    assert len(merged) == len(recomputed)
    assert np.allclose(merged["brier_new"], merged["brier_old"], atol=1e-12)
    assert np.allclose(merged["ece_new"], merged["ece_old"], atol=1e-12)


def test_reliability_is_measured_against_the_argmax_not_the_thresholded_label() -> None:
    """M6/M7 predict by a tuned threshold, so y_pred is the wrong reference.

    Verified by construction rather than by assertion about the models: the
    helper is handed a frame whose stored ``y_pred`` deliberately disagrees with
    the argmax, and the accuracy it reports must follow the argmax.
    """
    import pandas as pd

    from src.evaluation.calibration_analysis import _confidence_and_correct

    frame = pd.DataFrame(
        {
            "y_true": [1, 0],
            "y_pred": [0, 1],  # the opposite of the argmax in both rows
            "proba_0": [0.2, 0.9],
            "proba_1": [0.8, 0.1],
        }
    )
    confidence, correct, _ = _confidence_and_correct(frame, (0, 1))
    assert np.allclose(confidence, [0.8, 0.9])
    assert np.allclose(correct, [1.0, 1.0])


def test_ece_grows_with_the_bin_count_on_real_predictions() -> None:
    """T78.3 -- bins are configurable, and the choice is not cosmetic.

    Not an arbitrary property: a finer grid removes within-bin averaging, so it
    can only find at least as much miscalibration. A run where ECE fell with
    more bins would mean the binning is wrong.
    """
    sweep = _csv("calibration_bin_sensitivity.csv")
    binary = sweep[(sweep["run"] == "EXP-A1") & (sweep["n_bins"].isin([5, 20]))]
    if not len(binary):
        pytest.skip("bin sweep does not cover EXP-A1")
    coarse = binary[binary["n_bins"] == 5].set_index("model_id")["ece_mean"]
    fine = binary[binary["n_bins"] == 20].set_index("model_id")["ece_mean"]
    shared = coarse.index.intersection(fine.index)
    assert len(shared)
    assert (fine[shared] >= coarse[shared] - 1e-9).all()


# ---------------------------------------------------------------------------
# clause 1 -- Brier and ECE for EVERY model on EVERY task
# ---------------------------------------------------------------------------


def test_every_model_of_every_run_has_a_brier_and_an_ece() -> None:
    """Checked against each run's own predictions.parquet, not a hard-coded list.

    The parquets are gitignored, so on CI and on a fresh clone there is nothing
    to check the summary against and this SKIPS. It must not pass vacuously
    either: `available` counts the runs whose predictions were actually read,
    and a zero there is a skip rather than a silently-satisfied loop. That is
    the standing rule in Docs/note.md, and the first version of this test broke
    CI by asserting on a counter no run had incremented.
    """
    from src.evaluation.calibration_analysis import load_predictions, run_specs

    summary = _csv("calibration_summary.csv")
    measured = 0
    available = 0
    for spec in run_specs():
        frame = load_predictions(spec)
        if frame is None:
            continue
        available += 1
        expected = set(frame["model_id"].astype(str))
        block = summary[summary["run"] == spec.run]
        assert set(block["model_id"].astype(str)) == expected, spec.run
        assert block["brier_mean"].notna().all(), spec.run
        assert block["ece_mean"].notna().all(), spec.run
        assert (block["n_folds"] == frame["fold_label"].nunique()).all(), spec.run
        measured += len(block)

    if not available:
        pytest.skip("no run has a predictions.parquet here (they are gitignored)")
    assert measured >= 40


def test_no_calibration_measure_is_out_of_its_possible_range() -> None:
    summary = _csv("calibration_summary.csv")
    n_classes = {"binary": 2, "pascal_a": 4, "pascal_b": 3, "circor_murmur": 3, "circor_outcome": 2}
    assert (summary["ece_mean"] >= 0).all()
    assert (summary["ece_mean"] <= 1).all()
    assert (summary["brier_mean"] >= 0).all()
    # Multiclass Brier sums over classes, so its ceiling is 2, not 1.
    assert (summary["brier_mean"] <= 2).all()
    for task, classes in n_classes.items():
        block = summary[summary["task"] == task]
        if not len(block):
            continue
        floor = 1.0 / classes
        assert (block["mean_confidence_mean"] >= floor - 1e-9).all(), task


def test_no_calibration_metric_is_suspiciously_perfect() -> None:
    """The standing rule, applied to calibration: a 0.0 ECE is a bug report."""
    summary = _csv("calibration_summary.csv")
    suspicious = summary[(summary["ece_mean"] < 1e-6) | (summary["brier_mean"] < 1e-6)]
    assert suspicious.empty, suspicious.to_string()


# ---------------------------------------------------------------------------
# clause 2 -- sigmoid versus isotonic, for the SVM
# ---------------------------------------------------------------------------


def test_sigmoid_and_isotonic_were_compared_on_the_identical_folds() -> None:
    comparison = _csv("svm_calibration_method_comparison.csv")
    assert set(comparison["method"]) == {"sigmoid", "isotonic"}
    assert set(comparison["model_id"]) == {"M3"}

    folds = comparison.groupby("method")["fold_label"].apply(lambda s: set(s.astype(str)))
    assert folds.loc["sigmoid"] == folds.loc["isotonic"]
    assert len(folds.loc["sigmoid"]) == 25, "the comparison must run the full 5x5 map"

    # Same folds means the same held-out rows, or "identical folds" is a claim
    # about labels rather than about data.
    sizes = comparison.pivot_table(index="fold_label", columns="method", values="n_test")
    assert (sizes["sigmoid"] == sizes["isotonic"]).all()
    assert comparison["brier"].notna().all()
    assert comparison["ece"].notna().all()


def test_the_two_calibrators_actually_produced_different_models() -> None:
    """A comparison where both arms agree exactly did not compare anything."""
    comparison = _csv("svm_calibration_method_comparison.csv")
    wide = comparison.pivot_table(index="fold_label", columns="method", values="brier")
    assert not np.allclose(wide["sigmoid"], wide["isotonic"])


# ---------------------------------------------------------------------------
# clause 3 -- the chosen calibration is recorded
# ---------------------------------------------------------------------------


def test_the_chosen_calibration_is_recorded_and_names_the_final_model() -> None:
    import json

    from src.models.calibration import CALIBRATION_METHODS
    from src.utils.evidence import PROJECT_ROOT

    path = _root() / "final_model_calibration.json"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/30_calibration_analysis.py")
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["svm_calibration_method"] in CALIBRATION_METHODS
    assert record["svm_calibration_source"] == "configs/models.yaml calibration block"
    assert record["final_model_has_post_hoc_calibrator"] is False
    assert record["ensembles_carrying_the_svm"] == ["M6", "M7"]

    manifest = json.loads(
        (PROJECT_ROOT / "models_saved" / "binary" / "final" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["final_model_id"] == manifest["selected_model_id"]

    # The recorded method has to be the one the config would build, not a
    # remembered value: a config edit that nobody re-ran this script after must
    # fail here rather than leave a stale claim in an output.
    from src.models.calibration import calibration_settings

    assert record["svm_calibration_method"] == calibration_settings()["method"]


def test_t23_states_which_calibration_the_final_model_uses() -> None:
    import json

    from src.utils.evidence import PROJECT_ROOT

    meta_path = (
        PROJECT_ROOT / SECTION / "T23_calibration_and_confidence_summary.meta.json"
    )
    if not meta_path.is_file():
        pytest.skip(str(meta_path) + " does not exist; run scripts/30_calibration_analysis.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    notes = " ".join(meta.get("notes", []))
    assert "T78.4" in notes
    assert "sigmoid" in notes and "isotonic" in notes
    assert "FINAL BINARY MODEL" in notes
    assert "screening" in notes.lower()


# ---------------------------------------------------------------------------
# the artifacts
# ---------------------------------------------------------------------------


def test_t23_exists_and_matches_the_summary_it_was_built_from() -> None:
    table = _section_csv("T23_calibration_and_confidence_summary.csv")
    summary = _csv("calibration_summary.csv")
    assert len(table) == len(summary)
    merged = table.merge(
        summary, on=["run", "model_id"], suffixes=("_table", "_source"), how="inner"
    )
    assert len(merged) == len(table)
    assert np.allclose(merged["brier_mean_table"], merged["brier_mean_source"])
    assert np.allclose(merged["ece_mean_table"], merged["ece_mean_source"])


@pytest.mark.parametrize(
    ("figure_id", "slug"),
    [("G33", "G33_confidence_histogram"), ("G34", "G34_calibration_curve")],
)
def test_the_figures_exist_with_the_csv_they_were_drawn_from(figure_id: str, slug: str) -> None:
    import pandas as pd

    from src.reporting.graphs import figures_dir

    directory = figures_dir()
    png = directory / (slug + ".png")
    csv = directory / (slug + ".csv")
    if not png.is_file():
        pytest.skip(str(png) + " does not exist; run scripts/30_calibration_analysis.py")
    assert csv.is_file(), figure_id + " has a PNG with no source CSV"
    frame = pd.read_csv(csv)
    assert len(frame)
    assert not frame.isna().all().all()
    assert png.stat().st_size > 10_000


def test_g34_is_the_confidence_curve_not_the_positive_class_curve() -> None:
    """The two reliability definitions must not be confused for one another.

    `curves.calibration_frame` writes positive-class probability against the
    observed positive rate and is defined for the binary task only. G34's curve
    is confidence against accuracy and exists for four-class PASCAL A too. A
    binary point can appear in both; a 4-class point below 0.25 confidence
    cannot exist in either, and no G34 point may fall below its own floor.
    """
    import pandas as pd

    from src.reporting.graphs import figures_dir

    path = figures_dir() / "G34_calibration_curve.csv"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/30_calibration_analysis.py")
    frame = pd.read_csv(path)
    assert set(frame["kind"]) == {"confidence_vs_accuracy"}
    floors = 1.0 / frame["n_classes"].astype(float)
    assert (frame["confidence_mean"] >= floors - 1e-9).all()
    assert (frame["accuracy_mean"] >= 0).all()
    assert (frame["accuracy_mean"] <= 1).all()
