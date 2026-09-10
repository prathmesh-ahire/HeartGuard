"""T80.7 -- the failure analysis gate.

The gate: **every false positive and false negative is categorised,
cross-referenced to the diagnosis field, and named example records appear in the
report.**

"Every" is checked by reconciling the categorised counts back against the raw
error set rather than by trusting the pipeline that produced both -- a grouping
that silently dropped rows with a missing key would otherwise pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/10_robustness"
RUN_DIR = "failure_analysis"


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION / RUN_DIR


def _csv(name: str) -> Any:
    import pandas as pd

    path = _root() / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/32_failure_analysis.py")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# the module, on real predictions
# ---------------------------------------------------------------------------


def test_error_types_partition_every_prediction() -> None:
    """TP/TN/FP/FN with nothing left over -- a '?' row means a label outside {0,1}."""
    from src.evaluation.failure_analysis import ERROR_TYPES

    frame = _csv("binary_predictions_labelled.csv")
    assert set(frame["error_type"]) <= set(ERROR_TYPES)
    assert not (frame["error_type"] == "?").any()
    expected = frame["y_true"].to_numpy() != frame["y_pred"].to_numpy()
    assert (frame["is_error"].to_numpy().astype(bool) == expected).all()


def test_confidence_is_in_the_predicted_class_not_the_positive_class() -> None:
    """What a dashboard shows beside the answer, not P(abnormal).

    For a 'normal' prediction the positive-class probability is the model's
    confidence in being WRONG, and binning errors by it would invert the panel.
    """
    frame = _csv("binary_predictions_labelled.csv")
    assert (frame["confidence"] >= 0.5 - 1e-9).all()
    positive = frame[frame["y_pred"] == 1]
    negative = frame[frame["y_pred"] == 0]
    assert np.allclose(positive["confidence"], positive["proba_1"], atol=1e-9)
    assert np.allclose(negative["confidence"], 1.0 - negative["proba_1"], atol=1e-9)


# ---------------------------------------------------------------------------
# clause 1 -- every error is categorised
# ---------------------------------------------------------------------------


def test_every_error_is_accounted_for_in_every_grouping() -> None:
    """Counts reconcile back to the raw error set, per source and per grouping."""
    from src.evaluation.failure_analysis import CATEGORIES

    frame = _csv("binary_predictions_labelled.csv")
    categorised = _csv("failures_by_category.csv")

    for source, block in frame.groupby("source"):
        total_errors = int(block["is_error"].sum())
        for category in CATEGORIES:
            part = categorised[
                (categorised["source"] == source) & (categorised["category"] == category)
            ]
            assert len(part), source + " has no " + category + " grouping"
            assert int(part["n_errors"].sum()) == total_errors, (source, category)
            assert int(part["n_predictions"].sum()) == len(block), (source, category)
            assert int(part["n_fp"].sum() + part["n_fn"].sum()) == total_errors


def test_the_error_set_holds_only_errors_and_holds_all_of_them() -> None:
    frame = _csv("binary_predictions_labelled.csv")
    errors = _csv("false_positives_and_negatives.csv")
    assert len(errors) == int(frame["is_error"].sum())
    assert errors["is_error"].all()
    assert set(errors["error_type"]) <= {"FP", "FN"}


def test_the_two_rates_use_the_denominators_they_claim() -> None:
    categorised = _csv("failures_by_category.csv")
    positive = categorised[categorised["n_positive"] > 0]
    negative = categorised[categorised["n_negative"] > 0]
    assert np.allclose(
        positive["false_negative_rate"], positive["n_fn"] / positive["n_positive"], atol=1e-9
    )
    assert np.allclose(
        negative["false_positive_rate"], negative["n_fp"] / negative["n_negative"], atol=1e-9
    )
    assert np.allclose(
        categorised["error_rate"], categorised["n_errors"] / categorised["n_predictions"]
    )


def test_the_reporting_floor_is_applied_to_each_rate_own_denominator() -> None:
    """The training-c case: 31 records, above the floor, but only 7 of them normal.

    Flooring on the group total alone published a false-positive rate of 1.000
    over seven recordings without a mark on it.
    """
    from src.evaluation.failure_analysis import MIN_GROUP

    categorised = _csv("failures_by_category.csv")
    for column in ("fn_below_reporting_floor", "fp_below_reporting_floor"):
        assert column in categorised.columns
    assert (
        categorised["fn_below_reporting_floor"].astype(bool)
        == (categorised["n_positive_records"] < MIN_GROUP)
    ).all()
    assert (
        categorised["fp_below_reporting_floor"].astype(bool)
        == (categorised["n_negative_records"] < MIN_GROUP)
    ).all()

    # No rate at 1.0 may be unflagged unless its denominator really is large.
    extreme = categorised[(categorised["false_positive_rate"] >= 0.999)]
    for _, row in extreme.iterrows():
        assert bool(row["fp_below_reporting_floor"]) or int(row["n_negative_records"]) >= MIN_GROUP


def test_a_record_wrong_on_every_fold_is_distinguished_from_one_wrong_once() -> None:
    records = _csv("record_failures.csv")
    assert {"n_wrong", "n_evaluations", "consistently_wrong"} <= set(records.columns)
    assert (records["n_wrong"] <= records["n_evaluations"]).all()
    assert (
        records["consistently_wrong"].astype(bool)
        == (records["n_wrong"] == records["n_evaluations"])
    ).all()

    in_domain = records[records["source"].str.contains("in-domain")]
    if len(in_domain):
        assert (in_domain["n_evaluations"] == 5).all(), "the repeated map scores each record 5x"
        # Both kinds must actually exist, or the distinction is untested.
        wrong = in_domain[in_domain["n_wrong"] > 0]
        assert wrong["consistently_wrong"].any()
        assert (~wrong["consistently_wrong"].astype(bool)).any()


# ---------------------------------------------------------------------------
# clause 2 -- cross-referenced to the diagnosis field
# ---------------------------------------------------------------------------


def test_failures_are_cross_referenced_to_the_physionet_diagnosis_field() -> None:
    import pandas as pd

    from src.utils.config import load_config

    categorised = _csv("failures_by_category.csv")
    block = categorised[categorised["category"] == "diagnosis_class"]
    assert len(block), "no diagnosis cross-reference was produced"

    master = pd.read_csv(
        pd.io.common.stringify_path(
            load_config("paths").require("outputs.dataset_audit") + "/metadata_master.csv"
        ),
        low_memory=False,
    )
    real = set(master.loc[master["dataset_source"] == "D1", "diagnosis_class"].dropna())
    assert set(block["level"]) <= real, "a diagnosis level appeared that the corpus does not have"
    # The pathologies actually missed must be identifiable, not just counted.
    assert block["false_negative_rate"].notna().any()
    assert (block["n_records"] > 0).all()


def test_the_confused_pairs_cover_every_multiclass_run_that_exists() -> None:
    from src.evaluation.failure_analysis import MULTICLASS_RUNS
    from src.utils.evidence import PROJECT_ROOT

    pairs = _csv("confused_class_pairs.csv")
    expected = {
        run
        for run, directory in MULTICLASS_RUNS
        if (PROJECT_ROOT / directory / "confusion_matrices.json").is_file()
    }
    assert set(pairs["run"]) == expected
    assert (pairs["true_class"] != pairs["predicted_class"]).all()
    assert (pairs["share_of_true_class"] > 0).all()
    assert (pairs["share_of_true_class"] <= 1.0 + 1e-9).all()
    assert (pairs["rank_in_run_model"] >= 1).all()


# ---------------------------------------------------------------------------
# clause 3 -- named example records appear in the report
# ---------------------------------------------------------------------------


def test_the_narrative_report_names_real_records_that_were_really_wrong() -> None:
    from src.utils.evidence import PROJECT_ROOT

    path = _root() / "failure_report.md"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/32_failure_analysis.py")
    text = path.read_text(encoding="utf-8")

    examples = _csv("named_example_records.csv")
    assert len(examples) >= 5
    for record_uid in examples["record_uid"]:
        assert str(record_uid) in text, str(record_uid) + " is not named in the report"

    # The named records must be real corpus records that really were errors.
    import pandas as pd

    master = pd.read_csv(
        PROJECT_ROOT / "outputs/01_dataset_audit/metadata_master.csv", low_memory=False
    )
    assert set(examples["record_uid"]) <= set(master["record_uid"])
    assert (examples["n_wrong"] > 0).all()
    assert examples["selection_rule"].notna().all()

    assert "screening" in text.lower()
    assert "not a diagnostic tool" in text
    assert "nan%" not in text, "a rate that does not exist was printed as nan%"


def test_the_report_regenerates_identically(tmp_path) -> None:
    """Rule 5 applied to prose: the narrative must not drift between runs."""
    from src.reporting.failure_report import write_failure_report

    first = _root() / "failure_report.md"
    if not first.is_file():
        pytest.skip(str(first) + " does not exist; run scripts/32_failure_analysis.py")

    second = write_failure_report(
        tmp_path / "failure_report.md",
        categorised=_csv("failures_by_category.csv"),
        records=_csv("record_failures.csv"),
        examples=_csv("named_example_records.csv"),
        pairs=_csv("confused_class_pairs.csv"),
        sources=tuple(
            line.strip().strip("- `")
            for line in first.read_text(encoding="utf-8").splitlines()
            if line.startswith("- `")
        ),
        command="python scripts/32_failure_analysis.py",
    )
    assert second.read_text(encoding="utf-8") == first.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the artifacts
# ---------------------------------------------------------------------------


def test_t27_exists_and_reconciles_with_the_frame_behind_it() -> None:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    root = PROJECT_ROOT / SECTION
    csv = root / "T27_false_positive_and_false_negative_analysis.csv"
    if not csv.is_file():
        pytest.skip(str(csv) + " does not exist; run scripts/32_failure_analysis.py")
    table = pd.read_csv(csv)
    source = _csv("failures_by_category.csv")
    assert len(table) == len(source)
    for suffix in (".docx", ".tex", ".md", ".meta.json"):
        assert (root / ("T27_false_positive_and_false_negative_analysis" + suffix)).is_file()


def test_g35_exists_with_the_csv_it_was_drawn_from() -> None:
    import pandas as pd

    from src.reporting.graphs import figures_dir

    slug = "G35_false_positive_and_false_negative_distribution"
    directory = figures_dir()
    png = directory / (slug + ".png")
    if not png.is_file():
        pytest.skip(str(png) + " does not exist; run scripts/32_failure_analysis.py")
    frame = pd.read_csv(directory / (slug + ".csv"))
    assert len(frame)
    assert "fn_below_reporting_floor" in frame.columns
    assert png.stat().st_size > 10_000
