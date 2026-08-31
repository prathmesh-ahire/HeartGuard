"""T68.7 and T69.7 -- the CirCor murmur and outcome gates.

Split the same way as the other experiment gates: anything reading a committed
CSV or JSON runs on CI, anything needing a ``predictions.parquet`` skips, because
``*.parquet`` is gitignored and CI has never seen one.

The two gates in a sentence each:

* **T68.7** -- both murmur variants ran, the three-class one including the 68
  Unknown patients and the two-class one over the 874 known, and both are
  reported rather than one being quietly preferred.
* **T69.7** -- the outcome labels came from the per-patient ``.txt`` files and not
  from ``training_data.csv``, patient-level aggregation was evaluated, and the
  noise that label propagation introduces is documented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/08_circor_external_validation"
CIRCOR_MODELS = ("M3", "M4", "M5", "M6", "M7")
MURMUR_RUNS = ("EXP-C1-three_class", "EXP-C1-two_class")


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(*parts: str) -> Any:
    import pandas as pd

    path = _root().joinpath(*parts)
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run the CirCor experiments")
    return pd.read_csv(path)


def _parquet(*parts: str) -> Any:
    import pandas as pd

    path = _root().joinpath(*parts)
    if not path.is_file():
        pytest.skip(str(path) + " is gitignored; run the experiment locally")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# configuration -- runs on CI
# ---------------------------------------------------------------------------


def test_the_murmur_variants_are_declared_as_two_separate_label_spaces() -> None:
    """T68.2 -- three-class and two-class are different questions, not a switch."""
    from src.evaluation.experiment import Experiment

    exp = Experiment.load("EXP-C1")
    assert set(exp.variants) == {"three_class", "two_class"}

    three = exp.for_variant("three_class")
    two = exp.for_variant("two_class")
    assert three.n_classes == 3
    assert two.n_classes == 2
    assert "Unknown" in three.label_space
    assert "Unknown" not in two.label_space
    # Different folders, or one would overwrite the other.
    assert three.output_dir().name != two.output_dir().name


def test_the_outcome_label_source_is_declared_as_the_txt_files() -> None:
    """T69.1 -- the label is NOT in training_data.csv, and the config says so.

    A loader built around the CSV produces a murmur-only pipeline and this
    experiment dies. The declaration is what stops that being rediscovered.
    """
    from src.evaluation.experiment import Experiment

    spec = Experiment.load("EXP-C2").spec
    assert spec.get("label_source") == "patient_txt_files"
    assert spec.get("label_field") == "#Outcome"
    assert spec.get("label_space") == {"Normal": 0, "Abnormal": 1}


def test_a_nested_search_can_now_be_given_a_wall_clock_ceiling() -> None:
    """Phase 68 fix -- `Budget.max_seconds` existed but nothing reached it.

    Every nested search before this ran unbounded, so a fold whose search drew an
    expensive gradient-boosting point was indistinguishable from a hang.
    """
    from src.evaluation.tuned import NestedSearchPlanner

    assert NestedSearchPlanner().max_seconds is None  # unchanged default
    assert NestedSearchPlanner(max_seconds=90).max_seconds == 90.0
    # The ceiling must reach the cache key, or two different budgets would share
    # one cached search and the second would silently inherit the first's.
    import inspect

    source = inspect.getsource(NestedSearchPlanner._search_key)
    assert "max_seconds" in source


# ---------------------------------------------------------------------------
# the fold map -- committed CSV, runs on CI
# ---------------------------------------------------------------------------


def _fold_map() -> Any:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "subject_split_map.csv"
    if not path.is_file():
        pytest.skip("DA-07 fold map missing")
    return pd.read_csv(path)


def test_circor_is_patient_grouped_with_no_patient_on_both_sides() -> None:
    """Rule 3 -- CirCor patient ids are native, so there is no excuse for a leak."""
    frame = _fold_map()
    block = frame[frame["task"] == "circor_murmur"]
    assert block["record_uid"].nunique() == 3163
    assert block["split_group"].nunique() == 942

    records = block.drop_duplicates("record_uid")[["record_uid", "split_group"]]
    group_of = dict(zip(records["record_uid"], records["split_group"], strict=True))
    for fold, chunk in block.groupby("fold"):
        test_uids = set(chunk["record_uid"])
        train_uids = set(block["record_uid"]) - test_uids
        shared = {group_of[u] for u in test_uids} & {group_of[u] for u in train_uids}
        assert not shared, "fold " + str(fold) + ": " + str(len(shared)) + " patient(s) leak"


def test_the_outcome_task_carries_the_486_456_patient_split() -> None:
    """T69.7 clause 1 -- the counts that prove the txt files were parsed."""
    frame = _fold_map()
    block = frame[frame["task"] == "circor_outcome"].drop_duplicates("split_group")
    counts = block["class_name"].value_counts().to_dict()
    assert counts.get("Normal") == 486, counts
    assert counts.get("Abnormal") == 456, counts
    assert sum(counts.values()) == 942


# ---------------------------------------------------------------------------
# gates over what is on disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run", MURMUR_RUNS)
def test_both_murmur_variants_ran_over_the_full_grid(run: str) -> None:
    """T68.7 -- both variants, five models, five folds each, nothing missing."""
    frame = _csv(run, "per_fold_metrics.csv")
    assert set(frame["model_id"]) == set(CIRCOR_MODELS)
    for model_id, block in frame.groupby("model_id"):
        assert len(block) == 5, model_id + " has " + str(len(block)) + " folds"


def test_the_three_class_variant_kept_the_unknown_patients() -> None:
    """T68.7 -- the headline variant is the one that includes Unknown."""
    frame = _csv("EXP-C1-three_class", "per_fold_metrics.csv")
    assert "recall_Unknown" in frame.columns
    assert (np.asarray(frame["support_Unknown"], dtype=float) > 0).all()


def test_the_two_class_variant_dropped_exactly_the_unknown_records() -> None:
    """T68.2 -- 874 of 942 patients, and the drop is the Unknown ones only."""
    frame = _fold_map()
    murmur = frame[frame["task"] == "circor_murmur"].drop_duplicates("record_uid")
    unknown = int((murmur["class_name"] == "Unknown").sum())
    predictions = _parquet("EXP-C1-two_class", "predictions.parquet")
    per_fold_records = int(predictions.groupby("model_id")["record_uid"].nunique().iloc[0])
    assert per_fold_records == len(murmur) - unknown
    assert set(predictions["y_true"]) <= {0, 1}
    assert "recall_Unknown" not in _csv("EXP-C1-two_class", "per_fold_metrics.csv").columns


def test_both_murmur_variants_appear_in_t13() -> None:
    """T68.7 -- 'report both; do not silently pick one', checked on the table."""
    t13 = _csv("T13_circor_murmur_results.csv")
    assert set(t13["run"]) == set(MURMUR_RUNS), sorted(set(t13["run"]))
    for run in MURMUR_RUNS:
        assert set(t13[t13["run"] == run]["model_id"]) == set(CIRCOR_MODELS)


@pytest.mark.parametrize(
    ("table", "expected_runs"),
    [
        ("T13_circor_murmur_results.csv", MURMUR_RUNS),
        ("T14_circor_outcome_results.csv", ("EXP-C2",)),
    ],
)
def test_every_aggregation_rule_was_evaluated(table: str, expected_runs: tuple) -> None:
    """T68.4 / T69.4 -- all three rules, at patient level, for every model."""
    from src.evaluation.aggregation import AGGREGATION_RULES

    frame = _csv(table)
    assert set(frame["run"]) == set(expected_runs)
    patient = frame[frame["level"] == "patient"]
    assert set(patient["rule"]) == set(AGGREGATION_RULES), sorted(set(patient["rule"]))
    assert "recording" in set(frame["level"])
    for run in expected_runs:
        block = patient[patient["run"] == run]
        for rule in AGGREGATION_RULES:
            assert set(block[block["rule"] == rule]["model_id"]) == set(CIRCOR_MODELS)


def test_patient_level_covers_every_patient_exactly_once_per_fold() -> None:
    """The aggregation must not drop or duplicate a patient."""
    from src.evaluation.aggregation import aggregate_predictions

    predictions = _parquet("EXP-C2", "predictions.parquet")
    patients = aggregate_predictions(predictions, rule="max")
    for (model_id, fold_label), block in patients.groupby(["model_id", "fold_label"]):
        assert block["patient_id"].is_unique, str(model_id) + " " + str(fold_label)
    total = patients[patients["model_id"] == "M3"]["patient_id"].nunique()
    assert total == 942, total
    assert int(patients["n_recordings"].sum() / patients["model_id"].nunique()) == 3163


def test_t15_reports_the_signed_cost_of_aggregating() -> None:
    """T68.6 -- what patient-level aggregation actually buys, per rule."""
    t15 = _csv("T15_recording_vs_patient_level.csv")
    for column in ("sensitivity_recording", "sensitivity_patient", "sensitivity_delta"):
        assert column in t15.columns
    # The deltas must be internally consistent, not independently computed.
    # `sensitivity` is a binary metric, so it is legitimately NaN for the
    # three-class murmur variant -- but ONLY there, and that is asserted rather
    # than tolerated, otherwise a NaN could hide a genuine mismatch.
    stored = np.asarray(t15["sensitivity_delta"], dtype=float)
    delta = np.asarray(t15["sensitivity_patient"], dtype=float) - np.asarray(
        t15["sensitivity_recording"], dtype=float
    )
    missing = ~np.isfinite(stored)
    assert set(t15.loc[missing, "run"]) <= {"EXP-C1-three_class"}, (
        "sensitivity is undefined outside the three-class variant: "
        + ", ".join(sorted(set(t15.loc[missing, "run"])))
    )
    assert np.allclose(delta, stored, equal_nan=True)
    assert np.isfinite(stored[~missing]).all()
    # And the binary runs must actually be present, or the check above is vacuous.
    assert (~missing).sum() > 0


def test_the_label_propagation_caveat_is_on_the_record() -> None:
    """T69.2 / T69.7 -- the noise propagation introduces must be documented.

    CirCor labels a patient; the model scores a recording. Giving every recording
    its patient's label asserts a murmur is audible at every location it was
    recorded from, which is false -- and that mislabelling is inside the training
    data, not just the evaluation.
    """
    path = _root() / "circor_label_propagation.md"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/18_circor_tables.py")
    text = path.read_text(encoding="utf-8")
    assert "#Outcome" in text
    assert "486" in text and "456" in text
    assert "propagat" in text.lower()
    assert "not a diagnostic tool" in text.lower()


@pytest.mark.parametrize("run", (*MURMUR_RUNS, "EXP-C2"))
def test_no_circor_metric_is_suspiciously_perfect(run: str) -> None:
    """The standing near-perfect rule, applied before these ship.

    Specificity legitimately reaches 0.996 on the two-class murmur task, and that
    is NOT leakage: M3 predicts Present for 5.9% of recordings against a true
    prevalence of 20.5%, so near-perfect specificity is the arithmetic
    consequence of rarely using the positive class. The check is therefore on
    balanced accuracy and AUC, which a degenerate predictor cannot inflate.
    """
    frame = _csv(run, "per_fold_metrics.csv")
    for column in ("balanced_accuracy", "macro_f1", "roc_auc", "ovr_auc_macro"):
        if column not in frame.columns:
            continue
        worst = float(np.nanmax(np.asarray(frame[column], dtype=float)))
        assert worst < 0.99, run + " " + column + " reaches " + format(worst, ".4f")
