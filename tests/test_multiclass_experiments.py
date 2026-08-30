"""T66.7 and T67.7 -- the PASCAL A and PASCAL B gates.

Structured the way the binary gates are. Everything that reads a committed CSV,
JSON or markdown file runs on CI; everything that needs the feature matrix or a
``predictions.parquet`` skips -- never passes -- because ``*.parquet`` is
gitignored and CI has never seen one. That distinction is not cosmetic: three
red builds in this project have come from a test asserting the existence of a
gitignored input.

The two gates in one sentence each:

* **T66.7** -- repeated 5x2 CV really ran, per-class recall exists for all four
  classes, and every headline metric carries a confidence interval given n=124.
* **T67.7** -- subject-grouped CV really used the subject groups with no subject
  on both sides of a fold, and the report states explicitly that sets A and B
  were never merged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

PASCAL_A_CLASSES = ("normal", "murmur", "extrahls", "artifact")
PASCAL_B_CLASSES = ("normal", "murmur", "extrastole")
MULTICLASS_MODELS = ("M1", "M3", "M4", "M5", "M6", "M7")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _experiment(exp_id: str) -> Any:
    from src.evaluation.experiment import Experiment

    return Experiment.load(exp_id)


def _committed(exp_id: str, name: str) -> Any:
    import pandas as pd

    path = _experiment(exp_id).output_dir() / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/11_run_experiment.py --exp " + exp_id)
    return pd.read_csv(path)


def _table(filename: str) -> Any:
    import pandas as pd

    path = _experiment("EXP-B1").output_dir().parent / filename
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/15_multiclass_tables.py")
    return pd.read_csv(path)


def _text(filename: str) -> str:
    path = _experiment("EXP-B1").output_dir().parent / filename
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/15_multiclass_tables.py")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# configuration only -- runs on CI
# ---------------------------------------------------------------------------


def test_the_two_pascal_tasks_are_separate_label_spaces() -> None:
    """Rule 4, checked on the declarations rather than trusted."""
    b1 = _experiment("EXP-B1")
    b2 = _experiment("EXP-B2")

    assert b1.task == "pascal_a"
    assert b2.task == "pascal_b"
    assert b1.task != b2.task
    assert set(b1.label_space) == set(PASCAL_A_CLASSES)
    assert set(b2.label_space) == set(PASCAL_B_CLASSES)
    assert b1.cv != b2.cv, "two tasks sharing one CV scheme would share a fold map"


def test_pascal_a_uses_repeated_5x2_because_of_the_19_record_class() -> None:
    """T66.2 -- 5 repeats of 2 folds, not 5 folds.

    Five folds would put roughly four ``extrahls`` records in a test fold. The
    scheme is declared, so this asserts the declaration rather than re-deriving
    the reasoning.
    """
    scheme = _experiment("EXP-B1").cv_scheme()
    assert int(scheme["n_splits"]) == 2
    assert int(scheme["n_repeats"]) == 5
    assert int(scheme["total_folds"]) == 10


def test_the_declared_class_weight_is_verified_not_decorative() -> None:
    """``class_weight`` in experiments.yaml is checked against configs/models.yaml.

    Before Phase 66 the field was parsed into ``Experiment.class_weight`` and
    never read again, so editing it looked like it did something and did not.
    """
    import dataclasses

    from src.evaluation.experiment import ExperimentError, assert_declared_class_weight

    for exp_id in ("EXP-B1", "EXP-B2"):
        exp = _experiment(exp_id)
        assert exp.class_weight == "balanced"
        assert_declared_class_weight(exp)  # must not raise

    # A declaration the models do not honour must stop the run.
    impossible = dataclasses.replace(_experiment("EXP-B1"), class_weight="not_a_real_weight")
    with pytest.raises(ExperimentError, match="class_weight"):
        assert_declared_class_weight(impossible)


def test_both_tracks_declare_every_mandatory_model() -> None:
    """T66.3 / T67.3 -- "all mandatory models", resolved against models.yaml.

    M8 is ``mandatory: false`` (an optional extra the binary track runs), so the
    six declared here are the complete mandatory set rather than a subset of it.
    """
    from src.utils.config import load_config

    catalogue = load_config("models").require("models")
    mandatory = {
        model_id
        for model_id, spec in catalogue.items()
        if bool(spec.get("mandatory"))
    }
    assert mandatory == set(MULTICLASS_MODELS)
    for exp_id in ("EXP-B1", "EXP-B2"):
        declared = set(_experiment(exp_id).models)
        assert mandatory <= declared, exp_id + " is missing " + ", ".join(
            sorted(mandatory - declared)
        )


def test_every_multiclass_model_actually_carries_balanced_weights() -> None:
    """T66.3 -- the requirement is about the estimators, so check the estimators."""
    from src.evaluation.experiment import _resolved_class_weight

    for model_id in MULTICLASS_MODELS:
        source, value = _resolved_class_weight(model_id)
        assert value == "balanced", source + " -> " + repr(value)


# ---------------------------------------------------------------------------
# the fold map -- committed CSV, runs on CI
# ---------------------------------------------------------------------------


def _fold_map() -> Any:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "subject_split_map.csv"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run the Phase 43 fold-map builder")
    return pd.read_csv(path)


def test_pascal_a_fold_map_is_five_repeats_of_two_folds() -> None:
    """T66.7 clause 1 -- repeated 5x2 CV ran."""
    frame = _fold_map()
    block = frame[frame["task"] == "pascal_a"]
    assert sorted(block["repeat"].unique()) == [0, 1, 2, 3, 4]
    assert sorted(block["fold"].unique()) == [0, 1]
    assert block["record_uid"].nunique() == 124
    # Each record is tested exactly once per repeat, which is what makes a
    # single repeat a duplicate-free out-of-fold prediction set.
    for repeat, chunk in block.groupby("repeat"):
        assert chunk["record_uid"].nunique() == 124, "repeat " + str(repeat)
        assert len(chunk) == 124, "repeat " + str(repeat) + " lists a record twice"


def test_pascal_a_records_no_subject_ids_rather_than_inventing_them() -> None:
    """Rule 3 -- where an ID cannot be derived, that is recorded, not faked."""
    frame = _fold_map()
    block = frame[frame["task"] == "pascal_a"]
    assert not block["subject_derived"].any()
    # The one duplicated recording Phase 17 found shares a group so grouped CV
    # cannot put the same audio on both sides. 124 records, 123 groups.
    assert block["split_group"].nunique() == 123


def test_pascal_b_grouped_cv_used_165_subjects_not_167_sessions() -> None:
    """T67.7 clause 1, with the gate's own number deliberately corrected.

    The task list says 167. That is the count of recording *sessions*: three
    subjects were recorded twice on the same day, two of them with both sessions
    labelled. Grouping on the session would put one person on both sides of a
    fold. The decision to group on the subject number -- and to change gate
    T13.7's 167 to 165 -- was taken on 2026-08-25 and is recorded in
    ``Docs/note.md``. This test pins the corrected figure so the deviation
    cannot quietly revert.
    """
    frame = _fold_map()
    block = frame[frame["task"] == "pascal_b"]
    assert block["record_uid"].nunique() == 461
    assert block["subject_derived"].all()
    assert block["split_group"].nunique() == 165
    assert block["subject_id"].nunique() == 165


def test_no_pascal_b_subject_appears_on_both_sides_of_a_fold() -> None:
    """Rule 3 measured on the map, per fold, rather than assumed from the scheme."""
    frame = _fold_map()
    block = frame[frame["task"] == "pascal_b"]
    records = block.drop_duplicates("record_uid")[["record_uid", "split_group"]]
    group_of = dict(zip(records["record_uid"], records["split_group"], strict=True))

    for (repeat, fold), chunk in block.groupby(["repeat", "fold"]):
        test_uids = set(chunk["record_uid"])
        train_uids = set(block["record_uid"]) - test_uids
        shared = {group_of[uid] for uid in test_uids} & {group_of[uid] for uid in train_uids}
        assert not shared, (
            "r" + str(repeat) + "f" + str(fold) + ": " + str(len(shared)) + " subject(s) leak"
        )


def test_no_record_belongs_to_both_pascal_tasks() -> None:
    """Rule 4, measured. This is the check that would catch an actual merge."""
    from src.reporting.pascal_statements import verify_sets_never_merged

    facts = verify_sets_never_merged(_fold_map())
    assert facts["shared_records"] == 0
    assert facts["n_records_a"] == 124
    assert facts["n_records_b"] == 461
    # A shared class NAME is expected and is not a merge -- see the docstring on
    # verify_sets_never_merged.
    assert set(facts["shared_class_names"]) == {"normal", "murmur"}


# ---------------------------------------------------------------------------
# interval arithmetic -- pure, runs on CI
# ---------------------------------------------------------------------------


def test_the_fold_interval_reports_its_own_n() -> None:
    from src.reporting.multiclass_report import fold_interval

    interval = fold_interval([0.5, 0.6, 0.7, 0.8, 0.9])
    assert interval["n"] == 5
    assert interval["lo"] < interval["mean"] < interval["hi"]

    # A NaN is excluded from n rather than propagated into the mean.
    with_nan = fold_interval([0.5, float("nan"), 0.7])
    assert with_nan["n"] == 2
    assert np.isfinite(with_nan["mean"])

    # One value cannot support an interval, and must say so rather than
    # returning a zero-width one.
    single = fold_interval([0.5])
    assert single["n"] == 1
    assert not np.isfinite(single["lo"])


def test_the_record_bootstrap_does_not_pool_repeats() -> None:
    """Pooling five repeats would enter each record five times and shrink the CI.

    Built so the pooled frame is exactly five copies of one repeat: a bootstrap
    that pooled them would see n=500 instead of n=100 and return an interval
    roughly sqrt(5) narrower. The per-repeat interval must be unchanged by the
    duplication.
    """
    import pandas as pd

    from src.reporting.multiclass_report import record_interval

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 100)
    y_pred = np.where(rng.random(100) < 0.7, y_true, (y_true + 1) % 3)

    one = pd.DataFrame({"repeat": 0, "y_true": y_true, "y_pred": y_pred})
    five = pd.concat([one.assign(repeat=repeat) for repeat in range(5)], ignore_index=True)

    single = record_interval(one, labels=(0, 1, 2), metrics=["accuracy"], n_resamples=400)[
        "accuracy"
    ]
    repeated = record_interval(five, labels=(0, 1, 2), metrics=["accuracy"], n_resamples=400)[
        "accuracy"
    ]

    assert single["n_records"] == 100
    assert repeated["n_records"] == 100, "the bootstrap saw a pooled frame, not a repeat"
    single_width = single["hi"] - single["lo"]
    repeated_width = repeated["hi"] - repeated["lo"]
    assert abs(single_width - repeated_width) < 0.25 * single_width


def test_the_fast_metrics_match_sklearn_exactly() -> None:
    """The bootstrap derives six metrics from one confusion matrix instead of
    calling sklearn six times per draw. That is only safe if the arithmetic is
    identical, including the awkward cases: a class absent from a draw, a class
    predicted but never true, and balanced accuracy's different averaging.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    from src.reporting.multiclass_report import confusion_metrics

    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        n_classes = int(rng.integers(2, 5))
        labels = list(range(n_classes))
        size = int(rng.integers(5, 120))
        y_true = rng.integers(0, n_classes, size)
        y_pred = rng.integers(0, n_classes, size)
        fast = confusion_metrics(y_true, y_pred, labels=labels)
        reference = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_precision": precision_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            ),
        }
        for name, expected in reference.items():
            worst = max(worst, abs(fast[name] - float(expected)))
    assert worst < 1e-12, "fast metrics drifted from sklearn by " + str(worst)


# ---------------------------------------------------------------------------
# gates over what is on disk, once Phases 66/67 have run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("exp_id", "expected_folds"), [("EXP-B1", 10), ("EXP-B2", 5)])
def test_the_committed_run_covers_every_fold_and_model(exp_id: str, expected_folds: int) -> None:
    """T66.7 / T67.7 -- the shipped table is the whole grid, not one pass of it.

    This is the check the EXP-A2 two-pass truncation would have failed: a second
    run with ``--models M6 M7`` overwrote a table that had held seven models.
    """
    frame = _committed(exp_id, "per_fold_metrics.csv")
    exp = _experiment(exp_id)
    assert set(frame["model_id"]) == set(exp.models), (
        exp_id + " table holds " + ", ".join(sorted(set(frame["model_id"])))
    )
    for model_id, block in frame.groupby("model_id"):
        assert len(block) == expected_folds, model_id + " has " + str(len(block)) + " folds"
        assert block["fold_label"].nunique() == expected_folds


@pytest.mark.parametrize(
    ("exp_id", "classes"),
    [("EXP-B1", PASCAL_A_CLASSES), ("EXP-B2", PASCAL_B_CLASSES)],
)
def test_per_class_recall_exists_for_every_class(exp_id: str, classes: tuple) -> None:
    """T66.7 clause 2 / rule 6 -- per-class recall, for all of them, not the macro alone."""
    frame = _committed(exp_id, "per_fold_metrics.csv")
    for name in classes:
        column = "recall_" + name
        assert column in frame.columns, "no " + column + " in " + exp_id
        values = np.asarray(frame[column], dtype=float)
        assert np.isfinite(values).all(), column + " holds a non-finite value"
        support = np.asarray(frame["support_" + name], dtype=float)
        assert (support > 0).all(), name + " has an empty test fold somewhere"


def test_the_multiclass_search_optimizes_macro_f1_not_a_binary_objective() -> None:
    """Rule 4 reaches the search too: a multiclass task must not be scored as binary."""
    from src.optimization.base import objective_for

    for task in ("pascal_a", "pascal_b"):
        objective = objective_for(task)
        assert objective.kind == "multiclass"
        assert objective.name == "macro_f1"
    assert objective_for("binary").name == "balanced_accuracy"


@pytest.mark.parametrize("exp_id", ["EXP-B1", "EXP-B2"])
def test_every_shipped_row_was_tuned_inside_its_own_training_fold(exp_id: str) -> None:
    """The nested-CV claim, checked on every row rather than on the command line.

    A row whose ``planner_tuning`` is missing or says ``config defaults`` was not
    tuned inside its fold, and the run cannot be described as nested.
    """
    import json

    frame = _committed(exp_id, "per_fold_metrics.csv")
    assert "planner_tuning" in frame.columns, exp_id + " carries no tuning provenance"
    tuning = frame["planner_tuning"].astype(str)
    assert (tuning == "nested-bayes").all(), sorted(set(tuning))

    # The two kinds of row carry different evidence, because an ensemble runs one
    # search PER MEMBER rather than one of its own: an individual model records
    # the trial count it spent, an ensemble records every member's inner score.
    ensembles = frame["model_id"].isin(("M6", "M7"))

    trials = np.asarray(frame.loc[~ensembles, "planner_n_trials"], dtype=float)
    assert trials.size, "no individual-model rows to check"
    assert (trials > 0).all(), "an individual model records zero search trials"

    members = ("M3", "M4", "M5")
    for _, row in frame[ensembles].iterrows():
        scores = json.loads(str(row["planner_inner_scores"]))
        assert set(scores) == set(members), row["model_id"] + " " + row["fold_label"]
        assert all(np.isfinite(float(v)) for v in scores.values())
        # Stronger than a trial count: the member scores must equal what the
        # standalone member scored on the SAME fold, which is what proves the
        # ensemble reused that fold's search rather than running its own.
        for member in members:
            standalone = frame[
                (frame["model_id"] == member) & (frame["fold_label"] == row["fold_label"])
            ]
            if standalone.empty:
                continue
            expected = float(standalone["planner_inner_best_score"].iloc[0])
            assert abs(float(scores[member]) - expected) < 1e-9, (
                row["model_id"] + " " + row["fold_label"] + " " + member
            )


@pytest.mark.parametrize("exp_id", ["EXP-B1", "EXP-B2"])
def test_the_multiclass_metric_set_is_complete(exp_id: str) -> None:
    """Rule 6 -- macro-F1, weighted-F1, balanced accuracy and OvR AUC all present."""
    frame = _committed(exp_id, "per_fold_metrics.csv")
    for column in (
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "ovr_auc_macro",
        "accuracy",
    ):
        assert column in frame.columns, exp_id + " is missing " + column


@pytest.mark.parametrize(
    ("filename", "n_models"), [("T11_pascal_a_results.csv", 6), ("T12_pascal_b_results.csv", 6)]
)
def test_every_headline_metric_carries_a_confidence_interval(filename: str, n_models: int) -> None:
    """T66.7 clause 3 -- the n=124 requirement, and T67's by the same argument.

    Both intervals are required. A fold interval alone would describe only how
    much the score moves when the split moves, which at 124 records is the
    smaller of the two uncertainties.
    """
    from src.reporting.multiclass_report import _RECORD_METRICS, MULTICLASS_HEADLINE

    table = _table(filename)
    assert len(table) == n_models

    for metric in MULTICLASS_HEADLINE:
        mean_column = metric + "_mean"
        if mean_column not in table.columns:
            continue
        fold_ci = table[metric + "_fold_ci"]
        assert fold_ci.notna().all(), metric + " has a missing fold interval"
        assert (fold_ci != "n/a").all(), metric + " fold interval is n/a"
        if metric in _RECORD_METRICS:
            record_ci = table[metric + "_record_ci"]
            assert (record_ci != "n/a").all(), (
                metric + " has no record interval; predictions.parquet was missing"
            )


@pytest.mark.parametrize(
    ("filename", "classes"),
    [
        ("T11_pascal_a_results_per_class.csv", PASCAL_A_CLASSES),
        ("T12_pascal_b_results_per_class.csv", PASCAL_B_CLASSES),
    ],
)
def test_the_per_class_table_covers_every_model_and_class(filename: str, classes: tuple) -> None:
    table = _table(filename)
    assert set(table["class_name"]) == set(classes)
    for model_id, block in table.groupby("model_id"):
        assert set(block["class_name"]) == set(classes), model_id
    assert "recall_fold_ci" in table.columns
    assert (table["recall_fold_ci"] != "n/a").all()


def test_the_coverage_check_names_a_collapsed_predictor() -> None:
    """A constant predictor and a partial one must be detected, not averaged over."""
    import pandas as pd

    from src.reporting.multiclass_report import coverage

    constant = pd.DataFrame({"y_pred": [0] * 20})
    facts = coverage(constant, labels=(0, 1, 2))
    assert facts["degenerate"] is True
    assert facts["predicts_all_classes"] is False
    assert facts["missing_classes"] == "1;2"
    assert facts["n_classes_predicted"] == 1

    partial = pd.DataFrame({"y_pred": [0] * 10 + [1] * 10})
    facts = coverage(partial, labels=(0, 1, 2))
    assert facts["degenerate"] is False, "two classes is not a constant predictor"
    assert facts["predicts_all_classes"] is False
    assert facts["missing_classes"] == "2"

    complete = pd.DataFrame({"y_pred": [0, 1, 2] * 7})
    facts = coverage(complete, labels=(0, 1, 2))
    assert facts["predicts_all_classes"] is True
    assert facts["missing_classes"] == ""


@pytest.mark.parametrize(
    ("filename", "classes"),
    [
        ("T11_pascal_a_results.csv", PASCAL_A_CLASSES),
        ("T12_pascal_b_results.csv", PASCAL_B_CLASSES),
    ],
)
def test_a_model_that_never_predicts_a_class_is_named_in_the_report(
    filename: str, classes: tuple
) -> None:
    """T67.5 -- the imbalance finding must be visible, not buried in an average.

    On EXP-B2, M3 and M6 never emit `extrastole` at all, and M6 has the HIGHEST
    accuracy of the six models precisely because of it. A results table alone
    cannot show that, so the caveats file has to name every such model, and this
    test fails if a future run adds one and leaves it unnamed.
    """
    table = _table(filename)
    for column in ("n_classes_predicted", "predicts_all_classes", "degenerate"):
        assert column in table.columns, filename + " carries no coverage column"
    assert (table["n_classes_declared"] == len(classes)).all()

    incomplete = table[~table["predicts_all_classes"].astype(bool)]
    text = _text(filename.replace(".csv", "_caveats.md"))
    assert "## Models that never predict some class" in text
    if incomplete.empty:
        assert "Every model emitted every class at least once" in text
        return
    for model_id in incomplete["model_id"]:
        assert ("| " + str(model_id) + " |") in text, (
            str(model_id) + " never predicts every class but is not named in the caveats"
        )


def test_the_pascal_a_report_says_artifact_is_not_a_cardiac_class() -> None:
    """T66.6 -- the sentence CLAUDE.md requires, in the generated deliverable."""
    from src.reporting.pascal_statements import ARTIFACT_STATEMENT

    text = _text("T11_pascal_a_results_caveats.md")
    assert ARTIFACT_STATEMENT in text
    assert "RECORDING-QUALITY" in text
    assert "NOT a four-class cardiac classifier" in text
    # Rule 3's disclosure, which belongs beside it.
    assert "subject_derived=False" in text
    # Rule 7.
    assert "not a diagnostic tool" in text.lower()


def test_the_pascal_b_report_says_the_sets_were_never_merged() -> None:
    """T67.6 / T67.7 clause 2 -- explicit, in the report, not only in the code."""
    from src.reporting.pascal_statements import MERGE_STATEMENT

    text = _text("T12_pascal_b_results_caveats.md")
    assert MERGE_STATEMENT in text
    assert "165" in text, "the subject-group count is not stated"
    assert "session" in text.lower(), "the 165-vs-167 distinction is not explained"
    assert "not a diagnostic tool" in text.lower()


@pytest.mark.parametrize("exp_id", ["EXP-B1", "EXP-B2"])
def test_no_multiclass_metric_is_suspiciously_perfect(exp_id: str) -> None:
    """The standing near-perfect rule, applied to these tracks before they ship.

    At n=124 with one 19-record class a fold can legitimately reach 1.0 on a
    single class, so the check is on the aggregate metrics rather than per-class
    ones -- a macro-F1 at 1.0 over a whole fold is the symptom worth stopping for.
    """
    frame = _committed(exp_id, "per_fold_metrics.csv")
    for column in ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1"):
        values = np.asarray(frame[column], dtype=float)
        worst = float(np.nanmax(values))
        assert worst < 0.99, (
            exp_id + " " + column + " reaches " + format(worst, ".4f") + "; investigate "
            "leakage, a scaler fitted outside the fold, or a duplicate record "
            "before recording it"
        )


@pytest.mark.parametrize("exp_id", ["EXP-B1", "EXP-B2"])
def test_the_output_contract_is_complete_on_disk(exp_id: str) -> None:
    """Every non-parquet file of the six-file contract exists for both runs."""
    from src.evaluation.experiment import OUTPUT_CONTRACT

    directory = _experiment(exp_id).output_dir()
    # `output_dir()` calls ensure_dir, so the folder exists whether or not the
    # experiment ever ran. The run is evidenced by its metrics table, not by
    # the directory.
    if not (directory / "per_fold_metrics.csv").is_file():
        pytest.skip(str(directory) + " holds no run; run scripts/11_run_experiment.py")
    for name in OUTPUT_CONTRACT:
        path = directory / name
        if path.suffix == ".parquet":
            continue  # gitignored; CI has never seen one
        assert path.is_file(), "missing " + name + " in " + str(directory)
        assert path.stat().st_size > 0, name + " is empty"


@pytest.mark.parametrize("exp_id", ["EXP-B1", "EXP-B2"])
def test_the_predictions_cover_every_record_once_per_repeat(exp_id: str) -> None:
    """Skips on CI -- predictions.parquet is gitignored."""
    import pandas as pd

    path = _experiment(exp_id).output_dir() / "predictions.parquet"
    if not path.is_file():
        pytest.skip(str(path) + " is gitignored; run the experiment locally")
    frame = pd.read_parquet(path)
    exp = _experiment(exp_id)
    expected = 124 if exp.task == "pascal_a" else 461
    for (model_id, repeat), block in frame.groupby(["model_id", "repeat"]):
        assert len(block) == expected, str(model_id) + " r" + str(repeat)
        assert block["record_uid"].nunique() == expected
    assert set(frame["y_true"]) <= set(exp.label_space.values())
    assert set(frame["y_pred"]) <= set(exp.label_space.values())
