"""Classification metrics (Phase 45, gate T45.7).

The gate: hand-computed values for a small confusion matrix, and agreement with
scikit-learn on the macro and per-class figures.

Both halves matter and neither is redundant. Comparing only against sklearn
would pass even if this module fed it the wrong arguments -- a swapped
``pos_label``, a label ordering derived from the data -- because sklearn would
answer the wrong question consistently. So the core quantities are worked out by
hand from a 2x2 table written into the test, and sklearn is used as the
independent check on the aggregation.

The hand-worked case, used throughout:

    12 records, positive label = 1
    TP = 4, FN = 2, FP = 3, TN = 3

    sensitivity = 4/6  = 0.6667      specificity = 3/6  = 0.5
    precision   = 4/7  = 0.5714      accuracy    = 7/12 = 0.5833
    F1 = 2*4 / (2*4 + 3 + 2) = 8/13  = 0.6154
    balanced accuracy = (0.6667 + 0.5) / 2 = 0.5833
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.evaluation import metrics as mt

# TP=4, FN=2, FP=3, TN=3
Y_TRUE = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PRED = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])

HAND = {
    "tp": 4.0,
    "fn": 2.0,
    "fp": 3.0,
    "tn": 3.0,
    "sensitivity": 4 / 6,
    "specificity": 3 / 6,
    "precision": 4 / 7,
    "accuracy": 7 / 12,
    "f1": 8 / 13,
    "balanced_accuracy": (4 / 6 + 3 / 6) / 2,
}


# ---------------------------------------------------------------------------
# T45.7 -- hand-computed values
# ---------------------------------------------------------------------------


def test_binary_metrics_match_the_hand_computed_table():
    result = mt.binary_metrics(Y_TRUE, Y_PRED)
    for key, expected in HAND.items():
        assert result[key] == pytest.approx(expected), key


def test_the_confusion_matrix_matches_the_hand_computed_counts():
    matrix = mt.confusion(Y_TRUE, Y_PRED, labels=[0, 1])
    assert matrix.tolist() == [[3, 3], [2, 4]]  # [[TN, FP], [FN, TP]]
    assert matrix.sum() == len(Y_TRUE)


def test_specificity_is_the_negative_class_recall():
    """sklearn has no specificity; this is the definition, checked by hand."""
    assert mt.specificity_score(Y_TRUE, Y_PRED, [0, 1], 1) == pytest.approx(3 / 6)
    # Symmetry: specificity for one class is sensitivity for the other.
    flipped = mt.binary_metrics(Y_TRUE, Y_PRED, labels=[0, 1], positive_label=0)
    assert flipped["sensitivity"] == pytest.approx(3 / 6)
    assert flipped["specificity"] == pytest.approx(4 / 6)


def test_a_report_never_contains_accuracy_alone():
    """Research rule 6, enforced on the shape of the returned dict."""
    result = mt.binary_metrics(Y_TRUE, Y_PRED)
    assert set(mt.BINARY_KEYS) <= set(result)
    for required in ("sensitivity", "specificity", "f1", "balanced_accuracy"):
        assert required in result


# ---------------------------------------------------------------------------
# T45.7 -- agreement with scikit-learn
# ---------------------------------------------------------------------------


def test_binary_figures_agree_with_sklearn():
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    result = mt.binary_metrics(Y_TRUE, Y_PRED)
    assert result["accuracy"] == pytest.approx(accuracy_score(Y_TRUE, Y_PRED))
    assert result["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(Y_TRUE, Y_PRED)
    )
    assert result["sensitivity"] == pytest.approx(recall_score(Y_TRUE, Y_PRED))
    assert result["precision"] == pytest.approx(precision_score(Y_TRUE, Y_PRED))
    assert result["f1"] == pytest.approx(f1_score(Y_TRUE, Y_PRED))
    assert result["mcc"] == pytest.approx(matthews_corrcoef(Y_TRUE, Y_PRED))


def test_macro_and_per_class_agree_with_sklearn():
    """T45.7's second clause, on a four-class problem."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    rng = np.random.default_rng(42)
    labels = [0, 1, 2, 3]
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 4, size=200)

    result = mt.multiclass_metrics(y_true, y_pred, labels=labels)
    assert result["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    assert result["weighted_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    for index, label in enumerate(labels):
        assert result["precision_" + str(label)] == pytest.approx(precision[index])
        assert result["recall_" + str(label)] == pytest.approx(recall[index])
        assert result["f1_" + str(label)] == pytest.approx(f1[index])
        assert result["support_" + str(label)] == pytest.approx(support[index])


def test_auc_agrees_with_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    scores = rng.uniform(size=100)
    result = mt.binary_metrics(y_true, (scores > 0.5).astype(int), scores)

    assert result["roc_auc"] == pytest.approx(roc_auc_score(y_true, scores))
    assert result["pr_auc"] == pytest.approx(average_precision_score(y_true, scores))


# ---------------------------------------------------------------------------
# fixed class ordering (T45.4)
# ---------------------------------------------------------------------------


def test_a_class_missing_from_a_fold_still_gets_a_row():
    """The failure this design exists to prevent."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    matrix = mt.confusion(y_true, y_pred, labels=[0, 1, 2, 3])

    assert matrix.shape == (4, 4)
    assert matrix[2].sum() == 0 and matrix[3].sum() == 0
    assert matrix[:2, :2].tolist() == [[1, 1], [0, 2]]


def test_labels_are_required_and_not_inferred():
    with pytest.raises(TypeError):
        mt.confusion(Y_TRUE, Y_PRED)  # type: ignore[call-arg]


def test_a_prediction_outside_the_fixed_ordering_is_rejected():
    """Silently dropping it would make the matrix rows stop summing to support."""
    with pytest.raises(ValueError, match="absent from the fixed ordering"):
        mt.confusion(np.array([0, 1]), np.array([0, 9]), labels=[0, 1])


def test_duplicate_labels_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        mt.confusion(Y_TRUE, Y_PRED, labels=[0, 1, 1])


def test_normalized_rows_sum_to_one_where_there_is_support():
    normalized = mt.confusion_normalized(Y_TRUE, Y_PRED, labels=[0, 1])
    assert np.allclose(normalized.sum(axis=1), 1.0)
    assert normalized[1, 1] == pytest.approx(4 / 6)


def test_an_empty_class_row_normalizes_to_zero_not_nan():
    normalized = mt.confusion_normalized(
        np.array([0, 0, 1]), np.array([0, 1, 1]), labels=[0, 1, 2]
    )
    assert np.isfinite(normalized).all()
    assert normalized[2].tolist() == [0.0, 0.0, 0.0]


def test_column_normalization_is_available_and_differs():
    by_true = mt.confusion_normalized(Y_TRUE, Y_PRED, labels=[0, 1], axis="true")
    by_pred = mt.confusion_normalized(Y_TRUE, Y_PRED, labels=[0, 1], axis="pred")
    assert np.allclose(by_pred.sum(axis=0), 1.0)
    assert not np.allclose(by_true, by_pred)


def test_an_unknown_normalization_axis_is_rejected():
    with pytest.raises(ValueError, match="axis must be"):
        mt.confusion_normalized(Y_TRUE, Y_PRED, labels=[0, 1], axis="both")


# ---------------------------------------------------------------------------
# undefined rather than invented
# ---------------------------------------------------------------------------


def test_auc_is_nan_when_a_fold_holds_one_class():
    """Not 0.5, not 1.0 -- undefined. An invented 0.5 would average into a result."""
    y_true = np.zeros(10, dtype=int)
    result = mt.binary_metrics(y_true, np.zeros(10, dtype=int), np.linspace(0, 1, 10))
    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["pr_auc"])
    assert np.isnan(result["mcc"])
    assert result["accuracy"] == 1.0


def test_ovr_auc_is_nan_when_a_class_is_absent():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    proba = np.tile([0.4, 0.3, 0.3], (4, 1))
    result = mt.multiclass_metrics(y_true, y_pred, proba, labels=[0, 1, 2])
    assert np.isnan(result["ovr_auc_macro"])


def test_probability_shape_is_validated():
    with pytest.raises(ValueError, match="n_samples"):
        mt.multiclass_metrics(
            np.array([0, 1]), np.array([0, 1]), np.zeros((2, 5)), labels=[0, 1, 2]
        )


def test_a_two_column_probability_array_picks_the_right_column():
    """Handing over the wrong column is a silent, plausible-looking error."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    positive = np.array([0.1, 0.2, 0.8, 0.9])
    both = np.column_stack([1 - positive, positive])

    one_d = mt.binary_metrics(y_true, y_pred, positive)
    two_d = mt.binary_metrics(y_true, y_pred, both)
    assert one_d["roc_auc"] == pytest.approx(two_d["roc_auc"]) == 1.0


def test_binary_metrics_reject_more_than_two_labels():
    with pytest.raises(ValueError, match="exactly two labels"):
        mt.binary_metrics(Y_TRUE, Y_PRED, labels=[0, 1, 2])


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="disagree on shape"):
        mt.binary_metrics(Y_TRUE, Y_PRED[:-1])


# ---------------------------------------------------------------------------
# T45.5 -- calibration
# ---------------------------------------------------------------------------


def test_brier_score_of_a_perfect_confident_prediction_is_zero():
    y_true = np.array([0, 1, 1, 0])
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    assert mt.brier_score(y_true, proba, labels=[0, 1]) == pytest.approx(0.0)


def test_brier_score_matches_the_hand_computed_value():
    """One record, p=0.7 on the true class: (0.7-1)^2 + (0.3-0)^2 = 0.18."""
    value = mt.brier_score(np.array([1]), np.array([[0.3, 0.7]]), labels=[0, 1])
    assert value == pytest.approx(0.18)


def test_brier_score_agrees_with_sklearn_in_the_binary_case():
    from sklearn.metrics import brier_score_loss

    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=50)
    positive = rng.uniform(size=50)

    ours = mt.brier_score(y_true, positive, labels=[0, 1])
    # sklearn's binary form counts only the positive column; the multiclass form
    # counts both, which doubles it for two classes.
    assert ours == pytest.approx(2 * brier_score_loss(y_true, positive))


def test_a_label_absent_from_the_ordering_is_rejected_by_brier():
    with pytest.raises(ValueError, match="absent from labels"):
        mt.brier_score(np.array([7]), np.array([[0.5, 0.5]]), labels=[0, 1])


def test_perfect_calibration_gives_zero_ece():
    """Confidence 1.0 and always correct: no gap in any bin."""
    y_true = np.array([0, 1, 0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert mt.expected_calibration_error(y_true, proba, labels=[0, 1]) == pytest.approx(0.0)


def test_total_overconfidence_gives_ece_of_one():
    """Confidence 1.0 and always wrong: |0 - 1| = 1 in the single occupied bin."""
    y_true = np.array([0, 0, 0, 0])
    proba = np.tile([0.0, 1.0], (4, 1))
    assert mt.expected_calibration_error(y_true, proba, labels=[0, 1]) == pytest.approx(1.0)


def test_ece_is_hand_computable_on_two_bins():
    """Half the records at confidence 1.0 and correct, half at 0.6 and wrong.

    ECE = 0.5*|1 - 1| + 0.5*|0 - 0.6| = 0.3
    """
    y_true = np.array([1, 1, 0, 0])
    proba = np.array([[0.0, 1.0], [0.0, 1.0], [0.4, 0.6], [0.4, 0.6]])
    value = mt.expected_calibration_error(y_true, proba, labels=[0, 1], n_bins=10)
    assert value == pytest.approx(0.3)


def test_ece_bin_count_is_configurable():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    positive = rng.uniform(size=200)
    coarse = mt.expected_calibration_error(y_true, positive, labels=[0, 1], n_bins=2)
    fine = mt.expected_calibration_error(y_true, positive, labels=[0, 1], n_bins=20)
    assert coarse != fine
    assert 0.0 <= coarse <= 1.0 and 0.0 <= fine <= 1.0


def test_a_confidence_of_exactly_one_lands_in_the_last_bin():
    """An off-by-one in the bin edges would put it out of range."""
    y_true = np.array([1, 1])
    proba = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert mt.expected_calibration_error(y_true, proba, labels=[0, 1], n_bins=5) == 0.0


def test_zero_bins_is_rejected():
    with pytest.raises(ValueError, match="n_bins"):
        mt.expected_calibration_error(Y_TRUE, Y_PRED, labels=[0, 1], n_bins=0)


# ---------------------------------------------------------------------------
# T45.6 -- bootstrap
# ---------------------------------------------------------------------------


def _accuracy(y_true: Any, y_pred: Any) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def test_bootstrap_brackets_the_point_estimate():
    result = mt.bootstrap_ci(_accuracy, Y_TRUE, Y_PRED, n_resamples=500)
    assert result["point"] == pytest.approx(HAND["accuracy"])
    assert result["lower"] <= result["point"] <= result["upper"]
    assert result["n_valid"] > 0


def test_bootstrap_is_reproducible_at_a_fixed_seed():
    """Rule 5: an unseeded interval moves on every run and is not a result."""
    first = mt.bootstrap_ci(_accuracy, Y_TRUE, Y_PRED, n_resamples=300, seed=42)
    second = mt.bootstrap_ci(_accuracy, Y_TRUE, Y_PRED, n_resamples=300, seed=42)
    assert first == second

    different = mt.bootstrap_ci(_accuracy, Y_TRUE, Y_PRED, n_resamples=300, seed=7)
    assert different["lower"] != first["lower"] or different["upper"] != first["upper"]


def test_a_wider_alpha_gives_a_narrower_interval():
    narrow = mt.bootstrap_ci(_accuracy, Y_TRUE, Y_PRED, n_resamples=800, alpha=0.5)
    wide = mt.bootstrap_ci(_accuracy, Y_TRUE, Y_PRED, n_resamples=800, alpha=0.01)
    assert (narrow["upper"] - narrow["lower"]) < (wide["upper"] - wide["lower"])


def test_a_perfect_classifier_has_a_degenerate_interval():
    y = np.array([0, 1, 0, 1, 1, 0])
    result = mt.bootstrap_ci(_accuracy, y, y, n_resamples=200)
    assert result["point"] == 1.0
    assert result["lower"] == result["upper"] == 1.0


def test_unscorable_resamples_are_excluded_rather_than_substituted():
    """A resample with one class cannot give an AUC; it must not become 0.5."""

    def auc(y_true: Any, y_pred: Any, y_proba: Any) -> float:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_proba))

    rng = np.random.default_rng(42)
    y_true = np.array([0, 1, 1, 1, 1, 1, 1, 1])
    scores = rng.uniform(size=8)
    result = mt.bootstrap_ci(auc, y_true, y_true, scores, n_resamples=300)

    assert result["n_valid"] < result["n_resamples"]
    assert result["n_valid"] > 0


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_metrics_frame_aggregates_folds():
    folds = [
        {"accuracy": 0.8, "roc_auc": 0.9},
        {"accuracy": 0.6, "roc_auc": 0.7},
    ]
    frame = mt.metrics_frame(folds).set_index("metric")
    assert frame.loc["accuracy", "mean"] == pytest.approx(0.7)
    assert frame.loc["accuracy", "min"] == pytest.approx(0.6)
    assert frame.loc["accuracy", "max"] == pytest.approx(0.8)
    assert int(frame.loc["accuracy", "n"]) == 2


def test_an_undefined_fold_lowers_n_rather_than_the_mean():
    """A NaN AUC must not be averaged in as a zero."""
    folds = [
        {"accuracy": 0.8, "roc_auc": 0.9},
        {"accuracy": 0.6, "roc_auc": float("nan")},
    ]
    frame = mt.metrics_frame(folds).set_index("metric")
    assert frame.loc["roc_auc", "mean"] == pytest.approx(0.9)
    assert int(frame.loc["roc_auc", "n"]) == 1
    assert int(frame.loc["roc_auc", "n_folds"]) == 2
    assert int(frame.loc["accuracy", "n"]) == 2


def test_metrics_frame_rejects_an_empty_run():
    with pytest.raises(ValueError, match="no folds"):
        mt.metrics_frame([])
