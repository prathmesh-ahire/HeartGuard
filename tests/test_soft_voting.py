"""Soft-voting ensembles M6 and M7 (Phase 50, gate T50.7).

The gate: equal weights equal the arithmetic mean, weights normalise to 1, and
neither weight optimization nor threshold selection ever touches the outer test
fold; confirm the per-fold threshold is recorded next to its fixed-0.5
counterpart.

The leakage clause is the one that needs machinery rather than an assertion.
Weights chosen on the test fold produce a *better* number, not a broken one, and
nothing in the output distinguishes them from honest weights. So the tests below
work by construction: a canary member that is perfect on the test rows and
useless on the training rows would win all the weight if the optimiser could see
the test fold, and must win none of it if it cannot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.ensemble import soft_voting as sv

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture
def imbalanced() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """~20% positive with repeated subjects, like the primary track."""
    rng = np.random.default_rng(42)
    n = 400
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + rng.normal(scale=0.7, size=n) > 1.0).astype(int)
    groups = np.array(["s" + str(index // 4) for index in range(n)])
    return X, y, groups


class _ConstantProbability(BaseEstimator, ClassifierMixin):
    """A member whose probabilities are fixed. Makes fusion arithmetic checkable."""

    def __init__(self, positive: float = 0.5) -> None:
        self.positive = positive

    def fit(self, X: Any, y: Any) -> _ConstantProbability:
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n = np.asarray(X).shape[0]
        return np.tile([1.0 - self.positive, self.positive], (n, 1))

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]


class _MemorisesTrainingRows(BaseEstimator, ClassifierMixin):
    """Perfect on rows it was fitted on, confidently WRONG on anything else.

    The canary for T50.5. A weight optimiser scoring its members on their
    *training* predictions sees this as flawless and hands it all the weight; one
    scoring out-of-fold predictions sees an anti-predictor and must give it none.

    Off-fold it must be **anti-correlated with the truth**, and getting there
    took two attempts that are worth recording because both failed the same way.
    Returning 0.5 for unseen rows shrinks the fused score toward the middle;
    returning a confident constant shifts it. Both are *monotone* transforms of
    the honest member's score, so the tuned threshold undoes them exactly, every
    weight vector ties, and the canary proves nothing. Only a member that
    reorders records can change a threshold-tuned objective at all.

    That is a property of the ensemble, not of this test: **M7's weights are
    identifiable only up to rank-preserving transformations**. A member whose
    out-of-fold scores are constant, or a monotone function of another member's,
    is invisible to the objective and can be handed any weight without cost.
    `_pick_among_ties` exists because of exactly this.
    """

    def fit(self, X: Any, y: Any) -> _MemorisesTrainingRows:
        features = np.asarray(X)
        self.classes_ = np.unique(y)
        self.n_features_in_ = features.shape[1]
        self.seen_ = {
            row.tobytes(): int(label)
            for row, label in zip(features, np.asarray(y), strict=True)
        }
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        features = np.asarray(X)
        out = np.empty((features.shape[0], 2))
        for index, row in enumerate(features):
            label = self.seen_.get(row.tobytes())
            if label is None:
                # Unseen: rank records by the INVERSE of the real signal, so the
                # damage cannot be undone by moving the threshold.
                positive = 1.0 / (1.0 + np.exp(row[0]))
                out[index] = [1.0 - positive, positive]
            else:
                out[index] = [1.0 - label, float(label)]
        return out

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# T50.1 / T50.6 -- fusion arithmetic
# ---------------------------------------------------------------------------


def test_equal_weights_equal_the_arithmetic_mean():
    """The gate's first clause, stated directly."""
    rng = np.random.default_rng(0)
    stack = rng.dirichlet([1, 1], size=(4, 25))
    assert np.allclose(sv.fuse_probabilities(stack), stack.mean(axis=0))
    assert np.allclose(sv.fuse_probabilities(stack, [1, 1, 1, 1]), stack.mean(axis=0))
    assert np.allclose(
        sv.fuse_probabilities(stack, [7.5, 7.5, 7.5, 7.5]), stack.mean(axis=0)
    )


def test_weights_normalise_to_one():
    """The gate's second clause. Raw scores in, a convex combination out."""
    rng = np.random.default_rng(1)
    stack = rng.dirichlet([1, 1, 1], size=(3, 30))
    fused = sv.fuse_probabilities(stack, [2.0, 3.0, 5.0])
    scaled = sv.fuse_probabilities(stack, [20.0, 30.0, 50.0])
    assert np.allclose(fused, scaled)
    assert np.allclose(fused.sum(axis=1), 1.0)


def test_fusion_is_a_convex_combination_so_it_stays_a_probability():
    rng = np.random.default_rng(2)
    stack = rng.dirichlet([1, 1, 1, 1], size=(5, 40))
    fused = sv.fuse_probabilities(stack, rng.random(5))
    assert (fused >= 0).all() and (fused <= 1).all()
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert fused.min() >= stack.min(axis=0).min()


def test_malformed_weights_are_refused():
    stack = np.full((3, 5, 2), 0.5)
    with pytest.raises(sv.EnsembleError, match="non-negative"):
        sv.fuse_probabilities(stack, [1.0, -1.0, 1.0])
    with pytest.raises(sv.EnsembleError, match="sum to zero"):
        sv.fuse_probabilities(stack, [0.0, 0.0, 0.0])
    with pytest.raises(sv.EnsembleError, match="expected 3 weights"):
        sv.fuse_probabilities(stack, [1.0, 1.0])
    with pytest.raises(sv.EnsembleError, match="n_members, n_samples, n_classes"):
        sv.fuse_probabilities(np.full((5, 2), 0.5))


def test_the_simplex_grid_is_reproducible_and_sums_to_one():
    """The lattice is exact; the division into floats is not, and says so.

    `0.05 + 0.10 + 0.85` is not exactly 1.0 in binary, so the rows sum to 1 only
    to within representation error. What must be exact is reproducibility --
    the same grid on every machine and every call -- because the weights are a
    published result (rule 5).
    """
    for resolution in (0.5, 0.25, 0.05):
        grid = sv.simplex_grid(3, resolution)
        assert np.allclose(grid.sum(axis=1), 1.0)
        assert (grid >= 0).all()
    assert sv.simplex_grid(3, 0.05).shape == (231, 3)
    assert np.array_equal(sv.simplex_grid(3, 0.05), sv.simplex_grid(3, 0.05))


def test_ties_are_broken_toward_equal_weights():
    """A tied optimum must not hand weight to a member that did not earn it.

    Measured case: a member whose out-of-fold probabilities were a constant 0.5
    took 0.95 of the weight without changing the score, because shrinking toward
    0.5 is a monotone transform the threshold undoes. Grid order decided it.
    """
    grid = sv.simplex_grid(2, 0.05)
    scores = np.where(grid[:, 0] > 0, 1.0, 0.0)  # everything non-degenerate ties
    weights, n_tied = sv._pick_among_ties(grid, scores)

    assert n_tied == 20
    assert np.allclose(weights, [0.5, 0.5])

    # A genuine winner still wins.
    scores = np.zeros(len(grid))
    scores[np.argmax(grid[:, 0])] = 1.0
    winner, tied = sv._pick_among_ties(grid, scores)
    assert tied == 1
    assert winner[0] == 1.0


# ---------------------------------------------------------------------------
# T50.4 -- the threshold
# ---------------------------------------------------------------------------


def test_the_threshold_beats_a_fixed_half_on_an_imbalanced_score():
    rng = np.random.default_rng(42)
    y = (rng.random(500) < 0.2).astype(int)
    scores = np.clip(0.15 + 0.4 * y + rng.normal(scale=0.15, size=500), 0, 1)

    choice = sv.select_threshold(y, scores, objective="balanced_accuracy")
    assert choice.threshold < 0.5
    assert choice.balanced_accuracy >= choice.fixed_half_balanced_accuracy
    assert choice.sensitivity > choice.fixed_half_sensitivity


def test_the_fixed_half_comparison_is_always_recorded():
    """The gate's last clause -- the difference must be visible, not assumed."""
    rng = np.random.default_rng(7)
    y = (rng.random(200) < 0.3).astype(int)
    scores = rng.random(200)
    row = sv.select_threshold(y, scores).as_dict()

    for key in (
        "threshold",
        "threshold_sensitivity",
        "threshold_balanced_accuracy",
        "fixed_half_sensitivity",
        "fixed_half_balanced_accuracy",
        "sensitivity_gained",
    ):
        assert key in row, key
    assert row["sensitivity_gained"] == pytest.approx(
        row["threshold_sensitivity"] - row["fixed_half_sensitivity"]
    )


def test_youden_and_balanced_accuracy_pick_the_same_threshold():
    """Youden is 2*balanced_accuracy - 1 -- a monotone transform, not a choice."""
    rng = np.random.default_rng(11)
    y = (rng.random(300) < 0.25).astype(int)
    scores = np.clip(0.2 + 0.35 * y + rng.normal(scale=0.2, size=300), 0, 1)
    assert sv.select_threshold(y, scores, objective="balanced_accuracy").threshold == (
        sv.select_threshold(y, scores, objective="youden").threshold
    )


def test_the_shipped_objective_does_not_saturate_sensitivity():
    """`balanced_accuracy` was chosen 2026-08-27 over a sensitivity-weighted one.

    The alternative is algebraically 1.5*sens + 0.5*spec and drove the threshold
    to a test sensitivity of 1.000 at specificity 0.619. This pins the property
    that decision turned on.
    """
    from src.utils.config import load_config

    for model_id in ("M6", "M7"):
        configured = load_config("models").get("models." + model_id + ".defaults.objective")
        assert configured == "balanced_accuracy", model_id

    # An all-positive rule must never win under the shipped objective.
    scorer = sv.OBJECTIVES["balanced_accuracy"]
    assert scorer(1.0, 0.0) < scorer(0.85, 0.85)


def test_an_unknown_objective_is_refused():
    with pytest.raises(sv.EnsembleError, match="unknown objective"):
        sv.select_threshold([0, 1], [0.2, 0.8], objective="f1_ish")


# ---------------------------------------------------------------------------
# T50.2 / T50.3 -- M6 and M7
# ---------------------------------------------------------------------------


def test_equal_weight_voting_gives_every_member_the_same_share(imbalanced: Any):
    X, y, _ = imbalanced
    members = [
        ("a", _ConstantProbability(0.2)),
        ("b", _ConstantProbability(0.6)),
        ("c", _ConstantProbability(0.7)),
    ]
    fitted = sv.SoftVotingEnsemble(members, weights="equal", tune_threshold=False).fit(X, y)

    assert np.allclose(fitted.weights_, 1 / 3)
    assert fitted.predict_proba(X[:3])[0, 1] == pytest.approx((0.2 + 0.6 + 0.7) / 3)


def test_explicit_weights_are_normalised_and_applied(imbalanced: Any):
    X, y, _ = imbalanced
    members = [("a", _ConstantProbability(0.0)), ("b", _ConstantProbability(1.0))]
    fitted = sv.SoftVotingEnsemble(
        members, weights=[1.0, 3.0], tune_threshold=False
    ).fit(X, y)

    assert np.allclose(fitted.weights_, [0.25, 0.75])
    assert fitted.predict_proba(X[:2])[0, 1] == pytest.approx(0.75)


def test_optimised_weights_live_on_the_simplex(imbalanced: Any):
    X, y, groups = imbalanced
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    members = [
        ("lr", LogisticRegression(max_iter=500)),
        ("tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("dud", _ConstantProbability(0.5)),
    ]
    fitted = sv.SoftVotingEnsemble(
        members, weights="optimized", groups=groups, inner_cv=3
    ).fit(X, y)

    assert fitted.weights_.sum() == pytest.approx(1.0)
    assert (fitted.weights_ >= 0).all()
    assert len(fitted.weights_) == 3


def test_hard_voting_is_refused(imbalanced: Any):
    X, y, _ = imbalanced
    with pytest.raises(sv.EnsembleError, match="only soft voting"):
        sv.SoftVotingEnsemble(
            [("a", _ConstantProbability(0.5))], voting="hard"
        ).fit(X, y)


def test_a_member_that_saw_fewer_classes_is_refused_rather_than_mis_fused():
    """Averaging by position would add one member's P(normal) to another's P(abnormal)."""
    proba = np.full((4, 2), 0.5)
    with pytest.raises(sv.EnsembleError, match="never saw class"):
        sv._align_columns(proba, np.array([0]), np.array([0, 1]), "m")

    reordered = sv._align_columns(
        np.array([[0.1, 0.9]]), np.array([1, 0]), np.array([0, 1]), "m"
    )
    assert reordered[0, 0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# T50.5 / T50.7 -- no leakage into the weights or the threshold
# ---------------------------------------------------------------------------


def test_weights_are_chosen_on_out_of_fold_probabilities_not_training_ones(
    imbalanced: Any,
):
    """The canary. A memoriser is perfect on training rows and useless off them.

    If the optimiser scored members on their training predictions it would give
    this member all the weight, because there it is flawless. Scored out of fold
    it is a coin flip, so it must get little or none.
    """
    X, y, groups = imbalanced
    from sklearn.linear_model import LogisticRegression

    members = [
        ("honest", LogisticRegression(max_iter=500, class_weight="balanced")),
        ("memoriser", _MemorisesTrainingRows()),
    ]
    fitted = sv.SoftVotingEnsemble(
        members, weights="optimized", groups=groups, inner_cv=3
    ).fit(X, y)

    weights = dict(zip(["honest", "memoriser"], fitted.weights_.tolist(), strict=True))
    assert weights["memoriser"] < weights["honest"], (
        "the memoriser won " + str(round(weights["memoriser"], 3)) + " of the weight; "
        "the optimiser is scoring members on rows they were fitted on"
    )


def test_optimised_weights_never_lose_to_equal_weights_in_fold(imbalanced: Any):
    """M7 must be at least as good as M6 on the data both were chosen on.

    Equal weights are a point on the simplex grid, so an optimiser that searched
    it properly cannot score below them in-fold. It can still lose on the OUTER
    test fold -- that is honest overfitting of the inner CV and a real result --
    but losing in-fold would mean the search or its tie-breaking is broken.
    """
    X, y, groups = imbalanced
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    def members() -> list[tuple[str, Any]]:
        return [
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ("tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ]

    equal = sv.SoftVotingEnsemble(
        members(), weights="equal", groups=groups, inner_cv=3
    ).fit(X, y)
    optimised = sv.SoftVotingEnsemble(
        members(), weights="optimized", groups=groups, inner_cv=3
    ).fit(X, y)

    assert (
        optimised.threshold_choice_.balanced_accuracy
        >= equal.threshold_choice_.balanced_accuracy - 1e-9
    )


def test_the_inner_cv_covers_the_training_fold_and_nothing_else(imbalanced: Any):
    X, y, groups = imbalanced
    from sklearn.linear_model import LogisticRegression

    fitted = sv.SoftVotingEnsemble(
        [("lr", LogisticRegression(max_iter=500))],
        weights="optimized", groups=groups, inner_cv=3,
    ).fit(X, y)

    assert fitted.fit_report_.n_oof_rows == len(y)
    assert fitted.fit_report_.grouped_inner_cv is True
    assert fitted.oof_fused_.shape == (len(y), 2)
    assert np.isfinite(fitted.oof_fused_).all()


def test_the_inner_cv_keeps_subjects_on_one_side_when_groups_are_given(
    imbalanced: Any,
):
    """Research rule 3, applied to the split that chooses the weights."""
    X, y, groups = imbalanced
    ensemble = sv.SoftVotingEnsemble(
        [("a", _ConstantProbability(0.5))], groups=groups, inner_cv=3
    )
    ensemble.classes_ = np.unique(y)
    splits, grouped = ensemble._inner_splits(X, y)

    assert grouped is True
    for train_index, held_index in splits:
        assert not (set(groups[train_index]) & set(groups[held_index]))


def test_mismatched_groups_are_refused(imbalanced: Any):
    """Groups from the full matrix instead of the training fold IS the leak."""
    X, y, groups = imbalanced
    with pytest.raises(sv.EnsembleError, match="training fold's own groups"):
        sv.SoftVotingEnsemble(
            [("a", _ConstantProbability(0.5))],
            weights="optimized", groups=groups[:10], inner_cv=3,
        ).fit(X, y)


def test_the_threshold_is_chosen_on_training_rows_only(imbalanced: Any):
    """A threshold fitted on the test fold would be free, and invisible.

    Fitting on a slice and predicting on the rest: the threshold must have been
    derived from the slice's out-of-fold probabilities, so it cannot depend on
    the held-back rows at all.
    """
    X, y, groups = imbalanced
    from sklearn.linear_model import LogisticRegression

    train = np.arange(0, 300)
    held = np.arange(300, 400)

    def build() -> Any:
        return sv.SoftVotingEnsemble(
            [("lr", LogisticRegression(max_iter=500, class_weight="balanced"))],
            weights="equal", groups=groups[train], inner_cv=3,
        )

    first = build().fit(X[train], y[train])
    # Corrupting the held-back labels must not change anything that was decided.
    corrupted = y.copy()
    corrupted[held] = 1 - corrupted[held]
    second = build().fit(X[train], corrupted[train])

    assert first.threshold_ == second.threshold_
    assert np.array_equal(first.weights_, second.weights_)
    assert first.threshold_choice_.n_scored_rows == len(train)


def test_a_thresholded_model_says_it_does_not_predict_by_argmax(imbalanced: Any):
    """Otherwise a probability gate would fail a working decision rule."""
    X, y, groups = imbalanced
    from sklearn.linear_model import LogisticRegression

    fitted = sv.SoftVotingEnsemble(
        [("lr", LogisticRegression(max_iter=500, class_weight="balanced"))],
        groups=groups, inner_cv=3,
    ).fit(X, y)

    proba = fitted.predict_proba(X)
    predicted = fitted.predict(X)
    by_argmax = np.asarray(fitted.classes_)[proba.argmax(axis=1)]

    if fitted.threshold_ != 0.5:
        assert not fitted.predicts_by_argmax
        assert not np.array_equal(predicted, by_argmax)
    assert np.array_equal(predicted, np.where(proba[:, 1] >= fitted.threshold_, 1, 0))


def test_predict_matches_argmax_exactly_when_the_threshold_is_a_half(imbalanced: Any):
    """The binary rule generalises the documented argmax rather than replacing it."""
    X, y, _ = imbalanced
    fitted = sv.SoftVotingEnsemble(
        [("a", _ConstantProbability(0.3)), ("b", _ConstantProbability(0.8))],
        weights="equal", tune_threshold=False,
    ).fit(X, y)

    assert fitted.threshold_ == 0.5
    assert fitted.predicts_by_argmax
    proba = fitted.predict_proba(X)
    assert np.array_equal(
        fitted.predict(X), np.asarray(fitted.classes_)[proba.argmax(axis=1)]
    )


def test_two_fits_of_the_ensemble_agree_exactly(imbalanced: Any):
    """Rule 5: the simplex grid needs no seed, so this must be exact."""
    X, y, groups = imbalanced
    from sklearn.linear_model import LogisticRegression

    def build() -> Any:
        return sv.SoftVotingEnsemble(
            [
                ("lr", LogisticRegression(max_iter=500)),
                ("dud", _ConstantProbability(0.5)),
            ],
            weights="optimized", groups=groups, inner_cv=3,
        )

    first, second = build().fit(X, y), build().fit(X, y)
    assert np.array_equal(first.weights_, second.weights_)
    assert first.threshold_ == second.threshold_
    assert np.array_equal(first.predict_proba(X), second.predict_proba(X))


def test_an_unfitted_ensemble_refuses_to_predict():
    ensemble = sv.SoftVotingEnsemble([("a", _ConstantProbability(0.5))])
    with pytest.raises(sv.EnsembleError, match="not fitted"):
        ensemble.predict_proba(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------


def test_m6_and_m7_differ_only_in_their_weights():
    """A baseline with a worse decision rule would flatter M7 for free."""
    from src.models import estimators as est

    m6, m7 = est.build_estimator("M6"), est.build_estimator("M7")
    assert m6.weights == "equal"
    assert m7.weights == "optimized"
    for attribute in ("inner_cv", "objective", "tune_threshold", "voting"):
        assert getattr(m6, attribute) == getattr(m7, attribute), attribute
    assert [name for name, _ in m6.estimators] == [name for name, _ in m7.estimators]


def test_members_are_calibrated_per_the_measured_verdicts():
    """M4 stays raw because calibrating it made its ECE worse (Phase 48)."""
    from src.models.calibration import CalibratedSVM

    members = dict(sv.ensemble_members("M6"))
    assert set(members) == {"M3", "M4", "M5"}
    assert isinstance(members["M3"], CalibratedSVM), "an SVM has no probabilities"
    assert isinstance(members["M5"], CalibratedSVM), "boosting is over-confident"
    assert not isinstance(members["M4"], CalibratedSVM), "calibrating RF made ECE worse"
