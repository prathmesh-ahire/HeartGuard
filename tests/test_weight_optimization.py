"""T60.7 -- SO-05 ensemble weight optimization: the constraint, and the stability check.

The gate T60.7 names is narrow and exact: optimized weights are non-negative and
sum to 1, and per-fold weight variance is reported. Both are asserted here
against the emitted files, not against a fresh computation, because what a
deliverable claims is what the file says.

The unit tests around them run on synthetic probability stacks, which is
legitimate for a weight search in a way it would not be for a feature extractor:
the input to this optimizer IS a probability array, so a constructed one exercises
exactly the same code path as a fitted one. The fold-safety claim is the exception
and is checked against the real matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="module")
def toy() -> tuple[np.ndarray, np.ndarray]:
    """Three members with genuinely different skill, so weights have work to do."""
    rng = np.random.default_rng(42)
    n = 400
    y = (rng.random(n) < 0.35).astype(int)
    stack = []
    for noise in (0.20, 0.35, 0.55):
        signal = np.clip(y + rng.normal(0.0, noise, n), 0.01, 0.99)
        stack.append(np.column_stack([1.0 - signal, signal]))
    return np.stack(stack), y


# ---------------------------------------------------------------------------
# T60.3 -- the constraint
# ---------------------------------------------------------------------------


def test_normalize_projects_anything_onto_the_simplex() -> None:
    from src.optimization import weights as wt

    for raw in ([3.0, 1.0, 1.0], [-2.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1e-9, 1e-9, 1e-9]):
        vector = wt.normalize_weights(raw)
        assert vector.sum() == pytest.approx(1.0)
        assert (vector >= 0).all()


def test_an_all_zero_vector_becomes_equal_weights_not_an_error() -> None:
    from src.optimization import weights as wt

    assert wt.normalize_weights([0.0, 0.0, 0.0]) == pytest.approx(wt.equal_weights(3))


@pytest.mark.parametrize("n_members", [2, 3, 4, 5])
def test_the_candidate_grid_always_contains_equal_weights(n_members: int) -> None:
    """The fix. A shrinkage rule whose target is not a candidate cannot reach it."""
    from src.optimization import weights as wt

    grid = wt.candidate_grid(n_members, 0.05)
    distance = np.linalg.norm(grid - wt.equal_weights(n_members), axis=1)
    assert np.isclose(distance, 0.0).any()
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert (grid >= 0).all()


def test_the_raw_lattice_does_not_contain_equal_weights_for_three_members() -> None:
    """Proof the fix above is fixing something real, not guarding a non-problem."""
    from src.ensemble.soft_voting import simplex_grid

    lattice = simplex_grid(3, 0.05)
    distance = np.linalg.norm(lattice - np.full(3, 1.0 / 3.0), axis=1)
    assert not np.isclose(distance, 0.0).any()
    assert distance.min() == pytest.approx(0.0408, abs=1e-3)


def test_m7s_own_weight_search_now_offers_equal_weights() -> None:
    """The same guarantee where it actually decides a shipped model."""
    from src.ensemble.soft_voting import weight_candidates

    grid = weight_candidates(3, 0.05)
    distance = np.linalg.norm(grid - np.full(3, 1.0 / 3.0), axis=1)
    assert np.isclose(distance, 0.0).any()


# ---------------------------------------------------------------------------
# T60.1 / T60.2 -- the optimizers
# ---------------------------------------------------------------------------


def test_every_optimizer_returns_a_legal_weight_vector(toy: Any) -> None:
    from src.optimization import weights as wt

    stack, y = toy
    members = ("A", "B", "C")
    results = [
        wt.optimize_grid(stack, y, members, n_standard_errors=1.0),
        wt.optimize_slsqp(stack, y, members, surrogate="objective"),
        wt.optimize_slsqp(stack, y, members, surrogate="log_loss"),
    ]
    for result in results:
        assert result.weights_sum == pytest.approx(1.0)
        assert all(weight >= 0 for weight in result.weights)
        assert len(result.weights) == len(members)
        assert np.isfinite(result.score)


def test_the_grid_is_deterministic(toy: Any) -> None:
    from src.optimization import weights as wt

    stack, y = toy
    first = wt.optimize_grid(stack, y, ("A", "B", "C"))
    second = wt.optimize_grid(stack, y, ("A", "B", "C"))
    assert first.weights == second.weights
    assert first.score == pytest.approx(second.score)


def test_slsqp_on_the_step_objective_does_not_move(toy: Any) -> None:
    """The measurement behind the claim that the objective is not differentiable.

    Balanced accuracy is piecewise constant in the weights, so a finite-difference
    gradient is exactly zero everywhere between threshold crossings and SLSQP
    reports success without leaving its start. This is why T60.1's grid exists,
    and it should be a number in the repository rather than an assertion in prose.
    """
    from src.optimization import weights as wt

    stack, y = toy
    result = wt.optimize_slsqp(stack, y, ("A", "B", "C"), surrogate="objective")
    assert result.moved_from_start == pytest.approx(0.0, abs=1e-9)
    assert result.detail["success"] is True


def test_slsqp_on_the_smooth_surrogate_does_move(toy: Any) -> None:
    """Without this, the test above would only prove SLSQP was wired up wrong."""
    from src.optimization import weights as wt

    stack, y = toy
    result = wt.optimize_slsqp(stack, y, ("A", "B", "C"), surrogate="log_loss")
    assert result.moved_from_start > 0.01
    assert result.n_iterations > 1


def test_the_one_standard_error_rule_shrinks_toward_equal_weights(toy: Any) -> None:
    from src.optimization import weights as wt

    stack, y = toy
    guarded = wt.optimize_grid(stack, y, ("A", "B", "C"), n_standard_errors=1.0)
    argmax = wt.optimize_grid(stack, y, ("A", "B", "C"), n_standard_errors=None)
    centre = wt.equal_weights(3)
    assert np.linalg.norm(np.asarray(guarded.weights) - centre) <= (
        np.linalg.norm(np.asarray(argmax.weights) - centre) + 1e-12
    )
    assert guarded.detail["n_within_margin"] >= 1


def test_an_unknown_surrogate_is_refused(toy: Any) -> None:
    from src.optimization import weights as wt

    stack, y = toy
    with pytest.raises(wt.WeightSearchError):
        wt.optimize_slsqp(stack, y, ("A", "B", "C"), surrogate="nonsense")


# ---------------------------------------------------------------------------
# T60.4 -- fold safety, against the real matrix
# ---------------------------------------------------------------------------


def test_weights_are_chosen_without_touching_the_outer_test_fold() -> None:
    from src.models import smoke as sm
    from src.optimization import driver as od
    from src.optimization import weights as wt

    try:
        data = sm.load_task_data("binary")
        fold = od.outer_folds_for("binary", data, repeats=[0], folds=[0])[0]
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 unavailable (" + type(error).__name__ + "): " + str(error))

    results, detail = wt.optimize_fold(
        data, fold, ("M1", "M3"), n_inner_splits=3, seed=42
    )
    assert detail["outer_test_rows_touched"] == 0
    assert detail["n_out_of_fold_rows"] == len(np.asarray(fold.train_index))
    for result in results:
        assert result.weights_sum == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T60.7 -- the gate over what is on disk
# ---------------------------------------------------------------------------


def _section() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.search_optimization"))


def _load(name: str) -> Any:
    import pandas as pd

    path = _section() / "SO-05" / name
    if not path.exists():
        pytest.skip(
            str(path) + " does not exist; run scripts/08_run_weight_optimization.py first"
        )
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def test_every_emitted_weight_vector_is_non_negative_and_sums_to_one() -> None:
    """T60.7, first half, asserted on the file rather than on a re-computation."""
    frame = _load("weight_search.csv")
    assert len(frame) > 0
    weight_columns = [column for column in frame.columns if column.startswith("w_")]
    assert len(weight_columns) >= 2

    for column in weight_columns:
        assert (frame[column] >= -1e-12).all(), column
    assert np.allclose(frame[weight_columns].sum(axis=1), 1.0)
    assert np.allclose(frame["weights_sum"], 1.0)


def test_per_fold_weight_variance_is_reported() -> None:
    """T60.7, second half. The stability check must exist and be populated."""
    stability = _load("weight_stability.csv")
    for column in ("method", "member", "n_folds", "mean_weight", "std_weight",
                   "min_weight", "max_weight", "range_weight"):
        assert column in stability.columns, column
    assert len(stability) > 0
    assert stability["std_weight"].notna().all()
    assert (stability["range_weight"] >= 0).all()
    assert (stability["max_weight"] >= stability["min_weight"]).all()


def test_the_stability_check_covers_more_than_one_fold() -> None:
    """A standard deviation over one fold is zero and says nothing."""
    stability = _load("weight_stability.csv")
    assert int(stability["n_folds"].max()) > 1


def test_equal_weights_is_compared_against_every_optimizer_per_fold() -> None:
    """T60.5: the delta table must carry both arms on the same fold."""
    delta = _load("equal_vs_optimized.csv")
    assert {"outer_fold", "method", "equal_score", "optimized_score", "delta"} <= set(
        delta.columns
    )
    assert len(delta) > 0
    assert np.allclose(delta["delta"], delta["optimized_score"] - delta["equal_score"])
    assert delta["method"].nunique() >= 2


def test_the_final_weight_vector_records_its_constraint_and_spread() -> None:
    """T60.6: the shipped vector, its per-member spread, and how it was constrained."""
    final = _load("final_weights.json")
    assert final["mean_weights_sum"] == pytest.approx(1.0)
    assert all(weight >= 0 for weight in final["mean_weights"])
    assert len(final["mean_weights"]) == len(final["members"])
    assert final.get("constraint")
    assert set(final["per_member_std"]) == set(final["members"])
    assert final["n_folds"] == len(final["per_fold_weights"])
    for weights in final["per_fold_weights"].values():
        assert sum(weights) == pytest.approx(1.0)


def test_no_fold_touched_its_own_outer_test_rows() -> None:
    final = _load("final_weights.json")
    for detail in final["fold_detail"]:
        assert detail["outer_test_rows_touched"] == 0, detail["outer_fold"]
