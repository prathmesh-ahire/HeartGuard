"""T58.7 / T59.7 -- the GA and PSO mask searches, their traces and their comparison.

The algorithms are exercised against the REAL D1 matrix and the real DA-07 fold
map at a deliberately tiny budget, because what has to be proved here is not that
a GA can descend a fitness -- it is that *this* evaluator, on *these* folds, never
reaches an outer test row and returns the same answer twice under seed 42.

The on-disk gates skip when their inputs are absent, so a fresh clone reports
"skipped" rather than a false pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

TINY = {"population_size": 6, "generations": 3, "elitism": 1, "tournament_size": 2}


@pytest.fixture(scope="module")
def data() -> Any:
    from src.models import smoke as sm

    return sm.load_task_data("binary")


@pytest.fixture(scope="module")
def fold(data: Any) -> Any:
    from src.optimization import driver as od

    return od.outer_folds_for("binary", data, repeats=[0], folds=[0])[0]


def _evaluator(data: Any, fold: Any) -> Any:
    from src.optimization.masks import MaskEvaluator

    return MaskEvaluator(data, fold, model_id="M1", n_inner_splits=3, seed=42)


# ---------------------------------------------------------------------------
# the shared evaluator
# ---------------------------------------------------------------------------


def test_the_evaluator_never_touches_an_outer_test_row(data: Any, fold: Any) -> None:
    """Measured, not assumed: the ledger records every row any fit saw."""
    evaluator = _evaluator(data, fold)
    rng = np.random.default_rng(0)
    for _ in range(3):
        evaluator.fitness(rng.random(evaluator.n_features) < 0.5)
    evaluator.assert_no_outer_leakage()
    assert evaluator.as_dict()["outer_test_rows_touched"] == 0


def test_the_leakage_check_would_actually_fail_if_a_test_row_leaked(
    data: Any, fold: Any
) -> None:
    """Without this, the test above proves only that the assertion runs."""
    from src.evaluation.cv import LeakageError
    from src.optimization.base import SearchError

    evaluator = _evaluator(data, fold)
    evaluator.ledger.record(np.asarray(fold.test_index)[:5], np.asarray(fold.test_index)[5:10])
    with pytest.raises((AssertionError, LeakageError, SearchError)):
        evaluator.assert_no_outer_leakage()


def test_the_cache_returns_the_same_fitness_without_refitting(
    data: Any, fold: Any
) -> None:
    evaluator = _evaluator(data, fold)
    mask = np.zeros(evaluator.n_features, dtype=bool)
    mask[:30] = True
    first = evaluator.fitness(mask)
    second = evaluator.fitness(mask)
    assert first.fitness == second.fitness
    assert not first.cached and second.cached
    assert evaluator.n_calls == 2
    assert evaluator.n_fitted == 1


def test_an_all_zero_mask_is_refused(data: Any, fold: Any) -> None:
    from src.optimization.masks import MaskError

    evaluator = _evaluator(data, fold)
    with pytest.raises(MaskError):
        evaluator.fitness(np.zeros(evaluator.n_features, dtype=bool))


def test_repair_lifts_a_mask_to_the_minimum_feature_count() -> None:
    from src.optimization.masks import repair_mask

    rng = np.random.default_rng(1)
    mask = np.zeros(138, dtype=bool)
    mask[3] = True
    repaired = repair_mask(mask, min_features=10, rng=rng)
    assert int(repaired.sum()) == 10
    assert repaired[3]


# ---------------------------------------------------------------------------
# T58.1-T58.4 -- the GA
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ga_result(data: Any, fold: Any) -> Any:
    from src.optimization import genetic as ga

    return ga.run_ga(_evaluator(data, fold), ga.GAConfig(**TINY), seed=42)


def test_the_ga_records_per_generation_best_and_mean_fitness(ga_result: Any) -> None:
    """T58.4, and the shape T58.7 gates on."""
    frame = ga_result.trace_frame()
    assert len(frame) == TINY["generations"]
    assert list(frame["generation"]) == list(range(TINY["generations"]))
    for column in ("best_fitness", "mean_fitness", "worst_fitness"):
        assert frame[column].notna().all()
    assert (frame["best_fitness"] <= frame["mean_fitness"] + 1e-12).all()
    assert (frame["mean_fitness"] <= frame["worst_fitness"] + 1e-12).all()


def test_the_ga_best_fitness_never_worsens(ga_result: Any) -> None:
    """Elitism's whole job. A rising trace would mean the elites are not carried.

    Asserted on ``best_fitness``, the generation's own best, NOT on
    ``best_so_far``: with elitism the two coincide, so the strong claim is
    available here and is the one worth making. PSO has no elitism and only the
    monotone column holds there.
    """
    best = ga_result.trace_frame()["best_fitness"].to_numpy()
    assert np.all(np.diff(best) <= 1e-12)


def test_the_ga_is_reproducible_under_seed_42(data: Any, fold: Any, ga_result: Any) -> None:
    """Rule 5, on the whole search and not only on one operator."""
    from src.optimization import genetic as ga

    again = ga.run_ga(_evaluator(data, fold), ga.GAConfig(**TINY), seed=42)
    assert again.best_fitness == pytest.approx(ga_result.best_fitness)
    assert np.array_equal(again.best_mask, ga_result.best_mask)


def test_the_ga_respects_the_minimum_feature_count(ga_result: Any) -> None:
    assert ga_result.n_selected >= ga_result.config["min_features"]


def test_the_ga_fitness_is_the_configured_j(data: Any, fold: Any, ga_result: Any) -> None:
    """T58.2: the fitness must be J, not some other score wearing its name."""
    from src.optimization import multi_objective as mo

    evaluator = _evaluator(data, fold)
    scored = evaluator.fitness(ga_result.best_mask)
    recomputed = mo.score_j(
        scored.macro_f1, evaluator.names_of(ga_result.best_mask)
    )
    assert scored.fitness == pytest.approx(recomputed.value)
    assert ga_result.best_fitness == pytest.approx(scored.fitness)


def test_the_joint_chromosome_returns_weights_on_the_simplex(data: Any, fold: Any) -> None:
    """T58.5, at the smallest budget that still exercises crossover and mutation."""
    from src.optimization import genetic as ga

    config = ga.JointGAConfig(
        members=("M1",), population_size=4, generations=2, elitism=1, tournament_size=2
    )
    result = ga.run_joint_ga(data, fold, config, n_inner_splits=3, seed=42)
    assert result.best_weights
    assert len(result.best_weights) == len(config.members)
    assert sum(result.best_weights) == pytest.approx(1.0)
    assert all(weight >= 0 for weight in result.best_weights)
    assert result.evaluator["outer_test_rows_touched"] == 0


# ---------------------------------------------------------------------------
# T59.1-T59.4 -- PSO
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pso_result(data: Any, fold: Any) -> Any:
    from src.optimization import swarm as pso

    config = pso.PSOConfig(swarm_size=6, iterations=3)
    return pso.run_binary_pso(_evaluator(data, fold), config, seed=42)


def test_pso_records_per_iteration_best_fitness(pso_result: Any) -> None:
    """T59.4."""
    frame = pso_result.trace_frame()
    assert len(frame) == 3
    assert frame["best_fitness"].notna().all()
    assert (frame["best_fitness"] <= frame["mean_fitness"] + 1e-12).all()


def test_pso_best_so_far_is_monotone_even_though_its_iteration_best_is_not(
    pso_result: Any,
) -> None:
    """The swarm has no elitism, so only the running best can be claimed monotone."""
    frame = pso_result.trace_frame()
    assert np.all(np.diff(frame["best_so_far"].to_numpy()) <= 1e-12)
    assert (frame["best_so_far"] <= frame["best_fitness"] + 1e-12).all()


def test_pso_is_reproducible_under_seed_42(data: Any, fold: Any, pso_result: Any) -> None:
    from src.optimization import swarm as pso

    again = pso.run_binary_pso(
        _evaluator(data, fold), pso.PSOConfig(swarm_size=6, iterations=3), seed=42
    )
    assert again.best_fitness == pytest.approx(pso_result.best_fitness)
    assert np.array_equal(again.best_mask, pso_result.best_mask)


def test_pso_and_ga_declare_the_same_budget_at_the_configured_settings() -> None:
    """T59.5 compares them at equal budget; the config must actually make them equal."""
    from src.optimization import genetic as ga
    from src.optimization import swarm as pso

    assert ga.load_ga_config().max_evaluations == pso.load_pso_config().max_evaluations


def test_the_velocity_clamp_keeps_the_flip_probability_off_the_rails() -> None:
    from src.optimization.swarm import _sigmoid, load_pso_config

    clamp = load_pso_config().velocity_clamp
    assert 0.01 < float(_sigmoid(np.asarray([-clamp]))[0]) < 0.5
    assert 0.5 < float(_sigmoid(np.asarray([clamp]))[0]) < 0.99


def test_the_weight_swarm_stays_on_the_simplex(data: Any, fold: Any) -> None:
    """T59.1, and the constraint T60.3 will restate: non-negative and summing to 1."""
    from src.optimization import swarm as pso

    config = pso.WeightSwarmConfig(members=("M1", "M3"), swarm_size=6, iterations=4)
    result = pso.run_weight_pso(data, fold, config, n_inner_splits=3, seed=42)
    assert sum(result.best_weights) == pytest.approx(1.0)
    assert all(weight >= 0 for weight in result.best_weights)
    assert result.detail["outer_test_rows_touched"] == 0
    assert len(result.iterations) == 4


def test_out_of_fold_probabilities_cover_every_training_row_once(
    data: Any, fold: Any
) -> None:
    """The weight swarm's cache is only honest if every row is predicted out of fold."""
    from src.optimization.swarm import member_oof_probabilities

    stack, y_oof, _, detail = member_oof_probabilities(
        data, fold, ("M1",), n_inner_splits=3, seed=42
    )
    assert stack.shape[0] == 1
    assert stack.shape[1] == y_oof.size
    assert y_oof.size == len(np.asarray(fold.train_index))
    assert detail["n_out_of_fold_rows"] == y_oof.size
    assert np.allclose(stack.sum(axis=2), 1.0)


# ---------------------------------------------------------------------------
# T58.7 / T59.7 -- the gates over what is on disk
# ---------------------------------------------------------------------------


def _section() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.search_optimization"))


def _load(exp: str, name: str) -> Any:
    import pandas as pd

    path = _section() / exp / name
    if not path.exists():
        pytest.skip(
            str(path) + " does not exist; run scripts/07_run_population_search.py first"
        )
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def _skip_reason_recorded(token: str) -> bool:
    from src.utils.config import load_config

    path = Path(load_config("paths").require("outputs.root")) / "missing_outputs_report.txt"
    return path.is_file() and token in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("exp", ["SO-03a", "SO-03b"])
def test_the_convergence_trace_exists_with_per_generation_fitness(exp: str) -> None:
    """T58.7 and T59.7: the trace, or a recorded compute-bound skip."""
    path = _section() / exp / "convergence.csv"
    if not path.exists():
        assert _skip_reason_recorded(exp), (
            exp + " has no convergence trace and no skip reason in "
            "missing_outputs_report.txt; a missing output must be one or the other"
        )
        pytest.skip(exp + " skipped, and the reason is recorded")

    frame = _load(exp, "convergence.csv")
    assert len(frame) > 0
    for column in ("generation", "best_fitness", "best_so_far", "mean_fitness"):
        assert column in frame.columns
    for label, block in frame.groupby("outer_fold"):
        assert list(block["generation"]) == list(range(len(block))), label
        assert block["best_fitness"].notna().all()
        assert block["mean_fitness"].notna().all()
        # `best_so_far`, not `best_fitness`: PSO's per-iteration best legitimately
        # worsens because the swarm keeps no elite. The running best cannot.
        assert np.all(np.diff(block["best_so_far"].to_numpy()) <= 1e-9), (
            label + ": the running best fitness worsened between generations"
        )


@pytest.mark.parametrize("exp", ["SO-03a", "SO-03b"])
def test_the_selected_mask_is_a_real_proper_subset(exp: str) -> None:
    from src.feature_extraction import registry

    best = _load(exp, "best_subset.json")
    known = set(registry.feature_names())
    for label, summary in best["by_fold"].items():
        names = summary["features"]
        assert 0 < len(names) < len(known), label
        assert set(names) <= known, label
        assert summary["n_selected"] == len(names)
        assert summary["evaluator"]["outer_test_rows_touched"] == 0, label


def test_the_ga_versus_pso_comparison_used_identical_folds_budget_and_seed() -> None:
    """T59.7, the part that would otherwise be taken on trust."""
    import pandas as pd

    path = _section() / "ga_vs_pso_comparison.csv"
    if not path.exists():
        pytest.skip(
            str(path) + " does not exist; run "
            "`python scripts/07_run_population_search.py --compare`"
        )
    frame = pd.read_csv(path)
    assert len(frame) > 0

    ga_best = _load("SO-03a", "best_subset.json")["by_fold"]
    pso_best = _load("SO-03b", "best_subset.json")["by_fold"]
    for row in frame.itertuples():
        label = str(row.outer_fold)
        assert label in ga_best and label in pso_best
        assert int(ga_best[label]["config"]["max_evaluations"]) == int(row.budget_evaluations)
        assert int(pso_best[label]["config"]["max_evaluations"]) == int(row.budget_evaluations)
        assert int(ga_best[label]["evaluator"]["seed"]) == int(row.seed)
        assert int(pso_best[label]["evaluator"]["seed"]) == int(row.seed)
        assert ga_best[label]["model_id"] == pso_best[label]["model_id"]
        assert (
            int(ga_best[label]["evaluator"]["n_inner_splits"])
            == int(pso_best[label]["evaluator"]["n_inner_splits"])
        )


def test_the_joint_chromosome_output_states_it_is_not_comparable_to_m7() -> None:
    """T58.5 ran on a reduced member set; the file has to say so, not the chat log."""
    path = _section() / "SO-03a" / "joint_weights.json"
    if not path.exists():
        pytest.skip(str(path) + " does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "note" in payload and "M7" in payload["note"]
    for summary in payload["by_fold"].values():
        weights = summary["best_weights"]
        assert sum(weights) == pytest.approx(1.0)
        assert all(weight >= 0 for weight in weights)


def test_the_weight_swarm_output_is_on_the_simplex() -> None:
    """T59.1 on disk."""
    path = _section() / "SO-03b" / "weight_swarm.json"
    if not path.exists():
        pytest.skip(str(path) + " does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for label, summary in payload["by_fold"].items():
        assert summary["weights_sum"] == pytest.approx(1.0), label
        assert all(weight >= 0 for weight in summary["best_weights"]), label
        assert summary["detail"]["outer_test_rows_touched"] == 0, label
