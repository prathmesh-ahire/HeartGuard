"""T61.7 -- SO-06: J exactly as documented, a bounded time term, and exported Pareto data.

T61.1 through T61.3 were built in Phase 57, because T57.5 needed J to select the
shipped subset before Phase 61 existed. They are gated here, where the task list
asks for them, against the same implementation.

The formula is checked term by term rather than against a stored number. A test
that compares J to a value someone once observed proves the code has not changed;
a test that reassembles the documented formula proves the code computes what the
blueprint says it computes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T61.1 / T61.2 -- the objective and its weights
# ---------------------------------------------------------------------------


def test_j_is_the_documented_formula_term_by_term() -> None:
    """J = alpha(1 - MacroF1) + beta(SelectedFeatures/138) + gamma(NormalizedTime)."""
    from src.feature_extraction import registry
    from src.optimization import multi_objective as mo

    weights = mo.load_weights()
    cost_model = mo.load_cost_model()
    names = list(registry.feature_names())[:37]

    scored = mo.score_j(0.79, names, weights=weights, cost_model=cost_model)
    expected = (
        weights.alpha * (1.0 - 0.79)
        + weights.beta * (37.0 / weights.n_features_total)
        + weights.gamma * cost_model.normalized(mo.families_needed(names))
    )
    assert scored.value == pytest.approx(expected)
    assert scored.n_selected == 37
    assert scored.performance == pytest.approx(0.79)


def test_j_is_minimised_so_a_better_model_scores_lower() -> None:
    from src.feature_extraction import registry
    from src.optimization import multi_objective as mo

    names = list(registry.feature_names())[:40]
    worse = mo.score_j(0.70, names)
    better = mo.score_j(0.90, names)
    assert better.value < worse.value


def test_more_features_never_lowers_j_at_equal_performance() -> None:
    from src.feature_extraction import registry
    from src.optimization import multi_objective as mo

    names = list(registry.feature_names())
    small = mo.score_j(0.83, names[:20])
    large = mo.score_j(0.83, names[:120])
    assert small.value < large.value


def test_the_weights_come_from_config_not_from_a_default_in_code() -> None:
    """T61.2. A J traceable only to a literal in a module is not traceable."""
    from src.optimization import multi_objective as mo
    from src.utils.config import load_config

    configured = load_config("models").require("optimization.multi_objective")
    weights = mo.load_weights()
    assert weights.alpha == pytest.approx(float(configured["alpha"]))
    assert weights.beta == pytest.approx(float(configured["beta"]))
    assert weights.gamma == pytest.approx(float(configured["gamma"]))


def test_negative_or_non_finite_weights_are_refused() -> None:
    from src.optimization import multi_objective as mo

    with pytest.raises(mo.MultiObjectiveError):
        mo.JWeights(alpha=-0.1)
    with pytest.raises(mo.MultiObjectiveError):
        mo.JWeights(alpha=0.0, beta=0.0, gamma=0.0)


# ---------------------------------------------------------------------------
# T61.3 -- the inference-time term
# ---------------------------------------------------------------------------


def test_the_inference_time_term_is_bounded_in_the_unit_interval() -> None:
    """T61.3, stated as the gate states it, over every reachable family subset."""
    from itertools import combinations

    from src.optimization import multi_objective as mo

    cost_model = mo.load_cost_model()
    families = sorted(cost_model.seconds)
    for size in range(1, len(families) + 1):
        for subset in combinations(families, size):
            value = cost_model.normalized(subset)
            assert 0.0 <= value <= 1.0, subset
    assert cost_model.normalized(families) == pytest.approx(1.0)


def test_the_slowest_configuration_is_exactly_one() -> None:
    """"Normalized against the slowest configuration" has to mean literally 1.0."""
    from src.feature_extraction import registry
    from src.optimization import multi_objective as mo

    cost_model = mo.load_cost_model()
    everything = mo.families_needed(list(registry.feature_names()))
    assert cost_model.normalized(everything) == pytest.approx(1.0)


def test_an_unknown_family_is_refused_rather_than_charged_zero() -> None:
    from src.optimization import multi_objective as mo

    with pytest.raises(mo.MultiObjectiveError):
        mo.load_cost_model().normalized(["not_a_family"])


# ---------------------------------------------------------------------------
# the Pareto machinery
# ---------------------------------------------------------------------------


def test_domination_is_the_textbook_relation() -> None:
    from src.optimization import pareto as pt

    # maximise the first, minimise the second
    sense = [1, -1]
    others = np.asarray([[0.9, 10.0], [0.5, 50.0]])
    assert pt.is_dominated([0.8, 20.0], others, sense)       # beaten on both
    assert not pt.is_dominated([0.95, 30.0], others, sense)  # better somewhere
    assert not pt.is_dominated([0.9, 10.0], others, sense)   # equal, not worse


def test_the_front_keeps_the_dominated_points_in_the_table() -> None:
    import pandas as pd

    from src.optimization import pareto as pt

    frame = pd.DataFrame(
        {
            "macro_f1": [0.90, 0.80, 0.85],
            "n_selected": [10.0, 20.0, 10.0],
            "normalized_inference_time": [1.0, 1.0, 1.0],
        }
    )
    out = pt.pareto_front(frame)
    assert len(out) == len(frame)
    assert out["on_front"].tolist() == [True, False, False]


def test_the_weighting_grid_is_a_normalised_simplex() -> None:
    from src.optimization import pareto as pt

    grid = pt.weighting_grid(0.05)
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert (grid >= 0).all()
    assert grid.shape[1] == 3


# ---------------------------------------------------------------------------
# T61.7 -- the gate over what is on disk
# ---------------------------------------------------------------------------


def _section() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.search_optimization"))


def _load(name: str) -> Any:
    import pandas as pd

    path = _section() / "SO-06" / name
    if not path.exists():
        pytest.skip(str(path) + " does not exist; run scripts/09_run_multi_objective.py")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def test_the_pareto_data_is_exported(  # T61.7
) -> None:
    configurations = _load("pareto_configurations.csv")
    front = _load("pareto_front.csv")
    assert len(configurations) > len(front) > 0
    assert configurations["on_front"].sum() == len(front)
    for column in ("ranker", "k", "macro_f1", "n_selected", "normalized_inference_time"):
        assert column in front.columns


def test_nothing_on_the_front_is_dominated_by_anything_in_the_table() -> None:
    """Re-derives the relation from the exported numbers rather than trusting the flag."""
    from src.optimization import pareto as pt

    configurations = _load("pareto_configurations.csv")
    columns = list(pt.OBJECTIVE_SENSE)
    sense = [pt.OBJECTIVE_SENSE[name] for name in columns]
    matrix = configurations[columns].to_numpy(dtype=float)
    for index, on_front in enumerate(configurations["on_front"]):
        dominated = pt.is_dominated(matrix[index], np.delete(matrix, index, axis=0), sense)
        assert dominated != bool(on_front), configurations.iloc[index][["ranker", "k"]].to_dict()


def test_the_recorded_j_definition_matches_the_implementation() -> None:
    """T61.7: 'implemented exactly as documented', checked against the emitted record."""
    from src.optimization import multi_objective as mo

    definition = _load("j_definition.json")
    weights = mo.load_weights()
    assert definition["minimised"] is True
    assert "alpha*(1 - MacroF1)" in definition["formula"]
    assert "beta*(SelectedFeatures/138)" in definition["formula"]
    assert "gamma*NormalizedInferenceTime" in definition["formula"]
    assert definition["weights"]["alpha"] == pytest.approx(weights.alpha)
    assert definition["weights"]["beta"] == pytest.approx(weights.beta)
    assert definition["weights"]["gamma"] == pytest.approx(weights.gamma)


def test_the_exported_inference_time_column_is_bounded() -> None:
    configurations = _load("pareto_configurations.csv")
    assert configurations["normalized_inference_time"].between(0.0, 1.0).all()


def test_the_weighting_sweep_covers_the_simplex_and_records_its_selection() -> None:
    sweep = _load("weighting_sweep.csv")
    assert len(sweep) > 0
    assert np.allclose(sweep[["alpha", "beta", "gamma"]].sum(axis=1), 1.0)
    assert (sweep[["alpha", "beta", "gamma"]] >= 0).all().all()
    for column in ("selected_ranker", "selected_k", "selected_macro_f1"):
        assert column in sweep.columns
    assert sweep["guarded"].nunique() == 2, "both raw and guarded sweeps must be exported"


def test_the_operating_point_is_the_subset_so04_actually_ships() -> None:
    """T61.5, and the rule that there is only one final feature subset in this project."""
    operating = _load("operating_point.json")
    settings_path = _section() / "SO-04" / "so04_settings.json"
    if not settings_path.exists():
        pytest.skip(str(settings_path) + " does not exist")
    so04 = json.loads(settings_path.read_text(encoding="utf-8"))["chosen"]
    assert operating["ranker"] == so04["ranker"]
    assert int(operating["k"]) == int(so04["k"])
    assert operating["on_pareto_front"] is True
    assert 0.0 <= operating["share_of_weight_space"] <= 1.0
