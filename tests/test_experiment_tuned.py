"""T65.7 -- the optimized run is nested, selects by the documented rule, and persists.

The expensive part of Phase 65 is the nested search itself, and a test cannot
re-run nine hours of it. What it can do, and does here, is prove the three
properties the gate actually asks about:

* the search inside a fold never reads that fold's test rows (nested, measured
  by :class:`~src.optimization.base.RowLedger`, not asserted);
* selection follows sensitivity-then-balanced-accuracy and would differ from a
  raw-accuracy rule when the two disagree;
* the persisted model carries its feature list and says which point it was
  fitted at.

Everything reaching the gitignored matrix or a not-yet-produced artifact skips
rather than fails, so a fresh clone and CI report "skipped".
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# T07 -- the fixed selected point
# ---------------------------------------------------------------------------


def _selected() -> dict[str, dict[str, Any]]:
    from src.evaluation.tuned import TuningError, selected_parameters

    try:
        return selected_parameters()
    except (TuningError, FileNotFoundError) as error:
        pytest.skip("T07 / best_parameters.json unavailable: " + str(error))


def test_selected_parameters_cover_every_searched_model() -> None:
    points = _selected()
    assert {"M1", "M3", "M4", "M5", "M8"} <= set(points)
    for model_id, params in points.items():
        assert params, model_id + " selected an empty parameter set"


def test_selected_parameters_keep_none_as_a_value_not_a_blank() -> None:
    """`class_weight=None` is a decision SO-02 made, not a missing entry.

    A naive `read_csv` of T07 turns the string "None" into NaN; this is the
    reason `selected_parameters` reads the typed JSON and only cross-checks
    against T07. See the 2026-08-28 entry in Docs/note.md.
    """
    points = _selected()
    assert "class_weight" in points["M3"]
    assert points["M3"]["class_weight"] is None
    assert points["M4"]["class_weight"] == "balanced"


def test_selected_points_are_legal_in_their_declared_search_space() -> None:
    from src.models import spaces

    points = _selected()
    for model_id, params in points.items():
        space = spaces.load_space(model_id)
        assert not space.violations(params), (
            model_id + " selected a point its own space rejects: " + str(params)
        )


# ---------------------------------------------------------------------------
# T65.3 -- the SO-04 subset
# ---------------------------------------------------------------------------


def _subset() -> tuple[list[int], list[str]]:
    from src.evaluation.tuned import TuningError, subset_columns

    try:
        return subset_columns()
    except (TuningError, FileNotFoundError) as error:
        pytest.skip("SO-04 subset unavailable: " + str(error))


def test_subset_columns_resolve_against_the_feature_registry() -> None:
    from src.feature_extraction.registry import feature_names

    columns, names = _subset()
    registry = list(feature_names())
    assert len(columns) == len(names)
    assert len(set(columns)) == len(columns)
    for position, name in zip(columns, names, strict=True):
        assert registry[position] == name


def test_subset_pipeline_selects_exactly_those_columns() -> None:
    """The subset is a pipeline step, so it is fitted inside the fold like the rest."""
    from src.evaluation.tuned import subset_pipeline_config
    from src.models import pipeline as pl

    columns, names = _subset()
    config = subset_pipeline_config()
    assert config["selector"]["kind"] == "fixed_subset"
    assert config["selector"]["source"] == "SO-04"

    from sklearn.linear_model import LogisticRegression

    built = pl.build_pipeline(LogisticRegression(max_iter=200), config=config, n_features=138)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(60, 138))
    targets = np.array([0, 1] * 30)
    built.fit(features, targets)

    selector = built.named_steps["selector"]
    assert list(selector.get_support(indices=True)) == list(columns)
    assert built.named_steps["estimator"].n_features_in_ == len(names)


# ---------------------------------------------------------------------------
# T65.4 -- the selection rule
# ---------------------------------------------------------------------------


def test_selection_prioritises_sensitivity_over_raw_accuracy() -> None:
    """Rule 6, on a case constructed so the two rules disagree.

    Built by hand rather than taken from a run: a real table where accuracy and
    sensitivity happen to agree would let a rule that silently sorted on
    accuracy pass this test.
    """
    import pandas as pd

    from src.evaluation.tuned import select_final_model

    aggregate = pd.DataFrame(
        [
            {
                "model_id": "high_accuracy",
                "accuracy_mean": 0.92,
                "sensitivity_mean": 0.55,
                "balanced_accuracy_mean": 0.75,
            },
            {
                "model_id": "high_sensitivity",
                "accuracy_mean": 0.84,
                "sensitivity_mean": 0.88,
                "balanced_accuracy_mean": 0.85,
            },
        ]
    )
    selection = select_final_model(aggregate)
    assert selection["model_id"] == "high_sensitivity"
    assert selection["accuracy_would_have_chosen"] == "high_accuracy"
    assert selection["rule_and_accuracy_agree"] is False
    assert [row["model_id"] for row in selection["ranking"]] == [
        "high_sensitivity",
        "high_accuracy",
    ]


def test_selection_breaks_a_sensitivity_tie_on_balanced_accuracy() -> None:
    import pandas as pd

    from src.evaluation.tuned import select_final_model

    aggregate = pd.DataFrame(
        [
            {"model_id": "A", "sensitivity_mean": 0.80, "balanced_accuracy_mean": 0.81},
            {"model_id": "B", "sensitivity_mean": 0.80, "balanced_accuracy_mean": 0.86},
        ]
    )
    assert select_final_model(aggregate)["model_id"] == "B"


def test_the_experiment_declares_the_rule_the_code_applies() -> None:
    from src.evaluation.experiment import Experiment

    assert Experiment.load("EXP-A2").selection_rule == ("sensitivity", "balanced_accuracy")


# ---------------------------------------------------------------------------
# the planners
# ---------------------------------------------------------------------------


def test_nested_planner_key_changes_with_its_recipe() -> None:
    """Resume must not reuse a unit searched under a different budget or method."""
    from src.evaluation.tuned import NestedSearchPlanner

    class _Fold:
        label = "r0f0"
        train_uids = ("a", "b")

    fold, data = _Fold(), None
    base = NestedSearchPlanner(method="bayes", trials=12, model_trials={"M5": 5})
    same = NestedSearchPlanner(method="bayes", trials=12, model_trials={"M5": 5})
    cheaper = NestedSearchPlanner(method="bayes", trials=6, model_trials={"M5": 5})
    random = NestedSearchPlanner(method="random", trials=12, model_trials={"M5": 5})

    assert base.key_material("M4", fold, data) == same.key_material("M4", fold, data)
    assert base.key_material("M4", fold, data) != cheaper.key_material("M4", fold, data)
    assert base.key_material("M4", fold, data) != random.key_material("M4", fold, data)
    # An ensemble's recipe is its members' budgets -- M6/M7 have none of their own.
    assert set(base.key_material("M7", fold, data)["budgets"]) == {"M3", "M4", "M5"}


def test_subset_planner_refuses_to_start_its_own_search() -> None:
    """It must reuse EXP-A2's points, not produce a second, different set."""
    from src.evaluation.tuned import SubsetPlanner, TuningError

    try:
        planner = SubsetPlanner()
    except (TuningError, FileNotFoundError) as error:
        pytest.skip("SO-04 subset unavailable: " + str(error))

    assert planner.reuse_only is True
    assert planner.pipeline_config is not None
    assert planner.search_pipeline_config is None

    class _Fold:
        label = "r9f9"
        train_uids = ("a",)
        train_index = np.array([0, 1])

    class _Data:
        task = "binary"

    with pytest.raises(TuningError, match="no cached"):
        planner.search_point("M4", _Fold(), _Data())


def test_tuned_members_actually_reach_the_ensemble() -> None:
    """T65.2 -- an optimized ensemble must have optimized members, not defaults."""
    from src.ensemble.soft_voting import ensemble_members

    members = dict(ensemble_members("M7", {"M4": {"n_estimators": 137}}))
    forest = members["M4"]
    assert forest.get_params()["n_estimators"] == 137
    # The others keep their config defaults rather than being reset.
    assert dict(ensemble_members("M7"))["M4"].get_params()["n_estimators"] != 137


def test_an_unknown_member_override_is_refused() -> None:
    from src.ensemble.soft_voting import EnsembleError, ensemble_members

    with pytest.raises(EnsembleError, match="no member"):
        ensemble_members("M7", {"M1": {"C": 1.0}})


# ---------------------------------------------------------------------------
# nested behaviour on the real matrix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data() -> Any:
    from src.models import smoke as sm

    try:
        return sm.load_task_data("binary")
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 matrix unavailable (" + type(error).__name__ + "): " + str(error))


@pytest.fixture(scope="module")
def fold(data: Any) -> Any:
    from src.optimization import driver as od

    try:
        return od.outer_folds_for("binary", data, repeats=[0], folds=[0])[0]
    except Exception as error:  # noqa: BLE001 - a missing DA-07 map is a skip
        pytest.skip("DA-07 fold 0 unavailable (" + type(error).__name__ + "): " + str(error))


@pytest.mark.needs_data
def test_the_nested_search_never_touches_the_outer_test_fold(
    data: Any, fold: Any, tmp_path: Any
) -> None:
    """Rule 2, measured on a real search rather than assumed from the code path.

    Three trials of the cheapest model: enough to exercise the ledger, cheap
    enough to run in a test. The property being checked does not depend on the
    budget.
    """
    from src.evaluation.tuned import NestedSearchPlanner

    planner = NestedSearchPlanner(
        method="random", trials=3, model_trials={}, inner_splits=2, cache_dir=tmp_path
    )
    payload = planner.search_point("M1", fold, data)

    assert payload["outer_test_rows_touched"] == 0
    assert payload["n_trials"] == 3
    assert payload["n_trials_ok"] >= 1
    assert payload["fold"] == fold.label

    from src.models import spaces

    assert not spaces.load_space("M1").violations(payload["best_params"])

    # Cached, so the second call costs nothing and returns the same point.
    again = planner.search_point("M1", fold, data)
    assert again["best_params"] == payload["best_params"]
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.needs_data
def test_the_planner_hands_the_searched_point_to_the_estimator(
    data: Any, fold: Any, tmp_path: Any
) -> None:
    from src.evaluation.tuned import NestedSearchPlanner

    planner = NestedSearchPlanner(
        method="random", trials=3, model_trials={}, inner_splits=2, cache_dir=tmp_path
    )
    planned = planner.plan("M1", fold, data)
    estimator = planned.factory()
    for name, value in planned.params.items():
        assert estimator.get_params()[name] == value
    assert planned.extra["tuning"] == "nested-random"
    assert "search inside this training fold" in planned.note


# ---------------------------------------------------------------------------
# gates over what Phase 65 actually produced
# ---------------------------------------------------------------------------


def _per_fold(exp_id: str) -> Any:
    import pandas as pd

    from src.evaluation.experiment import Experiment

    path = Experiment.load(exp_id).output_dir() / "per_fold_metrics.csv"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run Phase 65")
    return pd.read_csv(path)


def test_exp_a2_every_fold_was_tuned_by_a_nested_search() -> None:
    """T65.7 -- not one row may have fallen back to defaults or a fixed point."""
    frame = _per_fold("EXP-A2")
    assert "planner_tuning" in frame.columns
    assert set(frame["planner_tuning"]) == {"nested-bayes"}
    assert len(frame) == 25 * len(set(frame["model_id"]))


def test_exp_a2_pairs_fold_for_fold_with_exp_a1() -> None:
    """T65.1 -- the outer loop is the same 25-fold map, or Phase 81 cannot pair them."""
    baseline = _per_fold("EXP-A1")
    optimized = _per_fold("EXP-A2")
    assert set(baseline["fold_label"]) == set(optimized["fold_label"])
    shared = sorted(set(baseline["model_id"]) & set(optimized["model_id"]))
    assert shared, "the two runs share no model to pair"
    for model_id in shared:
        left = baseline[baseline["model_id"] == model_id]
        right = optimized[optimized["model_id"] == model_id]
        assert sorted(left["fold_label"]) == sorted(right["fold_label"])
        merged = left.merge(right, on="fold_label", suffixes=("_a1", "_a2"))
        assert (merged["n_test_a1"] == merged["n_test_a2"]).all()


def test_the_final_model_is_persisted_with_its_feature_list() -> None:
    """T65.6 -- a reloaded model must be able to detect a column reordering."""
    from pathlib import Path

    from src.utils.config import load_config

    directory = Path(load_config("paths").require("models_saved")) / "binary" / "final"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(str(manifest_path) + " does not exist; run Phase 65")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from src.feature_extraction.registry import feature_names

    assert manifest["feature_names"] == list(feature_names())
    assert manifest["n_features"] == 138
    assert manifest["task"] == "binary"
    assert manifest["selected_model_id"]
    assert manifest["selection_rule"] == ["sensitivity", "balanced_accuracy"]
    assert manifest["hyperparameter_source"]
    assert "screening" in manifest["disclaimer"].lower()
    assert (directory / "model.joblib").is_file()


def test_t09_compares_the_two_ensembles_fold_by_fold() -> None:
    """T65.5 -- paired over the same folds, with the per-fold deltas kept."""
    import pandas as pd

    from src.evaluation.experiment import Experiment

    path = Experiment.load("EXP-A2").output_dir().parent / "T09_ensemble_weight_comparison.csv"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/13_finalize_binary_model.py")

    frame = pd.read_csv(path)
    assert "EXP-A2" in set(frame["exp_id"])
    block = frame[frame["exp_id"] == "EXP-A2"]
    assert len(block) == 25
    for metric in ("balanced_accuracy", "sensitivity", "specificity"):
        assert {"M6_" + metric, "M7_" + metric, "delta_" + metric} <= set(block.columns)
        assert np.allclose(
            block["delta_" + metric].to_numpy(dtype=float),
            (block["M7_" + metric] - block["M6_" + metric]).to_numpy(dtype=float),
            atol=1e-9,
        )
