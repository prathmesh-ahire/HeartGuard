"""T55.7 / T56.7 -- the emitted SO-01 and SO-02 artifacts are complete and honest.

These are gate tests over what is actually on disk in
``outputs/05_search_optimization/``, not over what the code would produce if
run. The distinction matters: every other test in this repository proves the
search *can* behave, and this one proves the files the write-up will quote were
produced by a search that did.

Each test skips when its inputs are absent, so a fresh clone that has not run
the searches yet reports "skipped", never a false pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SEARCHED_MODELS = ("M1", "M3", "M4", "M5", "M8")
BAYES_MODELS = ("M3", "M4", "M5")


def _section() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.search_optimization"))


def _load(exp: str, name: str) -> Any:
    import pandas as pd

    path = _section() / exp / name
    if not path.exists():
        pytest.skip(str(path) + " does not exist; run scripts/05_run_search.py first")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# T55.7 -- SO-01
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def so01_trials() -> Any:
    return _load("SO-01", "trials.csv")


@pytest.fixture(scope="module")
def so01_best() -> Any:
    return _load("SO-01", "best_parameters.json")


def test_so01_has_a_trial_history_for_every_searched_model(so01_trials: Any) -> None:
    present = set(so01_trials["model_id"].unique())
    assert set(SEARCHED_MODELS) <= present, sorted(set(SEARCHED_MODELS) - present)


def test_so01_trial_history_is_complete_not_summarised(so01_trials: Any) -> None:
    """One row per trial, numbered from zero with no gaps, per model and fold."""
    for (model_id, fold), block in so01_trials.groupby(["model_id", "outer_fold"]):
        assert block["trial"].tolist() == list(range(len(block))), (model_id, fold)
        assert block["seconds"].gt(0).all(), (model_id, fold)
        assert block["n_inner_folds"].eq(block["n_inner_folds"].iloc[0]).all()


def test_so01_records_every_parameter_of_every_searched_space(so01_trials: Any) -> None:
    from src.models import spaces

    for model_id in SEARCHED_MODELS:
        block = so01_trials[so01_trials["model_id"] == model_id]
        if block.empty:
            pytest.skip(model_id + " was not searched")
        space = spaces.load_space(model_id)
        for name in space.names:
            column = "param_" + name
            assert column in block.columns, (model_id, name)
            assert block[column].notna().any(), (model_id, name)


def test_so01_best_parameters_exist_for_every_model(so01_best: Any) -> None:
    searches = {entry["model_id"]: entry for entry in so01_best["searches"]}
    assert set(SEARCHED_MODELS) <= set(searches)
    for model_id in SEARCHED_MODELS:
        entry = searches[model_id]
        assert entry["best_params"], model_id
        assert entry["best_score"] is not None, model_id
        assert entry["n_trials_ok"] > 0, model_id


def test_so01_best_parameters_are_legal_points_in_the_declared_space(so01_best: Any) -> None:
    """A best point outside its own space would mean the log and the space disagree."""
    from src.models import spaces

    for entry in so01_best["searches"]:
        space = spaces.load_space(entry["model_id"])
        assert space.is_valid(entry["best_params"]), (
            entry["model_id"],
            space.violations(entry["best_params"]),
        )


def test_so01_searches_touched_no_outer_test_row(so01_best: Any) -> None:
    """The T55.7 clause: the search ran on inner folds only.

    Read from the ledger count each search recorded at run time, so this is the
    number of test rows the estimators were actually handed -- not a re-derivation
    of what they should have been handed.
    """
    for entry in so01_best["searches"]:
        assert entry["outer_test_rows_touched"] == 0, entry["model_id"]
        assert entry["n_outer_test_rows"] > 0, entry["model_id"]
        assert entry["n_rows_touched"] > 0, entry["model_id"]


def test_so01_nested_outcomes_are_scored_on_the_outer_fold() -> None:
    frame = _load("SO-01", "nested_outcomes.csv")
    assert not frame.empty
    for column in ("inner_best_score", "outer_score", "n_train", "n_test"):
        assert column in frame.columns
    assert frame["n_test"].gt(0).all()
    assert frame["outer_score"].notna().all()


# ---------------------------------------------------------------------------
# T56.7 -- SO-02
# ---------------------------------------------------------------------------


def test_so02_capability_is_recorded_either_way() -> None:
    """Whether skopt ran or not, the reason is on disk (T56.1, T56.6)."""
    path = _section() / "SO-02" / "search_capability.json"
    from src.utils.config import load_config

    report = Path(load_config("paths").require("outputs.missing_outputs_report"))
    if not path.exists():
        if report.exists() and "SO-02" in report.read_text(encoding="utf-8"):
            return
        pytest.skip("SO-02 has not been run and no skip reason is recorded yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "available" in payload
    if not payload["available"]:
        assert payload["reason"], "an unavailable method must say why"
        assert "SO-02" in report.read_text(encoding="utf-8")


def test_so02_convergence_trace_exists_and_is_monotone() -> None:
    """T56.4: best-so-far versus trial index. It can only ever go up."""
    frame = _load("SO-02", "convergence.csv")
    assert not frame.empty
    for (model_id, fold), block in frame.groupby(["model_id", "outer_fold"]):
        best = block.sort_values("trial")["best_so_far"].ffill()
        assert best.is_monotonic_increasing, (model_id, fold)
        assert block["elapsed_seconds"].is_monotonic_increasing, (model_id, fold)


def test_so02_searched_the_primary_models() -> None:
    payload = _load("SO-02", "best_parameters.json")
    searched = {entry["model_id"] for entry in payload["searches"]}
    assert set(BAYES_MODELS) <= searched, sorted(set(BAYES_MODELS) - searched)


def test_so02_searches_touched_no_outer_test_row() -> None:
    payload = _load("SO-02", "best_parameters.json")
    for entry in payload["searches"]:
        assert entry["outer_test_rows_touched"] == 0, entry["model_id"]


# ---------------------------------------------------------------------------
# T56.5/T56.7 -- the equal-budget comparison
# ---------------------------------------------------------------------------


def test_equal_budget_comparison_exists_and_is_matched() -> None:
    import pandas as pd

    path = _section() / "search_method_comparison.csv"
    if not path.exists():
        pytest.skip(str(path) + " does not exist; run 05_run_search.py --compare")
    frame = pd.read_csv(path)
    assert not frame.empty
    for column in (
        "inner_best_score_random",
        "inner_best_score_bayes",
        "outer_score_random",
        "outer_score_bayes",
        "inner_delta_bayes_minus_random",
        "outer_delta_bayes_minus_random",
    ):
        assert column in frame.columns
    # Matched pairs: one row per (model, outer fold), never a model twice.
    assert not frame.duplicated(subset=["model_id", "outer_fold"]).any()


def test_the_two_methods_really_did_get_the_same_budget() -> None:
    """"Equal budget" is checked against the recorded budgets, not assumed (T56.3)."""
    import pandas as pd

    path = _section() / "search_method_comparison.csv"
    if not path.exists():
        pytest.skip("no comparison on disk yet")
    pairs = pd.read_csv(path)

    budgets: dict[str, dict[str, int]] = {}
    for exp, method in (("SO-01", "random"), ("SO-02", "bayes")):
        payload = _load(exp, "best_parameters.json")
        budgets[method] = {
            entry["model_id"]: int(entry["budget"]["max_trials"])
            for entry in payload["searches"]
        }
    for model_id in pairs["model_id"].unique():
        assert budgets["random"][model_id] == budgets["bayes"][model_id], model_id


def test_no_trial_count_exceeded_its_budget() -> None:
    """A search that overran its budget would make the comparison meaningless."""
    for exp in ("SO-01", "SO-02"):
        payload = _load(exp, "best_parameters.json")
        for entry in payload["searches"]:
            assert entry["n_trials"] <= entry["budget"]["max_trials"], (exp, entry["model_id"])
            assert entry["stop_reason"], (exp, entry["model_id"])
