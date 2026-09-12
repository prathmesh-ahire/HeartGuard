"""Deployed task models: the selection rule, and what the manifests may not claim.

`models_saved/<task>/final/` is what the prediction pages score against. Two
things have to hold and neither is self-evident from the diff:

1. The label space the deployed model was fitted on is the label space the API
   declares for that task. A murmur model fitted on `circor_murmur` codes and
   served under `("Absent", "Present", "Unknown")` is only correct because those
   two orderings agree, and nothing enforced that agreement before this file.
2. No manifest may present an in-sample fit as a performance estimate. The
   deployed models are refit on every labelled record; their numbers live in
   the cross-validated run the manifest names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.inference.predictor import TASKS
from src.models import deploy
from src.utils.constants import label_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_every_task_other_than_binary_has_a_declared_deployment() -> None:
    """Binary is T65.6's; the other four are this module's, and none is left out."""
    declared = {spec.task for spec in deploy.DEPLOYMENTS}
    assert declared == set(TASKS) - {"binary"}


def test_each_deployment_fits_the_label_space_its_api_task_declares() -> None:
    for spec in deploy.DEPLOYMENTS:
        assert label_names(spec.data_task) == TASKS[spec.task].classes, spec.task


def test_the_selection_rule_is_never_accuracy() -> None:
    """Research rule 6: sensitivity and balanced accuracy decide, never accuracy."""
    for spec in deploy.DEPLOYMENTS:
        assert "accuracy" not in spec.rule, spec.task
        assert spec.rule[0] in {"sensitivity", "macro_f1"}, spec.task
        assert spec.rule[-1] == "balanced_accuracy", spec.task


def test_a_binary_task_is_ranked_on_sensitivity_and_a_multiclass_one_on_macro_f1() -> None:
    by_task = {spec.task: spec for spec in deploy.DEPLOYMENTS}
    assert by_task["outcome"].rule[0] == "sensitivity"
    for task in ("pascal_a", "pascal_b", "murmur"):
        assert by_task[task].rule[0] == "macro_f1"


def test_the_murmur_deployment_ranks_the_headline_three_class_variant() -> None:
    """The page declares three classes; the two-class run is the 874-patient one."""
    spec = next(item for item in deploy.DEPLOYMENTS if item.task == "murmur")
    assert spec.run_dir.endswith("EXP-C1-three_class")
    assert len(TASKS["murmur"].classes) == 3


def test_the_deploy_fold_holds_every_row_and_tests_none() -> None:
    """A deployment fit is in sample by construction; nothing is held out to score."""

    @dataclass
    class _Data:
        task: str = "binary"
        y: tuple[int, ...] = (0, 1, 0, 1)
        groups: tuple[str, ...] = ("a", "b", "c", "d")
        record_uids: tuple[str, ...] = ("u1", "u2", "u3", "u4")

    fold = deploy._deploy_fold(_Data())
    assert fold.n_train == 4
    assert fold.n_test == 0
    assert fold.label == "r-1f-1"
    assert fold.scheme == deploy.DEPLOY_FOLD_LABEL


@pytest.mark.parametrize("spec", deploy.DEPLOYMENTS, ids=lambda spec: spec.task)
def test_the_ranking_reads_that_task_s_own_committed_run(spec: deploy.DeploymentSpec) -> None:
    source = PROJECT_ROOT / spec.run_dir / "aggregate_metrics.csv"
    if not source.is_file():
        pytest.skip(spec.run_dir + " is not present in this checkout")
    selection = deploy.rank_models(spec)
    assert selection["ranking_source"].endswith("aggregate_metrics.csv")
    assert selection["n_models_ranked"] >= 5
    # The ranking is sorted by the declared rule, most important metric first.
    leading = [row[spec.rule[0]] for row in selection["ranking"]]
    assert leading == sorted(leading, reverse=True)
    assert selection["selected_model_id"] == selection["ranking"][0]["model_id"]


@pytest.mark.parametrize("task", sorted(TASKS))
def test_a_saved_manifest_names_its_estimate_elsewhere(task: str) -> None:
    """An in-sample fit must never read as its own performance estimate."""
    manifest_path = PROJECT_ROOT / "models_saved" / TASKS[task].model_dir / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no deployed model for " + task + " in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fold"] == "all-records"
    assert manifest["task"] == task
    assert manifest["n_features"] == 138
    assert "accuracy" not in manifest["selection_rule"]
    assert "NOT anything measured on these rows" in manifest["note"]
    assert "Not a diagnostic device" in manifest["disclaimer"]
    assert manifest["n_records_fitted"] > 0


@pytest.mark.parametrize("spec", deploy.DEPLOYMENTS, ids=lambda spec: spec.task)
def test_a_saved_model_records_the_search_that_chose_its_point(
    spec: deploy.DeploymentSpec,
) -> None:
    manifest_path = PROJECT_ROOT / "models_saved" / spec.task / "final" / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no deployed model for " + spec.task + " in this checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert deploy.DEPLOY_FOLD_LABEL in manifest["hyperparameter_source"]
    assert manifest["published_table"] == spec.table_id
    assert manifest["data_task"] == spec.data_task
    assert manifest["n_classes"] == len(TASKS[spec.task].classes)
    searched = manifest["search"]
    assert searched, spec.task
    for payload in searched.values():
        assert payload["n_trials_ok"] > 0
        assert payload["n_inner_folds"] >= 2
