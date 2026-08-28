"""T63.7 -- the experiment runner honours its output contract and resumes.

Two halves. The first is configuration-only and runs anywhere, including CI: it
proves every declared experiment can be built into an :class:`Experiment`, that
the contract in the code and the contract in the YAML are the same list, and
that the dependency order the chaining script computes is a real topological
order.

The second half runs a real experiment on the real D1 matrix. It skips -- never
passes -- when the matrix is absent, because ``*.parquet`` is gitignored and CI
has never seen it. Same guard as ``tests/test_feature_selection.py``.

Nothing here writes into ``outputs/``. Every run goes to ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

BASELINE_MODELS = ("M1", "M3", "M4", "M5", "M6", "M8")


# ---------------------------------------------------------------------------
# configuration only -- runs on CI
# ---------------------------------------------------------------------------


def test_output_contract_matches_the_config() -> None:
    """The contract is declared twice; the two declarations must agree.

    ``Experiment.load`` raises on a mismatch, so this is really a check that the
    guard is wired up rather than that the strings happen to be equal today.
    """
    from src.evaluation.experiment import OUTPUT_CONTRACT
    from src.utils.config import load_config

    declared = tuple(load_config("experiments").require("defaults")["output_contract"])
    assert declared == OUTPUT_CONTRACT


@pytest.mark.parametrize(
    "exp_id",
    ["EXP-A1", "EXP-A2", "EXP-B1", "EXP-B2", "EXP-C1", "EXP-C2", "EXP-G1"],
)
def test_every_runnable_experiment_loads(exp_id: str) -> None:
    from src.evaluation.experiment import Experiment

    exp = Experiment.load(exp_id)
    assert exp.exp_id == exp_id
    assert exp.task
    assert exp.models
    assert exp.cv
    assert exp.output_section.startswith(("06_", "07_", "08_", "09_", "10_"))
    assert exp.seed == 42
    assert exp.scoring in {"balanced_accuracy", "macro_f1"}
    assert exp.report_metrics
    # T63.3 -- outputs/<section>/<EXP-ID>/, resolved through configs/paths.yaml.
    assert exp.output_dir().name == exp_id


@pytest.mark.parametrize("exp_id", ["EXP-C3", "EXP-E1", "EXP-E2"])
def test_analysis_only_experiments_are_refused(exp_id: str) -> None:
    """An analysis-only run has no fold map of its own; running it would be wrong."""
    from src.evaluation.experiment import Experiment, ExperimentError

    with pytest.raises(ExperimentError, match="analysis_only"):
        Experiment.load(exp_id)


@pytest.mark.parametrize("exp_id", ["EXP-F1", "EXP-F2"])
def test_ablation_grids_are_refused_until_their_phase(exp_id: str) -> None:
    """EXP-F1/F2 loop over configurations, not models. Phase 74/75 handles them.

    Asserted rather than left to fail later so the gap is a recorded decision:
    the runner must not quietly invent a model list for an ablation grid.
    """
    from src.evaluation.experiment import Experiment, ExperimentError

    with pytest.raises(ExperimentError, match="ablation grid"):
        Experiment.load(exp_id)


def test_label_spaces_are_never_merged() -> None:
    """Research rule 4, asserted on the declarations the runner actually reads."""
    from src.evaluation.experiment import Experiment

    tasks = {
        exp_id: Experiment.load(exp_id).task
        for exp_id in ("EXP-A1", "EXP-A2", "EXP-B1", "EXP-B2", "EXP-C1", "EXP-C2")
    }
    assert tasks["EXP-A1"] == tasks["EXP-A2"] == "binary"
    assert tasks["EXP-B1"] == "pascal_a"
    assert tasks["EXP-B2"] == "pascal_b"
    assert tasks["EXP-C1"] == "circor_murmur"
    assert tasks["EXP-C2"] == "circor_outcome"
    assert len(set(tasks.values())) == 5


def test_dependency_order_is_topological() -> None:
    """T63.6 -- every dependency appears before the run that needs it."""
    import importlib.util

    from src.utils.config import load_config

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_all_experiments", root / "scripts" / "12_run_all_experiments.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    order = module.dependency_order()
    table = load_config("experiments").require("experiments")
    assert set(order) == set(table)
    position = {exp_id: index for index, exp_id in enumerate(order)}
    for exp_id, declaration in table.items():
        for dependency in declaration.get("depends_on", []):
            assert position[dependency] < position[exp_id], (
                exp_id + " runs before its dependency " + dependency
            )


# ---------------------------------------------------------------------------
# fixtures over the real matrix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data() -> Any:
    from src.models import smoke as sm

    try:
        return sm.load_task_data("binary")
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 matrix unavailable (" + type(error).__name__ + "): " + str(error))


@pytest.fixture(scope="module")
def experiment() -> Any:
    from src.evaluation.experiment import Experiment

    return Experiment.load("EXP-A1")


@pytest.fixture(scope="module")
def one_run(data: Any, experiment: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """M1 over repeat 0 -- five real folds of the real matrix, in a temp tree.

    M1 is the cheapest of the six declared models (~0.2 s a fold), so this is a
    complete end-to-end pass of the runner for about a second of CPU. It proves
    the plumbing, not a result.
    """
    from src.evaluation import experiment as ex

    directory = tmp_path_factory.mktemp("exp63")
    result = ex.run_experiment(
        experiment, data=data, models=["M1"], repeats=[0], out_dir=directory
    )
    written = ex.write_outputs(result, out_dir=directory, command="pytest")
    return result, written, directory


# ---------------------------------------------------------------------------
# T63.7 -- the output contract
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_output_contract_is_complete(one_run: Any) -> None:
    from src.evaluation.experiment import OUTPUT_CONTRACT

    _, _, directory = one_run
    target = directory / "EXP-A1"
    for name in OUTPUT_CONTRACT:
        path = target / name
        assert path.is_file(), "missing contract file " + name
        assert path.stat().st_size > 0, name + " is empty"


@pytest.mark.needs_data
def test_per_fold_metrics_hold_one_row_per_fold(one_run: Any) -> None:
    import pandas as pd

    _, _, directory = one_run
    frame = pd.read_csv(directory / "EXP-A1" / "per_fold_metrics.csv")
    assert len(frame) == 5
    assert sorted(frame["fold_label"]) == ["r0f0", "r0f1", "r0f2", "r0f3", "r0f4"]
    # Rule 6 -- never accuracy alone.
    for column in ("sensitivity", "specificity", "f1", "balanced_accuracy", "roc_auc"):
        assert column in frame.columns
        assert frame[column].notna().all()


@pytest.mark.needs_data
def test_aggregate_is_derived_from_the_per_fold_values(one_run: Any) -> None:
    """The mean and SD must be recomputable from the per-fold table.

    T64.6 needs the individual fold values kept for the Phase 81 paired tests,
    so the aggregate has to be a *view* of them rather than a separately
    accumulated number that could drift.
    """
    import pandas as pd

    _, _, directory = one_run
    per_fold = pd.read_csv(directory / "EXP-A1" / "per_fold_metrics.csv")
    aggregate = pd.read_csv(directory / "EXP-A1" / "aggregate_metrics.csv")
    row = aggregate[aggregate["model_id"] == "M1"].iloc[0]

    for metric in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc"):
        values = per_fold[metric].to_numpy(dtype=float)
        assert row[metric + "_mean"] == pytest.approx(float(np.mean(values)), abs=1e-9)
        assert row[metric + "_sd"] == pytest.approx(float(np.std(values, ddof=1)), abs=1e-9)
        assert int(row[metric + "_n"]) == len(values)
    assert int(row["n_folds"]) == 5


@pytest.mark.needs_data
def test_predictions_cover_every_test_row_once_per_repeat(one_run: Any, data: Any) -> None:
    import pandas as pd

    result, _, directory = one_run
    predictions = pd.read_parquet(directory / "EXP-A1" / "predictions.parquet")
    expected = sum(len(fold.test_uids) for fold in result.folds)
    assert len(predictions) == expected
    # One repeat, so every record is predicted exactly once.
    assert predictions["record_uid"].nunique() == len(predictions)
    assert set(predictions["record_uid"]) == set(data.record_uids)
    assert {"proba_0", "proba_1"} <= set(predictions.columns)
    totals = predictions[["proba_0", "proba_1"]].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)


@pytest.mark.needs_data
def test_confusion_matrices_agree_with_the_predictions(one_run: Any) -> None:
    import pandas as pd

    _, _, directory = one_run
    payload = json.loads((directory / "EXP-A1" / "confusion_matrices.json").read_text())
    predictions = pd.read_parquet(directory / "EXP-A1" / "predictions.parquet")

    assert payload["labels"] == [0, 1]
    assert payload["class_names"] == ["normal", "abnormal"]
    total = np.asarray(payload["models"]["M1"]["total"], dtype=int)
    assert total.sum() == len(predictions)
    for true_label in (0, 1):
        for predicted_label in (0, 1):
            expected = int(
                (
                    (predictions["y_true"] == true_label)
                    & (predictions["y_pred"] == predicted_label)
                ).sum()
            )
            assert total[true_label, predicted_label] == expected


@pytest.mark.needs_data
def test_config_snapshot_records_what_actually_ran(one_run: Any) -> None:
    import yaml

    _, _, directory = one_run
    snapshot = yaml.safe_load((directory / "EXP-A1" / "config_snapshot.yaml").read_text())
    assert snapshot["exp_id"] == "EXP-A1"
    assert snapshot["seed"] == 42
    assert snapshot["cv"] == "repeated_5x5_grouped"
    assert snapshot["cv_scheme"]["total_folds"] == 25
    # The subset actually run is recorded, so a five-fold run can never be read
    # as the 25-fold protocol.
    assert snapshot["run"]["models_run"] == ["M1"]
    assert snapshot["run"]["n_folds_run"] == 5
    assert snapshot["run"]["folds_run"] == ["r0f0", "r0f1", "r0f2", "r0f3", "r0f4"]


@pytest.mark.needs_data
def test_run_manifest_is_embedded_beside_the_results(one_run: Any) -> None:
    """T63.3 -- a results folder must be attributable without a lookup elsewhere."""
    _, _, directory = one_run
    manifest = json.loads((directory / "EXP-A1" / "run_manifest.json").read_text())
    assert manifest.get("seed") == 42
    assert manifest.get("package_versions")
    assert manifest.get("git") is not None


@pytest.mark.needs_data
def test_fold_membership_has_no_subject_on_both_sides(one_run: Any, data: Any) -> None:
    """Research rule 3, checked against the membership that was written down."""
    import pandas as pd

    _, _, directory = one_run
    membership = pd.read_parquet(directory / "EXP-A1" / "fold_membership.parquet")
    group_of = dict(zip(data.record_uids, [str(g) for g in data.groups], strict=True))

    for label, block in membership.groupby("fold_label"):
        train = block[block["split"] == "train"]["record_uid"]
        test = block[block["split"] == "test"]["record_uid"]
        assert not (set(train) & set(test)), label + ": a record is in both splits"
        train_groups = {group_of[uid] for uid in train}
        test_groups = {group_of[uid] for uid in test}
        assert not (train_groups & test_groups), label + ": a subject is in both splits"


# ---------------------------------------------------------------------------
# T63.4 -- resume-on-restart
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_resume_skips_a_completed_fold(one_run: Any, data: Any, experiment: Any) -> None:
    from src.evaluation import experiment as ex

    first, _, directory = one_run
    assert first.n_computed == 5
    assert first.n_resumed == 0

    second = ex.run_experiment(
        experiment, data=data, models=["M1"], repeats=[0], out_dir=directory
    )
    assert second.n_computed == 0
    assert second.n_resumed == 5

    # A resumed unit must be the same measurement, not merely the same shape.
    before = first.per_fold_frame().set_index("fold_label")
    after = second.per_fold_frame().set_index("fold_label")
    for metric in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc", "brier"):
        assert np.allclose(
            before[metric].to_numpy(dtype=float),
            after[metric].to_numpy(dtype=float),
            atol=0,
        ), metric + " changed when resumed"


@pytest.mark.needs_data
def test_a_stale_checkpoint_is_recomputed_not_reused(
    one_run: Any, data: Any, experiment: Any
) -> None:
    """Resume is keyed on what would change the numbers, not on the file existing.

    A checkpoint whose key no longer matches is the dangerous case: reusing it
    produces a results table that is partly from one configuration and partly
    from another, with nothing on disk saying so.
    """
    from src.evaluation import experiment as ex

    _, _, directory = one_run
    checkpoint = directory / "EXP-A1" / ex.CHECKPOINT_DIRNAME / "M1__r0f0.json"
    payload = json.loads(checkpoint.read_text())
    payload["unit_key"] = "0" * 16
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    result = ex.run_experiment(
        experiment, data=data, models=["M1"], repeats=[0], out_dir=directory
    )
    assert result.n_computed == 1
    assert result.n_resumed == 4


@pytest.mark.needs_data
def test_no_resume_recomputes_everything(one_run: Any, data: Any, experiment: Any) -> None:
    from src.evaluation import experiment as ex

    _, _, directory = one_run
    result = ex.run_experiment(
        experiment, data=data, models=["M1"], repeats=[0], out_dir=directory, resume=False
    )
    assert result.n_computed == 5
    assert result.n_resumed == 0


@pytest.mark.needs_data
def test_the_runner_refuses_an_undeclared_model(data: Any, experiment: Any, tmp_path: Path) -> None:
    from src.evaluation import experiment as ex

    with pytest.raises(ex.ExperimentError, match="does not declare"):
        ex.run_experiment(
            experiment, data=data, models=["M7"], repeats=[0], out_dir=tmp_path
        )


# ---------------------------------------------------------------------------
# gates over what is on disk, once Phase 64/65 have run
# ---------------------------------------------------------------------------


def _committed(exp_id: str, name: str) -> Any:
    import pandas as pd

    from src.evaluation.experiment import Experiment

    path = Experiment.load(exp_id).output_dir() / name
    if not path.is_file():
        pytest.skip(
            str(path) + " does not exist; run scripts/11_run_experiment.py --exp " + exp_id
        )
    return pd.read_csv(path)


@pytest.mark.parametrize("exp_id", ["EXP-A1", "EXP-A2"])
def test_committed_run_used_the_full_fold_map(exp_id: str) -> None:
    """T64.7 / T65.7 -- the shipped table covers all 25 folds, not a subset."""
    frame = _committed(exp_id, "per_fold_metrics.csv")
    from src.evaluation.experiment import Experiment

    exp = Experiment.load(exp_id)
    expected = int(exp.cv_scheme()["total_folds"])
    for model_id, block in frame.groupby("model_id"):
        assert len(block) == expected, model_id + " ran " + str(len(block)) + " folds"
        assert sorted(block["fold_label"]) == sorted(
            "r" + str(r) + "f" + str(f) for r in range(5) for f in range(5)
        )
    assert set(frame["model_id"]) == set(exp.models)
