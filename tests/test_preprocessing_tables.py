"""Preprocessing tables, ablation grid and evidence registration (T29.7).

The T29.7 gate has three clauses: PP-07 is generated from the live config, PP-09
ran all four filter/normalization configurations, and every PP artifact is
registered in the evidence index.

The middle clause was unrunnable at Phase 29 -- PP-09 needs the locked 138
features and a trained model, which Parts IV and V build -- and until Phase 46
existed this module instead asserted that the gap was *declared*: four arms
defined and distinct, PP-09 registered as ``missing`` rather than quietly
absent, and the deferral written into ``missing_outputs_report.txt``. That
placeholder test (``test_pp09_is_a_declared_gap_not_a_silent_one``) was written
to fail the moment PP-09 appeared, as the prompt to close the deferral. It has
been replaced by the assertions below, which check the table itself: four arms,
one varying factor pair, deltas against the shipped configuration, and means
that reconcile with the per-fold values behind them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.preprocessing import ablation, artifacts

# ===========================================================================
# T29.1 -- PP-07
# ===========================================================================


def test_settings_table_is_generated_from_the_live_config(
    signal_config: Any, tmp_path: Path
) -> None:
    """A hand-typed settings table is a hand-typed number (rule 1)."""
    import pandas as pd

    path = ablation.write_settings(tmp_path, signal_config)
    table = pd.read_csv(path, keep_default_na=False)

    assert list(table.columns) == list(ablation.SETTINGS_COLUMNS)
    assert set(table["stage"]) == {"resample", "filter", "normalization", "framing"}

    def value_of(setting: str) -> str:
        rows = table[table["setting"] == setting]
        assert len(rows) == 1, setting
        return str(rows.iloc[0]["value"])

    assert value_of("target sampling rate") == str(signal_config.require("resample.target_fs"))
    assert value_of("low cutoff") == str(signal_config.require("filter.low_hz"))
    assert value_of("high cutoff") == str(signal_config.require("filter.high_hz"))
    assert value_of("design order") == str(signal_config.require("filter.order"))
    assert value_of("resampling method") == str(signal_config.require("resample.method"))
    assert value_of("normalization method") == str(signal_config.require("normalization.method"))


def test_setting_names_are_unique_across_the_whole_table() -> None:
    """T04 is read on the page, without the stage column to disambiguate it."""
    names = [row["setting"] for row in ablation.settings_rows()]
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, duplicates


def test_settings_table_reports_the_effective_order_not_just_the_prototype() -> None:
    """The -6 dB / 16th-order distinction has to survive into the thesis table."""
    rows = {row["setting"]: row["value"] for row in ablation.settings_rows()}
    assert rows["design order"] == 4
    assert rows["effective order applied"] == 16


def test_settings_table_follows_a_config_change(signal_config: Any) -> None:
    import copy

    from src.utils.config import Config

    data = copy.deepcopy(signal_config.as_dict())
    data["filter"]["low_hz"] = 25
    rows = {row["setting"]: row["value"] for row in ablation.settings_rows(Config("signal", data))}
    assert rows["low cutoff"] == 25


# ===========================================================================
# T29.2 -- the four arms
# ===========================================================================


def test_grid_is_the_two_by_two_the_task_asks_for() -> None:
    assert len(ablation.ABLATION_GRID) == 4
    combinations = {
        (arm.filter_enabled, arm.normalization_enabled) for arm in ablation.ABLATION_GRID
    }
    assert combinations == {(True, True), (True, False), (False, True), (False, False)}

    shipped = [
        arm
        for arm in ablation.ABLATION_GRID
        if arm.filter_enabled and arm.normalization_enabled
    ]
    assert len(shipped) == 1
    assert shipped[0].arm_id == "PP-A"


def test_each_arm_resolves_to_a_distinct_config_hash() -> None:
    """Four arms that share a hash would be one arm run four times."""
    rows = ablation.arm_rows()
    hashes = [row["config_hash"] for row in rows]
    assert len(set(hashes)) == 4, rows

    shipped = [row for row in rows if row["is_shipped_configuration"]]
    assert len(shipped) == 1
    assert shipped[0]["arm_id"] == "PP-A"


def test_grid_cache_dirs_are_relative_to_the_project() -> None:
    """An absolute D:\\... path in a deliverable is wrong on the next machine."""
    for row in ablation.arm_rows():
        assert not Path(row["cache_dir"]).is_absolute(), row
        assert row["cache_dir"].startswith("cache/preprocessed/")


def test_grid_is_written_with_its_columns(tmp_path: Path) -> None:
    import pandas as pd

    path = ablation.write_grid(tmp_path)
    table = pd.read_csv(path)
    assert list(table.columns) == list(ablation.GRID_COLUMNS)
    assert len(table) == 4


@pytest.mark.needs_data
def test_the_four_arms_actually_produce_different_signals(
    master_frame: Any, project_root: Path
) -> None:
    """The claim the grid rests on, checked on a real recording."""
    import numpy as np

    from src.preprocessing.pipeline import preprocess

    row = master_frame.iloc[0]
    signals = {}
    for arm in ablation.ABLATION_GRID:
        result = preprocess(
            project_root / str(row["file_path"]),
            ablation.arm_config(arm),
            record_uid=str(row["record_uid"]),
            use_cache=False,
        )
        signals[arm.arm_id] = result.signal

    ids = list(signals)
    for i, first in enumerate(ids):
        for second in ids[i + 1 :]:
            assert not np.array_equal(signals[first], signals[second]), (first, second)

    # The normalized arms are on a unit scale; the un-normalized ones are not.
    assert float(signals["PP-A"].std()) == pytest.approx(1.0, abs=1e-3)
    assert float(signals["PP-C"].std()) == pytest.approx(1.0, abs=1e-3)
    assert float(signals["PP-B"].std()) < 0.5
    assert float(signals["PP-D"].std()) < 0.5


# ===========================================================================
# T29.5 / T29.7 -- registration and the declared gap
# ===========================================================================


def test_the_manifest_covers_pp01_through_pp09() -> None:
    ids = [entry[0] for entry in artifacts.PP_ARTIFACTS]
    assert ids == ["PP-0" + str(n) for n in range(1, 10)]
    for _, filename, description, source in artifacts.PP_ARTIFACTS:
        assert filename.endswith((".png", ".csv"))
        assert description and source


def test_every_pp_artifact_registers_with_its_real_status(tmp_path: Path) -> None:
    """Status comes from the filesystem, so a row can never claim a missing file."""
    index = tmp_path / "evidence_index.csv"
    directory = tmp_path / "02_preprocessing"
    directory.mkdir()
    (directory / "preprocessing_settings.csv").write_text("stage\n", encoding="utf-8")

    rows = artifacts.register_preprocessing_artifacts(directory, index_path=index)
    status = {row["evidence_id"]: row["status"] for row in rows}

    assert status["PP-07"] == "ok"
    assert status["PP-01"] == "missing"
    assert status["PP-09"] == "missing"


def test_real_index_has_every_pp_artifact_registered() -> None:
    """T29.7 -- registration, against the committed evidence index."""
    from src.utils.evidence import read_evidence

    registered = {row["evidence_id"]: row for row in read_evidence()}
    for evidence_id, _, _, _ in (*artifacts.PP_ARTIFACTS, *artifacts.SUPPORTING_ARTIFACTS):
        assert evidence_id in registered, evidence_id + " is not in the evidence index"
        assert registered[evidence_id]["source_data"], evidence_id + " has no source_data"


def test_pp01_through_pp09_exist_on_disk() -> None:
    """T29.7 -- PP-07, PP-08 and PP-09 exist, and so do the six figures."""
    state = artifacts.verify_preprocessing_artifacts()
    for expected in ("PP-0" + str(n) for n in range(1, 10)):
        assert expected in state["present"], expected + " is missing from " + state["directory"]


# ===========================================================================
# T29.3 / T29.4 -- PP-09, the ablation that was deferred at Phase 29
# ===========================================================================


def _pp09() -> Any:
    import pandas as pd

    from src.preprocessing import ablation_run

    path = ablation_run.ablation_path()
    if not path.is_file():
        pytest.skip("PP-09 has not been generated: " + str(path))
    return pd.read_csv(path)


def test_pp09_ran_all_four_filter_normalization_configurations() -> None:
    """The T29.7 clause that could not pass until Phase 46 existed."""
    table = _pp09()
    assert len(table) == 4, table["arm_id"].tolist()
    combinations = {
        (bool(row["filter_enabled"]), bool(row["normalization_enabled"]))
        for row in table.to_dict("records")
    }
    assert combinations == {(True, True), (True, False), (False, True), (False, False)}
    assert set(table["arm_id"]) == {arm.arm_id for arm in ablation.ABLATION_GRID}


def test_pp09_arms_differ_only_in_preprocessing() -> None:
    """One model, one task, one fold map. Otherwise it is four unrelated runs."""
    table = _pp09()
    for column in ("model_id", "task", "scheme", "n_folds", "n_records", "n_subjects"):
        assert table[column].nunique() == 1, column + " varies across the arms"
    assert table["config_hash"].nunique() == 4, "two arms shared a preprocessing config"


def test_pp09_deltas_are_taken_against_the_shipped_configuration() -> None:
    from src.preprocessing import ablation_run

    table = _pp09()
    shipped = table[table["is_shipped_configuration"]]
    assert len(shipped) == 1
    assert str(shipped.iloc[0]["arm_id"]) == ablation_run.REFERENCE_ARM

    for metric in ablation_run.METRICS:
        assert metric + "_delta" in table.columns, metric
        # The reference arm's delta against itself is exactly zero, not nearly.
        assert float(shipped.iloc[0][metric + "_delta"]) == 0.0, metric
        others = table[~table["is_shipped_configuration"]]
        recomputed = others[metric] - float(shipped.iloc[0][metric])
        assert others[metric + "_delta"].to_numpy() == pytest.approx(
            recomputed.to_numpy(), abs=1e-12
        ), metric


def test_pp09_reports_more_than_accuracy() -> None:
    """Rule 6, in the one table whose whole point is a single delta."""
    table = _pp09()
    for metric in ("sensitivity", "specificity", "f1", "balanced_accuracy", "roc_auc"):
        assert metric in table.columns
        assert table[metric].notna().all(), metric


def test_pp09_means_match_the_per_fold_values_behind_them() -> None:
    """A mean nobody can check is a hand-typed number with extra steps."""
    import pandas as pd

    from src.preprocessing import ablation_run

    table = _pp09()
    path = ablation_run.per_fold_path()
    if not path.is_file():
        pytest.skip("per-fold values not written: " + str(path))
    per_fold = pd.read_csv(path)

    for row in table.to_dict("records"):
        block = per_fold[per_fold["arm_id"] == row["arm_id"]]
        assert len(block) == int(row["n_folds"]), row["arm_id"]
        for metric in ablation_run.METRICS:
            values = block[metric].to_numpy(dtype=float)
            finite = values[~pd.isna(values)]
            assert float(row[metric]) == pytest.approx(float(finite.mean()), abs=1e-9), (
                row["arm_id"],
                metric,
            )
