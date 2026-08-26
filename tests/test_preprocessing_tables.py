"""Preprocessing tables, ablation grid and evidence registration (T29.7).

The T29.7 gate has three clauses. Two can pass today: PP-07 exists and is
generated from the live config, and every PP artifact is registered in the
evidence index. The third -- "the ablation ran all four filter/normalization
configurations" -- cannot, because PP-09 needs features and a model that Parts
IV and V build. That is recorded as a deferral in
``outputs/missing_outputs_report.txt`` and asserted here: the four arms must be
defined, distinct, and ready to run, and PP-09 must be registered as *missing*
rather than quietly absent.

``test_pp09_is_a_declared_gap_not_a_silent_one`` is the test that will start
failing once PP-09 is produced. That is intentional -- it is the reminder to
close the deferral rather than leave it open forever.
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


def test_pp01_through_pp08_exist_on_disk() -> None:
    """T29.7 -- PP-07 and PP-08 exist, and so do the six figures."""
    state = artifacts.verify_preprocessing_artifacts()
    for expected in ("PP-0" + str(n) for n in range(1, 9)):
        assert expected in state["present"], expected + " is missing from " + state["directory"]


def test_pp09_is_a_declared_gap_not_a_silent_one() -> None:
    """T29.3/T29.4 are deferred, and the deferral is written down.

    When PP-09 is finally produced (after Phase 46), this test fails -- on
    purpose. That failure is the prompt to close the deferral: flip T29.3/T29.4
    in todo.md, resolve the missing_outputs_report entry, and delete this test.
    """
    from src.utils.config import load_config

    state = artifacts.verify_preprocessing_artifacts()
    if "PP-09" in state["present"]:
        pytest.fail(
            "PP-09 now exists -- close the T29.3/T29.4 deferral in Docs/todo.md and "
            "outputs/missing_outputs_report.txt, then delete this test"
        )

    report = Path(load_config("paths").require("outputs.missing_outputs_report"))
    text = report.read_text(encoding="utf-8")
    assert "T29.3" in text and "PP-09" in text, "the PP-09 gap is not recorded"
    assert "after Phase 46" in text, "the deferral records no re-entry point"
