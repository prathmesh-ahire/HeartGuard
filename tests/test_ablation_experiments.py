"""The three ablation and supplementary gates: T74.7, T75.7, T76.7.

Each phase's gate is asserted against the **written artifacts**, not against the
functions that wrote them. A builder can be correct and the file it produced
still be wrong -- the file is what a reader audits and what Phase 88 renders.

The heavy runs are skipped rather than failed when their outputs are absent, so
this file is meaningful in CI (where `dataset/` and the result directories do
not exist) and strict on a machine that has run them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

#: T74.7 names these exactly. A1..A6 plus the full set; A8 is the SO-04 subset,
#: whose size is whatever FE-12 chose and is therefore checked against FE-12.
EXPECTED_COUNTS: dict[str, int] = {
    "A1": 24,
    "A2": 22,
    "A3": 39,
    "A4": 24,
    "A5": 46,
    "A6": 63,
    "A7": 138,
}

ABLATION_DIR = "outputs/09_ablation"


def _root() -> Path:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT


# ===========================================================================
# T74.7 -- the feature-family ablation
# ===========================================================================


def test_t74_7_the_configurations_resolve_to_the_declared_feature_counts() -> None:
    """24 / 22 / 39 / 24 / 46 / 63 / 138, and the subset from FE-12."""
    from src.evaluation import feature_ablation as fa

    for config_id, expected in EXPECTED_COUNTS.items():
        columns = fa.configuration_columns(config_id)
        assert len(columns) == expected, (
            config_id + " resolves to " + str(len(columns)) + " features, not "
            + str(expected)
        )
        assert len(set(columns)) == len(columns), config_id + " repeats a feature"


def test_t74_7_a8_is_fe12s_subset_read_from_disk_not_reselected() -> None:
    """A8 re-selected inside the ablation would be the one arm that saw its own test rows."""
    from src.evaluation import feature_ablation as fa

    try:
        shipped = set(fa.selected_subset_names())
    except FileNotFoundError as error:
        pytest.skip(str(error))
    assert set(fa.configuration_columns("A8")) == shipped


def test_t74_7_a7_is_the_full_locked_representation() -> None:
    """A7 must be the registry, in registry order -- it is the control arm."""
    from src.evaluation import feature_ablation as fa
    from src.feature_extraction.registry import feature_names

    assert fa.configuration_columns("A7") == tuple(feature_names())


def test_t74_7_every_configuration_is_a_subset_of_the_locked_138() -> None:
    from src.evaluation import feature_ablation as fa
    from src.feature_extraction.registry import feature_names

    locked = set(feature_names())
    for config_id in fa.configuration_ids():
        extra = set(fa.configuration_columns(config_id)) - locked
        assert not extra, config_id + " names features outside the registry: " + str(extra)


def test_t74_7_the_arms_are_distinct() -> None:
    """Two arms with the same columns would be one arm run twice."""
    from src.evaluation import feature_ablation as fa

    seen: dict[tuple[str, ...], str] = {}
    for config_id in fa.configuration_ids():
        columns = fa.configuration_columns(config_id)
        assert columns not in seen, (
            config_id + " and " + seen[columns] + " use identical feature sets"
        )
        seen[columns] = config_id


@pytest.mark.needs_data
def test_t74_7_every_arm_ran_on_the_same_folds_and_models() -> None:
    """The 'identical folds and seed' clause, against the written contracts."""
    import pandas as pd

    from src.evaluation import feature_ablation as fa

    frames: dict[str, Any] = {}
    for config_id in fa.configuration_ids():
        path = fa.arm_experiment(config_id).output_dir() / "per_fold_metrics.csv"
        if path.is_file():
            frames[config_id] = pd.read_csv(path)
    if len(frames) < len(fa.configuration_ids()):
        pytest.skip(
            "only " + str(len(frames)) + " of " + str(len(fa.configuration_ids()))
            + " EXP-F1 arms have been run"
        )

    reference_id, reference = next(iter(frames.items()))
    ref_units = set(zip(reference["model_id"], reference["repeat"], reference["fold"], strict=True))
    for config_id, frame in frames.items():
        units = set(zip(frame["model_id"], frame["repeat"], frame["fold"], strict=True))
        assert units == ref_units, (
            config_id + " ran a different (model, repeat, fold) set from " + reference_id
        )


@pytest.mark.needs_data
def test_t74_7_a7_reproduces_exp_a1() -> None:
    """The control. A drifting A7 makes every other arm's delta meaningless."""
    from src.evaluation import feature_ablation as fa

    path = fa.arm_experiment("A7").output_dir() / "per_fold_metrics.csv"
    if not path.is_file():
        pytest.skip("A7 has not been run")
    report = fa.assert_a7_reproduces_exp_a1()
    assert bool(report["matches"].all()), report.to_dict("records")


@pytest.mark.needs_data
def test_t74_7_t17_and_t18_exist_and_carry_their_arms() -> None:
    import pandas as pd

    section = _root() / ABLATION_DIR
    t17 = section / "T17_feature_family_ablation.csv"
    t18 = section / "T18_all_features_versus_selected_features.csv"
    if not t17.is_file():
        pytest.skip("T17 has not been generated")

    from src.evaluation import feature_ablation as fa

    table = pd.read_csv(t17)
    assert set(table["config_id"]) == set(fa.configuration_ids())
    for config_id, expected in EXPECTED_COUNTS.items():
        rows = table[table["config_id"] == config_id]
        assert set(rows["n_features"]) == {expected}, config_id

    assert t18.is_file(), "T17 exists but T18 does not"
    comparison = pd.read_csv(t18)
    assert {"n_features_all", "n_features_selected"} <= set(comparison.columns)
    assert set(comparison["n_features_all"]) == {138}


# ===========================================================================
# T75.7 -- the optimization ablation
# ===========================================================================


@pytest.mark.needs_data
def test_t75_7_each_stage_carries_an_incremental_delta() -> None:
    from src.evaluation import optimization_ablation as oa

    try:
        stages, _ = oa.build_stages()
    except FileNotFoundError as error:
        pytest.skip(str(error))

    assert len(stages) >= 2, "an ablation needs at least two stages"
    assert list(stages["stage_order"]) == sorted(stages["stage_order"])

    for metric in ("sensitivity", "balanced_accuracy"):
        assert metric + "_incremental" in stages.columns
        assert metric + "_cumulative" in stages.columns
        # The first stage is the baseline: no previous stage, zero cumulative.
        assert stages.iloc[0][metric + "_cumulative"] == 0.0
        rest = stages.iloc[1:][metric + "_incremental"]
        assert rest.notna().all(), "a later stage has no incremental delta"


@pytest.mark.needs_data
def test_t75_7_the_cumulative_delta_is_the_sum_of_the_incremental_ones() -> None:
    """Paired deltas over shared folds telescope. If they do not, they are not paired."""
    from src.evaluation import optimization_ablation as oa

    try:
        stages, _ = oa.build_stages()
    except FileNotFoundError as error:
        pytest.skip(str(error))

    for metric in ("sensitivity", "balanced_accuracy"):
        running = 0.0
        for position, row in enumerate(stages.to_dict("records")):
            if position:
                running += float(row[metric + "_incremental"])
            assert float(row[metric + "_cumulative"]) == pytest.approx(
                running, abs=1e-9
            ), (metric + " at stage " + str(row["stage"]))


@pytest.mark.needs_data
def test_t75_7_a9_and_a10_are_written_as_named_comparisons() -> None:
    section = _root() / ABLATION_DIR
    expected = (
        "A9_individual_vs_ensemble_default.csv",
        "A9_individual_vs_ensemble_tuned.csv",
        "A10_equal_vs_optimized_weights.csv",
    )
    present = [name for name in expected if (section / name).is_file()]
    if not present:
        pytest.skip("EXP-F2 has not been run")
    missing = [name for name in expected if name not in present]
    assert not missing, "missing comparison file(s): " + ", ".join(missing)


@pytest.mark.needs_data
def test_t75_7_t19_g23_and_g24_exist() -> None:
    section = _root() / ABLATION_DIR
    t19 = section / "T19_search_and_optimization_ablation.csv"
    if not t19.is_file():
        pytest.skip("T19 has not been generated")

    # G23/G24 moved to the figures directory in Phase 93, beside every other
    # G-figure and in the one figure registry. The assertions are unchanged.
    from src.reporting.graphs import figures_dir

    for name in (
        "G23_ensemble_weight_comparison.png",
        "G23_ensemble_weight_comparison.csv",
        "G24_baseline_versus_optimized.png",
        "G24_baseline_versus_optimized.csv",
    ):
        path = figures_dir() / name
        assert path.is_file(), name + " is missing"
        assert path.stat().st_size > 0, name + " is empty"


# ===========================================================================
# T76.7 -- the diagnosis track
# ===========================================================================


def test_t76_7_the_merge_policy_is_a_threshold_not_a_hand_picked_list() -> None:
    """T76.2 asks for a documented policy. A list of names is not a policy."""
    from src.evaluation import diagnosis_track as dt

    try:
        policy = dt.merge_report()
    except FileNotFoundError as error:
        pytest.skip(str(error))

    for row in policy.to_dict("records"):
        expected = row["n_records"] >= dt.MIN_CLASS_RECORDS
        assert bool(row["retained"]) is bool(expected), (
            row["diagnosis_class"] + " has " + str(row["n_records"])
            + " records but retained=" + str(row["retained"])
            + " against a threshold of " + str(dt.MIN_CLASS_RECORDS)
        )
        assert row["reason"], row["diagnosis_class"] + " has no recorded reason"
        if not expected:
            assert row["merged_into"] == dt.OTHER_CLASS


@pytest.mark.needs_data
def test_t76_7_the_merge_policy_is_written_to_disk() -> None:
    path = (
        _root() / "outputs" / "07_multiclass_results" / "EXP-G1"
        / "diagnosis_class_merge_policy.csv"
    )
    if not path.is_file():
        pytest.skip("EXP-G1 has not been run")
    assert path.stat().st_size > 0


@pytest.mark.needs_data
def test_t76_7_every_per_class_metric_carries_a_confidence_interval() -> None:
    """T76.6 -- CIs on every per-class metric, not just the headline."""
    import pandas as pd

    path = (
        _root() / "outputs" / "07_multiclass_results" / "EXP-G1"
        / "per_class_metrics.csv"
    )
    if not path.is_file():
        pytest.skip("EXP-G1 has not been run")

    table = pd.read_csv(path)
    for kind in ("recall", "precision", "f1"):
        column = kind + "_fold_ci"
        assert column in table.columns, "no interval column for " + kind
        assert table[column].notna().all(), kind + " has a row with no interval"
        assert (
            table[column].astype(str).str.contains(r"\[").all()
        ), kind + " intervals are not formatted as intervals"


@pytest.mark.needs_data
def test_t76_7_the_track_reports_per_class_recall_for_every_class() -> None:
    import pandas as pd

    from src.evaluation import diagnosis_track as dt

    path = (
        _root() / "outputs" / "07_multiclass_results" / "EXP-G1"
        / "per_class_metrics.csv"
    )
    if not path.is_file():
        pytest.skip("EXP-G1 has not been run")

    table = pd.read_csv(path)
    records = dt.diagnosis_records()
    classes = set(records["diagnosis_merged"].astype(str))
    for model_id, block in table.groupby("model_id"):
        assert set(block["class_name"].astype(str)) == classes, (
            str(model_id) + " does not report every class"
        )


@pytest.mark.needs_data
def test_t76_7_the_source_confound_baseline_is_reported() -> None:
    """No EXP-G1 number may be quoted without the no-audio baseline beside it.

    The diagnosis labels are nearly nested inside PhysioNet's sub-collections,
    and a source-only lookup outscores every model on this track. If that
    baseline ever stops being written, the remaining numbers read as audio-based
    diagnosis classification, which they are not.
    """
    import pandas as pd

    directory = _root() / "outputs" / "07_multiclass_results" / "EXP-G1"
    if not (directory / "aggregate_metrics.csv").is_file():
        pytest.skip("EXP-G1 has not been run")

    baseline_path = directory / "source_confound_baseline.csv"
    assert baseline_path.is_file(), (
        "EXP-G1 was reported without its source-only baseline"
    )
    assert (directory / "class_by_sub_collection.csv").is_file(), (
        "the class-by-sub-collection confound table is missing"
    )

    baseline = pd.read_csv(baseline_path)
    assert len(baseline) >= 2 and baseline["macro_f1"].notna().all()

    # And the headline table has to say so in words, not just ship a CSV.
    table = _root() / "outputs" / "07_multiclass_results" / "T-S1_physionet_diagnosis_multiclass.md"
    if table.is_file():
        text = table.read_text(encoding="utf-8").lower()
        assert "baseline" in text and "no audio" in text, (
            "T-S1 does not state the source-only baseline in its notes"
        )


@pytest.mark.needs_data
def test_t76_7_the_confound_is_measured_not_assumed() -> None:
    """The nesting of class inside sub-collection, recomputed from the master."""
    from src.evaluation import diagnosis_track as dt

    try:
        table = dt.class_by_source_table()
    except FileNotFoundError as error:
        pytest.skip(str(error))

    assert "largest_subset_share" in table.columns
    assert "n_subsets_present" in table.columns
    # This is the finding, pinned: if a later relabelling spreads the classes
    # across sub-collections, this fails and the confound note can be revisited.
    assert float(table["largest_subset_share"].max()) >= 0.99, (
        "no class is confined to a single sub-collection any more -- re-read the "
        "2026-09-10 confound entry in Docs/note.md before trusting it"
    )


@pytest.mark.needs_data
def test_t76_7_the_supplementary_tables_do_not_use_a_locked_table_id() -> None:
    """T13/T14 are CirCor. EXP-G1 is supplementary and gets a `-S` id."""
    section = _root() / "outputs" / "07_multiclass_results"
    if not (section / "EXP-G1").is_dir():
        pytest.skip("EXP-G1 has not been run")

    collisions = [
        path.name
        for path in section.glob("T1[34]_*.csv")
        if "diagnosis" in path.name.lower() or "physionet" in path.name.lower()
    ]
    assert not collisions, (
        "the diagnosis track wrote a locked T13/T14 id: " + ", ".join(collisions)
    )
