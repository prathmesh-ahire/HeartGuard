"""T82.7 -- the statistical validation gate.

The gate: **fold-wise mean and SD, bootstrap CIs and the paired tests all ran;
the normality check is recorded; and EVERY reported p-value states its n.**

The last clause is the one this project cares about most. Wilcoxon on five folds
cannot produce a p-value below 0.0625 however large the effect, so a p-value
printed without its sample size is not a result -- it is a number that looks like
one. Two tests here enforce it: one asserts the column exists and is populated
everywhere, and one asserts the underpowered rows are actually flagged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/12_statistics"

#: Every emitted frame that carries a p-value must carry these beside it.
P_VALUE_COMPANIONS = ("n", "n_description")


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(name: str) -> Any:
    import pandas as pd

    path = _root() / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/34_statistical_validation.py")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# the module's own arithmetic
# ---------------------------------------------------------------------------


def test_the_minimum_achievable_p_is_the_real_floor() -> None:
    """Checked against the definition, not against a remembered table.

    A two-sided sign/signed-rank test's smallest attainable p-value is
    2 * 0.5**n: every pair falling the same way. n=5 gives 0.0625, which is why
    a plain 5-fold comparison can never reject at 0.05.
    """
    from src.evaluation.statistics import ALPHA, minimum_achievable_p

    assert minimum_achievable_p("wilcoxon", 5) == pytest.approx(0.0625)
    assert minimum_achievable_p("wilcoxon", 5) > ALPHA
    assert minimum_achievable_p("wilcoxon", 6) == pytest.approx(0.03125)
    assert minimum_achievable_p("wilcoxon", 6) < ALPHA
    assert minimum_achievable_p("wilcoxon", 25) < 1e-6


def test_the_nemenyi_critical_difference_matches_its_definition() -> None:
    """CD = q_alpha * sqrt(k(k+1)/6n), q from the studentized range over sqrt(2)."""
    from scipy import stats

    from src.evaluation.statistics import nemenyi_critical_difference

    k, n = 7, 25
    expected = (
        float(stats.studentized_range.ppf(0.95, k, np.inf))
        / np.sqrt(2.0)
        * np.sqrt(k * (k + 1) / (6.0 * n))
    )
    assert nemenyi_critical_difference(k, n) == pytest.approx(expected)


def test_the_record_level_partition_holds_every_record_once() -> None:
    """A repeated map counts each record five times; the tests must not.

    Checked on the run itself rather than asserted: `partition_predictions`
    raises if a duplicate survives, so this proves the guard fires on real data.
    """
    from src.evaluation.statistics import RUNS, partition_predictions

    spec = next(s for s in RUNS if s.run == "EXP-A2")
    frame = partition_predictions(spec)
    if frame is None:
        pytest.skip("EXP-A2 predictions.parquet unavailable")
    for _model, block in frame.groupby("model_id"):
        assert not block["record_uid"].duplicated().any()
    assert frame["repeat"].nunique() == 1


# ---------------------------------------------------------------------------
# clause 1 -- fold-wise mean and SD
# ---------------------------------------------------------------------------


def test_foldwise_mean_and_sd_exist_for_every_model_and_headline_metric() -> None:
    from src.evaluation.statistics import HEADLINE_METRICS

    summary = _csv("foldwise_mean_sd.csv")
    assert len(summary)
    assert set(summary["metric"]) <= set(HEADLINE_METRICS)
    assert summary["mean"].notna().all()
    assert summary["n_folds"].min() >= 2

    binary = summary[summary["run"] == "EXP-A2"]
    assert (binary["n_folds"] == 25).all(), "the headline binary run is the 5x5 map"
    for metric in ("sensitivity", "balanced_accuracy", "specificity", "roc_auc"):
        assert metric in set(binary["metric"]), metric
    assert binary["sd"].notna().all()
    assert (binary["ci_low"] <= binary["mean"]).all()
    assert (binary["ci_high"] >= binary["mean"]).all()


# ---------------------------------------------------------------------------
# clause 2 -- bootstrap CIs
# ---------------------------------------------------------------------------


def test_bootstrap_intervals_exist_and_bracket_their_point_estimate() -> None:
    frame = _csv("bootstrap_auc_ci.csv")
    computed = frame[frame["status"] == "computed"]
    assert len(computed), "no bootstrap interval was computed"
    assert (computed["ci_low"] <= computed["point"]).all()
    assert (computed["point"] <= computed["ci_high"]).all()
    assert (computed["ci_width"] > 0).all()
    assert (computed["n_valid_resamples"] > computed["n_resamples"] * 0.9).all()
    assert (computed["seed"] == 42).all()
    # The multiclass runs are recorded as skipped with a reason, not dropped.
    skipped = frame[frame["status"] == "skipped"]
    if len(skipped):
        assert skipped["reason"].str.len().min() > 20


def test_the_bootstrap_resamples_records_not_the_repeated_pool() -> None:
    """n must be the corpus size, not five times it."""
    frame = _csv("bootstrap_auc_ci.csv")
    binary = frame[(frame["run"] == "EXP-A2") & (frame["status"] == "computed")]
    if not len(binary):
        pytest.skip("EXP-A2 has no bootstrap row")
    assert (binary["n"] == 3240).all(), binary[["model_id", "n"]].to_string()
    assert binary["n_description"].str.contains("exactly once").all()


# ---------------------------------------------------------------------------
# clause 3 -- the paired tests, and the normality check
# ---------------------------------------------------------------------------


def test_the_paired_tests_ran_on_every_model_pair() -> None:
    import pandas as pd

    paired = _csv("paired_fold_tests.csv")
    metrics = _csv("per_fold_metrics_all_runs.csv")
    for run in ("EXP-A1", "EXP-A2"):
        models = sorted(set(metrics[metrics["run"] == run]["model_id"]))
        expected = len(models) * (len(models) - 1) // 2
        block = paired[(paired["run"] == run) & (paired["metric"] == "sensitivity")]
        assert len(block) == expected, (run, len(block), expected)
    assert pd.notna(paired["p_value"]).all()


def test_both_tests_are_computed_and_the_normality_check_is_recorded() -> None:
    """T82.4: the choice must be auditable, so the rejected test is kept too."""
    paired = _csv("paired_fold_tests.csv")
    for column in (
        "shapiro_statistic",
        "shapiro_p_value",
        "differences_are_normal",
        "normality_check",
        "wilcoxon_p_value",
        "t_p_value",
        "chosen_test",
    ):
        assert column in paired.columns, column

    assert paired["normality_check"].notna().all()
    assert paired["normality_check"].str.contains("Shapiro-Wilk").all()
    assert set(paired["chosen_test"]) <= {"paired t-test", "Wilcoxon signed-rank"}
    # The chosen test's p-value must be the one reported.
    normal = paired[paired["differences_are_normal"].astype(bool)]
    other = paired[~paired["differences_are_normal"].astype(bool)]
    assert np.allclose(normal["p_value"], normal["t_p_value"], equal_nan=True)
    assert np.allclose(other["p_value"], other["wilcoxon_p_value"], equal_nan=True)
    # Both branches must actually occur, or the check is untested on this data.
    assert len(normal) and len(other)


def test_the_friedman_omnibus_ran_and_gates_its_own_post_hoc() -> None:
    omnibus = _csv("friedman_omnibus.csv")
    assert len(omnibus)
    assert (omnibus["k_models"] >= 3).all()
    assert omnibus["p_value"].notna().all()
    assert (omnibus["posthoc_run"].astype(bool) == omnibus["significant"].astype(bool)).all()

    import pandas as pd

    path = _root() / "nemenyi_posthoc.csv"
    if path.is_file():
        posthoc = pd.read_csv(path)
        rejected = omnibus[omnibus["significant"].astype(bool)]
        assert set(zip(posthoc["run"], posthoc["metric"], strict=True)) <= set(
            zip(rejected["run"], rejected["metric"], strict=True)
        ), "a post-hoc was computed where the omnibus did not reject"


def test_mcnemar_reports_the_discordant_count_as_its_n() -> None:
    frame = _csv("mcnemar_paired_predictions.csv")
    assert len(frame)
    assert (frame["n"] == frame["n_correct_a_only"] + frame["n_correct_b_only"]).all()
    assert frame["n_description"].str.contains("DISCORDANT").all()
    totals = (
        frame["n_both_correct"]
        + frame["n_both_wrong"]
        + frame["n_correct_a_only"]
        + frame["n_correct_b_only"]
    )
    assert (totals == frame["n_records_compared"]).all()


def test_a_pair_that_never_disagreed_reports_no_p_value_at_all() -> None:
    """n=0 discordant means no test exists, and 1.0 would claim one was run.

    M6 and M7 are the same fitted ensemble on the CirCor runs, so they agree on
    every record. The first version emitted p=1.0 there, which reads as "tested,
    found no difference" -- the exact confusion this module exists to prevent.
    """
    frame = _csv("mcnemar_paired_predictions.csv")
    identical = frame[frame["n"] == 0]
    if not len(identical):
        pytest.skip("no model pair agreed on every record in this run set")
    assert identical["p_value"].isna().all()
    assert identical["statistic"].isna().all()
    assert identical["note"].str.contains("not tested").all()


# ---------------------------------------------------------------------------
# clause 4 -- EVERY p-value states its n
# ---------------------------------------------------------------------------


def test_every_frame_with_a_p_value_carries_its_n_and_what_n_means() -> None:
    import pandas as pd

    root = _root()
    checked = 0
    for path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(path)
        if "p_value" not in frame.columns:
            continue
        for column in P_VALUE_COMPANIONS:
            assert column in frame.columns, path.name + " has a p-value but no " + column
        reported = frame[frame["p_value"].notna()]
        assert reported["n"].notna().all(), path.name
        assert (reported["n"] > 0).all(), path.name
        assert reported["n_description"].notna().all(), path.name
        assert (reported["n_description"].str.len() > 10).all(), path.name
        checked += 1
    assert checked >= 3, "expected at least three p-value-bearing frames"


def test_underpowered_comparisons_are_flagged_and_the_flag_is_right() -> None:
    """A 5-fold run must be marked, and a 25-fold run must not be."""
    from src.evaluation.statistics import ALPHA

    paired = _csv("paired_fold_tests.csv")
    assert "underpowered" in paired.columns
    assert (
        paired["underpowered"].astype(bool)
        == (paired["minimum_achievable_p"] >= ALPHA)
    ).all()

    five = paired[paired["n"] == 5]
    if len(five):
        assert five["underpowered"].all(), "n=5 cannot reach 0.05 and must be flagged"
        assert np.allclose(five["minimum_achievable_p"], 0.0625)
    assert not paired[paired["n"] == 25]["underpowered"].any()


def test_an_effect_size_accompanies_every_p_value() -> None:
    """T82.6. A p-value with no effect size beside it is half a result."""
    paired = _csv("paired_fold_tests.csv")
    reported = paired[paired["p_value"].notna()]
    assert reported["effect_size"].notna().all()
    assert reported["effect_size_name"].notna().all()
    assert set(reported["effect_size_name"]) <= {
        "Cohen d (paired)",
        "rank-biserial correlation",
    }

    omnibus = _csv("friedman_omnibus.csv")
    assert omnibus["effect_size"].notna().all()
    assert omnibus["effect_size_name"].str.contains("Kendall").all()
    assert ((omnibus["effect_size"] >= 0) & (omnibus["effect_size"] <= 1)).all()


def test_the_design_record_states_why_each_test_uses_the_sample_it_does() -> None:
    import json

    path = _root() / "statistical_design.json"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/34_statistical_validation.py")
    design = json.loads(path.read_text(encoding="utf-8"))
    assert design["alpha"] == 0.05
    assert design["bootstrap_seed"] == 42
    assert "Shapiro-Wilk" in design["normality_check"]
    assert "five times" in design["why_one_repeat_for_record_level_tests"]
    assert set(design["n_description"]) == {"fold", "record"}
