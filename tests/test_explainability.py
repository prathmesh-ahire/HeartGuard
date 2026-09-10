"""T81.7 -- the explainability gate.

The gate: **permutation importance is the primary reported measure, family-level
aggregation exists, and the per-sample explanation works for one real record.**

"Works" is not "returns something". A per-sample explanation that does not add
up to the decision it is explaining is wrong, so the strongest test here
reconstructs the model's own `decision_function` from the contributions and
asserts the two agree to 1e-9. That check runs on a real corpus recording, not
on a synthetic vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/04_models/explainability"
FEATURES_SECTION = "outputs/03_features"


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(name: str) -> Any:
    import pandas as pd

    path = _root() / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/33_explainability.py")
    return pd.read_csv(path)


def _demo_record() -> Any:
    """The first supervised abnormal D1 record, by uid -- the script's own rule."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs/01_dataset_audit/metadata_master.csv"
    if not path.is_file():
        pytest.skip("metadata_master.csv unavailable")
    master = pd.read_csv(path, low_memory=False)
    candidates = master[
        (master["dataset_source"] == "D1")
        & (master["binary_label"] == 1)
        & master["use_in_supervised"].astype(bool)
    ].sort_values("record_uid")
    if not len(candidates):
        pytest.skip("no supervised abnormal D1 record")
    return candidates.iloc[0]


# ---------------------------------------------------------------------------
# clause 1 -- permutation is the PRIMARY reported measure
# ---------------------------------------------------------------------------


def test_fe11_reports_permutation_not_impurity() -> None:
    """The impurity ranking is the foil; leading with it would publish the foil."""
    import pandas as pd

    from src.reporting.importance_report import PRIMARY_KIND
    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / FEATURES_SECTION / "FE-11_top_feature_importance.csv"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/33_explainability.py")
    frame = pd.read_csv(path)
    assert len(frame)
    assert set(frame["kind"]) == {PRIMARY_KIND}, set(frame["kind"])
    assert PRIMARY_KIND == "permutation"


def test_permutation_importance_was_measured_on_held_out_rows_across_folds() -> None:
    per_fold = _csv("importance_per_fold.csv")
    permutation = per_fold[per_fold["kind"] == "permutation"]
    assert len(permutation)
    assert set(permutation["computed_on"]) == {"held-out fold"}
    assert (permutation["n_scored_rows"] > 0).all()
    assert (permutation["n_repeats"] > 0).all()
    assert permutation["scoring"].str.contains("balanced_accuracy").all(), (
        "rule 6: accuracy-scored permutation under-weights the minority class"
    )
    # More than one fold, or there is no spread to report.
    assert permutation["fold_label"].nunique() >= 5


def test_the_folds_used_are_a_complete_partition_of_the_corpus() -> None:
    """Every record held out exactly once across the five, so nothing is double-counted."""
    from src.evaluation.cv import load_folds, resolve_folds
    from src.explainability.global_importance import FOLD_REPEAT
    from src.models.smoke import load_task_data

    try:
        data = load_task_data("binary")
    except Exception as error:  # noqa: BLE001 - any missing input is a skip
        pytest.skip("D1 matrix unavailable (" + type(error).__name__ + ")")

    resolved = resolve_folds(load_folds("binary"), data.record_uids)
    folds = [f for f in resolved if f.repeat == FOLD_REPEAT]
    assert len(folds) == 5
    held: list[str] = []
    for fold in folds:
        held.extend(fold.test_uids)
    assert len(held) == len(set(held)) == len(data.record_uids)


def test_impurity_is_reported_beside_it_and_the_two_disagree() -> None:
    """The disagreement is the finding; identical rankings would mean a bug."""
    aggregated = _csv("importance_summary.csv")
    kinds = set(aggregated["kind"])
    assert "permutation" in kinds
    if "impurity" not in kinds:
        pytest.skip("no tree model was included in this run")

    both = aggregated.pivot_table(
        index=["model_id", "feature"], columns="kind", values="rank"
    ).dropna()
    assert len(both)
    assert not np.allclose(both["impurity"], both["permutation"]), (
        "the two measures produced identical rankings, which they should not"
    )
    correlation = float(
        np.corrcoef(both["impurity"].to_numpy(), both["permutation"].to_numpy())[0, 1]
    )
    assert -1.0 <= correlation <= 1.0


def test_a_negative_permutation_importance_is_kept_not_clipped() -> None:
    """Negative means shuffling helped -- a real measurement, not a floor at zero."""
    per_fold = _csv("importance_per_fold.csv")
    permutation = per_fold[per_fold["kind"] == "permutation"]
    assert (permutation["importance"] < 0).any(), (
        "no negative importance anywhere suggests the values were clipped"
    )


def test_the_rank_spread_is_reported_beside_the_rank() -> None:
    """A feature ranked 3rd with a rank SD of 30 is unstable, not third-best."""
    aggregated = _csv("importance_summary.csv")
    for column in ("rank_mean", "rank_sd", "rank_best", "rank_worst", "n_folds_positive"):
        assert column in aggregated.columns, column
    multi = aggregated[aggregated["n_folds"] > 1]
    assert multi["rank_sd"].notna().all()
    assert (multi["rank_best"] <= multi["rank_worst"]).all()


# ---------------------------------------------------------------------------
# clause 2 -- family-level aggregation exists
# ---------------------------------------------------------------------------


def test_family_aggregation_covers_the_six_declared_families() -> None:
    from src.feature_extraction.registry import feature_names, spec_for

    family = _csv("feature_family_importance.csv")
    assert len(family)
    declared = {spec_for(name).family for name in feature_names()}
    assert len(declared) == 6, declared
    for (model_id, kind), block in family.groupby(["model_id", "kind"]):
        assert set(block["family"]) == declared, (model_id, kind, set(block["family"]))
        assert int(block["n_features"].sum()) == 138, (model_id, kind)


def test_family_shares_are_over_the_positive_part_and_sum_to_one() -> None:
    """A signed grand total can sit near zero and turn every share into nonsense."""
    family = _csv("feature_family_importance.csv")
    per_fold = _csv("feature_family_importance_per_fold.csv")
    assert "share_of_positive" in per_fold.columns

    for keys, block in per_fold.groupby(["model_id", "kind", "fold_label"]):
        total = float(block["share_of_positive"].sum())
        assert np.isclose(total, 1.0, atol=1e-6), (keys, total)
    assert (per_fold["share_of_positive"] >= -1e-9).all()
    assert family["rank"].notna().all()


# ---------------------------------------------------------------------------
# clause 3 -- the per-sample explanation works on a REAL record
# ---------------------------------------------------------------------------


def test_the_per_sample_explanation_reconstructs_its_own_prediction() -> None:
    """The strongest clause: contributions + base must equal the decision.

    Run end to end on a real corpus recording -- audio in, prediction and
    explanation out -- not on a prepared vector.
    """
    from src.explainability.per_sample import RECONSTRUCTION_TOLERANCE, explain_prediction
    from src.utils.evidence import PROJECT_ROOT

    record = _demo_record()
    path = PROJECT_ROOT / str(record["file_path"])
    if not path.is_file():
        pytest.skip("the corpus recording is unavailable (dataset/ is gitignored)")

    result, explanation = explain_prediction(path, record_uid=str(record["record_uid"]))

    assert explanation.reconstruction_error <= RECONSTRUCTION_TOLERANCE
    assert np.isclose(
        explanation.base_value + explanation.total_contribution,
        explanation.decision_value,
        atol=RECONSTRUCTION_TOLERANCE,
    )
    # The explanation must describe the prediction that was actually made.
    assert explanation.predicted_class == result.predicted_class
    assert np.isclose(explanation.probability, result.confidence, atol=1e-9)
    assert explanation.n_features == len(explanation.contributions) == 138


def test_the_explanation_payload_is_what_a_page_needs_and_says_what_it_is_not() -> None:
    import json

    path = _root() / "per_sample_explanation.json"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/33_explainability.py")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["n_features"] == 138
    assert payload["n_shown"] == 20
    assert len(payload["top_contributions"]) == 20
    assert payload["record_uid"]
    assert payload["selection_rule"]
    assert "log-odds" in payload["units"]
    assert payload["reconstruction_error"] < 1e-9

    top = payload["top_contributions"]
    magnitudes = [abs(item["contribution"]) for item in top]
    assert magnitudes == sorted(magnitudes, reverse=True), "not ranked by magnitude"
    for item in top:
        assert item["feature"]
        assert item["family"]
        assert item["direction"].startswith("toward ")
        assert np.isfinite(item["contribution"])

    caveats = " ".join(payload["caveats"])
    assert "not in probability" in caveats
    assert "not a clinical explanation" in caveats
    assert "screening" in caveats.lower()


def test_a_wrong_decomposition_is_refused_rather_than_reported() -> None:
    """The reconstruction check must actually fire, not merely be present.

    Built from a deliberately inconsistent estimator: its coefficients say one
    thing and its decision_function says another. A function that reported the
    contributions anyway would be publishing an explanation of a decision the
    model did not make.
    """
    from src.explainability.per_sample import ExplanationError, explain_vector

    names = ("time_mean", "time_std", "time_var")
    with pytest.raises(ExplanationError, match="do not reconstruct"):
        explain_vector(
            _InconsistentPipeline(),
            np.array([1.0, 2.0, 3.0]),
            names,
            class_names=("normal", "abnormal"),
        )


class _Estimator:
    """coef_ . z + intercept_ = 1.0, but decision_function insists on 99.0."""

    coef_ = np.array([[1.0, 0.0, 0.0]])
    intercept_ = np.array([0.0])
    classes_ = np.array([0, 1])

    def decision_function(self, X: Any) -> Any:
        return np.array([99.0])


class _Identity:
    def transform(self, X: Any) -> Any:
        return np.asarray(X, dtype=float)


class _InconsistentPipeline:
    """The smallest object shaped like the project's fitted pipeline."""

    def __init__(self) -> None:
        self.named_steps = {"scaler": _Identity(), "estimator": _Estimator()}

    def predict_proba(self, X: Any) -> Any:
        return np.array([[0.2, 0.8]])


# ---------------------------------------------------------------------------
# the artifacts
# ---------------------------------------------------------------------------


def test_shap_either_ran_or_recorded_why_it_did_not() -> None:
    """T81.3 -- a missing optional dependency leaves a record, not a gap."""
    import json

    path = _root() / "shap_report.json"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/33_explainability.py")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] in {"computed", "skipped", "failed"}
    if report["status"] == "computed":
        assert report["models_computed"]
        assert report["shap_version"]
        assert (_root() / "shap_importance.csv").is_file()
        frame = _csv("shap_importance.csv")
        assert (frame["mean_abs_shap"] >= 0).all()
        assert frame["rank"].notna().all()
    else:
        assert len(str(report["reason"])) > 20, "a skip must state its reason"


def test_the_excluded_models_carry_a_stated_reason() -> None:
    coverage = _csv("explainability_coverage.csv")
    excluded = coverage[coverage["excluded_reason"].notna() & (coverage["excluded_reason"] != "")]
    assert len(excluded), "M6/M7 are excluded and the reason must travel with the output"
    assert excluded["excluded_reason"].str.len().min() > 40


@pytest.mark.parametrize(
    ("figure_id", "slug"),
    [
        ("G18", "G18_feature_importance_plot"),
        ("G19", "G19_top_20_features_chart"),
    ],
)
def test_the_figures_exist_with_the_csv_they_were_drawn_from(figure_id: str, slug: str) -> None:
    import pandas as pd

    from src.reporting.graphs import figures_dir

    directory = figures_dir()
    png = directory / (slug + ".png")
    if not png.is_file():
        pytest.skip(str(png) + " does not exist; run scripts/33_explainability.py")
    frame = pd.read_csv(directory / (slug + ".csv"))
    assert len(frame)
    assert not frame.isna().all().all()
    assert png.stat().st_size > 10_000


def test_fe11_carries_the_measure_caveats_in_its_notes() -> None:
    import json

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / FEATURES_SECTION / "FE-11_top_feature_importance.meta.json"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/33_explainability.py")
    notes = " ".join(json.loads(path.read_text(encoding="utf-8")).get("notes", []))
    assert "PERMUTATION IS THE REPORTED MEASURE" in notes
    assert "CAN BE NEGATIVE" in notes
    assert "T81.4" in notes
    assert "screening" in notes.lower()
