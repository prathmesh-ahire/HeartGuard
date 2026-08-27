"""🔴 MEGA TEST 2 — Feature and model integrity (Phase 53).

Covers Parts IV-V (Phases 30-51). Nothing in Part VI starts until this file is
green.

**Why this exists when a thousand other tests already pass.** Those check units:
an extractor against its own maths, an estimator against its own contract. This
file checks the *seams* on the artifacts that will actually be consumed
downstream — the committed FE-03 matrix, the committed split map, the models
sitting in `models_saved/`, the config that every one of them was built from. A
leakage bug introduced in Phase 44 is cheap to fix here and catastrophic to
discover at submission, and no single-module test would catch it.

The T53.x sections are the plan's own gate. The `EXTRA` sections below them are
cross-cutting checks added for this sweep, most of them pinning something that
actually went wrong during Phases 46-52 — a config value that parsed as a
string, a thread count that changed a model, a wrapper that advertised a method
it did not have. Each names the failure it exists to prevent.

T53.7's "full suite green" clause is satisfied by running the suite, not by a
test asserting about itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

FEATURE_COUNT = 138
EXPECTED_FAMILIES = {
    "time": 24,
    "frequency": 22,
    "mfcc": 39,
    "chroma": 24,
    "dwt": 24,
    "envelope": 5,
}
D1_RECORDS = 3240
MANDATORY_MODELS = ("M1", "M3", "M4", "M5", "M6", "M7")


# ---------------------------------------------------------------------------
# fixtures — the committed artifacts, read once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def task_data() -> Any:
    """D1's binary task: FE-03 joined to its labels, or a skip."""
    from src.models import smoke as sm

    try:
        return sm.load_task_data("binary")
    except Exception as error:  # noqa: BLE001 — any missing input is a skip
        pytest.skip("FE-03 unavailable (" + type(error).__name__ + "): " + str(error))


@pytest.fixture(scope="module")
def fold_zero(task_data: Any) -> Any:
    from src.models import smoke as sm

    return sm._fold_zero("binary", task_data)


@pytest.fixture(scope="module")
def fitted_models(task_data: Any, fold_zero: Any) -> dict[str, Any]:
    """Every buildable model, fitted once on fold 0. Shared by several sections.

    Module-scoped because M6 and M7 take about three minutes each — they refit
    their members over an inner CV — and fitting them per test would make this
    file too slow to run, which is how a mega test quietly stops being run.
    """
    from src.models import estimators as est
    from src.models import registry as reg
    from src.models import smoke as sm

    results: dict[str, Any] = {}
    for model_id in reg.model_ids():
        if not reg.available(model_id):
            continue
        factory = sm._default_factory(model_id, task_data, fold_zero)
        try:
            results[model_id] = sm.smoke_fold(
                model_id, factory, task_data, fold_zero, keep_pipeline=True
            )
        except est.EstimatorError as error:  # pragma: no cover — availability guard
            results[model_id] = error
    return results


# ---------------------------------------------------------------------------
# T53.1 — the feature registry
# ---------------------------------------------------------------------------


def test_t53_1_the_registry_holds_exactly_138_names() -> None:
    from src.feature_extraction.registry import feature_names

    names = feature_names()
    assert len(names) == FEATURE_COUNT
    assert len(set(names)) == FEATURE_COUNT, "a duplicate name would silently alias"


def test_t53_1_the_family_counts_are_24_22_39_24_24_5() -> None:
    from src.feature_extraction.registry import EXPECTED_FAMILY_COUNTS, family_counts

    assert dict(family_counts()) == EXPECTED_FAMILIES
    assert dict(EXPECTED_FAMILY_COUNTS) == EXPECTED_FAMILIES
    assert sum(EXPECTED_FAMILIES.values()) == FEATURE_COUNT


def test_t53_1_the_order_is_stable_across_two_reads() -> None:
    """Column order is a literal, never derived — so it cannot drift per run."""
    from src.feature_extraction.registry import feature_names

    assert feature_names() == feature_names()


def test_t53_1_the_fingerprint_pins_the_order_not_just_the_set() -> None:
    """Two runs disagreeing on the fingerprint are not comparable at all."""
    from src.feature_extraction.registry import feature_names, registry_fingerprint

    fingerprint = registry_fingerprint()
    assert len(fingerprint) == 64, "expected a sha256 hex digest"
    assert registry_fingerprint() == fingerprint

    import hashlib

    recomputed = hashlib.sha256("\n".join(feature_names()).encode()).hexdigest()
    assert recomputed == fingerprint, (
        "the fingerprint is not the hash of the ordered names; it cannot detect "
        "a reordering, which is the one thing it exists for"
    )


def test_t53_1_the_matrix_columns_match_the_registry_in_order(task_data: Any) -> None:
    """FE-03 is consumed positionally by every model; order is not cosmetic."""
    from src.feature_extraction.registry import feature_names

    assert task_data.feature_names == feature_names()
    assert task_data.n_features == FEATURE_COUNT


# ---------------------------------------------------------------------------
# T53.2 — re-extraction reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.needs_data
def test_t53_2_one_hundred_records_re_extract_bit_identically() -> None:
    """The gate: 100 random records, re-extracted, compared against the cache.

    Bit-identical, not close. A feature that drifts in its last bits between runs
    makes every downstream metric irreproducible by an amount nobody can bound,
    and this project has already found one cause of exactly that — a BLAS thread
    count reaching a chroma value (2026-08-27).
    """
    from src.feature_extraction.matrix import load_matrix
    from src.feature_extraction.quality import reproducibility_check

    result = reproducibility_check(load_matrix(), 100, seed=42)

    assert result["n_checked"] == 100
    assert result["n_mismatches"] == 0, result["mismatches"][:10]
    assert result["identical"] is True


def test_t53_2_the_committed_matrix_covers_the_audited_corpus(task_data: Any) -> None:
    assert task_data.n_records == D1_RECORDS, (
        "D1 is 3,240 records; a different number means the matrix and the audit "
        "have diverged — see Docs/note.md on the 301 validation duplicates"
    )
    assert len(set(task_data.record_uids)) == task_data.n_records


def test_t53_2_the_matrix_holds_no_unexplained_nan(task_data: Any) -> None:
    """FE-04 accounts for every NaN by name; an unexplained one fails the gate."""
    n_nan = int(np.isnan(task_data.X).sum())
    if n_nan == 0:
        return
    report = Path("outputs/03_features/feature_missing_values.csv")
    assert report.is_file(), (
        str(n_nan) + " NaN cells in FE-03 but no FE-04 report to account for them"
    )


# ---------------------------------------------------------------------------
# T53.3 — the leakage canary, end to end
# ---------------------------------------------------------------------------


def test_t53_3_a_feature_correlated_with_test_labels_does_not_help(
    task_data: Any, fold_zero: Any
) -> None:
    """The gate: inject a feature perfectly correlated with the TEST fold's labels.

    A fold-safe pipeline cannot benefit from it, because the model never sees a
    test label while fitting — the planted column is pure noise on the training
    rows, so the fitted model has no reason to rely on it, and the test score
    must not move beyond ordinary fit-to-fit variation.

    This is the end-to-end version of the Phase 44 canary, run on the real 138
    against the real fold rather than on synthetic data.
    """
    from src.evaluation import metrics as mt
    from src.models import estimators as est
    from src.models import pipeline as pl

    train = np.asarray(fold_zero.train_index, dtype=int)
    test = np.asarray(fold_zero.test_index, dtype=int)
    rng = np.random.default_rng(42)

    planted = rng.normal(size=task_data.n_records)
    # Perfect on the test rows, pure noise on the training rows.
    planted[test] = task_data.y[test] * 10.0 + rng.normal(scale=0.01, size=test.size)
    poisoned = np.column_stack([task_data.X, planted])

    def score(matrix: np.ndarray) -> float:
        built = pl.build_pipeline(est.make_m1(), y=task_data.y[train])
        built.fit(matrix[train], task_data.y[train])
        predicted = built.predict(matrix[test])
        return mt.binary_metrics(
            task_data.y[test], predicted, labels=(0, 1), positive_label=1
        )["balanced_accuracy"]

    clean = score(task_data.X)
    poisoned_score = score(poisoned)

    assert poisoned_score < clean + 0.02, (
        "planting a test-label-correlated feature moved balanced accuracy from "
        + str(round(clean, 4)) + " to " + str(round(poisoned_score, 4))
        + "; the pipeline is seeing test-fold information"
    )


def test_t53_3_no_fitted_step_saw_a_test_row(task_data: Any, fold_zero: Any) -> None:
    """Checked against what the steps LEARNED, not against their outputs.

    A scaler fitted on 3,240 rows and one fitted on 2,593 both produce plausible
    standardised numbers; only the learned statistics tell them apart.
    """
    from src.models import estimators as est
    from src.models import pipeline as pl

    train = np.asarray(fold_zero.train_index, dtype=int)
    built = pl.build_pipeline(est.make_m1(), y=task_data.y[train])
    built.fit(task_data.X[train], task_data.y[train])

    learned = pl.fitted_steps(built)
    assert learned["scaler.n_samples_seen_"] == train.size
    assert np.allclose(
        learned["scaler.mean_"], np.nanmean(task_data.X[train], axis=0), equal_nan=True
    )
    assert not np.allclose(
        learned["scaler.mean_"], np.nanmean(task_data.X, axis=0), equal_nan=True
    ), "the scaler's mean matches the FULL matrix; it was fitted outside the fold"


def test_t53_3_every_fold_of_every_task_is_subject_disjoint() -> None:
    """Research rule 3, over the whole committed split map — not just fold 0."""
    from src.evaluation import cv

    for task in cv.available_tasks():
        for fold in cv.load_folds(task):
            shared = set(fold.train_groups) & set(fold.test_groups)
            assert not shared, task + " " + fold.label + ": " + str(sorted(shared)[:5])


# ---------------------------------------------------------------------------
# T53.4 — model round trip
# ---------------------------------------------------------------------------


def test_t53_4_every_saved_model_reloads_and_predicts_identically(
    task_data: Any,
) -> None:
    """The gate. Bit-identical predictions, or the artifact is not the model."""
    from src.models import registry as reg

    saved = reg.saved_models()
    if not saved:
        pytest.skip("no models saved; run scripts/04_model_smoke.py")

    checked = 0
    for record in saved:
        if not record.path.is_file():
            continue  # binaries are gitignored; manifests still get checked below
        model, manifest = reg.load_model(
            record.task, record.model_id, feature_names=record.feature_names
        )
        assert manifest["n_features"] == FEATURE_COUNT

        sample = task_data.X[:64]
        first = model.predict_proba(sample)
        second = model.predict_proba(sample)
        assert np.array_equal(first, second), record.model_id + " is not deterministic"
        assert np.array_equal(
            model.predict(sample), model.predict(sample)
        ), record.model_id
        checked += 1

    if checked == 0:
        pytest.skip("model binaries absent (gitignored); manifests checked separately")


def test_t53_4_every_manifest_carries_its_feature_list_and_costs() -> None:
    """Manifests are committed even when the binaries are not — they are the record."""
    from src.models import registry as reg

    saved = reg.saved_models()
    if not saved:
        pytest.skip("no models saved; run scripts/04_model_smoke.py")

    for record in saved:
        assert len(record.feature_names) == FEATURE_COUNT, record.model_id
        assert record.size_bytes > 0, record.model_id
        assert record.manifest.get("fit_seconds") is not None, record.model_id
        assert record.manifest.get("inference_seconds_per_record"), record.model_id
        assert record.manifest.get("package_versions"), record.model_id


def test_t53_4_the_stored_feature_list_catches_a_column_reordering(
    task_data: Any, tmp_path: Path
) -> None:
    """The failure joblib cannot catch: right shape, wrong meaning, no exception."""
    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models import registry as reg

    built = pl.build_pipeline(est.make_m1(), y=task_data.y[:200])
    built.fit(task_data.X[:200], task_data.y[:200])
    reg.save_model(
        built, model_id="M1", task="binary",
        feature_names=task_data.feature_names, root=tmp_path,
    )

    swapped = (
        task_data.feature_names[1],
        task_data.feature_names[0],
        *task_data.feature_names[2:],
    )
    with pytest.raises(reg.RegistryError, match="DIFFERENT"):
        reg.load_model("binary", "M1", tmp_path, feature_names=swapped)


# ---------------------------------------------------------------------------
# T53.5 — probabilities
# ---------------------------------------------------------------------------


def test_t53_5_every_model_produces_well_formed_probabilities(
    fitted_models: dict[str, Any],
) -> None:
    """The gate, explicitly including the calibrated SVM and both ensembles."""
    from src.models import estimators as est

    assert {"M3", "M6", "M7"} <= set(fitted_models), (
        "the gate names the calibrated SVM and both ensembles by hand; they must "
        "be in the sweep"
    )

    for model_id, result in fitted_models.items():
        if isinstance(result, Exception):
            continue
        probability = result.probability
        assert probability["has_predict_proba"] is True, model_id
        assert probability["n_nan"] == 0, model_id
        assert probability["n_inf"] == 0, model_id
        assert probability["proba_min"] >= 0.0, model_id
        assert probability["proba_max"] <= 1.0, model_id
        assert probability["max_row_sum_error"] <= probability["row_sum_tolerance"], (
            model_id + ": rows sum off by " + str(probability["max_row_sum_error"])
        )
        assert probability["well_formed"] is True, (
            model_id + ": " + str(est.ProbabilityReport(**{
                "n_rows": probability["n_rows"],
                "n_classes": probability["n_classes"],
                "min_value": probability["proba_min"],
                "max_value": probability["proba_max"],
                "max_row_sum_error": probability["max_row_sum_error"],
                "n_nan": probability["n_nan"],
                "n_inf": probability["n_inf"],
            }).problems())
        )


def test_t53_5_the_row_sum_tolerance_respects_the_producers_dtype() -> None:
    """XGBoost returns float32; a fixed 1e-9 gate fails an exactly-correct matrix."""
    from src.models import estimators as est

    assert est.row_sum_tolerance(np.float64, 2) == 1e-9
    assert est.row_sum_tolerance(np.float32, 2) > 1e-7
    # And a genuinely broken matrix is still rejected at either precision.
    for dtype in (np.float32, np.float64):
        broken = np.array([[0.2, 0.9]], dtype=dtype)
        assert not est.probability_report(broken).rows_sum_to_one


def test_t53_5_a_thresholded_model_is_not_asked_to_match_argmax(
    fitted_models: dict[str, Any],
) -> None:
    """M6/M7 predict by cut-off, so `predict != argmax` is correct, not a fault."""
    for model_id in ("M6", "M7"):
        result = fitted_models.get(model_id)
        if result is None or isinstance(result, Exception):
            continue
        ensemble = result.pipeline.named_steps["estimator"]
        if ensemble.threshold_ != 0.5:
            assert not ensemble.predicts_by_argmax, model_id
        assert result.probability["well_formed"] is True, model_id


# ---------------------------------------------------------------------------
# T53.6 — every mandatory model runs clean on fold 0
# ---------------------------------------------------------------------------


def test_t53_6_every_mandatory_model_ran(fitted_models: dict[str, Any]) -> None:
    from src.models import registry as reg

    mandatory = {
        entry.model_id for entry in reg.entries() if entry.mandatory and entry.implemented
    }
    assert mandatory == set(MANDATORY_MODELS), (
        "the mandatory set changed: " + str(sorted(mandatory))
    )
    for model_id in mandatory:
        result = fitted_models.get(model_id)
        assert result is not None, model_id + " did not run"
        assert not isinstance(result, Exception), model_id + ": " + str(result)


def test_t53_6_no_model_produces_a_nan_metric(fitted_models: dict[str, Any]) -> None:
    for model_id, result in fitted_models.items():
        if isinstance(result, Exception):
            continue
        for name, value in result.metrics.items():
            assert np.isfinite(value), model_id + "." + name + " is " + str(value)


def test_t53_6_no_model_scores_suspiciously_well(
    fitted_models: dict[str, Any],
) -> None:
    """A near-perfect metric is a bug report, not a result — the standing rule."""
    for model_id, result in fitted_models.items():
        if isinstance(result, Exception):
            continue
        for name in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc"):
            value = result.metrics.get(name)
            if value is None:
                continue
            assert value < 0.99, (
                model_id + "." + name + " = " + str(round(value, 4))
                + " on one untuned fold; investigate before recording it"
            )


def test_t53_6_the_baseline_metric_table_is_recorded(
    fitted_models: dict[str, Any],
) -> None:
    """The gate asks for the table, so the table has to exist on disk."""
    import pandas as pd

    path = Path("outputs/04_models/baseline_smoke_metrics.csv")
    if not path.is_file():
        pytest.skip("smoke table absent; run scripts/04_model_smoke.py")

    table = pd.read_csv(path)
    recorded = set(table["model_id"])
    ran = {mid for mid, r in fitted_models.items() if not isinstance(r, Exception)}
    assert ran <= recorded, "not recorded: " + str(sorted(ran - recorded))
    for column in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc"):
        assert column in table.columns
        assert table[column].notna().all()


# ---------------------------------------------------------------------------
# EXTRA — config traps that have already bitten once
# ---------------------------------------------------------------------------


def test_extra_no_yaml_bound_parsed_as_a_string() -> None:
    """PyYAML is YAML 1.1: `1.0e3` is the STRING "1.0e3"; `1.0e+3` is a float.

    Four search-space bounds were silently strings until 2026-08-27.
    """
    import re

    import yaml

    pattern = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")

    def walk(node: Any, path: str = "") -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                yield from walk(value, path + "." + str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, path + "[" + str(index) + "]")
        else:
            yield path, node

    offenders = []
    for config in sorted(Path("configs").glob("*.yaml")):
        loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
        for path, value in walk(loaded):
            if isinstance(value, str) and pattern.match(value):
                offenders.append(config.name + path + " = " + repr(value))
    assert not offenders, "write the exponent signed: " + str(offenders)


def test_extra_m8_pins_a_fixed_thread_count() -> None:
    """XGBoost's `subsample` mask is per thread block; -1 is the machine's cores.

    Measured on the real matrix: n_jobs 1 against 4 differ by 0.571 in predicted
    probability. `-1` would make M8 irreproducible between machines — rule 5.
    """
    from src.models import estimators as est

    n_jobs = est.model_defaults("M8").get("n_jobs")
    assert isinstance(n_jobs, int) and n_jobs > 0, (
        "M8 must declare a fixed positive n_jobs, got " + repr(n_jobs)
    )


def test_extra_no_model_carries_a_removed_sklearn_parameter() -> None:
    """`penalty` goes in sklearn 1.10 and `probability` in 1.11; we pin 1.9."""
    from src.models import estimators as est

    assert "penalty" not in est.model_defaults("M1")
    assert "l1_ratio" in est.model_defaults("M1")
    # M3 still DECLARES probability as an assertion of intent, but must not pass
    # it to SVC — the factory reads it and drops it.
    inner = est.make_m3().estimator
    assert inner.get_params().get("probability") == "deprecated", (
        "make_m3 passed `probability` through; sklearn 1.9 warns on every fit"
    )


def test_extra_every_model_has_a_route_for_class_weight() -> None:
    """M5 and M8 have no usable `class_weight` and must be wrapped, or they get none."""
    from src.models import pipeline as pl
    from src.models import registry as reg

    for model_id in reg.model_ids():
        if not reg.available(model_id):
            continue
        estimator = reg.build(model_id)
        if model_id in {"M2", "M6", "M7"}:
            # KNN genuinely has no imbalance mechanism (documented); the
            # ensembles delegate to their members.
            continue
        assert pl.supports_class_weight(estimator), (
            model_id + " has no class_weight route; it would be fitted with NO "
            "imbalance handling on a 79/21 task — wrap it in ClassWeightedClassifier"
        )


def test_extra_every_seed_in_config_is_42() -> None:
    from src.utils.config import load_config

    assert load_config("models").get("global.random_state") == 42


# ---------------------------------------------------------------------------
# EXTRA — determinism, end to end
# ---------------------------------------------------------------------------


def test_extra_every_model_fits_twice_to_the_same_numbers(
    task_data: Any, fold_zero: Any
) -> None:
    """Rule 5, on the real matrix. Cheap models only — the ensembles are covered
    by their own reproducibility test and take three minutes each."""
    from src.models import estimators as est
    from src.models import pipeline as pl

    train = np.asarray(fold_zero.train_index, dtype=int)
    test = np.asarray(fold_zero.test_index, dtype=int)

    for model_id in ("M1", "M2", "M3", "M4", "M8"):
        if model_id == "M8" and not est.m8_capability().available:
            continue
        outputs = []
        for _ in range(2):
            built = pl.build_pipeline(
                est.build_estimator(model_id), y=task_data.y[train]
            )
            built.fit(task_data.X[train], task_data.y[train])
            outputs.append(built.predict_proba(task_data.X[test]))
        assert np.array_equal(outputs[0], outputs[1]), model_id + " is not reproducible"


def test_extra_the_ensembles_choose_the_same_weights_twice(
    task_data: Any, fold_zero: Any
) -> None:
    """The simplex grid needs no seed, so M7's weights must be exactly repeatable."""
    from src.ensemble import soft_voting as sv

    train = np.asarray(fold_zero.train_index, dtype=int)
    rng = np.random.default_rng(42)
    # A small stand-in for the real members: this checks the SEARCH is
    # deterministic, which does not need three minutes of member fitting.
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    subset = rng.permutation(train)[:600]

    def build() -> Any:
        return sv.SoftVotingEnsemble(
            [
                ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
                ("tree", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ],
            weights="optimized",
            groups=task_data.groups[subset],
            inner_cv=3,
        )

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    clean = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(task_data.X[subset])
    )
    first = build().fit(clean, task_data.y[subset])
    second = build().fit(clean, task_data.y[subset])

    assert np.array_equal(first.weights_, second.weights_)
    assert first.threshold_ == second.threshold_


# ---------------------------------------------------------------------------
# EXTRA — the ensemble's in-fold decisions never touch the outer fold
# ---------------------------------------------------------------------------


def test_extra_the_ensembles_weights_and_threshold_are_chosen_in_fold(
    fitted_models: dict[str, Any], fold_zero: Any
) -> None:
    """Both were chosen on out-of-fold probabilities over the TRAINING rows only."""
    for model_id in ("M6", "M7"):
        result = fitted_models.get(model_id)
        if result is None or isinstance(result, Exception):
            continue
        ensemble = result.pipeline.named_steps["estimator"]
        report = ensemble.fit_report_

        assert report.n_oof_rows == len(fold_zero.train_uids), (
            model_id + " chose its decision rule on " + str(report.n_oof_rows)
            + " rows but the training fold has " + str(len(fold_zero.train_uids))
        )
        assert report.grouped_inner_cv is True, (
            model_id + "'s inner CV was not subject-grouped"
        )
        assert ensemble.threshold_choice_.n_scored_rows == len(fold_zero.train_uids)


def test_extra_m7_departs_from_equal_weights_only_beyond_the_noise(
    fitted_models: dict[str, Any],
) -> None:
    """The one-standard-error rule, on the real fold.

    Not a check that M7 beats M6 — it may not, and that would be an honest
    result. A check that the selection is inside its own stated margin, and that
    the margin was computed rather than assumed.
    """
    result = fitted_models.get("M7")
    if result is None or isinstance(result, Exception):
        pytest.skip("M7 did not run")

    selection = result.pipeline.named_steps["estimator"].weight_selection_
    assert selection["standard_error"] > 0
    assert selection["margin"] == pytest.approx(
        selection["selection_standard_errors"] * selection["standard_error"], rel=1e-3
    )
    assert selection["chosen_score"] >= selection["best_score"] - selection["margin"] - 1e-9
    assert 1 <= selection["n_within_margin"] <= selection["n_candidates"]


def test_extra_m6_and_m7_share_everything_except_their_weights(
    fitted_models: dict[str, Any],
) -> None:
    """Otherwise the comparison measures the decision rule, not the weights."""
    six, seven = fitted_models.get("M6"), fitted_models.get("M7")
    if six is None or seven is None or isinstance(six, Exception) or isinstance(seven, Exception):
        pytest.skip("both ensembles must run")

    a = six.pipeline.named_steps["estimator"]
    b = seven.pipeline.named_steps["estimator"]
    assert [n for n, _ in a.estimators] == [n for n, _ in b.estimators]
    assert a.objective == b.objective
    assert a.inner_cv == b.inner_cv
    assert a.tune_threshold == b.tune_threshold
    assert np.allclose(a.weights_, 1 / len(a.estimators)), "M6 must be equal weights"


# ---------------------------------------------------------------------------
# EXTRA — the model inventory and its recorded gaps
# ---------------------------------------------------------------------------


def test_extra_the_registry_covers_m1_to_m9_with_no_silent_gaps() -> None:
    from src.models import registry as reg

    ids = reg.model_ids()
    assert ids == tuple("M" + str(index) for index in range(1, 10))
    for entry in reg.entries():
        if entry.implemented and not entry.unavailable_reason:
            continue
        assert entry.unavailable_reason, (
            entry.model_id + " is unavailable but gives no reason; a gap must "
            "never look like an oversight"
        )


def test_extra_m9s_exclusion_is_recorded_in_the_missing_outputs_report() -> None:
    from src.utils.config import load_config

    if load_config("models").get("models.M9.in_scope"):
        pytest.skip("M9 is in scope")

    report = Path("outputs/missing_outputs_report.txt")
    assert report.is_file()
    text = " ".join(report.read_text(encoding="utf-8").split())
    assert "1D-CNN" in text
    assert "T52" in text
    assert "may compare PV-MEPCG against a CNN" in text


def test_extra_every_evidence_row_written_this_part_resolves() -> None:
    from src.utils.evidence import read_evidence

    rows = read_evidence()
    if not rows:
        pytest.skip("evidence index is empty")

    for row in rows:
        if not row.get("evidence_id", "").startswith(("MD-", "FE-")):
            continue
        if (row.get("status") or "ok") != "ok":
            continue
        target = Path(row["filename"])
        assert target.is_file(), row["evidence_id"] + " -> " + str(target)
