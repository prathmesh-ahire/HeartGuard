"""🔴 MEGA TEST 3 — Experiment and result integrity (Phase 77).

Covers Parts VI-VII (Phases 53-76). Nothing in Part VIII starts until this file
is green.

**What this file is for.** MEGA TEST 2 checked the features and the models.
This one checks the *results* — the committed experiment outputs that every
table, figure and claim in the thesis will be built from. By this point the
project has fourteen result directories written over three weeks by several
different runners, and the failure modes that matter are no longer "does this
function work" but "do these artifacts still say the same thing":

* a run that cannot be reproduced from its own manifest (T77.1),
* a search that scored a trial on a fold it was later evaluated against (T77.2),
* a metric that is too good and nobody looked at it (T77.3),
* two label spaces that quietly merged (T77.4),
* a transfer result written up as a generalization failure (T77.5).

T77.6 is [TEST/MANUAL] — a human has to read the PASCAL A caveats — and is not
asserted here. T77.7's "full suite green" clause is satisfied by running the
suite, not by a test asserting about itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

#: Every committed run that writes a six-file contract, with the label space it
#: is allowed to use. Rule 4: these five vocabularies never mix.
RESULT_RUNS: tuple[tuple[str, str, str], ...] = (
    ("06_binary_results/EXP-A1", "binary", "EXP-A1"),
    ("06_binary_results/EXP-A2", "binary", "EXP-A2"),
    ("06_binary_results/EXP-A2-so04_subset", "binary", "EXP-A2"),
    ("07_multiclass_results/EXP-B1", "pascal_a", "EXP-B1"),
    ("07_multiclass_results/EXP-B1-defaults", "pascal_a", "EXP-B1"),
    ("07_multiclass_results/EXP-B2", "pascal_b", "EXP-B2"),
    ("07_multiclass_results/EXP-B2-defaults", "pascal_b", "EXP-B2"),
    ("08_circor_external_validation/EXP-C1-two_class", "circor_murmur", "EXP-C1"),
    ("08_circor_external_validation/EXP-C1-three_class", "circor_murmur", "EXP-C1"),
    ("08_circor_external_validation/EXP-C2", "circor_outcome", "EXP-C2"),
)

#: The five declared label spaces and the class codes each may contain.
LABEL_SPACES: dict[str, set[int]] = {
    "binary": {0, 1},
    "pascal_a": {0, 1, 2, 3},
    "pascal_b": {0, 1, 2},
    "circor_murmur": {0, 1, 2},
    "circor_outcome": {0, 1},
}

#: The standing rule's threshold. A metric at or above this is a symptom.
SUSPICIOUS = 0.99

METRIC_COLUMNS = (
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "roc_auc",
    "ovr_auc_macro",
)


def _root() -> Path:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / "outputs"


def _existing_runs() -> list[tuple[str, str, str, Path]]:
    found = []
    for directory, task, exp_id in RESULT_RUNS:
        path = _root() / directory
        if (path / "per_fold_metrics.csv").is_file():
            found.append((directory, task, exp_id, path))
    return found


@pytest.fixture(scope="module")
def runs() -> list[tuple[str, str, str, Path]]:
    found = _existing_runs()
    if not found:
        pytest.skip("no committed experiment results to audit")
    return found


# ---------------------------------------------------------------------------
# T77.1 — the seed-discipline proof
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
@pytest.mark.slow
def test_t77_1_exp_a1_fold_zero_reproduces_its_stored_metrics() -> None:
    """Re-fit EXP-A1's fold 0 and land on the committed numbers exactly.

    This is the whole project's reproducibility claim in one assertion: same
    seed, same fold map, same config, same numbers. Anything less than an exact
    match means some part of the pipeline carries state that the manifest does
    not record, and every stored metric becomes an observation rather than a
    result.

    M1 only, and fold 0 only: the point is bit-exactness, and re-fitting the
    ensembles over 25 folds to prove it would cost 90 minutes to test the same
    property.

    It runs against the REAL output directory on purpose. ``resume=False``
    refreshes one entry under ``_checkpoints/``, which is gitignored scratch,
    and writes nothing else -- ``write_outputs`` is not called, so the committed
    contract files are untouched. Pointing it at a tmp dir would exercise a
    different path from the one that produced the numbers being checked, which
    is the one thing this test exists to verify.
    """
    import pandas as pd

    from src.evaluation import experiment as ex

    directory = _root() / "06_binary_results" / "EXP-A1"
    if not (directory / "per_fold_metrics.csv").is_file():
        pytest.skip("EXP-A1 has not been run")

    stored = pd.read_csv(directory / "per_fold_metrics.csv")
    stored = stored[
        (stored["model_id"] == "M1") & (stored["repeat"] == 0) & (stored["fold"] == 0)
    ]
    if stored.empty:
        pytest.skip("EXP-A1 holds no M1 r0f0 row")

    result = ex.run_experiment(
        "EXP-A1",
        models=["M1"],
        repeats=[0],
        folds=[0],
        out_dir=None,
        resume=False,
    )
    fresh = result.per_fold_frame()
    fresh = fresh[(fresh["repeat"] == 0) & (fresh["fold"] == 0)]
    assert len(fresh) == 1

    for metric in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc", "f1"):
        if metric not in stored.columns or metric not in fresh.columns:
            continue
        assert float(fresh.iloc[0][metric]) == pytest.approx(
            float(stored.iloc[0][metric]), abs=1e-12
        ), metric + " did not reproduce"


def test_t77_1_every_run_manifest_records_its_seed_and_fold_map(
    runs: list[tuple[str, str, str, Path]]
) -> None:
    """A run that does not record how it was seeded cannot be re-run (rule 5)."""
    for directory, _, _, path in runs:
        manifest_path = path / "run_manifest.json"
        assert manifest_path.is_file(), directory + " has no run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        blob = json.dumps(manifest)
        assert "42" in blob, directory + " records no seed"
        assert (path / "fold_membership.parquet").is_file(), (
            directory + " did not write its fold membership, so no later test can "
            "demonstrate which folds it actually saw"
        )


# ---------------------------------------------------------------------------
# T77.2 — no search trial ever saw an outer test fold
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_t77_2_no_search_trial_scored_an_outer_test_row() -> None:
    """Every inner split sits strictly inside its own outer training fold (rule 2).

    Checked against the written ``inner_fold_map.csv``, not against the search
    code: the code can be right and the artifact still wrong, and the artifact
    is the evidence a reader would audit.
    """
    import pandas as pd

    from src.evaluation import cv as cv_module

    maps = sorted((_root() / "05_search_optimization").glob("*/inner_fold_map.csv"))
    if not maps:
        pytest.skip("no search inner-fold maps to audit")

    outer = {fold.label: fold for fold in cv_module.load_folds("binary")}
    checked = 0
    for path in maps:
        table = pd.read_csv(path)
        if "outer_fold" not in table.columns or "record_uid" not in table.columns:
            continue
        for label, block in table.groupby("outer_fold"):
            fold = outer.get(str(label))
            if fold is None:
                continue
            forbidden = set(fold.test_uids)
            used = set(block["record_uid"].astype(str))
            leaked = used & forbidden
            assert not leaked, (
                str(path.parent.name) + " outer fold " + str(label) + ": "
                + str(len(leaked)) + " outer-TEST record(s) appear in an inner "
                "split, e.g. " + ", ".join(sorted(leaked)[:5])
            )
            checked += 1
    assert checked, "no outer fold label in any inner map matched the DA-07 map"


@pytest.mark.needs_data
def test_t77_2_inner_splits_are_subject_disjoint() -> None:
    """A subject on both sides of an inner split leaks within the search (rule 3)."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    maps = sorted((_root() / "05_search_optimization").glob("*/inner_fold_map.csv"))
    if not maps:
        pytest.skip("no search inner-fold maps to audit")

    master = pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv",
        low_memory=False,
    )
    group_of = dict(
        zip(master["record_uid"].astype(str), master["split_group"].astype(str), strict=True)
    )

    for path in maps:
        table = pd.read_csv(path)
        if not {"outer_fold", "inner_fold", "role", "record_uid"} <= set(table.columns):
            continue
        table["group"] = [group_of.get(str(u), "?") for u in table["record_uid"]]
        for (outer_fold, inner_fold), block in table.groupby(["outer_fold", "inner_fold"]):
            roles = {str(r): set(g["group"]) for r, g in block.groupby("role")}
            train = roles.get("train", set())
            validate = roles.get("val", roles.get("valid", roles.get("test", set())))
            overlap = (train & validate) - {"?"}
            assert not overlap, (
                path.parent.name + " outer " + str(outer_fold) + " inner "
                + str(inner_fold) + ": " + str(len(overlap))
                + " subject(s) on both sides"
            )


# ---------------------------------------------------------------------------
# T77.3 — nothing is suspiciously good without an explanation
# ---------------------------------------------------------------------------


def _suspicious_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    table = pd.read_csv(path)
    hits = []
    for column in METRIC_COLUMNS:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        for index in values[values >= SUSPICIOUS].index:
            row = table.loc[index]
            hits.append(
                {
                    "file": str(path),
                    "metric": column,
                    "value": float(values[index]),
                    "model_id": str(row.get("model_id", "")),
                    "fold": str(row.get("fold", "")),
                    "repeat": str(row.get("repeat", "")),
                }
            )
    return hits


@pytest.mark.needs_data
def test_t77_3_no_aggregate_metric_is_at_or_above_099(
    runs: list[tuple[str, str, str, Path]]
) -> None:
    """A near-perfect AGGREGATE metric is a leak until proven otherwise.

    Aggregates, not per-fold values: a single fold scoring 1.0 on 24 test
    records is ordinary small-sample luck, while a model averaging 0.99 across
    25 folds is one of the four causes in the standing rule. The per-fold scan
    lives in its own test below and reports rather than fails.
    """
    offenders = []
    for _directory, _, _, path in runs:
        aggregate = path / "aggregate_metrics.csv"
        if aggregate.is_file():
            offenders.extend(_suspicious_rows(aggregate))

    assert not offenders, (
        "aggregate metric(s) at or above " + str(SUSPICIOUS)
        + " — investigate against the four causes (subject leak, scaler fitted "
        "on the full matrix, duplicate record across splits, label joined on the "
        "wrong key) and record the finding in Docs/note.md before this passes: "
        + json.dumps(offenders[:10], indent=1)
    )


@pytest.mark.needs_data
def test_t77_3_per_fold_outliers_are_enumerated_not_hidden(
    runs: list[tuple[str, str, str, Path]], capsys: Any
) -> None:
    """Every per-fold metric at or above 0.99, listed with its fold.

    This one does not fail on a hit -- a perfect specificity on one CirCor fold
    is expected when a model rarely predicts the positive class. It fails only
    if a whole MODEL is perfect across every fold, which no amount of small-fold
    luck explains.
    """
    import pandas as pd

    everywhere: list[str] = []
    found: list[dict[str, Any]] = []
    for directory, _, _, path in runs:
        per_fold = path / "per_fold_metrics.csv"
        if not per_fold.is_file():
            continue
        hits = _suspicious_rows(per_fold)
        found.extend(hits)

        table = pd.read_csv(per_fold)
        for metric in METRIC_COLUMNS:
            if metric not in table.columns:
                continue
            values = pd.to_numeric(table[metric], errors="coerce")
            for model_id, block in table.assign(_v=values).groupby("model_id"):
                clean = block["_v"].dropna()
                if len(clean) >= 5 and bool((clean >= SUSPICIOUS).all()):
                    everywhere.append(
                        directory + " " + str(model_id) + " " + metric
                        + " >= " + str(SUSPICIOUS) + " on all " + str(len(clean))
                        + " folds"
                    )

    with capsys.disabled():
        print(
            "\nT77.3: "
            + str(len(found))
            + " per-fold metric value(s) at or above "
            + str(SUSPICIOUS)
            + " across "
            + str(len({h["file"] for h in found}))
            + " file(s)."
        )
        for hit in found[:20]:
            print(
                "  "
                + Path(hit["file"]).parent.name
                + "  "
                + hit["model_id"]
                + "  r"
                + hit["repeat"]
                + "f"
                + hit["fold"]
                + "  "
                + hit["metric"]
                + "="
                + format(hit["value"], ".4f")
            )

    assert not everywhere, (
        "a model is at or above " + str(SUSPICIOUS) + " on EVERY fold, which "
        "small-sample luck does not explain: " + "; ".join(everywhere)
    )


# ---------------------------------------------------------------------------
# T77.4 — the five label spaces never merged
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_t77_4_every_run_uses_exactly_one_declared_label_space(
    runs: list[tuple[str, str, str, Path]]
) -> None:
    """The class codes in each run's predictions belong to one vocabulary (rule 4).

    PASCAL A and PASCAL B both number ``normal`` 0 and ``murmur`` 1, so a merge
    produces no error and no impossible value -- only a wider set of codes than
    the narrower space allows. That is exactly what this checks.
    """
    import pandas as pd

    for directory, task, _, path in runs:
        predictions = path / "predictions.parquet"
        if not predictions.is_file():
            continue
        frame = pd.read_parquet(predictions)
        allowed = LABEL_SPACES[task]
        for column in ("y_true", "y_pred"):
            if column not in frame.columns:
                continue
            seen = {int(v) for v in pd.unique(frame[column].dropna())}
            assert seen <= allowed, (
                directory + " (" + task + ") has " + column + " codes " + str(sorted(seen))
                + " but the declared label space is " + str(sorted(allowed))
            )


@pytest.mark.needs_data
def test_t77_4_pascal_a_and_b_share_no_record(
    runs: list[tuple[str, str, str, Path]]
) -> None:
    """The specific merge rule 4 exists to prevent, checked on the artifacts."""
    import pandas as pd

    def uids(directory: str) -> set[str]:
        path = _root() / directory / "predictions.parquet"
        if not path.is_file():
            return set()
        return set(pd.read_parquet(path)["record_uid"].astype(str))

    a = uids("07_multiclass_results/EXP-B1")
    b = uids("07_multiclass_results/EXP-B2")
    if not a or not b:
        pytest.skip("one of the PASCAL runs is missing")
    assert not (a & b), (
        str(len(a & b)) + " record(s) appear in both PASCAL A and PASCAL B results"
    )


@pytest.mark.needs_data
def test_t77_4_each_experiment_declares_the_task_its_output_uses() -> None:
    """The config's declared task matches the directory the results landed in."""
    from src.evaluation.experiment import Experiment

    for directory, task, exp_id, _path in _existing_runs():
        declared = Experiment.load(exp_id).task
        assert declared == task, (
            exp_id + " declares task " + declared + " but " + directory
            + " is audited as " + task
        )


# ---------------------------------------------------------------------------
# T77.5 — EXP-D1 is framed as transfer, and CirCor was never retuned
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_t77_5_exp_d1_carries_the_population_note() -> None:
    """The adult-to-paediatric framing travels with the numbers, not beside them."""
    path = _root() / "08_circor_external_validation" / "EXP-D1" / "population_mismatch.json"
    if not path.is_file():
        pytest.skip("EXP-D1 has not been run")

    payload = json.loads(path.read_text(encoding="utf-8"))
    blob = json.dumps(payload).lower()
    for token in ("adult", "paediatric", "transfer"):
        assert token in blob, "EXP-D1 metadata does not mention " + token
    assert "framing_rule" in payload, "EXP-D1 records no framing rule"
    assert payload.get("written_before_any_metric") is True, (
        "the population note must be written BEFORE the metrics, or it is a "
        "post-hoc excuse rather than a pre-registered expectation"
    )


@pytest.mark.needs_data
def test_t77_5_no_retuning_touched_circor() -> None:
    """A model retuned on CirCor is no longer an external validation."""
    path = _root() / "08_circor_external_validation" / "EXP-D1" / "population_mismatch.json"
    if not path.is_file():
        pytest.skip("EXP-D1 has not been run")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("retuning_performed") is False, (
        "EXP-D1 records retuning_performed=" + repr(payload.get("retuning_performed"))
    )
    assert payload.get("retuning_allowed") is False, (
        "EXP-D1 records retuning as allowed; external validation forbids it"
    )


@pytest.mark.needs_data
def test_t77_5_no_search_ran_on_a_circor_task() -> None:
    """No trial log names a CirCor task. Checked, not assumed."""
    import pandas as pd

    trials = sorted((_root() / "05_search_optimization").glob("*/trials.csv"))
    if not trials:
        pytest.skip("no search trial logs")

    for path in trials:
        table = pd.read_csv(path)
        if "task" not in table.columns:
            continue
        tasks = {str(t) for t in table["task"].dropna().unique()}
        circor = {t for t in tasks if t.startswith("circor")}
        assert not circor, path.parent.name + " searched on " + ", ".join(sorted(circor))


# ---------------------------------------------------------------------------
# EXTRA — the two ablations Phases 74-75 added
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_extra_feature_ablation_arms_share_one_fold_map() -> None:
    """T17's whole claim is 'identical folds'. This is that claim, checked."""
    import pandas as pd

    from src.evaluation import feature_ablation as fa

    membership: dict[str, Any] = {}
    for config_id in fa.configuration_ids():
        path = fa.arm_experiment(config_id).output_dir() / "fold_membership.parquet"
        if path.is_file():
            frame = pd.read_parquet(path).sort_values(
                ["repeat", "fold", "record_uid"], kind="mergesort"
            )
            membership[config_id] = frame.reset_index(drop=True)
    if len(membership) < 2:
        pytest.skip("fewer than two EXP-F1 arms have been run")

    reference_id, reference = next(iter(membership.items()))
    for config_id, frame in membership.items():
        assert len(frame) == len(reference), (
            config_id + " covers " + str(len(frame)) + " fold-record pairs but "
            + reference_id + " covers " + str(len(reference))
        )
        assert list(frame["record_uid"]) == list(reference["record_uid"]), (
            config_id + " and " + reference_id + " do not name the same records"
        )
        assert list(frame["fold"]) == list(reference["fold"]), (
            config_id + " and " + reference_id + " disagree on fold assignment"
        )


@pytest.mark.needs_data
def test_extra_optimization_stages_are_paired_on_shared_folds() -> None:
    """An incremental delta across different folds is not a delta."""
    from src.evaluation import optimization_ablation as oa

    try:
        _stages, per_fold = oa.build_stages()
    except FileNotFoundError as error:
        pytest.skip(str(error))

    counts = per_fold.groupby("stage").size()
    assert counts.nunique() == 1, (
        "the stages cover different fold counts: " + counts.to_dict().__str__()
    )
    keys = {
        stage: set(
            zip(block["repeat"].astype(int), block["fold"].astype(int), strict=True)
        )
        for stage, block in per_fold.groupby("stage")
    }
    reference = next(iter(keys.values()))
    for stage, folds in keys.items():
        assert folds == reference, stage + " does not share the other stages' folds"


@pytest.mark.needs_data
def test_extra_diagnosis_track_is_subject_disjoint_and_covers_every_class() -> None:
    """EXP-G1's own fold map, checked the way DA-07's five are."""
    from src.evaluation import diagnosis_track as dt

    try:
        folds = dt.load_folds()
    except FileNotFoundError as error:
        pytest.skip(str(error))

    records = dt.diagnosis_records()
    label_of = dict(zip(records["record_uid"].astype(str), records["y"].astype(int), strict=True))
    classes = set(label_of.values())

    for fold in folds:
        assert not (set(fold.train_groups) & set(fold.test_groups)), (
            "fold " + fold.label + " shares a subject across the split"
        )
        present = {label_of[uid] for uid in fold.test_uids if uid in label_of}
        assert present == classes, (
            "fold " + fold.label + " is missing class(es) "
            + str(sorted(classes - present))
            + "; a per-class recall with no support is not a number"
        )


@pytest.mark.needs_data
def test_extra_leave_one_source_out_is_reported_and_its_verdict_is_pinned() -> None:
    """EXP-F3 exists, and the finding that no model transfers is asserted.

    Pinned deliberately. EXP-F3 is the only measurement in the project that
    bounds what the pooled figures may be claimed to mean: leave-one-
    sub-collection-out takes AUC from 0.92-0.94 to 0.43-0.53, and **AUC is
    rank-based so class-prior shift cannot explain it**. If a later change makes
    a model transfer, this test fails and the standing rule in Docs/note.md
    ("the binary model does not transfer to an unseen recording collection")
    must be revisited rather than silently outlived.
    """
    import pandas as pd

    directory = _root() / "09_ablation" / "EXP-F3"
    if not (directory / "per_fold_metrics.csv").is_file():
        pytest.skip("EXP-F3 has not been run")

    per_fold = pd.read_csv(directory / "per_fold_metrics.csv")
    assert per_fold["held_out_source"].nunique() == 6, (
        "EXP-F3 must hold out all six PhysioNet sub-collections"
    )
    # Subject-disjointness is structural here (no subject spans a collection),
    # and that structure is what makes the experiment valid at all.
    from src.evaluation import source_holdout as sh

    records = sh.holdout_records()
    spanning = records.groupby("split_group")[sh.SOURCE_COLUMN].nunique()
    assert not int((spanning > 1).sum())

    comparison = directory / "pooled_vs_holdout.csv"
    assert comparison.is_file(), "EXP-F3 ran but its pooled comparison is missing"
    table = pd.read_csv(comparison)
    assert "roc_auc_pooled" in table.columns and "roc_auc_holdout" in table.columns

    best = float(table["roc_auc_holdout"].max())
    assert best < 0.75, (
        "a model now reaches held-out AUC " + format(best, ".4f")
        + " on an unseen sub-collection. That contradicts the standing rule in "
        "Docs/note.md; re-read the 2026-09-10 EXP-F3 entry before changing it"
    )
    assert float(table["roc_auc_pooled"].min()) > 0.85, (
        "the pooled AUCs no longer look like EXP-A1's; the comparison is not "
        "against the run it claims to be"
    )


@pytest.mark.needs_data
def test_extra_diagnosis_track_never_touches_a_normal_record() -> None:
    """EXP-G1 is the abnormal subtypes. A normal record here is a merged space."""
    import pandas as pd

    from src.evaluation import diagnosis_track as dt
    from src.utils.evidence import PROJECT_ROOT

    records = dt.diagnosis_records()
    master = pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv",
        low_memory=False,
    )
    binary = dict(
        zip(master["record_uid"].astype(str), master["binary_label"], strict=True)
    )
    normals = [
        uid for uid in records["record_uid"].astype(str) if binary.get(uid) == 0
    ]
    assert not normals, (
        str(len(normals)) + " normal record(s) in the diagnosis track, e.g. "
        + ", ".join(normals[:5])
    )
