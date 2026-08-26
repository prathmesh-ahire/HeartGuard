"""🔴 MEGA TEST 1 — Data and preprocessing integrity (Phase 30).

Covers Parts I-III. Nothing in Part IV starts until this file is green.

**Why this exists when 476 other tests already pass.** Those tests check units:
a loader against its own parsing, a filter against its own response. This file
checks the *joins between them* on the artifacts that will actually be consumed
downstream -- the committed DA-08, the committed split map, the committed PP-08,
the 7,536 signals sitting in `cache/preprocessed/`. A leak introduced in Phase 12
is cheap to fix here and catastrophic to discover at submission, and it would not
be caught by any test that only looks at one module.

The T30.x tasks are the plan's own gate. The ``EXTRA`` sections below them are
cross-cutting checks added for this sweep: dataset immutability, cache integrity,
cross-artifact count agreement, and end-to-end determinism.

T30.6 is [TEST/MANUAL] and is deliberately not implemented here -- a person has
to look at the figures. T30.7's "full suite green" clause is satisfied by running
the suite, not by a test asserting about itself.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.needs_data

CORPUS_SIZE = 7536
SUPERVISED_SIZE = 6988
AUDITED_COUNTS = {
    "D1": 3541,   # 3,240 training + 301 validation copies
    "D2": 176,    # 124 labelled + 52 unlabelled
    "D3": 656,    # 461 labelled + 195 unlabelled
    "D4": 3163,   # 942 patients
}


# ---------------------------------------------------------------------------
# fixtures -- the committed artifacts, read once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def master() -> Any:
    from src.data_loader import master as ms

    return ms.load_master()


@pytest.fixture(scope="module")
def split_map() -> Any:
    from src.data_loader.splits import load_split_map

    return load_split_map()


@pytest.fixture(scope="module")
def quality() -> Any:
    from src.preprocessing.quality import scan_quality

    return scan_quality()


@pytest.fixture(scope="module")
def root() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("project_root"))


# ===========================================================================
# T30.1 -- the audit reproduces its counts from an empty output directory
# ===========================================================================


@pytest.mark.slow
def test_t30_1_audit_reproduces_the_audited_counts_from_scratch(tmp_path: Path) -> None:
    """Re-run the whole audit into an empty directory and check the four families.

    These are the numbers in CLAUDE.md's dataset table, and two of them
    contradict the source documents (CirCor 942 patients / 3,163 recordings;
    PhysioNet 3,240 rather than 3,541 usable). Reality wins, and this is where
    that is re-established rather than assumed.
    """
    import importlib.util
    import sys

    import pandas as pd

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "01_run_dataset_audit.py"
    spec = importlib.util.spec_from_file_location("audit_script", script_path)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    sys.modules["audit_script"] = script
    spec.loader.exec_module(script)

    out_dir = tmp_path / "audit"
    out_dir.mkdir()
    summary = script.run_audit(
        script.parse_args(["--out-dir", str(out_dir), "--skip-verify"])
    )

    assert summary["mode"] == "full"
    assert summary["n_records"] == CORPUS_SIZE
    assert summary["n_supervised"] == SUPERVISED_SIZE

    fresh = pd.read_csv(out_dir / "metadata_master.csv", keep_default_na=False)
    assert len(fresh) == CORPUS_SIZE
    assert fresh["dataset_source"].value_counts().to_dict() == AUDITED_COUNTS

    # PhysioNet: 3,240 training records, 301 validation copies.
    d1 = fresh[fresh["dataset_source"] == "D1"]
    assert int((d1["subset"] == "validation").sum()) == 301
    assert int((d1["subset"] != "validation").sum()) == 3240

    # PASCAL A: 124 labelled. PASCAL B: 461 labelled.
    labelled = fresh[fresh["is_unlabeled"].astype(str).str.lower() == "false"]
    assert int((labelled["dataset_source"] == "D2").sum()) == 124
    assert int((labelled["dataset_source"] == "D3").sum()) == 461

    # CirCor: 942 patients over 3,163 recordings.
    d4 = fresh[fresh["dataset_source"] == "D4"]
    assert len(d4) == 3163
    assert d4["subject_id"].nunique() == 942


def test_t30_1_committed_master_carries_the_same_counts(master: Any) -> None:
    """The artifact on disk, which is what everything downstream actually reads."""
    assert len(master) == CORPUS_SIZE
    assert master["dataset_source"].value_counts().to_dict() == AUDITED_COUNTS
    assert int(master["use_in_supervised"].sum()) == SUPERVISED_SIZE
    assert master[master["dataset_source"] == "D4"]["subject_id"].nunique() == 942


# ===========================================================================
# T30.2 -- no subject leaks across any fold of any task
# ===========================================================================


def test_t30_2_zero_subject_overlap_in_every_fold_of_every_task(split_map: Any) -> None:
    """Rule 3, checked on the committed DA-07 rather than a rebuilt copy."""
    from src.data_loader.splits import assert_no_leakage, iter_folds

    assert_no_leakage(split_map)

    # The map stores one row per record per repeat, carrying the fold it is the
    # TEST member of; the train side is the rest of that repeat.
    subjects = dict(zip(split_map["record_uid"], split_map["subject_id"], strict=True))
    checked = 0
    for task in sorted(set(split_map["task"])):
        for repeat, fold, train_uids, test_uids in iter_folds(split_map, task):
            train = {subjects[uid] for uid in train_uids}
            test = {subjects[uid] for uid in test_uids}
            assert not (train & test), (task, repeat, fold, sorted(train & test)[:5])
            assert test, (task, repeat, fold)
            assert train, (task, repeat, fold)
            checked += 1

    # Five tasks x 5 repeats x 5 folds; a silently empty map would pass the
    # overlap check above by having nothing to overlap.
    assert checked >= 25, checked
    assert split_map["task"].nunique() >= 5


def test_t30_2_the_301_validation_records_appear_in_no_fold(
    master: Any, split_map: Any
) -> None:
    from src.data_loader.splits import assert_validation_excluded

    assert_validation_excluded(split_map, master)

    validation = set(
        master[(master["dataset_source"] == "D1") & (master["subset"] == "validation")][
            "record_uid"
        ]
    )
    assert len(validation) == 301
    assert not (validation & set(split_map["record_uid"]))


def test_t30_2_every_supervised_record_is_placed_exactly_once_per_repeat(
    split_map: Any,
) -> None:
    for (task, repeat), group in split_map.groupby(["task", "repeat"]):
        assert group["record_uid"].is_unique, (task, repeat)
        assert set(group["fold"]) == set(range(int(group["n_splits"].iloc[0]))), (task, repeat)


# ===========================================================================
# T30.3 -- preprocessing is bit-identical twice, and the cache is used
# ===========================================================================


@pytest.mark.slow
def test_t30_3_two_hundred_records_preprocess_identically_twice(
    master: Any, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 5 end to end, across all four datasets, into a throwaway cache."""
    from src.preprocessing.pipeline import cache_stats, preprocess
    from src.utils import config as config_module

    monkeypatch.setenv("HEARTGUARD__PATHS__CACHE__PREPROCESSED", str(tmp_path / "cache"))
    config_module.clear_cache()
    try:
        sample = master.sample(n=200, random_state=42)
        assert sample["dataset_source"].nunique() == 4

        first: dict[str, np.ndarray] = {}
        for row in sample.itertuples(index=False):
            result = preprocess(root / str(row.file_path), record_uid=str(row.record_uid))
            assert result.from_cache is False, row.record_uid
            first[str(row.record_uid)] = result.signal

        assert cache_stats()["n_files"] == 200

        for row in sample.itertuples(index=False):
            result = preprocess(root / str(row.file_path), record_uid=str(row.record_uid))
            assert result.from_cache is True, row.record_uid
            assert np.array_equal(result.signal, first[str(row.record_uid)]), row.record_uid

        # Recomputed from the WAV, not read back from the cache.
        for row in sample.head(25).itertuples(index=False):
            fresh = preprocess(
                root / str(row.file_path), record_uid=str(row.record_uid), use_cache=False
            )
            assert np.array_equal(fresh.signal, first[str(row.record_uid)]), row.record_uid
    finally:
        config_module.clear_cache()


# ===========================================================================
# T30.4 -- the duration extremes
# ===========================================================================


def test_t30_4_duration_extremes_survive_preprocessing(master: Any, root: Path) -> None:
    """0.76 s and 122 s: no error, and no silent truncation."""
    from src.preprocessing.pipeline import preprocess

    shortest = master.loc[master["duration_sec"].idxmin()]
    longest = master.loc[master["duration_sec"].idxmax()]

    assert float(shortest["duration_sec"]) < 1.0
    assert float(longest["duration_sec"]) > 100.0

    for row in (shortest, longest):
        result = preprocess(
            root / str(row["file_path"]), record_uid=str(row["record_uid"]), use_cache=False
        )
        expected = round(float(row["n_samples"]) * result.fs / float(row["original_fs"]))

        assert result.n_samples == pytest.approx(expected, abs=1), row["record_uid"]
        assert result.duration_sec == pytest.approx(float(row["duration_sec"]), abs=1e-2)
        assert np.isfinite(result.signal).all(), row["record_uid"]
        assert result.filter_info["filter_too_short"] is False, row["record_uid"]
        assert result.filter_info["filter_padlen_reduced"] is False, row["record_uid"]
        # A z-scored signal that is all zeros is a truncation that kept its length.
        assert float(result.signal.std()) == pytest.approx(1.0, abs=1e-3), row["record_uid"]


# ===========================================================================
# T30.5 -- no Heartbeat_Sound leakage, no duplicate uids
# ===========================================================================


def test_t30_5_no_heartbeat_sound_file_is_a_supervised_record(master: Any) -> None:
    """The 100% duplicate of set_a + set_b. Including it doubles the data silently."""
    paths = master["file_path"].astype(str)
    assert not paths.str.contains("Heartbeat_Sound", case=False).any()

    supervised = master[master["use_in_supervised"]]
    assert not supervised["file_path"].astype(str).str.contains(
        "Heartbeat_Sound", case=False
    ).any()
    assert len(supervised) == SUPERVISED_SIZE


def test_t30_5_no_record_uid_is_duplicated(master: Any) -> None:
    assert master["record_uid"].is_unique
    assert master["record_uid"].notna().all()
    assert (master["record_uid"].astype(str).str.strip() != "").all()


def test_t30_5_no_duplicate_audio_reaches_a_supervised_set(master: Any) -> None:
    """Content hashes, not filenames: the same audio under two names is still one."""
    supervised = master[master["use_in_supervised"]]
    hashes = supervised["content_sha256"].astype(str)
    hashes = hashes[hashes.str.len() > 0]
    duplicated = hashes[hashes.duplicated(keep=False)]

    if not duplicated.empty:
        # The one known and documented case: PASCAL A's re-encoded pair, which is
        # kept under one subject_id (Docs/note.md, 2026-08-26 Phases 16-18).
        offenders = supervised[supervised["content_sha256"].isin(set(duplicated))]
        assert offenders["dataset_source"].unique().tolist() == ["D2"], (
            offenders[["record_uid", "dataset_source"]].to_dict("records")
        )
        assert offenders["subject_id"].nunique() == 1


# ===========================================================================
# EXTRA 1 -- dataset immutability
# ===========================================================================


@pytest.mark.slow
def test_extra_dataset_files_are_unchanged_since_the_audit(master: Any, root: Path) -> None:
    """`dataset/` is read-only input. Re-derive content hashes and compare.

    A recording edited, re-encoded or replaced since the audit would invalidate
    every count, duplicate decision and quality flag downstream, and nothing else
    in the project would notice.
    """
    from src.data_loader.integrity import _content_hash, load_thresholds
    from src.preprocessing.io import load_wav, to_mono

    thresholds = load_thresholds()
    sample = master[master["content_sha256"].astype(str).str.len() > 0].sample(
        n=60, random_state=42
    )
    assert sample["dataset_source"].nunique() == 4

    for row in sample.itertuples(index=False):
        samples, fs = load_wav(root / str(row.file_path))
        digest = _content_hash(to_mono(samples), fs, thresholds)
        assert digest == str(row.content_sha256), row.record_uid


def test_extra_every_master_file_path_still_resolves(master: Any, root: Path) -> None:
    missing = [
        str(row.record_uid)
        for row in master.itertuples(index=False)
        if not (root / str(row.file_path)).is_file()
    ]
    assert not missing, missing[:10]


# ===========================================================================
# EXTRA 2 -- the preprocessed cache is complete and healthy
# ===========================================================================


@pytest.mark.slow
def test_extra_preprocessed_cache_covers_the_corpus(master: Any) -> None:
    """All 7,536 signals cached under the current config hash, and readable."""
    from src.preprocessing.pipeline import cache_root, cache_stats, config_hash

    digest = config_hash()
    stats = cache_stats()
    assert stats["n_files"] == CORPUS_SIZE, stats
    assert stats["megabytes"] > 500

    directory = cache_root()
    assert directory.name == digest

    missing = [
        str(uid) for uid in master["record_uid"] if not (directory / (str(uid) + ".npz")).is_file()
    ]
    assert not missing, missing[:10]


@pytest.mark.slow
def test_extra_cached_signals_match_their_master_rows(master: Any) -> None:
    """A cache entry whose length disagrees with DA-08 is a silent truncation."""
    from src.preprocessing.pipeline import cache_path

    sample = master.sample(n=200, random_state=7)
    for row in sample.itertuples(index=False):
        with np.load(cache_path(str(row.record_uid)), allow_pickle=False) as bundle:
            signal = np.asarray(bundle["signal"], dtype=np.float32)

        expected = round(float(row.n_samples) * 2000 / float(row.original_fs))
        assert abs(signal.size - expected) <= 1, row.record_uid
        assert np.isfinite(signal).all(), row.record_uid
        assert float(signal.std()) == pytest.approx(1.0, abs=1e-2), row.record_uid


# ===========================================================================
# EXTRA 3 -- artifacts agree with each other
# ===========================================================================


def test_extra_pp08_covers_exactly_the_master_records(master: Any, quality: Any) -> None:
    assert len(quality) == len(master)
    assert list(quality["record_uid"]) == list(master["record_uid"])
    assert quality["duration_sec"].sum() == pytest.approx(
        master["duration_sec"].sum(), rel=1e-3
    )


def test_extra_pp08_flags_are_internally_consistent(quality: Any) -> None:
    """`is_low_quality` must be exactly the OR of its four components."""
    composite = (
        quality["is_noisy"] | quality["is_short"] | quality["is_clipped"] | quality["is_silent"]
    )
    assert (quality["is_low_quality"] == composite).all()

    # Every flagged record names a reason, and every unflagged one names none.
    reasons = quality["quality_reasons"].astype(str)
    assert (reasons[quality["is_low_quality"]].str.len() > 0).all()
    assert (reasons[~quality["is_low_quality"]].str.len() == 0).all()


def test_extra_audit_summary_tables_agree_with_master(master: Any, root: Path) -> None:
    """DA-01/DA-02 are derived views; a disagreement means one is stale."""
    import pandas as pd

    audit_dir = root / "outputs" / "01_dataset_audit"
    inventory = pd.read_csv(audit_dir / "dataset_inventory.csv", keep_default_na=False)

    for dataset, expected in AUDITED_COUNTS.items():
        rows = inventory[inventory["dataset_source"].astype(str) == dataset]
        assert not rows.empty, dataset
        assert int(rows["total_files"].astype(int).sum()) == expected, dataset

    distribution = pd.read_csv(audit_dir / "class_distribution.csv", keep_default_na=False)
    binary = distribution[
        (distribution["task"] == "binary") & (distribution["scope"] == "supervised")
    ]
    assert not binary.empty
    supervised_binary = master[master["use_in_supervised"] & master["binary_label"].notna()]
    assert int(binary["n_records"].astype(int).sum()) == len(supervised_binary)


def test_extra_evidence_index_rows_resolve_or_are_declared_gaps(root: Path) -> None:
    """Every registered artifact exists, except gaps recorded in the report."""
    from src.utils.evidence import read_evidence

    report = (root / "outputs" / "missing_outputs_report.txt").read_text(encoding="utf-8")
    for row in read_evidence():
        path = root / row["filename"]
        if row["status"] == "ok":
            assert path.is_file(), row["evidence_id"] + " claims ok but is absent"
            assert path.stat().st_size > 0, row["evidence_id"]
        else:
            assert row["evidence_id"] in report, (
                row["evidence_id"] + " is missing but not declared in "
                "missing_outputs_report.txt"
            )


def test_extra_no_evidence_row_points_outside_the_project(root: Path) -> None:
    """Regression: the committed index once held nine rows inside a temp folder.

    A ``--out-dir`` run (the T22.7 audit-script test) rewrote DA-01..DA-09 to
    paths under ``AppData/Local/Temp/pytest-of-.../``, all claiming
    ``status=ok`` for files deleted when the test finished. Every gate passed,
    because every gate checked artifacts rather than the index.
    """
    from src.utils.evidence import read_evidence

    rows = read_evidence()
    assert rows, "the evidence index is empty"
    for row in rows:
        path = Path(row["filename"])
        assert not path.is_absolute(), row["evidence_id"] + " -> " + row["filename"]
        assert "pytest" not in row["filename"].lower(), row["evidence_id"]
        assert (root / path).resolve().is_relative_to(root.resolve()), row["evidence_id"]


def test_extra_registering_an_outside_artifact_cannot_touch_the_real_index(
    tmp_path: Path,
) -> None:
    """The structural guard, exercised rather than trusted."""
    from src.utils.evidence import index_for_artifact, read_evidence, register_evidence

    before = {row["evidence_id"]: row["filename"] for row in read_evidence()}
    assert "DA-08" in before

    outside = tmp_path / "metadata_master.csv"
    outside.write_text("record_uid", encoding="utf-8")
    assert index_for_artifact(outside) == tmp_path / "evidence_index.csv"

    register_evidence(
        evidence_id="DA-08",
        filename=outside,
        metric_or_asset="deliberate pollution attempt",
    )

    after = {row["evidence_id"]: row["filename"] for row in read_evidence()}
    assert after["DA-08"] == before["DA-08"]
    assert (tmp_path / "evidence_index.csv").is_file()


def test_extra_declared_gaps_match_the_blocked_tasks(root: Path) -> None:
    """A `[!]` or `[-]` task in todo.md requires an entry in the report."""
    todo = (root / "Docs" / "todo.md").read_text(encoding="utf-8")
    report = (root / "outputs" / "missing_outputs_report.txt").read_text(encoding="utf-8")

    blocked = [
        line.split("**")[1]
        for line in todo.splitlines()
        if line.startswith(("- [!]", "- [-]")) and "**" in line
    ]
    assert blocked, "no blocked tasks found -- has the marker format changed?"
    for task in blocked:
        assert task in report, task + " is blocked in todo.md but not in the report"


# ===========================================================================
# EXTRA 4 -- the five label spaces stay separate
# ===========================================================================


def test_extra_no_record_carries_a_label_for_a_foreign_task(master: Any) -> None:
    """Rule 4, on the committed table."""
    from src.data_loader.master import assert_no_cross_task_bleed

    assert_no_cross_task_bleed(master)


def test_extra_each_task_frame_is_the_dataset_it_belongs_to(master: Any) -> None:
    from src.data_loader.master import task_frame

    expected = {
        "binary": {"D1"},
        "pascal_a": {"D2"},
        "pascal_b": {"D3"},
        "circor_murmur": {"D4"},
        "circor_outcome": {"D4"},
    }
    for task, datasets in expected.items():
        frame = task_frame(master, task)
        assert not frame.empty, task
        assert set(frame["dataset_source"]) == datasets, task
        assert frame["y"].notna().all(), task


def test_extra_pascal_a_and_b_are_never_pooled(master: Any) -> None:
    """Both number `normal` 0 and `murmur` 1; pooling them is silent corruption."""
    from src.data_loader.master import task_frame

    a = task_frame(master, "pascal_a")
    b = task_frame(master, "pascal_b")
    assert not set(a["record_uid"]) & set(b["record_uid"])
    assert set(master["multiclass_task"].dropna().unique()) >= {"pascal_a", "pascal_b"}


# ===========================================================================
# EXTRA 5 -- configuration invariants
# ===========================================================================


def test_extra_every_config_loads_and_validates() -> None:
    from src.utils.config import load_all

    configs = load_all(validate=True, use_cache=False)
    assert set(configs) == {"paths", "signal", "features", "models", "experiments"}


def test_extra_the_locked_138_still_sums(features_config: Any) -> None:
    counts = {
        family: int(features_config.require("families." + family + ".count"))
        for family in ("time", "frequency", "mfcc", "chroma", "dwt", "envelope")
    }
    assert counts == {
        "time": 24,
        "frequency": 22,
        "mfcc": 39,
        "chroma": 24,
        "dwt": 24,
        "envelope": 5,
    }
    assert sum(counts.values()) == 138
    assert int(features_config.require("expected_total")) == 138


def test_extra_seed_is_42_everywhere(config: dict[str, Any]) -> None:
    assert int(config["models"].require("global.random_state")) == 42
    assert int(config["experiments"].require("defaults.seed")) == 42

    from src.utils.seed import GLOBAL_SEED

    assert GLOBAL_SEED == 42


def test_extra_passband_stays_below_nyquist(signal_config: Any) -> None:
    fs = int(signal_config.require("resample.target_fs"))
    high = float(signal_config.require("filter.high_hz"))
    low = float(signal_config.require("filter.low_hz"))
    assert 0 < low < high < fs / 2
    assert tuple(signal_config.require("quality.snr_proxy.in_band")) == (low, high)


# ===========================================================================
# EXTRA 6 -- end-to-end determinism
# ===========================================================================


def test_extra_the_whole_chain_is_reproducible(master: Any, root: Path) -> None:
    """Same record, same seed, twice: identical signal and identical metrics."""
    from src.preprocessing.pipeline import preprocess
    from src.utils.seed import set_global_seed

    row = master[master["dataset_source"] == "D2"].iloc[0]
    path = root / str(row["file_path"])

    set_global_seed(42)
    first = preprocess(path, record_uid=str(row["record_uid"]), use_cache=False)
    set_global_seed(42)
    second = preprocess(path, record_uid=str(row["record_uid"]), use_cache=False)

    assert np.array_equal(first.signal, second.signal)
    assert first.quality == second.quality
    assert first.steps == second.steps
    assert first.config_hash == second.config_hash


def test_extra_pp08_regenerates_byte_identically(quality: Any, tmp_path: Path) -> None:
    """A deliverable that differs between two runs is not reproducible."""
    from src.preprocessing.quality import write_quality_flags

    first = write_quality_flags(quality, tmp_path / "a")
    second = write_quality_flags(quality, tmp_path / "b")
    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_extra_committed_pp08_matches_a_fresh_write(quality: Any, tmp_path: Path) -> None:
    """The committed PP-08 is what the current code produces, not an older run."""
    from src.preprocessing.quality import quality_flags_path, write_quality_flags

    fresh = write_quality_flags(quality, tmp_path)
    committed = quality_flags_path()
    assert committed.is_file()
    assert hashlib.sha256(fresh.read_bytes()).hexdigest() == (
        hashlib.sha256(committed.read_bytes()).hexdigest()
    ), "outputs/02_preprocessing/signal_quality_flags.csv is stale -- regenerate it"
