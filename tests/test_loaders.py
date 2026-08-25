"""Every loader against the real dataset tree (T22.4).

The other loader test files each go deep on one dataset. This one is the
breadth check: each of the four loaders, plus the hardening added in Phase 22 --
``--limit`` sampling and the metadata cache -- exercised against the files on
disk rather than a fixture.

Two properties matter most and are easy to lose:

* ``--limit`` must keep every class. A smoke sample that is 20 PhysioNet
  ``normal`` records runs green and proves nothing about the abnormal path.
* the cache must return the same table the builder would have. A cache that
  drifts from its source is worse than no cache, because everything downstream
  keeps working and is quietly wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.data_loader import cache as ch

pytestmark = pytest.mark.needs_data


# ===========================================================================
# the four loaders
# ===========================================================================


def test_physionet_loader_returns_the_audited_shape() -> None:
    from src.data_loader.physionet import (
        N_ALL_ROWS,
        N_TRAINING_RECORDS,
        load_physionet,
    )

    table = load_physionet()
    assert len(table) == N_ALL_ROWS == 3541
    assert int(table["use_in_supervised"].sum()) == N_TRAINING_RECORDS == 3240
    assert set(table["binary_label_name"]) == {"normal", "abnormal"}
    assert table["record_uid"].is_unique
    assert (table["original_fs"] == 2000).all()


def test_pascal_loader_returns_both_sets_with_their_own_label_spaces() -> None:
    from src.data_loader.pascal import load_pascal

    table = load_pascal()
    assert len(table) == 832
    counts = table["dataset_source"].value_counts().to_dict()
    assert counts == {"D3": 656, "D2": 176}
    set_a = table[table["dataset_source"] == "D2"]
    set_b = table[table["dataset_source"] == "D3"]
    assert (set_a["original_fs"] == 44100).all()
    assert (set_b["original_fs"] == 4000).all()
    # Rule 4 at the loader level: the two vocabularies never mix.
    a_classes = set(set_a.loc[~set_a["is_unlabeled"], "multiclass_label_name"])
    b_classes = set(set_b.loc[~set_b["is_unlabeled"], "multiclass_label_name"])
    assert a_classes == {"normal", "murmur", "extrahls", "artifact"}
    assert b_classes == {"normal", "murmur", "extrastole"}


def test_circor_loader_returns_both_tasks_side_by_side() -> None:
    from src.data_loader.circor import N_PATIENTS, N_RECORDINGS, load_circor

    table = load_circor(with_segmentation=False)
    assert len(table) == N_RECORDINGS == 3163
    assert table["patient_id"].nunique() == N_PATIENTS == 942
    assert set(table["murmur"]) == {"Absent", "Present", "Unknown"}
    # The outcome label lives only in the per-patient .txt files, never in
    # training_data.csv. A loader built around the CSV loses EXP-C2 entirely.
    assert set(table["outcome"]) == {"Normal", "Abnormal"}
    assert (table["original_fs"] == 4000).all()


def test_catalog_concatenates_all_four_without_losing_a_record() -> None:
    from src.data_loader.catalog import build_catalog

    catalog = build_catalog()
    assert len(catalog) == 7536
    assert catalog["record_uid"].is_unique
    assert set(catalog["dataset_source"]) == {"D1", "D2", "D3", "D4"}


# ===========================================================================
# T22.1 -- --limit
# ===========================================================================


def test_limit_keeps_every_physionet_class_in_every_subset() -> None:
    from src.data_loader.physionet import load_physionet

    table = load_physionet(limit=10)
    for subset, block in table.groupby("subset"):
        assert len(block) <= 10, subset
        assert set(block["binary_label_name"]) == {"normal", "abnormal"}, subset


def test_limit_keeps_every_circor_murmur_class() -> None:
    """20 records must contain Absent, Present *and* Unknown.

    Unknown is 4.9% of CirCor. A head(20) sample would contain none of it, and
    the three-class path would go untested by every smoke run.
    """
    from src.data_loader.circor import load_circor

    table = load_circor(with_segmentation=False, limit=20)
    assert len(table) == 20
    assert set(table["murmur"]) == {"Absent", "Present", "Unknown"}


def test_limit_keeps_every_pascal_class_in_both_sets() -> None:
    from src.data_loader.pascal import load_pascal

    table = load_pascal(limit=20)
    for dataset, expected in (
        ("D2", {"normal", "murmur", "extrahls", "artifact"}),
        ("D3", {"normal", "murmur", "extrastole"}),
    ):
        block = table[table["dataset_source"] == dataset]
        assert len(block) <= 20, dataset
        labelled = block[~block["is_unlabeled"]]
        assert set(labelled["multiclass_label_name"]) == expected, dataset


def test_limit_is_deterministic() -> None:
    """Rule 5. No RNG is involved, and the test proves it rather than assuming."""
    from src.data_loader.circor import load_circor

    first = load_circor(with_segmentation=False, limit=15)
    second = load_circor(with_segmentation=False, limit=15)
    assert list(first["record_uid"]) == list(second["record_uid"])


def test_limit_rejects_a_non_positive_value() -> None:
    import pandas as pd

    frame = pd.DataFrame({"record_uid": ["a", "b"], "dataset_source": ["D1", "D1"]})
    with pytest.raises(ValueError, match="positive integer"):
        ch.apply_limit(frame, 0)


def test_a_limited_run_refuses_to_write_audit_outputs() -> None:
    """A 60-row DA-08 on disk is indistinguishable from a complete one."""
    from src.data_loader.circor import load_circor
    from src.data_loader.pascal import load_pascal
    from src.data_loader.physionet import load_physionet

    for loader in (load_physionet, load_pascal, load_circor):
        with pytest.raises(ValueError, match="--limit run"):
            loader(limit=5, write_outputs=True)


def test_limit_larger_than_the_table_returns_everything() -> None:
    from src.data_loader.pascal import load_pascal

    assert len(load_pascal(limit=10_000)) == 832


# ===========================================================================
# T22.2 -- the metadata cache
# ===========================================================================


def test_cache_key_changes_when_a_metadata_file_changes(tmp_path: Path) -> None:
    source = tmp_path / "meta.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    first = ch.metadata_cache_key("t", metadata_files=[source])
    assert first == ch.metadata_cache_key("t", metadata_files=[source])

    source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert ch.metadata_cache_key("t", metadata_files=[source]) != first


def test_cache_key_changes_when_a_file_is_added_to_the_tree(tmp_path: Path) -> None:
    tree = tmp_path / "audio"
    tree.mkdir()
    (tree / "one.wav").write_bytes(b"\x00" * 16)
    first = ch.metadata_cache_key("t", trees=[tree])

    (tree / "two.wav").write_bytes(b"\x00" * 16)
    assert ch.metadata_cache_key("t", trees=[tree]) != first


def test_cache_key_changes_with_the_cache_version(monkeypatch: Any) -> None:
    """A schema change must invalidate every cache written before it."""
    first = ch.metadata_cache_key("t")
    monkeypatch.setattr(ch, "CACHE_VERSION", ch.CACHE_VERSION + 1)
    assert ch.metadata_cache_key("t") != first


def test_tree_signature_reports_a_missing_root(tmp_path: Path) -> None:
    signature = ch.tree_signature(tmp_path / "nope")
    assert signature["exists"] is False


def test_cached_table_returns_the_builder_result_and_then_reuses_it() -> None:
    import pandas as pd

    calls: list[int] = []

    def build() -> Any:
        calls.append(1)
        return pd.DataFrame({"x": [1, 2, 3]})

    name = "pytest_cache_probe"
    ch.clear_metadata_cache(name)
    try:
        first = ch.cached_table(name, build, extra={"probe": True})
        second = ch.cached_table(name, build, extra={"probe": True})
        assert len(calls) == 1, "the second call should have hit the cache"
        assert first.equals(second)

        # A different input must not be served from the first key's file.
        third = ch.cached_table(name, build, extra={"probe": False})
        assert len(calls) == 2
        assert third.equals(first)
    finally:
        ch.clear_metadata_cache(name)


def test_disabling_the_cache_always_rebuilds() -> None:
    import pandas as pd

    calls: list[int] = []

    def build() -> Any:
        calls.append(1)
        return pd.DataFrame({"x": [1]})

    ch.cached_table("pytest_disabled_probe", build, enabled=False)
    ch.cached_table("pytest_disabled_probe", build, enabled=False)
    assert len(calls) == 2
    assert ch.clear_metadata_cache("pytest_disabled_probe") == 0


@pytest.mark.slow
def test_the_cache_returns_the_same_table_the_builder_would_have() -> None:
    """The failure mode that matters: a cache that has drifted from its source.

    Everything downstream keeps working, and is quietly wrong.
    """
    from src.data_loader.pascal import load_pascal

    cached = load_pascal(use_cache=True)
    fresh = load_pascal(use_cache=False)
    assert list(cached.columns) == list(fresh.columns)
    assert len(cached) == len(fresh)
    assert list(cached["record_uid"]) == list(fresh["record_uid"])
    assert list(cached["multiclass_label_name"]) == list(fresh["multiclass_label_name"])
    assert list(cached["subject_id"]) == list(fresh["subject_id"])
