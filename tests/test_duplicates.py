"""Duplicate detection gate (T17.7).

The corpus-wide tests need the full scan, including the 832 ``Heartbeat_Sound/``
files that are not records -- T17.3 is about proving they duplicate set_a +
set_b, so they have to be hashed alongside everything else.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.data_loader import duplicates as dup

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scanned() -> tuple[Any, Any]:
    """Scan of every WAV plus the Heartbeat_Sound helper files."""
    from src.data_loader.catalog import build_catalog
    from src.data_loader.integrity import scan_corpus
    from src.data_loader.pascal import heartbeat_sound_root

    catalog = build_catalog()
    extra = {
        ("HB_" + p.parent.name + "_" + p.stem): p
        for p in sorted(heartbeat_sound_root().rglob("*.wav"))
    }
    return scan_corpus(catalog, extra_files=extra)


@pytest.fixture(scope="module")
def report(scanned: tuple[Any, Any]) -> Any:
    scan, envelopes = scanned
    return dup.run_duplicate_detection(scan, envelopes)


# ===========================================================================
# T17.7 -- the gate
# ===========================================================================


@pytest.mark.needs_data
@pytest.mark.slow
def test_all_832_heartbeat_sound_files_are_flagged_and_dropped(
    scanned: tuple[Any, Any], report: Any
) -> None:
    """T17.3/T17.7 -- the whole point of this phase.

    Including any of these doubles the PASCAL corpus and puts the same recording
    in both train and test. All 832 must be detected as duplicates and every one
    marked drop.
    """
    scan, _ = scanned
    heartbeat_scanned = scan[scan["record_uid"].str.startswith(dup.HEARTBEAT_PREFIX)]
    assert len(heartbeat_scanned) == 832

    flagged = report[report["record_uid"].str.startswith(dup.HEARTBEAT_PREFIX)]
    assert flagged["record_uid"].nunique() == 832
    assert (flagged["decision"] == "drop").all()
    assert (flagged["method"] == "raw_sha256").all()

    # Each one points at the set_a or set_b record it copies, and never the
    # other way round.
    assert (flagged["duplicate_of"].str.startswith(("D2_set_a_", "D3_set_b_"))).all()


@pytest.mark.needs_data
@pytest.mark.slow
def test_no_pascal_record_survives_twice(scanned: tuple[Any, Any], report: Any) -> None:
    """T17.7 -- after applying the report, every PASCAL file appears once."""
    scan, _ = scanned
    dropped = set(report.loc[report["decision"] == "drop", "record_uid"])

    pascal = scan[scan["dataset_source"].isin(["D2", "D3"])]
    kept = pascal[~pascal["record_uid"].isin(dropped)]
    assert len(kept) == 832                     # 176 set_a + 656 set_b, once each
    assert kept["raw_sha256"].nunique() == 832  # and no two of them share bytes

    # No kept PASCAL record shares content with any other kept record anywhere.
    survivors = scan[~scan["record_uid"].isin(dropped)]
    assert survivors["content_sha256"].nunique() == len(survivors)


@pytest.mark.needs_data
@pytest.mark.slow
def test_physionet_validation_copies_are_dropped_not_their_twins(report: Any) -> None:
    """The other real duplication in this corpus, and the direction matters.

    All 301 validation records are byte-identical to a training record. The
    training copy is the one kept: that is where the subject grouping and the
    appendix annotations attach.
    """
    validation = report[report["record_uid"].str.contains("_validation_")]
    assert validation["record_uid"].nunique() == 301
    assert (validation["decision"] == "drop").all()
    assert (validation["duplicate_of"].str.contains("_training-")).all()

    kept = report[report["decision"] == "keep"]
    assert not kept["record_uid"].str.contains("_validation_").any()


@pytest.mark.needs_data
@pytest.mark.slow
def test_exact_and_content_hashes_agree_on_this_corpus(scanned: tuple[Any, Any]) -> None:
    """T17.1/T17.2 -- the content hash finds nothing the byte hash missed.

    Worth asserting rather than assuming: it means this corpus contains no
    re-encoded duplicates, which is a finding about the data, not a property of
    the method. If a future dataset drop broke it, the content layer is what
    would catch it.
    """
    scan, _ = scanned
    byte_groups = dup.exact_duplicate_groups(scan)
    content_groups = dup.content_duplicate_groups(scan)

    byte_members = {uid for group in byte_groups for uid, _, _ in group}
    content_members = {uid for group in content_groups for uid, _, _ in group}
    assert content_members == byte_members

    # 1,133 groups of two. 832 are a set_a/set_b record beside its
    # Heartbeat_Sound copy; 301 are a PhysioNet training record beside its
    # validation copy. Members, not groups: both halves of every pair count.
    assert len(byte_groups) == 832 + 301
    assert all(len(group) == 2 for group in byte_groups)
    assert len(byte_members) == 2 * (832 + 301) == 2266


@pytest.mark.needs_data
@pytest.mark.slow
def test_cross_dataset_check_between_pascal_b_and_circor(
    scanned: tuple[Any, Any],
) -> None:
    """T17.5 -- two 4 kHz paediatric corpora, and they share no material.

    If they did, EXP-D1's cross-corpus design would be measuring memorisation
    rather than transfer, so the absence is the result worth recording.
    """
    scan, envelopes = scanned
    from src.data_loader.integrity import load_thresholds

    threshold = load_thresholds().near_duplicate_correlation
    pairs = dup.cross_dataset_pairs(scan, envelopes, threshold, "D3", "D4")
    assert pairs == []

    shared = set(scan.loc[scan["dataset_source"] == "D3", "content_sha256"]) & set(
        scan.loc[scan["dataset_source"] == "D4", "content_sha256"]
    )
    assert shared == set()


@pytest.mark.needs_data
@pytest.mark.slow
def test_near_duplicates_are_reported_for_review_never_dropped(report: Any) -> None:
    """T17.4 -- a 0.98 envelope correlation is a hint, not a verdict."""
    near = report[report["method"].str.startswith("envelope_correlation")]
    assert (near["decision"] == "review").all()
    assert (near["similarity"] >= 0.98).all()
    assert (near["similarity"] <= 1.0 + 1e-9).all()


@pytest.mark.needs_data
@pytest.mark.slow
def test_duplicate_report_is_written(report: Any, tmp_path: Any) -> None:
    """T17.6 -- DA-06, with a decision on every row."""
    import pandas as pd

    target = dup.write_duplicate_report(report, tmp_path)
    assert target.name == "duplicate_report.csv"

    written = pd.read_csv(target)
    assert list(written.columns) == list(dup.DUPLICATE_REPORT_COLUMNS)
    assert set(written["decision"]) <= {"keep", "drop", "review"}
    assert written["decision"].notna().all()

    # Every drop names what it is a duplicate of; every keep names nothing.
    dropped = written[written["decision"] == "drop"]
    assert dropped["duplicate_of"].notna().all()
    assert (dropped["duplicate_of"].astype(str) != "").all()


# ===========================================================================
# pure-function tests -- no dataset required
# ===========================================================================


def test_keep_priority_order() -> None:
    """The rule that decides which member of a group survives."""
    heartbeat = dup.keep_priority("HB_normal_x", "heartbeat_sound")
    validation = dup.keep_priority("D1_validation_a0001", "D1")
    training = dup.keep_priority("D1_training-a_a0001", "D1")
    assert training < validation < heartbeat


def test_keep_priority_is_deterministic_between_equals() -> None:
    """Rule 5: two runs of the same command must pick the same survivor."""
    first = dup.keep_priority("D3_set_b_b", "D3")
    second = dup.keep_priority("D3_set_b_a", "D3")
    assert second < first


def _scan_frame(rows: list[dict[str, Any]]) -> Any:
    import pandas as pd

    return pd.DataFrame(rows)


def test_exact_duplicate_groups_sorts_the_keeper_first() -> None:
    scan = _scan_frame(
        [
            {
                "record_uid": "HB_normal_x",
                "dataset_source": "heartbeat_sound",
                "file_path": "hb/x.wav",
                "raw_sha256": "aaa",
                "content_sha256": "ccc",
            },
            {
                "record_uid": "D3_set_b_x",
                "dataset_source": "D3",
                "file_path": "set_b/x.wav",
                "raw_sha256": "aaa",
                "content_sha256": "ccc",
            },
        ]
    )
    groups = dup.exact_duplicate_groups(scan)
    assert len(groups) == 1
    assert groups[0][0][0] == "D3_set_b_x"
    assert groups[0][1][0] == "HB_normal_x"


def test_singletons_are_not_groups() -> None:
    scan = _scan_frame(
        [
            {
                "record_uid": "A",
                "dataset_source": "D1",
                "file_path": "a.wav",
                "raw_sha256": "one",
                "content_sha256": "one",
            },
            {
                "record_uid": "B",
                "dataset_source": "D1",
                "file_path": "b.wav",
                "raw_sha256": "two",
                "content_sha256": "two",
            },
        ]
    )
    assert dup.exact_duplicate_groups(scan) == []


def test_empty_hashes_are_never_grouped() -> None:
    """An unreadable file has no hash; two of them are not duplicates."""
    scan = _scan_frame(
        [
            {
                "record_uid": "A",
                "dataset_source": "D1",
                "file_path": "a.wav",
                "raw_sha256": "",
                "content_sha256": "",
            },
            {
                "record_uid": "B",
                "dataset_source": "D1",
                "file_path": "b.wav",
                "raw_sha256": "",
                "content_sha256": "",
            },
        ]
    )
    assert dup.exact_duplicate_groups(scan) == []


def test_near_duplicate_pairs_finds_the_obvious_case() -> None:
    import numpy as np

    scan = _scan_frame(
        [
            {"record_uid": "A", "dataset_source": "D3"},
            {"record_uid": "B", "dataset_source": "D3"},
            {"record_uid": "C", "dataset_source": "D3"},
        ]
    )
    base = np.sin(np.linspace(0, 8, 64)).astype(np.float32)
    base /= np.linalg.norm(base)
    other = np.cos(np.linspace(0, 8, 64)).astype(np.float32)
    other /= np.linalg.norm(other)
    envelopes = np.stack([base, base * 1.0, other])

    pairs = dup.near_duplicate_pairs(scan, envelopes, 0.98, within="D3")
    assert [(a, b) for a, b, _ in pairs] == [("A", "B")]


def test_near_duplicate_pairs_reports_each_pair_once() -> None:
    """Upper triangle only -- no self-pairs and no (B, A) beside (A, B)."""
    import numpy as np

    scan = _scan_frame(
        [{"record_uid": name, "dataset_source": "D3"} for name in "ABCD"]
    )
    vector = np.ones(8, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    envelopes = np.stack([vector] * 4)

    pairs = dup.near_duplicate_pairs(scan, envelopes, 0.9, within="D3")
    assert len(pairs) == 6                       # 4 choose 2
    assert all(a < b for a, b, _ in pairs)
