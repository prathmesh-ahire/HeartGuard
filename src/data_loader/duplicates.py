"""Duplicate detection across the corpus (Phase 17).

Four layers, cheapest first, because each catches something the next cannot.

**Raw-byte SHA-256** (T17.1) finds files that are literally the same bytes. It
catches ``Heartbeat_Sound/`` and the PhysioNet ``validation/`` folder, which are
this corpus's two real duplication problems.

**Audio-content hash** (T17.2) finds the same recording stored differently --
re-encoded, different container, different bit depth. Computed on mono audio
resampled to one rate and rounded, so two encodings of one signal agree.

**Envelope correlation** (T17.4) finds recordings that are *nearly* the same:
overlapping excerpts of one session, or one signal saved twice with a level
change. Nothing a hash can see.

**Cross-dataset** (T17.5) asks whether PASCAL B and CirCor -- both 4 kHz
paediatric stethoscope corpora, collected years apart -- share any material. If
they did, EXP-D1's train-on-one-test-on-the-other design would be measuring
memorisation rather than transfer.

The keep/drop decision (T17.6) is deliberate, not incidental: within a duplicate
group exactly one member is kept, and which one is chosen by an explicit rule
rather than by whatever order the files were scanned in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "DUPLICATE_REPORT_COLUMNS",
    "HEARTBEAT_PREFIX",
    "heartbeat_extra_files",
    "exact_duplicate_groups",
    "content_duplicate_groups",
    "near_duplicate_pairs",
    "cross_dataset_pairs",
    "keep_priority",
    "build_duplicate_report",
    "write_duplicate_report",
    "run_duplicate_detection",
]

log = get_logger(__name__)

HEARTBEAT_PREFIX = "HB_"


def heartbeat_extra_files(root: Path | None = None) -> dict[str, Path]:
    """The 832 ``Heartbeat_Sound/`` files, keyed for the scan's ``extra_files``.

    These are not records -- T17.3 hashes them only to prove they duplicate
    set_a + set_b -- but they still need a stable, unique key. The class folder
    goes into it because the same basename can legitimately appear under two
    class folders, and PASCAL A is already known to file one recording under
    two different labels.

    One definition, used by the audit script and the Phase 17 tests alike. Three
    independent reconstructions of this key produced two different answers once
    already.
    """
    from src.data_loader.pascal import heartbeat_sound_root

    base = root or heartbeat_sound_root()
    return {
        HEARTBEAT_PREFIX + path.parent.name + "_" + path.stem: path
        for path in sorted(base.rglob("*.wav"))
    }

DUPLICATE_REPORT_COLUMNS: tuple[str, ...] = (
    "group_id",
    "method",
    "record_uid",
    "dataset_source",
    "file_path",
    "duplicate_of",
    "similarity",
    "decision",
    "reason",
)


def _audit_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return ensure_dir(out_dir)
    from src.utils.config import load_config

    return ensure_dir(load_config("paths").require("outputs.dataset_audit"))


# ---------------------------------------------------------------------------
# the keep/drop rule (T17.6)
# ---------------------------------------------------------------------------


def keep_priority(record_uid: str, dataset_source: str) -> tuple[int, str]:
    """Sort key deciding which member of a duplicate group is kept.

    Lower sorts first and is kept. The order encodes three judgements:

    1. A ``Heartbeat_Sound/`` entry is never kept. It is a label helper, not a
       recording, and T17.3 requires all 832 to be dropped.
    2. A PhysioNet ``validation/`` copy is never kept over its training twin.
       The training subset is where the record belongs and where its subject
       grouping and annotations attach.
    3. Otherwise the lexicographically smallest ``record_uid``, purely so the
       choice is deterministic across runs -- rule 5.
    """
    if record_uid.startswith(HEARTBEAT_PREFIX) or dataset_source == "heartbeat_sound":
        return (2, record_uid)
    if "_validation_" in record_uid:
        return (1, record_uid)
    return (0, record_uid)


def _group_by(scan: Any, column: str) -> list[list[tuple[str, str, str]]]:
    """Groups of two or more rows sharing a non-empty value in ``column``."""
    buckets: dict[str, list[tuple[str, str, str]]] = {}
    for row in scan.itertuples(index=False):
        key = str(getattr(row, column) or "")
        if not key:
            continue
        buckets.setdefault(key, []).append(
            (row.record_uid, row.dataset_source, row.file_path)
        )
    return [sorted(v, key=lambda m: keep_priority(m[0], m[1]))
            for v in buckets.values() if len(v) > 1]


def exact_duplicate_groups(scan: Any) -> list[list[tuple[str, str, str]]]:
    """Groups sharing a raw-byte SHA-256 (T17.1)."""
    return _group_by(scan, "raw_sha256")


def content_duplicate_groups(scan: Any) -> list[list[tuple[str, str, str]]]:
    """Groups sharing an audio-content hash (T17.2).

    A superset of the byte-identical groups: anything with the same bytes has the
    same content. What matters is the *difference* -- a content group that is not
    also a byte group is a re-encoded duplicate, which is the case a byte hash
    misses entirely.
    """
    return _group_by(scan, "content_sha256")


# ---------------------------------------------------------------------------
# near duplicates (T17.4, T17.5)
# ---------------------------------------------------------------------------


def _correlate(
    left: Any, right: Any, threshold: float, *, same_set: bool, chunk: int = 512
) -> list[tuple[int, int, float]]:
    """Index pairs whose envelope correlation is at or above ``threshold``.

    Envelopes arrive z-scored to unit norm, so the correlation is a dot product
    and the whole comparison is one matrix multiply. Chunked over rows because
    the full 3,541 x 3,541 product is 50 MB and there is no reason to hold it.
    """
    import numpy as np

    pairs: list[tuple[int, int, float]] = []
    for start in range(0, left.shape[0], chunk):
        stop = min(start + chunk, left.shape[0])
        block = left[start:stop] @ right.T
        rows, columns = np.nonzero(block >= threshold)
        for row, column in zip(rows, columns, strict=False):
            i = start + int(row)
            j = int(column)
            if same_set and i >= j:
                continue          # upper triangle only; skips self-pairs too
            # Clipped because this is a correlation and cannot exceed 1. The
            # envelopes are unit-normalised in float32, so an identical pair
            # comes back as 1.000001 -- float error, not information, and a
            # similarity above 1.0 in a published table invites a fair question
            # nobody wants to answer.
            pairs.append((i, j, min(1.0, max(-1.0, float(block[row, column])))))
    return pairs


def near_duplicate_pairs(
    scan: Any,
    envelopes: Any,
    threshold: float,
    *,
    within: str | None = None,
) -> list[tuple[str, str, float]]:
    """Near-duplicate pairs by envelope correlation, within one dataset (T17.4).

    Restricted to one dataset at a time by default because that is what T17.4
    asks for, and because a cross-dataset comparison is a different question with
    a different answer -- see :func:`cross_dataset_pairs`.
    """
    import numpy as np

    mask = (
        np.ones(len(scan), dtype=bool)
        if within is None
        else (scan["dataset_source"] == within).to_numpy()
    )
    indices = np.nonzero(mask)[0]
    if indices.size < 2:
        return []

    subset = envelopes[indices]
    uids = scan["record_uid"].to_numpy()[indices]
    return [
        (str(uids[i]), str(uids[j]), score)
        for i, j, score in _correlate(subset, subset, threshold, same_set=True)
    ]


def cross_dataset_pairs(
    scan: Any,
    envelopes: Any,
    threshold: float,
    left: str,
    right: str,
) -> list[tuple[str, str, float]]:
    """Near-duplicate pairs between two datasets (T17.5)."""
    import numpy as np

    left_index = np.nonzero((scan["dataset_source"] == left).to_numpy())[0]
    right_index = np.nonzero((scan["dataset_source"] == right).to_numpy())[0]
    if left_index.size == 0 or right_index.size == 0:
        return []

    uids = scan["record_uid"].to_numpy()
    return [
        (str(uids[left_index[i]]), str(uids[right_index[j]]), score)
        for i, j, score in _correlate(
            envelopes[left_index], envelopes[right_index], threshold, same_set=False
        )
    ]


# ---------------------------------------------------------------------------
# DA-06 (T17.3, T17.6)
# ---------------------------------------------------------------------------


def build_duplicate_report(
    scan: Any,
    envelopes: Any,
    *,
    threshold: float,
    cross_dataset: tuple[str, str] = ("D3", "D4"),
) -> Any:
    """Assemble **DA-06** with a keep/drop decision per row (T17.6)."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    counter = 0

    byte_groups = exact_duplicate_groups(scan)
    byte_members = {uid for group in byte_groups for uid, _, _ in group}
    for group in byte_groups:
        counter += 1
        group_id = "exact_" + str(counter).zfill(4)
        keeper = group[0]
        for position, (uid, dataset, path) in enumerate(group):
            is_heartbeat = uid.startswith(HEARTBEAT_PREFIX)
            rows.append(
                {
                    "group_id": group_id,
                    "method": "raw_sha256",
                    "record_uid": uid,
                    "dataset_source": dataset,
                    "file_path": path,
                    "duplicate_of": "" if position == 0 else keeper[0],
                    "similarity": 1.0,
                    "decision": "keep" if position == 0 else "drop",
                    "reason": (
                        "byte-identical group; kept by the T17.6 priority rule"
                        if position == 0
                        else (
                            "Heartbeat_Sound is a label helper and a full duplicate "
                            "of set_a + set_b (T17.3)"
                            if is_heartbeat
                            else "byte-identical to " + keeper[0]
                        )
                    ),
                }
            )

    # T17.2 -- only groups the byte hash did NOT already find are informative.
    for group in content_duplicate_groups(scan):
        members = {uid for uid, _, _ in group}
        if members <= byte_members:
            continue
        counter += 1
        group_id = "content_" + str(counter).zfill(4)
        keeper = group[0]
        for position, (uid, dataset, path) in enumerate(group):
            rows.append(
                {
                    "group_id": group_id,
                    "method": "content_sha256",
                    "record_uid": uid,
                    "dataset_source": dataset,
                    "file_path": path,
                    "duplicate_of": "" if position == 0 else keeper[0],
                    "similarity": 1.0,
                    "decision": "keep" if position == 0 else "drop",
                    "reason": (
                        "identical audio content stored differently"
                        if position
                        else "kept by the T17.6 priority rule"
                    ),
                }
            )

    # T17.4 / T17.5 -- near duplicates are reported, never dropped automatically.
    # A 0.98 envelope correlation is a strong hint and not proof; two recordings
    # of the same patient at the same site legitimately look alike.
    #
    # Pairs already resolved by an exact or content group are skipped. Without
    # this the 301 PhysioNet validation copies -- byte-identical, already marked
    # drop above -- reappear here as 301 "review" rows and bury the handful of
    # pairs the envelope layer actually found on its own. A second detector
    # agreeing about a known duplicate is not a new finding.
    resolved = {
        frozenset((row["record_uid"], row["duplicate_of"]))
        for row in rows
        if row["duplicate_of"]
    }
    records = scan[scan["dataset_source"] != "heartbeat_sound"]
    for dataset in sorted(set(records["dataset_source"])):
        for left, right, score in near_duplicate_pairs(
            scan, envelopes, threshold, within=dataset
        ):
            if frozenset((left, right)) in resolved:
                continue
            counter += 1
            rows.append(
                {
                    "group_id": "near_" + str(counter).zfill(4),
                    "method": "envelope_correlation",
                    "record_uid": right,
                    "dataset_source": dataset,
                    "file_path": "",
                    "duplicate_of": left,
                    "similarity": round(score, 6),
                    "decision": "review",
                    "reason": (
                        "envelope correlation >= " + format(threshold, ".2f")
                        + " within " + dataset + "; not proof of duplication"
                    ),
                }
            )

    for left, right, score in cross_dataset_pairs(
        scan, envelopes, threshold, *cross_dataset
    ):
        counter += 1
        rows.append(
            {
                "group_id": "cross_" + str(counter).zfill(4),
                "method": "envelope_correlation_cross_dataset",
                "record_uid": right,
                "dataset_source": cross_dataset[1],
                "file_path": "",
                "duplicate_of": left,
                "similarity": round(score, 6),
                "decision": "review",
                "reason": (
                    "envelope correlation >= " + format(threshold, ".2f")
                    + " between " + cross_dataset[0] + " and " + cross_dataset[1]
                ),
            }
        )

    return pd.DataFrame(rows, columns=list(DUPLICATE_REPORT_COLUMNS))


def write_duplicate_report(report: Any, out_dir: str | Path | None = None) -> Path:
    """Write **DA-06** ``duplicate_report.csv`` (T17.6)."""
    target = _audit_dir(out_dir) / "duplicate_report.csv"
    save_csv(report, target)
    log.info(
        "wrote %s (%d rows: %s)",
        target.name,
        len(report),
        ", ".join(
            key + "=" + str(value)
            for key, value in report["decision"].value_counts().to_dict().items()
        )
        or "none",
    )
    return target


def run_duplicate_detection(
    scan: Any,
    envelopes: Any,
    *,
    threshold: float | None = None,
    write_outputs: bool = False,
    out_dir: str | Path | None = None,
) -> Any:
    """Phase 17 end to end. Returns the DA-06 report."""
    if threshold is None:
        from src.data_loader.integrity import load_thresholds

        threshold = load_thresholds().near_duplicate_correlation

    report = build_duplicate_report(scan, envelopes, threshold=threshold)

    # T17.3 -- stated as an assertion, not a hope. All 832 Heartbeat_Sound files
    # must appear in the report and every one must be marked drop.
    heartbeat = report[report["record_uid"].str.startswith(HEARTBEAT_PREFIX)]
    n_heartbeat = int(
        scan["record_uid"].str.startswith(HEARTBEAT_PREFIX).sum()
    )
    if n_heartbeat:
        if heartbeat["record_uid"].nunique() != n_heartbeat:
            raise ValueError(
                "only " + str(heartbeat["record_uid"].nunique()) + " of "
                + str(n_heartbeat) + " Heartbeat_Sound files were detected as "
                "duplicates -- T17.3 requires all of them"
            )
        not_dropped = heartbeat[heartbeat["decision"] != "drop"]
        if not not_dropped.empty:
            raise ValueError(
                str(len(not_dropped)) + " Heartbeat_Sound file(s) are not marked "
                "drop -- including any of them doubles the PASCAL corpus"
            )

    if write_outputs:
        write_duplicate_report(report, out_dir)
    return report
