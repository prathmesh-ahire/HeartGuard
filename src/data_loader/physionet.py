"""PhysioNet/CinC Challenge 2016 loader (Phases 09-11).

This module turns the ``dataset/archive (3)`` tree into one record table, joins
the online-appendix annotations onto it, and derives subject groups for
leakage-safe cross-validation. Nothing here reads audio samples -- only headers,
reference files and CSVs. Sample-level work belongs to Phase 16 onwards.

Four things about this corpus are not obvious from the source documents, and are
the reason several functions below look defensive rather than simple.

**The validation folder is not extra data.** All 301 of its WAV files are
byte-identical to a same-named file in ``training-a..f``, and their
``REFERENCE.csv`` labels agree. The often-quoted "3,541 records" therefore counts
301 recordings twice. Those rows are still loaded -- the audit has to be able to
show the duplication rather than quietly hide it -- but each carries
``is_duplicate=True``, a ``duplicate_of`` pointer at its training twin, and
``use_in_supervised=False``. The unique corpus is 3,240 records, 2,575 normal /
665 abnormal. See ``Docs/note.md``.

**The validation folder has no .hea files at all**, so header facts there come
from the WAV header itself. ``header_source`` records which was used.

**Only training-a carries a second ECG channel**, and only for 405 of its 409
records -- ``a0041``, ``a0117``, ``a0220`` and ``a0233`` declare one signal, not
two. A flag derived from "is this training-a" would be wrong for those four, so
the flag is read from each header.

**Subject grouping is not what the record names suggest.** The recoverable
grouping lives in the online appendix's ``Original record name`` and
``# Raw record`` columns, not in the challenge record names. Details in
:func:`derive_subject_physionet`.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.constants import map_physionet_reference
from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "HeaderInfo",
    "RecordRef",
    "SubjectDerivation",
    "DATASET_ID",
    "TRAINING_SUBSETS",
    "VALIDATION_SUBSET",
    "ALL_SUBSETS",
    "EXPECTED_RECORD_COUNTS",
    "N_TRAINING_RECORDS",
    "N_ALL_ROWS",
    "N_APPENDIX_ROWS",
    "N_DIAGNOSIS_CLASSES",
    "APPENDIX_COLUMNS",
    "ORDINAL_ANNOTATION_COLUMNS",
    "DIAGNOSIS_MEANING_ALIASES",
    "physionet_root",
    "subset_dir",
    "parse_header",
    "read_wav_header",
    "load_reference",
    "load_reference_sqi",
    "list_records",
    "build_record_table",
    "write_label_conflicts",
    "load_appendix",
    "load_diagnosis_meanings",
    "annotation_code_maps",
    "enrich_with_appendix",
    "write_unannotated_report",
    "derive_subject_physionet",
    "add_subject_ids",
    "write_subject_derivation",
    "load_physionet",
]

log = get_logger(__name__)

DATASET_ID = "D1"

TRAINING_SUBSETS: tuple[str, ...] = (
    "training-a",
    "training-b",
    "training-c",
    "training-d",
    "training-e",
    "training-f",
)
VALIDATION_SUBSET = "validation"
ALL_SUBSETS: tuple[str, ...] = (*TRAINING_SUBSETS, VALIDATION_SUBSET)

# Audited on disk 2026-08-22, re-verified 2026-08-25. A subset that comes back
# with a different count means the tree changed, and every downstream count in
# the write-up is then stale -- so it is an error, not a warning.
EXPECTED_RECORD_COUNTS: dict[str, int] = {
    "training-a": 409,
    "training-b": 490,
    "training-c": 31,
    "training-d": 55,
    "training-e": 2141,
    "training-f": 114,
    VALIDATION_SUBSET: 301,
}

N_TRAINING_RECORDS = sum(EXPECTED_RECORD_COUNTS[s] for s in TRAINING_SUBSETS)   # 3240
N_ALL_ROWS = sum(EXPECTED_RECORD_COUNTS.values())                               # 3541


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def physionet_root() -> Path:
    """Absolute path of ``dataset/archive (3)``, from ``configs/paths.yaml``."""
    from src.utils.config import load_config

    return Path(load_config("paths").require("dataset.d1_physionet.root"))


def subset_dir(subset: str, root: Path | None = None) -> Path:
    """Directory for one subset, validated against :data:`ALL_SUBSETS`."""
    if subset not in ALL_SUBSETS:
        raise ValueError(
            "unknown PhysioNet subset " + repr(subset) + " -- must be one of: "
            + ", ".join(ALL_SUBSETS)
        )
    return (root or physionet_root()) / subset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(path: Path) -> str:
    """Project-relative POSIX path, so the tables stay portable."""
    try:
        return path.resolve().relative_to(_project_root()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def _audit_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return ensure_dir(out_dir)
    from src.utils.config import load_config

    return ensure_dir(load_config("paths").require("outputs.dataset_audit"))


# ---------------------------------------------------------------------------
# headers (T09.2, T09.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeaderInfo:
    """One parsed WFDB ``.hea`` file, or a WAV header standing in for one."""

    record: str
    n_signals: int
    fs_hz: int
    n_samples: int
    signal_files: tuple[str, ...]
    signal_descriptions: tuple[str, ...]
    comment: str
    source: str          # "hea" | "wav"

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs_hz if self.fs_hz else 0.0

    @property
    def has_ecg_channel(self) -> bool:
        return any(d.upper() == "ECG" for d in self.signal_descriptions)

    @property
    def ecg_signal_file(self) -> str:
        for name, desc in zip(self.signal_files, self.signal_descriptions, strict=False):
            if desc.upper() == "ECG":
                return name
        return ""


def parse_header(path: str | Path) -> HeaderInfo:
    """Parse a WFDB ``.hea`` file.

    Format, using ``training-a/a0001.hea`` as the example::

        a0001 2 2000 71332                     <- record, n_signals, fs, n_samples
        a0001.wav 16+44 1 16 0 0 0 0 PCG       <- one line per signal
        a0001.dat 16 1000 16 0 0 367 0 ECG
        # Abnormal                             <- comment carrying the label

    The declared signal count is cross-checked against the number of signal
    lines actually present. A header claiming two signals while listing one is a
    corrupt file, and trusting either number blindly would put a wrong
    ``has_ecg_channel`` into the audit.
    """
    header_path = Path(path)
    lines = [ln.strip() for ln in header_path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError("empty header file: " + str(header_path))

    fields = lines[0].split()
    if len(fields) < 4:
        raise ValueError(
            "malformed header line in " + str(header_path) + ": " + repr(lines[0])
        )
    record = fields[0]
    n_signals = int(fields[1])
    fs_hz = int(float(fields[2]))
    n_samples = int(fields[3])

    signal_files: list[str] = []
    signal_descriptions: list[str] = []
    comments: list[str] = []
    for line in lines[1:]:
        if line.startswith("#"):
            comments.append(line.lstrip("#").strip())
            continue
        parts = line.split()
        signal_files.append(parts[0])
        signal_descriptions.append(parts[-1] if len(parts) > 1 else "")

    if len(signal_files) != n_signals:
        raise ValueError(
            str(header_path) + " declares " + str(n_signals) + " signal(s) but lists "
            + str(len(signal_files)) + " signal line(s)"
        )

    return HeaderInfo(
        record=record,
        n_signals=n_signals,
        fs_hz=fs_hz,
        n_samples=n_samples,
        signal_files=tuple(signal_files),
        signal_descriptions=tuple(signal_descriptions),
        comment=comments[0] if comments else "",
        source="hea",
    )


def read_wav_header(path: str | Path) -> HeaderInfo:
    """Header facts straight from a WAV file, for records with no ``.hea``.

    Every validation record needs this: that directory ships WAV audio and
    ``REFERENCE.csv`` only. No comment line exists there, so ``comment`` is empty
    and the ``.hea``-vs-``REFERENCE`` cross-check (T09.6) has nothing to compare
    -- which is recorded as such, not treated as agreement.
    """
    import soundfile as sf

    wav_path = Path(path)
    info = sf.info(str(wav_path))
    return HeaderInfo(
        record=wav_path.stem,
        n_signals=int(info.channels),
        fs_hz=int(info.samplerate),
        n_samples=int(info.frames),
        signal_files=(wav_path.name,),
        signal_descriptions=("PCG",),
        comment="",
        source="wav",
    )


# ---------------------------------------------------------------------------
# reference files (T09.4, T09.5)
# ---------------------------------------------------------------------------


def load_reference(subset: str, root: Path | None = None) -> dict[str, int]:
    """``REFERENCE.csv`` for one subset as ``{record: -1 | 1}``."""
    path = subset_dir(subset, root) / "REFERENCE.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "no REFERENCE.csv for subset " + subset + ": " + str(path)
        )

    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            record = row[0].strip()
            if record in out:
                raise ValueError(
                    "record " + record + " appears twice in " + str(path)
                    + " -- a duplicated reference row can silently override a label"
                )
            out[record] = int(row[1])
    return out


def load_reference_sqi(
    subset: str, root: Path | None = None
) -> dict[str, dict[str, int]]:
    """``REFERENCE-SQI.csv`` as ``{record: {"reference": -1|1, "sqi": 0|1}}``.

    Three columns: record, the same -1/1 label as ``REFERENCE.csv``, and a
    signal-quality flag (1 = usable, 0 = poor; 2,876 / 364 across the training
    set). The validation subset has no SQI file, so this returns ``{}`` there
    rather than raising -- an absent file is a documented property of that
    subset, not a fault.
    """
    path = subset_dir(subset, root) / "REFERENCE-SQI.csv"
    if not path.is_file():
        return {}

    out: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            out[row[0].strip()] = {
                "reference": int(row[1]) if len(row) > 1 and row[1].strip() else 0,
                "sqi": int(row[2]) if len(row) > 2 and row[2].strip() else 0,
            }
    return out


# ---------------------------------------------------------------------------
# record listing (T09.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordRef:
    """A record located on disk, before any label or header is read."""

    record_id: str
    subset: str
    wav_path: Path
    header_path: Path | None

    @property
    def record_uid(self) -> str:
        # Matches the master-metadata convention of T19.3:
        # {dataset}_{subset}_{record_id}.
        return DATASET_ID + "_" + self.subset + "_" + self.record_id


def list_records(
    subsets: tuple[str, ...] | list[str] | None = None,
    root: Path | None = None,
    *,
    check_counts: bool = True,
) -> list[RecordRef]:
    """Every record in ``subsets`` (default: all seven directories).

    ``RECORDS``, the WAV files on disk and ``REFERENCE.csv`` are three
    independent listings of the same records, and all three agree in this
    corpus. The scan is driven by the WAV files -- the thing that actually has to
    exist -- and the other two are cross-checked against it in
    :func:`build_record_table`.
    """
    base = root or physionet_root()
    if not base.is_dir():
        raise FileNotFoundError(
            "PhysioNet root not found: " + str(base) + " -- check dataset/ is present "
            "and configs/paths.yaml points at it"
        )

    wanted = tuple(subsets) if subsets else ALL_SUBSETS
    out: list[RecordRef] = []
    for subset in wanted:
        directory = subset_dir(subset, base)
        if not directory.is_dir():
            raise FileNotFoundError(
                "missing PhysioNet subset directory: " + str(directory)
            )

        found = sorted(directory.glob("*.wav"), key=lambda p: p.stem)
        if check_counts and subset in EXPECTED_RECORD_COUNTS:
            expected = EXPECTED_RECORD_COUNTS[subset]
            if len(found) != expected:
                raise ValueError(
                    "subset " + subset + " holds " + str(len(found)) + " WAV files but "
                    + str(expected) + " were audited -- the dataset tree has changed, "
                    "so every count in the write-up is stale"
                )

        for wav in found:
            hea = wav.with_suffix(".hea")
            out.append(
                RecordRef(
                    record_id=wav.stem,
                    subset=subset,
                    wav_path=wav,
                    header_path=hea if hea.is_file() else None,
                )
            )
    return out


# ---------------------------------------------------------------------------
# duplicate detection between validation/ and training-*
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_duplicate_map(records: list[RecordRef]) -> dict[str, str]:
    """Map each validation ``record_uid`` to the training ``record_uid`` it copies.

    Compared by content, not by name. A same-named pair that turned out to differ
    would be a far more interesting finding than a duplicate, and shows up here
    as an absent mapping rather than as a false match.
    """
    training = {r.record_id: r for r in records if r.subset in TRAINING_SUBSETS}
    validation = [r for r in records if r.subset == VALIDATION_SUBSET]
    if not validation or not training:
        return {}

    mapping: dict[str, str] = {}
    for ref in validation:
        twin = training.get(ref.record_id)
        if twin is None:
            continue
        if ref.wav_path.stat().st_size != twin.wav_path.stat().st_size:
            continue
        if _sha256(ref.wav_path) == _sha256(twin.wav_path):
            mapping[ref.record_uid] = twin.record_uid
    return mapping


# ---------------------------------------------------------------------------
# the record table (T09.2 - T09.6)
# ---------------------------------------------------------------------------


def _cross_check_header_comment(
    header: HeaderInfo, mapped: dict[str, Any]
) -> tuple[int | None, bool, str]:
    """T09.6 -- compare the ``.hea`` comment with the REFERENCE.csv label."""
    comment = header.comment.strip()
    if header.source != "hea":
        return None, False, "no .hea file in this subset -- nothing to cross-check"
    if not comment:
        return None, True, "header carries no # comment line"

    normalized = comment.lower()
    if normalized not in ("normal", "abnormal"):
        return None, True, "unrecognised header comment: " + repr(comment)

    hea_label = 0 if normalized == "normal" else 1
    if hea_label != mapped["binary_label"]:
        return (
            hea_label,
            True,
            "header says " + comment + " but REFERENCE.csv says "
            + str(mapped["binary_label_name"]),
        )
    return hea_label, False, ""


def build_record_table(
    subsets: tuple[str, ...] | list[str] | None = None,
    root: Path | None = None,
    *,
    check_counts: bool = True,
    detect_duplicates: bool = True,
) -> Any:
    """Build the PhysioNet record table -- one row per record, all seven subsets.

    Covers T09.2 (header parse), T09.3 (ECG-channel flag), T09.4 (reference
    labels, with the both-ways completeness assertion), T09.5 (signal quality)
    and T09.6 (the ``.hea``-comment cross-check).
    """
    import pandas as pd

    base = root or physionet_root()
    records = list_records(subsets, base, check_counts=check_counts)
    duplicates = _validation_duplicate_map(records) if detect_duplicates else {}

    by_subset: dict[str, list[RecordRef]] = {}
    for ref in records:
        by_subset.setdefault(ref.subset, []).append(ref)

    rows: list[dict[str, Any]] = []
    for subset, refs in by_subset.items():
        reference = load_reference(subset, base)
        sqi_table = load_reference_sqi(subset, base)
        on_disk = {r.record_id for r in refs}

        # T09.4 -- both directions. A label with no WAV is as broken as a WAV
        # with no label, and checking one direction hides half the failures.
        missing_label = sorted(on_disk - set(reference))
        missing_file = sorted(set(reference) - on_disk)
        if missing_label:
            raise ValueError(
                subset + ": " + str(len(missing_label)) + " WAV file(s) have no "
                "REFERENCE.csv label, first few: " + ", ".join(missing_label[:5])
            )
        if missing_file:
            raise ValueError(
                subset + ": " + str(len(missing_file)) + " REFERENCE.csv label(s) have "
                "no WAV file, first few: " + ", ".join(missing_file[:5])
            )

        records_index = subset_dir(subset, base) / "RECORDS"
        indexed = {
            ln.strip()
            for ln in records_index.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        if indexed != on_disk:
            raise ValueError(
                subset + ": the RECORDS index and the WAV files on disk disagree ("
                + str(len(indexed ^ on_disk)) + " record(s) differ)"
            )

        for ref in refs:
            header = (
                parse_header(ref.header_path)
                if ref.header_path is not None
                else read_wav_header(ref.wav_path)
            )

            reference_value = reference[ref.record_id]
            mapped = map_physionet_reference(reference_value)
            hea_label, conflict, conflict_reason = _cross_check_header_comment(
                header, mapped
            )

            sqi_row = sqi_table.get(ref.record_id)
            if sqi_row is not None and sqi_row["reference"] != reference_value:
                conflict = True
                conflict_reason = (
                    (conflict_reason + "; " if conflict_reason else "")
                    + "REFERENCE-SQI.csv label " + str(sqi_row["reference"])
                    + " disagrees with REFERENCE.csv " + str(reference_value)
                )

            duplicate_of = duplicates.get(ref.record_uid, "")
            rows.append(
                {
                    "record_uid": ref.record_uid,
                    "dataset_source": DATASET_ID,
                    "subset": subset,
                    "split": "validation" if subset == VALIDATION_SUBSET else "train",
                    "record_id": ref.record_id,
                    "file_path": _relative(ref.wav_path),
                    "header_path": (
                        _relative(ref.header_path) if ref.header_path else ""
                    ),
                    "header_source": header.source,
                    "n_signals": header.n_signals,
                    "original_fs": header.fs_hz,
                    "n_samples": header.n_samples,
                    "duration_sec": round(header.duration_sec, 6),
                    "has_ecg_channel": header.has_ecg_channel,
                    "ecg_signal_file": header.ecg_signal_file,
                    "hea_comment": header.comment.strip(),
                    "hea_label": hea_label,
                    "reference_value": reference_value,
                    "binary_label": mapped["binary_label"],
                    "binary_label_name": mapped["binary_label_name"],
                    "sqi_available": sqi_row is not None,
                    "sqi": sqi_row["sqi"] if sqi_row is not None else None,
                    "label_conflict": conflict,
                    "label_conflict_reason": conflict_reason,
                    "is_duplicate": bool(duplicate_of),
                    "duplicate_of": duplicate_of,
                    "use_in_supervised": not duplicate_of,
                }
            )

    table = pd.DataFrame(rows)
    table = table.sort_values(["subset", "record_id"], kind="stable")
    table = table.reset_index(drop=True)

    if table["record_uid"].duplicated().any():
        raise ValueError("record_uid is not unique in the PhysioNet record table")

    log.info(
        "PhysioNet: %d rows across %d subsets (%d unique recordings, %d duplicated)",
        len(table),
        table["subset"].nunique(),
        int(table["use_in_supervised"].sum()),
        int(table["is_duplicate"].sum()),
    )
    return table


def write_label_conflicts(table: Any, out_dir: str | Path | None = None) -> Path:
    """Write ``physionet_label_conflicts.csv`` (T09.6).

    Written even when empty. An absent file is indistinguishable from a check
    that never ran; a file with a header and no rows is a positive statement that
    the cross-check happened and found nothing.
    """
    target = _audit_dir(out_dir) / "physionet_label_conflicts.csv"
    columns = [
        "record_uid",
        "subset",
        "record_id",
        "header_source",
        "hea_comment",
        "hea_label",
        "reference_value",
        "binary_label",
        "binary_label_name",
        "label_conflict_reason",
    ]
    conflicts = table.loc[table["label_conflict"], columns]
    save_csv(conflicts, target)
    log.info("wrote %s (%d conflict row(s))", target.name, len(conflicts))
    return target


# ===========================================================================
# PHASE 10 -- annotation enrichment (T10.1 - T10.6)
# ===========================================================================

# The appendix annotates 3,153 of the 3,240 training records. The 87 gaps are
# all in training-e and are listed, not dropped (T10.6).
N_APPENDIX_ROWS = 3153

# T10.4 -- twelve diagnosis categories once surrounding whitespace is stripped.
# The raw column contains both "Normal " and "Normal"; without the strip they
# read as two categories and the count comes out at 13.
N_DIAGNOSIS_CLASSES = 12

# T10.2 -- source column -> the name it gets in the record table. Declared
# explicitly because the source headers embed their own code legends and run to
# 400 characters; joining on them by hand is how a typo silently produces an
# all-NaN column.
APPENDIX_COLUMNS: dict[str, str] = {
    "Challenge record name": "record_id",
    "Database": "appendix_database",
    "Original record name": "original_record_name",
    "Diagnosis": "diagnosis",
    "Class (-1=normal 1=abnormal)": "appendix_class",
    "# Beat (automated algorithm)": "n_beats_auto",
    "# Beats requiring hand correction": "n_beats_hand_corrected",
    "Gender": "gender",
    "Age (year)": "age_years",
    "Height (m)": "height_m",
    "Weight (kg)": "weight_kg",
    "BMI": "bmi",
    "Smoker": "smoker",
    "Degree of disease": "degree_of_disease",
    "Subject ID": "native_subject_id",
    "# Raw record": "raw_record",
    "# Recording in each subject": "n_recordings_in_subject",
    "Transducer site on body": "transducer_site",
    "Recording state": "recording_state",
}

# T10.5 -- the coded annotation columns. Key is the short name used in the
# record table; value is the prefix its source column starts with. Matching on a
# prefix rather than the full header keeps this readable, since each of these
# headers carries its entire code legend inline.
#
# `murmur_location` and `abdominal_sounds` are beyond T10.5's list. They cost
# nothing to parse, sit in the same block of columns, and abdominal sounds in
# particular is a noise channel the robustness track (T71.3) can calibrate
# against, so leaving them behind would mean re-reading this file later.
#
# **Every one of these legends starts at 2, and every one of these columns also
# contains 0 in the data** -- 415 rows for murmur location, 26-37 rows for the
# rest, plus a single 1 in two of them. The appendix never says what 0 means.
# From the cross-tabulation it reads as "not applicable" (a murmur location of 0
# always accompanies "Murmurs: None"), but that is an inference, so the code is
# kept and the label is left blank rather than filled in with a guess.
# :func:`load_appendix` logs the counts on every load so the gap stays visible.
ORDINAL_ANNOTATION_COLUMNS: dict[str, str] = {
    "murmur": "Murmurs (",
    "murmur_location": "Murmur Location (",
    "arrhythmia": "Arrhythmia (",
    "respiration_noise": "Respiration noise (",
    "ambient_noise": "Ambient noise (",
    "recording_noise": "Recording noise (",
    "abdominal_sounds": "Abdominal sounds (",
}

# T10.3 -- one alias, and it is a real inconsistency between the two appendix
# files rather than a parsing bug: the training-set CSV labels the training-c
# control group "Controls", while the meanings CSV lists it as "Normal". The raw
# value is preserved in `diagnosis_class`; only the meaning lookup is redirected.
DIAGNOSIS_MEANING_ALIASES: dict[tuple[str, str], str] = {
    ("training-c", "Controls"): "Normal",
}

_CODE_LEGEND = re.compile(r"\(([^()]*)\)\s*$")
_CODE_PAIR = re.compile(r"(\d+)\s*=\s*(.*?)(?=[,;\s]*\b\d+\s*=|$)")


def _appendix_path(root: Path | None = None) -> Path:
    if root is not None:
        return root / "annotations" / "Online_Appendix_training_set.csv"
    from src.utils.config import load_config

    return Path(load_config("paths").require("dataset.d1_physionet.appendix_training"))


def _diagnosis_meanings_path(root: Path | None = None) -> Path:
    if root is not None:
        return root / "annotations" / "Online_Appendix_Diagnosis_meanings.csv"
    from src.utils.config import load_config

    return Path(load_config("paths").require("dataset.d1_physionet.appendix_diagnosis"))


def _find_column(columns: list[str], prefix: str) -> str:
    matches = [c for c in columns if c.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one appendix column starting with " + repr(prefix)
            + ", found " + str(len(matches))
        )
    return matches[0]


def load_appendix(root: Path | None = None) -> Any:
    """Load ``Online_Appendix_training_set.csv`` (T10.1).

    ``utf-8-sig`` is not optional. The file carries a UTF-8 BOM, so reading it as
    plain ``utf-8`` names the first column ``﻿Challenge record name`` -- the
    key column then matches nothing on the join and every annotation arrives as
    NaN, with no error anywhere.

    Returned with this project's column names, plus a ``diagnosis_class`` column
    (T10.4) and one ``<name>_code`` / ``<name>_label`` pair per coded annotation
    column (T10.5).
    """
    import pandas as pd

    path = _appendix_path(root)
    if not path.is_file():
        raise FileNotFoundError("PhysioNet appendix not found: " + str(path))

    raw = pd.read_csv(path, encoding="utf-8-sig")
    source_columns = list(raw.columns)

    missing = [c for c in APPENDIX_COLUMNS if c not in source_columns]
    if missing:
        raise ValueError(
            "the PhysioNet appendix is missing expected column(s): "
            + ", ".join(missing)
        )

    frame = raw[list(APPENDIX_COLUMNS)].rename(columns=APPENDIX_COLUMNS).copy()

    for column in ("record_id", "appendix_database", "original_record_name",
                   "diagnosis", "gender", "transducer_site", "recording_state"):
        frame[column] = frame[column].astype("string").str.strip()

    # T10.4 -- the 12 categories. Whitespace-stripped only; the raw vocabulary is
    # kept as-is so "Normal: NHC" and "Normal: MARS500" stay distinct from
    # "Normal", which is what makes the extended-multiclass track possible.
    frame["diagnosis_class"] = frame["diagnosis"]

    # T10.5 -- codes and their meanings, with the meanings read out of the column
    # header itself rather than transcribed here. A transcription would be one
    # more place for the legend to drift away from the file it describes.
    legends = annotation_code_maps(root)
    for short, prefix in ORDINAL_ANNOTATION_COLUMNS.items():
        source = _find_column(source_columns, prefix)
        codes = pd.to_numeric(raw[source], errors="coerce").astype("Int64")
        frame[short + "_code"] = codes
        frame[short + "_label"] = codes.map(legends[short]).astype("string")

        unmapped = codes.notna() & frame[short + "_label"].isna()
        if unmapped.any():
            log.warning(
                "appendix column %r: %d row(s) carry code(s) %s that its own legend "
                "does not define -- code kept, label left blank rather than guessed",
                short,
                int(unmapped.sum()),
                sorted({int(v) for v in codes[unmapped].dropna().unique()}),
            )

    if len(frame) != N_APPENDIX_ROWS:
        raise ValueError(
            "the PhysioNet appendix holds " + str(len(frame)) + " rows but "
            + str(N_APPENDIX_ROWS) + " were audited"
        )
    if frame["record_id"].duplicated().any():
        raise ValueError("the PhysioNet appendix repeats a challenge record name")

    n_classes = frame["diagnosis_class"].nunique()
    if n_classes != N_DIAGNOSIS_CLASSES:
        raise ValueError(
            "expected " + str(N_DIAGNOSIS_CLASSES) + " diagnosis categories, found "
            + str(n_classes) + ": " + ", ".join(sorted(frame["diagnosis_class"].unique()))
        )
    return frame


def annotation_code_maps(root: Path | None = None) -> dict[str, dict[int, str]]:
    """Code -> meaning for every coded annotation column (T10.5).

    Parsed out of the column headers, which carry their own legend inline::

        Murmurs (2=None 3=Weak 4=Strong 6=Unclear)

    Reading the legend from the file means the mapping cannot drift away from the
    data it describes, and it also catches the awkward cases automatically: the
    murmur-location legend is comma-separated where the rest are space-separated,
    and the noise legends run to 16-18 levels with embedded ``<``, ``>`` and
    ``.`` characters.
    """
    import pandas as pd

    path = _appendix_path(root)
    columns = list(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)

    out: dict[str, dict[int, str]] = {}
    for short, prefix in ORDINAL_ANNOTATION_COLUMNS.items():
        source = _find_column(columns, prefix)
        legend = _CODE_LEGEND.search(source)
        if legend is None:
            raise ValueError("no code legend in appendix column header: " + source)
        pairs = {
            int(code): meaning.strip(" ,;")
            for code, meaning in _CODE_PAIR.findall(legend.group(1))
        }
        if not pairs:
            raise ValueError("empty code legend in appendix column header: " + source)
        out[short] = pairs
    return out


def load_diagnosis_meanings(root: Path | None = None) -> Any:
    """Load ``Online_Appendix_Diagnosis_meanings.csv`` (T10.3).

    Keyed by ``(database, diagnosis)``: the same short code means different
    things in different subsets, and "Normal" appears once per database with its
    own wording. A lookup on the diagnosis alone would collapse those.
    """
    import pandas as pd

    path = _diagnosis_meanings_path(root)
    if not path.is_file():
        raise FileNotFoundError("diagnosis meanings file not found: " + str(path))

    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.rename(
        columns={
            "Database": "appendix_database",
            "Diagnosis": "diagnosis_class",
            "Meaning": "diagnosis_meaning",
        }
    )
    for column in ("appendix_database", "diagnosis_class", "diagnosis_meaning"):
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def enrich_with_appendix(table: Any, root: Path | None = None) -> Any:
    """Join the appendix onto the record table (T10.2, T10.3, T10.6).

    Two joins that are easy to get wrong:

    **Validation rows.** The appendix covers training records only. Each
    validation row is a byte-identical copy of a training record, so it is
    annotated through its ``duplicate_of`` twin rather than left blank -- a blank
    would read as "this record has no annotation", which is not true of any of
    them.

    **The diagnosis meaning.** Joined on ``(database, diagnosis)``, not on the
    diagnosis alone. See :func:`load_diagnosis_meanings`.
    """
    import pandas as pd

    appendix = load_appendix(root)
    meanings = load_diagnosis_meanings(root)

    aliased = meanings.copy()
    annotated = pd.merge(
        appendix,
        aliased,
        on=["appendix_database", "diagnosis_class"],
        how="left",
    )

    # Fill the one documented alias, then insist nothing else is unmapped: an
    # unmapped diagnosis means the two appendix files have drifted apart again,
    # and that is a finding, not something to leave as a blank cell.
    for (database, diagnosis), replacement in DIAGNOSIS_MEANING_ALIASES.items():
        mask = (
            (annotated["appendix_database"] == database)
            & (annotated["diagnosis_class"] == diagnosis)
        )
        lookup = aliased.loc[
            (aliased["appendix_database"] == database)
            & (aliased["diagnosis_class"] == replacement),
            "diagnosis_meaning",
        ]
        if mask.any() and not lookup.empty:
            annotated.loc[mask, "diagnosis_meaning"] = lookup.iloc[0]

    unmapped = annotated.loc[annotated["diagnosis_meaning"].isna(), "diagnosis_class"]
    if not unmapped.empty:
        raise ValueError(
            "diagnosis code(s) with no meaning in the meanings file: "
            + ", ".join(sorted(set(unmapped.dropna())))
        )

    enriched = table.copy()

    # The join key. Challenge record names are globally unique across subsets
    # (a*, b*, ... f*), but the key is still built from the record a row actually
    # refers to, so a validation row resolves through its training twin.
    twin_record = (
        enriched["duplicate_of"]
        .map(dict(zip(enriched["record_uid"], enriched["record_id"], strict=False)))
    )
    enriched["_join_record"] = enriched["record_id"].where(
        ~enriched["is_duplicate"], twin_record
    )

    merged = pd.merge(
        enriched,
        annotated.rename(columns={"record_id": "_join_record"}),
        on="_join_record",
        how="left",
        validate="many_to_one",
    )
    merged["appendix_matched"] = merged["appendix_database"].notna()
    merged = merged.drop(columns=["_join_record"])

    # The appendix carries its own copy of the -1/1 class. It agrees with
    # REFERENCE.csv for all 3,153 annotated records today; if that ever stops
    # being true it is a label-provenance problem, so it is checked rather than
    # assumed.
    matched = merged[merged["appendix_matched"]]
    disagreement = matched[matched["appendix_class"] != matched["reference_value"]]
    if not disagreement.empty:
        raise ValueError(
            str(len(disagreement)) + " record(s) where the appendix Class column "
            "disagrees with REFERENCE.csv, first few: "
            + ", ".join(disagreement["record_uid"].head(5))
        )

    training = merged[merged["subset"].isin(TRAINING_SUBSETS)]
    n_matched = int(training["appendix_matched"].sum())
    log.info(
        "PhysioNet appendix: %d of %d training records annotated (%d unmatched)",
        n_matched,
        len(training),
        len(training) - n_matched,
    )
    return merged


def write_unannotated_report(table: Any, out_dir: str | Path | None = None) -> Path:
    """Write ``physionet_unannotated.csv`` (T10.6).

    The 87 training-e records with no appendix row. They are listed here and kept
    in the record table with their REFERENCE.csv label intact -- they are usable
    for the binary task and only unusable for anything that needs the appendix,
    which is a different statement from "dropped".
    """
    target = _audit_dir(out_dir) / "physionet_unannotated.csv"
    columns = [
        "record_uid",
        "subset",
        "record_id",
        "file_path",
        "binary_label",
        "binary_label_name",
        "duration_sec",
    ]
    unmatched = table.loc[
        table["subset"].isin(TRAINING_SUBSETS) & ~table["appendix_matched"], columns
    ]
    save_csv(unmatched, target)
    log.info("wrote %s (%d unannotated record(s))", target.name, len(unmatched))
    return target


# ===========================================================================
# PHASE 11 -- subject derivation (T11.1 - T11.6)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class SubjectDerivation:
    """One record's subject group, and how confident we are in it."""

    subject_id: str
    pattern: str
    subject_derived: bool


# T11.1 - T11.3. These match the appendix's `Original record name`, NOT the
# challenge record name: `a0001` carries no subject information, its original
# name `C45S1` does. Deliberately not anchored at the end -- four training-a
# records are `C16S0b`, `C12S3b`, `C16S2b`, `C13S1b` and four training-b records
# are `S34f2.e4k_data` and friends. An anchored regex misses all eight, and each
# miss invents a singleton subject for a person whose other recordings are
# already in the corpus, which is exactly the leak grouping exists to prevent.
_PATTERN_A = re.compile(r"^C(\d+)S(\d+)", re.IGNORECASE)
_PATTERN_B = re.compile(r"^S(\d+)f(\d+)", re.IGNORECASE)
_PATTERN_C = re.compile(r"^id(\d+)", re.IGNORECASE)

# T11.4. training-e is the subset where the written plan and the data disagree.
# See derive_subject_physionet.
_PATTERN_E_NORMAL = re.compile(r"^e\d+$", re.IGNORECASE)
_PATTERN_E_CAD = re.compile(r"^\d+$")

_SUBSET_LETTER = {
    "training-a": "a",
    "training-b": "b",
    "training-c": "c",
    "training-d": "d",
    "training-e": "e",
    "training-f": "f",
    VALIDATION_SUBSET: "v",
}


def derive_subject_physionet(
    record_id: str,
    subset: str,
    original_record_name: str | None = None,
    raw_record: object = None,
) -> SubjectDerivation:
    """Derive one record's subject group (T11.1 - T11.5).

    ==============  ====================================  =========  =========
    Subset          Basis                                 Groups     Derived
    ==============  ====================================  =========  =========
    training-a      original name ``C<subj>S<sess>``      60         yes
    training-b      original name ``S<subj>f<n>_data``    106         yes
    training-c      original name ``id<subj>``            31         yes
    training-e      appendix ``# Raw record`` + cohort    404 + 87   mixed
    training-d      none recoverable                      55         no
    training-f      none recoverable                      114        no
    ==============  ====================================  =========  =========

    **training-e is a deliberate deviation from T11.4, agreed 2026-08-25.** The
    task text says to treat each training-e record as its own subject. The
    appendix says otherwise: its 2,054 annotated training-e records carry only
    284 distinct ``# Raw record`` values, a median of 8 recordings each and up to
    16. Following the task text would put eight slices of one person's recording
    on both sides of a fold, in the subset that is 66% of all PhysioNet data.

    **And the raw-record numbering collides between two cohorts.** The 1,871
    normal records have original names like ``e01430``; the 183 CAD records have
    purely numeric original names equal to their own raw-record number. Both
    number from 1, so raw record 14 is one person in the normal cohort and a
    different person in the CAD cohort. Keying on the bare number merges them --
    and since the cohorts are label-pure, that would merge a normal subject with
    an abnormal one. The cohort is therefore part of the key, and is read from
    the shape of the original name, not from the label.

    The 87 training-e records with no appendix row get their own group with
    ``subject_derived=False``.
    """
    original = (original_record_name or "").strip()
    letter = _SUBSET_LETTER.get(subset, subset)
    fallback = SubjectDerivation(
        subject_id=letter + "_rec_" + record_id,
        pattern=subset + ": no recoverable pattern -- record is its own group",
        subject_derived=False,
    )

    if subset == "training-a":
        match = _PATTERN_A.match(original)
        if match:
            return SubjectDerivation(
                subject_id="a_C" + str(int(match.group(1))),
                pattern="training-a: original record name C<subject>S<session>",
                subject_derived=True,
            )
        return fallback

    if subset == "training-b":
        match = _PATTERN_B.match(original)
        if match:
            return SubjectDerivation(
                subject_id="b_S" + str(int(match.group(1))),
                pattern="training-b: original record name S<subject>f<file>_data",
                subject_derived=True,
            )
        return fallback

    if subset == "training-c":
        match = _PATTERN_C.match(original)
        if match:
            return SubjectDerivation(
                subject_id="c_id" + str(int(match.group(1))),
                pattern="training-c: original record name id<subject>",
                subject_derived=True,
            )
        return fallback

    if subset == "training-e":
        number = _as_int(raw_record)
        if number is None or not original:
            return SubjectDerivation(
                subject_id="e_rec_" + record_id,
                pattern="training-e: no appendix row -- record is its own group",
                subject_derived=False,
            )
        if _PATTERN_E_NORMAL.match(original):
            cohort = "n"
        elif _PATTERN_E_CAD.match(original):
            cohort = "p"
        else:
            return SubjectDerivation(
                subject_id="e_rec_" + record_id,
                pattern=(
                    "training-e: original record name " + repr(original)
                    + " matches neither cohort -- record is its own group"
                ),
                subject_derived=False,
            )
        return SubjectDerivation(
            subject_id="e_" + cohort + "R" + str(number),
            pattern="training-e: appendix # Raw record, namespaced by cohort",
            subject_derived=True,
        )

    return fallback


def _as_int(value: object) -> int | None:
    """Coerce an appendix numeric cell to ``int``, or ``None`` if it is blank.

    The appendix's numeric columns arrive as float64 because they contain NaN, so
    ``# Raw record`` 14 reads as ``14.0``. Formatting that into a subject id
    would produce ``e_nR14.0``, which is a different string from ``e_nR14`` and
    would split one subject in two if any code path ever produced the other form.
    """
    if value is None:
        return None
    try:
        import math

        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return int(number)


def add_subject_ids(table: Any) -> Any:
    """Add ``subject_id``, ``subject_derived`` and ``subject_pattern`` (T11.1-T11.5).

    Requires the appendix columns, so run :func:`enrich_with_appendix` first.

    Validation rows inherit the subject of the training record they duplicate.
    They are excluded from supervised use anyway, but a subject id that differs
    from the twin's would make them look like new people in any table that counts
    subjects -- and would quietly defeat the grouping if they were ever included.
    """
    out = table.copy()
    if "original_record_name" not in out.columns:
        raise ValueError(
            "add_subject_ids needs the appendix columns -- call "
            "enrich_with_appendix(table) first"
        )

    derivations = [
        derive_subject_physionet(
            record_id=row.record_id,
            subset=row.subset,
            original_record_name=(
                None if _is_missing(row.original_record_name)
                else str(row.original_record_name)
            ),
            raw_record=row.raw_record,
        )
        for row in out.itertuples(index=False)
    ]
    out["subject_id"] = [d.subject_id for d in derivations]
    out["subject_derived"] = [d.subject_derived for d in derivations]
    out["subject_pattern"] = [d.pattern for d in derivations]

    if out["is_duplicate"].any():
        by_uid = dict(zip(out["record_uid"], out.index, strict=False))
        for position in out.index[out["is_duplicate"]]:
            twin = by_uid.get(out.at[position, "duplicate_of"])
            if twin is None:
                continue
            out.at[position, "subject_id"] = out.at[twin, "subject_id"]
            out.at[position, "subject_derived"] = out.at[twin, "subject_derived"]
            out.at[position, "subject_pattern"] = (
                "validation: inherited from the duplicated training record "
                + str(out.at[twin, "record_uid"])
            )

    if out["subject_id"].isna().any() or (out["subject_id"] == "").any():
        raise ValueError("some PhysioNet records ended up with no subject_id")

    _verify_training_b_against_native(out)

    supervised = out[out["use_in_supervised"]]
    log.info(
        "PhysioNet subjects: %d groups over %d records (%d records with a derived "
        "subject, %d record-level fallbacks)",
        supervised["subject_id"].nunique(),
        len(supervised),
        int(supervised["subject_derived"].sum()),
        int((~supervised["subject_derived"]).sum()),
    )
    return out


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        import math

        return isinstance(value, float) and math.isnan(value)
    except (TypeError, ValueError):
        return False


def _verify_training_b_against_native(table: Any) -> None:
    """T11.2 -- the derived training-b subjects must equal the native ones.

    training-b is the only subset where the appendix supplies a real ``Subject
    ID`` column (490 of 3,153 rows). That makes it the one place where a derived
    grouping can be checked against ground truth rather than argued for, so it is
    checked on every load: if ``S34f2.e4k_data`` ever stopped resolving to
    subject 34, the same silent failure would be happening unobserved in
    training-a and training-c, which have no native column to compare against.
    """
    rows = table[
        (table["subset"] == "training-b") & table["native_subject_id"].notna()
    ]
    if rows.empty:
        return

    expected = ["b_S" + str(_as_int(value)) for value in rows["native_subject_id"]]
    mismatched = [
        uid
        for uid, derived, native in zip(
            rows["record_uid"], rows["subject_id"], expected, strict=False
        )
        if derived != native
    ]
    if mismatched:
        raise ValueError(
            str(len(mismatched)) + " training-b record(s) whose derived subject "
            "disagrees with the appendix Subject ID column, first few: "
            + ", ".join(mismatched[:5])
        )


def write_subject_derivation(table: Any, out_dir: str | Path | None = None) -> Path:
    """Write ``physionet_subject_derivation.csv`` (T11.6)."""
    target = _audit_dir(out_dir) / "physionet_subject_derivation.csv"
    columns = [
        "record_uid",
        "subset",
        "record_id",
        "original_record_name",
        "native_subject_id",
        "raw_record",
        "subject_pattern",
        "subject_id",
        "subject_derived",
        "use_in_supervised",
    ]
    save_csv(table[columns], target)
    log.info(
        "wrote %s (%d rows, %d distinct subjects)",
        target.name,
        len(table),
        table["subject_id"].nunique(),
    )
    return target


# ---------------------------------------------------------------------------
# one call for the whole dataset
# ---------------------------------------------------------------------------


def load_physionet(
    root: Path | None = None,
    *,
    subsets: tuple[str, ...] | list[str] | None = None,
    write_outputs: bool = False,
    out_dir: str | Path | None = None,
) -> Any:
    """Record table, appendix annotations and subject groups in one call.

    ``write_outputs=True`` also emits the three Part II audit files:
    ``physionet_label_conflicts.csv`` (T09.6), ``physionet_unannotated.csv``
    (T10.6) and ``physionet_subject_derivation.csv`` (T11.6).
    """
    table = build_record_table(subsets, root)
    table = enrich_with_appendix(table, root)
    table = add_subject_ids(table)

    if write_outputs:
        write_label_conflicts(table, out_dir)
        write_unannotated_report(table, out_dir)
        write_subject_derivation(table, out_dir)
    return table
