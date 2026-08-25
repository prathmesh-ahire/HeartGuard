"""PASCAL Classifying Heart Sounds Challenge loader, sets A and B (Phases 12-13).

set_a is **D2 / task ``pascal_a``** (4 classes) and set_b is **D3 / task
``pascal_b``** (3 classes). They are two separate tasks and are never merged --
rule 4. ``extrahls`` (set_a) and ``extrastole`` (set_b) are different phenomena
that happen to share the code 2, and this module never puts them in one column.

Five things about this corpus will waste a session each if they are not known up
front.

**Labels come from folder membership, not from the CSVs.** ``Heartbeat_Sound/``
holds the same 832 recordings re-foldered by class, and that foldering is the
authoritative label index (T12.2). It is a **label helper only** -- it is a 100%
duplicate of set_a + set_b, so including it as a training source doubles the
PASCAL data and puts the same recording in both train and test.

**Neither CSV's filenames match what is on disk, and they fail differently.**
set_b.csv prefixes every row with ``Btraining_`` and doubles that prefix on the
noisy files (``Btraining_normal_Btraining_noisynormal_125_...``); zero of its 656
rows match a filename literally. set_a.csv is fine for its 124 labelled rows but
drops the ``Aunlabelledtest`` prefix entirely from its 52 unlabelled ones,
listing them as ``set_a/__201012172010.wav``. Both are resolved by
:func:`normalize_pascal_name`.

**149 set_b files carry a noise qualifier that breaks the obvious regex.**
``normal_noisynormal_125_...`` and ``murmur_noisymurmur_162_...`` use single
underscores where every other file uses a double. A ``<label>__(\\d+)_...``
pattern matches 507 of 656 and drops the other 149 into whatever fallback exists.
This is the "~149-record outlier group" of T13.3: it was never a subject, it was
a fallback bucket. See :func:`parse_set_b_name`.

**set_b's subject count is 165, not the 167 in T13.1.** 167 counts (subject,
timestamp) pairs -- recording sessions. Three subjects were recorded twice on one
day and two of them have both sessions labelled, so session-level grouping splits
two people across folds. See :func:`derive_subject_pascal`.

**set_a_timing.csv's ``location`` column is a sample index, not a body site.**
Nothing to do with the ``recording_location`` of T13.2, which is set_b's
auscultation site. See :func:`load_set_a_timing`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.constants import (
    TASK_PASCAL_A,
    TASK_PASCAL_B,
    UNLABELED_CLASS,
    encode_label,
    label_names,
)
from src.utils.io import ensure_dir, save_csv
from src.utils.logging_setup import get_logger

__all__ = [
    "PascalRecord",
    "SetBName",
    "SubjectDerivation",
    "SET_A",
    "SET_B",
    "PASCAL_SETS",
    "SET_DATASET_ID",
    "SET_TASK",
    "UNLABELLED_MARKER",
    "EXPECTED_CLASS_COUNTS",
    "EXPECTED_LABELLED_TOTALS",
    "EXPECTED_UNLABELLED_COUNTS",
    "N_SET_B_SUBJECTS",
    "N_SET_B_SESSIONS",
    "N_NOISY_SET_B_FILES",
    "TIMING_FS",
    "pascal_root",
    "set_dir",
    "heartbeat_sound_root",
    "normalize_pascal_name",
    "parse_set_b_name",
    "load_label_index",
    "load_set_csv",
    "list_records",
    "build_record_table",
    "write_label_conflicts",
    "derive_subject_pascal",
    "add_subject_ids",
    "load_set_a_timing",
    "write_subject_derivation",
    "load_pascal",
]

log = get_logger(__name__)

SET_A = "set_a"
SET_B = "set_b"
PASCAL_SETS: tuple[str, ...] = (SET_A, SET_B)

SET_DATASET_ID: dict[str, str] = {SET_A: "D2", SET_B: "D3"}
SET_TASK: dict[str, str] = {SET_A: TASK_PASCAL_A, SET_B: TASK_PASCAL_B}

# The prefix each set uses for its unlabelled files on disk. set_a.csv omits it,
# which is the whole reason normalize_pascal_name needs to know the set.
UNLABELLED_MARKER: dict[str, str] = {
    SET_A: "Aunlabelledtest",
    SET_B: "Bunlabelledtest",
}

# T12.6 -- audited on disk 2026-08-22, re-verified 2026-08-25 against the
# Heartbeat_Sound foldering. These are asserted, not logged: a PASCAL count that
# has drifted means the label index changed underneath every downstream result.
EXPECTED_CLASS_COUNTS: dict[str, dict[str, int]] = {
    SET_A: {"artifact": 40, "extrahls": 19, "murmur": 34, "normal": 31},
    SET_B: {"extrastole": 46, "murmur": 95, "normal": 320},
}
EXPECTED_LABELLED_TOTALS: dict[str, int] = {SET_A: 124, SET_B: 461}
EXPECTED_UNLABELLED_COUNTS: dict[str, int] = {SET_A: 52, SET_B: 195}

# T13.1 / T13.3 -- see the module docstring.
N_SET_B_SUBJECTS = 165        # distinct people among the 461 labelled records
N_SET_B_SESSIONS = 167        # distinct (subject, timestamp) pairs -- NOT people
N_NOISY_SET_B_FILES = 149     # 120 noisynormal + 29 noisymurmur

# set_a is recorded at 44.1 kHz and set_a_timing.csv indexes samples at that
# rate, not at the 2 kHz pipeline target.
TIMING_FS = 44100


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def pascal_root() -> Path:
    """Absolute path of ``dataset/archive (2)``."""
    from src.utils.config import load_config

    return Path(load_config("paths").require("dataset.d2_pascal_a.root")).parent


def set_dir(dataset: str, root: Path | None = None) -> Path:
    """Directory for ``set_a`` or ``set_b``."""
    if dataset not in PASCAL_SETS:
        raise ValueError(
            "unknown PASCAL set " + repr(dataset) + " -- must be one of: "
            + ", ".join(PASCAL_SETS)
        )
    return (root or pascal_root()) / dataset


def heartbeat_sound_root() -> Path:
    """Absolute path of ``dataset/Heartbeat_Sound`` -- the label index only.

    Never a training source. It is a byte-for-byte duplicate of set_a + set_b,
    and a glob wide enough to pick it up doubles the PASCAL corpus silently.
    """
    from src.utils.config import load_config

    return Path(load_config("paths").require("dataset.heartbeat_sound.root"))


def _csv_path(dataset: str, root: Path | None = None) -> Path:
    base = root or pascal_root()
    return base / (dataset + ".csv")


def _timing_path(root: Path | None = None) -> Path:
    return (root or pascal_root()) / "set_a_timing.csv"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(path: Path) -> str:
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
# filename normalisation (T12.3)
# ---------------------------------------------------------------------------

_BTRAINING = re.compile(r"Btraining_")
_UNDERSCORES = re.compile(r"_+")


def normalize_pascal_name(name: str, dataset: str | None = None) -> str:
    """Canonical key for one PASCAL recording, from any of its three spellings.

    The same recording is written three different ways across this dataset::

        set_b/Btraining_extrastole_127_1306764300147_C2.wav          (set_b.csv)
        extrastole__127_1306764300147_C2.wav                         (on disk)
        extrastole__127_1306764300147_C2.wav                    (Heartbeat_Sound)

    and, for the noisy files, a fourth::

        Btraining_normal_Btraining_noisynormal_125_1306332456645_C.wav  (CSV)
        normal_noisynormal_125_1306332456645_C.wav                      (disk)

    Two rules collapse all of them onto one key: drop every ``Btraining_`` token,
    then squeeze runs of ``_`` to one. That resolves 656 of 656 set_b rows.

    ``dataset`` is needed for one case only. set_a.csv writes its 52 unlabelled
    rows as ``set_a/__201012172010.wav`` -- the ``Aunlabelledtest`` prefix is
    simply missing, so the key would begin with a separator and match nothing.
    A leading separator therefore means "unlabelled", and the set's marker is put
    back. Without ``dataset`` those 52 rows raise rather than silently failing to
    join, because an unresolved filename that returns *something* is how a label
    ends up attached to the wrong recording.
    """
    stem = name.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith(".wav"):
        stem = stem[:-4]

    stem = _BTRAINING.sub("", stem)
    stem = _UNDERSCORES.sub("_", stem)

    if stem.startswith("_"):
        if dataset is None:
            raise ValueError(
                "cannot normalise " + repr(name) + ": its label prefix is missing "
                "(set_a.csv does this for unlabelled rows) and no dataset was given "
                "to say which unlabelled marker to restore"
            )
        if dataset not in UNLABELLED_MARKER:
            raise ValueError("unknown PASCAL set " + repr(dataset))
        stem = UNLABELLED_MARKER[dataset] + stem

    return stem


# ---------------------------------------------------------------------------
# set_b filename structure (T13.1, T13.2, T13.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SetBName:
    """The four fields encoded in a set_b filename."""

    label_component: str      # "normal", "normal_noisynormal", "Bunlabelledtest"
    noise_qualifier: str      # "noisynormal" | "noisymurmur" | ""
    subject: str              # the numeric patient id
    timestamp: str            # epoch milliseconds -- the recording session
    location_raw: str         # "A", "B1", "D31", "B_1", ...
    location: str             # the auscultation site letter, "A".."F"
    location_repeat: str      # the trailing repeat index, "" when absent


# Deliberately tolerant of the noise qualifier and of a single OR double
# separator. The strict `^[A-Za-z]+__(\d+)_(\d+)_(\w+)$` reading matches 507 of
# 656 files; the 149 it misses are the `_noisy` ones, and they are the whole of
# T13.3's phantom outlier group.
_SET_B_NAME = re.compile(
    r"^(?P<label>[A-Za-z]+?)(?:_(?P<noise>noisy[a-z]+))?"
    r"_(?P<subject>\d+)_(?P<timestamp>\d+)_(?P<location>.+)$"
)
_LOCATION = re.compile(r"^(?P<letter>[A-Za-z])_?(?P<repeat>\d*)$")


def parse_set_b_name(name: str) -> SetBName:
    """Split a set_b filename into subject, session, location and noise flag."""
    stem = normalize_pascal_name(name, SET_B)
    match = _SET_B_NAME.match(stem)
    if match is None:
        raise ValueError(
            "set_b filename does not parse: " + repr(name) + " (canonical form "
            + repr(stem) + ")"
        )

    raw_location = match.group("location")
    site = _LOCATION.match(raw_location)
    if site is None:
        # Never silently blank: an unparsed suffix must be visible, because
        # `recording_location` feeds the per-site analysis in Phase 70.
        raise ValueError(
            "set_b location suffix does not parse: " + repr(raw_location)
            + " in " + repr(name)
        )

    noise = match.group("noise") or ""
    label = match.group("label")
    return SetBName(
        label_component=label + ("_" + noise if noise else ""),
        noise_qualifier=noise,
        subject=match.group("subject"),
        timestamp=match.group("timestamp"),
        location_raw=raw_location,
        location=site.group("letter").upper(),
        location_repeat=site.group("repeat"),
    )


# ---------------------------------------------------------------------------
# the authoritative label index (T12.2)
# ---------------------------------------------------------------------------


def load_label_index(root: Path | None = None) -> dict[str, str]:
    """``{canonical key: class folder}`` from ``Heartbeat_Sound/`` (T12.2).

    832 files across six folders, one of which (``unlabel``) is not a class.
    Verified here rather than trusted: a filename appearing under two class
    folders would be an ambiguous label, and the total is asserted so a partial
    copy of the folder cannot pass as the full index.
    """
    base = root or heartbeat_sound_root()
    if not base.is_dir():
        raise FileNotFoundError(
            "Heartbeat_Sound/ not found: " + str(base) + " -- it is the "
            "authoritative PASCAL label source, so nothing can be labelled without it"
        )

    index: dict[str, str] = {}
    seen: dict[str, list[str]] = {}
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        for wav in folder.glob("*.wav"):
            key = normalize_pascal_name(wav.name)
            seen.setdefault(key, []).append(folder.name)
            index[key] = folder.name

    ambiguous = {k: v for k, v in seen.items() if len(set(v)) > 1}
    if ambiguous:
        raise ValueError(
            str(len(ambiguous)) + " Heartbeat_Sound file(s) appear under more than "
            "one class folder, first few: " + ", ".join(sorted(ambiguous)[:5])
        )

    total = sum(len(v) for v in seen.values())
    if total != 832:
        raise ValueError(
            "Heartbeat_Sound/ holds " + str(total) + " files but 832 were audited"
        )
    return index


# ---------------------------------------------------------------------------
# the challenge CSVs (T12.4)
# ---------------------------------------------------------------------------


def load_set_csv(dataset: str, root: Path | None = None) -> Any:
    """``set_a.csv`` / ``set_b.csv`` with a canonical key column.

    A cross-check only. The labels used everywhere else come from
    :func:`load_label_index`; these rows exist so a disagreement can be reported
    (T12.4) instead of one source being picked on faith.
    """
    import pandas as pd

    path = _csv_path(dataset, root)
    if not path.is_file():
        raise FileNotFoundError("PASCAL CSV not found: " + str(path))

    frame = pd.read_csv(path)
    frame["canonical_key"] = [
        normalize_pascal_name(name, dataset) for name in frame["fname"]
    ]
    frame["csv_label"] = (
        frame["label"].astype("string").str.strip().fillna(UNLABELED_CLASS)
    )
    frame["csv_sublabel"] = frame["sublabel"].astype("string").str.strip().fillna("")

    if frame["canonical_key"].duplicated().any():
        raise ValueError(
            dataset + ".csv normalises two different rows onto the same key -- the "
            "normalizer is losing information"
        )
    return frame[["canonical_key", "fname", "csv_label", "csv_sublabel"]]


# ---------------------------------------------------------------------------
# record listing (T12.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PascalRecord:
    """A PASCAL recording located on disk, before labelling."""

    record_id: str
    dataset: str
    wav_path: Path

    @property
    def canonical_key(self) -> str:
        return normalize_pascal_name(self.wav_path.name, self.dataset)

    @property
    def record_uid(self) -> str:
        return SET_DATASET_ID[self.dataset] + "_" + self.dataset + "_" + self.record_id


def list_records(
    dataset: str | None = None, root: Path | None = None
) -> list[PascalRecord]:
    """Every WAV in ``set_a`` and/or ``set_b`` (T12.1).

    Scans the two challenge directories only. ``Heartbeat_Sound/`` is not
    scanned here by design -- it is read for labels and nothing else.
    """
    base = root or pascal_root()
    wanted = (dataset,) if dataset else PASCAL_SETS

    out: list[PascalRecord] = []
    for name in wanted:
        directory = set_dir(str(name), base)
        if not directory.is_dir():
            raise FileNotFoundError("missing PASCAL directory: " + str(directory))
        for wav in sorted(directory.glob("*.wav"), key=lambda p: p.stem):
            out.append(
                PascalRecord(record_id=wav.stem, dataset=str(name), wav_path=wav)
            )
    return out


# ---------------------------------------------------------------------------
# the record table (T12.4 - T12.6)
# ---------------------------------------------------------------------------


def build_record_table(
    dataset: str | None = None,
    root: Path | None = None,
    *,
    check_counts: bool = True,
) -> Any:
    """Build the PASCAL record table -- one row per recording in set_a + set_b.

    Covers T12.4 (folder-versus-CSV cross-check), T12.5 (the set split and the
    unlabelled exclusion) and T12.6 (the exact-count assertion).
    """
    import pandas as pd
    import soundfile as sf

    base = root or pascal_root()
    records = list_records(dataset, base)
    index = load_label_index()

    csv_by_set = {
        name: load_set_csv(name, base).set_index("canonical_key")
        for name in ({r.dataset for r in records})
    }

    rows: list[dict[str, Any]] = []
    for record in records:
        key = record.canonical_key
        class_folder = index.get(key)
        if class_folder is None:
            raise ValueError(
                "no Heartbeat_Sound entry for " + record.wav_path.name
                + " (canonical key " + repr(key) + ") -- the label index is "
                "incomplete, so this recording cannot be labelled"
            )

        task = SET_TASK[record.dataset]
        is_unlabeled = class_folder == UNLABELED_CLASS

        # Rule 4, enforced per record: a set_a folder name must be in the
        # pascal_a vocabulary and a set_b one in pascal_b. `extrahls` turning up
        # in set_b, or `extrastole` in set_a, would mean the two label spaces
        # had been crossed.
        if not is_unlabeled and class_folder not in label_names(task):
            raise ValueError(
                "class folder " + repr(class_folder) + " is not in the "
                + task + " vocabulary (" + ", ".join(label_names(task)) + ") -- "
                "found on " + record.wav_path.name
            )

        csv_frame = csv_by_set[record.dataset]
        csv_label = ""
        csv_sublabel = ""
        conflict = False
        reason = ""
        if key in csv_frame.index:
            csv_label = str(csv_frame.at[key, "csv_label"])
            csv_sublabel = str(csv_frame.at[key, "csv_sublabel"])
            if csv_label != class_folder:
                conflict = True
                reason = (
                    "Heartbeat_Sound says " + class_folder + " but "
                    + record.dataset + ".csv says " + csv_label
                )
        else:
            conflict = True
            reason = record.dataset + ".csv has no row for this recording"

        info = sf.info(str(record.wav_path))
        noise_qualifier = ""
        if record.dataset == SET_B:
            noise_qualifier = parse_set_b_name(record.wav_path.name).noise_qualifier

        rows.append(
            {
                "record_uid": record.record_uid,
                "dataset_source": SET_DATASET_ID[record.dataset],
                "subset": record.dataset,
                "record_id": record.record_id,
                "canonical_key": key,
                "file_path": _relative(record.wav_path),
                "task": task,
                "class_folder": class_folder,
                "label_source": "Heartbeat_Sound/" + class_folder,
                "multiclass_label": (
                    None if is_unlabeled else encode_label(task, class_folder)
                ),
                "multiclass_label_name": "" if is_unlabeled else class_folder,
                "csv_label": csv_label,
                "csv_sublabel": csv_sublabel,
                "noise_qualifier": noise_qualifier,
                "label_conflict": conflict,
                "label_conflict_reason": reason,
                "is_unlabeled": is_unlabeled,
                "use_in_supervised": not is_unlabeled,
                "original_fs": int(info.samplerate),
                "n_samples": int(info.frames),
                "n_channels": int(info.channels),
                "duration_sec": round(info.frames / info.samplerate, 6),
            }
        )

    table = pd.DataFrame(rows)
    table = table.sort_values(["subset", "record_id"], kind="stable")
    table = table.reset_index(drop=True)

    if table["record_uid"].duplicated().any():
        raise ValueError("record_uid is not unique in the PASCAL record table")

    if check_counts:
        _assert_counts(table, dataset)

    log.info(
        "PASCAL: %d recordings (%d labelled, %d unlabelled) across %d set(s)",
        len(table),
        int(table["use_in_supervised"].sum()),
        int(table["is_unlabeled"].sum()),
        table["subset"].nunique(),
    )
    return table


def _assert_counts(table: Any, dataset: str | None) -> None:
    """T12.6 -- fail loudly, per class, not just on the totals.

    Totals alone would pass if two classes swapped counts, which is exactly what
    a broken label join looks like.
    """
    for name in PASCAL_SETS:
        if dataset and name != dataset:
            continue
        rows = table[table["subset"] == name]
        labelled = rows[rows["use_in_supervised"]]

        found = labelled["class_folder"].value_counts().to_dict()
        expected = EXPECTED_CLASS_COUNTS[name]
        if found != expected:
            raise ValueError(
                name + " class counts are " + repr(found) + " but "
                + repr(expected) + " were audited -- the label index has drifted"
            )
        if len(labelled) != EXPECTED_LABELLED_TOTALS[name]:
            raise ValueError(
                name + " holds " + str(len(labelled)) + " labelled records, expected "
                + str(EXPECTED_LABELLED_TOTALS[name])
            )
        n_unlabelled = int(rows["is_unlabeled"].sum())
        if n_unlabelled != EXPECTED_UNLABELLED_COUNTS[name]:
            raise ValueError(
                name + " holds " + str(n_unlabelled) + " unlabelled records, expected "
                + str(EXPECTED_UNLABELLED_COUNTS[name])
            )


def write_label_conflicts(table: Any, out_dir: str | Path | None = None) -> Path:
    """Write ``pascal_label_conflicts.csv`` (T12.4).

    Written even when empty, for the same reason as the PhysioNet one: an absent
    file cannot be told apart from a check that never ran.
    """
    target = _audit_dir(out_dir) / "pascal_label_conflicts.csv"
    columns = [
        "record_uid",
        "subset",
        "record_id",
        "canonical_key",
        "class_folder",
        "csv_label",
        "csv_sublabel",
        "label_conflict_reason",
    ]
    conflicts = table.loc[table["label_conflict"], columns]
    save_csv(conflicts, target)
    log.info("wrote %s (%d conflict row(s))", target.name, len(conflicts))
    return target


# ===========================================================================
# PHASE 13 -- subject derivation, locations and timing (T13.1 - T13.6)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class SubjectDerivation:
    """One record's subject group, session, site, and how it was obtained."""

    subject_id: str
    session_id: str
    recording_location: str
    location_repeat: str
    pattern: str
    subject_derived: bool


def derive_subject_pascal(record_id: str, dataset: str) -> SubjectDerivation:
    """Derive subject, session and auscultation site from a filename (T13.1-T13.4).

    **set_b (T13.1, T13.2).** ``<label>[_noisy<label>]_<subject>_<timestamp>_<site>``.
    The subject is the numeric patient id; the timestamp is the recording
    session; the trailing letter is the auscultation site.

    *The grouping key is the subject, not the session.* T13.1 quotes 167 groups;
    that is the count of distinct (subject, timestamp) pairs among the 461
    labelled records. Three subjects -- 109, 240 and 245 -- were recorded twice
    on the same day, minutes to ninety minutes apart, and two of them have both
    sessions labelled. Grouping on the session would put those two people on both
    sides of a fold. The real count is **165 subjects**; the session id is kept as
    its own column so the 167 figure stays reproducible without being the key.
    Settled with the user on 2026-08-25.

    *Sites are wider than T13.2 lists.* The real suffix set is
    A, A1, A2, B, B1, B2, B3, B_1, C, C1, C2, D, D1, D2, D3, D31, D4, E, F --
    six sites with a repeat index, not the five suffixes the task names. The
    letter goes in ``recording_location`` and the index in ``location_repeat``.
    ``D31`` is genuinely ambiguous (D repeat 31, or D3 repeat 1); it is recorded
    as read rather than guessed at, and it is one file.

    **set_a (T13.4).** Filenames are ``<label>__<timestamp>`` and carry no
    subject information at all, so each record is its own group with
    ``subject_derived=False``. PASCAL A results are therefore record-level and
    must never be described as subject-level.
    """
    if dataset == SET_A:
        return SubjectDerivation(
            subject_id="a_" + record_id,
            session_id="",
            recording_location="",
            location_repeat="",
            pattern="set_a: timestamp-only filename -- record is its own group",
            subject_derived=False,
        )

    if dataset != SET_B:
        raise ValueError("unknown PASCAL set " + repr(dataset))

    parsed = parse_set_b_name(record_id)
    return SubjectDerivation(
        subject_id="b_" + parsed.subject,
        session_id="b_" + parsed.subject + "_" + parsed.timestamp,
        recording_location=parsed.location,
        location_repeat=parsed.location_repeat,
        pattern=(
            "set_b: filename <label>[_noisy<label>]_<subject>_<timestamp>_<site>"
        ),
        subject_derived=True,
    )


def add_subject_ids(table: Any) -> Any:
    """Add subject, session and location columns (T13.1 - T13.4)."""
    out = table.copy()
    derivations = [
        derive_subject_pascal(row.record_id, row.subset)
        for row in out.itertuples(index=False)
    ]
    out["subject_id"] = [d.subject_id for d in derivations]
    out["session_id"] = [d.session_id for d in derivations]
    out["recording_location"] = [d.recording_location for d in derivations]
    out["location_repeat"] = [d.location_repeat for d in derivations]
    out["subject_pattern"] = [d.pattern for d in derivations]
    out["subject_derived"] = [d.subject_derived for d in derivations]

    if (out["subject_id"] == "").any():
        raise ValueError("some PASCAL records ended up with no subject_id")

    set_b = out[(out["subset"] == SET_B) & out["use_in_supervised"]]
    if not set_b.empty:
        n_subjects = set_b["subject_id"].nunique()
        n_sessions = set_b["session_id"].nunique()
        if n_subjects != N_SET_B_SUBJECTS or n_sessions != N_SET_B_SESSIONS:
            raise ValueError(
                "set_b grouping has drifted: " + str(n_subjects) + " subjects / "
                + str(n_sessions) + " sessions over the labelled records, expected "
                + str(N_SET_B_SUBJECTS) + " / " + str(N_SET_B_SESSIONS)
            )
        log.info(
            "PASCAL set_b: %d subjects over %d labelled records (%d sessions; "
            "%d subject(s) recorded more than once)",
            n_subjects,
            len(set_b),
            n_sessions,
            n_sessions - n_subjects,
        )
    return out


def load_set_a_timing(table: Any | None = None, root: Path | None = None) -> Any:
    """Load ``set_a_timing.csv`` into an S1/S2 annotation table (T13.5).

    390 rows: 21 set_a recordings, all of them ``normal``, hand-annotated with
    195 S1 and 195 S2 events over 5 to 19 cardiac cycles each.

    **The source column named ``location`` is a sample index, not a body site.**
    It is a position within the recording at set_a's native 44.1 kHz -- checked
    against every file, and it never exceeds the file length. It is renamed to
    ``sample_index`` here and paired with ``time_sec``, because a column called
    ``location`` sitting next to the ``recording_location`` of T13.2 is a
    collision waiting to be joined on.

    Consumed by Phase 80 (cycle analysis) and T113.6 (the dashboard
    cardiac-cycle viewer) -- it is not left unused.
    """
    import pandas as pd

    path = _timing_path(root)
    if not path.is_file():
        raise FileNotFoundError("set_a_timing.csv not found: " + str(path))

    frame = pd.read_csv(path)
    frame = frame.rename(columns={"location": "sample_index"})
    frame["canonical_key"] = [
        normalize_pascal_name(name, SET_A) for name in frame["fname"]
    ]
    frame["sound"] = frame["sound"].astype("string").str.strip().str.upper()
    frame["time_sec"] = frame["sample_index"] / TIMING_FS

    unexpected = set(frame["sound"]) - {"S1", "S2"}
    if unexpected:
        raise ValueError(
            "set_a_timing.csv holds unexpected sound label(s): "
            + ", ".join(sorted(unexpected))
        )
    if (frame["sample_index"] < 0).any():
        raise ValueError("set_a_timing.csv holds a negative sample index")

    if table is not None:
        known = set(table.loc[table["subset"] == SET_A, "canonical_key"])
        missing = sorted(set(frame["canonical_key"]) - known)
        if missing:
            raise ValueError(
                "set_a_timing.csv annotates " + str(len(missing)) + " recording(s) "
                "that are not in set_a, first few: " + ", ".join(missing[:5])
            )

    log.info(
        "PASCAL set_a timing: %d annotations over %d recording(s)",
        len(frame),
        frame["canonical_key"].nunique(),
    )
    return frame[
        ["canonical_key", "fname", "cycle", "sound", "sample_index", "time_sec"]
    ]


def write_subject_derivation(table: Any, out_dir: str | Path | None = None) -> Path:
    """Write ``pascal_subject_derivation.csv`` (T13.6)."""
    target = _audit_dir(out_dir) / "pascal_subject_derivation.csv"
    columns = [
        "record_uid",
        "subset",
        "record_id",
        "canonical_key",
        "subject_pattern",
        "subject_id",
        "session_id",
        "recording_location",
        "location_repeat",
        "noise_qualifier",
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


def load_pascal(
    root: Path | None = None,
    *,
    dataset: str | None = None,
    write_outputs: bool = False,
    out_dir: str | Path | None = None,
) -> Any:
    """Record table with labels, subjects, sessions and locations in one call.

    ``write_outputs=True`` also emits ``pascal_label_conflicts.csv`` (T12.4) and
    ``pascal_subject_derivation.csv`` (T13.6).
    """
    table = build_record_table(dataset, root)
    table = add_subject_ids(table)

    if write_outputs:
        write_label_conflicts(table, out_dir)
        write_subject_derivation(table, out_dir)
    return table
