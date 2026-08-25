"""CirCor DigiScope 2022 loader (Phases 14-15).

**D4**, and the only dataset here carrying two separate tasks:
``circor_murmur`` (Absent / Present / Unknown, per patient) and
``circor_outcome`` (Normal / Abnormal, per patient). Never merged -- rule 4.

Six things about this corpus are not obvious, and every one of them has cost a
session somewhere.

**The directory level is doubled.** The data lives at
``archive/training_data/training_data/``. A path built as
``archive/training_data/`` matches zero WAV files and returns an empty list
rather than raising -- the worst possible failure mode.

**``Outcome`` is not in ``training_data.csv``.** It exists only as
``#Outcome: Normal|Abnormal`` inside the per-patient ``.txt`` files. The CSV has
22 columns and none of them is Outcome, so a pipeline built on the CSV silently
has no outcome task at all.

**``RECORDS`` and ``SHA256SUMS.txt`` are relative to the *middle* directory.**
They list paths as ``training_data/13918_AV``, which resolves against
``archive/training_data/`` -- not against ``archive/``, where the two files
themselves sit. Resolved against the wrong base, all 10,431 entries report as
missing and the integrity check "fails" completely.

**All 942 patient ``.txt`` files fail their published checksum, and every WAV,
HEA and TSV passes.** 9,489 of 9,489 signal files are byte-perfect; 942 of 942
text files differ. That distribution is not corruption -- it is a manifest
generated from an earlier revision of the text files. Reported, not hidden. See
:func:`verify_integrity`.

**``training_data.csv`` and the ``.txt`` files disagree about ``Age`` for 73
patients**, and about nothing else. The txt files win, for the reason in
:func:`load_demographics`.

**One recording has no segmentation, and one segmentation has no recording.**
``50782_MV_1.wav`` exists with no ``.tsv``; ``50782_MV.tsv`` exists with no
``.wav`` and holds a single malformed row. They are not the same file wearing
two names, and neither is adopted as the other. See :func:`load_segmentation`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.constants import (
    CIRCOR_LOCATIONS,
    TASK_CIRCOR_MURMUR,
    TASK_CIRCOR_OUTCOME,
    encode_label,
    label_names,
)
from src.utils.io import ensure_dir, save_csv, save_parquet
from src.utils.logging_setup import get_logger

__all__ = [
    "PatientFile",
    "RecordingRef",
    "DATASET_ID",
    "SUBSET",
    "N_PATIENTS",
    "N_RECORDINGS",
    "EXPECTED_MURMUR_COUNTS",
    "EXPECTED_OUTCOME_COUNTS",
    "EXPECTED_LOCATION_COUNTS",
    "SEGMENTATION_STATES",
    "ANNOTATED_STATES",
    "METADATA_KEYS",
    "MURMUR_DESCRIPTORS",
    "NAN_LITERAL",
    "circor_root",
    "circor_archive_root",
    "parse_patient_file",
    "load_patient_files",
    "load_demographics",
    "build_patient_table",
    "build_record_table",
    "write_demographics_conflicts",
    "parse_circor_header",
    "load_segmentation",
    "segmentation_summary",
    "build_segmentation_table",
    "write_segmentation_artifacts",
    "write_unsegmented_report",
    "verify_integrity",
    "write_integrity_report",
    "load_circor",
]

log = get_logger(__name__)

DATASET_ID = "D4"
SUBSET = "training_data"

# T14.6 -- audited on disk 2026-08-22, re-verified 2026-08-25.
N_PATIENTS = 942
N_RECORDINGS = 3163
EXPECTED_MURMUR_COUNTS: dict[str, int] = {"Absent": 695, "Present": 179, "Unknown": 68}
EXPECTED_OUTCOME_COUNTS: dict[str, int] = {"Normal": 486, "Abnormal": 456}
EXPECTED_LOCATION_COUNTS: dict[str, int] = {
    "AV": 800,
    "PV": 766,
    "TV": 732,
    "MV": 861,
    "Phc": 4,
}

# T15.2 -- the TSV state alphabet. 0 means "not annotated here", which is a
# legitimate value covering 5,938 segments, not a missing one.
SEGMENTATION_STATES: dict[int, str] = {
    0: "unannotated",
    1: "S1",
    2: "systole",
    3: "S2",
    4: "diastole",
}
ANNOTATED_STATES: tuple[int, ...] = (1, 2, 3, 4)

# T14.3 / T14.4 -- every key present in all 942 patient files, in file order.
METADATA_KEYS: tuple[str, ...] = (
    "Age",
    "Sex",
    "Height",
    "Weight",
    "Pregnancy status",
    "Murmur",
    "Murmur locations",
    "Most audible location",
    "Systolic murmur timing",
    "Systolic murmur shape",
    "Systolic murmur grading",
    "Systolic murmur pitch",
    "Systolic murmur quality",
    "Diastolic murmur timing",
    "Diastolic murmur shape",
    "Diastolic murmur grading",
    "Diastolic murmur pitch",
    "Diastolic murmur quality",
    "Outcome",
    "Campaign",
    "Additional ID",
)

# T14.4 -- the ten systolic/diastolic descriptors, as a group.
MURMUR_DESCRIPTORS: tuple[str, ...] = tuple(
    phase + " murmur " + attribute
    for phase in ("Systolic", "Diastolic")
    for attribute in ("timing", "shape", "grading", "pitch", "quality")
)

# The source files write a missing value as the literal string "nan". Kept as a
# named constant because `== "nan"` scattered through the parser reads like a
# float check and is not one.
NAN_LITERAL = "nan"

_COLUMN_NAMES = {key: key.lower().replace(" ", "_") for key in METADATA_KEYS}

# Segment times are floats with microsecond noise: six recordings have a final
# segment ending up to 201 us past the end of the WAV. One sample at 4 kHz is
# 250 us, so a one-sample tolerance covers all of it while still catching a real
# overrun.
_TIME_TOLERANCE_SEC = 1.0 / 4000.0

_RECORD_NAME = re.compile(
    r"^(?P<patient>\d+)_(?P<location>" + "|".join(CIRCOR_LOCATIONS) + r")"
    r"(?:_(?P<index>\d+))?$"
)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def circor_root() -> Path:
    """The doubled data directory, ``archive/training_data/training_data``."""
    from src.utils.config import load_config

    return Path(load_config("paths").require("dataset.d4_circor.root"))


def circor_archive_root() -> Path:
    """``archive/`` -- where ``training_data.csv``, ``RECORDS`` and
    ``SHA256SUMS.txt`` actually live, one level *above* the doubled directory."""
    from src.utils.config import load_config

    return Path(load_config("paths").require("dataset.d4_circor.demographics_csv")).parent


def _manifest_base(root: Path | None = None) -> Path:
    """The base that ``RECORDS`` and ``SHA256SUMS.txt`` entries resolve against.

    Their paths read ``training_data/13918_AV.wav``, so the base is the middle
    ``training_data/`` directory -- the parent of the data directory, and the
    child of the directory the two manifests themselves sit in.
    """
    return (root or circor_root()).parent


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
# the per-patient .txt files (T14.2 - T14.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordingRef:
    """One line of a patient file: a location and its three companion files."""

    location: str
    record_id: str
    header_file: str
    wav_file: str
    tsv_file: str          # "" when the patient file lists no segmentation


@dataclass(frozen=True, slots=True)
class PatientFile:
    """One parsed ``<patient>.txt``."""

    patient_id: str
    n_locations: int
    fs_hz: int
    recordings: tuple[RecordingRef, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def murmur(self) -> str:
        return self.metadata.get("Murmur", "")

    @property
    def outcome(self) -> str:
        """The field that exists nowhere else. See the module docstring."""
        return self.metadata.get("Outcome", "")


def parse_patient_file(path: str | Path) -> PatientFile:
    """Parse one CirCor patient ``.txt`` file (T14.2 - T14.4).

    Structure, using ``13918.txt``::

        13918 4 4000                                        <- id, n_loc, fs
        AV 13918_AV.hea 13918_AV.wav 13918_AV.tsv           <- one per location
        PV 13918_PV.hea 13918_PV.wav 13918_PV.tsv
        ...
        #Age: Child                                         <- 21 metadata keys
        #Murmur: Present
        #Outcome: Abnormal

    Two details the format hides.

    A location line normally carries four fields, but ``50782.txt`` has one with
    three -- ``MV 50782_MV_1.hea 50782_MV_1.wav`` and no TSV. That is a real
    recording with no segmentation, so a strict four-field parse would reject a
    valid patient. The TSV slot is left empty instead, and T15.5 reports it.

    A location can also repeat: 17 patients have two recordings at one site,
    named ``<patient>_<LOC>_1`` and ``_2``. ``n_locations`` counts *recordings*,
    not distinct sites, which is why 3 patients declare 6 and 10 declare 5 while
    only five sites exist.
    """
    file_path = Path(path)
    lines = [ln.strip() for ln in file_path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError("empty CirCor patient file: " + str(file_path))

    header = lines[0].split()
    if len(header) < 3:
        raise ValueError(
            "malformed CirCor header line in " + str(file_path) + ": " + repr(lines[0])
        )
    patient_id, n_locations, fs_hz = header[0], int(header[1]), int(header[2])
    if patient_id != file_path.stem:
        raise ValueError(
            "CirCor patient file " + file_path.name + " declares patient id "
            + repr(patient_id) + " -- filename and content disagree"
        )

    recordings: list[RecordingRef] = []
    for line in lines[1 : 1 + n_locations]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(
                "malformed location line in " + str(file_path) + ": " + repr(line)
            )
        location, header_file, wav_file = parts[0], parts[1], parts[2]
        if location not in CIRCOR_LOCATIONS:
            raise ValueError(
                "unknown auscultation location " + repr(location) + " in "
                + str(file_path) + " (valid: " + ", ".join(CIRCOR_LOCATIONS) + ")"
            )
        recordings.append(
            RecordingRef(
                location=location,
                record_id=Path(wav_file).stem,
                header_file=header_file,
                wav_file=wav_file,
                tsv_file=parts[3] if len(parts) > 3 else "",
            )
        )
    if len(recordings) != n_locations:
        raise ValueError(
            str(file_path) + " declares " + str(n_locations) + " location(s) but "
            "lists " + str(len(recordings))
        )

    metadata: dict[str, str] = {}
    for line in lines[1 + n_locations :]:
        if not line.startswith("#"):
            raise ValueError(
                "unexpected non-metadata line in " + str(file_path) + ": " + repr(line)
            )
        key, _, value = line[1:].partition(":")
        metadata[key.strip()] = value.strip()

    missing = [k for k in METADATA_KEYS if k not in metadata]
    if missing:
        raise ValueError(
            str(file_path) + " is missing metadata key(s): " + ", ".join(missing)
        )
    return PatientFile(
        patient_id=patient_id,
        n_locations=n_locations,
        fs_hz=fs_hz,
        recordings=tuple(recordings),
        metadata=metadata,
    )


def load_patient_files(root: Path | None = None) -> list[PatientFile]:
    """Every patient file under the doubled data directory (T14.1, T14.2)."""
    base = root or circor_root()
    if not base.is_dir():
        raise FileNotFoundError(
            "CirCor data directory not found: " + str(base) + " -- note the DOUBLED "
            "training_data/training_data path; the shallower one matches no files"
        )

    files = sorted(base.glob("*.txt"), key=lambda p: p.stem)
    if not files:
        raise FileNotFoundError(
            "no patient .txt files under " + str(base) + " -- this is what the "
            "doubled-directory mistake looks like: a valid path with nothing in it"
        )
    return [parse_patient_file(p) for p in files]


# ---------------------------------------------------------------------------
# demographics CSV (T14.5)
# ---------------------------------------------------------------------------


def load_demographics(root: Path | None = None) -> Any:
    """Load ``training_data.csv`` (T14.5).

    ``keep_default_na=False`` is deliberate. The file writes missing values as
    the literal string ``nan``, exactly as the ``.txt`` files do, and letting
    pandas convert those to real NaN makes the two sources look different in 74
    places where they actually agree.

    **This file has no ``Outcome`` column.** It is demographics and murmur
    description only; the outcome task comes from the txt files.
    """
    import pandas as pd

    path = (
        (root / "training_data.csv") if root is not None
        else Path(_config_path("dataset.d4_circor.demographics_csv"))
    )
    if not path.is_file():
        raise FileNotFoundError("CirCor demographics CSV not found: " + str(path))

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "Outcome" in frame.columns:
        raise ValueError(
            "training_data.csv unexpectedly has an Outcome column -- the loader "
            "reads Outcome from the patient .txt files and that assumption has "
            "just changed"
        )
    frame = frame.rename(columns={"Patient ID": "patient_id"})
    for column in frame.columns:
        frame[column] = frame[column].astype("string").str.strip()
    return frame


def _config_path(key: str) -> str:
    from src.utils.config import load_config

    return str(load_config("paths").require(key))


# ---------------------------------------------------------------------------
# the patient table (T14.5, T14.6)
# ---------------------------------------------------------------------------


def _clean(value: str) -> str:
    """Source ``nan`` -> empty string, so missing reads as missing everywhere."""
    text = (value or "").strip()
    return "" if text.lower() == NAN_LITERAL else text


def build_patient_table(root: Path | None = None) -> Any:
    """One row per patient, with both task labels and all metadata (T14.2-T14.5)."""
    import pandas as pd

    patients = load_patient_files(root)
    rows: list[dict[str, Any]] = []
    for patient in patients:
        meta = patient.metadata
        murmur = _clean(meta["Murmur"])
        outcome = _clean(meta["Outcome"])
        if murmur not in label_names(TASK_CIRCOR_MURMUR):
            raise ValueError(
                "patient " + patient.patient_id + " has murmur value " + repr(murmur)
                + " outside the circor_murmur vocabulary"
            )
        if outcome not in label_names(TASK_CIRCOR_OUTCOME):
            raise ValueError(
                "patient " + patient.patient_id + " has outcome value " + repr(outcome)
                + " outside the circor_outcome vocabulary"
            )

        row: dict[str, Any] = {
            "patient_id": patient.patient_id,
            "n_recordings": patient.n_locations,
            "locations": "+".join(r.location for r in patient.recordings),
            "fs_hz": patient.fs_hz,
            "murmur": murmur,
            "murmur_label": encode_label(TASK_CIRCOR_MURMUR, murmur),
            "outcome": outcome,
            "outcome_label": encode_label(TASK_CIRCOR_OUTCOME, outcome),
            "outcome_source": "patient .txt #Outcome (absent from training_data.csv)",
        }
        for key in METADATA_KEYS:
            if key in ("Murmur", "Outcome"):
                continue
            row[_COLUMN_NAMES[key]] = _clean(meta[key])
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("patient_id", kind="stable")
    table = table.reset_index(drop=True)

    if len(table) != N_PATIENTS:
        raise ValueError(
            "CirCor holds " + str(len(table)) + " patients but " + str(N_PATIENTS)
            + " were audited"
        )
    if table["patient_id"].duplicated().any():
        raise ValueError("CirCor patient_id is not unique")

    murmur_counts = table["murmur"].value_counts().to_dict()
    if murmur_counts != EXPECTED_MURMUR_COUNTS:
        raise ValueError(
            "CirCor murmur counts are " + repr(murmur_counts) + " but "
            + repr(EXPECTED_MURMUR_COUNTS) + " were audited"
        )
    outcome_counts = table["outcome"].value_counts().to_dict()
    if outcome_counts != EXPECTED_OUTCOME_COUNTS:
        raise ValueError(
            "CirCor outcome counts are " + repr(outcome_counts) + " but "
            + repr(EXPECTED_OUTCOME_COUNTS) + " were audited -- and Outcome comes "
            "only from the patient .txt files, so this is where a CSV-based "
            "shortcut shows up"
        )
    return table


def _demographics_conflicts(table: Any, root: Path | None = None) -> Any:
    """Field-by-field comparison of the CSV against the txt files (T14.5)."""
    import pandas as pd

    csv = load_demographics(root)
    if set(csv["patient_id"]) != set(table["patient_id"]):
        raise ValueError(
            "training_data.csv and the patient .txt files disagree about which "
            "patients exist"
        )

    by_id = table.set_index("patient_id")
    rows: list[dict[str, Any]] = []
    for _, csv_row in csv.iterrows():
        patient_id = str(csv_row["patient_id"])
        txt_row = by_id.loc[patient_id]
        for source_column in csv.columns:
            if source_column == "patient_id":
                continue
            target = (
                "locations" if source_column == "Locations"
                else _COLUMN_NAMES.get(source_column, source_column)
            )
            if target not in by_id.columns and target != "murmur":
                continue
            csv_value = _clean(str(csv_row[source_column]))
            txt_value = _clean(str(txt_row[target]))
            if csv_value != txt_value:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "field": source_column,
                        "training_data_csv": csv_value,
                        "patient_txt": txt_value,
                        "authoritative": "patient_txt",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "patient_id",
            "field",
            "training_data_csv",
            "patient_txt",
            "authoritative",
        ],
    )


def write_demographics_conflicts(
    table: Any, root: Path | None = None, out_dir: str | Path | None = None
) -> Path:
    """Write ``circor_demographics_conflicts.csv`` (T14.5).

    The two sources agree on patient ids and on every field except ``Age``,
    where they disagree for 73 of 942 patients -- real category differences
    (``Infant`` versus ``Child``, ``Young Adult`` versus ``Adolescent``), not a
    parsing artefact, and ``Young Adult`` exists only in the CSV.

    The txt files are treated as authoritative, consistently with ``Outcome``
    already coming from them, and with the checksum evidence that the txt files
    were revised *after* the published manifest was generated while the CSV still
    matches it. Both values are written out so the choice stays inspectable.
    """
    target = _audit_dir(out_dir) / "circor_demographics_conflicts.csv"
    conflicts = _demographics_conflicts(table, root)
    save_csv(conflicts, target)
    log.info(
        "wrote %s (%d field conflict(s) over %d patient(s))",
        target.name,
        len(conflicts),
        conflicts["patient_id"].nunique() if len(conflicts) else 0,
    )
    return target


# ---------------------------------------------------------------------------
# headers (T15.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CircorHeader:
    record: str
    n_signals: int
    fs_hz: int
    n_samples: int
    location: str


def parse_circor_header(path: str | Path) -> _CircorHeader:
    """Parse a CirCor ``.hea`` (T15.1).

    ``13918_AV 1 4000 41152`` plus one signal line whose trailing description is
    the auscultation location.
    """
    header_path = Path(path)
    lines = [ln.strip() for ln in header_path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if len(lines) < 2:
        raise ValueError("malformed CirCor header: " + str(header_path))

    fields = lines[0].split()
    if len(fields) < 4:
        raise ValueError(
            "malformed CirCor header line in " + str(header_path) + ": "
            + repr(lines[0])
        )
    signal = lines[1].split()
    return _CircorHeader(
        record=fields[0],
        n_signals=int(fields[1]),
        fs_hz=int(float(fields[2])),
        n_samples=int(fields[3]),
        location=signal[-1] if len(signal) > 1 else "",
    )


# ===========================================================================
# PHASE 15 -- segmentation (T15.2 - T15.5)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class _Segmentation:
    """One parsed ``.tsv``: aligned start/end/state triples, plus any problems."""

    record_id: str
    segments: tuple[tuple[float, float, int], ...]
    issues: tuple[str, ...]

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def load_segmentation(
    path: str | Path, *, duration_sec: float | None = None
) -> _Segmentation:
    """Parse one CirCor ``.tsv`` segmentation file (T15.2).

    Three tab-separated columns -- start, end, state -- and no header. States map
    through :data:`SEGMENTATION_STATES`: 1 = S1, 2 = systole, 3 = S2,
    4 = diastole, 0 = unannotated.

    **Rows are normalised, and every normalisation is reported rather than
    applied silently.** Three things were found across the 3,162 real files:

    *Two files list every segment twice.* ``50690_MV_2.tsv`` (104 rows, 53
    unique) and ``50690_TV.tsv`` (76 rows, 39 unique) contain an exact duplicate
    of each row, appended out of order -- which is what makes them look like
    8.6-second and 7.2-second backwards jumps. Dropping exact duplicates and
    sorting by start time resolves both to zero overlaps and loses nothing: no
    value is changed, and no row is dropped that is not a byte-identical copy of
    one that is kept. These are the only two such files in the corpus.

    *One file has a genuine 11.7 ms overlap.* ``50150_MV.tsv`` has a diastole
    segment ending 11.7 ms after the next S1 begins. It is real, it is small, and
    it is left exactly as it is -- recorded as an issue, not edited.

    *Six files end a few microseconds past their WAV.* Up to 201 us, against a
    250 us sample period, so it is float formatting rather than an overrun. A
    one-sample tolerance absorbs it; anything larger is reported.
    """
    tsv_path = Path(path)
    raw: list[tuple[float, float, int]] = []
    issues: list[str] = []

    for number, line in enumerate(
        tsv_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) != 3:
            raise ValueError(
                str(tsv_path) + " line " + str(number) + " has " + str(len(parts))
                + " field(s), expected 3: " + repr(line)
            )
        start, end, state = float(parts[0]), float(parts[1]), int(float(parts[2]))
        if state not in SEGMENTATION_STATES:
            issues.append(
                "invalid state " + str(state) + " on line " + str(number)
                + " (valid: " + ", ".join(str(s) for s in SEGMENTATION_STATES) + ")"
            )
        if end < start:
            issues.append(
                "segment " + str(number) + " ends before it starts ("
                + str(start) + " -> " + str(end) + ")"
            )
        if start < 0:
            issues.append("segment " + str(number) + " starts before zero")
        raw.append((start, end, state))

    deduplicated = sorted(set(raw))
    if len(deduplicated) != len(raw):
        issues.append(
            "dropped " + str(len(raw) - len(deduplicated)) + " exact duplicate row(s) "
            "and sorted by start time"
        )
    elif deduplicated != raw:
        issues.append("rows were not in start-time order; sorted")

    for index in range(1, len(deduplicated)):
        overlap = deduplicated[index - 1][1] - deduplicated[index][0]
        if overlap > _TIME_TOLERANCE_SEC:
            issues.append(
                "segments " + str(index) + " and " + str(index + 1) + " overlap by "
                + format(overlap, ".6f") + " s"
            )

    if duration_sec is not None and deduplicated:
        overrun = deduplicated[-1][1] - duration_sec
        if overrun > _TIME_TOLERANCE_SEC:
            issues.append(
                "last segment ends " + format(overrun, ".6f")
                + " s past the end of the recording"
            )

    return _Segmentation(
        record_id=tsv_path.stem,
        segments=tuple(deduplicated),
        issues=tuple(issues),
    )


def segmentation_summary(
    segmentation: _Segmentation, duration_sec: float
) -> dict[str, Any]:
    """Annotated fraction and estimated cycle count for one recording (T15.3).

    The cycle count is the number of **S1 segments** -- one S1 per cardiac cycle.
    Counting complete S1/systole/S2/diastole runs instead would undercount every
    recording whose annotation starts mid-cycle, which is most of them.
    """
    annotated = sum(
        end - start
        for start, end, state in segmentation.segments
        if state in ANNOTATED_STATES
    )
    n_cycles = sum(1 for _, _, state in segmentation.segments if state == 1)
    per_state = {
        name: sum(
            end - start
            for start, end, state in segmentation.segments
            if state == code
        )
        for code, name in SEGMENTATION_STATES.items()
    }
    return {
        "n_segments": segmentation.n_segments,
        "annotated_sec": round(annotated, 6),
        "annotated_fraction": (
            round(annotated / duration_sec, 6) if duration_sec > 0 else 0.0
        ),
        "n_cycles": n_cycles,
        "unannotated_sec": round(per_state["unannotated"], 6),
        "s1_sec": round(per_state["S1"], 6),
        "systole_sec": round(per_state["systole"], 6),
        "s2_sec": round(per_state["S2"], 6),
        "diastole_sec": round(per_state["diastole"], 6),
        "segmentation_issues": "; ".join(segmentation.issues),
        "has_segmentation": bool(
            segmentation.segments
            and any(s in ANNOTATED_STATES for _, _, s in segmentation.segments)
        ),
    }


# ---------------------------------------------------------------------------
# the record table (T14.6, T15.1, T15.3)
# ---------------------------------------------------------------------------


def build_record_table(
    root: Path | None = None,
    *,
    patients: Any | None = None,
    with_segmentation: bool = True,
) -> Any:
    """One row per recording -- 3,163 of them -- with patient labels attached.

    Patient-level labels (murmur, outcome, demographics) are broadcast to every
    recording of that patient. That is correct for these two tasks: both are
    diagnosed per patient, not per auscultation site. It is also exactly why
    ``patient_id`` must be the grouping key for cross-validation -- four
    recordings of one child sharing a label is four chances to leak.
    """
    import pandas as pd
    import soundfile as sf

    base = root or circor_root()
    patient_files = load_patient_files(base)
    patient_table = build_patient_table(base) if patients is None else patients
    patient_meta = patient_table.set_index("patient_id")

    rows: list[dict[str, Any]] = []
    for patient in patient_files:
        meta = patient_meta.loc[patient.patient_id]
        murmur_locations = [
            part for part in str(meta["murmur_locations"]).split("+") if part
        ]

        for recording in patient.recordings:
            match = _RECORD_NAME.match(recording.record_id)
            if match is None:
                raise ValueError(
                    "CirCor record name does not parse: " + recording.record_id
                )
            if match.group("location") != recording.location:
                raise ValueError(
                    "record " + recording.record_id + " is listed under location "
                    + recording.location + " but its name says "
                    + str(match.group("location"))
                )

            wav_path = base / recording.wav_file
            header_path = base / recording.header_file
            if not wav_path.is_file():
                raise FileNotFoundError("missing CirCor WAV: " + str(wav_path))

            header = parse_circor_header(header_path)
            info = sf.info(str(wav_path))

            # T15.1 -- the header is cross-checked against the WAV, not trusted.
            # A header claiming a length the audio does not have would put a
            # wrong duration into every downstream table.
            if header.fs_hz != info.samplerate:
                raise ValueError(
                    recording.record_id + ": header says " + str(header.fs_hz)
                    + " Hz, WAV says " + str(info.samplerate)
                )
            if header.n_samples != info.frames:
                raise ValueError(
                    recording.record_id + ": header says " + str(header.n_samples)
                    + " samples, WAV says " + str(info.frames)
                )
            if header.record != recording.record_id:
                raise ValueError(
                    "header in " + str(header_path) + " names record "
                    + repr(header.record) + ", expected " + repr(recording.record_id)
                )

            duration = info.frames / info.samplerate
            row: dict[str, Any] = {
                "record_uid": DATASET_ID + "_" + SUBSET + "_" + recording.record_id,
                "dataset_source": DATASET_ID,
                "subset": SUBSET,
                "record_id": recording.record_id,
                "patient_id": patient.patient_id,
                "subject_id": patient.patient_id,
                "subject_derived": False,   # native id -- nothing was derived
                "recording_location": recording.location,
                "location_index": match.group("index") or "",
                "file_path": _relative(wav_path),
                "header_path": _relative(header_path),
                "tsv_path": (
                    _relative(base / recording.tsv_file) if recording.tsv_file else ""
                ),
                "original_fs": int(info.samplerate),
                "n_samples": int(info.frames),
                "n_channels": int(info.channels),
                "duration_sec": round(duration, 6),
                "header_fs": header.fs_hz,
                "header_n_samples": header.n_samples,
                "header_location": header.location,
                "murmur": meta["murmur"],
                "murmur_label": int(meta["murmur_label"]),
                "outcome": meta["outcome"],
                "outcome_label": int(meta["outcome_label"]),
                "murmur_locations": meta["murmur_locations"],
                "most_audible_location": meta["most_audible_location"],
                "is_murmur_location": recording.location in murmur_locations,
                "is_most_audible": (
                    recording.location == str(meta["most_audible_location"])
                ),
                "use_in_supervised": True,
            }
            for key in METADATA_KEYS:
                if key in ("Murmur", "Outcome", "Murmur locations",
                           "Most audible location"):
                    continue
                row[_COLUMN_NAMES[key]] = meta[_COLUMN_NAMES[key]]

            if with_segmentation:
                tsv_path = base / recording.tsv_file if recording.tsv_file else None
                if tsv_path is not None and tsv_path.is_file():
                    parsed = load_segmentation(tsv_path, duration_sec=duration)
                    row.update(segmentation_summary(parsed, duration))
                    row["segmentation_available"] = True
                else:
                    row.update(
                        {
                            "n_segments": 0,
                            "annotated_sec": 0.0,
                            "annotated_fraction": 0.0,
                            "n_cycles": 0,
                            "unannotated_sec": 0.0,
                            "s1_sec": 0.0,
                            "systole_sec": 0.0,
                            "s2_sec": 0.0,
                            "diastole_sec": 0.0,
                            "segmentation_issues": (
                                "no .tsv listed in the patient file"
                                if not recording.tsv_file
                                else "listed .tsv does not exist on disk"
                            ),
                            "has_segmentation": False,
                            "segmentation_available": False,
                        }
                    )
            rows.append(row)

    table = pd.DataFrame(rows).sort_values("record_id", kind="stable")
    table = table.reset_index(drop=True)

    if len(table) != N_RECORDINGS:
        raise ValueError(
            "CirCor holds " + str(len(table)) + " recordings but "
            + str(N_RECORDINGS) + " were audited"
        )
    if table["record_uid"].duplicated().any():
        raise ValueError("CirCor record_uid is not unique")

    location_counts = table["recording_location"].value_counts().to_dict()
    if location_counts != EXPECTED_LOCATION_COUNTS:
        raise ValueError(
            "CirCor location counts are " + repr(location_counts) + " but "
            + repr(EXPECTED_LOCATION_COUNTS) + " were audited"
        )

    log.info(
        "CirCor: %d recordings over %d patients (%d segmented, %d not)",
        len(table),
        table["patient_id"].nunique(),
        int(table["has_segmentation"].sum()) if with_segmentation else 0,
        int((~table["has_segmentation"]).sum()) if with_segmentation else 0,
    )
    return table


# ---------------------------------------------------------------------------
# segmentation artifacts (T15.4, T15.5)
# ---------------------------------------------------------------------------


def build_segmentation_table(table: Any, root: Path | None = None) -> Any:
    """Every segment of every recording, as one long table (T15.4).

    ~262,000 rows: ``record_uid, patient_id, segment_index, start_sec, end_sec,
    duration_sec, state, state_name``. This is the artifact Phase 80 (cycle
    analysis) and T113.6 (the dashboard cardiac-cycle viewer) consume -- T15.4 is
    explicit that the segmentation must not be parsed and thrown away.
    """
    import pandas as pd

    base = root or circor_root()
    rows: list[dict[str, Any]] = []
    for record in table.itertuples(index=False):
        if not record.tsv_path:
            continue
        # Resolved against `base` by filename rather than by replaying the stored
        # project-relative path, so a caller pointing at a copy of the tree gets
        # that copy's segmentations. Every CirCor .tsv sits flat in one directory.
        tsv_path = base / Path(record.tsv_path).name
        if not tsv_path.is_file():
            continue
        parsed = load_segmentation(tsv_path, duration_sec=record.duration_sec)
        for index, (start, end, state) in enumerate(parsed.segments):
            rows.append(
                {
                    "record_uid": record.record_uid,
                    "patient_id": record.patient_id,
                    "segment_index": index,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": round(end - start, 6),
                    "state": state,
                    "state_name": SEGMENTATION_STATES.get(state, "invalid"),
                }
            )
    frame = pd.DataFrame(rows)
    log.info(
        "CirCor segmentation: %d segments over %d recording(s)",
        len(frame),
        frame["record_uid"].nunique() if len(frame) else 0,
    )
    return frame


def write_segmentation_artifacts(
    table: Any, root: Path | None = None, out_dir: str | Path | None = None
) -> dict[str, Path]:
    """Write the per-segment Parquet and the per-recording summary CSV (T15.4)."""
    from src.utils.config import load_config

    segments = build_segmentation_table(table, root)
    cache_dir = ensure_dir(load_config("paths").require("cache.metadata"))
    parquet = cache_dir / "circor_segmentation.parquet"
    save_parquet(segments, parquet)

    summary_columns = [
        "record_uid",
        "patient_id",
        "record_id",
        "recording_location",
        "duration_sec",
        "segmentation_available",
        "has_segmentation",
        "n_segments",
        "n_cycles",
        "annotated_sec",
        "annotated_fraction",
        "unannotated_sec",
        "s1_sec",
        "systole_sec",
        "s2_sec",
        "diastole_sec",
        "segmentation_issues",
    ]
    summary = _audit_dir(out_dir) / "circor_segmentation_summary.csv"
    save_csv(table[summary_columns], summary)
    log.info("wrote %s and %s", parquet.name, summary.name)
    return {"segments": parquet, "summary": summary}


def write_unsegmented_report(
    table: Any, root: Path | None = None, out_dir: str | Path | None = None
) -> Path:
    """Write ``circor_unsegmented.csv`` (T15.5).

    Two distinct problems land here, and they are not the same file:

    * ``50782_MV_1`` -- a real recording, listed in ``RECORDS``, whose patient
      file names no ``.tsv`` and for which none exists.
    * ``50782_MV.tsv`` -- a segmentation file with no recording, holding one
      malformed row (``0  0  28``: zero length, and 28 is not a state).

    It is tempting to read the second as a mis-named copy of the first. It is
    not adopted: the association is a guess, and the file is unusable anyway.
    Both are listed so the pair is visible.
    """
    import pandas as pd

    base = root or circor_root()
    columns = [
        "record_uid",
        "record_id",
        "patient_id",
        "recording_location",
        "duration_sec",
        "segmentation_issues",
    ]
    unsegmented = table.loc[~table["has_segmentation"], columns].copy()
    unsegmented["kind"] = "recording without usable segmentation"

    # Orphan .tsv files: present on disk, belonging to no listed recording.
    listed = {
        Path(p).stem for p in table["tsv_path"] if p
    }
    orphans = sorted(
        p for p in base.glob("*.tsv") if p.stem not in listed
    )
    orphan_rows = []
    for path in orphans:
        parsed = load_segmentation(path)
        orphan_rows.append(
            {
                "record_uid": "",
                "record_id": path.stem,
                "patient_id": path.stem.split("_")[0],
                "recording_location": "",
                "duration_sec": float("nan"),
                "segmentation_issues": (
                    "orphan .tsv: no recording of this name is listed in any patient "
                    "file" + ("; " + "; ".join(parsed.issues) if parsed.issues else "")
                ),
                "kind": "segmentation without a recording",
            }
        )

    report = pd.concat(
        [unsegmented, pd.DataFrame(orphan_rows, columns=[*columns, "kind"])],
        ignore_index=True,
    )
    target = _audit_dir(out_dir) / "circor_unsegmented.csv"
    save_csv(report, target)
    log.info(
        "wrote %s (%d recording(s) without segmentation, %d orphan .tsv file(s))",
        target.name,
        len(unsegmented),
        len(orphan_rows),
    )
    return target


# ---------------------------------------------------------------------------
# integrity (T15.6)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_integrity(root: Path | None = None, *, hash_files: bool = True) -> Any:
    """Verify ``RECORDS`` and ``SHA256SUMS.txt`` across the CirCor tree (T15.6).

    **Both manifests resolve against the middle ``training_data/`` directory**,
    not against ``archive/`` where they themselves sit. Their entries read
    ``training_data/13918_AV.wav``; joined to the wrong base, all 10,431 report
    as missing and the check appears to fail catastrophically.

    **The expected result is not "everything matches".** Every one of the 9,489
    WAV, HEA and TSV files matches its published SHA-256, and every one of the
    942 patient ``.txt`` files does not. 0-of-942 is not what corruption looks
    like -- it is a manifest generated from an earlier revision of those files,
    which is consistent with ``training_data.csv`` (which still matches its own
    checksum) disagreeing with the txt files about ``Age``.

    The signal data this project actually processes is therefore verified
    byte-perfect, and the metadata gap is reported rather than glossed over.
    """
    import pandas as pd

    base = root or circor_root()
    manifest_base = _manifest_base(base)
    archive = circor_archive_root()

    rows: list[dict[str, Any]] = []

    records_path = archive / "RECORDS"
    if not records_path.is_file():
        raise FileNotFoundError("CirCor RECORDS not found: " + str(records_path))
    listed = [
        line.strip()
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for entry in listed:
        wav = manifest_base / (entry + ".wav")
        rows.append(
            {
                "manifest": "RECORDS",
                "entry": entry,
                "file_type": ".wav",
                "status": "ok" if wav.is_file() else "missing",
                "detail": "" if wav.is_file() else "listed in RECORDS but not on disk",
            }
        )

    checksums_path = archive / "SHA256SUMS.txt"
    if not checksums_path.is_file():
        raise FileNotFoundError("CirCor SHA256SUMS.txt not found: " + str(checksums_path))
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        candidate = manifest_base / name
        if not candidate.exists():
            candidate = archive / name
        if not candidate.is_file():
            rows.append(
                {
                    "manifest": "SHA256SUMS",
                    "entry": name,
                    "file_type": Path(name).suffix.lower(),
                    "status": "missing",
                    "detail": "listed in SHA256SUMS.txt but not on disk",
                }
            )
            continue
        if not hash_files:
            continue
        actual = _sha256(candidate)
        matches = actual == expected.strip()
        rows.append(
            {
                "manifest": "SHA256SUMS",
                "entry": name,
                "file_type": Path(name).suffix.lower(),
                "status": "ok" if matches else "checksum_mismatch",
                "detail": "" if matches else "expected " + expected.strip()[:16]
                + "..., got " + actual[:16] + "...",
            }
        )

    report = pd.DataFrame(
        rows, columns=["manifest", "entry", "file_type", "status", "detail"]
    )
    missing = report[report["status"] == "missing"]
    if not missing.empty:
        raise ValueError(
            str(len(missing)) + " CirCor file(s) are listed in a manifest but absent "
            "from disk, first few: " + ", ".join(missing["entry"].head(5))
        )

    mismatched = report[report["status"] == "checksum_mismatch"]
    by_type = mismatched["file_type"].value_counts().to_dict()
    if by_type and set(by_type) != {".txt"}:
        raise ValueError(
            "CirCor checksum mismatches outside the patient .txt files: "
            + repr(by_type) + " -- the signal data is supposed to be byte-perfect"
        )
    if mismatched.empty:
        log.info("CirCor integrity: every manifest entry present and matching")
    else:
        checksummed = report[report["manifest"] == "SHA256SUMS"]
        signal = checksummed[checksummed["file_type"].isin([".wav", ".hea", ".tsv"])]
        log.warning(
            "CirCor integrity: all %d signal file(s) (wav/hea/tsv) match their "
            "published SHA-256, and %d patient .txt file(s) do not -- 0 of %d, "
            "which is a manifest predating a revision of those files, not "
            "corruption. RECORDS: %d/%d recordings present.",
            int((signal["status"] == "ok").sum()),
            len(mismatched),
            len(mismatched),
            int((report["manifest"] == "RECORDS").sum()),
            int((report["manifest"] == "RECORDS").sum()),
        )
    return report


def write_integrity_report(
    report: Any, out_dir: str | Path | None = None
) -> Path:
    """Write ``circor_integrity.csv`` (T15.6)."""
    target = _audit_dir(out_dir) / "circor_integrity.csv"
    save_csv(report, target)
    log.info("wrote %s (%d manifest entries)", target.name, len(report))
    return target


# ---------------------------------------------------------------------------
# one call for the whole dataset
# ---------------------------------------------------------------------------


def load_circor(
    root: Path | None = None,
    *,
    with_segmentation: bool = True,
    write_outputs: bool = False,
    out_dir: str | Path | None = None,
    verify: bool = False,
    limit: int | None = None,
    use_cache: bool = True,
) -> Any:
    """Patient labels, recordings and segmentation summaries in one call.

    ``verify=True`` additionally runs the SHA-256 pass over 585 MB, which takes
    around 40 seconds -- off by default so an ordinary load stays quick.

    ``limit`` caps the table at N recordings for smoke runs (T22.1). The sample
    interleaves the murmur classes, so a 20-record limit still contains Absent,
    Present and Unknown rather than 20 Absents. Audit outputs are never written
    from a limited table.
    """
    from src.data_loader.cache import apply_limit, cached_table

    def _build() -> Any:
        return build_record_table(root, with_segmentation=with_segmentation)

    archive = circor_archive_root()
    table = cached_table(
        "circor",
        _build,
        metadata_files=[
            archive / "training_data.csv",
            archive / "RECORDS",
        ],
        trees=[root or circor_root()],
        extra={"with_segmentation": bool(with_segmentation)},
        enabled=use_cache,
    )

    if limit is not None:
        if write_outputs:
            raise ValueError(
                "refusing to write CirCor audit outputs from a --limit run: "
                "a partial DA artifact reads as a complete one"
            )
        return apply_limit(
            table, limit, by=("dataset_source",), stratify=("murmur", "outcome")
        )

    if write_outputs:
        patients = build_patient_table(root)
        write_demographics_conflicts(patients, root, out_dir)
        write_segmentation_artifacts(table, root, out_dir)
        write_unsegmented_report(table, root, out_dir)
        if verify:
            write_integrity_report(verify_integrity(root), out_dir)
    return table
