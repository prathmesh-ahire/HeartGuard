"""Dataset identifiers and label vocabularies (Phase 05, tasks T05.1-T05.5).

**Rule 4 is enforced here, structurally.** The five label spaces -- ``binary``,
``pascal_a``, ``pascal_b``, ``circor_murmur``, ``circor_outcome`` -- are separate
tasks and are never merged.

The trap this module exists to close: the *names* deliberately collide across
tasks, and so do the *codes*.

===========  ==========  ==========  ==========  ==========
code         binary      pascal_a    pascal_b    circor_murmur
===========  ==========  ==========  ==========  ==========
0            normal      normal      normal      Absent
1            abnormal    murmur      murmur      Present
2            --          extrahls    extrastole  Unknown
3            --          artifact    --          --
===========  ==========  ==========  ==========  ==========

A bare ``1`` is meaningless. It is ``murmur`` under PASCAL A, ``murmur`` under
PASCAL B (a *different* task with a different class set), ``Present`` under
CirCor murmur and ``abnormal`` under the binary task. Code ``2`` is the sharpest
case: ``extrahls`` in PASCAL A and ``extrastole`` in PASCAL B -- two different
cardiac phenomena, one integer.

So every label is addressed as a **(task, code)** or **(task, name)** pair, and
:func:`namespaced_id` produces globally unique identifiers such as
``pascal_a:murmur``. There is deliberately no ``decode(code)`` that takes a bare
integer, because there is no correct answer to it.

All mappings are read-only at runtime (``MappingProxyType``): a label map cannot
be mutated by accident halfway through a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "DatasetInfo",
    "DATASETS",
    "DATASET_IDS",
    "TASKS",
    "TASK_BINARY",
    "TASK_PASCAL_A",
    "TASK_PASCAL_B",
    "TASK_CIRCOR_MURMUR",
    "TASK_CIRCOR_OUTCOME",
    "LABEL_MAPS",
    "INVERSE_LABEL_MAPS",
    "TASK_DATASET",
    "TASK_KIND",
    "PHYSIONET_REFERENCE_MAP",
    "CIRCOR_LOCATIONS",
    "PASCAL_CLASS_FOLDERS",
    "UNLABELED_CLASS",
    "encode_label",
    "decode_label",
    "namespaced_id",
    "label_names",
    "label_codes",
    "n_classes",
    "is_binary_task",
    "map_physionet_reference",
    "all_namespaced_ids",
]


# ---------------------------------------------------------------------------
# datasets (T05.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Identity and audited facts for one dataset family."""

    dataset_id: str
    canonical_name: str
    short_name: str
    paths_key: str
    native_fs: int
    n_records: int
    n_subjects: int | None
    subject_ids: str          # "native" | "derived" | "partial" | "none"
    tasks: tuple[str, ...]

    def __str__(self) -> str:
        return self.dataset_id + " (" + self.short_name + ")"


# Counts are the AUDITED figures from the real files (2026-08-22), not the
# figures printed in the source documents. Where they disagree the discrepancy
# is reported, not silently substituted -- see Docs/note.md.
DATASETS: Mapping[str, DatasetInfo] = MappingProxyType(
    {
        "D1": DatasetInfo(
            dataset_id="D1",
            canonical_name="PhysioNet/CinC Challenge 2016",
            short_name="PhysioNet 2016",
            paths_key="dataset.d1_physionet",
            native_fs=2000,
            n_records=3541,          # 3,240 training (a-f) + 301 validation
            n_subjects=None,         # only partially recoverable
            subject_ids="partial",
            tasks=("binary",),
        ),
        "D2": DatasetInfo(
            dataset_id="D2",
            canonical_name="PASCAL Classifying Heart Sounds Challenge, Dataset A",
            short_name="PASCAL set_a",
            paths_key="dataset.d2_pascal_a",
            native_fs=44100,
            n_records=124,           # labeled; 52 unlabelled excluded
            n_subjects=None,
            subject_ids="none",      # timestamp-only filenames carry no subject
            tasks=("pascal_a",),
        ),
        "D3": DatasetInfo(
            dataset_id="D3",
            canonical_name="PASCAL Classifying Heart Sounds Challenge, Dataset B",
            short_name="PASCAL set_b",
            paths_key="dataset.d3_pascal_b",
            native_fs=4000,
            n_records=461,           # labeled; 195 unlabelled excluded
            n_subjects=167,          # filename-derived
            subject_ids="derived",
            tasks=("pascal_b",),
        ),
        "D4": DatasetInfo(
            dataset_id="D4",
            canonical_name="CirCor DigiScope Phonocardiogram Dataset (PhysioNet 2022)",
            short_name="CirCor 2022",
            paths_key="dataset.d4_circor",
            native_fs=4000,
            n_records=3163,
            n_subjects=942,
            subject_ids="native",
            tasks=("circor_murmur", "circor_outcome"),
        ),
    }
)

DATASET_IDS: tuple[str, ...] = tuple(DATASETS)


# ---------------------------------------------------------------------------
# tasks (T05.1)
# ---------------------------------------------------------------------------

TASK_BINARY = "binary"
TASK_PASCAL_A = "pascal_a"
TASK_PASCAL_B = "pascal_b"
TASK_CIRCOR_MURMUR = "circor_murmur"
TASK_CIRCOR_OUTCOME = "circor_outcome"

TASKS: tuple[str, ...] = (
    TASK_BINARY,
    TASK_PASCAL_A,
    TASK_PASCAL_B,
    TASK_CIRCOR_MURMUR,
    TASK_CIRCOR_OUTCOME,
)

TASK_DATASET: Mapping[str, str] = MappingProxyType(
    {
        TASK_BINARY: "D1",
        TASK_PASCAL_A: "D2",
        TASK_PASCAL_B: "D3",
        TASK_CIRCOR_MURMUR: "D4",
        TASK_CIRCOR_OUTCOME: "D4",
    }
)

TASK_KIND: Mapping[str, str] = MappingProxyType(
    {
        TASK_BINARY: "binary",
        TASK_PASCAL_A: "multiclass",
        TASK_PASCAL_B: "multiclass",
        TASK_CIRCOR_MURMUR: "multiclass",   # 3-class headline variant
        TASK_CIRCOR_OUTCOME: "binary",
    }
)


# ---------------------------------------------------------------------------
# label maps (T05.2 - T05.5)
# ---------------------------------------------------------------------------

# T05.2 -- binary. PhysioNet ships -1/1; see PHYSIONET_REFERENCE_MAP below.
_BINARY = {"normal": 0, "abnormal": 1}

# T05.3 -- PASCAL A, four classes.
# `artifact` is a RECORDING-QUALITY label, not a cardiac class. A model trained
# on this vocabulary is not a four-class cardiac classifier and must never be
# described as one (T66.6).
_PASCAL_A = {"normal": 0, "murmur": 1, "extrahls": 2, "artifact": 3}

# T05.4 -- PASCAL B, three classes. NOT merged with PASCAL A: `extrastole` and
# `extrahls` are different phenomena that happen to share the code 2.
_PASCAL_B = {"normal": 0, "murmur": 1, "extrastole": 2}

# T05.5 -- CirCor. Capitalised because that is exactly how the values appear in
# the source files; the loaders match on them literally.
_CIRCOR_MURMUR = {"Absent": 0, "Present": 1, "Unknown": 2}
_CIRCOR_OUTCOME = {"Normal": 0, "Abnormal": 1}

LABEL_MAPS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        TASK_BINARY: MappingProxyType(dict(_BINARY)),
        TASK_PASCAL_A: MappingProxyType(dict(_PASCAL_A)),
        TASK_PASCAL_B: MappingProxyType(dict(_PASCAL_B)),
        TASK_CIRCOR_MURMUR: MappingProxyType(dict(_CIRCOR_MURMUR)),
        TASK_CIRCOR_OUTCOME: MappingProxyType(dict(_CIRCOR_OUTCOME)),
    }
)

INVERSE_LABEL_MAPS: Mapping[str, Mapping[int, str]] = MappingProxyType(
    {
        task: MappingProxyType({code: name for name, code in mapping.items()})
        for task, mapping in LABEL_MAPS.items()
    }
)

# T05.2 -- PhysioNet REFERENCE.csv encodes normal as -1 and abnormal as 1.
# The original value is preserved in metadata by map_physionet_reference().
PHYSIONET_REFERENCE_MAP: Mapping[int, int] = MappingProxyType({-1: 0, 1: 1})

# T05.5 -- CirCor auscultation locations. Phc has n=4 across the whole corpus
# and is flagged as statistically uninformative rather than reported (T70.4).
CIRCOR_LOCATIONS: tuple[str, ...] = ("AV", "PV", "TV", "MV", "Phc")

# Folder names in dataset/Heartbeat_Sound/, the authoritative PASCAL label
# source. `unlabel` is not a class -- it marks records excluded from supervised
# tracks but kept in metadata (T12.5).
UNLABELED_CLASS = "unlabel"
PASCAL_CLASS_FOLDERS: tuple[str, ...] = (
    "artifact",
    "extrahls",
    "extrastole",
    "murmur",
    "normal",
    UNLABELED_CLASS,
)


# ---------------------------------------------------------------------------
# accessors
# ---------------------------------------------------------------------------


def _require_task(task: str) -> Mapping[str, int]:
    mapping = LABEL_MAPS.get(task)
    if mapping is None:
        raise KeyError(
            "unknown task " + repr(task) + " -- must be one of: " + ", ".join(TASKS)
        )
    return mapping


def label_names(task: str) -> tuple[str, ...]:
    """Class names for ``task``, ordered by code."""
    mapping = _require_task(task)
    return tuple(name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1]))


def label_codes(task: str) -> tuple[int, ...]:
    """Class codes for ``task``, ascending."""
    return tuple(sorted(_require_task(task).values()))


def n_classes(task: str) -> int:
    return len(_require_task(task))


def is_binary_task(task: str) -> bool:
    return TASK_KIND[task] == "binary" if task in TASK_KIND else False


def encode_label(task: str, name: str) -> int:
    """Name -> code, within one task."""
    mapping = _require_task(task)
    if name not in mapping:
        raise KeyError(
            "label " + repr(name) + " is not in task " + repr(task)
            + " (valid: " + ", ".join(label_names(task)) + ")"
        )
    return mapping[name]


def decode_label(task: str, code: int) -> str:
    """Code -> name, **within one task**.

    There is deliberately no single-argument ``decode(code)``: a bare integer
    means different things in different tasks, so decoding without naming the
    task has no correct answer.
    """
    _require_task(task)
    inverse = INVERSE_LABEL_MAPS[task]
    if code not in inverse:
        raise KeyError(
            "code " + repr(code) + " is not valid for task " + repr(task)
            + " (valid: " + ", ".join(str(c) for c in label_codes(task)) + ")"
        )
    return inverse[code]


def namespaced_id(task: str, name: str) -> str:
    """Globally unique label id, e.g. ``pascal_a:murmur``.

    Bare class names collide across tasks (``murmur`` is in both PASCAL A and
    PASCAL B; ``normal`` is in three). Namespacing is what makes a label safe to
    put in a shared table, figure legend or evidence-index row.
    """
    encode_label(task, name)   # validates both task and name
    return task + ":" + name


def all_namespaced_ids() -> tuple[str, ...]:
    """Every label across every task, namespaced. Guaranteed unique."""
    return tuple(
        namespaced_id(task, name) for task in TASKS for name in label_names(task)
    )


def map_physionet_reference(value: int) -> dict[str, Any]:
    """Map a PhysioNet REFERENCE.csv value to the binary label space (T05.2).

    Returns the mapped code alongside the original value, so the raw -1/1 is
    preserved in metadata rather than being thrown away by the conversion.
    """
    # `int(1.5)` silently truncates to 1, which would turn a malformed value into
    # a confident "abnormal". A float is accepted only when it is exactly
    # integral -- pandas reads a REFERENCE.csv column as float64 whenever the
    # column contains a NaN, so -1.0 is legitimate and must still work.
    if isinstance(value, bool):
        raise ValueError("PhysioNet reference value must not be a bool: " + repr(value))
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                "PhysioNet reference value " + repr(value) + " is not integral -- "
                "refusing to truncate it into a binary label"
            )
        raw = int(value)
    elif isinstance(value, int):
        raw = int(value)
    else:
        try:
            raw = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "PhysioNet reference value is not an integer: " + repr(value)
            ) from exc

    if raw not in PHYSIONET_REFERENCE_MAP:
        raise ValueError(
            "PhysioNet reference value " + repr(raw) + " is not -1 or 1 -- refusing "
            "to guess a binary label from an unexpected code"
        )

    code = PHYSIONET_REFERENCE_MAP[raw]
    return {
        "task": TASK_BINARY,
        "binary_label": code,
        "binary_label_name": decode_label(TASK_BINARY, code),
        "original_value": raw,
        "original_scheme": "physionet_reference_csv(-1=normal, 1=abnormal)",
    }
