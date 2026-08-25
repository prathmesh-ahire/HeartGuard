"""Label vocabulary tests (Phase 05, task T05.6).

Asserts that every label map is bijective and that no task's label vocabulary
overlaps another's.

**What "non-overlapping" means here, and why.** The bare class *names* collide
by design -- ``normal`` appears in three tasks, ``murmur`` in two -- and so do
the integer *codes*, since every task starts at 0. Those collisions are a
property of the source datasets, not a defect to be renamed away.

The requirement that matters, and the one enforced below, is that a label can
never be silently reinterpreted under the wrong task. That is established by:

1. each map being bijective, so name and code agree one-to-one within a task;
2. every label being addressable only as a ``(task, ...)`` pair, with
   ``namespaced_id`` producing globally unique ids that ARE disjoint across
   tasks;
3. decoding refusing to work without a task; and
4. an explicit test that the *same code means different things in different
   tasks* -- the collision is asserted to exist, which is precisely why rule 4
   forbids merging the label spaces.

Point 4 is the one worth keeping. If a future change ever made the vocabularies
genuinely interchangeable, that test would fail and force the question rather
than letting a quiet merge through.
"""

from __future__ import annotations

import itertools

import pytest

from src.utils import constants as K

# ---------------------------------------------------------------------------
# task and dataset identity (T05.1)
# ---------------------------------------------------------------------------


def test_five_tasks_exactly():
    assert K.TASKS == (
        "binary",
        "pascal_a",
        "pascal_b",
        "circor_murmur",
        "circor_outcome",
    )
    assert len(K.TASKS) == len(set(K.TASKS)) == 5


def test_four_datasets_exactly():
    assert K.DATASET_IDS == ("D1", "D2", "D3", "D4")
    for dataset_id, info in K.DATASETS.items():
        assert info.dataset_id == dataset_id
        assert info.canonical_name and info.short_name
        assert info.native_fs in (2000, 4000, 44100)
        assert info.subject_ids in {"native", "derived", "partial", "none"}


def test_every_task_belongs_to_exactly_one_dataset():
    assert set(K.TASK_DATASET) == set(K.TASKS)
    for task, dataset_id in K.TASK_DATASET.items():
        assert dataset_id in K.DATASETS
        assert task in K.DATASETS[dataset_id].tasks


def test_every_dataset_task_is_a_known_task():
    declared = {t for info in K.DATASETS.values() for t in info.tasks}
    assert declared == set(K.TASKS)


# ---------------------------------------------------------------------------
# bijectivity (T05.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", K.TASKS)
def test_label_map_is_bijective(task):
    mapping = K.LABEL_MAPS[task]
    names = list(mapping.keys())
    codes = list(mapping.values())

    assert len(names) == len(set(names)), f"{task}: duplicate class name"
    assert len(codes) == len(set(codes)), f"{task}: two classes share a code"
    assert len(names) == len(codes)

    inverse = K.INVERSE_LABEL_MAPS[task]
    assert len(inverse) == len(mapping)
    for name, code in mapping.items():
        assert inverse[code] == name
        assert K.encode_label(task, name) == code
        assert K.decode_label(task, code) == name


@pytest.mark.parametrize("task", K.TASKS)
def test_codes_are_contiguous_from_zero(task):
    """0..n-1 with no gaps -- sklearn assumes this for class indices."""
    assert K.label_codes(task) == tuple(range(K.n_classes(task)))


@pytest.mark.parametrize("task", K.TASKS)
def test_label_names_ordered_by_code(task):
    names = K.label_names(task)
    assert [K.encode_label(task, n) for n in names] == sorted(
        K.encode_label(task, n) for n in names
    )


def test_expected_class_counts():
    assert K.n_classes("binary") == 2
    assert K.n_classes("pascal_a") == 4
    assert K.n_classes("pascal_b") == 3
    assert K.n_classes("circor_murmur") == 3
    assert K.n_classes("circor_outcome") == 2


def test_exact_label_maps():
    """The maps specified in T05.2-T05.5, verbatim."""
    assert dict(K.LABEL_MAPS["binary"]) == {"normal": 0, "abnormal": 1}
    assert dict(K.LABEL_MAPS["pascal_a"]) == {
        "normal": 0, "murmur": 1, "extrahls": 2, "artifact": 3
    }
    assert dict(K.LABEL_MAPS["pascal_b"]) == {"normal": 0, "murmur": 1, "extrastole": 2}
    assert dict(K.LABEL_MAPS["circor_murmur"]) == {"Absent": 0, "Present": 1, "Unknown": 2}
    assert dict(K.LABEL_MAPS["circor_outcome"]) == {"Normal": 0, "Abnormal": 1}


# ---------------------------------------------------------------------------
# non-overlap across tasks (T05.6 / T05.7)
# ---------------------------------------------------------------------------


def test_namespaced_ids_are_globally_unique():
    """The addressable vocabularies do not overlap across tasks."""
    ids = K.all_namespaced_ids()
    assert len(ids) == len(set(ids))
    assert len(ids) == sum(K.n_classes(t) for t in K.TASKS) == 14


def test_no_two_tasks_share_a_namespaced_label():
    for a, b in itertools.combinations(K.TASKS, 2):
        va = {K.namespaced_id(a, n) for n in K.label_names(a)}
        vb = {K.namespaced_id(b, n) for n in K.label_names(b)}
        assert va.isdisjoint(vb), f"{a} and {b} share a namespaced label"


def test_bare_names_DO_collide_which_is_why_namespacing_exists():
    """Guards the reason rule 4 exists: bare names are not task-identifying."""
    assert "normal" in K.LABEL_MAPS["binary"]
    assert "normal" in K.LABEL_MAPS["pascal_a"]
    assert "normal" in K.LABEL_MAPS["pascal_b"]
    assert "murmur" in K.LABEL_MAPS["pascal_a"]
    assert "murmur" in K.LABEL_MAPS["pascal_b"]


def test_same_code_means_different_things_across_tasks():
    """The sharpest collision: code 2 is a different phenomenon in each task.

    ``extrahls`` (an extra heart sound) and ``extrastole`` (an extra systole)
    are not the same finding. Merging PASCAL A and B would silently equate them.
    """
    assert K.decode_label("pascal_a", 2) == "extrahls"
    assert K.decode_label("pascal_b", 2) == "extrastole"
    assert K.decode_label("circor_murmur", 2) == "Unknown"

    assert K.decode_label("binary", 1) == "abnormal"
    assert K.decode_label("pascal_a", 1) == "murmur"
    assert K.decode_label("circor_murmur", 1) == "Present"
    assert K.decode_label("circor_outcome", 1) == "Abnormal"


def test_a_code_valid_in_one_task_can_be_invalid_in_another():
    """Code 3 exists only in PASCAL A. Every other task must reject it."""
    assert K.decode_label("pascal_a", 3) == "artifact"
    for task in ("binary", "pascal_b", "circor_murmur", "circor_outcome"):
        with pytest.raises(KeyError):
            K.decode_label(task, 3)


def test_cross_task_encode_is_rejected():
    """A PASCAL B class must not encode under PASCAL A, and vice versa."""
    with pytest.raises(KeyError):
        K.encode_label("pascal_a", "extrastole")
    with pytest.raises(KeyError):
        K.encode_label("pascal_b", "extrahls")
    with pytest.raises(KeyError):
        K.encode_label("pascal_b", "artifact")
    with pytest.raises(KeyError):
        K.encode_label("binary", "murmur")
    with pytest.raises(KeyError):
        K.encode_label("circor_outcome", "Unknown")
    # CirCor values are capitalised in the source files; lowercase must not slip through.
    with pytest.raises(KeyError):
        K.encode_label("circor_murmur", "absent")


def test_unknown_task_is_rejected():
    for bad in ("pascal", "murmur", "PASCAL_A", "", "circor"):
        with pytest.raises(KeyError):
            K.encode_label(bad, "normal")
        with pytest.raises(KeyError):
            K.decode_label(bad, 0)


# ---------------------------------------------------------------------------
# PhysioNet -1/1 mapping (T05.2)
# ---------------------------------------------------------------------------


def test_physionet_reference_mapping():
    assert dict(K.PHYSIONET_REFERENCE_MAP) == {-1: 0, 1: 1}

    normal = K.map_physionet_reference(-1)
    assert normal["binary_label"] == 0
    assert normal["binary_label_name"] == "normal"
    assert normal["original_value"] == -1

    abnormal = K.map_physionet_reference(1)
    assert abnormal["binary_label"] == 1
    assert abnormal["binary_label_name"] == "abnormal"
    assert abnormal["original_value"] == 1


def test_physionet_original_value_is_preserved():
    """T05.2 requires the raw -1/1 to survive the conversion into metadata."""
    for raw in (-1, 1):
        result = K.map_physionet_reference(raw)
        assert result["original_value"] == raw
        assert "physionet" in result["original_scheme"]


def test_physionet_rejects_unexpected_codes():
    """0 is the dangerous one: it is a valid *output* code and an invalid input."""
    for bad in (0, 2, -2, 99):
        with pytest.raises(ValueError):
            K.map_physionet_reference(bad)
    for junk in ("normal", None, ""):
        with pytest.raises((ValueError, TypeError)):
            K.map_physionet_reference(junk)


def test_physionet_rejects_non_integral_floats_rather_than_truncating():
    """``int(1.5)`` is 1, which would turn a malformed value into "abnormal"."""
    for bad in (1.5, -0.5, 0.9, -1.2):
        with pytest.raises(ValueError):
            K.map_physionet_reference(bad)
    with pytest.raises(ValueError):
        K.map_physionet_reference(float("nan"))


def test_physionet_accepts_integral_floats_from_pandas():
    """pandas reads a column as float64 as soon as it contains a NaN, so -1.0
    and 1.0 reach this function legitimately and must still map."""
    assert K.map_physionet_reference(-1.0)["binary_label"] == 0
    assert K.map_physionet_reference(1.0)["binary_label"] == 1
    assert K.map_physionet_reference(-1.0)["original_value"] == -1


def test_physionet_rejects_bools():
    """``True == 1`` in Python, so a bool would map to "abnormal" unchallenged."""
    for bad in (True, False):
        with pytest.raises(ValueError):
            K.map_physionet_reference(bad)


def test_physionet_accepts_numpy_integers():
    """Loaders hand over numpy scalars, not Python ints."""
    np = pytest.importorskip("numpy")
    assert K.map_physionet_reference(np.int64(-1))["binary_label"] == 0
    assert K.map_physionet_reference(np.int64(1))["binary_label"] == 1
    assert K.map_physionet_reference(np.float64(-1.0))["binary_label"] == 0
    with pytest.raises(ValueError):
        K.map_physionet_reference(np.float64(1.5))


# ---------------------------------------------------------------------------
# CirCor locations and PASCAL folders (T05.5)
# ---------------------------------------------------------------------------


def test_circor_locations():
    assert K.CIRCOR_LOCATIONS == ("AV", "PV", "TV", "MV", "Phc")
    assert len(set(K.CIRCOR_LOCATIONS)) == 5


def test_pascal_class_folders_cover_both_label_spaces():
    folders = set(K.PASCAL_CLASS_FOLDERS)
    assert folders >= set(K.LABEL_MAPS["pascal_a"])
    assert folders >= set(K.LABEL_MAPS["pascal_b"])
    assert K.UNLABELED_CLASS in folders
    # `unlabel` is a folder, never a class in either task.
    assert K.UNLABELED_CLASS not in K.LABEL_MAPS["pascal_a"]
    assert K.UNLABELED_CLASS not in K.LABEL_MAPS["pascal_b"]


# ---------------------------------------------------------------------------
# immutability
# ---------------------------------------------------------------------------


def test_label_maps_cannot_be_mutated_at_runtime():
    with pytest.raises(TypeError):
        K.LABEL_MAPS["pascal_a"]["sneaky"] = 4        # type: ignore[index]
    with pytest.raises(TypeError):
        K.LABEL_MAPS["new_task"] = {}                  # type: ignore[index]
    with pytest.raises(TypeError):
        K.INVERSE_LABEL_MAPS["binary"][2] = "third"    # type: ignore[index]
    assert K.n_classes("pascal_a") == 4


def test_dataset_info_is_frozen():
    import dataclasses

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        K.DATASETS["D1"].native_fs = 8000              # type: ignore[misc]
