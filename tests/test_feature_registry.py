"""The registry holds exactly 138 names in one fixed order (T31.6, T31.7).

The two assertions that matter are the count and the *stability* of the order.
A count check alone would pass on a registry whose columns shuffle between runs,
and that failure is invisible: no error, no wrong-looking number, just a model
trained on one column meaning and explained under another. So the order is
checked across two separate interpreter processes with different hash seeds --
the condition under which a set- or dict-derived ordering actually breaks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.feature_extraction import base as fb
from src.feature_extraction import registry as reg

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# composition (T31.6)
# ---------------------------------------------------------------------------


def test_registry_holds_exactly_138_features():
    assert len(reg.FEATURE_SPECS) == 138
    assert len(reg.FEATURE_NAMES) == 138
    assert reg.EXPECTED_TOTAL == 138


def test_no_duplicate_names():
    names = list(reg.FEATURE_NAMES)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == [], "duplicate feature names: " + ", ".join(duplicates)
    assert len(set(names)) == 138


def test_family_counts_match_the_locked_composition():
    assert reg.family_counts() == {
        "time": 24,
        "frequency": 22,
        "mfcc": 39,
        "chroma": 24,
        "dwt": 24,
        "envelope": 5,
    }
    assert sum(reg.family_counts().values()) == 138


def test_registry_agrees_with_features_yaml():
    problems = reg.validate_against_config()
    assert problems == [], "\n".join(problems)


def test_every_spec_is_fully_populated():
    for spec in reg.FEATURE_SPECS:
        assert spec.name.strip(), "empty name at index " + str(spec.index)
        assert spec.family in reg.FAMILY_ORDER
        assert spec.extractor.strip()
        assert spec.equation.strip(), spec.name + " has no equation reference"
        assert spec.description.strip(), spec.name + " has no description"


def test_indices_are_contiguous_and_match_position():
    for position, spec in enumerate(reg.FEATURE_SPECS):
        assert spec.index == position
        assert reg.index_of(spec.name) == position


def test_families_occupy_contiguous_blocks_in_declared_order():
    """Column order is family-blocked, so a family is one slice, not scattered."""
    seen: list[str] = []
    for spec in reg.FEATURE_SPECS:
        if not seen or seen[-1] != spec.family:
            seen.append(spec.family)
    assert tuple(seen) == reg.FAMILY_ORDER


def test_names_are_prefixed_by_their_family():
    prefixes = {
        "time": "time_",
        "frequency": "freq_",
        "mfcc": "mfcc_",
        "chroma": "chroma_",
        "dwt": "dwt_",
        "envelope": "env_",
    }
    for spec in reg.FEATURE_SPECS:
        assert spec.name.startswith(prefixes[spec.family]), spec.name


def test_lookup_helpers_round_trip():
    for spec in reg.FEATURE_SPECS:
        assert reg.spec_for(spec.name) is spec
        assert reg.family_of(spec.name) == spec.family
    with pytest.raises(reg.RegistryError):
        reg.spec_for("time_not_a_feature")
    with pytest.raises(reg.RegistryError):
        reg.feature_names("spectrogram")


def test_as_records_is_138_rows_ready_for_fe01():
    records = reg.as_records()
    assert len(records) == 138
    assert set(records[0]) == {
        "index",
        "name",
        "family",
        "extractor",
        "equation",
        "unit",
        "description",
    }


# ---------------------------------------------------------------------------
# stable ordering (T31.3, T31.7)
# ---------------------------------------------------------------------------


_ORDER_PROBE = (
    "import sys; sys.path.insert(0, r'{root}');"
    "from src.feature_extraction import registry as r;"
    "print(r.registry_fingerprint());"
    "print('|'.join(r.FEATURE_NAMES))"
)


def _probe_order(seed: str) -> tuple[str, list[str]]:
    """Read the name order out of a fresh interpreter with a given hash seed."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    out = subprocess.run(
        [sys.executable, "-c", _ORDER_PROBE.format(root=str(ROOT))],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        check=True,
    ).stdout.splitlines()
    return out[0].strip(), out[1].strip().split("|")


def test_order_is_identical_across_two_interpreter_sessions():
    """Two processes, two hash seeds, same 138 names in the same order.

    ``PYTHONHASHSEED`` differs between the runs on purpose: it is what makes a
    set-derived or dict-key-derived ordering vary, and it is randomized by
    default, so an unstable registry would fail here but pass an in-process
    check.
    """
    first_hash, first_names = _probe_order("0")
    second_hash, second_names = _probe_order("12345")

    assert first_names == second_names
    assert first_hash == second_hash
    assert first_names == list(reg.FEATURE_NAMES)
    assert first_hash == reg.registry_fingerprint()
    assert len(first_names) == 138


def test_fingerprint_is_a_sha256_of_the_name_list():
    import hashlib

    expected = hashlib.sha256("\n".join(reg.FEATURE_NAMES).encode("utf-8")).hexdigest()
    assert reg.registry_fingerprint() == expected
    assert len(reg.registry_fingerprint()) == 64


# ---------------------------------------------------------------------------
# extractor binding
# ---------------------------------------------------------------------------


class _Dummy(fb.BaseFeatureExtractor):
    family = "envelope"
    name = "dummy_envelope"

    def __init__(self, names, values=None, boom=None):
        super().__init__(None)
        self._names = tuple(names)
        self._values = values
        self._boom = boom

    def feature_names(self):
        return self._names

    def _compute(self, signal, fs, flags):
        if self._boom is not None:
            raise self._boom
        flags.append("dummy")
        return self._values


def test_register_extractor_rejects_a_mismatched_name_list():
    good = reg.feature_names("envelope")
    with pytest.raises(reg.RegistryError):
        reg.register_extractor(_Dummy(good[:-1]))
    with pytest.raises(reg.RegistryError):
        reg.register_extractor(_Dummy(tuple(reversed(good))))


def test_unbuilt_family_raises_a_named_error_not_an_import_error():
    """Phases 32-37 land one at a time; an unbuilt family must say so clearly."""
    missing = [
        family
        for family in reg.FAMILY_ORDER
        if not (ROOT / (reg.FAMILY_MODULES[family].replace(".", "/") + ".py")).is_file()
    ]
    for family in missing:
        with pytest.raises(reg.ExtractorNotRegistered):
            reg.get_extractor(family)


# ---------------------------------------------------------------------------
# the NaN policy and shape contract (T31.4)
# ---------------------------------------------------------------------------


def test_extractor_returns_all_nan_instead_of_raising():
    names = reg.feature_names("envelope")
    result = _Dummy(names, boom=RuntimeError("synthetic failure")).extract(
        np.ones(100), 2000, record_uid="R-1"
    )
    assert result.failed is True
    assert "synthetic failure" in (result.error or "")
    assert list(result.values) == list(names)
    assert result.n_missing == len(names)
    assert result.record_uid == "R-1"


def test_a_missing_key_becomes_nan_and_a_stray_key_fails_the_family():
    names = reg.feature_names("envelope")
    partial = dict.fromkeys(names[:-1], 1.0)
    result = _Dummy(names, values=partial).extract(np.ones(100), 2000)
    assert result.failed is False
    assert list(result.values) == list(names)
    assert np.isnan(result.values[names[-1]])
    assert result.flags == ("dummy",)

    stray = dict.fromkeys(names, 1.0)
    stray["env_not_registered"] = 1.0
    bad = _Dummy(names, values=stray).extract(np.ones(100), 2000)
    assert bad.failed is True
    assert "unregistered" in (bad.error or "")


def test_degenerate_input_is_reported_not_raised():
    names = reg.feature_names("envelope")
    values = dict.fromkeys(names, 0.0)
    for bad in (np.array([]), np.array([1.0, np.nan, 2.0]), np.ones((3, 3))):
        result = _Dummy(names, values=values).extract(bad, 2000)
        assert result.failed is True
        assert result.n_missing == len(names)


def test_timings_accumulate_per_family():
    names = reg.feature_names("envelope")
    values = dict.fromkeys(names, 0.0)
    fb.reset_timings()
    try:
        extractor = _Dummy(names, values=values)
        for _ in range(3):
            extractor.extract(np.ones(1000), 2000)
        _Dummy(names, boom=ValueError("x")).extract(np.ones(1000), 2000)

        table = fb.timing_table()
        assert set(table) == {"envelope"}
        entry = table["envelope"]
        assert entry.calls == 4
        assert entry.failures == 1
        assert entry.total_seconds > 0
        assert entry.min_seconds <= entry.mean_seconds <= entry.max_seconds
    finally:
        fb.reset_timings()
