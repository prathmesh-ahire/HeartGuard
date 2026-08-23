"""Environment tests (Phase 06, task T06.4).

Asserts every pinned package is installed **at its pinned version** and actually
imports. Installing and importing are different things on Windows: numba and
llvmlite against the wrong numpy install cleanly and then die at import.

This duplicates part of ``scripts/verify_env.py`` on purpose. That script is a
human-facing pre-flight check with advisories and colour; this is the machine
gate that runs in CI on every push. Coupling them would mean a change to the
script's reporting could quietly disable the CI check.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

REQ_FILES = (
    "requirements.txt",
    "requirements-extra.txt",
    "requirements-api.txt",
    "requirements-report.txt",
    "requirements-dev.txt",
)

# Distribution name -> importable module name, where the two differ.
IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "scikit-optimize": "skopt",
    "PyWavelets": "pywt",
    "PyYAML": "yaml",
    "python-multipart": "multipart",
    "python-docx": "docx",
    "pytest-cov": "pytest_cov",
}

PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==(\S+)$")

EXPECTED_PYTHON = "3.11.9"


def _parse(path: Path) -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN_RE.match(line)
        assert match is not None, f"{path.name}: unparseable pin {line!r}"
        pins.append((match.group(1), match.group(2)))
    return pins


def _all_pins() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for name in REQ_FILES:
        path = ROOT / name
        if path.is_file():
            out += [(name, dist, pin) for dist, pin in _parse(path)]
    return out


ALL_PINS = _all_pins()
PIN_IDS = [f"{dist}" for _, dist, _ in ALL_PINS]


# ---------------------------------------------------------------------------
# interpreter
# ---------------------------------------------------------------------------


def test_python_version_is_locked():
    import platform

    assert platform.python_version() == EXPECTED_PYTHON, (
        "this project is locked to Python " + EXPECTED_PYTHON
    )


def test_running_from_a_virtualenv():
    assert sys.prefix != sys.base_prefix, "not running inside a virtual environment"


# ---------------------------------------------------------------------------
# requirements files
# ---------------------------------------------------------------------------


def test_all_requirements_files_exist():
    for name in REQ_FILES:
        assert (ROOT / name).is_file(), f"{name} is missing"


def test_every_requirement_is_pinned_exactly():
    """No ranges, no bare names. A floating pin makes a run unreproducible."""
    assert ALL_PINS, "no pins were parsed from any requirements file"
    for source, dist, pin in ALL_PINS:
        assert re.fullmatch(r"[0-9][0-9A-Za-z.\-+]*", pin), (
            f"{source}: {dist} pin {pin!r} does not look like an exact version"
        )


def test_no_distribution_is_pinned_twice_at_different_versions():
    seen: dict[str, tuple[str, str]] = {}
    for source, dist, pin in ALL_PINS:
        key = dist.lower()
        if key in seen:
            prev_source, prev_pin = seen[key]
            assert prev_pin == pin, (
                f"{dist} pinned to {prev_pin} in {prev_source} but {pin} in {source}"
            )
        else:
            seen[key] = (source, pin)


# ---------------------------------------------------------------------------
# installed versions and imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "dist", "pin"), ALL_PINS, ids=PIN_IDS
)
def test_pinned_package_is_installed_at_that_version(source, dist, pin):
    try:
        installed = version(dist)
    except PackageNotFoundError:
        pytest.fail(f"{dist} is pinned to {pin} in {source} but is not installed")
    assert installed == pin, f"{dist}: installed {installed}, pinned {pin} ({source})"


@pytest.mark.parametrize(
    ("source", "dist", "pin"), ALL_PINS, ids=PIN_IDS
)
def test_pinned_package_imports(source, dist, pin):
    module = IMPORT_NAME.get(dist, dist.replace("-", "_"))
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001 - any import failure is a failure
        pytest.fail(f"import {module} ({dist}) raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# the specific combinations that break on Windows + Python 3.11
# ---------------------------------------------------------------------------


def test_numba_and_numpy_are_compatible():
    """librosa's JIT paths die at import when numba and numpy disagree."""
    import numba
    import numpy as np

    assert numba.__version__
    assert np.__version__


def test_librosa_core_paths_work_at_the_2khz_target_rate():
    """librosa defaults assume 22 kHz speech; this project runs at 2 kHz."""
    import warnings

    import librosa
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.sin(2 * np.pi * 50 * np.arange(4000) / 2000).astype(np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=2000, n_mfcc=13, fmin=20, fmax=400, n_mels=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=2000, n_chroma=12)

    assert mfcc.shape[0] == 13
    assert chroma.shape[0] == 12
    assert np.isfinite(mfcc).all()
    assert np.isfinite(chroma).all()


def test_pywavelets_supports_the_locked_wavelet():
    """db4, 5 levels -> 6 sub-bands. The locked DWT configuration."""
    import numpy as np
    import pywt

    assert "db4" in pywt.wavelist(kind="discrete")
    coeffs = pywt.wavedec(np.random.default_rng(42).normal(size=4000), "db4", level=5)
    assert len(coeffs) == 6


def test_skopt_runs_not_just_imports():
    """scikit-optimize imports cleanly and then fails at .fit() on some sklearn
    versions. An import check is not sufficient evidence for this package."""
    import warnings

    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from skopt import BayesSearchCV
    from skopt.space import Real

    X, y = make_classification(n_samples=40, n_features=5, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = BayesSearchCV(
            SVC(), {"C": Real(0.1, 10, "log-uniform")}, n_iter=2, cv=2, random_state=42
        )
        search.fit(X, y)
    assert 0.1 <= search.best_params_["C"] <= 10


def test_external_gradient_boosting_available():
    """M8 has a capability check (T49.3); both are installed on this machine."""
    import lightgbm
    import xgboost

    assert xgboost.__version__
    assert lightgbm.__version__


def test_no_cuda_is_expected_not_a_failure():
    """CPU-only by hardware. Any task assuming a GPU is wrong for this machine."""
    import xgboost

    assert xgboost.__version__, "xgboost must work on CPU"
