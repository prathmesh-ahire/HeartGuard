"""Shared test fixtures (Phase 06, task T06.2).

CLI options and marker-driven skipping live in the root ``conftest.py``;
``pytest_addoption`` is only honoured there.

**Every fixture that writes goes to ``tmp_path``.** No test writes into the real
``outputs/`` tree. A test that appends to the real evidence index or the real run
manifest pollutes deliverables with rows that point at temporary files, and
Phase 102 asserts every evidence row resolves to a real file on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.make_synthetic_pcg import (  # noqa: E402
    DEFAULT_FS,
    SyntheticPCG,
    make_edge_case_signals,
    make_synthetic_pcg,
)

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    """The real dataset tree. Only meaningful in ``needs_data`` tests."""
    return ROOT / "dataset"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """A throwaway ``outputs/``-shaped tree, one per test.

    Mirrors the real layout so code that resolves ``outputs.<section>`` has
    somewhere to write without touching the project's actual deliverables.
    """
    out = tmp_path / "outputs"
    for section in (
        "00_evidence_index",
        "01_dataset_audit",
        "02_preprocessing",
        "03_features",
        "04_models",
        "logs",
        "configs",
    ):
        (out / section).mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """All five configuration files, loaded and validated."""
    from src.utils.config import load_all

    return load_all()


@pytest.fixture(scope="session")
def signal_config(config: dict[str, Any]) -> Any:
    return config["signal"]


@pytest.fixture(scope="session")
def features_config(config: dict[str, Any]) -> Any:
    return config["features"]


@pytest.fixture(scope="session")
def paths_config(config: dict[str, Any]) -> Any:
    return config["paths"]


# ---------------------------------------------------------------------------
# synthetic signals
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pcg() -> SyntheticPCG:
    """A 5-second, 72 bpm synthetic PCG at the 2 kHz target rate.

    Deterministic: seed 42, identical on every call and every machine.
    """
    return make_synthetic_pcg(duration_sec=5.0, fs=DEFAULT_FS, heart_rate_bpm=72.0)


@pytest.fixture
def synthetic_signal(synthetic_pcg: SyntheticPCG) -> np.ndarray:
    """Just the samples, for tests that do not need the ground truth."""
    return synthetic_pcg.signal


@pytest.fixture
def synthetic_pcg_factory():
    """Build synthetic PCGs with custom parameters inside a test."""
    return make_synthetic_pcg


@pytest.fixture
def edge_case_signals() -> dict[str, np.ndarray]:
    """Degenerate and boundary signals: shortest real, silence, constant, clipped."""
    return make_edge_case_signals(DEFAULT_FS)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_every_test():
    """Reseed before every test.

    Autouse and deliberate: without it, a test that consumes randomness changes
    what the *next* test sees, so a suite can pass in one order and fail in
    another. That failure mode is miserable to diagnose and trivial to prevent.
    """
    from src.utils.seed import set_global_seed

    set_global_seed(42)
    yield


@pytest.fixture
def rng() -> np.random.Generator:
    from src.utils.seed import get_rng

    return get_rng(42)


# ---------------------------------------------------------------------------
# real records (Phases 23-25)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def master_frame() -> Any:
    """DA-08, read from disk. Only meaningful in ``needs_data`` tests."""
    from src.data_loader import master as ms

    return ms.load_master()


@pytest.fixture(scope="session")
def sample_records(master_frame: Any, project_root: Path):
    """``sample_records("D2", 5)`` -> a deterministic list of real WAV paths.

    Sampled with seed 42 so the same files are exercised on every run and a
    failure can be reproduced from the record uid alone. Unlabelled and
    duplicate records are kept: preprocessing has to survive every file on disk,
    not only the ones that reach a classifier.
    """

    def pick(dataset_source: str, n: int = 5, *, seed: int = 42) -> list[tuple[str, Path]]:
        subset = master_frame[master_frame["dataset_source"] == dataset_source]
        if subset.empty:
            return []
        take = subset.sample(n=min(n, len(subset)), random_state=seed)
        return [
            (str(row.record_uid), project_root / str(row.file_path))
            for row in take.itertuples()
        ]

    return pick
