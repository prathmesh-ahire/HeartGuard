"""Root conftest — ``sys.path``, CLI options and marker-driven skipping.

**Why the path insert.** Without it, ``import src.utils...`` works under
``python -m pytest`` (which puts the working directory on ``sys.path``) and
fails under a bare ``pytest`` (which does not). CI runs the bare form -- T08.2
-- so the difference is not cosmetic: the suite would pass locally and fail the
build.

**Why the hooks live here rather than in tests/conftest.py.**
``pytest_addoption`` is only honoured in the rootdir conftest or a plugin, so
``--runslow`` and ``--no-data`` must be declared at this level. The collection
hook that acts on them is kept alongside for coherence. ``tests/conftest.py``
(T06.2) holds the fixtures.

**Marker policy.**

``slow``
    Skipped by default, included with ``--runslow``. A default run stays fast
    enough that people actually run it.

``needs_data``
    Auto-skipped when ``dataset/`` is absent. The 1.3 GB corpus is gitignored
    and never reaches GitHub, so every data-dependent test is excluded in CI by
    design. ``--no-data`` forces the same behaviour locally, which is how the
    skip path gets exercised on a machine that *does* have the dataset -- an
    untested skip path is indistinguishable from a broken one.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = ROOT / "dataset"

#: The project's real run manifest. Tests must never append to it; a test that
#: wants to *read* it does so explicitly through this constant.
REAL_RUN_MANIFEST = ROOT / "outputs" / "00_evidence_index" / "run_manifest.json"

MANIFEST_ENV = "HEARTGUARD__PATHS__OUTPUTS__RUN_MANIFEST"

_manifest_sandbox: Path | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked `slow` (skipped by default)",
    )
    parser.addoption(
        "--no-data",
        action="store_true",
        default=False,
        help=(
            "pretend dataset/ is absent, so `needs_data` tests skip. Exercises the "
            "CI path on a machine that has the dataset."
        ),
    )


def dataset_available(config: pytest.Config | None = None) -> bool:
    """True when the real dataset tree is present and not suppressed."""
    if config is not None and config.getoption("--no-data"):
        return False
    return DATASET_ROOT.is_dir() and any(DATASET_ROOT.iterdir())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "_internal: reserved; not used by project tests"
    )
    _redirect_run_manifest()


def _redirect_run_manifest() -> None:
    """Point the run manifest at a throwaway file for the whole test session.

    ``tests/conftest.py`` promises that no test writes into the real
    ``outputs/`` tree. Until now that promise held for artifacts and for the
    evidence index but *not* for the run manifest, because
    ``run_manifest._default_manifest_path()`` resolves through
    ``configs/paths.yaml`` and ignores whatever ``--out-dir`` a script was given.
    Every audit test that called ``run_audit`` therefore appended a real row to
    the project's own provenance file, and any such test killed mid-run left a
    row stuck at ``status="running"`` forever.

    Rule 5 makes the manifest a deliverable: it is what maps a published number
    back to the run that produced it. Rows from ``--limit 10`` test invocations
    map to nothing, and a permanent "running" row reads as an interrupted
    project run. Both are noise in a file whose entire value is that it is
    trustworthy.

    The redirect happens in ``pytest_configure`` rather than in an autouse
    fixture because it must land before the first test module is imported --
    a module that loads config at import time would otherwise cache the real
    path and ignore the override.
    """
    global _manifest_sandbox

    if os.environ.get(MANIFEST_ENV):  # an explicit override wins; don't fight it
        return

    _manifest_sandbox = Path(tempfile.mkdtemp(prefix="pvmepcg-manifest-"))
    os.environ[MANIFEST_ENV] = str(_manifest_sandbox / "run_manifest.json")

    from src.utils.config import clear_cache

    clear_cache()  # anything already loaded holds the real path


def pytest_unconfigure(config: pytest.Config) -> None:
    global _manifest_sandbox

    if _manifest_sandbox is None:
        return
    os.environ.pop(MANIFEST_ENV, None)
    shutil.rmtree(_manifest_sandbox, ignore_errors=True)
    _manifest_sandbox = None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_slow = pytest.mark.skip(reason="slow test; pass --runslow to include")
    if config.getoption("--no-data"):
        data_reason = "dataset/ suppressed by --no-data"
    else:
        data_reason = (
            "dataset/ not present -- the 1.3 GB corpus is gitignored and never "
            "reaches CI, so data-dependent tests are excluded by design"
        )
    skip_data = pytest.mark.skip(reason=data_reason)

    have_data = dataset_available(config)
    run_slow = config.getoption("--runslow")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "needs_data" in item.keywords and not have_data:
            item.add_marker(skip_data)


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Make the marker state visible in every run's header.

    A suite that reports "40 passed" while silently skipping every data test is
    the shape of a green build that proves nothing. Printing the state means a
    reader can see which half ran.
    """
    return [
        "PV-MEPCG: dataset="
        + ("present" if dataset_available(config) else "absent -> needs_data will SKIP")
        + ", slow="
        + ("included" if config.getoption("--runslow") else "SKIPPED (--runslow to include)"),
        "PV-MEPCG: run manifest -> "
        + (str(_manifest_sandbox) if _manifest_sandbox else "NOT REDIRECTED (real outputs/)"),
    ]
