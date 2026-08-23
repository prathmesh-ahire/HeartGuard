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

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = ROOT / "dataset"


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
        + ("included" if config.getoption("--runslow") else "SKIPPED (--runslow to include)")
    ]
