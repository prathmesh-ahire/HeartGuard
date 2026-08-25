"""Timing instrumentation (Phase 04, task T04.4).

A ``@timed`` decorator and a ``timer`` context manager, both writing durations
into the active run manifest.

These numbers are not diagnostics -- they are a **deliverable**. The complexity
table (T25/T26) reports training time, inference time and per-family extraction
cost, and the 1D-CNN scope decision (T52.1) is made against the Phase 75
timings. A duration that was never recorded is a table cell that has to be
re-measured by re-running a six-hour job.

``perf_counter`` is used throughout: it is monotonic and unaffected by the
system clock being adjusted mid-run.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

__all__ = ["timed", "timer", "Stopwatch", "format_duration"]

F = TypeVar("F", bound=Callable[..., Any])


def format_duration(seconds: float) -> str:
    """Human-readable duration: ``412 ms``, ``3.42 s``, ``5m 03s``, ``2h 14m``."""
    if seconds < 1:
        return str(round(seconds * 1000, 1)) + " ms"
    if seconds < 60:
        return str(round(seconds, 2)) + " s"
    if seconds < 3600:
        minutes, rest = divmod(seconds, 60)
        return str(int(minutes)) + "m " + str(int(rest)).zfill(2) + "s"
    hours, rest = divmod(seconds, 3600)
    return str(int(hours)) + "h " + str(int(rest // 60)).zfill(2) + "m"


def _record(label: str, seconds: float) -> None:
    """Write a duration into the active manifest, if there is one."""
    # Instrumentation must never break the work it is measuring.
    with contextlib.suppress(Exception):
        from src.utils.run_manifest import current_run

        run = current_run()
        if run is not None:
            run.record_timing(label, seconds)


class Stopwatch:
    """Result object returned by :func:`timer`. ``.seconds`` is set on exit."""

    __slots__ = ("label", "started", "seconds")

    def __init__(self, label: str) -> None:
        self.label = label
        self.started: float | None = None
        self.seconds: float | None = None

    def __repr__(self) -> str:
        if self.seconds is None:
            return "Stopwatch(" + self.label + ", running)"
        return "Stopwatch(" + self.label + ", " + format_duration(self.seconds) + ")"


@contextmanager
def timer(
    label: str,
    *,
    log: bool = True,
    record: bool = True,
    level: str = "info",
) -> Iterator[Stopwatch]:
    """Time a block.

    ::

        with timer("extract:physionet") as sw:
            run_extraction()
        print(sw.seconds)

    The duration is recorded even when the block raises, tagged ``[failed]`` --
    a run that died after four hours still tells you it took four hours.
    """
    watch = Stopwatch(label)
    watch.started = time.perf_counter()
    failed = False
    try:
        yield watch
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.perf_counter() - watch.started
        watch.seconds = elapsed
        key = label + " [failed]" if failed else label
        if record:
            _record(key, elapsed)
        if log:
            with contextlib.suppress(Exception):
                from src.utils.logging_setup import get_logger

                logger = get_logger("pvmepcg.timing")
                getattr(logger, level, logger.info)(
                    key + " took " + format_duration(elapsed)
                )


def timed(
    label_or_func: str | Callable[..., Any] | None = None,
    *,
    log: bool = True,
    record: bool = True,
) -> Any:
    """Decorator recording a function's wall time.

    Usable bare or with a label::

        @timed
        def extract(...): ...

        @timed("extract:physionet")
        def extract(...): ...

    Defaults to ``module.qualname`` when no label is given.
    """

    def decorate(func: Callable[..., Any], label: str | None) -> Callable[..., Any]:
        resolved = label or (func.__module__ + "." + func.__qualname__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timer(resolved, log=log, record=record):
                return func(*args, **kwargs)

        return wrapper

    if callable(label_or_func):
        return decorate(label_or_func, None)

    def outer(func: Callable[..., Any]) -> Callable[..., Any]:
        return decorate(func, label_or_func)

    return outer
