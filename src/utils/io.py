"""Atomic save/load helpers (Phase 04, task T04.5).

Every writer here is **atomic**: content goes to a temporary file in the same
directory and is then moved into place with :func:`os.replace`, which is atomic
on both Windows and POSIX. A reader never sees a half-written file, and an
interrupted run leaves the previous version intact rather than a truncated one.
This matters because long extraction and search runs get interrupted, and a
zero-byte or half-written CSV that still *looks* like an output is exactly the
kind of thing that silently poisons a downstream table.

Every writer creates its parent directories.

**JSON and NaN.** ``json.dumps`` emits a bare ``NaN`` token for a float NaN,
which is not valid JSON -- ``JSON.parse`` rejects it outright. It works on most
records and fails on the degenerate ones, which are precisely the records the
NaN policy exists to track. :func:`save_json` therefore coerces NaN and +/-Inf
to ``null`` by default and refuses to emit the invalid token.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "ensure_dir",
    "atomic_path",
    "save_csv",
    "load_csv",
    "save_json",
    "load_json",
    "save_parquet",
    "load_parquet",
    "save_png",
    "save_pickle",
    "load_pickle",
    "sanitize_for_json",
]


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` as a directory (with parents) and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def atomic_path(target: str | os.PathLike, *, suffix: str = "") -> Iterator[Path]:
    """Yield a temporary path that is moved onto ``target`` on clean exit.

    The temporary file is created in the same directory as ``target`` so the
    final move stays on one filesystem and is therefore genuinely atomic. If the
    body raises, the temporary file is removed and ``target`` is left untouched.
    """
    dest = Path(target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent), prefix="." + dest.name + ".", suffix=suffix or ".tmp"
    )
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write(target: str | os.PathLike, writer: Callable[[Path], None]) -> Path:
    dest = Path(target)
    with atomic_path(dest, suffix=dest.suffix or ".tmp") as tmp:
        writer(tmp)
    return dest


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def save_csv(df: Any, path: str | os.PathLike, *, index: bool = False, **kwargs: Any) -> Path:
    """Write a DataFrame to CSV atomically."""
    return _atomic_write(
        path, lambda tmp: df.to_csv(tmp, index=index, encoding="utf-8", **kwargs)
    )


def load_csv(path: str | os.PathLike, **kwargs: Any) -> Any:
    import pandas as pd

    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def sanitize_for_json(obj: Any) -> Any:
    """Recursively make ``obj`` JSON-safe.

    NaN and +/-Inf become ``None``. numpy scalars become Python scalars, numpy
    arrays become lists, Paths become strings, sets become sorted lists. This is
    the single place where the numeric-type boundary is crossed, so a numpy
    ``int64`` never reaches ``json.dumps`` and raises "not JSON serializable"
    three hours into a run.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, set):
        return sorted(sanitize_for_json(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)

    # numpy / pandas scalars and arrays, without importing numpy at module load.
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return sanitize_for_json(tolist())
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return sanitize_for_json(item())
        except (ValueError, TypeError):
            pass

    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        return isoformat()

    return str(obj)


def save_json(
    obj: Any,
    path: str | os.PathLike,
    *,
    indent: int = 2,
    sanitize: bool = True,
    sort_keys: bool = False,
) -> Path:
    """Write ``obj`` to JSON atomically, with NaN/Inf coerced to ``null``.

    ``allow_nan=False`` is passed deliberately: if sanitizing is disabled and a
    NaN survives, this raises rather than emitting a bare ``NaN`` token that
    every strict JSON parser will reject.
    """
    payload = sanitize_for_json(obj) if sanitize else obj

    def write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=indent, allow_nan=False, sort_keys=sort_keys)
            fh.write("\n")

    return _atomic_write(path, write)


def load_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------


def save_parquet(df: Any, path: str | os.PathLike, **kwargs: Any) -> Path:
    """Write a DataFrame to Parquet atomically (pyarrow engine)."""
    kwargs.setdefault("engine", "pyarrow")
    kwargs.setdefault("index", False)
    return _atomic_write(path, lambda tmp: df.to_parquet(tmp, **kwargs))


def load_parquet(path: str | os.PathLike, **kwargs: Any) -> Any:
    import pandas as pd

    kwargs.setdefault("engine", "pyarrow")
    return pd.read_parquet(path, **kwargs)


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------


def save_png(
    fig: Any,
    path: str | os.PathLike,
    *,
    dpi: int = 300,
    close: bool = True,
    **kwargs: Any,
) -> Path:
    """Save a matplotlib figure atomically at publication dpi."""
    kwargs.setdefault("bbox_inches", "tight")
    result = _atomic_write(path, lambda tmp: fig.savefig(tmp, dpi=dpi, format="png", **kwargs))
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# pickle / joblib
# ---------------------------------------------------------------------------


def save_pickle(obj: Any, path: str | os.PathLike, *, use_joblib: bool = True) -> Path:
    """Serialize ``obj`` atomically. joblib by default -- it is far more
    efficient for the large numpy arrays inside a fitted sklearn estimator."""

    def write(tmp: Path) -> None:
        if use_joblib:
            import joblib

            joblib.dump(obj, tmp)
        else:
            with tmp.open("wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return _atomic_write(path, write)


def load_pickle(path: str | os.PathLike, *, use_joblib: bool = True) -> Any:
    if use_joblib:
        import joblib

        return joblib.load(path)
    with Path(path).open("rb") as fh:
        # Deserializing only artifacts this project wrote into models_saved/.
        # No untrusted input reaches this path.
        return pickle.load(fh)  # noqa: S301
