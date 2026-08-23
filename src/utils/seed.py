"""Global seeding (Phase 04, task T04.1).

Rule 5: fixed seed 42, everywhere. Two runs of the same command must produce
identical numbers.

**The PYTHONHASHSEED caveat, which is easy to get wrong.** Python reads
``PYTHONHASHSEED`` once, at interpreter start. Setting ``os.environ`` from
inside a running process does **not** change that process's hash seed -- the
randomization is already fixed. What it does do is propagate to child processes,
which is what makes joblib workers deterministic.

So :func:`set_global_seed` sets the variable (for children) and separately
*reports* whether it was set early enough to affect this process, via
:func:`hash_seed_status`. It never claims a guarantee it cannot make. Any code
whose result depends on ``set`` or ``dict`` iteration order should not rely on
this being handled -- it should sort explicitly.

Practically this only bites where iteration order of a set leaks into a result.
The feature registry enforces a fixed ordering explicitly (T31.3) rather than
depending on hash stability, which is the right pattern.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

__all__ = [
    "GLOBAL_SEED",
    "set_global_seed",
    "get_rng",
    "hash_seed_status",
    "seed_state",
]

GLOBAL_SEED = 42

_HASH_SEED_AT_START: str | None = os.environ.get("PYTHONHASHSEED")
_applied_seed: int | None = None


def set_global_seed(seed: int = GLOBAL_SEED) -> int:
    """Seed every global RNG this project uses. Returns the seed applied.

    Covers:
      * ``random`` -- the stdlib global RNG
      * ``numpy.random`` -- the legacy global RNG that scikit-learn's
        ``random_state=None`` paths fall back to
      * ``PYTHONHASHSEED`` -- for child processes (see the module docstring)
    """
    global _applied_seed

    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    # Affects child processes only; this interpreter's hash seed is already set.
    os.environ["PYTHONHASHSEED"] = str(seed)

    _applied_seed = seed
    return seed


def get_rng(seed: int | None = None) -> Any:
    """Return a fresh ``numpy.random.Generator``.

    Prefer this over the legacy global ``numpy.random`` functions in new code:
    an explicit generator cannot be perturbed by an unrelated library reseeding
    the global state mid-run.
    """
    import numpy as np

    return np.random.default_rng(GLOBAL_SEED if seed is None else seed)


def hash_seed_status() -> dict[str, Any]:
    """Report whether ``PYTHONHASHSEED`` was set early enough to matter here."""
    at_start = _HASH_SEED_AT_START
    effective = at_start is not None and at_start != "random"
    return {
        "value_at_interpreter_start": at_start,
        "value_now": os.environ.get("PYTHONHASHSEED"),
        "hash_randomization_enabled": bool(sys.flags.hash_randomization),
        # True only if the variable was already set when the interpreter
        # started. Setting it later cannot retroactively change this process.
        "effective_for_this_process": effective,
        "effective_for_child_processes": os.environ.get("PYTHONHASHSEED") is not None,
        "note": (
            "PYTHONHASHSEED is read once at interpreter start. Set it in the "
            "shell before launching Python if this process's own hash order "
            "must be deterministic; code should not depend on it regardless."
        ),
    }


def seed_state() -> dict[str, Any]:
    """The seed block recorded in the run manifest."""
    return {
        "global_seed": GLOBAL_SEED,
        "applied_seed": _applied_seed,
        "seed_applied": _applied_seed is not None,
        "python_hash_seed": hash_seed_status(),
    }
