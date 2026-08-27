"""SO-01 -- randomized hyperparameter search over the declared spaces (Phase 55).

Random search is the documented minimum for this project, and it is also the
control the advanced methods are judged against: SO-02 (Bayesian) only earns its
place if it beats this at an *equal* trial budget (T56.3, T56.5). So the two
share :class:`~src.optimization.base.BaseSearch`'s loop and differ in one method.

**Why this is not ``sklearn.model_selection.RandomizedSearchCV``.** The task
names that class, and the sampling is identical to it -- each dimension drawn
independently from its declared distribution, uniform in log space where the
config says ``log_uniform``. Three requirements of this project are things that
class cannot do, and all three are load-bearing rather than cosmetic:

1. **Constrained spaces.** ``RandomizedSearchCV`` samples every dimension
   independently and has no notion of an illegal combination. M1's ``lbfgs``
   solver accepts only an L2 penalty and *raises* on ``l1_ratio=1.0``; M2's ``p``
   is read only under ``metric="minkowski"``, so half of that space is duplicates
   of the other half. Both are declared as named constraints in
   ``configs/models.yaml`` and applied by :meth:`SearchSpace.repair` /
   :meth:`SearchSpace.is_valid` before dispatch. A search that discovered its own
   space by catching solver exceptions would report a trial count that does not
   mean what it says.
2. **The fold map is loaded, not derived.** ``cv=`` would have to be handed an
   iterable that already encodes DA-07's outer fold and the inner splits cut from
   it -- at which point the class is contributing only its loop.
3. **The budget and the trial log.** T54.4 wants every trial's parameters, score,
   per-inner-fold scores and duration; T54.5 wants a wall-clock ceiling with
   graceful termination between trials. ``cv_results_`` has no durations per
   candidate and ``RandomizedSearchCV`` has no wall-clock budget at all.

The deviation is deliberate and recorded in ``Docs/note.md``. What is *not*
deviated from: the sampling distributions come from the same
:meth:`SearchSpace.to_distributions` scipy objects that would have been passed to
``RandomizedSearchCV``, and ``tests/test_search_no_leakage.py`` pins that the
draws match sklearn's ``ParameterSampler`` on an unconstrained space.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from src.optimization.base import BaseSearch, Trial
from src.utils.logging_setup import get_logger

__all__ = ["RandomizedSearch"]

log = get_logger("optimization.randomized")


class RandomizedSearch(BaseSearch):
    """Independent draws from the declared space, repaired and filtered.

    The generator is seeded once per search from ``seed`` and consumed in order,
    so trial *k* of a re-run is the same point as trial *k* of the first run --
    which is what makes a truncated search (a wall-clock stop) a prefix of the
    full one rather than a different search.
    """

    method: ClassVar[str] = "random"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(self.seed)

    def _propose(self, history: Sequence[Trial]) -> dict[str, Any]:
        del history  # random search is memoryless -- that is the whole point
        return self.space.sample(self._rng)
