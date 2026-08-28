"""Tuned and nested-search planners for EXP-A2 (Phase 65).

Three planners, in increasing cost and increasing honesty:

:class:`TunedPlanner`
    One fixed hyperparameter point per model -- the ``final_selected`` column of
    T07, chosen by SO-02 on outer fold r0f0. Cheap, reproducible, and **not
    nested**: r0f0's training rows are other folds' test rows, so a 25-fold run
    under this planner carries a selection bias. It exists for the deployed
    model (T65.6) and for tasks where a per-fold search is out of budget, and
    every artifact it produces says so.

:class:`NestedSearchPlanner`
    A search inside every outer training fold. Expensive and correct: the outer
    test fold is never read by the search, so the resulting metric is a genuine
    nested-CV estimate. This is what EXP-A2 runs (T65.1).

:class:`SubsetPlanner`
    :class:`NestedSearchPlanner`'s per-fold points, reused against the SO-04
    20-feature pipeline (T65.3). The hyperparameters were tuned on all 138
    features and are applied to 20; that is a stated approximation, not a
    silent one -- re-searching inside the subset would double Phase 65's cost
    for a variant that is a sensitivity check rather than a headline.

**The search budget.** A nested search over the 25-fold map at the Phase 55-56
budget (40 trials per model, 15 for M5) measures out at ~31 h of CPU on this
machine. The user chose the reduced budget on 2026-08-28: 12 trials per model,
5 for M5, three inner splits, all 25 outer folds. That is a weaker search than
the single-fold SO-01/SO-02 study and is recorded as a limitation rather than
presented as equivalent. See ``Docs/note.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.experiment import ModelPlanner, PlannedModel
from src.utils.logging_setup import get_logger

__all__ = [
    "DEFAULT_TRIALS",
    "DEFAULT_MODEL_TRIALS",
    "ENSEMBLE_MEMBERS",
    "TuningError",
    "selected_parameters",
    "subset_columns",
    "subset_pipeline_config",
    "TunedPlanner",
    "NestedSearchPlanner",
    "SubsetPlanner",
    "select_final_model",
]

log = get_logger("evaluation.tuned")

#: Decided by the user on 2026-08-28 against the measured ~31 h alternative.
DEFAULT_TRIALS = 12
DEFAULT_MODEL_TRIALS: dict[str, int] = {"M5": 5}

#: M6/M7 declare no hyperparameters of their own; what gets tuned is their
#: members. Read from config rather than hardcoded, but defaulted here so a
#: planner can state its recipe without loading anything.
ENSEMBLE_MEMBERS: tuple[str, ...] = ("M3", "M4", "M5")

SEARCH_CACHE_DIRNAME = "_nested_search_cache"


class TuningError(RuntimeError):
    """A tuned point cannot be assembled from what is on disk."""


# ---------------------------------------------------------------------------
# T07 -- the fixed selected point
# ---------------------------------------------------------------------------


def _best_parameters(exp: str) -> dict[str, dict[str, Any]]:
    """``best_parameters.json`` for one search, as ``{model_id: params}``.

    The JSON is read rather than T07's CSV because it holds **typed** values: an
    ``int`` is an int, ``class_weight: null`` is None. T07 is the human-readable
    view of the same decision and is used to cross-check which method won, not
    to recover the values -- round-tripping ``"None"`` through a CSV is exactly
    the trap ``search_report.read_t07`` exists to avoid.
    """
    from src.optimization.base import search_dir

    path = search_dir(exp) / "best_parameters.json"
    if not path.is_file():
        raise TuningError(
            "no " + exp + " results at " + str(path) + "; run scripts/05_run_search.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    found: dict[str, dict[str, Any]] = {}
    for search in payload.get("searches", []):
        found[str(search["model_id"])] = dict(search.get("best_params", {}))
    return found


def selected_parameters() -> dict[str, dict[str, Any]]:
    """The T07 ``final_selected`` point per model, with typed values.

    SO-02 where it searched the model, SO-01 otherwise -- the rule T07 itself
    applies. The result is cross-checked against T07's ``final_source`` column
    so the two views of the same decision cannot drift apart unnoticed.
    """
    from src.reporting.search_report import read_t07

    by_exp = {"SO-01": _best_parameters("SO-01"), "SO-02": _best_parameters("SO-02")}
    table = read_t07()

    chosen: dict[str, dict[str, Any]] = {}
    for model_id, block in table.groupby("model_id"):
        sources = {str(value) for value in block["final_source"]}
        if sources == {"not searched"}:
            continue
        if len(sources) != 1:
            raise TuningError(
                model_id + " has more than one final_source in T07: " + str(sorted(sources))
            )
        source = sources.pop()
        if source not in by_exp:
            raise TuningError(model_id + " names an unknown final_source " + repr(source))
        params = by_exp[source].get(model_id)
        if params is None:
            raise TuningError(
                "T07 says " + model_id + " was selected from " + source
                + " but that run's best_parameters.json has no entry for it"
            )
        chosen[str(model_id)] = params
    if not chosen:
        raise TuningError("T07 records no searched model; nothing to tune with")
    return chosen


# ---------------------------------------------------------------------------
# SO-04 -- the selected feature subset (T65.3)
# ---------------------------------------------------------------------------


def subset_columns() -> tuple[list[int], list[str]]:
    """The SO-04 subset as ``(column positions, feature names)``.

    Positions are re-derived from the feature registry rather than trusted from
    the CSV's own ``column`` field: the registry is the authority on column
    order, and a subset applied at the wrong positions selects 20 arbitrary
    features while looking entirely correct.
    """
    import pandas as pd

    from src.feature_extraction.registry import feature_names
    from src.utils.config import load_config

    path = Path(load_config("paths").require("outputs.features")) / "selected_feature_subset.csv"
    if not path.is_file():
        raise TuningError(
            "no SO-04 subset at " + str(path) + "; run scripts/06_run_feature_selection.py"
        )
    frame = pd.read_csv(path)
    names = [str(name) for name in frame["feature"]]
    registry = list(feature_names())
    missing = [name for name in names if name not in registry]
    if missing:
        raise TuningError(
            "SO-04 names " + str(len(missing)) + " feature(s) absent from the registry, e.g. "
            + ", ".join(missing[:5])
        )
    positions = [registry.index(name) for name in names]
    stored = [int(value) for value in frame["column"]]
    if positions != stored:
        raise TuningError(
            "SO-04's stored column positions disagree with the feature registry; "
            "the subset was written against a different feature order"
        )
    return positions, names


def subset_pipeline_config() -> dict[str, Any]:
    """The Phase 44 pipeline config with the SO-04 subset switched on.

    A ``fixed_subset`` selector, so the restriction happens **inside** the
    pipeline and therefore inside the training fold, exactly like every other
    learned step. Slicing the matrix before the split would be equivalent here
    (nothing is learned from the data) but it would put the one selection step
    that can be done outside the fold outside it, and the next one would follow.
    """
    from src.utils.config import load_config

    columns, names = subset_columns()
    config = dict(load_config("models").get("pipeline") or {})
    config["selector"] = {
        "enabled": True,
        "kind": "fixed_subset",
        "k": len(columns),
        "columns": list(columns),
        "source": "SO-04",
        "feature_names": list(names),
    }
    return config


# ---------------------------------------------------------------------------
# the planners
# ---------------------------------------------------------------------------


class TunedPlanner(ModelPlanner):
    """The single T07 point, applied to every fold. NOT nested -- see the module docstring."""

    name = "tuned"

    def __init__(self, *, pipeline_config: dict[str, Any] | None = None) -> None:
        self.pipeline_config = pipeline_config
        self._points: dict[str, dict[str, Any]] | None = None

    @property
    def points(self) -> dict[str, dict[str, Any]]:
        if self._points is None:
            self._points = selected_parameters()
        return self._points

    def key_material(self, model_id: str, fold: Any, data: Any) -> dict[str, Any]:
        if model_id in {"M6", "M7"}:
            params = {member: self.points.get(member, {}) for member in ENSEMBLE_MEMBERS}
        else:
            params = self.points.get(model_id, {})
        return {"planner": self.name, "model_id": model_id, "params": params}

    def plan(self, model_id: str, fold: Any, data: Any) -> PlannedModel:
        from src.models import estimators as est

        if model_id in {"M6", "M7"}:
            member_params = {
                member: self.points[member]
                for member in ENSEMBLE_MEMBERS
                if member in self.points
            }
            train_groups = np.asarray(data.groups, dtype=object)[
                np.asarray(fold.train_index, dtype=int)
            ]

            def build_ensemble() -> Any:
                return est.make_ensemble(
                    model_id, groups=train_groups, member_params=member_params
                )

            return PlannedModel(
                factory=build_ensemble,
                params={"members": member_params},
                pipeline_config=self.pipeline_config,
                note="T07 selected point per member; NOT nested",
                extra={"tuning": "T07-fixed"},
            )

        params = dict(self.points.get(model_id, {}))

        def build() -> Any:
            return est.build_estimator(model_id, **params)

        return PlannedModel(
            factory=build,
            params=params,
            pipeline_config=self.pipeline_config,
            note="T07 selected point; NOT nested",
            extra={"tuning": "T07-fixed"},
        )


class NestedSearchPlanner(ModelPlanner):
    """A hyperparameter search inside every outer training fold (T65.1).

    The search never reads the outer test fold -- :class:`~src.optimization.base.RowLedger`
    asserts it on every result -- so the outer score is a nested estimate rather
    than a search score.

    Searches are cached on disk under ``outputs/05_search_optimization/_nested_search_cache/``
    and keyed by everything that would change the point they choose. Two things
    make that worth doing: M6 and M7 need the same three member searches on the
    same fold, and the SO-04 subset variant reuses the full-matrix points rather
    than paying for a second set.
    """

    name = "nested"

    def __init__(
        self,
        *,
        method: str = "bayes",
        trials: int = DEFAULT_TRIALS,
        model_trials: dict[str, int] | None = None,
        inner_splits: int = 3,
        pipeline_config: dict[str, Any] | None = None,
        search_pipeline_config: dict[str, Any] | None = None,
        cache_dir: str | Path | None = None,
        reuse_only: bool = False,
    ) -> None:
        self.method = method
        self.trials = int(trials)
        self.model_trials = dict(DEFAULT_MODEL_TRIALS if model_trials is None else model_trials)
        self.inner_splits = int(inner_splits)
        #: Applied to the final per-fold fit.
        self.pipeline_config = pipeline_config
        #: Applied inside the search. Left as the default 138-feature pipeline
        #: even for the subset variant, so both variants tune on the same object
        #: and their cached searches are shared rather than duplicated.
        self.search_pipeline_config = search_pipeline_config
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.reuse_only = bool(reuse_only)

    # -- budget -------------------------------------------------------------

    def budget_for(self, model_id: str) -> int:
        return int(self.model_trials.get(model_id, self.trials))

    def _searched(self, model_id: str) -> tuple[str, ...]:
        """Which model ids actually get searched for ``model_id``.

        M6 and M7 declare no hyperparameters; what an "optimized ensemble" means
        is that its **members** are tuned, so those are what get searched.
        """
        if model_id in {"M6", "M7"}:
            from src.utils.config import load_config

            declared = load_config("models").get("models." + model_id + ".members")
            return tuple(str(name) for name in (declared or ENSEMBLE_MEMBERS))
        return (model_id,)

    # -- caching ------------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        from src.utils.config import load_config
        from src.utils.io import ensure_dir

        if self._cache_dir is None:
            root = Path(load_config("paths").require("outputs.search_optimization"))
            self._cache_dir = root / SEARCH_CACHE_DIRNAME
        return Path(ensure_dir(self._cache_dir))

    def _search_key(self, model_id: str, fold: Any, data: Any) -> str:
        import hashlib

        from src.models import spaces

        material = {
            "method": self.method,
            "model_id": model_id,
            "trials": self.budget_for(model_id),
            "inner_splits": self.inner_splits,
            "fold": fold.label,
            "task": getattr(data, "task", ""),
            "n_train": int(np.asarray(fold.train_index).size),
            "train_uids": hashlib.sha256(
                "|".join(sorted(fold.train_uids)).encode("utf-8")
            ).hexdigest()[:16],
            "space": [
                dimension.describe() for dimension in spaces.load_space(model_id).dimensions
            ],
            "pipeline": self.search_pipeline_config,
            "seed": 42,
        }
        text = json.dumps(material, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, model_id: str, fold: Any, key: str) -> Path:
        return self.cache_dir / (
            self.method + "__" + model_id + "__" + fold.label + "__" + key + ".json"
        )

    def search_point(self, model_id: str, fold: Any, data: Any) -> dict[str, Any]:
        """The best point for one model on one outer fold, searched or cached."""
        key = self._search_key(model_id, fold, data)
        path = self._cache_path(model_id, fold, key)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            log.info("reusing cached %s search: %s %s", self.method, model_id, fold.label)
            return payload
        if self.reuse_only:
            raise TuningError(
                "no cached "
                + self.method
                + " search for "
                + model_id
                + " on "
                + fold.label
                + " at "
                + str(path)
                + "; run EXP-A2 with --planner nested first"
            )

        from src.optimization import base as ob
        from src.optimization import driver as od

        objective = ob.objective_for(getattr(data, "task", "binary"))
        search = od.build_search(
            self.method,
            model_id,
            objective=objective,
            budget=ob.Budget(max_trials=self.budget_for(model_id)),
            pipeline_config=self.search_pipeline_config,
        )
        result = search.run(data, fold, n_inner_splits=self.inner_splits)
        result.assert_no_outer_leakage()
        best = result.best
        if best is None:
            raise TuningError(
                model_id + " " + fold.label + ": every trial failed; no point to use"
            )
        payload = {
            "method": self.method,
            "model_id": model_id,
            "fold": fold.label,
            "objective": objective.name,
            "best_params": dict(best.params),
            "inner_best_score": float(best.score),
            "n_trials": len(result.trials),
            "n_trials_ok": len(result.successful),
            "seconds": round(float(result.seconds), 3),
            "stop_reason": result.stop_reason,
            "n_inner_folds": result.n_inner_folds,
            "outer_test_rows_touched": len(
                result.ledger.touched & set(result.outer_test_index)
            ),
            "search_key": key,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    # -- the planner interface ----------------------------------------------

    def key_material(self, model_id: str, fold: Any, data: Any) -> dict[str, Any]:
        return {
            "planner": self.name,
            "model_id": model_id,
            "method": self.method,
            "inner_splits": self.inner_splits,
            "reuse_only": self.reuse_only,
            "search_pipeline": self.search_pipeline_config,
            "budgets": {
                searched: self.budget_for(searched) for searched in self._searched(model_id)
            },
        }

    def plan(self, model_id: str, fold: Any, data: Any) -> PlannedModel:
        from src.models import estimators as est

        searched = self._searched(model_id)
        points = {name: self.search_point(name, fold, data) for name in searched}
        inner_scores = {
            name: round(float(payload["inner_best_score"]), 6)
            for name, payload in points.items()
        }
        touched = sum(int(payload["outer_test_rows_touched"]) for payload in points.values())
        if touched:
            raise TuningError(
                model_id
                + " "
                + fold.label
                + ": a cached search reports "
                + str(touched)
                + " outer test row(s) touched"
            )

        if model_id in {"M6", "M7"}:
            member_params = {
                name: dict(payload["best_params"]) for name, payload in points.items()
            }
            train_groups = np.asarray(data.groups, dtype=object)[
                np.asarray(fold.train_index, dtype=int)
            ]

            def build_ensemble() -> Any:
                return est.make_ensemble(
                    model_id, groups=train_groups, member_params=member_params
                )

            return PlannedModel(
                factory=build_ensemble,
                params={"members": member_params},
                pipeline_config=self.pipeline_config,
                note=self.method + " search inside this training fold, per member",
                extra={
                    "tuning": "nested-" + self.method,
                    "inner_scores": json.dumps(inner_scores, sort_keys=True),
                    "search_seconds": round(
                        sum(float(p["seconds"]) for p in points.values()), 2
                    ),
                },
            )

        params = dict(points[model_id]["best_params"])

        def build() -> Any:
            return est.build_estimator(model_id, **params)

        return PlannedModel(
            factory=build,
            params=params,
            pipeline_config=self.pipeline_config,
            note=self.method + " search inside this training fold",
            extra={
                "tuning": "nested-" + self.method,
                "inner_best_score": inner_scores[model_id],
                "search_seconds": round(float(points[model_id]["seconds"]), 2),
                "n_trials": int(points[model_id]["n_trials"]),
            },
        )


class SubsetPlanner(NestedSearchPlanner):
    """EXP-A2's per-fold points, refitted on the SO-04 20-feature pipeline (T65.3).

    ``reuse_only`` by default: it must not start its own searches. If the cache
    is missing an entry, that is a signal the nested run has not happened yet,
    not an invitation to spend another nine hours producing a second set of
    points that would then differ from the ones the headline table used.
    """

    name = "nested_subset"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("reuse_only", True)
        kwargs["pipeline_config"] = subset_pipeline_config()
        kwargs["search_pipeline_config"] = None
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# T65.4 -- the selection rule
# ---------------------------------------------------------------------------


def select_final_model(
    aggregate: Any,
    *,
    rule: tuple[str, ...] = ("sensitivity", "balanced_accuracy"),
) -> dict[str, Any]:
    """Rank models by the documented rule and return the winner with its ranking.

    Research rule 6: selection prioritises **sensitivity and balanced accuracy**,
    never raw accuracy. Implemented as a lexicographic sort on the mean of each
    named metric in order, so "prioritise" means a defined ordering rather than
    an unstated weighting -- and the full ranking is returned, so a close second
    is visible instead of being collapsed into a single winner.
    """
    import pandas as pd

    if aggregate is None or len(aggregate) == 0:
        raise TuningError("no aggregate metrics to select from")
    frame = pd.DataFrame(aggregate).copy()
    columns = [metric + "_mean" for metric in rule]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TuningError(
            "the aggregate table has no column(s) " + ", ".join(missing)
            + "; the selection rule cannot be applied"
        )

    ordered = frame.sort_values(columns, ascending=False).reset_index(drop=True)
    ordered.insert(0, "rank", range(1, len(ordered) + 1))
    winner = ordered.iloc[0]
    accuracy_winner = (
        str(frame.sort_values("accuracy_mean", ascending=False).iloc[0]["model_id"])
        if "accuracy_mean" in frame.columns
        else ""
    )
    return {
        "model_id": str(winner["model_id"]),
        "rule": list(rule),
        "ranking": ordered[["rank", "model_id", *columns]].to_dict(orient="records"),
        "accuracy_would_have_chosen": accuracy_winner,
        "rule_and_accuracy_agree": accuracy_winner == str(winner["model_id"]),
    }
