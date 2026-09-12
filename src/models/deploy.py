"""Deployed models for the four non-binary tasks (Phase 120 prerequisite).

``models_saved/binary/final/`` was written by T65.6. The other four label spaces
-- PASCAL A, PASCAL B, CirCor murmur and CirCor outcome -- have complete
experiments (EXP-B1, EXP-B2, EXP-C1, EXP-C2) but no *deployed* model, because no
task in ``Docs/todo.md`` ever asked for one. The prediction pages built in Phase
116 therefore rendered "No model is deployed for this task yet" for four of the
five tasks, and screenshots 9 and 10 of Phase 120 ("multiclass prediction
output", "CirCor murmur/outcome output") had nothing to show but that empty
state. This module closes that gap, following T65.6's pattern exactly rather
than inventing a second one.

## The rule, stated before it is applied

**Which model.** The ranking is read from the task's own committed
``aggregate_metrics.csv`` -- the same file its published table was built from --
under research rule 6's ordering: ``sensitivity`` then ``balanced_accuracy``
for a binary task, ``macro_f1`` then ``balanced_accuracy`` for a multiclass one.
Never raw accuracy. What accuracy *would* have chosen is recorded next to the
winner so a disagreement is visible rather than implied, exactly as
``final_model_selection.json`` does for binary.

**Which hyperparameters.** The deployed point is a single search over every
labelled record of that task, run with the same method, budget and space the
experiment's own folds used. This is the generalization of what T07 is for
binary -- one search, whose point the deployed model takes -- and not a new
rule. It is *not* one of the nested per-fold points: those are 5 or 10 different
points that estimate performance, not a model.

## What this does NOT claim

The saved model is fitted **in sample** on every labelled record of its task. Its
performance estimate is the cross-validated run named in its manifest and
nothing measured on these rows. Every manifest says so in its own ``note``, and
``src/reporting/samples.py`` already says the same thing on the page.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "DEPLOY_FOLD_LABEL",
    "DEPLOYMENTS",
    "DeploymentError",
    "DeploymentSpec",
    "deploy_task",
    "rank_models",
    "search_deployed_point",
]

log = get_logger("models.deploy")

#: The synthetic fold the deployment search runs on: every labelled row trains,
#: nothing is held out. It is labelled rather than numbered so it can never be
#: confused with an outer fold of the experiment's own map.
DEPLOY_FOLD_LABEL = "deploy-all-records"


class DeploymentError(RuntimeError):
    """A task cannot be deployed, with the reason."""


@dataclass(frozen=True)
class DeploymentSpec:
    """One task's deployment: where its ranking comes from and how it is ranked."""

    #: The inference task key (``src.inference.predictor.TASKS``) and therefore
    #: the directory under ``models_saved/``.
    task: str
    #: The pipeline task key (``src.data_loader.master.TASK_LABEL_COLUMNS``).
    data_task: str
    #: The committed run directory whose ``aggregate_metrics.csv`` ranks the
    #: models. It is the run behind that task's published table.
    run_dir: str
    #: The published table the run directory backs, for the manifest.
    table_id: str
    #: Research rule 6's ordering for this task's kind, most important first.
    rule: tuple[str, ...]


DEPLOYMENTS: tuple[DeploymentSpec, ...] = (
    DeploymentSpec(
        "pascal_a",
        "pascal_a",
        "outputs/07_multiclass_results/EXP-B1",
        "T11",
        ("macro_f1", "balanced_accuracy"),
    ),
    DeploymentSpec(
        "pascal_b",
        "pascal_b",
        "outputs/07_multiclass_results/EXP-B2",
        "T12",
        ("macro_f1", "balanced_accuracy"),
    ),
    # The three-class variant, not the two-class one: `headline: true` in
    # configs/experiments.yaml, and the label space the deployed page declares
    # (Absent / Present / Unknown) is that variant's.
    DeploymentSpec(
        "murmur",
        "circor_murmur",
        "outputs/08_circor_external_validation/EXP-C1-three_class",
        "T13",
        ("macro_f1", "balanced_accuracy"),
    ),
    DeploymentSpec(
        "outcome",
        "circor_outcome",
        "outputs/08_circor_external_validation/EXP-C2",
        "T14",
        ("sensitivity", "balanced_accuracy"),
    ),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def spec_for(task: str) -> DeploymentSpec:
    """The declared deployment for ``task``."""
    for spec in DEPLOYMENTS:
        if spec.task == task:
            return spec
    raise DeploymentError(
        "no deployment declared for " + repr(task) + "; declared: "
        + ", ".join(item.task for item in DEPLOYMENTS)
    )


def rank_models(spec: DeploymentSpec) -> dict[str, Any]:
    """Rank one run's models under ``spec.rule``, and note accuracy's choice.

    Ensembles are ranked alongside their members: M6 and M7 are models the
    experiment ran, and excluding them would be choosing the winner by a rule
    that was never declared.
    """
    import pandas as pd

    path = _project_root() / spec.run_dir / "aggregate_metrics.csv"
    if not path.is_file():
        raise DeploymentError(
            "no ranking source for " + spec.task + ": " + str(path) + " does not exist"
        )
    frame = pd.read_csv(path)

    columns = [metric + "_mean" for metric in spec.rule]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DeploymentError(
            spec.task + ": " + path.name + " has no " + ", ".join(missing)
            + " -- the selection rule cannot be applied to it"
        )

    ordered = frame.sort_values(columns, ascending=False, kind="mergesort")
    ranking = [
        {
            "model_id": str(row["model_id"]),
            **{metric: float(row[metric + "_mean"]) for metric in spec.rule},
        }
        for _, row in ordered.iterrows()
    ]
    accuracy_choice = None
    if "accuracy_mean" in frame.columns:
        accuracy_choice = str(
            frame.sort_values("accuracy_mean", ascending=False, kind="mergesort").iloc[0][
                "model_id"
            ]
        )
    return {
        "selected_model_id": ranking[0]["model_id"],
        "rule": list(spec.rule),
        "ranking": ranking,
        "accuracy_would_have_chosen": accuracy_choice,
        "ranking_source": spec.run_dir + "/aggregate_metrics.csv",
        "n_models_ranked": len(ranking),
    }


def _deploy_fold(data: Any) -> Any:
    """Every labelled row in the training side, nothing held out."""
    import numpy as np

    from src.evaluation.cv import Fold

    n = len(data.y)
    return Fold(
        task=data.task,
        scheme=DEPLOY_FOLD_LABEL,
        # -1/-1, so `Fold.label` reads `r-1f-1`: the search cache is keyed by
        # that label as well as by the training uids, and a deployment search
        # must be unmistakable in the cache directory rather than sitting next
        # to a real r0f0 entry distinguished only by its hash suffix.
        repeat=-1,
        fold=-1,
        train_uids=tuple(data.record_uids),
        test_uids=(),
        train_groups=tuple(str(group) for group in data.groups),
        test_groups=(),
        train_index=np.arange(n, dtype=int),
        test_index=np.asarray([], dtype=int),
    )


def search_deployed_point(
    model_id: str, data: Any, *, cache_dir: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    """One search per searchable model, over every labelled record.

    Returns the payload per searched id: M6 and M7 declare no hyperparameters of
    their own, so what gets searched for them is their members.
    """
    from src.evaluation.tuned import NestedSearchPlanner

    planner = NestedSearchPlanner(cache_dir=cache_dir)
    fold = _deploy_fold(data)
    points: dict[str, dict[str, Any]] = {}
    for name in planner._searched(model_id):
        log.info("deployment search: %s on %d records of %s", name, len(data.y), data.task)
        points[name] = planner.search_point(name, fold, data)
    return points


def deploy_task(
    spec: DeploymentSpec,
    *,
    model_id: str | None = None,
    cache_dir: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Rank, search, refit on every labelled record, and save ``<task>/final``."""
    import numpy as np

    from src.evaluation.tuned import ENSEMBLE_MEMBERS
    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models import registry as reg
    from src.models import smoke as sm

    selection = rank_models(spec)
    chosen = model_id or selection["selected_model_id"]

    data = sm.load_task_data(spec.data_task)
    points = search_deployed_point(chosen, data, cache_dir=cache_dir)

    if chosen in {"M6", "M7"}:
        member_params = {
            member: points[member]["best_params"]
            for member in ENSEMBLE_MEMBERS
            if member in points
        }
        estimator = est.make_ensemble(
            chosen,
            groups=np.asarray(data.groups, dtype=object),
            member_params=member_params,
        )
        parameters: dict[str, Any] = {"members": member_params}
    else:
        parameters = dict(points[chosen]["best_params"])
        estimator = est.build_estimator(chosen, **parameters)

    built = pl.build_pipeline(estimator, y=data.y, n_features=len(data.feature_names))
    started = time.perf_counter()
    built.fit(data.X, data.y)
    fit_seconds = time.perf_counter() - started

    saved = reg.save_model(
        built,
        model_id="final",
        task=spec.task,
        feature_names=data.feature_names,
        fit_seconds=fit_seconds,
        X_sample=data.X[: min(64, len(data.y))],
        fold="all-records",
        root=root,
        extra={
            "selected_model_id": chosen,
            "selection_rule": selection["rule"],
            "selection_ranking": selection["ranking"],
            "accuracy_would_have_chosen": selection["accuracy_would_have_chosen"],
            "ranking_source": selection["ranking_source"],
            "published_table": spec.table_id,
            "hyperparameters": parameters,
            "hyperparameter_source": (
                "one "
                + str(next(iter(points.values()))["method"])
                + " search over every labelled record of "
                + spec.data_task
                + " at the run's own budget (fold "
                + DEPLOY_FOLD_LABEL
                + ")"
            ),
            "search": {
                name: {
                    "objective": payload["objective"],
                    "inner_best_score": payload["inner_best_score"],
                    "n_trials": payload["n_trials"],
                    "n_trials_ok": payload["n_trials_ok"],
                    "seconds": payload["seconds"],
                    "n_inner_folds": payload["n_inner_folds"],
                }
                for name, payload in points.items()
            },
            "data_task": spec.data_task,
            "n_records_fitted": len(data.y),
            "n_classes": len({int(value) for value in data.y}),
            "note": (
                "Refitted on every labelled "
                + spec.data_task
                + " record. Its performance estimate is "
                + spec.run_dir.rsplit("/", 1)[-1]
                + "'s cross-validation ("
                + spec.table_id
                + "), NOT anything measured on these rows. The deployed "
                + "hyperparameter point is a single search over all records, the "
                + "same split between a nested estimate and a deployed fit that "
                + "models_saved/binary/final/ records."
            ),
            "disclaimer": (
                "Academic screening and decision-support prototype. Not a "
                "diagnostic device and not a substitute for clinical assessment."
            ),
        },
    )
    log.info(
        "deployed %s: %s refit on %d records -> %s (%.3f MB)",
        spec.task,
        chosen,
        len(data.y),
        saved.path,
        saved.size_mb,
    )
    return {
        "task": spec.task,
        "selected_model_id": chosen,
        "selection": selection,
        "hyperparameters": parameters,
        "n_records_fitted": len(data.y),
        "fit_seconds": round(fit_seconds, 3),
        "path": str(saved.path),
        "size_mb": saved.size_mb,
    }
