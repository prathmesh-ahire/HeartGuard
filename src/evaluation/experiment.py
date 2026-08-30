"""The experiment runner: one declared run, one output contract (Phase 63).

Everything from Part VII onward is "fit these models on those folds and write
the numbers down". Doing that once, here, rather than once per phase is what
makes EXP-A1 and EXP-A2 comparable fold-for-fold at all -- the two runs share
this code, this fold map and this metric set, so a difference between them is a
difference in the models rather than a difference in two scripts that were
written a week apart.

Three things this module refuses to do, each for a reason:

* **It does not derive folds.** They come from the DA-07 map through
  :mod:`src.evaluation.cv`, resolved against the matrix row order.
* **It does not decide what a model is.** A :class:`ModelPlanner` does, and the
  planner is what changes between a baseline run and a tuned one. The runner
  never knows whether it is running default or searched hyperparameters.
* **It does not aggregate away the per-fold values.** ``per_fold_metrics.csv``
  is the primary artifact and the aggregate is derived from it. The paired
  tests in Phase 81 need the 25 individual numbers; a mean and an SD cannot be
  un-averaged.

Resume-on-restart (T63.4) is per (model, fold). A completed unit is reloaded
from its checkpoint only when the **unit key** matches -- a hash over the
experiment, model, fold membership, planner recipe, pipeline config, feature
registry fingerprint and seed. A checkpoint written under a different config is
recomputed rather than reused, because a partially-stale results table is worse
than a slow one.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "CHECKPOINT_DIRNAME",
    "OUTPUT_CONTRACT",
    "ExperimentError",
    "Experiment",
    "PlannedModel",
    "ModelPlanner",
    "DefaultPlanner",
    "UnitResult",
    "ExperimentResult",
    "assert_declared_class_weight",
    "run_experiment",
    "write_outputs",
    "load_per_fold_metrics",
]

log = get_logger("evaluation.experiment")

CHECKPOINT_DIRNAME = "_checkpoints"

#: T63.2. Also the value of ``defaults.output_contract`` in
#: ``configs/experiments.yaml`` -- asserted equal at load time so the two cannot
#: drift apart silently.
OUTPUT_CONTRACT: tuple[str, ...] = (
    "per_fold_metrics.csv",
    "aggregate_metrics.csv",
    "predictions.parquet",
    "confusion_matrices.json",
    "config_snapshot.yaml",
    "run_manifest.json",
)

#: Written alongside the contract. Not part of it, but Phase 81's paired tests
#: have to be able to *demonstrate* that two runs saw identical folds, and the
#: only way to do that afterwards is to have written the membership down.
MEMBERSHIP_FILENAME = "fold_membership.parquet"


class ExperimentError(RuntimeError):
    """The experiment cannot be assembled or run as declared."""


#: Sentinel for "the members do not agree, so the ensemble has no single value".
_MIXED = object()


def _resolved_class_weight(model_id: str) -> tuple[str, Any]:
    """One model's effective ``class_weight``, following M6/M7 down to its members.

    Returns ``(source, value)`` where ``source`` names what was inspected, so a
    failure says *which* member disagreed rather than only that the ensemble did.
    """
    from src.models.estimators import model_spec

    spec = model_spec(model_id)
    members = [str(name) for name in (spec.get("members") or [])]
    if members:
        values = {name: _resolved_class_weight(name)[1] for name in members}
        if len({repr(value) for value in values.values()}) != 1:
            described = ", ".join(
                name + "=" + repr(value) for name, value in values.items()
            )
            return model_id + " members " + described, _MIXED
        return model_id + " members", next(iter(values.values()))
    return model_id + " defaults", (spec.get("defaults") or {}).get("class_weight")


def assert_declared_class_weight(experiment: Experiment) -> None:
    """``class_weight`` in experiments.yaml is a claim about the models -- check it.

    The field is **not** applied as an override, deliberately. ``class_weight`` is
    a *searchable* dimension for M3 and M4 (``choices: ["balanced", null]``), so
    forcing it would silently delete half of those spaces from every nested run.
    It is verified against ``configs/models.yaml`` instead: if an experiment
    declares ``class_weight: balanced`` and any model it runs does not carry it,
    the run stops.

    Before Phase 66 the field was read into ``Experiment.class_weight`` and never
    used again -- decorative config, which is worse than no config, because
    editing it looked like it did something. See ``Docs/note.md``.
    """
    declared = experiment.class_weight
    if declared is None:
        return
    wrong = []
    for model_id in experiment.models:
        source, value = _resolved_class_weight(model_id)
        if value is _MIXED or value != declared:
            wrong.append(source + " -> " + repr(value))
    if wrong:
        raise ExperimentError(
            experiment.exp_id
            + " declares class_weight="
            + repr(declared)
            + " but configs/models.yaml disagrees: "
            + "; ".join(wrong)
        )



# ---------------------------------------------------------------------------
# T63.1 -- the Experiment object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """One entry of ``configs/experiments.yaml``, validated on the way in.

    Frozen because an experiment definition is an input, not a working value. A
    run that quietly edited its own declaration -- dropping a model that failed,
    say -- would produce an output whose config snapshot no longer describes it.
    """

    exp_id: str
    title: str
    phase: int
    task: str
    dataset: str
    models: tuple[str, ...]
    cv: str
    output_section: str
    tuned: bool = False
    nested: bool = False
    #: A named side-run of the same declaration -- T65.3's SO-04 subset variant.
    #: It gets its own contract-complete folder (``<EXP-ID>-<variant>``) rather
    #: than extra columns in the headline run, so neither can overwrite or be
    #: mistaken for the other.
    variant: str = ""
    emits: tuple[str, ...] = ()
    label_space: dict[str, int] = field(default_factory=dict)
    class_weight: str | None = None
    notes: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, exp_id: str, *, config: Any = None) -> Experiment:
        from src.utils.config import load_config

        loaded = load_config("experiments") if config is None else config
        table = loaded.require("experiments")
        if exp_id not in table:
            raise ExperimentError(
                "configs/experiments.yaml declares no "
                + repr(exp_id)
                + "; known: "
                + ", ".join(sorted(table))
            )
        spec = dict(table[exp_id])
        defaults = dict(loaded.require("defaults"))

        contract = tuple(defaults.get("output_contract", ()))
        if contract and contract != OUTPUT_CONTRACT:
            raise ExperimentError(
                "configs/experiments.yaml defaults.output_contract "
                + str(list(contract))
                + " disagrees with src.evaluation.experiment.OUTPUT_CONTRACT "
                + str(list(OUTPUT_CONTRACT))
            )

        if spec.get("analysis_only", False):
            raise ExperimentError(
                exp_id
                + " is analysis_only: it stratifies predictions produced by "
                + "other experiments and is not run through this runner"
            )
        for key in ("task", "cv", "output_section"):
            if key not in spec:
                raise ExperimentError(exp_id + " has no " + key + " in its declaration")
        if not spec.get("models"):
            if spec.get("configurations"):
                # EXP-F1/EXP-F2 are ablation grids: their unit of work is a
                # *configuration* (a feature-family subset, an optimization
                # stage), not a model list. Phases 74-75 extend the runner to
                # loop over them. Refusing here rather than inventing a model
                # list keeps that a deliberate decision.
                raise ExperimentError(
                    exp_id
                    + " declares `configurations`, not `models`: it is an "
                    + "ablation grid and needs the Phase 74/75 runner"
                )
            raise ExperimentError(exp_id + " declares no models")

        return cls(
            exp_id=exp_id,
            title=str(spec.get("title", exp_id)),
            phase=int(spec.get("phase", 0)),
            task=str(spec["task"]),
            dataset=str(spec.get("dataset", "")),
            models=tuple(str(m) for m in spec["models"]),
            cv=str(spec["cv"]),
            output_section=str(spec["output_section"]),
            tuned=bool(spec.get("tuned", False)),
            nested=bool(spec.get("nested", False)),
            emits=tuple(str(e) for e in spec.get("emits", ())),
            label_space=dict(spec.get("label_space", {}) or {}),
            class_weight=spec.get("class_weight"),
            notes=str(spec.get("notes", "") or "").strip(),
            spec=spec,
            defaults=defaults,
        )

    # -- derived properties -------------------------------------------------

    @property
    def n_classes(self) -> int:
        return len(self.label_space)

    @property
    def is_binary(self) -> bool:
        """Two declared labels. Not "the fold happens to hold two classes"."""
        return self.n_classes == 2

    @property
    def scoring(self) -> str:
        table = dict(self.defaults.get("scoring", {}))
        return str(table["binary" if self.is_binary else "multiclass"])

    @property
    def report_metrics(self) -> tuple[str, ...]:
        key = "report_metrics_binary" if self.is_binary else "report_metrics_multiclass"
        return tuple(str(m) for m in self.defaults.get(key, ()))

    @property
    def selection_rule(self) -> tuple[str, ...]:
        return tuple(str(m) for m in self.defaults.get("selection_rule", ()))

    @property
    def seed(self) -> int:
        return int(self.defaults.get("seed", 42))

    @property
    def class_names(self) -> tuple[str, ...]:
        """Label names in label order, so per-class columns are readable."""
        if not self.label_space:
            return ()
        return tuple(
            name for name, _ in sorted(self.label_space.items(), key=lambda kv: kv[1])
        )

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(sorted(int(v) for v in self.label_space.values()))

    def cv_scheme(self, *, config: Any = None) -> dict[str, Any]:
        from src.utils.config import load_config

        loaded = load_config("experiments") if config is None else config
        schemes = loaded.require("cv_schemes")
        if self.cv not in schemes:
            raise ExperimentError(
                self.exp_id + " names cv scheme " + repr(self.cv) + " which is not declared"
            )
        return dict(schemes[self.cv])

    @property
    def run_id(self) -> str:
        """``EXP-A2`` or ``EXP-A2-so04_subset``. What names the output folder."""
        return self.exp_id + ("-" + self.variant if self.variant else "")

    def output_dir(self, out_dir: str | Path | None = None) -> Path:
        """``outputs/<section>/<EXP-ID>/`` -- T63.3. A variant gets its own folder."""
        from src.utils.config import load_config
        from src.utils.io import ensure_dir

        if out_dir is not None:
            root = Path(out_dir)
        else:
            key_by_dir = {
                "06_binary_results": "outputs.binary_results",
                "07_multiclass_results": "outputs.multiclass_results",
                "08_circor_external_validation": "outputs.circor_external_validation",
                "09_ablation": "outputs.ablation",
                "10_robustness": "outputs.robustness",
            }
            key = key_by_dir.get(self.output_section)
            if key is None:
                raise ExperimentError(
                    self.exp_id
                    + ": output_section "
                    + repr(self.output_section)
                    + " has no entry in configs/paths.yaml"
                )
            root = Path(load_config("paths").require(key))
        return Path(ensure_dir(root / self.run_id))

    def as_snapshot(self) -> dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "run_id": self.run_id,
            "variant": self.variant,
            "title": self.title,
            "phase": self.phase,
            "task": self.task,
            "dataset": self.dataset,
            "models": list(self.models),
            "cv": self.cv,
            "cv_scheme": self.cv_scheme(),
            "tuned": self.tuned,
            "nested": self.nested,
            "emits": list(self.emits),
            "label_space": dict(self.label_space),
            "scoring": self.scoring,
            "report_metrics": list(self.report_metrics),
            "selection_rule": list(self.selection_rule),
            "seed": self.seed,
            "output_section": self.output_section,
            "declaration": self.spec,
        }


# ---------------------------------------------------------------------------
# planners -- what "the model" means for this run
# ---------------------------------------------------------------------------


@dataclass
class PlannedModel:
    """One model, built for one fold: how to construct it and what to record."""

    factory: Callable[[], Any]
    params: dict[str, Any] = field(default_factory=dict)
    pipeline_config: dict[str, Any] | None = None
    note: str = ""
    #: Anything the planner produced that belongs in the results table -- an
    #: inner search score, the number of trials it ran, the fold it tuned on.
    extra: dict[str, Any] = field(default_factory=dict)


class ModelPlanner:
    """Turns ``(model_id, fold)`` into a fitted-per-fold model specification.

    Split into two methods on purpose. ``key_material`` must be **cheap and
    deterministic**: it is evaluated before the checkpoint lookup, so a nested
    planner that runs a 20-minute search inside ``plan`` still resumes without
    running it. Putting the search's *recipe* in the key and the search's
    *result* in the plan is what makes that safe -- change the recipe and the
    key changes, so a stale checkpoint is never reused.
    """

    name = "planner"

    def key_material(self, model_id: str, fold: Any, data: Any) -> dict[str, Any]:
        raise NotImplementedError

    def plan(self, model_id: str, fold: Any, data: Any) -> PlannedModel:
        raise NotImplementedError


class DefaultPlanner(ModelPlanner):
    """Config defaults, no tuning -- the EXP-A1 baseline (T64.1).

    The ensembles get the **training fold's** subject groups so their inner CV
    is subject-aware, exactly as the Phase 46 smoke runner does. That closure is
    the reason a planner takes the fold rather than only the model id.
    """

    name = "default"

    def __init__(self, *, pipeline_config: dict[str, Any] | None = None) -> None:
        self.pipeline_config = pipeline_config

    def key_material(self, model_id: str, fold: Any, data: Any) -> dict[str, Any]:
        return {"planner": self.name, "model_id": model_id, "params": {}}

    def plan(self, model_id: str, fold: Any, data: Any) -> PlannedModel:
        from src.models import estimators as est

        if model_id in {"M6", "M7"}:
            train_groups = np.asarray(data.groups, dtype=object)[
                np.asarray(fold.train_index, dtype=int)
            ]

            def build_ensemble() -> Any:
                return est.make_ensemble(model_id, groups=train_groups)

            factory: Callable[[], Any] = build_ensemble
        else:

            def build() -> Any:
                return est.build_estimator(model_id)

            factory = build

        return PlannedModel(
            factory=factory,
            params={},
            pipeline_config=self.pipeline_config,
            note="config defaults",
        )


# ---------------------------------------------------------------------------
# one unit of work: one model on one fold
# ---------------------------------------------------------------------------


@dataclass
class UnitResult:
    """One (model, fold): its predictions, its metrics, and how it was built."""

    exp_id: str
    model_id: str
    task: str
    repeat: int
    fold: int
    fold_label: str
    n_train: int
    n_test: int
    n_features: int
    fit_seconds: float
    predict_seconds: float
    classes: tuple[int, ...]
    metrics: dict[str, float]
    confusion: list[list[int]]
    params: dict[str, Any]
    note: str
    extra: dict[str, Any]
    unit_key: str
    record_uids: tuple[str, ...]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    resumed: bool = False

    def metric_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "exp_id": self.exp_id,
            "model_id": self.model_id,
            "task": self.task,
            "repeat": self.repeat,
            "fold": self.fold,
            "fold_label": self.fold_label,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
        }
        for name, value in self.metrics.items():
            row[name] = value
        row["fit_seconds"] = round(float(self.fit_seconds), 6)
        row["predict_seconds"] = round(float(self.predict_seconds), 6)
        for name, value in sorted(self.extra.items()):
            row["planner_" + name] = value
        row["params"] = json.dumps(self.params, sort_keys=True, default=str)
        row["note"] = self.note
        return row

    def prediction_frame(self) -> Any:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "exp_id": self.exp_id,
                "model_id": self.model_id,
                "repeat": self.repeat,
                "fold": self.fold,
                "fold_label": self.fold_label,
                "record_uid": list(self.record_uids),
                "y_true": np.asarray(self.y_true).astype(int),
                "y_pred": np.asarray(self.y_pred).astype(int),
            }
        )
        if self.y_proba is not None:
            proba = np.asarray(self.y_proba, dtype=float)
            for index, klass in enumerate(self.classes):
                frame["proba_" + str(klass)] = proba[:, index]
        return frame

    # -- checkpointing ------------------------------------------------------

    def as_checkpoint(self) -> dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "model_id": self.model_id,
            "task": self.task,
            "repeat": self.repeat,
            "fold": self.fold,
            "fold_label": self.fold_label,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "classes": list(self.classes),
            "metrics": self.metrics,
            "confusion": self.confusion,
            "params": self.params,
            "note": self.note,
            "extra": self.extra,
            "unit_key": self.unit_key,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any], predictions: Any) -> UnitResult:
        classes = tuple(int(c) for c in payload["classes"])
        proba_columns = ["proba_" + str(c) for c in classes]
        has_proba = all(column in predictions.columns for column in proba_columns)
        return cls(
            exp_id=str(payload["exp_id"]),
            model_id=str(payload["model_id"]),
            task=str(payload["task"]),
            repeat=int(payload["repeat"]),
            fold=int(payload["fold"]),
            fold_label=str(payload["fold_label"]),
            n_train=int(payload["n_train"]),
            n_test=int(payload["n_test"]),
            n_features=int(payload["n_features"]),
            fit_seconds=float(payload["fit_seconds"]),
            predict_seconds=float(payload["predict_seconds"]),
            classes=classes,
            metrics=dict(payload["metrics"]),
            confusion=[list(row) for row in payload["confusion"]],
            params=dict(payload["params"]),
            note=str(payload["note"]),
            extra=dict(payload.get("extra", {})),
            unit_key=str(payload["unit_key"]),
            record_uids=tuple(predictions["record_uid"].astype(str)),
            y_true=predictions["y_true"].to_numpy(dtype=int),
            y_pred=predictions["y_pred"].to_numpy(dtype=int),
            y_proba=(
                predictions[proba_columns].to_numpy(dtype=float) if has_proba else None
            ),
            resumed=True,
        )


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _unit_key(
    exp: Experiment,
    model_id: str,
    fold: Any,
    data: Any,
    planner: ModelPlanner,
    pipeline_config: dict[str, Any] | None,
) -> str:
    """Everything that would change this unit's numbers, hashed.

    Fold *membership* is hashed rather than the fold label: two runs can agree on
    "r0f0" and disagree on which records that names, and a checkpoint reused
    across that difference is a silently wrong results table.
    """
    from src.feature_extraction.registry import registry_fingerprint

    material = {
        "exp_id": exp.run_id,
        "task": exp.task,
        "cv": exp.cv,
        "model_id": model_id,
        "fold": fold.label,
        "train_uids": _stable_hash({"u": sorted(fold.train_uids)}),
        "test_uids": _stable_hash({"u": sorted(fold.test_uids)}),
        "n_features": int(np.asarray(data.X).shape[1]),
        "feature_registry": registry_fingerprint(),
        "pipeline": pipeline_config,
        "seed": exp.seed,
        "planner": planner.key_material(model_id, fold, data),
    }
    return _stable_hash(material)


def _checkpoint_paths(directory: Path, model_id: str, fold_label: str) -> tuple[Path, Path]:
    stem = model_id + "__" + fold_label
    return directory / (stem + ".json"), directory / (stem + ".parquet")


def _score(
    exp: Experiment,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: Sequence[int],
) -> tuple[dict[str, float], list[list[int]]]:
    from src.evaluation import metrics as mt

    labels = tuple(int(c) for c in classes)
    if len(labels) == 2:
        scores = mt.binary_metrics(y_true, y_pred, y_proba, labels=labels, positive_label=1)
    else:
        names = exp.class_names if len(exp.class_names) == len(labels) else None
        scores = mt.multiclass_metrics(
            y_true, y_pred, y_proba, labels=labels, class_names=names
        )
    if y_proba is not None:
        scores["brier"] = mt.brier_score(y_true, y_proba, labels=labels)
        scores["ece"] = mt.expected_calibration_error(y_true, y_proba, labels=labels)
    matrix = mt.confusion(y_true, y_pred, labels)
    return (
        {key: float(value) for key, value in scores.items()},
        [[int(cell) for cell in row] for row in matrix],
    )


def _run_unit(
    exp: Experiment,
    model_id: str,
    fold: Any,
    data: Any,
    planner: ModelPlanner,
    unit_key: str,
) -> UnitResult:
    from src.evaluation import cv as cv_module
    from src.models import estimators as est
    from src.models import pipeline as pl

    train_index = np.asarray(fold.train_index, dtype=int)
    test_index = np.asarray(fold.test_index, dtype=int)
    features = np.asarray(data.X)
    targets = np.asarray(data.y, dtype=int)
    groups = np.asarray(data.groups, dtype=object)

    # Rule 3, checked against the arrays actually fed to the estimator -- a
    # correct map resolved against a mis-ordered matrix passes a map-level check
    # and still trains on the wrong rows.
    cv_module.assert_group_disjoint(fold)
    shared = set(groups[train_index].tolist()) & set(groups[test_index].tolist())
    if shared:
        raise cv_module.LeakageError(
            exp.exp_id
            + " "
            + model_id
            + " "
            + fold.label
            + ": "
            + str(len(shared))
            + " subject group(s) on both sides of the split after resolving "
            "against the matrix"
        )
    if set(train_index.tolist()) & set(test_index.tolist()):
        raise cv_module.LeakageError(
            exp.exp_id + " " + model_id + " " + fold.label + ": train and test rows overlap"
        )

    planned = planner.plan(model_id, fold, data)
    built = pl.build_pipeline(
        planned.factory(),
        config=planned.pipeline_config,
        y=targets[train_index],
        n_features=int(features.shape[1]),
    )

    started = time.perf_counter()
    built.fit(features[train_index], targets[train_index])
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    y_pred = np.asarray(built.predict(features[test_index]))
    y_proba = (
        np.asarray(built.predict_proba(features[test_index]))
        if est.has_predict_proba(built)
        else None
    )
    predict_seconds = time.perf_counter() - started

    classes = tuple(
        int(c) for c in np.asarray(getattr(built, "classes_", np.unique(targets)))
    )
    metrics, matrix = _score(exp, targets[test_index], y_pred, y_proba, classes)

    result = UnitResult(
        exp_id=exp.exp_id,
        model_id=model_id,
        task=exp.task,
        repeat=int(fold.repeat),
        fold=int(fold.fold),
        fold_label=fold.label,
        n_train=int(train_index.size),
        n_test=int(test_index.size),
        n_features=int(features.shape[1]),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        classes=classes,
        metrics=metrics,
        confusion=matrix,
        params=dict(planned.params),
        note=planned.note,
        extra=dict(planned.extra),
        unit_key=unit_key,
        record_uids=tuple(str(uid) for uid in fold.test_uids),
        y_true=targets[test_index],
        y_pred=y_pred.astype(int),
        y_proba=y_proba,
    )
    headline = "balanced_accuracy" if exp.is_binary else "macro_f1"
    log.info(
        "%s %s %s: %s %.4f, fit %.1fs",
        exp.exp_id,
        model_id,
        fold.label,
        headline,
        metrics.get(headline, float("nan")),
        fit_seconds,
    )
    return result


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    """Every unit of one experiment, plus what it took to produce them."""

    exp: Experiment
    units: list[UnitResult] = field(default_factory=list)
    folds: tuple[Any, ...] = ()
    seconds: float = 0.0
    n_resumed: int = 0
    n_computed: int = 0

    @property
    def models(self) -> tuple[str, ...]:
        seen: list[str] = []
        for unit in self.units:
            if unit.model_id not in seen:
                seen.append(unit.model_id)
        return tuple(seen)

    def per_fold_frame(self) -> Any:
        import pandas as pd

        frame = pd.DataFrame([unit.metric_row() for unit in self.units])
        if frame.empty:
            return frame
        return frame.sort_values(["model_id", "repeat", "fold"]).reset_index(drop=True)

    def predictions_frame(self) -> Any:
        import pandas as pd

        frames = [unit.prediction_frame() for unit in self.units]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def aggregate_frame(self) -> Any:
        """Mean +/- SD per model across folds (T64.6).

        ``nanstd`` with ``ddof=1``: the sample SD, because 25 folds are a sample
        of the fold population and the population SD would understate it.
        ``nan`` values are excluded rather than treated as zero -- a fold whose
        OvR AUC is undefined has no AUC, it does not have an AUC of nothing.
        """
        import pandas as pd

        per_fold = self.per_fold_frame()
        if per_fold.empty:
            return pd.DataFrame()

        skip = {
            "exp_id", "model_id", "task", "repeat", "fold", "fold_label",
            "params", "note",
        }
        numeric = [
            column
            for column in per_fold.columns
            if column not in skip
            and pd.api.types.is_numeric_dtype(per_fold[column])
        ]

        rows = []
        for model_id, block in per_fold.groupby("model_id", sort=False):
            row: dict[str, Any] = {
                "exp_id": self.exp.exp_id,
                "model_id": model_id,
                "task": self.exp.task,
                "cv": self.exp.cv,
                "n_folds": len(block),
            }
            for column in numeric:
                values = block[column].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                row[column + "_mean"] = (
                    float(np.mean(finite)) if finite.size else float("nan")
                )
                row[column + "_sd"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
                )
                row[column + "_n"] = int(finite.size)
            rows.append(row)
        return pd.DataFrame(rows)

    def confusion_payload(self) -> dict[str, Any]:
        labels = list(self.units[0].classes) if self.units else []
        names = (
            list(self.exp.class_names)
            if len(self.exp.class_names) == len(labels)
            else [str(label) for label in labels]
        )
        payload: dict[str, Any] = {
            "exp_id": self.exp.exp_id,
            "task": self.exp.task,
            "labels": labels,
            "class_names": names,
            "note": (
                "`total` is the element-wise sum over folds. For a repeated CV "
                "map that counts every record once per repeat, so its support is "
                "n_records x n_repeats, not n_records."
            ),
            "models": {},
        }
        for model_id in self.models:
            units = [unit for unit in self.units if unit.model_id == model_id]
            total = np.zeros((len(labels), len(labels)), dtype=int)
            per_fold = {}
            for unit in units:
                matrix = np.asarray(unit.confusion, dtype=int)
                total += matrix
                per_fold[unit.fold_label] = matrix.tolist()
            payload["models"][model_id] = {
                "total": total.tolist(),
                "per_fold": per_fold,
                "n_folds": len(units),
            }
        return payload


def run_experiment(
    exp: Experiment | str,
    *,
    data: Any = None,
    models: Sequence[str] | None = None,
    repeats: Sequence[int] | None = None,
    folds: Sequence[int] | None = None,
    planner: ModelPlanner | None = None,
    pipeline_config: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    resume: bool = True,
) -> ExperimentResult:
    """Run one experiment over its declared fold map, resuming completed units.

    ``repeats``/``folds`` subset the map. That is a **budget** decision and is
    recorded as one: the fold labels in every output say which folds actually
    ran, so a five-fold run can never be mistaken for the 25-fold protocol.
    """
    import pandas as pd

    from src.evaluation import cv as cv_module
    from src.models import smoke as sm
    from src.utils.io import ensure_dir

    experiment = Experiment.load(exp) if isinstance(exp, str) else exp
    chosen_models = tuple(models) if models else experiment.models
    unknown = [m for m in chosen_models if m not in experiment.models]
    if unknown:
        raise ExperimentError(
            experiment.exp_id
            + " does not declare model(s) "
            + ", ".join(unknown)
            + "; declared: "
            + ", ".join(experiment.models)
        )

    assert_declared_class_weight(experiment)

    loaded = sm.load_task_data(experiment.task) if data is None else data
    resolved = cv_module.resolve_folds(
        cv_module.load_folds(experiment.task), loaded.record_uids
    )
    selected = tuple(
        fold
        for fold in resolved
        if (repeats is None or fold.repeat in set(repeats))
        and (folds is None or fold.fold in set(folds))
    )
    if not selected:
        raise ExperimentError(
            experiment.exp_id
            + ": no fold matches repeats="
            + str(repeats)
            + " folds="
            + str(folds)
        )

    declared_folds = int(experiment.cv_scheme().get("total_folds", len(resolved)))
    if len(resolved) != declared_folds:
        raise ExperimentError(
            experiment.exp_id
            + ": the DA-07 map holds "
            + str(len(resolved))
            + " folds for task "
            + experiment.task
            + " but scheme "
            + experiment.cv
            + " declares "
            + str(declared_folds)
        )

    active_planner = planner or DefaultPlanner(pipeline_config=pipeline_config)
    directory = experiment.output_dir(out_dir)
    checkpoints = Path(ensure_dir(directory / CHECKPOINT_DIRNAME))

    result = ExperimentResult(exp=experiment, folds=selected)
    started = time.perf_counter()

    for model_id in chosen_models:
        for fold in selected:
            key = _unit_key(
                experiment, model_id, fold, loaded, active_planner, pipeline_config
            )
            meta_path, pred_path = _checkpoint_paths(checkpoints, model_id, fold.label)
            unit = None

            if resume and meta_path.is_file() and pred_path.is_file():
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    log.warning("unreadable checkpoint %s (%s); recomputing", meta_path, error)
                    payload = {}
                if payload.get("unit_key") == key:
                    unit = UnitResult.from_checkpoint(payload, pd.read_parquet(pred_path))
                    result.n_resumed += 1
                    log.info("resumed %s %s %s", experiment.exp_id, model_id, fold.label)
                elif payload:
                    log.warning(
                        "%s %s %s: checkpoint key %s != %s -- recomputing",
                        experiment.exp_id,
                        model_id,
                        fold.label,
                        payload.get("unit_key"),
                        key,
                    )

            if unit is None:
                unit = _run_unit(experiment, model_id, fold, loaded, active_planner, key)
                unit.prediction_frame().to_parquet(pred_path, index=False)
                meta_path.write_text(
                    json.dumps(unit.as_checkpoint(), indent=2, default=str),
                    encoding="utf-8",
                )
                result.n_computed += 1

            result.units.append(unit)

    result.seconds = time.perf_counter() - started
    log.info(
        "%s: %d unit(s) -- %d computed, %d resumed -- in %.1f s",
        experiment.exp_id,
        len(result.units),
        result.n_computed,
        result.n_resumed,
        result.seconds,
    )
    return result


# ---------------------------------------------------------------------------
# T63.2 / T63.3 -- the output contract
# ---------------------------------------------------------------------------


def write_outputs(
    result: ExperimentResult,
    *,
    out_dir: str | Path | None = None,
    run: Any = None,
    command: str = "",
) -> dict[str, Path]:
    """Write the six contract files (plus fold membership) and register them.

    ``run_manifest.json`` is written **into the experiment directory**, not only
    appended to the global manifest: a results folder that has to be joined
    against a file elsewhere to find out what produced it is one refactor away
    from being unattributable.
    """
    import pandas as pd
    import yaml

    from src.utils.evidence import register_evidence
    from src.utils.io import save_csv, save_json

    exp = result.exp
    directory = exp.output_dir(out_dir)
    written: dict[str, Path] = {}

    written["per_fold_metrics.csv"] = save_csv(
        result.per_fold_frame(), directory / "per_fold_metrics.csv"
    )
    written["aggregate_metrics.csv"] = save_csv(
        result.aggregate_frame(), directory / "aggregate_metrics.csv"
    )

    predictions_path = directory / "predictions.parquet"
    result.predictions_frame().to_parquet(predictions_path, index=False)
    written["predictions.parquet"] = predictions_path

    written["confusion_matrices.json"] = save_json(
        result.confusion_payload(), directory / "confusion_matrices.json"
    )

    snapshot = exp.as_snapshot()
    snapshot["run"] = {
        "models_run": list(result.models),
        "folds_run": [fold.label for fold in result.folds],
        "n_folds_run": len(result.folds),
        "n_units": len(result.units),
        "n_computed": result.n_computed,
        "n_resumed": result.n_resumed,
        "seconds": round(result.seconds, 2),
        "command": command,
    }
    snapshot_path = directory / "config_snapshot.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(json.loads(json.dumps(snapshot, default=str)), sort_keys=False),
        encoding="utf-8",
    )
    written["config_snapshot.yaml"] = snapshot_path

    membership = pd.DataFrame(
        [
            {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "fold_label": fold.label,
                "record_uid": uid,
                "split": split,
            }
            for fold in result.folds
            for split, uids in (("train", fold.train_uids), ("test", fold.test_uids))
            for uid in uids
        ]
    )
    membership_path = directory / MEMBERSHIP_FILENAME
    membership.to_parquet(membership_path, index=False)
    written[MEMBERSHIP_FILENAME] = membership_path

    # The manifest goes last: it records the artifacts, so it has to be written
    # after they exist or its own listing would be a promise rather than a fact.
    from src.utils.evidence import is_inside_project
    from src.utils.run_manifest import current_run

    manifest = run if run is not None else current_run()
    # A run written outside the project -- a scratch `--out-dir`, a pytest
    # tmp_path -- must not leave provenance rows in the project's manifest
    # pointing at files that will not exist tomorrow. Same rule
    # `register_evidence` applies to the evidence index, applied here because a
    # scratch smoke run did exactly that on 2026-08-28. The embedded
    # `run_manifest.json` beside the results is written either way, so the
    # scratch run is still fully attributable -- just not from the project index.
    inside = is_inside_project(directory)
    if manifest is not None and inside:
        for path in written.values():
            manifest.record_artifact(path)
        manifest.record_timing(exp.run_id, result.seconds)
        payload = manifest.to_dict()
    elif manifest is not None:
        log.info("%s is outside the project; not recorded in the run manifest", directory)
        payload = manifest.to_dict()
    else:
        from src.utils.run_manifest import environment_info, git_info, package_versions

        payload = {
            "name": exp.run_id,
            "seed": exp.seed,
            "git": git_info(),
            "environment": environment_info(),
            "package_versions": package_versions(),
            "note": "written without an active run manifest",
        }
    written["run_manifest.json"] = save_json(payload, directory / "run_manifest.json")

    missing = [name for name in OUTPUT_CONTRACT if not (directory / name).is_file()]
    if missing:
        raise ExperimentError(
            exp.exp_id + ": output contract incomplete, missing " + ", ".join(missing)
        )

    if not inside:
        for name, path in written.items():
            log.info("%s -> %s", name, path)
        return written

    register_evidence(
        exp.run_id,
        written["per_fold_metrics.csv"],
        metric_or_asset=exp.title + " -- per-fold metrics over " + exp.cv,
        experiment_id=exp.exp_id,
        dataset=exp.dataset,
        model=", ".join(result.models),
        source_data="outputs/03_features/all_features_matrix.parquet",
        command=command,
    )
    register_evidence(
        exp.run_id + "-AGG",
        written["aggregate_metrics.csv"],
        metric_or_asset=exp.title + " -- mean +/- SD across folds",
        experiment_id=exp.exp_id,
        dataset=exp.dataset,
        command=command,
    )
    for name, path in written.items():
        log.info("%s -> %s", name, path)
    return written


def load_per_fold_metrics(exp_id: str, *, out_dir: str | Path | None = None) -> Any:
    """Read one experiment's per-fold table, for the paired tests in Part VIII."""
    import pandas as pd

    experiment = Experiment.load(exp_id)
    path = experiment.output_dir(out_dir) / "per_fold_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "no per-fold metrics at " + str(path) + "; run " + exp_id + " first"
        )
    return pd.read_csv(path)
