"""The model registry and on-disk persistence (Phase 51).

One place that answers "what is M4, how do I build it, what can be tuned about
it, and is it available on this machine" -- so that the search framework, the
experiment runner, the API and the complexity table all read the same answer.
Before this module those questions were answered by whichever script needed them,
and a model that is one thing to the search and another to the report is a
result nobody can reproduce.

**A saved model is a pipeline plus its feature-name list, never a bare
estimator.** The pipeline carries the imputer's medians and the scaler's mean and
scale, all fitted on one training fold; without them the estimator is being fed
numbers on a different scale from the ones it learned. And without the feature
names, a matrix whose columns arrive in a different order still has the right
shape, so nothing raises -- the model simply reads `mfcc_04_mean` where
`time_zcr_mean` should be and returns a confident, meaningless answer. The 138
names are a literal in the registry precisely so this check is possible; here it
is enforced at load time.

Every save records what the complexity table (T26) needs: file size on disk,
training time, and single-record inference time -- the last measured one record
at a time, because that is the deployment case. Batch throughput divided by batch
size understates per-record latency by whatever the vectorisation buys, which for
a 500-tree forest is a large factor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "MANIFEST_FILENAME",
    "MODEL_FILENAME",
    "RegistryError",
    "ModelEntry",
    "SavedModel",
    "model_ids",
    "entry",
    "entries",
    "build",
    "available",
    "registry_frame",
    "model_dir",
    "save_model",
    "load_model",
    "saved_models",
    "measure_inference",
]

log = get_logger("models.registry")

MANIFEST_FILENAME = "manifest.json"
MODEL_FILENAME = "model.joblib"


class RegistryError(RuntimeError):
    """The model cannot be described, built, saved or reloaded as asked."""


# ---------------------------------------------------------------------------
# T51.1 -- the registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    """Everything known about one model id, assembled from config and code."""

    model_id: str
    name: str
    estimator_path: str
    mandatory: bool
    implemented: bool
    is_ensemble: bool
    members: tuple[str, ...] = ()
    calibrated: bool = False
    n_search_dimensions: int = 0
    search_constraints: tuple[str, ...] = ()
    unavailable_reason: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "estimator": self.estimator_path,
            "mandatory": self.mandatory,
            "implemented": self.implemented,
            "is_ensemble": self.is_ensemble,
            "members": "|".join(self.members),
            "calibrated": self.calibrated,
            "n_search_dimensions": self.n_search_dimensions,
            "search_constraints": "|".join(self.search_constraints),
            "unavailable_reason": self.unavailable_reason,
        }


def model_ids() -> tuple[str, ...]:
    """Every id declared in config, in declaration order -- M1..M9."""
    from src.utils.config import load_config

    declared = load_config("models").get("models") or {}
    return tuple(str(key) for key in declared)


def entry(model_id: str) -> ModelEntry:
    """Describe one model without building it.

    Deliberately does not construct the estimator: this is what a report or a
    dashboard needs, and it must work for a model whose optional dependency is
    missing. Availability is a separate question, answered by :func:`available`.
    """
    from src.models import estimators as est
    from src.models import spaces

    spec = est.model_spec(model_id)
    implemented = model_id in est.IMPLEMENTED_MODELS

    dimensions = 0
    constraints: tuple[str, ...] = ()
    try:
        space = spaces.load_space(model_id)
        dimensions = len(space)
        constraints = tuple(item.name for item in space.constraints)
    except spaces.SpaceError:  # a model may legitimately declare no space
        pass

    reason = ""
    if not implemented:
        reason = est._ADDED_IN_PHASE.get(model_id, "not implemented")
    elif model_id == "M8":
        capability = est.m8_capability()
        if not capability.available:
            reason = capability.reason

    members = tuple(str(name) for name in (spec.get("members") or []))
    return ModelEntry(
        model_id=model_id,
        name=str(spec.get("name", model_id)),
        estimator_path=str(spec.get("estimator") or ""),
        mandatory=bool(spec.get("mandatory", False)),
        implemented=implemented,
        is_ensemble=bool(members),
        members=members,
        calibrated=bool(spec.get("calibrate", False))
        or bool(spec.get("calibrate_members")),
        n_search_dimensions=dimensions,
        search_constraints=constraints,
        unavailable_reason=reason,
    )


def entries() -> tuple[ModelEntry, ...]:
    return tuple(entry(model_id) for model_id in model_ids())


def available(model_id: str) -> bool:
    """Whether this model can actually be built on this machine right now."""
    return not entry(model_id).unavailable_reason


def build(model_id: str, **overrides: Any) -> Any:
    """An unfitted estimator for ``model_id``.

    A thin pass-through to the factories on purpose. The registry's job is to be
    the single *description* of the models; duplicating construction logic here
    would create a second definition that could drift from the first.
    """
    from src.models import estimators as est

    return est.build_estimator(model_id, **overrides)


def registry_frame() -> Any:
    """The registry as a table -- the model inventory the write-up quotes."""
    import pandas as pd

    return pd.DataFrame([item.as_row() for item in entries()])


# ---------------------------------------------------------------------------
# T51.2 / T51.5 -- persistence
# ---------------------------------------------------------------------------


@dataclass
class SavedModel:
    """A model on disk, and what was recorded about it when it was written."""

    model_id: str
    task: str
    path: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("feature_names") or ())

    @property
    def size_bytes(self) -> int:
        return int(self.manifest.get("model_bytes", -1))

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024**2 if self.size_bytes >= 0 else float("nan")


def models_root(root: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if root is not None:
        return Path(ensure_dir(root))
    return Path(ensure_dir(load_config("paths").require("models_saved")))


def model_dir(task: str, model_id: str, root: str | Path | None = None) -> Path:
    """``models_saved/{task}/{model_id}/`` -- T51.2's layout, exactly."""
    from src.utils.io import ensure_dir

    return Path(ensure_dir(models_root(root) / str(task) / str(model_id)))


def save_model(
    model: Any,
    *,
    model_id: str,
    task: str,
    feature_names: tuple[str, ...] | list[str],
    root: str | Path | None = None,
    fit_seconds: float = float("nan"),
    X_sample: Any = None,
    fold: str = "",
    extra: dict[str, Any] | None = None,
) -> SavedModel:
    """Write the fitted pipeline plus the manifest the complexity table needs.

    ``feature_names`` is not optional and is not derived from the model. An
    estimator knows how many columns it was fitted on and nothing about what they
    meant, so the names have to come from the caller that assembled the matrix --
    and once written, they are what :func:`load_model` checks a future matrix
    against.
    """
    import joblib

    from src.utils.run_manifest import package_versions

    names = tuple(str(name) for name in feature_names)
    if not names:
        raise RegistryError(
            "refusing to save " + model_id + " without its feature names; a "
            "reloaded model cannot detect a column reordering without them"
        )

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and int(expected) != len(names):
        raise RegistryError(
            model_id + " was fitted on " + str(int(expected)) + " columns but "
            + str(len(names)) + " feature names were supplied"
        )

    directory = model_dir(task, model_id, root)
    path = directory / MODEL_FILENAME
    joblib.dump(model, path)
    size = int(path.stat().st_size)

    inference = measure_inference(model, X_sample) if X_sample is not None else {}

    manifest: dict[str, Any] = {
        "model_id": model_id,
        "task": task,
        "fold": fold,
        "n_features": len(names),
        "feature_names": list(names),
        "model_bytes": size,
        "model_mb": round(size / 1024**2, 6),
        "fit_seconds": None if np.isnan(fit_seconds) else round(float(fit_seconds), 6),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "package_versions": package_versions(),
        "estimator_class": type(model).__name__,
    }
    manifest.update(inference)
    if extra:
        manifest.update(extra)

    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    log.info(
        "saved %s/%s: %.3f MB, fit %.2fs",
        task,
        model_id,
        size / 1024**2,
        fit_seconds if not np.isnan(fit_seconds) else float("nan"),
    )
    return SavedModel(model_id=model_id, task=task, path=path, manifest=manifest)


def load_model(
    task: str,
    model_id: str,
    root: str | Path | None = None,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Reload a saved model, refusing a feature list that does not match.

    Passing ``feature_names`` turns a silent catastrophe into an exception. The
    failure this prevents is not a crash: a matrix with the same 138 columns in a
    different order has the right shape, so the model predicts happily and is
    wrong on every row in a way no metric on that run can reveal.
    """
    import joblib

    directory = model_dir(task, model_id, root)
    path = directory / MODEL_FILENAME
    if not path.is_file():
        raise RegistryError("no saved model at " + str(path))

    manifest_path = directory / MANIFEST_FILENAME
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model = joblib.load(path)

    if feature_names is not None:
        stored = tuple(manifest.get("feature_names") or ())
        supplied = tuple(str(name) for name in feature_names)
        if not stored:
            raise RegistryError(
                "the manifest for " + task + "/" + model_id + " holds no feature "
                "names; the column alignment cannot be checked"
            )
        if stored != supplied:
            raise RegistryError(_misalignment_message(task, model_id, stored, supplied))

    return model, manifest


def _misalignment_message(
    task: str, model_id: str, stored: tuple[str, ...], supplied: tuple[str, ...]
) -> str:
    if set(stored) == set(supplied):
        first = next(
            index for index, pair in enumerate(zip(stored, supplied, strict=False))
            if pair[0] != pair[1]
        )
        return (
            task + "/" + model_id + " was fitted on the same features in a DIFFERENT "
            "ORDER -- position " + str(first) + " holds " + repr(supplied[first])
            + " but the model expects " + repr(stored[first])
            + ". Predictions would be silently wrong, not merely worse."
        )
    missing = [name for name in stored if name not in supplied]
    added = [name for name in supplied if name not in stored]
    return (
        task + "/" + model_id + " expects " + str(len(stored)) + " features, got "
        + str(len(supplied)) + "; missing " + str(missing[:5])
        + ", unexpected " + str(added[:5])
    )


def saved_models(root: str | Path | None = None) -> tuple[SavedModel, ...]:
    """Every model currently on disk, with its manifest."""
    base = models_root(root)
    found: list[SavedModel] = []
    for manifest_path in sorted(base.glob("*/*/" + MANIFEST_FILENAME)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        found.append(
            SavedModel(
                model_id=manifest.get("model_id", manifest_path.parent.name),
                task=manifest.get("task", manifest_path.parent.parent.name),
                path=manifest_path.parent / MODEL_FILENAME,
                manifest=manifest,
            )
        )
    return tuple(found)


# ---------------------------------------------------------------------------
# T51.3 / T51.4 -- size and timing
# ---------------------------------------------------------------------------


def measure_inference(
    model: Any, X: Any, *, n_records: int = 20, repeats: int = 3
) -> dict[str, Any]:
    """Single-record and batch inference cost, measured separately on purpose.

    T51.4 asks for **single-record** inference time, and that is not batch time
    divided by batch size. A 500-tree forest amortises its tree traversal across
    a batch; the deployment case -- one recording arrives at ``POST /predict`` --
    gets none of that. Both are reported so the complexity table can say which
    it means.

    The median of ``repeats`` passes, not the mean: a single scheduler hiccup on
    a 4-core laptop moves a mean far more than it moves a median.
    """
    features = np.asarray(X)
    if features.ndim != 2 or features.shape[0] == 0:
        raise RegistryError(
            "need a 2-D sample with at least one row, got shape " + str(features.shape)
        )

    sample = features[: min(n_records, features.shape[0])]

    single: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for row in sample:
            model.predict(row.reshape(1, -1))
        single.append((time.perf_counter() - started) / sample.shape[0])

    batch: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict(sample)
        batch.append((time.perf_counter() - started) / sample.shape[0])

    return {
        "inference_seconds_per_record": round(float(np.median(single)), 8),
        "inference_seconds_per_record_batched": round(float(np.median(batch)), 8),
        "inference_batch_speedup": round(
            float(np.median(single)) / max(float(np.median(batch)), 1e-12), 2
        ),
        "inference_n_records": int(sample.shape[0]),
        "inference_repeats": int(repeats),
    }
