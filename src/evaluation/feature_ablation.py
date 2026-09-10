"""EXP-F1 -- the feature-family ablation, A1 through A8 (Phase 74).

The question is which of the six feature families carry the signal. The only
way to answer it credibly is to change **nothing else**: same records, same
DA-07 fold map, same seed, same models, same pipeline. So every configuration
here is a *column subset of one already-extracted matrix* rather than a
re-extraction -- there is exactly one FE-03 matrix in this project, and eight
runs over eight subsets of it cannot disagree about what a feature value means.

The eight configurations are declared in ``configs/experiments.yaml``, not here:

===  =========================================  ===========
A1   time                                        24
A2   frequency                                   22
A3   mfcc                                        39
A4   dwt                                         24
A5   time + frequency                            46
A6   mfcc + dwt                                  63
A7   all six families                           138
A8   the SO-04 selected subset                   20
===  =========================================  ===========

**A7 is EXP-A1's exact configuration**, which makes it a control rather than a
ninth data point: :func:`assert_a7_reproduces_exp_a1` checks the re-run against
the committed EXP-A1 numbers. If A7 drifts, the ablation is measuring the
runner and not the features, and no row in T17 means what it says.

**Chroma and envelope get no solo arm.** T74.1-T74.5 name eight configurations
and those six; chroma (24) and envelope (5) appear only inside A7. That is the
task list's shape, not an oversight here, and T17 says so in its notes rather
than leaving a reader to wonder why four of six families were singled out.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "EXP_ID",
    "ABLATION_MODELS",
    "SELECTED_SUBSET_ID",
    "configuration_ids",
    "configuration_spec",
    "configuration_columns",
    "selected_subset_names",
    "subset_data",
    "arm_experiment",
    "run_configuration",
    "run_feature_ablation",
    "assert_a7_reproduces_exp_a1",
    "collect_per_fold",
    "summarise",
]

log = get_logger("evaluation.feature_ablation")

EXP_ID = "EXP-F1"

#: The five individual models. **The ensembles M6/M7 are deliberately absent.**
#: EXP-F1 asks what the feature families are worth with the model held fixed;
#: what ensembling is worth is EXP-F2's entire subject (Phase 75), measured on
#: the full 138 where it belongs. Including M6 here would also have cost 5,351 s
#: per configuration -- 12 hours over the eight arms -- to answer a question
#: another experiment already answers, on folds it already shares.
ABLATION_MODELS: tuple[str, ...] = ("M1", "M3", "M4", "M5", "M8")

#: A8's feature list comes from FE-12, the shipped subset, never re-selected
#: here: re-running selection inside the ablation would fit it to all 25 folds
#: at once and make A8 the one arm that saw its own test rows.
SELECTED_SUBSET_ID = "SO-04"


# ---------------------------------------------------------------------------
# the configurations
# ---------------------------------------------------------------------------


def _spec() -> dict[str, Any]:
    from src.utils.config import load_config

    table = load_config("experiments").require("experiments")
    if EXP_ID not in table:
        raise KeyError("configs/experiments.yaml declares no " + EXP_ID)
    return dict(table[EXP_ID])


def configuration_ids() -> tuple[str, ...]:
    return tuple(_spec().get("configurations", {}))


def configuration_spec(config_id: str) -> dict[str, Any]:
    configurations = _spec().get("configurations", {})
    if config_id not in configurations:
        raise KeyError(
            EXP_ID + " declares no configuration " + repr(config_id)
            + "; known: " + ", ".join(sorted(configurations))
        )
    return dict(configurations[config_id])


def selected_subset_names() -> tuple[str, ...]:
    """FE-12's shipped subset, read from disk in its stored rank order."""
    import pandas as pd

    from src.utils.config import load_config

    path = Path(load_config("paths").require("outputs.features")) / "selected_feature_subset.csv"
    if not path.is_file():
        raise FileNotFoundError(
            str(path) + " is missing; A8 cannot be run without the FE-12 subset"
        )
    table = pd.read_csv(path).sort_values("rank", kind="mergesort")
    return tuple(str(name) for name in table["feature"])


def configuration_columns(config_id: str) -> tuple[str, ...]:
    """The feature names one configuration keeps, in registry order.

    Registry order, not the order the families were named: the column ordering
    of a fitted matrix is part of what makes a run reproducible, and two arms
    that agree on *which* features they use must also agree on their order.
    """
    from src.feature_extraction.registry import feature_names

    spec = configuration_spec(config_id)
    families = spec.get("families")

    if families == "selected_subset":
        wanted = set(selected_subset_names())
        selected = tuple(name for name in feature_names() if name in wanted)
        missing = wanted - set(selected)
        if missing:
            raise ValueError(
                config_id + ": " + str(len(missing))
                + " selected feature(s) are not in the registry: "
                + ", ".join(sorted(missing)[:5])
            )
        return selected

    if not isinstance(families, list) or not families:
        raise ValueError(config_id + " declares no families and no selected_subset")

    gathered: list[str] = []
    for family in families:
        gathered.extend(feature_names(str(family)))
    ordered = tuple(name for name in feature_names() if name in set(gathered))

    declared = spec.get("n_features")
    if isinstance(declared, int) and len(ordered) != declared:
        raise ValueError(
            config_id + " declares n_features=" + str(declared)
            + " but its families resolve to " + str(len(ordered))
        )
    return ordered


def subset_data(data: Any, columns: tuple[str, ...]) -> Any:
    """``TaskData`` restricted to ``columns``, everything else untouched."""
    import dataclasses

    import numpy as np

    index = {name: position for position, name in enumerate(data.feature_names)}
    missing = [name for name in columns if name not in index]
    if missing:
        raise ValueError(
            str(len(missing)) + " configuration feature(s) absent from the matrix, e.g. "
            + ", ".join(missing[:5])
        )
    take = np.array([index[name] for name in columns], dtype=int)
    return dataclasses.replace(
        data, X=np.asarray(data.X)[:, take], feature_names=tuple(columns)
    )


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------


def arm_experiment(config_id: str, models: tuple[str, ...] = ABLATION_MODELS) -> Any:
    """EXP-F1 as a runnable ``Experiment`` for one configuration.

    Built by replacing fields on the loaded EXP-A1 declaration rather than by
    writing a fresh one, so the task, the CV scheme and the label space are the
    ones EXP-A1 used by construction -- ``identical_folds_as: EXP-A1`` is the
    config's requirement and this is how it is honoured rather than asserted.
    """
    import dataclasses

    from src.evaluation.experiment import Experiment

    spec = _spec()
    anchor = str(spec.get("identical_folds_as", "EXP-A1"))
    base = Experiment.load(anchor)
    if base.cv != str(spec["cv"]) or base.task != str(spec["task"]):
        raise ValueError(
            EXP_ID + " declares task/cv " + str(spec["task"]) + "/" + str(spec["cv"])
            + " but its fold anchor " + anchor + " uses " + base.task + "/" + base.cv
        )
    unknown = [m for m in models if m not in base.models]
    if unknown:
        raise ValueError(anchor + " does not declare model(s) " + ", ".join(unknown))

    return dataclasses.replace(
        base,
        exp_id=EXP_ID,
        title=str(spec.get("title", EXP_ID)) + " -- " + config_id,
        phase=int(spec.get("phase", 74)),
        models=tuple(models),
        output_section=str(spec.get("output_section", "09_ablation")),
        variant=config_id,
        emits=tuple(str(e) for e in spec.get("emits", ())),
        notes=(
            config_id + ": " + _label(config_id) + ". Column subset of the single "
            "FE-03 matrix; folds, seed, models and pipeline identical to " + anchor + "."
        ),
    )


def _label(config_id: str) -> str:
    spec = configuration_spec(config_id)
    families = spec.get("families")
    if families == "selected_subset":
        return "the " + SELECTED_SUBSET_ID + " selected subset"
    return " + ".join(str(f) for f in (families or ()))


def run_configuration(
    config_id: str,
    *,
    data: Any = None,
    models: tuple[str, ...] = ABLATION_MODELS,
    out_dir: str | Path | None = None,
    resume: bool = True,
) -> Any:
    """Run one configuration over the full fold map and write its contract."""
    from src.evaluation.experiment import run_experiment, write_outputs
    from src.models import smoke as sm

    experiment = arm_experiment(config_id, models)
    columns = configuration_columns(config_id)
    loaded = sm.load_task_data(experiment.task) if data is None else data
    narrowed = subset_data(loaded, columns)

    log.info(
        "%s %s: %d feature(s), %d model(s) -- %s",
        EXP_ID,
        config_id,
        len(columns),
        len(models),
        _label(config_id),
    )
    started = time.perf_counter()
    result = run_experiment(experiment, data=narrowed, out_dir=out_dir, resume=resume)
    write_outputs(result, out_dir=out_dir)
    log.info("%s %s done in %.1f min", EXP_ID, config_id, (time.perf_counter() - started) / 60)
    return result


def run_feature_ablation(
    *,
    configurations: tuple[str, ...] | None = None,
    models: tuple[str, ...] = ABLATION_MODELS,
    out_dir: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Every configuration, over one loaded matrix.

    The matrix is loaded **once** and subset per arm. Reloading it per arm would
    be eight chances for a different file to be picked up half way through a
    three-hour run.
    """
    from src.models import smoke as sm

    chosen = configurations or configuration_ids()
    spec = _spec()
    data = sm.load_task_data(str(spec["task"]))
    results: dict[str, Any] = {}
    for config_id in chosen:
        results[config_id] = run_configuration(
            config_id, data=data, models=models, out_dir=out_dir, resume=resume
        )
    return results


# ---------------------------------------------------------------------------
# the A7 control
# ---------------------------------------------------------------------------


def assert_a7_reproduces_exp_a1(*, out_dir: str | Path | None = None, atol: float = 1e-9) -> Any:
    """A7 is EXP-A1's configuration, so it must reproduce EXP-A1's numbers.

    This is the seed- and fold-discipline proof for the whole ablation: if the
    all-138 arm does not land on the committed EXP-A1 rows to nine decimals,
    something in the runner path differs and every other arm's delta is
    contaminated by that difference rather than by its feature subset.
    """
    import numpy as np
    import pandas as pd

    from src.evaluation.experiment import Experiment

    anchor = pd.read_csv(
        Experiment.load("EXP-A1").output_dir() / "per_fold_metrics.csv"
    )
    ours = pd.read_csv(arm_experiment("A7").output_dir(out_dir) / "per_fold_metrics.csv")

    keys = ["model_id", "repeat", "fold"]
    metrics = [
        column
        for column in ("sensitivity", "specificity", "f1", "balanced_accuracy", "roc_auc",
                       "accuracy")
        if column in anchor.columns and column in ours.columns
    ]
    merged = ours[keys + metrics].merge(
        anchor[keys + metrics], on=keys, how="inner", suffixes=("_f1", "_a1")
    )
    if merged.empty:
        raise AssertionError("A7 and EXP-A1 share no (model, repeat, fold) rows")

    rows = []
    for metric in metrics:
        difference = np.abs(
            merged[metric + "_f1"].to_numpy(dtype=float)
            - merged[metric + "_a1"].to_numpy(dtype=float)
        )
        rows.append(
            {
                "metric": metric,
                "n_units": len(merged),
                "max_abs_difference": float(np.nanmax(difference)),
                "matches": bool(np.nanmax(difference) <= atol),
            }
        )
    report = pd.DataFrame(rows)
    failed = report[~report["matches"]]
    log.info(
        "A7 vs EXP-A1: %d unit(s), max difference %.3g (%s)",
        int(report["n_units"].iloc[0]),
        float(report["max_abs_difference"].max()),
        "match" if failed.empty else "MISMATCH",
    )
    if not failed.empty:
        raise AssertionError(
            "A7 does not reproduce EXP-A1 on " + ", ".join(failed["metric"])
            + "; the ablation runner is not EXP-A1's path"
        )
    return report


# ---------------------------------------------------------------------------
# collecting the arms
# ---------------------------------------------------------------------------


def collect_per_fold(
    configurations: tuple[str, ...] | None = None, *, out_dir: str | Path | None = None
) -> Any:
    """Every arm's per-fold rows in one frame, tagged by configuration."""
    import pandas as pd

    frames = []
    for config_id in configurations or configuration_ids():
        path = arm_experiment(config_id).output_dir(out_dir) / "per_fold_metrics.csv"
        if not path.is_file():
            log.warning("%s: %s missing, skipping", config_id, path)
            continue
        block = pd.read_csv(path)
        width = len(configuration_columns(config_id))

        # The runner writes its own `n_features` -- the width of the matrix it
        # was actually handed. Cross-check it against the configuration's
        # declared width rather than overwriting it: a disagreement means the
        # arm ran on a different feature set from the one T17 will label it
        # with, which is the one error this table could not survive.
        if "n_features" in block.columns:
            recorded = set(block["n_features"].dropna().astype(int))
            if recorded and recorded != {width}:
                raise ValueError(
                    config_id + " ran on " + str(sorted(recorded))
                    + " feature(s) but its configuration declares " + str(width)
                )
            block = block.drop(columns=["n_features"])

        block.insert(0, "config_id", config_id)
        block.insert(1, "families", _label(config_id))
        block.insert(2, "n_features", width)
        frames.append(block)
    if not frames:
        raise FileNotFoundError("no EXP-F1 configuration has been run yet")
    return pd.concat(frames, ignore_index=True)


#: T17's headline metrics. Sensitivity first: rule 6 ranks by it.
HEADLINE: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "accuracy",
)


def summarise(per_fold: Any) -> Any:
    """Mean and SD per (configuration, model), plus the delta against A7.

    A7 is the reference because it is the full proposed representation -- every
    other arm's number answers "what is lost by dropping the rest", which is
    the question a family ablation exists to answer.
    """
    import numpy as np
    import pandas as pd

    rows = []
    for (config_id, model_id), block in per_fold.groupby(["config_id", "model_id"], sort=False):
        row: dict[str, Any] = {
            "config_id": config_id,
            "families": str(block["families"].iloc[0]),
            "n_features": int(block["n_features"].iloc[0]),
            "model_id": model_id,
            "n_folds": len(block),
        }
        for metric in HEADLINE:
            if metric not in block.columns:
                continue
            values = block[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[metric] = float(np.mean(finite)) if finite.size else float("nan")
            row[metric + "_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    reference = table[table["config_id"] == "A7"].set_index("model_id")
    for metric in HEADLINE:
        if metric not in table.columns:
            continue
        table[metric + "_delta_vs_A7"] = [
            float(row[metric]) - float(reference.loc[row["model_id"], metric])
            if row["model_id"] in reference.index
            else float("nan")
            for row in table.to_dict("records")
        ]

    order = {config_id: index for index, config_id in enumerate(configuration_ids())}
    table = table.sort_values(
        ["config_id", "model_id"], key=lambda s: s.map(order) if s.name == "config_id" else s
    )
    return table.reset_index(drop=True)
