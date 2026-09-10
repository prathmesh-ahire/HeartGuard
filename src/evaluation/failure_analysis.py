"""Which records the model gets wrong, and what they have in common (Phase 80).

Every headline metric is an average over records. This module goes back to the
records: every false positive and false negative of the final binary model, each
one carrying the four things that might explain it -- how long the recording is,
whether the quality flags called it noisy, which sub-collection it came from, and
how confident the model was when it got it wrong -- plus, for PhysioNet, the
diagnosis the record actually carries.

## A record is wrong five times, or once, and the difference is the finding

The binary map is repeated 5x5 grouped CV: every record is held out **once per
repeat**, so each record is scored five times by five separately-fitted models.
Counting "false positives" over the raw prediction rows would therefore report
five times the corpus and hide the only interesting distinction there is.

:func:`record_failures` collapses to one row per record with ``n_wrong`` out of
``n_evaluations``. A record wrong in 5 of 5 is a **consistent** failure -- five
independently fitted models agreed, so it is a property of the recording rather
than of one fold's fit. A record wrong in 1 of 5 sits on a decision boundary.
Those two need different sentences in a write-up and the same count would give
them the same one.

## Which model, and which run

The final binary model is **M1**, and the run whose numbers every binary table
reports is **EXP-A2**. So the in-domain failures are EXP-A2's M1 rows.

The deployed artifact in `models_saved/binary/final/` is refitted on all 3,240
labelled records and therefore has no out-of-fold predictions at all -- asking it
which records it gets wrong would be asking about its training set. Its only
honest error set is **EXP-D1**, where that exact model scored 3,163 CirCor
recordings it had never seen. Both are analysed, labelled, and never pooled: one
is 25-fold in-domain cross-validation, the other a single adult-to-paediatric
transfer whose population mismatch is already documented.

## The diagnosis cross-reference is a description, not a subgroup result

PhysioNet's `diagnosis_class` is present for 3,149 of 3,240 records and is very
unevenly filled -- CAD 389, AS 12. A per-diagnosis error rate over 12 records is
a description of those 12 records. `n_records` travels with every rate and
anything under :data:`MIN_GROUP` is flagged rather than dropped, because a
suppressed group reads as an absence of failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "MIN_GROUP",
    "ERROR_TYPES",
    "CATEGORIES",
    "FailureError",
    "FailureSource",
    "SOURCES",
    "MULTICLASS_RUNS",
    "record_context",
    "prediction_failures",
    "record_failures",
    "categorise",
    "confusion_pairs",
    "example_records",
]

log = get_logger("evaluation.failure")

#: Below this many records a per-group rate is reported with a flag rather than
#: quoted. Never dropped: a missing group reads as "no failures here".
MIN_GROUP = 30

ERROR_TYPES: tuple[str, ...] = ("TN", "FP", "FN", "TP")

#: The four groupings T80.2 asks for, in the order the tables print them.
CATEGORIES: tuple[str, ...] = (
    "duration_band",
    "noise_flag",
    "subset",
    "confidence_band",
)


class FailureError(RuntimeError):
    """A failure analysis cannot be run as asked."""


@dataclass(frozen=True)
class FailureSource:
    """One binary run whose errors are analysed, and how to read it."""

    name: str
    directory: str
    model_id: str
    kind: str
    note: str


SOURCES: tuple[FailureSource, ...] = (
    FailureSource(
        "EXP-A2 M1 (in-domain)",
        "outputs/06_binary_results/EXP-A2",
        "M1",
        "cross-validated",
        (
            "The final model's configuration, scored out-of-fold on 25 repeated "
            "grouped folds of PhysioNet. Every record is evaluated five times, "
            "once per repeat, by five separately fitted models."
        ),
    ),
    FailureSource(
        "EXP-D1 M1 (external)",
        "outputs/08_circor_external_validation/EXP-D1",
        "M1",
        "external holdout",
        (
            "The DEPLOYED model, refitted on all 3,240 PhysioNet records, scoring "
            "3,163 CirCor recordings once. Adult-to-paediatric transfer: the "
            "population mismatch dominates and these errors are not comparable "
            "with the in-domain ones."
        ),
    ),
)

#: Every multiclass confusion matrix T80.4 mines for confused pairs.
MULTICLASS_RUNS: tuple[tuple[str, str], ...] = (
    ("EXP-B1", "outputs/07_multiclass_results/EXP-B1"),
    ("EXP-B1-defaults", "outputs/07_multiclass_results/EXP-B1-defaults"),
    ("EXP-B2", "outputs/07_multiclass_results/EXP-B2"),
    ("EXP-B2-defaults", "outputs/07_multiclass_results/EXP-B2-defaults"),
    ("EXP-C1-three_class", "outputs/08_circor_external_validation/EXP-C1-three_class"),
    ("EXP-C1-two_class", "outputs/08_circor_external_validation/EXP-C1-two_class"),
    ("EXP-C2", "outputs/08_circor_external_validation/EXP-C2"),
)

#: Confidence bands for T80.2. Cut at 0.5 (a binary model cannot go below it),
#: then in decades, with the top band split because that is where the dangerous
#: errors live -- a wrong answer at 0.95 is a different problem from one at 0.55.
_CONFIDENCE_EDGES: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001)
_CONFIDENCE_LABELS: tuple[str, ...] = (
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-0.95",
    "0.95-1.00",
)


def record_context(*, root: Path | None = None) -> Any:
    """Per-record duration band, quality flags, subset and diagnosis, in one frame.

    Every column is *loaded* from the artifact that owns it -- duration bands
    from the same function T18.3 used, noise from PP-08's flags, diagnosis from
    the master metadata -- so a failure table cannot silently disagree with the
    tables those artifacts feed.
    """
    import pandas as pd

    from src.evaluation.duration import duration_bands
    from src.utils.config import load_config

    audit = Path(load_config("paths").require("outputs.dataset_audit"))
    preprocessing = Path(load_config("paths").require("outputs.preprocessing"))
    if root is not None:  # pragma: no cover - test seam
        audit = root / "01_dataset_audit"
        preprocessing = root / "02_preprocessing"

    master = pd.read_csv(audit / "metadata_master.csv", low_memory=False)
    context = master[
        [
            "record_uid",
            "dataset_source",
            "subset",
            "subject_id",
            "duration_sec",
            "recording_location",
            "diagnosis_class",
            "binary_label_name",
        ]
    ].copy()

    quality = pd.read_csv(preprocessing / "signal_quality_flags.csv")
    context = context.merge(
        quality[["record_uid", "is_noisy", "is_low_quality", "is_short", "quality_reasons"]],
        on="record_uid",
        how="left",
    )
    context["noise_flag"] = np.where(
        context["is_noisy"].fillna(False).astype(bool), "noisy (PP-08)", "clean (PP-08)"
    )

    bands = duration_bands().set_index("record_uid")["duration_band"]
    context["duration_band"] = context["record_uid"].map(bands)
    return context


def _confidence_band(values: Any) -> Any:
    import pandas as pd

    return pd.cut(
        np.asarray(values, dtype=float),
        bins=list(_CONFIDENCE_EDGES),
        labels=list(_CONFIDENCE_LABELS),
        right=False,
        include_lowest=True,
    ).astype(str)


def prediction_failures(source: FailureSource, *, root: Path | None = None) -> Any:
    """Every stored prediction of one model, labelled TP/TN/FP/FN and contextualised.

    Returns *all* predictions, not only the wrong ones. An error count without
    the denominator beside it cannot become a rate, and a table of failures with
    no successes in it invites exactly that mistake.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    base = root or PROJECT_ROOT
    path = base / source.directory / "predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(str(path) + "; the run has not been executed")

    frame = pd.read_parquet(path)
    frame = frame[frame["model_id"].astype(str) == source.model_id].copy()
    if not len(frame):
        raise FailureError(source.name + ": " + source.model_id + " is not in " + str(path))
    if "proba_1" not in frame.columns:
        raise FailureError(source.name + ": not a binary run (no proba_1 column)")

    truth = frame["y_true"].to_numpy(dtype=int)
    predicted = frame["y_pred"].to_numpy(dtype=int)
    frame["error_type"] = np.select(
        [
            (truth == 0) & (predicted == 0),
            (truth == 0) & (predicted == 1),
            (truth == 1) & (predicted == 0),
            (truth == 1) & (predicted == 1),
        ],
        list(ERROR_TYPES),
        default="?",
    )
    frame["is_error"] = truth != predicted
    # Confidence in the class that was PREDICTED, which is what a user of the
    # dashboard sees beside the answer -- not the probability of the positive
    # class, which for a normal prediction is the confidence in being wrong.
    frame["confidence"] = np.where(
        predicted == 1, frame["proba_1"].to_numpy(dtype=float), 1.0 - frame["proba_1"]
    )
    frame["confidence_band"] = _confidence_band(frame["confidence"])

    context = record_context(root=root)
    merged = frame.merge(context, on="record_uid", how="left")
    if merged["duration_band"].isna().any():
        missing = int(merged["duration_band"].isna().sum())
        raise FailureError(
            source.name + ": " + str(missing) + " prediction row(s) have no record context; "
            "the predictions and the audit disagree about which records exist"
        )
    merged.insert(0, "source", source.name)
    merged.insert(1, "source_kind", source.kind)
    return merged


def record_failures(failures: Any) -> Any:
    """One row per record: how often it was wrong, and how sure the model was.

    ``consistently_wrong`` means every evaluation of that record was an error.
    On the repeated map that is five independently fitted models agreeing, which
    makes it a statement about the recording; a record wrong once in five is a
    statement about a decision boundary.
    """
    import pandas as pd

    keys = ["source", "record_uid"]
    carried = [
        "dataset_source",
        "subset",
        "subject_id",
        "duration_sec",
        "duration_band",
        "noise_flag",
        "is_low_quality",
        "quality_reasons",
        "diagnosis_class",
        "binary_label_name",
        "recording_location",
    ]
    rows: list[dict[str, Any]] = []
    for values, block in failures.groupby(keys, sort=True):
        row = dict(zip(keys, values, strict=True))
        first = block.iloc[0]
        for column in carried:
            if column in block.columns:
                row[column] = first[column]
        wrong = block[block["is_error"]]
        row["y_true"] = int(first["y_true"])
        row["n_evaluations"] = len(block)
        row["n_wrong"] = len(wrong)
        row["error_rate"] = float(len(wrong)) / float(len(block))
        row["consistently_wrong"] = bool(len(wrong) == len(block))
        row["error_type"] = (
            str(wrong["error_type"].iloc[0]) if len(wrong) else "correct"
        )
        row["mean_confidence"] = float(block["confidence"].mean())
        row["mean_confidence_when_wrong"] = (
            float(wrong["confidence"].mean()) if len(wrong) else float("nan")
        )
        row["max_confidence_when_wrong"] = (
            float(wrong["confidence"].max()) if len(wrong) else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def categorise(failures: Any, by: str) -> Any:
    """Error rate within each level of one grouping, with its denominator.

    Computed over prediction rows rather than over records, deliberately: the
    question "what fraction of predictions in this band were wrong" is what a
    band-wise error rate means, and it is the quantity that reconciles with the
    run's own metrics. The record-level view is :func:`record_failures`.
    """
    import pandas as pd

    if by not in failures.columns:
        raise FailureError("no column " + repr(by) + " to categorise by")

    rows: list[dict[str, Any]] = []
    for values, block in failures.groupby(["source", by], sort=True):
        source, level = values
        counts = block["error_type"].value_counts()
        n = len(block)
        n_wrong = int(block["is_error"].sum())
        rows.append(
            {
                "source": str(source),
                "category": by,
                "level": str(level),
                "n_predictions": n,
                "n_records": int(block["record_uid"].nunique()),
                "n_positive": int((block["y_true"] == 1).sum()),
                "n_negative": int((block["y_true"] == 0).sum()),
                **{"n_" + kind.lower(): int(counts.get(kind, 0)) for kind in ERROR_TYPES},
                "n_errors": n_wrong,
                "error_rate": float(n_wrong) / n if n else float("nan"),
                "false_negative_rate": (
                    float(counts.get("FN", 0)) / float((block["y_true"] == 1).sum())
                    if (block["y_true"] == 1).any()
                    else float("nan")
                ),
                "false_positive_rate": (
                    float(counts.get("FP", 0)) / float((block["y_true"] == 0).sum())
                    if (block["y_true"] == 0).any()
                    else float("nan")
                ),
                "mean_confidence_when_wrong": (
                    float(block.loc[block["is_error"], "confidence"].mean())
                    if n_wrong
                    else float("nan")
                ),
                "below_reporting_floor": bool(block["record_uid"].nunique() < MIN_GROUP),
            }
        )
    return pd.DataFrame(rows)


def confusion_pairs(
    runs: tuple[tuple[str, str], ...] = MULTICLASS_RUNS, *, root: Path | None = None
) -> Any:
    """T80.4 -- every off-diagonal cell of every stored confusion matrix, ranked.

    ``share_of_true_class`` is the number that matters: 40 confusions out of 320
    records of a class is a different statement from 40 out of 46, and the raw
    count alone cannot tell them apart. ``total`` in the stored payload is the
    element-wise sum over folds, so on a repeated map its support is the corpus
    times the number of repeats -- ``n_repeats_counted`` records that rather
    than quietly dividing it out.
    """
    import json

    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    base = root or PROJECT_ROOT
    rows: list[dict[str, Any]] = []
    for run, directory in runs:
        path = base / directory / "confusion_matrices.json"
        if not path.is_file():
            log.warning("%s: no confusion_matrices.json, skipping", run)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = list(payload["class_names"])
        for model_id, entry in payload["models"].items():
            matrix = np.asarray(entry["total"], dtype=float)
            n_folds = int(entry.get("n_folds", 0))
            supports = matrix.sum(axis=1)
            for true_index, true_name in enumerate(names):
                for pred_index, pred_name in enumerate(names):
                    if true_index == pred_index:
                        continue
                    count = float(matrix[true_index, pred_index])
                    if count <= 0:
                        continue
                    support = float(supports[true_index])
                    rows.append(
                        {
                            "run": run,
                            "task": str(payload.get("task", "")),
                            "model_id": str(model_id),
                            "true_class": true_name,
                            "predicted_class": pred_name,
                            "pair": true_name + " -> " + pred_name,
                            "n_confused": count,
                            "n_true_class_summed": support,
                            "share_of_true_class": count / support if support else float("nan"),
                            "n_folds_summed": n_folds,
                        }
                    )

    frame = pd.DataFrame(rows)
    if not len(frame):
        return frame
    frame["rank_in_run_model"] = (
        frame.groupby(["run", "model_id"])["n_confused"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return frame.sort_values(
        ["run", "model_id", "n_confused"], ascending=[True, True, False]
    ).reset_index(drop=True)


def example_records(records: Any, *, per_group: int = 5) -> Any:
    """The concrete records T80.6 has to name, chosen by a stated rule.

    The rule: among the records that were wrong on **every** evaluation, the
    ones the model was most confident about. That is the worst case a screening
    prototype has -- confidently wrong, repeatably -- and picking by confidence
    rather than by eye means the list regenerates identically.
    """
    import pandas as pd

    wrong = records[records["n_wrong"] > 0]
    picks: list[pd.DataFrame] = []
    for values, block in wrong.groupby(["source", "error_type"], sort=True):
        consistent = block[block["consistently_wrong"]]
        pool = consistent if len(consistent) >= per_group else block
        chosen = pool.nlargest(per_group, "mean_confidence_when_wrong").copy()
        chosen["selection_rule"] = (
            "highest mean confidence among consistently-wrong records"
            if len(consistent) >= per_group
            else "highest mean confidence among all wrong records "
            "(fewer than " + str(per_group) + " were consistently wrong)"
        )
        chosen["group"] = str(values[0]) + " / " + str(values[1])
        picks.append(chosen)
    if not picks:
        return wrong.head(0)
    return pd.concat(picks, ignore_index=True)
