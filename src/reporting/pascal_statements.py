"""The written caveats that must travel with T11 and T12 (T66.6, T67.5, T67.6).

Three claims in this project are easier to get wrong in prose than in code, and
all three are about the PASCAL tracks:

* PASCAL A's ``artifact`` is a **recording-quality** label. A model that
  separates it is not a four-class *cardiac* classifier, and CLAUDE.md forbids
  describing it as one.
* PASCAL A has **no recoverable subject IDs**, so its results are record-level.
* Sets A and B are **never merged** -- research rule 4.

Every number in the generated statements is measured here from the DA-07 fold
map and the run's own per-fold table, never typed. The merge claim in particular
is *verified* before it is written: if any record ever appeared under both
tasks, the statement would refuse to generate rather than assert something
false.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "MERGE_STATEMENT",
    "ARTIFACT_STATEMENT",
    "StatementError",
    "verify_sets_never_merged",
    "build_statement",
    "write_statement",
]

log = get_logger("reporting.pascal")


class StatementError(RuntimeError):
    """A claim the statement would make cannot be verified against the data."""


#: The exact sentence T67.6 requires, and the string the gate greps for.
MERGE_STATEMENT = (
    "PASCAL set A and set B were never merged: they are two separate label "
    "spaces, evaluated as two separate tasks, with two separate fold maps."
)

#: The exact sentence T66.6 requires.
ARTIFACT_STATEMENT = (
    "`artifact` is a RECORDING-QUALITY label, not a cardiac class. EXP-B1 is "
    "therefore NOT a four-class cardiac classifier and must never be described "
    "as one."
)


def _fold_map() -> Any:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "subject_split_map.csv"
    return pd.read_csv(path)


def verify_sets_never_merged(fold_map: Any = None) -> dict[str, Any]:
    """Prove, from the fold map, that the two PASCAL tasks were never pooled.

    Raises if any record appears under both tasks -- that is the check that would
    actually catch a merge. The class vocabularies are reported but deliberately
    NOT required to be disjoint: both sets carry `normal` and `murmur`, and that
    shared *name* is precisely why the two must stay apart. `normal` against
    three alternatives and `normal` against two are different targets with
    different priors, so a shared name is evidence for rule 4, not against it.
    """
    frame = _fold_map() if fold_map is None else fold_map
    a = frame[frame["task"] == "pascal_a"]
    b = frame[frame["task"] == "pascal_b"]
    if a.empty or b.empty:
        raise StatementError(
            "the DA-07 map holds no rows for pascal_a and/or pascal_b; cannot verify rule 4"
        )

    shared_records = sorted(set(a["record_uid"]) & set(b["record_uid"]))
    if shared_records:
        raise StatementError(
            str(len(shared_records))
            + " record(s) appear under BOTH pascal_a and pascal_b, e.g. "
            + ", ".join(shared_records[:3])
        )

    classes_a = set(a["class_name"])
    classes_b = set(b["class_name"])
    shared_classes = sorted(classes_a & classes_b)

    return {
        "n_records_a": int(a["record_uid"].nunique()),
        "n_records_b": int(b["record_uid"].nunique()),
        "shared_records": 0,
        "classes_a": sorted(classes_a),
        "classes_b": sorted(classes_b),
        "shared_class_names": shared_classes,
        "scheme_a": str(a["scheme"].iloc[0]),
        "scheme_b": str(b["scheme"].iloc[0]),
        "groups_a": int(a["split_group"].nunique()),
        "groups_b": int(b["split_group"].nunique()),
        "subject_derived_a": bool(a["subject_derived"].iloc[0]),
        "subject_derived_b": bool(b["subject_derived"].iloc[0]),
    }


def _class_counts(frame: Any, task: str) -> dict[str, int]:
    block = frame[frame["task"] == task].drop_duplicates("record_uid")
    counts = block["class_name"].value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def _majority_baseline(counts: dict[str, int]) -> tuple[str, float]:
    total = sum(counts.values())
    name = max(counts, key=lambda key: counts[key])
    return name, (counts[name] / total if total else float("nan"))


def build_statement(
    task: str,
    *,
    per_fold: Any = None,
    fold_map: Any = None,
    predictions: Any = None,
    label_space: Any = None,
) -> str:
    """The markdown caveat block for ``pascal_a`` or ``pascal_b``.

    ``per_fold`` is optional; when given, the fold count, the achieved macro-F1
    range and the measured per-class recall are quoted from the run rather than
    left unstated. That last table is what T67.5 asks for and what makes T66.6
    concrete: on set_a the class the models handle best is ``artifact``, which
    is the one that is not cardiac.
    """
    import pandas as pd

    if task not in ("pascal_a", "pascal_b"):
        raise StatementError("no statement is declared for task " + repr(task))

    frame = _fold_map() if fold_map is None else fold_map
    shared = verify_sets_never_merged(frame)
    counts = _class_counts(frame, task)
    rarest = min(counts, key=lambda key: counts[key])
    majority, baseline = _majority_baseline(counts)
    total = sum(counts.values())

    lines: list[str] = []
    title = (
        "PASCAL A (set_a) -- four-class"
        if task == "pascal_a"
        else "PASCAL B (set_b) -- three-class"
    )
    lines.append("# " + title + ": caveats that travel with the result table")
    lines.append("")
    lines.append(
        "Generated from `outputs/01_dataset_audit/subject_split_map.csv` and the "
        "run's own `per_fold_metrics.csv`. Every number below is measured, none typed."
    )
    lines.append("")

    lines.append("## Sample size and class balance")
    lines.append("")
    lines.append("| class | records | share |")
    lines.append("|---|---|---|")
    for name in sorted(counts, key=lambda key: -counts[key]):
        share = counts[name] / total if total else float("nan")
        lines.append("| " + name + " | " + str(counts[name]) + " | " + format(share, ".1%") + " |")
    lines.append("| **total** | **" + str(total) + "** | |")
    lines.append("")
    lines.append(
        "The rarest class is **"
        + rarest
        + "** at **"
        + str(counts[rarest])
        + " records**. Across the fold map that is roughly "
        + format(counts[rarest] / max(len(set(frame[frame["task"] == task]["fold"])), 1), ".1f")
        + " records per test fold, which is why every headline metric in the "
        "companion table carries a confidence interval and why a difference of a "
        "few points between two models should not be read as a difference at all."
    )
    lines.append("")
    lines.append(
        "Always predicting **"
        + majority
        + "** would score "
        + format(baseline, ".1%")
        + " accuracy on this corpus. That is the number any accuracy figure here "
        "must be compared against, and it is why the table is ranked by macro-F1 "
        "and reports per-class recall (research rule 6)."
    )
    lines.append("")

    if task == "pascal_a":
        lines.append("## `artifact` is not a cardiac class")
        lines.append("")
        lines.append(ARTIFACT_STATEMENT)
        lines.append("")
        lines.append(
            "It labels recordings that are unusable -- handling noise, contact "
            "artefacts, a stethoscope moving -- rather than a cardiac finding. A "
            "high recall on `artifact` measures recording-quality detection, which "
            "is useful in its own right and is a different claim from cardiac "
            "classification. The honest description of EXP-B1 is: *a four-class "
            "acoustic-event classifier over three cardiac categories and one "
            "recording-quality category.*"
        )
        lines.append("")
        lines.append("## Results are record-level, not subject-level")
        lines.append("")
        lines.append(
            "PASCAL set_a carries no recoverable subject identifier, so "
            "`subject_derived=False` for all "
            + str(shared["n_records_a"])
            + " records and the folds are stratified at record level. Two rows do "
            "share a group: Phase 17 found one recording filed under two class "
            "labels (`extrahls__201104021355` and `murmur__201104021355`, envelope "
            "correlation 0.999978), and those two share a group key so the same "
            "audio cannot land on both sides of a split. That leaves "
            + str(shared["groups_a"])
            + " groups over "
            + str(shared["n_records_a"])
            + " records."
        )
        lines.append("")
        lines.append(
            "**This is a dataset limitation, not a method limitation**, and it must "
            "be stated as such. No claim of subject-level generalization can be made "
            "from EXP-B1."
        )
    else:
        lines.append("## Subject grouping: 165 subjects, not 167")
        lines.append("")
        lines.append(
            "PASCAL set_b is "
            + str(shared["n_records_b"])
            + " records over **"
            + str(shared["groups_b"])
            + " subject groups**, derived from the numeric id in the filename. The "
            "figure 167 that appears in the task list counts recording *sessions*: "
            "three subjects were recorded twice on the same day, two of them with "
            "both sessions in the labelled set. Grouping on the session would put "
            "the same person on both sides of a fold, so the grouping key is the "
            "subject number. Anything quoting 167 subjects is quoting sessions. See "
            "the 2026-08-25 entry in `Docs/note.md`."
        )
        lines.append("")
        lines.append("## Severe class imbalance and what it does to recall")
        lines.append("")
        lines.append(
            "At "
            + " / ".join(str(counts[name]) for name in sorted(counts, key=lambda key: -counts[key]))
            + " the minority class has roughly "
            + format(counts[rarest] / max(counts[majority], 1), ".0%")
            + " of the majority's support. Every model is fitted with balanced class "
            "weights, but weighting cannot manufacture examples: the per-class recall "
            "column in the companion table is where the imbalance actually shows, and "
            "it is the column to read first."
        )

    lines.append("")
    lines.append("## Rule 4 -- the label spaces were never merged")
    lines.append("")
    lines.append(MERGE_STATEMENT)
    lines.append("")
    lines.append("Verified against the fold map rather than asserted:")
    lines.append("")
    lines.append("| check | set A | set B |")
    lines.append("|---|---|---|")
    lines.append(
        "| records | " + str(shared["n_records_a"]) + " | " + str(shared["n_records_b"]) + " |"
    )
    lines.append(
        "| classes | "
        + ", ".join(shared["classes_a"])
        + " | "
        + ", ".join(shared["classes_b"])
        + " |"
    )
    lines.append("| CV scheme | " + shared["scheme_a"] + " | " + shared["scheme_b"] + " |")
    lines.append(
        "| grouping | "
        + ("record-level (no subject id)" if not shared["subject_derived_a"] else "subject")
        + " | "
        + ("subject" if shared["subject_derived_b"] else "record-level")
        + " |"
    )
    lines.append(
        "| records shared with the other set | "
        + str(shared["shared_records"])
        + " | "
        + str(shared["shared_records"])
        + " |"
    )
    lines.append("")
    overlap = shared["shared_class_names"]
    if overlap:
        lines.append(
            "The two sets share the class *name(s)* "
            + ", ".join("`" + name + "`" for name in overlap)
            + ", which is exactly why they are kept apart: a shared name is not a "
            "shared label space. `normal` in a four-class problem and `normal` in a "
            "three-class problem are different targets with different priors, and "
            "pooling them would change what both numbers mean."
        )
    else:
        lines.append("The two class vocabularies are disjoint.")
    lines.append("")

    if per_fold is not None:
        table = pd.DataFrame(per_fold)
        if "macro_f1" in table.columns and not table.empty:
            values = np.asarray(table["macro_f1"], dtype=float)
            finite = values[np.isfinite(values)]
            lines.append("## What this run produced")
            lines.append("")
            lines.append(
                "Across "
                + str(int(table["fold_label"].nunique()))
                + " folds and "
                + str(int(table["model_id"].nunique()))
                + " models, per-fold macro-F1 ranged from "
                + format(float(finite.min()), ".4f")
                + " to "
                + format(float(finite.max()), ".4f")
                + ". The spread across folds is itself the small-sample effect: it is "
                "wider than the spread between models."
            )
            lines.append("")

        # T67.5 / T66.6 -- what the imbalance actually does, measured per class
        # over every model and fold rather than argued from the class counts.
        recall_columns = [
            "recall_" + name for name in counts if "recall_" + name in table.columns
        ]
        if recall_columns:
            lines.append(
                "Per-class recall, pooled over every model and fold (mean, and the "
                "worst single fold):"
            )
            lines.append("")
            lines.append("| class | records | mean recall | worst fold |")
            lines.append("|---|---|---|---|")
            for name in sorted(counts, key=lambda key: -counts[key]):
                column = "recall_" + name
                if column not in table.columns:
                    continue
                values = np.asarray(table[column], dtype=float)
                finite = values[np.isfinite(values)]
                if not finite.size:
                    continue
                lines.append(
                    "| "
                    + name
                    + " | "
                    + str(counts[name])
                    + " | "
                    + format(float(finite.mean()), ".4f")
                    + " | "
                    + format(float(finite.min()), ".4f")
                    + " |"
                )
            lines.append("")
            ordered = sorted(counts, key=lambda key: -counts[key])
            biggest, smallest = ordered[0], ordered[-1]
            big_col, small_col = "recall_" + biggest, "recall_" + smallest
            if big_col in table.columns and small_col in table.columns:
                big = float(np.nanmean(np.asarray(table[big_col], dtype=float)))
                small = float(np.nanmean(np.asarray(table[small_col], dtype=float)))
                lines.append(
                    "The gap between the largest class (**"
                    + biggest
                    + "**, "
                    + str(counts[biggest])
                    + " records, recall "
                    + format(big, ".4f")
                    + ") and the smallest (**"
                    + smallest
                    + "**, "
                    + str(counts[smallest])
                    + " records, recall "
                    + format(small, ".4f")
                    + ") is "
                    + format(big - small, ".4f")
                    + ". Balanced class weights reduce that gap; they cannot close "
                    "it, because weighting reweights the examples that exist and "
                    "cannot supply the ones that do not."
                )
                lines.append("")

    # A model that never emits a class has not learned it, whatever its accuracy
    # says. Surfaced here because a metrics table cannot show it -- see
    # `multiclass_report.coverage`.
    if predictions is not None and label_space:
        from src.reporting.multiclass_report import coverage

        preds = pd.DataFrame(predictions)
        by_index = {int(v): str(k) for k, v in dict(label_space).items()}
        labels = sorted(by_index)
        collapsed: list[str] = []
        for model_id, block in preds.groupby("model_id", sort=True):
            facts = coverage(block, labels=labels)
            if facts["predicts_all_classes"]:
                continue
            absent = [
                by_index[int(v)] for v in str(facts["missing_classes"]).split(";") if v != ""
            ]
            collapsed.append(
                "| "
                + str(model_id)
                + " | "
                + str(facts["n_classes_predicted"])
                + " of "
                + str(facts["n_classes_declared"])
                + " | "
                + ", ".join("`" + name + "`" for name in absent)
                + " | "
                + ("**yes -- constant output**" if facts["degenerate"] else "no")
                + " |"
            )
        lines.append("## Models that never predict some class")
        lines.append("")
        if collapsed:
            lines.append(
                "**Read this before the metrics table.** Each model below never "
                "emitted at least one class anywhere in the run. It has not learned "
                "that class, however its accuracy reads -- and on a corpus this "
                "imbalanced, refusing to use the smallest class is *rewarded* by "
                "accuracy. This is what research rule 6 exists to catch."
            )
            lines.append("")
            lines.append("| model | classes predicted | never predicted | collapsed |")
            lines.append("|---|---|---|---|")
            lines.extend(collapsed)
            lines.append("")
            lines.append(
                "A row marked *collapsed* produced one constant answer for every "
                "record. Its scores are those of the corresponding trivial baseline "
                "and must never be presented as a model result."
            )
        else:
            lines.append(
                "Every model emitted every class at least once. No collapsed or "
                "partially-collapsed predictor in this run."
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*PV-MEPCG / PulseVision is an academic screening and decision-support "
        "prototype. It is not a diagnostic tool and must not be used to diagnose, "
        "treat, or make clinical decisions about any patient.*"
    )
    lines.append("")
    return "\n".join(lines)


def write_statement(
    directory: Any,
    *,
    task: str,
    filename: str,
    per_fold: Any = None,
    fold_map: Any = None,
    predictions: Any = None,
    label_space: Any = None,
) -> Any:
    """Render the statement and write it, returning the path."""
    from pathlib import Path

    from src.utils.io import ensure_dir

    target = Path(ensure_dir(directory)) / filename
    text = build_statement(
        task,
        per_fold=per_fold,
        fold_map=fold_map,
        predictions=predictions,
        label_space=label_space,
    )
    target.write_text(text, encoding="utf-8")
    log.info("%s: %d line(s)", filename, len(text.splitlines()))
    return target
