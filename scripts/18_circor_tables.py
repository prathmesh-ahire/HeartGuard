"""Emit T13, T14 and T15 from the completed CirCor runs (T68.5, T68.6, T69.5).

Three runs feed this: EXP-C1 in both declared variants (three-class, matching the
2022 Challenge, and two-class over the 874 known patients) and EXP-C2, the clinical
outcome task. Each is scored at **recording level** and again at **patient level**
under all three aggregation rules, because CirCor labels a patient while the model
scores a recording.

    python scripts/18_circor_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("circor_tables")

SECTION = "outputs/08_circor_external_validation"

#: (run directory, experiment id, variant, positive label, emitted table)
RUNS = (
    ("EXP-C1-three_class", "EXP-C1", "three_class", 1, "T13"),
    ("EXP-C1-two_class", "EXP-C1", "two_class", 1, "T13"),
    ("EXP-C2", "EXP-C2", "", 1, "T14"),
)


def _summarise(frame, keys):
    """mean +/- SD over folds for every numeric metric, grouped by ``keys``."""
    import numpy as np
    import pandas as pd

    skip = set(keys) | {"fold_label", "n_units", "support", "n_classes", "n_declared_classes"}
    metrics = [
        c
        for c in frame.columns
        if c not in skip and pd.api.types.is_numeric_dtype(frame[c])
    ]
    rows = []
    for values, block in frame.groupby(list(keys), sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values, strict=True))
        row["n_folds"] = len(block)
        row["n_units_mean"] = float(np.mean(block["n_units"]))
        for metric in metrics:
            series = np.asarray(block[metric], dtype=float)
            finite = series[np.isfinite(series)]
            row[metric + "_mean"] = float(finite.mean()) if finite.size else float("nan")
            row[metric + "_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)



def _write_propagation_caveat(root: Path, combined) -> Path:
    """T69.2 / T69.7 -- the noise patient-to-recording label propagation introduces.

    Every number here is measured from the DA-07 map and the runs themselves. The
    point is not that propagation is avoidable -- it is not, CirCor labels the
    patient -- but that its cost is stated rather than left for a reader to
    discover.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    folds = pd.read_csv(
        PROJECT_ROOT / "outputs" / "01_dataset_audit" / "subject_split_map.csv"
    )
    outcome = folds[folds["task"] == "circor_outcome"]
    patients = outcome.drop_duplicates("split_group")
    per_patient = outcome.groupby("split_group").size()
    murmur = folds[folds["task"] == "circor_murmur"].drop_duplicates("record_uid")
    counts = patients["class_name"].value_counts().to_dict()

    lines = [
        "# CirCor: what patient-to-recording label propagation costs",
        "",
        "Generated from `outputs/01_dataset_audit/subject_split_map.csv` and the "
        "EXP-C1/EXP-C2 runs. Every number is measured.",
        "",
        "## Where the outcome label comes from",
        "",
        "`Outcome` is **not** in `training_data.csv`. It exists only inside the "
        "per-patient `.txt` files as `#Outcome: Normal|Abnormal`, and a loader "
        "built around the CSV produces a murmur-only pipeline. Parsed from the "
        "txt files: **"
        + str(counts.get("Normal", 0))
        + " Normal / "
        + str(counts.get("Abnormal", 0))
        + " Abnormal** over "
        + str(len(patients))
        + " patients.",
        "",
        "## The propagation, and the noise it introduces",
        "",
        "CirCor labels a **patient**; the model scores a **recording**. Each "
        "patient carries "
        + format(float(per_patient.mean()), ".2f")
        + " recordings on average (min "
        + str(int(per_patient.min()))
        + ", max "
        + str(int(per_patient.max()))
        + "), taken at up to five auscultation locations. Training assigns every "
        "recording its patient's label, which asserts that the finding is present "
        "in **every** recording of that patient.",
        "",
        "**That assertion is false for a murmur and questionable for an outcome.** "
        "A murmur audible at the pulmonary valve need not be audible at the mitral "
        "valve; CirCor ships a `Most audible location` field precisely because "
        "location matters. So an unknown share of the "
        + str(int((murmur['class_name'] == 'Present').sum()))
        + " recordings labelled `Present` contain no audible murmur, and they are "
        "in the **training** data, not only the evaluation.",
        "",
        "Three consequences that must travel with any CirCor number:",
        "",
        "1. **Recording-level metrics are pessimistic by an unmeasured amount.** A "
        "model penalised for not hearing a murmur that is not there is being "
        "scored against a wrong label, not making an error.",
        "2. **The label noise is not random.** It concentrates in patients with "
        "many recordings and in findings that are locally audible, so it "
        "correlates with exactly the structure Phase 70's location analysis "
        "examines.",
        "3. **Patient-level aggregation partly undoes it**, which is why all three "
        "rules are reported. See `T15_recording_vs_patient_level.csv`.",
        "",
        "## Outcome is near-balanced, unlike murmur",
        "",
        "The outcome task is "
        + str(counts.get("Normal", 0))
        + "/"
        + str(counts.get("Abnormal", 0))
        + " -- close to balanced, and the only task in this project that is. "
        "Accuracy is therefore *less* misleading here than elsewhere, but it is "
        "still not the reporting metric: sensitivity and balanced accuracy lead, "
        "per research rule 6. Murmur, by contrast, is "
        + " / ".join(
            str(k) + " " + str(v)
            for k, v in murmur["class_name"].value_counts().items()
        )
        + " at recording level and must never be read the same way.",
        "",
    ]

    patient_rows = combined[combined["level"] == "patient"]
    if not patient_rows.empty and "sensitivity" in patient_rows.columns:
        lines.append("## Measured effect of each aggregation rule")
        lines.append("")
        lines.append("| rule | sensitivity | specificity | balanced accuracy |")
        lines.append("|---|---|---|---|")
        for rule, block in patient_rows.groupby("rule", sort=True):
            lines.append(
                "| "
                + str(rule)
                + " | "
                + format(float(block["sensitivity"].mean()), ".4f")
                + " | "
                + format(float(block["specificity"].mean()), ".4f")
                + " | "
                + format(float(block["balanced_accuracy"].mean()), ".4f")
                + " |"
            )
        recording = combined[combined["level"] == "recording"]
        lines.append(
            "| *recording level* | "
            + format(float(recording["sensitivity"].mean()), ".4f")
            + " | "
            + format(float(recording["specificity"].mean()), ".4f")
            + " | "
            + format(float(recording["balanced_accuracy"].mean()), ".4f")
            + " |"
        )
        lines.append("")
        lines.append(
            "Aggregation moves the operating point rather than creating "
            "information: balanced accuracy barely changes while sensitivity and "
            "specificity trade against each other. **No rule is declared the "
            "winner** -- that is a clinical judgement, and all three are reported."
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

    path = root / "circor_label_propagation.md"
    path.write_text(chr(10).join(lines), encoding="utf-8")
    log.info("%s: %d line(s)", path.name, len(lines))
    return path


def main() -> int:
    import pandas as pd

    from src.evaluation.aggregation import evaluate_aggregations
    from src.evaluation.experiment import Experiment
    from src.utils.run_manifest import start_run

    root = Path(SECTION)
    run = start_run("circor_tables")
    written: dict[str, Path] = {}
    every: list[pd.DataFrame] = []

    for directory, exp_id, variant, positive, _table_id in RUNS:
        path = root / directory / "predictions.parquet"
        if not path.is_file():
            log.error("%s missing; run the experiment first", path)
            return 2

        exp = Experiment.load(exp_id)
        if variant:
            exp = exp.for_variant(variant)
        labels = exp.labels
        names = exp.class_names

        predictions = pd.read_parquet(path)
        scored = evaluate_aggregations(
            predictions,
            labels=labels,
            positive_label=positive,
            class_names=names,
        )
        scored.insert(0, "run", directory)
        scored.insert(1, "task", exp.task)
        scored.insert(2, "n_declared_classes", len(labels))
        every.append(scored)

        # Per-fold values are kept -- Phase 82's paired tests consume them, and an
        # aggregate that cannot be re-derived from its folds is not checkable.
        fold_path = root / directory / "per_fold_by_level.csv"
        scored.to_csv(fold_path, index=False)
        written[fold_path.name + " (" + directory + ")"] = fold_path

    combined = pd.concat(every, ignore_index=True)

    # T13 -- murmur, both variants, both levels.
    murmur = combined[combined["task"] == "circor_murmur"]
    t13 = _summarise(murmur, ("run", "model_id", "level", "rule"))
    t13_path = root / "T13_circor_murmur_results.csv"
    t13.to_csv(t13_path, index=False)
    written["T13"] = t13_path

    # T14 -- outcome.
    outcome = combined[combined["task"] == "circor_outcome"]
    t14 = _summarise(outcome, ("run", "model_id", "level", "rule"))
    t14_path = root / "T14_circor_outcome_results.csv"
    t14.to_csv(t14_path, index=False)
    written["T14"] = t14_path

    # T15 -- recording level against patient level, as a signed delta per rule.
    # This is the table that says what aggregation actually buys, so it is built
    # by joining the two levels rather than by eyeballing two other tables.
    rec = combined[combined["level"] == "recording"]
    pat = combined[combined["level"] == "patient"]
    metrics = [
        m
        for m in ("sensitivity", "specificity", "balanced_accuracy", "macro_f1", "accuracy")
        if m in combined.columns
    ]
    rec_mean = rec.groupby(["run", "model_id"])[metrics].mean()
    rows = []
    for (runid, model_id, rule), block in pat.groupby(["run", "model_id", "rule"], sort=True):
        if (runid, model_id) not in rec_mean.index:
            continue
        base = rec_mean.loc[(runid, model_id)]
        row = {"run": runid, "model_id": model_id, "rule": rule}
        for metric in metrics:
            patient_value = float(block[metric].mean())
            row[metric + "_recording"] = float(base[metric])
            row[metric + "_patient"] = patient_value
            row[metric + "_delta"] = patient_value - float(base[metric])
        rows.append(row)
    t15 = pd.DataFrame(rows)
    t15_path = root / "T15_recording_vs_patient_level.csv"
    t15.to_csv(t15_path, index=False)
    written["T15"] = t15_path

    caveat_path = _write_propagation_caveat(root, combined)
    written["label propagation caveat"] = caveat_path

    for path in written.values():
        run.record_artifact(path)
    run.set("runs", [r[0] for r in RUNS])
    run.finish(status="ok")

    print()
    print(
        t15.groupby("rule")[[m + "_delta" for m in metrics]]
        .mean()
        .round(4)
        .to_string()
    )
    print()
    for name, path in written.items():
        print(f"{name:34s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
