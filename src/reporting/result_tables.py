"""T08-T15, the core result tables, rendered through the table engine (Phase 87).

The numbers already exist. Phases 64-69 wrote them as working CSVs beside the
runs that produced them -- ``T08_individual_model_comparison.csv``,
``T11_pascal_a_results.csv``, ``T13_circor_murmur_results.csv`` and so on -- and
those files are read by name by ``scripts/13_finalize_binary_model.py``, the
Phase 82 statistics and several tests. They are therefore **not overwritten**.
This module renders the deliverable tables under the titles the output
specification uses, in all four formats with provenance, and records the working
CSV and the experiment output behind it as sources.

Three consequences worth knowing
--------------------------------
**Two files carry each of the ids T08-T15.** ``T08_individual_model_comparison.csv``
is the working table (EXP-A1 only, ``mean +/- SD`` strings at four places);
``T08_physionet_individual_model_comparison.*`` is the deliverable. The evidence
index row for each id points at the deliverable once this module has run. A
guard refuses to write a deliverable whose file name equals one of its own
sources: the obvious title for T13, "CirCor Murmur Results", slugs to the
working file's exact name and would have replaced it.

**T08 and T10 carry both binary runs**, EXP-A1 (default hyperparameters) and
EXP-A2 (nested-tuned), where the working files carry EXP-A1 alone. The shipped
model comes from EXP-A2; a model-comparison table without it would not show the
model the project ships.

**Nothing here computes a metric.** T08 and T10 aggregate per-fold rows with the
same functions that wrote the working files; T09 takes means of paired per-fold
differences; T11-T15 select and reshape. Every sentence in a note that quotes a
number is generated from a source CSV, never typed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.reporting.tables import (
    Column,
    Table,
    TableSpec,
    build_table,
    format_value,
    portable_path,
)
from src.utils.logging_setup import get_logger

__all__ = [
    "CORE_TABLE_IDS",
    "RESULT_TABLE_IDS",
    "LOCATIONS",
    "BINARY_RUNS",
    "location_of",
    "table_frame",
    "audit_table",
    "build_t08",
    "build_t09",
    "build_t10",
    "build_t11",
    "build_t12",
    "build_t13",
    "build_t14",
    "build_t15",
    "build_core_tables",
]

log = get_logger("reporting.result_tables")

DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support "
    "prototype, not a diagnostic tool."
)

CORE_TABLE_IDS: tuple[str, ...] = ("T08", "T09", "T10", "T11", "T12", "T13", "T14", "T15")

#: Every numbered result table, T08-T30: the directory it is written into and its
#: file stem. Declared rather than globbed, so a renamed or missing table fails
#: the gate instead of being silently skipped. T20/T21 sit in
#: ``09_robustness_analysis`` because Phases 72-73 wrote them there rather than
#: in ``10_robustness``; see Docs/note.md.
LOCATIONS: dict[str, tuple[str, str]] = {
    "T08": ("outputs/06_binary_results", "T08_physionet_individual_model_comparison"),
    "T09": ("outputs/06_binary_results", "T09_equal_weight_versus_optimized_ensemble"),
    "T10": ("outputs/06_binary_results", "T10_physionet_fold_wise_results"),
    "T11": ("outputs/07_multiclass_results", "T11_pascal_a_multiclass_results"),
    "T12": ("outputs/07_multiclass_results", "T12_pascal_b_multiclass_results"),
    "T13": (
        "outputs/08_circor_external_validation",
        "T13_circor_murmur_classification_results",
    ),
    "T14": ("outputs/08_circor_external_validation", "T14_circor_clinical_outcome_results"),
    "T15": (
        "outputs/08_circor_external_validation",
        "T15_recording_level_versus_subject_level_results",
    ),
    "T16": ("outputs/08_circor_external_validation", "T16_cross_dataset_generalization"),
    "T17": ("outputs/09_ablation", "T17_feature_family_ablation"),
    "T18": ("outputs/09_ablation", "T18_all_features_versus_selected_features"),
    "T19": ("outputs/09_ablation", "T19_search_and_optimization_ablation"),
    "T20": ("outputs/09_robustness_analysis", "T20_noise_robustness"),
    "T21": ("outputs/09_robustness_analysis", "T21_duration_robustness"),
    "T22": ("outputs/08_circor_external_validation", "T22_auscultation_location_analysis"),
    "T23": ("outputs/10_robustness", "T23_calibration_and_confidence_summary"),
    "T24": ("outputs/11_complexity", "T24_complexity_analysis"),
    "T25": ("outputs/11_complexity", "T25_training_and_inference_time"),
    "T26": ("outputs/11_complexity", "T26_model_size_and_memory"),
    "T27": ("outputs/10_robustness", "T27_false_positive_and_false_negative_analysis"),
    "T28": ("outputs/12_statistics", "T28_statistical_significance_comparison"),
    "T29": ("outputs/00_evidence_index", "T29_objective_to_evidence_mapping"),
    "T30": ("outputs/00_evidence_index", "T30_final_conclusion_matrix"),
}

RESULT_TABLE_IDS: tuple[str, ...] = tuple(LOCATIONS)

#: The two binary runs, in the order they appear in T08 and T10.
BINARY_RUNS: tuple[str, ...] = ("EXP-A1", "EXP-A2")

#: Research rule 6: sensitivity and balanced accuracy lead, accuracy comes last.
_BINARY_METRICS: tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "accuracy",
)
_DEFAULT_RULE: tuple[str, ...] = ("sensitivity", "balanced_accuracy")

_CIRCOR_RUN_ORDER = {"EXP-C1-three_class": 0, "EXP-C1-two_class": 1, "EXP-C2": 2}
_LEVEL_ORDER = {"recording": 0, "patient": 1}
_RULE_ORDER = {"none": 0, "max": 1, "mean": 2, "any_present": 3}

_CI = re.compile(r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _root() -> Path:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT


def location_of(table_id: str) -> tuple[Path, str]:
    """``(directory, stem)`` for one numbered result table."""
    directory, stem = LOCATIONS[table_id]
    return _root() / directory, stem


def table_frame(table_id: str) -> Any:
    """The CSV -- the source of truth -- of one written table."""
    directory, stem = location_of(table_id)
    return _read(directory / (stem + ".csv"))


_SUFFIXES: tuple[str, ...] = (".csv", ".md", ".docx", ".tex", ".meta.json")


def audit_table(table_id: str) -> list[str]:
    """Everything wrong with one written table, or an empty list.

    One definition of "a finished table", used by the Phase 87-89 gates and by
    ``scripts/37_result_tables.py --check``:

    * all four renderings plus the ``.meta.json`` exist and are non-empty;
    * every rendering shows exactly the CSV under the kinds and places the meta
      recorded, with no two columns under one header;
    * the meta names an experiment and at least one source, and every source
      is recorded repo-relative, exists, and still has the recorded digest;
    * each source file and the experiment id are named in every rendering, and
      the screening-only statement travels with the table.

    The repo-relative clause exists because T23-T28 were first written with
    ``D:/Projects/HeartGuard/...`` source paths, which resolve on one machine.
    """
    import json

    import pandas as pd
    from docx import Document

    from src.reporting import tables as tb

    directory, stem = location_of(table_id)
    paths = {suffix: directory / (stem + suffix) for suffix in _SUFFIXES}
    problems = [
        table_id + ": missing or empty " + suffix
        for suffix, path in paths.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if problems:
        return problems

    frame = pd.read_csv(paths[".csv"])
    meta = json.loads(paths[".meta.json"].read_text(encoding="utf-8"))
    if meta.get("table_id") != table_id:
        problems.append(table_id + ": meta names table " + str(meta.get("table_id")))
    if not meta.get("exp_id"):
        problems.append(table_id + ": blank experiment id")
    if not meta.get("sources"):
        problems.append(table_id + ": no source recorded")

    expected = tb.expected_body(frame, meta)
    rendered_md: list[list[str]] = []
    for reader, suffix in (
        (tb.read_docx_table, ".docx"),
        (tb.read_latex_table, ".tex"),
        (tb.read_markdown_table, ".md"),
    ):
        rendered = reader(paths[suffix])
        if suffix == ".md":
            rendered_md = rendered
        if rendered[1:] == expected:
            continue
        detail = "row count " + str(len(rendered) - 1) + " vs " + str(len(expected))
        for row, (shown, wanted) in enumerate(zip(rendered[1:], expected, strict=False)):
            if shown != wanted:
                column = next(
                    i for i, (a, b) in enumerate(zip(shown, wanted, strict=False)) if a != b
                )
                detail = (
                    "row "
                    + str(row)
                    + " column "
                    + str(frame.columns[column])
                    + ": shows "
                    + repr(shown[column])
                    + ", CSV renders "
                    + repr(wanted[column])
                )
                break
        problems.append(table_id + suffix + ": does not render its CSV (" + detail + ")")

    if rendered_md:
        header = rendered_md[0]
        duplicated = sorted({h for h in header if header.count(h) > 1})
        if duplicated:
            problems.append(table_id + ": duplicate header(s) " + ", ".join(duplicated))

    for fingerprint in meta.get("sources", []):
        recorded = str(fingerprint.get("path", ""))
        if re.match(r"^([A-Za-z]:|/)", recorded):
            problems.append(table_id + ": machine-specific source path " + recorded)
            continue
        source = _root() / recorded
        if not source.is_file():
            problems.append(table_id + ": source missing " + recorded)
            continue
        if tb.content_digest(source)[0] != fingerprint.get("sha256"):
            problems.append(table_id + ": stale -- " + recorded + " changed since it was built")

    docx_text = " ".join(p.text for p in Document(str(paths[".docx"])).paragraphs)
    renderings = {
        ".md": paths[".md"].read_text(encoding="utf-8"),
        ".tex": paths[".tex"].read_text(encoding="utf-8"),
        ".docx": docx_text,
    }
    for suffix, blob in renderings.items():
        for fingerprint in meta.get("sources", []):
            name = Path(str(fingerprint.get("path", ""))).name
            if name and name not in blob:
                problems.append(table_id + suffix + ": does not name its source " + name)
        if meta.get("exp_id") and str(meta["exp_id"]) not in blob:
            problems.append(table_id + suffix + ": omits its experiment id")
    lowered = renderings[".md"].lower()
    if "not a diagnostic tool" not in lowered and "screening" not in lowered:
        problems.append(table_id + ": carries no screening-only statement")
    return problems


def _section(name: str) -> Path:
    return _root() / "outputs" / name


def _read(path: Path) -> Any:
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(
            "source CSV not found: "
            + portable_path(path)
            + " -- the phase that writes it has not run in this checkout"
        )
    return pd.read_csv(path)


def _m(value: Any) -> str:
    """A number as the metric rule renders it, for use inside a generated note."""
    return format_value(value, "metric")


def _finish(spec: TableSpec, frame: Any) -> Table:
    """Bind the frame, refusing a deliverable that would overwrite a source."""
    _, stem = LOCATIONS[spec.table_id]
    if spec.slug() != stem:
        raise ValueError(
            spec.table_id + ": title slugs to " + spec.slug() + ", not the declared " + stem
        )
    if stem + ".csv" in {Path(source).name for source in spec.sources}:
        raise ValueError(
            spec.table_id + ": the deliverable would overwrite its own source " + stem + ".csv"
        )
    return build_table(spec, frame)


def _within_corpus_note() -> tuple[str, str | None]:
    """The EXP-F3 caveat, with its numbers read from T-S5 rather than typed."""
    path = _section("09_ablation") / "T-S5_leave_one_source_out_generalization.csv"
    if not path.is_file():
        return (
            "Every PhysioNet figure is cross-validated WITHIN the PhysioNet 2016 "
            "corpus and is not a claim of generalization or field performance.",
            None,
        )
    loso = _read(path)
    return (
        "Every PhysioNet figure here is cross-validated WITHIN the PhysioNet 2016 "
        "corpus. Holding out one recording sub-collection at a time (EXP-F3, T-S5) "
        "takes ROC-AUC from "
        + _m(loso["roc_auc_pooled"].min())
        + "-"
        + _m(loso["roc_auc_pooled"].max())
        + " pooled to "
        + _m(loso["roc_auc_holdout"].min())
        + "-"
        + _m(loso["roc_auc_holdout"].max())
        + " across the "
        + str(len(loso))
        + " models, so no number here is a claim of generalization, "
        "deployment-readiness or field performance.",
        portable_path(path),
    )


def _significance(
    run: str, metric: str, model_a: str, model_b: str
) -> tuple[dict[str, Any] | None, str | None]:
    """One row of the Phase 82 paired tests, looked up in either model order."""
    path = _section("12_statistics") / "paired_fold_tests_interpreted.csv"
    if not path.is_file():
        return None, None
    tests = _read(path)
    for left, right in ((model_a, model_b), (model_b, model_a)):
        match = tests[
            (tests["run"] == run)
            & (tests["metric"] == metric)
            & (tests["model_a"] == left)
            & (tests["model_b"] == right)
        ]
        if len(match):
            return match.iloc[0].to_dict(), portable_path(path)
    return None, portable_path(path)


# ---------------------------------------------------------------------------
# T08 -- PhysioNet individual model comparison
# ---------------------------------------------------------------------------


def _trials_sentence(exp_id: str, per_fold: Any) -> str | None:
    """What the nested search actually spent, read from the per-fold planner log."""
    if "planner_n_trials" not in per_fold.columns:
        return None
    trials = per_fold.dropna(subset=["planner_n_trials"])
    if not len(trials):
        return None
    counts = trials.groupby("model_id")["planner_n_trials"].median()
    parts = [str(model) + " " + str(int(value)) for model, value in counts.items()]
    return (
        exp_id + " is nested: every outer fold ran its own reduced-budget search on inner "
        "folds of its training rows only. Median search trials per outer fold: "
        + ", ".join(parts)
        + ". Write it up as that budget, never as 'tuned' unqualified."
    )


def build_t08(command: str = "") -> Table:
    import pandas as pd

    from src.evaluation.experiment import Experiment
    from src.reporting.experiment_report import build_t08 as summarise

    frames: list[Any] = []
    sources: list[str] = []
    notes: list[str] = []
    for exp_id in BINARY_RUNS:
        experiment = Experiment.load(exp_id)
        path = _section("06_binary_results") / exp_id / "per_fold_metrics.csv"
        per_fold = _read(path)
        rule = tuple(getattr(experiment, "selection_rule", None) or _DEFAULT_RULE)
        table = summarise(per_fold, metrics=_BINARY_METRICS, rule=rule)
        table.insert(0, "run", exp_id)
        table.insert(1, "tuning", "nested search" if experiment.tuned else "defaults")
        frames.append(table)
        sources.append(portable_path(path))

        by_rule = table.iloc[0]
        by_accuracy = table.sort_values("accuracy_mean", ascending=False).iloc[0]
        if by_accuracy["model_id"] != by_rule["model_id"]:
            notes.append(
                exp_id
                + ": ranked by "
                + str(by_rule["ranked_by"])
                + ", "
                + str(by_rule["model_id"])
                + " is first; ranked by accuracy, "
                + str(by_accuracy["model_id"])
                + " would be first ("
                + _m(by_accuracy["accuracy_mean"])
                + "), and the rule ranks it "
                + str(int(by_accuracy["rank"]))
                + ". The difference is why accuracy never ranks a model here."
            )
        sentence = _trials_sentence(exp_id, per_fold) if experiment.tuned else None
        if sentence:
            notes.append(sentence)

    frame = pd.concat(frames, ignore_index=True)

    selection_path = _section("06_binary_results") / "final_model_selection.csv"
    if selection_path.is_file():
        selection = _read(selection_path)
        final = selection.sort_values("rank").iloc[0]
        runner_up = selection.sort_values("rank").iloc[1]
        sentence = (
            "The shipped final binary model is "
            + str(final["model_id"])
            + ", first in EXP-A2 by the pre-registered rule."
        )
        test, test_path = _significance(
            "EXP-A2", "sensitivity", str(final["model_id"]), str(runner_up["model_id"])
        )
        if test is not None:
            sentence += (
                " It is not distinguishable from "
                + str(runner_up["model_id"])
                + " on sensitivity ("
                + str(test["chosen_test"])
                + ", n="
                + str(int(test["n"]))
                + " folds, p="
                + format_value(test["p_value"], "p_value")
                + ", Holm-corrected "
                + format_value(test["p_holm"], "p_value")
                + "), so the table does not show that "
                + str(final["model_id"])
                + " outperforms the ensemble; it shows they are tied and "
                + str(final["model_id"])
                + " is simpler."
            )
            if test_path:
                sources.append(test_path)
        notes.append(sentence)
        sources.append(portable_path(selection_path))

    within, loso_path = _within_corpus_note()
    notes.append(within)
    if loso_path:
        sources.append(loso_path)
    notes.append(
        "Mean and SD are over the 25 folds of the repeated 5x5 subject-grouped map "
        "(sample SD, ddof=1). Folds of a repeated CV overlap, so the SD is a spread, "
        "not a standard error; T28 carries the paired tests."
    )
    notes.append(
        "M9 (1D-CNN) is out of scope on CPU-only hardware and was not run; no "
        "comparison against a convolutional network is made."
    )
    notes.append(DISCLAIMER)

    columns: list[Column] = [
        Column("run", "Run"),
        Column("tuning", "Hyperparameters"),
        Column("rank", "Rank", kind="count"),
        Column("model_id", "ID"),
        Column("model_name", "Model"),
        Column("n_folds", "Folds", kind="count"),
    ]
    for metric in ("sensitivity", "specificity", "balanced_accuracy"):
        label = metric.replace("_", " ").capitalize()
        columns.append(Column(metric + "_mean", label, kind="metric"))
        columns.append(Column(metric + "_sd", label + " SD", kind="metric"))
    columns.extend(
        [
            Column("f1_mean", "F1", kind="metric"),
            Column("roc_auc_mean", "ROC-AUC", kind="metric"),
            Column("roc_auc_sd", "ROC-AUC SD", kind="metric"),
            Column("accuracy_mean", "Accuracy", kind="metric"),
        ]
    )

    spec = TableSpec(
        table_id="T08",
        title="PhysioNet Individual Model Comparison",
        caption=(
            "Every model on the PhysioNet 2016 binary task (normal versus abnormal), "
            "as the mean and SD over the identical 25 folds of the repeated 5x5 "
            "subject-grouped map, in two runs: EXP-A1 with default hyperparameters "
            "and EXP-A2 with a nested per-fold search. Within each run models are "
            "ranked by sensitivity, then balanced accuracy -- the project's "
            "pre-registered selection rule -- and accuracy is shown last because it "
            "ranks nothing."
        ),
        sources=tuple(dict.fromkeys(sources)),
        columns=tuple(columns),
        exp_id=", ".join(BINARY_RUNS),
        objective="O1 (normal versus abnormal), O3 (model comparison)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=tuple(notes),
        command=command,
    )
    return _finish(spec, frame)


# ---------------------------------------------------------------------------
# T09 -- equal-weight versus optimized ensemble
# ---------------------------------------------------------------------------


def build_t09(command: str = "") -> Table:
    import numpy as np
    import pandas as pd

    path = _section("06_binary_results") / "T09_ensemble_weight_comparison.csv"
    comparison = _read(path)
    sources = [portable_path(path)]
    test_source: str | None = None

    rows: list[dict[str, Any]] = []
    for metric in _BINARY_METRICS:
        delta_column = "delta_" + metric
        if delta_column not in comparison.columns:
            continue
        delta = comparison[delta_column].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "metric": metric,
            "n_folds": len(delta),
            "m6_mean": float(np.mean(comparison["M6_" + metric])),
            "m7_mean": float(np.mean(comparison["M7_" + metric])),
            "delta_mean": float(np.mean(delta)),
            "delta_sd": float(np.std(delta, ddof=1)) if len(delta) > 1 else float("nan"),
            "n_folds_m7_better": int(np.sum(delta > 0)),
            "n_folds_m7_worse": int(np.sum(delta < 0)),
            "n_folds_identical": int(np.sum(delta == 0)),
        }
        test, test_path = _significance("EXP-A2", metric, "M6", "M7")
        test_source = test_path or test_source
        row["test"] = str(test["chosen_test"]) if test is not None else "not run"
        row["p_value"] = float(test["p_value"]) if test is not None else float("nan")
        row["p_holm"] = float(test["p_holm"]) if test is not None else float("nan")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if test_source:
        sources.append(test_source)

    identical = int(frame.loc[frame["metric"] == "balanced_accuracy", "n_folds_identical"].iloc[0])
    significant = frame[frame["p_holm"] < 0.05]
    verdict = (
        "No metric differs significantly between M6 and M7 after Holm correction "
        "(n=" + str(int(frame["n_folds"].iloc[0])) + " folds per test)."
        if not len(significant)
        else "Significant after Holm correction: "
        + ", ".join(str(m) for m in significant["metric"])
        + "."
    )

    spec = TableSpec(
        table_id="T09",
        title="Equal-Weight versus Optimized Ensemble",
        caption=(
            "The equal-weight soft-voting ensemble (M6) against the same ensemble "
            "with weights chosen per fold by the SO-05 weight search (M7), on the "
            "identical 25 EXP-A2 folds. Delta is M7 minus M6 on each fold, so a "
            "positive delta favours the optimized weights; 'identical' counts folds "
            "on which the two produced the same value."
        ),
        sources=tuple(sources),
        columns=(
            Column("metric", "Metric"),
            Column("n_folds", "Folds", kind="count"),
            Column("m6_mean", "M6 equal weights", kind="metric"),
            Column("m7_mean", "M7 optimized weights", kind="metric"),
            Column("delta_mean", "Mean delta (M7 - M6)", kind="metric", places=4),
            Column("delta_sd", "Delta SD", kind="metric", places=4),
            Column("n_folds_m7_better", "Folds M7 better", kind="count"),
            Column("n_folds_m7_worse", "Folds M7 worse", kind="count"),
            Column("n_folds_identical", "Folds identical", kind="count"),
            Column("test", "Test"),
            Column("p_value", "p (raw)", kind="p_value"),
            Column("p_holm", "p (Holm)", kind="p_value"),
        ),
        exp_id="EXP-A2",
        objective="O3 (ensemble), O5 (ensemble-weight optimization)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=(
            "On "
            + str(identical)
            + " of "
            + str(int(frame["n_folds"].iloc[0]))
            + " folds M7's balanced accuracy equals M6's exactly: the weight search "
            "returned equal weights or weights that change no prediction. A weight "
            "search that lands on equal weights is a result -- the members are too "
            "close in quality for the data to support preferring one.",
            verdict,
            "Deltas are shown at four places because they are smaller than the "
            "three-place metric rule can display.",
            "M7's weights are chosen by a one-standard-error rule that shrinks "
            "toward equal weights, a selection rule fixed before any result was "
            "seen.",
            DISCLAIMER,
        ),
        command=command,
    )
    return _finish(spec, frame)


# ---------------------------------------------------------------------------
# T10 -- PhysioNet fold-wise results, mean +/- SD
# ---------------------------------------------------------------------------


def build_t10(command: str = "") -> Table:
    import pandas as pd

    from src.reporting.experiment_report import build_t10_summary

    frames: list[Any] = []
    sources: list[str] = []
    for exp_id in BINARY_RUNS:
        path = _section("06_binary_results") / exp_id / "per_fold_metrics.csv"
        summary = build_t10_summary(_read(path), metrics=_BINARY_METRICS)
        summary["run"] = exp_id
        frames.append(summary)
        sources.append(portable_path(path))
    frame = pd.concat(frames, ignore_index=True)

    order = {metric: position for position, metric in enumerate(_BINARY_METRICS)}
    frame["_metric_order"] = frame["metric"].map(order)
    frame = (
        frame.sort_values(["run", "model_id", "_metric_order"], kind="mergesort")
        .drop(columns=["_metric_order"])
        .reset_index(drop=True)
    )

    undefined = frame[frame["n_folds_defined"] < frame["n_folds"]]
    notes = [
        "SD is the sample SD across folds (ddof=1). The 25 training sets of a "
        "repeated CV overlap heavily, so dividing it by the square root of 25 would "
        "give an interval far too narrow; it is reported as a spread.",
        "Every per-fold value behind these rows is kept in the run's "
        "per_fold_metrics.csv and in T10_fold_wise_results.csv; the Phase 82 paired "
        "tests are computed on them.",
    ]
    if len(undefined):
        notes.append(
            "Folds with an undefined value are excluded from that metric's mean: "
            + ", ".join(
                str(r.run) + " " + str(r.model_id) + " " + str(r.metric)
                for r in undefined.itertuples(index=False)
            )
            + "."
        )
    within, loso_path = _within_corpus_note()
    notes.append(within)
    if loso_path:
        sources.append(loso_path)
    notes.append(DISCLAIMER)

    spec = TableSpec(
        table_id="T10",
        title="PhysioNet Fold-wise Results",
        caption=(
            "The fold-wise distribution of every reported metric for every model on "
            "the PhysioNet 2016 binary task: mean, SD, minimum and maximum over the "
            "25 folds of the repeated 5x5 subject-grouped map, for EXP-A1 (default "
            "hyperparameters) and EXP-A2 (nested search)."
        ),
        sources=tuple(sources),
        columns=(
            Column("run", "Run"),
            Column("model_id", "Model"),
            Column("metric", "Metric"),
            Column("n_folds", "Folds", kind="count"),
            Column("n_folds_defined", "Folds defined", kind="count"),
            Column("mean", "Mean", kind="metric"),
            Column("sd", "SD", kind="metric"),
            Column("min", "Min", kind="metric"),
            Column("max", "Max", kind="metric"),
        ),
        exp_id=", ".join(BINARY_RUNS),
        objective="O1 (normal versus abnormal), O3 (model comparison)",
        dataset="D1 PhysioNet 2016 (3,240 records, 857 subjects)",
        notes=tuple(notes),
        command=command,
    )
    return _finish(spec, frame)


# ---------------------------------------------------------------------------
# T11 / T12 -- PASCAL multiclass
# ---------------------------------------------------------------------------


def _interval(text: Any) -> tuple[float, float] | None:
    match = _CI.search(str(text))
    return (float(match.group(1)), float(match.group(2))) if match else None


def _pascal(
    table_id: str,
    title: str,
    exp_id: str,
    working: str,
    caption: str,
    dataset: str,
    specific_notes: tuple[str, ...],
    command: str,
) -> Table:
    from src.evaluation.experiment import Experiment

    section = _section("07_multiclass_results")
    head_path = section / working
    per_class_path = section / working.replace(".csv", "_per_class.csv")
    head = _read(head_path)
    per_class = _read(per_class_path)

    experiment = Experiment.load(exp_id)
    classes = tuple(experiment.class_names)
    names_by_label = {int(v): k for k, v in dict(experiment.label_space).items()}

    frame = head.copy()
    recall = per_class.pivot(index="model_id", columns="class_name", values="recall_mean")
    for name in classes:
        frame["recall_" + name] = frame["model_id"].map(recall[name])

    support = per_class[per_class["model_id"] == per_class["model_id"].iloc[0]].set_index(
        "class_name"
    )["support_mean"]
    majority = str(support.idxmax())
    majority_share = float(support.max() / support.sum())
    beaten = frame[frame["accuracy_mean"] <= majority_share]["model_id"].tolist()

    notes: list[str] = [
        "Ranked by " + str(head["ranked_by"].iloc[0]) + "; per-class recall is shown "
        "for every class (research rule 6), and accuracy is shown last.",
        "Always predicting the majority class '"
        + majority
        + "' scores accuracy "
        + _m(majority_share)
        + (
            ". "
            + ", ".join(beaten)
            + " do not beat that on accuracy, which is why accuracy never ranks a "
            "model on this task."
            if beaten
            else "; every model beats it on accuracy."
        ),
    ]

    partial = frame[frame["n_classes_predicted"] < len(classes)]
    if len(partial):
        described = []
        for row in partial.itertuples(index=False):
            missing = str(getattr(row, "missing_classes", ""))
            labels = [
                names_by_label.get(int(float(part)), part)
                for part in re.split(r"[|;, ]+", missing)
                if part and part.lower() != "nan"
            ]
            described.append(
                str(row.model_id) + " never predicts " + (", ".join(labels) or "one class")
            )
        notes.append("Class coverage: " + "; ".join(described) + ".")

    intervals = [_interval(text) for text in frame["macro_f1_record_ci"]]
    if all(intervals):
        bounds = [b for b in intervals if b is not None]
        overlap_all = all(
            a[0] <= b[1] and b[0] <= a[1] for i, a in enumerate(bounds) for b in bounds[i + 1 :]
        )
        notes.append(
            (
                "Every model's record-level macro-F1 interval overlaps every other's, "
                "so the ranking orders models this sample cannot distinguish."
            )
            if overlap_all
            else "At least one pair of record-level macro-F1 intervals does not overlap."
        )
    notes.append(
        "Record-level intervals are a bootstrap over the stored out-of-fold "
        "predictions; fold-level intervals and SDs are in the working file "
        + working
        + ", and the generated caveats are in "
        + working.replace(".csv", "_caveats.md")
        + "."
    )
    notes.extend(specific_notes)
    notes.append(DISCLAIMER)

    columns: list[Column] = [
        Column("model_id", "ID"),
        Column("model_name", "Model"),
        Column("n_folds", "Folds", kind="count"),
        Column("macro_f1_mean", "Macro-F1", kind="metric"),
        Column("macro_f1_sd", "SD", kind="metric"),
        Column("macro_f1_record_ci", "Macro-F1 95% CI", kind="preformatted"),
        Column("balanced_accuracy_mean", "Balanced accuracy", kind="metric"),
        Column("balanced_accuracy_record_ci", "Bal. acc. 95% CI", kind="preformatted"),
    ]
    columns.extend(Column("recall_" + name, name + " recall", kind="metric") for name in classes)
    columns.extend(
        [
            Column("accuracy_mean", "Accuracy", kind="metric"),
            Column("n_classes_predicted", "Classes predicted", kind="count"),
            Column("degenerate", "Degenerate"),
        ]
    )

    spec = TableSpec(
        table_id=table_id,
        title=title,
        caption=caption,
        sources=(portable_path(head_path), portable_path(per_class_path)),
        columns=tuple(columns),
        exp_id=exp_id,
        objective="O6 (multiclass classification)",
        dataset=dataset,
        notes=tuple(notes),
        command=command,
    )
    return _finish(spec, frame)


def build_t11(command: str = "") -> Table:
    n_records = _read(_section("07_multiclass_results") / "T11_pascal_a_results.csv")[
        "n_records"
    ].iloc[0]
    return _pascal(
        "T11",
        "PASCAL A Multiclass Results",
        "EXP-B1",
        "T11_pascal_a_results.csv",
        (
            "PASCAL set_a, four classes (normal, murmur, extra heart sound, artifact), "
            "over the repeated 5x2 stratified map with a nested per-fold search. "
            "Mean and SD over folds, with record-level bootstrap 95% intervals, "
            "because at "
            + format_value(n_records, "count")
            + " records a point estimate alone would mislead."
        ),
        "D2 PASCAL set_a (labelled records only)",
        (
            "'artifact' is a RECORDING-QUALITY label, not a cardiac class. This is not "
            "a four-class cardiac classifier and must never be described as one.",
            "PASCAL set_a carries no subject identifiers, so results are record-level.",
            "The headline is the pre-registered tuned run even where defaults scored "
            "higher; the paired comparison is T11_pascal_a_results_tuning_comparison.csv.",
            "PASCAL A and PASCAL B are separate tasks with separate label spaces and "
            "were never merged.",
        ),
        command,
    )


def build_t12(command: str = "") -> Table:
    head = _read(_section("07_multiclass_results") / "T12_pascal_b_results.csv")
    n_folds = int(head["n_folds"].iloc[0])
    return _pascal(
        "T12",
        "PASCAL B Multiclass Results",
        "EXP-B2",
        "T12_pascal_b_results.csv",
        (
            "PASCAL set_b, three classes (normal, murmur, extrasystole), over a "
            "subject-grouped "
            + str(n_folds)
            + "-fold map with a nested per-fold search. Mean and SD over folds, with "
            "record-level bootstrap 95% intervals."
        ),
        "D3 PASCAL set_b (labelled records, filename-derived subjects)",
        (
            "This map is a plain "
            + str(n_folds)
            + "-fold split, not a repeated one, so any paired test on it has n="
            + str(n_folds)
            + " and cannot reach significance in some configurations regardless of "
            "effect size.",
            "PASCAL A and PASCAL B are separate tasks with separate label spaces and "
            "were never merged.",
        ),
        command,
    )


# ---------------------------------------------------------------------------
# T13 / T14 / T15 -- CirCor
# ---------------------------------------------------------------------------


def _circor_sorted(frame: Any) -> Any:
    ordered = frame.copy()
    ordered["_run"] = ordered["run"].map(_CIRCOR_RUN_ORDER).fillna(99)
    ordered["_level"] = ordered["level"].map(_LEVEL_ORDER).fillna(99) if "level" in ordered else 0
    ordered["_rule"] = ordered["rule"].map(_RULE_ORDER).fillna(99)
    return (
        ordered.sort_values(["_run", "_level", "_rule", "model_id"], kind="mergesort")
        .drop(columns=["_run", "_level", "_rule"])
        .reset_index(drop=True)
    )


def _circor_sources(working: Path, runs: tuple[str, ...]) -> tuple[str, ...]:
    found = [portable_path(working)]
    for run in runs:
        path = _section("08_circor_external_validation") / run / "per_fold_by_level.csv"
        if path.is_file():
            found.append(portable_path(path))
    return tuple(found)


_CIRCOR_COMMON: tuple[str, ...] = (
    "CirCor labels a PATIENT and the model scores a RECORDING, so every recording "
    "inherits its patient's label, including recordings from locations where the "
    "finding is not audible. Recording-level figures are pessimistic by an "
    "unmeasured amount; see circor_label_propagation.md.",
    "Patient-level rows aggregate a patient's recordings under three rules (max, "
    "mean, any_present). Aggregation moves the operating point; no rule is declared "
    "the winner, because that is a clinical judgement.",
    "CirCor is a predominantly paediatric cohort; nothing here transfers a "
    "PhysioNet result to it (that is EXP-D1, T16).",
)


def _models_note(frame: Any) -> str:
    models = sorted(str(m) for m in frame["model_id"].unique())
    return (
        "Models: "
        + ", ".join(models)
        + ". "
        + (
            "The shipped binary model M1 is not among them: the CirCor tracks carry "
            "their own model set."
            if "M1" not in models
            else ""
        )
    ).strip()


def build_t13(command: str = "") -> Table:
    working = _section("08_circor_external_validation") / "T13_circor_murmur_results.csv"
    frame = _circor_sorted(_read(working))

    patient_auc = frame.loc[frame["level"] == "patient", "roc_auc_mean"]
    notes = [
        "The three-class variant (Absent / Present / Unknown) is the headline, "
        "matching the 2022 Challenge. The two-class variant excludes the patients "
        "whose murmur is Unknown and is reported alongside, never instead.",
        _models_note(frame),
        "Per-class recall columns apply to the three-class variant; sensitivity and "
        "specificity (murmur Present as positive) apply to the two-class variant. "
        "'n/a' marks a metric the variant does not define.",
    ]
    if patient_auc.isna().all():
        notes.append(
            "ROC-AUC was computed at recording level only; the patient-level rows "
            "carry none, and 'n/a' there is an absent measurement, not a zero."
        )
    notes.extend(_CIRCOR_COMMON)
    notes.append(DISCLAIMER)

    spec = TableSpec(
        table_id="T13",
        title="CirCor Murmur Classification Results",
        caption=(
            "CirCor DigiScope 2022 murmur classification, patient-grouped 5-fold "
            "cross-validation with a nested per-fold search, scored at recording "
            "level and at patient level under each aggregation rule. Mean over the "
            "5 folds with the SD of balanced accuracy."
        ),
        sources=_circor_sources(working, ("EXP-C1-three_class", "EXP-C1-two_class")),
        columns=(
            Column("run", "Variant"),
            Column("model_id", "Model"),
            Column("level", "Level"),
            Column("rule", "Rule"),
            Column("n_folds", "Folds", kind="count"),
            Column("n_units_mean", "Units / fold", kind="mean_count"),
            Column("balanced_accuracy_mean", "Balanced accuracy", kind="metric"),
            Column("balanced_accuracy_sd", "SD", kind="metric"),
            Column("macro_f1_mean", "Macro-F1", kind="metric"),
            Column("recall_Absent_mean", "Absent recall", kind="metric"),
            Column("recall_Present_mean", "Present recall", kind="metric"),
            Column("recall_Unknown_mean", "Unknown recall", kind="metric"),
            Column("sensitivity_mean", "Sensitivity", kind="metric"),
            Column("specificity_mean", "Specificity", kind="metric"),
            Column("roc_auc_mean", "ROC-AUC", kind="metric"),
            Column("accuracy_mean", "Accuracy", kind="metric"),
        ),
        exp_id="EXP-C1",
        objective="O6 (large-sample validation), O1 (external corpus)",
        dataset="D4 CirCor DigiScope 2022 (942 patients, 3,163 recordings)",
        notes=tuple(notes),
        command=command,
    )
    return _finish(spec, frame)


def build_t14(command: str = "") -> Table:
    working = _section("08_circor_external_validation") / "T14_circor_outcome_results.csv"
    frame = _circor_sorted(_read(working))

    patients = frame[frame["level"] == "patient"].iloc[0]
    share = patients["n_positive_mean"] / (
        patients["n_positive_mean"] + patients["n_negative_mean"]
    )
    notes = [
        "The outcome label is NOT in training_data.csv: it is parsed from the "
        "'#Outcome:' line of each patient's .txt file.",
        "Abnormal is the positive class and makes up "
        + format_value(share * 100.0, "percent")
        + "% of the patients in an average test fold -- close to balanced, the only "
        "near-balanced task in this project -- but sensitivity and balanced "
        "accuracy still lead.",
        _models_note(frame),
    ]
    notes.extend(_CIRCOR_COMMON)
    notes.append(DISCLAIMER)

    spec = TableSpec(
        table_id="T14",
        title="CirCor Clinical Outcome Results",
        caption=(
            "CirCor DigiScope 2022 clinical outcome (normal versus abnormal), "
            "patient-grouped 5-fold cross-validation with a nested per-fold search, "
            "scored at recording level and at patient level under each aggregation "
            "rule. Mean and SD over the 5 folds."
        ),
        sources=_circor_sources(working, ("EXP-C2",)),
        columns=(
            Column("model_id", "Model"),
            Column("level", "Level"),
            Column("rule", "Rule"),
            Column("n_folds", "Folds", kind="count"),
            Column("n_units_mean", "Units / fold", kind="mean_count"),
            Column("sensitivity_mean", "Sensitivity", kind="metric"),
            Column("sensitivity_sd", "Sensitivity SD", kind="metric"),
            Column("specificity_mean", "Specificity", kind="metric"),
            Column("specificity_sd", "Specificity SD", kind="metric"),
            Column("balanced_accuracy_mean", "Balanced accuracy", kind="metric"),
            Column("balanced_accuracy_sd", "Balanced accuracy SD", kind="metric"),
            Column("f1_mean", "F1", kind="metric"),
            Column("roc_auc_mean", "ROC-AUC", kind="metric"),
            Column("accuracy_mean", "Accuracy", kind="metric"),
        ),
        exp_id="EXP-C2",
        objective="O6 (large-sample validation), O1 (external corpus)",
        dataset="D4 CirCor DigiScope 2022 (942 patients, 3,163 recordings)",
        notes=tuple(notes),
        command=command,
    )
    return _finish(spec, frame)


def build_t15(command: str = "") -> Table:
    working = _section("08_circor_external_validation") / "T15_recording_vs_patient_level.csv"
    raw = _read(working)
    raw["level"] = "patient"
    frame = _circor_sorted(raw).drop(columns=["level"])

    by_rule = frame.groupby("rule")["balanced_accuracy_delta"].mean()
    summary = ", ".join(
        str(rule) + " " + ("+" if value >= 0 else "") + _m(value)
        for rule, value in sorted(by_rule.items(), key=lambda kv: _RULE_ORDER.get(kv[0], 99))
    )
    small = bool((by_rule.abs() < 0.05).all())
    notes = [
        "Delta is patient level minus recording level for the same model and run, so "
        "a positive delta means aggregation raised the metric.",
        "Mean balanced-accuracy delta per rule, over every run and model: "
        + summary
        + "."
        + (
            " Every rule moves balanced accuracy by less than 0.05 on average: "
            "aggregation shifts the operating point between sensitivity and "
            "specificity more than it adds information."
            if small
            else ""
        ),
        "CirCor is the only corpus here with both levels: PhysioNet and PASCAL are "
        "scored per recording with subjects grouped across folds, never aggregated "
        "to a subject-level prediction.",
    ]
    notes.extend(_CIRCOR_COMMON[:2])
    notes.append(DISCLAIMER)

    columns: list[Column] = [
        Column("run", "Run"),
        Column("model_id", "Model"),
        Column("rule", "Rule"),
    ]
    for metric in ("balanced_accuracy", "sensitivity", "specificity", "macro_f1"):
        label = metric.replace("_", " ")
        columns.extend(
            [
                Column(metric + "_recording", label + " (recording)", kind="metric"),
                Column(metric + "_patient", label + " (patient)", kind="metric"),
                Column(metric + "_delta", label + " delta", kind="metric"),
            ]
        )

    spec = TableSpec(
        table_id="T15",
        title="Recording-Level versus Subject-Level Results",
        caption=(
            "What patient-level aggregation does to every CirCor result: the "
            "recording-level mean beside the patient-level mean under each "
            "aggregation rule, with the signed difference. In CirCor the subject is "
            "the patient."
        ),
        sources=_circor_sources(working, ("EXP-C1-three_class", "EXP-C1-two_class", "EXP-C2")),
        columns=tuple(columns),
        exp_id="EXP-C1, EXP-C2",
        objective="O6 (large-sample validation)",
        dataset="D4 CirCor DigiScope 2022 (942 patients, 3,163 recordings)",
        notes=tuple(notes),
        command=command,
    )
    return _finish(spec, frame)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_BUILDERS = {
    "T08": build_t08,
    "T09": build_t09,
    "T10": build_t10,
    "T11": build_t11,
    "T12": build_t12,
    "T13": build_t13,
    "T14": build_t14,
    "T15": build_t15,
}


def build_core_tables(
    table_ids: tuple[str, ...] = CORE_TABLE_IDS, *, command: str = ""
) -> dict[str, Table]:
    """Build the requested T08-T15 tables without writing anything."""
    built: dict[str, Table] = {}
    for table_id in table_ids:
        if table_id not in _BUILDERS:
            raise KeyError("not a Phase 87 table: " + table_id)
        built[table_id] = _BUILDERS[table_id](command)
        log.info("%s built (%d rows)", table_id, len(built[table_id].frame))
    return built
