"""The Q1 / IEEE paper asset pack (Phase 103).

``outputs/Q1_PAPER_ASSETS/`` holds the twenty assets of the extraction spec's
section 15 -- nine tables, two figures, five graphs, three algorithms and a
results narrative -- and nothing else. None of them computes anything.

**Every number is a selection, never a calculation.** A Q1 table or graph is a
list of :class:`Block` s: one generated CSV, a row filter, and a column mapping.
Each output row carries the file it came from (``source_file``), and
:func:`verify_q1_pack` re-derives every Q1 CSV from its sources and compares
them cell by cell (T103.7, T105.5). A block cannot round, average or rescale,
so there is nothing for a Q1 table to get wrong that its source did not.

**The narrative is written from facts.** Each number in
``Q1_results_narrative.docx`` is a :class:`Fact` -- one CSV, one row, one
column -- read when the document is built, and listed with its source in
``Q1_results_narrative_sources.csv`` and in the document itself. The sentences
around the numbers are prose; the numbers inside them are not.

The final model and the top multiclass models are read from the result files
(``final_model_selection.csv``, the first row of T11/T12), never named here.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.evidence import PROJECT_ROOT, register_evidence
from src.utils.logging_setup import get_logger

__all__ = [
    "Block",
    "Fact",
    "Q1_DIR",
    "COMMAND",
    "build_frame",
    "read_source",
    "q1_tables",
    "q1_graphs",
    "narrative_facts",
    "verify_q1_pack",
    "write_q1_pack",
]

log = get_logger(__name__)

Q1_DIR = "outputs/Q1_PAPER_ASSETS"
COMMAND = "python scripts/43_q1_assets.py"

DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support prototype. "
    "It does not diagnose, and nothing here is a diagnosis, a treatment "
    "recommendation or a substitute for clinical judgement."
)

# Source files, repo-relative.
T01 = "outputs/01_dataset_audit/T01_dataset_inventory.csv"
T02 = "outputs/01_dataset_audit/T02_class_distribution_and_imbalance_ratio.csv"
T05 = "outputs/03_features/T05_feature_inventory_and_counts.csv"
T07 = "outputs/05_search_optimization/T07_search_space_and_best_parameters.csv"
T08 = "outputs/06_binary_results/T08_physionet_individual_model_comparison.csv"
T09 = "outputs/06_binary_results/T09_equal_weight_versus_optimized_ensemble.csv"
T11 = "outputs/07_multiclass_results/T11_pascal_a_multiclass_results.csv"
T12 = "outputs/07_multiclass_results/T12_pascal_b_multiclass_results.csv"
T13 = "outputs/08_circor_external_validation/T13_circor_murmur_classification_results.csv"
T14 = "outputs/08_circor_external_validation/T14_circor_clinical_outcome_results.csv"
T16 = "outputs/08_circor_external_validation/T16_cross_dataset_generalization.csv"
T17 = "outputs/09_ablation/T17_feature_family_ablation.csv"
T18 = "outputs/09_ablation/T18_all_features_versus_selected_features.csv"
T19 = "outputs/09_ablation/T19_search_and_optimization_ablation.csv"
TS5 = "outputs/09_ablation/T-S5_leave_one_source_out_generalization.csv"
T20 = "outputs/09_robustness_analysis/T20_noise_robustness.csv"
T21 = "outputs/09_robustness_analysis/T21_duration_robustness.csv"
T24 = "outputs/11_complexity/T24_complexity_analysis.csv"
T28 = "outputs/12_statistics/T28_statistical_significance_comparison.csv"
SELECTION = "outputs/06_binary_results/final_model_selection.csv"
FIG = "outputs/13_figures_diagrams/"
G12 = FIG + "G12_binary_roc_curve.csv"
G13 = FIG + "G13_binary_precision_recall_curve.csv"
G14 = FIG + "G14_binary_confusion_matrix.csv"
G15 = FIG + "G15_pascal_a_confusion_matrix.csv"
G16 = FIG + "G16_pascal_b_confusion_matrix.csv"
G24 = FIG + "G24_baseline_versus_optimized.csv"
G25 = FIG + "G25_inference_time_versus_performance.csv"
G29 = FIG + "G29_cross_dataset_performance_drop.csv"
ALG = "outputs/14_algorithms/"


def _root() -> Path:
    return PROJECT_ROOT


def read_source(path: str) -> Any:
    """Read a source CSV the one way every block and every check reads it.

    ``keep_default_na=False`` with only the empty field as missing: T07 holds
    the string ``None`` as a chosen value, and pandas' default reads it as NaN
    (the Quick Reference trap). A blank stays missing; a word stays a word.
    """
    import pandas as pd

    return pd.read_csv(_root() / path, keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """One source CSV, filtered and renamed. Selection only: no arithmetic."""

    source: str
    #: (source column, output column), in output order.
    columns: tuple[tuple[str, str], ...]
    #: (column, allowed values); compared as strings so "True" matches True.
    filters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    #: (output column, value) for every row of this block.
    constants: tuple[tuple[str, str], ...] = ()
    #: (source column, metric label): one output row per label, in ``metric``/``value``.
    melt: tuple[tuple[str, str], ...] = ()


def _select(block: Block) -> Any:
    frame = read_source(block.source)
    for column, allowed in block.filters:
        frame = frame[frame[column].astype(str).isin(allowed)]
    return frame.reset_index(drop=True)


def build_frame(blocks: tuple[Block, ...] | list[Block]) -> Any:
    """Concatenate the blocks; every row names its source file."""
    import pandas as pd

    parts = []
    for block in blocks:
        selected = _select(block)
        if selected.empty:
            raise ValueError(
                block.source + ": the block's filters select no rows " + str(block.filters)
            )
        base = selected[[source for source, _ in block.columns]].rename(columns=dict(block.columns))
        if block.melt:
            melted = []
            for source, label in block.melt:
                part = base.copy()
                part["metric"] = label
                part["value"] = selected[source].to_numpy()
                melted.append(part)
            base = pd.concat(melted, ignore_index=True)
        for column, value in block.constants:
            base.insert(0, column, value)
        base["source_file"] = block.source
        parts.append(base)
    return pd.concat(parts, ignore_index=True)


def _final_model() -> str:
    selection = read_source(SELECTION).sort_values("rank")
    return str(selection.iloc[0]["model_id"])


def _top_model(path: str) -> str:
    return str(read_source(path).iloc[0]["model_id"])


# ---------------------------------------------------------------------------
# tables (T103.1-T103.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Q1Table:
    table_id: str
    title: str
    caption: str
    blocks: tuple[Block, ...]
    #: (output column, header, kind, places); kinds are the table engine's.
    columns: tuple[tuple[str, str, str | None, int | None], ...]
    exp_id: str = ""
    objective: str = ""
    dataset: str = ""
    notes: tuple[str, ...] = ()


def q1_tables() -> tuple[Q1Table, ...]:
    final = _final_model()
    within = (
        "Within the PhysioNet 2016 corpus (subject-grouped cross-validation); "
        "not a generalization claim."
    )
    return (
        Q1Table(
            "Q1_T01",
            "Dataset Summary",
            "The four corpora, the recordings each contributes and the task it carries. "
            "Label spaces are never merged.",
            (
                Block(
                    T01,
                    (
                        ("dataset_source", "dataset"),
                        ("dataset_name", "dataset_name"),
                        ("usable_files", "recordings_used"),
                        ("total_files", "files_in_folder"),
                        ("n_subjects", "subjects"),
                        ("subject_id_origin", "subject_id_origin"),
                        ("n_classes", "classes"),
                        ("original_fs", "native_fs_hz"),
                        ("target_fs", "analysis_fs_hz"),
                        ("total_hours", "total_hours"),
                        ("role", "role"),
                    ),
                ),
            ),
            (
                ("dataset", "ID", "text", None),
                ("dataset_name", "Dataset", "text", None),
                ("recordings_used", "Recordings used", "count", None),
                ("files_in_folder", "Files in folder", "count", None),
                ("subjects", "Subjects", "count", None),
                ("subject_id_origin", "Subject IDs", "text", None),
                ("classes", "Classes", "count", None),
                ("native_fs_hz", "Native fs (Hz)", "count", None),
                ("analysis_fs_hz", "Analysis fs (Hz)", "count", None),
                ("total_hours", "Hours", "seconds", None),
                ("role", "Role", "text", None),
                ("source_file", "Source", "text", None),
            ),
            objective="O1, O6",
            dataset="D1-D4",
            notes=(
                "'Files in folder' exceeds 'recordings used' where material is unlabelled or, "
                "for PhysioNet, where the validation folder duplicates training records.",
            ),
        ),
        Q1Table(
            "Q1_T02",
            "Feature Summary",
            "The six feature families of the 138-feature vector, counted from the "
            "feature registry.",
            (
                Block(
                    T05,
                    (
                        ("family", "family"),
                        ("extractor", "extractor"),
                        ("counted_features", "n_features"),
                        ("first_index", "first_index"),
                        ("matches_expected", "matches_locked_count"),
                    ),
                ),
            ),
            (
                ("family", "Family", "text", None),
                ("extractor", "Extractor", "text", None),
                ("n_features", "Features", "count", None),
                ("first_index", "First column", "count", None),
                ("matches_locked_count", "Matches locked count", "text", None),
                ("source_file", "Source", "text", None),
            ),
            objective="O4",
            notes=(
                "138 features are computed; seven carry no unique information (five are pinned by "
                "z-normalization, two are exact duplicates), so 131 are unique.",
            ),
        ),
        Q1Table(
            "Q1_T03",
            "Model and Search Configuration",
            "Searched hyperparameters, their ranges, and the value the final configuration uses.",
            (
                Block(
                    T07,
                    (
                        ("model_id", "model_id"),
                        ("parameter", "parameter"),
                        ("distribution", "distribution"),
                        ("range_or_choices", "range_or_choices"),
                        ("final_selected", "final_selected"),
                        ("final_source", "final_source"),
                    ),
                ),
            ),
            (
                ("model_id", "Model", "text", None),
                ("parameter", "Parameter", "text", None),
                ("distribution", "Distribution", "text", None),
                ("range_or_choices", "Range or choices", "text", None),
                ("final_selected", "Selected", "text", None),
                ("final_source", "Chosen by", "text", None),
                ("source_file", "Source", "text", None),
            ),
            exp_id="SO-01, SO-02",
            objective="O5",
            dataset="D1 PhysioNet 2016",
            notes=(
                "SO-01 and SO-02 search one outer fold (r0f0); EXP-A2's nested search re-tunes on "
                "every fold at a reduced budget. 'None' is a chosen value, not a missing one.",
            ),
        ),
        Q1Table(
            "Q1_T04",
            "Individual versus Ensemble",
            "Every model under the nested search on PhysioNet 2016, ranked by sensitivity then "
            "balanced accuracy. " + within,
            (
                Block(
                    T08,
                    (
                        ("rank", "rank"),
                        ("model_id", "model_id"),
                        ("model_name", "model_name"),
                        ("n_folds", "n_folds"),
                        ("sensitivity_mean", "sensitivity_mean"),
                        ("sensitivity_sd", "sensitivity_sd"),
                        ("specificity_mean", "specificity_mean"),
                        ("balanced_accuracy_mean", "balanced_accuracy_mean"),
                        ("balanced_accuracy_sd", "balanced_accuracy_sd"),
                        ("f1_mean", "f1_mean"),
                        ("roc_auc_mean", "roc_auc_mean"),
                        ("accuracy_mean", "accuracy_mean"),
                    ),
                    filters=(("run", ("EXP-A2",)),),
                ),
            ),
            (
                ("rank", "Rank", "count", None),
                ("model_id", "Model", "text", None),
                ("model_name", "Name", "text", None),
                ("n_folds", "Folds", "count", None),
                ("sensitivity_mean", "Sensitivity", "metric", None),
                ("sensitivity_sd", "Sens. SD", "metric", None),
                ("specificity_mean", "Specificity", "metric", None),
                ("balanced_accuracy_mean", "Balanced acc.", "metric", None),
                ("balanced_accuracy_sd", "Bal. acc. SD", "metric", None),
                ("f1_mean", "F1", "metric", None),
                ("roc_auc_mean", "ROC-AUC", "metric", None),
                ("accuracy_mean", "Accuracy", "metric", None),
                ("source_file", "Source", "text", None),
            ),
            exp_id="EXP-A2",
            objective="O1, O3",
            dataset="D1 PhysioNet 2016",
            notes=(
                "The final model "
                + final
                + " and the equal-weight ensemble M6 are statistically tied "
                "on sensitivity; " + final + " is the simpler model and is what the pre-registered "
                "rule selects. Neither is described as outperforming the other.",
            ),
        ),
        Q1Table(
            "Q1_T05",
            "Feature Ablation",
            "Feature-family ablation for the final model "
            + final
            + " (A1-A8), same folds and seed. "
            + within,
            (
                Block(
                    T17,
                    (
                        ("config_id", "config_id"),
                        ("families", "families"),
                        ("n_features", "n_features"),
                        ("sensitivity", "sensitivity"),
                        ("specificity", "specificity"),
                        ("balanced_accuracy", "balanced_accuracy"),
                        ("balanced_accuracy_sd", "balanced_accuracy_sd"),
                        ("balanced_accuracy_delta_vs_A7", "balanced_accuracy_delta_vs_A7"),
                        ("roc_auc", "roc_auc"),
                    ),
                    filters=(("model_id", (final,)),),
                ),
            ),
            (
                ("config_id", "Config", "text", None),
                ("families", "Families", "text", None),
                ("n_features", "Features", "count", None),
                ("sensitivity", "Sensitivity", "metric", None),
                ("specificity", "Specificity", "metric", None),
                ("balanced_accuracy", "Balanced acc.", "metric", None),
                ("balanced_accuracy_sd", "SD", "metric", None),
                ("balanced_accuracy_delta_vs_A7", "Delta vs all 138", "metric", None),
                ("roc_auc", "ROC-AUC", "metric", None),
                ("source_file", "Source", "text", None),
            ),
            exp_id="EXP-F1",
            objective="O2, O5",
            dataset="D1 PhysioNet 2016",
        ),
        Q1Table(
            "Q1_T06",
            "Optimization Ablation",
            "Baseline, tuned and optimized-ensemble stages in pipeline order (A9, A10). " + within,
            (
                Block(
                    T19,
                    (
                        ("stage_order", "stage_order"),
                        ("stage", "stage"),
                        ("comparison", "comparison"),
                        ("source_run", "source_run"),
                        ("model_id", "model_id"),
                        ("sensitivity", "sensitivity"),
                        ("specificity", "specificity"),
                        ("balanced_accuracy", "balanced_accuracy"),
                        ("balanced_accuracy_incremental", "balanced_accuracy_incremental"),
                        ("roc_auc", "roc_auc"),
                    ),
                ),
            ),
            (
                ("stage_order", "Order", "count", None),
                ("stage", "Stage", "text", None),
                ("comparison", "Comparison", "text", None),
                ("source_run", "Run", "text", None),
                ("model_id", "Model", "text", None),
                ("sensitivity", "Sensitivity", "metric", None),
                ("specificity", "Specificity", "metric", None),
                ("balanced_accuracy", "Balanced acc.", "metric", None),
                ("balanced_accuracy_incremental", "Change vs previous", "metric", None),
                ("roc_auc", "ROC-AUC", "metric", None),
                ("source_file", "Source", "text", None),
            ),
            exp_id="EXP-F2",
            objective="O3, O5",
            dataset="D1 PhysioNet 2016",
            notes=(
                "The stages are the pipeline's order, not a ranking; fold SDs overlap "
                "between every pair.",
            ),
        ),
        Q1Table(
            "Q1_T07",
            "Multiclass Results",
            "PASCAL A (four-class) and PASCAL B (three-class), two separate tasks, "
            "ranked by macro-F1.",
            (
                Block(
                    T11,
                    (
                        ("model_id", "model_id"),
                        ("model_name", "model_name"),
                        ("n_folds", "n_folds"),
                        ("macro_f1_mean", "macro_f1_mean"),
                        ("macro_f1_sd", "macro_f1_sd"),
                        ("macro_f1_record_ci", "macro_f1_record_ci"),
                        ("balanced_accuracy_mean", "balanced_accuracy_mean"),
                        ("n_classes_predicted", "n_classes_predicted"),
                        ("degenerate", "degenerate"),
                    ),
                    constants=(("dataset", "PASCAL A (EXP-B1)"),),
                ),
                Block(
                    T12,
                    (
                        ("model_id", "model_id"),
                        ("model_name", "model_name"),
                        ("n_folds", "n_folds"),
                        ("macro_f1_mean", "macro_f1_mean"),
                        ("macro_f1_sd", "macro_f1_sd"),
                        ("macro_f1_record_ci", "macro_f1_record_ci"),
                        ("balanced_accuracy_mean", "balanced_accuracy_mean"),
                        ("n_classes_predicted", "n_classes_predicted"),
                        ("degenerate", "degenerate"),
                    ),
                    constants=(("dataset", "PASCAL B (EXP-B2)"),),
                ),
            ),
            (
                ("dataset", "Dataset", "text", None),
                ("model_id", "Model", "text", None),
                ("model_name", "Name", "text", None),
                ("n_folds", "Folds", "count", None),
                ("macro_f1_mean", "Macro-F1", "metric", None),
                ("macro_f1_sd", "SD", "metric", None),
                ("macro_f1_record_ci", "Record-level 95% CI", "text", None),
                ("balanced_accuracy_mean", "Balanced acc.", "metric", None),
                ("n_classes_predicted", "Classes predicted", "count", None),
                ("degenerate", "Degenerate", "text", None),
                ("source_file", "Source", "text", None),
            ),
            exp_id="EXP-B1, EXP-B2",
            objective="O6",
            dataset="D2 PASCAL A; D3 PASCAL B",
            notes=(
                "PASCAL A has 124 records with 19 in one class, and every model's "
                "record-level interval "
                "overlaps every other's: the ordering is not a distinguishable ranking.",
                "PASCAL's 'artifact' label is a recording-quality category, not a cardiac class.",
            ),
        ),
        Q1Table(
            "Q1_T08",
            "External Validation",
            "CirCor murmur and outcome (subject-wise), PhysioNet-to-CirCor transfer without "
            "retuning, and the final model on an unseen PhysioNet recording collection.",
            (
                Block(
                    T13,
                    (("model_id", "model_id"), ("level", "level"), ("rule", "rule")),
                    filters=(("run", ("EXP-C1-three_class",)), ("level", ("recording",))),
                    constants=(("evaluation", "CirCor murmur, three-class (EXP-C1)"),),
                    melt=(
                        ("balanced_accuracy_mean", "balanced_accuracy"),
                        ("macro_f1_mean", "macro_f1"),
                    ),
                ),
                Block(
                    T14,
                    (("model_id", "model_id"), ("level", "level"), ("rule", "rule")),
                    filters=(("level", ("recording",)),),
                    constants=(("evaluation", "CirCor clinical outcome (EXP-C2)"),),
                    melt=(
                        ("sensitivity_mean", "sensitivity"),
                        ("specificity_mean", "specificity"),
                        ("balanced_accuracy_mean", "balanced_accuracy"),
                        ("roc_auc_mean", "roc_auc"),
                    ),
                ),
                Block(
                    T16,
                    (
                        ("level", "level"),
                        ("rule", "rule"),
                        ("metric", "metric"),
                        ("external_value", "value"),
                    ),
                    filters=(
                        ("level", ("recording",)),
                        ("metric", ("sensitivity", "specificity", "balanced_accuracy", "roc_auc")),
                    ),
                    constants=(
                        ("model_id", final),
                        ("evaluation", "PhysioNet to CirCor, no retuning (EXP-D1)"),
                    ),
                ),
                Block(
                    TS5,
                    (("model_id", "model_id"),),
                    filters=(("model_id", (final,)),),
                    constants=(
                        ("rule", "none"),
                        ("level", "recording"),
                        ("evaluation", "Unseen PhysioNet recording collection (EXP-F3)"),
                    ),
                    melt=(
                        ("sensitivity_holdout", "sensitivity"),
                        ("specificity_holdout", "specificity"),
                        ("balanced_accuracy_holdout", "balanced_accuracy"),
                        ("roc_auc_holdout", "roc_auc"),
                    ),
                ),
            ),
            (
                ("evaluation", "Evaluation", "text", None),
                ("model_id", "Model", "text", None),
                ("level", "Level", "text", None),
                ("rule", "Aggregation", "text", None),
                ("metric", "Metric", "text", None),
                ("value", "Value", "metric", None),
                ("source_file", "Source", "text", None),
            ),
            exp_id="EXP-C1, EXP-C2, EXP-D1, EXP-F3",
            objective="O1, O3, O6",
            dataset="D4 CirCor 2022; D1 PhysioNet 2016",
            notes=(
                "EXP-D1 is adult-to-paediatric transfer across acquisition systems; its drop is a "
                "population and device effect, not evidence that the method fails in general.",
                "EXP-F3: holding out one PhysioNet recording collection takes ROC-AUC "
                "to chance level. "
                "The binary model does not transfer to an unseen recording setup, and no "
                "generalization or deployment claim is made.",
            ),
        ),
        Q1Table(
            "Q1_T09",
            "Complexity and Robustness",
            "Model footprint and speed (one fold fit), and the final model "
            + final
            + " across noise and duration conditions.",
            (
                Block(
                    T24,
                    (
                        ("model_id", "model_id"),
                        ("configuration", "condition"),
                        ("n_features", "n_features"),
                        ("fit_seconds", "fit_seconds"),
                        ("single_predict_seconds", "single_predict_seconds"),
                        ("model_mb", "model_mb"),
                        ("balanced_accuracy", "balanced_accuracy"),
                        ("sensitivity", "sensitivity"),
                    ),
                    constants=(("analysis", "complexity (fold r0f0 fit)"),),
                ),
                Block(
                    T20,
                    (
                        ("model_id", "model_id"),
                        ("group", "condition"),
                        ("balanced_accuracy_mean", "balanced_accuracy"),
                        ("sensitivity_mean", "sensitivity"),
                    ),
                    filters=(
                        ("model_id", (final,)),
                        ("run", ("EXP-A2", "EXP-E1-AWGN")),
                        ("reported", ("True",)),
                    ),
                    constants=(("analysis", "noise (EXP-E1)"),),
                ),
                Block(
                    T21,
                    (
                        ("model_id", "model_id"),
                        ("group", "condition"),
                        ("balanced_accuracy_mean", "balanced_accuracy"),
                        ("sensitivity_mean", "sensitivity"),
                    ),
                    filters=(
                        ("model_id", (final,)),
                        ("run", ("EXP-A2", "EXP-E2-TRUNC")),
                        ("reported", ("True",)),
                    ),
                    constants=(("analysis", "duration (EXP-E2)"),),
                ),
            ),
            (
                ("analysis", "Analysis", "text", None),
                ("model_id", "Model", "text", None),
                ("condition", "Condition", "text", None),
                ("n_features", "Features", "count", None),
                ("fit_seconds", "Fit (s)", "seconds", None),
                ("single_predict_seconds", "One prediction (s)", "seconds", 4),
                ("model_mb", "Size (MB)", "metric", None),
                ("balanced_accuracy", "Balanced acc.", "metric", None),
                ("sensitivity", "Sensitivity", "metric", None),
                ("source_file", "Source", "text", None),
            ),
            exp_id="EXP-A2, EXP-E1, EXP-E2",
            objective="O4, O5",
            dataset="D1 PhysioNet 2016",
            notes=(
                "Complexity rows are one fold's fit on this CPU-only machine; an ensemble's wall "
                "clock varies with what its member search picked on that fold.",
                "Observational noise and duration groups are not controlled comparisons; the AWGN "
                "and truncation rows are.",
            ),
        ),
    )


def _table_spec(table: Q1Table) -> Any:
    from src.reporting.tables import Column, TableSpec

    return TableSpec(
        table_id=table.table_id,
        title=table.title,
        caption=table.caption,
        sources=tuple(dict.fromkeys(block.source for block in table.blocks)),
        columns=tuple(
            Column(name, header, kind=kind, places=places)
            for name, header, kind, places in table.columns
        ),
        exp_id=table.exp_id,
        objective=table.objective,
        dataset=table.dataset,
        notes=(*table.notes, DISCLAIMER),
        command=COMMAND,
    )


# ---------------------------------------------------------------------------
# graphs (T103.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Q1Graph:
    figure_id: str
    title: str
    caption: str
    blocks: tuple[Block, ...]
    draw: Callable[[Any], Any] = field(compare=False)
    exp_id: str = ""
    objective: str = ""
    dataset: str = ""
    size: str = "double"


def _palette(n: int) -> list[str]:
    from src.reporting.graphs import OKABE_ITO

    colours = list(OKABE_ITO)
    return [colours[i % len(colours)] for i in range(n)]


def _draw_roc_pr(frame: Any) -> Any:
    from src.reporting.graphs import subplots

    figure, axes = subplots("double", ncols=2)
    for axis, panel, xlabel, ylabel in (
        (axes[0], "ROC", "False positive rate", "True positive rate"),
        (axes[1], "PR", "Recall", "Precision"),
    ):
        block = frame[frame["panel"] == panel]
        models = list(dict.fromkeys(block["model_id"]))
        for colour, model in zip(_palette(len(models)), models, strict=True):
            curve = block[block["model_id"] == model]
            label = model + " (" + format(float(curve["auc"].iloc[0]), ".3f") + ")"
            axis.plot(curve["x"], curve["mean"], color=colour, label=label)
        if panel == "ROC":
            axis.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(
            ("ROC, AUC" if panel == "ROC" else "Precision-recall, AP") + " in legend", fontsize=8
        )
        axis.legend(fontsize=6, loc="lower right" if panel == "ROC" else "lower left")
    return figure


def _confusion(axis: Any, block: Any, title: str) -> None:
    import numpy as np

    labels = list(dict.fromkeys(block.sort_values("true_index")["true_class"]))
    size = len(labels)
    share = np.zeros((size, size))
    counts = np.zeros((size, size), dtype=int)
    for row in block.itertuples(index=False):
        share[int(row.true_index), int(row.pred_index)] = float(row.row_share_all_folds)
        counts[int(row.true_index), int(row.pred_index)] = int(row.count_all_folds)
    axis.imshow(share, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(size):
        for j in range(size):
            axis.text(
                j,
                i,
                format(share[i, j], ".2f") + "\n" + format(counts[i, j], ",d"),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if share[i, j] > 0.6 else "black",
            )
    axis.set_xticks(range(size), labels, rotation=30, fontsize=7)
    axis.set_yticks(range(size), labels, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title, fontsize=8)


def _draw_binary_confusion(frame: Any) -> Any:
    from src.reporting.graphs import subplots

    figure, axis = subplots("single")
    model = str(frame["model_id"].iloc[0])
    _confusion(axis, frame, model + ", all folds (row share, count)")
    return figure


def _draw_multiclass_confusion(frame: Any) -> Any:
    from src.reporting.graphs import subplots

    figure, axes = subplots("double", ncols=2)
    for axis, panel in zip(axes, list(dict.fromkeys(frame["panel"])), strict=True):
        block = frame[frame["panel"] == panel]
        _confusion(axis, block, panel + ", " + str(block["model_id"].iloc[0]))
    return figure


def _draw_ablation(frame: Any) -> Any:
    from src.reporting.graphs import subplots

    figure, axes = subplots("double", ncols=2)
    for axis, panel in zip(axes, list(dict.fromkeys(frame["panel"])), strict=True):
        block = frame[frame["panel"] == panel].reset_index(drop=True)
        positions = list(range(len(block)))
        axis.barh(positions, block["value"], xerr=block["sd"], color=_palette(1)[0], alpha=0.85)
        axis.set_yticks(positions, block["label"], fontsize=6)
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Balanced accuracy (mean, SD error bar)")
        axis.set_title(panel, fontsize=8)
    return figure


def _draw_generalization(frame: Any) -> Any:
    import numpy as np

    from src.reporting.graphs import subplots

    figure, axes = subplots("double", ncols=2)
    drop = frame[frame["panel"] == "cross-dataset"].reset_index(drop=True)
    positions = np.arange(len(drop))
    axes[0].bar(
        positions - 0.2, drop["in_domain_mean"], width=0.4, label="within PhysioNet (EXP-A2)"
    )
    axes[0].bar(positions + 0.2, drop["external_value"], width=0.4, label="on CirCor (EXP-D1)")
    axes[0].set_xticks(positions, drop["metric"], fontsize=6, rotation=30, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=6)
    axes[0].set_title("Transfer without retuning", fontsize=8)

    cost = frame[frame["panel"] == "complexity"].reset_index(drop=True)
    axes[1].scatter(cost["single_predict_seconds"], cost["balanced_accuracy"], color=_palette(2)[1])
    ordered = cost.sort_values("single_predict_seconds").itertuples(index=False)
    for index, row in enumerate(ordered):
        axes[1].annotate(
            str(row.model_id),
            (row.single_predict_seconds, row.balanced_accuracy),
            xytext=(4, 6 if index % 2 == 0 else -9),
            textcoords="offset points",
            fontsize=6,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("One prediction (s, log scale)")
    axes[1].set_ylabel("Balanced accuracy (EXP-A2)")
    axes[1].set_title("Inference time versus performance", fontsize=8)
    return figure


_CM_COLUMNS = (
    ("run", "run"),
    ("model_id", "model_id"),
    ("true_index", "true_index"),
    ("true_class", "true_class"),
    ("pred_index", "pred_index"),
    ("pred_class", "pred_class"),
    ("count_all_folds", "count_all_folds"),
    ("row_share_all_folds", "row_share_all_folds"),
    ("n_folds_all", "n_folds_all"),
)


def q1_graphs() -> tuple[Q1Graph, ...]:
    final = _final_model()
    pascal_b = _top_model(T12)
    curve = (
        ("run", "run"),
        ("model_id", "model_id"),
        ("model_name", "model_name"),
        ("x", "x"),
        ("mean", "mean"),
        ("sd", "sd"),
        ("n_folds", "n_folds"),
    )
    return (
        Q1Graph(
            "Q1_G01",
            "ROC and PR Combined",
            "Mean ROC and precision-recall curves over the EXP-A2 folds for every model, within "
            "the PhysioNet 2016 corpus. Legend values are mean ROC-AUC and average precision.",
            (
                Block(
                    G12,
                    (*curve, ("roc_auc_mean", "auc")),
                    filters=(("run", ("EXP-A2",)),),
                    constants=(("panel", "ROC"),),
                ),
                Block(
                    G13,
                    (*curve, ("pr_auc_mean", "auc")),
                    filters=(("run", ("EXP-A2",)),),
                    constants=(("panel", "PR"),),
                ),
            ),
            _draw_roc_pr,
            exp_id="EXP-A2",
            objective="O1",
            dataset="D1 PhysioNet 2016",
        ),
        Q1Graph(
            "Q1_G02",
            "Binary Confusion Matrix",
            "Confusion matrix of the final model "
            + final
            + " over all EXP-A2 folds; each cell shows "
            "the row share and the count.",
            (Block(G14, _CM_COLUMNS, filters=(("model_id", (final,)),)),),
            _draw_binary_confusion,
            exp_id="EXP-A2",
            objective="O1",
            dataset="D1 PhysioNet 2016",
            size="single",
        ),
        Q1Graph(
            "Q1_G03",
            "Multiclass Confusion Matrix",
            "Confusion matrices of the top model by macro-F1 on PASCAL A and PASCAL B, "
            "over all folds. "
            "Two separate tasks; 'artifact' is a recording-quality label.",
            (
                Block(G15, _CM_COLUMNS, constants=(("panel", "PASCAL A"),)),
                Block(
                    G16,
                    _CM_COLUMNS,
                    filters=(("model_id", (pascal_b,)),),
                    constants=(("panel", "PASCAL B"),),
                ),
            ),
            _draw_multiclass_confusion,
            exp_id="EXP-B1, EXP-B2",
            objective="O6",
            dataset="D2 PASCAL A; D3 PASCAL B",
        ),
        Q1Graph(
            "Q1_G04",
            "Ablation and Optimization",
            "Left: feature-family ablation for "
            + final
            + ". Right: the optimization stages in pipeline "
            "order. Error bars are fold SDs and overlap throughout; neither panel is a ranking.",
            (
                Block(
                    T17,
                    (
                        ("config_id", "label"),
                        ("n_features", "n_features"),
                        ("balanced_accuracy", "value"),
                        ("balanced_accuracy_sd", "sd"),
                        ("model_id", "model_id"),
                    ),
                    filters=(("model_id", (final,)),),
                    constants=(("panel", "Feature-family ablation (EXP-F1)"),),
                ),
                Block(
                    G24,
                    (
                        ("stage", "label"),
                        ("balanced_accuracy", "value"),
                        ("balanced_accuracy_sd", "sd"),
                        ("model_id", "model_id"),
                    ),
                    constants=(("panel", "Optimization stages (EXP-F2)"),),
                ),
            ),
            _draw_ablation,
            exp_id="EXP-F1, EXP-F2",
            objective="O3, O5",
            dataset="D1 PhysioNet 2016",
        ),
        Q1Graph(
            "Q1_G05",
            "Generalization and Complexity",
            "Left: the final model within PhysioNet and on CirCor without retuning (EXP-D1). "
            "Right: one-prediction time against balanced accuracy for every model.",
            (
                Block(
                    G29,
                    (
                        ("metric", "metric"),
                        ("level", "level"),
                        ("rule", "rule"),
                        ("in_domain_mean", "in_domain_mean"),
                        ("external_value", "external_value"),
                    ),
                    constants=(("panel", "cross-dataset"),),
                ),
                Block(
                    G25,
                    (
                        ("model_id", "model_id"),
                        ("single_predict_seconds", "single_predict_seconds"),
                        ("balanced_accuracy", "balanced_accuracy"),
                        ("model_mb", "model_mb"),
                    ),
                    constants=(("panel", "complexity"),),
                ),
            ),
            _draw_generalization,
            exp_id="EXP-A2, EXP-D1",
            objective="O1, O5",
            dataset="D1 PhysioNet 2016; D4 CirCor 2022",
        ),
    )


# ---------------------------------------------------------------------------
# figures and algorithms (T103.4, T103.6)
# ---------------------------------------------------------------------------

#: Q1 id -> (published stem, source stems). One source is a copy; two are stacked.
Q1_FIGURES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Q1_F01": ("Q1_F01_proposed_architecture", ("F01_overall_pv_mepcg_proposed_architecture",)),
    "Q1_F02": (
        "Q1_F02_feature_and_ensemble_workflow",
        ("F08_feature_family_fusion", "F10_svm_rf_gb_optimized_soft_voting_architecture"),
    ),
}

Q1_ALGORITHMS: dict[str, tuple[str, str]] = {
    "Q1_ALG01": ("Q1_ALG01_feature_extraction", "ALG-04_multi_domain_138_feature_extraction"),
    "Q1_ALG02": ("Q1_ALG02_search_optimization", "ALG-07_hyperparameter_optimization"),
    "Q1_ALG03": ("Q1_ALG03_proposed_ensemble", "ALG-12_optimized_weight_soft_voting"),
}


def _svg_size(text: str) -> tuple[float, float, str]:
    root = re.search(r"<svg\b[^>]*>", text)
    if root is None:
        raise ValueError("no <svg> root")
    width = re.search(r'\bwidth="([\d.]+)([a-z]*)"', root.group(0))
    height = re.search(r'\bheight="([\d.]+)([a-z]*)"', root.group(0))
    if width is None or height is None:
        raise ValueError("svg root has no width/height")
    return float(width.group(1)), float(height.group(1)), width.group(2)


def _stack_svg(sources: list[Path], target: Path, gap: float = 12.0) -> Path:
    """Stack SVGs vertically as nested <svg> elements: still vector, still editable."""
    parts: list[str] = []
    offset = 0.0
    widths: list[float] = []
    unit = ""
    for source in sources:
        text = source.read_text(encoding="utf-8")
        width, height, unit = _svg_size(text)
        body = re.sub(r"<\?xml[^>]*\?>", "", text)
        body = re.sub(r"<!DOCTYPE[^>]*>", "", body)
        body = re.sub(r"<svg\b", '<svg x="0" y="' + format(offset, "g") + unit + '"', body, count=1)
        parts.append(body.strip())
        widths.append(width)
        offset += height + gap
    total = offset - gap
    outer = (
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'version="1.1" width="'
        + format(max(widths), "g")
        + unit
        + '" height="'
        + format(total, "g")
        + unit
        + '">\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )
    target.write_text(outer, encoding="utf-8")
    return target


def _stack_png(sources: list[Path], target: Path, gap: int = 36) -> Path:
    from PIL import Image

    images = [Image.open(source).convert("RGB") for source in sources]
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    top = 0
    for image in images:
        canvas.paste(image, ((width - image.width) // 2, top))
        top += image.height + gap
    canvas.save(target, dpi=(300, 300))
    return target


def _copy_figures(target: Path, evidence_index: str | Path | None) -> dict[str, list[Path]]:
    written: dict[str, list[Path]] = {}
    for q1_id, (stem, sources) in Q1_FIGURES.items():
        paths = [_root() / FIG / (source + ".png") for source in sources]
        svgs = [_root() / FIG / (source + ".svg") for source in sources]
        png, svg = target / (stem + ".png"), target / (stem + ".svg")
        if len(sources) == 1:
            shutil.copyfile(paths[0], png)
            shutil.copyfile(svgs[0], svg)
        else:
            _stack_png(paths, png)
            _stack_svg(svgs, svg)
        register_evidence(
            q1_id,
            png,
            metric_or_asset=stem.split("_", 2)[-1].replace("_", " "),
            objective="O1, O3",
            source_data="; ".join(FIG + source + ".svg" for source in sources),
            command=COMMAND,
            index_path=evidence_index,
        )
        written[q1_id] = [png, svg]
    return written


def _copy_algorithms(target: Path, evidence_index: str | Path | None) -> dict[str, list[Path]]:
    written: dict[str, list[Path]] = {}
    for q1_id, (stem, source) in Q1_ALGORITHMS.items():
        files = []
        for suffix in (".docx", ".txt"):
            destination = target / (stem + suffix)
            shutil.copyfile(_root() / ALG / (source + suffix), destination)
            files.append(destination)
        register_evidence(
            q1_id,
            files[0],
            metric_or_asset=source.split("_", 1)[0]
            + " as "
            + stem.split("_", 2)[-1].replace("_", " "),
            objective="O3, O4, O5",
            source_data=ALG + source + ".txt",
            command=COMMAND,
            index_path=evidence_index,
        )
        written[q1_id] = files
    return written


# ---------------------------------------------------------------------------
# the narrative (T103.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """One number (or name) in the narrative: a CSV, a row, a column."""

    fact_id: str
    source: str
    column: str
    filters: tuple[tuple[str, str], ...] = ()
    #: "value" (exactly one row), "first" (first matching row), "sum", "count".
    op: str = "value"
    kind: str = "metric"
    places: int = 3

    def raw(self) -> Any:
        frame = read_source(self.source)
        for column, value in self.filters:
            frame = frame[frame[column].astype(str) == value]
        if self.op == "count":
            return len(frame)
        if self.op == "sum":
            return frame[self.column].sum()
        if frame.empty:
            raise ValueError(
                self.fact_id + ": no row in " + self.source + " for " + str(self.filters)
            )
        if self.op == "value" and len(frame) != 1:
            raise ValueError(
                self.fact_id + ": " + str(len(frame)) + " rows match " + str(self.filters)
            )
        return frame.iloc[0][self.column]

    def written(self) -> str:
        value = self.raw()
        if self.kind == "text":
            return str(value)
        if self.kind == "count":
            return format(round(float(value)), ",d")
        return format(float(value), "." + str(self.places) + "f")

    def selector(self) -> str:
        where = ", ".join(column + "=" + value for column, value in self.filters) or "all rows"
        return self.op + " of " + self.column + " where " + where


def narrative_facts() -> dict[str, Fact]:
    final = _final_model()
    facts: list[Fact] = [Fact("final_model", SELECTION, "model_id", op="first", kind="text")]

    def add(fact_id: str, source: str, column: str, **kwargs: Any) -> None:
        facts.append(Fact(fact_id, source, column, **kwargs))

    for dataset in ("D1", "D2", "D3", "D4"):
        where: tuple[tuple[str, str], ...] = (("dataset_source", dataset),)
        add(dataset + "_name", T01, "dataset_name", filters=where, kind="text")
        add(dataset + "_records", T01, "usable_files", filters=where, kind="count")
        add(dataset + "_subjects", T01, "n_subjects", filters=where, kind="count")
    add(
        "D1_abnormal_share",
        T02,
        "share",
        filters=(("dataset_source", "D1"), ("task", "binary"), ("class", "abnormal")),
    )
    add("n_features", T05, "counted_features", op="sum", kind="count")
    add("n_families", T05, "family", op="count", kind="count")

    binary = (("run", "EXP-A2"),)
    for model, prefix in ((final, "final"), ("M6", "m6")):
        where = (*binary, ("model_id", model))
        add(prefix + "_n_folds", T08, "n_folds", filters=where, kind="count")
        for column in (
            "sensitivity_mean",
            "sensitivity_sd",
            "specificity_mean",
            "balanced_accuracy_mean",
            "roc_auc_mean",
        ):
            add(prefix + "_" + column, T08, column, filters=where)
    pair = (("run", "EXP-A2"), ("metric", "sensitivity"), ("model_a", final), ("model_b", "M6"))
    add("tie_test", T28, "chosen_test", filters=pair, kind="text")
    add("tie_n", T28, "n", filters=pair, kind="count")
    add("tie_p", T28, "p_value", filters=pair)

    weights = (("metric", "balanced_accuracy"),)
    add("m7_delta", T09, "delta_mean", filters=weights, places=4)
    add("m7_identical", T09, "n_folds_identical", filters=weights, kind="count")
    add("m7_folds", T09, "n_folds", filters=weights, kind="count")

    subset = (("model_id", final),)
    add("subset_selected", T18, "n_features_selected", filters=subset, kind="count")
    add("subset_all", T18, "n_features_all", filters=subset, kind="count")
    add("subset_bal_selected", T18, "balanced_accuracy_selected", filters=subset)
    add("subset_bal_all", T18, "balanced_accuracy_all", filters=subset)

    for stage in read_source(T19)["stage"].astype(str):
        where = (("stage", stage),)
        add("stage_" + stage + "_label", T19, "comparison", filters=where, kind="text")
        add("stage_" + stage + "_bal", T19, "balanced_accuracy", filters=where)

    for source, prefix in ((T11, "pascal_a"), (T12, "pascal_b")):
        add(prefix + "_model", source, "model_id", op="first", kind="text")
        add(prefix + "_macro_f1", source, "macro_f1_mean", op="first")
        add(prefix + "_ci", source, "macro_f1_record_ci", op="first", kind="text")

    transfer = (("metric", "balanced_accuracy"), ("level", "recording"), ("rule", "none"))
    add("d1_in_domain", T16, "in_domain_mean", filters=transfer)
    add("d1_external", T16, "external_value", filters=transfer)
    add("f3_auc_pooled", TS5, "roc_auc_pooled", filters=subset)
    add("f3_auc_holdout", TS5, "roc_auc_holdout", filters=subset)
    add("f3_collections", TS5, "n_folds_holdout", filters=subset, kind="count")

    for model, prefix in ((final, "final"), ("M6", "m6")):
        where = (("model_id", model),)
        add(prefix + "_mb", T24, "model_mb", filters=where)
        add(prefix + "_predict_s", T24, "single_predict_seconds", filters=where, places=4)

    for source, analysis, group, fact_id in (
        (T20, "observational_pp08", "clean", "noise_clean"),
        (T20, "observational_pp08", "noisy", "noise_noisy"),
        (T21, "observational_bands", "long", "duration_long"),
        (T21, "observational_bands", "medium", "duration_medium"),
    ):
        where = (("analysis", analysis), ("run", "EXP-A2"), ("model_id", final), ("group", group))
        add(fact_id, source, "balanced_accuracy_mean", filters=where)
    return {fact.fact_id: fact for fact in facts}


def _narrative_paragraphs(values: dict[str, str]) -> list[tuple[str, str]]:
    """(style, text) pairs. Every digit in the text arrives through ``values``."""
    v = values
    final = v["final_model"]
    stages = [
        key[len("stage_") : -len("_label")]
        for key in v
        if key.startswith("stage_") and key.endswith("_label")
    ]
    stage_text = "; ".join(
        v["stage_" + s + "_label"] + " " + v["stage_" + s + "_bal"] for s in stages
    )
    return [
        ("Title", "PV-MEPCG / PulseVision: results narrative for the Q1 / IEEE paper"),
        (
            "Normal",
            "Every number in this document is read from a generated result file when the document "
            "is built. The file, row and column behind each one are listed in the table at the end "
            "and in Q1_results_narrative_sources.csv. " + DISCLAIMER,
        ),
        ("Heading 1", "Datasets and representation"),
        (
            "Normal",
            "The binary track uses "
            + v["D1_name"]
            + ": "
            + v["D1_records"]
            + " labelled recordings from "
            + v["D1_subjects"]
            + " derived subjects, of which a share of "
            + v["D1_abnormal_share"]
            + " is abnormal. "
            + v["D2_name"]
            + " ("
            + v["D2_records"]
            + " recordings) and "
            + v["D3_name"]
            + " ("
            + v["D3_records"]
            + " recordings, "
            + v["D3_subjects"]
            + " subjects) carry two separate "
            "multiclass tasks, and "
            + v["D4_name"]
            + " ("
            + v["D4_records"]
            + " recordings from "
            + v["D4_subjects"]
            + " patients) carries subject-wise murmur and outcome validation. Each "
            "recording is represented by "
            + v["n_features"]
            + " engineered features in "
            + v["n_families"]
            + " families (Table Q1_T02).",
        ),
        ("Heading 1", "Binary screening within the PhysioNet corpus"),
        (
            "Normal",
            "Under the nested search (EXP-A2, "
            + v["final_n_folds"]
            + " subject-grouped folds), the final "
            "model "
            + final
            + " reached sensitivity "
            + v["final_sensitivity_mean"]
            + " (SD "
            + v["final_sensitivity_sd"]
            + "), specificity "
            + v["final_specificity_mean"]
            + ", balanced "
            "accuracy "
            + v["final_balanced_accuracy_mean"]
            + " and ROC-AUC "
            + v["final_roc_auc_mean"]
            + ". The equal-weight ensemble M6 reached sensitivity "
            + v["m6_sensitivity_mean"]
            + " and "
            "balanced accuracy "
            + v["m6_balanced_accuracy_mean"]
            + ". The two are not distinguishable on "
            "sensitivity ("
            + v["tie_test"]
            + ", n = "
            + v["tie_n"]
            + ", p = "
            + v["tie_p"]
            + "); "
            + final
            + " is the simpler model and is the one the pre-registered selection rule "
            "returns, not a model "
            "that outperformed the ensemble (Table Q1_T04, Figures Q1_G01 and Q1_G02).",
        ),
        (
            "Normal",
            "Optimizing the ensemble weights (M7) changed mean balanced accuracy by "
            + v["m7_delta"]
            + " against equal weights, with "
            + v["m7_identical"]
            + " of "
            + v["m7_folds"]
            + " folds "
            "returning identical results: equal weighting is the evidenced default. The "
            "search-selected "
            "feature subset keeps "
            + v["subset_selected"]
            + " of "
            + v["subset_all"]
            + " features; for "
            + final
            + " its balanced accuracy is "
            + v["subset_bal_selected"]
            + " against "
            + v["subset_bal_all"]
            + " with all features.",
        ),
        ("Heading 1", "Ablation and optimization"),
        (
            "Normal",
            "Balanced accuracy along the optimization stages, in pipeline order (Table "
            "Q1_T06, Figure "
            "Q1_G04): "
            + stage_text
            + ". The fold standard deviations overlap between every pair of "
            "stages, so the sequence is not read as a ranking.",
        ),
        ("Heading 1", "Multiclass tasks"),
        (
            "Normal",
            "On PASCAL A the highest macro-F1 is "
            + v["pascal_a_model"]
            + " at "
            + v["pascal_a_macro_f1"]
            + " (record-level interval "
            + v["pascal_a_ci"]
            + "); on PASCAL B it is "
            + v["pascal_b_model"]
            + " at "
            + v["pascal_b_macro_f1"]
            + " ("
            + v["pascal_b_ci"]
            + "). The intervals overlap across "
            "models, and PASCAL A's artifact label is a recording-quality category, not "
            "a cardiac class "
            "(Table Q1_T07, Figure Q1_G03).",
        ),
        ("Heading 1", "External validation and its limits"),
        (
            "Normal",
            "Trained on PhysioNet and applied to CirCor without retuning (EXP-D1), recording-level "
            "balanced accuracy moves from "
            + v["d1_in_domain"]
            + " within PhysioNet to "
            + v["d1_external"]
            + ": adult-to-paediatric transfer across acquisition systems. Holding out "
            "one PhysioNet "
            "recording collection at a time (EXP-F3, "
            + v["f3_collections"]
            + " collections), the ROC-AUC "
            "of "
            + final
            + " falls from "
            + v["f3_auc_pooled"]
            + " pooled to "
            + v["f3_auc_holdout"]
            + ". The model does not transfer to an unseen recording setup; every result above is a "
            "within-corpus estimate, and no generalization or deployment claim is made "
            "(Table Q1_T08, "
            "Figure Q1_G05).",
        ),
        ("Heading 1", "Complexity and robustness"),
        (
            "Normal",
            final
            + " occupies "
            + v["final_mb"]
            + " MB and scores one recording in "
            + v["final_predict_s"]
            + " s once its features exist; M6 occupies "
            + v["m6_mb"]
            + " MB and takes "
            + v["m6_predict_s"]
            + " s. For "
            + final
            + ", balanced accuracy is "
            + v["noise_clean"]
            + " on recordings the quality "
            "flag marks clean and "
            + v["noise_noisy"]
            + " on those it marks noisy, and "
            + v["duration_long"]
            + " on long against "
            + v["duration_medium"]
            + " on medium-length recordings. These groups are "
            "observational; the controlled noise and truncation rows are in Table Q1_T09.",
        ),
        ("Heading 1", "Limitations"),
        (
            "Normal",
            "PhysioNet has no independent held-out test set, so the binary results are "
            "cross-validated "
            "only. The PhysioNet diagnosis-subtype task is not learnable once recording "
            "source is held "
            "constant. The deep-learning baseline (M9, a one-dimensional CNN) was not run on this "
            "CPU-only machine, so no comparison with deep learning is made. " + DISCLAIMER,
        ),
    ]


_NUMBER = re.compile(r"(?<![A-Za-z0-9_\-./=])\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    """Numeric tokens a reader would take as values, not as parts of identifiers."""
    return [match.group(0).rstrip(",") for match in _NUMBER.finditer(text)]


def _write_narrative(target: Path, evidence_index: str | Path | None) -> tuple[Path, Path]:
    import pandas as pd
    from docx import Document
    from docx.shared import Pt

    facts = narrative_facts()
    values = {fact_id: fact.written() for fact_id, fact in facts.items()}

    rows = [
        {
            "fact_id": fact_id,
            "value_as_written": values[fact_id],
            "value_full_precision": str(fact.raw()),
            "source_file": fact.source,
            "selector": fact.selector(),
        }
        for fact_id, fact in facts.items()
    ]
    sources_csv = target / "Q1_results_narrative_sources.csv"
    from src.utils.io import save_csv

    save_csv(pd.DataFrame(rows), sources_csv)

    document = Document()
    for style, text in _narrative_paragraphs(values):
        document.add_paragraph(text, style=style)
    document.add_paragraph("Where every number comes from", style="Heading 1")
    grid = document.add_table(rows=1, cols=4)
    grid.style = "Table Grid"
    for cell, header in zip(
        grid.rows[0].cells, ("Fact", "Value", "Source file", "Selector"), strict=True
    ):
        cell.text = header
    for row in rows:
        cells = grid.add_row().cells
        for cell, key in zip(
            cells, ("fact_id", "value_as_written", "source_file", "selector"), strict=True
        ):
            cell.text = row[key]
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(7)
    docx_path = target / "Q1_results_narrative.docx"
    document.save(str(docx_path))

    register_evidence(
        "Q1_NARRATIVE",
        docx_path,
        metric_or_asset="results narrative written from generated numbers",
        objective="O1-O6",
        source_data=str(sources_csv),
        command=COMMAND,
        index_path=evidence_index,
    )
    return docx_path, sources_csv


# ---------------------------------------------------------------------------
# writing and verifying
# ---------------------------------------------------------------------------


def q1_dir(out_dir: str | Path | None = None) -> Path:
    return Path(out_dir) if out_dir is not None else _root() / Q1_DIR


def write_q1_pack(
    out_dir: str | Path | None = None, *, evidence_index: str | Path | None = None
) -> dict[str, list[Path]]:
    """Write all twenty assets and ``Q1_asset_index.csv``."""
    import pandas as pd

    from src.reporting.graphs import Graph, GraphSpec, write_graph
    from src.reporting.tables import build_table, write_table
    from src.utils.io import ensure_dir, save_csv

    target = Path(ensure_dir(q1_dir(out_dir)))
    written: dict[str, list[Path]] = {}
    index: list[dict[str, str]] = []

    for table in q1_tables():
        paths = write_table(
            build_table(_table_spec(table), build_frame(table.blocks)),
            target,
            evidence_index=evidence_index,
        )
        written[table.table_id] = list(paths.values())
        index.append(
            {
                "asset_id": table.table_id,
                "kind": "table",
                "derived_from": "; ".join(dict.fromkeys(b.source for b in table.blocks)),
                "how": "row filter and column selection",
            }
        )

    for graph in q1_graphs():
        spec = GraphSpec(
            figure_id=graph.figure_id,
            title=graph.title,
            caption=graph.caption,
            sources=tuple(dict.fromkeys(block.source for block in graph.blocks)),
            exp_id=graph.exp_id,
            objective=graph.objective,
            dataset=graph.dataset,
            notes=(DISCLAIMER,),
            command=COMMAND,
            size=graph.size,
        )
        paths = write_graph(
            Graph(spec=spec, frame=build_frame(graph.blocks), draw=graph.draw),
            target,
            formats=("png", "svg"),
            evidence_index=evidence_index,
            registry=target / "figure_registry.csv",
        )
        written[graph.figure_id] = list(paths.values())
        index.append(
            {
                "asset_id": graph.figure_id,
                "kind": "graph",
                "derived_from": "; ".join(spec.sources),
                "how": "re-plotted from the selected rows",
            }
        )

    for q1_id, copied in _copy_figures(target, evidence_index).items():
        written[q1_id] = copied
        index.append(
            {
                "asset_id": q1_id,
                "kind": "figure",
                "derived_from": "; ".join(FIG + s + ".svg" for s in Q1_FIGURES[q1_id][1]),
                "how": "copied" if len(Q1_FIGURES[q1_id][1]) == 1 else "stacked, vector preserved",
            }
        )
    for q1_id, copied in _copy_algorithms(target, evidence_index).items():
        written[q1_id] = copied
        index.append(
            {
                "asset_id": q1_id,
                "kind": "algorithm",
                "derived_from": ALG + Q1_ALGORITHMS[q1_id][1] + ".docx",
                "how": "copied",
            }
        )

    docx_path, sources_csv = _write_narrative(target, evidence_index)
    written["Q1_NARRATIVE"] = [docx_path, sources_csv]
    index.append(
        {
            "asset_id": "Q1_NARRATIVE",
            "kind": "narrative",
            "derived_from": sources_csv.name,
            "how": "sentences around facts read from result files",
        }
    )

    save_csv(pd.DataFrame(index), target / "Q1_asset_index.csv")
    return written


def _frames_equal(expected: Any, actual: Any) -> list[str]:
    import numpy as np

    problems: list[str] = []
    if list(expected.columns) != list(actual.columns):
        return [
            "columns differ: " + str(list(expected.columns)) + " vs " + str(list(actual.columns))
        ]
    if len(expected) != len(actual):
        return ["row count " + str(len(actual)) + ", expected " + str(len(expected))]
    for column in expected.columns:
        for index, (want, got) in enumerate(
            zip(expected[column].tolist(), actual[column].tolist(), strict=True)
        ):
            try:
                a, b = float(want), float(got)
            except (TypeError, ValueError):
                if str(want) != str(got) and not (_blank(want) and _blank(got)):
                    problems.append(
                        column + "[" + str(index) + "]: " + repr(got) + " != source " + repr(want)
                    )
                continue
            if np.isnan(a) and np.isnan(b):
                continue
            if not np.isclose(a, b, rtol=1e-12, atol=0.0):
                problems.append(
                    column + "[" + str(index) + "]: " + repr(got) + " != source " + repr(want)
                )
    return problems


def _blank(value: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value)) or str(value) == ""
    except (TypeError, ValueError):
        return False


def verify_q1_pack(out_dir: str | Path | None = None) -> dict[str, list[str]]:
    """Re-derive every Q1 CSV from its sources and compare (T103.7, T105.5).

    Returns problems per asset; an empty list everywhere is a pass. The
    narrative is checked two ways: each fact re-read from its source must give
    the value written, and every number in the DOCX must be one of those values.
    """
    import pandas as pd

    target = q1_dir(out_dir)
    problems: dict[str, list[str]] = {}

    for table in q1_tables():
        written = target / (_table_spec(table).slug() + ".csv")
        if not written.is_file():
            problems[table.table_id] = ["missing " + written.name]
            continue
        actual = pd.read_csv(written, keep_default_na=False, na_values=[""])
        problems[table.table_id] = _frames_equal(build_frame(table.blocks), actual)

    for graph in q1_graphs():
        from src.reporting.graphs import GraphSpec

        slug = GraphSpec(
            figure_id=graph.figure_id, title=graph.title, caption="", sources=()
        ).slug()
        written = target / (slug + ".csv")
        if not written.is_file() or not (target / (slug + ".png")).is_file():
            problems[graph.figure_id] = ["missing " + slug + ".csv or .png"]
            continue
        actual = pd.read_csv(written, keep_default_na=False, na_values=[""])
        problems[graph.figure_id] = _frames_equal(build_frame(graph.blocks), actual)

    narrative: list[str] = []
    sources_csv = target / "Q1_results_narrative_sources.csv"
    docx_path = target / "Q1_results_narrative.docx"
    if not sources_csv.is_file() or not docx_path.is_file():
        narrative.append("narrative or its sources CSV missing")
    else:
        from docx import Document

        recorded = pd.read_csv(sources_csv, keep_default_na=False, dtype=str)
        facts = narrative_facts()
        allowed: set[str] = set()
        for row in recorded.itertuples(index=False):
            fact = facts.get(row.fact_id)
            if fact is None:
                narrative.append(row.fact_id + ": no such fact")
                continue
            if fact.written() != row.value_as_written:
                narrative.append(
                    row.fact_id
                    + ": written "
                    + row.value_as_written
                    + ", source now gives "
                    + fact.written()
                )
            allowed.add(row.value_as_written)
            allowed.update(numbers_in(" " + row.value_as_written))
        document = Document(str(docx_path))
        texts = [paragraph.text for paragraph in document.paragraphs]
        for grid in document.tables:
            for grid_row in grid.rows[1:]:
                texts.append(grid_row.cells[1].text)
        for text in texts:
            for number in numbers_in(" " + text):
                if number not in allowed:
                    narrative.append("untraced number " + number + " in: " + text[:80])
    problems["Q1_NARRATIVE"] = narrative

    for q1_id, (stem, _) in Q1_FIGURES.items():
        problems[q1_id] = [s for s in (stem + ".png", stem + ".svg") if not (target / s).is_file()]
    for q1_id, (stem, _) in Q1_ALGORITHMS.items():
        problems[q1_id] = [s for s in (stem + ".docx", stem + ".txt") if not (target / s).is_file()]
    return problems
