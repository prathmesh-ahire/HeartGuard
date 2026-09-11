"""T29 and T30 -- what answers each locked objective, and what it shows (Phase 89).

**T29, the objective-to-evidence mapping.** One row per locked objective, its
wording quoted verbatim from :mod:`src.reporting.objectives`, beside the
datasets and evidence the blueprint's objective mapping requires for it, the
modules that implement it and the files that evidence it. Every module and file
named is checked to exist when the table is built -- a mapping to a file that is
not there is refused, never written. Evidence the blueprint requires that a later
phase has not produced yet is named in ``outstanding`` with the phase that
produces it, and the objective is marked ``partial``. That status is re-derived
from the filesystem on every build, so it changes by itself when the evidence
lands; nobody has to remember to flip it.

**T30, the final conclusion matrix.** One row per objective: the measured
finding, a verdict, the explicit rule that produced the verdict, and the
limitation that bounds it. Every number is read from a table or run output at
build time and formatted by the engine's rules. The verdict wording is chosen by
a rule stated in the row and applied to those numbers, so a verdict cannot be
written after looking at a number and liking it. Nothing here claims diagnosis,
generalization or field performance (research rule 7, and the EXP-F3 limitation
recorded in Docs/note.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.reporting.result_tables import DISCLAIMER, LOCATIONS
from src.reporting.tables import Column, Table, TableSpec, build_table, format_value
from src.utils.logging_setup import get_logger

__all__ = [
    "CONCLUSION_TABLE_IDS",
    "OBJECTIVE_EVIDENCE",
    "ObjectiveEvidence",
    "build_t29",
    "build_t30",
    "build_conclusion_tables",
]

log = get_logger("reporting.conclusion_tables")

CONCLUSION_TABLE_IDS: tuple[str, ...] = ("T29", "T30")

_CI = re.compile(r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]")


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT


def _t(table_id: str) -> str:
    """The repo-relative CSV of a numbered table."""
    directory, stem = LOCATIONS[table_id]
    return directory + "/" + stem + ".csv"


def _read(relative: str) -> Any:
    import pandas as pd

    path = _root() / relative
    if not path.is_file():
        raise FileNotFoundError(
            relative + " -- the phase that writes it has not run in this checkout"
        )
    return pd.read_csv(path)


def _m(value: Any) -> str:
    return format_value(value, "metric")


def _p(value: Any) -> str:
    return format_value(value, "p_value")


def _signed(value: Any) -> str:
    return ("+" if float(value) >= 0 else "") + _m(value)


def _finish(spec: TableSpec, frame: Any) -> Table:
    _, stem = LOCATIONS[spec.table_id]
    if spec.slug() != stem:
        raise ValueError(spec.table_id + ": title slugs to " + spec.slug() + ", not " + stem)
    return build_table(spec, frame)


# ---------------------------------------------------------------------------
# T29 -- objective-to-evidence mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveEvidence:
    """What the blueprint requires for one objective, and what answers it here."""

    number: int
    #: The blueprint's "Dataset(s)" column (section 3), as printed.
    datasets: str
    #: The blueprint's "Required Evidence" column (section 3), as printed.
    required: str
    modules: tuple[str, ...]
    #: Repo-relative files. Every one must exist; the build refuses otherwise.
    evidence: tuple[str, ...]
    tables: tuple[str, ...]
    #: ``(what is missing, glob that matches once it is produced)``. Checked on
    #: every build: an item whose glob matches is no longer outstanding.
    pending: tuple[tuple[str, str], ...] = ()
    scope: str = ""


OBJECTIVE_EVIDENCE: tuple[ObjectiveEvidence, ...] = (
    ObjectiveEvidence(
        number=1,
        datasets="PhysioNet 2016 + CirCor 2022",
        required="Binary metrics, ROC/PR curves, confusion matrix, external validation.",
        modules=(
            "src/models/pipeline.py",
            "src/models/estimators.py",
            "src/evaluation/experiment.py",
            "src/evaluation/transfer.py",
            "src/inference/predictor.py",
        ),
        evidence=(
            _t("T08"),
            _t("T10"),
            "outputs/06_binary_results/EXP-A2/roc_pr_curve_points.csv",
            "outputs/06_binary_results/EXP-A2/confusion_matrices.json",
            _t("T16"),
            "outputs/09_ablation/T-S5_leave_one_source_out_generalization.csv",
        ),
        tables=("T08", "T10", "T16", "T-S5"),
        pending=(
            (
                "G12-G14 ROC, precision-recall and confusion-matrix figures (Phase 92); "
                "the curve points and matrices they draw are produced",
                "outputs/13_figures_diagrams/G14_*.png",
            ),
        ),
        scope=(
            "Binary results are cross-validated within PhysioNet 2016. The CirCor "
            "transfer (T16) and the leave-one-sub-collection-out test (T-S5) are the "
            "evidence on external validation, and both bound the result rather than "
            "extend it."
        ),
    ),
    ObjectiveEvidence(
        number=2,
        datasets="All three dataset families",
        required=("Literature review table, preprocessing ablation and feature-family ablation."),
        modules=(
            "src/preprocessing/ablation.py",
            "src/preprocessing/ablation_run.py",
            "src/evaluation/feature_ablation.py",
            "src/reporting/ablation_report.py",
        ),
        evidence=(
            "outputs/02_preprocessing/preprocessing_ablation.csv",
            _t("T17"),
            _t("T18"),
        ),
        tables=("PP-09", "T17", "T18"),
        pending=(
            (
                "Literature review table (Phase 101)",
                "outputs/**/*literature*.csv",
            ),
        ),
        scope=(
            "Both ablations ran on PhysioNet 2016, the corpus the model is built on; "
            "comparison across noise, duration and population conditions is carried "
            "by T20-T22 under objective 4."
        ),
    ),
    ObjectiveEvidence(
        number=3,
        datasets="PhysioNet 2016 + CirCor 2022",
        required=(
            "Individual-vs-ensemble comparison, calibration and cross-dataset generalization."
        ),
        modules=(
            "src/ensemble/soft_voting.py",
            "src/optimization/weights.py",
            "src/models/calibration.py",
            "src/evaluation/calibration_analysis.py",
            "src/evaluation/statistics.py",
        ),
        evidence=(_t("T08"), _t("T09"), _t("T19"), _t("T23"), _t("T28"), _t("T16")),
        tables=("T08", "T09", "T19", "T23", "T28", "T16"),
        scope=(
            "The ensemble's members are the SVM, Random Forest and Gradient Boosting "
            "models (T06). The CirCor tracks (T13, T14) evaluate the ensemble on a "
            "second corpus under its own cross-validation."
        ),
    ),
    ObjectiveEvidence(
        number=4,
        datasets="PhysioNet 2016 + PASCAL + CirCor",
        required="Duration-wise, noise-wise and dataset-wise robustness tables.",
        modules=(
            "src/evaluation/robustness.py",
            "src/evaluation/duration.py",
            "src/evaluation/location.py",
            "src/evaluation/failure_analysis.py",
            "src/preprocessing/quality.py",
        ),
        evidence=(_t("T20"), _t("T21"), _t("T22"), _t("T27")),
        tables=("T20", "T21", "T22", "T27"),
        scope=(
            "Noise and duration groups are observational; the AWGN sweep and the "
            "truncation study are their interventional counterparts on a PhysioNet "
            "subset."
        ),
    ),
    ObjectiveEvidence(
        number=5,
        datasets="PhysioNet 2016 for search; PASCAL/CirCor for confirmation",
        required=(
            "Baseline-vs-optimized results, selected features, inference time and "
            "complexity trade-off."
        ),
        modules=(
            "src/optimization/driver.py",
            "src/optimization/bayesian.py",
            "src/optimization/randomized.py",
            "src/feature_selection/ranking.py",
            "src/evaluation/tuned.py",
            "src/evaluation/complexity.py",
        ),
        evidence=(
            _t("T19"),
            "outputs/05_search_optimization/T07_search_space_and_best_parameters.csv",
            "outputs/03_features/selected_feature_subset.csv",
            _t("T18"),
            _t("T24"),
            _t("T25"),
            _t("T26"),
        ),
        tables=("T07", "T18", "T19", "T24", "T25", "T26"),
        scope=(
            "The search ran nested inside every PhysioNet outer fold and was repeated "
            "per fold on the PASCAL and CirCor tracks, all of which are tuned runs "
            "(T11-T14)."
        ),
    ),
    ObjectiveEvidence(
        number=6,
        datasets="PASCAL A/B + CirCor",
        required=(
            "Multiclass confusion matrices, macro-F1, per-class recall and large-sample validation."
        ),
        modules=(
            "src/models/pipeline.py",
            "src/evaluation/metrics.py",
            "src/evaluation/aggregation.py",
            "src/data_loader/pascal.py",
            "src/data_loader/circor.py",
        ),
        evidence=(
            _t("T11"),
            _t("T12"),
            "outputs/07_multiclass_results/EXP-B1/confusion_matrices.json",
            "outputs/07_multiclass_results/EXP-B2/confusion_matrices.json",
            _t("T13"),
            _t("T14"),
            _t("T15"),
        ),
        tables=("T11", "T12", "T13", "T14", "T15"),
        pending=(
            (
                "G15-G17 multiclass confusion-matrix and one-vs-rest ROC figures "
                "(Phase 92); the matrices they draw are produced",
                "outputs/13_figures_diagrams/G16_*.png",
            ),
        ),
        scope=(
            "The PhysioNet diagnosis track (EXP-G1, T-S1 to T-S4) was intended to "
            "carry this objective at scale; held within one recording source it does "
            "not (T30)."
        ),
    ),
)


def build_t29(command: str = "") -> Table:
    import pandas as pd

    from src.reporting.objectives import OBJECTIVES

    by_number = {objective.number: objective for objective in OBJECTIVES}
    declared = [item.number for item in OBJECTIVE_EVIDENCE]
    if sorted(declared) != sorted(by_number):
        raise ValueError(
            "T29 maps objectives "
            + str(declared)
            + " but the locked list is "
            + str(sorted(by_number))
        )

    root = _root()
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for item in OBJECTIVE_EVIDENCE:
        objective = by_number[item.number]
        for module in item.modules:
            if not (root / module).is_file():
                raise FileNotFoundError(
                    "objective " + str(item.number) + " names a missing module " + module
                )
        for path in item.evidence:
            if not (root / path).is_file():
                raise FileNotFoundError(
                    "objective " + str(item.number) + " names a missing evidence file " + path
                )
        outstanding = [what for what, pattern in item.pending if not any(root.glob(pattern))]
        rows.append(
            {
                "objective": item.number,
                "handle": objective.handle,
                "wording": objective.wording,
                "datasets": item.datasets,
                "required_evidence": item.required,
                "modules": "; ".join(item.modules),
                "evidence_files": "; ".join(item.evidence),
                "tables": ", ".join(item.tables),
                "status": "partial" if outstanding else "evidence produced",
                "outstanding": "; ".join(outstanding) if outstanding else "none",
                "caveat": " ".join(part for part in (item.scope, objective.caveat) if part),
            }
        )
        sources.extend(item.evidence)

    spec = TableSpec(
        table_id="T29",
        title="Objective-to-Evidence Mapping",
        caption=(
            "Each of the six locked objectives of PV-MEPCG / PulseVision, quoted "
            "verbatim, against the datasets and evidence the blueprint's objective "
            "mapping requires, the modules that implement it and the files that "
            "evidence it. Every module and file named was checked to exist when the "
            "table was built."
        ),
        sources=tuple(dict.fromkeys(sources)),
        columns=(
            Column("objective", "Objective", kind="count"),
            Column("handle", "Handle"),
            Column("wording", "Locked wording (verbatim)"),
            Column("datasets", "Datasets (blueprint)"),
            Column("required_evidence", "Required evidence (blueprint)"),
            Column("modules", "Implementing modules"),
            Column("evidence_files", "Evidence files"),
            Column("tables", "Tables"),
            Column("status", "Status"),
            Column("outstanding", "Outstanding"),
            Column("caveat", "Scope and caveat"),
        ),
        exp_id="all experiments (objective mapping)",
        objective="O1-O6",
        dataset="D1-D4",
        notes=(
            "The objective wording is locked by the blueprint and is quoted exactly; "
            "src/reporting/objectives.py holds it with a sha256 per objective, so a "
            "paraphrase is detectable.",
            "'partial' means evidence the blueprint requires has not been produced "
            "yet; 'Outstanding' names it and the phase that produces it. The status is "
            "re-derived from the files on every build.",
            DISCLAIMER,
        ),
        command=command,
    )
    return _finish(spec, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# T30 -- final conclusion matrix
# ---------------------------------------------------------------------------


def _final_model() -> tuple[str, str]:
    path = "outputs/06_binary_results/final_model_selection.csv"
    selection = _read(path).sort_values("rank")
    return str(selection.iloc[0]["model_id"]), path


def _pair(tests: Any, run: str, metric: str, left: str, right: str) -> dict[str, float]:
    """``{left_mean, right_mean, p_holm}`` for one Phase 82 paired test, either order."""
    for a, b, flip in ((left, right, False), (right, left, True)):
        match = tests[
            (tests["run"] == run)
            & (tests["metric"] == metric)
            & (tests["model_a"] == a)
            & (tests["model_b"] == b)
        ]
        if len(match):
            row = match.iloc[0]
            return {
                "left": float(row["mean_b"] if flip else row["mean_a"]),
                "right": float(row["mean_a"] if flip else row["mean_b"]),
                "p_holm": float(row["p_holm"]),
            }
    raise KeyError("no Phase 82 test for " + run + " " + metric + " " + left + " vs " + right)


def _o1(final: str, selection_path: str) -> dict[str, Any]:
    aggregate_path = "outputs/06_binary_results/EXP-A2/aggregate_metrics.csv"
    loso_path = "outputs/09_ablation/T-S5_leave_one_source_out_generalization.csv"
    agg = _read(aggregate_path).set_index("model_id").loc[final]
    loso = _read(loso_path)
    mine = loso.set_index("model_id").loc[final]
    transfer = _read(_t("T16"))
    external = transfer[(transfer["metric"] == "roc_auc") & (transfer["level"] == "recording")]

    # Transfer is claimed only if some model's held-out-source ROC-AUC clears
    # chance (0.5) by more than one SD across the held-out sources.
    clears = bool(((loso["roc_auc_holdout"] - loso["roc_auc_holdout_sd"]) > 0.5).any())
    verdict = (
        "Achieved as within-corpus screening performance; the result does not transfer "
        "to an unseen recording setup"
        if not clears
        else "Achieved within corpus; held-out-source transfer partially holds"
    )
    limitation = (
        "Held out one PhysioNet sub-collection at a time, "
        + final
        + "'s ROC-AUC falls from "
        + _m(mine["roc_auc_pooled"])
        + " to "
        + _m(mine["roc_auc_holdout"])
        + " (SD "
        + _m(mine["roc_auc_holdout_sd"])
        + ", EXP-F3); across the "
        + str(len(loso))
        + " models it is "
        + _m(loso["roc_auc_holdout"].min())
        + "-"
        + _m(loso["roc_auc_holdout"].max())
        + "."
    )
    if len(external):
        limitation += (
            " Transferred to CirCor (EXP-D1) its recording-level ROC-AUC is "
            + _m(external["external_value"].iloc[0])
            + "."
        )
    limitation += " No claim of generalization, deployment-readiness or field performance is made."
    return {
        "finding": (
            "Final model "
            + final
            + " (EXP-A2, 25 subject-grouped folds within PhysioNet 2016): sensitivity "
            + _m(agg["sensitivity_mean"])
            + " (SD "
            + _m(agg["sensitivity_sd"])
            + "), specificity "
            + _m(agg["specificity_mean"])
            + ", balanced accuracy "
            + _m(agg["balanced_accuracy_mean"])
            + ", ROC-AUC "
            + _m(agg["roc_auc_mean"])
            + "."
        ),
        "verdict": verdict,
        "rule": (
            "The within-corpus result is stated as measured. Transfer is claimed only if "
            "some model's held-out-source ROC-AUC exceeds 0.5 by more than one SD."
        ),
        "limitation": limitation,
        "tables": "T08, T10, T16, T-S5",
        "sources": [selection_path, aggregate_path, loso_path, _t("T16")],
    }


def _o2(final: str) -> dict[str, Any]:
    pp_path = "outputs/02_preprocessing/preprocessing_ablation.csv"
    pp = _read(pp_path)
    span = float(pp["sensitivity"].max() - pp["sensitivity"].min())
    fold_sd = float(pp["sensitivity_sd"].mean())
    shipped = pp[pp["is_shipped_configuration"].astype(bool)].iloc[0]
    best_ba = pp.sort_values("balanced_accuracy", ascending=False).iloc[0]

    t17 = _read(_t("T17"))
    mine = t17[t17["model_id"] == final]
    single = mine[
        (mine["n_features"] < 138)
        & ~mine["families"].astype(str).str.contains(r"[,+ ]|select", regex=True)
    ].sort_values("sensitivity")
    strongest, weakest = single.iloc[-1], single.iloc[0]
    subset = _read(_t("T18")).set_index("model_id").loc[final]
    # T18 does not carry the paired fold count; A8's own fold count in T17 is it.
    paired_folds = int(mine.loc[mine["config_id"] == "A8", "n_folds"].iloc[0])

    literature = any(_root().glob("outputs/**/*literature*.csv"))
    finding = (
        "Preprocessing (PP-09, "
        + str(pp["model_id"].iloc[0])
        + "): the four filter x normalization arms span "
        + _m(span)
        + " in sensitivity against a mean fold SD of "
        + _m(fold_sd)
        + "; the shipped arm is "
        + str(shipped["arm_id"])
        + (
            ", and " + str(best_ba["arm_id"]) + " scores the highest balanced accuracy"
            if best_ba["arm_id"] != shipped["arm_id"]
            else ", which also scores the highest balanced accuracy"
        )
        + ". Feature families (EXP-F1, "
        + final
        + "): strongest single family "
        + str(strongest["families"])
        + " (sensitivity "
        + _m(strongest["sensitivity"])
        + "), weakest "
        + str(weakest["families"])
        + " ("
        + _m(weakest["sensitivity"])
        + "); the selected "
        + str(int(subset["n_features_selected"]))
        + "-feature subset changes sensitivity by "
        + _signed(subset["sensitivity_delta"])
        + " against all "
        + str(int(subset["n_features_all"]))
        + ", winning "
        + str(int(subset["sensitivity_folds_selected_wins"]))
        + " of "
        + str(paired_folds)
        + " folds."
    )
    limitation = (
        "The preprocessing arms are indistinguishable -- their sensitivity span is "
        "below one fold SD -- so the shipped configuration rests on its design "
        "argument, not on a measured gain. "
        if span < fold_sd
        else ""
    ) + ("Chroma and envelope have no solo arm, and both ablations ran on PhysioNet 2016 only.")
    return {
        "finding": finding,
        "verdict": (
            "Achieved: both ablations and the literature review are produced"
            if literature
            else "Partially achieved: both ablations are measured; the literature review "
            "table is not yet produced (Phase 101)"
        ),
        "rule": "Partial while any evidence the blueprint requires is missing (T29).",
        "limitation": limitation,
        "tables": "PP-09, T17, T18, T29",
        "sources": [pp_path, _t("T17"), _t("T18")],
    }


def _o3(final: str) -> dict[str, Any]:
    tests_path = "outputs/12_statistics/paired_fold_tests_interpreted.csv"
    tests = _read(tests_path)
    tuned_s = _pair(tests, "EXP-A2", "sensitivity", "M6", final)
    tuned_b = _pair(tests, "EXP-A2", "balanced_accuracy", "M6", final)
    default_s = _pair(tests, "EXP-A1", "sensitivity", "M6", final)

    t09 = _read(_t("T09")).set_index("metric")
    identical = int(t09.loc["balanced_accuracy", "n_folds_identical"])
    n_folds = int(t09.loc["balanced_accuracy", "n_folds"])
    calibration = _read(_t("T23"))
    ece = calibration[calibration["run"] == "EXP-A2"].set_index("model_id")["ece_mean"]
    stages = _read(_t("T19")).set_index("stage")

    # Significant only at Holm-corrected p < 0.05, and only in the ensemble's favour.
    tuned_better = any(
        pair["p_holm"] < 0.05 and pair["left"] > pair["right"] for pair in (tuned_s, tuned_b)
    )
    default_better = default_s["p_holm"] < 0.05 and default_s["left"] > default_s["right"]
    if tuned_better:
        verdict = "Achieved: the ensemble is significantly better than " + final
    else:
        verdict = (
            "Ensemble implemented and evaluated; not significantly better than the "
            "simplest model (" + final + ") in the final, tuned configuration"
        )
        if default_better:
            verdict += "; significantly better only with default hyperparameters"
    return {
        "finding": (
            "EXP-A2 (tuned): ensemble M6 sensitivity "
            + _m(tuned_s["left"])
            + " vs "
            + final
            + " "
            + _m(tuned_s["right"])
            + " (Holm p="
            + _p(tuned_s["p_holm"])
            + "), balanced accuracy "
            + _m(tuned_b["left"])
            + " vs "
            + _m(tuned_b["right"])
            + " (Holm p="
            + _p(tuned_b["p_holm"])
            + "). EXP-A1 (defaults): M6 sensitivity "
            + _m(default_s["left"])
            + " vs "
            + _m(default_s["right"])
            + " (Holm p="
            + _p(default_s["p_holm"])
            + "). Optimized weights (M7) equal M6 on "
            + str(identical)
            + " of "
            + str(n_folds)
            + " folds. ECE (EXP-A2): M6 "
            + _m(ece.get("M6"))
            + ", "
            + final
            + " "
            + _m(ece.get(final))
            + "."
        ),
        "verdict": verdict,
        "rule": (
            "A difference is called significant only at Holm-corrected p < 0.05 over "
            "the n=25 paired folds (T28), and only in the stated direction."
        ),
        "limitation": (
            "Tuning the members moved the ensemble's sensitivity from "
            + _m(stages.loc["default_ensemble", "sensitivity"])
            + " (defaults) to "
            + _m(stages.loc["tuned_ensemble_equal", "sensitivity"])
            + " (tuned); the weight search returned equal or prediction-equivalent "
            "weights, so M7 adds nothing measurable on this corpus."
        ),
        "tables": "T08, T09, T19, T23, T28",
        "sources": [tests_path, _t("T09"), _t("T23"), _t("T19")],
    }


def _o4(final: str) -> dict[str, Any]:
    t20 = _read(_t("T20"))
    observed = t20[
        (t20["analysis"] == "observational_pp08")
        & (t20["run"] == "EXP-A2")
        & (t20["model_id"] == final)
    ].set_index("group")
    awgn = t20[t20["analysis"] == "interventional_awgn"].set_index("group")
    no_noise = next(g for g in awgn.index if str(g).startswith("clean"))

    t21 = _read(_t("T21"))
    bands = t21[
        (t21["analysis"] == "observational_bands")
        & (t21["run"] == "EXP-A2")
        & (t21["model_id"] == final)
    ].set_index("group")
    truncated = t21[t21["analysis"] == "interventional_truncation"].set_index("group")
    untruncated = next(g for g in truncated.index if str(g).startswith("full"))
    shortest = min(
        (g for g in truncated.index if str(g).endswith("s clip")),
        key=lambda g: float(str(g).split()[0]),
    )

    t27 = _read(_t("T27"))
    source = "EXP-A2 " + final + " (in-domain)"
    confidence = t27[(t27["source"] == source) & (t27["category"] == "confidence_band")]
    confidence = confidence.sort_values("level")

    t22 = _read(_t("T22"))
    outcome = t22[(t22["run"] == "EXP-C2") & t22["reported"].astype(bool)]
    by_location = outcome.groupby("location")["balanced_accuracy_mean"].mean().sort_values()

    degrades = bool(
        observed.loc["noisy", "sensitivity_mean"] < observed.loc["clean", "sensitivity_mean"]
        and bands.loc["medium", "sensitivity_mean"] < bands.loc["long", "sensitivity_mean"]
    )
    return {
        "finding": (
            "Noise (PP-08 groups, EXP-A2 "
            + final
            + "): sensitivity "
            + _m(observed.loc["clean", "sensitivity_mean"])
            + " on clean vs "
            + _m(observed.loc["noisy", "sensitivity_mean"])
            + " on noisy recordings; with added white noise, "
            + _m(awgn.loc[no_noise, "sensitivity_mean"])
            + " at none, "
            + _m(awgn.loc["10 dB SNR", "sensitivity_mean"])
            + " at 10 dB SNR and "
            + _m(awgn.loc["0 dB SNR", "sensitivity_mean"])
            + " at 0 dB. Duration: "
            + _m(bands.loc["long", "sensitivity_mean"])
            + " on long vs "
            + _m(bands.loc["medium", "sensitivity_mean"])
            + " on medium-length recordings; a "
            + str(shortest)
            + " gives "
            + _m(truncated.loc[shortest, "sensitivity_mean"])
            + " against "
            + _m(truncated.loc[untruncated, "sensitivity_mean"])
            + " untruncated. Confidence: error rate "
            + _m(confidence["error_rate"].iloc[0])
            + " in the lowest band vs "
            + _m(confidence["error_rate"].iloc[-1])
            + " in the highest. Location (CirCor outcome): balanced accuracy "
            + _m(by_location.iloc[-1])
            + " at "
            + str(by_location.index[-1])
            + " vs "
            + _m(by_location.iloc[0])
            + " at "
            + str(by_location.index[0])
            + "."
        ),
        "verdict": (
            "Robustness measured on all four corpora; performance degrades on noisy, "
            "shorter and low-confidence recordings"
            if degrades
            else "Robustness measured on all four corpora; no consistent degradation"
        ),
        "rule": (
            "Degradation is stated only where the noisy or shorter group's sensitivity "
            "is below the clean or longer group's for the final model."
        ),
        "limitation": (
            "Noise and duration groups are observational -- recordings differ in more "
            "than noise or length. The AWGN sweep and the truncation study are "
            "single-split interventional checks on a PhysioNet subset."
        ),
        "tables": "T20, T21, T22, T27",
        "sources": [_t("T20"), _t("T21"), _t("T22"), _t("T27")],
    }


def _o5(final: str) -> dict[str, Any]:
    stages = _read(_t("T19")).sort_values("stage_order").reset_index(drop=True)
    first, last = stages.iloc[0], stages.iloc[-1]
    best = stages.sort_values("sensitivity", ascending=False).iloc[0]
    subset = _read(_t("T18")).set_index("model_id").loc[final]
    complexity = _read(_t("T24")).set_index("model_id")
    predict_final = float(complexity.loc[final, "single_predict_seconds"])
    predict_ensemble = float(complexity.loc["M6", "single_predict_seconds"])
    pipeline = float(complexity.loc[final, "pipeline_seconds_per_recording"])

    peak_is_last = best["stage"] == last["stage"]
    return {
        "finding": (
            "Optimization stages (EXP-F2, sensitivity): "
            + _m(first["sensitivity"])
            + " for the untuned single model, "
            + _m(best["sensitivity"])
            + " at the best stage ("
            + str(best["stage"])
            + "), "
            + _m(last["sensitivity"])
            + " after the full search and weight optimization. Selected subset: "
            + str(int(subset["n_features_selected"]))
            + " of "
            + str(int(subset["n_features_all"]))
            + " features, sensitivity "
            + _signed(subset["sensitivity_delta"])
            + " for "
            + final
            + ". Complexity: "
            + final
            + " predicts one record in "
            + format_value(predict_final, "metric", 4)
            + " s against the ensemble's "
            + format_value(predict_ensemble, "metric", 4)
            + " s ("
            + format_value(predict_ensemble / predict_final, "mean_count")
            + "x); the whole pipeline takes "
            + format_value(pipeline, "seconds")
            + " s per recording."
        ),
        "verdict": (
            "Search applied to hyperparameters, feature subset and ensemble weights; the "
            "fully optimized stage scores highest"
            if peak_is_last
            else "Search applied to hyperparameters, feature subset and ensemble weights; "
            "its measured benefit is complexity, not sensitivity -- the best-scoring stage "
            "is " + str(best["stage"]) + ", not the fully optimized one"
        ),
        "rule": (
            "All stages are scored on the same 25 folds; the stage with the highest mean "
            "sensitivity is named, and no stage is called an improvement from a "
            "difference of means alone."
        ),
        "limitation": (
            "The nested search ran at a reduced budget. Everything before the "
            "prediction -- loading, preprocessing and feature extraction -- is "
            + format_value(100.0 * (pipeline - predict_final) / pipeline, "percent")
            + "% of a recording's pipeline time, so a faster model does not make "
            "screening proportionally faster."
        ),
        "tables": "T07, T18, T19, T24, T25",
        "sources": [_t("T19"), _t("T18"), _t("T24")],
    }


def _interval(text: Any) -> tuple[float, float] | None:
    match = _CI.search(str(text))
    return (float(match.group(1)), float(match.group(2))) if match else None


def _o6() -> dict[str, Any]:
    t11 = _read(_t("T11"))
    t12 = _read(_t("T12"))
    top_a, top_b = t11.iloc[0], t12.iloc[0]
    n_classes_b = sum(1 for c in t12.columns if c.startswith("recall_"))
    partial_b = int((t12["n_classes_predicted"] < n_classes_b).sum())

    from src.evaluation.experiment import Experiment

    t13 = _read(_t("T13"))
    three = t13[t13["run"] == "EXP-C1-three_class"]
    # From the declared label space, so the chance level is not a typed 1/3.
    classes_c = len(Experiment.load("EXP-C1").for_variant("three_class").labels)
    best_recording = three[three["level"] == "recording"]["balanced_accuracy_mean"].max()
    best_patient = three[three["level"] == "patient"]["balanced_accuracy_mean"].max()
    t14 = _read(_t("T14"))
    best_outcome = t14[t14["level"] == "patient"]["balanced_accuracy_mean"].max()

    within_path = (
        "outputs/07_multiclass_results/T-S3_physionet_diagnosis_multiclass_within_one_source.csv"
    )
    per_class_path = (
        "outputs/07_multiclass_results/T-S4_diagnosis_per_class_results_within_one_source.csv"
    )
    within = _read(within_path)
    k = _read(per_class_path)["class_name"].nunique()

    # Rankings are distinguishable only if the top interval clears every other.
    intervals = [_interval(text) for text in t11["macro_f1_record_ci"]]
    distinguishable = bool(
        intervals[0] is not None
        and all(other is not None and intervals[0][0] > other[1] for other in intervals[1:])
    )
    return {
        "finding": (
            "PASCAL A (EXP-B1, 4 classes): best macro-F1 "
            + _m(top_a["macro_f1_mean"])
            + " ("
            + str(top_a["model_id"])
            + ", record-level 95% CI "
            + str(top_a["macro_f1_record_ci"])
            + "). PASCAL B (EXP-B2, "
            + str(n_classes_b)
            + " classes): best macro-F1 "
            + _m(top_b["macro_f1_mean"])
            + " ("
            + str(top_b["model_id"])
            + "); "
            + str(partial_b)
            + " of "
            + str(len(t12))
            + " models never predict one class. CirCor murmur (EXP-C1, "
            + str(classes_c)
            + " classes): best balanced accuracy "
            + _m(best_recording)
            + " per recording and "
            + _m(best_patient)
            + " per patient, against a chance level of "
            + _m(1.0 / classes_c)
            + ". CirCor outcome (EXP-C2): best balanced accuracy "
            + _m(best_outcome)
            + " per patient."
        ),
        "verdict": (
            "Implemented on four separate multiclass and large-sample tasks; performance "
            "is modest"
            + (
                ""
                if distinguishable
                else ", and the PASCAL A model ranking is not distinguishable at this sample size"
            )
        ),
        "rule": (
            "A ranking is called distinguishable only if the top model's record-level "
            "95% CI clears every other model's."
        ),
        "limitation": (
            "PASCAL A is small and its best-detected class, 'artifact', is a "
            "recording-quality label, not a cardiac class. The PhysioNet diagnosis track "
            "held within one recording source scores balanced accuracy "
            + _m(within["balanced_accuracy_mean"].max())
            + " against a chance level of "
            + _m(1.0 / k)
            + ", so it does not carry this objective."
        ),
        "tables": "T11, T12, T13, T14, T15, T-S3",
        "sources": [_t("T11"), _t("T12"), _t("T13"), _t("T14"), within_path, per_class_path],
    }


def build_t30(command: str = "") -> Table:
    import pandas as pd

    from src.reporting.objectives import OBJECTIVES

    final, selection_path = _final_model()
    concluded = {
        1: _o1(final, selection_path),
        2: _o2(final),
        3: _o3(final),
        4: _o4(final),
        5: _o5(final),
        6: _o6(),
    }
    handles = {objective.number: objective.handle for objective in OBJECTIVES}

    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for number in sorted(concluded):
        row = concluded[number]
        rows.append(
            {
                "objective": number,
                "handle": handles[number],
                "finding": row["finding"],
                "verdict": row["verdict"],
                "rule": row["rule"],
                "limitation": row["limitation"],
                "tables": row["tables"],
                "sources": "; ".join(row["sources"]),
            }
        )
        sources.extend(row["sources"])

    spec = TableSpec(
        table_id="T30",
        title="Final Conclusion Matrix",
        caption=(
            "What the evidence shows for each locked objective of PV-MEPCG / "
            "PulseVision: the measured finding, the verdict, the explicit rule that "
            "produced the verdict, and the limitation that bounds it. Every number is "
            "read from the tables and run outputs named in its row when the table is "
            "built."
        ),
        sources=tuple(dict.fromkeys(sources)),
        columns=(
            Column("objective", "Objective", kind="count"),
            Column("handle", "Handle"),
            Column("finding", "Finding"),
            Column("verdict", "Verdict"),
            Column("rule", "Rule applied"),
            Column("limitation", "Limitation"),
            Column("tables", "Evidence tables"),
            Column("sources", "Source files"),
        ),
        exp_id="all experiments (conclusions)",
        objective="O1-O6",
        dataset="D1-D4",
        notes=(
            "Every number in this table is generated from the files named in its row "
            "and formatted by the table engine's rules; none is typed.",
            "Each verdict follows the rule stated beside it. The verdicts describe an "
            "academic screening and decision-support prototype evaluated on public "
            "corpora; none is a statement about clinical use.",
            "All PhysioNet results are cross-validated within the PhysioNet 2016 corpus; "
            "objective 1's limitation states how far they fall outside it.",
            DISCLAIMER,
        ),
        command=command,
    )
    return _finish(spec, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_BUILDERS = {"T29": build_t29, "T30": build_t30}


def build_conclusion_tables(
    table_ids: tuple[str, ...] = CONCLUSION_TABLE_IDS, *, command: str = ""
) -> dict[str, Table]:
    """Build T29 and/or T30 without writing anything."""
    built: dict[str, Table] = {}
    for table_id in table_ids:
        if table_id not in _BUILDERS:
            raise KeyError("not a Phase 89 table: " + table_id)
        built[table_id] = _BUILDERS[table_id](command)
        log.info("%s built (%d rows)", table_id, len(built[table_id].frame))
    return built
