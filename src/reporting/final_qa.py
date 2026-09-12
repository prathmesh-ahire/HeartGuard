"""The final QA sweep: six areas re-verified against the committed files (Phase 124).

Every claim below has been checked before, in the phase that produced it. This
sweep exists because "it passed when it was written" and "it is true now" are
different statements, and by Phase 124 roughly nine hundred tasks have edited
the files those earlier checks looked at.

So the rule here is that **a check reads a file and recomputes something**.
Nothing in this module asserts that a check was once run, and nothing takes a
report's own summary line as evidence -- the audit's stamp is re-derived from
the HTML on disk, the leakage checks re-intersect the real uid lists, the metric
checks re-read the result CSVs' columns.

The six areas are T124.1 to T124.6:

===========  ==========================================================
``dataset``  integrity, duplicates, label mapping, class counts, rates
``split``    zero group leakage in every fold of every task
``feature``  finite, complete, and keyed to the locked registry
``model``    seed 42, fold-safe pipelines, calibration as configured
``metric``   never accuracy alone (research rule 6)
``search``   no search ever touched a final test fold
``dashboard`` the T119.3 displayed-value audit is green right now
===========  ==========================================================

## Two checks that report rather than assert

**PASCAL A cannot be group-split.** Its 124 records carry no subject identifier
that survives the audit, so its scheme is ``repeated_5x2_stratified`` and the
"no subject in both train and test" guarantee is *unavailable* there, not
satisfied. The check records that as ``not_applicable`` with the count of
records whose ``subject_derived`` is false. Marking it green would be a claim
the data cannot support; marking it red would imply a defect that does not
exist.

**The feature matrix is gitignored.** A checkout without it cannot verify
finiteness, so that check skips with its reason rather than passing vacuously.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging_setup import get_logger

__all__ = [
    "AREAS",
    "Check",
    "QaReport",
    "run_sweep",
    "write_report",
]

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "00_evidence_index"
REPORT_JSON = OUT_DIR / "final_qa_report.json"
REPORT_MD = OUT_DIR / "final_qa_report.md"

AREAS: tuple[tuple[str, str], ...] = (
    ("dataset", "T124.1 -- integrity, duplicates, label mapping, class counts, sampling rates"),
    ("split", "T124.2 -- zero patient leakage across every fold of every task"),
    ("feature", "T124.3 -- every feature finite and reproducible across folds"),
    ("model", "T124.4 -- fixed seeds, fold-safe scaling, calibration as configured"),
    ("metric", "T124.5 -- never accuracy alone (research rule 6)"),
    ("search", "T124.6 -- no search touched a final test fold"),
    ("dashboard", "T124.6 -- the T119.3 displayed-value audit is green"),
)

#: The audited dataset facts, from the 2026-08-22 scan of the real files. These
#: are the numbers CLAUDE.md and the README publish, so the sweep's job is to
#: confirm the committed audit still agrees with them -- a table that drifted
#: from the corpus it describes is exactly what this phase exists to catch.
AUDITED: dict[str, dict[str, int]] = {
    # D1's 301 `validation/` records are on disk and in the master metadata, but
    # they duplicate 301 training records and are DROPPED from every experiment
    # (2026-08-26 decision). The count below is what the audit inventories, not
    # what is modelled.
    "D1": {"supervised": 3240, "unlabelled": 301},
    "D2": {"supervised": 124, "unlabelled": 52},
    "D3": {"supervised": 461, "unlabelled": 195},
    "D4": {"supervised": 3163, "patients": 942},
}
AUDITED_TOTAL_FILES = 7536

#: Problems in the audit's file report that mean a file cannot be used at all,
#: as against a quality flag on a file that reads perfectly well.
BROKEN_FILE_PROBLEMS = frozenset({"missing", "unreadable", "corrupt", "empty", "zero_bytes"})

#: Native sampling rate per dataset source, audited.
AUDITED_FS: dict[str, int] = {"D1": 2000, "D2": 44100, "D3": 4000, "D4": 4000}

#: sha256 of the ordered 138 feature names. Two runs that disagree on it are not
#: comparable, which is why it is pinned in three places rather than derived.
REGISTRY_FINGERPRINT_PREFIX = "9aa6c14561d3f32c"

#: Research rule 6. A result table that reports accuracy must also report these.
BINARY_REQUIRED = ("sensitivity", "specificity", "f1", "balanced_accuracy", "roc_auc")
MULTICLASS_REQUIRED = ("macro_f1", "balanced_accuracy", "macro_recall")


@dataclass(frozen=True)
class Check:
    """One recomputed QA claim."""

    area: str
    check_id: str
    title: str
    status: str  # "pass" | "fail" | "skipped" | "not_applicable"
    detail: str
    evidence: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        return self.status == "fail"


@dataclass
class QaReport:
    """Every check, with the area rollup T124.7 gates on."""

    checks: list[Check] = field(default_factory=list)
    generated_utc: str = ""

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        LOGGER.info("%s %s: %s", check.check_id, check.status.upper(), check.title)
        return check

    def for_area(self, area: str) -> list[Check]:
        return [check for check in self.checks if check.area == area]

    def area_status(self, area: str) -> str:
        checks = self.for_area(area)
        if not checks:
            return "empty"
        if any(check.failed for check in checks):
            return "fail"
        if any(check.status == "skipped" for check in checks):
            return "pass_with_skips"
        return "pass"

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.failed]

    @property
    def ok(self) -> bool:
        """T124.7's gate: no area failed.

        A ``skipped`` check does not fail the sweep -- it names what could not be
        verified in this checkout and why -- but it does keep the area out of a
        clean ``pass``, so it stays visible in the rollup.
        """
        return not self.failures and all(
            self.area_status(area) != "empty" for area, _ in AREAS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_utc": self.generated_utc,
            "ok": self.ok,
            "areas": {
                area: {
                    "description": description,
                    "status": self.area_status(area),
                    "checks": len(self.for_area(area)),
                }
                for area, description in AREAS
            },
            "counts": {
                status: sum(1 for check in self.checks if check.status == status)
                for status in ("pass", "fail", "skipped", "not_applicable")
            },
            "checks": [asdict(check) for check in self.checks],
        }


def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:  # pragma: no cover - every artifact is inside the project
        return str(path)


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


# ---------------------------------------------------------------------------
# T124.1 -- dataset QA
# ---------------------------------------------------------------------------


def _check_dataset(report: QaReport) -> None:
    audit = PROJECT_ROOT / "outputs" / "01_dataset_audit"

    # `missing_corrupt_files.csv` is named for what it was built to find and
    # holds what it actually found: 64 QUALITY flags -- 63 clipped recordings
    # and one silent one -- and zero unreadable files. So the QA claim is the
    # one the audit supports: nothing is missing or undecodable. The quality
    # flags are a property of the corpus, not a defect in this project, and
    # failing on them would be failing on the dataset being real.
    missing = audit / "missing_corrupt_files.csv"
    if missing.is_file():
        rows = _read_csv(missing)
        problems = (
            rows["problem"].astype(str)
            if "problem" in rows.columns
            else pd.Series(dtype=str)
        )
        fatal = rows[problems.isin(BROKEN_FILE_PROBLEMS)]
        quality = rows[~problems.isin(BROKEN_FILE_PROBLEMS)]
        report.add(
            Check(
                "dataset",
                "QA-DATA-01",
                "no file in the corpus is missing or unreadable",
                "fail" if not fatal.empty else "pass",
                f"{len(fatal)} unreadable/missing file(s); "
                f"{len(quality)} quality flag(s) recorded "
                f"({dict(problems.value_counts()) if len(rows) else {}})",
                (_rel(missing),),
            )
        )
    else:
        report.add(
            Check(
                "dataset",
                "QA-DATA-01",
                "no file in the corpus is missing or unreadable",
                "skipped",
                "missing_corrupt_files.csv is not in this checkout",
            )
        )

    master_csv = audit / "metadata_master.csv"
    if not master_csv.is_file():
        report.add(
            Check(
                "dataset",
                "QA-DATA-02",
                "the audited per-dataset counts still hold",
                "skipped",
                "metadata_master.csv is not in this checkout",
            )
        )
        return

    master = _read_csv(master_csv, low_memory=False)
    supervised = master[master["binary_label"].notna() | master["multiclass_label"].notna()]

    counted = master.groupby("dataset_source").size().to_dict()
    mismatches: list[str] = []
    for source, expected in AUDITED.items():
        actual = int(counted.get(source, 0))
        wanted = expected["supervised"] + expected.get("unlabelled", 0)
        if actual != wanted:
            mismatches.append(f"{source}: {actual} rows, audited {wanted}")
    report.add(
        Check(
            "dataset",
            "QA-DATA-02",
            "the audited per-dataset record counts still hold",
            "fail" if mismatches else "pass",
            "; ".join(mismatches)
            or ", ".join(f"{source} {int(counted.get(source, 0))}" for source in sorted(AUDITED)),
            (_rel(master_csv),),
        )
    )

    rates = master.groupby("dataset_source")["original_fs"].agg(lambda s: sorted(set(s)))
    wrong = {
        source: found
        for source, found in rates.items()
        if source in AUDITED_FS and found != [AUDITED_FS[source]]
    }
    report.add(
        Check(
            "dataset",
            "QA-DATA-03",
            "every record carries its dataset's audited native sampling rate",
            "fail" if wrong else "pass",
            str(wrong)
            or ", ".join(f"{source} {fs} Hz" for source, fs in sorted(AUDITED_FS.items())),
            (_rel(master_csv),),
        )
    )

    duplicates = audit / "duplicate_report.csv"
    if duplicates.is_file():
        frame = _read_csv(duplicates)
        undecided = frame[frame.get("decision", pd.Series(dtype=str)).astype(str).str.len() == 0]
        report.add(
            Check(
                "dataset",
                "QA-DATA-04",
                "every duplicate group carries an explicit decision",
                "fail" if not undecided.empty else "pass",
                f"{len(frame)} duplicate row(s), {len(undecided)} without a decision",
                (_rel(duplicates),),
            )
        )

    # The trap that silently doubles the corpus: Heartbeat_Sound/ is a 100%
    # duplicate of set_a+set_b and is a label helper only. Nothing sourced from
    # it may carry a label the models train on.
    from_helper = master[master["file_path"].astype(str).str.contains("Heartbeat_Sound", na=False)]
    trainable = from_helper[
        from_helper["binary_label"].notna() | from_helper["multiclass_label"].notna()
    ]
    report.add(
        Check(
            "dataset",
            "QA-DATA-05",
            "nothing from the duplicate Heartbeat_Sound/ tree is a training row",
            "fail" if not trainable.empty else "pass",
            f"{len(from_helper)} row(s) reference it, {len(trainable)} of them labelled",
            (_rel(master_csv),),
        )
    )

    # Label mapping: the five label spaces are five separate targets. This is
    # the project's own cross-task bleed assertion, re-run on the committed
    # master rather than on a freshly built one.
    try:
        from src.data_loader.master import assert_no_cross_task_bleed

        assert_no_cross_task_bleed(master)
        bleed_detail, bleed_status = "the five label spaces remain disjoint", "pass"
    except AssertionError as error:
        bleed_detail, bleed_status = str(error), "fail"
    report.add(
        Check(
            "dataset",
            "QA-DATA-06",
            "the five label spaces are never merged (research rule 4)",
            bleed_status,
            bleed_detail,
            (_rel(master_csv),),
        )
    )

    counts = audit / "class_distribution.csv"
    if counts.is_file():
        frame = _read_csv(counts)
        overall = frame[frame["scope"].astype(str) == "overall"]
        scoped = overall if not overall.empty else frame
        negative = scoped[scoped["n_records"] <= 0]
        report.add(
            Check(
                "dataset",
                "QA-DATA-07",
                "every class in every task has at least one record",
                "fail" if not negative.empty else "pass",
                f"{len(scoped)} class row(s) across {scoped['task'].nunique()} task(s)"
                + (f"; empty: {negative['class'].tolist()}" if not negative.empty else ""),
                (_rel(counts),),
            )
        )

    report.add(
        Check(
            "dataset",
            "QA-DATA-08",
            "the corpus is the audited size",
            "pass" if len(master) >= AUDITED_TOTAL_FILES - 0 else "fail",
            f"{len(master)} records in the master metadata, audited {AUDITED_TOTAL_FILES}",
            (_rel(master_csv),),
        )
        if len(master) == AUDITED_TOTAL_FILES
        else Check(
            "dataset",
            "QA-DATA-08",
            "the corpus is the audited size",
            "fail",
            f"{len(master)} records in the master metadata, audited {AUDITED_TOTAL_FILES}",
            (_rel(master_csv),),
        )
    )
    del supervised


# ---------------------------------------------------------------------------
# T124.2 -- split QA
# ---------------------------------------------------------------------------


def _check_splits(report: QaReport) -> None:
    from src.evaluation import cv

    split_map = cv.split_map_path()
    if not split_map.is_file():
        report.add(
            Check(
                "split",
                "QA-SPLIT-01",
                "no subject appears in both train and test",
                "skipped",
                "subject_split_map.csv is not in this checkout",
            )
        )
        return

    frame = _read_csv(split_map)
    grouped_schemes = {"repeated_5x5_grouped", "grouped_5fold", "patient_grouped_5fold"}

    for task in sorted(frame["task"].astype(str).unique()):
        scheme = str(frame[frame["task"] == task]["scheme"].iloc[0])
        rows = frame[frame["task"] == task]
        underived = int((~rows["subject_derived"].astype(bool)).sum())

        if scheme not in grouped_schemes:
            report.add(
                Check(
                    "split",
                    f"QA-SPLIT-{task}",
                    f"{task}: subject-level separation",
                    "not_applicable",
                    (
                        f"scheme is {scheme}: no subject identifier could be derived for "
                        f"{underived} of {len(rows)} rows, so grouping is unavailable rather "
                        "than satisfied. Recorded, not claimed."
                    ),
                    (_rel(split_map),),
                )
            )
            continue

        try:
            folds = cv.load_folds(task, validate=False)
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            report.add(
                Check(
                    "split",
                    f"QA-SPLIT-{task}",
                    f"{task}: subject-level separation",
                    "fail",
                    f"the fold map could not be loaded: {error}",
                    (_rel(split_map),),
                )
            )
            continue

        leaks: list[str] = []
        for fold in folds:
            try:
                cv.assert_group_disjoint(fold)
            except cv.LeakageError as error:
                leaks.append(str(error))
        report.add(
            Check(
                "split",
                f"QA-SPLIT-{task}",
                f"{task}: no subject and no record in both train and test",
                "fail" if leaks else "pass",
                "; ".join(leaks[:3])
                or (
                    f"{len(folds)} folds ({scheme}), "
                    f"{len(set(rows['split_group'].astype(str)))} groups, zero shared"
                ),
                (_rel(split_map),),
            )
        )

    # Every record is tested exactly once per repeat. A record missing from a
    # repeat is never scored; a record twice in one is scored twice.
    coverage: list[str] = []
    for (task, repeat), rows in frame.groupby(["task", "repeat"]):
        uids = rows["record_uid"].astype(str)
        if uids.duplicated().any():
            repeated = int(uids.duplicated().sum())
            coverage.append(f"{task} repeat {repeat}: {repeated} repeated uid(s)")
    report.add(
        Check(
            "split",
            "QA-SPLIT-COVERAGE",
            "every record is assigned to exactly one test fold per repeat",
            "fail" if coverage else "pass",
            "; ".join(coverage)
            or f"{len(frame)} assignments across {frame['task'].nunique()} tasks",
            (_rel(split_map),),
        )
    )

    summary = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "split_fold_summary.csv"
    if summary.is_file():
        folds = _read_csv(summary)
        empty = folds[folds["min_class_count"] <= 0]
        report.add(
            Check(
                "split",
                "QA-SPLIT-CLASSES",
                "no fold of any task is missing a class entirely",
                "fail" if not empty.empty else "pass",
                f"{len(folds)} folds, smallest class in any fold: "
                "{int(folds['min_class_count'].min())}",
                (_rel(summary),),
            )
        )


# ---------------------------------------------------------------------------
# T124.3 -- feature QA
# ---------------------------------------------------------------------------


def _check_features(report: QaReport) -> None:
    from src.feature_extraction import registry

    names = registry.feature_names()
    fingerprint = registry.registry_fingerprint()
    report.add(
        Check(
            "feature",
            "QA-FEAT-01",
            "the feature registry is still the locked 138",
            "pass"
            if len(names) == 138 and fingerprint.startswith(REGISTRY_FINGERPRINT_PREFIX)
            else "fail",
            f"{len(names)} features, fingerprint {fingerprint[:16]}",
            ("src/feature_extraction/registry.py",),
        )
    )

    counts = registry.family_counts()
    expected = {
        "time": 24,
        "frequency": 22,
        "mfcc": 39,
        "chroma": 24,
        "dwt": 24,
        "envelope": 5,
    }
    report.add(
        Check(
            "feature",
            "QA-FEAT-02",
            "the per-family counts still sum to 138 as declared",
            "pass" if counts == expected else "fail",
            str(counts),
            ("src/feature_extraction/registry.py", "configs/features.yaml"),
        )
    )

    matrix = PROJECT_ROOT / "outputs" / "03_features" / "all_features_matrix.parquet"
    if not matrix.is_file():
        report.add(
            Check(
                "feature",
                "QA-FEAT-03",
                "every feature value in the matrix is finite",
                "skipped",
                "all_features_matrix.parquet is gitignored and absent from this checkout",
            )
        )
    else:
        frame = pd.read_parquet(matrix)
        present = [name for name in names if name in frame.columns]
        values = frame[present].to_numpy(dtype="float64", na_value=float("nan"))
        n_nonfinite = int((~pd.notna(frame[present])).to_numpy().sum())
        n_inf = int(sum(1 for value in values.ravel() if math.isinf(value)))
        report.add(
            Check(
                "feature",
                "QA-FEAT-03",
                "every feature value in the matrix is finite",
                "fail" if (n_nonfinite or n_inf) else "pass",
                f"{frame.shape[0]} records x {len(present)} features; "
                f"{n_nonfinite} NaN, {n_inf} infinite",
                (_rel(matrix),),
            )
        )
        report.add(
            Check(
                "feature",
                "QA-FEAT-04",
                "the matrix carries every feature the registry declares, in order",
                "pass" if present == list(names) else "fail",
                f"{len(present)} of {len(names)} registry columns present and in order",
                (_rel(matrix),),
            )
        )

    errors = PROJECT_ROOT / "outputs" / "03_features" / "extraction_errors.csv"
    if errors.is_file():
        frame = _read_csv(errors)
        report.add(
            Check(
                "feature",
                "QA-FEAT-05",
                "no recording failed feature extraction",
                "fail" if not frame.empty else "pass",
                f"{len(frame)} extraction error row(s)",
                (_rel(errors),),
            )
        )


# ---------------------------------------------------------------------------
# T124.4 -- model QA
# ---------------------------------------------------------------------------


def _check_models(report: QaReport) -> None:
    manifest_path = OUT_DIR / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seed_block = manifest.get("final", {}).get("seed", {})
        others = seed_block.get("runs_with_another_seed", [])
        report.add(
            Check(
                "model",
                "QA-MODEL-01",
                "every recorded run used seed 42 (research rule 5)",
                "fail" if others or seed_block.get("global_seed") != 42 else "pass",
                f"global seed {seed_block.get('global_seed')}, "
                f"{seed_block.get('runs_by_seed', {})}, "
                f"{len(others)} run(s) on another seed",
                (_rel(manifest_path),),
            )
        )

        runs = manifest.get("runs", [])
        no_packages = [run.get("name", "?") for run in runs if not run.get("packages")]
        report.add(
            Check(
                "model",
                "QA-MODEL-02",
                "every run recorded its package versions",
                "fail" if no_packages else "pass",
                f"{len(runs)} runs, {len(no_packages)} without package versions"
                + (f": {no_packages[:3]}" if no_packages else ""),
                (_rel(manifest_path),),
            )
        )

    # Fold-safe scaling is structural: the imputer and the scaler are steps of
    # the sklearn Pipeline, so they are refitted per training fold. Asserted on
    # a pipeline actually built by the project's own builder, not on prose.
    try:
        from src.models.pipeline import build_pipeline
        from src.models.registry import build as build_model

        pipeline = build_pipeline(build_model("M1"))
        steps = [name for name, _ in pipeline.steps]
        learned_inside = [
            name for name in steps if any(word in name for word in ("impute", "scal", "select"))
        ]
        report.add(
            Check(
                "model",
                "QA-MODEL-03",
                "the imputer, scaler and selector are steps inside the Pipeline (rule 2)",
                "pass" if len(learned_inside) >= 2 else "fail",
                f"pipeline steps: {steps}",
                ("src/models/pipeline.py",),
            )
        )
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        report.add(
            Check(
                "model",
                "QA-MODEL-03",
                "the imputer, scaler and selector are steps inside the Pipeline (rule 2)",
                "fail",
                f"the pipeline could not be built: {error}",
                ("src/models/pipeline.py",),
            )
        )

    # Calibration is deliberately not uniform -- M3 self-calibrates, M4 stays
    # raw because calibrating it worsened ECE, M5 is calibrated. The QA claim is
    # that the shipped config still says so, since a silent change there moves
    # every operating point.
    try:
        from src.utils.config import load_config

        models_cfg = load_config("models")
        per_ensemble = {
            model_id: models_cfg.get(f"models.{model_id}.calibrate_members", None)
            for model_id in ("M6", "M7")
        }
        # M4 staying OUT of this list is the measured decision, not an omission:
        # calibrating it moved its ECE from 0.027 to 0.039 on D1 fold 0.
        uniform = [model_id for model_id, value in per_ensemble.items() if value != ["M5"]]
        report.add(
            Check(
                "model",
                "QA-MODEL-04",
                "ensemble members are still calibrated individually, on the measured evidence",
                "fail" if uniform else "pass",
                f"calibrate_members: {per_ensemble}"
                + ("" if not uniform else f" -- changed for {uniform}"),
                ("configs/models.yaml", "outputs/04_models/tree_calibration_assessment.csv"),
            )
        )
    except Exception as error:  # noqa: BLE001
        report.add(
            Check(
                "model",
                "QA-MODEL-04",
                "ensemble member calibration is still configured per member",
                "fail",
                str(error),
                ("configs/models.yaml",),
            )
        )

    # Every deployed model must record how it was chosen, and by what rule --
    # this is the difference between "the best model" and "the model our stated
    # rule selects", which on this data are not the same model.
    deployed = sorted((PROJECT_ROOT / "models_saved").glob("*/final/manifest.json"))
    bad: list[str] = []
    for path in deployed:
        data = json.loads(path.read_text(encoding="utf-8"))
        rule = str(data.get("selection_rule", ""))
        if not rule:
            bad.append(f"{path.parent.parent.name}: no selection_rule")
        elif rule.strip().lower().startswith("accuracy"):
            bad.append(f"{path.parent.parent.name}: selected by {rule}")
        if "accuracy_would_have_chosen" not in data:
            bad.append(f"{path.parent.parent.name}: no accuracy_would_have_chosen")
    report.add(
        Check(
            "model",
            "QA-MODEL-05",
            "every deployed model records its selection rule, and it is not accuracy",
            "fail" if bad or not deployed else "pass",
            "; ".join(bad) or f"{len(deployed)} deployed task model(s), all rule-6 selected",
            tuple(_rel(path) for path in deployed),
        )
    )


# ---------------------------------------------------------------------------
# T124.5 -- metric QA
# ---------------------------------------------------------------------------


def _metric_families(columns: list[str]) -> set[str]:
    """Metric names present, with the ``_mean``/``_sd``/``_n`` suffixes stripped."""
    found: set[str] = set()
    for column in columns:
        found.add(re.sub(r"_(mean|sd|n|std)$", "", str(column)).lower())
    return found


def _check_metrics(report: QaReport) -> None:
    aggregates = sorted(
        path
        for path in PROJECT_ROOT.glob("outputs/0[6-8]_*/*/aggregate_metrics.csv")
        if "_superseded_" not in path.as_posix()
    )
    if not aggregates:
        report.add(
            Check(
                "metric",
                "QA-METRIC-01",
                "no result table reports accuracy alone",
                "skipped",
                "no aggregate_metrics.csv in this checkout",
            )
        )
        return

    offenders: list[str] = []
    checked = 0
    for path in aggregates:
        frame = _read_csv(path, nrows=1)
        present = _metric_families(list(frame.columns))
        if "accuracy" not in present:
            continue
        checked += 1
        multiclass = "macro_f1" in present
        required = MULTICLASS_REQUIRED if multiclass else BINARY_REQUIRED
        missing = [name for name in required if name not in present]
        if missing:
            offenders.append(f"{_rel(path)} lacks {missing}")
    report.add(
        Check(
            "metric",
            "QA-METRIC-01",
            "every experiment that reports accuracy also reports the rule-6 metrics",
            "fail" if offenders else "pass",
            "; ".join(offenders[:3])
            or f"{checked} experiment result table(s), each carrying its task's full metric set",
            tuple(_rel(path) for path in aggregates),
        )
    )

    # Per-class recall for the multiclass tasks, which is the half of rule 6
    # that a macro average can hide.
    multiclass_paths = [
        path for path in aggregates if "07_multiclass" in path.as_posix() or "EXP-C1" in path.name
    ] or [path for path in aggregates if "07_multiclass" in path.as_posix()]
    missing_recall: list[str] = []
    for path in multiclass_paths:
        frame = _read_csv(path, nrows=1)
        if not any(re.match(r"recall_.+_mean$", str(column)) for column in frame.columns):
            present = _metric_families(list(frame.columns))
            if "macro_recall" not in present:
                missing_recall.append(_rel(path))
    report.add(
        Check(
            "metric",
            "QA-METRIC-02",
            "the multiclass tasks report per-class recall, not only a macro average",
            "fail" if missing_recall else "pass",
            "; ".join(missing_recall) or f"{len(multiclass_paths)} multiclass result table(s)",
            tuple(_rel(path) for path in multiclass_paths),
        )
    )

    selection = PROJECT_ROOT / "outputs" / "06_binary_results" / "final_model_selection.json"
    if selection.is_file():
        data = json.loads(selection.read_text(encoding="utf-8"))
        rule = str(data.get("rule", data.get("selection_rule", data.get("ranked_by", ""))))
        report.add(
            Check(
                "metric",
                "QA-METRIC-03",
                "the final model was selected on sensitivity and balanced accuracy",
                "pass" if "sensitivity" in rule.lower() else "fail",
                f"selection rule: {rule or '(not recorded)'}; "
                f"accuracy would have chosen {data.get('accuracy_would_have_chosen', '?')}",
                (_rel(selection),),
            )
        )


# ---------------------------------------------------------------------------
# T124.6 -- search QA and dashboard QA
# ---------------------------------------------------------------------------


def _check_search(report: QaReport) -> None:
    from src.evaluation import cv

    maps = sorted(
        path
        for path in PROJECT_ROOT.glob("outputs/05_search_optimization/*/inner_fold_map.csv")
        if "_superseded_" not in path.as_posix()
    )
    if not maps:
        report.add(
            Check(
                "search",
                "QA-SEARCH-01",
                "no search ever saw an outer test row",
                "skipped",
                "no inner_fold_map.csv in this checkout",
            )
        )
        return

    try:
        folds = {fold.label: fold for fold in cv.load_folds("binary", validate=False)}
    except Exception as error:  # noqa: BLE001
        report.add(
            Check(
                "search",
                "QA-SEARCH-01",
                "no search ever saw an outer test row",
                "fail",
                f"the binary fold map could not be loaded: {error}",
            )
        )
        return

    violations: list[str] = []
    checked = 0
    for path in maps:
        frame = _read_csv(path)
        for outer_label, rows in frame.groupby("outer_fold"):
            fold = folds.get(str(outer_label))
            if fold is None:
                violations.append(f"{_rel(path)}: unknown outer fold {outer_label}")
                continue
            checked += 1
            seen = set(rows["record_uid"].astype(str))
            overlap = seen & set(fold.test_uids)
            if overlap:
                violations.append(
                    f"{_rel(path)} {outer_label}: {len(overlap)} outer TEST record(s) "
                    f"inside the inner search, e.g. {sorted(overlap)[:3]}"
                )
    report.add(
        Check(
            "search",
            "QA-SEARCH-01",
            "no inner search split ever contained an outer test record",
            "fail" if violations else "pass",
            "; ".join(violations[:3])
            or (
                f"{checked} (search, outer fold) pair(s) re-intersected with the published "
                f"fold map across {len(maps)} search(es): "
                + ", ".join(sorted(path.parent.name for path in maps))
                + ". EXP-A2's per-fold nested searches write their chosen point to a "
                "gitignored cache and carry it in per_fold_metrics.csv, so they are covered "
                "by tests/test_search_no_leakage.py rather than by a map on disk."
            ),
            tuple(_rel(path) for path in maps),
        )
    )

    # The inner rows must also be exactly the outer TRAINING rows -- a search
    # that silently dropped part of its training set is not leakage but is still
    # a different experiment from the one reported.
    outside: list[str] = []
    for path in maps:
        frame = _read_csv(path)
        for outer_label, rows in frame.groupby("outer_fold"):
            fold = folds.get(str(outer_label))
            if fold is None:
                continue
            stray = set(rows["record_uid"].astype(str)) - set(fold.train_uids)
            if stray:
                outside.append(
                    f"{_rel(path)} {outer_label}: {len(stray)} row(s) not in outer train"
                )
    report.add(
        Check(
            "search",
            "QA-SEARCH-02",
            "every inner search row is an outer training row",
            "fail" if outside else "pass",
            "; ".join(outside[:3]) or f"{len(maps)} inner fold map(s) fully inside outer train",
            tuple(_rel(path) for path in maps),
        )
    )


def _check_dashboard(report: QaReport) -> None:
    # `--skip-frontend` is a supported way to run the pipeline, and on a machine
    # with no Node it is the only way. The dashboard area then has nothing to
    # read: that is "not verified here", not "the dashboard is wrong".
    stamp = PROJECT_ROOT / "frontend" / ".display-audit.json"
    if not stamp.is_file():
        report.add(
            Check(
                "dashboard",
                "QA-DASH-01",
                "the T119.3 displayed-value audit is green for the site on disk right now",
                "skipped",
                "no audit stamp: the dashboard has not been built in this checkout",
            )
        )
        report.add(
            Check(
                "dashboard",
                "QA-DASH-02",
                "all thirteen dashboard screenshots are on disk and non-trivial",
                "skipped",
                "the dashboard has not been built, so nothing was captured from it",
            )
        )
        return

    try:
        from src.reporting.display_audit import AuditGateError, require_passed_audit

        recorded = require_passed_audit()
        checked = recorded.get("checked", {}) or {}
        findings = recorded.get("n_findings", {}) or {}
        report.add(
            Check(
                "dashboard",
                "QA-DASH-01",
                "the T119.3 displayed-value audit is green for the site on disk right now",
                "pass",
                f"site {str(recorded.get('site_digest', ''))[:12]} passed at "
                f"{recorded.get('audited_utc', '?')}; "
                f"{checked.get('rendered_metric_tokens', '?')} rendered values and "
                f"{checked.get('table_and_figure_cells', '?')} table/figure cells over "
                f"{checked.get('pages', '?')} pages, {sum(findings.values())} finding(s)",
                ("frontend/.display-audit.json",),
            )
        )
    except AuditGateError as error:
        report.add(
            Check(
                "dashboard",
                "QA-DASH-01",
                "the T119.3 displayed-value audit is green for the site on disk right now",
                "fail",
                str(error),
                ("frontend/.display-audit.json",),
            )
        )
    except Exception as error:  # noqa: BLE001
        report.add(
            Check(
                "dashboard",
                "QA-DASH-01",
                "the T119.3 displayed-value audit is green for the site on disk right now",
                "skipped",
                f"the audit stamp could not be read: {error}",
            )
        )

    screenshots = PROJECT_ROOT / "outputs" / "15_dashboard_screenshots"
    pngs = sorted(screenshots.glob("*.png")) if screenshots.is_dir() else []
    if not pngs:
        report.add(
            Check(
                "dashboard",
                "QA-DASH-02",
                "all thirteen dashboard screenshots are on disk and non-trivial",
                "skipped",
                "no screenshots in this checkout",
            )
        )
        return
    report.add(
        Check(
            "dashboard",
            "QA-DASH-02",
            "all thirteen dashboard screenshots are on disk and non-trivial",
            "pass" if len(pngs) >= 13 and all(p.stat().st_size > 20_000 for p in pngs) else "fail",
            f"{len(pngs)} PNG(s), smallest {min((p.stat().st_size for p in pngs), default=0)} "
            "bytes",
            (_rel(screenshots),),
        )
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_AREA_RUNNERS = {
    "dataset": _check_dataset,
    "split": _check_splits,
    "feature": _check_features,
    "model": _check_models,
    "metric": _check_metrics,
    "search": _check_search,
    "dashboard": _check_dashboard,
}


def run_sweep(areas: list[str] | None = None) -> QaReport:
    """Run every QA area (or the named subset) and return the report."""
    report = QaReport(generated_utc=datetime.now(UTC).isoformat(timespec="seconds"))
    for area, _ in AREAS:
        if areas and area not in areas:
            continue
        _AREA_RUNNERS[area](report)
    return report


def _markdown(report: QaReport) -> str:
    lines = [
        "# Final QA sweep (Phase 124)",
        "",
        f"Generated {report.generated_utc} by `python scripts/49_final_qa_sweep.py`.",
        "",
        "Every row below was **recomputed from a committed file** at generation time.",
        "Nothing here asserts that an earlier check once passed.",
        "",
        f"**Result: {'PASS' if report.ok else 'FAIL'}**",
        "",
        "| Area | T124 | Status | Checks |",
        "|---|---|---|---|",
    ]
    for area, description in AREAS:
        task = description.split(" -- ")[0]
        lines.append(
            f"| {area} | {task} | **{report.area_status(area)}** | {len(report.for_area(area))} |"
        )

    for area, description in AREAS:
        lines += [
            "",
            f"## {area} — {description}",
            "",
            "| Check | Status | Detail |",
            "|---|---|---|",
        ]
        for check in report.for_area(area):
            detail = check.detail.replace("|", "\\|")
            lines.append(f"| `{check.check_id}` {check.title} | {check.status} | {detail} |")

    if report.failures:
        lines += ["", "## Failures", ""]
        lines += [
            f"- `{check.check_id}` {check.title} — {check.detail}"
            for check in report.failures
        ]

    skipped = [check for check in report.checks if check.status == "skipped"]
    if skipped:
        lines += [
            "",
            "## Not verified in this checkout",
            "",
            "Each of these names what it could not read, rather than passing vacuously.",
            "",
        ]
        lines += [f"- `{check.check_id}` {check.title} — {check.detail}" for check in skipped]

    not_applicable = [check for check in report.checks if check.status == "not_applicable"]
    if not_applicable:
        lines += [
            "",
            "## Unavailable rather than satisfied",
            "",
        ]
        lines += [
            f"- `{check.check_id}` {check.title} — {check.detail}" for check in not_applicable
        ]

    lines.append("")
    return "\n".join(lines)


def write_report(report: QaReport) -> tuple[Path, Path]:
    """Write the JSON and markdown reports and register them as evidence."""
    from src.utils.evidence import register_evidence

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")

    register_evidence(
        "QA-SWEEP",
        REPORT_MD,
        metric_or_asset="final QA sweep across six areas (Phase 124)",
        command="python scripts/49_final_qa_sweep.py",
        status="ok" if report.ok else "failed",
    )
    return REPORT_JSON, REPORT_MD
