"""Phase 124: the QA sweep itself has to be able to fail.

A sweep that reports green is worth exactly as much as its ability to report
red, so most of what is tested here is the failure path: a leaking fold, an
accuracy-only result table, a deployed model selected on accuracy, a corrupt
file in the audit. Each is planted and the sweep is asked to notice.

The live run against the real `outputs/` is one test at the end. It is the
`[TEST]` gate T124.7 asks for, and it skips rather than fails on a checkout
that has no `outputs/` -- CI has none.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.reporting import final_qa
from src.reporting.final_qa import Check, QaReport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = PROJECT_ROOT / "outputs" / "00_evidence_index" / "final_qa_report.json"


def _check(area: str = "dataset", status: str = "pass") -> Check:
    return Check(area, f"QA-{area.upper()}-X", "a check", status, "detail")


class TestRollup:
    def test_one_failure_fails_the_sweep(self) -> None:
        report = QaReport()
        for area, _ in final_qa.AREAS:
            report.add(_check(area))
        assert report.ok
        report.add(_check("dataset", "fail"))
        assert not report.ok
        assert report.area_status("dataset") == "fail"

    def test_an_area_with_no_checks_is_not_a_pass(self) -> None:
        """An area that ran nothing has verified nothing.

        The failure mode this guards is a runner that raises early, leaves its
        area empty, and lets a report full of green rows call itself complete.
        """
        report = QaReport()
        for area, _ in final_qa.AREAS[:-1]:
            report.add(_check(area))
        assert not report.ok
        assert report.area_status(final_qa.AREAS[-1][0]) == "empty"

    def test_a_skip_is_visible_but_not_a_failure(self) -> None:
        report = QaReport()
        for area, _ in final_qa.AREAS:
            report.add(_check(area))
        report.add(_check("feature", "skipped"))
        assert report.ok
        assert report.area_status("feature") == "pass_with_skips"

    def test_not_applicable_does_not_masquerade_as_a_pass(self) -> None:
        """PASCAL A's unavailable grouping must not read as a satisfied one."""
        report = QaReport()
        report.add(_check("split", "not_applicable"))
        assert not report.for_area("split")[0].passed
        assert not report.for_area("split")[0].failed

    def test_the_dict_counts_every_status(self) -> None:
        report = QaReport()
        report.add(_check("split", "pass"))
        report.add(_check("split", "fail"))
        report.add(_check("split", "skipped"))
        report.add(_check("split", "not_applicable"))
        counts = report.to_dict()["counts"]
        assert counts == {"pass": 1, "fail": 1, "skipped": 1, "not_applicable": 1}


class TestPlantedFailures:
    """Each check is shown a broken repository and must report it."""

    def test_a_leaking_fold_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.evaluation import cv

        leaking = cv.Fold(
            task="binary",
            scheme="repeated_5x5_grouped",
            repeat=0,
            fold=0,
            train_uids=("a", "b"),
            test_uids=("b", "c"),
            train_groups=("s1", "s2"),
            test_groups=("s2", "s3"),
        )
        monkeypatch.setattr(cv, "load_folds", lambda task, validate=True: (leaking,))

        report = QaReport()
        final_qa._check_splits(report)
        binary = [check for check in report.for_area("split") if check.check_id.endswith("binary")]
        assert binary and binary[0].failed
        assert "both train and test" in binary[0].title

    def test_an_accuracy_only_result_table_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_root = tmp_path
        experiment = fake_root / "outputs" / "06_binary_results" / "EXP-FAKE"
        experiment.mkdir(parents=True)
        pd.DataFrame(
            [{"exp_id": "EXP-FAKE", "model_id": "M1", "accuracy_mean": 0.9, "accuracy_sd": 0.01}]
        ).to_csv(experiment / "aggregate_metrics.csv", index=False)
        monkeypatch.setattr(final_qa, "PROJECT_ROOT", fake_root)

        report = QaReport()
        final_qa._check_metrics(report)
        first = report.for_area("metric")[0]
        assert first.failed
        assert "sensitivity" in first.detail

    def test_a_model_deployed_on_accuracy_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deployed = tmp_path / "models_saved" / "binary" / "final"
        deployed.mkdir(parents=True)
        (deployed / "manifest.json").write_text(
            json.dumps({"selection_rule": "accuracy", "accuracy_would_have_chosen": "M3"}),
            encoding="utf-8",
        )
        monkeypatch.setattr(final_qa, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(final_qa, "OUT_DIR", tmp_path / "nowhere")

        report = QaReport()
        final_qa._check_models(report)
        selection = [c for c in report.for_area("model") if c.check_id == "QA-MODEL-05"]
        assert selection and selection[0].failed
        assert "selected by accuracy" in selection[0].detail

    def test_an_unreadable_file_fails_but_a_quality_flag_does_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The distinction the audit's own report makes, and the sweep must keep.

        63 clipped recordings and one silent one are properties of a real
        corpus. A file that cannot be read is a defect. Collapsing the two
        would either hide the defect or fail the project for the dataset being
        what it is.
        """
        audit = tmp_path / "outputs" / "01_dataset_audit"
        audit.mkdir(parents=True)
        monkeypatch.setattr(final_qa, "PROJECT_ROOT", tmp_path)

        pd.DataFrame([{"record_uid": "x", "problem": "clipped"}]).to_csv(
            audit / "missing_corrupt_files.csv", index=False
        )
        report = QaReport()
        final_qa._check_dataset(report)
        assert report.for_area("dataset")[0].passed

        pd.DataFrame([{"record_uid": "x", "problem": "unreadable"}]).to_csv(
            audit / "missing_corrupt_files.csv", index=False
        )
        report = QaReport()
        final_qa._check_dataset(report)
        assert report.for_area("dataset")[0].failed

    def test_a_search_that_saw_an_outer_test_row_is_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.evaluation import cv

        search = tmp_path / "outputs" / "05_search_optimization" / "SO-FAKE"
        search.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "outer_fold": "r0f0",
                    "inner_fold": 0,
                    "role": "train",
                    "row": 0,
                    "record_uid": "a",
                },
                {"outer_fold": "r0f0", "inner_fold": 0, "role": "val", "row": 1, "record_uid": "z"},
            ]
        ).to_csv(search / "inner_fold_map.csv", index=False)

        fold = cv.Fold(
            task="binary",
            scheme="repeated_5x5_grouped",
            repeat=0,
            fold=0,
            train_uids=("a", "b"),
            test_uids=("z",),
            train_groups=("s1", "s2"),
            test_groups=("s3",),
        )
        monkeypatch.setattr(final_qa, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(cv, "load_folds", lambda task, validate=True: (fold,))

        report = QaReport()
        final_qa._check_search(report)
        first = report.for_area("search")[0]
        assert first.failed
        assert "outer TEST record" in first.detail


class TestMetricFamilies:
    def test_suffixes_are_stripped_so_a_metric_is_recognised(self) -> None:
        families = final_qa._metric_families(
            ["accuracy_mean", "accuracy_sd", "sensitivity_n", "roc_auc_mean"]
        )
        assert {"accuracy", "sensitivity", "roc_auc"} <= families

    def test_rule_six_requires_more_than_balanced_accuracy(self) -> None:
        assert "sensitivity" in final_qa.BINARY_REQUIRED
        assert "specificity" in final_qa.BINARY_REQUIRED
        assert "macro_f1" in final_qa.MULTICLASS_REQUIRED


class TestLiveSweep:
    """T124.7 -- the gate, run against the real committed outputs."""

    @pytest.fixture(scope="class")
    def report(self) -> dict:
        if not REPORT_JSON.is_file():
            pytest.skip("the QA sweep has not been run in this checkout")
        return json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    def test_the_recorded_sweep_is_green(self, report: dict) -> None:
        failing = [check for check in report["checks"] if check["status"] == "fail"]
        assert report["ok"], f"failing checks: {[c['check_id'] for c in failing]}"

    def test_all_six_areas_ran(self, report: dict) -> None:
        for area, _ in final_qa.AREAS:
            assert report["areas"][area]["checks"] > 0, f"{area} produced no checks"

    def test_the_report_names_its_evidence(self, report: dict) -> None:
        """A check with no evidence path cannot be re-verified by a reader."""
        without = [
            check["check_id"]
            for check in report["checks"]
            if not check["evidence"] and check["status"] not in {"skipped", "not_applicable"}
        ]
        assert not without, f"checks with no evidence path: {without}"
