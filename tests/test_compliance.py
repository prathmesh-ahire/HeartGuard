"""Phase 125: the compliance review, and whether it can still say no.

Every check here is green on the real repository, which is the least
informative possible state for a test suite to confirm. So each check is shown
a repository that violates it and must report the violation -- and, just as
importantly, is shown the legitimate lookalike and must stay quiet:

  "does not diagnose"                    is not a diagnostic claim
  "the PhysioNet diagnosis track"        is a dataset name
  "a one-vs-rest treatment"              is not a clinical act
  "Okabe-Ito guarantees eight hues"      is not a clinical certainty claim
  objective 4's "medical diagnostic systems"  is locked wording, quoted

A word list that cannot tell those apart produces a review nobody reads, so the
false-positive cases are tested as carefully as the true ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.reporting import compliance
from src.reporting.compliance import ComplianceReport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = PROJECT_ROOT / "outputs" / "00_evidence_index" / "compliance_review.json"


def _fires(line: str, *, in_code: bool = False, rule: str | None = None) -> bool:
    """Does any rule (or the named one) fire on this line, with no neighbours?"""
    for candidate in compliance.LANGUAGE_RULES:
        if rule and candidate.name != rule:
            continue
        if candidate.hits(line, window=line, in_code=in_code):
            return True
    return False


class TestLanguageRulesFire:
    @pytest.mark.parametrize(
        "line",
        [
            "PulseVision diagnoses the recording as abnormal.",
            "The model provides a diagnosis for each patient.",
            "Use this as a diagnostic tool in the clinic.",
            "The recommended treatment for a positive patient is referral.",
            "The system prescribes a follow-up echocardiogram.",
            "This replaces a cardiologist for screening visits.",
            "PV-MEPCG is a medical device for murmur screening.",
            "The method is clinically validated on paediatric patients.",
            "A positive result confirms the diagnosis.",
            "The screen detects disease before symptoms appear.",
        ],
    )
    def test_a_claim_is_reported(self, line: str) -> None:
        assert _fires(line), f"no rule fired on: {line}"

    @pytest.mark.parametrize(
        "line",
        [
            "It does not diagnose, treat or prescribe.",
            "This is not a diagnostic tool and not a medical device.",
            "PV-MEPCG never replaces a clinician's assessment.",
            "No diagnosis, treatment or prescription is made or implied.",
            "Academic screening prototype; it is not a diagnosis.",
        ],
    )
    def test_a_denial_is_not_a_claim(self, line: str) -> None:
        assert not _fires(line), f"a denial was reported as a claim: {line}"

    @pytest.mark.parametrize(
        "line",
        [
            "PhysioNet carries a diagnosis track of 3,240 records.",
            "The run needs a one-vs-rest treatment, which is G17's job.",
            "Okabe-Ito guarantees the eight hues stay distinguishable.",
            "StratifiedGroupKFold guarantees this cannot happen.",
            "These flags are diagnostics and grouping keys.",
            "A PCG's diagnostic content lives at 20-400 Hz.",
        ],
    )
    def test_the_legitimate_lookalikes_stay_quiet(self, line: str) -> None:
        assert not _fires(line, in_code=True), f"false positive on: {line}"

    def test_prose_gets_no_code_latitude(self) -> None:
        """The corpus-field allowance is for code, not for reader-facing prose."""
        line = 'frame["diagnosis_class"] = frame["diagnosis"]'
        assert not _fires(line, in_code=True)

    def test_the_rule_file_is_exempt_but_only_it(self) -> None:
        assert "src/reporting/compliance.py" in compliance.SELF_EXEMPT
        assert "src/reporting/objectives.py" in compliance.SELF_EXEMPT
        assert len(compliance.SELF_EXEMPT) <= 5, "the exemption list is growing into a loophole"

    def test_a_quoted_objective_is_not_a_claim(self) -> None:
        quotations = compliance._locked_quotations()
        assert quotations, "the locked objectives could not be read"
        objective = quotations[3]
        assert compliance._is_quoted_objective(objective, quotations)
        assert not compliance._is_quoted_objective("we provide a diagnosis", quotations)

    def test_a_quotation_inside_one_table_cell_is_recognised(self) -> None:
        quotations = compliance._locked_quotations()
        row = f"| 4 | Short, noisy recordings | {quotations[3]} | PhysioNet | evidence |"
        assert compliance._is_quoted_objective(row, quotations)


class TestPlantedViolations:
    def test_a_hand_typed_number_in_a_table_is_caught(self, tmp_path: Path) -> None:
        table = tmp_path / "outputs" / "06_binary_results"
        table.mkdir(parents=True)
        pd.DataFrame([{"model_id": "M1", "sensitivity_mean": 0.8479933}]).to_csv(
            table / "T99_fake.csv", index=False
        )
        (table / "T99_fake.meta.json").write_text(
            json.dumps(
                {"command": "python scripts/x.py", "sources": [{"path": "a", "exists": True}]}
            ),
            encoding="utf-8",
        )
        (table / "T99_fake.md").write_text(
            "| Model | Sensitivity |\n|---|---|\n| M1 | 0.999 |\n", encoding="utf-8"
        )
        report = ComplianceReport()
        compliance._check_generated(report, tmp_path)
        findings = report.for_check("generated")
        assert findings and "0.999" in findings[0].text

    def test_a_number_that_does_re_derive_is_not_caught(self, tmp_path: Path) -> None:
        table = tmp_path / "outputs" / "06_binary_results"
        table.mkdir(parents=True)
        pd.DataFrame([{"model_id": "M1", "sensitivity_mean": 0.8479933}]).to_csv(
            table / "T99_fake.csv", index=False
        )
        (table / "T99_fake.meta.json").write_text(
            json.dumps(
                {"command": "python scripts/x.py", "sources": [{"path": "a", "exists": True}]}
            ),
            encoding="utf-8",
        )
        (table / "T99_fake.md").write_text(
            "| Model | Sensitivity |\n|---|---|\n| M1 | 0.848 |\n", encoding="utf-8"
        )
        report = ComplianceReport()
        compliance._check_generated(report, tmp_path)
        assert not report.for_check("generated")

    def test_a_table_with_no_sidecar_is_caught(self, tmp_path: Path) -> None:
        table = tmp_path / "outputs" / "06_binary_results"
        table.mkdir(parents=True)
        pd.DataFrame([{"a": 1}]).to_csv(table / "T99_fake.csv", index=False)
        (table / "T99_fake.md").write_text("| a |\n|---|\n| 1 |\n", encoding="utf-8")
        report = ComplianceReport()
        compliance._check_generated(report, tmp_path)
        assert any("sidecar" in finding.why for finding in report.for_check("generated"))

    def test_a_near_perfect_metric_with_no_explanation_is_caught(self, tmp_path: Path) -> None:
        experiment = tmp_path / "outputs" / "06_binary_results" / "EXP-FAKE"
        experiment.mkdir(parents=True)
        pd.DataFrame([{"model_id": "M1", "sensitivity_mean": 0.9999}]).to_csv(
            experiment / "aggregate_metrics.csv", index=False
        )
        report = ComplianceReport()
        compliance._check_perfection(report, tmp_path)
        findings = report.for_check("perfection")
        assert findings and "leaked subject" in findings[0].why

    def test_a_missing_disclaimer_is_caught(self, tmp_path: Path) -> None:
        report = ComplianceReport()
        compliance._check_disclaimer(report, tmp_path)
        assert report.for_check("disclaimer"), "an empty tree passed the disclaimer check"

    def test_a_documented_count_that_contradicts_the_audit_is_caught(self, tmp_path: Path) -> None:
        audit = tmp_path / "outputs" / "01_dataset_audit"
        audit.mkdir(parents=True)
        pd.DataFrame({"dataset_source": ["D1"] * 10}).to_csv(
            audit / "metadata_master.csv", index=False
        )
        report = ComplianceReport()
        compliance._check_counts(report, tmp_path)
        assert any("D1" in finding.text for finding in report.for_check("counts"))


class TestExplainedPerfection:
    def test_every_explained_value_is_real(self) -> None:
        """An allowlist entry for a value that does not exist is a fiction.

        The first version of this dict named `EXP-C1-two_class:M4`, which has no
        near-perfect metric at all -- the 0.9900 belongs to M3. An entry nobody
        can find in the data is exactly the kind of unfalsifiable claim the
        review exists to prevent.
        """
        for key in compliance.EXPLAINED_PERFECTION:
            experiment, model, column = key.split(":")
            matches = list(PROJECT_ROOT.glob(f"outputs/*/{experiment}/aggregate_metrics.csv"))
            if not matches:
                pytest.skip(f"{experiment} is not in this checkout")
            frame = pd.read_csv(matches[0])
            row = frame[frame["model_id"].astype(str) == model]
            assert not row.empty, f"{key}: no such model row"
            value = float(row.iloc[0][column])
            assert value >= compliance.PERFECTION_THRESHOLD, (
                f"{key} is {value}, below the threshold -- this entry explains nothing"
            )

    def test_the_threshold_is_below_the_highest_real_metric(self) -> None:
        """A threshold above every value in the project cannot ever fire."""
        assert compliance.PERFECTION_THRESHOLD <= 0.99


class TestLiveReview:
    """T125.7 -- the gate, against the real repository."""

    @pytest.fixture(scope="class")
    def report(self) -> dict:
        if not REPORT_JSON.is_file():
            pytest.skip("the compliance review has not been run in this checkout")
        return json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    def test_the_recorded_review_is_clean(self, report: dict) -> None:
        summary = {
            finding["check"]: finding["where"] + ":" + str(finding["line"])
            for finding in report["findings"]
        }
        assert report["ok"], f"compliance findings: {summary}"

    def test_every_check_ran(self, report: dict) -> None:
        for name, _ in compliance.CHECKS:
            assert name in report["checks"]
            assert report["checks"][name]["note"], f"{name} recorded no scope"

    def test_the_review_scanned_a_real_amount_of_the_repository(self, report: dict) -> None:
        """A scan that silently matched nothing would also report zero findings."""
        assert report["scanned"]["language_files"] > 100
        assert report["scanned"]["language_lines"] > 10_000
        assert report["scanned"]["rendered_tables"] > 20
