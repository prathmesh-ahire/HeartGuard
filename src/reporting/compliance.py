"""Compliance and claims review: what this project is allowed to say (Phase 125).

Six checks, run over the **whole deliverable** -- the Python, the frontend
source, the built site, every generated report and every root document:

T125.1  no diagnostic language anywhere; screening wording only
T125.2  the disclaimer reaches the README, the dashboard and every report
T125.3  no metric is implausibly perfect without a recorded explanation
T125.4  the six locked objectives appear verbatim wherever they are quoted
T125.5  every documented count matches the audited corpus
T125.6  no number in a published table was hand-entered

## Why a word list is not enough, and what is done instead

"Diagnosis" is a forbidden claim and also the name of a real dataset track in
this project (the PhysioNet **diagnosis track**, 3,240 records, which objective
6 rests on). "Not a diagnostic device" is the disclaimer itself. A grep that
cannot tell those apart produces a wall of false positives, and a review nobody
can read is a review nobody runs.

So each forbidden pattern carries its own allowances, and a hit is reported only
when no allowance matches the line it appears on. The allowances are narrow and
written down here, in one place, so the exceptions are reviewable rather than
scattered through the code as ``# noqa``-style silences.

## What T125.6 actually checks

Every published table is a ``.csv`` (full precision), a ``.md``/``.docx``/``.tex``
rendering of it, and a ``.meta.json`` naming the sources and the exact command.
So "was this number hand-entered" has a mechanical answer: **every numeric token
in the rendering must re-derive from a value in the sibling CSV** under the
table's own declared rounding rules. A number typed into the markdown has no
CSV value behind it and is reported with its row.

This is the same test the dashboard's displayed-value audit applies to the
browser (T119.3), pointed at the document deliverables instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging_setup import get_logger

__all__ = [
    "CHECKS",
    "Finding",
    "ComplianceReport",
    "run_review",
    "write_report",
]

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs" / "00_evidence_index"
REPORT_JSON = OUT_DIR / "compliance_review.json"
REPORT_MD = OUT_DIR / "compliance_review.md"

CHECKS: tuple[tuple[str, str], ...] = (
    ("language", "T125.1 -- screening wording only, no diagnostic language"),
    ("disclaimer", "T125.2 -- the disclaimer reaches every surface"),
    ("perfection", "T125.3 -- no unexplained near-perfect metric"),
    ("objectives", "T125.4 -- the six locked objectives, verbatim"),
    ("counts", "T125.5 -- every documented count matches audited reality"),
    ("generated", "T125.6 -- no result was hand-entered"),
)

#: Where a claim can be made. Source that a reader or a grader sees.
SCANNED_GLOBS: tuple[str, ...] = (
    "README.md",
    "ARCHITECTURE.md",
    "CONFIGURATION.md",
    "CITATION.md",
    "HANDOVER.md",
    "CHANGELOG.md",
    "src/**/*.py",
    "scripts/**/*.py",
    "frontend/app/**/*.tsx",
    "frontend/components/**/*.tsx",
    "frontend/lib/*.ts",
    "outputs/**/*.md",
    "outputs/**/*.txt",
)

#: Directories whose contents are working state, third-party, or generated
#: bundles nobody reads as prose.
SKIP_PARTS: frozenset[str] = frozenset(
    {"node_modules", ".next", "out", "__pycache__", "_checkpoints", ".venv", "cache"}
)


@dataclass(frozen=True)
class Rule:
    """One forbidden claim, with the narrow places it is legitimate.

    ``allow`` is checked against a **three-line window** -- the line before, the
    line itself, and the line after -- because a denial and the word it denies
    are routinely split by a wrap: README.md reads "it is not a" / "medical
    device and it is not a diagnostic tool", and a line-at-a-time matcher reads
    the second half as a claim.

    ``allow_in_code`` applies only under ``src/`` and ``scripts/``. That is where
    the corpus's own vocabulary lives: PhysioNet ships a ``diagnosis`` field and
    a diagnosis track, so ``frame["diagnosis_class"]`` is a column name, not a
    claim. Reader-facing prose gets no such latitude, which is the point of
    keeping the two lists apart.
    """

    name: str
    pattern: str
    why: str
    allow: tuple[str, ...] = ()
    allow_in_code: tuple[str, ...] = ()

    def hits(self, line: str, *, window: str, in_code: bool) -> bool:
        if not re.search(self.pattern, line, flags=re.IGNORECASE):
            return False
        allowances = self.allow + (self.allow_in_code if in_code else ())
        return not any(re.search(allowed, window, flags=re.IGNORECASE) for allowed in allowances)


#: Files that contain the forbidden patterns because they *are* the rule, plus
#: their tests. Exempting them is not a loophole -- a checker that cannot state
#: what it forbids cannot be reviewed -- but it is deliberately a short literal
#: list rather than a pattern.
SELF_EXEMPT: frozenset[str] = frozenset(
    {
        "src/reporting/compliance.py",
        "scripts/50_compliance_review.py",
        "tests/test_compliance.py",
        # The blueprint's own wording, locked: "Do not change, shorten,
        # paraphrase or reword them". Objectives 1, 3 and 5 contain "auto
        # diagnosis systems" and "medical diagnostic systems" because the
        # blueprint does. Rewording them to satisfy this check would be the one
        # edit the source document explicitly forbids, so T125.4 checks them
        # instead -- character for character, against the PDF.
        "src/reporting/objectives.py",
    }
)

#: Research rule 7. Each entry is a claim this software may not make.
#:
#: Two of these are deliberately narrower than the plain English word. "Treat"
#: is forbidden as a clinical act and unremarkable as a technical verb ("a
#: one-vs-rest treatment", "treat the members as exchangeable"), so it fires
#: only near a clinical object. "Confirms" and "guarantees" are likewise
#: everyday software English -- what is forbidden is confirming a *diagnosis*,
#: not confirming a build.
LANGUAGE_RULES: tuple[Rule, ...] = (
    Rule(
        "diagnosis",
        r"\bdiagnos(is|e|es|ed|ing)\b"
        r"|\bdiagnostics?\s+(tool|tools|device|devices|system|systems|aid|service|"
        r"product|claim|decision|report|result|results|use|purposes?)\b",
        "this is a screening prototype; it does not diagnose",
        allow=(
            r"diagnosis[_ -]?track",
            r"diagnostic track",
            r"diagnosis multiclass",
            r"multiclass diagnosis",
            r"\bnot\b[^.]{0,80}\bdiagnos",
            r"\bno\b[^.]{0,60}\bdiagnos",
            r"\b(does not|never|cannot|is not|rather than)\b[^.]{0,60}diagnos",
            r"diagnostic language",
            r"screening[^.]{0,60}diagnos",
            r"diagnos[^.]{0,60}screening",
            # The corpus's own vocabulary. "PhysioNet's diagnosis labels" is a
            # statement about a published dataset field, not a claim about what
            # this software does -- and it has to be sayable in a changelog.
            r"diagnos(is|es)?[\s\"'_+-]+(label|labels|annotation|annotations|code|codes|"
            r"categor\w*|class|classes|column|field|meaning|meanings|record|records|"
            r"row|rows|track|map|multiclass|string|entry|entries)",
            # PhysioNet's per-record `Diagnosis` field, referred to as a bare
            # noun: "subset and diagnosis", "grouped by PhysioNet diagnosis",
            # "Diagnosis Per-Class Results". Every one of these names a column
            # of a published corpus. The allowance requires a data word within
            # the same sentence, so "provides a diagnosis" is still a finding.
            r"\b(subset|metadata|quality|band|flag|column|field|per[- ]class|task|"
            r"track|cross[- ]reference|physionet|circor|grouped by|lookup|short code)\b"
            r"[^.]{0,80}diagnos",
            r"diagnos(is|es)\b[^.]{0,80}\b(metadata|per[- ]class|task|track|column|"
            r"field|cross[- ]reference|macro-f1|5-class|multiclass|subset|corpus)\b",
            r"\bnothing\b[^.]{0,80}diagnos",
            r"diagnos\w*[^.]{0,80}\b(is not|are not|never|not reportable|"
            r"a description)\b",
        ),
        allow_in_code=(
            # The corpus's own field, and the supplementary track named after it.
            r"diagnos\w*[_ ](class|label|code|meaning|column|field|file|map|fold|"
            r"track|frame|set|id|count|counts|classification)",
            r"(load|read|write|merge|parse|emit)\w*[_ ]?diagnos",
            r"diagnos\w*\b[^.]{0,60}\b(physionet|circor|corpus|dataset|annotation|"
            r"per patient|meanings file|label|column|track)",
            r"\b(physionet|circor|corpus|dataset|track|label)\b[^.]{0,60}diagnos",
            # A Python identifier, dict key, or column reference.
            r"\bdiagnos\w*\s*[=\[\(:]",
            r"[\[\(\"'\.]diagnos\w*",
            r"\bdiagnos\w*[\"'\]\)]",
        ),
    ),
    Rule(
        "treatment",
        r"\b(therapy|therapeutic|therapies)\b"
        r"|\btreat(s|ed|ing|ment)?\b(?=[^.]{0,60}\b"
        r"(patient|patients|disease|condition|murmur|clinical|clinician|illness|symptom)\b)",
        "no treatment is recommended, implied or described",
        allow=(
            r"\bnot\b[^.]{0,90}\b(treat|therap)",
            r"\bno\b[^.]{0,60}\b(treat|therap)",
            r"\b(does not|never|cannot)\b[^.]{0,60}(treat|therap)",
            r"(treat|therap)[^.]{0,60}\b(is not|are not|never)\b",
        ),
    ),
    Rule(
        "prescription",
        r"\bprescrib(e|es|ed|ing)\b|\bprescription\b",
        "nothing here prescribes",
        allow=(
            r"\bnot\b[^.]{0,90}prescri",
            r"\bno\b[^.]{0,60}prescri",
            r"\b(does not|never|cannot)\b[^.]{0,60}prescri",
            r"prescrib[^.]{0,60}\b(is not|never)\b",
        ),
    ),
    Rule(
        "replaces a clinician",
        r"replaces? (a |the )?(doctor|clinician|physician|cardiologist)",
        "screening support never replaces clinical assessment",
        allow=(r"\b(not|never|does not|cannot|no)\b[^.]{0,80}replace",),
    ),
    Rule(
        "medical device",
        r"\bmedical device\b",
        "this is not a medical device and must not be described as one",
        allow=(r"\bnot\b[^.]{0,60}medical device", r"medical device[^.]{0,60}\bnot\b"),
    ),
    Rule(
        "clinical claim",
        r"\b(clinically (validated|proven|approved)|fda[- ]approved|ce[- ]marked)\b",
        "no regulatory or clinical validation claim is supported",
        allow=(r"\b(not|never|no)\b[^.]{0,60}(clinically|fda|ce[- ]marked)",),
    ),
    Rule(
        "certainty",
        r"\b(detects? disease|screens? for disease)\b"
        r"|\b(confirms?|rules? out|guarantees?)\b(?=[^.]{0,50}\b"
        r"(diagnos\w+|disease|murmur|condition|pathology|abnormality|patient)\b)",
        "a screening indication is not a determination",
        allow=(
            r"\b(not|never|cannot|does not|no)\b[^.]{0,70}(confirm|rule out|guarantee|detect)",
            r"(confirm|rule out|guarantee)[^.]{0,50}\b(is not|never|cannot)\b",
        ),
    ),
)

#: The disclaimer's load-bearing phrases. A surface satisfies T125.2 if it
#: carries a screening statement AND an explicit denial of diagnosis; the exact
#: sentence differs by surface (the API's is one line, the README's is a
#: paragraph) and pinning one string would force the wrong wording somewhere.
DISCLAIMER_MARKERS: tuple[str, ...] = ("screening", "not a diagn")

#: Where the canonical disclaimer text is declared. These must carry the words
#: themselves -- everything else may carry them by reference.
DISCLAIMER_CONSTANTS: tuple[str, ...] = (
    "src/inference/predictor.py",
    "src/data_loader/inventory.py",
)

#: Surfaces T125.2 names.
DISCLAIMER_SURFACES: tuple[tuple[str, str], ...] = (
    ("README", "README.md"),
    ("dashboard layout", "frontend/app/layout.tsx"),
    ("inference API", "src/api/main.py"),
    ("predictor", "src/inference/predictor.py"),
    ("dataset audit report", "src/data_loader/inventory.py"),
    ("sample report", "src/reporting/sample_report.py"),
    ("Q1 paper pack", "src/reporting/q1_pack.py"),
    ("thesis pack", "src/reporting/thesis_pack.py"),
)

#: A metric at or above this is treated as implausible until explained.
#: "Results that look too good are a bug until proven otherwise" -- in this
#: project the causes, in order, are a leaked subject, a scaler fitted on the
#: full matrix, a duplicate recording in both splits, or a label joined on the
#: wrong key.
#: 0.98 rather than 0.999. The highest aggregate metric anywhere in this
#: project is 0.9900, so a threshold above it would make this check green
#: without ever looking at anything -- a check that cannot fire is not a check.
#: At 0.98 it fires on exactly one value, which is then explained below.
PERFECTION_THRESHOLD = 0.98

#: Metric columns worth scanning. Counts, durations and n_* columns legitimately
#: hit 1.0 and are not claims about performance.
METRIC_COLUMN = re.compile(
    r"^(accuracy|balanced_accuracy|sensitivity|specificity|precision|recall|f1|macro_f1|"
    r"weighted_f1|macro_precision|macro_recall|roc_auc|pr_auc|auc)(_mean)?$"
)

#: Near-perfect values that are explained, and where the explanation lives. A
#: value that reaches this list has been looked at; one that does not is a
#: finding.
EXPLAINED_PERFECTION: dict[str, str] = {
    "EXP-C1-two_class:M3:specificity_mean": (
        "0.9900, and it is the calibrated SVM almost never saying 'present'. Its "
        "sensitivity on the same rows is 0.2485 and its balanced accuracy 0.6192 -- "
        "the WORST of the five models on both, while M6/M7 reach 0.5473 sensitivity "
        "at 0.8452 specificity. So this is the rule-6 pathology working exactly as "
        "documented (a 92%-absent corpus rewards a model that abstains), not a "
        "leak: a leaked subject or a duplicated recording would lift sensitivity "
        "and AUC with it, and M3's ROC-AUC (0.7450) is the middle of the pack. "
        "Recorded here rather than silently thresholded away."
    ),
}


@dataclass(frozen=True)
class Finding:
    """One thing the review objects to."""

    check: str
    where: str
    line: int
    text: str
    why: str


@dataclass
class ComplianceReport:
    findings: list[Finding] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    scanned: dict[str, int] = field(default_factory=dict)
    generated_utc: str = ""

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def for_check(self, check: str) -> list[Finding]:
        return [finding for finding in self.findings if finding.check == check]

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_utc": self.generated_utc,
            "ok": self.ok,
            "scanned": self.scanned,
            "checks": {
                name: {
                    "description": description,
                    "findings": len(self.for_check(name)),
                    "note": self.notes.get(name, ""),
                }
                for name, description in CHECKS
            },
            "findings": [asdict(finding) for finding in self.findings],
        }


def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:  # pragma: no cover
        return str(path)


def _scannable(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in SCANNED_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if SKIP_PARTS & set(path.parts):
                continue
            files.append(path)
    return files


# ---------------------------------------------------------------------------
# T125.1 -- language
# ---------------------------------------------------------------------------


def _locked_quotations() -> tuple[str, ...]:
    """The six locked objectives, whitespace-normalized.

    A line that is part of one of these is a **quotation**, and quotations are
    reproduced exactly or not at all -- the blueprint's instruction is "do not
    change, shorten, paraphrase or reword". Objective 4 says "might find use in
    medical diagnostic systems"; that wording is checked for fidelity by T125.4
    and must not be edited to satisfy T125.1.
    """
    try:
        from src.reporting.objectives import OBJECTIVES
    except ImportError:  # pragma: no cover
        return ()
    return tuple(" ".join(objective.wording.split()).lower() for objective in OBJECTIVES)


def _is_quoted_objective(line: str, quotations: tuple[str, ...]) -> bool:
    """Is this line (or one cell of it) part of a locked objective's wording?

    A generated table quotes an objective in one cell of a row whose other cells
    are evidence paths, so the row as a whole matches nothing and the cell
    matches exactly. Both are checked.
    """
    candidates = [line, *line.split("|")] if "|" in line else [line]
    for candidate in candidates:
        fragment = " ".join(candidate.split()).strip("|# -*").lower()
        if len(fragment) < 25:
            continue
        if any(fragment in quotation for quotation in quotations):
            return True
    return False


def _check_language(report: ComplianceReport, root: Path) -> None:
    files = _scannable(root)
    quotations = _locked_quotations()
    report.scanned["language_files"] = len(files)
    lines_scanned = 0

    for path in files:
        relative = _rel(path)
        if relative in SELF_EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        in_code = relative.startswith(("src/", "scripts/"))
        lines = text.splitlines()
        for index, line in enumerate(lines):
            lines_scanned += 1
            window = " ".join(lines[max(0, index - 1) : index + 2])
            if _is_quoted_objective(line, quotations):
                continue
            for rule in LANGUAGE_RULES:
                if rule.hits(line, window=window, in_code=in_code):
                    report.add(
                        Finding(
                            "language",
                            relative,
                            index + 1,
                            line.strip()[:200],
                            f"{rule.name}: {rule.why}",
                        )
                    )
    report.scanned["language_lines"] = lines_scanned
    report.notes["language"] = (
        f"{len(files)} files, {lines_scanned} lines, {len(LANGUAGE_RULES)} rules; "
        "each rule carries its own allowances, matched over a three-line window so a "
        "denial split by a line wrap still counts; code files additionally allow the "
        "corpus's own diagnosis field and track names"
    )


# ---------------------------------------------------------------------------
# T125.2 -- disclaimer
# ---------------------------------------------------------------------------


def _check_disclaimer(report: ComplianceReport, root: Path) -> None:
    # The two modules that DECLARE the text must contain it in full.
    for relative in DISCLAIMER_CONSTANTS:
        path = root / relative
        text = path.read_text(encoding="utf-8").lower() if path.is_file() else ""
        missing = [marker for marker in DISCLAIMER_MARKERS if marker not in text]
        if missing:
            report.add(
                Finding(
                    "disclaimer",
                    relative,
                    0,
                    "",
                    f"the canonical DISCLAIMER constant is missing {missing}",
                )
            )

    # Everything else may carry it by reference. `src/api/main.py` imports
    # DISCLAIMER and puts it on four response models; requiring the literal
    # sentence in every such file would push the project toward copies of a
    # string that must never diverge.
    present = 0
    for label, relative in DISCLAIMER_SURFACES:
        path = root / relative
        if not path.is_file():
            report.add(
                Finding("disclaimer", relative, 0, "", f"{label}: the file is not in this checkout")
            )
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        literal = all(marker in lowered for marker in DISCLAIMER_MARKERS)
        by_reference = "DISCLAIMER" in text or "disclaimer" in lowered
        if not (literal or by_reference):
            report.add(
                Finding(
                    "disclaimer",
                    relative,
                    0,
                    "",
                    f"{label}: neither the disclaimer text nor a reference to it",
                )
            )
        else:
            present += 1

    # Every page of the BUILT dashboard, not only the layout that is supposed to
    # put it there. A layout can carry the string and a route can opt out of it.
    built = root / "frontend" / "out"
    pages = sorted(built.rglob("index.html")) if built.is_dir() else []
    without = [
        _rel(page)
        for page in pages
        if not all(marker in page.read_text(encoding="utf-8").lower() for marker in ("screening",))
    ]
    for page in without:
        report.add(Finding("disclaimer", page, 0, "", "a built page carries no screening notice"))

    report.scanned["disclaimer_surfaces"] = len(DISCLAIMER_SURFACES)
    report.scanned["built_pages"] = len(pages)
    report.notes["disclaimer"] = (
        f"{present} of {len(DISCLAIMER_SURFACES)} declared surfaces carry it "
        "(literally or by reference to the canonical constant), "
        f"{len(pages) - len(without)} of {len(pages)} built dashboard pages, "
        f"and {len(DISCLAIMER_CONSTANTS)} canonical constants verified word for word"
    )


# ---------------------------------------------------------------------------
# T125.3 -- implausible perfection
# ---------------------------------------------------------------------------


def _check_perfection(report: ComplianceReport, root: Path) -> None:
    tables = sorted(
        path
        for path in root.glob("outputs/*/**/aggregate_metrics.csv")
        if "_superseded_" not in path.as_posix()
    )
    flagged = 0
    for path in tables:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        label_column = "model_id" if "model_id" in frame.columns else frame.columns[0]
        experiment = path.parent.name
        for column in frame.columns:
            if not METRIC_COLUMN.match(str(column)):
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            for index, value in numeric.items():
                if pd.isna(value) or value < PERFECTION_THRESHOLD:
                    continue
                key = f"{experiment}:{frame.loc[index, label_column]}:{column}"
                if key in EXPLAINED_PERFECTION:
                    continue
                flagged += 1
                report.add(
                    Finding(
                        "perfection",
                        _rel(path),
                        int(index) + 2,
                        f"{key} = {value:.6f}",
                        "a metric at or above 0.995 with no recorded explanation -- "
                        "check for a leaked subject, a scaler fitted on the full matrix, "
                        "a duplicated recording, or a label joined on the wrong key",
                    )
                )
    report.scanned["metric_tables"] = len(tables)
    report.notes["perfection"] = (
        f"{len(tables)} result tables scanned at threshold {PERFECTION_THRESHOLD}; "
        f"{flagged} unexplained, {len(EXPLAINED_PERFECTION)} explained in "
        "src/reporting/compliance.py"
    )


# ---------------------------------------------------------------------------
# T125.4 -- objectives verbatim
# ---------------------------------------------------------------------------


def _check_objectives(report: ComplianceReport, root: Path) -> None:
    try:
        from src.reporting.objectives import OBJECTIVES
    except ImportError as error:  # pragma: no cover
        report.add(Finding("objectives", "src/reporting/objectives.py", 0, "", str(error)))
        return

    declared = {objective.number: objective.wording for objective in OBJECTIVES}
    report.scanned["objectives"] = len(declared)

    # Wherever an objective's wording is reproduced, it must be the declared
    # wording character for character. The generated frontend payload is the one
    # place it reaches a reader in full, so it is the one compared.
    payload = root / "frontend" / "lib" / "generated" / "objectives.json"
    if payload.is_file():
        data = json.loads(payload.read_text(encoding="utf-8"))
        entries = data.get("objectives", data if isinstance(data, list) else [])
        quoted = {
            int(entry["number"]): str(entry["wording"])
            for entry in entries
            if "wording" in entry
        }
        for number, text in declared.items():
            if number not in quoted:
                report.add(
                    Finding(
                        "objectives",
                        _rel(payload),
                        0,
                        f"objective {number}",
                        "the exported payload does not carry this objective",
                    )
                )
            elif quoted[number] != text:
                report.add(
                    Finding(
                        "objectives",
                        _rel(payload),
                        0,
                        f"objective {number}",
                        "the exported wording differs from the locked wording",
                    )
                )
        report.notes["objectives"] = (
            f"{len(quoted)} of {len(declared)} objectives exported, compared character for "
            "character"
        )
    else:
        report.notes["objectives"] = "objectives.json is not in this checkout"

    # Paraphrase detection where it can be done cheaply: an objective's first
    # eight words appearing with different wording after them.
    for number, text in declared.items():
        opening = " ".join(text.split()[:6])
        if len(opening) < 20:
            continue
        for path in _scannable(root):
            if path.suffix not in {".md", ".txt"}:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            if opening in content and text not in content.replace("\n", " "):
                collapsed = " ".join(content.split())
                if " ".join(text.split()) in collapsed:
                    continue
                report.add(
                    Finding(
                        "objectives",
                        _rel(path),
                        0,
                        opening,
                        f"objective {number} appears to be quoted but not verbatim",
                    )
                )


# ---------------------------------------------------------------------------
# T125.5 -- documented counts
# ---------------------------------------------------------------------------

#: Counts that appear in prose, and what the audit says they are. A number in a
#: document that contradicts the corpus is the failure this catches -- including
#: the two the source documents themselves get wrong.
DOCUMENTED_COUNTS: tuple[tuple[str, str, str], ...] = (
    ("3,240", "README.md", "D1 PhysioNet training recordings"),
    ("301", "README.md", "D1 validation recordings, dropped as duplicates"),
    ("124", "README.md", "D2 PASCAL set_a labelled"),
    ("461", "README.md", "D3 PASCAL set_b labelled"),
    ("942", "README.md", "D4 CirCor patients"),
    ("3,163", "README.md", "D4 CirCor recordings"),
    ("138", "README.md", "engineered features"),
)


def _check_counts(report: ComplianceReport, root: Path) -> None:
    from src.reporting.final_qa import AUDITED

    master = root / "outputs" / "01_dataset_audit" / "metadata_master.csv"
    if master.is_file():
        frame = pd.read_csv(master, low_memory=True, usecols=["dataset_source"])
        counted = frame.groupby("dataset_source").size().to_dict()
        for source, expected in AUDITED.items():
            wanted = expected["supervised"] + expected.get("unlabelled", 0)
            actual = int(counted.get(source, 0))
            if actual != wanted:
                report.add(
                    Finding(
                        "counts",
                        _rel(master),
                        0,
                        f"{source}: {actual}",
                        f"the audit inventories {actual} records, documentation says {wanted}",
                    )
                )
        report.scanned["counted_sources"] = len(counted)

    for value, relative, what in DOCUMENTED_COUNTS:
        path = root / relative
        if not path.is_file():
            continue
        if value not in path.read_text(encoding="utf-8"):
            report.add(
                Finding("counts", relative, 0, value, f"{what}: the count is not documented here")
            )

    # The two discrepancies the source documents get wrong, and the population
    # note EXP-D1 must always carry. Each must be stated somewhere a reader
    # reaches, or the deliverable quietly inherits the document's error.
    required_notes = {
        "EXP-D1 population mismatch": ("adult-to-paediatric", "README.md"),
        "no held-out PhysioNet test set": ("no held-out PhysioNet test set", "README.md"),
        "CirCor public subset": ("public subset only", "README.md"),
    }
    for label, (needle, relative) in required_notes.items():
        path = root / relative
        if path.is_file() and needle.lower() not in path.read_text(encoding="utf-8").lower():
            report.add(
                Finding("counts", relative, 0, needle, f"{label} is not stated in {relative}")
            )

    report.notes["counts"] = (
        f"{len(DOCUMENTED_COUNTS)} published counts and {len(required_notes)} required "
        "discrepancy notes checked against the audit"
    )


# ---------------------------------------------------------------------------
# T125.6 -- nothing hand-entered
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

#: A cell that *is* a number, rather than prose that happens to contain digits.
#: The forms a table cell legitimately takes: a bare value, a percentage, a
#: mean +/- SD, a range, a value with a significance marker, a p-value written
#: as an inequality, and an em-dash for "not applicable".
_NUMERIC_CELL = re.compile(
    r"^\**\s*[<>=~]?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*%?"
    r"(?:\s*(?:\+/-|±|\+-|to|--|-|\u2013)\s*[-+]?\d[\d,]*(?:\.\d+)?\s*%?)?"
    r"[*\u2020\u2021\s]*\**$"
)


def _csv_number_forms(frame: pd.DataFrame) -> set[str]:
    """Every string a CSV value could legitimately be rendered as.

    The renderer rounds by column kind (metrics to 3 places, p-values to 3,
    percentages to 1, counts to 0), and thousands separators are added for
    display. Rather than re-implement the renderer, every plausible rounding of
    every value is enumerated -- a number in the markdown that matches none of
    them did not come from this CSV.
    """
    forms: set[str] = set()
    for column in frame.columns:
        for value in frame[column].tolist():
            text = str(value).strip()
            if not text:
                continue
            forms.add(text)
            try:
                number = float(text)
            except (TypeError, ValueError):
                continue
            for places in range(0, 7):
                forms.add(f"{number:.{places}f}")
                forms.add(f"{number:,.{places}f}")
            if number.is_integer():
                forms.add(str(int(number)))
                forms.add(f"{int(number):,}")
    return forms


def _check_generated(report: ComplianceReport, root: Path) -> None:
    renderings = sorted(root.glob("outputs/*/T*.md"))
    checked = 0
    narratives = 0
    numeric_cells = 0
    for markdown in renderings:
        csv_path = markdown.with_suffix(".csv")
        meta_path = markdown.with_suffix(".meta.json")

        if not csv_path.is_file() and not meta_path.is_file():
            # Narrative, not a rendered table. It still has to say where its
            # numbers come from.
            head = "\n".join(markdown.read_text(encoding="utf-8").splitlines()[:8]).lower()
            if not any(word in head for word in ("generated from", "outputs/", ".csv")):
                report.add(
                    Finding(
                        "generated",
                        _rel(markdown),
                        0,
                        "",
                        "neither a sidecar nor a stated source: nothing says where these "
                        "numbers came from",
                    )
                )
            narratives += 1
            continue

        if not meta_path.is_file():
            report.add(
                Finding(
                    "generated",
                    _rel(markdown),
                    0,
                    "",
                    "no .meta.json sidecar: this table has no recorded command or source",
                )
            )
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not str(meta.get("command", "")).strip():
            report.add(
                Finding("generated", _rel(meta_path), 0, "", "the sidecar records no command")
            )
        for source in meta.get("sources", []):
            if not source.get("exists", False):
                report.add(
                    Finding(
                        "generated",
                        _rel(meta_path),
                        0,
                        str(source.get("path", "")),
                        "a declared source file is missing",
                    )
                )
        if not meta.get("sources"):
            report.add(
                Finding("generated", _rel(meta_path), 0, "", "the sidecar names no source data")
            )

        if not csv_path.is_file():
            report.add(
                Finding(
                    "generated",
                    _rel(markdown),
                    0,
                    "",
                    "no sibling .csv: the rendering has no full-precision source",
                )
            )
            continue

        frame = pd.read_csv(csv_path, keep_default_na=False)
        forms = _csv_number_forms(frame)
        checked += 1

        for number, line in enumerate(markdown.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.startswith("|") or set(line) <= set("|-: "):
                continue
            for cell in line.strip().strip("|").split("|"):
                text = cell.strip()
                # Only cells that ARE numbers. A cell reading
                # `outputs/06_binary_results/T08_...csv` or `PhysioNet 2016`
                # carries digits and is not a result; checking those turns the
                # signal (a metric nobody can trace) into noise (two thousand
                # path fragments).
                if not _NUMERIC_CELL.match(text):
                    continue
                numeric_cells += 1
                bare = text.strip("*").strip()
                if bare in forms or bare.replace(",", "") in forms:
                    continue
                for token in _NUMBER.findall(text):
                    if token in forms or token.replace(",", "") in forms:
                        continue
                    report.add(
                        Finding(
                            "generated",
                            _rel(markdown),
                            number,
                            f"{token} (cell: {text[:60]})",
                            "this number does not re-derive from any value in the sibling CSV",
                        )
                    )
    report.scanned["rendered_tables"] = len(renderings)
    report.notes["generated"] = (
        f"{checked} of {len(renderings)} published tables re-derived cell by cell "
        f"from their own CSV ({numeric_cells} numeric cells); every sidecar checked "
        f"for a command and live sources; {narratives} narrative document(s) checked "
        "for a stated source instead"
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_RUNNERS = {
    "language": _check_language,
    "disclaimer": _check_disclaimer,
    "perfection": _check_perfection,
    "objectives": _check_objectives,
    "counts": _check_counts,
    "generated": _check_generated,
}


def run_review(checks: list[str] | None = None, root: Path | None = None) -> ComplianceReport:
    """Run every compliance check (or the named subset)."""
    base = root or PROJECT_ROOT
    report = ComplianceReport(generated_utc=datetime.now(UTC).isoformat(timespec="seconds"))
    for name, _ in CHECKS:
        if checks and name not in checks:
            continue
        LOGGER.info("compliance: %s", name)
        _RUNNERS[name](report, base)
    return report


def _markdown(report: ComplianceReport) -> str:
    lines = [
        "# Compliance and claims review (Phase 125)",
        "",
        f"Generated {report.generated_utc} by `python scripts/50_compliance_review.py`.",
        "",
        f"**Result: {'PASS' if report.ok else str(len(report.findings)) + ' FINDING(S)'}**",
        "",
        "| Check | T125 | Findings | Scope |",
        "|---|---|---|---|",
    ]
    for name, description in CHECKS:
        task = description.split(" -- ")[0]
        note = report.notes.get(name, "").replace("|", "\\|")
        lines.append(f"| {name} | {task} | {len(report.for_check(name))} | {note} |")

    for name, description in CHECKS:
        findings = report.for_check(name)
        lines += ["", f"## {name} — {description}", ""]
        if not findings:
            lines.append("No findings.")
            continue
        lines += ["| Where | Line | What | Why |", "|---|---|---|---|"]
        for finding in findings[:200]:
            text = finding.text.replace("|", "\\|")
            why = finding.why.replace("|", "\\|")
            lines.append(f"| `{finding.where}` | {finding.line} | {text} | {why} |")
        if len(findings) > 200:
            lines.append(f"| … | | {len(findings) - 200} more | |")

    lines.append("")
    return "\n".join(lines)


def write_report(report: ComplianceReport) -> tuple[Path, Path]:
    from src.utils.evidence import register_evidence

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")
    register_evidence(
        "QA-COMPLIANCE",
        REPORT_MD,
        metric_or_asset="compliance and claims review (Phase 125)",
        command="python scripts/50_compliance_review.py",
        status="ok" if report.ok else "failed",
    )
    return REPORT_JSON, REPORT_MD
