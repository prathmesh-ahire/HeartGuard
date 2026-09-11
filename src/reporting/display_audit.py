"""The displayed-value audit (T119.3), the manifest check (T119.5) and the screenshot gate (T119.4).

`scripts/16_check_no_hardcoded_metrics.py` reads the SOURCE of the pages and
catches a number typed into one. It cannot see what a page actually rendered: a
template that printed a `values` float instead of its `display` string, a
payload the site was built against before `outputs/` changed, a footer naming a
run that never happened. This module reads the BUILT site instead.

What it checks
--------------
1. **Screen to source.** Every page under `frontend/out/` is crawled and every
   metric-shaped token in its visible text (three or more decimals, or a
   percentage) must be a string some `generated/` payload carries. Numbers in a
   payload's numeric fields do not count -- they position marks and are never
   text -- so a page that renders one fails here.
2. **Source to generated.** Every table cell and every inlined figure cell is
   re-formatted from its committed CSV through `tables.format_value` and must
   equal the exported string. A stale `generated/` fails.
3. **Evidence is current.** Every evidence row's recorded sha256 must equal the
   digest of the committed file today, so the evidence browser cannot vouch for
   a file that has since changed.
4. **The footer is the run (T119.5).** Every page must show the manifest's run
   id, commit and export time; the run id must exist in the project's run
   manifest as an `export_frontend_data` run with the same commit; and every
   source the manifest fingerprinted must still have that fingerprint.

The gate
--------
A passing audit writes a report stamped with a digest of the exact HTML it read.
`require_passed_audit` refuses unless that report exists, passed, and still
matches the site on disk -- so a screenshot (Phase 120) can only be taken of the
build that was audited, not of one rebuilt afterwards.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "AuditGateError",
    "AuditReport",
    "METRIC_LIKE",
    "default_report_path",
    "require_passed_audit",
    "run_audit",
    "site_digest",
    "write_report",
]

METRIC_LIKE = re.compile(r"\b\d+\.\d{3,}\b|\b\d+(?:\.\d+)?%")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")

#: Findings shown per check before the rest are counted rather than listed.
_SHOW = 25


class AuditGateError(RuntimeError):
    """The site on disk has not passed the displayed-value audit."""


@dataclass
class AuditReport:
    """What one audit read and found. ``passed`` is the only verdict."""

    site_digest: str
    run_id: str
    git_commit: str | None
    checked: dict[str, int] = field(default_factory=dict)
    findings: dict[str, list[str]] = field(default_factory=dict)
    audited_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return not any(self.findings.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.passed else "fail",
            "audited_utc": self.audited_utc,
            "site_digest": self.site_digest,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "checked": self.checked,
            "n_findings": {key: len(value) for key, value in self.findings.items()},
            "findings": {key: value[:_SHOW] for key, value in self.findings.items()},
        }


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def _paths() -> Any:
    from src.utils.config import load_config

    return load_config("paths")


def _project_root() -> Path:
    return Path(_paths().require("project_root"))


def _out_dir(override: str | Path | None) -> Path:
    return (
        Path(override) if override is not None else Path(_paths().require("frontend.static_export"))
    )


def _generated_dir(override: str | Path | None) -> Path:
    return Path(override) if override is not None else Path(_paths().require("frontend.generated"))


def default_report_path() -> Path:
    """Beside the build it describes, and gitignored like it: a stamp, not a deliverable."""
    return Path(_paths().require("frontend.root")) / ".display-audit.json"


# ---------------------------------------------------------------------------
# reading the site
# ---------------------------------------------------------------------------


def _pages(out: Path) -> list[Path]:
    return sorted(path for path in out.rglob("*.html") if "evidence" not in path.parts)


def _visible_text(raw: str) -> str:
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


def site_digest(out_dir: str | Path | None = None) -> str:
    """sha256 over every exported page, newline-normalised, in path order."""
    out = _out_dir(out_dir)
    digest = hashlib.sha256()
    for page in _pages(out):
        digest.update(page.relative_to(out).as_posix().encode("utf-8"))
        digest.update(page.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _route(out: Path, page: Path) -> str:
    relative = page.relative_to(out).as_posix()
    if relative.endswith("index.html"):
        return "/" + relative[: -len("index.html")]
    return "/" + relative


# ---------------------------------------------------------------------------
# the four checks
# ---------------------------------------------------------------------------


def _strings(node: Any, into: set[str]) -> None:
    if isinstance(node, str):
        into.add(node)
        into.update(METRIC_LIKE.findall(node))
    elif isinstance(node, dict):
        for value in node.values():
            _strings(value, into)
    elif isinstance(node, list):
        for value in node:
            _strings(value, into)


def _allowed(generated: Path) -> set[str]:
    """Every string any generated payload carries -- the only legal source of display text."""
    allowed: set[str] = set()
    for path in sorted(generated.rglob("*.json")):
        _strings(json.loads(path.read_text(encoding="utf-8")), allowed)
    return allowed


def _screen_to_source(out: Path, generated: Path, report: AuditReport) -> None:
    allowed = _allowed(generated)
    findings: list[str] = []
    tokens = 0
    pages = _pages(out)
    for page in pages:
        text = _TIMESTAMP.sub(" ", _visible_text(page.read_text(encoding="utf-8")))
        for token in METRIC_LIKE.findall(text):
            tokens += 1
            if token not in allowed and token.rstrip("%") not in allowed:
                findings.append(
                    _route(out, page) + " shows " + token + ", which no payload carries"
                )
    report.checked["pages"] = len(pages)
    report.checked["rendered_metric_tokens"] = tokens
    report.findings["screen_to_source"] = sorted(set(findings))


def _frame_cells(payload: dict[str, Any], label: str, findings: list[str]) -> int:
    import pandas as pd

    from src.reporting.tables import format_value

    source = _project_root() / str(payload["source_csv"])
    if not source.is_file():
        findings.append(label + ": source " + str(payload["source_csv"]) + " is missing")
        return 0
    frame = pd.read_csv(source)
    if int(payload["n_rows"]) != len(frame):
        findings.append(
            label + ": " + str(payload["n_rows"]) + " rows exported, CSV has " + str(len(frame))
        )
        return 0
    cells = 0
    for column in payload.get("columns", []):
        name = str(column["name"])
        if name not in frame.columns:
            findings.append(label + "." + name + ": column is not in the CSV")
            continue
        for position, shown in enumerate(column["display"]):
            cells += 1
            expected = format_value(frame[name].iloc[position], str(column["kind"]))
            if shown != expected:
                findings.append(
                    label
                    + "."
                    + name
                    + "["
                    + str(position)
                    + "]: shows "
                    + repr(shown)
                    + ", the CSV formats to "
                    + repr(expected)
                )
    return cells


def _source_to_generated(generated: Path, report: AuditReport) -> None:
    findings: list[str] = []
    cells = 0
    tables = json.loads((generated / "tables.json").read_text(encoding="utf-8"))
    for table_id, payload in tables.items():
        cells += _frame_cells(payload, table_id, findings)
    figure_dir = generated / "figures"
    for path in sorted(figure_dir.glob("G*.json")) if figure_dir.is_dir() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("data_omitted"):
            cells += _frame_cells(payload, str(payload["id"]), findings)
    report.checked["tables"] = len(tables)
    report.checked["table_and_figure_cells"] = cells
    report.findings["source_to_generated"] = findings


def _evidence_current(generated: Path, report: AuditReport) -> None:
    from src.reporting.tables import content_digest

    evidence = json.loads((generated / "evidence.json").read_text(encoding="utf-8"))
    findings: list[str] = []
    digests: dict[str, str | None] = {}
    for entry in evidence:
        relative = str(entry["generated_from"])
        if relative not in digests:
            path = _project_root() / relative
            digests[relative] = content_digest(path)[0] if path.is_file() else None
        if digests[relative] is None:
            findings.append(entry["key"] + ": " + relative + " no longer exists")
        elif digests[relative] != entry["generated_from_sha256"]:
            findings.append(entry["key"] + ": " + relative + " changed since the export")
    report.checked["evidence_entries"] = len(evidence)
    report.findings["evidence_current"] = sorted(set(findings))


def _manifest_is_the_run(
    out: Path, generated: Path, report: AuditReport, run_manifest: str | Path | None = None
) -> None:
    from src.reporting.tables import content_digest

    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    findings: list[str] = []

    run_id = str(manifest.get("run_id") or "")
    commit = manifest.get("git_commit")
    if not run_id:
        findings.append("the generated manifest records no run id")
    else:
        manifest_path = Path(run_manifest) if run_manifest is not None else default_run_manifest()
        runs: dict[str, Any] = {}
        if manifest_path.is_file():
            runs = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            findings.append(
                "the project's run manifest " + manifest_path.as_posix() + " is missing"
            )
        match = next((run for run in runs.get("runs", []) if run.get("run_id") == run_id), None)
        if runs and match is None:
            findings.append("run " + run_id + " is not in the project's run manifest")
        elif match is not None:
            if match.get("name") != "export_frontend_data":
                findings.append(
                    "run " + run_id + " is a " + str(match.get("name")) + " run, not an export"
                )
            recorded = (match.get("git") or {}).get("commit")
            if recorded and commit and recorded != commit:
                findings.append("the manifest commit differs from the run's own record")

    for source in manifest.get("sources", []):
        path = _project_root() / str(source["path"])
        if not path.is_file():
            findings.append("manifest source " + str(source["path"]) + " no longer exists")
        elif content_digest(path)[0] != source["sha256"]:
            findings.append("manifest source " + str(source["path"]) + " changed since the export")

    shown = [run_id, str(manifest.get("exported_utc", ""))]
    if commit:
        shown.append(str(commit)[:12])
    pages = _pages(out)
    for page in pages:
        text = _visible_text(page.read_text(encoding="utf-8"))
        for value in shown:
            if value and value not in text:
                findings.append(_route(out, page) + ": the footer does not show " + value)
    report.checked["manifest_sources"] = len(manifest.get("sources", []))
    report.findings["footer_is_the_run"] = findings


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def default_run_manifest() -> Path:
    """The project's run manifest, through `configs/paths.yaml`.

    A parameter elsewhere because pytest's conftest redirects this key to an
    empty temporary file (so a test can never write the real manifest); a test
    auditing the real build names the committed file explicitly.
    """
    return Path(_paths().require("outputs.run_manifest"))


def run_audit(
    *,
    out_dir: str | Path | None = None,
    generated_dir: str | Path | None = None,
    run_manifest: str | Path | None = None,
) -> AuditReport:
    """Crawl the built site and run all four checks. Never raises on a finding."""
    out = _out_dir(out_dir)
    generated = _generated_dir(generated_dir)
    if not (out / "index.html").is_file():
        raise FileNotFoundError("no built site at " + str(out) + "; run `npm run build` first")
    manifest = json.loads((generated / "manifest.json").read_text(encoding="utf-8"))
    report = AuditReport(
        site_digest=site_digest(out),
        run_id=str(manifest.get("run_id") or ""),
        git_commit=manifest.get("git_commit"),
    )
    _screen_to_source(out, generated, report)
    _source_to_generated(generated, report)
    _evidence_current(generated, report)
    _manifest_is_the_run(out, generated, report, run_manifest)
    return report


def write_report(report: AuditReport, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else default_report_path()
    target.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def require_passed_audit(
    *, out_dir: str | Path | None = None, report_path: str | Path | None = None
) -> dict[str, Any]:
    """T119.4: refuse unless THIS build passed the audit. Returns the report."""
    path = Path(report_path) if report_path is not None else default_report_path()
    if not path.is_file():
        raise AuditGateError(
            "no displayed-value audit has been run on this build; run "
            "scripts/45_audit_displayed_values.py before taking a screenshot"
        )
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded.get("status") != "pass":
        raise AuditGateError(
            "the last displayed-value audit FAILED ("
            + json.dumps(recorded.get("n_findings"))
            + "); "
            "no screenshot may be taken of this site"
        )
    current = site_digest(out_dir)
    if recorded.get("site_digest") != current:
        raise AuditGateError(
            "the site was rebuilt after it was audited (digest "
            + str(recorded.get("site_digest"))[:12]
            + " audited, "
            + current[:12]
            + " on disk); re-run the audit"
        )
    loaded: dict[str, Any] = recorded
    return loaded
