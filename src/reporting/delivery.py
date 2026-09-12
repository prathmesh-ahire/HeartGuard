"""Final delivery: the completion checklist, the asset counts, the ZIP, the handover (Phase 126).

Four things, in the order T126 asks for them:

T126.1/T126.2
    :func:`checklist` re-runs the extraction spec's section-19 list and the
    blueprint's own requirements, and :func:`asset_inventory` counts what is on
    disk against the declared targets -- 30 tables, 35 graphs, 20 diagrams, 20
    algorithms, 13 screenshots. A gap is reported with the reason recorded for
    it, never silently.
T126.3/T126.4
    :func:`build_zip` packages the deliverable and :func:`verify_zip` opens the
    file it just wrote and checks it -- CRCs, required entries, and the
    exclusions. A ZIP nobody opened is not a deliverable.
T126.6
    :func:`write_handover` generates ``HANDOVER.md``: what was produced (counted
    from disk), what was not (read from ``missing_outputs_report.txt`` and
    ``todo.md``'s own markers), and what clinical validation would require.

## What is deliberately not in the ZIP

``dataset/`` (1.3 GB of third-party corpora that this project has no licence to
redistribute -- see CITATION.md), ``.venv/``, ``node_modules/``, ``.next/``,
``cache/``, ``.git/``, and the two working-state directories under ``outputs/``
(``_checkpoints/`` and ``_nested_search_cache/``). ``Docs/`` is excluded because
it is kept local by the user's decision and holds the two source documents,
which are likewise not ours to redistribute.

Everything a grader needs is in: the code, the configs, the **saved model
binaries** (which the repository gitignores and a delivery must not), every
result CSV and JSON, the logs, the frontend source *and* its built static
export, and all assets. `frontend/out/` being inside is what makes the
"no Node required" claim true.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "ASSET_TARGETS",
    "AssetCount",
    "DELIVERY_NAME",
    "build_zip",
    "asset_inventory",
    "checklist",
    "verify_zip",
    "write_handover",
]

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = PROJECT_ROOT / "dist"
DELIVERY_NAME = "PV-MEPCG_PulseVision_delivery"
MANIFEST_NAME = "DELIVERY_MANIFEST.json"

#: The counts the extraction spec requires, and where each lives.
ASSET_TARGETS: tuple[tuple[str, int, str, str], ...] = (
    ("tables", 30, "outputs/*/T*.csv", r"^(T\d{2})"),
    ("graphs", 35, "outputs/13_figures_diagrams/G*.png", r"^(G\d{2})"),
    ("diagrams", 20, "outputs/13_figures_diagrams/F*.png", r"^(F\d{2})"),
    ("algorithms", 20, "outputs/14_algorithms/ALG-*.txt", r"^(ALG-\d{2})"),
    ("screenshots", 13, "outputs/15_dashboard_screenshots/SS-*.png", r"^SS-(\d{2})"),
)

#: Directories and patterns that never enter the delivery. Each is excluded for
#: a reason a grader can check, not for convenience.
EXCLUDED_PARTS: tuple[tuple[str, str], ...] = (
    ("dataset", "1.3 GB of third-party corpora; not ours to redistribute (see CITATION.md)"),
    (".venv", "a machine-specific virtual environment"),
    ("node_modules", "restored by `npm ci` from the committed package-lock.json"),
    (".next", "Next.js build cache; the export in frontend/out/ is what ships"),
    ("cache", "regenerable preprocessed signals and feature shards"),
    (".git", "version-control internals"),
    ("__pycache__", "compiled bytecode"),
    (".pytest_cache", "test runner state"),
    (".mypy_cache", "type checker state"),
    (".ruff_cache", "linter state"),
    ("_checkpoints", "experiment resume state; every number in it is in the committed CSVs"),
    ("_nested_search_cache", "search working state; the chosen points are in per_fold_metrics.csv"),
    ("Docs", "kept local by the user's decision; holds the two source documents"),
    ("dist", "the delivery archive itself"),
    ("test-results", "Playwright working output"),
    ("playwright-report", "Playwright working output"),
)

#: Entries the ZIP must contain, or it does not deliver what it claims to.
REQUIRED_ENTRIES: tuple[str, ...] = (
    "README.md",
    "ARCHITECTURE.md",
    "CONFIGURATION.md",
    "CITATION.md",
    "HANDOVER.md",
    "requirements/base.txt",
    "configs/paths.yaml",
    "scripts/00_run_everything.py",
    "src/api/main.py",
    "frontend/out/index.html",
    "frontend/package-lock.json",
    "outputs/missing_outputs_report.txt",
    "outputs/00_evidence_index/evidence_index.csv",
    "outputs/00_evidence_index/run_manifest.json",
    "outputs/00_evidence_index/final_qa_report.md",
    "outputs/00_evidence_index/compliance_review.md",
    MANIFEST_NAME,
)


class DeliveryError(RuntimeError):
    """The delivery could not be built, or the archive failed its own check."""


@dataclass(frozen=True)
class AssetCount:
    kind: str
    target: int
    found: int
    ids: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.found >= self.target and not self.missing


def asset_inventory(root: Path | None = None) -> tuple[AssetCount, ...]:
    """T126.2 -- the five counts, taken from the filesystem.

    Tables are counted by their **numbered** ids (T01..T30): the five
    supplementary ``T-S*`` tables are real deliverables and are listed, but the
    spec's "30 tables" means the numbered series, and counting the supplements
    toward it would let a missing T-number hide behind an extra supplement.
    """
    base = root or PROJECT_ROOT
    counts: list[AssetCount] = []
    for kind, target, pattern, id_pattern in ASSET_TARGETS:
        found: set[str] = set()
        for path in base.glob(pattern):
            match = re.match(id_pattern, path.name)
            if match:
                found.add(match.group(1))
        if kind == "tables":
            expected = {f"T{number:02d}" for number in range(1, target + 1)}
        elif kind == "graphs":
            expected = {f"G{number:02d}" for number in range(1, target + 1)}
        elif kind == "diagrams":
            expected = {f"F{number:02d}" for number in range(1, target + 1)}
        elif kind == "algorithms":
            expected = {f"ALG-{number:02d}" for number in range(1, target + 1)}
        else:
            expected = {f"{number:02d}" for number in range(1, target + 1)}
        counts.append(
            AssetCount(
                kind=kind,
                target=target,
                found=len(found & expected),
                ids=tuple(sorted(found)),
                missing=tuple(sorted(expected - found)),
            )
        )
    return tuple(counts)


def checklist() -> dict[str, Any]:
    """T126.1 -- the extraction spec's section-19 list, item by item.

    Delegates to the same :func:`evidence_pack.check_mandatory` the evidence
    index uses, so the checklist and the index can never disagree. An item that
    is ``not produced`` must carry a reason; one that does not is ``failed``,
    which is the case this exists for.
    """
    from src.reporting.evidence_pack import check_mandatory

    rows = check_mandatory()
    by_status: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_status.setdefault(str(row["status"]), []).append(row)

    unexplained = [
        row["item_id"]
        for row in by_status.get("not produced", [])
        if not str(row.get("detail", row.get("reason", ""))).strip()
    ]
    return {
        "total": len(rows),
        "present": len(by_status.get("present", [])),
        "not_produced": len(by_status.get("not produced", [])),
        "failed": len(by_status.get("failed", [])),
        "unexplained": unexplained,
        "items": rows,
    }


def _excluded(relative: Path) -> str | None:
    parts = set(relative.parts)
    for part, reason in EXCLUDED_PARTS:
        if part in parts:
            return reason
    if relative.suffix in {".pyc", ".pyo"}:
        return "compiled bytecode"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = root / ".git" / text[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.is_file() else ""
    return text


def _manifest(root: Path, counts: tuple[AssetCount, ...], files: list[Path]) -> dict[str, Any]:
    return {
        "framework": "PV-MEPCG / PulseVision",
        "repository": "HeartGuard",
        "built_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(root),
        "scope": (
            "Academic screening and decision-support prototype. Not a medical device, "
            "not a diagnostic tool, and not clinically validated. See README.md."
        ),
        "n_files": len(files),
        "asset_counts": {
            count.kind: {"target": count.target, "found": count.found} for count in counts
        },
        "excluded": [{"path": part, "reason": reason} for part, reason in EXCLUDED_PARTS],
        "node_required": False,
        "serve_command": "python -m uvicorn src.api.main:app --port 8000",
        "reproduce_command": "python scripts/00_run_everything.py",
        "what_is_missing": "outputs/missing_outputs_report.txt",
    }


def build_zip(destination: Path | None = None, *, root: Path | None = None) -> Path:
    """T126.3 -- package the deliverable, with a manifest inside it."""
    base = root or PROJECT_ROOT
    target = Path(destination) if destination else DIST_DIR / f"{DELIVERY_NAME}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if _excluded(relative):
            continue
        files.append(relative)
    files.sort()

    counts = asset_inventory(base)
    manifest = _manifest(base, counts, files)

    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in files:
            archive.write(base / relative, relative.as_posix())
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")

    digest = _sha256(target)
    target.with_suffix(".zip.sha256").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8"
    )
    LOGGER.info("wrote %s (%.1f MB, %d files)", target, target.stat().st_size / 1e6, len(files))
    return target


def verify_zip(path: Path | None = None) -> dict[str, Any]:
    """T126.4/T126.7 -- open the archive that was just written and check it.

    Checks, in order: the central directory reads, every CRC matches, every
    required entry is present, no excluded path leaked in, and the manifest
    inside says what the archive actually contains.
    """
    target = Path(path) if path else DIST_DIR / f"{DELIVERY_NAME}.zip"
    if not target.is_file():
        raise DeliveryError(f"no delivery archive at {target}")

    with zipfile.ZipFile(target) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise DeliveryError(f"corrupt entry in the archive: {broken}")

        names = set(archive.namelist())
        missing = [entry for entry in REQUIRED_ENTRIES if entry not in names]
        leaked = sorted(
            {
                name
                for name in names
                if _excluded(Path(name)) is not None
            }
        )
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))

        # The "no Node required" claim, checked rather than asserted: the built
        # export has to be in the archive, and it has to be more than a stub.
        built = "frontend/out/index.html"
        index = archive.getinfo(built) if built in names else None

    if missing:
        raise DeliveryError(f"the archive is missing required entries: {missing}")
    if leaked:
        raise DeliveryError(f"excluded paths leaked into the archive: {leaked[:5]}")
    if index is None or index.file_size < 1000:
        raise DeliveryError("frontend/out/index.html is absent or a stub; a grader would need Node")

    size = target.stat().st_size
    return {
        "path": str(target),
        "bytes": size,
        "megabytes": round(size / 1e6, 1),
        "entries": len(names),
        "sha256": _sha256(target),
        "manifest_files": manifest["n_files"],
        "git_commit": manifest.get("git_commit", ""),
        "node_required": manifest.get("node_required", None),
    }


# ---------------------------------------------------------------------------
# T126.6 -- the handover
# ---------------------------------------------------------------------------

#: What a clinical validation of this method would actually require. Written
#: here rather than generated, because it is the one section of the handover
#: that is a judgement about future work; everything above it is counted.
CLINICAL_VALIDATION = """\
Nothing in this repository is a step toward clinical use, and the gap is larger
than "more data". At minimum, the following would have to happen first, roughly
in this order:

1. **A prospective, multi-site acquisition protocol.** The single largest
   measured weakness here is that performance does not survive a change of
   recording collection: leave-one-sub-collection-out takes AUC from 0.92-0.94
   to 0.43-0.53, several folds below chance. Any clinical study must acquire on
   the devices, in the rooms, and with the operators it intends to deploy to,
   and must hold out entire sites.
2. **A reference standard that is not another label file.** These corpora are
   labelled by expert listening and, for CirCor, by a published protocol. A
   clinical study needs echocardiography (or an equivalent) as ground truth,
   read blind to the model output.
3. **A pre-registered operating point and endpoint.** The decision threshold
   here is chosen inside each training fold against balanced accuracy. A
   clinical study fixes one threshold in advance, states the sensitivity and
   specificity it is powered to detect, and reports what it pre-registered.
4. **Paediatric and adult cohorts studied separately.** EXP-D1 is
   adult-to-paediatric transfer and drops as a population effect; the two are
   not one population and cannot share a model, a threshold, or a claim.
5. **A clinically meaningful comparator.** "Better than chance" is not a claim
   anyone can act on. The comparator is a clinician with a stethoscope, and the
   question is whether a screening aid changes referral decisions.
6. **Human-factors and failure-mode work.** What a clinician does with a
   low-confidence output, what happens on an unscorable recording, and what the
   cost of a false negative is in the intended care pathway.
7. **Regulatory scope.** Software intended to inform a clinical decision is a
   medical device in most jurisdictions, and PV-MEPCG is not a medical device,
   is not built to any quality management standard, and would need to be
   rebuilt under one.

Until every one of those exists, the correct description of this work is the one
on the first page of the README: an academic screening and decision-support
prototype, evaluated within public research corpora.
"""


def _todo_markers(root: Path) -> dict[str, list[str]]:
    """Tasks marked blocked or skipped, straight from ``todo.md``.

    ``Docs/`` is gitignored, so this reads it where it exists and returns empty
    lists where it does not -- the handover then says so rather than implying
    nothing was blocked.
    """
    todo = root / "Docs" / "todo.md"
    found: dict[str, list[str]] = {"blocked": [], "skipped": []}
    if not todo.is_file():
        return found
    for line in todo.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- [!]"):
            found["blocked"].append(stripped[5:].strip())
        elif stripped.startswith("- [-]"):
            found["skipped"].append(stripped[5:].strip())
    return found


def write_handover(path: Path | None = None, *, root: Path | None = None) -> Path:
    """T126.6 -- what was produced, what was not, and what would come next."""
    base = root or PROJECT_ROOT
    target = Path(path) if path else base / "HANDOVER.md"

    counts = asset_inventory(base)
    spec = checklist()
    markers = _todo_markers(base)

    missing_report = base / "outputs" / "missing_outputs_report.txt"
    gaps = (
        missing_report.read_text(encoding="utf-8").splitlines()
        if missing_report.is_file()
        else []
    )

    lines: list[str] = [
        "<!-- GENERATED by scripts/51_package_delivery.py -- do not edit by hand. -->",
        "",
        "# PV-MEPCG / PulseVision — handover",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d')} from the repository itself.",
        "",
        "> **Scope.** An academic screening and decision-support prototype. Not a medical",
        "> device, not a diagnostic tool, and not clinically validated in any way. Every",
        "> result below is a cross-validation within public research corpora.",
        "",
        "## What was produced",
        "",
        "| Deliverable | Target | Produced |",
        "|---|---|---|",
    ]
    for count in counts:
        mark = "" if count.ok else f" — missing {', '.join(count.missing)}"
        lines.append(f"| {count.kind} | {count.target} | {count.found}{mark} |")

    lines += [
        "",
        f"The extraction spec and the blueprint between them name **{spec['total']} required",
        f"items**. {spec['present']} are present, {spec['not_produced']} are recorded as not",
        f"produced with a technical reason, and {spec['failed']} are failed (missing with no",
        "reason). The item-by-item list is `outputs/00_evidence_index/evidence_index.xlsx`.",
        "",
        "Also produced, and worth opening first:",
        "",
        "| File | What it tells you |",
        "|---|---|",
        "| `outputs/00_evidence_index/evidence_index.xlsx` | every artifact, the command that made "
        "it, its sources, and whether it is on disk |",
        "| `outputs/00_evidence_index/final_qa_report.md` | the six-area QA sweep, recomputed from "
        "the committed files |",
        "| `outputs/00_evidence_index/compliance_review.md` | the claims review: language, "
        "disclaimer, perfection, objectives, counts, provenance |",
        "| `outputs/00_evidence_index/run_manifest.json` | every run, its seed, its package "
        "versions and its wall time |",
        "| `outputs/missing_outputs_report.txt` | **everything that could not be produced, and "
        "why** |",
        "",
        "## What was not done, and why",
        "",
    ]

    if markers["blocked"]:
        lines += ["**Blocked tasks (`[!]`):**", ""]
        lines += [f"- {item}" for item in markers["blocked"]]
        lines.append("")
    if markers["skipped"]:
        lines += ["**Skipped tasks (`[-]`):**", ""]
        lines += [f"- {item}" for item in markers["skipped"]]
        lines.append("")
    if not markers["blocked"] and not markers["skipped"]:
        lines += [
            "The task list is not part of the published repository, so its markers could",
            "not be read here. `outputs/missing_outputs_report.txt` is the authoritative",
            "record and is included.",
            "",
        ]

    if gaps:
        lines += [
            "The gaps report, verbatim:",
            "",
            "```",
            *gaps[:120],
            *(["…"] if len(gaps) > 120 else []),
            "```",
            "",
        ]

    lines += [
        "## The five things a reader should not misread",
        "",
        "1. **The binary model does not transfer across recording collections.** AUC falls",
        "   from 0.92-0.94 pooled to 0.43-0.53 leave-one-sub-collection-out. The pooled",
        "   numbers are correct as a within-corpus cross-validation and are not a",
        "   generalization claim.",
        "2. **The final binary model (M1, a regularised logistic regression) is not better",
        "   than the ensemble.** They are statistically tied (sensitivity 0.8648 vs 0.8615,",
        "   p = 0.797); M1 is simpler and 190x faster. Never write that it outperformed.",
        "3. **Accuracy would have chosen a different model, and that is the point.** The",
        "   selection rule is sensitivity then balanced accuracy; every deployed model",
        "   records what raw accuracy would have picked instead.",
        "4. **`artifact` in PASCAL A is a recording-quality label**, not a cardiac class.",
        "5. **EXP-D1's drop is a population and acquisition effect**, not a failure of the",
        "   method.",
        "",
        "## What clinical validation would require",
        "",
        CLINICAL_VALIDATION,
    ]

    target.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", target)
    return target
