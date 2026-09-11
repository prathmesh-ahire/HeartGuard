"""The thesis asset pack (Phase 104).

``outputs/THESIS_ASSETS/`` is a chapter-shaped copy of the deliverables that
already exist -- nothing in it is generated here except the reproducibility
appendix and the indexes. Every copy is byte-identical to its source and the
manifest records both sha256 digests, so a thesis figure can always be traced
back to the file a phase wrote and, through that file's own provenance, to the
run that produced it.

Chapter placement follows the blueprint's section 17:

Ch1 Introduction     problem, objectives, scope -- the architecture, track and novelty diagrams
Ch2 Literature       LIT-01 / LIT-02
Ch3 Methodology      datasets, preprocessing, features, models, search: T01-T07, G01-G10,
                     F02 and F04-F10, ALG-01..ALG-12, the equations reference
Ch4 Implementation   pipelines, dashboard, reports: F11-F17, ALG-13..ALG-19, the
                     reproducibility appendix (dashboard screenshots are Part X)
Ch5 Results          T08-T28, G11-G35
Ch6 Conclusion       T29 objective achievement, T30 conclusion matrix, F18, F20, ALG-20
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.evidence import PROJECT_ROOT, read_evidence, register_evidence
from src.utils.logging_setup import get_logger

__all__ = [
    "CHAPTERS",
    "THESIS_DIR",
    "COMMAND",
    "EXPECTED",
    "chapter_of",
    "planned_copies",
    "write_thesis_pack",
    "verify_thesis_pack",
]

log = get_logger(__name__)

THESIS_DIR = "outputs/THESIS_ASSETS"
COMMAND = "python scripts/44_thesis_assets.py"
MANIFEST = "thesis_asset_manifest.csv"
FIG = "outputs/13_figures_diagrams"
ALG = "outputs/14_algorithms"

CHAPTERS = (
    "Ch1_Introduction",
    "Ch2_Literature",
    "Ch3_Methodology",
    "Ch4_Implementation",
    "Ch5_Results",
    "Ch6_Conclusion",
)

#: T104.7's reconciliation: the counts the extraction spec names.
EXPECTED = {"table": 30, "graph": 35, "diagram": 20, "algorithm": 20}

DISCLAIMER = (
    "PV-MEPCG / PulseVision is an academic screening and decision-support prototype. "
    "It does not diagnose, and nothing in it is a diagnosis, a treatment recommendation "
    "or a substitute for clinical judgement."
)


def _number(asset_id: str) -> int:
    digits = "".join(ch for ch in asset_id if ch.isdigit())
    return int(digits)


def chapter_of(asset_id: str) -> str:
    """The chapter folder an asset belongs in."""
    if asset_id.startswith("LIT-"):
        return "Ch2_Literature"
    if asset_id.startswith("EQUATIONS"):
        return "Ch3_Methodology"
    if asset_id.startswith("ALG-"):
        n = _number(asset_id)
        return (
            "Ch3_Methodology" if n <= 12 else "Ch4_Implementation" if n <= 19 else "Ch6_Conclusion"
        )
    kind, n = asset_id[0], _number(asset_id)
    if kind == "T":
        return "Ch3_Methodology" if n <= 7 else "Ch5_Results" if n <= 28 else "Ch6_Conclusion"
    if kind == "G":
        return "Ch3_Methodology" if n <= 10 else "Ch5_Results"
    if kind == "F":
        if n in (1, 3, 19):
            return "Ch1_Introduction"
        if n in (18, 20):
            return "Ch6_Conclusion"
        return "Ch3_Methodology" if n <= 10 else "Ch4_Implementation"
    raise ValueError("no chapter for " + asset_id)


@dataclass(frozen=True)
class Copy:
    asset_id: str
    kind: str
    chapter: str
    source: str  # repo-relative


def _tables() -> list[Copy]:
    import json

    copies: list[Copy] = []
    for number in range(1, 31):
        table_id = "T" + str(number).zfill(2)
        metas = sorted(
            p
            for p in (PROJECT_ROOT / "outputs").rglob(table_id + "_*.meta.json")
            if "_checkpoints" not in p.parts and "THESIS_ASSETS" not in p.parts
        )
        if len(metas) != 1:
            raise FileNotFoundError(
                table_id + ": expected one .meta.json, found " + str(len(metas))
            )
        meta = json.loads(metas[0].read_text(encoding="utf-8"))
        stem = metas[0].name[: -len(".meta.json")]
        folder = metas[0].parent.relative_to(PROJECT_ROOT).as_posix()
        for suffix in (".csv", ".docx", ".meta.json"):
            copies.append(
                Copy(table_id, "table", chapter_of(table_id), folder + "/" + stem + suffix)
            )
        if meta["table_id"] != table_id:
            raise ValueError(str(metas[0]) + " declares " + str(meta["table_id"]))
    return copies


def _graphs() -> list[Copy]:
    from src.reporting.graphs import read_registry

    rows = {
        row["figure_id"]: row for row in read_registry(PROJECT_ROOT / FIG / "figure_registry.csv")
    }
    copies: list[Copy] = []
    for number in range(1, 36):
        graph_id = "G" + str(number).zfill(2)
        row = rows[graph_id]
        stem = row["filename"][: -len(".png")]
        for name in (row["filename"], row["source_csv"], stem + ".meta.json"):
            copies.append(Copy(graph_id, "graph", chapter_of(graph_id), FIG + "/" + name))
    return copies


def _diagrams() -> list[Copy]:
    copies: list[Copy] = []
    for number in range(1, 21):
        figure_id = "F" + str(number).zfill(2)
        svgs = sorted((PROJECT_ROOT / FIG).glob(figure_id + "_*.svg"))
        if len(svgs) != 1:
            raise FileNotFoundError(figure_id + ": expected one SVG, found " + str(len(svgs)))
        stem = svgs[0].name[: -len(".svg")]
        for suffix in (".svg", ".png", ".meta.json"):
            copies.append(
                Copy(figure_id, "diagram", chapter_of(figure_id), FIG + "/" + stem + suffix)
            )
    return copies


def _algorithms() -> list[Copy]:
    copies: list[Copy] = []
    for number in range(1, 21):
        alg_id = "ALG-" + str(number).zfill(2)
        txts = sorted((PROJECT_ROOT / ALG).glob(alg_id + "_*.txt"))
        if len(txts) != 1:
            raise FileNotFoundError(alg_id + ": expected one .txt, found " + str(len(txts)))
        stem = txts[0].name[: -len(".txt")]
        for suffix in (".docx", ".txt"):
            copies.append(Copy(alg_id, "algorithm", chapter_of(alg_id), ALG + "/" + stem + suffix))
    for suffix in (".docx", ".tex", ".csv"):
        copies.append(
            Copy("EQUATIONS", "equations", "Ch3_Methodology", ALG + "/equations_reference" + suffix)
        )
    return copies


def _literature() -> list[Copy]:
    folder = "outputs/16_literature_review"
    copies: list[Copy] = []
    for lit_id in ("LIT-01", "LIT-02"):
        for path in sorted((PROJECT_ROOT / folder).glob(lit_id + "_*")):
            if path.suffix in (".csv", ".docx") or path.name.endswith(".meta.json"):
                copies.append(
                    Copy(lit_id, "literature", "Ch2_Literature", folder + "/" + path.name)
                )
    return copies


def planned_copies() -> list[Copy]:
    return _tables() + _graphs() + _diagrams() + _algorithms() + _literature()


def _sha256(path: Path) -> str:
    """Content digest, newline-normalized for text (``tables.content_digest``).

    A raw byte hash calls a git-checked-out CRLF copy of an LF file "changed";
    the text is the same, so the digest must be too. Binary files hash raw.
    """
    from src.reporting.tables import content_digest

    return content_digest(path)[0]


# ---------------------------------------------------------------------------
# the reproducibility appendix (T104.6)
# ---------------------------------------------------------------------------


def _write_appendix(target: Path) -> Path:
    import json

    from docx import Document
    from docx.shared import Pt

    manifest = json.loads(
        (PROJECT_ROOT / "outputs/00_evidence_index/run_manifest.json").read_text(encoding="utf-8")
    )
    final = manifest.get("final")
    if final is None:
        raise RuntimeError(
            "run_manifest.json has no final block; run scripts/42_evidence_index.py first"
        )

    document = Document()
    document.add_paragraph("Appendix: Reproducibility of PV-MEPCG / PulseVision", style="Title")
    document.add_paragraph(
        "Everything in this appendix is read from outputs/00_evidence_index/run_manifest.json "
        "(its final block, written by python scripts/42_evidence_index.py) and from the evidence "
        "index when this file is generated. " + DISCLAIMER
    )

    def table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
        grid = document.add_table(rows=1, cols=len(headers))
        grid.style = "Table Grid"
        for cell, header in zip(grid.rows[0].cells, headers, strict=True):
            cell.text = header
        for row in rows:
            cells = grid.add_row().cells
            for cell, value in zip(cells, row, strict=True):
                cell.text = value
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(7)

    environment = final["environment"]
    document.add_paragraph("Environment", style="Heading 1")
    table(
        ("Property", "Value"),
        [
            ("Python", str(environment.get("python_version"))),
            ("Platform", str(environment.get("platform"))),
            ("Processor", str(environment.get("processor"))),
            ("CPU cores", str(environment.get("cpu_count"))),
            ("GPU", str(environment.get("gpu"))),
            ("Git commit at finalization", str(final["git"].get("commit"))),
            ("Finalized", str(final["finalized_utc"])),
        ],
    )

    document.add_paragraph("Seed discipline", style="Heading 1")
    seed = final["seed"]
    document.add_paragraph(
        "Global seed: "
        + str(seed["global_seed"])
        + ". Runs by recorded seed: "
        + ", ".join(key + " -> " + str(value) for key, value in seed["runs_by_seed"].items())
        + ". Runs recording a different seed: "
        + (", ".join(seed["runs_with_another_seed"]) or "none")
        + "."
    )

    document.add_paragraph("Pinned packages", style="Heading 1")
    table(
        ("Package", "Version"),
        [(name, str(version)) for name, version in sorted(final["packages"].items())],
    )

    document.add_paragraph("Wall time per pipeline stage", style="Heading 1")
    document.add_paragraph(
        "Runs record the pipeline stage (the script's run name). A run still marked running "
        "ended without finishing -- killed, crashed or interrupted -- and has no duration; none "
        "is estimated."
    )
    table(
        (
            "Stage",
            "Runs",
            "Finished",
            "Unfinished",
            "Total wall (s)",
            "Longest run (s)",
            "First started (UTC)",
        ),
        [
            (
                str(stage["stage"]),
                str(stage["n_runs"]),
                str(stage["n_finished"]),
                str(stage["n_unfinished"]),
                format(stage["total_wall_seconds"], ".1f"),
                format(stage["max_wall_seconds"], ".1f"),
                str(stage["first_started_utc"]),
            )
            for stage in final["stage_timing"]
        ],
    )

    document.add_paragraph("Reproduction commands", style="Heading 1")
    document.add_paragraph(
        "Every command recorded in the evidence index, with the evidence ids it produces. "
        "Run from the repository root with the virtual environment active."
    )
    by_command: dict[str, list[str]] = {}
    for row in read_evidence():
        if row.get("command"):
            by_command.setdefault(row["command"], []).append(row["evidence_id"])
    table(
        ("Command", "Evidence ids"),
        [(command, ", ".join(ids)) for command, ids in sorted(by_command.items())],
    )

    path = target / "Ch4_Implementation" / "reproducibility_appendix.docx"
    document.save(str(path))
    return path


# ---------------------------------------------------------------------------
# writing and verifying
# ---------------------------------------------------------------------------


def thesis_dir(out_dir: str | Path | None = None) -> Path:
    return Path(out_dir) if out_dir is not None else PROJECT_ROOT / THESIS_DIR


_MANIFEST_COLUMNS = ("chapter", "asset_id", "kind", "file", "source", "sha256")


def write_thesis_pack(
    out_dir: str | Path | None = None, *, evidence_index: str | Path | None = None
) -> dict[str, Any]:
    target = thesis_dir(out_dir)
    for chapter in CHAPTERS:
        (target / chapter).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for copy in planned_copies():
        source = PROJECT_ROOT / copy.source
        destination = target / copy.chapter / source.name
        shutil.copyfile(source, destination)
        rows.append(
            {
                "chapter": copy.chapter,
                "asset_id": copy.asset_id,
                "kind": copy.kind,
                "file": copy.chapter + "/" + source.name,
                "source": copy.source,
                "sha256": _sha256(destination),
            }
        )

    appendix = _write_appendix(target)
    rows.append(
        {
            "chapter": "Ch4_Implementation",
            "asset_id": "THESIS-REPRODUCIBILITY",
            "kind": "appendix",
            "file": "Ch4_Implementation/" + appendix.name,
            "source": (
                "outputs/00_evidence_index/run_manifest.json; "
                "outputs/00_evidence_index/evidence_index.csv"
            ),
            "sha256": _sha256(appendix),
        }
    )

    for chapter in CHAPTERS:
        chapter_rows = [row for row in rows if row["chapter"] == chapter]
        index = target / chapter / "chapter_index.csv"
        _write_csv(index, chapter_rows)
        register_evidence(
            "THESIS-" + chapter,
            index,
            metric_or_asset=chapter
            + ": "
            + ", ".join(
                f"{n} {k}" for k, n in sorted(Counter(r["kind"] for r in chapter_rows).items())
            ),
            objective="O1-O6",
            source_data="; ".join(dict.fromkeys(row["source"] for row in chapter_rows)),
            command=COMMAND,
            index_path=evidence_index,
        )
    register_evidence(
        "THESIS-REPRODUCIBILITY",
        appendix,
        metric_or_asset=(
            "reproducibility appendix: environment, seeds, packages, stage timing, commands"
        ),
        source_data="outputs/00_evidence_index/run_manifest.json",
        command=COMMAND,
        index_path=evidence_index,
    )
    manifest = target / MANIFEST
    _write_csv(manifest, rows)
    counts = Counter(row["kind"] for row in rows)
    assets = {kind: len({r["asset_id"] for r in rows if r["kind"] == kind}) for kind in counts}
    return {"manifest": manifest, "files": len(rows), "assets": assets}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def verify_thesis_pack(out_dir: str | Path | None = None) -> list[str]:
    """T104.7: folders populated, counts reconcile, every copy identical to its source."""
    target = thesis_dir(out_dir)
    manifest = target / MANIFEST
    if not manifest.is_file():
        return ["no " + MANIFEST]
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    problems: list[str] = []
    for chapter in CHAPTERS:
        folder = target / chapter
        files = (
            [p for p in folder.glob("*") if p.name != "chapter_index.csv"]
            if folder.is_dir()
            else []
        )
        if not files:
            problems.append(chapter + " is empty")
    for kind, expected in EXPECTED.items():
        found = len({row["asset_id"] for row in rows if row["kind"] == kind})
        if found != expected:
            problems.append(kind + ": " + str(found) + " assets, expected " + str(expected))
    for row in rows:
        copied = target / row["file"]
        if not copied.is_file():
            problems.append("missing " + row["file"])
            continue
        if _sha256(copied) != row["sha256"]:
            problems.append("changed since packing: " + row["file"])
        if row["kind"] != "appendix":
            source = PROJECT_ROOT / row["source"]
            if not source.is_file():
                problems.append("source gone: " + row["source"])
            elif _sha256(source) != row["sha256"]:
                problems.append("stale copy (source changed): " + row["file"])
    required = {
        "table": (".csv", ".docx"),
        "graph": (".png", ".csv"),
        "diagram": (".svg", ".png"),
        "algorithm": (".docx", ".txt"),
    }
    for kind, suffixes in required.items():
        for asset_id in {row["asset_id"] for row in rows if row["kind"] == kind}:
            have = {Path(row["file"]).suffix for row in rows if row["asset_id"] == asset_id}
            for suffix in suffixes:
                if suffix not in have:
                    problems.append(asset_id + " has no " + suffix)
    return problems
