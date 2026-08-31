"""The six locked objectives, verbatim, with what in this repository answers them.

T114.1 asks for "the six locked objectives verbatim" on the home page, and
T125.4 asks for a check that they appear verbatim wherever quoted, "with no
paraphrasing". The blueprint's own instruction is blunter than either:

    The following objectives are locked. Do not change, shorten, paraphrase or
    reword them in implementation documents, reports, thesis chapters, paper
    drafts or client-facing material.

## Why they are declared here rather than typed into a page

`Docs/` is gitignored, so the blueprint PDF does not reach CI and no build-time
step can re-read it. The wording therefore has to live in the repository — and
the wrong place for it is a `.tsx` file, where nothing would ever check it
again and a well-meaning edit to "fix" the grammar of objective 5 would be
invisible.

So this module follows the pattern `architecture.py` and `equations.py` already
use: the content is **declared once in Python and checked against the
repository at export time**. Each objective names the modules and `outputs/`
directories that answer it, and :func:`objectives_payload` refuses to emit an
objective whose evidence is missing. Each also carries the sha256 of its own
wording, so "verbatim" is a comparison a test can make rather than a claim a
reviewer has to eyeball.

:func:`verify_against_source` re-extracts the wording from the blueprint PDF and
compares it, digest for digest. It runs where `Docs/` is present — which is this
machine and not CI — and is the check that keeps the declaration honest.

## Two transcription notes, neither of them a correction

* **Objective 2** is printed across a line break as `state-` / `of-the-art`. The
  wrap is closed to `state-of-the-art`. Dropping the hyphen instead would give
  `stateof-the-art`, which is what a naive dehyphenation produces and is wrong.
* **Objective 3** contains typographic quotation marks around `"ensemble"`
  (U+201C, U+201D), not ASCII quotes. They are preserved. Normalising them
  would be a change to locked wording, however small it looks.

## What "answers an objective" means here

The evidence lists are deliberately modest. An objective is answered by code
that exists and outputs that were produced; it is not answered by a number being
good. Objective 6 in particular is scoped honestly: PASCAL A carries 19 samples
in one of its four classes, which is why the PhysioNet diagnosis track exists,
and the note on that objective says so rather than implying four-class PASCAL
carries it alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "BLUEPRINT_PDF",
    "BLUEPRINT_SHA256",
    "OBJECTIVES",
    "Objective",
    "objectives_payload",
    "verify_against_source",
    "wording_digest",
]

log = get_logger("reporting.objectives")

#: Where the locked wording came from. Gitignored, so this is provenance rather
#: than a build dependency: the wording below is the committed copy.
BLUEPRINT_PDF = "Docs/Kharat_Mam_Developer_Blueprint_PulseVision_PCG_Heart_Sound_Classification.pdf"

#: sha256 of that PDF at the time the wording was extracted (page 2, section 1,
#: "Final Locked Objectives"). If the blueprint is ever reissued this changes,
#: and `verify_against_source` is how the difference is found rather than
#: assumed absent.
BLUEPRINT_SHA256 = "18ccac2d86f3ffdc4cbb6457fdf7abe1c499b9b41675efc80d0cfe91cb10c73d"

BLUEPRINT_PAGE = 2


@dataclass(frozen=True)
class Objective:
    """One locked objective and the repository evidence that answers it."""

    number: int
    #: The blueprint's wording, unchanged. Never edited, never paraphrased.
    wording: str
    #: A short handle for the objective. NOT a substitute for the wording, and
    #: never rendered in place of it -- it labels a card whose body is the
    #: wording itself.
    handle: str
    #: Repository-relative module paths that implement this objective. Verified.
    modules: tuple[str, ...]
    #: `outputs/` subdirectories evidencing it. Verified.
    evidence_dirs: tuple[str, ...]
    #: What a reader should know before reading the objective as answered.
    caveat: str | None = None
    #: Required when `evidence_dirs` is empty. An objective with no evidence yet
    #: says which phase produces it, rather than pointing at a directory that
    #: does not exist -- inventing a path to satisfy the check would defeat it.
    pending_reason: str | None = None


OBJECTIVES: tuple[Objective, ...] = (
    Objective(
        number=1,
        wording=(
            "To introduce a new approach for the classification of normal and abnormal "
            "heart sound recordings using a machine learning algorithm. An appropriate "
            "algorithm for auto diagnosis systems of heart diseases, capable of "
            "distinguishing most known pathological states, must be developed."
        ),
        handle="Normal versus abnormal classification",
        modules=(
            "src/models/pipeline.py",
            "src/models/estimators.py",
            "src/inference/predictor.py",
        ),
        evidence_dirs=("outputs/06_binary_results",),
        caveat=(
            "This is a screening and decision-support prototype. The blueprint's "
            "phrase “auto diagnosis systems” is its wording, not a claim made "
            "by this implementation: nothing here diagnoses, and every result is "
            "reported as a screening signal."
        ),
    ),
    Objective(
        number=2,
        wording=(
            "To review recently published preprocessing, feature extraction, and "
            "classification techniques, as well as the state-of-the-art in "
            "phonocardiogram (PCG) signal analysis."
        ),
        handle="State-of-the-art review",
        modules=("src/reporting/tables.py",),
        evidence_dirs=(),
        pending_reason=(
            "The literature review is Phase 101 and has not run. No outputs "
            "directory is named here on purpose: pointing at one that does not "
            "exist, or at a neighbouring directory that holds something else, "
            "would turn this check into a formality."
        ),
    ),
    Objective(
        number=3,
        wording=(
            "To utilise ensemble machine learning algorithms that combine the "
            "predictions of several learning models into a single “ensemble” "
            "model to improve overall performance."
        ),
        handle="Heterogeneous ensemble",
        modules=("src/ensemble/soft_voting.py", "src/optimization/weights.py"),
        evidence_dirs=("outputs/05_search_optimization",),
    ),
    Objective(
        number=4,
        wording=(
            "To understand heart sound classification techniques that might find use in "
            "medical diagnostic systems; the objective is to accurately classify normal "
            "and abnormal heart sounds from single, short, and potentially noisy "
            "recordings."
        ),
        handle="Short, noisy, single recordings",
        modules=("src/preprocessing/pipeline.py", "src/feature_extraction/extractor.py"),
        evidence_dirs=("outputs/02_preprocessing", "outputs/03_features"),
    ),
    Objective(
        number=5,
        wording=(
            "Apply a search algorithm to improve the performance of the diagnostic "
            "system in terms of accuracy, complexity, and the range of distinguishable "
            "heart sounds."
        ),
        handle="Search-driven optimization",
        modules=("src/optimization/driver.py", "src/feature_selection/ranking.py"),
        evidence_dirs=("outputs/05_search_optimization",),
    ),
    Objective(
        number=6,
        wording=(
            "To introduce a greater number of classes (types of heart sounds) to test "
            "the final system further, making it capable of handling more tests on a "
            "larger number of samples."
        ),
        handle="More classes, more samples",
        modules=("src/models/pipeline.py",),
        evidence_dirs=("outputs/07_multiclass_results",),
        caveat=(
            "PASCAL A has 124 records with 19 in one class, which is too few to carry "
            "this objective alone. The PhysioNet diagnosis track (3,240 records) is "
            "what answers it at scale. PASCAL's “artifact” label is a "
            "recording-quality category, not a cardiac class, so the four-class model "
            "is never described as a four-class cardiac classifier."
        ),
    ),
)


def wording_digest(wording: str) -> str:
    """sha256 of an objective's wording, as UTF-8.

    The unit "verbatim" is measured in. A test comparing two digests catches a
    changed quotation mark, a dropped hyphen and a silently Americanised
    "utilize", none of which a reader reliably notices.
    """
    return hashlib.sha256(wording.encode("utf-8")).hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def objectives_payload() -> dict[str, Any]:
    """The six objectives as exportable data, each checked against the repository.

    Raises when a declared module or evidence directory is missing, so an
    objective cannot be presented as answered by code that is not there. An
    evidence directory that exists but is empty is reported as `pending` rather
    than dropped: "this has not run yet" and "this does not exist" are different
    statements and the page should be able to make either.
    """
    root = _project_root()
    rows: list[dict[str, Any]] = []

    for objective in OBJECTIVES:
        for module in objective.modules:
            if not (root / module).is_file():
                raise FileNotFoundError(
                    "objective "
                    + str(objective.number)
                    + " names a module that does not exist: "
                    + module
                )

        if not objective.evidence_dirs and not objective.pending_reason:
            raise ValueError(
                "objective "
                + str(objective.number)
                + " declares no evidence directory and gives no reason"
            )

        evidence: list[dict[str, Any]] = []
        for directory in objective.evidence_dirs:
            path = root / directory
            if not path.is_dir():
                raise FileNotFoundError(
                    "objective "
                    + str(objective.number)
                    + " names an outputs directory that does not exist: "
                    + directory
                )
            n_files = sum(1 for item in path.rglob("*") if item.is_file())
            evidence.append(
                {
                    "dir": directory,
                    "n_files": n_files,
                    "status": "produced" if n_files else "pending",
                }
            )

        rows.append(
            {
                "number": objective.number,
                "label": "Objective " + str(objective.number),
                "handle": objective.handle,
                "wording": objective.wording,
                "wording_sha256": wording_digest(objective.wording),
                "modules": list(objective.modules),
                "evidence": evidence,
                "caveat": objective.caveat,
                "pending_reason": objective.pending_reason,
                "status": (
                    "produced"
                    if evidence and all(item["status"] == "produced" for item in evidence)
                    else "pending"
                ),
            }
        )

    return {
        "n_objectives": len(rows),
        "source": BLUEPRINT_PDF,
        "source_page": BLUEPRINT_PAGE,
        "source_sha256": BLUEPRINT_SHA256,
        "locked_notice": (
            "These objectives are locked by the source document: they are not "
            "changed, shortened, paraphrased or reworded anywhere in this project. "
            "The wording below is quoted exactly, and each is published with the "
            "sha256 of its own text so a paraphrase is detectable rather than "
            "arguable."
        ),
        "transcription_notes": [
            "Objective 2 is printed across a line break as 'state-' / 'of-the-art'; "
            "the wrap is closed to 'state-of-the-art'. Dropping the hyphen instead "
            "gives 'stateof-the-art', which is what a naive dehyphenation produces.",
            "Objective 3 uses typographic quotation marks (U+201C, U+201D) around "
            "'ensemble'. They are preserved rather than normalised to ASCII.",
        ],
        "objectives": rows,
    }


def verify_against_source(pdf_path: str | Path | None = None) -> dict[str, Any]:
    """Re-extract the objectives from the blueprint and compare, digest by digest.

    Returns a report rather than raising, so a caller can distinguish "the PDF is
    not here" (the CI case, and not a failure) from "the PDF says something else"
    (a real finding). Needs `pypdf`, which is deliberately not a project
    dependency — it is installed for this check and removed again, the same way
    Phase 113 extracted the equations.
    """
    source = Path(pdf_path) if pdf_path is not None else _project_root() / BLUEPRINT_PDF
    if not source.is_file():
        return {"checked": False, "reason": "blueprint PDF not present: " + str(source)}

    try:
        from pypdf import PdfReader
    except ImportError:
        return {"checked": False, "reason": "pypdf is not installed"}

    import re

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    text = PdfReader(str(source)).pages[BLUEPRINT_PAGE - 1].extract_text() or ""
    block = text.split("Objective No. Locked Objective Wording", 1)[-1]
    block = block.split("2. Dataset Inventory")[0]
    parts = re.split(r"Objective\s+([1-6])\s*\n", block)

    extracted: dict[int, str] = {}
    for index in range(1, len(parts), 2):
        body = " ".join(parts[index + 1].split())
        # Close a hyphenated line wrap WITHOUT losing the hyphen; see the
        # objective-2 transcription note.
        body = re.sub(r"-\s+(?=[a-z])", "-", body)
        extracted[int(parts[index])] = body

    mismatches = [
        {
            "number": objective.number,
            "declared": objective.wording,
            "extracted": extracted.get(objective.number),
        }
        for objective in OBJECTIVES
        if extracted.get(objective.number) != objective.wording
    ]

    return {
        "checked": True,
        "pdf_sha256": digest,
        "pdf_sha256_matches": digest == BLUEPRINT_SHA256,
        "n_extracted": len(extracted),
        "mismatches": mismatches,
    }
