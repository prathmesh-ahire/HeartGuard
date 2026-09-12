"""The thirteen dashboard screenshots (Phase 120), declared once, in Python.

The extraction spec's section 14 names thirteen screenshots. They are declared
here -- number, route, what has to be on screen, and the caption -- and consumed
from two directions:

* ``frontend/screenshots/dashboard.spec.ts`` reads ``capture_plan.json`` and
  drives a real browser against the real built site and the real inference API.
  Nothing about *which* pages are captured, or in what order, lives in the
  TypeScript.
* :func:`finalize` turns what the browser wrote into ``captions.md``,
  ``screenshot_index.csv`` and thirteen ``SS-nn`` rows in the evidence index.

Two rules this file exists to enforce
-------------------------------------

**A screenshot is of an audited build.** ``playwright.screenshots.config.ts``
runs :func:`src.reporting.display_audit.require_passed_audit` as its global
setup (T119.4), so the capture aborts before the browser opens if the site on
disk is not the one the displayed-value audit passed on.

**A screenshot of an empty state is a failure, not a screenshot.** T120.7 asks
for "no placeholder or empty chart in any of them", so every spec carries
``must_contain`` / ``must_not_contain`` text that the browser asserts *before*
the shutter, and :func:`finalize` refuses any file the capture did not write or
that came back suspiciously small.

The six prediction captures need a running inference service and a local
``dataset/`` copy, because the built-in samples are resolved from the operator's
own corpus rather than committed (see ``src/reporting/samples.py``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "CAPTURE_PLAN",
    "PLAN_FILENAME",
    "ScreenshotError",
    "ShotSpec",
    "finalize",
    "plan_payload",
    "write_plan",
]

log = get_logger("reporting.screenshots")

PLAN_FILENAME = "capture_plan.json"
CAPTIONS_FILENAME = "captions.md"
INDEX_FILENAME = "screenshot_index.csv"

#: Below this a PNG is a blank or error frame rather than a page. The smallest
#: real capture in this set is ~90 kB; 20 kB is a floor, not a target.
MIN_BYTES = 20_000

DISCLAIMER_MARK = "Screening and research use only."


class ScreenshotError(RuntimeError):
    """A screenshot is missing, empty, or was taken of the wrong thing."""


@dataclass(frozen=True)
class ShotSpec:
    """One numbered screenshot and everything the browser needs to take it."""

    number: int
    #: The extraction spec's own name for this screenshot (section 14).
    title: str
    #: The route to open, relative to the site root.
    route: str
    #: What the reader is looking at, written for a thesis figure caption.
    caption: str
    #: Steps the browser performs before the shutter. Each is a dict the
    #: TypeScript spec knows how to execute; see `dashboard.spec.ts`.
    steps: tuple[dict[str, Any], ...] = ()
    #: Text that must be on screen when the shutter fires.
    must_contain: tuple[str, ...] = ()
    #: Text that must NOT be -- an unbuilt route, an empty state, a chart that
    #: failed to draw. T120.7's "no placeholder" as an assertion.
    must_not_contain: tuple[str, ...] = (
        "Route scaffolded; content not built yet.",
        "No model is deployed for this task yet",
        "This chart could not be drawn.",
        "Nothing scored yet",
    )
    #: Overrides the route-derived filename slug, for a shot whose capture
    #: route is not the page it is a screenshot OF.
    slug: str = ""
    #: Capture the whole scrollable page rather than the viewport.
    full_page: bool = True
    #: Viewport height. A full-page capture still needs one; a long results page
    #: reads better from a taller viewport because the sticky header and the
    #: lazy `Reveal` sections settle once rather than repeatedly.
    height: int = 1400

    @property
    def shot_id(self) -> str:
        return "SS-" + str(self.number).zfill(2)

    @property
    def filename(self) -> str:
        slug = self.slug or self.route.strip("/").replace("/", "_") or "home"
        return self.shot_id + "_" + slug + ".png"


def _sample(sample_id: str) -> dict[str, Any]:
    """Pick a built-in dataset sample by id in the prediction panel."""
    return {"action": "sample", "sample_id": sample_id}


def _click(name: str) -> dict[str, Any]:
    return {"action": "click", "name": name}


def _await_result() -> dict[str, Any]:
    return {"action": "await_result"}


#: The disclaimer is asserted on every capture: T120.7 requires it visible in
#: all thirteen, and asserting it per spec would let one be forgotten.
CAPTURE_PLAN: tuple[ShotSpec, ...] = (
    ShotSpec(
        1,
        "Home and project overview",
        "/",
        "The PV-MEPCG / PulseVision dashboard home page: the framework in one "
        "line, the six locked research objectives, the pipeline from recording "
        "to screening indication, and the scope-and-safety notice that appears "
        "on every route.",
        must_contain=("PV-MEPCG",),
    ),
    ShotSpec(
        2,
        "Dataset inventory and class distribution",
        "/dataset/",
        "The four public PCG corpora as audited against the files on disk "
        "(T01): per-dataset inventory, class balance and the duration profile. "
        "Every count is read from outputs/01_dataset_audit/, never from the "
        "source documents, which disagree with the files in two places.",
    ),
    ShotSpec(
        3,
        "Signal upload and waveform",
        "/predict/binary/",
        "Choosing a recording on the binary screening page. A built-in dataset "
        "sample has been selected and its waveform drawn from audio served by "
        "the local inference service; no corpus audio is committed to the "
        "repository. The model has not been run yet -- this is the upload and "
        "preview step alone.",
        steps=(_sample("binary-abnormal"),),
        must_contain=("1. Choose a recording", "D1_training-b_b0033"),
        # `Nothing scored yet` is the correct state here: the point of this
        # screenshot is the step BEFORE scoring.
        must_not_contain=(
            "Route scaffolded; content not built yet.",
            "No model is deployed for this task yet",
            "This chart could not be drawn.",
        ),
    ),
    ShotSpec(
        4,
        "Preprocessing before/after",
        "/preprocessing/",
        "The preprocessing chain (T02): resampling, band-pass filtering, "
        "normalization and signal-quality assessment, with the same recording "
        "before and after each stage and the spectral effect of the filter.",
    ),
    ShotSpec(
        5,
        "Feature extraction summary",
        "/features/",
        "The locked 138-feature registry across six families (T03), the "
        "fold-safe subset selected inside the training folds, and the family "
        "each selected feature belongs to. The column order is a literal in "
        "the registry and is fingerprinted, not derived.",
    ),
    ShotSpec(
        6,
        "Model comparison dashboard",
        "/models/",
        "The declared models and their fold-wise comparison (T04, T08). "
        "Sensitivity, specificity, F1, balanced accuracy and AUC are reported "
        "together; accuracy alone never decides anything here.",
    ),
    ShotSpec(
        7,
        "Search optimization dashboard",
        "/optimization/",
        "The search space, the methods compared at equal budget, the selected "
        "parameters and what the search actually bought (T05-T07). Every "
        "search shown ran inside a training fold.",
    ),
    ShotSpec(
        8,
        "Binary prediction output",
        "/predict/binary/",
        "A live binary screening indication for one PhysioNet 2016 recording, "
        "produced by POST /predict against the deployed model. The panel shows "
        "the class probabilities, the confidence margin, the operating "
        "threshold and the corpus label, and states that the recording's "
        "stored out-of-fold probability is five numbers rather than one.",
        steps=(_sample("binary-abnormal"), _click("2. Run the screening model"), _await_result()),
        must_contain=("3. Result",),
    ),
    ShotSpec(
        9,
        "Multiclass prediction output",
        "/predict/multiclass/",
        "A live PASCAL A four-class acoustic-event output with the full "
        "probability distribution. PASCAL A and PASCAL B are offered as two "
        "separate label spaces on this page and are never merged; `artifact` "
        "is a recording-quality label, not a cardiac class.",
        steps=(_sample("pascal-a-murmur"), _click("2. Run the screening model"), _await_result()),
        must_contain=("3. Result", "PASCAL A"),
    ),
    ShotSpec(
        10,
        "CirCor murmur/outcome output",
        "/predict/murmur/",
        "The CirCor page with both of its label spaces exercised: a live "
        "clinical-outcome indication for one recording above, and below it the "
        "same subject collapsed from all four auscultation locations to a "
        "patient-level murmur indication under every declared aggregation "
        "rule. Murmur and outcome are separate tasks with separate models.",
        steps=(
            _click("Clinical outcome"),
            _sample("circor-murmur-present"),
            _click("2. Run the screening model"),
            _await_result(),
            _click("Score this subject at every location"),
            {"action": "await_patient"},
        ),
        must_contain=("Recording level and patient level",),
        height=1800,
    ),
    ShotSpec(
        11,
        "Robustness analytics",
        "/robustness/",
        "How the results move under added noise, shortened recordings, a "
        "different corpus and a different auscultation location (T19-T22, "
        "T-S5). The leave-one-sub-collection-out result is shown beside the "
        "pooled one rather than in place of it.",
        height=1800,
    ),
    ShotSpec(
        12,
        "Feature importance/explainability",
        # The per-sample half of this page reads the last prediction out of
        # `sessionStorage`, so the capture scores a recording first and then
        # navigates. A capture that opened this route cold would photograph the
        # "no prediction yet" state, which is exactly what T120.7 forbids.
        "/predict/binary/",
        "Global feature importance and family-level contribution (T23), and "
        "the per-sample explanation of the prediction made in this browser "
        "tab: the contribution of each feature to that recording's own "
        "decision, computed by the API over the exact vector it scored.",
        steps=(
            _sample("binary-abnormal"),
            _click("2. Run the screening model"),
            _await_result(),
            {"action": "goto", "route": "/explainability/"},
        ),
        must_contain=("Explainability",),
        must_not_contain=(
            "Route scaffolded; content not built yet.",
            "No model is deployed for this task yet",
            "This chart could not be drawn.",
            "Nothing scored yet",
            "No prediction has been made in this browser tab yet",
        ),
        slug="explainability",
        height=1800,
    ),
    ShotSpec(
        13,
        "Exported report preview",
        "/reports/",
        "The report export panel after a recording report has really been "
        "generated and saved by the running service, with the objective-"
        "coverage table (T29) rendered below it -- the same content the "
        "downloadable DOCX carries. A DOCX cannot be previewed inside the "
        "browser, so what is shown is the export in its completed state plus "
        "the rendered content of the document it produced.",
        steps=(
            {"action": "select", "label": "Built-in sample", "value": "binary-abnormal"},
            _click("Generate recording report"),
            {"action": "await_saved"},
        ),
        must_contain=("Objective coverage",),
        height=1800,
    ),
)


def plan_payload() -> dict[str, Any]:
    """The capture plan as the Playwright spec consumes it."""
    return {
        "min_bytes": MIN_BYTES,
        "disclaimer": DISCLAIMER_MARK,
        "shots": [
            {**asdict(spec), "shot_id": spec.shot_id, "filename": spec.filename}
            for spec in CAPTURE_PLAN
        ],
    }


def _out_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if out_dir is not None:
        return Path(ensure_dir(out_dir))
    return Path(ensure_dir(load_config("paths").require("outputs.dashboard_screenshots")))


def write_plan(out_dir: str | Path | None = None) -> Path:
    """Write ``capture_plan.json`` next to where the PNGs will land."""
    target = _out_dir(out_dir) / PLAN_FILENAME
    target.write_text(
        json.dumps(plan_payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("capture plan: %d shots -> %s", len(CAPTURE_PLAN), target)
    return target


@dataclass
class Finalized:
    """What :func:`finalize` produced, and anything it could not."""

    captions: Path
    index: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def finalize(
    out_dir: str | Path | None = None, *, register: bool = True, strict: bool = True
) -> Finalized:
    """Caption, index and register the thirteen captures (T120.6).

    Refuses a capture that does not exist or that is too small to be a rendered
    page: a caption under a blank PNG is exactly the kind of unchecked artifact
    research rule 1 exists to prevent.
    """
    directory = _out_dir(out_dir)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for spec in CAPTURE_PLAN:
        path = directory / spec.filename
        size = path.stat().st_size if path.is_file() else 0
        if size < MIN_BYTES:
            missing.append(
                spec.shot_id
                + " "
                + spec.filename
                + (" is missing" if size == 0 else " is only " + str(size) + " bytes")
            )
            continue
        rows.append(
            {
                "screenshot_id": spec.shot_id,
                "number": spec.number,
                "title": spec.title,
                "route": spec.route,
                "filename": spec.filename,
                "bytes": size,
                "caption": spec.caption,
            }
        )

    if missing and strict:
        raise ScreenshotError(
            "the capture run did not produce "
            + str(len(missing))
            + " of "
            + str(len(CAPTURE_PLAN))
            + " screenshots: "
            + "; ".join(missing)
        )

    index_path = directory / INDEX_FILENAME
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "screenshot_id",
                "number",
                "title",
                "route",
                "filename",
                "bytes",
                "caption",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    captions_path = directory / CAPTIONS_FILENAME
    lines = [
        "# Dashboard screenshots (Phase 120)",
        "",
        "PV-MEPCG / PulseVision. Thirteen captures of the built dashboard, taken "
        "by `frontend/screenshots/dashboard.spec.ts` against the static export in "
        "`frontend/out/` served by the FastAPI process, with the live inference "
        "service answering `/predict`.",
        "",
        "Every capture is gated on `scripts/45_audit_displayed_values.py --gate` "
        "(T119.4): the audit must have passed on byte-for-byte the site that was "
        "photographed. The scope-and-safety notice is asserted present in all "
        "thirteen before the shutter fires.",
        "",
        "**Academic screening and decision-support prototype. Not a diagnostic "
        "device and not a substitute for clinical assessment.**",
        "",
    ]
    for row in rows:
        lines += [
            "## " + row["screenshot_id"] + " — " + row["title"],
            "",
            "![" + row["title"] + "](" + row["filename"] + ")",
            "",
            "**Figure "
            + str(row["number"])
            + ".** "
            + row["caption"]
            + " Route: `"
            + row["route"]
            + "`.",
            "",
        ]
    if missing:
        lines += ["## Not captured", ""] + ["- " + item for item in missing] + [""]
    captions_path.write_text("\n".join(lines), encoding="utf-8")

    if register:
        from src.utils.evidence import register_evidence

        for row in rows:
            register_evidence(
                row["screenshot_id"],
                directory / row["filename"],
                metric_or_asset="Dashboard screenshot " + str(row["number"]) + ": " + row["title"],
                objective="Dashboard",
                source_data="frontend/lib/generated (build-time codegen from outputs/)",
                command="python scripts/47_dashboard_screenshots.py",
            )
        register_evidence(
            "SS-INDEX",
            index_path,
            metric_or_asset="Dashboard screenshot index with captions",
            objective="Dashboard",
            source_data=str(directory / PLAN_FILENAME),
            command="python scripts/47_dashboard_screenshots.py",
        )

    log.info("finalized %d screenshots -> %s", len(rows), directory)
    return Finalized(captions=captions_path, index=index_path, rows=rows, missing=missing)
