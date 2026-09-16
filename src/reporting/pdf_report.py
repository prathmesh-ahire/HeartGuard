"""Product Reports page: per-recording and model-summary PDFs (Phase 134).

## Why PDF here and DOCX in `sample_report.py`

`sample_report.py` documents a deliberate 2026 decision to use DOCX for the
Part VII/IX thesis deliverables. This module does not reopen that decision --
it answers a different question. The Reports page (Part XII) is downloaded by
an operator, not a thesis committee, and needs a file every viewer can open
with no application installed; DOCX does not guarantee that, PDF does. Both
`reportlab` (writer) and `pypdf` (reader, used by the T134.7 test) are
pure-Python wheels with no Windows compiler-toolchain requirement, the same bar
`tinydb` was checked against in Phase 129.

## What a per-recording report is built from

`render_recording_pdf` takes the dict `HistoryStore.present()` returns --
never a fresh prediction. **The audio is never stored** (see
`history_store.py`), so a report built from a live re-score could show a
different number than what History actually saved, which is exactly the kind
of two-numbers-for-one-fact bug research rule 1 exists to prevent. Every number
in the report is therefore `record["display"][...]`, the same
`tables.format_value` output the History and Insights pages render -- T134.6
is met by construction, not by a second rounding pass.

The **waveform is the one optional part**: it can only be redrawn for a
built-in sample (its audio is on disk under `dataset/`), never for an upload or
a microphone recording. The caller passes `waveform_png` only when it already
resolved the record to a sample id; this module never guesses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inference.predictor import DISCLAIMER
from src.utils.io import ensure_dir
from src.utils.logging_setup import get_logger

__all__ = [
    "WAVEFORM_UNAVAILABLE",
    "render_model_summary_pdf",
    "render_recording_pdf",
]

log = get_logger("reporting.pdf_report")

WAVEFORM_UNAVAILABLE = (
    "Waveform not available. The original audio is not kept once a recording "
    "is analysed, except for the built-in sample library."
)

_TITLE = "PV-MEPCG / PulseVision"


def _styles() -> Any:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    base.add(ParagraphStyle("PVTitle", parent=base["Title"], fontSize=16, spaceAfter=4))
    base.add(ParagraphStyle("PVH2", parent=base["Heading2"], spaceBefore=14, spaceAfter=6))
    base.add(
        ParagraphStyle(
            "PVDisclaimer",
            parent=base["Normal"],
            fontSize=8.5,
            textColor=colors.HexColor("#7a1f1f"),
            borderPadding=6,
            spaceAfter=10,
        )
    )
    base.add(ParagraphStyle("PVMuted", parent=base["Normal"], textColor=colors.grey, fontSize=9))
    return base


def _kv_table(rows: list[tuple[str, str]]) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=[140, 320])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def render_recording_pdf(
    record: dict[str, Any],
    out_path: str | Path,
    *,
    waveform_png: str | Path | None = None,
) -> Path:
    """One History record to one PDF. Every value is `record["display"][...]`.

    `record` is exactly what `HistoryStore.present()` returns. `waveform_png`
    is an already-rendered PNG path (see module docstring); pass `None` when
    the record's audio was never kept.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer

    styles = _styles()
    target = Path(out_path)
    ensure_dir(target.parent)

    display = record.get("display") or {}
    story: list[Any] = [
        Paragraph(_TITLE + " -- Recording report", styles["PVTitle"]),
        Paragraph(DISCLAIMER, styles["PVDisclaimer"]),
        Paragraph(str(record.get("file_name") or "n/a"), styles["Heading3"]),
        Paragraph(str(record.get("task_title") or record.get("task") or "n/a"), styles["PVMuted"]),
        Spacer(1, 8),
        _kv_table(
            [
                ("Date analysed", str(record.get("created_at") or "n/a")),
                ("Result", str(display.get("result") or "n/a")),
                ("Confidence", str(display.get("confidence") or "n/a")),
                (
                    "Low confidence",
                    "yes" if record.get("low_confidence") else "no",
                ),
                ("Batch", str(record.get("batch_id") or "single analysis")),
            ]
        ),
        Paragraph("Class probabilities", styles["PVH2"]),
        _kv_table(
            [(name, value) for name, value in (display.get("probabilities") or {}).items()]
            or [("n/a", "n/a")]
        ),
        Paragraph("Notes", styles["PVH2"]),
        Paragraph(str(record.get("notes") or "No notes recorded.").replace("\n", "<br/>")),
        Paragraph("Tags", styles["PVH2"]),
        Paragraph(", ".join(record.get("tags") or []) or "No tags."),
        Paragraph("Waveform", styles["PVH2"]),
    ]
    if waveform_png is not None and Path(waveform_png).is_file():
        story.append(Image(str(waveform_png), width=420, height=170))
    else:
        story.append(Paragraph(WAVEFORM_UNAVAILABLE, styles["PVMuted"]))

    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        title=_TITLE + " recording report",
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    doc.build(story)
    log.info("wrote recording report -> %s", target)
    return target


def render_model_summary_pdf(out_path: str | Path, *, generated_dir: str | Path) -> Path:
    """The plain-language model summary (T134.5): every task, and the shipped
    binary model's headline cross-validated figures.

    Reads `frontend/lib/generated/prediction.json` (task titles and
    descriptions -- the same text the Insights reliability panel shows) and
    `tables.json`'s T08 (the PhysioNet model comparison, T08's own EXP-A2/M1
    row, which is the shipped final binary model per the 2026-08-30 decision
    in `Docs/note.md`). Both files are already Python-formatted by
    `scripts/17_export_frontend_data.py`; nothing here rounds a number.
    """
    import json

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = _styles()
    target = Path(out_path)
    ensure_dir(target.parent)
    base = Path(generated_dir)

    prediction = json.loads((base / "prediction.json").read_text(encoding="utf-8"))
    tables = json.loads((base / "tables.json").read_text(encoding="utf-8"))

    story: list[Any] = [
        Paragraph(_TITLE + " -- Model summary", styles["PVTitle"]),
        Paragraph(DISCLAIMER, styles["PVDisclaimer"]),
        Paragraph(
            "What each check does",
            styles["PVH2"],
        ),
    ]
    for task in prediction.get("tasks", []):
        story.append(Paragraph("<b>" + str(task.get("title", "")) + "</b>", styles["Normal"]))
        story.append(Paragraph(str(task.get("description", "")), styles["PVMuted"]))
        story.append(Spacer(1, 4))

    headline = _t08_row(tables.get("T08") or {}, run="EXP-A2", model_id="M1")
    story.append(Paragraph("Binary screening: shipped model", styles["PVH2"]))
    if headline:
        story.append(
            Paragraph(
                "The deployed binary model (M1, regularised logistic regression), "
                "measured over the same 25 cross-validation folds used throughout "
                "this project:",
                styles["Normal"],
            )
        )
        story.append(
            _kv_table(
                [
                    ("Sensitivity", headline.get("sensitivity_mean", "n/a")),
                    ("Specificity", headline.get("specificity_mean", "n/a")),
                    ("Balanced accuracy", headline.get("balanced_accuracy_mean", "n/a")),
                    ("ROC AUC", headline.get("roc_auc_mean", "n/a")),
                ]
            )
        )
    else:
        story.append(
            Paragraph("The headline comparison table is not available.", styles["PVMuted"])
        )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "These figures are cross-validated within the PhysioNet 2016 corpus "
            "only; holding out an entire recording source at a time drops "
            "performance sharply (EXP-F3), so none of this is a claim of "
            "generalisation or deployment-readiness. " + DISCLAIMER,
            styles["PVMuted"],
        )
    )

    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        title=_TITLE + " model summary",
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    doc.build(story)
    log.info("wrote model summary report -> %s", target)
    return target


def _t08_row(table: dict[str, Any], *, run: str, model_id: str) -> dict[str, str] | None:
    """One row of a column-oriented `tables.json` entry, by `run` and `model_id`.

    `tables.json` stores each table as parallel per-column `display` arrays
    (see `scripts/17_export_frontend_data.py`); this walks them by index
    rather than assuming a row-oriented shape.
    """
    columns = {col["name"]: col for col in table.get("columns") or []}
    runs = (columns.get("run") or {}).get("display") or []
    models = (columns.get("model_id") or {}).get("display") or []
    index = next(
        (
            i
            for i, (r, m) in enumerate(zip(runs, models, strict=False))
            if r == run and m == model_id
        ),
        None,
    )
    if index is None:
        return None
    return {
        name: (col.get("display") or [])[index]
        for name, col in columns.items()
        if index < len(col.get("display") or [])
    }
