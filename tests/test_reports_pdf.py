"""Phase 134's gate (T134.7): the Reports store, the PDF generators, and the
new History/Reports API endpoints they sit behind.

T134.7 asks for a PDF and a CSV generated from known records, opened and
checked against History rather than reasoned about. Every check below opens
the actual bytes -- `pypdf` for the PDF, `csv.reader` for the CSV -- the same
round-trip discipline `test_reports_api.py` uses for the DOCX reports.

Every store here lives in `tmp_path`; `conftest.py` also redirects
`cache.reports_db` / `cache.reports_dir` for the whole session, so no test
here can write into the operator's real `cache/reports/`.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from src.api import report_store as rs
from src.api.history_store import HistoryStore
from src.api.main import create_app
from src.inference.predictor import DISCLAIMER
from src.reporting.pdf_report import (
    WAVEFORM_UNAVAILABLE,
    render_model_summary_pdf,
    render_recording_pdf,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED = PROJECT_ROOT / "frontend" / "lib" / "generated"


def _pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _row(store: HistoryStore, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "file_name": "a0001.wav",
        "task": "binary",
        "result": "normal",
        "confidence": 0.91,
        "probabilities": {"normal": 0.91, "abnormal": 0.09},
    }
    fields.update(overrides)
    return store.create(**fields)


# ---------------------------------------------------------------------------
# ReportStore (T134.4)
# ---------------------------------------------------------------------------


def test_the_session_never_writes_into_the_real_reports_store() -> None:
    assert "pvmepcg-manifest-" in str(rs.default_reports_db_path())


def test_save_then_get_returns_the_same_bytes(tmp_path: Path) -> None:
    store = rs.ReportStore(tmp_path / "r.json", tmp_path / "files")
    row = store.save(
        kind="recording",
        title="a.wav",
        filename="a.pdf",
        content_type="application/pdf",
        content=b"%PDF-fake",
    )
    found = store.get(row["id"])
    assert found is not None
    content, meta = found
    assert content == b"%PDF-fake"
    assert meta["filename"] == "a.pdf"


def test_list_recent_is_newest_first_and_prunes_past_the_cap(tmp_path: Path) -> None:
    store = rs.ReportStore(tmp_path / "r.json", tmp_path / "files")
    for i in range(rs.MAX_ENTRIES + 5):
        store.save(
            kind="recording",
            title=str(i),
            filename=str(i) + ".pdf",
            content_type="application/pdf",
            content=str(i).encode(),
        )
    rows = store.list_recent(limit=rs.MAX_ENTRIES)
    assert len(rows) == rs.MAX_ENTRIES
    assert rows[0]["title"] == str(rs.MAX_ENTRIES + 4)  # newest first
    files = list((tmp_path / "files").iterdir())
    assert len(files) == rs.MAX_ENTRIES  # pruned rows' files are deleted too


def test_get_of_an_unknown_id_is_none(tmp_path: Path) -> None:
    store = rs.ReportStore(tmp_path / "r.json", tmp_path / "files")
    assert store.get("no-such-id") is None


# ---------------------------------------------------------------------------
# render_recording_pdf (T134.1 / T134.6)
# ---------------------------------------------------------------------------


def test_a_recording_pdf_carries_every_stored_value_and_the_disclaimer(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "h.json")
    created = _row(
        store,
        file_name="clinic-upload.wav",
        notes="Repeat at the next visit",
        tags=["clinic-a", "repeat"],
        source_kind="upload",
    )
    record = store.get(created["id"])
    assert record is not None

    target = tmp_path / "out.pdf"
    render_recording_pdf(record, target, waveform_png=None)
    text = _pdf_text(target.read_bytes())

    # T134.6: every number is the SAME string History's own `display` carries.
    assert record["file_name"] in text
    assert record["display"]["result"] in text
    assert record["display"]["confidence"] in text
    for value in record["display"]["probabilities"].values():
        assert value in text
    assert "Repeat at the next visit" in text
    assert "clinic-a" in text
    assert "repeat" in text
    assert DISCLAIMER in text
    # No stored audio for an upload -- the note explains why, not a blank section.
    assert WAVEFORM_UNAVAILABLE in text


def test_a_recording_pdf_with_no_notes_or_tags_says_so_plainly(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "h.json")
    record = store.get(_row(store)["id"])
    assert record is not None
    target = tmp_path / "out.pdf"
    render_recording_pdf(record, target)
    text = _pdf_text(target.read_bytes())
    assert "No notes recorded." in text
    assert "No tags." in text


def test_a_recording_pdf_embeds_a_given_waveform_image(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png = tmp_path / "wave.png"
    fig.savefig(png)
    plt.close(fig)

    store = HistoryStore(tmp_path / "h.json")
    record = store.get(_row(store, source_kind="sample")["id"])
    assert record is not None
    target = tmp_path / "out.pdf"
    render_recording_pdf(record, target, waveform_png=png)
    assert WAVEFORM_UNAVAILABLE not in _pdf_text(target.read_bytes())
    assert target.stat().st_size > 3000  # an embedded PNG is bigger than the note alone


# ---------------------------------------------------------------------------
# render_model_summary_pdf (T134.5 / T134.6)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GENERATED.is_dir(), reason="frontend/lib/generated/ has not been built")
def test_the_model_summary_pdf_matches_tables_json_exactly() -> None:
    import json

    tables = json.loads((GENERATED / "tables.json").read_text(encoding="utf-8"))
    t08 = tables["T08"]
    columns = {c["name"]: c for c in t08["columns"]}
    runs = columns["run"]["display"]
    models = columns["model_id"]["display"]
    index = next(
        i
        for i, (r, m) in enumerate(zip(runs, models, strict=False))
        if r == "EXP-A2" and m == "M1"
    )
    expected_sensitivity = columns["sensitivity_mean"]["display"][index]
    expected_specificity = columns["specificity_mean"]["display"][index]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "summary.pdf"
        render_model_summary_pdf(target, generated_dir=GENERATED)
        text = _pdf_text(target.read_bytes())

    assert expected_sensitivity in text
    assert expected_specificity in text
    assert DISCLAIMER in text
    prediction = json.loads((GENERATED / "prediction.json").read_text(encoding="utf-8"))
    for task in prediction["tasks"]:
        assert task["title"] in text


# ---------------------------------------------------------------------------
# API: per-recording PDF, model summary, recent reports, filtered CSV export
# ---------------------------------------------------------------------------


@pytest.fixture()
def api(tmp_path: Path) -> Any:
    app = create_app(
        static_root=PROJECT_ROOT / "does-not-exist",
        preload=False,
        history_path=tmp_path / "h.json",
        reports_db_path=tmp_path / "r.json",
        reports_dir_path=tmp_path / "rfiles",
    )
    with TestClient(app) as client:
        yield client


def test_t134_1_the_recording_pdf_endpoint_matches_the_history_record(api: Any) -> None:
    created = api.post(
        "/api/history",
        json={
            "file_name": "upload.wav",
            "task": "binary",
            "result": "abnormal",
            "confidence": 0.77,
            "probabilities": {"normal": 0.23, "abnormal": 0.77},
        },
    ).json()

    response = api.get("/api/history/" + created["id"] + "/report.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    text = _pdf_text(response.content)
    assert "abnormal" in text
    assert "0.770" in text or "77" in text

    # T134.4: it is now in the recent-reports list, and re-downloads identically.
    recent = api.get("/api/reports").json()
    assert recent["items"], "the generated report was not logged"
    report_id = recent["items"][0]["id"]
    again = api.get("/api/reports/" + report_id)
    assert again.content == response.content


def test_t134_1_an_unknown_record_is_404(api: Any) -> None:
    assert api.get("/api/history/does-not-exist/report.pdf").status_code == 404


def test_t134_4_an_unknown_report_id_is_404(api: Any) -> None:
    assert api.get("/api/reports/does-not-exist").status_code == 404


@pytest.mark.skipif(not GENERATED.is_dir(), reason="frontend/lib/generated/ has not been built")
def test_t134_5_the_model_summary_endpoint_serves_a_pdf(api: Any) -> None:
    response = api.get("/api/reports/model-summary.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert DISCLAIMER in _pdf_text(response.content)


def test_t134_2_the_filtered_export_is_a_csv_of_every_matching_row(api: Any) -> None:
    for i in range(3):
        api.post(
            "/api/history",
            json={
                "file_name": "r" + str(i) + ".wav",
                "task": "binary",
                "result": "normal",
                "confidence": 0.9,
                "probabilities": {"normal": 0.9, "abnormal": 0.1},
            },
        )
    api.post(
        "/api/history",
        json={
            "file_name": "murmur.wav",
            "task": "murmur",
            "result": "Present",
            "confidence": 0.6,
            "probabilities": {"Present": 0.6, "Absent": 0.3, "Unknown": 0.1},
        },
    )

    response = api.get("/api/history/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(response.text)))
    assert len(rows) == 1 + 4  # header + 4 rows, every task, none merged

    # T134.2: the task filter narrows it, exactly like the History page's own filter.
    only_binary = api.get("/api/history/export?task=binary")
    rows2 = list(csv.reader(io.StringIO(only_binary.text)))
    assert len(rows2) == 1 + 3
    assert all(row[3] == "Binary screening (PhysioNet 2016)" for row in rows2[1:])


def test_t134_2_no_match_is_a_400_not_an_empty_csv(api: Any) -> None:
    response = api.get("/api/history/export?task=binary")
    assert response.status_code == 400
