"""Phase 114's gate: is every number on pages 1-3 traceable to a source CSV?

T114.6 and T114.7 both ask for the same thing, from two directions, and this
file does both because either alone is a hole.

**Source to screen.** Every value the page states is recomputed here from the
CSV in `outputs/` — through `tables.format_value`, the project's single rounding
authority — and asserted to appear in the exported HTML. If a page ever shows a
different number from the one its source holds, the string it should have shown
is missing and this fails.

**Screen to source.** The exported HTML is then scanned for anything that
*looks* like a metric: a decimal with three or more places, or a percentage.
Every one of those must be in the allowed set built from the sources. This is
the direction that catches a fabricated number — a value typed into a page by
hand is, by construction, not in the set.

Neither half is a substitute for the other. The first misses an invented number
sitting beside the correct ones; the second misses a correct-looking number that
is off by a rounding rule.

**Why the HTML and not the React tree.** The static export is what a reader
actually receives. A test against the component would prove the component works
and say nothing about whether the page was rebuilt after the exporter changed.

The whole file skips when `frontend/out/` is absent: it is a build artifact, and
CI builds it in the frontend job rather than the Python one.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "frontend" / "out"
GENERATED = PROJECT_ROOT / "frontend" / "lib" / "generated"
OUTPUTS = PROJECT_ROOT / "outputs"

PAGES = {
    "/": "index.html",
    "/dataset/": "dataset/index.html",
    "/preprocessing/": "preprocessing/index.html",
}

#: Anything matching this in the page text has to be justified by a source.
#: Three decimal places is the metric rounding rule, so it is the signature of a
#: reported value; a percentage is the other form a metric is stated in.
METRIC_LIKE = re.compile(r"\b\d+\.\d{3,}\b|\b\d+(?:\.\d+)?%")

#: Numbers that are structural rather than measured: task and phase references
#: printed in an eyebrow, sample rates named in prose, the four-digit years of
#: the corpora. Each is listed individually rather than covered by a pattern,
#: because a pattern wide enough to cover them is wide enough to hide a metric.
STRUCTURAL: frozenset[str] = frozenset(
    {
        "114.4",  # eyebrow: the task this section implements
        "114.5",
    }
)


def _pages_built() -> bool:
    return all((OUT / name).is_file() for name in PAGES.values())


pytestmark = pytest.mark.skipif(
    not _pages_built(),
    reason="frontend/out/ is not built in this checkout; run npm run build",
)


def _text(page: str) -> str:
    """The rendered text of a page: no scripts, no styles, no markup."""
    raw = (OUT / PAGES[page]).read_text(encoding="utf-8")
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


#: An ISO-8601 instant. The footer prints the export timestamp, whose seconds
#: field (`…T03:23:40.450045+00:00`) is a decimal with six places and therefore
#: metric-shaped. It is removed before the scan rather than allow-listed as a
#: literal, because the literal changes on every export and pinning it would
#: mean the gate had to be edited every time the site was rebuilt. The
#: timestamp itself is checked against the manifest separately.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")


def _generated(name: str) -> Any:
    return json.loads((GENERATED / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# source -> screen
# ---------------------------------------------------------------------------


def test_the_home_tiles_are_the_audited_corpus_counts() -> None:
    """Each tile is recomputed from metadata_master.csv, not read from the export."""
    import pandas as pd

    from src.reporting.tables import format_value

    frame = pd.read_csv(OUTPUTS / "01_dataset_audit" / "metadata_master.csv")
    text = _text("/")

    for source, part in frame.groupby("dataset_source", sort=True):
        modelled = int(part["use_in_supervised"].astype(bool).sum())
        assert format_value(modelled, "count") in text, (
            "the home page does not show " + str(source) + "'s modelled count"
        )
        assert format_value(len(part), "count") in text, (
            "the home page does not show " + str(source) + "'s file count"
        )


def test_the_home_page_quotes_all_six_objectives_verbatim() -> None:
    """Not a number, but the same rule: the source text, unchanged.

    Compared against `src/reporting/objectives.py`, which carries the sha256 of
    each wording. A page that paraphrased even one word would fail here, which
    is what T125.4 asks for.
    """
    from src.reporting.objectives import OBJECTIVES, wording_digest

    text = " ".join(_text("/").split())
    for objective in OBJECTIVES:
        assert objective.wording in text, (
            "objective " + str(objective.number) + " is not quoted verbatim on the home page"
        )
        assert wording_digest(objective.wording)[:16] in text


@pytest.mark.parametrize(
    ("page", "table_ids"),
    [("/dataset/", ("T01", "T02", "T03")), ("/preprocessing/", ("T04",))],
)
def test_every_table_cell_on_the_page_equals_its_source_csv_cell(
    page: str, table_ids: tuple[str, ...]
) -> None:
    """Recompute each cell from the CSV the table's own meta.json names."""
    import pandas as pd

    from src.reporting.tables import format_value, infer_kind

    tables = _generated("tables.json")
    text = _text(page)
    checked = 0

    for table_id in table_ids:
        payload = tables[table_id]
        source = _table_source(table_id)
        frame = pd.read_csv(source)
        for column in payload["columns"]:
            name = column["name"]
            if name not in frame.columns:
                continue
            kind = column.get("kind") or infer_kind(name, frame[name])
            for position, shown in enumerate(column["display"]):
                expected = format_value(frame[name].iloc[position], kind)
                assert shown == expected, (
                    table_id
                    + "."
                    + name
                    + " row "
                    + str(position)
                    + ": export shows "
                    + repr(shown)
                    + ", source gives "
                    + repr(expected)
                )
                assert shown in text, (
                    table_id
                    + "."
                    + name
                    + " row "
                    + str(position)
                    + " ("
                    + repr(shown)
                    + ") is not on "
                    + page
                )
                checked += 1

    assert checked > 60, "too few cells checked to mean anything: " + str(checked)


def _table_source(table_id: str) -> Path:
    """The CSV a generated table was built from, per its committed meta.json."""
    for meta in OUTPUTS.rglob(table_id + "_*.meta.json"):
        return meta.with_name(meta.name.replace(".meta.json", ".csv"))
    raise AssertionError("no meta.json found for " + table_id)


def test_the_quality_indicators_are_the_recorded_measurements() -> None:
    """Preprocessing page: each indicator against signal_quality_flags.csv."""
    import pandas as pd

    from src.reporting.tables import format_value

    examples = _generated("preprocessing_examples.json")
    if not examples["available"]:
        pytest.skip("no preprocessing example was exported: " + str(examples["reason"]))

    frame = pd.read_csv(OUTPUTS / "02_preprocessing" / "signal_quality_flags.csv")
    checked = 0
    for record in examples["records"]:
        row = frame[frame["record_uid"] == record["record_uid"]]
        assert not row.empty, record["record_uid"] + " is not in signal_quality_flags.csv"
        for item in record["quality"]:
            kind = (
                "count"
                if item["name"] in ("fs", "original_fs")
                else ("seconds" if item["name"] == "duration_sec" else "metric")
            )
            expected = format_value(row[item["name"]].iloc[0], kind)
            assert item["display"] == expected, item["name"] + " on " + record["record_uid"]
            checked += 1
    assert checked >= 40, "too few indicators checked: " + str(checked)


def test_no_quality_indicator_is_printed_at_raw_float_precision() -> None:
    """A measurement shown to fifteen significant figures claims an accuracy it has not.

    This was a real defect: `infer_kind` classifies `snr_proxy_db` as text,
    which renders the value unchanged, so the page showed -19.557766... until
    the kinds were declared explicitly.
    """
    examples = _generated("preprocessing_examples.json")
    if not examples["available"]:
        pytest.skip("no preprocessing example was exported")
    for record in examples["records"]:
        for item in record["quality"]:
            if "." not in item["display"]:
                continue
            decimals = len(item["display"].split(".")[1])
            assert decimals <= 3, (
                item["name"]
                + " is shown as "
                + item["display"]
                + ", which is "
                + str(decimals)
                + " decimal places"
            )


# ---------------------------------------------------------------------------
# screen -> source
# ---------------------------------------------------------------------------


def _allowed_strings() -> set[str]:
    """Every value the generated payloads say a page may state."""
    allowed: set[str] = set(STRUCTURAL)

    def admit(value: str) -> None:
        """Admit a source string and every metric-shaped token inside it.

        A T04 cell reads "128 ms, 50% overlap": the cell is sourced, but the
        scan sees `50%` on its own. Tokenising the allowed strings the same way
        the page is tokenised keeps the comparison exact rather than falling
        back to a substring test, which would admit `0.848` because some
        unrelated cell happens to contain `10.8481`.
        """
        allowed.add(value)
        allowed.update(METRIC_LIKE.findall(value))

    for name in ("tables.json", "figures.json"):
        for payload in _generated(name).values():
            for column in payload.get("columns", []):
                for value in column.get("display", []):
                    admit(str(value))

    summary = _generated("dataset_summary.json")
    for row in summary["summary"]:
        for key, value in row.items():
            if key.endswith("_display"):
                admit(str(value))

    examples = _generated("preprocessing_examples.json")
    for record in examples.get("records", []):
        for item in record["quality"]:
            admit(str(item["display"]))
        admit(str(record["fs"]))
        admit(str(record["native_fs"]))

    for objective in _generated("objectives.json")["objectives"]:
        admit(objective["wording_sha256"][:16])

    return allowed


@pytest.mark.parametrize("page", list(PAGES))
def test_nothing_that_looks_like_a_metric_is_unaccounted_for(page: str) -> None:
    """Every metric-shaped token on the page must come from a generated payload.

    This is the direction that catches an invented number. A value typed into a
    `.tsx` file is not in `generated/`, so it is not in the allowed set, so it
    fails here — which is the same guarantee
    `scripts/16_check_no_hardcoded_metrics.py` gives at the source level, taken
    at the rendered-output level where a template expression could still have
    produced one.
    """
    allowed = _allowed_strings()
    text = _TIMESTAMP.sub(" ", _text(page))

    unexplained = sorted(
        {
            token
            for token in METRIC_LIKE.findall(text)
            if token not in allowed and token.rstrip("%") not in allowed
        }
    )
    assert unexplained == [], (
        page + " states values with no source in generated/: " + ", ".join(unexplained)
    )


@pytest.mark.parametrize("page", list(PAGES))
def test_the_page_carries_the_screening_disclaimer_and_the_framework_name(page: str) -> None:
    text = _text(page)
    assert "PV-MEPCG" in text or "PulseVision" in text
    assert "HeartGuard" not in text, (
        "HeartGuard is the repository name; deliverables say PV-MEPCG / PulseVision"
    )
    lowered = text.lower()
    assert "not a diagnostic device" in lowered or "does not diagnose" in lowered
    for forbidden in ("diagnosis of", "treatment plan", "replaces a doctor"):
        assert forbidden not in lowered, "clinical language on " + page + ": " + forbidden


def test_the_dataset_page_states_both_populations_rather_than_one() -> None:
    """The scope trap, checked on the page rather than trusted to a caption.

    note.md records a table that put 21.98 corpus hours beside 3,240 supervised
    records — both numbers correct, the row impossible. The page must show the
    file count and the modelled count together, and say they differ.
    """
    text = " ".join(_text("/dataset/").split())
    summary = _generated("dataset_summary.json")
    for row in summary["summary"]:
        assert row["n_files_display"] in text
        assert row["n_modelled_display"] in text
    assert "modelled" in text.lower()
    assert summary["scope_note"] in text


def test_the_footer_timestamp_is_the_one_the_exporter_recorded() -> None:
    """The instant removed from the metric scan is checked here instead."""
    manifest = _generated("manifest.json")
    for page in PAGES:
        text = _text(page)
        assert manifest["exported_utc"] in text, (
            page + " does not carry the export timestamp from manifest.json"
        )
        assert manifest["git_commit"][:12] in text
