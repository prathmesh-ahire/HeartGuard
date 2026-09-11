"""Phase 117's gate: are pages 10-12 traceable, and does the evidence browser resolve?

Same two directions as `tests/test_pages_1_3.py` and `tests/test_pages_4_6.py`.

**Source to screen.** Every cell of the robustness tables is recomputed from its
committed CSV through `tables.format_value` and asserted to be on the page. The
explainability numbers are recomputed from `outputs/04_models/explainability/`,
and every number on the limitations page from the file it names.

**Screen to source.** Everything metric-shaped in the rendered text must be a
string some generated payload carries, so a number typed into a page fails.

**T117.7's own clause.** The evidence browser must resolve at least one link per
page to a real CSV. "Resolve" is taken literally: the link's target is a file in
the built site, and its bytes digest to the value `evidence.json` recorded AND to
the committed file in `outputs/` today -- so a stale copy fails as well as a
missing one.

Skips when `frontend/out/` is absent. Runs in the frontend CI job.
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
    "/robustness/": "robustness/index.html",
    "/explainability/": "explainability/index.html",
    "/reports/": "reports/index.html",
    "/limitations/": "limitations/index.html",
}

ROBUSTNESS_TABLES = ("T16", "T20", "T21", "T22", "T23", "T27", "T-S5")

METRIC_LIKE = re.compile(r"\b\d+\.\d{3,}\b|\b\d+(?:\.\d+)?%")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")
_HREF = re.compile(r'href="(/evidence/[^"#?]+)"')

#: Task references printed in an eyebrow. Listed individually.
STRUCTURAL: frozenset[str] = frozenset({"117.3", "117.4"})

pytestmark = pytest.mark.skipif(
    not all((OUT / name).is_file() for name in PAGES.values()),
    reason="frontend/out/ is not built in this checkout; run npm run build",
)


def _raw(page: str) -> str:
    return (OUT / PAGES[page]).read_text(encoding="utf-8")


def _text(page: str) -> str:
    raw = re.sub(r"<script.*?</script>", " ", _raw(page), flags=re.S)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split())


def _generated(name: str) -> Any:
    return json.loads((GENERATED / name).read_text(encoding="utf-8"))


def _table(table_id: str) -> dict[str, Any]:
    tables = _generated("tables.json")
    assert table_id in tables, table_id + " was not exported"
    return dict(tables[table_id])


# ---------------------------------------------------------------------------
# T117.1 -- robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_id", ROBUSTNESS_TABLES)
def test_every_robustness_cell_is_its_source_csv_cell_and_is_on_the_page(table_id: str) -> None:
    import pandas as pd

    from src.reporting.tables import format_value

    payload = _table(table_id)
    frame = pd.read_csv(PROJECT_ROOT / payload["source_csv"])
    assert payload["n_rows"] == len(frame), table_id + " row count differs from its CSV"

    text = _text("/robustness/")
    missing: list[str] = []
    for column in payload["columns"]:
        for position, shown in enumerate(column["display"]):
            expected = format_value(frame[column["name"]].iloc[position], column["kind"])
            assert shown == expected, table_id + "." + column["name"] + " row " + str(position)
            if METRIC_LIKE.search(shown) and shown not in text:
                missing.append(column["name"] + "[" + str(position) + "]=" + shown)
    assert missing == [], table_id + " cells absent from /robustness/: " + ", ".join(missing[:10])


def test_the_robustness_page_frames_the_cross_dataset_result_as_transfer() -> None:
    text = _text("/robustness/")
    assert (
        "cross-dataset transfer from an adult cohort to a predominantly paediatric cohort" in text
    )
    assert "within-corpus cross-validation" in text


def test_every_robustness_figure_is_its_canonical_png() -> None:
    figures = _generated("figures.json")
    raw = _raw("/robustness/")
    for figure_id in ("G29", "G30", "G31", "G32", "G34", "G35"):
        png = figures[figure_id]["png"]
        assert png and ('src="' + png + '"') in raw, figure_id + " PNG is not on the page"
        assert (OUT / png.lstrip("/")).is_file(), figure_id + " PNG is not in the export"


# ---------------------------------------------------------------------------
# T117.2 -- explainability
# ---------------------------------------------------------------------------


def test_global_importance_is_recomputed_from_the_importance_summary() -> None:
    import pandas as pd

    from src.reporting.tables import format_value

    payload = _generated("explainability.json")
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))
    frame = pd.read_csv(OUTPUTS / "04_models" / "explainability" / "importance_summary.csv")
    text = _text("/explainability/")

    for group in payload["importance"]:
        rows = frame[
            (frame["task"] == group["task"])
            & (frame["model_id"] == group["model_id"])
            & (frame["kind"] == group["kind"])
        ].set_index("feature")
        assert len(group["rows"]) == min(20, len(rows))
        assert [row["rank"] for row in group["rows"]] == sorted(
            row["rank"] for row in group["rows"]
        )
        for row in group["rows"]:
            source = rows.loc[row["feature"]]
            assert row["importance_display"] == format_value(source["importance_mean"], "metric")
            assert row["importance_sd_display"] == format_value(source["importance_sd"], "metric")

    # The first group is the one the static HTML renders.
    for row in payload["importance"][0]["rows"]:
        assert row["feature"] in text
        assert row["importance_display"] in text


def test_family_shares_are_recomputed_from_the_family_importance() -> None:
    import pandas as pd

    from src.reporting.tables import format_value

    payload = _generated("explainability.json")
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))
    frame = pd.read_csv(OUTPUTS / "04_models" / "explainability" / "feature_family_importance.csv")
    text = _text("/explainability/")
    for group in payload["families"]:
        rows = frame[
            (frame["task"] == group["task"])
            & (frame["model_id"] == group["model_id"])
            & (frame["kind"] == group["kind"])
        ].set_index("family")
        for row in group["rows"]:
            expected = format_value(rows.loc[row["family"], "share_of_positive_mean"], "metric")
            assert row["share_display"] == expected
            assert row["share_display"] + " ± " + row["share_sd_display"] in text


def test_the_stored_explanation_is_the_committed_decomposition() -> None:
    from src.reporting.tables import format_value

    payload = _generated("explainability.json")["example"]
    if not payload["available"]:
        pytest.skip("no stored per-sample explanation")
    source = json.loads(
        (OUTPUTS / "04_models" / "explainability" / "per_sample_explanation.json").read_text(
            encoding="utf-8"
        )
    )
    text = _text("/explainability/")
    assert payload["record_uid"] == source["record_uid"]
    assert payload["probability_display"] == format_value(source["probability"], "metric")
    assert len(payload["rows"]) == len(source["top_contributions"])
    for shown, stored in zip(payload["rows"], source["top_contributions"], strict=True):
        assert shown["feature"] == stored["feature"]
        assert shown["contribution_display"] == format_value(stored["contribution"], "metric")
        assert shown["contribution_display"] in text


def test_the_live_explanation_says_so_when_nothing_has_been_scored() -> None:
    """The static HTML has no session: it must render the stated absence."""
    text = _text("/explainability/")
    assert "No prediction has been made in this browser tab yet" in text


# ---------------------------------------------------------------------------
# T117.5 -- limitations
# ---------------------------------------------------------------------------


def test_every_limitations_number_is_read_from_its_named_file() -> None:
    import pandas as pd

    from src.reporting.tables import format_value

    payload = _generated("limitations.json")
    text = _text("/limitations/")

    population = json.loads(
        (
            OUTPUTS / "08_circor_external_validation" / "EXP-D1" / "population_mismatch.json"
        ).read_text(encoding="utf-8")
    )
    block = payload["population"]
    assert block["train_age_median_display"] == format_value(
        population["train"]["age_median_years"], "mean_count"
    )
    assert block["test_n_patients_display"] == format_value(
        population["test"]["n_patients"], "count"
    )
    assert block["test_share_paediatric_display"] == (
        format_value(population["test"]["share_paediatric_of_recorded"] * 100.0, "percent") + "%"
    )
    for key in (
        "train_n_with_age_display",
        "train_age_median_display",
        "train_share_under_18_display",
        "test_n_patients_display",
        "test_n_paediatric_display",
        "test_share_paediatric_display",
    ):
        assert block[key] in text, key + " is not on /limitations/"

    t02 = pd.read_csv(
        OUTPUTS / "01_dataset_audit" / "T02_class_distribution_and_imbalance_ratio.csv"
    )
    for track in payload["pascal"]["tracks"]:
        rows = t02[t02["task"] == track["task"]]
        assert track["n_records_display"] == format_value(int(rows["n_records"].sum()), "count")
        assert track["smallest_n_display"] == format_value(int(rows["n_records"].min()), "count")
        assert track["n_records_display"] in text

    circor = t02[t02["task"] == "circor_outcome"]
    assert payload["circor"]["n_patients_display"] == format_value(
        int(circor["n_subjects"].sum()), "count"
    )
    assert payload["circor"]["n_patients_display"] in text

    ts5 = pd.read_csv(OUTPUTS / "09_ablation" / "T-S5_leave_one_source_out_generalization.csv")
    m1 = ts5[ts5["model_id"] == "M1"].iloc[0]
    holdout = payload["source_holdout"]
    assert holdout["auc_pooled_display"] == format_value(m1["roc_auc_pooled"], "metric")
    assert holdout["auc_holdout_display"] == format_value(m1["roc_auc_holdout"], "metric")
    assert holdout["auc_holdout_display"] in text


def test_the_limitations_page_states_all_three_caveats_t117_5_names() -> None:
    text = _text("/limitations/")
    assert "adult to paediatric" in text
    assert "The PASCAL tracks are small" in text
    assert "CirCor is the public subset only" in text
    assert "is NOT evidence that PV-MEPCG fails to generalize" in text


# ---------------------------------------------------------------------------
# T117.3 / T117.4 -- reports and the evidence browser
# ---------------------------------------------------------------------------


def test_the_reports_page_offers_all_three_reports() -> None:
    text = _text("/reports/")
    assert "Generate recording report" in text
    assert "Generate experiment report" in text
    assert "Download objective-coverage report" in text
    # T29 is rendered on the page it is downloadable from.
    for value in _table("T29")["columns"][0]["display"]:
        assert value in text


def test_every_evidence_entry_is_listed_in_the_browser() -> None:
    evidence = _generated("evidence.json")
    text = _text("/reports/")
    raw = _raw("/reports/")
    assert len(evidence) > 500, "the evidence index is implausibly small"
    for entry in evidence:
        assert entry["key"] in text, entry["key"] + " is not in the evidence browser"
        if entry["url"]:
            assert 'href="' + entry["url"] + '"' in raw


def _resolves(href: str, evidence: list[dict[str, Any]]) -> None:
    from src.reporting.tables import content_digest

    relative = href[len("/evidence/") :]
    served = OUT / "evidence" / relative
    assert served.is_file(), href + " does not resolve to a file in the built site"
    assert served.suffix.lower() in (".csv", ".json"), href + " is not a CSV or JSON"
    recorded = {
        entry["generated_from_sha256"] for entry in evidence if entry["generated_from"] == relative
    }
    digest = content_digest(served)[0]
    if recorded:
        assert digest in recorded, href + " served bytes differ from the digest in evidence.json"
    committed = PROJECT_ROOT / relative
    assert committed.is_file(), relative + " is not in outputs/"
    assert content_digest(committed)[0] == digest, href + " is stale against outputs/"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_at_least_one_evidence_link_per_page_resolves_to_a_real_csv(page: str) -> None:
    """T117.7: the clause, taken literally -- and applied to every link, not one."""
    evidence = _generated("evidence.json")
    links = sorted(set(_HREF.findall(_raw(page))))
    csv_links = [href for href in links if href.endswith(".csv")]
    assert csv_links, page + " links no CSV through the evidence browser"
    for href in links:
        _resolves(href, evidence)


# ---------------------------------------------------------------------------
# screen -> source
# ---------------------------------------------------------------------------


def _collect_strings(node: Any, into: set[str]) -> None:
    if isinstance(node, str):
        into.add(node)
        into.update(METRIC_LIKE.findall(node))
    elif isinstance(node, dict):
        for value in node.values():
            _collect_strings(value, into)
    elif isinstance(node, list):
        for value in node:
            _collect_strings(value, into)


def _allowed() -> set[str]:
    allowed: set[str] = set(STRUCTURAL)
    for name in (
        "tables.json",
        "figures.json",
        "explainability.json",
        "limitations.json",
        "reports.json",
        "prediction.json",
    ):
        _collect_strings(_generated(name), allowed)
    return allowed


@pytest.mark.parametrize("page", sorted(PAGES))
def test_nothing_that_looks_like_a_metric_is_unaccounted_for(page: str) -> None:
    allowed = _allowed()
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


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_page_uses_screening_language_and_the_framework_name(page: str) -> None:
    text = _text(page)
    assert "PV-MEPCG" in text or "PulseVision" in text
    assert "HeartGuard" not in text
    lowered = text.lower()
    assert "does not diagnose" in lowered
    for forbidden in ("diagnosis of", "treatment plan", "replaces a doctor"):
        assert forbidden not in lowered, "clinical language on " + page + ": " + forbidden
