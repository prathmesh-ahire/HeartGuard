"""The T86.7 gate: T01-T07 exist in every format and name their sources (Phase 86).

The gate has two halves, and the second is the one that matters.

**Existence.** All seven tables, each in CSV, Markdown, DOCX and LaTeX, plus the
``.meta.json``, non-empty, in the directory the table belongs to.

**Traceability.** Every table names the source CSV it was built from, that file
still exists, and its **sha256 still matches the digest recorded when the table
was written**. That last check is what makes the gate worth having: audit CSVs
get regenerated, and a table built from a superseded one is indistinguishable
from a current one by looking at it. If this test fails on a digest, the tables
are stale and `python scripts/18_setup_tables.py` is the fix -- it is not a
reason to relax the assertion.

Values are cross-checked back against the source rather than against literals
typed here, because a literal in a test is a hand-typed number under rule 1 just
as much as a literal in a table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.reporting import tables as tb
from src.reporting.setup_tables import (
    SETUP_TABLE_IDS,
    SUPERVISED_SCOPE,
    audit_dir,
    destination_for,
    outputs_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SUFFIXES = (".csv", ".md", ".docx", ".tex", ".meta.json")

#: The stem each table is written under, derived from its title exactly as
#: TableSpec.slug() does. Kept here so the gate fails on a renamed file rather
#: than silently checking nothing.
STEMS = {
    "T01": "T01_dataset_inventory",
    "T02": "T02_class_distribution_and_imbalance_ratio",
    "T03": "T03_recording_duration_and_sampling_summary",
    "T04": "T04_preprocessing_configuration",
    "T05": "T05_feature_inventory_and_counts",
    "T06": "T06_model_hyperparameter_configuration",
    "T07": "T07_search_space_and_best_parameters",
}


def _paths(table_id: str) -> dict[str, Path]:
    directory = destination_for(table_id)
    stem = STEMS[table_id]
    return {suffix: directory / (stem + suffix) for suffix in REQUIRED_SUFFIXES}


@pytest.fixture(scope="module")
def generated() -> dict[str, dict[str, Path]]:
    """Every table's paths, or a module-wide skip on a checkout that has none."""
    built = {table_id: _paths(table_id) for table_id in SETUP_TABLE_IDS}
    if not built["T01"][".csv"].is_file():
        pytest.skip("T01-T07 not generated here (run scripts/18_setup_tables.py)")
    return built


# ---------------------------------------------------------------------------
# T86.7, first half -- every table in every format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_id", SETUP_TABLE_IDS)
def test_table_exists_in_all_four_formats_plus_provenance(
    generated: dict[str, dict[str, Path]], table_id: str
) -> None:
    for suffix, path in generated[table_id].items():
        assert path.is_file(), table_id + " is missing " + suffix
        assert path.stat().st_size > 0, table_id + suffix + " is empty"


@pytest.mark.parametrize("table_id", SETUP_TABLE_IDS)
def test_every_rendering_agrees_with_the_csv(
    generated: dict[str, dict[str, Path]], table_id: str
) -> None:
    paths = generated[table_id]
    meta = json.loads(paths[".meta.json"].read_text(encoding="utf-8"))
    source = pd.read_csv(paths[".csv"])
    kinds = meta["column_kinds"]

    expected_body = [
        [tb.format_value(source[column].iloc[row], kinds[column]) for column in source.columns]
        for row in range(len(source))
    ]
    for reader, suffix in (
        (tb.read_docx_table, ".docx"),
        (tb.read_latex_table, ".tex"),
        (tb.read_markdown_table, ".md"),
    ):
        rendered = reader(paths[suffix])
        assert rendered[1:] == expected_body, (
            table_id + suffix + " does not render the CSV under its own rounding rules"
        )


@pytest.mark.parametrize("table_id", SETUP_TABLE_IDS)
def test_row_count_survives_every_writer(
    generated: dict[str, dict[str, Path]], table_id: str
) -> None:
    paths = generated[table_id]
    expected = len(pd.read_csv(paths[".csv"]))
    assert len(tb.read_docx_table(paths[".docx"])) == expected + 1
    assert len(tb.read_latex_table(paths[".tex"])) == expected + 1
    assert len(tb.read_markdown_table(paths[".md"])) == expected + 1


# ---------------------------------------------------------------------------
# T86.7, second half -- each table records the source file it was built from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table_id", SETUP_TABLE_IDS)
def test_meta_names_at_least_one_source_and_an_explicit_experiment_id(
    generated: dict[str, dict[str, Path]], table_id: str
) -> None:
    meta = json.loads(generated[table_id][".meta.json"].read_text(encoding="utf-8"))
    assert meta["table_id"] == table_id
    assert meta["exp_id"], table_id + " has a blank experiment id"
    assert meta["sources"], table_id + " records no source file"
    assert meta["framework"] == "PV-MEPCG / PulseVision"


@pytest.mark.parametrize("table_id", SETUP_TABLE_IDS)
def test_recorded_sources_exist_and_are_not_stale(
    generated: dict[str, dict[str, Path]], table_id: str
) -> None:
    meta = json.loads(generated[table_id][".meta.json"].read_text(encoding="utf-8"))
    for fingerprint in meta["sources"]:
        source = PROJECT_ROOT / fingerprint["path"]
        assert source.is_file(), (
            table_id + " names a source that does not exist: " + fingerprint["path"]
        )
        current = hashlib.sha256(source.read_bytes()).hexdigest()
        assert current == fingerprint["sha256"], (
            table_id + " was built from an older " + fingerprint["path"] + ". The"
            " table is stale: regenerate it with"
            " `python scripts/18_setup_tables.py`. Do not relax this assertion."
        )


@pytest.mark.parametrize("table_id", SETUP_TABLE_IDS)
def test_the_source_file_is_named_inside_every_human_readable_rendering(
    generated: dict[str, dict[str, Path]], table_id: str
) -> None:
    from docx import Document

    paths = generated[table_id]
    meta = json.loads(paths[".meta.json"].read_text(encoding="utf-8"))
    names = [Path(f["path"]).name for f in meta["sources"]]

    docx_text = " ".join(p.text for p in Document(str(paths[".docx"])).paragraphs)
    renderings = {
        ".md": paths[".md"].read_text(encoding="utf-8"),
        ".tex": paths[".tex"].read_text(encoding="utf-8"),
        ".docx": docx_text,
    }
    for suffix, blob in renderings.items():
        for name in names:
            assert name in blob, table_id + suffix + " does not name " + name
        assert meta["exp_id"] in blob, table_id + suffix + " omits the experiment id"


# ---------------------------------------------------------------------------
# the numbers themselves, re-derived from the source rather than typed here
# ---------------------------------------------------------------------------


def test_t01_matches_da01_row_for_row(generated: dict[str, dict[str, Path]]) -> None:
    table = pd.read_csv(generated["T01"][".csv"])
    source = pd.read_csv(audit_dir() / "dataset_inventory.csv")
    assert list(table["dataset_source"]) == list(source["dataset_source"])
    assert list(table["total_files"]) == list(source["total_files"])
    assert list(table["usable_files"]) == list(source["usable_files"])


def test_t02_reports_the_supervised_scope_and_the_five_label_spaces(
    generated: dict[str, dict[str, Path]],
) -> None:
    table = pd.read_csv(generated["T02"][".csv"])
    source = pd.read_csv(audit_dir() / "class_distribution.csv")
    supervised = source[source["scope"] == SUPERVISED_SCOPE]

    assert len(table) == len(supervised)
    assert table["n_records"].sum() == supervised["n_records"].sum()
    # Rule 4: five separate label spaces, never merged into one target.
    assert set(table["task"]) == set(supervised["task"])
    assert len(set(table["task"])) == 5


def test_t03_pairs_each_duration_row_with_its_own_sampling_row(
    generated: dict[str, dict[str, Path]],
) -> None:
    table = pd.read_csv(generated["T03"][".csv"])
    durations = pd.read_csv(audit_dir() / "recording_duration_summary.csv")
    sampling = pd.read_csv(audit_dir() / "sampling_rate_summary.csv")
    supervised = durations[(durations["scope"] == SUPERVISED_SCOPE) & (durations["class"] == "ALL")]

    assert len(table) == len(supervised)
    for row in table.itertuples(index=False):
        expected_fs = sampling.loc[
            sampling["dataset_source"] == row.dataset_source, "original_fs"
        ].iloc[0]
        assert row.original_fs == expected_fs
        expected_n = supervised.loc[supervised["dataset_source"] == row.dataset_source, "n"].iloc[0]
        assert row.n == expected_n


def test_t03_does_not_mix_a_corpus_wide_total_into_a_supervised_summary(
    generated: dict[str, dict[str, Path]],
) -> None:
    """DA-04's total_hours covers ALL files; T03's rows cover the supervised subset.

    Carrying both in one row put 21.98 corpus hours beside 3,240 supervised
    PhysioNet records, whose actual duration is 20.2 h. The column was removed
    rather than relabelled; T01 reports corpus hours, with its scope named.
    """
    table = pd.read_csv(generated["T03"][".csv"])
    assert "total_hours" not in table.columns


def test_t05_counts_the_locked_138_features_in_registry_order(
    generated: dict[str, dict[str, Path]],
) -> None:
    table = pd.read_csv(generated["T05"][".csv"])
    inventory = pd.read_csv(outputs_dir("features") / "feature_inventory.csv")

    assert table["counted_features"].sum() == 138
    assert table["counted_features"].sum() == len(inventory)
    assert (table["counted_features"] == table["expected_count"]).all()
    # Registry order, not alphabetical: the 138-vector's column order is fixed.
    assert list(table["first_index"]) == sorted(table["first_index"])
    assert list(table["family"]) == ["time", "frequency", "mfcc", "chroma", "dwt", "envelope"]


def test_t06_declares_m9_as_an_explicit_exclusion_rather_than_omitting_it(
    generated: dict[str, dict[str, Path]],
) -> None:
    table = pd.read_csv(generated["T06"][".csv"])
    registry = pd.read_csv(outputs_dir("models") / "model_registry.csv")
    assert list(table["model_id"]) == list(registry["model_id"])

    m9 = table[table["model_id"] == "M9"].iloc[0]
    assert not bool(m9["implemented"])
    assert isinstance(m9["unavailable_reason"], str) and m9["unavailable_reason"]


def test_t07_keeps_hyperparameter_values_at_full_precision(
    generated: dict[str, dict[str, Path]],
) -> None:
    """A rounded C cannot be pasted back into a config; rule 5 outranks tidiness."""
    meta = json.loads(generated["T07"][".meta.json"].read_text(encoding="utf-8"))
    assert meta["column_kinds"]["so_01_selected"] == "text"
    assert meta["column_kinds"]["final_selected"] == "text"
    # Scores in the same table are metrics and DO follow the 3-decimal rule.
    assert meta["column_kinds"]["so_01_score"] == "metric"

    rendered = tb.read_markdown_table(generated["T07"][".md"])
    source = pd.read_csv(
        outputs_dir("search_optimization") / "search_space_and_best_parameters.csv"
    )
    shipped = str(source["final_selected"].iloc[0])
    assert any(shipped in cell for row in rendered for cell in row), (
        "T07 rounded a hyperparameter value that must reproduce the run"
    )
