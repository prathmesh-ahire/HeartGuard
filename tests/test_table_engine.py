"""The T85.7 gate: one table through every writer, cell for cell (Phase 85).

The claim being tested is narrow and load-bearing: **the CSV is the numbers and
the other three files are renderings of it.** So the test does not check that
the DOCX "looks right" -- it reads every rendered cell back out of the DOCX, the
LaTeX and the Markdown and compares each one against the value it should have
been given the CSV's full-precision value and the column's declared kind. A
writer that silently rounds differently, drops a column, or renders a NaN as
``0`` fails here rather than in a thesis table.

The rounding rules (T85.6) are asserted as *values*, not as configuration:
0.856789 must appear as ``0.857`` and never ``0.86`` or ``0.8568``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import tables as tb

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Deliberately awkward: a NaN, an inf, a LaTeX metacharacter, a big count."""
    return pd.DataFrame(
        {
            "model_name": ["SVM (RBF)", "Random Forest", "Gradient_Boosting & co."],
            "n_records": [3240, 461, 124],
            "sensitivity": [0.856789, 0.7125, np.nan],
            "share_pct": [79.4753, 20.52468, 100.0],
            "fit_seconds": [12.3456, 0.5, np.inf],
            "summary": ["0.8588 +/- 0.0255", "0.7100 +/- 0.0300", "n/a"],
        }
    )


@pytest.fixture
def sample_spec() -> tb.TableSpec:
    return tb.TableSpec(
        table_id="T99",
        title="Table Engine Self Check",
        caption="Synthetic table used by the T85.7 gate. Not a result.",
        sources=("outputs/01_dataset_audit/dataset_inventory.csv",),
        columns=(
            tb.Column("model_name", header="Model", kind="text"),
            tb.Column("n_records", header="Records", kind="count"),
            tb.Column("sensitivity", header="Sensitivity", kind="metric"),
            tb.Column("share_pct", header="Share (%)", kind="percent"),
            tb.Column("fit_seconds", header="Fit (s)", kind="seconds"),
            tb.Column("summary", header="Mean +/- SD", kind="preformatted"),
        ),
        exp_id="EXP-SELFCHECK",
        objective="O0 (engine self-check)",
        notes=("A note that must survive into every writer.",),
        command="pytest tests/test_table_engine.py",
    )


@pytest.fixture
def written(
    sample_spec: tb.TableSpec, sample_frame: pd.DataFrame, tmp_path: Path
) -> dict[str, Path]:
    table = tb.build_table(sample_spec, sample_frame)
    return tb.write_table(table, tmp_path, evidence_index=tmp_path / "evidence_index.csv")


# ---------------------------------------------------------------------------
# T85.6 -- the rounding rules, asserted as literal rendered values
# ---------------------------------------------------------------------------


def test_metric_renders_at_three_decimals() -> None:
    assert tb.format_value(0.856789, "metric") == "0.857"
    assert tb.format_value(0.7125, "metric") == "0.713"
    assert tb.format_value(1.0, "metric") == "1.000"


def test_percent_renders_at_one_decimal() -> None:
    assert tb.format_value(79.4753, "percent") == "79.5"
    assert tb.format_value(100.0, "percent") == "100.0"


def test_count_renders_as_a_separated_integer() -> None:
    assert tb.format_value(3240, "count") == "3,240"
    assert tb.format_value(124.0, "count") == "124"


def test_nan_renders_as_na_and_never_as_zero() -> None:
    assert tb.format_value(np.nan, "metric") == tb.NA_TEXT
    assert tb.format_value(None, "metric") == tb.NA_TEXT
    assert tb.format_value(pd.NA, "count") == tb.NA_TEXT
    assert tb.format_value(np.nan, "metric") != "0.000"


def test_infinity_is_labelled_not_formatted() -> None:
    assert tb.format_value(np.inf, "seconds") == "inf"
    assert tb.format_value(-np.inf, "seconds") == "-inf"


def test_preformatted_and_text_pass_through_untouched() -> None:
    assert tb.format_value("0.8588 +/- 0.0255", "preformatted") == "0.8588 +/- 0.0255"
    assert tb.format_value("SVM (RBF)", "text") == "SVM (RBF)"


def test_declared_places_override_the_kind_default() -> None:
    assert tb.format_value(0.856789, "metric", places=4) == "0.8568"


# ---------------------------------------------------------------------------
# T85.2 -- the CSV keeps full precision
# ---------------------------------------------------------------------------


def test_csv_preserves_full_precision(written: dict[str, Path], sample_frame: pd.DataFrame) -> None:
    back = pd.read_csv(written["csv"])
    assert back["sensitivity"].iloc[0] == pytest.approx(0.856789, abs=0.0)
    assert back["share_pct"].iloc[1] == pytest.approx(20.52468, abs=0.0)
    # And the CSV carries no formatting, no provenance columns, no rounding.
    assert list(back.columns) == list(sample_frame.columns)


# ---------------------------------------------------------------------------
# T85.7 -- every writer renders the SAME numbers
# ---------------------------------------------------------------------------


def _expected_rows(spec: tb.TableSpec, frame: pd.DataFrame) -> list[list[str]]:
    table = tb.build_table(spec, frame)
    display = tb.formatted_frame(table)
    return [[str(c) for c in display.columns]] + [
        [str(v) for v in row] for row in display.itertuples(index=False)
    ]


def test_all_four_writers_agree_cell_for_cell(
    written: dict[str, Path], sample_spec: tb.TableSpec, sample_frame: pd.DataFrame
) -> None:
    expected = _expected_rows(sample_spec, sample_frame)

    from_docx = tb.read_docx_table(written["docx"])
    from_latex = tb.read_latex_table(written["latex"])
    from_md = tb.read_markdown_table(written["md"])

    assert from_docx == expected, "DOCX disagrees with the formatted CSV"
    assert from_latex == expected, "LaTeX disagrees with the formatted CSV"
    assert from_md == expected, "Markdown disagrees with the formatted CSV"


def test_rendered_values_are_the_csv_values_under_the_rounding_rules(
    written: dict[str, Path],
) -> None:
    """The renderings are re-derived from the CSV itself, not from the fixture."""
    source = pd.read_csv(written["csv"])
    kinds = json.loads(written["meta"].read_text(encoding="utf-8"))["column_kinds"]

    body = tb.read_docx_table(written["docx"])[1:]
    for row_index, row in enumerate(body):
        for column_index, column in enumerate(source.columns):
            expected = tb.format_value(source[column].iloc[row_index], kinds[column])
            assert row[column_index] == expected, "row " + str(row_index) + " column " + str(column)


def test_a_nan_survives_as_na_in_every_rendering(written: dict[str, Path]) -> None:
    for reader, key in (
        (tb.read_docx_table, "docx"),
        (tb.read_latex_table, "latex"),
        (tb.read_markdown_table, "md"),
    ):
        flat = [cell for row in reader(written[key]) for cell in row]
        assert tb.NA_TEXT in flat, key + " lost the NaN"
        assert "0.000" not in flat, key + " rendered a NaN as a number"


def test_latex_metacharacters_are_escaped_and_round_trip(written: dict[str, Path]) -> None:
    raw = written["latex"].read_text(encoding="utf-8")
    assert "Gradient\\_Boosting \\& co." in raw
    assert "\\toprule" in raw and "\\bottomrule" in raw
    assert tb.read_latex_table(written["latex"])[3][0] == "Gradient_Boosting & co."


# ---------------------------------------------------------------------------
# T85.5 -- provenance
# ---------------------------------------------------------------------------


def test_every_writer_records_the_experiment_id_and_the_source_file(
    written: dict[str, Path],
) -> None:
    latex = written["latex"].read_text(encoding="utf-8")
    markdown = written["md"].read_text(encoding="utf-8")
    docx_text = " ".join(
        p.text for p in __import__("docx").Document(str(written["docx"])).paragraphs
    )
    for blob, name in ((latex, "latex"), (markdown, "md"), (docx_text, "docx")):
        assert "EXP-SELFCHECK" in blob, name + " lost the experiment id"
        assert "dataset_inventory.csv" in blob, name + " lost the source file"


def test_meta_json_fingerprints_every_source(written: dict[str, Path]) -> None:
    meta = json.loads(written["meta"].read_text(encoding="utf-8"))
    assert meta["table_id"] == "T99"
    assert meta["exp_id"] == "EXP-SELFCHECK"
    assert meta["framework"] == "PV-MEPCG / PulseVision"
    assert meta["rounding_rules"]["metric"] == 3
    assert meta["rounding_rules"]["percent"] == 1
    assert meta["column_kinds"]["sensitivity"] == "metric"
    assert len(meta["sources"]) == 1
    fingerprint = meta["sources"][0]
    if fingerprint["exists"]:
        assert len(fingerprint["sha256"]) == 64
        assert fingerprint["bytes"] > 0


def test_a_setup_table_records_an_explicit_not_experiment_bound_marker(
    sample_frame: pd.DataFrame, tmp_path: Path
) -> None:
    spec = tb.TableSpec(
        table_id="T98",
        title="No Experiment",
        caption="Setup table.",
        sources=("outputs/01_dataset_audit/dataset_inventory.csv",),
    )
    out = tb.write_table(
        tb.build_table(spec, sample_frame),
        tmp_path,
        evidence_index=tmp_path / "evidence_index.csv",
    )
    meta = json.loads(out["meta"].read_text(encoding="utf-8"))
    assert meta["exp_id"] == tb.NOT_EXPERIMENT_BOUND
    assert tb.NOT_EXPERIMENT_BOUND in out["md"].read_text(encoding="utf-8")


def test_evidence_registration_lands_in_the_given_index(
    written: dict[str, Path], tmp_path: Path
) -> None:
    index = tmp_path / "evidence_index.csv"
    assert index.is_file()
    rows = pd.read_csv(index)
    row = rows[rows["evidence_id"] == "T99"].iloc[0]
    assert row["status"] == "ok"
    assert "dataset_inventory.csv" in row["source_data"]


# ---------------------------------------------------------------------------
# refusals -- the engine must not produce a plausible-looking wrong table
# ---------------------------------------------------------------------------


def test_a_missing_declared_column_raises_rather_than_dropping_it(
    sample_spec: tb.TableSpec, sample_frame: pd.DataFrame
) -> None:
    with pytest.raises(KeyError, match="missing declared column"):
        tb.build_table(sample_spec, sample_frame.drop(columns=["sensitivity"]))


def test_an_empty_frame_is_refused(sample_spec: tb.TableSpec) -> None:
    empty = pd.DataFrame({c.name: [] for c in sample_spec.columns})
    with pytest.raises(ValueError, match="empty table"):
        tb.build_table(sample_spec, empty)


def test_an_unknown_writer_is_refused(
    sample_spec: tb.TableSpec, sample_frame: pd.DataFrame, tmp_path: Path
) -> None:
    table = tb.build_table(sample_spec, sample_frame)
    with pytest.raises(ValueError, match="unknown table writer"):
        tb.write_table(table, tmp_path, formats=("csv", "pdf"))


# ---------------------------------------------------------------------------
# the same engine over a real project CSV, not a synthetic one
# ---------------------------------------------------------------------------


def test_engine_round_trips_a_real_audit_csv(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "outputs" / "01_dataset_audit" / "class_distribution.csv"
    if not source.is_file():
        pytest.skip("DA-02 class_distribution.csv not present (pipeline not run here)")

    frame = pd.read_csv(source)
    spec = tb.TableSpec(
        table_id="T97",
        title="Real Source Round Trip",
        caption="DA-02 rendered through the engine.",
        sources=(source.relative_to(PROJECT_ROOT).as_posix(),),
        columns=(
            tb.Column("dataset_source", header="ID", kind="text"),
            tb.Column("class", header="Class", kind="text"),
            tb.Column("n_records", header="Records", kind="count"),
            tb.Column("share", header="Share", kind="metric"),
            tb.Column("imbalance_ratio", header="Imbalance", kind="metric"),
        ),
    )
    out = tb.write_table(
        tb.build_table(spec, frame),
        tmp_path,
        evidence_index=tmp_path / "evidence_index.csv",
    )
    expected = _expected_rows(spec, frame)
    assert tb.read_docx_table(out["docx"]) == expected
    assert tb.read_latex_table(out["latex"]) == expected
    assert tb.read_markdown_table(out["md"]) == expected

    # And the counts are not quietly transformed on the way through.
    rendered = pd.read_csv(out["csv"])
    assert rendered["n_records"].sum() == frame["n_records"].sum()


# ---------------------------------------------------------------------------
# the source digest must mean the same thing on Windows and on CI
# ---------------------------------------------------------------------------


def test_content_digest_ignores_line_endings_for_text(tmp_path: Path) -> None:
    """The regression test for a red build.

    `.gitattributes` declares `*.csv text eol=lf`, so the repository stores LF
    while a CSV written by pandas on Windows has CRLF in the working tree. A raw
    byte digest therefore reports every table "stale" on CI and clean locally,
    which is exactly what happened. The digest is a CONTENT identity.
    """
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n3,4\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n3,4\r\n")

    assert lf.read_bytes() != crlf.read_bytes()
    assert tb.content_digest(lf)[0] == tb.content_digest(crlf)[0]
    assert tb.content_digest(lf)[1] == "sha256/lf"


def test_content_digest_still_detects_a_real_edit(tmp_path: Path) -> None:
    original = tmp_path / "a.csv"
    edited = tmp_path / "b.csv"
    original.write_bytes(b"a,b\n1,2\n")
    edited.write_bytes(b"a,b\n1,3\n")
    assert tb.content_digest(original)[0] != tb.content_digest(edited)[0]


def test_content_digest_does_not_normalize_a_binary_file(tmp_path: Path) -> None:
    """Rewriting CRLF inside a .docx or a .png would change its actual content."""
    raw = tmp_path / "figure.png"
    raw.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\r\n\x02")
    digest, method = tb.content_digest(raw)
    assert method == "sha256/raw"
    import hashlib

    assert digest == hashlib.sha256(raw.read_bytes()).hexdigest()


def test_source_fingerprint_records_which_digest_method_it_used(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\r\n1,2\r\n")
    fingerprint = tb.source_fingerprint(source)
    assert fingerprint["exists"] is True
    assert fingerprint["digest_method"] == "sha256/lf"
    assert fingerprint["sha256"] == tb.content_digest(source)[0]

    missing = tb.source_fingerprint(tmp_path / "nope.csv")
    assert missing["exists"] is False
    assert missing["sha256"] is None
    assert missing["digest_method"] is None
