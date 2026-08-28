"""T62.7 -- Part VI reporting: T07, G20, G21, G22 exist with their sources, and
every SO artifact is registered in the evidence index.

A figure is a set of numbers with axes on it, so the gate on a figure is the gate
on a number: it must exist, and the CSV it was drawn from must exist beside it.
Checking only that a PNG is on disk would pass on a blank canvas.

G20 gets one extra assertion the other two do not need, because it is the figure
that can most easily be wrong while looking right -- see the module docstring of
``src/reporting/search_report.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _section() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.search_optimization"))


def _figures() -> Path:
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.figures_diagrams"))


def _evidence() -> Any:
    import pandas as pd

    from src.utils.config import load_config

    path = (
        Path(load_config("paths").require("outputs.evidence_index"))
        / "evidence_index.csv"
        if "outputs.evidence_index" in load_config("paths")
        else Path(load_config("paths").require("outputs.root"))
        / "00_evidence_index"
        / "evidence_index.csv"
    )
    if not path.exists():
        pytest.skip(str(path) + " does not exist")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# T62.1 -- T07
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t07() -> Any:
    from src.reporting import search_report as sr

    path = _section() / sr.T07_FILENAME
    if not path.exists():
        pytest.skip(str(path) + " does not exist; run scripts/10_search_reports.py")
    return sr.read_t07(path)


def test_reading_t07_naively_would_erase_a_real_chosen_value() -> None:
    """Why `read_t07` exists, stated as a measurement rather than a comment.

    `class_weight=None` is a value SO-02 actually selected for M3. pandas treats
    the string "None" as missing by default, so a naive read turns that decision
    into a blank -- indistinguishable from "not recorded".
    """
    import pandas as pd

    from src.reporting import search_report as sr

    path = _section() / sr.T07_FILENAME
    if not path.exists():
        pytest.skip(str(path) + " does not exist")
    naive = pd.read_csv(path)
    careful = sr.read_t07(path)

    assert careful["final_selected"].notna().all()
    erased = int(naive["final_selected"].isna().sum())
    assert erased > 0, "expected pandas to erase at least one None; the trap moved"
    assert (careful.loc[naive["final_selected"].isna(), "final_selected"] == "None").all()


def test_t07_names_a_range_and_a_distribution_for_every_searched_parameter(
    t07: Any,
) -> None:
    """T62.1 asks for variable, range, distribution and final selected value."""
    for column in ("model_id", "parameter", "distribution", "range_or_choices",
                   "final_selected", "final_source"):
        assert column in t07.columns, column
    assert len(t07) > 0
    assert t07["parameter"].notna().all()
    assert (t07["distribution"].astype(str).str.len() > 0).all()
    assert (t07["range_or_choices"].astype(str).str.len() > 0).all()


def test_t07_covers_every_model_that_was_actually_searched(t07: Any) -> None:
    searched = set(
        t07.loc[t07["so_01_searched"] | t07["so_02_searched"], "model_id"].unique()
    )
    assert searched, "T07 records no model as searched"
    for model_id in searched:
        block = t07[t07["model_id"] == model_id]
        assert (block["final_source"] != "").any(), model_id


def test_t07_final_values_are_traceable_to_a_named_method(t07: Any) -> None:
    """No selected value may appear without saying which search produced it.

    And no cell may be blank: an empty string round-trips through CSV as NaN and
    renders as the literal "nan", which a reader cannot distinguish from a value.
    """
    from src.reporting.search_report import NOT_SEARCHED

    assert t07["final_source"].notna().all(), "T07 has NaN in final_source"
    assert t07["final_selected"].notna().all(), "T07 has NaN in final_selected"
    assert set(t07["final_source"].unique()) <= {"SO-01", "SO-02", NOT_SEARCHED}

    searched = t07[t07["final_source"] != NOT_SEARCHED]
    assert len(searched) > 0
    assert (searched["final_selected"] != NOT_SEARCHED).all()


# ---------------------------------------------------------------------------
# T62.2 / T62.3 / T62.4 -- the figures, each with its source CSV
# ---------------------------------------------------------------------------


FIGURE_SOURCES: dict[str, tuple[str, ...]] = {
    "G20": ("SO-01/convergence.csv", "SO-02/convergence.csv",
            "SO-03a/convergence.csv", "SO-03b/convergence.csv"),
    "G21": ("SO-04/all_features_vs_selected.csv",),
    "G22": ("SO-04/feature_count_curve.csv",),
}


@pytest.mark.parametrize("figure_id", ["G20", "G21", "G22"])
def test_the_figure_exists_and_is_not_an_empty_canvas(figure_id: str) -> None:
    from src.reporting import search_report as sr

    path = _figures() / sr.FIGURES[figure_id]
    if not path.exists():
        pytest.skip(str(path) + " does not exist; run scripts/10_search_reports.py")
    assert path.stat().st_size > 20_000, (
        figure_id + " is only " + str(path.stat().st_size) + " bytes; a 300-dpi plot "
        "with data on it is larger than that"
    )


@pytest.mark.parametrize("figure_id", ["G20", "G21", "G22"])
def test_the_figure_has_its_source_csv_beside_it(figure_id: str) -> None:
    """T62.7: 'with their source CSVs'. A figure whose source is gone is a claim."""
    from src.reporting import search_report as sr

    if not (_figures() / sr.FIGURES[figure_id]).exists():
        pytest.skip(figure_id + " has not been generated")
    for relative in FIGURE_SOURCES[figure_id]:
        assert (_section() / relative).is_file(), figure_id + " lost " + relative


def test_g20_draws_the_two_search_families_on_separate_axes() -> None:
    """The one way this figure can be wrong while looking right.

    SO-01/SO-02 record `best_so_far` as a RISING balanced accuracy; SO-03a/SO-03b
    record it as a FALLING J. The two ranges overlap, so a shared axis produces a
    plausible figure asserting that the mask searches converged far below the
    hyperparameter searches -- comparing a cost against an accuracy. The builder
    must therefore emit more than one axis.
    """
    import pandas as pd

    hyper = _section() / "SO-01" / "convergence.csv"
    mask = _section() / "SO-03a" / "convergence.csv"
    if not (hyper.is_file() and mask.is_file()):
        pytest.skip("both convergence traces are needed for this comparison")

    accuracy = pd.read_csv(hyper)["best_so_far"]
    cost = pd.read_csv(mask)["best_so_far"]
    # The hazard is real: the ranges genuinely overlap on this data.
    assert accuracy.min() < cost.max() or cost.min() < accuracy.max()
    # Rising versus falling, which is what makes a shared axis meaningless.
    assert accuracy.iloc[-1] >= accuracy.iloc[0]
    assert cost.iloc[-1] <= cost.iloc[0]

    source = Path("src/reporting/search_report.py").read_text(encoding="utf-8")
    assert "nrows=2" in source, "G20 must not be drawn on a single shared axis"


# ---------------------------------------------------------------------------
# T62.5 -- the evidence index
# ---------------------------------------------------------------------------


def test_every_so_artifact_on_disk_is_registered_in_the_evidence_index() -> None:
    """T62.5, driven from the files rather than from a hand-kept list."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "search_reports_script", Path("scripts/10_search_reports.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    section = _section()
    present = [
        (evidence_id, relative)
        for evidence_id, relative, _ in module.SO_ARTIFACTS
        if (section / relative).is_file()
    ]
    if not present:
        pytest.skip("no SO artifacts on disk yet")

    index = _evidence()
    registered = set(index["evidence_id"].astype(str))
    missing = [evidence_id for evidence_id, _ in present if evidence_id not in registered]
    assert not missing, "on disk but not in the evidence index: " + ", ".join(sorted(missing))


@pytest.mark.parametrize("figure_id", ["G20", "G21", "G22"])
def test_each_figure_is_registered(figure_id: str) -> None:
    from src.reporting import search_report as sr

    if not (_figures() / sr.FIGURES[figure_id]).exists():
        pytest.skip(figure_id + " has not been generated")
    index = _evidence()
    assert figure_id in set(index["evidence_id"].astype(str))


def test_no_registered_search_artifact_is_recorded_as_missing() -> None:
    """A row pointing at a file that is not there is worse than no row."""
    index = _evidence()
    search_rows = index[
        index["evidence_id"].astype(str).str.startswith(("SO-", "T07", "G2", "FE-12"))
    ]
    if search_rows.empty:
        pytest.skip("no search artifacts registered yet")
    broken = search_rows[search_rows["status"] != "ok"]
    assert broken.empty, broken[["evidence_id", "filename", "status"]].to_dict("records")


# ---------------------------------------------------------------------------
# T62.6 -- the changelog
# ---------------------------------------------------------------------------


def _changelog_phases() -> set[int]:
    """Every phase number CHANGELOG.md claims, expanding its `Phases A-B` ranges.

    The file groups phases into one entry where they shipped together, which is
    its established convention -- so this has to read a range as covering every
    phase inside it rather than looking for one heading per phase.
    """
    import re

    text = Path("CHANGELOG.md").read_text(encoding="utf-8")
    covered: set[int] = set()
    # The heading may separate the two numbers with a hyphen or an en dash; the
    # dash class is written as an escape so this file itself stays plain ASCII
    # and ruff's ambiguous-character rule has nothing to object to.
    dashes = "-" + chr(0x2013)
    pattern = r"^## Phases? (\d+)(?:\s*[" + dashes + r"]\s*(\d+))?"
    for first, last in re.findall(pattern, text, re.M):
        covered.update(range(int(first), int(last or first) + 1))
    return covered


@pytest.mark.parametrize("phase", [57, 58, 59, 60, 61, 62])
def test_the_changelog_records_each_part_vi_phase(phase: int) -> None:
    """T62.6. A phase whose gate passed but which never reached CHANGELOG.md is invisible."""
    if not Path("CHANGELOG.md").exists():
        pytest.skip("CHANGELOG.md does not exist")
    covered = _changelog_phases()
    assert phase in covered, (
        "CHANGELOG.md has no entry covering Phase " + str(phase)
        + "; it covers " + str(sorted(covered))
    )


def test_the_changelog_range_parser_actually_expands_ranges() -> None:
    """Otherwise the check above could pass by matching nothing at all."""
    covered = _changelog_phases()
    assert len(covered) > 6
    assert 1 in covered, "Phase 01 should still be recorded"


def test_the_operating_point_and_fe12_agree_on_the_shipped_subset() -> None:
    """One project, one final feature subset. Two files that could disagree is one too many."""
    import pandas as pd

    from src.utils.config import load_config

    operating_path = _section() / "SO-06" / "operating_point.json"
    subset_path = (
        Path(load_config("paths").require("outputs.features")) / "selected_feature_subset.csv"
    )
    if not (operating_path.is_file() and subset_path.is_file()):
        pytest.skip("SO-06 or FE-12 has not been generated")

    operating = json.loads(operating_path.read_text(encoding="utf-8"))
    subset = pd.read_csv(subset_path)
    assert int(operating["k"]) == len(subset)
    assert operating["ranker"] == str(subset["ranker"].iloc[0])
