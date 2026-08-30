"""The T91.7 gate: G01-G10 exist with their source CSVs, and G02 matches DA-02.

Everything here is checked against the **emitted artifacts**, and every expected
value is re-derived from the upstream CSV rather than typed in. A literal in a
test is a hand-typed number under rule 1 exactly as much as a literal in a
figure caption.

The signal figures (G05-G09) are checked through their committed source CSVs
rather than by re-running the extractors, so this gate needs no audio and no
dataset. What it can still prove from those CSVs is the part that matters: that
the window is the documented one, that the shapes match the configured
extractor settings, and that the values are finite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import graphs as gr
from src.reporting.data_graphs import (
    DATA_GRAPH_IDS,
    DURATION_BIN_EDGES,
    FAMILY_ORDER,
    SUPERVISED_SCOPE,
    WINDOW_SECONDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEMS = {
    "G01": "G01_dataset_recording_counts",
    "G02": "G02_class_distribution",
    "G03": "G03_class_distribution_shares",
    "G04": "G04_recording_duration_histogram",
    "G05": "G05_before_and_after_filtering",
    "G06": "G06_normal_versus_abnormal_spectrogram",
    "G07": "G07_mfcc_heatmap",
    "G08": "G08_chroma_heatmap",
    "G09": "G09_wavelet_decomposition",
    "G10": "G10_feature_family_counts",
}


def _audit(name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "01_dataset_audit" / name


@pytest.fixture(scope="module")
def figures() -> dict[str, dict[str, Path]]:
    directory = gr.figures_dir()
    built = {
        figure_id: {
            suffix: directory / (STEMS[figure_id] + suffix)
            for suffix in (".png", ".csv", ".meta.json")
        }
        for figure_id in DATA_GRAPH_IDS
    }
    if not built["G01"][".png"].is_file():
        pytest.skip("G01-G10 not generated here (run scripts/19_data_graphs.py)")
    return built


@pytest.fixture(scope="module")
def registry() -> dict[str, dict[str, str]]:
    rows = gr.read_registry()
    if not rows:
        pytest.skip("figure_registry.csv not present")
    return {row["figure_id"]: row for row in rows}


# ---------------------------------------------------------------------------
# T91.7, first half -- all ten exist, each with the CSV that produced it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("figure_id", DATA_GRAPH_IDS)
def test_figure_exists_with_a_non_empty_png_csv_and_provenance(
    figures: dict[str, dict[str, Path]], figure_id: str
) -> None:
    for suffix, path in figures[figure_id].items():
        assert path.is_file(), figure_id + " is missing " + suffix
        assert path.stat().st_size > 0, figure_id + suffix + " is empty"
    png = figures[figure_id][".png"]
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.stat().st_size > 20_000, figure_id + " PNG is too small to be a 300 dpi figure"


@pytest.mark.parametrize("figure_id", DATA_GRAPH_IDS)
def test_the_plotted_csv_is_non_empty_and_finite(
    figures: dict[str, dict[str, Path]], figure_id: str
) -> None:
    frame = pd.read_csv(figures[figure_id][".csv"])
    assert len(frame) > 0
    numeric = frame.select_dtypes(include="number")
    assert numeric.shape[1] > 0, figure_id + " plotted no numeric values"
    assert np.isfinite(numeric.to_numpy(float)).all(), figure_id + " plotted a NaN or an infinity"


@pytest.mark.parametrize("figure_id", DATA_GRAPH_IDS)
def test_meta_records_the_plotted_csv_its_digest_and_the_upstream_sources(
    figures: dict[str, dict[str, Path]], figure_id: str
) -> None:
    from src.reporting.tables import content_digest

    paths = figures[figure_id]
    meta = json.loads(paths[".meta.json"].read_text(encoding="utf-8"))

    assert meta["figure_id"] == figure_id
    assert meta["framework"] == "PV-MEPCG / PulseVision"
    assert meta["dpi"] == 300
    assert meta["exp_id"], figure_id + " has a blank experiment id"
    assert meta["sources"], figure_id + " records no upstream source"
    assert meta["plotted_csv"] == paths[".csv"].name
    digest, method = content_digest(paths[".csv"])
    assert meta["plotted_csv_digest_method"] == method
    assert meta["plotted_csv_sha256"] == digest, (
        figure_id + " was drawn from a different CSV than the one on disk"
    )
    assert meta["plotted_rows"] == len(pd.read_csv(paths[".csv"]))


def test_the_registry_numbers_all_ten_in_series_order(registry: dict[str, dict[str, str]]) -> None:
    """G10 must be figure 10, not figure 5 from a partial batch."""
    for figure_id in DATA_GRAPH_IDS:
        assert figure_id in registry, figure_id + " is not in figure_registry.csv"
        expected = int(figure_id[1:])
        assert int(registry[figure_id]["figure_number"]) == expected, (
            figure_id
            + " holds figure number "
            + registry[figure_id]["figure_number"]
            + ", not "
            + str(expected)
        )


@pytest.mark.parametrize("figure_id", DATA_GRAPH_IDS)
def test_the_registry_row_points_at_files_that_exist(
    figures: dict[str, dict[str, Path]],
    registry: dict[str, dict[str, str]],
    figure_id: str,
) -> None:
    # `figures` is requested for its skip, not its value: a checkout that has the
    # registry but not the artifacts must report skipped, not failed.
    assert figures
    directory = gr.figures_dir()
    row = registry[figure_id]
    assert (directory / row["filename"]).is_file()
    assert (directory / row["source_csv"]).is_file()
    assert row["caption"].strip()


# ---------------------------------------------------------------------------
# T91.7, second half -- G02's counts match DA-02 EXACTLY
# ---------------------------------------------------------------------------


def test_g02_class_counts_match_da02_exactly(figures: dict[str, dict[str, Path]]) -> None:
    plotted = pd.read_csv(figures["G02"][".csv"])
    source = pd.read_csv(_audit("class_distribution.csv"))
    supervised = source[source["scope"] == SUPERVISED_SCOPE].reset_index(drop=True)

    assert len(plotted) == len(supervised)
    key = ["dataset_source", "task", "class"]
    merged = plotted.merge(supervised, on=key, how="inner", suffixes=("_plot", "_da02"))
    assert len(merged) == len(supervised), "G02 and DA-02 disagree on which rows exist"
    assert (merged["n_records_plot"] == merged["n_records_da02"]).all()
    assert merged["share_plot"].equals(merged["share_da02"])
    assert int(plotted["n_records"].sum()) == int(supervised["n_records"].sum())


def test_g02_keeps_the_five_label_spaces_separate(figures: dict[str, dict[str, Path]]) -> None:
    """Rule 4. A single merged bar chart would be the visual form of merging targets."""
    plotted = pd.read_csv(figures["G02"][".csv"])
    assert len(set(plotted["task"])) == 5
    # No class name may appear under two tasks without its task distinguishing it.
    for task, block in plotted.groupby("task"):
        assert block["class"].is_unique, "duplicate class within " + str(task)


def test_g03_plots_the_same_counts_as_g02(figures: dict[str, dict[str, Path]]) -> None:
    """The pie is a second view of one measurement, not a second measurement."""
    bars = pd.read_csv(figures["G02"][".csv"])
    pies = pd.read_csv(figures["G03"][".csv"])
    pd.testing.assert_frame_equal(bars, pies)


# ---------------------------------------------------------------------------
# the rest of the corpus figures, re-derived from their own sources
# ---------------------------------------------------------------------------


def test_g01_counts_match_da01(figures: dict[str, dict[str, Path]]) -> None:
    plotted = pd.read_csv(figures["G01"][".csv"])
    source = pd.read_csv(_audit("dataset_inventory.csv"))
    assert list(plotted["dataset_source"]) == list(source["dataset_source"])
    assert list(plotted["total_files"]) == list(source["total_files"])
    assert list(plotted["usable_files"]) == list(source["usable_files"])
    # The derived column is the difference, not a third measurement.
    assert (plotted["unlabelled_files"] == plotted["total_files"] - plotted["usable_files"]).all()


def test_g04_bins_account_for_every_supervised_record(figures: dict[str, dict[str, Path]]) -> None:
    plotted = pd.read_csv(figures["G04"][".csv"])
    master = pd.read_csv(_audit("metadata_master.csv"))
    supervised = master[master["use_in_supervised"].astype(bool)]

    assert int(plotted["n_records"].sum()) == len(supervised), (
        "a recording falls outside the declared bin edges"
    )
    for dataset, block in plotted.groupby("dataset_source"):
        expected = int((supervised["dataset_source"] == dataset).sum())
        assert int(block["n_records"].sum()) == expected


def test_g04_uses_the_declared_bin_edges_not_data_derived_ones(
    figures: dict[str, dict[str, Path]],
) -> None:
    """Derived edges would silently rebin every published histogram on a rerun."""
    plotted = pd.read_csv(figures["G04"][".csv"])
    edges = sorted(set(plotted["bin_low_sec"]) | set(plotted["bin_high_sec"]))
    assert edges == list(DURATION_BIN_EDGES)


def test_g10_sums_to_the_locked_138_in_registry_order(figures: dict[str, dict[str, Path]]) -> None:
    plotted = pd.read_csv(figures["G10"][".csv"])
    inventory = pd.read_csv(PROJECT_ROOT / "outputs" / "03_features" / "feature_inventory.csv")
    assert list(plotted["family"]) == list(FAMILY_ORDER)
    assert int(plotted["n_features"].sum()) == 138
    assert int(plotted["n_features"].sum()) == len(inventory)
    assert list(plotted["first_index"]) == sorted(plotted["first_index"])


# ---------------------------------------------------------------------------
# the signal figures, checked through their committed CSVs
# ---------------------------------------------------------------------------


def test_g05_plots_the_documented_window_of_raw_against_filtered(
    figures: dict[str, dict[str, Path]],
) -> None:
    plotted = pd.read_csv(figures["G05"][".csv"])
    assert list(plotted.columns) == ["time_sec", "raw_amplitude", "filtered_amplitude"]
    span = float(plotted["time_sec"].iloc[-1]) - float(plotted["time_sec"].iloc[0])
    assert span == pytest.approx(WINDOW_SECONDS, abs=0.01)
    # Filtering must have changed the signal, and not to zero.
    assert not np.allclose(plotted["raw_amplitude"], plotted["filtered_amplitude"])
    assert float(np.abs(plotted["filtered_amplitude"]).max()) > 0.0


def test_g06_carries_both_roles_on_one_shared_frequency_axis(
    figures: dict[str, dict[str, Path]],
) -> None:
    plotted = pd.read_csv(figures["G06"][".csv"])
    assert set(plotted["role"]) == {"normal", "abnormal"}
    normal = plotted[plotted["role"] == "normal"]["frequency_hz"].to_numpy(float)
    abnormal = plotted[plotted["role"] == "abnormal"]["frequency_hz"].to_numpy(float)
    assert np.array_equal(normal, abnormal), "the two panels use different frequency bins"
    times = [float(c[2:]) for c in plotted.columns if c.startswith("t_")]
    assert max(times) <= WINDOW_SECONDS


@pytest.mark.parametrize(
    ("figure_id", "row_column", "config_key"),
    [("G07", "coefficient", "mfcc.n_mfcc"), ("G08", "chroma_bin", "chroma.n_chroma")],
)
def test_time_frequency_matrices_have_the_configured_number_of_rows(
    figures: dict[str, dict[str, Path]],
    figure_id: str,
    row_column: str,
    config_key: str,
) -> None:
    from src.utils.config import load_config

    plotted = pd.read_csv(figures[figure_id][".csv"])
    expected = int(load_config("features").require("families." + config_key))
    assert len(plotted) == expected
    assert row_column in plotted.columns
    times = [float(c[2:]) for c in plotted.columns if c.startswith("t_")]
    assert times and max(times) <= WINDOW_SECONDS


def test_g07_shows_all_thirteen_coefficients_including_c0_and_c1(
    figures: dict[str, dict[str, Path]],
) -> None:
    """c0 and c1 are drawn as lines, but they are still in the data (see FE-06)."""
    plotted = pd.read_csv(figures["G07"][".csv"])
    assert list(plotted["coefficient"]) == list(range(len(plotted)))
    values = plotted[[c for c in plotted.columns if c.startswith("t_")]].to_numpy(float)
    # The reason for the split: c0/c1 do not share a range with c2-c12.
    rest_span = float(values[2:].max() - values[2:].min())
    assert float(values[:2].max()) > float(values[2:].max()) + rest_span * 0.2


def test_g09_covers_every_subband_of_the_configured_decomposition(
    figures: dict[str, dict[str, Path]],
) -> None:
    from src.utils.config import load_config

    plotted = pd.read_csv(figures["G09"][".csv"])
    level = int(load_config("features").require("families.dwt.level"))
    subbands = list(dict.fromkeys(plotted["subband"]))
    # One approximation plus `level` detail bands.
    assert len(subbands) == level + 1
    assert subbands[0].startswith("cA")
    assert float(plotted["time_sec"].max()) <= WINDOW_SECONDS
    # Each sub-band roughly halves in length going down the levels.
    lengths = [int((plotted["subband"] == name).sum()) for name in subbands]
    assert lengths == sorted(lengths), "sub-band lengths are not monotonic"
