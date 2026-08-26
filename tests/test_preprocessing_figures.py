"""Preprocessing figure gate (Phase 28, supporting T28.7).

T28.7 is a [TEST/MANUAL] check -- a person opens the six figures and looks at
them. These tests cover the parts of that check a person cannot reliably do by
eye: that PP-05 and PP-06 were drawn on **the same** colour scale (two viridis
images look plausible side by side whether or not they share one), that the
record selection is deterministic rather than whatever the merge happened to
order first, and that every figure is a real 300 dpi image rather than a
zero-byte file with the right name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.preprocessing import figures as fig
from src.reporting import plot_style as style

# ===========================================================================
# T28.6 -- the shared style module
# ===========================================================================


def test_style_is_serif_colourblind_safe_and_300dpi() -> None:
    assert style.DPI == 300
    assert style.RC_PARAMS["savefig.dpi"] == 300
    assert style.RC_PARAMS["font.family"] == "serif"
    # DejaVu Serif ships with matplotlib; anything else may silently fall back.
    assert style.RC_PARAMS["font.serif"][0] == "DejaVu Serif"
    assert style.SEQUENTIAL_CMAP == "viridis"
    assert len(style.OKABE_ITO) == 8
    assert len(set(style.OKABE_ITO)) == 8


def test_apply_style_installs_the_palette_and_agg() -> None:
    import matplotlib

    style.apply_style()
    import matplotlib.pyplot as plt

    assert matplotlib.get_backend().lower() == "agg"
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle[:2] == list(style.OKABE_ITO[:2])
    assert plt.rcParams["font.family"] == ["serif"]


def test_styled_context_restores_the_previous_settings() -> None:
    import matplotlib.pyplot as plt

    style.apply_style()
    before = plt.rcParams["font.size"]
    with style.styled(**{"font.size": 22}):
        assert plt.rcParams["font.size"] == 22
    assert plt.rcParams["font.size"] == before


def test_class_color_wraps() -> None:
    assert style.class_color(0) == style.OKABE_ITO[0]
    assert style.class_color(8) == style.OKABE_ITO[0]


def test_source_stamp_is_wrapped_not_one_long_line() -> None:
    """A single long stamp stretches the canvas under bbox_inches='tight'."""
    import matplotlib.pyplot as plt

    style.apply_style()
    figure, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot([0, 1], [0, 1])
    style.annotate_source(figure, "line one\nline two")
    texts = [t.get_text() for t in figure.texts]
    plt.close(figure)

    assert texts == ["line one\nline two"]


# ===========================================================================
# selection rules
# ===========================================================================


@pytest.mark.needs_data
def test_record_selection_is_deterministic() -> None:
    first = fig.select_records()
    second = fig.select_records()
    assert {k: v["record_uid"] for k, v in first.items()} == {
        k: v["record_uid"] for k, v in second.items()
    }


@pytest.mark.needs_data
def test_selected_records_satisfy_their_stated_rules() -> None:
    chosen = fig.select_records()

    assert chosen["normal"]["binary_label_name"] == "normal"
    assert chosen["abnormal"]["binary_label_name"] == "abnormal"
    for key in ("normal", "abnormal"):
        assert chosen[key]["dataset_source"] == "D1"
        assert not chosen[key]["is_low_quality"]
        assert not chosen[key]["is_duplicate"]

    drifty = chosen["drifty"]
    # The legibility cap: above it the filtered panel is a flat line (see the
    # module docstring). Below zero there would be no drift to show.
    assert 0.0 < float(drifty["drift_ratio_db"]) <= fig.PP03_DRIFT_CAP_DB
    assert 8.0 <= float(drifty["duration_sec"]) <= 20.0


# ===========================================================================
# T28.5 -- the shared colour scale
# ===========================================================================


@pytest.mark.needs_data
def test_spectrogram_pair_shares_one_colour_scale(tmp_path: Path) -> None:
    chosen = fig.select_records()
    normal, abnormal, scale = fig.plot_spectrogram_pair(
        chosen["normal"],
        chosen["abnormal"],
        tmp_path / "normal_spectrogram.png",
        tmp_path / "abnormal_spectrogram.png",
    )

    assert normal.is_file() and abnormal.is_file()
    assert scale.vmax_db > scale.vmin_db
    assert scale.vmax_db - scale.vmin_db == pytest.approx(
        fig.SPECTROGRAM_DYNAMIC_RANGE_DB, abs=1e-6
    )
    # The label is what a reader compares between the two printed figures, so it
    # must state the numbers rather than merely claim a shared scale.
    assert format(scale.vmin_db, ".1f") in scale.label()
    assert format(scale.vmax_db, ".1f") in scale.label()


# ===========================================================================
# T28.1 - T28.5 -- the artifacts themselves
# ===========================================================================


@pytest.mark.needs_data
@pytest.mark.slow
def test_all_six_figures_are_written(tmp_path: Path) -> None:
    from PIL import Image

    paths = fig.generate_all(tmp_path)

    assert sorted(paths) == ["PP-01", "PP-02", "PP-03", "PP-04", "PP-05", "PP-06"]
    for artifact, path in paths.items():
        assert path.name == fig.PP_FIGURES[artifact]
        assert path.is_file(), artifact
        assert path.stat().st_size > 20_000, artifact

        with Image.open(path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (0, 0))
        # 300 dpi at the project's page widths.
        assert dpi[0] == pytest.approx(300, abs=1), artifact
        assert width > 1500 and height > 500, (artifact, width, height)


@pytest.mark.needs_data
def test_pp_figures_exist_in_the_real_output_directory(paths_config: Any) -> None:
    """The committed artifacts, not a tmp_path copy of them."""
    directory = Path(paths_config.require("outputs.preprocessing"))
    for artifact, name in fig.PP_FIGURES.items():
        path = directory / name
        assert path.is_file(), artifact + " (" + name + ") has not been generated"
        assert path.stat().st_size > 20_000, artifact
