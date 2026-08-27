"""FE-05, FE-06, G10 and the class overlays (Phase 42, gate T42.7).

T42.7 is a [TEST/MANUAL] gate -- a human opens the plots. These tests cover what
a human cannot check reliably by eye: that G10's bar heights come from the
registry rather than from a literal, that the separation ranking is ordered as
claimed, and that every promised file exists and is non-empty.

The one thing asserted hardest is the thing that would be invisible in the
figure: **the class-separation ranking must never be used to select features.**
It is computed over the whole corpus including future test folds, which is fine
for a picture and a leak for a model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.feature_extraction import distributions as dist
from src.feature_extraction import figures as fig
from src.feature_extraction import matrix as mx
from src.feature_extraction.registry import (
    EXPECTED_FAMILY_COUNTS,
    FAMILY_ORDER,
    FEATURE_NAMES,
    feature_names,
)

LOCKED_COMPOSITION = [24, 22, 39, 24, 24, 5]


def _fake_matrix(n_rows: int = 60) -> Any:
    """Two separable classes so the ranking has something real to find."""
    import pandas as pd

    rng = np.random.default_rng(42)
    labels = np.array([0, 1] * (n_rows // 2))
    data: dict[str, Any] = {
        "record_uid": ["R" + format(index, "03d") for index in range(n_rows)],
        "dataset_source": ["D1"] * n_rows,
        "binary_label": labels.astype(float),
        "binary_label_name": ["normal" if label == 0 else "abnormal" for label in labels],
        "use_in_supervised": [True] * n_rows,
        "duration_sec": rng.uniform(5.0, 30.0, n_rows),
    }
    for name in FEATURE_NAMES:
        data[name] = rng.normal(size=n_rows)
    # One feature is made strongly class-dependent; it must rank first.
    data[FEATURE_NAMES[3]] = rng.normal(size=n_rows) + 6.0 * labels
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# T42.3 / G10 -- the counts
# ---------------------------------------------------------------------------


def test_the_registry_holds_the_locked_composition():
    """G10's subject, checked at the source rather than off the picture."""
    counts = [len(feature_names(family)) for family in FAMILY_ORDER]
    assert counts == LOCKED_COMPOSITION
    assert [EXPECTED_FAMILY_COUNTS[family] for family in FAMILY_ORDER] == counts
    assert sum(counts) == len(FEATURE_NAMES) == 138


def test_g10_is_drawn_and_named(tmp_path: Any):
    path = fig.plot_family_counts(tmp_path / fig.G10_FILENAME)
    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.name == "feature_family_count_chart.png"


def test_g10_refuses_to_draw_a_registry_that_disagrees_with_config(monkeypatch: Any):
    """If the registry ever drifts from 24/22/39/24/24/5, G10 must not draw it.

    A chart that quietly renders 23/22/39/24/24/5 is worse than a crash: it is a
    wrong number in a thesis figure, and nothing else in the pipeline would
    notice at that point.
    """
    from src.feature_extraction import registry

    wrong = dict(registry.EXPECTED_FAMILY_COUNTS)
    wrong["time"] = 23
    monkeypatch.setattr(registry, "EXPECTED_FAMILY_COUNTS", wrong)
    with pytest.raises(ValueError, match="locked composition"):
        fig.plot_family_counts("unused.png")


# ---------------------------------------------------------------------------
# T42.1 / T42.4 -- the ranking
# ---------------------------------------------------------------------------


def test_the_ranking_puts_the_separable_feature_first():
    ranking = dist.rank_by_separation(_fake_matrix())
    assert ranking.iloc[0]["feature"] == FEATURE_NAMES[3]
    assert float(ranking.iloc[0]["abs_cohens_d"]) > 2.0


def test_the_ranking_covers_every_feature_and_is_ordered():
    ranking = dist.rank_by_separation(_fake_matrix())
    assert len(ranking) == len(FEATURE_NAMES)
    assert set(ranking["feature"]) == set(FEATURE_NAMES)
    assert list(ranking["rank"]) == list(range(1, len(FEATURE_NAMES) + 1))
    assert ranking["abs_cohens_d"].is_monotonic_decreasing


def test_cohens_d_sign_follows_the_class_means():
    ranking = dist.rank_by_separation(_fake_matrix()).set_index("feature")
    row = ranking.loc[FEATURE_NAMES[3]]
    assert float(row["mean_class1"]) > float(row["mean_class0"])
    assert float(row["cohens_d"]) > 0


def test_the_ranking_module_exposes_no_selector():
    """Rule 2 in code: there must be nothing here shaped like feature selection.

    ``rank_by_separation`` returns a table on purpose. If a helper ever appears
    that hands back a column list, it becomes the obvious thing to pass to a
    model -- fitted on the whole corpus, test folds included.
    """
    exported = set(dist.__all__)
    forbidden = {"select_features", "top_features", "selected_columns", "best_features"}
    assert not exported & forbidden


def test_a_dataset_without_two_classes_is_rejected():
    frame = _fake_matrix()
    frame["binary_label"] = 0.0
    with pytest.raises(ValueError, match="binary classes"):
        dist.rank_by_separation(frame)


def test_unlabelled_and_excluded_records_are_left_out():
    frame = _fake_matrix()
    frame.loc[0:9, "use_in_supervised"] = False
    frame.loc[10:13, "binary_label"] = np.nan
    subset = dist.labelled_subset(frame)
    assert len(subset) == len(frame) - 14


def test_fe05_writes_a_panel_per_top_feature(tmp_path: Any):
    written = dist.write_distribution_artifacts(_fake_matrix(), tmp_path)

    panels = sorted(written["FE-05-dir"].glob("*.png"))
    assert len(panels) == dist.TOP_N_PLOTS
    assert all(panel.stat().st_size > 0 for panel in panels)
    # Panels are named by rank so the directory sorts into ranking order.
    assert panels[0].name.startswith("01_")
    assert written["separation_csv"].is_file()
    assert written["overlays"].is_file()
    assert written["overlays"].name == dist.OVERLAY_FILENAME


def test_fe05_registers_a_file_not_a_directory(tmp_path: Any):
    """The evidence index verifies with is_file(); a directory row would pass
    that check while being empty, so FE-05 registers its manifest instead."""
    import pandas as pd

    written = dist.write_distribution_artifacts(_fake_matrix(), tmp_path)

    assert written["FE-05"].is_file()
    assert written["FE-05"].name == dist.FE05_MANIFEST

    manifest = pd.read_csv(written["FE-05"])
    assert len(manifest) == dist.TOP_N_PLOTS
    assert list(manifest["rank"]) == list(range(1, dist.TOP_N_PLOTS + 1))
    # Every row names a panel that actually exists.
    for row in manifest.itertuples(index=False):
        assert (written["FE-05-dir"] / row.filename).is_file(), row.filename


# ---------------------------------------------------------------------------
# T42.2 -- FE-06
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_fe06_mfcc_heatmap_is_drawn(tmp_path: Any):
    path = fig.plot_mfcc_heatmap(tmp_path / "mfcc_heatmap.png")
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.needs_data
def test_fe06_is_reproducible_from_the_same_record(tmp_path: Any):
    """The representative record is chosen by rule, so two runs pick the same one."""
    first = fig.plot_mfcc_heatmap(tmp_path / "a.png")
    second = fig.plot_mfcc_heatmap(tmp_path / "b.png")
    assert first.stat().st_size == second.stat().st_size


# ---------------------------------------------------------------------------
# the emitted artifacts (gate T42.7's automated half)
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_every_phase_42_artifact_exists_on_disk():
    directory = mx.matrix_path().parent
    if not directory.joinpath(dist.SEPARATION_FILENAME).is_file():
        pytest.skip("Phase 42 artifacts not built; run scripts/03_feature_reports.py")

    assert (directory / fig.FE_FIGURES["FE-06"]).stat().st_size > 0
    assert (directory / dist.OVERLAY_FILENAME).stat().st_size > 0
    assert (fig.diagrams_dir() / fig.G10_FILENAME).stat().st_size > 0

    panels = sorted((directory / dist.FE05_DIRNAME).glob("*.png"))
    assert len(panels) == dist.TOP_N_PLOTS


@pytest.mark.needs_data
def test_phase_42_artifacts_are_registered_in_the_evidence_index():
    from src.utils.evidence import read_evidence

    rows = {row["evidence_id"]: row for row in read_evidence()}
    for evidence_id in ("FE-05", "FE-06", "G10"):
        if evidence_id not in rows:
            pytest.skip("Phase 42 artifacts not registered yet")
        assert rows[evidence_id]["status"] == "ok", evidence_id
