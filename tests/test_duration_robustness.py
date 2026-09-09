"""T73.7 -- the EXP-E2 duration robustness gate.

The gate: **duration bands match the T18.3 assignments, the shortest recordings
are analysed separately, and T21 and G31 exist.**

"Match the T18.3 assignments" is checked by calling the same function T18 used
and comparing, not by re-deriving the cut from the thresholds -- re-deriving it
would pass even if the two had drifted apart, which is the failure the clause is
there to catch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/09_robustness_analysis"
RUN_DIR = "EXP-E2"

N_CORPUS = 7536
#: Measured 2026-08-31 from metadata_master under the configured 5 s / 20 s cuts.
BAND_COUNTS = {"short": 339, "medium": 3234, "long": 3963}
SHORTEST_SECONDS = 0.76325


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(*parts: str) -> Any:
    import pandas as pd

    path = _root().joinpath(*parts)
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/24_duration_robustness.py")
    return pd.read_csv(path)


def _section_csv(name: str) -> Any:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / SECTION / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/24_duration_robustness.py")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# clause 1 -- the bands are T18.3's bands
# ---------------------------------------------------------------------------


def test_the_bands_are_the_same_assignment_t18_made() -> None:
    """T73.1 / T73.7 -- same function, not a re-derived cut that could drift."""
    import pandas as pd

    from src.data_loader.summaries import assign_duration_bands
    from src.evaluation.duration import DURATION_BANDS, duration_bands
    from src.utils.evidence import PROJECT_ROOT

    frame = duration_bands()
    assert len(frame) == N_CORPUS
    assert frame["record_uid"].is_unique
    assert set(frame["duration_band"]) == set(DURATION_BANDS)
    assert frame["duration_band"].value_counts().to_dict() == BAND_COUNTS

    reference = assign_duration_bands(
        pd.read_csv(
            PROJECT_ROOT / "outputs" / "01_dataset_audit" / "metadata_master.csv", low_memory=False
        )
    )
    merged = frame.merge(
        reference[["record_uid", "duration_band"]], on="record_uid", suffixes=("", "_ref")
    )
    assert len(merged) == N_CORPUS
    assert (merged["duration_band"] == merged["duration_band_ref"]).all()


def test_the_bands_respect_the_configured_thresholds() -> None:
    """A band is defined by its cut, and the cut lives in configs/signal.yaml."""
    from src.data_loader.integrity import load_thresholds
    from src.evaluation.duration import duration_bands

    thresholds = load_thresholds()
    frame = duration_bands()
    short = frame[frame["duration_band"] == "short"]
    medium = frame[frame["duration_band"] == "medium"]
    long = frame[frame["duration_band"] == "long"]
    assert (short["duration_sec"] < thresholds.short_below).all()
    assert (medium["duration_sec"] >= thresholds.short_below).all()
    assert (medium["duration_sec"] <= thresholds.long_above).all()
    assert (long["duration_sec"] > thresholds.long_above).all()


def test_duration_is_confounded_with_corpus_and_that_is_why_truncation_exists() -> None:
    """The caveat T21 carries, asserted rather than trusted to stay true.

    Every recording under 5 s is PASCAL. If that ever stopped being the case the
    caveat printed on T21 would be wrong, which is worse than not printing one.
    """
    from src.evaluation.duration import duration_bands

    frame = duration_bands()
    short_sources = set(frame[frame["duration_band"] == "short"]["dataset_source"])
    assert short_sources <= {"D2", "D3"}, sorted(short_sources)
    assert "D1" not in short_sources
    assert "D4" not in short_sources


# ---------------------------------------------------------------------------
# clause 2 -- the shortest recordings are analysed separately
# ---------------------------------------------------------------------------


def test_the_shortest_recordings_are_analysed_in_their_own_right() -> None:
    """T73.3 -- PASCAL B reaches 0.763 s, and that is measured, not assumed."""
    diagnostics = _csv(RUN_DIR, "short_record_diagnostics.csv")
    assert len(diagnostics) >= 10
    assert (diagnostics["duration_sec"] < 5.0).all()
    shortest = diagnostics.iloc[0]
    assert np.isclose(float(shortest["duration_sec"]), SHORTEST_SECONDS, atol=1e-4)

    # The finding is NOT that extraction breaks -- it does not, anywhere.
    assert int(diagnostics["n_nan_features"].sum()) == 0
    # It is that a window this short cannot contain enough heart sounds for a
    # rate to mean anything. env_peak_rate counts S1 and S2 at ~2.4/s.
    assert float(shortest["implied_n_heart_sounds"]) < 5.0
    assert diagnostics["n_samples"].min() > 0


def test_the_feature_shift_between_short_and_long_records_is_quantified() -> None:
    """Naming which features move beats asserting that some must."""
    shifts = _csv(RUN_DIR, "short_vs_long_feature_shift.csv")
    assert len(shifts) == 138
    assert set(shifts.columns) >= {"feature", "family", "standardized_shift"}
    finite = np.asarray(shifts["standardized_shift"], dtype=float)
    assert np.isfinite(finite).any()
    # Sorted by magnitude, so the top row is the feature that moves most.
    magnitudes = np.abs(finite[np.isfinite(finite)])
    assert magnitudes[0] == magnitudes.max()


# ---------------------------------------------------------------------------
# clause 3 -- T21 and G31 exist and agree
# ---------------------------------------------------------------------------


def test_t21_carries_both_kinds_of_evidence_labelled() -> None:
    """T73.5 -- observational bands and the truncation study, distinguishable."""
    t21 = _section_csv("T21_duration_robustness.csv")
    assert set(t21["analysis"]) == {"observational_bands", "interventional_truncation"}

    observational = t21[t21["analysis"] == "observational_bands"]
    assert len(set(observational["task"])) >= 3, sorted(set(observational["task"]))
    assert {"short", "medium", "long"} & set(observational["group"])

    interventional = t21[t21["analysis"] == "interventional_truncation"]
    levels = set(interventional["group"])
    for level in ("10 s clip", "5 s clip", "3 s clip"):
        assert level in levels, sorted(levels)
    assert any(g.startswith("full") for g in levels), sorted(levels)


def test_g31_was_drawn_from_the_same_numbers_as_t21() -> None:
    """T73.6 -- the figure exists and its CSV is T21's frame."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    figures = PROJECT_ROOT / "outputs" / "13_figures_diagrams"
    csv_path = figures / "G31_duration_wise_performance.csv"
    if not csv_path.is_file():
        pytest.skip("G31 not generated; run scripts/24_duration_robustness.py")
    assert (figures / "G31_duration_wise_performance.png").is_file()

    plotted = pd.read_csv(csv_path)
    t21 = _section_csv("T21_duration_robustness.csv")
    keys = ["analysis", "run", "model_id", "group"]
    merged = plotted.merge(t21, on=keys, suffixes=("_g", "_t"))
    assert len(merged) == len(plotted)
    assert np.allclose(
        np.asarray(merged["balanced_accuracy_mean_g"], dtype=float),
        np.asarray(merged["balanced_accuracy_mean_t"], dtype=float),
        equal_nan=True,
    )


def test_the_truncation_study_holds_length_as_the_only_variable() -> None:
    """T73.4 -- four lengths, the same records, and a control that is untouched."""
    truncation = _csv(RUN_DIR, "truncation_metrics.csv")
    assert len(truncation) == 4, sorted(truncation["clip_seconds"])
    assert int(truncation["n_nan_features"].sum()) == 0

    indexed = truncation.set_index("clip_seconds")
    # Every clip is at most its nominal length, and the control is the longest.
    for seconds in (10.0, 5.0, 3.0):
        assert float(indexed.loc[seconds, "realised_seconds_mean"]) <= seconds + 1e-6
    assert float(indexed.loc[np.inf, "realised_seconds_mean"]) > 10.0
    # Every level scores the same records.
    assert truncation["n_units"].nunique() == 1


def test_no_duration_metric_is_suspiciously_perfect() -> None:
    """The standing near-perfect rule on the subgroup table."""
    t21 = _section_csv("T21_duration_robustness.csv")
    reported = t21[t21["reported"]]
    for column in ("balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"):
        if column not in reported.columns:
            continue
        values = np.asarray(reported[column], dtype=float)
        if not np.isfinite(values).any():
            continue
        worst = float(np.nanmax(values))
        assert worst < 0.99, column + " reaches " + format(worst, ".4f")


# ---------------------------------------------------------------------------
# needs the truncation parquet -- skips on CI
# ---------------------------------------------------------------------------


def test_the_untruncated_control_reproduces_the_stored_feature_matrix() -> None:
    """As in the noise sweep: without this, a 10 s drop could be a pipeline drift."""
    import pandas as pd

    from src.feature_extraction.matrix import load_matrix
    from src.feature_extraction.registry import feature_names

    path = _root() / RUN_DIR / "truncation_features.parquet"
    if not path.is_file():
        pytest.skip(str(path) + " is gitignored; run --stage truncation locally")

    frame = pd.read_parquet(path)
    control = frame[~np.isfinite(frame["clip_seconds"])].set_index("record_uid").sort_index()
    assert len(control) > 0

    names = list(feature_names())
    stored = load_matrix().set_index("record_uid").loc[control.index, names]
    difference = np.abs(control[names].to_numpy(dtype=float) - stored.to_numpy(dtype=float))
    assert float(np.nanmax(difference)) == 0.0


def test_a_clip_is_a_window_of_the_original_not_always_its_head() -> None:
    """Always clipping the head would confound length with position."""
    from src.evaluation.duration import truncate

    signal = np.arange(20000, dtype=float)
    starts = set()
    for seed in range(8):
        rng = np.random.default_rng([42, seed])
        clip = truncate(signal, 2000, 3.0, rng)
        assert clip.size == 6000
        starts.add(float(clip[0]))
    assert len(starts) > 1, "every clip started at the same sample"
    # And it must be reproducible for a given seed.
    first = truncate(signal, 2000, 3.0, np.random.default_rng([42, 0]))
    again = truncate(signal, 2000, 3.0, np.random.default_rng([42, 0]))
    assert np.array_equal(first, again)
    # A clip longer than the signal returns the signal, not an error or padding.
    whole = truncate(signal, 2000, 999.0, np.random.default_rng([42]))
    assert np.array_equal(whole, signal)
