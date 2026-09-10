"""T84.7 -- the cardiac-cycle and segmentation gate.

The gate: **the cycle table covers all segmented CirCor recordings, PASCAL A
timing is included, and the annotated-waveform figure renders real
segmentation.**

"Real segmentation" is checked against the corpus `.tsv` itself, not against the
figure's own CSV -- a figure drawn from a synthetic or a re-detected overlay
would satisfy a self-consistency check and fail this one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/10_robustness"

#: Audited 2026-08-22 and pinned in CLAUDE.md.
N_CIRCOR_RECORDINGS = 3163
#: One recording has no .tsv (50782_MV_1); the rest are segmented.
N_CIRCOR_SEGMENTED = 3162
#: set_a_timing.csv: 390 annotations over 21 recordings.
N_PASCAL_ANNOTATED = 21


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(name: str) -> Any:
    import pandas as pd

    path = _root() / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/36_cycle_analysis.py")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# clause 1 -- every segmented CirCor recording is covered
# ---------------------------------------------------------------------------


def test_the_cycle_table_covers_every_circor_recording() -> None:
    cycles = _csv("segmentation_cycle_statistics.csv")
    circor = cycles[cycles["dataset"].str.startswith("D4")]
    assert len(circor) == N_CIRCOR_RECORDINGS, len(circor)
    assert circor["record_uid"].is_unique
    assert int(circor["has_segmentation"].sum()) == N_CIRCOR_SEGMENTED

    segmented = circor[circor["has_segmentation"].astype(bool)]
    assert segmented["n_cycles"].notna().all()
    assert (segmented["n_cycles"] > 0).all()
    for phase in ("s1", "systole", "s2", "diastole"):
        assert segmented["mean_" + phase + "_sec"].notna().all(), phase
        assert (segmented["mean_" + phase + "_sec"] > 0).all(), phase

    # The unsegmented recording is present with a stated reason, not dropped.
    absent = circor[~circor["has_segmentation"].astype(bool)]
    assert len(absent) == N_CIRCOR_RECORDINGS - N_CIRCOR_SEGMENTED
    assert absent["segmentation_issues"].str.len().min() > 0


def test_the_cycle_counts_agree_with_the_phase_15_audit() -> None:
    """Two independent passes over the same TSVs must not disagree about cycles.

    The Phase 15 summary counts S1 segments and so does this module; if they
    differ, one of them is parsing the files differently.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    audit_path = PROJECT_ROOT / "outputs/01_dataset_audit/circor_segmentation_summary.csv"
    if not audit_path.is_file():
        pytest.skip("the Phase 15 segmentation summary is unavailable")
    audit = pd.read_csv(audit_path).set_index("record_uid")
    cycles = _csv("segmentation_cycle_statistics.csv").set_index("record_uid")
    shared = audit.index.intersection(cycles.index)
    assert len(shared) >= N_CIRCOR_SEGMENTED

    mine = cycles.loc[shared, "n_cycles"].fillna(0).to_numpy(dtype=float)
    theirs = audit.loc[shared, "n_cycles"].fillna(0).to_numpy(dtype=float)
    assert np.array_equal(mine, theirs), "the two passes disagree about cycle counts"


def test_a_phase_mean_is_over_segments_not_a_total_divided_by_a_count() -> None:
    """The distinction the module was written for, verified on real data.

    Wherever the two differ the recording's annotation starts or ends mid-cycle,
    which is most of them. If they agreed everywhere, the mean would have been
    computed the lazy way.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    audit_path = PROJECT_ROOT / "outputs/01_dataset_audit/circor_segmentation_summary.csv"
    if not audit_path.is_file():
        pytest.skip("the Phase 15 segmentation summary is unavailable")
    audit = pd.read_csv(audit_path).set_index("record_uid")
    cycles = _csv("segmentation_cycle_statistics.csv").set_index("record_uid")
    shared = audit.index.intersection(cycles.index)

    naive = audit.loc[shared, "systole_sec"] / audit.loc[shared, "n_cycles"].replace(0, np.nan)
    proper = cycles.loc[shared, "mean_systole_sec"]
    mask = naive.notna() & proper.notna()
    differ = (~np.isclose(naive[mask], proper[mask], rtol=1e-6)).sum()
    assert differ > 0, (
        "every recording's per-segment mean equalled total/count, which means "
        "the mean was not computed per segment"
    )


def test_the_systole_to_diastole_ratio_is_a_per_cycle_mean() -> None:
    """It must not equal the ratio of the two totals -- see the module docstring."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    audit_path = PROJECT_ROOT / "outputs/01_dataset_audit/circor_segmentation_summary.csv"
    if not audit_path.is_file():
        pytest.skip("the Phase 15 segmentation summary is unavailable")
    audit = pd.read_csv(audit_path).set_index("record_uid")
    cycles = _csv("segmentation_cycle_statistics.csv").set_index("record_uid")
    shared = audit.index.intersection(cycles.index)

    naive = audit.loc[shared, "systole_sec"] / audit.loc[shared, "diastole_sec"].replace(
        0, np.nan
    )
    proper = cycles.loc[shared, "systole_to_diastole_ratio"]
    mask = naive.notna() & proper.notna()
    assert (~np.isclose(naive[mask], proper[mask], rtol=1e-6)).sum() > 0
    assert (cycles["n_paired_cycles"].fillna(0) >= 0).all()
    # A ratio without paired cycles behind it must not exist.
    ratios = cycles[cycles["systole_to_diastole_ratio"].notna()]
    assert (ratios["n_paired_cycles"] > 0).all()


# ---------------------------------------------------------------------------
# clause 2 -- PASCAL A timing is included, with its limits stated
# ---------------------------------------------------------------------------


def test_pascal_a_timing_is_included_and_its_missing_columns_stay_missing() -> None:
    cycles = _csv("segmentation_cycle_statistics.csv")
    pascal = cycles[cycles["dataset"].str.startswith("D2")]
    assert len(pascal) == N_PASCAL_ANNOTATED, len(pascal)
    assert pascal["timing_kind"].str.contains("instants").all()

    # Systole and diastole are derivable from the instants; S1/S2 durations are not.
    assert pascal["mean_systole_sec"].notna().all()
    assert pascal["mean_diastole_sec"].notna().all()
    assert (pascal["mean_systole_sec"] > 0).all()
    assert pascal["mean_s1_sec"].isna().all(), (
        "S1 has no duration in an instant annotation and must not be imputed"
    )
    assert pascal["mean_s2_sec"].isna().all()
    assert pascal["n_cycles"].notna().all()


def test_the_coverage_summary_records_the_corpora_with_no_annotation() -> None:
    """An absent row would read as an oversight; a zero row is a fact."""
    coverage = _csv("segmentation_coverage_summary.csv")
    datasets = set(coverage["dataset"])
    assert any(d.startswith("D4") for d in datasets)
    assert any(d.startswith("D2") for d in datasets)
    assert any(d.startswith("D3") for d in datasets), "set_b must be recorded as unannotated"
    assert any(d.startswith("D1") for d in datasets), "PhysioNet must be recorded as unannotated"

    none = coverage[coverage["timing_kind"] == "none"]
    assert len(none) == 2
    assert (none["n_with_annotation"] == 0).all()
    assert none["note"].str.len().min() > 20


# ---------------------------------------------------------------------------
# clause 3 -- the figure renders REAL segmentation
# ---------------------------------------------------------------------------


def test_the_annotated_waveform_bands_are_the_corpus_tsv_rows() -> None:
    """Checked against the corpus file, not against the figure's own CSV."""
    import json

    from src.reporting.segmentation import read_segmentation, resolve_sources

    root = _root()
    png = root / "segmentation_annotated_waveform.png"
    if not png.is_file():
        pytest.skip(str(png) + " does not exist; run scripts/36_cycle_analysis.py")
    assert png.stat().st_size > 10_000

    meta = json.loads(
        (root / "segmentation_annotated_waveform.meta.json").read_text(encoding="utf-8")
    )
    bands = _csv("segmentation_annotated_waveform_segments.csv")

    try:
        _wav, tsv, _origin = resolve_sources(str(meta["record_id"]))
    except FileNotFoundError:
        pytest.skip("neither the corpus nor the committed copy of the sample is present")
    truth = read_segmentation(tsv)

    assert len(bands) == len(truth) == meta["n_segments_total"]
    assert np.allclose(bands["start_sec"], [s["start"] for s in truth])
    assert np.allclose(bands["end_sec"], [s["end"] for s in truth])
    assert list(bands["state"]) == [s["label"] for s in truth]


def test_all_four_cardiac_phases_are_drawn_not_only_two() -> None:
    """The first version keyed its colours on the display name and drew S1/S2 only.

    A lookup miss on a colour map is a missing band, not an error, so the bug was
    invisible until someone looked at the picture. This asserts every phase that
    is present in the window actually has a colour.
    """
    from src.reporting.cycle_report import PHASE_COLOURS

    bands = _csv("segmentation_annotated_waveform_segments.csv")
    window = bands[bands["in_window"].astype(bool)]
    assert len(window) >= 8, "the window must contain at least two complete cycles"

    keys = set(window["phase_key"]) - {"unannotated"}
    assert keys == {"s1", "systole", "s2", "diastole"}, keys
    for key in keys:
        assert key in PHASE_COLOURS, key + " has no colour and would be drawn blank"


def test_the_plotted_samples_are_real_audio_at_the_right_rate() -> None:
    import json

    root = _root()
    meta_path = root / "segmentation_annotated_waveform.meta.json"
    if not meta_path.is_file():
        pytest.skip(str(meta_path) + " does not exist; run scripts/36_cycle_analysis.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    samples = _csv("segmentation_annotated_waveform_samples.csv")

    assert meta["sample_rate_hz"] == 4000, "CirCor is 4 kHz"
    expected = round(
        (meta["window_end_sec"] - meta["window_start_sec"]) * meta["sample_rate_hz"]
    )
    assert abs(len(samples) - expected) <= 1
    assert samples["amplitude"].std() > 0, "a flat trace is not a recording"
    assert np.isfinite(samples["amplitude"]).all()


# ---------------------------------------------------------------------------
# T84.3 and T84.4
# ---------------------------------------------------------------------------


def test_annotation_quality_was_tested_against_confidence_and_against_the_errors() -> None:
    correlations = _csv("segmentation_confidence_correlation.csv")
    assert len(correlations)
    assert set(correlations["target"]) >= {"confidence", "is_error"}
    assert "annotated_fraction" in set(correlations["predictor"])
    assert correlations["n"].notna().all()
    assert (correlations["n"] > 0).all()
    assert correlations["n_description"].str.len().min() > 10
    assert (correlations["rho"].abs() <= 1.0 + 1e-9).all()
    assert correlations["test"].str.contains("Spearman").all()


def test_the_class_comparison_says_it_is_context_and_not_a_result() -> None:
    """T84.4 -- no cycle timing is among the 138 features, and it must say so."""
    comparison = _csv("segmentation_timing_by_class.csv")
    assert len(comparison)
    assert (~comparison["is_a_feature"].astype(bool)).all()
    assert comparison["caveat"].str.contains("CONTEXT, NOT A RESULT").all()
    assert comparison["caveat"].str.contains("138 features").all()

    groupings = set(comparison["grouping"])
    assert "murmur present vs absent" in groupings
    assert "clinical outcome normal vs abnormal" in groupings

    from src.feature_extraction.registry import feature_names

    names = set(feature_names())
    for column in ("mean_systole_sec", "mean_diastole_sec", "systole_to_diastole_ratio"):
        assert column not in names, column + " is a feature after all; the caveat is wrong"


def test_the_seg01_table_exists_in_every_format() -> None:
    root = _root()
    slug = "SEG-01_segmentation_cycle_statistics"
    if not (root / (slug + ".csv")).is_file():
        pytest.skip(slug + " does not exist; run scripts/36_cycle_analysis.py")
    for suffix in (".csv", ".docx", ".tex", ".md", ".meta.json"):
        assert (root / (slug + suffix)).is_file(), slug + suffix

    import json

    notes = " ".join(
        json.loads((root / (slug + ".meta.json")).read_text(encoding="utf-8")).get("notes", [])
    )
    assert "CONTEXT, NOT A RESULT" in notes
    assert "PER-CYCLE MEAN" in notes
    assert "T84.3" in notes
    assert "screening" in notes.lower()


def test_the_waveform_figure_is_not_in_the_numbered_figure_registry() -> None:
    """Deliberate: the 35 G-figures are a counted set and this is not one of them."""
    from src.reporting.graphs import read_registry

    ids = {row["figure_id"] for row in read_registry()}
    assert "SEG-02" not in ids
    assert not any(str(i).startswith("SEG") for i in ids)

    from src.utils.evidence import read_evidence

    rows = {row["evidence_id"] for row in read_evidence()}
    assert "SEG-02" in rows, "the figure must still be registered as evidence"
