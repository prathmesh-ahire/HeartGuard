"""T70.7 -- the CirCor auscultation-location gate (EXP-C3).

The gate in one sentence: **per-location metrics exist for AV, PV, TV and MV,
and Phc (n=4) is flagged as statistically uninformative rather than reported as
a result.**

Split the usual way. Everything reading a committed CSV runs on CI; the two
tests that need a ``predictions.parquet`` skip there, because ``*.parquet`` is
gitignored and CI has never seen one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SECTION = "outputs/08_circor_external_validation"
RUNS = ("EXP-C1-three_class", "EXP-C1-two_class", "EXP-C2")
CIRCOR_MODELS = ("M3", "M4", "M5", "M6", "M7")

#: From configs/experiments.yaml EXP-C3. Asserted, not trusted: the whole point
#: of the gate is that Phc is four recordings.
DECLARED_COUNTS = {"AV": 800, "PV": 766, "TV": 732, "MV": 861, "Phc": 4}


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(*parts: str) -> Any:
    import pandas as pd

    path = _root().joinpath(*parts)
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/21_location_analysis.py")
    return pd.read_csv(path)


def _parquet(*parts: str) -> Any:
    import pandas as pd

    path = _root().joinpath(*parts)
    if not path.is_file():
        pytest.skip(str(path) + " is gitignored; run the experiment locally")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# the location vocabulary itself
# ---------------------------------------------------------------------------


def test_the_corpus_holds_exactly_the_five_declared_locations() -> None:
    """The counts EXP-C3 is configured around, checked against the audit."""
    from src.evaluation.location import location_table

    table = location_table()
    assert len(table) == 3163
    assert table["record_uid"].is_unique
    counts = table["location"].value_counts().to_dict()
    assert counts == DECLARED_COUNTS, counts
    assert table["patient_id"].nunique() == 942


def test_phc_is_the_only_location_below_the_reportable_floor() -> None:
    """T70.4 -- the exclusion is a rule about size, not a hand-picked name."""
    from src.evaluation.location import (
        EXCLUDED_LOCATIONS,
        MIN_RECORDS_PER_LOCATION,
        REPORTED_LOCATIONS,
    )

    small = {k for k, v in DECLARED_COUNTS.items() if v < MIN_RECORDS_PER_LOCATION}
    assert small == set(EXCLUDED_LOCATIONS) == {"Phc"}
    assert set(REPORTED_LOCATIONS) == set(DECLARED_COUNTS) - small
    # Four recordings over five folds is under one per fold, which is the reason
    # the flag exists rather than a wide interval.
    assert DECLARED_COUNTS["Phc"] < 5


def test_the_location_is_cross_checked_against_the_uid() -> None:
    """43 recordings carry a repeat suffix a naive split misreads as the location.

    ``D4_training_data_49748_AV_2`` is the patient's second aortic recording. A
    ``rsplit("_", 1)`` reads its location as ``2``. The parser must handle that,
    and it must still be checking the two sources against each other.
    """
    from src.evaluation.location import _UID, location_table

    table = location_table()
    repeats = [uid for uid in table["record_uid"] if uid.rsplit("_", 1)[-1].isdigit()]
    assert len(repeats) == 43, len(repeats)
    for uid in repeats:
        match = _UID.match(uid)
        assert match is not None, uid
        assert match.group("location") in DECLARED_COUNTS, uid

    # And the check is live: a corrupted location must raise, not be resolved.
    import pandas as pd

    from src.evaluation.location import LocationError, attach_locations

    frame = pd.DataFrame({"record_uid": ["D4_training_data_00000_AV"], "model_id": ["M3"]})
    with pytest.raises(LocationError, match="no auscultation location"):
        attach_locations(frame)


# ---------------------------------------------------------------------------
# T22 -- the gate proper
# ---------------------------------------------------------------------------


def test_t22_reports_every_valve_location_for_every_model_and_run() -> None:
    """T70.7 clause 1 -- AV, PV, TV and MV are all present and all reportable."""
    from src.evaluation.location import REPORTED_LOCATIONS

    t22 = _csv("T22_auscultation_location_analysis.csv")
    assert set(t22["run"]) == set(RUNS), sorted(set(t22["run"]))
    for run in RUNS:
        block = t22[t22["run"] == run]
        for location in REPORTED_LOCATIONS:
            cell = block[block["location"] == location]
            assert set(cell["model_id"]) == set(CIRCOR_MODELS), run + " " + location
            assert cell["reported"].all(), run + " " + location + " is not marked reportable"
            assert (cell["n_folds"] == 5).all()


def test_t22_carries_phc_flagged_rather_than_dropping_it() -> None:
    """T70.7 clause 2 -- flagged, with a stated reason, never quoted."""
    t22 = _csv("T22_auscultation_location_analysis.csv")
    phc = t22[t22["location"] == "Phc"]
    assert len(phc) > 0, "Phc was dropped; the gate requires it flagged, not removed"
    assert not phc["reported"].any()
    assert (phc["n_records_corpus"] == 4).all()
    reasons = phc["exclusion_reason"].astype(str)
    assert (reasons.str.len() > 0).all()
    assert reasons.str.contains("excluded").all()
    # Every reportable row must carry an EMPTY reason, or "flagged" would mean
    # nothing -- a reason on every row is the same as a reason on none.
    reported = t22[t22["reported"]]
    assert reported["exclusion_reason"].isna().all() or (
        reported["exclusion_reason"].astype(str).str.strip().eq("").all()
    )


def test_no_reported_location_metric_is_suspiciously_perfect() -> None:
    """The standing near-perfect rule, applied to the subgroup table.

    Checked on the metrics a degenerate predictor CANNOT inflate. Specificity
    is deliberately not among them: M3 reaches 0.9949 at PV, and that was
    investigated before it was recorded. It predicts ``Present`` for 5.0-6.3% of
    recordings at every location against a true prevalence of ~20.1%, so a
    near-perfect specificity is the arithmetic consequence of rarely using the
    positive class, not a leak. The same phenomenon was cleared at corpus level
    in Phase 68.

    So the rule here is stricter than a flat threshold, not looser: balanced
    accuracy, macro-F1 and accuracy are capped, AND any specificity at or above
    0.99 must be accompanied by a sensitivity below 0.5. A genuine leak lifts
    both sides at once, and that is what this would catch.
    """
    t22 = _csv("T22_auscultation_location_analysis.csv")
    reported = t22[t22["reported"]]
    assert len(reported) > 0

    for column in ("balanced_accuracy_mean", "macro_f1_mean", "accuracy_mean"):
        if column not in reported.columns:
            continue
        values = np.asarray(reported[column], dtype=float)
        if not np.isfinite(values).any():
            continue
        worst = float(np.nanmax(values))
        assert worst < 0.99, column + " reaches " + format(worst, ".4f")

    specificity = np.asarray(reported["specificity_mean"], dtype=float)
    sensitivity = np.asarray(reported["sensitivity_mean"], dtype=float)
    suspicious = np.isfinite(specificity) & (specificity >= 0.99)
    for index in np.flatnonzero(suspicious):
        row = reported.iloc[int(index)]
        assert sensitivity[index] < 0.5, (
            str(row["run"])
            + " "
            + str(row["model_id"])
            + " "
            + str(row["location"])
            + ": specificity "
            + format(float(specificity[index]), ".4f")
            + " WITH sensitivity "
            + format(float(sensitivity[index]), ".4f")
            + " -- both sides high is the leakage signature, investigate before "
            "recording this"
        )

    phc = t22[t22["location"] == "Phc"]
    perfect = np.nanmax(np.asarray(phc["balanced_accuracy_mean"], dtype=float))
    assert perfect >= 0.8, (
        "Phc no longer produces an implausible number; if that is real, this "
        "gate's premise changed and it must be revisited deliberately"
    )


def test_g32_was_drawn_from_the_same_numbers_as_t22() -> None:
    """T70.6 -- the figure's CSV is the table's frame, not a parallel derivation."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    figures = PROJECT_ROOT / "outputs" / "13_figures_diagrams"
    csv_path = figures / "G32_location_performance.csv"
    if not csv_path.is_file():
        pytest.skip("G32 not generated; run scripts/21_location_analysis.py")
    assert (figures / "G32_location_performance.png").is_file()

    plotted = pd.read_csv(csv_path)
    t22 = _csv("T22_auscultation_location_analysis.csv")
    keys = ["run", "model_id", "location"]
    merged = plotted.merge(t22, on=keys, suffixes=("_g", "_t"))
    assert len(merged) == len(plotted)
    assert np.allclose(
        np.asarray(merged["balanced_accuracy_mean_g"], dtype=float),
        np.asarray(merged["balanced_accuracy_mean_t"], dtype=float),
        equal_nan=True,
    )
    # The excluded location must reach the figure too, or a reader sees four
    # bars and cannot tell whether the fifth was removed or never existed.
    assert "Phc" in set(plotted["location"])


# ---------------------------------------------------------------------------
# T70.5 -- the Most audible location cross-check
# ---------------------------------------------------------------------------


def test_the_model_scores_higher_where_the_murmur_was_heard_loudest() -> None:
    """T70.5 -- against a field the model never saw.

    This is the check that the per-location numbers mean something. If the model
    were keying on a per-patient confound rather than on the murmur, its
    probability would not track the position a clinician marked loudest, and the
    difference would sit at zero.
    """
    check = _csv("circor_most_audible_location_check.csv")
    assert set(check["model_id"]) == set(CIRCOR_MODELS)
    assert set(check["most_audible_location"]) <= set(DECLARED_COUNTS)
    for model_id, block in check.groupby("model_id"):
        delta = float(np.mean(np.asarray(block["delta"], dtype=float)))
        assert delta > 0, model_id + " scores no higher at the loudest location"
        assert len(block) > 100, model_id + " has only " + str(len(block)) + " patients"


def test_the_most_audible_note_states_the_size_as_well_as_the_sign() -> None:
    """A positive difference that is small must not be written up as a large one."""
    path = _root() / "circor_most_audible_location.md"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/21_location_analysis.py")
    text = path.read_text(encoding="utf-8")
    assert "Most audible location" in text
    assert "not a diagnostic tool" in text.lower()
    assert "not large" in text.lower()


# ---------------------------------------------------------------------------
# needs a parquet -- skips on CI
# ---------------------------------------------------------------------------


def test_stratification_partitions_every_prediction_exactly_once() -> None:
    """No recording may be counted twice or dropped by the stratification."""
    from src.evaluation.location import attach_locations

    predictions = _parquet("EXP-C2", "predictions.parquet")
    attached = attach_locations(predictions)
    assert len(attached) == len(predictions)
    assert attached["location"].notna().all()
    per_model = attached.groupby(["model_id", "location"]).size().unstack(fill_value=0)
    for model_id, row in per_model.iterrows():
        assert row.to_dict() == DECLARED_COUNTS, str(model_id)


def test_the_per_fold_frame_behind_t22_re_derives_it() -> None:
    """An aggregate that cannot be rebuilt from its folds is not checkable."""
    from src.reporting.location_report import summarise_locations

    frames = [_csv(run, "per_fold_by_location.csv") for run in RUNS]
    import pandas as pd

    rebuilt = summarise_locations(pd.concat(frames, ignore_index=True))
    t22 = _csv("T22_auscultation_location_analysis.csv")
    keys = ["run", "model_id", "location"]
    merged = rebuilt.merge(t22, on=keys, suffixes=("_r", "_t"))
    assert len(merged) == len(t22)
    for column in ("balanced_accuracy_mean", "specificity_mean"):
        assert np.allclose(
            np.asarray(merged[column + "_r"], dtype=float),
            np.asarray(merged[column + "_t"], dtype=float),
            equal_nan=True,
        )
