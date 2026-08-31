"""T71.7 -- the EXP-D1 cross-dataset transfer gate.

Three clauses, and the first two are about process rather than about a number:

1. **The population mismatch is recorded in the experiment metadata BEFORE the
   metric.** Checked two ways: the metadata carries no metric-like value at all,
   and its timestamp precedes T16's.
2. **No retuning occurred on CirCor.** The applied model's hyperparameters must
   be exactly the saved PhysioNet model's, and that model must report having been
   fitted on the PhysioNet task and record count.
3. **The T71.5 framing rule is honoured.** Wherever an EXP-D1 artifact uses the
   word "generaliz", it must also carry the sentence saying a large drop is the
   expected population effect. A number this low is trivially misread, and the
   misreading is what this clause exists to prevent.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pytest

SECTION = "outputs/08_circor_external_validation"
RUN_DIR = "EXP-D1"

#: PhysioNet supervised records. The transfer model must have been fitted on
#: these and nothing else.
N_PHYSIONET_TRAIN = 3240
N_CIRCOR_RECORDINGS = 3163
N_CIRCOR_PATIENTS = 942


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _run_dir() -> Any:
    return _root() / RUN_DIR


def _json(path: Any) -> dict:
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/22_cross_dataset.py")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _csv(*parts: str) -> Any:
    import pandas as pd

    path = _root().joinpath(*parts)
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/22_cross_dataset.py")
    return pd.read_csv(path)


def _stamp(text: str) -> datetime:
    return datetime.fromisoformat(str(text))


# ---------------------------------------------------------------------------
# clause 1 -- the framing was written first
# ---------------------------------------------------------------------------


def test_the_population_mismatch_is_recorded_and_holds_no_metric() -> None:
    """T71.1 -- the file whose entire value is that it predates the result."""
    from src.evaluation.transfer import METADATA_FILENAME, _metric_like_keys

    metadata = _json(_run_dir() / METADATA_FILENAME)
    assert metadata["experiment_id"] == "EXP-D1"
    assert metadata["written_before_any_metric"] is True
    assert metadata["retuning_allowed"] is False
    assert metadata["retuning_performed"] is False
    assert not _metric_like_keys(metadata), sorted(_metric_like_keys(metadata))

    train, test = metadata["train"], metadata["test"]
    # Measured, not quoted from the blueprint.
    assert train["cohort"] == "adult"
    assert train["age_median_years"] == 25.0
    assert train["share_under_18"] < 0.01
    assert test["cohort"] == "predominantly paediatric"
    assert test["n_patients"] == N_CIRCOR_PATIENTS
    assert test["share_paediatric_of_recorded"] > 0.95
    # The two corpora record age on different scales, and that is stated.
    assert "band" in str(test["age_scale"]).lower()


def test_the_metadata_predates_the_table_it_frames() -> None:
    """T71.7 -- "before the metric" is checked, not asserted in a comment."""
    from src.evaluation.transfer import METADATA_FILENAME

    metadata = _json(_run_dir() / METADATA_FILENAME)
    table_meta = _json(_root() / "T16_cross_dataset_generalization.meta.json")
    written = _stamp(metadata["written_utc"])
    generated = _stamp(table_meta["generated_utc"])
    assert written < generated, (
        "the population metadata was stamped "
        + str(written)
        + " but T16 was generated "
        + str(generated)
        + "; the framing must be recorded before the result exists"
    )
    assert METADATA_FILENAME in {p.name for p in _run_dir().glob("*.json")}


# ---------------------------------------------------------------------------
# clause 2 -- nothing was retuned
# ---------------------------------------------------------------------------


def test_the_applied_model_is_the_saved_physionet_model_unchanged() -> None:
    """T71.2 / T71.7 -- no retuning, no recalibration, no partial refit."""
    from src.evaluation.transfer import METADATA_FILENAME
    from src.utils.evidence import PROJECT_ROOT

    metadata = _json(_run_dir() / METADATA_FILENAME)
    saved = _json(PROJECT_ROOT / "models_saved" / "binary" / "final" / "manifest.json")

    recorded = metadata["model"]
    assert recorded["fitted_on_task"] == "binary"
    assert recorded["n_records_fitted"] == N_PHYSIONET_TRAIN
    assert recorded["selected_model_id"] == saved["selected_model_id"]
    assert recorded["hyperparameters"] == saved["hyperparameters"]
    # The hyperparameters came from the PhysioNet search, not from anything that
    # saw CirCor.
    assert "T07" in str(recorded["hyperparameter_source"])


def test_the_external_evaluation_covers_the_whole_corpus_at_both_levels() -> None:
    """T71.3 / T71.4 -- all 3,163 recordings, all 942 patients, all three rules."""
    from src.evaluation.aggregation import AGGREGATION_RULES

    metrics = _csv(RUN_DIR, "metrics_by_level.csv")
    recording = metrics[metrics["level"] == "recording"]
    assert len(recording) == 1
    assert int(recording["n_units"].iloc[0]) == N_CIRCOR_RECORDINGS

    patient = metrics[metrics["level"] == "patient"]
    assert set(patient["rule"]) == set(AGGREGATION_RULES), sorted(set(patient["rule"]))
    assert (patient["n_units"] == N_CIRCOR_PATIENTS).all()


# ---------------------------------------------------------------------------
# clause 3 -- the framing rule
# ---------------------------------------------------------------------------


def test_the_drop_is_quantified_against_the_in_domain_result() -> None:
    """T71.5 -- a signed delta against EXP-A2, with both fold counts stated."""
    drops = _csv(RUN_DIR, "degradation.csv")
    assert set(drops["in_domain_exp"]) == {"EXP-A2"}
    assert (drops["in_domain_n_folds"] == 25).all()
    assert (drops["external_n_folds"] == 1).all()

    recording = drops[(drops["level"] == "recording")].set_index("metric")
    for metric in ("sensitivity", "specificity", "balanced_accuracy", "roc_auc"):
        assert metric in recording.index, metric
        row = recording.loc[metric]
        assert np.isclose(
            float(row["delta"]),
            float(row["external_value"]) - float(row["in_domain_mean"]),
        )
    # The drop is large and negative. That is the expected result, not a failure
    # -- but if it ever stops being large, the framing must be revisited rather
    # than left in place describing a result it no longer describes.
    assert float(recording.loc["balanced_accuracy", "delta"]) < -0.15


def test_every_artifact_using_the_word_generalize_carries_the_framing() -> None:
    """T71.7 clause 3 -- the rule that stops this becoming "the method fails"."""
    from src.reporting.transfer_report import TRANSFER_HEADLINE

    marker = "expected consequence of that mismatch"
    checked = 0
    for path in sorted(_run_dir().glob("*.md")) + sorted(_root().glob("T16_*.md")):
        text = path.read_text(encoding="utf-8")
        if "generaliz" not in text.lower():
            continue
        checked += 1
        assert marker in text or "EXPECTED consequence" in text, path.name
    for path in sorted(_root().glob("T16_*.meta.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(payload)
        if "generaliz" not in blob.lower():
            continue
        checked += 1
        assert TRANSFER_HEADLINE.split(".")[0] in payload["caption"], path.name
    assert checked >= 2, "no EXP-D1 artifact mentioned generalization; the check is vacuous"


def test_the_framing_note_names_all_three_causes_of_the_drop() -> None:
    """Attributing the whole drop to age would be the next wrong write-up.

    Two other causes are already measured elsewhere in this project: PhysioNet's
    sub-collections behave like six different datasets, and CirCor's Outcome is a
    clinical decision label rather than an acoustic one.
    """
    path = _run_dir() / "population_mismatch.md"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/22_cross_dataset.py")
    text = path.read_text(encoding="utf-8").lower()
    assert "population" in text
    assert "sub-collections" in text or "six different datasets" in text
    assert "clinical decision" in text
    assert "no retuning" in text
    assert "not a diagnostic tool" in text


def test_g29_plots_the_recording_level_comparison_only() -> None:
    """Patient-level aggregation beside the in-domain bar would read as a fix."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = (
        PROJECT_ROOT / "outputs" / "13_figures_diagrams" / "G29_cross_dataset_performance_drop.csv"
    )
    if not path.is_file():
        pytest.skip("G29 not generated; run scripts/22_cross_dataset.py")
    plotted = pd.read_csv(path)
    assert set(plotted["level"]) == {"recording"}
    assert (plotted["delta"] < 0).all(), "a positive delta on transfer needs explaining"
    assert (
        PROJECT_ROOT / "outputs" / "13_figures_diagrams" / "G29_cross_dataset_performance_drop.png"
    ).is_file()
