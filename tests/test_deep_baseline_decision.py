"""The 1D-CNN (M9) scope decision (Phase 52, gate T52.7).

The gate: confirm the 1D-CNN decision is recorded either as an implemented model
or as an entry in `missing_outputs_report.txt` with its technical reason.

An optional model that is simply absent looks identical to one that was
forgotten. This module is what makes the difference checkable: whichever way the
decision went, it has to be findable in config, in the registry, and -- if it
went the other way -- in the missing-outputs report with a reason a reader can
evaluate. "Not in the results table" is never an acceptable record of a
deliberate exclusion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import estimators as est
from src.models import registry as reg

REPORT = Path("outputs/missing_outputs_report.txt")


def _report_text() -> str:
    if not REPORT.is_file():
        pytest.skip("missing_outputs_report.txt is absent")
    return REPORT.read_text(encoding="utf-8")


def _flowed(text: str) -> str:
    """The report is wrapped prose, so collapse whitespace before matching.

    A phrase that happens to straddle a line break is the same phrase. Matching
    the raw text makes the assertion depend on where the wrapping fell, which
    would fail on a reflow that changed nothing.
    """
    return " ".join(text.split())


def test_the_decision_is_recorded_one_way_or_the_other():
    """The gate, stated directly."""
    from src.utils.config import load_config

    in_scope = load_config("models").get("models.M9.in_scope")
    assert in_scope is not None, (
        "M9's scope is still undecided (in_scope: null); T52.1 is a user call "
        "and cannot be left open past Phase 52"
    )

    if in_scope:
        assert "M9" in est.IMPLEMENTED_MODELS
        assert reg.available("M9")
        return

    assert "M9" not in est.IMPLEMENTED_MODELS
    assert not reg.available("M9")
    assert "1D-CNN" in _report_text()


def test_the_exclusion_carries_a_technical_reason():
    """Ground rule 1: never invented, never silently omitted, always explained."""
    from src.utils.config import load_config

    if load_config("models").get("models.M9.in_scope"):
        pytest.skip("M9 is in scope; there is no exclusion to justify")

    text = _report_text()
    start = text.index("1D-CNN")
    entry = _flowed(text[start - 400 : start + 2500])

    assert "Reason:" in entry
    assert "Impact:" in entry
    # The reason must be technical and specific, not "not needed".
    assert "CPU" in entry
    assert "T52" in entry


def test_the_exclusion_reason_reaches_anyone_who_asks_for_the_model():
    """A caller must not have to know to go looking for the report."""
    from src.utils.config import load_config

    if load_config("models").get("models.M9.in_scope"):
        pytest.skip("M9 is in scope")

    with pytest.raises(est.EstimatorError) as caught:
        est.build_estimator("M9")
    message = str(caught.value)
    assert "missing_outputs_report" in message
    assert "excluded" in message


def test_the_registry_shows_the_exclusion_rather_than_omitting_the_row():
    """An absent row reads as an oversight; a row with a reason reads as a decision."""
    frame = reg.registry_frame()
    row = frame[frame["model_id"] == "M9"]
    assert len(row) == 1, "M9 must stay in the registry even when excluded"
    assert row["unavailable_reason"].item()


def test_no_claim_may_compare_against_a_cnn():
    """The exclusion's downstream consequence, written where it will be read."""
    from src.utils.config import load_config

    if load_config("models").get("models.M9.in_scope"):
        pytest.skip("M9 is in scope")
    assert "may compare PV-MEPCG against a CNN" in _flowed(_report_text())
