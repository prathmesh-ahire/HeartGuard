"""🔴 MEGA TEST 5 — the full system (Phase 121).

Covers Part X (Phases 106-120) and everything it stands on. Nothing in Part XI
starts until this file is green.

MEGA TEST 4 asked whether the deliverables were all there. This one asks whether
the *system* is: whether one command rebuilds it, whether the guard rail that
protects every displayed number actually fails when it should, whether the built
site still matches the CSVs behind it, and whether live inference agrees with
what the experiments recorded for the same recording.

* T121.1 -- the one-command reproduction resolves and covers every producer;
* T121.2 -- a planted metric literal FAILS the build, and the clean tree passes;
* T121.3 -- the displayed-value audit is green over every exported page;
* T121.4 -- a known corpus recording, scored live, agrees with its stored
  out-of-fold prediction from EXP-A2;
* T121.5 -- a degenerate recording is refused with a reason, never scored;
* T121.7 -- the thirteen screenshots exist, are registered, and are of the
  audited build.

T121.6 is [TEST/MANUAL]: a human clicks through all twelve routes with the API
stopped and then running. Playwright's `e2e/api-stopped.spec.ts` covers the
automatable half of it and is not re-run here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.pipeline import run_all
from src.reporting import display_audit as da
from src.reporting import screenshots as ss
from src.utils.evidence import PROJECT_ROOT, read_evidence

OUT = PROJECT_ROOT / "frontend" / "out"
GENERATED = PROJECT_ROOT / "frontend" / "lib" / "generated"
SCREENSHOTS = PROJECT_ROOT / "outputs" / "15_dashboard_screenshots"
REAL_MANIFEST = PROJECT_ROOT / "outputs" / "00_evidence_index" / "run_manifest.json"

built = pytest.mark.skipif(
    not (OUT / "index.html").is_file(),
    reason="frontend/out/ is not built in this checkout; run npm run build",
)


# ---------------------------------------------------------------------------
# T121.1 -- one command
# ---------------------------------------------------------------------------


def test_t121_1_one_command_resolves_every_stage_with_no_manual_step() -> None:
    """Every stage names a real command, and the chain ends at the screenshots."""
    payload = run_all.run_stages(list(run_all.STAGES), dry_run=True)
    assert payload["n_stages"] == len(run_all.STAGES) >= 60
    assert all(entry["status"] == "dry-run" for entry in payload["stages"])
    assert run_all.STAGES[-2].stage_id == "screenshots"
    # No stage may need a human between two commands: every one of them is argv.
    for stage in run_all.STAGES:
        assert stage.argv, stage.stage_id


def test_t121_1_the_runner_reaches_the_frontend_without_a_separate_invocation() -> None:
    """T122.2: the export, install, build and guard rail are stages, not a README step."""
    ids = [stage.stage_id for stage in run_all.STAGES if stage.frontend]
    assert ids == ["npm_ci", "frontend_build", "screenshots", "evidence_index_final"]


# ---------------------------------------------------------------------------
# T121.2 -- the guard rail, proven rather than assumed
# ---------------------------------------------------------------------------


GUARD = PROJECT_ROOT / "scripts" / "16_check_no_hardcoded_metrics.py"


def _guard(frontend_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GUARD), "--frontend", str(frontend_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_t121_2_the_clean_tree_passes_the_guard_rail() -> None:
    result = _guard(PROJECT_ROOT / "frontend")
    assert result.returncode == 0, result.stdout + result.stderr


def test_t121_2_a_planted_metric_literal_fails_the_guard_rail(tmp_path: Path) -> None:
    """The same literal Phase 119 planted for T119.7, on a copy of the real tree."""
    import shutil

    root = tmp_path / "frontend"
    (root / "app" / "limitations").mkdir(parents=True)
    shutil.copytree(
        PROJECT_ROOT / "frontend" / "components", root / "components", dirs_exist_ok=True
    )
    shutil.copytree(PROJECT_ROOT / "frontend" / "app", root / "app", dirs_exist_ok=True)

    page = root / "app" / "limitations" / "page.tsx"
    original = page.read_text(encoding="utf-8")
    page.write_text(
        original.replace(
            "export default function",
            "const accuracy = 95.82;\n\nexport default function",
            1,
        ),
        encoding="utf-8",
    )
    planted = _guard(root)
    assert planted.returncode != 0, "a hand-typed metric passed the guard rail"
    assert "95.82" in planted.stdout + planted.stderr

    page.write_text(original, encoding="utf-8")
    assert _guard(root).returncode == 0


# ---------------------------------------------------------------------------
# T121.3 -- the displayed-value audit over the built site
# ---------------------------------------------------------------------------


@built
def test_t121_3_every_rendered_metric_matches_its_source() -> None:
    report = da.run_audit(run_manifest=REAL_MANIFEST)
    assert report.passed, json.dumps(report.to_dict(), indent=2)[:4000]
    assert report.checked.get("pages", 0) >= 12


@built
def test_t121_3_the_screenshot_gate_is_open_on_the_site_on_disk() -> None:
    """T119.4 -- and therefore that the screenshots below are of THIS build.

    The stamp is gitignored (it describes one build), so a checkout that has the
    committed `out/` but has not run the audit itself skips rather than fails.
    """
    if not da.default_report_path().is_file():
        pytest.skip("no audit stamp in this checkout; run npm run build")
    da.require_passed_audit()


# ---------------------------------------------------------------------------
# T121.4 -- live inference against the stored out-of-fold prediction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deployed_binary() -> object:
    """The binary bundle, or a module-wide skip.

    `models_saved/**/*.joblib` is gitignored, so CI has the manifests and not the
    binaries. A test that reached the predictor unguarded would ERROR there; the
    standing rule is that a gitignored input is a skip, never a failure.
    """
    try:
        from src.inference.predictor import load_bundle

        return load_bundle("binary")
    except Exception as error:  # noqa: BLE001 -- a missing bundle is a skip, not a failure
        pytest.skip("binary model unavailable (" + type(error).__name__ + ")")


@pytest.mark.parametrize("sample_id", ["binary-abnormal", "binary-normal"])
def test_t121_4_a_known_record_agrees_with_its_stored_out_of_fold_prediction(
    deployed_binary: object, sample_id: str
) -> None:
    """The deployed model's class must match what the 25-fold map recorded.

    Not the probability: the deployed bundle is refit **in sample** on all 3,240
    records, so its probability for a training record is not comparable with an
    out-of-fold one and asserting equality would be asserting leakage. What is
    comparable is the decision, on the two samples whose five repeats agree.
    """
    from src.inference.predictor import predict_recording
    from src.reporting.samples import SAMPLES, resolve_sample, stored_reference

    spec = next(item for item in SAMPLES if item.sample_id == sample_id)
    path = resolve_sample(sample_id)
    if path is None or not Path(path).is_file():
        pytest.skip(sample_id + " is not reachable; dataset/ is not in this checkout")

    reference = stored_reference().get(spec.record_uid)
    if reference is None:
        pytest.skip("EXP-A2 predictions are not in this checkout")
    assert reference["folds_agree"], sample_id + " was chosen because its repeats agree"

    outcome = predict_recording(path, task="binary")
    stored_class = reference["predicted_classes"][0]
    assert outcome.predicted_class == stored_class, (
        sample_id
        + ": live "
        + outcome.predicted_class
        + " vs stored out-of-fold "
        + stored_class
    )
    assert outcome.predicted_class == reference["true_class"], (
        sample_id + " is a record both the corpus and the folds agree on"
    )
    assert outcome.n_features == 138
    assert outcome.n_missing_features == 0


# ---------------------------------------------------------------------------
# T121.5 -- a degenerate recording
# ---------------------------------------------------------------------------


def _wav(path: Path, samples: list[int], fs: int = 2000) -> Path:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(fs)
        handle.writeframes(b"".join(int(v).to_bytes(2, "little", signed=True) for v in samples))
    return path


def test_t121_5_a_degenerate_recording_returns_null_with_an_explanation(
    deployed_binary: object, tmp_path: Path
) -> None:
    """Silence carries no cardiac information, so it gets no probability at all.

    This test was written because the system got it wrong: before Phase 121, a
    fully silent 8 s recording came back **abnormal at confidence 1.000** with
    24 of 138 features imputed from the training median. Maximum confidence over
    zero information is the worst output this dashboard could produce, and it
    was the default one.
    """
    from src.inference.predictor import predict_recording

    silent = _wav(tmp_path / "silence.wav", [0] * 16000)
    outcome = predict_recording(silent, task="binary")

    assert outcome.scorable is False
    assert outcome.predicted_class == ""
    assert outcome.predicted_index == -1
    assert set(outcome.probabilities.values()) == {None}
    assert outcome.confidence is None
    assert outcome.margin is None
    assert outcome.not_scorable_reason and len(outcome.not_scorable_reason) > 40


def test_t121_5_a_scorable_recording_is_still_scored(
    deployed_binary: object, tmp_path: Path
) -> None:
    """The guard must refuse silence without refusing a real recording."""
    import math

    from src.inference.predictor import predict_recording

    tone = _wav(
        tmp_path / "tone.wav",
        [int(6000 * math.sin(2 * math.pi * 40 * n / 2000)) for n in range(2000 * 8)],
    )
    outcome = predict_recording(tone, task="binary")
    assert outcome.scorable is True
    assert outcome.predicted_class in {"normal", "abnormal"}
    assert outcome.confidence is not None


def test_t121_5_a_too_short_recording_states_the_bound(tmp_path: Path) -> None:
    from src.inference.predictor import (
        MIN_DURATION_SECONDS,
        AudioValidationError,
        predict_recording,
    )

    tiny = _wav(tmp_path / "tiny.wav", [1000, -1000] * 100)
    with pytest.raises(AudioValidationError) as raised:
        predict_recording(tiny, task="binary")
    assert format(MIN_DURATION_SECONDS, ".2f") in str(raised.value)


def test_t121_5_the_api_returns_null_probabilities_and_an_explanation(
    tmp_path: Path,
) -> None:
    """200 with nulls and a reason -- what the page renders as "Not scored"."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import app

    silent = _wav(tmp_path / "silence.wav", [0] * 16000)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("silence.wav", silent.read_bytes(), "audio/wav")},
            data={"task": "binary"},
        )
    if response.status_code == 503:
        pytest.skip("no binary model bundle in this checkout")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scorable"] is False
    assert set(body["probabilities"].values()) == {None}
    assert body["confidence"] is None
    assert len(body["not_scorable_reason"]) > 40
    # The page renders `display.*`, so the strings must say n/a rather than 0.
    assert set(body["display"]["probabilities"].values()) == {"n/a"}
    assert body["display"]["confidence"] == "n/a"
    assert body["explanation"]["available"] is False


def test_t121_5_a_corrupt_upload_is_a_4xx_with_its_reason(tmp_path: Path) -> None:
    """A file that is not audio never reaches the model; it is refused outright."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api.main import app

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"RIFF    WAVEthis is not audio")
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("corrupt.wav", corrupt.read_bytes(), "audio/wav")},
            data={"task": "binary"},
        )
    assert response.status_code >= 400
    detail = response.json().get("detail", "")
    assert isinstance(detail, str) and len(detail) > 10, response.text


# ---------------------------------------------------------------------------
# T121.6's automatable half -- the patient collapse is about the right class
# ---------------------------------------------------------------------------


def test_t121_6_the_patient_collapse_uses_the_class_t15_was_computed_on() -> None:
    """The dashboard and T15 must collapse over the SAME class.

    `_positive_class` used to be `classes[-1]`, which for the three-class murmur
    space is `Unknown` -- the annotator's own third category. The page therefore
    showed a patient-level indication built from "the annotator could not tell",
    labelled the same way T15's `Present` collapse is. Two classes hid it: for
    every other task `classes[-1]` happens to be index 1.
    """
    from src.api.main import _positive_class
    from src.evaluation.aggregation import POSITIVE_LABEL
    from src.inference.predictor import TASKS

    assert POSITIVE_LABEL == 1
    assert _positive_class(list(TASKS["murmur"].classes)) == "Present"
    assert _positive_class(list(TASKS["outcome"].classes)) == "Abnormal"
    assert _positive_class(list(TASKS["binary"].classes)) == "abnormal"
    # scripts/18_circor_tables.py passes this index for all three CirCor runs.
    circor = (PROJECT_ROOT / "scripts" / "18_circor_tables.py").read_text(encoding="utf-8")
    assert circor.count('"EXP-C1-three_class", "EXP-C1", "three_class", 1,') == 1


def test_t121_6_a_result_states_the_operating_point_of_ITS_OWN_task() -> None:
    """One binary note used to be served to all five tasks.

    A PASCAL A four-class result carried "plain argmax at 0.5 ... every stored
    prediction in outputs/06_binary_results/ was produced the same way" -- a
    threshold that does not exist on a four-class task and a provenance claim
    about a directory that has nothing to do with PASCAL A. It is in screenshots
    SS-09 and SS-10, which is where it was finally noticed.
    """
    from src.inference.predictor import TASKS, _operating_point_note

    for task, spec in TASKS.items():
        note = _operating_point_note(task, len(spec.classes))
        if len(spec.classes) == 2:
            assert "0.5" in note, task
            assert "06_binary_results" not in note or task == "binary", task
        else:
            assert "0.5" not in note, task
            assert "06_binary_results" not in note, task
            assert "no single operating point" in note, task


# ---------------------------------------------------------------------------
# T121.7 -- the screenshots are of the audited build and are registered
# ---------------------------------------------------------------------------


def test_t121_7_the_thirteen_screenshots_exist_and_are_registered() -> None:
    """Read-only: a test must not rewrite the deliverable it is checking."""
    import csv

    index = SCREENSHOTS / "screenshot_index.csv"
    if not index.is_file():
        pytest.skip("the gated capture has not been run in this checkout")

    with index.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["screenshot_id"]: row for row in csv.DictReader(handle)}
    assert len(rows) == 13

    for shot in ss.CAPTURE_PLAN:
        row = rows.get(shot.shot_id)
        assert row is not None, shot.shot_id + " is not in the index"
        assert row["filename"] == shot.filename
        assert row["caption"] == shot.caption
        png = SCREENSHOTS / shot.filename
        assert png.is_file(), shot.filename
        # The same floor `finalize` refuses on: a blank frame is not a capture.
        assert png.stat().st_size >= ss.MIN_BYTES, shot.shot_id

    registered = {row["evidence_id"]: row for row in read_evidence()}
    for shot in ss.CAPTURE_PLAN:
        row = registered.get(shot.shot_id)
        assert row is not None, shot.shot_id + " is not in the evidence index"
        assert row["status"] == "ok", shot.shot_id
        assert row["filename"].endswith(shot.filename), shot.shot_id


def test_t121_7_every_screenshot_has_a_caption_that_says_what_it_shows() -> None:
    captions = SCREENSHOTS / "captions.md"
    if not captions.is_file():
        pytest.skip("the gated capture has not been run in this checkout")
    text = captions.read_text(encoding="utf-8")
    assert "Not a diagnostic device" in text
    for shot in ss.CAPTURE_PLAN:
        assert shot.caption in text, shot.shot_id + " has no caption"
        assert "![" + shot.title + "](" + shot.filename + ")" in text, shot.shot_id
