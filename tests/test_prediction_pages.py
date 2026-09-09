"""Phase 116's gate: the prediction pages, and what an upload actually returns.

T116.7 asks that a known record uploaded through the UI reproduce the stored
out-of-fold prediction for that record. Two things about that sentence decide
how this file is written.

**The deployed bundle cannot reproduce an out-of-fold value, and must not be
made to look as though it can.** `models_saved/binary/final/` is M1 refitted on
all 3,240 labelled records. Its probability for record X is an in-sample one;
X's stored value is held out. Asserting they agree would be asserting something
false, and "fixing" it with a tolerance would be worse.

So the gate is one record carried along the whole chain, in two legs that meet
in the middle:

* **Leg A -- the wire adds nothing.** The record's real bytes are POSTed as
  multipart to `/predict`, exactly as the browser's `FormData` sends them, and
  the response is asserted equal to a direct `predict_recording` call on the same
  file. The API transports a result; it does not compute or reshape one.
* **Leg B -- the predictor reproduces the fold.** The same file, through the same
  `predict_recording`, against a refit of the fold the stored prediction came
  from (EXP-A1/r0f0/M1, config defaults) must return that fold's stored
  probability to within the measured single-record bound, and its class exactly.

Together those say: the bytes a browser uploads reach a predictor that
reproduces the experiment.

**Single-record probabilities are bounded in ULP, not equal.** Scoring one row is
a GEMV where the fold was a GEMM, so the dot product accumulates in a different
order. Measured over 40 held-out records of EXP-A1/r0f0/M1: max 8 ULP
(4.44e-16), zero class flips. The bound here is 32 ULP with that measurement
cited, the same constant `tests/test_inference.py` uses and for the same reason.
Class decisions are asserted exactly; a class is not a float and cannot drift.

**Browser-level driving is Phase 118's.** T116.7 says "through the UI", and the
strongest thing available before Playwright exists is the real HTTP request the
UI makes, with the real bytes, against the real app. That is what leg A is. The
page's own call is pinned structurally -- `lib/api.ts` posts `file` and `task`
to `/predict` -- and Phase 118's smoke test drives the rendered control.

Everything needing `dataset/` or a `.parquet` skips: both are gitignored, so CI
runs the structural half of this file and nothing else.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from src.api.main import _patient_rules, create_app
from src.evaluation.aggregation import AGGREGATION_RULES
from src.inference.predictor import TASKS, load_bundle, predict_recording
from src.reporting.samples import (
    PATIENT_GROUP,
    SAMPLES,
    prediction_payload,
    resolve_sample,
    sample_locations,
)
from src.reporting.tables import format_value

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
APP = FRONTEND / "app" / "predict"
COMPONENTS = FRONTEND / "components" / "predict"
GENERATED = FRONTEND / "lib" / "generated"
MATRIX = PROJECT_ROOT / "outputs" / "03_features" / "all_features_matrix.parquet"
EXP_DIR = PROJECT_ROOT / "outputs" / "06_binary_results" / "EXP-A1"
BUNDLE_DIR = PROJECT_ROOT / "models_saved" / "binary" / "final"

#: The fold the gate reproduces. EXP-A1 is `tuned: false`, so a refit needs no
#: search and the gate is cheap enough to run.
FOLD_LABEL = "r0f0"
MODEL_ID = "M1"

#: See the module docstring: measured at 8 ULP over 40 records, bounded at 32.
SINGLE_RECORD_ULP = 32


@pytest.fixture(scope="module")
def client() -> Any:
    """The real app, with no preload: the bundle loads on demand either way."""
    return TestClient(create_app(preload=False))


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return prediction_payload()


# ---------------------------------------------------------------------------
# T116.6 -- the built-in samples
# ---------------------------------------------------------------------------


def test_every_declared_sample_resolves_in_the_audit(payload: dict[str, Any]) -> None:
    """A pinned id that no longer names a record is a corpus change, not a skip."""
    assert len(payload["samples"]) == len(SAMPLES)
    uids = {entry["record_uid"] for entry in payload["samples"]}
    assert uids == {spec.record_uid for spec in SAMPLES}


def test_no_corpus_audio_is_committed_for_the_samples() -> None:
    """The licence decision, asserted rather than left to a docstring.

    Only `85197_TV.wav` is committed, under the CirCor ODC-By grant that
    `frontend/public/NOTICE.md` records. A PhysioNet or PASCAL WAV appearing in
    `frontend/public/` would be redistribution under terms nobody verified.
    """
    committed = sorted(path.name for path in (FRONTEND / "public").rglob("*.wav"))
    assert committed == ["85197_TV.wav"], (
        "unexpected audio in frontend/public/: "
        + ", ".join(committed)
        + ". Corpus recordings are served from the operator's own dataset/ copy "
        "by GET /samples/{id}/audio; they are not committed."
    )


def test_the_samples_cover_a_confident_and_a_borderline_binary_record(
    payload: dict[str, Any],
) -> None:
    """T116.2's warning must demonstrate on a real record, not a contrived one."""
    binary = [s for s in payload["samples"] if "binary" in s["tasks"]]
    assert len(binary) >= 3
    references = [s["reference"] for s in binary if s["reference"] is not None]
    if not references:
        pytest.skip("EXP-A2 predictions are not in this checkout (gitignored)")
    assert any(not ref["folds_agree"] for ref in references), (
        "no borderline sample: every binary sample's repeats agree, so the "
        "low-confidence path cannot be shown without manufacturing one"
    )
    assert any(ref["folds_agree"] for ref in references)


def test_a_binary_sample_carries_all_five_stored_probabilities(
    payload: dict[str, Any],
) -> None:
    """Five, not one. Quoting one would be picking a fold."""
    references = [s["reference"] for s in payload["samples"] if s["reference"] is not None]
    if not references:
        pytest.skip("EXP-A2 predictions are not in this checkout (gitignored)")
    for reference in references:
        assert reference["n_repeats"] == 5
        assert len(reference["probabilities"]) == 5
        assert len(reference["fold_labels"]) == len(set(reference["fold_labels"]))
        assert reference["model_id"] == "M1"


def test_stored_probabilities_are_displayed_at_the_project_places(
    payload: dict[str, Any],
) -> None:
    """`format_value` is the only rounding authority; the payload used it."""
    for sample in payload["samples"]:
        reference = sample["reference"]
        if reference is None:
            continue
        for value, shown in zip(
            reference["probabilities"], reference["probabilities_display"], strict=True
        ):
            assert shown == format_value(value, "metric")
        assert reference["mean_probability_display"] == format_value(
            reference["mean_probability"], "metric"
        )


def test_the_patient_group_is_one_subject_at_several_locations(
    payload: dict[str, Any],
) -> None:
    """T116.4 cannot be shown from one recording each of four subjects."""
    subject, ids = PATIENT_GROUP
    group = payload["patient_group"]
    assert group["subject_id"] == subject
    assert list(ids) == group["sample_ids"]
    locations = sample_locations()
    assigned = [locations[sample_id] for sample_id in ids]
    assert len(set(assigned)) == len(assigned), "the group repeats an auscultation location"
    members = [s for s in payload["samples"] if s["sample_id"] in ids]
    assert {s["subject_id"] for s in members} == {subject}


def test_the_declared_tasks_are_the_five_label_spaces(payload: dict[str, Any]) -> None:
    assert [entry["task"] for entry in payload["tasks"]] == list(TASKS)
    for entry in payload["tasks"]:
        assert entry["classes"] == list(TASKS[entry["task"]].classes)


def test_the_samples_endpoint_reports_a_reason_when_it_cannot_serve_one(
    client: Any,
) -> None:
    rows = client.get("/samples").json()
    assert len(rows) == len(SAMPLES)
    for row in rows:
        assert row["available"] is (row["reason"] is None), (
            row["sample_id"] + " is neither available nor explained"
        )


def test_an_unknown_sample_is_refused_with_the_declared_ids(client: Any) -> None:
    response = client.get("/samples/not-a-sample/audio")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "not-a-sample" in detail
    assert SAMPLES[0].sample_id in detail


# ---------------------------------------------------------------------------
# T116.1 / T116.3 -- what a prediction response carries
# ---------------------------------------------------------------------------


def _first_servable(task: str) -> str:
    for spec in SAMPLES:
        if task in spec.tasks and resolve_sample(spec.sample_id) is not None:
            return spec.sample_id
    pytest.skip("no sample for task " + task + " is on this machine (dataset/ absent)")


def test_a_prediction_carries_a_display_string_for_every_number(client: Any) -> None:
    """The client renders `display.*`; nothing downstream rounds anything."""
    sample_id = _first_servable("binary")
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout")
    body = client.post("/predict/sample", data={"sample_id": sample_id, "task": "binary"}).json()

    display = body["display"]
    for name, value in body["probabilities"].items():
        assert display["probabilities"][name] == format_value(value, "metric")
        assert display["probabilities_percent"][name] == (
            format_value(None, "percent")
            if value is None
            else format_value(float(value) * 100.0, "percent") + "%"
        )
    assert display["confidence"] == format_value(body["confidence"], "metric")
    assert display["margin"] == format_value(body["margin"], "metric")
    assert display["n_features"] == format_value(body["n_features"], "count")


def test_low_confidence_is_the_servers_decision_not_the_pages(client: Any) -> None:
    """A page that inferred it from a probability would be a second rule."""
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout")
    sample_id = _first_servable("binary")
    body = client.post("/predict/sample", data={"sample_id": sample_id, "task": "binary"}).json()
    assert isinstance(body["low_confidence"], bool)
    assert body["low_confidence"] == (float(body["margin"]) < body["low_confidence_margin"])


def test_every_prediction_carries_the_disclaimer(client: Any) -> None:
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout")
    body = client.post(
        "/predict/sample", data={"sample_id": _first_servable("binary"), "task": "binary"}
    ).json()
    assert "not a diagnostic device" in body["disclaimer"].lower()


def test_an_unbuilt_task_says_why_rather_than_scoring_anyway(client: Any) -> None:
    """PASCAL and CirCor have no deployed bundle; the page must show the reason."""
    unavailable = [name for name, row in _task_status(client).items() if not row["available"]]
    if not unavailable:
        pytest.skip("every declared task has a deployed model in this checkout")
    task = unavailable[0]
    sample_id = next(
        (s.sample_id for s in SAMPLES if task in s.tasks and resolve_sample(s.sample_id)),
        None,
    )
    if sample_id is None:
        pytest.skip("no sample for " + task + " is on this machine")
    response = client.post("/predict/sample", data={"sample_id": sample_id, "task": task})
    assert response.status_code == 503
    assert task in response.json()["detail"]


def _task_status(client: Any) -> dict[str, Any]:
    return {row["task"]: row for row in client.get("/tasks").json()}


def test_an_unknown_task_names_the_five_spaces(client: Any) -> None:
    response = client.post(
        "/predict/sample", data={"sample_id": SAMPLES[0].sample_id, "task": "murmur_or_binary"}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "never merged" in detail
    for name in TASKS:
        assert name in detail


# ---------------------------------------------------------------------------
# T116.4 -- the patient-level collapse
# ---------------------------------------------------------------------------


def test_all_three_declared_rules_are_returned_in_order(client: Any) -> None:
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout")
    ids = [s.sample_id for s in SAMPLES if "binary" in s.tasks and resolve_sample(s.sample_id)]
    if len(ids) < 2:
        pytest.skip("fewer than two binary samples are on this machine")
    body = client.post(
        "/predict/patient", data={"sample_ids": ",".join(ids), "task": "binary"}
    ).json()
    assert [rule["rule"] for rule in body["rules"]] == list(AGGREGATION_RULES)
    assert body["n_recordings"] == len(ids)
    for rule in body["rules"]:
        assert rule["score_display"] == format_value(rule["score"], "metric")


def test_the_live_rules_agree_with_the_experiment_aggregator() -> None:
    """The pin.

    `_patient_rules` repeats the arithmetic of
    `src.evaluation.aggregation.aggregate_predictions` because that function
    collapses a frame carrying `y_true` -- it asserts a subject's recordings
    agree on their label, which is right for an experiment and meaningless at
    inference time, where there is no label. Repeating arithmetic is how two
    implementations drift, so this runs both over the same scores and requires
    the same answer.
    """
    from src.evaluation.aggregation import aggregate_predictions

    scores = [0.91, 0.12, 0.34, 0.48]
    classes = ["normal", "abnormal"]

    class Stub:
        def __init__(self, score: float) -> None:
            self.probabilities = {"normal": 1.0 - score, "abnormal": score}
            self.predicted_class = "abnormal" if score >= 0.5 else "normal"

    # `_patient_rules` reads two attributes off each result; a stub carrying
    # exactly those is the point -- constructing four real PredictResponse
    # objects would test Pydantic, not the collapse.
    stubs: list[Any] = [Stub(value) for value in scores]
    live = {rule.rule: rule for rule in _patient_rules(stubs, classes)}

    # Real record_uids: `aggregate_predictions` resolves the subject through the
    # DA-07 map and refuses a uid that is not in it, which is the right refusal
    # -- a fabricated id would silently become its own patient.
    uids = [
        entry["record_uid"]
        for entry in prediction_payload()["samples"]
        if entry["sample_id"] in PATIENT_GROUP[1]
    ]
    if len(uids) != len(scores):
        pytest.skip("the patient group is not resolvable in this checkout")
    frame = [
        {
            "record_uid": uid,
            "model_id": "M1",
            "fold_label": "r0f0",
            "y_true": 1,
            "y_pred": int(score >= 0.5),
            "proba_1": score,
        }
        for uid, score in zip(uids, scores, strict=True)
    ]

    for rule_name in AGGREGATION_RULES:
        table = aggregate_predictions(frame, rule=rule_name)
        assert len(table) == 1
        expected_class = classes[int(table.iloc[0]["y_pred"])]
        assert live[rule_name].predicted_class == expected_class, (
            rule_name + ": the live collapse and the experiment aggregator disagree"
        )
        assert live[rule_name].score == pytest.approx(float(table.iloc[0]["score"]))


# ---------------------------------------------------------------------------
# T116.7 -- the gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fold_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Refit EXP-A1/r0f0/M1 and save it as a loadable bundle.

    Loaded through `load_task_data`, never a hand-rolled join: the loader sorts
    by `record_uid`, and lbfgs takes a different path through a differently
    ordered matrix -- a refit disagreed by 1.2e-3 with matching row counts until
    that was fixed, which reads exactly like float noise and is not.
    """
    import joblib
    import pandas as pd

    membership = EXP_DIR / "fold_membership.parquet"
    predictions = EXP_DIR / "predictions.parquet"
    if not membership.is_file() or not predictions.is_file() or not MATRIX.is_file():
        pytest.skip("EXP-A1 parquets are not in this checkout (gitignored)")

    from src.models import estimators as est
    from src.models import pipeline as pl
    from src.models.smoke import load_task_data

    data = load_task_data("binary")
    folds = pd.read_parquet(membership)
    fold = folds[folds["fold_label"] == FOLD_LABEL]
    position = {uid: index for index, uid in enumerate(data.record_uids)}
    train = np.sort(
        np.array([position[uid] for uid in fold[fold["split"] == "train"]["record_uid"]], dtype=int)
    )
    test = np.sort(
        np.array([position[uid] for uid in fold[fold["split"] == "test"]["record_uid"]], dtype=int)
    )

    built = pl.build_pipeline(
        est.build_estimator(MODEL_ID),
        config=None,
        y=data.y[train],
        n_features=int(data.X.shape[1]),
    )
    built.fit(data.X[train], data.y[train])

    directory = tmp_path_factory.mktemp("phase116_fold_bundle")
    joblib.dump(built, directory / "model.joblib")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "task": "binary",
                "fold": FOLD_LABEL,
                "n_features": int(data.X.shape[1]),
                "feature_names": list(data.feature_names),
                "note": "refitted in tests/test_prediction_pages.py for the T116.7 gate",
            }
        ),
        encoding="utf-8",
    )

    stored = pd.read_parquet(predictions)
    stored = stored[(stored["model_id"] == MODEL_ID) & (stored["fold_label"] == FOLD_LABEL)]
    matrix = pd.read_parquet(MATRIX).set_index("record_uid")["file_path"]

    return {
        "directory": directory,
        "data": data,
        "test_index": test,
        "stored": stored.set_index("record_uid"),
        "paths": matrix,
    }


def test_an_uploaded_record_reproduces_its_stored_out_of_fold_prediction(
    client: Any, fold_bundle: dict[str, Any]
) -> None:
    """The gate: real bytes, over HTTP, to the value EXP-A1 stored.

    Leg A is the wire: the multipart POST the browser makes must return exactly
    what a direct `predict_recording` call returns for the same file.
    Leg B is the science: that same file, against a refit of the fold the stored
    prediction came from, must reproduce the stored probability.
    """
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout")

    data = fold_bundle["data"]
    stored = fold_bundle["stored"]
    paths = fold_bundle["paths"]
    fold_model = load_bundle("binary", path=fold_bundle["directory"])

    checked = 0
    for index in fold_bundle["test_index"]:
        uid = data.record_uids[int(index)]
        if uid not in stored.index:
            continue
        wav = Path(str(paths.loc[uid]))
        if not wav.is_file():
            pytest.skip("dataset/ is not present in this checkout")

        # -- leg A: the wire adds nothing ---------------------------------
        raw = wav.read_bytes()
        response = client.post(
            "/predict",
            files={"file": (wav.name, raw, "audio/wav")},
            data={"task": "binary"},
        )
        assert response.status_code == 200, response.text
        over_http = response.json()
        direct = predict_recording(wav, task="binary").to_dict()
        assert over_http["predicted_class"] == direct["predicted_class"]
        for name, value in direct["probabilities"].items():
            assert over_http["probabilities"][name] == value, (
                "the HTTP path returned a different probability for "
                + name
                + " than predict_recording did on the same file; the API is "
                "transforming a result rather than transporting one"
            )
        assert over_http["source"] == wav.name

        # -- leg B: the predictor reproduces the fold ----------------------
        result = predict_recording(wav, task="binary", bundle=fold_model)
        expected = float(stored.loc[uid, "proba_1"])
        actual = result.probabilities["abnormal"]

        expected_class = "abnormal" if expected >= 0.5 else "normal"
        assert result.predicted_class == expected_class, (
            uid + ": stored " + expected_class + ", the upload path said " + result.predicted_class
        )

        allowed = SINGLE_RECORD_ULP * float(np.spacing(abs(expected)))
        assert abs(actual - expected) <= allowed, (
            uid
            + ": the upload path produced "
            + repr(actual)
            + " where "
            + FOLD_LABEL
            + " stored "
            + repr(expected)
            + " -- "
            + format(abs(actual - expected) / float(np.spacing(abs(expected))), ".1f")
            + " ULP, beyond the "
            + str(SINGLE_RECORD_ULP)
            + " that single-row versus batched BLAS accounts for."
        )
        assert abs(expected - 0.5) > allowed * 1000
        checked += 1
        if checked == 2:
            break

    assert checked == 2, "no held-out record could be carried through the upload path"


def test_the_sample_endpoint_and_the_upload_endpoint_agree(client: Any) -> None:
    """The built-in demo is the upload path, not a shortcut beside it."""
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout")
    sample_id = _first_servable("binary")
    path = resolve_sample(sample_id)
    assert path is not None

    by_sample = client.post(
        "/predict/sample", data={"sample_id": sample_id, "task": "binary"}
    ).json()
    by_upload = client.post(
        "/predict",
        files={"file": (path.name, path.read_bytes(), "audio/wav")},
        data={"task": "binary"},
    ).json()

    assert by_sample["predicted_class"] == by_upload["predicted_class"]
    assert by_sample["probabilities"] == by_upload["probabilities"]
    # Everything except the timings, which are wall clock and are SUPPOSED to
    # differ between two runs. Asserting they matched would be asserting the
    # machine is deterministic about how long work takes.
    shown_sample = {k: v for k, v in by_sample["display"].items() if k != "timings_seconds"}
    shown_upload = {k: v for k, v in by_upload["display"].items() if k != "timings_seconds"}
    assert shown_sample == shown_upload
    assert set(by_sample["display"]["timings_seconds"]) == set(
        by_upload["display"]["timings_seconds"]
    )


# ---------------------------------------------------------------------------
# The pages themselves
# ---------------------------------------------------------------------------


PAGES = (
    APP / "binary" / "page.tsx",
    APP / "multiclass" / "page.tsx",
    APP / "murmur" / "page.tsx",
)


def test_the_three_pages_exist_and_are_no_longer_placeholders() -> None:
    for page in PAGES:
        body = page.read_text(encoding="utf-8")
        assert "PagePlaceholder" not in body, page.name + " is still a placeholder"
        assert "PredictionPanel" in body


def test_no_prediction_component_formats_a_number() -> None:
    """The API rounds; the page renders. Same rule as the chart layer."""
    for path in sorted(COMPONENTS.glob("*.tsx")):
        body = path.read_text(encoding="utf-8")
        for forbidden in ("toFixed(", "toPrecision(", "Math.round("):
            assert forbidden not in body, path.name + " formats a number client-side: " + forbidden


def test_no_page_or_component_hardcodes_a_metric() -> None:
    """The 95.82% rule, applied to the files this phase added.

    The real guard rail is imported rather than reimplemented. A second regex
    here would be a second definition of "metric-shaped", and the two would
    disagree the first time either was tuned: a hand-rolled version of this
    flagged `width: '100%'` in a canvas style on its first run, which the real
    scanner exempts on purpose. CI scans the whole tree; this aims the same
    scanner at the files this phase added, so a regression fails in the test
    that owns them.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "metric_guard", PROJECT_ROOT / "scripts" / "16_check_no_hardcoded_metrics.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    # Registered before exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module loaded by path is not there.
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)

    findings: list[Any] = []
    suppressed: list[Any] = []
    for path in [*PAGES, *sorted(COMPONENTS.glob("*.tsx"))]:
        found, allowed = guard.scan_file(path)
        findings.extend(found)
        suppressed.extend(allowed)
    assert not findings, "\n".join(str(finding) for finding in findings)
    # A suppression is legitimate but must never be quiet. There should be zero.
    assert not suppressed, "these files suppress the guard rail: " + str(suppressed)


def test_the_pages_import_prediction_data_only_from_generated() -> None:
    for path in [*PAGES, *sorted(COMPONENTS.glob("*.tsx"))]:
        body = path.read_text(encoding="utf-8")
        for match in re.finditer(r"from '(@/lib/generated[^']*)'", body):
            assert match.group(1) in (
                "@/lib/generated",
                "@/lib/generated/types",
                "@/lib/generated/prediction",
            ), path.name + " imports generated data from " + match.group(1)


def test_the_payload_is_not_in_the_barrel() -> None:
    """Three routes of fifteen use it; in `index.ts` the other twelve pay for it.

    Measured on the build that introduced it: the shared chunk went from
    125.3 kB to 128.2 kB against a 130 kB budget, and moving the payload to its
    own module gave that back.
    """
    index = (GENERATED / "index.ts").read_text(encoding="utf-8")
    assert "prediction.json" not in index
    assert (GENERATED / "prediction.ts").is_file()


def test_every_generated_json_is_imported_exactly_once() -> None:
    """The same assertion T109.6 makes, re-run for the module this phase added."""
    modules = [path for path in GENERATED.glob("*.ts") if path.name != "types.ts"]
    imported: dict[str, int] = {}
    for module in modules:
        for match in re.finditer(r"from '\./([A-Za-z_]+\.json)'", module.read_text("utf-8")):
            imported[match.group(1)] = imported.get(match.group(1), 0) + 1
    for name, count in imported.items():
        assert count == 1, name + " is imported by " + str(count) + " generated modules"
    assert imported.get("prediction.json") == 1


def test_the_result_card_renders_the_disclaimer_and_the_low_confidence_warning() -> None:
    """T116.2: on every result, from the component rather than from a page."""
    body = (COMPONENTS / "ResultCard.tsx").read_text(encoding="utf-8")
    assert "result.disclaimer" in body
    assert "low_confidence" in body
    assert 'role="alert"' in body


def test_a_failed_prediction_renders_an_error_not_an_empty_result() -> None:
    for name in ("PredictionPanel.tsx", "PatientPanel.tsx"):
        body = (COMPONENTS / name).read_text(encoding="utf-8")
        assert "ErrorState" in body, name + " has no error affordance"


def test_the_client_posts_the_fields_the_api_declares() -> None:
    """The shape T116.7 drives: `file` and `task`, multipart, to `/predict`."""
    body = (FRONTEND / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "body.append('file', file)" in body
    assert "body.append('task', task)" in body
    assert "'/predict'" in body
    # `.toFixed(` -- the call, not the word: this module's own docstring says
    # there is no toFixed in it, and matching the bare word would flag that.
    assert ".toFixed(" not in body


def test_the_multiclass_page_offers_two_separate_label_spaces() -> None:
    """Never merged: two tasks on one page is not a seven-class selector."""
    body = (APP / "multiclass" / "page.tsx").read_text(encoding="utf-8")
    assert "'pascal_a'" in body
    assert "'pascal_b'" in body
    assert "merges neither" in body


def test_the_batch_panel_exports_csv_from_display_strings() -> None:
    body = (COMPONENTS / "BatchPanel.tsx").read_text(encoding="utf-8")
    assert "display.confidence" in body
    assert "text/csv" in body
    assert "toFixed" not in body
