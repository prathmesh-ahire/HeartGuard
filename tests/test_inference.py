"""Phase 106's gate: does the deployed path reproduce the experiment?

T106.7 asks that a known training record reproduce its stored out-of-fold
prediction **through the public entry point**. That sentence needs care, because
the obvious way to satisfy it is wrong.

The deployed bundle in `models_saved/binary/final/` was refitted on all 3,240
labelled records, so it has seen every record it could be asked about. Comparing
its probability for record X against X's stored *out-of-fold* probability
compares a training-set prediction with a held-out one; they are different
quantities and the comparison would either fail, or be "fixed" by loosening a
tolerance until it passed -- which would quietly assert that in-sample and
held-out predictions agree.

So the test refits the fold model the stored prediction actually came from --
EXP-A1, fold r0f0, M1, config defaults, the same 2,593 training rows -- saves it
as a bundle, and loads it **through `load_bundle`**. `predict_recording` then
takes a held-out record from its raw WAV to a probability, and that probability
must equal the parquet's to the bit. Every stage is exercised: load, resample,
band-pass, normalise, 138 features, impute, scale, predict.

**Where exact equality holds, and where it provably cannot.**

Features are bit-identical and asserted with `array_equal`. Fold probabilities
recomputed **as a batch** are bit-identical to the parquet and asserted with
`==` over all 647 rows. Class decisions are asserted exactly.

One quantity cannot be exact, and it was measured rather than assumed: the
probability for a **single** recording. `predict_proba` on a 647x138 block and on
a 1x138 row are different BLAS calls -- a GEMM and a GEMV -- which accumulate the
dot product in a different order, so the logit differs in its last bits. Measured
over 40 held-out records of EXP-A1/r0f0/M1: max 8 ULP, 4.44e-16 absolute, **zero
class flips**. That is the same root cause the Part VII re-run found in `brier`
and `ece`, surfacing here because the deployed API scores one upload at a time.

So the single-record assertion is bounded in ULP with that measurement cited, and
the exact assertion is kept at the level where it holds -- batch reproduction --
rather than the whole comparison being loosened to a tolerance.

Everything needing `dataset/` or a `.parquet` skips: both are gitignored, so CI
runs the structural half of this file and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.inference.predictor import (
    DISCLAIMER,
    LOW_CONFIDENCE_MARGIN,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    TASKS,
    AudioValidationError,
    ModelBundle,
    ModelUnavailableError,
    available_tasks,
    load_bundle,
    predict_recording,
    task_report,
    validate_recording,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX = PROJECT_ROOT / "outputs" / "03_features" / "all_features_matrix.parquet"
EXP_DIR = PROJECT_ROOT / "outputs" / "06_binary_results" / "EXP-A1"
BUNDLE_DIR = PROJECT_ROOT / "models_saved" / "binary" / "final"

#: The fold and model whose stored predictions this file reproduces. EXP-A1 is
#: `tuned: false`, so its models take config defaults and a refit needs no
#: search -- which is what makes this reproducible in a test at all.
FOLD_LABEL = "r0f0"
MODEL_ID = "M1"

#: How far a SINGLE-record probability may sit from the batch-computed one.
#:
#: Measured, not chosen: over 40 held-out records of EXP-A1/r0f0/M1 the largest
#: gap was 8 ULP (4.44e-16 absolute) with zero class flips, because scoring one
#: row is a GEMV where scoring the fold is a GEMM and the two accumulate the dot
#: product differently. 32 leaves headroom for another BLAS without letting
#: anything through that a decision could notice: at a probability of 0.9, 32
#: ULP is 3.6e-15.
SINGLE_RECORD_ULP = 32


@pytest.fixture(scope="module")
def deployed_bundle() -> ModelBundle:
    if not (BUNDLE_DIR / "model.joblib").is_file():
        pytest.skip("no saved binary model in this checkout")
    return load_bundle("binary")


@pytest.fixture(scope="module")
def corpus_row() -> Any:
    """One labelled PhysioNet record, with its stored feature row."""
    import pandas as pd

    if not MATRIX.is_file():
        pytest.skip("all_features_matrix.parquet is not in this checkout (gitignored)")
    frame = pd.read_parquet(MATRIX)
    row = frame[frame["record_uid"] == "D1_training-a_a0005"]
    if row.empty:
        pytest.skip("the reference record is not in this matrix")
    record = row.iloc[0]
    if not Path(str(record["file_path"])).is_file():
        pytest.skip("dataset/ is not present in this checkout")
    return record


@pytest.fixture(scope="module")
def refit_fold(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Refit EXP-A1/r0f0/M1 and save it as a loadable bundle.

    Uses `load_task_data`, the same loader the experiment runner used -- not a
    hand-rolled join. Row order is part of the result: `load_task_data` sorts by
    `record_uid`, and fitting the same rows in the parquet's natural order gives
    probabilities that differ in the third decimal. A test that rebuilt the join
    itself would be testing its own reimplementation.
    """
    import joblib
    import pandas as pd

    membership = EXP_DIR / "fold_membership.parquet"
    predictions = EXP_DIR / "predictions.parquet"
    if not membership.is_file() or not predictions.is_file():
        pytest.skip("EXP-A1 prediction parquets are not in this checkout (gitignored)")
    if not MATRIX.is_file():
        pytest.skip("all_features_matrix.parquet is not in this checkout (gitignored)")

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

    directory = tmp_path_factory.mktemp("fold_bundle")
    joblib.dump(built, directory / "model.joblib")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "task": "binary",
                "fold": FOLD_LABEL,
                "n_features": int(data.X.shape[1]),
                "feature_names": list(data.feature_names),
                "note": "refitted in tests/test_inference.py to reproduce a stored fold",
            }
        ),
        encoding="utf-8",
    )

    stored = pd.read_parquet(predictions)
    stored = stored[(stored["model_id"] == MODEL_ID) & (stored["fold_label"] == FOLD_LABEL)]

    return {
        "directory": directory,
        "data": data,
        "test_index": test,
        "stored": stored.set_index("record_uid"),
    }


# ---------------------------------------------------------------------------
# T106.6 / T106.7 -- the gate
# ---------------------------------------------------------------------------


def test_a_held_out_record_reproduces_its_stored_out_of_fold_probability(
    refit_fold: dict[str, Any],
) -> None:
    """The gate. Raw WAV to probability, through the public entry point."""
    import pandas as pd

    data = refit_fold["data"]
    stored = refit_fold["stored"]
    bundle = load_bundle("binary", path=refit_fold["directory"])

    frame = pd.read_parquet(MATRIX)
    paths = frame.set_index("record_uid")["file_path"]

    checked = 0
    for index in refit_fold["test_index"]:
        uid = data.record_uids[int(index)]
        if uid not in stored.index:
            continue
        wav = Path(str(paths.loc[uid]))
        if not wav.is_file():
            pytest.skip("dataset/ is not present in this checkout")

        result = predict_recording(wav, task="binary", bundle=bundle)
        expected = float(stored.loc[uid, "proba_1"])
        actual = result.probabilities["abnormal"]

        # The DECISION is exact. A tolerance has no business here: a class is
        # not a float and cannot drift.
        expected_class = "abnormal" if expected >= 0.5 else "normal"
        assert result.predicted_class == expected_class, (
            uid + ": stored " + expected_class + ", deployed path said " + result.predicted_class
        )

        # The PROBABILITY is bounded in ULP, for the measured reason in the
        # module docstring: one row is a GEMV, the fold was a GEMM.
        allowed = SINGLE_RECORD_ULP * float(np.spacing(abs(expected)))
        assert abs(actual - expected) <= allowed, (
            uid
            + ": the deployed path produced "
            + repr(actual)
            + " where "
            + FOLD_LABEL
            + " stored "
            + repr(expected)
            + " -- a gap of "
            + format(abs(actual - expected) / float(np.spacing(abs(expected))), ".1f")
            + " ULP, beyond the "
            + str(SINGLE_RECORD_ULP)
            + " that single-row versus batched BLAS accounts for. This is a real "
            "divergence in preprocessing, extraction or the pipeline, not float noise."
        )

        # And the decision is not an accident of the tolerance: the probability
        # is nowhere near the threshold compared with how far it may drift.
        assert abs(expected - 0.5) > allowed * 1000
        checked += 1
        if checked == 2:  # two records is the assertion; 647 is a benchmark
            break

    assert checked == 2, "no held-out record could be checked"


def test_the_whole_fold_reproduces_the_stored_probabilities_exactly(
    refit_fold: dict[str, Any],
) -> None:
    """Batch reproduction, asserted with `==` over every held-out row.

    This is the exact half of the gate. The single-record test above is bounded
    in ULP because one row is a different BLAS call; here the fold is scored the
    way the experiment scored it, and the answer must be bit-identical. If this
    ever needs a tolerance, something real has changed -- the refit, the loader's
    row order, or the pipeline -- and no amount of float noise explains it.
    """
    data = refit_fold["data"]
    stored = refit_fold["stored"]
    bundle = load_bundle("binary", path=refit_fold["directory"])

    index = refit_fold["test_index"]
    uids = [data.record_uids[int(i)] for i in index]
    proba = bundle.pipeline.predict_proba(data.X[index])[:, 1]
    expected = stored.loc[uids, "proba_1"].to_numpy(dtype=np.float64)

    assert np.array_equal(proba, expected), (
        "the refitted fold does not reproduce its stored probabilities; max |diff| = "
        + repr(float(np.abs(proba - expected).max()))
    )


def test_the_features_are_the_stored_training_features_bit_for_bit(
    corpus_row: Any,
) -> None:
    """T106.2, stated directly: the inference path recomputes, never reimplements.

    A divergence here is the failure mode this phase exists to prevent -- a
    plausible probability for every upload, wrong for all of them, with nothing
    crashing.
    """
    from src.feature_extraction.extractor import extract_all
    from src.preprocessing.pipeline import preprocess

    prepared = preprocess(
        str(corpus_row["file_path"]), record_uid=str(corpus_row["record_uid"]), use_cache=False
    )
    extracted = extract_all(prepared.signal, prepared.fs, record_uid=str(corpus_row["record_uid"]))

    names = [name for name in extracted.values if name in corpus_row.index]
    assert len(names) == 138, "the registry did not round-trip through the matrix"

    fresh = np.array([extracted.values[name] for name in names], dtype=np.float64)
    stored = corpus_row[names].to_numpy(dtype=np.float64)
    assert np.array_equal(fresh, stored, equal_nan=True), (
        "the features computed at inference differ from the ones the models were "
        "fitted on; max |diff| = " + repr(float(np.nanmax(np.abs(fresh - stored))))
    )


def test_two_predictions_of_the_same_file_are_identical(
    corpus_row: Any, deployed_bundle: ModelBundle
) -> None:
    """Research rule 5 has no exception for inference."""
    first = predict_recording(str(corpus_row["file_path"]), bundle=deployed_bundle)
    second = predict_recording(str(corpus_row["file_path"]), bundle=deployed_bundle)
    assert first.probabilities == second.probabilities
    assert first.predicted_class == second.predicted_class


# ---------------------------------------------------------------------------
# T106.3 -- the structured result
# ---------------------------------------------------------------------------


def test_the_result_carries_everything_t106_3_asks_for(
    corpus_row: Any, deployed_bundle: ModelBundle
) -> None:
    result = predict_recording(str(corpus_row["file_path"]), bundle=deployed_bundle)

    assert result.predicted_class in TASKS["binary"].classes
    assert set(result.probabilities) == set(TASKS["binary"].classes)
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert isinstance(result.low_confidence, bool)
    for stage in ("validate", "load_model", "preprocess", "extract", "predict", "total"):
        assert stage in result.timings_seconds, "no timing for stage " + stage
        assert result.timings_seconds[stage] >= 0
    assert result.n_features == 138
    assert result.model["model_id"]
    assert result.disclaimer == DISCLAIMER


def test_the_result_states_its_operating_point(
    corpus_row: Any, deployed_bundle: ModelBundle
) -> None:
    """The deployed bundle has no in-fold threshold, and must not imply one."""
    result = predict_recording(str(corpus_row["file_path"]), bundle=deployed_bundle)
    assert result.operating_threshold == 0.5
    assert "no in-fold selected threshold" in result.operating_point_note
    predicted_from_threshold = "abnormal" if result.probabilities["abnormal"] >= 0.5 else "normal"
    assert result.predicted_class == predicted_from_threshold


def test_low_confidence_is_a_margin_not_a_probability_floor() -> None:
    """0.40 wins a three-class task and is a coin flip on a two-class one."""
    assert 0 < LOW_CONFIDENCE_MARGIN < 0.5


def test_the_result_serialises_without_numpy_scalars(
    corpus_row: Any, deployed_bundle: ModelBundle
) -> None:
    """T108.3 encodes; the result should not need rescuing first."""
    payload = predict_recording(str(corpus_row["file_path"]), bundle=deployed_bundle).to_dict()
    text = json.dumps(payload)
    for token in ("NaN", "Infinity"):
        assert token not in text, "a non-finite value reached the JSON payload"


# ---------------------------------------------------------------------------
# T106.4 -- task routing
# ---------------------------------------------------------------------------


def test_all_five_label_spaces_are_declared_and_kept_apart() -> None:
    """Research rule 4: five tasks, five targets, never merged."""
    assert set(TASKS) == {"binary", "pascal_a", "pascal_b", "murmur", "outcome"}
    assert TASKS["pascal_a"].classes == ("normal", "murmur", "extrahls", "artifact")
    assert TASKS["pascal_b"].classes == ("normal", "murmur", "extrastole")
    assert TASKS["murmur"].classes == ("Absent", "Present", "Unknown")
    assert TASKS["outcome"].classes == ("Normal", "Abnormal")
    # No two tasks share a model directory, which is how one would end up
    # serving another's label space.
    directories = [spec.model_dir for spec in TASKS.values()]
    assert len(set(directories)) == len(directories)


def test_pascal_a_says_artifact_is_not_a_cardiac_class() -> None:
    """The blueprint's most dangerous omission; CLAUDE.md names it."""
    description = TASKS["pascal_a"].description.lower()
    assert "recording-quality" in description
    assert "not a four-class cardiac" in description


def test_an_unbuilt_task_reports_why_rather_than_disappearing() -> None:
    rows = {row["task"]: row for row in task_report()}
    assert set(rows) == set(TASKS)
    for key, row in rows.items():
        if row["available"]:
            assert row["reason"] is None
        else:
            assert row["reason"], key + " is unavailable with no reason given"
            assert "has not been run" in row["reason"]


def test_asking_for_an_unbuilt_task_names_the_task(tmp_path: Path) -> None:
    unbuilt = [key for key in TASKS if key not in available_tasks()]
    if not unbuilt:
        pytest.skip("every task has a saved model in this checkout")
    with pytest.raises(ModelUnavailableError, match=unbuilt[0]):
        load_bundle(unbuilt[0])


def test_an_unknown_task_is_rejected() -> None:
    with pytest.raises(ModelUnavailableError, match="unknown task"):
        load_bundle("not_a_task")


# ---------------------------------------------------------------------------
# T106.5 -- input validation
# ---------------------------------------------------------------------------


def _write_wav(path: Path, seconds: float, fs: int = 4000, channels: int = 1) -> Path:
    import soundfile as sf

    from tests.fixtures.make_synthetic_pcg import make_synthetic_pcg

    if seconds <= 0:
        data = np.zeros((0, channels), dtype=np.float32)
    else:
        signal = make_synthetic_pcg(duration_sec=seconds, fs=fs).signal.astype(np.float32)
        data = signal if channels == 1 else np.column_stack([signal] * channels)
    sf.write(str(path), data, fs, subtype="PCM_16")
    return path


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(AudioValidationError, match="no such file"):
        validate_recording(tmp_path / "absent.wav")


def test_a_non_wav_file_is_refused_with_the_reason(tmp_path: Path) -> None:
    path = tmp_path / "recording.mp3"
    path.write_bytes(b"not really an mp3")
    with pytest.raises(AudioValidationError, match="not a WAV file"):
        validate_recording(path)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(AudioValidationError, match="empty"):
        validate_recording(path)


def test_a_file_that_is_not_audio_reports_the_decoder_message(tmp_path: Path) -> None:
    path = tmp_path / "text.wav"
    path.write_text("this is not a waveform", encoding="utf-8")
    with pytest.raises(AudioValidationError, match="could not be read as audio"):
        validate_recording(path)


def test_a_recording_shorter_than_the_corpus_minimum_is_refused(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "tiny.wav", 0.2)
    with pytest.raises(AudioValidationError, match="shortest recording"):
        validate_recording(path)


def test_a_recording_longer_than_the_corpus_maximum_is_refused(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "huge.wav", MAX_DURATION_SECONDS + 5.0, fs=1000)
    with pytest.raises(AudioValidationError, match="beyond the"):
        validate_recording(path)


def test_the_bounds_come_from_the_corpus_not_from_taste() -> None:
    """0.763 s and 121.998 s are the real extremes; the bounds bracket them."""
    from tests.fixtures.make_synthetic_pcg import duration_extremes

    extremes = duration_extremes()
    assert extremes["min_sec"] >= MIN_DURATION_SECONDS
    assert extremes["max_sec"] <= MAX_DURATION_SECONDS


def test_a_valid_wav_reports_what_was_read(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "ok.wav", 5.0, fs=4000)
    info = validate_recording(path)
    assert info["sample_rate_hz"] == 4000
    assert info["channels"] == 1
    assert info["duration_seconds"] == pytest.approx(5.0, abs=0.01)


def test_a_stereo_upload_is_downmixed_rather_than_refused(tmp_path: Path) -> None:
    """Mono conversion lives in the shared preprocessing path, not here."""
    path = _write_wav(tmp_path / "stereo.wav", 5.0, channels=2)
    info = validate_recording(path)
    assert info["channels"] == 2

    from src.preprocessing.pipeline import preprocess

    prepared = preprocess(path, use_cache=False)
    assert prepared.signal.ndim == 1


def test_a_synthetic_recording_predicts_without_crashing(
    tmp_path: Path, deployed_bundle: ModelBundle
) -> None:
    """Not an accuracy claim -- a synthetic signal has no true label.

    It asserts only that a well-formed recording the corpus never contained goes
    through every stage and produces a structurally valid result.
    """
    path = _write_wav(tmp_path / "synthetic.wav", 6.0)
    result = predict_recording(path, bundle=deployed_bundle)
    assert result.predicted_class in TASKS["binary"].classes
    assert 0.0 <= result.confidence <= 1.0
    assert result.n_missing_features <= 138


# ---------------------------------------------------------------------------
# T106.2 -- structural: no reimplementation
# ---------------------------------------------------------------------------


def test_the_predictor_calls_the_shared_pipeline_rather_than_its_own() -> None:
    """Grep-level, but it is the invariant the whole phase rests on."""
    source = (PROJECT_ROOT / "src" / "inference" / "predictor.py").read_text(encoding="utf-8")
    assert "from src.preprocessing.pipeline import preprocess" in source
    assert "from src.feature_extraction.extractor import extract_all" in source
    for forbidden in ("butter(", "sosfiltfilt", "librosa.feature", "resample("):
        assert forbidden not in source, (
            "predictor.py appears to implement " + forbidden + " itself instead of "
            "calling the code path the training runs used"
        )


def test_the_column_order_comes_from_the_manifest(deployed_bundle: ModelBundle) -> None:
    """A pipeline cannot notice that two columns arrived swapped."""
    assert len(deployed_bundle.feature_names) == 138
    assert len(set(deployed_bundle.feature_names)) == 138

    from src.feature_extraction.registry import feature_names

    assert list(deployed_bundle.feature_names) == list(feature_names())


def test_a_vector_is_built_by_name_not_by_position() -> None:
    from src.inference.predictor import _vector_for

    values = {"b": 2.0, "a": 1.0, "c": float("nan")}
    vector, missing = _vector_for(values, ("a", "b", "c", "d"))
    assert vector[0] == 1.0 and vector[1] == 2.0
    assert np.isnan(vector[2]) and np.isnan(vector[3])
    assert missing == ["c", "d"]
