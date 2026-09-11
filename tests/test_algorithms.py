"""Phase 98 gate: ALG-01 to ALG-10 exist in DOCX and plain text, and do what the code does.

T98.6 asks that every pseudocode block match the implementation "line for line
in behaviour"; T98.7 that three be spot-checked against it. The first half is
structural and covers all ten:

* every step cites a function that exists, resolved to a line in the repository;
* every parameter shown is re-read here from the config key it claims, and must
  equal the live value;
* no template types a number -- the numbers live in the parameter block;
* every formula the pseudocode quotes is asserted, at export, to still be in
  the function that computes it.

The second half runs the code. Five algorithms are re-implemented in this file
*from their rendered pseudocode* -- using the parameters the export printed --
and the result is compared with what the pipeline itself produces: ALG-03
(filter and z-score, including both short-signal branches), ALG-05 (what the
fold-safe pipeline learns, and from which rows), ALG-06 (the one-standard-error
guard and the consensus tie rule), ALG-08 (calibration fold count, averaging,
and argmax agreement) and ALG-10 (the balanced sample weights). T98.7 asks for
three; the extra two are the ones where a wording slip would change a result.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src.reporting.algorithms import (
    ALGORITHMS,
    AlgorithmError,
    _quoted,
    algorithm_for,
    render,
    resolve_reference,
    write_algorithms,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE_98 = ["ALG-" + str(index).zfill(2) for index in range(1, 11)]


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Path]]:
    target = tmp_path_factory.mktemp("algorithms")
    return write_algorithms(PHASE_98, target, evidence_index=target / "evidence_index.csv")


def _params(alg_id: str) -> dict[str, str]:
    return {p.name: p.value for p in render(algorithm_for(alg_id)).parameters}


# ---------------------------------------------------------------------------
# T98.7 -- they exist, in both formats, and say the same thing in both
# ---------------------------------------------------------------------------


def test_the_catalogue_declares_alg_01_to_alg_10_in_order() -> None:
    ids = [algorithm.alg_id for algorithm in ALGORITHMS]
    assert ids[: len(PHASE_98)] == PHASE_98
    for algorithm in ALGORITHMS[: len(PHASE_98)]:
        assert algorithm.task.startswith("T98."), algorithm.alg_id


@pytest.mark.parametrize("alg_id", PHASE_98)
def test_each_exists_as_text_and_docx_with_the_same_steps(
    alg_id: str, exported: dict[str, dict[str, Path]]
) -> None:
    from docx import Document

    paths = exported[alg_id]
    text = paths["txt"].read_text(encoding="utf-8")
    assert text.startswith("Algorithm " + alg_id + ": ")
    assert "PV-MEPCG / PulseVision" in text
    document = "\n".join(p.text for p in Document(str(paths["docx"])).paragraphs)
    rendered = render(algorithm_for(alg_id))
    assert len(rendered.lines) >= 5
    for _, line, _ in rendered.lines:
        assert line in text, alg_id + " text is missing: " + line
        assert line in document, alg_id + " docx is missing: " + line


def test_the_index_lists_every_exported_algorithm(exported: dict[str, dict[str, Path]]) -> None:
    import csv

    index = next(iter(exported.values()))["txt"].parent / "algorithm_index.csv"
    with index.open(encoding="utf-8", newline="") as handle:
        rows = {row["alg_id"]: row for row in csv.DictReader(handle)}
    assert sorted(rows) == PHASE_98
    for row in rows.values():
        assert row["implements"] and len(row["source_sha256"]) == 64


# ---------------------------------------------------------------------------
# T98.6 -- structural agreement, all ten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alg_id", PHASE_98)
def test_every_step_cites_a_function_that_exists(alg_id: str) -> None:
    algorithm = algorithm_for(alg_id)
    assert all(step.ref for step in algorithm.steps), alg_id + " has an uncited step"
    for reference in algorithm.references():
        path, line = resolve_reference(reference)
        assert (ROOT / path).is_file() and line >= 1


_NUMBER = re.compile(r"(?<![\w\-.{])(\d+(?:\.\d+)?)(?![\w}])")


@pytest.mark.parametrize("alg_id", PHASE_98)
def test_no_template_types_a_number(alg_id: str) -> None:
    """Numbers come from the parameter block; 0 and 1 are arithmetic, not data."""
    algorithm = algorithm_for(alg_id)
    texts = [algorithm.purpose, *algorithm.inputs, *algorithm.outputs, *algorithm.notes]
    texts += [step.text for step in algorithm.steps]
    for text in texts:
        stray = [n for n in _NUMBER.findall(text) if n not in {"0", "1"}]
        assert not stray, alg_id + " types " + str(stray) + " in: " + text


@pytest.mark.parametrize("alg_id", PHASE_98)
def test_every_parameter_shown_is_the_live_configured_value(alg_id: str) -> None:
    from src.reporting.algorithms import _show
    from src.utils.config import load_config

    checked = 0
    for parameter in render(algorithm_for(alg_id)).parameters:
        match = re.fullmatch(r"configs/(\w+)\.yaml ([\w.]+)", parameter.source)
        if not match:
            continue
        value = load_config(match.group(1)).require(match.group(2))
        if isinstance(value, dict):
            continue
        assert parameter.value == _show(value), parameter
        checked += 1
    assert checked or alg_id in {"ALG-01", "ALG-04"}


def test_a_quoted_fragment_that_left_the_code_is_an_error() -> None:
    with pytest.raises(AlgorithmError, match="no longer contains"):
        _quoted("src.preprocessing.io:resample_ratio", "Fraction(fs_in, fs_out)")


def test_a_reference_to_a_missing_function_is_an_error() -> None:
    with pytest.raises(AlgorithmError):
        resolve_reference("src.preprocessing.filters:no_such_function")


# ---------------------------------------------------------------------------
# T98.7 -- behavioural spot-checks: the pseudocode, run, against the code
# ---------------------------------------------------------------------------


def _signal(n: int, fs: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    return (
        0.6 * np.sin(2 * np.pi * 50 * t)
        + 0.3 * np.sin(2 * np.pi * 700 * t)
        + 0.2 * rng.standard_normal(n)
        + 0.4
    ).astype(np.float32)


def _alg03_by_hand(y: np.ndarray, p: dict[str, str]) -> np.ndarray:
    """ALG-03 steps 1-12, from the printed parameters only."""
    from scipy.signal import butter, sosfiltfilt

    fs, order = int(p["target_fs"]), int(p["order"])
    sos = butter(
        order, [float(p["low_hz"]), float(p["high_hz"])], btype="bandpass", fs=fs, output="sos"
    )
    padlen = int(p["padlen"])
    signal = y.astype(np.float64)
    if signal.size <= padlen:
        padlen = signal.size - 1
    too_short = padlen < int(p["min_padlen"])
    x_f = signal if too_short else sosfiltfilt(sos, signal, padlen=padlen)
    x_f = np.asarray(x_f, dtype=np.float32).astype(np.float64)
    x_f = x_f - x_f.mean()
    sigma = x_f.std()
    if sigma <= float(p["epsilon"]):
        return y
    return ((x_f - x_f.mean()) / sigma).astype(np.float32)


@pytest.mark.parametrize("n", [6000, 27, 10])
def test_alg03_pseudocode_reproduces_filter_and_normalize(n: int) -> None:
    """A normal record, one exactly at padlen, and one below min_padlen."""
    from src.preprocessing.filters import filter_signal
    from src.preprocessing.normalize import normalize_signal

    p = _params("ALG-03")
    fs = int(p["target_fs"])
    y = _signal(n, fs)
    pipeline = normalize_signal(filter_signal(y, fs).signal).signal
    np.testing.assert_allclose(pipeline, _alg03_by_hand(y, p), rtol=0, atol=2e-5)


def test_alg05_pipeline_learns_only_from_the_training_rows() -> None:
    from sklearn.linear_model import LogisticRegression

    from src.models.pipeline import build_pipeline, fitted_steps

    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 5))
    x[rng.random(x.shape) < 0.1] = np.nan
    y = (rng.random(80) < 0.3).astype(int)
    train, test = np.arange(60), np.arange(60, 80)

    p = _params("ALG-05")
    assert p["imputer"] == "median" and p["scaler"] == "standard"
    pipeline = build_pipeline(LogisticRegression(max_iter=500), y=y[train])
    pipeline.fit(x[train], y[train])
    learned = fitted_steps(pipeline)

    medians = np.nanmedian(x[train], axis=0)
    np.testing.assert_allclose(learned["imputer.statistics_"], medians)
    imputed = np.where(np.isnan(x[train]), medians, x[train])
    np.testing.assert_allclose(learned["scaler.mean_"], imputed.mean(axis=0))
    assert learned["scaler.n_samples_seen_"] == train.size
    assert pipeline[-1].class_weight == p["class_weight"]

    before = {key: np.array(value, copy=True) for key, value in learned.items()}
    pipeline.predict(x[test] * 1000.0)  # transforming test rows must teach it nothing
    for key, value in fitted_steps(pipeline).items():
        np.testing.assert_array_equal(np.asarray(value), before[key])


def test_alg06_guard_then_cheapest_then_smaller_k() -> None:
    """Steps 7-9, worked by hand on a frame whose answer the rule fixes."""
    import pandas as pd

    from src.feature_selection.sweep import choose_configuration, consensus_subset

    rows = []
    # a: best macro-F1, expensive. b: within one SE of a, cheap. c: cheapest of all,
    # but below the guard. d: ties b on J at a larger k.
    for ranker, k, f1s, j in (
        ("a", 60, [0.84, 0.86, 0.88], 0.30),
        ("b", 20, [0.845, 0.850, 0.855], 0.20),
        ("c", 10, [0.70, 0.71, 0.72], 0.10),
        ("d", 40, [0.845, 0.850, 0.855], 0.20),
    ):
        rows += [{"ranker": ranker, "k": k, "macro_f1": f, "j": j} for f in f1s]
    frame = pd.DataFrame(rows)

    n_se = float(_params("ALG-06")["n_se"])
    best = frame.groupby("ranker")["macro_f1"].agg(["mean", "std", "size"]).loc["a"]
    guard = best["mean"] - n_se * best["std"] / np.sqrt(best["size"])
    assert 0.84 < guard < 0.85  # b and d (mean 0.85) are inside, c is not

    ranker, k, detail = choose_configuration(frame, n_standard_errors=n_se)
    assert (ranker, k) == ("b", 20)
    assert detail["unguarded_j_ranker"] == "c"

    chosen, frequency = consensus_subset(
        {"f0": (0, 1, 2), "f1": (0, 1, 3), "f2": (0, 2, 3)}, 2, ["w", "x", "y", "z"]
    )
    # counts: 0 -> 3, then 1, 2, 3 -> 2 each; the tie goes to column 1 by position
    assert chosen == (0, 1) and frequency == {"w": 3, "x": 2}


def test_alg08_calibration_is_the_mean_of_the_configured_folds() -> None:
    from sklearn.datasets import make_classification

    from src.models.estimators import make_m3

    p = _params("ALG-08")
    x, y = make_classification(n_samples=150, n_features=6, weights=[0.75], random_state=42)
    model = make_m3().fit(x, y)
    folds = model.calibrated_.calibrated_classifiers_
    assert len(folds) == int(p["cv"])
    assert model.method == p["method"] and str(model.ensemble).lower() == p["ensemble"]

    proba = model.predict_proba(x)
    np.testing.assert_allclose(proba, np.mean([f.predict_proba(x) for f in folds], axis=0))
    np.testing.assert_array_equal(model.predict(x), model.classes_[np.argmax(proba, axis=1)])
    # Step 4: class_weight reaches the inner SVC, not the calibrator.
    assert model.calibrated_.estimator.class_weight == p["class_weight"]


def test_alg10_weights_are_n_over_k_times_class_count() -> None:
    from sklearn.utils.class_weight import compute_class_weight

    from src.models.estimators import make_m5
    from src.models.weighting import ClassWeightedClassifier, balanced_sample_weight

    y = np.array([0] * 30 + [1] * 10 + [2] * 5)
    by_hand = {c: y.size / (3 * np.sum(y == c)) for c in (0, 1, 2)}
    weights = balanced_sample_weight(y, "balanced")
    np.testing.assert_allclose(weights, [by_hand[c] for c in y])
    reference = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
    np.testing.assert_allclose([by_hand[c] for c in (0, 1, 2)], reference)

    p = _params("ALG-10")
    model = make_m5()
    assert isinstance(model, ClassWeightedClassifier) == (p["implementation"] == "classic")
    assert model.estimator.get_params()["n_estimators"] == int(p["n_estimators"])


def test_alg07_ties_go_to_the_earlier_trial() -> None:
    from src.optimization.base import SearchResult, Trial

    result = SearchResult(
        method="bayes", model_id="M1", task="binary", outer_label="r0f0", repeat=0, fold=0
    )
    result.trials = [
        Trial(index=0, params={"a": 1}, score=0.8),
        Trial(index=1, params={"a": 2}, score=0.8),
    ]
    assert result.best is not None and result.best.index == 0
