"""Phase 100 gate: every section 11 formula, checked numerically against the code that runs it.

T100.2 and T100.7 ask two things of the equations reference:

* **each implemented formula agrees numerically with its documented
  definition.** Every test below computes the formula by hand from its printed
  definition -- with numpy, never with the project's own helper -- and compares
  it with what the implementation returns on the same input;
* **each equation cross-references the file and line implementing it.** The
  line is resolved at export time from a fragment of the code
  (``CODE_REFERENCES``); the tests check the fragment is really on that line,
  that the line sits in the declared file, and that a fragment which has left
  the code fails the export.

The exported LaTeX and DOCX are read back: fifteen numbered LaTeX equations,
fifteen native Office Math paragraphs, each carrying its ``path:line``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.reporting.equations import EQUATIONS
from src.reporting.equations_reference import (
    CODE_REFERENCES,
    EquationReferenceError,
    cross_reference,
    reference_rows,
    write_equations_reference,
)

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(42)


def _equation(key: str):
    return next(equation for equation in EQUATIONS if equation.key == key)


# ---------------------------------------------------------------------------
# T100.5 -- file and line
# ---------------------------------------------------------------------------


def test_every_equation_resolves_to_a_line_holding_its_code() -> None:
    assert set(CODE_REFERENCES) == {equation.key for equation in EQUATIONS}
    for equation in EQUATIONS:
        reference = cross_reference(equation)
        lines = (ROOT / reference.path).read_text(encoding="utf-8").splitlines()
        line = " ".join(lines[reference.line - 1].split())
        assert " ".join(CODE_REFERENCES[equation.key].split()) in line, equation.key
        assert reference.path == equation.implemented_in
        body = (ROOT / reference.path).read_text(encoding="utf-8")
        assert equation.implements in body, equation.key


def test_a_fragment_that_left_the_code_fails_the_export(monkeypatch) -> None:
    from src.reporting import equations_reference as module

    monkeypatch.setitem(module.CODE_REFERENCES, "energy", "energy = sum_of_squares(x)")
    with pytest.raises(EquationReferenceError, match="no longer occurs"):
        cross_reference(_equation("energy"))


# ---------------------------------------------------------------------------
# T100.2 / T100.7 -- each formula, numerically
# ---------------------------------------------------------------------------


def test_eq1_butterworth_matches_the_documented_response_and_the_real_filter() -> None:
    from scipy.signal import butter, sosfreqz

    from src.preprocessing import filters as flt

    fs, low, high, n = 2000, flt.DEFAULT_LOW_HZ, flt.DEFAULT_HIGH_HZ, flt.DEFAULT_ORDER
    freqs = np.linspace(1.0, fs / 2 - 1.0, 400)
    # |H|^2 = 1 / (1 + (W/Wc)^2n), with the band-pass substitution and prewarp.
    warp = 2.0 * fs * np.tan(np.pi * freqs / fs)
    w1, w2 = (2.0 * fs * np.tan(np.pi * f / fs) for f in (low, high))
    transformed = (warp**2 - w1 * w2) / (warp * (w2 - w1))
    by_hand = 1.0 / np.sqrt(1.0 + transformed ** (2 * n))

    implemented = flt.butterworth_bandpass_magnitude(freqs, fs, low, high, n)
    np.testing.assert_allclose(implemented, by_hand, rtol=0, atol=1e-12)
    _, response = sosfreqz(
        butter(n, [low, high], btype="bandpass", fs=fs, output="sos"), freqs, fs=fs
    )
    np.testing.assert_allclose(implemented, np.abs(response), rtol=0, atol=1e-9)


def test_eq2_zscore() -> None:
    from src.preprocessing.normalize import zscore_normalize

    x = RNG.normal(3.0, 2.5, 5000)
    z, flagged = zscore_normalize(x)
    assert not flagged
    np.testing.assert_allclose(z, (x - x.mean()) / x.std(), rtol=0, atol=1e-6)


def test_eq3_eq4_energy_and_rms() -> None:
    from src.feature_extraction.time_domain import extract_time_features

    x = RNG.normal(0.2, 0.7, 6000)
    values = extract_time_features(x, 2000).values
    assert values["time_energy"] == pytest.approx(float(np.sum(x**2)), rel=1e-12)
    assert values["time_rms"] == pytest.approx(float(np.sqrt(np.mean(x**2))), rel=1e-12)


def test_eq5_spectral_centroid_is_the_frame_centre_of_mass() -> None:
    from src.feature_extraction.frequency import FrequencyExtractor

    fs = 2000
    t = np.arange(8000) / fs
    x = (
        np.sin(2 * np.pi * 60 * t)
        + 0.4 * np.sin(2 * np.pi * 250 * t)
        + 0.1 * RNG.normal(size=t.size)
    )
    extractor = FrequencyExtractor()
    magnitude, n_fft = extractor._stft_magnitude(x, [])
    f_k = np.fft.rfftfreq(n_fft, 1.0 / fs)
    per_frame = (f_k[:, None] * magnitude).sum(axis=0) / magnitude.sum(axis=0)
    values = extractor.extract(x, fs).values
    assert values["freq_centroid_mean"] == pytest.approx(float(per_frame.mean()), rel=1e-6)


def test_eq6_spectral_entropy_is_shannon_entropy_normalised() -> None:
    from src.feature_extraction.frequency import spectral_entropy

    psd = RNG.random(129)
    p = psd / psd.sum()
    documented = -np.sum(p * np.log(p))  # H_s = -sum p_k log p_k, natural log
    # The implementation is the same entropy in bits, divided by log2(N).
    assert spectral_entropy(psd) == pytest.approx(documented / np.log(psd.size), rel=1e-12)


def test_eq7_soft_voting_is_the_weight_normalised_average() -> None:
    from src.ensemble.soft_voting import fuse_probabilities

    stack = RNG.dirichlet(np.ones(3), size=(4, 50))  # (members, samples, classes)
    weights = np.array([2.0, 0.5, 1.0, 0.0])
    by_hand = sum(w * p for w, p in zip(weights, stack, strict=True)) / weights.sum()
    np.testing.assert_allclose(fuse_probabilities(stack, weights), by_hand, rtol=0, atol=1e-15)


def test_eq8_multiclass_prediction_is_the_argmax_of_the_fused_probability() -> None:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    from src.ensemble.soft_voting import SoftVotingEnsemble

    x, y = make_classification(
        n_samples=120, n_features=5, n_informative=3, n_classes=3, random_state=42
    )
    model = SoftVotingEnsemble(
        [("lr", LogisticRegression(max_iter=500)), ("nb", GaussianNB())]
    ).fit(x, y)
    np.testing.assert_array_equal(
        model.predict(x), model.classes_[np.argmax(model.predict_proba(x), axis=1)]
    )


def test_eq9_to_eq13_binary_metrics_from_the_confusion_counts() -> None:
    from src.evaluation.metrics import binary_metrics, specificity_score

    y = (RNG.random(400) < 0.3).astype(int)
    pred = np.where(RNG.random(400) < 0.8, y, 1 - y)
    tp = float(((y == 1) & (pred == 1)).sum())
    tn = float(((y == 0) & (pred == 0)).sum())
    fp = float(((y == 0) & (pred == 1)).sum())
    fn = float(((y == 1) & (pred == 0)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    precision = tp / (tp + fp)

    m = binary_metrics(y, pred)
    assert m["accuracy"] == pytest.approx((tp + tn) / (tp + tn + fp + fn))
    assert m["sensitivity"] == pytest.approx(sens)
    assert m["specificity"] == pytest.approx(spec)
    assert specificity_score(y, pred, (0, 1), 1) == pytest.approx(spec)
    assert m["f1"] == pytest.approx(2 * precision * sens / (precision + sens))
    assert m["balanced_accuracy"] == pytest.approx((sens + spec) / 2)


def test_eq14_macro_f1_is_the_mean_of_per_class_f1() -> None:
    from src.optimization.multi_objective import macro_f1

    y = RNG.integers(0, 3, 300)
    pred = np.where(RNG.random(300) < 0.7, y, RNG.integers(0, 3, 300))
    per_class = []
    for c in range(3):
        tp = np.sum((y == c) & (pred == c))
        p, r = tp / np.sum(pred == c), tp / np.sum(y == c)
        per_class.append(2 * p * r / (p + r))
    assert macro_f1(y, pred) == pytest.approx(float(np.mean(per_class)), rel=1e-12)


def test_eq15_j_is_the_documented_weighted_sum() -> None:
    from src.feature_extraction.registry import EXPECTED_TOTAL, FAMILY_ORDER, feature_names
    from src.optimization.multi_objective import FamilyCostModel, JWeights, score_j

    weights = JWeights()
    assert weights.n_features_total == EXPECTED_TOTAL  # the 138 the equation divides by
    seconds = {family: float(index + 1) for index, family in enumerate(FAMILY_ORDER)}
    cost = FamilyCostModel(seconds)
    subset = [name for name in feature_names() if name.startswith(("mfcc", "env"))][:17]
    needed = {name.split("_")[0] for name in subset}
    families = [family for family in FAMILY_ORDER if any(family.startswith(n) for n in needed)]
    time_term = sum(seconds[family] for family in families) / sum(seconds.values())

    score = score_j(0.81, subset, weights=weights, cost_model=cost)
    expected = (
        weights.alpha * (1 - 0.81)
        + weights.beta * len(subset) / EXPECTED_TOTAL
        + weights.gamma * time_term
    )
    assert score.value == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# T100.3, T100.4, T100.6 -- the exported files, read back
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    target = tmp_path_factory.mktemp("equations")
    return write_equations_reference(target, evidence_index=target / "evidence_index.csv")


def test_latex_holds_fifteen_labelled_equations_with_their_lines(exported) -> None:
    tex = exported["tex"].read_text(encoding="utf-8")
    assert tex.count(r"\begin{equation}") == len(EQUATIONS) == 15
    assert tex.count("{") == tex.count("}")
    for row in reference_rows():
        assert r"\label{eq:" + row["key"] + "}" in tex
        assert "% implemented in " + row["implemented_in"] + ":" + str(row["line"]) in tex


def test_docx_holds_fifteen_native_equations_and_every_cross_reference(exported) -> None:
    from docx import Document
    from docx.oxml.ns import qn

    document = Document(str(exported["docx"]))
    body = document.element.body
    assert len(list(body.iter(qn("m:oMathPara")))) == 15
    math_text = "".join(node.text or "" for node in body.iter(qn("m:t")))
    assert "\\" not in math_text, "a LaTeX command leaked into the Office Math"
    text = "\n".join(p.text for p in document.paragraphs)
    for row in reference_rows():
        assert row["implemented_in"] + ":" + str(row["line"]) in text
    assert "not a diagnostic tool" in text


def test_the_converter_builds_the_structures_word_expects() -> None:
    from src.reporting.omml import OmmlError, latex_to_omml

    xml = latex_to_omml(_equation("rms").latex)
    assert "<m:rad>" in xml and "<m:f>" in xml and "<m:nary>" in xml
    assert "<m:d>" in latex_to_omml(_equation("butterworth").latex)
    assert "<m:acc>" in latex_to_omml(_equation("prediction").latex)
    with pytest.raises(OmmlError):
        latex_to_omml(r"\unknowncommand{x}")
