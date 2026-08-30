"""The fifteen equations of blueprint section 11, as exportable data.

T113.5 asks the dashboard to render all fifteen. They are declared here rather
than typed into a `.tsx` file for the same reason the palette and the pipeline
steps are: a formula written into a page is a claim nothing checks, and a wrong
subscript in a thesis equation is the kind of error that survives every review
because it looks like typography.

Each entry carries four things:

* the LaTeX KaTeX renders;
* the symbol table, so ``\\sigma`` is defined rather than assumed;
* the **file that implements it**, verified to exist at export time, together
  with a symbol expected to appear in that file (T100.5 asks for exactly this
  cross-reference; doing it here means Phase 100 inherits it rather than
  redoing it);
* the blueprint's own words for what the formula is used for.

## Two transcription notes, recorded rather than silently corrected

**RMS.** Section 11 prints ``RMS = [(1/N)Sum x[n]^2]`` with the radical sign
trailing the bracket rather than enclosing it. Read literally that is not a
root-mean-square. It is rendered here as the square root of the mean square,
which is what the formula is called, what the use column says, and what
``time_domain.py`` computes.

**Spectral centroid.** The blueprint writes ``C = Sum f_k X_k / Sum X_k``
without parentheses. Rendered here with the sums explicitly bracketed, since the
unparenthesised form reads as ``Sum (f_k X_k / Sum X_k)``, which is not the
centre of mass.

Neither changes a result. Both are written down because a reader comparing the
dashboard against the blueprint will notice the difference, and should find the
reason here rather than guess at it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = ["EQUATIONS", "Equation", "equations_payload"]

log = get_logger("reporting.equations")


@dataclass(frozen=True)
class Equation:
    """One formula, its symbols, and the code that implements it."""

    number: int
    key: str
    name: str
    #: KaTeX source. Display mode; no surrounding delimiters.
    latex: str
    #: The blueprint's "Use" column, verbatim where it is a sentence.
    use: str
    #: Repository-relative implementing module. Verified to exist.
    implemented_in: str
    #: A name expected to appear in that module. Verified to appear.
    implements: str
    #: ``(symbol_latex, meaning)`` pairs. Every symbol in `latex` is defined.
    symbols: tuple[tuple[str, str], ...]
    #: Set when the rendering departs from the blueprint's typography.
    transcription_note: str | None = None


EQUATIONS: tuple[Equation, ...] = (
    Equation(
        1,
        "butterworth",
        "Butterworth response",
        r"\left|H(j\omega)\right|^{2} = \frac{1}{1 + \left(\dfrac{\omega}{\omega_c}\right)^{2n}}",
        "Preprocessing",
        "src/preprocessing/filters.py",
        "butterworth_bandpass_magnitude",
        (
            (r"H(j\omega)", "frequency response of the filter"),
            (r"\omega", "angular frequency, in radians per second"),
            (r"\omega_c", "cut-off angular frequency"),
            ("n", "filter order"),
        ),
    ),
    Equation(
        2,
        "zscore",
        "Z-score",
        r"z = \frac{x - \mu}{\sigma}",
        "Signal normalization",
        "src/preprocessing/normalize.py",
        "zscore_normalize",
        (
            ("x", "a sample of the signal"),
            (r"\mu", "mean of the signal"),
            (r"\sigma", "standard deviation of the signal"),
        ),
    ),
    Equation(
        3,
        "energy",
        "Energy",
        r"E = \sum_{n} x[n]^{2}",
        "Time-domain feature",
        "src/feature_extraction/time_domain.py",
        "energy",
        (
            ("E", "total signal energy"),
            ("x[n]", "the n-th sample"),
        ),
    ),
    Equation(
        4,
        "rms",
        "RMS",
        r"\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{n} x[n]^{2}}",
        "Effective amplitude",
        "src/feature_extraction/time_domain.py",
        "rms",
        (
            ("N", "number of samples"),
            ("x[n]", "the n-th sample"),
        ),
        transcription_note=(
            "Section 11 prints the radical after the bracket rather than over "
            "it. Rendered as the root of the mean square, which is what the "
            "name, the use column and the implementation all describe."
        ),
    ),
    Equation(
        5,
        "spectral_centroid",
        "Spectral centroid",
        r"C = \frac{\sum_{k} f_k X_k}{\sum_{k} X_k}",
        "Spectral centre of mass",
        "src/feature_extraction/frequency.py",
        "spectral_centroid",
        (
            ("C", "spectral centroid, in hertz"),
            ("f_k", "frequency of the k-th bin"),
            ("X_k", "magnitude of the k-th bin"),
        ),
        transcription_note=(
            "Bracketed here. The blueprint's unparenthesised form reads as a "
            "sum of ratios, which is not a centre of mass."
        ),
    ),
    Equation(
        6,
        "spectral_entropy",
        "Spectral entropy",
        r"H_s = -\sum_{k} p_k \log p_k",
        "Spectral irregularity",
        "src/feature_extraction/frequency.py",
        "spectral_entropy",
        (
            ("H_s", "spectral entropy"),
            ("p_k", "normalized power in the k-th bin, summing to one"),
        ),
    ),
    Equation(
        7,
        "soft_voting",
        "Soft voting",
        r"P_{\text{final}}(c) = \frac{\sum_{m} w_m P_m(c)}{\sum_{m} w_m}",
        "Weighted ensemble probability",
        "src/ensemble/soft_voting.py",
        "fuse_probabilities",
        (
            ("c", "a class"),
            ("m", "an ensemble member: SVM, Random Forest or Gradient Boosting"),
            ("w_m", "the weight of member m, non-negative"),
            ("P_m(c)", "member m's probability for class c"),
        ),
    ),
    Equation(
        8,
        "prediction",
        "Prediction",
        r"\hat{y} = \arg\max_{c} P_{\text{final}}(c)",
        "Final class",
        "src/evaluation/metrics.py",
        "argmax",
        (
            (r"\hat{y}", "the predicted class"),
            (r"P_{\text{final}}(c)", "the fused probability for class c"),
        ),
        transcription_note=(
            "For the binary task the deployed rule is not argmax at 0.5: T50.4 "
            "selects the decision threshold inside the training fold against "
            "sensitivity and balanced accuracy. This equation describes the "
            "multiclass rule and the fixed-threshold binary reference."
        ),
    ),
    Equation(
        9,
        "accuracy",
        "Accuracy",
        r"\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}",
        "Overall correctness",
        "src/evaluation/metrics.py",
        "binary_metrics",
        (
            ("TP", "true positives: abnormal, predicted abnormal"),
            ("TN", "true negatives: normal, predicted normal"),
            ("FP", "false positives: normal, predicted abnormal"),
            ("FN", "false negatives: abnormal, predicted normal"),
        ),
    ),
    Equation(
        10,
        "sensitivity",
        "Sensitivity",
        r"\mathrm{Sensitivity} = \frac{TP}{TP + FN}",
        "Abnormal-case detection",
        "src/evaluation/metrics.py",
        "binary_metrics",
        (
            ("TP", "true positives"),
            ("FN", "false negatives: the abnormal cases that were missed"),
        ),
    ),
    Equation(
        11,
        "specificity",
        "Specificity",
        r"\mathrm{Specificity} = \frac{TN}{TN + FP}",
        "Normal-case detection",
        "src/evaluation/metrics.py",
        "specificity_score",
        (
            ("TN", "true negatives"),
            ("FP", "false positives"),
        ),
    ),
    Equation(
        12,
        "f1",
        "F1-score",
        r"F_1 = \frac{2PR}{P + R}",
        "Precision-recall balance",
        "src/evaluation/metrics.py",
        "binary_metrics",
        (
            ("P", r"precision, $TP/(TP+FP)$"),
            ("R", r"recall, which is sensitivity, $TP/(TP+FN)$"),
        ),
    ),
    Equation(
        13,
        "balanced_accuracy",
        "Balanced accuracy",
        r"\mathrm{BA} = \frac{\mathrm{Sensitivity} + \mathrm{Specificity}}{2}",
        "Imbalance-aware binary metric",
        "src/evaluation/metrics.py",
        "binary_metrics",
        ((r"\mathrm{BA}", "balanced accuracy"),),
    ),
    Equation(
        14,
        "macro_f1",
        "Macro-F1",
        r"\text{Macro-}F_1 = \frac{1}{|C|}\sum_{c \in C} F_1(c)",
        "Multiclass evaluation",
        "src/optimization/multi_objective.py",
        "macro_f1",
        (
            ("C", "the set of classes"),
            ("F_1(c)", "the F1 score of class c taken as positive"),
        ),
    ),
    Equation(
        15,
        "objective_j",
        "Multi-objective score",
        r"J = \alpha\left(1 - \text{Macro-}F_1\right) "
        r"+ \beta\left(\frac{\text{SelectedFeatures}}{138}\right) "
        r"+ \gamma\left(\text{NormalizedInferenceTime}\right)",
        "Optimization objective",
        "src/optimization/multi_objective.py",
        "score_j",
        (
            ("J", "the cost being minimised"),
            (r"\alpha, \beta, \gamma", "weights on accuracy, compactness and speed"),
            (r"\text{SelectedFeatures}", "size of the selected subset, out of 138"),
            (
                r"\text{NormalizedInferenceTime}",
                "inference time scaled against the cost model",
            ),
        ),
    ),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def equations_payload() -> dict[str, Any]:
    """The fifteen equations, each cross-checked against the implementation.

    Raises if an equation names a module that does not exist, or a symbol that
    does not appear in it. An equation the code does not implement is a formula
    in a brochure.
    """
    root = _project_root()
    items: list[dict[str, Any]] = []
    for equation in EQUATIONS:
        module = root / equation.implemented_in
        if not module.is_file():
            raise FileNotFoundError(
                "equation "
                + str(equation.number)
                + " ("
                + equation.key
                + ") names "
                + equation.implemented_in
                + ", which does not exist"
            )
        body = module.read_text(encoding="utf-8", errors="replace")
        if equation.implements not in body:
            raise ValueError(
                "equation "
                + str(equation.number)
                + " ("
                + equation.key
                + ") claims "
                + equation.implemented_in
                + " implements it via "
                + equation.implements
                + ", which does not appear in that file"
            )
        items.append(
            {
                "number": equation.number,
                "key": equation.key,
                "name": equation.name,
                "latex": equation.latex,
                "use": equation.use,
                "implemented_in": equation.implemented_in,
                "implements": equation.implements,
                "symbols": [
                    {"symbol": symbol, "meaning": meaning} for symbol, meaning in equation.symbols
                ],
                "transcription_note": equation.transcription_note,
            }
        )

    if [item["number"] for item in items] != list(range(1, len(EQUATIONS) + 1)):
        raise ValueError("the equations are not numbered 1..n in order")

    return {
        "source": "Developer Blueprint, section 11 -- Equations and Optimization Formulas",
        "n_equations": len(items),
        "note": (
            "Declared in src/reporting/equations.py and verified at export time: "
            "every equation below names a module that exists and a symbol that "
            "appears in it. Two entries depart from the blueprint's typography; "
            "each says so and why."
        ),
        "equations": items,
    }
