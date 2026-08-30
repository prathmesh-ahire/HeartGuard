"""The twelve architecture steps, and the ensemble weights, as exportable data.

T112.3 asks for a scroll-driven walkthrough of "the 12 documented architecture
steps". Nothing in `outputs/` documented them: F01 (T95.1) draws the diagram but
belongs to Part IX, and the preprocessing module's ``PIPELINE_STEPS`` covers six
signal-level steps out of an end-to-end chain that starts at a corpus audit and
ends at a nested cross-validated score.

So the list is declared here, once, and **verified against the repository every
time it is exported**: each step names the module that implements it and the
`outputs/` directory that evidences it, and :func:`pipeline_payload` refuses to
emit a step whose module or directory is absent. A step is therefore a claim the
build checks, not a caption somebody typed into a page. When F01 is drawn in
Phase 95 it should read this list rather than restate it, for the same reason the
dashboard palette is exported rather than retyped.

The ensemble payload is read from SO-05's ``final_weights.json`` and carries one
number that matters more than the weights themselves: **21 of the 25 outer folds
chose weights identical to equal weighting.** A visualization that animates three
dramatically different weights into a vote would be showing something the search
did not find. The payload says so, and the component renders it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.reporting.tables import content_digest
from src.utils.logging_setup import get_logger

__all__ = [
    "ARCHITECTURE_STEPS",
    "ENSEMBLE_SOURCE",
    "ArchitectureStep",
    "ensemble_payload",
    "pipeline_payload",
]

log = get_logger("reporting.architecture")


@dataclass(frozen=True)
class ArchitectureStep:
    """One step of the pipeline, and the two things that make it checkable."""

    index: int
    key: str
    title: str
    summary: str
    #: Repository-relative path to the module that implements the step. Verified.
    module: str
    #: `outputs/` subdirectory holding this step's evidence. Verified.
    evidence_dir: str
    #: The research rule this step is where we keep, if it is one of them.
    rule: str | None = None


#: The twelve steps, in execution order. Every module path and evidence
#: directory below is asserted to exist at export time.
ARCHITECTURE_STEPS: tuple[ArchitectureStep, ...] = (
    ArchitectureStep(
        1,
        "ingest",
        "Corpus ingest and audit",
        "Four public corpora are opened, counted and checksummed against what "
        "their documentation claims. Where the two disagree, the files win and "
        "the discrepancy is recorded rather than reconciled quietly.",
        "src/data_loader/inventory.py",
        "01_dataset_audit",
    ),
    ArchitectureStep(
        2,
        "grouping",
        "Label harmonisation and subject grouping",
        "Five label spaces are kept separate, never merged, and every recording "
        "is attached to a subject where one can be derived. The subject, not the "
        "recording, is what a fold splits on.",
        "src/data_loader/splits.py",
        "01_dataset_audit",
        rule="No subject leakage: a subject never appears in both train and test.",
    ),
    ArchitectureStep(
        3,
        "resample",
        "Mono conversion and resampling to 2 kHz",
        "Recordings arrive at 44.1 kHz, 4 kHz and 2 kHz. All are reduced to mono "
        "and resampled to a common 2 kHz, which retains the whole diagnostic "
        "band of a heart sound and discards the rest.",
        "src/preprocessing/io.py",
        "02_preprocessing",
    ),
    ArchitectureStep(
        4,
        "quality",
        "Signal quality assessment",
        "Clipping, silence and a signal-to-noise proxy are measured here, in the "
        "middle of the chain, because the band-pass deletes the out-of-band term "
        "the proxy needs and normalisation deletes the levels the other two are "
        "defined against.",
        "src/preprocessing/quality.py",
        "02_preprocessing",
    ),
    ArchitectureStep(
        5,
        "filter",
        "Band-pass filtering",
        "A zero-phase band-pass keeps the murmur band and removes mains hum, "
        "handling noise and the low-frequency wander that breathing and handling "
        "put into a chest recording.",
        "src/preprocessing/filters.py",
        "02_preprocessing",
    ),
    ArchitectureStep(
        6,
        "normalize",
        "Amplitude normalisation",
        "Recordings differ by stethoscope, gain and chest wall. Normalisation "
        "removes the loudness difference so that a feature describes the sound "
        "rather than the microphone.",
        "src/preprocessing/normalize.py",
        "02_preprocessing",
    ),
    ArchitectureStep(
        7,
        "features",
        "Feature extraction, 138 across six families",
        "Time-domain, spectral, MFCC, chroma, wavelet and envelope descriptors "
        "are computed per recording. The registry is fixed: the same 138 columns "
        "in the same order for every corpus and every experiment.",
        "src/feature_extraction/extractor.py",
        "03_features",
    ),
    ArchitectureStep(
        8,
        "fold_local",
        "Fold-local imputation and scaling",
        "The imputer and the scaler are fitted on the training fold only and "
        "then applied to the held-out fold. Fitting either on the full matrix "
        "would leak the test fold's distribution into training.",
        "src/models/pipeline.py",
        "04_models",
        rule="Fold safety: nothing is fitted on data the outer test fold contributed to.",
    ),
    ArchitectureStep(
        9,
        "selection",
        "Feature selection inside the training fold",
        "Which of the 138 features survive is itself decided per fold, from the "
        "training rows only. A selector chosen once on the whole matrix is one "
        "of the commonest ways a published result becomes unreproducible.",
        "src/feature_selection/ranking.py",
        "05_search_optimization",
        rule="Fold safety: the selector never sees the outer test fold.",
    ),
    ArchitectureStep(
        10,
        "search",
        "Hyperparameter search, nested",
        "Genetic, swarm, Bayesian and randomised search run in an inner loop "
        "over the training fold. The outer fold is never scored during the "
        "search, so the reported number is not the number that was optimised.",
        "src/optimization/driver.py",
        "05_search_optimization",
        rule="Fold safety: the search is scored on inner splits only.",
    ),
    ArchitectureStep(
        11,
        "ensemble",
        "Weighted soft-voting ensemble",
        "SVM, Random Forest and Gradient Boosting each produce a class "
        "probability, and a weighted average of the three produces the vote. "
        "The weights are searched per fold under the same fold safety rule.",
        "src/ensemble/soft_voting.py",
        "05_search_optimization",
    ),
    ArchitectureStep(
        12,
        "evaluate",
        "Repeated nested cross-validation and reporting",
        "Every experiment reports sensitivity, specificity, F1, balanced "
        "accuracy and AUC across repeated subject-grouped folds. Accuracy alone "
        "is never the headline, and model selection prioritises sensitivity and "
        "balanced accuracy.",
        "src/evaluation/experiment.py",
        "06_binary_results",
        rule="Never accuracy alone: five metrics per binary experiment, always.",
    ),
)

#: SO-05's weight search, relative to the repository root.
ENSEMBLE_SOURCE = "outputs/05_search_optimization/SO-05/final_weights.json"

#: The member ids SO-05 optimises over, and their registry names.
_MEMBER_NAMES = {
    "M3": "SVM (RBF kernel)",
    "M4": "Random Forest",
    "M5": "Gradient Boosting",
}

_SHORT_NAMES = {"M3": "SVM", "M4": "RF", "M5": "GB"}

#: Fixed demonstration inputs for the ensemble visualization (T112.4).
#:
#: These are NOT predictions and NOT metrics -- they are three arbitrary,
#: constant probabilities that exist so a reader can see how the members
#: combine. They live here rather than in the component for one reason: the
#: **vote they produce is arithmetic**, and the client neither computes nor
#: rounds a displayed number. Doing the multiply-add in Python keeps that rule
#: intact for a figure that would otherwise be the one exception to it.
_DEMO_PROBABILITIES: tuple[float, ...] = (0.71, 0.58, 0.64)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pipeline_payload() -> dict[str, Any]:
    """The twelve steps, each verified against the repository.

    Raises if a step names a module or an evidence directory that is not there.
    A step that cannot point at its implementation is a caption, and a caption
    that describes work the pipeline does not do is exactly what this project
    refuses to publish.
    """
    root = _project_root()
    steps: list[dict[str, Any]] = []
    for step in ARCHITECTURE_STEPS:
        module = root / step.module
        evidence = root / "outputs" / step.evidence_dir
        if not module.is_file():
            raise FileNotFoundError(
                "architecture step "
                + str(step.index)
                + " ("
                + step.key
                + ") names "
                + step.module
                + ", which does not exist. Either the module moved and this list "
                "is stale, or the step is not implemented; both are findings."
            )
        if not evidence.is_dir():
            raise FileNotFoundError(
                "architecture step "
                + str(step.index)
                + " ("
                + step.key
                + ") claims evidence in outputs/"
                + step.evidence_dir
                + ", which does not exist"
            )
        steps.append(
            {
                "index": step.index,
                "key": step.key,
                "title": step.title,
                "summary": step.summary,
                "module": step.module,
                "evidence_dir": "outputs/" + step.evidence_dir,
                "rule": step.rule,
            }
        )

    if [s["index"] for s in steps] != list(range(1, len(ARCHITECTURE_STEPS) + 1)):
        raise ValueError("the architecture steps are not numbered 1..n in order")

    return {
        "n_steps": len(steps),
        "note": (
            "Declared in src/reporting/architecture.py and verified at export "
            "time: every step below names a module that exists and an outputs/ "
            "directory that exists."
        ),
        "steps": steps,
    }


def ensemble_payload() -> dict[str, Any]:
    """SO-05's searched voting weights, or a stated absence.

    Returns ``available: False`` with a reason rather than raising: the weight
    search is one experiment among many, and a dashboard that cannot build
    because one input has not been produced yet is worse than one that says so.
    """
    path = _project_root() / ENSEMBLE_SOURCE
    if not path.is_file():
        return {
            "available": False,
            "reason": (
                ENSEMBLE_SOURCE + " has not been produced yet; run SO-05 "
                "(Phase 62) before the ensemble view can show searched weights."
            ),
            "source": ENSEMBLE_SOURCE,
            "members": [],
        }

    raw = json.loads(path.read_text(encoding="utf-8"))
    member_ids = list(raw["members"])
    weights = [float(w) for w in raw["mean_weights"]]
    if len(weights) != len(member_ids):
        raise ValueError(ENSEMBLE_SOURCE + " has a weight count that does not match members")

    per_member_std = raw.get("per_member_std", {})
    equal = 1.0 / len(member_ids)
    digest, method = content_digest(path)

    members = [
        {
            "model_id": member_id,
            "name": _MEMBER_NAMES.get(member_id, member_id),
            "short_name": _SHORT_NAMES.get(member_id, member_id),
            "weight": weight,
            # Formatted here, in Python, because the client neither computes nor
            # declares a number. Three places matches the metric rounding rule.
            "weight_display": format(weight, ".3f"),
            "weight_std": float(per_member_std.get(member_id, 0.0)),
            "weight_std_display": format(float(per_member_std.get(member_id, 0.0)), ".3f"),
        }
        for member_id, weight in zip(member_ids, weights, strict=True)
    ]

    demonstration: dict[str, Any] | None = None
    if len(_DEMO_PROBABILITIES) == len(members):
        vote = sum(
            member["weight"] * probability
            for member, probability in zip(members, _DEMO_PROBABILITIES, strict=True)
        )
        demonstration = {
            "note": (
                "Three fixed demonstration probabilities, not predictions. No "
                "recording is involved: this shows how the members combine, and "
                "nothing here is a result. The vote is computed in Python from "
                "the searched weights above, so the browser renders text rather "
                "than arithmetic."
            ),
            "inputs": [
                {
                    "model_id": member["model_id"],
                    "short_name": member["short_name"],
                    "probability": probability,
                    "probability_display": format(probability, ".2f"),
                }
                for member, probability in zip(members, _DEMO_PROBABILITIES, strict=True)
            ],
            "vote": vote,
            "vote_display": format(vote, ".3f"),
        }

    n_folds = int(raw.get("n_folds", 0))
    identical = int(raw.get("folds_identical_to_equal", 0))
    return {
        "available": True,
        "source": ENSEMBLE_SOURCE,
        "sha256": digest,
        "digest_method": method,
        "experiment": raw.get("experiment", ""),
        "task": raw.get("task", ""),
        "objective": raw.get("objective", ""),
        "selection_rule": raw.get("selection_rule", ""),
        "n_folds": n_folds,
        "seed": raw.get("seed"),
        "equal_weight": equal,
        "equal_weight_display": format(equal, ".3f"),
        "folds_identical_to_equal": identical,
        "folds_identical_display": str(identical) + " of " + str(n_folds),
        "constraint": raw.get("constraint", ""),
        "members": members,
        "demonstration": demonstration,
        # The honest headline. A viewer who sees three bars fusing into a vote
        # will assume the search found three materially different weights; on
        # this corpus it mostly did not, and the visualization must say so.
        "interpretation": (
            "The searched weights sit close to equal weighting: "
            + str(identical)
            + " of the "
            + str(n_folds)
            + " outer folds chose weights identical to 1/3 each. The optimised "
            "vote is a small refinement of an equal vote on this corpus, not a "
            "large reweighting, and reading a big difference into the bars "
            "below would overstate what the search found."
        ),
    }
