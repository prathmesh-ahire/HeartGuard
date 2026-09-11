"""Literature review and state of the art -- the evidence for Objective 2 (Phase 101).

Objective 2 asks PV-MEPCG / PulseVision "to review recently published
preprocessing, feature extraction, and classification techniques, as well as the
state-of-the-art in phonocardiogram (PCG) signal analysis". Its required
evidence in the blueprint is a literature review table beside the two
ablations. This module is that table, in two parts:

**LIT-01, the review.** One row per study, on the schema T101.1 fixes (author,
year, dataset, preprocessing, feature families, classifier, validation scheme,
reported metric, sample size, stated limitation), seeded with the ten references
of the original college synopsis (T101.2), extended with the leading entries of
the PhysioNet/CinC 2016 and George B. Moody PhysioNet 2022 Challenges (T101.3)
and with recent work on ensembles, feature selection, deep learning and domain
shift (T101.4). Every row carries a positioning entry saying how PV-MEPCG
differs (T101.5) -- stated as a difference, never as a superiority claim.

**LIT-02, the indicative comparison.** The published PhysioNet 2016 numbers
beside PV-MEPCG's own, with the caveat T101.6 requires: the challenge entries
were scored once on a hidden test set that was never released, PV-MEPCG is
cross-validated inside the public training corpus, and the protocols are not
interchangeable. PV-MEPCG's numbers are **read from its result files at build
time**, never typed; the published ones are quotations, each with its source.

What a quoted value is, and is not
----------------------------------
Every study below was checked against its publisher, PubMed, Semantic Scholar or
official challenge record on 2026-09-11 (the ``verification`` column names the
record). Where that record does not state a field -- many 2003-2010 abstracts
give no sample size or headline metric -- the cell says so rather than filling
it from memory. Literature values are citations, not results of this project:
they are never re-computed, and nothing here is presented as PV-MEPCG's.

The ``limitation_source`` column separates a limitation the paper itself states
from one this review observes in the design (a 36-patient cohort, record-level
random splits on a corpus with repeated subjects), so an observation is never
put in an author's mouth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.logging_setup import get_logger

__all__ = [
    "GROUPS",
    "PUBLISHED_2016",
    "STUDIES",
    "SYNOPSIS_REFERENCES",
    "LITERATURE_TABLE_IDS",
    "build_lit01",
    "build_lit02",
    "literature_dir",
    "write_literature_tables",
]

log = get_logger("reporting.literature")

COMMAND = "python scripts/41_literature_review.py"
LITERATURE_TABLE_IDS: tuple[str, ...] = ("LIT-01", "LIT-02")
VERIFIED = "web-verified 2026-09-11 against "

AGGREGATE = "outputs/06_binary_results/EXP-A2/aggregate_metrics.csv"
SELECTION = "outputs/06_binary_results/final_model_selection.csv"
LOSO = "outputs/09_ablation/T-S5_leave_one_source_out_generalization.csv"
MODULE = "src/reporting/literature.py"

GROUPS: dict[str, str] = {
    "synopsis": "Synopsis reference",
    "challenge_2016": "PhysioNet/CinC 2016 benchmark",
    "challenge_2022": "PhysioNet 2022 benchmark",
    "recent": "Recent work",
}

#: The ten references of the college synopsis, as the synopsis labels them. T101.7
#: checks every one is covered by a row whose ``synopsis_label`` matches.
SYNOPSIS_REFERENCES: tuple[str, ...] = (
    "Muruganantham 2003",
    "Shui 2004",
    "Segaier 2005",
    "Jiang & Choi 2006",
    "Ahlstrom 2006",
    "Nigam 2008",
    "Babaei & Geranmayeh 2009",
    "Ari 2010",
    "Choi 2010",
    "Rangayyan & Lehner",
)

NOT_STATED = "not stated in the accessible record"


@dataclass(frozen=True)
class Study:
    study_id: str
    group: str
    authors: str
    year: str
    title: str
    venue: str
    dataset: str
    preprocessing: str
    features: str
    classifier: str
    validation: str
    reported_metric: str
    sample_size: str
    limitation: str
    limitation_source: str
    positioning: str
    verification: str
    source_url: str
    synopsis_label: str = ""


STUDIES: tuple[Study, ...] = (
    # --- the synopsis references (T101.2) -----------------------------------
    Study(
        "L01",
        "synopsis",
        "Rangayyan RM, Lehner RJ",
        "1987",
        "Phonocardiogram signal analysis: a review",
        "Critical Reviews in Biomedical Engineering 15(3):211-236",
        "none (narrative review)",
        "reviewed: filtering, envelope and segmentation methods of the period",
        "reviewed: time- and frequency-domain PCG descriptors",
        "reviewed; no classifier proposed",
        "not applicable (review)",
        "none (review)",
        "not applicable",
        "Predates public PCG corpora and machine-learning classifiers.",
        "observed",
        "A review of techniques; PV-MEPCG is an implemented, reproducible pipeline whose "
        "every number is generated and evaluated on four public corpora.",
        VERIFIED + "PubMed 3329595",
        "https://pubmed.ncbi.nlm.nih.gov/3329595/",
        "Rangayyan & Lehner",
    ),
    Study(
        "L02",
        "synopsis",
        "Muruganantham M, et al.",
        "2003",
        "Methods for classification of phonocardiogram",
        "TENCON 2003, Conference on Convergent Technologies for the Asia-Pacific Region",
        NOT_STATED,
        NOT_STATED,
        "wavelet-based features",
        "neural network, fuzzy system and wavelet approaches, compared",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "Method comparison without a public dataset or a stated validation protocol.",
        "observed",
        "Compared three classifier families on private data; PV-MEPCG compares eight "
        "classifiers and two soft-voting ensembles on one stored, subject-grouped fold map.",
        VERIFIED + "Semantic Scholar record (TENCON 2003)",
        "https://www.semanticscholar.org/paper/27b97a9bcd46cb79ed0b5f8403c5a31da8d1b379",
        "Muruganantham 2003",
    ),
    Study(
        "L03",
        "synopsis",
        "Shui",
        "2004",
        "AM13 Analysis of Heart Sound",
        "Thesis, National University of Singapore",
        NOT_STATED,
        NOT_STATED,
        "wavelet analysis",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "Known only through secondary citation; the full text was not accessible.",
        "observed",
        "Wavelet features alone; PV-MEPCG uses the wavelet (DWT) family as one of six "
        "(138 features) and measures each family's contribution by ablation (EXP-F1).",
        VERIFIED + "secondary citation in IJCA 77(4) review; primary text not accessible",
        "https://www.ijcaonline.org/archives/volume77/number4/13381-1001/",
        "Shui 2004",
    ),
    Study(
        "L04",
        "synopsis",
        "El-Segaier M, Lilja O, Lukkarinen S, Sornmo L, Sepponen R, Pesonen E",
        "2005",
        "Computer-based detection and analysis of heart sound and murmur",
        "Annals of Biomedical Engineering 33:937-942",
        "300 children with a cardiac murmur, electronic stethoscope, simultaneous ECG",
        "short-time Fourier transform",
        "S1/S2 timing referenced to ECG R- and T-waves; spectral content",
        "rule-based heart-sound detection (no learned classifier)",
        "agreement with the ECG reference",
        "S1 detected 100% within 0.05-0.2 R-R; S2 97%",
        "300 children",
        "Depends on a simultaneous ECG reference.",
        "observed",
        "Needs an ECG to find heart sounds; PV-MEPCG classifies from the PCG alone, "
        "segmentation-free.",
        VERIFIED + "PubMed 16060534",
        "https://pubmed.ncbi.nlm.nih.gov/16060534/",
        "Segaier 2005",
    ),
    Study(
        "L05",
        "synopsis",
        "Jiang Z, Choi S",
        "2006",
        "A cardiac sound characteristic waveform method for in-home heart disorder "
        "monitoring with electric stethoscope",
        "Expert Systems with Applications 31:286-298",
        "electric-stethoscope recordings",
        "single-degree-of-freedom model extracting a characteristic waveform",
        "time intervals T1, T2, T11, T12 against an adaptive threshold",
        "fuzzy c-means clustering for the threshold; rule-based identification",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "Hand-designed interval parameters from one waveform model.",
        "observed",
        "Timing parameters from a single model; PV-MEPCG learns from 138 multi-domain "
        "features validated across public multi-site corpora.",
        VERIFIED + "ScienceDirect record, doi 10.1016/j.eswa.2005.09.025",
        "https://www.sciencedirect.com/science/article/abs/pii/S0957417405002150",
        "Jiang & Choi 2006",
    ),
    Study(
        "L06",
        "synopsis",
        "Ahlstrom C, Hult P, Rask P, et al.",
        "2006",
        "Feature extraction for systolic heart murmur classification",
        "Annals of Biomedical Engineering 34(11):1666-1677",
        "36 patients: aortic stenosis, mitral insufficiency, physiological murmurs",
        NOT_STATED,
        "207 features: Shannon energy, wavelets, fractal dimensions, recurrence "
        "quantification; 14 selected",
        "neural network",
        NOT_STATED,
        "86% correct with the multi-domain subset vs 68% single-domain",
        "36 patients",
        "Small single-centre cohort.",
        "observed",
        "Closest synopsis precedent: multi-domain features with search-based (SFFS) "
        "selection. PV-MEPCG selects inside training folds on thousands of public "
        "recordings and reports the subset against all features.",
        VERIFIED + "PubMed 17019618",
        "https://pubmed.ncbi.nlm.nih.gov/17019618/",
        "Ahlstrom 2006",
    ),
    Study(
        "L07",
        "synopsis",
        "Nigam V, Priemer R",
        "2008",
        "A simplicity-based fuzzy clustering approach for detection and extraction of "
        "murmurs from the phonocardiogram",
        "Physiological Measurement 29:33-47",
        NOT_STATED,
        "murmur localisation within the cardiac cycle",
        "signal simplicity",
        "fuzzy clustering",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "The paper notes the variation of murmur amplitude and spectrum across patients "
        "hinders a generic segmentation.",
        "stated",
        "Segments murmurs; PV-MEPCG does not segment and classifies whole recordings, "
        "with CirCor murmur kept as its own task.",
        VERIFIED + "secondary: bibliographic record in the Chakrabarti & Saha critical review",
        "https://www.researchgate.net/publication/296640022",
        "Nigam 2008",
    ),
    Study(
        "L08",
        "synopsis",
        "Babaei S, Geranmayeh A",
        "2009",
        "Heart sound reproduction based on neural network classification of cardiac "
        "valve disorders using wavelet transforms of PCG signals",
        "Computers in Biology and Medicine 39(1):8-15",
        "clinical PCG samples: aortic insufficiency, aortic stenosis, pulmonary stenosis, normal",
        "db4 wavelet decomposition, five levels",
        "wavelet sub-band statistics",
        "neural network, or statistical averaging",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "Four classes from clinical samples of unstated size.",
        "observed",
        "Wavelet + neural network; PV-MEPCG adds five more feature families and a "
        "heterogeneous soft-voting ensemble.",
        VERIFIED + "PubMed 19081085",
        "https://pubmed.ncbi.nlm.nih.gov/19081085/",
        "Babaei & Geranmayeh 2009",
    ),
    Study(
        "L09",
        "synopsis",
        "Ari S, Hembram K, Saha G",
        "2010",
        "Detection of cardiac abnormality from PCG signal using LMS based least square "
        "SVM classifier",
        "Expert Systems with Applications 37(12):8019-8026",
        "normal, aortic insufficiency, aortic stenosis, atrial septal defect, mitral "
        "regurgitation, mitral stenosis",
        NOT_STATED,
        "wavelet features",
        "least-squares SVM improved with a least-mean-square step",
        NOT_STATED,
        "86.72% accuracy",
        NOT_STATED,
        "Accuracy alone, on a normal-versus-pathological task.",
        "observed",
        "Wavelet + one SVM; PV-MEPCG uses an RBF SVM as one member beside Random Forest "
        "and Gradient Boosting, with in-fold calibration, and never reports accuracy alone.",
        VERIFIED + "ScienceDirect record S0957417410004999",
        "https://www.sciencedirect.com/science/article/abs/pii/S0957417410004999",
        "Ari 2010",
    ),
    Study(
        "L10",
        "synopsis",
        "Choi S, Jiang Z",
        "2010",
        "Cardiac sound murmurs classification with autoregressive spectral analysis and "
        "multi-support vector machine technique",
        "Computers in Biology and Medicine 40(1):8-20",
        "489 cardiac sounds (196 normal, 293 abnormal) from 6 volunteers and 34 patients",
        "normalized autoregressive power spectral density (NAR-PSD)",
        "two spectral features: peak frequency and bandwidth of the NAR-PSD curve",
        "multi-support vector machine",
        NOT_STATED,
        NOT_STATED,
        "489 recordings, 40 subjects",
        "Many recordings per subject from 40 people.",
        "observed",
        "Two spectral features + SVM; PV-MEPCG groups every split by subject, so no "
        "person contributes to both training and test.",
        VERIFIED + "PubMed 19926081",
        "https://pubmed.ncbi.nlm.nih.gov/19926081/",
        "Choi 2010",
    ),
    # --- challenge benchmarks (T101.3) --------------------------------------
    Study(
        "B01",
        "challenge_2016",
        "Potes C, Parvaneh S, Rahman A, Conroy B",
        "2016",
        "Ensemble of feature-based and deep learning-based classifiers for detection of "
        "abnormal heart sounds",
        "Computing in Cardiology 43 (Challenge winner)",
        "PhysioNet/CinC 2016: training 2,575 normal / 665 abnormal; hidden test set",
        "cardiac-cycle segmentation; four frequency bands for the CNN",
        "124 time-frequency features",
        "AdaBoost variant + convolutional neural network, combined as an ensemble",
        "official hidden test set",
        "Se 0.9424, Sp 0.7781, overall (MAcc) 0.8602",
        "3,240 training recordings",
        "Modest specificity against very high sensitivity.",
        "observed",
        "Also an ensemble, but with a CNN member; PV-MEPCG is classical and CPU-only (the "
        "CNN is out of scope) and weights its members by an in-fold simplex search. It "
        "cannot be scored on the hidden test set.",
        VERIFIED + "CinC 2016 paper record and the official 2016 score listing",
        "https://www.cinc.org/archives/2016/pdf/182-399.pdf",
    ),
    Study(
        "B02",
        "challenge_2016",
        "Zabihi M, Bahrami Rad A, Kiranyaz S, Gabbouj M, Katsaggelos AK",
        "2016",
        "Heart sound anomaly and quality detection using ensemble of neural networks "
        "without segmentation",
        "Computing in Cardiology 43 (second place)",
        "PhysioNet/CinC 2016",
        "no segmentation",
        "18 of 40 time, frequency and time-frequency features, chosen by wrapper selection",
        "ensemble of 20 feed-forward neural networks",
        "official hidden test set (training score 0.9150 under 20-fold CV)",
        "Se 0.8691, Sp 0.8490, overall 0.8590",
        "3,240 training recordings",
        "Training-set score 5.6 points above the hidden-test score.",
        "observed",
        "The closest design: segmentation-free features, search-based selection and an "
        "ensemble. PV-MEPCG differs in heterogeneous members, one-standard-error "
        "selection rules, subject-grouped repeated CV and external validation on CirCor.",
        VERIFIED + "the paper (moody-challenge.physionet.org)",
        "https://moody-challenge.physionet.org/2016/papers/zabihi.pdf",
    ),
    Study(
        "B03",
        "challenge_2016",
        "Kay E, Agarwal A",
        "2016",
        "Official PhysioNet/CinC 2016 entry (third place)",
        "PhysioNet/CinC Challenge 2016 official score listing",
        "PhysioNet/CinC 2016",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "official hidden test set",
        "overall (MAcc) 0.8520",
        "3,240 training recordings",
        "Only the overall score is recorded in the official listing.",
        "observed",
        "A hidden-test score; PV-MEPCG has no access to that test set, which was never "
        "released, so any comparison is indicative.",
        VERIFIED + "official 2016 score listing",
        "https://physionet.org/content/challenge-2016/1.0.0/sources/2016-scoreinfo-with-authors",
    ),
    Study(
        "B04",
        "challenge_2022",
        "Team HearHeart (murmur track winner)",
        "2022",
        "George B. Moody PhysioNet Challenge 2022, murmur detection",
        "Reyna MA et al., PLOS Digital Health 2023 (Challenge summary)",
        "CirCor DigiScope: 3,163 public training recordings; hidden test 1,582 recordings",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "official hidden test set",
        "weighted accuracy 0.780 (murmur, first of the ranked teams)",
        "3,163 training recordings",
        "Scored on a test set that is not public.",
        "observed",
        "PV-MEPCG runs CirCor murmur as its own three-class task (EXP-C1) with "
        "patient-level aggregation, cross-validated on the public training set only; "
        "it is not scored with the Challenge's weighted-accuracy metric.",
        VERIFIED + "Reyna et al. 2023, PMC10495026",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10495026/",
    ),
    Study(
        "B05",
        "challenge_2022",
        "Team CUED_Acoustics (clinical outcome track winner)",
        "2022",
        "George B. Moody PhysioNet Challenge 2022, clinical outcome",
        "Reyna MA et al., PLOS Digital Health 2023 (Challenge summary)",
        "CirCor DigiScope: 3,163 public training recordings; hidden test 1,582 recordings",
        NOT_STATED,
        NOT_STATED,
        NOT_STATED,
        "official hidden test set",
        "cost 11,144 (clinical outcome, lowest of the ranked teams)",
        "3,163 training recordings",
        "The outcome metric is a cost model, not a classification rate.",
        "observed",
        "PV-MEPCG runs CirCor outcome as a separate binary task (EXP-C2) and tests "
        "adult-to-paediatric transfer (EXP-D1); it reports sensitivity and balanced "
        "accuracy rather than the Challenge cost.",
        VERIFIED + "Reyna et al. 2023, PMC10495026",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC10495026/",
    ),
    # --- recent work (T101.4) ------------------------------------------------
    Study(
        "R01",
        "recent",
        "Homsi MN, Warrick P",
        "2017",
        "Ensemble methods with outliers for phonocardiogram classification",
        "Physiological Measurement 38(8):1631",
        "PhysioNet/CinC 2016",
        "outlier detection by an interquartile-range threshold",
        "131 time, frequency, wavelet and statistical features, then feature reduction",
        "ensemble classifiers",
        "official hidden test set",
        "MAcc 0.801",
        "3,240 training recordings",
        "Outliers handled separately from the standard-range signals.",
        "stated",
        "An ensemble over a comparable classical feature set; PV-MEPCG adds optimized "
        "weights, in-fold feature selection and cross-dataset validation.",
        VERIFIED + "secondary: the Clifford et al. 2017 focus-issue review, which reports it",
        "https://iopscience.iop.org/article/10.1088/1361-6579/aa7ec8",
    ),
    Study(
        "R02",
        "recent",
        "Tang H, et al.",
        "2018",
        "PCG classification using multidomain features and SVM classifier",
        "BioMed Research International",
        "PhysioNet/CinC 2016: 3,153 recordings (2,488 normal, 665 abnormal)",
        "segmentation into heart-sound states",
        "515 features from nine domains (intervals, state spectra, amplitude, energy, "
        "cepstrum, cyclostationarity, higher-order statistics, entropy)",
        "SVM, RBF kernel",
        "repeated random record-level train/test splits (200 repetitions)",
        "Se 0.88, Sp 0.87, overall 0.88",
        "3,153 recordings",
        "Record-level random splits on a corpus where one subject can contribute several "
        "recordings.",
        "observed",
        "Multi-domain features like PV-MEPCG, but split by record; PV-MEPCG groups every "
        "split by subject and repeats it 5x5, the stricter protocol.",
        VERIFIED + "PMC6077676",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6077676/",
    ),
    Study(
        "R03",
        "recent",
        "Humayun AI, Ghaffarzadegan S, Ansari MI, Feng Z, Hasan T",
        "2020",
        "Towards domain invariant heart sound abnormality detection using learnable filterbanks",
        "IEEE Journal of Biomedical and Health Informatics 24(8):2189-2198",
        "multi-domain public PCG datasets (stethoscope and site variation)",
        "raw PCG into learnable time-convolution filterbanks",
        "learned (tConv FIR filterbank front end)",
        "convolutional neural network",
        "cross-domain evaluation",
        "up to 11.84% relative improvement in MAcc over the compared state of the art",
        NOT_STATED,
        "The paper states that stethoscope, environment and collection protocol degrade "
        "machine-learning heart-sound systems.",
        "stated",
        "Addresses the problem PV-MEPCG measures and does not solve: EXP-F3 "
        "(leave-one-sub-collection-out) shows the ranking collapses on an unseen "
        "recording setup. Domain adaptation is PV-MEPCG future work.",
        VERIFIED + "IEEE Xplore / arXiv 1910.00498",
        "https://arxiv.org/abs/1910.00498",
    ),
    Study(
        "R04",
        "recent",
        "Clifford GD, et al.",
        "2017",
        "Recent advances in heart sound analysis",
        "Physiological Measurement 38: E10",
        "review of the PhysioNet/CinC 2016 Challenge and focus issue",
        "reviewed",
        "reviewed",
        "reviewed",
        "not applicable (review)",
        "none (review)",
        "not applicable",
        "Not applicable (review).",
        "not available",
        "Context for the 2016 benchmark rows; PV-MEPCG's contribution is a reproducible, "
        "fold-safe classical pipeline across four corpora, not a new challenge score.",
        VERIFIED + "IOPscience record",
        "https://iopscience.iop.org/article/10.1088/1361-6579/aa7ec8",
    ),
    Study(
        "R05",
        "recent",
        "Chen W, et al.",
        "2024",
        "Artificial intelligence for heart sound classification: a review",
        "Expert Systems (Wiley), doi 10.1111/exsy.13535",
        "review",
        "reviewed",
        "reviewed: handcrafted and learned representations",
        "reviewed: classical machine learning and deep learning",
        "not applicable (review)",
        "none (review)",
        "not applicable",
        "Not applicable (review).",
        "not available",
        "The current state of the art is dominated by deep learning; PV-MEPCG is an "
        "interpretable, CPU-only classical baseline evaluated with subject-grouped CV "
        "and external validation, and makes no claim against deep models.",
        VERIFIED + "Wiley Online Library record",
        "https://onlinelibrary.wiley.com/doi/10.1111/exsy.13535",
    ),
)


@dataclass(frozen=True)
class Published:
    """A PhysioNet 2016 result as published, for LIT-02. Quoted, never computed."""

    study_id: str
    entry: str
    evaluation: str
    sensitivity: float
    specificity: float
    macc: float
    source_url: str


_NAN = float("nan")

PUBLISHED_2016: tuple[Published, ...] = (
    Published(
        "B01",
        "Potes et al. 2016 (Challenge 1st)",
        "official hidden test set, scored once",
        0.9424,
        0.7781,
        0.8602,
        "https://www.cinc.org/archives/2016/pdf/182-399.pdf",
    ),
    Published(
        "B02",
        "Zabihi et al. 2016 (Challenge 2nd)",
        "official hidden test set, scored once",
        0.8691,
        0.8490,
        0.8590,
        "https://moody-challenge.physionet.org/2016/papers/zabihi.pdf",
    ),
    Published(
        "B03",
        "Kay & Agarwal 2016 (Challenge 3rd)",
        "official hidden test set, scored once",
        _NAN,
        _NAN,
        0.8520,
        "https://physionet.org/content/challenge-2016/1.0.0/sources/2016-scoreinfo-with-authors",
    ),
    Published(
        "R01",
        "Homsi & Warrick 2017",
        "official hidden test set",
        _NAN,
        _NAN,
        0.801,
        "https://iopscience.iop.org/article/10.1088/1361-6579/aa7ec8",
    ),
    Published(
        "R02",
        "Tang et al. 2018",
        "repeated random record-level splits of the training corpus",
        0.88,
        0.87,
        0.88,
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6077676/",
    ),
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def literature_dir(out_dir: str | Path | None = None) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs.literature"))


# ---------------------------------------------------------------------------
# LIT-01
# ---------------------------------------------------------------------------


def build_lit01(command: str = "") -> Any:
    import pandas as pd

    from src.reporting.result_tables import DISCLAIMER
    from src.reporting.tables import Column, TableSpec, build_table

    rows = []
    for study in STUDIES:
        row = asdict(study)
        row["group"] = GROUPS[study.group]
        rows.append(row)

    text = (
        ("study_id", "ID"),
        ("group", "Group"),
        ("authors", "Authors"),
        ("year", "Year"),
        ("title", "Title"),
        ("venue", "Venue"),
        ("dataset", "Dataset"),
        ("preprocessing", "Preprocessing"),
        ("features", "Feature families"),
        ("classifier", "Classifier"),
        ("validation", "Validation scheme"),
        ("reported_metric", "Reported metric (as published)"),
        ("sample_size", "Sample size"),
        ("limitation", "Limitation"),
        ("limitation_source", "Limitation source"),
        ("positioning", "How PV-MEPCG differs"),
        ("verification", "Verified against"),
        ("source_url", "Source"),
        ("synopsis_label", "Synopsis reference"),
    )
    spec = TableSpec(
        table_id="LIT-01",
        title="Literature Review and State of the Art",
        caption=(
            "Published PCG preprocessing, feature-extraction and classification work: the "
            "ten references of the original synopsis, the leading PhysioNet/CinC 2016 and "
            "PhysioNet 2022 Challenge entries, and recent work on ensembles, feature "
            "selection, deep learning and domain shift, each with how PV-MEPCG / "
            "PulseVision differs from it."
        ),
        sources=(MODULE,),
        columns=tuple(Column(name, header, kind="text") for name, header in text),
        exp_id="n/a (literature review)",
        objective="O2",
        dataset="published studies",
        notes=(
            "Reported metrics are quoted from each publication or official challenge record "
            "and are not results of this project. A field the accessible record does not "
            "state is written as such, never filled in.",
            "'Limitation source' marks whether the paper states the limitation or this "
            "review observes it in the design.",
            "Rangayyan & Lehner is dated 1987 by its PubMed record (volume 15); the synopsis "
            "cites it as 1988.",
            DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# LIT-02
# ---------------------------------------------------------------------------


def _final_model() -> str:
    import pandas as pd

    selection = pd.read_csv(_root() / SELECTION).sort_values("rank")
    return str(selection.iloc[0]["model_id"])


def build_lit02(command: str = "") -> Any:
    import pandas as pd

    from src.reporting.result_tables import DISCLAIMER
    from src.reporting.tables import Column, TableSpec, build_table

    final = _final_model()
    aggregate = pd.read_csv(_root() / AGGREGATE).set_index("model_id").loc[final]
    loso = pd.read_csv(_root() / LOSO).set_index("model_id").loc[final]

    rows: list[dict[str, Any]] = [
        {
            "entry": item.entry,
            "evaluation": item.evaluation,
            "sensitivity": item.sensitivity,
            "specificity": item.specificity,
            "macc": item.macc,
            "origin": "published",
            "source": item.source_url,
        }
        for item in PUBLISHED_2016
    ]
    rows += [
        {
            "entry": "PV-MEPCG / PulseVision, final model " + final,
            "evaluation": (
                "repeated 5x5 subject-grouped cross-validation within the public training "
                "corpus (EXP-A2), mean over " + str(int(aggregate["n_folds"])) + " folds"
            ),
            "sensitivity": float(aggregate["sensitivity_mean"]),
            "specificity": float(aggregate["specificity_mean"]),
            "macc": float(aggregate["balanced_accuracy_mean"]),
            "origin": "this project, generated",
            "source": AGGREGATE,
        },
        {
            "entry": "PV-MEPCG / PulseVision, final model " + final + ", unseen collection",
            "evaluation": (
                "leave-one-sub-collection-out (EXP-F3), mean over "
                + str(int(loso["n_folds_holdout"]))
                + " held-out recording collections"
            ),
            "sensitivity": float(loso["sensitivity_holdout"]),
            "specificity": float(loso["specificity_holdout"]),
            "macc": float(loso["balanced_accuracy_holdout"]),
            "origin": "this project, generated",
            "source": LOSO,
        },
    ]
    spec = TableSpec(
        table_id="LIT-02",
        title="PhysioNet 2016 Published Comparison",
        caption=(
            "PV-MEPCG / PulseVision on PhysioNet/CinC 2016 beside published results. "
            "Indicative, not head-to-head: the split protocols differ, so the rows are not "
            "scores on the same records."
        ),
        sources=(AGGREGATE, SELECTION, LOSO, MODULE),
        columns=(
            Column("entry", "Entry", kind="text"),
            Column("evaluation", "Evaluation protocol", kind="text"),
            Column("sensitivity", "Sensitivity", kind="metric"),
            Column("specificity", "Specificity", kind="metric"),
            Column("macc", "MAcc (balanced accuracy)", kind="metric"),
            Column("origin", "Origin", kind="text"),
            Column("source", "Source", kind="text"),
        ),
        exp_id="EXP-A2; EXP-F3 (PV-MEPCG rows only)",
        objective="O2",
        dataset="D1 PhysioNet 2016",
        notes=(
            "CAVEAT -- split protocols differ, and the comparison is indicative, not "
            "head-to-head. The Challenge entries were scored once on the official hidden "
            "test set, which was never released; PV-MEPCG is cross-validated within the "
            "public training corpus, grouped by subject; Tang et al. used repeated random "
            "record-level splits. Different records, different protocols: no row "
            "outranks another on this table.",
            "MAcc, the Challenge's overall score, is (sensitivity + specificity) / 2 -- the "
            "same quantity as balanced accuracy.",
            "The unseen-collection row is the bound on the within-corpus row: the same model "
            "scored on a PhysioNet recording collection it never trained on (EXP-F3). No "
            "generalization claim is made from any row.",
            "Published values are quotations with their source; PV-MEPCG's are read from its "
            "result files when this table is built.",
            DISCLAIMER,
        ),
        command=command,
    )
    return build_table(spec, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_literature_tables(
    out_dir: str | Path | None = None, *, evidence_index: str | Path | None = None
) -> dict[str, dict[str, Path]]:
    from src.reporting.tables import write_table
    from src.utils.io import ensure_dir

    target = Path(ensure_dir(literature_dir(out_dir)))
    written = {
        table_id: write_table(builder(COMMAND), target, evidence_index=evidence_index)
        for table_id, builder in (("LIT-01", build_lit01), ("LIT-02", build_lit02))
    }
    log.info("literature tables: %s", ", ".join(written))
    return written
