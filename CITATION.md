# Citation and dataset credits

PV-MEPCG / PulseVision is an academic research prototype. It is built entirely
on three public heart-sound corpora, and **none of them is ours**. If you use
this repository, cite the datasets first — they are the contribution that made
any of it possible.

## Citing this work

> PV-MEPCG / PulseVision: a search-optimized heterogeneous ensemble for
> phonocardiogram heart-sound screening. Repository:
> <https://github.com/prathmesh-ahire/HeartGuard>

This is a student research prototype, not a published method and not a medical
device. It has had **no clinical validation of any kind**. See
[README.md](README.md#limitations) for what that means in practice, and
[the scope boundary](README.md#-scope-boundary--read-this-first) for what the
software may and may not be described as doing.

---

## D1 — PhysioNet/CinC Challenge 2016

**Used for:** the binary task (normal / abnormal), 3,240 training recordings,
2000 Hz. Every headline binary result in this project is a cross-validation
within this corpus.

Cite all three:

> Liu, C., Springer, D., Li, Q., Moody, B., Juan, R. A., Chorro, F. J.,
> Castells, F., Roig, J. M., Silva, I., Johnson, A. E. W., Syed, Z.,
> Schmidt, S. E., Papadaniil, C. D., Hadjileontiadis, L., Naseri, H.,
> Moukadem, A., Dieterlen, A., Brandt, C., Tang, H., Samieinasab, M.,
> Samieinasab, M. R., Sameni, R., Mark, R. G., Clifford, G. D. (2016).
> An open access database for the evaluation of heart sound algorithms.
> *Physiological Measurement*, 37(12), 2181–2213.
> <https://doi.org/10.1088/0967-3334/37/12/2181>

> Clifford, G. D., Liu, C., Moody, B., Springer, D., Silva, I., Li, Q.,
> Mark, R. G. (2016). Classification of normal/abnormal heart sound
> recordings: The PhysioNet/Computing in Cardiology Challenge 2016.
> *Computing in Cardiology*, 43, 609–612.

> Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M.,
> Ivanov, P. Ch., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K.,
> Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet:
> Components of a New Research Resource for Complex Physiologic Signals.
> *Circulation*, 101(23), e215–e220.
> <https://doi.org/10.1161/01.CIR.101.23.e215>

**Canonical source:** <https://physionet.org/content/challenge-2016/1.0.0/>
**Terms at source:** Open Data Commons Attribution License v1.0 (ODC-By 1.0).

## D2 / D3 — PASCAL Classifying Heart Sounds Challenge 2011

**Used for:** PASCAL A (4-class, 124 labelled records, 44100 Hz) and PASCAL B
(3-class, 461 labelled records, 4000 Hz). These are two separate label spaces
and are never merged.

> Bentley, P., Nordehn, G., Coimbra, M., Mannor, S. (2011).
> The PASCAL Classifying Heart Sounds Challenge 2011 (CHSC2011) Results.
> <http://www.peterjbentley.com/heartchallenge/index.html>

**Canonical source:** the challenge page above.
**Terms at source:** the challenge's own conditions of use. The dataset is
distributed for research and the challenge asks that the above result page be
cited.

> **`artifact` in PASCAL A is a recording-quality label, not a cardiac class.**
> The four-class model must never be described as a four-class *cardiac*
> classifier.

## D4 — CirCor DigiScope Phonocardiogram Dataset (PhysioNet 2022)

**Used for:** the CirCor murmur task (absent / present / unknown) and the CirCor
outcome task (normal / abnormal); 942 patients, 3,163 recordings, 4000 Hz. Also
the external-validation experiment EXP-D1 and the one redistributed sample
recording in the dashboard.

> Oliveira, J., Renna, F., Costa, P. D., Nogueira, M., Oliveira, C.,
> Ferreira, C., Jorge, A., Mattos, S., Hatem, T., Tavares, T., Elola, A.,
> Rad, A. B., Sameni, R., Clifford, G. D., Coimbra, M. T. (2022).
> The CirCor DigiScope Dataset: From Murmur Detection to Murmur Classification.
> *IEEE Journal of Biomedical and Health Informatics*, 26(6), 2524–2535.
> <https://doi.org/10.1109/JBHI.2021.3137048>

> Reyna, M. A., Kiarashi, Y., Elola, A., Oliveira, J., Renna, F., Gu, A.,
> Perez-Alday, E. A., Sadr, N., Sharma, A., Mattos, S., Coimbra, M. T.,
> Sameni, R., Rad, A. B., Clifford, G. D. (2022). Heart Murmur Detection
> from Phonocardiogram Recordings: The George B. Moody PhysioNet Challenge 2022.
> *Computing in Cardiology*, 49.

Plus the PhysioNet resource citation (Goldberger et al. 2000) given under D1.

**Canonical source:** <https://physionet.org/content/circor-heart-sound/1.0.3/>
**Licence:** [Open Data Commons Attribution License v1.0 (ODC-By 1.0)](https://opendatacommons.org/licenses/by/1-0/).
The grant travels with the corpus as `dataset/archive/LICENSE.txt`.

### The one redistributed recording

`frontend/public/85197_TV.wav` (93 KB, record `85197_TV`) and its expert
segmentation are the sole exception to "the dataset is never committed", so the
dashboard can draw a real S1 / systole / S2 / diastole overlay instead of a
synthetic animation. ODC-By §4.2(d) and §4.3 require notices that a WAV file
cannot carry, so they live in [`frontend/public/NOTICE.md`](frontend/public/NOTICE.md)
beside the audio, and the dashboard renders the attribution wherever that
recording plays. It is a **de-identified dataset sample**, not a patient and not
a case.

---

## What this repository could and could not verify about licensing

Stated plainly, because the honest answer is not uniform across the three:

| Corpus | Licence file present in this checkout | What we rely on |
|---|---|---|
| D4 CirCor | **Yes** — `dataset/archive/LICENSE.txt`, ODC-By 1.0 | the grant itself, read and followed clause by clause |
| D1 PhysioNet 2016 | No | the terms published at the canonical PhysioNet page |
| D2/D3 PASCAL | No | the terms published at the challenge page |

The D1 and D2/D3 copies used here came from redistributions that carry no
licence file. That is why **only the CirCor sample is redistributed** in this
repository: asserting a licence nobody verified is the same failure mode as
asserting a metric nobody computed. The other two corpora are cited, described
and analysed here, and you must obtain them from their canonical sources
yourself — see [README.md](README.md#dataset-placement).

## Software this is built on

scikit-learn, NumPy, SciPy, pandas, librosa, soundfile, PyWavelets, matplotlib,
seaborn, joblib, pyarrow, XGBoost, FastAPI, uvicorn, Next.js, React, Tailwind
CSS, ECharts, Plotly, Three.js, GSAP and KaTeX. Exact pinned versions are in
`requirements/` and `outputs/configs/pip_freeze.txt` (Python) and
`frontend/package-lock.json` (Node).
