# CirCor: what patient-to-recording label propagation costs

Generated from `outputs/01_dataset_audit/subject_split_map.csv` and the EXP-C1/EXP-C2 runs. Every number is measured.

## Where the outcome label comes from

`Outcome` is **not** in `training_data.csv`. It exists only inside the per-patient `.txt` files as `#Outcome: Normal|Abnormal`, and a loader built around the CSV produces a murmur-only pipeline. Parsed from the txt files: **486 Normal / 456 Abnormal** over 942 patients.

## The propagation, and the noise it introduces

CirCor labels a **patient**; the model scores a **recording**. Each patient carries 3.36 recordings on average (min 1, max 6), taken at up to five auscultation locations. Training assigns every recording its patient's label, which asserts that the finding is present in **every** recording of that patient.

**That assertion is false for a murmur and questionable for an outcome.** A murmur audible at the pulmonary valve need not be audible at the mitral valve; CirCor ships a `Most audible location` field precisely because location matters. So an unknown share of the 616 recordings labelled `Present` contain no audible murmur, and they are in the **training** data, not only the evaluation.

Three consequences that must travel with any CirCor number:

1. **Recording-level metrics are pessimistic by an unmeasured amount.** A model penalised for not hearing a murmur that is not there is being scored against a wrong label, not making an error.
2. **The label noise is not random.** It concentrates in patients with many recordings and in findings that are locally audible, so it correlates with exactly the structure Phase 70's location analysis examines.
3. **Patient-level aggregation partly undoes it**, which is why all three rules are reported. See `T15_recording_vs_patient_level.csv`.

## Outcome is near-balanced, unlike murmur

The outcome task is 486/456 -- close to balanced, and the only task in this project that is. Accuracy is therefore *less* misleading here than elsewhere, but it is still not the reporting metric: sensitivity and balanced accuracy lead, per research rule 6. Murmur, by contrast, is Absent 2391 / Present 616 / Unknown 156 at recording level and must never be read the same way.

## Measured effect of each aggregation rule

| rule | sensitivity | specificity | balanced accuracy |
|---|---|---|---|
| any_present | 0.6721 | 0.6217 | 0.5884 |
| max | 0.6176 | 0.6908 | 0.5914 |
| mean | 0.4101 | 0.8561 | 0.5631 |
| *recording level* | 0.4530 | 0.8165 | 0.5824 |

Aggregation moves the operating point rather than creating information: balanced accuracy barely changes while sensitivity and specificity trade against each other. **No rule is declared the winner** -- that is a clinical judgement, and all three are reported.

---

*PV-MEPCG / PulseVision is an academic screening and decision-support prototype. It is not a diagnostic tool and must not be used to diagnose, treat, or make clinical decisions about any patient.*
