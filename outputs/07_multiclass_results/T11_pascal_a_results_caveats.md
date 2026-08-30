# PASCAL A (set_a) -- four-class: caveats that travel with the result table

Generated from `outputs/01_dataset_audit/subject_split_map.csv` and the run's own `per_fold_metrics.csv`. Every number below is measured, none typed.

## Sample size and class balance

| class | records | share |
|---|---|---|
| artifact | 40 | 32.3% |
| murmur | 34 | 27.4% |
| normal | 31 | 25.0% |
| extrahls | 19 | 15.3% |
| **total** | **124** | |

The rarest class is **extrahls** at **19 records**. Across the fold map that is roughly 9.5 records per test fold, which is why every headline metric in the companion table carries a confidence interval and why a difference of a few points between two models should not be read as a difference at all.

Always predicting **artifact** would score 32.3% accuracy on this corpus. That is the number any accuracy figure here must be compared against, and it is why the table is ranked by macro-F1 and reports per-class recall (research rule 6).

## `artifact` is not a cardiac class

`artifact` is a RECORDING-QUALITY label, not a cardiac class. EXP-B1 is therefore NOT a four-class cardiac classifier and must never be described as one.

It labels recordings that are unusable -- handling noise, contact artefacts, a stethoscope moving -- rather than a cardiac finding. A high recall on `artifact` measures recording-quality detection, which is useful in its own right and is a different claim from cardiac classification. The honest description of EXP-B1 is: *a four-class acoustic-event classifier over three cardiac categories and one recording-quality category.*

## Results are record-level, not subject-level

PASCAL set_a carries no recoverable subject identifier, so `subject_derived=False` for all 124 records and the folds are stratified at record level. Two rows do share a group: Phase 17 found one recording filed under two class labels (`extrahls__201104021355` and `murmur__201104021355`, envelope correlation 0.999978), and those two share a group key so the same audio cannot land on both sides of a split. That leaves 123 groups over 124 records.

**This is a dataset limitation, not a method limitation**, and it must be stated as such. No claim of subject-level generalization can be made from EXP-B1.

## Rule 4 -- the label spaces were never merged

PASCAL set A and set B were never merged: they are two separate label spaces, evaluated as two separate tasks, with two separate fold maps.

Verified against the fold map rather than asserted:

| check | set A | set B |
|---|---|---|
| records | 124 | 461 |
| classes | artifact, extrahls, murmur, normal | extrastole, murmur, normal |
| CV scheme | repeated_5x2_stratified | grouped_5fold |
| grouping | record-level (no subject id) | subject |
| records shared with the other set | 0 | 0 |

The two sets share the class *name(s)* `murmur`, `normal`, which is exactly why they are kept apart: a shared name is not a shared label space. `normal` in a four-class problem and `normal` in a three-class problem are different targets with different priors, and pooling them would change what both numbers mean.

## What this run produced

Across 10 folds and 6 models, per-fold macro-F1 ranged from 0.5057 to 0.7017. The spread across folds is itself the small-sample effect: it is wider than the spread between models.

Per-class recall, pooled over every model and fold (mean, and the worst single fold):

| class | records | mean recall | worst fold |
|---|---|---|---|
| artifact | 40 | 0.9367 | 0.7000 |
| murmur | 34 | 0.7833 | 0.5294 |
| normal | 31 | 0.3602 | 0.0000 |
| extrahls | 19 | 0.4178 | 0.0000 |

The gap between the largest class (**artifact**, 40 records, recall 0.9367) and the smallest (**extrahls**, 19 records, recall 0.4178) is 0.5189. Balanced class weights reduce that gap; they cannot close it, because weighting reweights the examples that exist and cannot supply the ones that do not.

## Models that never predict some class

Every model emitted every class at least once. No collapsed or partially-collapsed predictor in this run.

---

*PV-MEPCG / PulseVision is an academic screening and decision-support prototype. It is not a diagnostic tool and must not be used to diagnose, treat, or make clinical decisions about any patient.*
