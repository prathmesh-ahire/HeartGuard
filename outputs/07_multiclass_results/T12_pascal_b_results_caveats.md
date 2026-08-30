# PASCAL B (set_b) -- three-class: caveats that travel with the result table

Generated from `outputs/01_dataset_audit/subject_split_map.csv` and the run's own `per_fold_metrics.csv`. Every number below is measured, none typed.

## Sample size and class balance

| class | records | share |
|---|---|---|
| normal | 320 | 69.4% |
| murmur | 95 | 20.6% |
| extrastole | 46 | 10.0% |
| **total** | **461** | |

The rarest class is **extrastole** at **46 records**. Across the fold map that is roughly 9.2 records per test fold, which is why every headline metric in the companion table carries a confidence interval and why a difference of a few points between two models should not be read as a difference at all.

Always predicting **normal** would score 69.4% accuracy on this corpus. That is the number any accuracy figure here must be compared against, and it is why the table is ranked by macro-F1 and reports per-class recall (research rule 6).

## Subject grouping: 165 subjects, not 167

PASCAL set_b is 461 records over **165 subject groups**, derived from the numeric id in the filename. The figure 167 that appears in the task list counts recording *sessions*: three subjects were recorded twice on the same day, two of them with both sessions in the labelled set. Grouping on the session would put the same person on both sides of a fold, so the grouping key is the subject number. Anything quoting 167 subjects is quoting sessions. See the 2026-08-25 entry in `Docs/note.md`.

## Severe class imbalance and what it does to recall

At 320 / 95 / 46 the minority class has roughly 14% of the majority's support. Every model is fitted with balanced class weights, but weighting cannot manufacture examples: the per-class recall column in the companion table is where the imbalance actually shows, and it is the column to read first.

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

Across 5 folds and 6 models, per-fold macro-F1 ranged from 0.3351 to 0.5496. The spread across folds is itself the small-sample effect: it is wider than the spread between models.

Per-class recall, pooled over every model and fold (mean, and the worst single fold):

| class | records | mean recall | worst fold |
|---|---|---|---|
| normal | 320 | 0.8453 | 0.5469 |
| murmur | 95 | 0.3807 | 0.1053 |
| extrastole | 46 | 0.0689 | 0.0000 |

The gap between the largest class (**normal**, 320 records, recall 0.8453) and the smallest (**extrastole**, 46 records, recall 0.0689) is 0.7764. Balanced class weights reduce that gap; they cannot close it, because weighting reweights the examples that exist and cannot supply the ones that do not.

## Models that never predict some class

**Read this before the metrics table.** Each model below never emitted at least one class anywhere in the run. It has not learned that class, however its accuracy reads -- and on a corpus this imbalanced, refusing to use the smallest class is *rewarded* by accuracy. This is what research rule 6 exists to catch.

| model | classes predicted | never predicted | collapsed |
|---|---|---|---|
| M3 | 2 of 3 | `extrastole` | no |
| M6 | 2 of 3 | `extrastole` | no |

A row marked *collapsed* produced one constant answer for every record. Its scores are those of the corresponding trivial baseline and must never be presented as a model result.

---

*PV-MEPCG / PulseVision is an academic screening and decision-support prototype. It is not a diagnostic tool and must not be used to diagnose, treat, or make clinical decisions about any patient.*
