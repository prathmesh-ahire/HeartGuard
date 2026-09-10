# EXP-D1 addendum — the acquisition shortcut is a third cause of the drop

**Written 2026-09-10, after Phase 76. Deliberately NOT merged into
`population_mismatch.json`.** That file is a pre-registration: it carries
`written_before_any_metric: true` and was written on 2026-08-31 before EXP-D1
produced a number. Editing measured results into it afterwards would make that
claim false, and re-generating it would destroy the provenance that gives it its
value. So this addendum sits beside it instead.

`population_mismatch.json` names two causes for EXP-D1's drop: the
adult-to-paediatric population mismatch, and the heterogeneity of PhysioNet's
six sub-collections. Phase 76 found a third, and measured it.

**The framing rule still holds, and this addendum sharpens it rather than
replacing it.** A large drop remains the expected consequence of that mismatch,
and EXP-D1 must still never be written up as "our method does not generalize".
What changes is the *attribution*: the drop is the expected consequence of that
mismatch **and** of a change of acquisition setting, and the measurement below
shows acquisition to be the larger of the two. Age is no longer the headline
cause; it is one of three.

## What was measured

A predictor that receives **no audio at all** — it maps a recording's PhysioNet
sub-collection to that collection's most common class, fitted inside each
training fold — scores as follows:

| Track | Metric | Source-only (no audio) | Best model | Model wins? |
|---|---|---|---|---|
| Binary (EXP-A1) | sensitivity | 0.4933 | M6 0.8791 | yes |
| Binary (EXP-A1) | balanced accuracy | 0.7191 | M6 0.8588 | yes |
| Diagnosis (EXP-G1, 5-class) | macro-F1 | 0.6345 | M5 0.6203 | **no** |

Source: `outputs/09_ablation/source_only_baselines.csv` and
`outputs/07_multiclass_results/EXP-G1/source_confound_baseline.csv`.

## Why it bears on EXP-D1

The sub-collections were recorded with different equipment in different
settings. A model fitted on the pooled corpus can therefore learn the
**acquisition signature** — stethoscope response, room noise, channel
characteristics — instead of cardiac content, because on this corpus that
signature is genuinely predictive of the label.

CirCor was recorded elsewhere, with different equipment. Whatever share of a
PhysioNet-fitted model's performance rests on the acquisition signature does not
transfer, and contributes to EXP-D1's drop **on top of** the age mismatch and
the class-balance heterogeneity already recorded.

## What this does and does not license

**It does not license** describing EXP-D1's drop as wholly a population effect.
There are three known causes and their shares have not been separated.

**It does not license** a claim that the binary model is *merely* fingerprinting.
On the pooled folds it clears the source-only baseline by 35.6 sensitivity
points and 14.0 balanced accuracy points, so within the corpus it is using far
more than acquisition. Read that alongside the update below, which shows the
same model at chance on a collection it has not seen: both are true, and
together they say the model learns real within-corpus structure that does not
survive a change of acquisition setting.

## Update, same day: the leave-one-sub-collection-out result

That measurement has now been run (EXP-F3, `outputs/09_ablation/EXP-F3/`), and
it changes this file's conclusion rather than merely supporting it.

| model | AUC pooled 25-fold | AUC leave-one-source-out |
|---|---|---|
| M1 | 0.9222 | 0.5330 |
| M6 | 0.9395 | 0.4438 |
| M4 | 0.9418 | 0.4256 |

Every model falls to between 0.43 and 0.53. **AUC 0.50 is random.** Several
folds are below chance, which is the signature of a feature-to-label sign flip
between collections — consistent with the already-recorded finding that all 20
top features by pooled Cohen's d reverse sign between sources.

**AUC is rank-based and therefore invariant to class balance**, so the prior
shift caused by holding out a collection cannot explain this. It is not a
calibration or threshold problem; the ranking is absent.

**What this means for EXP-D1.** The model fails to transfer between
sub-collections *within PhysioNet*, where there is no age mismatch whatsoever.
Acquisition is therefore a **larger** share of the PhysioNet-to-CirCor drop than
the adult-to-paediatric mismatch, not a third-order effect. The EXP-D1 write-up
must not present age as the primary cause.

**What it does not license** remains: no statement anywhere that this model
generalizes, is deployment-ready, or that its pooled figures predict field
performance. Those are now affirmatively contradicted by a measurement in this
repository, not merely unmeasured.

**What is still valid, unchanged:** every pooled result. EXP-A1's figures are a
correct repeated 5x5 subject-grouped cross-validation of the PhysioNet corpus.
Both sides of every fold span all six collections, so nothing about those
numbers is wrong — they simply describe within-corpus performance and must be
labelled as such.

## Related

- `Docs/note.md`, 2026-09-10 — the confound entry, the fixes applied, and the
  EXP-F3 entry.
- `outputs/09_ablation/EXP-F3/` and `T-S5_leave_one_source_out_generalization.*`
  — the leave-one-sub-collection-out measurement itself.
- `outputs/07_multiclass_results/EXP-G1-within_source/` — the de-confounded
  diagnosis task, run on a single sub-collection where the shortcut is
  arithmetically unavailable.
- PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic
  tool.
