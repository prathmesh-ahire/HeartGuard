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

**It does not license** a claim that the binary model is merely fingerprinting.
It clears the source-only baseline by 35.6 sensitivity points and 14.0 balanced
accuracy points, so it is demonstrably using far more than acquisition.

**It does not license** any statement about behaviour on an unseen recording
setup. The share of binary performance attributable to acquisition has **not**
been isolated. The measurement that would isolate it is a
leave-one-sub-collection-out evaluation — train on five collections, test on the
sixth, six times — which **has not been run**. Until it has, the honest position
is: the model is known to use more than acquisition, and is not known to be
independent of it.

## Related

- `Docs/note.md`, 2026-09-10 — the confound entry and the fix applied.
- `outputs/07_multiclass_results/EXP-G1-within_source/` — the de-confounded
  diagnosis task, run on a single sub-collection where the shortcut is
  arithmetically unavailable.
- PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic
  tool.
