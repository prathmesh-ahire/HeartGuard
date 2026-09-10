### T-S3 - PhysioNet Diagnosis Multiclass (within one source)

Supplementary multiclass evidence for Objective 6 on 292 abnormal PhysioNet recordings over 58 subjects, subject-grouped 5-fold, balanced class weighting. Eight diagnosis classes were merged to 3 by the policy in diagnosis_class_merge_policy.csv. Two intervals are reported on every metric because they answer different questions: the fold interval is a Student-t interval over 5 folds, the record interval a percentile bootstrap over recordings. Ranked by macro-F1.

| Model | Name | Folds | macro f1 | macro f1 SD | macro f1 95% CI (folds) | macro f1 95% CI (records) | balanced accuracy | balanced accuracy SD | balanced accuracy 95% CI (folds) | balanced accuracy 95% CI (records) | macro recall | macro recall SD | macro recall 95% CI (folds) | macro recall 95% CI (records) | macro precision | macro precision SD | macro precision 95% CI (folds) | macro precision 95% CI (records) | weighted f1 | weighted f1 SD | weighted f1 95% CI (folds) | weighted f1 95% CI (records) | accuracy | accuracy SD | accuracy 95% CI (folds) | accuracy 95% CI (records) | Classes predicted | Degenerate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M6 | Soft Voting (equal weights) | 5 | 0.333 | 0.071 | [0.2441, 0.4212] | [0.3001, 0.3806] | 0.373 | 0.065 | [0.2926, 0.4539] | [0.3316, 0.4109] | 0.373 | 0.065 | [0.2926, 0.4539] | [0.3316, 0.4109] | 0.323 | 0.072 | [0.2331, 0.4124] | [0.2853, 0.3685] | 0.435 | 0.096 | [0.3163, 0.5547] | [0.3863, 0.5078] | 0.491 | 0.089 | [0.3811, 0.6017] | [0.4281, 0.5479] | 2 | False |
| M5 | Gradient Boosting | 5 | 0.331 | 0.046 | [0.2739, 0.3887] | [0.2917, 0.3867] | 0.351 | 0.030 | [0.3138, 0.3888] | [0.3079, 0.3966] | 0.351 | 0.030 | [0.3138, 0.3888] | [0.3079, 0.3966] | 0.330 | 0.068 | [0.2461, 0.4137] | [0.2785, 0.4117] | 0.422 | 0.049 | [0.3614, 0.4834] | [0.3702, 0.4865] | 0.452 | 0.031 | [0.4137, 0.4911] | [0.3972, 0.5103] | 3 | False |
| M4 | Random Forest | 5 | 0.324 | 0.070 | [0.2376, 0.4106] | [0.2892, 0.3819] | 0.348 | 0.059 | [0.2752, 0.4214] | [0.3043, 0.3908] | 0.348 | 0.059 | [0.2752, 0.4214] | [0.3043, 0.3908] | 0.317 | 0.075 | [0.2242, 0.4106] | [0.2761, 0.4148] | 0.413 | 0.091 | [0.3001, 0.5256] | [0.3651, 0.4811] | 0.446 | 0.078 | [0.3490, 0.5433] | [0.3870, 0.5034] | 3 | False |
| M1 | Logistic Regression | 5 | 0.305 | 0.023 | [0.2771, 0.3331] | [0.2651, 0.3603] | 0.309 | 0.025 | [0.2776, 0.3400] | [0.2596, 0.3599] | 0.309 | 0.025 | [0.2776, 0.3400] | [0.2596, 0.3599] | 0.313 | 0.031 | [0.2751, 0.3519] | [0.2726, 0.3682] | 0.371 | 0.027 | [0.3376, 0.4050] | [0.3219, 0.4355] | 0.367 | 0.023 | [0.3383, 0.3954] | [0.3151, 0.4247] | 3 | False |
| M7 | Soft Voting (optimized weights) | 5 | 0.300 | 0.083 | [0.1961, 0.4033] | [0.2816, 0.3579] | 0.343 | 0.064 | [0.2640, 0.4222] | [0.3032, 0.3828] | 0.343 | 0.064 | [0.2640, 0.4222] | [0.3032, 0.3828] | 0.287 | 0.082 | [0.1854, 0.3890] | [0.2655, 0.3441] | 0.393 | 0.109 | [0.2574, 0.5278] | [0.3601, 0.4752] | 0.450 | 0.082 | [0.3479, 0.5513] | [0.3938, 0.5034] | 3 | False |
| M3 | SVM (RBF kernel) | 5 | 0.295 | 0.058 | [0.2238, 0.3671] | [0.2732, 0.3518] | 0.347 | 0.051 | [0.2841, 0.4108] | [0.3064, 0.3838] | 0.347 | 0.051 | [0.2841, 0.4108] | [0.3064, 0.3838] | 0.302 | 0.069 | [0.2169, 0.3872] | [0.2614, 0.3460] | 0.388 | 0.075 | [0.2950, 0.4817] | [0.3499, 0.4706] | 0.460 | 0.071 | [0.3717, 0.5484] | [0.4007, 0.5171] | 2 | False |

> **THE RECORDING SOURCE IS CONSTANT IN THIS TABLE.** Every record is from training-a, so the sub-collection shortcut that EXP-G1's full five-class track was found to be taking is arithmetically unavailable here: a source-only predictor degenerates to always predicting the majority class, and scores macro-F1 0.2097 / balanced accuracy 0.3333. Model(s) beating that trivial baseline: M6, M5, M4, M1, M7, M3. This is the de-confounded evidence; the five-class table (T-S1) is reported beside it as the negative result.

> This track covers the ABNORMAL recordings only. The 2,575 normal recordings are not a class here: including them would make the largest class 39 times the smallest and would restate the binary task, which EXP-A1 answers on all 3,240 records.

> 'Benign, MVP, Other pathologic' are the classes. 'Other pathologic' pools four conditions that are individually too small to estimate (MPC, AD, MR, AS). It is a statistical convenience and NOT a diagnosis -- no number for it may be read as a claim about any one of the four.

> n = 5 folds. Every interval here is wide and most of them overlap between models; this is supplementary evidence on a small sample, not a model ranking that separates its models.

> Nine subjects carry recordings under more than one diagnosis class. Grouping is by subject, so those subjects contribute to two classes within the same fold; no subject crosses a fold boundary.

> PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic tool.

_Table T-S3 -- PhysioNet Diagnosis Multiclass (within one source)_
_Experiment: EXP-G1_
_Objective: O6 (multiclass classification)_
_Source: outputs/01_dataset_audit/metadata_master.csv; outputs/01_dataset_audit/diagnosis_split_map.csv; outputs/03_features/all_features_matrix.parquet_
_Generated by PV-MEPCG / PulseVision at 2026-09-10T07:37:40.180020+00:00_
