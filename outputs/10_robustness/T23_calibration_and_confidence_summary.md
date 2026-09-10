### T23 - Calibration and Confidence Summary

Brier score and expected calibration error for every model on every task, computed inside each fold from the stored out-of-fold probabilities and averaged over folds. ECE is measured over the confidence of the predicted class in equal-width bins, the same definition src/evaluation/metrics.py uses, so it is defined identically for the binary and the multiclass tracks. 'Confidence - accuracy' is signed: positive means the model claims more certainty than its own hit rate justifies.

| Run | Task | Model | ECE bins | Folds | Scored rows | Brier mean | Brier SD | ECE mean | ECE SD | Confidence mean | Argmax accuracy | Confidence - accuracy | Gap SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-A1 | binary | M1 | 10 | 25 | 16,200 | 0.225 | 0.040 | 0.061 | 0.024 | 0.900 | 0.840 | 0.060 | 0.025 |
| EXP-A1 | binary | M3 | 10 | 25 | 16,200 | 0.183 | 0.026 | 0.057 | 0.020 | 0.826 | 0.877 | -0.051 | 0.026 |
| EXP-A1 | binary | M4 | 10 | 25 | 16,200 | 0.169 | 0.028 | 0.066 | 0.021 | 0.835 | 0.898 | -0.062 | 0.024 |
| EXP-A1 | binary | M5 | 10 | 25 | 16,200 | 0.174 | 0.038 | 0.034 | 0.019 | 0.904 | 0.880 | 0.023 | 0.023 |
| EXP-A1 | binary | M6 | 10 | 25 | 16,200 | 0.169 | 0.025 | 0.077 | 0.025 | 0.822 | 0.898 | -0.075 | 0.027 |
| EXP-A1 | binary | M8 | 10 | 25 | 16,200 | 0.166 | 0.035 | 0.061 | 0.017 | 0.952 | 0.896 | 0.056 | 0.019 |
| EXP-A2 | binary | M1 | 10 | 25 | 16,200 | 0.230 | 0.046 | 0.051 | 0.024 | 0.881 | 0.836 | 0.045 | 0.030 |
| EXP-A2 | binary | M3 | 10 | 25 | 16,200 | 0.185 | 0.024 | 0.053 | 0.022 | 0.831 | 0.878 | -0.047 | 0.029 |
| EXP-A2 | binary | M4 | 10 | 25 | 16,200 | 0.189 | 0.035 | 0.049 | 0.014 | 0.835 | 0.872 | -0.037 | 0.027 |
| EXP-A2 | binary | M5 | 10 | 25 | 16,200 | 0.196 | 0.057 | 0.061 | 0.060 | 0.846 | 0.867 | -0.021 | 0.075 |
| EXP-A2 | binary | M6 | 10 | 25 | 16,200 | 0.168 | 0.024 | 0.073 | 0.026 | 0.829 | 0.898 | -0.069 | 0.031 |
| EXP-A2 | binary | M7 | 10 | 25 | 16,200 | 0.168 | 0.024 | 0.073 | 0.027 | 0.829 | 0.898 | -0.070 | 0.031 |
| EXP-A2 | binary | M8 | 10 | 25 | 16,200 | 0.203 | 0.058 | 0.071 | 0.067 | 0.831 | 0.864 | -0.033 | 0.088 |
| EXP-B1 | pascal_a | M1 | 10 | 10 | 620 | 0.519 | 0.085 | 0.228 | 0.075 | 0.785 | 0.663 | 0.122 | 0.186 |
| EXP-B1 | pascal_a | M3 | 10 | 10 | 620 | 0.465 | 0.026 | 0.190 | 0.044 | 0.532 | 0.660 | -0.128 | 0.062 |
| EXP-B1 | pascal_a | M4 | 10 | 10 | 620 | 0.405 | 0.020 | 0.124 | 0.040 | 0.646 | 0.684 | -0.038 | 0.060 |
| EXP-B1 | pascal_a | M5 | 10 | 10 | 620 | 0.471 | 0.111 | 0.187 | 0.096 | 0.782 | 0.669 | 0.113 | 0.131 |
| EXP-B1 | pascal_a | M6 | 10 | 10 | 620 | 0.431 | 0.010 | 0.173 | 0.046 | 0.554 | 0.671 | -0.117 | 0.052 |
| EXP-B1 | pascal_a | M7 | 10 | 10 | 620 | 0.430 | 0.034 | 0.159 | 0.061 | 0.575 | 0.673 | -0.097 | 0.101 |
| EXP-B2 | pascal_b | M1 | 10 | 5 | 461 | 0.694 | 0.090 | 0.308 | 0.068 | 0.839 | 0.566 | 0.273 | 0.058 |
| EXP-B2 | pascal_b | M3 | 10 | 5 | 461 | 0.422 | 0.011 | 0.052 | 0.021 | 0.716 | 0.727 | -0.010 | 0.027 |
| EXP-B2 | pascal_b | M4 | 10 | 5 | 461 | 0.521 | 0.003 | 0.108 | 0.034 | 0.526 | 0.610 | -0.083 | 0.053 |
| EXP-B2 | pascal_b | M5 | 10 | 5 | 461 | 0.473 | 0.073 | 0.123 | 0.043 | 0.693 | 0.668 | 0.025 | 0.117 |
| EXP-B2 | pascal_b | M6 | 10 | 5 | 461 | 0.417 | 0.016 | 0.077 | 0.033 | 0.673 | 0.746 | -0.073 | 0.035 |
| EXP-B2 | pascal_b | M7 | 10 | 5 | 461 | 0.441 | 0.051 | 0.095 | 0.040 | 0.645 | 0.716 | -0.071 | 0.061 |
| EXP-C1-three_class | circor_murmur | M3 | 10 | 5 | 3,163 | 0.336 | 0.020 | 0.036 | 0.004 | 0.775 | 0.794 | -0.019 | 0.025 |
| EXP-C1-three_class | circor_murmur | M4 | 10 | 5 | 3,163 | 0.415 | 0.017 | 0.145 | 0.040 | 0.602 | 0.744 | -0.142 | 0.045 |
| EXP-C1-three_class | circor_murmur | M5 | 10 | 5 | 3,163 | 0.405 | 0.042 | 0.100 | 0.026 | 0.633 | 0.729 | -0.096 | 0.028 |
| EXP-C1-three_class | circor_murmur | M6 | 10 | 5 | 3,163 | 0.327 | 0.017 | 0.069 | 0.036 | 0.743 | 0.798 | -0.055 | 0.052 |
| EXP-C1-three_class | circor_murmur | M7 | 10 | 5 | 3,163 | 0.363 | 0.033 | 0.093 | 0.051 | 0.698 | 0.774 | -0.075 | 0.075 |
| EXP-C1-two_class | circor_murmur | M3 | 10 | 5 | 3,007 | 0.259 | 0.021 | 0.039 | 0.014 | 0.824 | 0.838 | -0.014 | 0.024 |
| EXP-C1-two_class | circor_murmur | M4 | 10 | 5 | 3,007 | 0.299 | 0.022 | 0.101 | 0.033 | 0.725 | 0.819 | -0.093 | 0.043 |
| EXP-C1-two_class | circor_murmur | M5 | 10 | 5 | 3,007 | 0.284 | 0.035 | 0.071 | 0.021 | 0.798 | 0.817 | -0.019 | 0.076 |
| EXP-C1-two_class | circor_murmur | M6 | 10 | 5 | 3,007 | 0.253 | 0.022 | 0.053 | 0.018 | 0.805 | 0.842 | -0.037 | 0.037 |
| EXP-C1-two_class | circor_murmur | M7 | 10 | 5 | 3,007 | 0.253 | 0.022 | 0.053 | 0.018 | 0.805 | 0.842 | -0.037 | 0.037 |
| EXP-C2 | circor_outcome | M3 | 10 | 5 | 3,163 | 0.476 | 0.009 | 0.044 | 0.026 | 0.575 | 0.607 | -0.032 | 0.034 |
| EXP-C2 | circor_outcome | M4 | 10 | 5 | 3,163 | 0.470 | 0.012 | 0.032 | 0.018 | 0.585 | 0.604 | -0.019 | 0.027 |
| EXP-C2 | circor_outcome | M5 | 10 | 5 | 3,163 | 0.470 | 0.017 | 0.039 | 0.013 | 0.602 | 0.599 | 0.003 | 0.035 |
| EXP-C2 | circor_outcome | M6 | 10 | 5 | 3,163 | 0.471 | 0.010 | 0.039 | 0.027 | 0.574 | 0.609 | -0.034 | 0.033 |
| EXP-C2 | circor_outcome | M7 | 10 | 5 | 3,163 | 0.471 | 0.010 | 0.039 | 0.027 | 0.574 | 0.609 | -0.034 | 0.033 |
| EXP-D1 | circor_outcome | M1 | 10 | 1 | 3,163 | 0.714 | n/a | 0.299 | n/a | 0.830 | 0.531 | 0.299 | n/a |

> ECE IS NOT BIN-COUNT INVARIANT. The reported value uses the bin count in the 'ECE bins' column; calibration_bin_sensitivity.csv carries the same predictions re-binned at 5, 10, 15 and 20 bins. An ECE quoted without its bin count is half a number.

> BRIER IS NOT COMPARABLE ACROSS LABEL SPACES. The multiclass form sums squared error over all classes, so its range is 0-2 where the binary form's is 0-1. Compare models within a run, never a PASCAL A Brier against a PhysioNet one.

> Reliability is measured against the ARGMAX of each model's own probability vector, not against its stored y_pred. M6 and M7 decide by an in-fold selected threshold rather than by argmax, so the two legitimately differ; 'Argmax accuracy' is therefore not the accuracy those two models' tables report.

> EXP-D1 is ONE external evaluation, not cross-validation: n_folds=1 and every SD is undefined. It is adult-to-paediatric transfer and the population mismatch dominates it.

> T78.4 -- the SVM's calibrator, compared on the IDENTICAL 25-fold map with everything else held fixed (n=25): isotonic brier 0.1877 / ECE 0.0566; sigmoid brier 0.1831 / ECE 0.0573. Lower is better for both. Fold wins on Brier: isotonic 6, sigmoid 19, tied 0. The configured method is sigmoid. THE FINAL BINARY MODEL IS M1, a logistic regression that contains no SVM and no post-hoc calibrator, so this comparison is evidence about the M6/M7 ensemble members and not about the deployed estimator.

> PV-MEPCG / PulseVision is an academic screening and decision-support prototype, not a diagnostic tool. A calibrated probability is not a clinical risk.

_Table T23 -- Calibration and Confidence Summary_
_Experiment: EXP-A1, EXP-A2, EXP-B1, EXP-B2, EXP-C1, EXP-C2, EXP-D1_
_Objective: O3 (model comparison), O4 (reliability)_
_Source: D:/Projects/HeartGuard/outputs/10_robustness/calibration/calibration_summary.csv; D:/Projects/HeartGuard/outputs/10_robustness/calibration/calibration_per_fold.csv; D:/Projects/HeartGuard/outputs/10_robustness/calibration/reliability_points.csv; D:/Projects/HeartGuard/outputs/10_robustness/calibration/confidence_histogram.csv; D:/Projects/HeartGuard/outputs/10_robustness/calibration/svm_calibration_method_comparison.csv_
_Generated by PV-MEPCG / PulseVision at 2026-09-10T11:10:12.541592+00:00_
