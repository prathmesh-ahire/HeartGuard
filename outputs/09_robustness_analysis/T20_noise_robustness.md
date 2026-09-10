### T20 - Noise Robustness

Two kinds of evidence about noise, labelled by the Analysis column. observational_pp08 partitions each experiment's stored out-of-fold predictions by the PP-08 quality flags -- nothing is altered, the groups differ in more than noise, and a gap is an association. interventional_awgn adds white Gaussian noise at fixed SNRs to a stratified PhysioNet sample and scores it with a model refitted without those recordings' subjects -- only the noise varies, so a gap is causal for additive white noise specifically. The five label spaces are reported separately and never pooled.

| Analysis | Run | Task | Model | Group | Records | Reportable | Folds | Units/fold | Mean SNR (dB) | sensitivity mean | sensitivity SD | specificity mean | specificity SD | balanced accuracy mean | balanced accuracy SD | roc auc mean | roc auc SD | macro f1 mean | macro f1 SD | accuracy mean | accuracy SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| observational_pp08 | EXP-A2 | binary | M1 | clean | 2,763 | True | 25 | 552.6 | 29.92 | 0.901 | 0.030 | 0.870 | 0.045 | 0.886 | 0.023 | 0.946 | 0.019 | n/a | n/a | 0.877 | 0.034 |
| observational_pp08 | EXP-A2 | binary | M1 | low_quality_other | 15 | False | 20 | 3.8 | 30.29 | 0.325 | 0.427 | 0.628 | 0.398 | 0.493 | 0.302 | 0.640 | 0.451 | n/a | n/a | 0.654 | 0.347 |
| observational_pp08 | EXP-A2 | binary | M1 | noisy | 462 | True | 25 | 92.4 | 31.00 | 0.595 | 0.239 | 0.604 | 0.132 | 0.600 | 0.110 | 0.652 | 0.151 | n/a | n/a | 0.598 | 0.107 |
| observational_pp08 | EXP-A2 | binary | M3 | clean | 2,763 | True | 25 | 552.6 | 29.92 | 0.737 | 0.057 | 0.931 | 0.030 | 0.834 | 0.033 | 0.942 | 0.020 | n/a | n/a | 0.889 | 0.027 |
| observational_pp08 | EXP-A2 | binary | M3 | low_quality_other | 15 | False | 20 | 3.8 | 30.29 | 0.329 | 0.446 | 0.927 | 0.172 | 0.600 | 0.226 | 1.000 | 0.000 | n/a | n/a | 0.840 | 0.258 |
| observational_pp08 | EXP-A2 | binary | M3 | noisy | 462 | True | 25 | 92.4 | 31.00 | 0.134 | 0.213 | 0.943 | 0.028 | 0.539 | 0.104 | 0.634 | 0.134 | n/a | n/a | 0.813 | 0.049 |
| observational_pp08 | EXP-A2 | binary | M4 | clean | 2,763 | True | 25 | 552.6 | 29.92 | 0.881 | 0.039 | 0.893 | 0.041 | 0.887 | 0.026 | 0.956 | 0.020 | n/a | n/a | 0.890 | 0.032 |
| observational_pp08 | EXP-A2 | binary | M4 | low_quality_other | 15 | False | 20 | 3.8 | 30.29 | 0.229 | 0.405 | 0.805 | 0.328 | 0.476 | 0.235 | 0.867 | 0.183 | n/a | n/a | 0.682 | 0.358 |
| observational_pp08 | EXP-A2 | binary | M4 | noisy | 462 | True | 25 | 92.4 | 31.00 | 0.387 | 0.232 | 0.849 | 0.091 | 0.618 | 0.112 | 0.650 | 0.147 | n/a | n/a | 0.772 | 0.075 |
| observational_pp08 | EXP-A2 | binary | M5 | clean | 2,763 | True | 25 | 552.6 | 29.92 | 0.874 | 0.051 | 0.895 | 0.050 | 0.884 | 0.031 | 0.955 | 0.020 | n/a | n/a | 0.890 | 0.038 |
| observational_pp08 | EXP-A2 | binary | M5 | low_quality_other | 15 | False | 20 | 3.8 | 30.29 | 0.229 | 0.405 | 0.847 | 0.299 | 0.493 | 0.232 | 0.907 | 0.146 | n/a | n/a | 0.714 | 0.353 |
| observational_pp08 | EXP-A2 | binary | M5 | noisy | 462 | True | 25 | 92.4 | 31.00 | 0.376 | 0.225 | 0.804 | 0.146 | 0.590 | 0.095 | 0.662 | 0.136 | n/a | n/a | 0.734 | 0.109 |
| observational_pp08 | EXP-A2 | binary | M6 | clean | 2,763 | True | 25 | 552.6 | 29.92 | 0.921 | 0.045 | 0.868 | 0.047 | 0.895 | 0.022 | 0.955 | 0.021 | n/a | n/a | 0.879 | 0.033 |
| observational_pp08 | EXP-A2 | binary | M6 | low_quality_other | 15 | False | 20 | 3.8 | 30.29 | 0.312 | 0.465 | 0.727 | 0.385 | 0.503 | 0.311 | 1.000 | 0.000 | n/a | n/a | 0.672 | 0.391 |
| observational_pp08 | EXP-A2 | binary | M6 | noisy | 462 | True | 25 | 92.4 | 31.00 | 0.431 | 0.259 | 0.752 | 0.146 | 0.592 | 0.096 | 0.659 | 0.140 | n/a | n/a | 0.698 | 0.103 |
| observational_pp08 | EXP-A2 | binary | M7 | clean | 2,763 | True | 25 | 552.6 | 29.92 | 0.920 | 0.045 | 0.868 | 0.048 | 0.894 | 0.023 | 0.955 | 0.021 | n/a | n/a | 0.879 | 0.034 |
| observational_pp08 | EXP-A2 | binary | M7 | low_quality_other | 15 | False | 20 | 3.8 | 30.29 | 0.312 | 0.465 | 0.727 | 0.385 | 0.503 | 0.311 | 1.000 | 0.000 | n/a | n/a | 0.672 | 0.391 |
| observational_pp08 | EXP-A2 | binary | M7 | noisy | 462 | True | 25 | 92.4 | 31.00 | 0.435 | 0.261 | 0.751 | 0.145 | 0.593 | 0.098 | 0.659 | 0.140 | n/a | n/a | 0.697 | 0.102 |
| observational_pp08 | EXP-A2 | binary | M8 | clean | 2,763 | True | 25 | 552.6 | 29.92 | 0.879 | 0.051 | 0.888 | 0.046 | 0.883 | 0.027 | 0.956 | 0.019 | n/a | n/a | 0.886 | 0.033 |
| observational_pp08 | EXP-A2 | binary | M8 | low_quality_other | 15 | False | 20 | 3.8 | 30.29 | 0.246 | 0.402 | 0.818 | 0.281 | 0.498 | 0.215 | 0.947 | 0.119 | n/a | n/a | 0.707 | 0.312 |
| observational_pp08 | EXP-A2 | binary | M8 | noisy | 462 | True | 25 | 92.4 | 31.00 | 0.433 | 0.233 | 0.798 | 0.135 | 0.615 | 0.101 | 0.654 | 0.145 | n/a | n/a | 0.733 | 0.101 |
| observational_pp08 | EXP-B1 | pascal_a | M1 | clean | 86 | True | 10 | 43.0 | 29.51 | n/a | n/a | n/a | n/a | 0.607 | 0.057 | n/a | n/a | 0.597 | 0.036 | 0.594 | 0.035 |
| observational_pp08 | EXP-B1 | pascal_a | M1 | low_quality_other | 7 | False | 10 | 3.5 | 19.62 | n/a | n/a | n/a | n/a | 0.447 | 0.341 | n/a | n/a | 0.244 | 0.217 | 0.372 | 0.301 |
| observational_pp08 | EXP-B1 | pascal_a | M1 | noisy | 31 | False | 10 | 15.5 | -0.31 | n/a | n/a | n/a | n/a | 0.747 | 0.241 | n/a | n/a | 0.248 | 0.029 | 0.904 | 0.084 |
| observational_pp08 | EXP-B1 | pascal_a | M3 | clean | 86 | True | 10 | 43.0 | 29.51 | n/a | n/a | n/a | n/a | 0.606 | 0.069 | n/a | n/a | 0.533 | 0.067 | 0.584 | 0.061 |
| observational_pp08 | EXP-B1 | pascal_a | M3 | low_quality_other | 7 | False | 10 | 3.5 | 19.62 | n/a | n/a | n/a | n/a | 0.257 | 0.362 | n/a | n/a | 0.135 | 0.200 | 0.225 | 0.333 |
| observational_pp08 | EXP-B1 | pascal_a | M3 | noisy | 31 | False | 10 | 15.5 | -0.31 | n/a | n/a | n/a | n/a | 0.743 | 0.272 | n/a | n/a | 0.244 | 0.008 | 0.952 | 0.057 |
| observational_pp08 | EXP-B1 | pascal_a | M4 | clean | 86 | True | 10 | 43.0 | 29.51 | n/a | n/a | n/a | n/a | 0.656 | 0.058 | n/a | n/a | 0.601 | 0.044 | 0.593 | 0.047 |
| observational_pp08 | EXP-B1 | pascal_a | M4 | low_quality_other | 7 | False | 10 | 3.5 | 19.62 | n/a | n/a | n/a | n/a | 0.647 | 0.239 | n/a | n/a | 0.315 | 0.135 | 0.697 | 0.204 |
| observational_pp08 | EXP-B1 | pascal_a | M4 | noisy | 31 | False | 10 | 15.5 | -0.31 | n/a | n/a | n/a | n/a | 0.728 | 0.263 | n/a | n/a | 0.241 | 0.007 | 0.934 | 0.054 |
| observational_pp08 | EXP-B1 | pascal_a | M5 | clean | 86 | True | 10 | 43.0 | 29.51 | n/a | n/a | n/a | n/a | 0.607 | 0.058 | n/a | n/a | 0.581 | 0.046 | 0.589 | 0.042 |
| observational_pp08 | EXP-B1 | pascal_a | M5 | low_quality_other | 7 | False | 10 | 3.5 | 19.62 | n/a | n/a | n/a | n/a | 0.531 | 0.389 | n/a | n/a | 0.273 | 0.189 | 0.562 | 0.396 |
| observational_pp08 | EXP-B1 | pascal_a | M5 | noisy | 31 | False | 10 | 15.5 | -0.31 | n/a | n/a | n/a | n/a | 0.716 | 0.254 | n/a | n/a | 0.240 | 0.008 | 0.923 | 0.058 |
| observational_pp08 | EXP-B1 | pascal_a | M6 | clean | 86 | True | 10 | 43.0 | 29.51 | n/a | n/a | n/a | n/a | 0.611 | 0.063 | n/a | n/a | 0.572 | 0.051 | 0.596 | 0.056 |
| observational_pp08 | EXP-B1 | pascal_a | M6 | low_quality_other | 7 | False | 10 | 3.5 | 19.62 | n/a | n/a | n/a | n/a | 0.385 | 0.346 | n/a | n/a | 0.216 | 0.220 | 0.370 | 0.329 |
| observational_pp08 | EXP-B1 | pascal_a | M6 | noisy | 31 | False | 10 | 15.5 | -0.31 | n/a | n/a | n/a | n/a | 0.731 | 0.260 | n/a | n/a | 0.242 | 0.007 | 0.940 | 0.051 |
| observational_pp08 | EXP-B1 | pascal_a | M7 | clean | 86 | True | 10 | 43.0 | 29.51 | n/a | n/a | n/a | n/a | 0.614 | 0.067 | n/a | n/a | 0.574 | 0.052 | 0.594 | 0.046 |
| observational_pp08 | EXP-B1 | pascal_a | M7 | low_quality_other | 7 | False | 10 | 3.5 | 19.62 | n/a | n/a | n/a | n/a | 0.390 | 0.346 | n/a | n/a | 0.236 | 0.211 | 0.415 | 0.362 |
| observational_pp08 | EXP-B1 | pascal_a | M7 | noisy | 31 | False | 10 | 15.5 | -0.31 | n/a | n/a | n/a | n/a | 0.734 | 0.270 | n/a | n/a | 0.242 | 0.008 | 0.941 | 0.058 |
| observational_pp08 | EXP-B2 | pascal_b | M1 | clean | 384 | True | 5 | 76.8 | 17.16 | n/a | n/a | n/a | n/a | 0.483 | 0.039 | n/a | n/a | 0.468 | 0.030 | 0.547 | 0.019 |
| observational_pp08 | EXP-B2 | pascal_b | M1 | low_quality_other | 63 | False | 5 | 12.6 | 19.70 | n/a | n/a | n/a | n/a | 0.491 | 0.134 | n/a | n/a | 0.324 | 0.131 | 0.615 | 0.117 |
| observational_pp08 | EXP-B2 | pascal_b | M1 | noisy | 14 | False | 4 | 3.5 | 3.66 | n/a | n/a | n/a | n/a | 0.812 | 0.239 | n/a | n/a | 0.437 | 0.198 | 0.839 | 0.236 |
| observational_pp08 | EXP-B2 | pascal_b | M3 | clean | 384 | True | 5 | 76.8 | 17.16 | n/a | n/a | n/a | n/a | 0.397 | 0.034 | n/a | n/a | 0.378 | 0.049 | 0.696 | 0.030 |
| observational_pp08 | EXP-B2 | pascal_b | M3 | low_quality_other | 63 | False | 5 | 12.6 | 19.70 | n/a | n/a | n/a | n/a | 0.750 | 0.250 | n/a | n/a | 0.410 | 0.114 | 0.937 | 0.063 |
| observational_pp08 | EXP-B2 | pascal_b | M3 | noisy | 14 | False | 4 | 3.5 | 3.66 | n/a | n/a | n/a | n/a | 0.688 | 0.239 | n/a | n/a | 0.304 | 0.169 | 0.652 | 0.341 |
| observational_pp08 | EXP-B2 | pascal_b | M4 | clean | 384 | True | 5 | 76.8 | 17.16 | n/a | n/a | n/a | n/a | 0.406 | 0.049 | n/a | n/a | 0.400 | 0.051 | 0.565 | 0.019 |
| observational_pp08 | EXP-B2 | pascal_b | M4 | low_quality_other | 63 | False | 5 | 12.6 | 19.70 | n/a | n/a | n/a | n/a | 0.688 | 0.207 | n/a | n/a | 0.398 | 0.127 | 0.876 | 0.115 |
| observational_pp08 | EXP-B2 | pascal_b | M4 | noisy | 14 | False | 4 | 3.5 | 3.66 | n/a | n/a | n/a | n/a | 0.675 | 0.236 | n/a | n/a | 0.306 | 0.068 | 0.705 | 0.223 |
| observational_pp08 | EXP-B2 | pascal_b | M5 | clean | 384 | True | 5 | 76.8 | 17.16 | n/a | n/a | n/a | n/a | 0.411 | 0.078 | n/a | n/a | 0.413 | 0.092 | 0.624 | 0.117 |
| observational_pp08 | EXP-B2 | pascal_b | M5 | low_quality_other | 63 | False | 5 | 12.6 | 19.70 | n/a | n/a | n/a | n/a | 0.735 | 0.233 | n/a | n/a | 0.408 | 0.116 | 0.922 | 0.052 |
| observational_pp08 | EXP-B2 | pascal_b | M5 | noisy | 14 | False | 4 | 3.5 | 3.66 | n/a | n/a | n/a | n/a | 0.787 | 0.253 | n/a | n/a | 0.414 | 0.189 | 0.804 | 0.243 |
| observational_pp08 | EXP-B2 | pascal_b | M6 | clean | 384 | True | 5 | 76.8 | 17.16 | n/a | n/a | n/a | n/a | 0.417 | 0.032 | n/a | n/a | 0.406 | 0.044 | 0.716 | 0.020 |
| observational_pp08 | EXP-B2 | pascal_b | M6 | low_quality_other | 63 | False | 5 | 12.6 | 19.70 | n/a | n/a | n/a | n/a | 0.750 | 0.250 | n/a | n/a | 0.410 | 0.114 | 0.937 | 0.063 |
| observational_pp08 | EXP-B2 | pascal_b | M6 | noisy | 14 | False | 4 | 3.5 | 3.66 | n/a | n/a | n/a | n/a | 0.729 | 0.208 | n/a | n/a | 0.354 | 0.126 | 0.714 | 0.254 |
| observational_pp08 | EXP-B2 | pascal_b | M7 | clean | 384 | True | 5 | 76.8 | 17.16 | n/a | n/a | n/a | n/a | 0.408 | 0.032 | n/a | n/a | 0.399 | 0.042 | 0.678 | 0.081 |
| observational_pp08 | EXP-B2 | pascal_b | M7 | low_quality_other | 63 | False | 5 | 12.6 | 19.70 | n/a | n/a | n/a | n/a | 0.750 | 0.250 | n/a | n/a | 0.410 | 0.114 | 0.937 | 0.063 |
| observational_pp08 | EXP-B2 | pascal_b | M7 | noisy | 14 | False | 4 | 3.5 | 3.66 | n/a | n/a | n/a | n/a | 0.771 | 0.208 | n/a | n/a | 0.392 | 0.141 | 0.777 | 0.211 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M3 | clean | 1,612 | True | 5 | 322.4 | 9.90 | 0.307 | 0.086 | 0.986 | 0.009 | 0.647 | 0.045 | 0.761 | 0.081 | n/a | n/a | 0.817 | 0.028 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M3 | low_quality_other | 7 | False | 4 | 1.8 | 10.17 | 0.000 | 0.000 | 1.000 | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | n/a | n/a | 0.750 | 0.289 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M3 | noisy | 1,388 | True | 5 | 277.6 | 0.97 | 0.141 | 0.059 | 0.994 | 0.002 | 0.567 | 0.029 | 0.704 | 0.052 | n/a | n/a | 0.863 | 0.009 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M4 | clean | 1,612 | True | 5 | 322.4 | 9.90 | 0.505 | 0.146 | 0.902 | 0.027 | 0.703 | 0.083 | 0.761 | 0.090 | n/a | n/a | 0.802 | 0.055 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M4 | low_quality_other | 7 | False | 4 | 1.8 | 10.17 | 0.250 | 0.500 | 1.000 | 0.000 | 0.625 | 0.250 | 1.000 | 0.000 | n/a | n/a | 0.875 | 0.250 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M4 | noisy | 1,388 | True | 5 | 277.6 | 0.97 | 0.291 | 0.100 | 0.936 | 0.022 | 0.613 | 0.042 | 0.675 | 0.064 | n/a | n/a | 0.837 | 0.015 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M5 | clean | 1,612 | True | 5 | 322.4 | 9.90 | 0.488 | 0.138 | 0.899 | 0.053 | 0.694 | 0.070 | 0.765 | 0.081 | n/a | n/a | 0.796 | 0.050 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M5 | low_quality_other | 7 | False | 4 | 1.8 | 10.17 | 0.250 | 0.500 | 1.000 | 0.000 | 0.625 | 0.250 | 1.000 | 0.000 | n/a | n/a | 0.875 | 0.250 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M5 | noisy | 1,388 | True | 5 | 277.6 | 0.97 | 0.298 | 0.114 | 0.939 | 0.037 | 0.619 | 0.043 | 0.694 | 0.070 | n/a | n/a | 0.841 | 0.022 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M6 | clean | 1,612 | True | 5 | 322.4 | 9.90 | 0.617 | 0.104 | 0.816 | 0.058 | 0.716 | 0.049 | 0.772 | 0.086 | n/a | n/a | 0.766 | 0.042 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M6 | low_quality_other | 7 | False | 4 | 1.8 | 10.17 | 0.250 | 0.500 | 1.000 | 0.000 | 0.625 | 0.250 | 1.000 | 0.000 | n/a | n/a | 0.875 | 0.250 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M6 | noisy | 1,388 | True | 5 | 277.6 | 0.97 | 0.417 | 0.046 | 0.874 | 0.050 | 0.646 | 0.017 | 0.702 | 0.051 | n/a | n/a | 0.805 | 0.039 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M7 | clean | 1,612 | True | 5 | 322.4 | 9.90 | 0.617 | 0.104 | 0.816 | 0.058 | 0.716 | 0.049 | 0.772 | 0.086 | n/a | n/a | 0.766 | 0.042 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M7 | low_quality_other | 7 | False | 4 | 1.8 | 10.17 | 0.250 | 0.500 | 1.000 | 0.000 | 0.625 | 0.250 | 1.000 | 0.000 | n/a | n/a | 0.875 | 0.250 |
| observational_pp08 | EXP-C1-two_class | circor_murmur | M7 | noisy | 1,388 | True | 5 | 277.6 | 0.97 | 0.417 | 0.046 | 0.874 | 0.050 | 0.646 | 0.017 | 0.702 | 0.051 | n/a | n/a | 0.805 | 0.039 |
| observational_pp08 | EXP-C2 | circor_outcome | M3 | clean | 1,655 | True | 5 | 331.0 | 9.84 | 0.433 | 0.059 | 0.772 | 0.040 | 0.603 | 0.024 | 0.637 | 0.020 | n/a | n/a | 0.608 | 0.029 |
| observational_pp08 | EXP-C2 | circor_outcome | M3 | low_quality_other | 8 | False | 4 | 2.0 | 10.17 | 0.292 | 0.344 | 0.000 | 0.000 | 0.292 | 0.344 | n/a | n/a | n/a | n/a | 0.292 | 0.344 |
| observational_pp08 | EXP-C2 | circor_outcome | M3 | noisy | 1,500 | True | 5 | 300.0 | 0.89 | 0.459 | 0.078 | 0.745 | 0.040 | 0.602 | 0.041 | 0.610 | 0.043 | n/a | n/a | 0.607 | 0.044 |
| observational_pp08 | EXP-C2 | circor_outcome | M4 | clean | 1,655 | True | 5 | 331.0 | 9.84 | 0.496 | 0.046 | 0.710 | 0.033 | 0.603 | 0.016 | 0.638 | 0.029 | n/a | n/a | 0.606 | 0.013 |
| observational_pp08 | EXP-C2 | circor_outcome | M4 | low_quality_other | 8 | False | 4 | 2.0 | 10.17 | 0.375 | 0.479 | 0.750 | 0.354 | 0.562 | 0.315 | n/a | n/a | n/a | n/a | 0.750 | 0.289 |
| observational_pp08 | EXP-C2 | circor_outcome | M4 | noisy | 1,500 | True | 5 | 300.0 | 0.89 | 0.470 | 0.091 | 0.725 | 0.048 | 0.597 | 0.036 | 0.619 | 0.038 | n/a | n/a | 0.601 | 0.037 |
| observational_pp08 | EXP-C2 | circor_outcome | M5 | clean | 1,655 | True | 5 | 331.0 | 9.84 | 0.498 | 0.050 | 0.704 | 0.034 | 0.601 | 0.021 | 0.634 | 0.028 | n/a | n/a | 0.604 | 0.018 |
| observational_pp08 | EXP-C2 | circor_outcome | M5 | low_quality_other | 8 | False | 4 | 2.0 | 10.17 | 0.292 | 0.344 | 0.750 | 0.354 | 0.479 | 0.172 | n/a | n/a | n/a | n/a | 0.667 | 0.236 |
| observational_pp08 | EXP-C2 | circor_outcome | M5 | noisy | 1,500 | True | 5 | 300.0 | 0.89 | 0.485 | 0.097 | 0.696 | 0.042 | 0.590 | 0.041 | 0.622 | 0.047 | n/a | n/a | 0.593 | 0.040 |
| observational_pp08 | EXP-C2 | circor_outcome | M6 | clean | 1,655 | True | 5 | 331.0 | 9.84 | 0.461 | 0.085 | 0.752 | 0.087 | 0.607 | 0.009 | 0.644 | 0.024 | n/a | n/a | 0.609 | 0.006 |
| observational_pp08 | EXP-C2 | circor_outcome | M6 | low_quality_other | 8 | False | 4 | 2.0 | 10.17 | 0.292 | 0.344 | 0.250 | 0.354 | 0.354 | 0.292 | n/a | n/a | n/a | n/a | 0.417 | 0.289 |
| observational_pp08 | EXP-C2 | circor_outcome | M6 | noisy | 1,500 | True | 5 | 300.0 | 0.89 | 0.454 | 0.130 | 0.719 | 0.071 | 0.587 | 0.040 | 0.621 | 0.040 | n/a | n/a | 0.593 | 0.038 |
| observational_pp08 | EXP-C2 | circor_outcome | M7 | clean | 1,655 | True | 5 | 331.0 | 9.84 | 0.461 | 0.085 | 0.752 | 0.087 | 0.607 | 0.009 | 0.644 | 0.024 | n/a | n/a | 0.609 | 0.006 |
| observational_pp08 | EXP-C2 | circor_outcome | M7 | low_quality_other | 8 | False | 4 | 2.0 | 10.17 | 0.292 | 0.344 | 0.250 | 0.354 | 0.354 | 0.292 | n/a | n/a | n/a | n/a | 0.417 | 0.289 |
| observational_pp08 | EXP-C2 | circor_outcome | M7 | noisy | 1,500 | True | 5 | 300.0 | 0.89 | 0.454 | 0.130 | 0.719 | 0.071 | 0.587 | 0.040 | 0.621 | 0.040 | n/a | n/a | 0.593 | 0.038 |
| interventional_awgn | EXP-E1-AWGN | binary | M1 | clean (no noise added) | 251 | True | 1 | 251.0 | inf | 0.673 | n/a | 0.894 | n/a | 0.784 | n/a | 0.903 | n/a | n/a | n/a | 0.849 | n/a |
| interventional_awgn | EXP-E1-AWGN | binary | M1 | 20 dB SNR | 251 | True | 1 | 251.0 | 21.53 | 0.635 | n/a | 0.930 | n/a | 0.782 | n/a | 0.903 | n/a | n/a | n/a | 0.869 | n/a |
| interventional_awgn | EXP-E1-AWGN | binary | M1 | 15 dB SNR | 251 | True | 1 | 251.0 | 16.64 | 0.423 | n/a | 0.945 | n/a | 0.684 | n/a | 0.892 | n/a | n/a | n/a | 0.837 | n/a |
| interventional_awgn | EXP-E1-AWGN | binary | M1 | 10 dB SNR | 251 | True | 1 | 251.0 | 11.90 | 0.173 | n/a | 0.975 | n/a | 0.574 | n/a | 0.839 | n/a | n/a | n/a | 0.809 | n/a |
| interventional_awgn | EXP-E1-AWGN | binary | M1 | 5 dB SNR | 251 | True | 1 | 251.0 | 7.48 | 0.096 | n/a | 0.995 | n/a | 0.546 | n/a | 0.800 | n/a | n/a | n/a | 0.809 | n/a |
| interventional_awgn | EXP-E1-AWGN | binary | M1 | 0 dB SNR | 251 | True | 1 | 251.0 | 3.68 | 0.000 | n/a | 1.000 | n/a | 0.500 | n/a | 0.746 | n/a | n/a | n/a | 0.793 | n/a |

> Cross-checked against PhysioNet's own REFERENCE-SQI annotation over 3240 records: agreement 0.8960, but recall of the SQI-poor records only 0.6923. The two are measuring related but DIFFERENT things, so a PP-08 group must never be described as 'the noisy recordings' without saying whose definition of noisy. PP-08's own SNR proxy does separate the two SQI classes in the expected direction (mean 30.18 dB good vs 29.35 dB poor).

> Additive white Gaussian noise is a laboratory stressor, not a model of stethoscope noise -- real contamination is coloured, non-stationary and often in-band (breathing, bowel sounds, rubbing). The AWGN curve bounds sensitivity to a broadband disturbance; it does not predict field performance.

> SNR is set against the RAW recording, before the 20-400 Hz bandpass. The filter then removes the out-of-band share, so the realised post-filter SNR is higher than the nominal level and is reported beside it rather than assumed.

> PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic tool.

_Table T20 -- Noise Robustness_
_Experiment: EXP-E1_
_Objective: O4 (robustness)_
_Source: outputs/02_preprocessing/signal_quality_flags.csv; outputs/09_robustness_analysis/EXP-E1/awgn_sweep_metrics.csv; outputs/09_robustness_analysis/EXP-E1/sqi_crosscheck.csv_
_Generated by PV-MEPCG / PulseVision at 2026-09-10T17:03:01.073021+00:00_
