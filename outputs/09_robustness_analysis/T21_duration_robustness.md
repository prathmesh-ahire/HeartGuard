### T21 - Duration Robustness

Two kinds of evidence about recording length, labelled by the Analysis column. observational_bands partitions each experiment's stored out-of-fold predictions by the T18.3 bands (short under 5 s, medium 5-20 s, long over 20 s), loaded from the same assignment T18 used. interventional_truncation clips the same PhysioNet recordings to 10, 5 and 3 seconds and re-extracts them, scored by a model refitted without their subjects. Only the second half can attribute anything to duration.

| Analysis | Run | Task | Model | Group | Records | Reportable | Folds | Units/fold | Median duration (s) | Shortest (s) | sensitivity mean | sensitivity SD | specificity mean | specificity SD | balanced accuracy mean | balanced accuracy SD | roc auc mean | roc auc SD | macro f1 mean | macro f1 SD | accuracy mean | accuracy SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| observational_bands | EXP-A2 | binary | M1 | long | 1,744 | True | 25 | 348.8 | 29.66 | 20.00 | 0.944 | 0.022 | 0.850 | 0.062 | 0.897 | 0.034 | 0.945 | 0.026 | n/a | n/a | 0.874 | 0.048 |
| observational_bands | EXP-A2 | binary | M1 | medium | 1,496 | True | 25 | 299.2 | 12.05 | 5.31 | 0.709 | 0.116 | 0.809 | 0.051 | 0.759 | 0.054 | 0.865 | 0.039 | n/a | n/a | 0.793 | 0.041 |
| observational_bands | EXP-A2 | binary | M3 | long | 1,744 | True | 25 | 348.8 | 29.66 | 20.00 | 0.838 | 0.041 | 0.893 | 0.045 | 0.866 | 0.032 | 0.938 | 0.026 | n/a | n/a | 0.879 | 0.036 |
| observational_bands | EXP-A2 | binary | M3 | medium | 1,496 | True | 25 | 299.2 | 12.05 | 5.31 | 0.306 | 0.109 | 0.973 | 0.011 | 0.640 | 0.054 | 0.870 | 0.037 | n/a | n/a | 0.879 | 0.026 |
| observational_bands | EXP-A2 | binary | M4 | long | 1,744 | True | 25 | 348.8 | 29.66 | 20.00 | 0.935 | 0.032 | 0.865 | 0.057 | 0.900 | 0.037 | 0.953 | 0.027 | n/a | n/a | 0.883 | 0.047 |
| observational_bands | EXP-A2 | binary | M4 | medium | 1,496 | True | 25 | 299.2 | 12.05 | 5.31 | 0.589 | 0.131 | 0.908 | 0.048 | 0.748 | 0.063 | 0.886 | 0.039 | n/a | n/a | 0.861 | 0.040 |
| observational_bands | EXP-A2 | binary | M5 | long | 1,744 | True | 25 | 348.8 | 29.66 | 20.00 | 0.916 | 0.033 | 0.876 | 0.066 | 0.896 | 0.040 | 0.951 | 0.028 | n/a | n/a | 0.886 | 0.053 |
| observational_bands | EXP-A2 | binary | M5 | medium | 1,496 | True | 25 | 299.2 | 12.05 | 5.31 | 0.610 | 0.146 | 0.889 | 0.067 | 0.750 | 0.059 | 0.887 | 0.037 | n/a | n/a | 0.847 | 0.047 |
| observational_bands | EXP-A2 | binary | M6 | long | 1,744 | True | 25 | 348.8 | 29.66 | 20.00 | 0.970 | 0.027 | 0.832 | 0.064 | 0.901 | 0.032 | 0.951 | 0.028 | n/a | n/a | 0.868 | 0.048 |
| observational_bands | EXP-A2 | binary | M6 | medium | 1,496 | True | 25 | 299.2 | 12.05 | 5.31 | 0.645 | 0.156 | 0.871 | 0.071 | 0.758 | 0.060 | 0.891 | 0.037 | n/a | n/a | 0.837 | 0.049 |
| observational_bands | EXP-A2 | binary | M7 | long | 1,744 | True | 25 | 348.8 | 29.66 | 20.00 | 0.970 | 0.027 | 0.832 | 0.064 | 0.901 | 0.032 | 0.951 | 0.027 | n/a | n/a | 0.867 | 0.049 |
| observational_bands | EXP-A2 | binary | M7 | medium | 1,496 | True | 25 | 299.2 | 12.05 | 5.31 | 0.646 | 0.156 | 0.870 | 0.071 | 0.758 | 0.060 | 0.891 | 0.037 | n/a | n/a | 0.836 | 0.049 |
| observational_bands | EXP-A2 | binary | M8 | long | 1,744 | True | 25 | 348.8 | 29.66 | 20.00 | 0.924 | 0.040 | 0.864 | 0.063 | 0.894 | 0.037 | 0.952 | 0.026 | n/a | n/a | 0.879 | 0.049 |
| observational_bands | EXP-A2 | binary | M8 | medium | 1,496 | True | 25 | 299.2 | 12.05 | 5.31 | 0.621 | 0.145 | 0.887 | 0.059 | 0.754 | 0.061 | 0.887 | 0.036 | n/a | n/a | 0.847 | 0.041 |
| observational_bands | EXP-B1 | pascal_a | M1 | medium | 116 | True | 10 | 58.0 | 8.97 | 5.75 | n/a | n/a | n/a | n/a | 0.654 | 0.063 | n/a | n/a | 0.637 | 0.038 | 0.690 | 0.030 |
| observational_bands | EXP-B1 | pascal_a | M1 | short | 8 | False | 10 | 4.0 | 2.57 | 0.94 | n/a | n/a | n/a | n/a | 0.324 | 0.249 | n/a | n/a | 0.144 | 0.103 | 0.285 | 0.209 |
| observational_bands | EXP-B1 | pascal_a | M3 | medium | 116 | True | 10 | 58.0 | 8.97 | 5.75 | n/a | n/a | n/a | n/a | 0.631 | 0.071 | n/a | n/a | 0.585 | 0.056 | 0.697 | 0.035 |
| observational_bands | EXP-B1 | pascal_a | M3 | short | 8 | False | 10 | 4.0 | 2.57 | 0.94 | n/a | n/a | n/a | n/a | 0.117 | 0.193 | n/a | n/a | 0.041 | 0.066 | 0.113 | 0.183 |
| observational_bands | EXP-B1 | pascal_a | M4 | medium | 116 | True | 10 | 58.0 | 8.97 | 5.75 | n/a | n/a | n/a | n/a | 0.672 | 0.047 | n/a | n/a | 0.627 | 0.028 | 0.700 | 0.025 |
| observational_bands | EXP-B1 | pascal_a | M4 | short | 8 | False | 10 | 4.0 | 2.57 | 0.94 | n/a | n/a | n/a | n/a | 0.362 | 0.178 | n/a | n/a | 0.174 | 0.095 | 0.450 | 0.244 |
| observational_bands | EXP-B1 | pascal_a | M5 | medium | 116 | True | 10 | 58.0 | 8.97 | 5.75 | n/a | n/a | n/a | n/a | 0.640 | 0.052 | n/a | n/a | 0.625 | 0.055 | 0.696 | 0.034 |
| observational_bands | EXP-B1 | pascal_a | M5 | short | 8 | False | 10 | 4.0 | 2.57 | 0.94 | n/a | n/a | n/a | n/a | 0.254 | 0.239 | n/a | n/a | 0.127 | 0.134 | 0.292 | 0.284 |
| observational_bands | EXP-B1 | pascal_a | M6 | medium | 116 | True | 10 | 58.0 | 8.97 | 5.75 | n/a | n/a | n/a | n/a | 0.628 | 0.056 | n/a | n/a | 0.593 | 0.040 | 0.698 | 0.033 |
| observational_bands | EXP-B1 | pascal_a | M6 | short | 8 | False | 10 | 4.0 | 2.57 | 0.94 | n/a | n/a | n/a | n/a | 0.275 | 0.278 | n/a | n/a | 0.139 | 0.146 | 0.278 | 0.285 |
| observational_bands | EXP-B1 | pascal_a | M7 | medium | 116 | True | 10 | 58.0 | 8.97 | 5.75 | n/a | n/a | n/a | n/a | 0.634 | 0.058 | n/a | n/a | 0.608 | 0.053 | 0.700 | 0.034 |
| observational_bands | EXP-B1 | pascal_a | M7 | short | 8 | False | 10 | 4.0 | 2.57 | 0.94 | n/a | n/a | n/a | n/a | 0.278 | 0.208 | n/a | n/a | 0.132 | 0.098 | 0.305 | 0.238 |
| observational_bands | EXP-B2 | pascal_b | M1 | long | 5 | False | 3 | 1.7 | 22.75 | 21.97 | n/a | n/a | n/a | n/a | 0.167 | 0.289 | n/a | n/a | 0.074 | 0.128 | 0.167 | 0.289 |
| observational_bands | EXP-B2 | pascal_b | M1 | medium | 229 | True | 5 | 45.8 | 9.08 | 5.02 | n/a | n/a | n/a | n/a | 0.532 | 0.085 | n/a | n/a | 0.503 | 0.045 | 0.579 | 0.032 |
| observational_bands | EXP-B2 | pascal_b | M1 | short | 227 | True | 5 | 45.4 | 2.92 | 0.76 | n/a | n/a | n/a | n/a | 0.454 | 0.073 | n/a | n/a | 0.420 | 0.066 | 0.562 | 0.046 |
| observational_bands | EXP-B2 | pascal_b | M3 | long | 5 | False | 3 | 1.7 | 22.75 | 21.97 | n/a | n/a | n/a | n/a | 1.000 | 0.000 | n/a | n/a | 0.444 | 0.192 | 1.000 | 0.000 |
| observational_bands | EXP-B2 | pascal_b | M3 | medium | 229 | True | 5 | 45.8 | 9.08 | 5.02 | n/a | n/a | n/a | n/a | 0.414 | 0.028 | n/a | n/a | 0.399 | 0.041 | 0.684 | 0.051 |
| observational_bands | EXP-B2 | pascal_b | M3 | short | 227 | True | 5 | 45.4 | 2.92 | 0.76 | n/a | n/a | n/a | n/a | 0.363 | 0.056 | n/a | n/a | 0.336 | 0.078 | 0.770 | 0.040 |
| observational_bands | EXP-B2 | pascal_b | M4 | long | 5 | False | 3 | 1.7 | 22.75 | 21.97 | n/a | n/a | n/a | n/a | 0.667 | 0.577 | n/a | n/a | 0.333 | 0.333 | 0.667 | 0.577 |
| observational_bands | EXP-B2 | pascal_b | M4 | medium | 229 | True | 5 | 45.8 | 9.08 | 5.02 | n/a | n/a | n/a | n/a | 0.410 | 0.067 | n/a | n/a | 0.394 | 0.056 | 0.536 | 0.042 |
| observational_bands | EXP-B2 | pascal_b | M4 | short | 227 | True | 5 | 45.4 | 2.92 | 0.76 | n/a | n/a | n/a | n/a | 0.417 | 0.085 | n/a | n/a | 0.423 | 0.087 | 0.680 | 0.016 |
| observational_bands | EXP-B2 | pascal_b | M5 | long | 5 | False | 3 | 1.7 | 22.75 | 21.97 | n/a | n/a | n/a | n/a | 0.833 | 0.289 | n/a | n/a | 0.296 | 0.064 | 0.833 | 0.289 |
| observational_bands | EXP-B2 | pascal_b | M5 | medium | 229 | True | 5 | 45.8 | 9.08 | 5.02 | n/a | n/a | n/a | n/a | 0.414 | 0.067 | n/a | n/a | 0.409 | 0.064 | 0.613 | 0.116 |
| observational_bands | EXP-B2 | pascal_b | M5 | short | 227 | True | 5 | 45.4 | 2.92 | 0.76 | n/a | n/a | n/a | n/a | 0.417 | 0.112 | n/a | n/a | 0.416 | 0.142 | 0.722 | 0.087 |
| observational_bands | EXP-B2 | pascal_b | M6 | long | 5 | False | 3 | 1.7 | 22.75 | 21.97 | n/a | n/a | n/a | n/a | 0.833 | 0.289 | n/a | n/a | 0.296 | 0.064 | 0.833 | 0.289 |
| observational_bands | EXP-B2 | pascal_b | M6 | medium | 229 | True | 5 | 45.8 | 9.08 | 5.02 | n/a | n/a | n/a | n/a | 0.436 | 0.020 | n/a | n/a | 0.428 | 0.031 | 0.710 | 0.044 |
| observational_bands | EXP-B2 | pascal_b | M6 | short | 227 | True | 5 | 45.4 | 2.92 | 0.76 | n/a | n/a | n/a | n/a | 0.396 | 0.059 | n/a | n/a | 0.387 | 0.093 | 0.786 | 0.041 |
| observational_bands | EXP-B2 | pascal_b | M7 | long | 5 | False | 3 | 1.7 | 22.75 | 21.97 | n/a | n/a | n/a | n/a | 0.833 | 0.289 | n/a | n/a | 0.296 | 0.064 | 0.833 | 0.289 |
| observational_bands | EXP-B2 | pascal_b | M7 | medium | 229 | True | 5 | 45.8 | 9.08 | 5.02 | n/a | n/a | n/a | n/a | 0.428 | 0.033 | n/a | n/a | 0.422 | 0.041 | 0.680 | 0.098 |
| observational_bands | EXP-B2 | pascal_b | M7 | short | 227 | True | 5 | 45.4 | 2.92 | 0.76 | n/a | n/a | n/a | n/a | 0.388 | 0.066 | n/a | n/a | 0.386 | 0.097 | 0.755 | 0.050 |
| observational_bands | EXP-C1-two_class | circor_murmur | M3 | long | 1,957 | True | 5 | 391.4 | 28.19 | 20.05 | 0.252 | 0.039 | 0.994 | 0.006 | 0.623 | 0.019 | 0.747 | 0.051 | n/a | n/a | 0.849 | 0.017 |
| observational_bands | EXP-C1-two_class | circor_murmur | M3 | medium | 1,050 | True | 5 | 210.0 | 16.18 | 5.15 | 0.228 | 0.094 | 0.983 | 0.010 | 0.605 | 0.051 | 0.729 | 0.099 | n/a | n/a | 0.821 | 0.043 |
| observational_bands | EXP-C1-two_class | circor_murmur | M4 | long | 1,957 | True | 5 | 391.4 | 28.19 | 20.05 | 0.398 | 0.061 | 0.937 | 0.012 | 0.667 | 0.031 | 0.735 | 0.066 | n/a | n/a | 0.832 | 0.013 |
| observational_bands | EXP-C1-two_class | circor_murmur | M4 | medium | 1,050 | True | 5 | 210.0 | 16.18 | 5.15 | 0.469 | 0.198 | 0.885 | 0.029 | 0.677 | 0.095 | 0.708 | 0.126 | n/a | n/a | 0.797 | 0.058 |
| observational_bands | EXP-C1-two_class | circor_murmur | M5 | long | 1,957 | True | 5 | 391.4 | 28.19 | 20.05 | 0.402 | 0.054 | 0.932 | 0.035 | 0.667 | 0.020 | 0.738 | 0.052 | n/a | n/a | 0.828 | 0.019 |
| observational_bands | EXP-C1-two_class | circor_murmur | M5 | medium | 1,050 | True | 5 | 210.0 | 16.18 | 5.15 | 0.445 | 0.215 | 0.894 | 0.055 | 0.670 | 0.094 | 0.727 | 0.120 | n/a | n/a | 0.801 | 0.063 |
| observational_bands | EXP-C1-two_class | circor_murmur | M6 | long | 1,957 | True | 5 | 391.4 | 28.19 | 20.05 | 0.519 | 0.053 | 0.871 | 0.057 | 0.695 | 0.041 | 0.753 | 0.055 | n/a | n/a | 0.802 | 0.044 |
| observational_bands | EXP-C1-two_class | circor_murmur | M6 | medium | 1,050 | True | 5 | 210.0 | 16.18 | 5.15 | 0.558 | 0.192 | 0.796 | 0.059 | 0.677 | 0.081 | 0.733 | 0.115 | n/a | n/a | 0.750 | 0.041 |
| observational_bands | EXP-C1-two_class | circor_murmur | M7 | long | 1,957 | True | 5 | 391.4 | 28.19 | 20.05 | 0.519 | 0.053 | 0.871 | 0.057 | 0.695 | 0.041 | 0.753 | 0.055 | n/a | n/a | 0.802 | 0.044 |
| observational_bands | EXP-C1-two_class | circor_murmur | M7 | medium | 1,050 | True | 5 | 210.0 | 16.18 | 5.15 | 0.558 | 0.192 | 0.796 | 0.059 | 0.677 | 0.081 | 0.733 | 0.115 | n/a | n/a | 0.750 | 0.041 |
| observational_bands | EXP-C2 | circor_outcome | M3 | long | 2,054 | True | 5 | 410.8 | 28.14 | 20.05 | 0.417 | 0.072 | 0.782 | 0.030 | 0.600 | 0.026 | 0.617 | 0.027 | n/a | n/a | 0.606 | 0.031 |
| observational_bands | EXP-C2 | circor_outcome | M3 | medium | 1,109 | True | 5 | 221.8 | 16.10 | 5.15 | 0.501 | 0.074 | 0.711 | 0.076 | 0.606 | 0.061 | 0.636 | 0.072 | n/a | n/a | 0.608 | 0.061 |
| observational_bands | EXP-C2 | circor_outcome | M4 | long | 2,054 | True | 5 | 410.8 | 28.14 | 20.05 | 0.448 | 0.078 | 0.742 | 0.042 | 0.595 | 0.023 | 0.620 | 0.026 | n/a | n/a | 0.599 | 0.025 |
| observational_bands | EXP-C2 | circor_outcome | M4 | medium | 1,109 | True | 5 | 221.8 | 16.10 | 5.15 | 0.553 | 0.105 | 0.669 | 0.091 | 0.611 | 0.064 | 0.647 | 0.073 | n/a | n/a | 0.612 | 0.062 |
| observational_bands | EXP-C2 | circor_outcome | M5 | long | 2,054 | True | 5 | 410.8 | 28.14 | 20.05 | 0.479 | 0.080 | 0.716 | 0.044 | 0.597 | 0.024 | 0.625 | 0.035 | n/a | n/a | 0.600 | 0.024 |
| observational_bands | EXP-C2 | circor_outcome | M5 | medium | 1,109 | True | 5 | 221.8 | 16.10 | 5.15 | 0.519 | 0.102 | 0.673 | 0.039 | 0.596 | 0.052 | 0.633 | 0.059 | n/a | n/a | 0.597 | 0.048 |
| observational_bands | EXP-C2 | circor_outcome | M6 | long | 2,054 | True | 5 | 410.8 | 28.14 | 20.05 | 0.431 | 0.096 | 0.756 | 0.065 | 0.594 | 0.017 | 0.626 | 0.028 | n/a | n/a | 0.598 | 0.017 |
| observational_bands | EXP-C2 | circor_outcome | M6 | medium | 1,109 | True | 5 | 221.8 | 16.10 | 5.15 | 0.511 | 0.157 | 0.697 | 0.105 | 0.604 | 0.058 | 0.647 | 0.073 | n/a | n/a | 0.606 | 0.054 |
| observational_bands | EXP-C2 | circor_outcome | M7 | long | 2,054 | True | 5 | 410.8 | 28.14 | 20.05 | 0.431 | 0.096 | 0.756 | 0.065 | 0.594 | 0.017 | 0.626 | 0.028 | n/a | n/a | 0.598 | 0.017 |
| observational_bands | EXP-C2 | circor_outcome | M7 | medium | 1,109 | True | 5 | 221.8 | 16.10 | 5.15 | 0.511 | 0.157 | 0.697 | 0.105 | 0.604 | 0.058 | 0.647 | 0.073 | n/a | n/a | 0.606 | 0.054 |
| interventional_truncation | EXP-E2-TRUNC | binary | M1 | full (untruncated) | 251 | True | 1 | 251.0 | 22.22 | 7.76 | 0.673 | n/a | 0.894 | n/a | 0.784 | n/a | 0.903 | n/a | n/a | n/a | 0.849 | n/a |
| interventional_truncation | EXP-E2-TRUNC | binary | M1 | 10 s clip | 251 | True | 1 | 251.0 | 9.65 | 7.76 | 0.615 | n/a | 0.925 | n/a | 0.770 | n/a | 0.912 | n/a | n/a | n/a | 0.861 | n/a |
| interventional_truncation | EXP-E2-TRUNC | binary | M1 | 5 s clip | 251 | True | 1 | 251.0 | 5.00 | 5.00 | 0.442 | n/a | 0.950 | n/a | 0.696 | n/a | 0.880 | n/a | n/a | n/a | 0.845 | n/a |
| interventional_truncation | EXP-E2-TRUNC | binary | M1 | 3 s clip | 251 | True | 1 | 251.0 | 3.00 | 3.00 | 0.462 | n/a | 0.910 | n/a | 0.686 | n/a | 0.847 | n/a | n/a | n/a | 0.817 | n/a |

> DURATION IS CONFOUNDED WITH CORPUS. Every recording under 5 s is PASCAL; PhysioNet has none and CirCor has none. A short-band deficit compared across corpora is therefore a PASCAL result wearing a duration label, which is why the truncation study exists.

> The shortest recording in the corpus is 0.763 s (D3_set_b_normal__296_1311682952647_A1). The extractor returns NO NaNs at any duration -- measured over all 7,536 records -- so nothing breaks. What does happen is subtler: env_peak_rate counts heart SOUNDS at roughly 2.4/s on this corpus, so a window that short can physically contain only about 2.0 of them, and a rate estimated from one or two events carries an enormous relative error while looking perfectly finite. Treat every frame-count-dependent feature on a sub-2-second record as an estimate with no usable precision, not as a measurement.

> Clips are taken from a randomly placed window, not from the start. The opening seconds of a recording carry the transducer being settled, so always clipping the head would confound length with position. The window start is drawn from the seeded generator.

> PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic tool.

_Table T21 -- Duration Robustness_
_Experiment: EXP-E2_
_Objective: O4 (robustness)_
_Source: outputs/09_robustness_analysis/EXP-E2/per_band_summary.csv; outputs/09_robustness_analysis/EXP-E2/truncation_metrics.csv; outputs/09_robustness_analysis/EXP-E2/short_record_diagnostics.csv_
_Generated by PV-MEPCG / PulseVision at 2026-09-10T17:03:31.183545+00:00_
