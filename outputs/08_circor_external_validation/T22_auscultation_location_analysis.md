### T22 - Auscultation Location Analysis

CirCor out-of-fold predictions stratified by the chest position the recording was taken from, for the murmur task in both label spaces and for the clinical outcome task. Nothing is retrained: this re-scores the predictions EXP-C1 and EXP-C2 already produced. Phc holds four recordings in the whole corpus and is carried with Reportable = False; its numbers are arithmetic noise and must not be quoted. Mean and SD are over the five patient-grouped folds.

| Run | Task | Model | Location | Recordings | Reportable | Folds | Units/fold | sensitivity mean | sensitivity SD | specificity mean | specificity SD | balanced accuracy mean | balanced accuracy SD | macro f1 mean | macro f1 SD | accuracy mean | accuracy SD | Why not reportable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-C1-three_class | circor_murmur | M3 | AV | 800 | True | 5 | 160.0 | n/a | n/a | n/a | n/a | 0.415 | 0.021 | 0.422 | 0.026 | 0.789 | 0.020 |  |
| EXP-C1-three_class | circor_murmur | M3 | MV | 861 | True | 5 | 172.2 | n/a | n/a | n/a | n/a | 0.406 | 0.011 | 0.409 | 0.016 | 0.772 | 0.015 |  |
| EXP-C1-three_class | circor_murmur | M3 | PV | 766 | True | 5 | 153.2 | n/a | n/a | n/a | n/a | 0.444 | 0.039 | 0.460 | 0.048 | 0.820 | 0.023 |  |
| EXP-C1-three_class | circor_murmur | M3 | Phc | 4 | False | 3 | 1.3 | n/a | n/a | n/a | n/a | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-three_class | circor_murmur | M3 | TV | 732 | True | 5 | 146.4 | n/a | n/a | n/a | n/a | 0.399 | 0.041 | 0.403 | 0.060 | 0.798 | 0.036 |  |
| EXP-C1-three_class | circor_murmur | M4 | AV | 800 | True | 5 | 160.0 | n/a | n/a | n/a | n/a | 0.542 | 0.033 | 0.506 | 0.032 | 0.694 | 0.030 |  |
| EXP-C1-three_class | circor_murmur | M4 | MV | 861 | True | 5 | 172.2 | n/a | n/a | n/a | n/a | 0.535 | 0.067 | 0.517 | 0.051 | 0.715 | 0.051 |  |
| EXP-C1-three_class | circor_murmur | M4 | PV | 766 | True | 5 | 153.2 | n/a | n/a | n/a | n/a | 0.564 | 0.040 | 0.567 | 0.058 | 0.788 | 0.039 |  |
| EXP-C1-three_class | circor_murmur | M4 | Phc | 4 | False | 3 | 1.3 | n/a | n/a | n/a | n/a | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-three_class | circor_murmur | M4 | TV | 732 | True | 5 | 146.4 | n/a | n/a | n/a | n/a | 0.523 | 0.066 | 0.529 | 0.061 | 0.784 | 0.029 |  |
| EXP-C1-three_class | circor_murmur | M5 | AV | 800 | True | 5 | 160.0 | n/a | n/a | n/a | n/a | 0.563 | 0.043 | 0.509 | 0.028 | 0.686 | 0.033 |  |
| EXP-C1-three_class | circor_murmur | M5 | MV | 861 | True | 5 | 172.2 | n/a | n/a | n/a | n/a | 0.510 | 0.036 | 0.491 | 0.039 | 0.690 | 0.061 |  |
| EXP-C1-three_class | circor_murmur | M5 | PV | 766 | True | 5 | 153.2 | n/a | n/a | n/a | n/a | 0.540 | 0.060 | 0.530 | 0.061 | 0.768 | 0.044 |  |
| EXP-C1-three_class | circor_murmur | M5 | Phc | 4 | False | 3 | 1.3 | n/a | n/a | n/a | n/a | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-three_class | circor_murmur | M5 | TV | 732 | True | 5 | 146.4 | n/a | n/a | n/a | n/a | 0.558 | 0.070 | 0.557 | 0.061 | 0.781 | 0.024 |  |
| EXP-C1-three_class | circor_murmur | M6 | AV | 800 | True | 5 | 160.0 | n/a | n/a | n/a | n/a | 0.417 | 0.037 | 0.424 | 0.045 | 0.794 | 0.020 |  |
| EXP-C1-three_class | circor_murmur | M6 | MV | 861 | True | 5 | 172.2 | n/a | n/a | n/a | n/a | 0.403 | 0.016 | 0.404 | 0.022 | 0.771 | 0.017 |  |
| EXP-C1-three_class | circor_murmur | M6 | PV | 766 | True | 5 | 153.2 | n/a | n/a | n/a | n/a | 0.439 | 0.047 | 0.454 | 0.059 | 0.821 | 0.026 |  |
| EXP-C1-three_class | circor_murmur | M6 | Phc | 4 | False | 3 | 1.3 | n/a | n/a | n/a | n/a | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-three_class | circor_murmur | M6 | TV | 732 | True | 5 | 146.4 | n/a | n/a | n/a | n/a | 0.413 | 0.036 | 0.421 | 0.049 | 0.809 | 0.023 |  |
| EXP-C1-three_class | circor_murmur | M7 | AV | 800 | True | 5 | 160.0 | n/a | n/a | n/a | n/a | 0.468 | 0.081 | 0.462 | 0.069 | 0.754 | 0.046 |  |
| EXP-C1-three_class | circor_murmur | M7 | MV | 861 | True | 5 | 172.2 | n/a | n/a | n/a | n/a | 0.441 | 0.049 | 0.447 | 0.053 | 0.747 | 0.037 |  |
| EXP-C1-three_class | circor_murmur | M7 | PV | 766 | True | 5 | 153.2 | n/a | n/a | n/a | n/a | 0.501 | 0.057 | 0.527 | 0.065 | 0.814 | 0.039 |  |
| EXP-C1-three_class | circor_murmur | M7 | Phc | 4 | False | 3 | 1.3 | n/a | n/a | n/a | n/a | 1.000 | 0.000 | 0.333 | 0.000 | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-three_class | circor_murmur | M7 | TV | 732 | True | 5 | 146.4 | n/a | n/a | n/a | n/a | 0.420 | 0.049 | 0.426 | 0.060 | 0.785 | 0.042 |  |
| EXP-C1-two_class | circor_murmur | M3 | AV | 755 | True | 5 | 151.0 | 0.217 | 0.046 | 0.983 | 0.008 | 0.600 | 0.024 | n/a | n/a | 0.829 | 0.015 |  |
| EXP-C1-two_class | circor_murmur | M3 | MV | 809 | True | 5 | 161.8 | 0.257 | 0.042 | 0.989 | 0.007 | 0.623 | 0.021 | n/a | n/a | 0.833 | 0.016 |  |
| EXP-C1-two_class | circor_murmur | M3 | PV | 733 | True | 5 | 146.6 | 0.290 | 0.097 | 0.995 | 0.005 | 0.643 | 0.048 | n/a | n/a | 0.854 | 0.016 |  |
| EXP-C1-two_class | circor_murmur | M3 | Phc | 4 | False | 3 | 1.3 | 0.667 | 0.577 | 1.000 | n/a | 0.833 | 0.289 | n/a | n/a | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-two_class | circor_murmur | M3 | TV | 706 | True | 5 | 141.2 | 0.216 | 0.098 | 0.993 | 0.007 | 0.605 | 0.052 | n/a | n/a | 0.836 | 0.023 |  |
| EXP-C1-two_class | circor_murmur | M4 | AV | 755 | True | 5 | 151.0 | 0.422 | 0.098 | 0.898 | 0.030 | 0.660 | 0.050 | n/a | n/a | 0.802 | 0.029 |  |
| EXP-C1-two_class | circor_murmur | M4 | MV | 809 | True | 5 | 161.8 | 0.396 | 0.100 | 0.906 | 0.025 | 0.651 | 0.056 | n/a | n/a | 0.797 | 0.039 |  |
| EXP-C1-two_class | circor_murmur | M4 | PV | 733 | True | 5 | 146.6 | 0.491 | 0.154 | 0.932 | 0.027 | 0.711 | 0.082 | n/a | n/a | 0.844 | 0.040 |  |
| EXP-C1-two_class | circor_murmur | M4 | Phc | 4 | False | 3 | 1.3 | 0.667 | 0.577 | 1.000 | n/a | 0.833 | 0.289 | n/a | n/a | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-two_class | circor_murmur | M4 | TV | 706 | True | 5 | 141.2 | 0.412 | 0.107 | 0.943 | 0.032 | 0.678 | 0.043 | n/a | n/a | 0.836 | 0.019 |  |
| EXP-C1-two_class | circor_murmur | M5 | AV | 755 | True | 5 | 151.0 | 0.423 | 0.135 | 0.904 | 0.040 | 0.663 | 0.055 | n/a | n/a | 0.807 | 0.023 |  |
| EXP-C1-two_class | circor_murmur | M5 | MV | 809 | True | 5 | 161.8 | 0.390 | 0.093 | 0.915 | 0.050 | 0.652 | 0.046 | n/a | n/a | 0.803 | 0.043 |  |
| EXP-C1-two_class | circor_murmur | M5 | PV | 733 | True | 5 | 146.6 | 0.485 | 0.094 | 0.920 | 0.045 | 0.703 | 0.045 | n/a | n/a | 0.833 | 0.033 |  |
| EXP-C1-two_class | circor_murmur | M5 | Phc | 4 | False | 3 | 1.3 | 0.667 | 0.577 | 1.000 | n/a | 0.833 | 0.289 | n/a | n/a | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-two_class | circor_murmur | M5 | TV | 706 | True | 5 | 141.2 | 0.390 | 0.134 | 0.938 | 0.045 | 0.664 | 0.048 | n/a | n/a | 0.827 | 0.018 |  |
| EXP-C1-two_class | circor_murmur | M6 | AV | 755 | True | 5 | 151.0 | 0.571 | 0.104 | 0.809 | 0.068 | 0.690 | 0.044 | n/a | n/a | 0.762 | 0.047 |  |
| EXP-C1-two_class | circor_murmur | M6 | MV | 809 | True | 5 | 161.8 | 0.490 | 0.090 | 0.832 | 0.050 | 0.661 | 0.053 | n/a | n/a | 0.759 | 0.047 |  |
| EXP-C1-two_class | circor_murmur | M6 | PV | 733 | True | 5 | 146.6 | 0.625 | 0.055 | 0.866 | 0.065 | 0.746 | 0.045 | n/a | n/a | 0.818 | 0.054 |  |
| EXP-C1-two_class | circor_murmur | M6 | Phc | 4 | False | 3 | 1.3 | 0.667 | 0.577 | 1.000 | n/a | 0.833 | 0.289 | n/a | n/a | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-two_class | circor_murmur | M6 | TV | 706 | True | 5 | 141.2 | 0.503 | 0.102 | 0.876 | 0.058 | 0.689 | 0.027 | n/a | n/a | 0.801 | 0.029 |  |
| EXP-C1-two_class | circor_murmur | M7 | AV | 755 | True | 5 | 151.0 | 0.571 | 0.104 | 0.809 | 0.068 | 0.690 | 0.044 | n/a | n/a | 0.762 | 0.047 |  |
| EXP-C1-two_class | circor_murmur | M7 | MV | 809 | True | 5 | 161.8 | 0.490 | 0.090 | 0.832 | 0.050 | 0.661 | 0.053 | n/a | n/a | 0.759 | 0.047 |  |
| EXP-C1-two_class | circor_murmur | M7 | PV | 733 | True | 5 | 146.6 | 0.625 | 0.055 | 0.866 | 0.065 | 0.746 | 0.045 | n/a | n/a | 0.818 | 0.054 |  |
| EXP-C1-two_class | circor_murmur | M7 | Phc | 4 | False | 3 | 1.3 | 0.667 | 0.577 | 1.000 | n/a | 0.833 | 0.289 | n/a | n/a | 1.000 | 0.000 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C1-two_class | circor_murmur | M7 | TV | 706 | True | 5 | 141.2 | 0.503 | 0.102 | 0.876 | 0.058 | 0.689 | 0.027 | n/a | n/a | 0.801 | 0.029 |  |
| EXP-C2 | circor_outcome | M3 | AV | 800 | True | 5 | 160.0 | 0.507 | 0.050 | 0.704 | 0.053 | 0.606 | 0.039 | n/a | n/a | 0.605 | 0.040 |  |
| EXP-C2 | circor_outcome | M3 | MV | 861 | True | 5 | 172.2 | 0.491 | 0.068 | 0.714 | 0.040 | 0.603 | 0.026 | n/a | n/a | 0.602 | 0.024 |  |
| EXP-C2 | circor_outcome | M3 | PV | 766 | True | 5 | 153.2 | 0.444 | 0.087 | 0.757 | 0.047 | 0.600 | 0.035 | n/a | n/a | 0.612 | 0.032 |  |
| EXP-C2 | circor_outcome | M3 | Phc | 4 | False | 2 | 2.0 | 0.667 | 0.471 | n/a | n/a | 0.667 | 0.471 | n/a | n/a | 0.667 | 0.471 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C2 | circor_outcome | M3 | TV | 732 | True | 5 | 146.4 | 0.316 | 0.086 | 0.862 | 0.046 | 0.589 | 0.046 | n/a | n/a | 0.609 | 0.045 |  |
| EXP-C2 | circor_outcome | M4 | AV | 800 | True | 5 | 160.0 | 0.522 | 0.067 | 0.659 | 0.036 | 0.590 | 0.024 | n/a | n/a | 0.590 | 0.024 |  |
| EXP-C2 | circor_outcome | M4 | MV | 861 | True | 5 | 172.2 | 0.548 | 0.059 | 0.690 | 0.045 | 0.619 | 0.022 | n/a | n/a | 0.619 | 0.021 |  |
| EXP-C2 | circor_outcome | M4 | PV | 766 | True | 5 | 153.2 | 0.458 | 0.107 | 0.719 | 0.065 | 0.588 | 0.044 | n/a | n/a | 0.598 | 0.042 |  |
| EXP-C2 | circor_outcome | M4 | Phc | 4 | False | 2 | 2.0 | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C2 | circor_outcome | M4 | TV | 732 | True | 5 | 146.4 | 0.377 | 0.072 | 0.806 | 0.029 | 0.592 | 0.033 | n/a | n/a | 0.608 | 0.030 |  |
| EXP-C2 | circor_outcome | M5 | AV | 800 | True | 5 | 160.0 | 0.515 | 0.083 | 0.644 | 0.036 | 0.579 | 0.034 | n/a | n/a | 0.579 | 0.033 |  |
| EXP-C2 | circor_outcome | M5 | MV | 861 | True | 5 | 172.2 | 0.556 | 0.075 | 0.639 | 0.034 | 0.597 | 0.029 | n/a | n/a | 0.597 | 0.028 |  |
| EXP-C2 | circor_outcome | M5 | PV | 766 | True | 5 | 153.2 | 0.453 | 0.091 | 0.728 | 0.062 | 0.590 | 0.055 | n/a | n/a | 0.601 | 0.054 |  |
| EXP-C2 | circor_outcome | M5 | Phc | 4 | False | 2 | 2.0 | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C2 | circor_outcome | M5 | TV | 732 | True | 5 | 146.4 | 0.418 | 0.085 | 0.793 | 0.066 | 0.606 | 0.017 | n/a | n/a | 0.620 | 0.014 |  |
| EXP-C2 | circor_outcome | M6 | AV | 800 | True | 5 | 160.0 | 0.505 | 0.118 | 0.663 | 0.079 | 0.584 | 0.021 | n/a | n/a | 0.584 | 0.020 |  |
| EXP-C2 | circor_outcome | M6 | MV | 861 | True | 5 | 172.2 | 0.517 | 0.092 | 0.682 | 0.082 | 0.599 | 0.024 | n/a | n/a | 0.598 | 0.022 |  |
| EXP-C2 | circor_outcome | M6 | PV | 766 | True | 5 | 153.2 | 0.426 | 0.110 | 0.753 | 0.092 | 0.590 | 0.030 | n/a | n/a | 0.602 | 0.030 |  |
| EXP-C2 | circor_outcome | M6 | Phc | 4 | False | 2 | 2.0 | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C2 | circor_outcome | M6 | TV | 732 | True | 5 | 146.4 | 0.353 | 0.112 | 0.852 | 0.050 | 0.603 | 0.039 | n/a | n/a | 0.622 | 0.034 |  |
| EXP-C2 | circor_outcome | M7 | AV | 800 | True | 5 | 160.0 | 0.505 | 0.118 | 0.663 | 0.079 | 0.584 | 0.021 | n/a | n/a | 0.584 | 0.020 |  |
| EXP-C2 | circor_outcome | M7 | MV | 861 | True | 5 | 172.2 | 0.517 | 0.092 | 0.682 | 0.082 | 0.599 | 0.024 | n/a | n/a | 0.598 | 0.022 |  |
| EXP-C2 | circor_outcome | M7 | PV | 766 | True | 5 | 153.2 | 0.426 | 0.110 | 0.753 | 0.092 | 0.590 | 0.030 | n/a | n/a | 0.602 | 0.030 |  |
| EXP-C2 | circor_outcome | M7 | Phc | 4 | False | 2 | 2.0 | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | n/a | n/a | 0.833 | 0.236 | excluded: 4 recording(s) in the whole corpus, under one per fold -- a single record moves any rate by up to 1.0, so this is arithmetic noise, not a weak estimate |
| EXP-C2 | circor_outcome | M7 | TV | 732 | True | 5 | 146.4 | 0.353 | 0.112 | 0.852 | 0.050 | 0.603 | 0.039 | n/a | n/a | 0.622 | 0.034 |  |

> Location read from metadata_master.recording_location and cross-checked against the record_uid, which encodes it independently; 43 recordings carry a repeat suffix (_1.._3) that a naive suffix split misreads.

> The patient-level label is propagated to every recording of that patient, so per-location sensitivity is measured against a label that may not be audible at that position. See circor_label_propagation.md.

> PV-MEPCG / PulseVision is an academic screening prototype, not a diagnostic tool.

_Table T22 -- Auscultation Location Analysis_
_Experiment: EXP-C3_
_Objective: O5 (external validation)_
_Source: outputs/08_circor_external_validation/EXP-C1-three_class/per_fold_by_location.csv; outputs/08_circor_external_validation/EXP-C1-two_class/per_fold_by_location.csv; outputs/08_circor_external_validation/EXP-C2/per_fold_by_location.csv_
_Generated by PV-MEPCG / PulseVision at 2026-08-31T07:58:17.207411+00:00_
