# Final QA sweep (Phase 124)

Generated 2026-09-12T14:09:06+00:00 by `python scripts/49_final_qa_sweep.py`.

Every row below was **recomputed from a committed file** at generation time.
Nothing here asserts that an earlier check once passed.

**Result: PASS**

| Area | T124 | Status | Checks |
|---|---|---|---|
| dataset | T124.1 | **pass** | 8 |
| split | T124.2 | **pass** | 7 |
| feature | T124.3 | **pass** | 5 |
| model | T124.4 | **pass** | 5 |
| metric | T124.5 | **pass** | 3 |
| search | T124.6 | **pass** | 2 |
| dashboard | T124.6 | **pass** | 2 |

## dataset — T124.1 -- integrity, duplicates, label mapping, class counts, sampling rates

| Check | Status | Detail |
|---|---|---|
| `QA-DATA-01` no file in the corpus is missing or unreadable | pass | 0 unreadable/missing file(s); 64 quality flag(s) recorded ({'clipped': np.int64(63), 'all_silent': np.int64(1)}) |
| `QA-DATA-02` the audited per-dataset record counts still hold | pass | D1 3541, D2 176, D3 656, D4 3163 |
| `QA-DATA-03` every record carries its dataset's audited native sampling rate | pass | {} |
| `QA-DATA-04` every duplicate group carries an explicit decision | pass | 2268 duplicate row(s), 0 without a decision |
| `QA-DATA-05` nothing from the duplicate Heartbeat_Sound/ tree is a training row | pass | 0 row(s) reference it, 0 of them labelled |
| `QA-DATA-06` the five label spaces are never merged (research rule 4) | pass | the five label spaces remain disjoint |
| `QA-DATA-07` every class in every task has at least one record | pass | 28 class row(s) across 5 task(s) |
| `QA-DATA-08` the corpus is the audited size | pass | 7536 records in the master metadata, audited 7536 |

## split — T124.2 -- zero patient leakage across every fold of every task

| Check | Status | Detail |
|---|---|---|
| `QA-SPLIT-binary` binary: no subject and no record in both train and test | pass | 25 folds (repeated_5x5_grouped), 857 groups, zero shared |
| `QA-SPLIT-circor_murmur` circor_murmur: no subject and no record in both train and test | pass | 5 folds (patient_grouped_5fold), 942 groups, zero shared |
| `QA-SPLIT-circor_outcome` circor_outcome: no subject and no record in both train and test | pass | 5 folds (patient_grouped_5fold), 942 groups, zero shared |
| `QA-SPLIT-pascal_a` pascal_a: subject-level separation | not_applicable | scheme is repeated_5x2_stratified: no subject identifier could be derived for 620 of 620 rows, so grouping is unavailable rather than satisfied. Recorded, not claimed. |
| `QA-SPLIT-pascal_b` pascal_b: no subject and no record in both train and test | pass | 5 folds (grouped_5fold), 165 groups, zero shared |
| `QA-SPLIT-COVERAGE` every record is assigned to exactly one test fold per repeat | pass | 23607 assignments across 5 tasks |
| `QA-SPLIT-CLASSES` no fold of any task is missing a class entirely | pass | 50 folds, smallest class in any fold: {int(folds['min_class_count'].min())} |

## feature — T124.3 -- every feature finite and reproducible across folds

| Check | Status | Detail |
|---|---|---|
| `QA-FEAT-01` the feature registry is still the locked 138 | pass | 138 features, fingerprint 9aa6c14561d3f32c |
| `QA-FEAT-02` the per-family counts still sum to 138 as declared | pass | {'time': 24, 'frequency': 22, 'mfcc': 39, 'chroma': 24, 'dwt': 24, 'envelope': 5} |
| `QA-FEAT-03` every feature value in the matrix is finite | pass | 7536 records x 138 features; 0 NaN, 0 infinite |
| `QA-FEAT-04` the matrix carries every feature the registry declares, in order | pass | 138 of 138 registry columns present and in order |
| `QA-FEAT-05` no recording failed feature extraction | pass | 0 extraction error row(s) |

## model — T124.4 -- fixed seeds, fold-safe scaling, calibration as configured

| Check | Status | Detail |
|---|---|---|
| `QA-MODEL-01` every recorded run used seed 42 (research rule 5) | pass | global seed 42, {'42': 310}, 0 run(s) on another seed |
| `QA-MODEL-02` every run recorded its package versions | pass | 325 runs, 0 without package versions |
| `QA-MODEL-03` the imputer, scaler and selector are steps inside the Pipeline (rule 2) | pass | pipeline steps: ['imputer', 'scaler', 'estimator'] |
| `QA-MODEL-04` ensemble members are still calibrated individually, on the measured evidence | pass | calibrate_members: {'M6': ['M5'], 'M7': ['M5']} |
| `QA-MODEL-05` every deployed model records its selection rule, and it is not accuracy | pass | 5 deployed task model(s), all rule-6 selected |

## metric — T124.5 -- never accuracy alone (research rule 6)

| Check | Status | Detail |
|---|---|---|
| `QA-METRIC-01` every experiment that reports accuracy also reports the rule-6 metrics | pass | 12 experiment result table(s), each carrying its task's full metric set |
| `QA-METRIC-02` the multiclass tasks report per-class recall, not only a macro average | pass | 6 multiclass result table(s) |
| `QA-METRIC-03` the final model was selected on sensitivity and balanced accuracy | pass | selection rule: ['sensitivity', 'balanced_accuracy']; accuracy would have chosen M3 |

## search — T124.6 -- no search touched a final test fold

| Check | Status | Detail |
|---|---|---|
| `QA-SEARCH-01` no inner search split ever contained an outer test record | pass | 2 (search, outer fold) pair(s) re-intersected with the published fold map across 2 search(es): SO-01, SO-02. EXP-A2's per-fold nested searches write their chosen point to a gitignored cache and carry it in per_fold_metrics.csv, so they are covered by tests/test_search_no_leakage.py rather than by a map on disk. |
| `QA-SEARCH-02` every inner search row is an outer training row | pass | 2 inner fold map(s) fully inside outer train |

## dashboard — T124.6 -- the T119.3 displayed-value audit is green

| Check | Status | Detail |
|---|---|---|
| `QA-DASH-01` the T119.3 displayed-value audit is green for the site on disk right now | pass | site fccc941dbb0b passed at 2026-09-12T12:39:50.738085+00:00; 4527 rendered values and 85134 table/figure cells over 16 pages, 0 finding(s) |
| `QA-DASH-02` all thirteen dashboard screenshots are on disk and non-trivial | pass | 13 PNG(s), smallest 80589 bytes |

## Unavailable rather than satisfied

- `QA-SPLIT-pascal_a` pascal_a: subject-level separation — scheme is repeated_5x2_stratified: no subject identifier could be derived for 620 of 620 rows, so grouping is unavailable rather than satisfied. Recorded, not claimed.
