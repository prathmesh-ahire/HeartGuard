# HeartGuard — Master Task Breakdown

**Framework name (deliverables):** PV-MEPCG / PulseVision
**Repo name:** HeartGuard
**Reference docs:** `Kharat_Mam_Developer_Blueprint...pdf` (what to build), `KP_KHARAT_22366.docx` (what to output)
**Build order:** Backend + pipeline complete first. Dashboard/UI only after Part IX.
**Companion files:** `CLAUDE.md` at the repo root (how to work on this project) and `Docs/note.md` (change log of errors, surprises and non-obvious decisions). Read `CLAUDE.md` at the start of every session; check `note.md` before debugging anything.

## Ground rules (apply to every task)
1. No fabricated numbers. Every metric traces to a real run + its source CSV/JSON.
2. Scaler, feature selector and search run **inside** the training fold only.
3. Subject-wise grouping wherever a subject ID exists. Never split a subject across folds.
4. Binary / multiclass / murmur-outcome label spaces stay separate. Never merged.
5. Fixed global seed (42). Every run writes its seed, fold map, params and package versions.
6. Screening / decision-support wording only. No diagnostic claims.
7. Anything that cannot be produced goes into `outputs/missing_outputs_report.txt` with the technical reason. Create the file empty in Phase 01 and append to it as gaps arise — never write it only at the end.
8. **Commit and push after every phase**, once its `[TEST]` gate passes. Never let more than one phase of work sit unpushed — GitHub is the recovery point if anything goes wrong locally.

## How to use this file

Work through tasks in order (Phase → Task, e.g. T09.4). Do not skip ahead — later phases depend on earlier ones.

**Status boxes:** `[ ]` todo · `[x]` done · `[~]` in progress · `[!]` blocked · `[-]` skipped (reason required in `note.md` **and** `missing_outputs_report.txt`)

**Task markers:**

| Marker | Meaning |
|---|---|
| *(unmarked)* | AI agent task — the default |
| **[MANUAL]** | **You** must do this: install something, create an account, make a judgement call the agent cannot make |
| **[TEST]** | Verification checkpoint. The agent runs it. **Do not move to the next phase until it passes.** |
| **[TEST/MANUAL]** | **You** must look at it — open the figure, click through the page, read the output and confirm it is right |

**Every phase ends with task `.7`, which is its [TEST] gate.** It verifies everything that phase added. A phase is not done until its gate passes.

**Five 🔴 MEGA TEST phases** sit at the major boundaries — Phases 30, 53, 77, 105 and 121. Each re-verifies everything important built up to that point, so a bug introduced in Phase 12 is caught at Phase 29 rather than at submission. Nothing in the following Part begins until a mega test is fully green.

| Mega test | Phase | Gates | Covers |
|---|---|---|---|
| MEGA 1 — Data & Preprocessing Integrity | 30 | Part IV | Phases 01-29 |
| MEGA 2 — Feature & Model Integrity | 53 | Part VI | Phases 31-52 |
| MEGA 3 — Experiment & Result Integrity | 77 | Part VIII | Phases 54-76 |
| MEGA 4 — Deliverable Completeness | 105 | Part X | Phases 78-104 |
| MEGA 5 — Full System | 121 | Part XI | Phases 106-120 |

**A [TEST] task never gets weakened to make it pass.** If a gate fails, fix the cause. If the gate itself is wrong, say so and change it deliberately, with a `note.md` entry explaining why.

## Verified dataset map (audited 2026-08-22)
| ID | Path | Contents | Fs | Subject IDs |
|----|------|----------|----|-------------|
| D1 | `dataset/archive (3)` | 3,240 train (a-f) + 301 validation; 2,725 normal / 816 abnormal | 2000 Hz | partial (derive) |
| D2 | `dataset/archive (2)/set_a` | 124 labeled (artifact 40, extrahls 19, murmur 34, normal 31) + 52 unlabeled | 44100 Hz | none |
| D3 | `dataset/archive (2)/set_b` | 461 labeled (extrastole 46, murmur 95, normal 320) + 195 unlabeled | 4000 Hz | filename-derived (167) |
| D4 | `dataset/archive/training_data/training_data` | 942 patients / 3,163 recordings | 4000 Hz | native |
| -- | `dataset/Heartbeat_Sound` | duplicate of D2+D3, foldered by class | -- | label helper only |

---

# PART I — FOUNDATION, ENVIRONMENT & CI

### Phase 01 — Repository Skeleton
- [x] **T01.1** Create `src/` package root with `__init__.py` and subpackages: `data_loader`, `preprocessing`, `feature_extraction`, `feature_selection`, `models`, `ensemble`, `optimization`, `evaluation`, `explainability`, `reporting`, `inference`, `api`, `utils`. Create top-level `frontend/` (populated in Phase 103).
- [x] **T01.2** Create `outputs/` tree exactly per doc section 17: `00_evidence_index`, `01_dataset_audit`, `02_preprocessing`, `03_features`, `04_models`, `05_search_optimization`, `06_binary_results`, `07_multiclass_results`, `08_circor_external_validation`, `09_ablation`, `10_robustness`, `11_complexity`, `12_statistics`, `13_figures_diagrams`, `14_algorithms`, `15_dashboard_screenshots`, `Q1_PAPER_ASSETS`, `THESIS_ASSETS`, `logs`, `configs`.
- [x] **T01.3** Create `models_saved/`, `cache/`, `tests/`, `scripts/`, `notebooks/`.
- [x] **T01.4** Add `.gitignore` excluding `dataset/`, `cache/`, `*.parquet`, `models_saved/*.pkl`, `__pycache__/`, `.venv/`, `node_modules/`, `.next/` — but NOT `frontend/out/`, which is committed so a grader without Node can still serve the dashboard.
- [x] **T01.5** Add `.gitkeep` to every empty output directory so the tree survives git.
- [x] **T01.6** Write `README.md` stub (project name, scope boundary disclaimer, quickstart placeholder) and create an empty `outputs/missing_outputs_report.txt` with its header, so every later skip has somewhere to go.
- [x] **T01.7** [TEST] Confirm the full `src/` and `outputs/` trees exist exactly as listed, every empty dir has a `.gitkeep`, and `.gitignore` already excludes `dataset/` before anything is staged.

### Phase 02 — Python Environment
- [x] **T02.1** [MANUAL] Create venv at `.venv` with Python 3.11.9; document activation for PowerShell and bash.
- [x] **T02.2** Write `requirements.txt` with pinned versions: numpy, scipy, pandas, scikit-learn, librosa, soundfile, PyWavelets, matplotlib, seaborn, joblib, pyarrow, tqdm, pyyaml.
- [x] **T02.3** Write `requirements-extra.txt`: xgboost, lightgbm, scikit-optimize, deap, pyswarms, shap, statsmodels.
- [x] **T02.4** Write `requirements-api.txt`: fastapi, uvicorn[standard], python-multipart, pydantic. Write `requirements-report.txt`: python-docx, openpyxl, kaleido.
- [x] **T02.5** Install all three sets; capture `pip freeze` to `outputs/configs/pip_freeze.txt`.
- [x] **T02.6** Write `scripts/verify_env.py` — imports every package, prints versions, checks Node LTS and npm are on PATH, exits nonzero on any failure.
- [x] **T02.7** [TEST] Run `scripts/verify_env.py` — every package imports at its pinned version, Node and npm resolve, and the script exits 0.

### Phase 03 — Configuration System
- [x] **T03.1** Write `configs/paths.yaml` — absolute dataset roots for D1–D4, output roots, cache root.
- [x] **T03.2** Write `configs/signal.yaml` — target_fs 2000, filter type butterworth, order 4, low 20, high 400, normalization zscore.
- [x] **T03.3** Write `configs/features.yaml` — per-family enable flags, MFCC n=13, chroma n=12, DWT wavelet db4 level 5, frame/hop sizes.
- [x] **T03.4** Write `configs/models.yaml` — default hyperparameters for LR, KNN, SVM, RF, GB, XGB/LGBM.
- [x] **T03.5** Write `configs/experiments.yaml` — definitions for EXP-A1..F2 (dataset, task, models, CV scheme).
- [x] **T03.6** Write `src/utils/config.py` — YAML loader with dotted-key access, env-var override, and schema validation.
- [x] **T03.7** [TEST] Load every YAML through `src/utils/config.py`, assert schema validation catches a deliberately malformed key, and confirm dotted-key access returns the expected values.

### Phase 04 — Reproducibility & Logging
- [x] **T04.1** Write `src/utils/seed.py` — `set_global_seed(42)` covering `random`, `numpy`, and `PYTHONHASHSEED`.
- [x] **T04.2** Write `src/utils/logging_setup.py` — dual console + rotating file handler into `outputs/logs/`.
- [x] **T04.3** Write `src/utils/run_manifest.py` — records run id, UTC timestamp, git commit, seed, config snapshot, package versions to `outputs/00_evidence_index/run_manifest.json`.
- [x] **T04.4** Write `src/utils/timing.py` — `@timed` decorator and context manager writing durations to the manifest.
- [x] **T04.5** Write `src/utils/io.py` — atomic save/load helpers for CSV, JSON, Parquet, PNG, pickle, all creating parent dirs.
- [x] **T04.6** Write `src/utils/evidence.py` — `register_evidence(...)` appending one row per generated artifact to the evidence index.
- [x] **T04.7** [TEST] Run a throwaway job end to end: confirm the seed is applied, a log file appears in `outputs/logs/`, the manifest records git commit and package versions, and `register_evidence` appends a row.

### Phase 05 — Constants & Label Vocabularies
- [x] **T05.1** Write `src/utils/constants.py` — dataset IDs D1–D4, canonical names, task names (`binary`, `pascal_a`, `pascal_b`, `circor_murmur`, `circor_outcome`).
- [x] **T05.2** Define binary label map: `normal=0`, `abnormal=1`; PhysioNet `-1 -> 0`, `1 -> 1`. Preserve original value in metadata.
- [x] **T05.3** Define PASCAL A label map: `normal=0, murmur=1, extrahls=2, artifact=3`.
- [x] **T05.4** Define PASCAL B label map: `normal=0, murmur=1, extrastole=2`.
- [x] **T05.5** Define CirCor maps: murmur `Absent=0, Present=1, Unknown=2`; outcome `Normal=0, Abnormal=1`; locations `AV, PV, TV, MV, Phc`.
- [x] **T05.6** Write `tests/test_constants.py` asserting all maps are bijective and mutually non-overlapping across tasks.
- [x] **T05.7** [TEST] Run `tests/test_constants.py` — all label maps bijective, and assert that no task's label vocabulary overlaps another's.

### Phase 06 — Test Harness
- [x] **T06.1** Add `pytest.ini` with testpaths, markers `slow` / `needs_data`, and strict-marker mode.
- [x] **T06.2** Write `tests/conftest.py` with fixtures: tiny synthetic PCG signal, temp output dir, loaded config.
- [x] **T06.3** Write `tests/fixtures/make_synthetic_pcg.py` — deterministic S1/S2 + noise generator for unit tests.
- [x] **T06.4** Write `tests/test_env.py` asserting every required package imports at the pinned version.
- [x] **T06.5** Write `scripts/run_tests.ps1` and `scripts/run_tests.sh`.
- [x] **T06.6** Confirm full suite runs green on the empty skeleton.
- [x] **T06.7** [TEST] Run the full pytest suite on the empty skeleton — green, with `slow` and `needs_data` markers correctly skipping.

### Phase 07 — Version Control
- [x] **T07.1** `git init`; set default branch `main`.
- [x] **T07.2** [MANUAL] Create the private GitHub repository, then add remote `https://github.com/prathmesh-ahire/HeartGuard.git`.
- [x] **T07.3** Verify `dataset/` and `cache/` are ignored; confirm nothing >50 MB is staged.
- [x] **T07.4** Commit the skeleton as `chore: project skeleton, env and config system`.
- [x] **T07.5** Add `CHANGELOG.md` with a Phase-completion log format.
- [x] **T07.6** Push `main` and verify the remote tree.
- [x] **T07.7** [TEST] Confirm `git status` shows no dataset or cache file staged, nothing over 50 MB is tracked, and the remote tree matches local after push.

### Phase 08 — Continuous Integration
- [ ] **T08.1** Create `.github/workflows/ci.yml` triggering on push and pull request to `main`, with a concurrency group so superseded runs cancel.
- [ ] **T08.2** Python job: set up 3.11, cache pip, install all requirements files, run `pytest -m "not needs_data"` — the 1.3 GB dataset is gitignored and never reaches GitHub, so data-dependent tests are excluded by design, not by accident.
- [ ] **T08.3** Quality job: run `ruff` and `mypy` over `src/`; add both to `requirements-dev.txt` and pin them.
- [ ] **T08.4** Frontend job (activates from Part X): Node LTS, `npm ci`, `npm run build`, then `scripts/07_check_no_hardcoded_metrics.py`. Guard it so it skips cleanly while `frontend/` does not yet exist.
- [ ] **T08.5** Add the CI status badge to `README.md`, and document in the workflow header exactly which checks CI can and cannot perform without the dataset.
- [ ] **T08.6** [MANUAL] Enable GitHub Actions on the repository and confirm the first run appears in the Actions tab.
- [ ] **T08.7** [TEST] Push a branch containing a deliberately failing test; confirm CI goes red and blocks. Fix it, confirm green. **CI that has never failed is not proven to work.**

---

# PART II — DATA LAYER & DATASET AUDIT

### Phase 09 — PhysioNet 2016 Loader (D1)
- [ ] **T09.1** Write `src/data_loader/physionet.py` with `list_records()` scanning `training-a..f` + `validation`.
- [ ] **T09.2** Parse each `.hea` header: record name, n signals, sampling rate, n samples, comment line (`# Normal` / `# Abnormal`).
- [ ] **T09.3** Detect and record which subsets carry a second ECG channel (`.dat`) — training-a only — as a metadata flag.
- [ ] **T09.4** Parse `REFERENCE.csv` per subset into `{record: -1|1}`; assert every WAV has a label and every label has a WAV.
- [ ] **T09.5** Parse `REFERENCE-SQI.csv` into a signal-quality column where present.
- [ ] **T09.6** Cross-check the `.hea` comment label against `REFERENCE.csv`; log disagreements to `outputs/01_dataset_audit/physionet_label_conflicts.csv`.
- [ ] **T09.7** [TEST] Load all 3,541 PhysioNet records; assert every WAV has a label and every label a WAV, and that the `.hea` comment agrees with `REFERENCE.csv` for all but the logged conflicts.

### Phase 10 — PhysioNet Annotation Enrichment
- [ ] **T10.1** Load `annotations/Online_Appendix_training_set.csv` (utf-8-sig) into a dataframe keyed by challenge record name.
- [ ] **T10.2** Join `Diagnosis`, `Gender`, `Age`, `Original record name`, `Transducer site on body` onto the record table.
- [ ] **T10.3** Load `Online_Appendix_Diagnosis_meanings.csv` and map each diagnosis code to its clinical meaning.
- [ ] **T10.4** Build a `diagnosis_class` column (12 categories) reserved for the optional extended-multiclass track.
- [ ] **T10.5** Parse murmur / arrhythmia / respiration-noise / ambient-noise / recording-noise codes into named ordinal columns.
- [ ] **T10.6** Report appendix coverage: 3,153 of 3,240 rows matched; log the unmatched records to `physionet_unannotated.csv`.
- [ ] **T10.7** [TEST] Assert 3,153 appendix rows joined, the 12 diagnosis categories are populated, and the unmatched records are listed rather than silently dropped.

### Phase 11 — PhysioNet Subject Derivation
- [ ] **T11.1** Write `derive_subject_physionet()` — training-a pattern `C(\d+)S(\d+)` maps to subject `a_C{n}`.
- [ ] **T11.2** training-b pattern `S(\d+)f(\d+)_data` maps to `b_S{n}`; validate against the 106 native subject IDs already present.
- [ ] **T11.3** training-c pattern `id(\d+)` maps to `c_id{n}`.
- [ ] **T11.4** training-e pattern `e(\d+)` — treat as one recording per subject; document the assumption explicitly.
- [ ] **T11.5** training-d and training-f — no recoverable pattern; each record becomes its own group with `subject_derived=False`.
- [ ] **T11.6** Emit `outputs/01_dataset_audit/physionet_subject_derivation.csv` with record, pattern used, subject id, confidence flag.
- [ ] **T11.7** [TEST] Assert every PhysioNet record has a `subject_id`, that training-b's derived IDs match the 106 native ones exactly, and that `subject_derived=False` is set wherever no pattern applied.

### Phase 12 — PASCAL Loader (D2 / D3)
- [ ] **T12.1** Write `src/data_loader/pascal.py` scanning `set_a/` and `set_b/` for WAV files.
- [ ] **T12.2** Build the authoritative label index from `Heartbeat_Sound/<class>/` folder membership (832 files, verified 100 percent name overlap with set_a+set_b).
- [ ] **T12.3** Write a filename normalizer resolving the set_b CSV mismatch (`Btraining_<label>_<id>...` versus on-disk `<label>__<id>...`).
- [ ] **T12.4** Cross-validate folder-derived labels against `set_a.csv` / `set_b.csv`; write disagreements to `pascal_label_conflicts.csv`.
- [ ] **T12.5** Split records into set_a (D2) and set_b (D3); exclude unlabelled files from supervised tracks but keep them in metadata.
- [ ] **T12.6** Assert final labeled counts: set_a = 124 (40/19/34/31), set_b = 461 (46/95/320). Fail loudly on mismatch.
- [ ] **T12.7** [TEST] Assert the labeled counts land exactly on set_a = 124 (40/19/34/31) and set_b = 461 (46/95/320); confirm the filename normalizer resolves every set_b CSV row.

### Phase 13 — PASCAL Subject Derivation & Timing
- [ ] **T13.1** Extract set_b subject id from filename pattern `<label>__(\d+)_<timestamp>_<loc>` yielding 167 groups.
- [ ] **T13.2** Extract the set_b auscultation-location suffix (`_A`, `_B`, `_C`, `_D`, `_D1`) into a `recording_location` column.
- [ ] **T13.3** Investigate the outlier group holding ~149 recordings; determine whether it is a real subject or a regex fallback, and document.
- [ ] **T13.4** Mark set_a as `subject_id = record_id`, `subject_derived=False` — timestamp-only filenames carry no subject information.
- [ ] **T13.5** Load `set_a_timing.csv` into an S1/S2 annotation table; it is consumed by Phase 80 (cycle analysis) and T113.6 (the dashboard cardiac-cycle viewer).
- [ ] **T13.6** Emit `outputs/01_dataset_audit/pascal_subject_derivation.csv`.
- [ ] **T13.7** [TEST] Assert 167 set_b subject groups, every record carries a location where the filename provides one, and the ~149-record outlier group has been explained in writing.

### Phase 14 — CirCor Loader (D4)
- [ ] **T14.1** Write `src/data_loader/circor.py` scanning `archive/training_data/training_data/` — note the doubled directory level.
- [ ] **T14.2** Parse each of the 942 patient `.txt` files: header line (patient id, n locations, fs) plus the per-location file triplets.
- [ ] **T14.3** Parse all `#Key: Value` fields, critically `#Outcome` — this is ABSENT from `training_data.csv` and exists only in the txt files.
- [ ] **T14.4** Parse `#Murmur`, `#Murmur locations`, `#Most audible location`, and the ten systolic/diastolic murmur descriptors.
- [ ] **T14.5** Load `training_data.csv` (942 rows) for demographics and join; assert patient-id agreement with the txt parse.
- [ ] **T14.6** Assert counts: 942 patients, 3,163 recordings, murmur 695/179/68, outcome 486/456, locations AV 800 / PV 766 / TV 732 / MV 861 / Phc 4.
- [ ] **T14.7** [TEST] Assert 942 patients, 3,163 recordings, murmur 695/179/68 and **outcome 486/456 parsed from the txt files** — the CSV has no Outcome column and must not be the source.

### Phase 15 — CirCor Segmentation & Header Parsing
- [ ] **T15.1** Parse each `.hea` file for record name, fs (4000) and sample count; cross-check against the WAV header.
- [ ] **T15.2** Write a `.tsv` segmentation parser (start, end, state) mapping states 1-4 to S1 / systole / S2 / diastole, 0 = unannotated.
- [ ] **T15.3** Compute per-recording annotated fraction and estimated heart-cycle count from the TSV.
- [ ] **T15.4** Store segmentation as a per-record artifact; it is consumed by Phase 80 (cycle analysis) and T113.6 (the dashboard cardiac-cycle viewer). Do not leave it unused.
- [ ] **T15.5** Flag recordings with zero usable segmentation into `circor_unsegmented.csv`.
- [ ] **T15.6** Verify `RECORDS` and `SHA256SUMS.txt` integrity across the CirCor tree; log any mismatch.
- [ ] **T15.7** [TEST] Parse every `.tsv`; assert state values are within 0-4, that segment times are monotonic and within the recording duration, and that unsegmented records are reported.

### Phase 16 — Audio Integrity Scan
- [ ] **T16.1** Write `src/data_loader/integrity.py` opening every WAV across D1-D4 and capturing fs, channels, frames, bit depth, duration.
- [ ] **T16.2** Flag unreadable, zero-length and truncated files (baseline audit found zero across 7,536 files — re-verify programmatically).
- [ ] **T16.3** Flag all-silent and constant-value recordings by peak and variance threshold.
- [ ] **T16.4** Flag clipped recordings where the fraction of full-scale samples exceeds a configured threshold.
- [ ] **T16.5** Flag records present on disk but missing a label, and labels missing a file, per dataset.
- [ ] **T16.6** Emit **DA-05** `outputs/01_dataset_audit/missing_corrupt_files.csv`.
- [ ] **T16.7** [TEST] Run the integrity scan across all 7,536 files; confirm zero unreadable and zero zero-length, matching the 2026-08-22 audit, and that `missing_corrupt_files.csv` is written even when empty.

### Phase 17 — Duplicate Detection
- [ ] **T17.1** Compute SHA-256 of raw bytes for every WAV and group exact duplicates.
- [ ] **T17.2** Compute an audio-content hash on decoded PCM after mono conversion and resampling, to catch re-encoded duplicates.
- [ ] **T17.3** Formally record `Heartbeat_Sound/` as a full duplicate of set_a+set_b (832/832) and exclude it from all supervised tracks.
- [ ] **T17.4** Detect near-duplicates via correlation of downsampled envelopes above a configured threshold, within each dataset.
- [ ] **T17.5** Check cross-dataset duplication between PASCAL B and CirCor, both 4 kHz sources.
- [ ] **T17.6** Emit **DA-06** `outputs/01_dataset_audit/duplicate_report.csv` with a keep/drop decision column.
- [ ] **T17.7** [TEST] Confirm all 832 `Heartbeat_Sound` files are flagged as duplicates and marked drop, and that no PASCAL record appears twice in any supervised set.

### Phase 18 — Duration & Sampling Summaries
- [ ] **T18.1** Compute per-dataset duration statistics: min, p25, median, p75, max, mean, SD.
- [ ] **T18.2** Compute per-class duration statistics within each dataset.
- [ ] **T18.3** Define robustness duration bands (short under 5 s, medium 5-20 s, long over 20 s) and assign every record.
- [ ] **T18.4** Emit **DA-03** `recording_duration_summary.csv`.
- [ ] **T18.5** Emit **DA-04** `sampling_rate_summary.csv` — original fs 2000 / 4000 / 44100 versus converted 2000 Hz, with counts.
- [ ] **T18.6** Emit **DA-02** `class_distribution.csv` — dataset by class counts plus imbalance ratio.
- [ ] **T18.7** [TEST] Cross-check the duration and sampling summaries against the audited figures (PhysioNet median 20.83 s, PASCAL B min 0.76 s, three native rates 2000/4000/44100).

### Phase 19 — Master Metadata Assembly
- [ ] **T19.1** Define the master schema: `record_uid, dataset_source, subset, subject_id, subject_derived, record_id, file_path, original_fs, duration_sec, n_samples, recording_location, binary_label, multiclass_label, murmur_label, outcome_label, diagnosis_class, quality_flags, is_duplicate, is_unlabeled, split_group`.
- [ ] **T19.2** Enforce that each record carries labels only for tasks it legitimately belongs to; all others NaN. Never coerce across label spaces.
- [ ] **T19.3** Generate a globally unique `record_uid` as `{dataset}_{subset}_{record_id}`.
- [ ] **T19.4** Merge the D1-D4 record tables into one dataframe and validate the schema with explicit dtype checks.
- [ ] **T19.5** Emit **DA-08** `dataset/metadata_master.csv` plus a Parquet twin for fast loading.
- [ ] **T19.6** Write `tests/test_master_metadata.py` — no duplicate uids, no cross-task label bleed, every file path resolves on disk.
- [ ] **T19.7** [TEST] Run `tests/test_master_metadata.py` — unique uids, every file path resolves, and **assert no record carries a label for a task it does not belong to**.

### Phase 20 — Split Map Generation
- [ ] **T20.1** Write `src/data_loader/splits.py` implementing 5-fold StratifiedGroupKFold keyed on `subject_id`.
- [ ] **T20.2** Generate the D1 binary split map as **repeated 5x5 StratifiedGroupKFold (25 folds)** — five repeats with different shuffles, subject-grouped. This is the protocol for the whole primary track: it gives stable mean and SD, and n=25 for the paired statistical tests, which 5 folds cannot supply. Verify no subject spans two folds within any repeat.
- [ ] **T20.3** Generate D2 (record-level stratified, repeated 5x2 given n=124) and D3 (subject-grouped) split maps.
- [ ] **T20.4** Generate D4 patient-wise split maps independently for the murmur task and the outcome task.
- [ ] **T20.5** Hold PhysioNet `validation/` (301 records) out as an untouched final test set, separate from cross-validation.
- [ ] **T20.6** Emit **DA-07** `subject_split_map.csv` and write `tests/test_no_leakage.py` asserting zero subject overlap in every fold of every task.
- [ ] **T20.7** [TEST] Run `tests/test_no_leakage.py` — zero subject overlap in every fold of every task; confirm the 301 validation records appear in no CV fold.

### Phase 21 — Dataset Inventory & Audit Reporting
- [ ] **T21.1** Emit **DA-01** `dataset_inventory.csv` — dataset, version, source, folder, total files, usable files, classes, subjects, fs, role.
- [ ] **T21.2** Write the discrepancy note: the CirCor public release is 942 patients / 3,163 recordings, not the blueprint figure of 1,568 / 5,272 (the remainder is the unreleased Challenge test set).
- [ ] **T21.3** Write the discrepancy note: PhysioNet on disk is 3,240 training plus 301 validation, not the blueprint figure of ~3,126.
- [ ] **T21.4** Write the limitation note: PASCAL A carries no recoverable subject IDs, so its results are record-level only.
- [ ] **T21.5** Generate **DA-09** `dataset_audit_report.docx` — a narrative audit covering every count, conflict, duplicate and limitation above.
- [ ] **T21.6** Register all DA-01 through DA-09 artifacts in the evidence index.
- [ ] **T21.7** [TEST] Confirm DA-01 through DA-09 all exist and are non-empty, and that the audit report states both count discrepancies and the PASCAL A subject limitation.

### Phase 22 — Data Layer Hardening
- [ ] **T22.1** Add a `--limit N` sampling mode to every loader for fast smoke runs.
- [ ] **T22.2** Add on-disk caching of parsed metadata keyed by a config-plus-mtime hash.
- [ ] **T22.3** Write `scripts/01_run_dataset_audit.py` executing Phases 08-20 end to end from a single command.
- [ ] **T22.4** Write `tests/test_loaders.py` covering each loader against the real tree, marked `needs_data`.
- [ ] **T22.5** Benchmark full-audit wall time and record it in the run manifest.
- [ ] **T22.6** Commit Part II and update `CHANGELOG.md`.
- [ ] **T22.7** [TEST] Run `scripts/01_run_dataset_audit.py` from scratch on a cleared output dir; confirm it completes, is idempotent on a second run, and records its wall time.

---

# PART III — SIGNAL PREPROCESSING

### Phase 23 — Audio IO & Resampling
- [ ] **T23.1** Write `src/preprocessing/io.py` — `load_wav(path)` returning float32 samples plus native fs, via soundfile.
- [ ] **T23.2** Implement mono conversion (channel mean) with a no-op fast path, since all 7,536 files are already mono.
- [ ] **T23.3** Implement `resample_to(x, fs_in, 2000)` using a high-quality polyphase or soxr resampler with anti-aliasing.
- [ ] **T23.4** Special-case the 44.1 kHz to 2 kHz decimation for PASCAL A (factor 22.05) and verify no aliasing via a swept-sine test.
- [ ] **T23.5** Verify the 4 kHz to 2 kHz path for CirCor and PASCAL B, and the 2 kHz no-op path for PhysioNet.
- [ ] **T23.6** Write `tests/test_resample.py` asserting output length, dtype, finiteness and preserved RMS within tolerance.
- [ ] **T23.7** [TEST] Run `tests/test_resample.py`; additionally resample a swept sine from 44.1 kHz and confirm no aliasing energy appears above 1 kHz.

### Phase 24 — Filtering
- [ ] **T24.1** Write `src/preprocessing/filters.py` — 4th-order Butterworth bandpass 20-400 Hz designed in SOS form.
- [ ] **T24.2** Apply the filter with `sosfiltfilt` for zero phase distortion; document the effective doubled order.
- [ ] **T24.3** Guard against short signals shorter than the filter padlen; fall back to reduced padding and flag the record.
- [ ] **T24.4** Validate the response against the documented Butterworth magnitude equation and plot the transfer function.
- [ ] **T24.5** Add a config-switchable no-filter path to support the preprocessing ablation (PP-09).
- [ ] **T24.6** Write `tests/test_filters.py` — a 5 Hz tone is attenuated, a 200 Hz tone passes, an 800 Hz tone is attenuated.
- [ ] **T24.7** [TEST] Run `tests/test_filters.py` — 5 Hz attenuated, 200 Hz passed, 800 Hz attenuated; confirm the shortest record (0.76 s) filters without a padlen error.

### Phase 25 — Normalization
- [ ] **T25.1** Implement per-record z-score normalization with a zero-variance guard.
- [ ] **T25.2** Implement peak normalization as an alternative path for the ablation.
- [ ] **T25.3** Implement optional DC-offset removal ahead of normalization.
- [ ] **T25.4** Add a config-switchable no-normalization path for the ablation.
- [ ] **T25.5** Assert post-normalization mean is approximately 0 and SD approximately 1 across a sample of every dataset.
- [ ] **T25.6** Write `tests/test_normalization.py`.
- [ ] **T25.7** [TEST] Run `tests/test_normalization.py`; sample 50 records per dataset and assert post-normalization mean ~0 and SD ~1.

### Phase 26 — Signal Quality Analysis
- [ ] **T26.1** Compute duration, RMS, peak amplitude and dynamic range per record.
- [ ] **T26.2** Compute a clipping ratio and a silence ratio per record.
- [ ] **T26.3** Compute an SNR proxy from the in-band (20-400 Hz) to out-of-band power ratio.
- [ ] **T26.4** Compute spectral flatness as a noise/artifact proxy and a zero-crossing anomaly score.
- [ ] **T26.5** Derive composite flags `is_noisy`, `is_short`, `is_low_quality` from configured thresholds, and calibrate thresholds against PhysioNet REFERENCE-SQI as a sanity reference.
- [ ] **T26.6** Emit **PP-08** `outputs/02_preprocessing/signal_quality_flags.csv` for all 7,536 records.
- [ ] **T26.7** [TEST] Generate quality flags for all 7,536 records; confirm no NaN flags, and sanity-check the noisy flag against PhysioNet `REFERENCE-SQI` agreement rate.

### Phase 27 — Preprocessing Pipeline & Cache
- [ ] **T27.1** Write `src/preprocessing/pipeline.py` — `preprocess(path, cfg)` chaining load, mono, resample, filter, normalize.
- [ ] **T27.2** Return a structured result carrying the signal, fs, applied-step list and quality metrics.
- [ ] **T27.3** Implement a preprocessed-signal cache under `cache/preprocessed/` keyed by record_uid plus a config hash.
- [ ] **T27.4** Add parallel batch preprocessing over all records via joblib, with a progress bar.
- [ ] **T27.5** Run full preprocessing across D1-D4 and record total wall time and cache size.
- [ ] **T27.6** Write `tests/test_pipeline_determinism.py` — two runs on the same input produce bit-identical output.
- [ ] **T27.7** [TEST] Run `tests/test_pipeline_determinism.py`; preprocess 200 random records twice and assert bit-identical output, then confirm the cache actually shortcuts the second pass.

### Phase 28 — Preprocessing Visual Outputs
- [ ] **T28.1** Emit **PP-01** `original_waveform_normal.png` from a representative PhysioNet normal record.
- [ ] **T28.2** Emit **PP-02** `original_waveform_abnormal.png` from a representative PhysioNet abnormal record.
- [ ] **T28.3** Emit **PP-03** `before_after_filtering.png` — raw versus 20-400 Hz filtered, same record, shared axes.
- [ ] **T28.4** Emit **PP-04** `normalization_comparison.png` — before and after z-score.
- [ ] **T28.5** Emit **PP-05** `normal_spectrogram.png` and **PP-06** `abnormal_spectrogram.png` with a shared colour scale.
- [ ] **T28.6** Write a shared plotting style module (300 dpi, serif, colourblind-safe palette) used by every figure in the project.
- [ ] **T28.7** [TEST/MANUAL] Open PP-01 through PP-06 and confirm the filtering figure visibly removes baseline drift and the two spectrograms share one colour scale.

### Phase 29 — Preprocessing Tables & Ablation
- [ ] **T29.1** Emit **PP-07** `preprocessing_settings.csv` — filter type, order, cutoffs, resample method, target fs, normalization.
- [ ] **T29.2** Define the preprocessing ablation grid: filter on/off crossed with normalization on/off, four configurations.
- [ ] **T29.3** Extract features and train a single fast baseline model under each of the four configurations on D1.
- [ ] **T29.4** Emit **PP-09** `preprocessing_ablation.csv` with the metric delta for each configuration.
- [ ] **T29.5** Register all PP-01 through PP-09 artifacts in the evidence index.
- [ ] **T29.6** Commit Part III and update `CHANGELOG.md`.
- [ ] **T29.7** [TEST] Confirm PP-07 through PP-09 exist, the ablation ran all four filter/normalization configurations, and every PP artifact is registered in the evidence index.

### Phase 30 — 🔴 MEGA TEST 1 — Data & Preprocessing Integrity
> Covers Parts I-III (Phases 01-28). Nothing in Part IV starts until every item here passes.
- [ ] **T30.1** [TEST] Re-run the complete dataset audit from an empty output dir; assert the counts still read PhysioNet 3,240+301, PASCAL A 124, PASCAL B 461, CirCor 942 patients / 3,163 recordings.
- [ ] **T30.2** [TEST] Assert zero subject overlap across every fold of every split map, and that the 301 PhysioNet validation records appear in no CV fold.
- [ ] **T30.3** [TEST] Preprocess 200 random records across all four datasets twice; assert bit-identical output and that the cache is actually reused.
- [ ] **T30.4** [TEST] Run preprocessing on the duration extremes — the 0.76 s PASCAL B record and the 122 s PhysioNet record — and confirm neither errors nor silently truncates.
- [ ] **T30.5** [TEST] Assert no `Heartbeat_Sound` file appears in any supervised record set, and that no `record_uid` is duplicated anywhere in master metadata.
- [ ] **T30.6** [TEST/MANUAL] Open PP-01 through PP-06 and the audit report; confirm the figures look right and that both count discrepancies are stated in writing.
- [ ] **T30.7** [TEST] Full pytest suite green with nothing skipped that should run; commit and tag `mega-1-data-integrity`.

---

# PART IV — FEATURE ENGINEERING (the locked 138)

**Locked composition:** Time 24 + Frequency 22 + MFCC 39 + Chroma 24 + DWT 24 + Envelope 5 = **138**

### Phase 31 — Feature Framework
- [ ] **T31.1** Write `src/feature_extraction/base.py` defining a `FeatureExtractor` protocol: `name`, `family`, `feature_names()`, `extract(signal, fs)`.
- [ ] **T31.2** Build a global feature registry mapping every feature name to its family, extractor and equation reference.
- [ ] **T31.3** Enforce a fixed, deterministic feature ordering so column indices are stable across every run.
- [ ] **T31.4** Add a NaN/Inf policy: extractors return NaN on failure, never raise, and every failure is logged with the record uid.
- [ ] **T31.5** Add per-family timing instrumentation for the complexity analysis later.
- [ ] **T31.6** Write `tests/test_feature_registry.py` asserting exactly 138 registered names with no duplicates.
- [ ] **T31.7** [TEST] Run `tests/test_feature_registry.py` — exactly 138 names, no duplicates, and the order is stable across two interpreter sessions.

### Phase 32 — Time-Domain Features (24)
- [ ] **T32.1** Basic statistics: mean, std, variance, min, max, range, median, IQR (8).
- [ ] **T32.2** Shape statistics: skewness, kurtosis (2).
- [ ] **T32.3** Energy measures: total energy, RMS, peak-to-peak, crest factor (4).
- [ ] **T32.4** Zero-crossing rate mean and std over frames (2).
- [ ] **T32.5** Complexity measures: Shannon entropy, sample entropy, Hjorth activity, Hjorth mobility, Hjorth complexity (5).
- [ ] **T32.6** Autocorrelation peak value, autocorrelation peak lag, and signal duration (3). Assert the family totals 24.
- [ ] **T32.7** [TEST] Assert the time family returns exactly 24 finite values on synthetic input and on one real record from each of the four datasets.

### Phase 33 — Frequency-Domain Features (22)
- [ ] **T33.1** Spectral centroid mean and std; spectral bandwidth mean and std (4).
- [ ] **T33.2** Spectral rolloff at 85 percent and 95 percent, each mean and std (4).
- [ ] **T33.3** Spectral flatness mean and std; spectral flux mean and std (4).
- [ ] **T33.4** Global spectral entropy, dominant frequency, peak power, total power (4).
- [ ] **T33.5** Relative band power in six bands: 20-50, 50-100, 100-150, 150-250, 250-350, 350-400 Hz (6).
- [ ] **T33.6** Verify Welch PSD parameters are fixed in config and that the family totals 22.
- [ ] **T33.7** [TEST] Assert the frequency family returns exactly 22 finite values and that the six band powers sum to no more than the total power.

### Phase 34 — MFCC Features (39)
- [ ] **T34.1** Configure the mel filterbank for a 2 kHz signal: n_mels, fmin 20, fmax 400 (Nyquist-safe), n_fft and hop from config.
- [ ] **T34.2** Compute 13 MFCC coefficients per frame.
- [ ] **T34.3** Aggregate 13 MFCC means (13).
- [ ] **T34.4** Aggregate 13 MFCC standard deviations (13).
- [ ] **T34.5** Compute delta-MFCC and aggregate 13 delta means (13). Assert the family totals 39.
- [ ] **T34.6** Guard very short recordings where the frame count is below the delta width; pad or degrade gracefully and flag.
- [ ] **T34.7** [TEST] Assert the MFCC family returns exactly 39 values; confirm `fmax` is below Nyquist and that a 0.76 s record degrades gracefully rather than raising.

### Phase 35 — Chroma Features (24)
- [ ] **T35.1** Compute a chroma-STFT representation adapted to the 20-400 Hz band.
- [ ] **T35.2** Aggregate 12 chroma bin means (12).
- [ ] **T35.3** Aggregate 12 chroma bin standard deviations (12). Assert the family totals 24.
- [ ] **T35.4** Document the physiological caveat: chroma is a musical-pitch construct applied here as a generic harmonic descriptor.
- [ ] **T35.5** Verify chroma stability on a synthetic constant-pitch signal.
- [ ] **T35.6** Emit **FE-07** `chroma_heatmap.png` from a representative record.
- [ ] **T35.7** [TEST] Assert the chroma family returns exactly 24 values and is stable on a synthetic constant-pitch signal.

### Phase 36 — Wavelet (DWT) Features (24)
- [ ] **T36.1** Configure a 5-level db4 discrete wavelet decomposition, giving sub-bands cA5, cD5, cD4, cD3, cD2, cD1.
- [ ] **T36.2** Compute per-sub-band energy (6).
- [ ] **T36.3** Compute per-sub-band standard deviation (6).
- [ ] **T36.4** Compute per-sub-band entropy (6).
- [ ] **T36.5** Compute per-sub-band mean absolute value (6). Assert the family totals 24.
- [ ] **T36.6** Guard signals too short for 5 levels; reduce the level and flag the record. Emit **FE-08** `wavelet_decomposition.png`.
- [ ] **T36.7** [TEST] Assert the DWT family returns exactly 24 values; confirm a signal too short for 5 levels reduces the level and sets a flag instead of failing.

### Phase 37 — Envelope Features (5)
- [ ] **T37.1** Compute the Hilbert analytic envelope of the preprocessed signal.
- [ ] **T37.2** Smooth the envelope with a configured low-pass and extract envelope mean and std (2).
- [ ] **T37.3** Extract envelope skewness and kurtosis (2).
- [ ] **T37.4** Detect envelope peaks and compute peak rate in peaks per second as a heart-rate proxy (1). Assert the family totals 5.
- [ ] **T37.5** Sanity-check the peak rate against a plausible physiological range and flag outliers.
- [ ] **T37.6** Emit **FE-09** `signal_envelope.png`.
- [ ] **T37.7** [TEST] Assert the envelope family returns exactly 5 values and that the peak rate falls in a plausible physiological range on real recordings.

### Phase 38 — Feature Assembly & Validation
- [ ] **T38.1** Write `src/feature_extraction/extractor.py` — `extract_all(signal, fs)` returning a 138-length ordered vector.
- [ ] **T38.2** Assert on every call that the vector length is exactly 138 and the name order matches the registry.
- [ ] **T38.3** Emit **FE-01** `feature_inventory.csv` — index, name, family, equation or reference, unit, description, for all 138.
- [ ] **T38.4** Emit **FE-02** `feature_family_summary.csv` — the six families with counts 24/22/39/24/24/5.
- [ ] **T38.5** Write `tests/test_extract_all.py` running on the synthetic signal and asserting all values are finite.
- [ ] **T38.6** Benchmark single-record extraction time per family and store it for the complexity table.
- [ ] **T38.7** [TEST] Run `tests/test_extract_all.py` — the vector is length 138, ordered per the registry, and all values finite on real records from all four datasets.

### Phase 39 — Batch Extraction Runner
- [ ] **T39.1** Write `scripts/02_extract_features.py` with `--dataset`, `--limit`, `--workers`, `--force` options.
- [ ] **T39.2** Parallelize with joblib over records, reading from the preprocessed cache.
- [ ] **T39.3** Write incremental checkpointing so an interrupted run resumes rather than restarting.
- [ ] **T39.4** Log every extraction failure with the record uid, family and exception to `outputs/03_features/extraction_errors.csv`.
- [ ] **T39.5** Write per-dataset Parquet shards under `cache/features/`.
- [ ] **T39.6** Add a `--smoke` mode processing 20 records per dataset for fast validation.
- [ ] **T39.7** [TEST] Run the extractor in `--smoke` mode (20 records per dataset); confirm checkpointing resumes correctly after a deliberate interrupt.

### Phase 40 — Run Extraction on All Datasets
- [ ] **T40.1** Extract features for PhysioNet training a-f (3,240 records) and record wall time.
- [ ] **T40.2** Extract features for the PhysioNet held-out validation set (301 records).
- [ ] **T40.3** Extract features for PASCAL set_a (176 records including unlabelled).
- [ ] **T40.4** Extract features for PASCAL set_b (656 records including unlabelled).
- [ ] **T40.5** Extract features for CirCor (3,163 records) and record wall time.
- [ ] **T40.6** Emit **FE-03** `all_features_matrix.parquet` — the merged matrix joined to master metadata by `record_uid`.
- [ ] **T40.7** [TEST] Confirm the merged matrix has one row per non-excluded record with 138 columns, joins cleanly to master metadata on `record_uid`, and records its wall time.

### Phase 41 — Feature Quality Assurance
- [ ] **T41.1** Emit **FE-04** `feature_missing_values.csv` — per-feature NaN and Inf counts with the responsible records.
- [ ] **T41.2** Identify constant and near-zero-variance features and report them (do not drop yet; dropping happens inside folds).
- [ ] **T41.3** Identify features with extreme outliers or unbounded ranges and document a clipping or transform policy.
- [ ] **T41.4** Re-extract a random sample of 50 records and assert bit-identical reproduction of the cached values.
- [ ] **T41.5** Check per-dataset feature distributions for domain shift and record the finding for the generalization analysis.
- [ ] **T41.6** Emit **FE-10** `feature_correlation_matrix.png` plus its source CSV.
- [ ] **T41.7** [TEST] Re-extract 50 random records and assert bit-identical values against the cache; confirm FE-04 accounts for every NaN and Inf with a named record.

### Phase 42 — Feature Visual Outputs
- [ ] **T42.1** Emit **FE-05** `feature_distribution_plots/` — histograms and boxplots for the top features by class separation.
- [ ] **T42.2** Emit **FE-06** `mfcc_heatmap.png` from a representative record.
- [ ] **T42.3** Emit **G10** `feature_family_count_chart.png` visualizing the 24/22/39/24/24/5 composition.
- [ ] **T42.4** Produce class-conditional distribution overlays for the ten most discriminative features on D1.
- [ ] **T42.5** Register all FE artifacts in the evidence index.
- [ ] **T42.6** Commit Part IV and update `CHANGELOG.md`.
- [ ] **T42.7** [TEST/MANUAL] Open the distribution plots and correlation heatmap; confirm the family counts in G10 read 24/22/39/24/24/5.

---

# PART V — MODELING CORE

### Phase 43 — Cross-Validation Infrastructure
- [ ] **T43.1** Write `src/evaluation/cv.py` loading the split maps from DA-07 rather than re-deriving folds at runtime.
- [ ] **T43.2** Implement a `run_cv(estimator_factory, X, y, groups, folds)` driver returning per-fold predictions and probabilities.
- [ ] **T43.3** Persist per-fold train/test index arrays and out-of-fold predictions to disk for later paired statistical tests.
- [ ] **T43.4** Implement repeated CV: 5x5 grouped for the primary D1 track, and 5x2 stratified for the small PASCAL A track.
- [ ] **T43.5** Assert at runtime, inside every fold, that the train and test group sets are disjoint.
- [ ] **T43.6** Write `tests/test_cv_driver.py` on synthetic data with known groups.
- [ ] **T43.7** [TEST] Run `tests/test_cv_driver.py`; assert the driver loads folds from DA-07 rather than re-deriving them, and that out-of-fold predictions cover every record exactly once.

### Phase 44 — Fold-Safe Pipeline
- [ ] **T44.1** Build an sklearn `Pipeline` of imputer, scaler, optional selector and estimator, so every step fits on train only.
- [ ] **T44.2** Configure the imputer (median) and scaler (StandardScaler) with their parameters exposed in config.
- [ ] **T44.3** Wire the feature-selection step so it can be disabled, use a fixed subset, or run a search inside the fold.
- [ ] **T44.4** Add class-weight and resampling options driven by the imbalance ratio of each task.
- [ ] **T44.5** Write `tests/test_fold_safety.py` — inject a leakage canary feature and assert the pipeline never sees test statistics.
- [ ] **T44.6** Document the fold-safety guarantee in the README as a QA claim.
- [ ] **T44.7** [TEST] Run `tests/test_fold_safety.py` — inject a leakage canary feature and confirm the pipeline never sees test-fold statistics.

### Phase 45 — Metrics Module
- [ ] **T45.1** Write `src/evaluation/metrics.py` — accuracy, precision, recall/sensitivity, specificity, F1.
- [ ] **T45.2** Add balanced accuracy, ROC-AUC, PR-AUC and MCC.
- [ ] **T45.3** Add macro-F1, weighted-F1, per-class precision/recall and one-vs-rest AUC for multiclass tasks.
- [ ] **T45.4** Add confusion-matrix computation with fixed class ordering and both raw and normalized forms.
- [ ] **T45.5** Add calibration metrics: Brier score and expected calibration error over configurable bins.
- [ ] **T45.6** Add bootstrap confidence intervals for any scalar metric; write `tests/test_metrics.py` against hand-computed values.
- [ ] **T45.7** [TEST] Run `tests/test_metrics.py` against hand-computed values for a small confusion matrix; confirm macro and per-class metrics agree with scikit-learn.

### Phase 46 — Baseline Models M1 and M2
- [ ] **T46.1** Implement **M1** Logistic Regression with the default configuration and balanced class weights.
- [ ] **T46.2** Define the M1 hyperparameter search space (C, penalty, solver, max_iter).
- [ ] **T46.3** Implement **M2** K-Nearest Neighbours (optional baseline).
- [ ] **T46.4** Define the M2 search space (n_neighbors, weights, metric, p).
- [ ] **T46.5** Verify both expose `predict_proba` for the soft-voting ensemble.
- [ ] **T46.6** Smoke-run both on D1 fold 0 and record the metrics.
- [ ] **T46.7** [TEST] Smoke-run M1 and M2 on D1 fold 0; confirm both expose `predict_proba` and produce no NaN.

### Phase 47 — SVM-RBF (M3)
- [ ] **T47.1** Implement **M3** SVC with an RBF kernel, `probability=False`, wrapped for explicit calibration.
- [ ] **T47.2** Wrap M3 in `CalibratedClassifierCV` (sigmoid and isotonic variants) fitted inside the training fold only.
- [ ] **T47.3** Define the M3 search space (C, gamma, class_weight).
- [ ] **T47.4** Benchmark M3 fit time on the full D1 feature matrix and decide whether to cap `cache_size` or subsample during search.
- [ ] **T47.5** Verify calibrated probabilities are well-formed: within [0,1] and summing to 1.
- [ ] **T47.6** Smoke-run M3 on D1 fold 0 and record metrics and fit time.
- [ ] **T47.7** [TEST] Confirm M3's calibrated probabilities lie in [0,1] and sum to 1, and record the fit time so the search budget can be planned.

### Phase 48 — Random Forest (M4)
- [ ] **T48.1** Implement **M4** RandomForestClassifier with a fixed seed and `n_jobs` from config.
- [ ] **T48.2** Define the M4 search space (n_estimators, max_depth, min_samples_leaf, max_features, class_weight).
- [ ] **T48.3** Expose impurity-based feature importances for the explainability module.
- [ ] **T48.4** Add permutation importance computed on the held-out fold as the more trustworthy alternative.
- [ ] **T48.5** Assess whether RF probabilities need calibration and record the decision with evidence.
- [ ] **T48.6** Smoke-run M4 on D1 fold 0 and record metrics, fit time and model size.
- [ ] **T48.7** [TEST] Smoke-run M4; confirm impurity and permutation importances are both produced and that permutation importance is computed on the held-out fold only.

### Phase 49 — Gradient Boosting (M5) and External Baseline (M8)
- [ ] **T49.1** Implement **M5** GradientBoostingClassifier (or HistGradientBoosting where the speed is needed) with a fixed seed.
- [ ] **T49.2** Define the M5 search space (n_estimators, learning_rate, max_depth, subsample, min_samples_leaf).
- [ ] **T49.3** Implement **M8** XGBoost or LightGBM behind a capability check that degrades gracefully if the package is unavailable.
- [ ] **T49.4** Define the M8 search space and set `n_jobs` and determinism flags.
- [ ] **T49.5** Verify M5 and M8 probability outputs and calibration behaviour.
- [ ] **T49.6** Smoke-run both on D1 fold 0 and record metrics, fit time and model size.
- [ ] **T49.7** [TEST] Smoke-run M5 and M8; if the external baseline is unavailable, confirm the capability check degrades gracefully and the reason is recorded.

### Phase 50 — Soft-Voting Ensembles (M6 and M7)
- [ ] **T50.1** Write `src/ensemble/soft_voting.py` implementing weighted probability averaging over calibrated base models.
- [ ] **T50.2** Implement **M6** equal-weight soft voting over SVM-RBF, RF and GB — the mandatory ensemble baseline.
- [ ] **T50.3** Implement **M7** optimized-weight soft voting with weights as tunable parameters — the proposed model.
- [ ] **T50.4** Implement the final prediction rule as argmax over the fused probability vector, matching the documented equations.
- [ ] **T50.5** Ensure base models are fitted inside the fold and their out-of-fold probabilities drive weight optimization, with no test leakage.
- [ ] **T50.6** Write `tests/test_soft_voting.py` — with equal weights the fused output equals the arithmetic mean; weights normalize to 1.
- [ ] **T50.7** [TEST] Run `tests/test_soft_voting.py` — equal weights equal the arithmetic mean, weights normalise to 1, and weight optimization never touches the outer test fold.

### Phase 51 — Model Registry & Persistence
- [ ] **T51.1** Write `src/models/registry.py` mapping model ids M1-M9 to factories, search spaces and metadata.
- [ ] **T51.2** Implement model save/load via joblib into `models_saved/{task}/{model_id}/`.
- [ ] **T51.3** Record model file size on disk for the complexity table (T26).
- [ ] **T51.4** Record training time and single-record inference time for every model.
- [ ] **T51.5** Store the fitted pipeline together with its feature-name list so inference cannot silently misalign columns.
- [ ] **T51.6** Write `tests/test_model_roundtrip.py` — a reloaded model reproduces identical predictions.
- [ ] **T51.7** [TEST] Run `tests/test_model_roundtrip.py` — every saved model reloads and reproduces identical predictions; confirm size and timing are recorded.

### Phase 52 — Optional Deep Baseline (M9)
- [ ] **T52.1** [MANUAL] Decide and document whether the 1D-CNN is in scope given CPU-only hardware — a user call, made against the Phase 75 complexity timings.
- [ ] **T52.2** If in scope, define a compact 1D-CNN over fixed-length preprocessed segments.
- [ ] **T52.3** Implement segment extraction with padding and cropping to a fixed length.
- [ ] **T52.4** Train under the same fold map as the classical models so the comparison is fair.
- [ ] **T52.5** Record accuracy alongside training time and model size for the complexity trade-off argument.
- [ ] **T52.6** If out of scope, record the exclusion and its technical reason in `missing_outputs_report.txt`.
- [ ] **T52.7** [TEST] Confirm the 1D-CNN decision is recorded either as an implemented model or as an entry in `missing_outputs_report.txt` with its technical reason.

### Phase 53 — 🔴 MEGA TEST 2 — Feature & Model Integrity
> Covers Parts IV-V (Phases 30-51). Nothing in Part VI starts until every item here passes.
- [ ] **T53.1** [TEST] Assert the feature registry holds exactly 138 names in a stable order, and that the family counts are 24/22/39/24/24/5.
- [ ] **T53.2** [TEST] Re-extract 100 random records and assert bit-identical values against the cached matrix.
- [ ] **T53.3** [TEST] Run the leakage canary end to end: inject a feature perfectly correlated with test-fold labels and confirm the fold-safe pipeline does not benefit from it.
- [ ] **T53.4** [TEST] Round-trip every saved model; assert reloaded predictions are identical and that the stored feature list prevents column misalignment.
- [ ] **T53.5** [TEST] Assert every model's probabilities lie in [0,1] and sum to 1 per row, including the calibrated SVM and both ensembles.
- [ ] **T53.6** [TEST] Smoke-run every mandatory model on D1 fold 0; no NaN, no crash, and record the baseline metric table for later comparison.
- [ ] **T53.7** [TEST] Full pytest suite green; commit and tag `mega-2-features-models`.

---

# PART VI — SEARCH & OPTIMIZATION

### Phase 54 — Search Framework
- [ ] **T54.1** Write `src/optimization/base.py` defining a common search interface: space, objective, budget, seed, history.
- [ ] **T54.2** Define the scoring objective per task: balanced accuracy plus sensitivity for binary, macro-F1 for multiclass.
- [ ] **T54.3** Implement nested cross-validation so search runs on inner folds and evaluation on the untouched outer fold.
- [ ] **T54.4** Log every trial (params, score, duration) to `outputs/05_search_optimization/{exp}/trials.csv`.
- [ ] **T54.5** Add a wall-clock and trial-count budget per search with graceful early termination.
- [ ] **T54.6** Write `tests/test_search_no_leakage.py` asserting the outer test fold is never scored during search.
- [ ] **T54.7** [TEST] Run `tests/test_search_no_leakage.py` — assert the outer test fold is never scored during any search, and that every trial is logged.

### Phase 55 — RandomizedSearchCV (SO-01)
- [ ] **T55.1** Implement the RandomizedSearchCV wrapper reading spaces from the model registry.
- [ ] **T55.2** Run the search for SVM-RBF on D1 and persist the best parameters.
- [ ] **T55.3** Run the search for Random Forest on D1 and persist the best parameters.
- [ ] **T55.4** Run the search for Gradient Boosting on D1 and persist the best parameters.
- [ ] **T55.5** Run searches for Logistic Regression and the external baseline on D1.
- [ ] **T55.6** Emit the SO-01 best-parameter JSON and trial history for every model.
- [ ] **T55.7** [TEST] Confirm best parameters and full trial histories exist for every model, and that the search ran only on inner folds.

### Phase 56 — Bayesian Optimization (SO-02)
- [ ] **T56.1** Implement a scikit-optimize `BayesSearchCV` wrapper behind an availability check.
- [ ] **T56.2** Convert each model search space to a skopt dimension specification.
- [ ] **T56.3** Run Bayesian optimization for the three primary models on D1 under an equal trial budget to RandomizedSearchCV.
- [ ] **T56.4** Record the convergence trace (best score versus trial index) for the convergence plot.
- [ ] **T56.5** [MANUAL] Compare Bayesian against Randomized at equal budget; the user confirms which becomes the final search method.
- [ ] **T56.6** Emit SO-02 outputs; if skopt is unavailable, record the reason in `missing_outputs_report.txt`.
- [ ] **T56.7** [TEST] Confirm the Bayesian convergence trace exists and the equal-budget comparison against RandomizedSearchCV is recorded, or the skip reason is logged.

### Phase 57 — Feature Selection Search (SO-04)
- [ ] **T57.1** Implement a filter baseline: mutual information and ANOVA F ranking with a top-k cut.
- [ ] **T57.2** Implement embedded selection via RF and GB importance thresholds.
- [ ] **T57.3** Implement RFECV over a compute-affordable estimator.
- [ ] **T57.4** Sweep k across a configured grid and record performance at each feature count for the trade-off curve.
- [ ] **T57.5** Select the final compact subset by the multi-objective score and emit **FE-12** `selected_feature_subset.csv`.
- [ ] **T57.6** Emit SO-04 comparison: all 138 features versus the selected subset, same folds and seed.
- [ ] **T57.7** [TEST] Confirm the feature-count sweep produced a performance curve and that the selected subset is smaller than 138 and reproducible.

### Phase 58 — Genetic Algorithm (SO-03a)
- [ ] **T58.1** Implement a binary-chromosome GA over the 138-feature mask using DEAP or a compact custom implementation.
- [ ] **T58.2** Define the fitness function as the multi-objective score J with configured alpha, beta and gamma.
- [ ] **T58.3** Configure population size, generations, crossover and mutation rates, and elitism, all in config.
- [ ] **T58.4** Run the GA inside training folds only and log per-generation best and mean fitness.
- [ ] **T58.5** Extend the chromosome to jointly encode ensemble weights alongside the feature mask.
- [ ] **T58.6** Emit the GA convergence trace and its selected subset; document skipping with a reason if compute-bound.
- [ ] **T58.7** [TEST] Confirm the GA convergence trace exists with per-generation fitness, or that the skip is recorded with a compute-bound reason.

### Phase 59 — Particle Swarm Optimization (SO-03b)
- [ ] **T59.1** Implement PSO over the continuous ensemble-weight simplex.
- [ ] **T59.2** Implement binary PSO over the feature mask as an alternative to the GA.
- [ ] **T59.3** Configure swarm size, inertia, cognitive and social coefficients, and iteration budget.
- [ ] **T59.4** Run PSO inside training folds and log per-iteration best fitness.
- [ ] **T59.5** Compare GA against PSO on identical folds, budget and seed.
- [ ] **T59.6** Emit the PSO outputs or record the skip reason.
- [ ] **T59.7** [TEST] Confirm the PSO run or its documented skip, and that the GA-versus-PSO comparison used identical folds, budget and seed.

### Phase 60 — Ensemble Weight Optimization (SO-05)
- [ ] **T60.1** Implement a coarse grid search over the three-model weight simplex as the deterministic baseline.
- [ ] **T60.2** Implement continuous weight optimization via SciPy constrained minimization on out-of-fold probabilities.
- [ ] **T60.3** Constrain weights to be non-negative and sum to one; document the normalization.
- [ ] **T60.4** Optimize weights on inner-fold out-of-fold probabilities only, never on the outer test fold.
- [ ] **T60.5** Emit SO-05: equal weights versus optimized weights, with the per-fold metric delta.
- [ ] **T60.6** Emit the final weight vector and its per-fold variance as a stability check.
- [ ] **T60.7** [TEST] Assert the optimized weights are non-negative and sum to 1, and that per-fold weight variance is reported as a stability check.

### Phase 61 — Multi-Objective Optimization (SO-06)
- [ ] **T61.1** Implement the documented objective J = alpha(1 - MacroF1) + beta(SelectedFeatures/138) + gamma(NormalizedInferenceTime).
- [ ] **T61.2** Expose alpha, beta and gamma in config and record the chosen values with their justification.
- [ ] **T61.3** Normalize inference time against the slowest configuration so the term is bounded in [0,1].
- [ ] **T61.4** Sweep the weighting to trace a performance-versus-complexity Pareto front.
- [ ] **T61.5** Select and record the final operating point on the front.
- [ ] **T61.6** Emit SO-06 outputs and the Pareto data as CSV.
- [ ] **T61.7** [TEST] Confirm the multi-objective score is implemented exactly as documented, the inference-time term is bounded in [0,1], and the Pareto data is exported.

### Phase 62 — Search Reporting
- [ ] **T62.1** Emit **T07** `search_space_and_best_parameters.csv` — variable, range, distribution, final selected value, per model.
- [ ] **T62.2** Emit **G20** `search_convergence_plot.png` overlaying Randomized, Bayesian, GA and PSO traces.
- [ ] **T62.3** Emit **G21** `all_features_vs_selected_features.png`.
- [ ] **T62.4** Emit **G22** `f1_accuracy_vs_feature_count.png` from the T57.4 sweep.
- [ ] **T62.5** Register every SO artifact in the evidence index.
- [ ] **T62.6** Commit Part VI and update `CHANGELOG.md`.
- [ ] **T62.7** [TEST] Confirm T07, G20, G21 and G22 exist with their source CSVs and that every SO artifact is in the evidence index.

---

# PART VII — EXPERIMENTAL RUNS

### Phase 63 — Experiment Runner
- [ ] **T63.1** Write `src/evaluation/experiment.py` — an `Experiment` object built from `configs/experiments.yaml`.
- [ ] **T63.2** Standardize the output contract: per-fold metrics CSV, aggregate metrics CSV, predictions Parquet, confusion matrices, config snapshot.
- [ ] **T63.3** Write every run into `outputs/<section>/<EXP-ID>/` with the run manifest embedded.
- [ ] **T63.4** Implement resume-on-restart so a completed fold is not recomputed.
- [ ] **T63.5** Write `scripts/03_run_experiment.py --exp EXP-A1` as the single entry point for all runs.
- [ ] **T63.6** Write `scripts/04_run_all_experiments.py` chaining every mandatory run in dependency order.
- [ ] **T63.7** [TEST] Run one experiment end to end via `scripts/03_run_experiment.py`; confirm the output contract is complete and that resume-on-restart skips a completed fold.

### Phase 64 — EXP-A1 PhysioNet Binary Baseline
- [ ] **T64.1** Run M1 Logistic Regression on D1 under the **repeated 5x5 grouped CV** map from T20.2, using default parameters. Every model in this phase uses the same 25 folds.
- [ ] **T64.2** Run M3 SVM-RBF and M4 Random Forest under the identical fold map.
- [ ] **T64.3** Run M5 Gradient Boosting and, where available, M8 XGBoost/LightGBM.
- [ ] **T64.4** Run M6 equal-weight soft voting over the calibrated SVM, RF and GB.
- [ ] **T64.5** Evaluate all models on the untouched 301-record PhysioNet validation set.
- [ ] **T64.6** Emit **T08** individual-model comparison and **T10** fold-wise results as mean plus or minus SD across all 25 folds, retaining the per-fold values for the paired tests in Phase 81.
- [ ] **T64.7** [TEST] Confirm all mandatory models ran on the identical 25-fold map, the 301-record validation set was evaluated, per-fold values are persisted for Phase 81, and T08 and T10 are produced with mean and SD.

### Phase 65 — EXP-A2 PhysioNet Binary Optimized
- [ ] **T65.1** Re-run every model with the SO-01/SO-02 tuned hyperparameters under nested CV, with the repeated 5x5 map as the outer loop so the optimized results are paired fold-for-fold with the T63 baselines.
- [ ] **T65.2** Run M7 optimized-weight soft voting with weights from SO-05.
- [ ] **T65.3** Run the optimized ensemble restricted to the SO-04 selected feature subset.
- [ ] **T65.4** Apply the documented selection rule: prioritize sensitivity and balanced accuracy, not raw accuracy.
- [ ] **T65.5** Emit **T09** equal-weight versus optimized-weight ensemble comparison.
- [ ] **T65.6** Persist the final selected model to `models_saved/binary/final/` with its full pipeline and feature list.
- [ ] **T65.7** [TEST] Confirm the optimized run used nested CV, that selection followed the sensitivity-and-balanced-accuracy rule, and that the final model is persisted with its feature list.

### Phase 66 — EXP-B1 PASCAL A Four-Class
- [ ] **T66.1** Assemble the D2 four-class dataset: 124 labeled records across normal, murmur, extrahls, artifact.
- [ ] **T66.2** Configure repeated stratified 5x2 CV given the very small sample size.
- [ ] **T66.3** Run all mandatory models with balanced class weights.
- [ ] **T66.4** Run both ensemble variants under the same folds.
- [ ] **T66.5** Emit **T11** PASCAL A results: macro-F1, weighted-F1, balanced accuracy, per-class precision and recall, one-vs-rest AUC.
- [ ] **T66.6** Document the small-sample limitation, report confidence intervals on every headline metric, and state that `artifact` is a **recording-quality** label rather than a cardiac class — so this is not a four-class cardiac classifier and must not be described as one.
- [ ] **T66.7** [TEST] Confirm repeated 5x2 CV ran, per-class recall is reported for all four classes, and confidence intervals accompany every headline metric given n=124.

### Phase 67 — EXP-B2 PASCAL B Three-Class
- [ ] **T67.1** Assemble the D3 three-class dataset: 461 labeled records across normal, murmur, extrastole.
- [ ] **T67.2** Configure subject-grouped repeated CV using the 167 filename-derived subject groups.
- [ ] **T67.3** Run all mandatory models and both ensembles.
- [ ] **T67.4** Emit **T12** PASCAL B results with the full multiclass metric set.
- [ ] **T67.5** Analyse the severe class imbalance (320 / 95 / 46) and its effect on per-class recall.
- [ ] **T67.6** Explicitly confirm in the report that sets A and B were never merged, per the locked instruction.
- [ ] **T67.7** [TEST] Confirm subject-grouped CV used the 167 groups, and that the report states explicitly that sets A and B were never merged.

### Phase 68 — EXP-C1 CirCor Murmur
- [ ] **T68.1** Assemble the D4 murmur task at recording level with patient-wise grouping.
- [ ] **T68.2** Run the murmur task **both ways**: 3-class including the 68 Unknown patients (headline, matches the 2022 Challenge) and 2-class on the 874 known patients. Report both; do not silently pick one.
- [ ] **T68.3** Run the best models from EXP-A2 under patient-wise 5-fold CV.
- [ ] **T68.4** Implement recording-to-patient aggregation (max, mean, and any-present rules) and evaluate each.
- [ ] **T68.5** Emit **T13** CirCor murmur results at both recording and patient level.
- [ ] **T68.6** Emit **T15** recording-level versus subject-level comparison.
- [ ] **T68.7** [TEST] Confirm **both** murmur variants ran — 3-class with the 68 Unknown patients and 2-class on the 874 known — and that both are reported.

### Phase 69 — EXP-C2 CirCor Outcome
- [ ] **T69.1** Build the outcome task from the `#Outcome` field parsed out of the patient txt files (486 Normal / 456 Abnormal).
- [ ] **T69.2** Propagate the patient-level outcome label to that patient's recordings, and document the noise this introduces.
- [ ] **T69.3** Run the best models under patient-wise 5-fold CV.
- [ ] **T69.4** Evaluate patient-level aggregation strategies for the outcome task.
- [ ] **T69.5** Emit **T14** CirCor clinical-outcome results.
- [ ] **T69.6** Note in the report that outcome is a near-balanced task, unlike murmur, and interpret metrics accordingly.
- [ ] **T69.7** [TEST] Confirm the outcome labels came from the txt files (486/456), that patient-level aggregation was evaluated, and that label propagation noise is documented.

### Phase 70 — EXP-C3 CirCor Location Analysis
- [ ] **T70.1** Stratify CirCor predictions by auscultation location: AV 800, PV 766, TV 732, MV 861, Phc 4.
- [ ] **T70.2** Compute per-location metrics for the murmur task.
- [ ] **T70.3** Compute per-location metrics for the outcome task.
- [ ] **T70.4** Exclude or flag Phc (n=4) as statistically uninformative.
- [ ] **T70.5** Cross-reference the `Most audible location` field against per-location model confidence.
- [ ] **T70.6** Emit **T22** auscultation-location analysis and **G32** the location performance chart.
- [ ] **T70.7** [TEST] Confirm per-location metrics exist for AV, PV, TV and MV, and that Phc (n=4) is flagged as statistically uninformative rather than reported as a result.

### Phase 71 — EXP-D1 Cross-Dataset Generalization
- [ ] **T71.1** Confirm label compatibility AND record the population mismatch up front: PhysioNet is adult (median age 25, 0.3% under 18); CirCor is ~98% paediatric (Child 598, Infant 191, Adolescent 66, Neonate 6). This is adult-to-paediatric transfer, not a like-for-like test.
- [ ] **T71.2** Train on the full PhysioNet training set with the finalized EXP-A2 configuration; no retuning of any kind.
- [ ] **T71.3** Test directly on all 3,163 CirCor recordings and report the metrics.
- [ ] **T71.4** Aggregate to CirCor patient level and report those metrics separately.
- [ ] **T71.5** Quantify degradation against the in-domain EXP-A2 result and frame it correctly: a large drop is the expected consequence of the population mismatch, NOT evidence that the method fails to generalize. Any write-up claiming otherwise is wrong.
- [ ] **T71.6** Emit **T16** cross-dataset generalization and **G29** the performance-drop chart.
- [ ] **T71.7** [TEST] Confirm the population mismatch is recorded in the experiment metadata **before** the metric, no retuning occurred on CirCor, and the framing rule in T71.5 is honoured.

### Phase 72 — EXP-E1 Noise Robustness
- [ ] **T72.1** Partition every test set into clean, noisy and low-confidence groups using the PP-08 quality flags.
- [ ] **T72.2** Compute metrics per noise group for the final model on each dataset.
- [ ] **T72.3** Cross-check the grouping against the PhysioNet REFERENCE-SQI values as an independent reference.
- [ ] **T72.4** Optionally add synthetic-noise injection at controlled SNR levels for a graded degradation curve.
- [ ] **T72.5** Emit **T20** noise robustness.
- [ ] **T72.6** Emit **G30** the noise-level robustness chart.
- [ ] **T72.7** [TEST] Confirm noise groups were derived from PP-08 flags and cross-checked against `REFERENCE-SQI`, and that T20 and G30 exist.

### Phase 73 — EXP-E2 Duration Robustness
- [ ] **T73.1** Partition test sets into short, medium and long bands using the T18.3 assignments.
- [ ] **T73.2** Compute metrics per duration band for the final model on each dataset.
- [ ] **T73.3** Analyse the shortest recordings specifically — PASCAL B reaches 0.76 s — for feature-extraction degradation.
- [ ] **T73.4** Optionally run a truncation study: evaluate the same recordings clipped to 3, 5 and 10 seconds.
- [ ] **T73.5** Emit **T21** duration robustness.
- [ ] **T73.6** Emit **G31** the duration-wise performance chart.
- [ ] **T73.7** [TEST] Confirm duration bands match the T18.3 assignments, that the shortest recordings are analysed separately, and that T21 and G31 exist.

### Phase 74 — EXP-F1 Feature Ablation A1-A10
- [ ] **T74.1** Run **A1** time-domain only (24 features) and **A2** frequency-domain only (22).
- [ ] **T74.2** Run **A3** MFCC only (39) and **A4** wavelet only (24).
- [ ] **T74.3** Run **A5** time plus frequency (46) and **A6** MFCC plus wavelet (63).
- [ ] **T74.4** Run **A7** all 138 features — the full proposed representation.
- [ ] **T74.5** Run **A8** the optimized feature subset from SO-04.
- [ ] **T74.6** Emit **T17** feature-family ablation and **T18** all-versus-selected, on identical folds and seed.
- [ ] **T74.7** [TEST] Confirm all ten ablation configurations ran on identical folds and seed, and that the feature counts per configuration are 24/22/39/24/46/63/138 and the selected subset.

### Phase 75 — EXP-F2 Optimization Ablation
- [ ] **T75.1** Run **A9** the best individual model versus the ensemble.
- [ ] **T75.2** Run **A10** equal-weight versus optimized-weight ensemble on identical folds.
- [ ] **T75.3** Run the three-way comparison: default parameters, tuned parameters, tuned plus optimized ensemble.
- [ ] **T75.4** Isolate the contribution of each optimization stage as an incremental delta.
- [ ] **T75.5** Emit **T19** the search and optimization ablation table.
- [ ] **T75.6** Emit **G23** and **G24**, the ensemble-weight and baseline-versus-optimized charts.
- [ ] **T75.7** [TEST] Confirm the optimization ablation isolates each stage as an incremental delta on identical folds, and that T19, G23 and G24 exist.

### Phase 76 — Optional Extended Multiclass Track
- [ ] **T76.1** Run the PhysioNet diagnosis-class track — **confirmed in scope** (CAD 287, MVP 134, Benign 118, Pathologic 62, MPC 23, AD 17, MR 12, AS 12). It is the only credible answer to Objective 6, since PASCAL A has just 19 samples in one of its four classes.
- [ ] **T76.2** If run, define a class-merge policy for the very small categories and document it explicitly.
- [ ] **T76.3** Configure subject-grouped CV and heavy class weighting for the long tail.
- [ ] **T76.4** Run the mandatory model set and report macro-F1 with per-class recall.
- [ ] **T76.5** Present this as supplementary evidence for Objective 6 on a far larger sample than PASCAL.
- [ ] **T76.6** State the small-sample limitation explicitly and report confidence intervals on every per-class metric.
- [ ] **T76.7** [TEST] Confirm the diagnosis track ran, the class-merge policy is documented, and per-class confidence intervals are reported.

### Phase 77 — 🔴 MEGA TEST 3 — Experiment & Result Integrity
> Covers Parts VI-VII (Phases 53-75). Nothing in Part VIII starts until every item here passes.
- [ ] **T77.1** [TEST] Re-run EXP-A1 fold 0 and assert it reproduces the stored metrics exactly — this is the seed-discipline proof for the whole project.
- [ ] **T77.2** [TEST] Audit every search trial log against the fold maps; assert no trial was ever scored on an outer test fold.
- [ ] **T77.3** [TEST] Scan every experiment output for any metric at or above 0.99 and investigate each one against the four causes in the standing rule; document the finding for each.
- [ ] **T77.4** [TEST] Assert each experiment's target vocabulary matches exactly one of the five declared label spaces — no merged or cross-contaminated targets anywhere.
- [ ] **T77.5** [TEST] Confirm EXP-D1's output metadata carries the adult-to-paediatric population note, and that no retuning touched CirCor.
- [ ] **T77.6** [TEST/MANUAL] Review the PASCAL A per-class results against n=124 with 19 in one class; confirm confidence intervals are reported and no over-strong claim is made.
- [ ] **T77.7** [TEST] Full pytest suite green; commit and tag `mega-3-experiments`.

---

# PART VIII — ANALYSIS

### Phase 78 — Calibration Analysis
- [ ] **T78.1** Collect out-of-fold predicted probabilities for every model on every task.
- [ ] **T78.2** Compute Brier score and expected calibration error per model.
- [ ] **T78.3** Build reliability curves with configurable bin counts.
- [ ] **T78.4** Compare sigmoid against isotonic calibration for the SVM and record which is used in the final model.
- [ ] **T78.5** Emit **T23** the calibration and confidence summary.
- [ ] **T78.6** Emit **G33** the confidence histogram and **G34** the calibration curve.
- [ ] **T78.7** [TEST] Confirm Brier and ECE are computed for every model, that sigmoid versus isotonic was compared for the SVM, and that the chosen calibration is recorded.

### Phase 79 — Complexity Analysis
- [ ] **T79.1** Record training time per model per fold from the timing instrumentation.
- [ ] **T79.2** Measure single-record end-to-end inference time: load, preprocess, extract, predict, over repeated trials.
- [ ] **T79.3** Decompose inference time by stage to show where the cost actually sits.
- [ ] **T79.4** Record on-disk model size and peak memory during fit.
- [ ] **T79.5** Emit **T24** complexity analysis, **T25** training and inference time, **T26** model size and memory.
- [ ] **T79.6** Emit **G25** inference time versus performance, **G26** training time, **G27** model size.
- [ ] **T79.7** [TEST] Confirm training time, per-stage inference time, model size and peak memory are recorded for every model, and that T24 through T26 exist.

### Phase 80 — Failure Analysis
- [ ] **T80.1** Extract all false positives and false negatives for the final binary model.
- [ ] **T80.2** Categorize failures by duration band, noise flag, dataset subset and predicted confidence.
- [ ] **T80.3** Cross-reference failures against the PhysioNet diagnosis field to see which pathologies are missed.
- [ ] **T80.4** Extract the most-confused class pairs from every multiclass confusion matrix.
- [ ] **T80.5** Emit **T27** the false-positive and false-negative analysis and **G35** the failure distribution chart.
- [ ] **T80.6** Write a narrative failure report naming concrete example records.
- [ ] **T80.7** [TEST] Confirm every false positive and false negative is categorised, cross-referenced to the diagnosis field, and that named example records appear in the report.

### Phase 81 — Explainability
- [ ] **T81.1** Compute RF and GB impurity importances for the final models.
- [ ] **T81.2** Compute permutation importance on held-out folds — the primary reported measure.
- [ ] **T81.3** Compute SHAP values for the tree models if the package is available; otherwise record the skip reason.
- [ ] **T81.4** Aggregate importance up to the feature-family level to show which of the six families carries the signal.
- [ ] **T81.5** Emit **FE-11** `top_feature_importance.csv`, **G18** the importance plot and **G19** the top-20 chart.
- [ ] **T81.6** Implement per-sample explanation output for the dashboard: the top contributing features for one prediction.
- [ ] **T81.7** [TEST] Confirm permutation importance is the primary reported measure, that family-level aggregation exists, and that per-sample explanation works for one real record.

### Phase 82 — Statistical Validation
- [ ] **T82.1** Report fold-wise values with mean and standard deviation for every headline metric.
- [ ] **T82.2** Compute bootstrap 95 percent confidence intervals for ROC-AUC on the key binary results.
- [ ] **T82.3** Run McNemar tests on paired predictions over the same test set for each model pair.
- [ ] **T82.4** Run Wilcoxon signed-rank or paired t-tests on fold-level paired comparisons from the repeated 5x5 map (**n=25**, established in T20.2), choosing per a documented normality check. Plain 5-fold would be n=5, which cannot reach p<0.05 in some configurations regardless of effect size — that is why the primary track is repeated.
- [ ] **T82.5** Run the Friedman test across all models over the repeated folds from T82.4, with a Nemenyi post-hoc where warranted. State the number of paired observations alongside every test statistic.
- [ ] **T82.6** Compute effect sizes (Cohen d or rank-biserial) alongside every p-value.
- [ ] **T82.7** [TEST] Confirm fold-wise mean and SD, bootstrap CIs and the paired tests all ran; verify the normality check is recorded, and **assert that every reported p-value states its n** — a p-value from 5 folds must never appear without that caveat.

### Phase 83 — Statistical Reporting
- [ ] **T83.1** Emit `outputs/12_statistics/statistical_significance_matrix.csv` — the model-by-model p-value matrix.
- [ ] **T83.2** Emit `statistical_summary_table.docx` with the test used, statistic, p-value, effect size and interpretation.
- [ ] **T83.3** Emit **T28** the statistical significance comparison table.
- [ ] **T83.4** Apply and document a multiple-comparison correction (Holm or Benjamini-Hochberg).
- [ ] **T83.5** State plainly wherever the proposed ensemble is not significantly better than a simpler model, and wherever the test was underpowered rather than negative — those are different findings and must not be conflated.
- [ ] **T83.6** Register every statistics artifact in the evidence index.
- [ ] **T83.7** [TEST] Confirm the significance matrix and summary DOCX exist, multiple-comparison correction is applied, and any non-significant ensemble advantage is stated plainly.

### Phase 84 — Cardiac Cycle & Segmentation Analysis
- [ ] **T84.1** Build a cycle-level table from the CirCor TSV segmentation (T15.2): per-recording cycle count, mean S1 / systole / S2 / diastole duration, and systole-to-diastole ratio.
- [ ] **T84.2** Build the same table for PASCAL A from `set_a_timing.csv` (T13.5), and record that no equivalent annotation exists for PASCAL B or PhysioNet.
- [ ] **T84.3** Test whether segmentation quality (annotated fraction) correlates with prediction confidence and with the failure cases from Phase 76.
- [ ] **T84.4** Compare cycle timing between classes descriptively — murmur present versus absent, normal versus abnormal — and state plainly that these are not among the 138 features, so this is context, not a result.
- [ ] **T84.5** Emit `outputs/10_robustness/segmentation_cycle_statistics.csv` plus a segmentation-coverage summary.
- [ ] **T84.6** Emit an annotated-waveform figure overlaying S1 / systole / S2 / diastole on a real recording — this is the asset the dashboard cardiac-cycle viewer consumes.
- [ ] **T84.7** [TEST] Confirm the cycle table covers all segmented CirCor recordings, that PASCAL A timing is included, and that the annotated-waveform figure renders real segmentation.

---

# PART IX — RESULT ASSET GENERATION

### Phase 85 — Table Engine
- [ ] **T85.1** Write `src/reporting/tables.py` — a builder that reads result CSVs and emits formatted tables.
- [ ] **T85.2** Implement a CSV writer preserving full numeric precision as the source of truth.
- [ ] **T85.3** Implement a DOCX writer via python-docx with publication styling and a caption.
- [ ] **T85.4** Implement a LaTeX writer for direct paper insertion.
- [ ] **T85.5** Enforce that every table records the experiment id and source file it was built from.
- [ ] **T85.6** Add rounding rules (3 decimals for metrics, 1 for percentages) applied consistently everywhere.
- [ ] **T85.7** [TEST] Generate one table through all four writers (CSV, DOCX, LaTeX) and confirm the numbers are identical and the rounding rules applied consistently.

### Phase 86 — Tables T01-T07 (Setup)
- [ ] **T86.1** Generate **T01** Dataset Inventory from DA-01.
- [ ] **T86.2** Generate **T02** Class Distribution and Imbalance Ratio from DA-02.
- [ ] **T86.3** Generate **T03** Recording Duration and Sampling Summary from DA-03 and DA-04.
- [ ] **T86.4** Generate **T04** Preprocessing Configuration from PP-07.
- [ ] **T86.5** Generate **T05** Feature Inventory and Counts from FE-01 and FE-02.
- [ ] **T86.6** Generate **T06** Model Hyperparameter Configuration and **T07** Search Space and Best Parameters.
- [ ] **T86.7** [TEST] Confirm T01 through T07 exist in all required formats and that each records the source file it was built from.

### Phase 87 — Tables T08-T15 (Core Results)
- [ ] **T87.1** Generate **T08** PhysioNet Individual Model Comparison.
- [ ] **T87.2** Generate **T09** Equal-Weight versus Optimized Ensemble.
- [ ] **T87.3** Generate **T10** PhysioNet Fold-wise Results as mean plus or minus SD.
- [ ] **T87.4** Generate **T11** PASCAL A and **T12** PASCAL B multiclass results.
- [ ] **T87.5** Generate **T13** CirCor Murmur and **T14** CirCor Clinical Outcome results.
- [ ] **T87.6** Generate **T15** Recording-Level versus Subject-Level results.
- [ ] **T87.7** [TEST] Confirm T08 through T15 exist and that each metric matches the experiment output CSV it came from.

### Phase 88 — Tables T16-T23 (Ablation, Robustness, Calibration)
- [ ] **T88.1** Generate **T16** Cross-Dataset Generalization.
- [ ] **T88.2** Generate **T17** Feature-Family Ablation and **T18** All Features versus Selected Features.
- [ ] **T88.3** Generate **T19** Search and Optimization Ablation.
- [ ] **T88.4** Generate **T20** Noise Robustness and **T21** Duration Robustness.
- [ ] **T88.5** Generate **T22** Auscultation Location Analysis.
- [ ] **T88.6** Generate **T23** Calibration and Confidence Summary.
- [ ] **T88.7** [TEST] Confirm T16 through T23 exist and that ablation rows reconcile with the EXP-F1 and EXP-F2 outputs.

### Phase 89 — Tables T24-T30 (Complexity, Stats, Conclusion)
- [ ] **T89.1** Generate **T24** Complexity Analysis and **T25** Training and Inference Time.
- [ ] **T89.2** Generate **T26** Model Size and Memory.
- [ ] **T89.3** Generate **T27** False Positive and False Negative Analysis.
- [ ] **T89.4** Generate **T28** Statistical Significance Comparison.
- [ ] **T89.5** Generate **T29** Objective-to-Evidence Mapping — one row per locked objective with module, dataset, evidence file and status.
- [ ] **T89.6** Generate **T30** Final Conclusion Matrix.
- [ ] **T89.7** [TEST] Confirm T24 through T30 exist, and that T29's objective-to-evidence mapping has a real file path for all six objectives.

### Phase 90 — Graph Engine
- [ ] **T90.1** Write `src/reporting/graphs.py` with a shared 300 dpi style, colourblind-safe palette and consistent fonts.
- [ ] **T90.2** Enforce that every graph writes both a PNG and the exact source CSV that produced it.
- [ ] **T90.3** Add a caption and figure-number registry so numbering stays stable across regenerations.
- [ ] **T90.4** Add an SVG export path for the diagrams that need to stay editable.
- [ ] **T90.5** Add a light-background print profile suited to thesis printing.
- [ ] **T90.6** Write `tests/test_graph_engine.py` asserting every generator produces a non-empty PNG plus CSV.
- [ ] **T90.7** [TEST] Run `tests/test_graph_engine.py`; confirm every generator emits a non-empty PNG **and** its source CSV, and that the caption registry keeps numbering stable across a regeneration.

### Phase 91 — Graphs G01-G10 (Data and Signal)
- [ ] **T91.1** **G01** dataset-wise recording count bar chart.
- [ ] **T91.2** **G02** class-distribution bar chart and **G03** class-distribution pie chart.
- [ ] **T91.3** **G04** recording-duration histogram.
- [ ] **T91.4** **G05** before and after filtering waveform.
- [ ] **T91.5** **G06** normal versus abnormal spectrogram comparison.
- [ ] **T91.6** **G07** MFCC heatmap, **G08** chroma heatmap, **G09** wavelet decomposition, **G10** feature-family count chart.
- [ ] **T91.7** [TEST] Confirm G01 through G10 exist with source CSVs, and that G02's class counts match DA-02 exactly.

### Phase 92 — Graphs G11-G19 (Model Performance)
- [ ] **T92.1** **G11** individual model comparison bar chart with error bars across folds.
- [ ] **T92.2** **G12** binary ROC curve overlaying every model with AUC in the legend.
- [ ] **T92.3** **G13** binary precision-recall curve.
- [ ] **T92.4** **G14** binary confusion matrix heatmap, raw and normalized.
- [ ] **T92.5** **G15** PASCAL A and **G16** PASCAL B multiclass confusion matrices; **G17** multiclass one-vs-rest ROC.
- [ ] **T92.6** **G18** feature importance plot and **G19** top-20 features chart.
- [ ] **T92.7** [TEST] Confirm G11 through G19 exist, that ROC AUC values in the legend match T08, and that confusion matrices use the fixed class ordering.

### Phase 93 — Graphs G20-G28 (Optimization and Complexity)
- [ ] **T93.1** **G20** search convergence plot across all search methods.
- [ ] **T93.2** **G21** all-features versus selected-features chart.
- [ ] **T93.3** **G22** F1 and accuracy versus feature count.
- [ ] **T93.4** **G23** equal-weight versus optimized-weight ensemble; **G24** baseline versus optimized model.
- [ ] **T93.5** **G25** inference time versus performance scatter; **G26** training-time comparison; **G27** model-size comparison.
- [ ] **T93.6** **G28** dataset-wise generalization chart.
- [ ] **T93.7** [TEST] Confirm G20 through G28 exist and that the feature-count curve matches the T57.4 sweep data.

### Phase 94 — Graphs G29-G35 (Robustness and Reliability)
- [ ] **T94.1** **G29** cross-dataset performance-drop chart.
- [ ] **T94.2** **G30** noise-level robustness chart.
- [ ] **T94.3** **G31** duration-wise performance chart.
- [ ] **T94.4** **G32** auscultation-location performance chart.
- [ ] **T94.5** **G33** confidence histogram and **G34** calibration curve.
- [ ] **T94.6** **G35** false-positive and false-negative distribution chart.
- [ ] **T94.7** [TEST] Confirm G29 through G35 exist and that the robustness charts reconcile with T20, T21 and T22.

### Phase 95 — Figures F01-F10 (Architecture Diagrams)
- [ ] **T95.1** **F01** overall PV-MEPCG proposed architecture.
- [ ] **T95.2** **F02** end-to-end PCG classification workflow (the 12 documented steps).
- [ ] **T95.3** **F03** three-dataset experimental-track diagram and **F04** dataset harmonization and metadata workflow.
- [ ] **T95.4** **F05** subject-wise data-splitting diagram and **F06** signal preprocessing architecture.
- [ ] **T95.5** **F07** 138-feature extraction architecture and **F08** feature-family fusion diagram.
- [ ] **T95.6** **F09** search-based feature-selection workflow and **F10** SVM-RF-GB optimized soft-voting architecture.
- [ ] **T95.7** [TEST/MANUAL] Confirm F01 through F10 render as both SVG and 300 dpi PNG and are legible at thesis width in greyscale.

### Phase 96 — Figures F11-F20 (Pipeline and Contribution Diagrams)
- [ ] **T96.1** **F11** binary classification pipeline and **F12** PASCAL multiclass pipeline.
- [ ] **T96.2** **F13** CirCor recording-to-subject aggregation flow.
- [ ] **T96.3** **F14** cross-dataset external-validation workflow and **F15** noise and duration robustness framework.
- [ ] **T96.4** **F16** dashboard system architecture — must show the build-time codegen boundary (`outputs/` to `generated/`) and the single live `/predict` endpoint — and **F17** prediction and report-generation flow.
- [ ] **T96.5** **F18** objective-to-module traceability diagram.
- [ ] **T96.6** **F19** novelty contribution diagram and **F20** final research-contribution summary.
- [ ] **T96.7** [TEST/MANUAL] Confirm F11 through F20 render in both formats and that all 20 diagrams share one visual language.

### Phase 97 — Diagram Production Pipeline
- [ ] **T97.1** Choose the diagram toolchain (Graphviz or Mermaid rendered to SVG) and pin it in requirements.
- [ ] **T97.2** Keep every diagram as source text in `src/reporting/diagrams/` so it is version-controlled and regenerable.
- [ ] **T97.3** Render each diagram to both SVG (editable) and 300 dpi PNG.
- [ ] **T97.4** Apply a single consistent visual language across all 20 diagrams: shapes, colours, arrow semantics.
- [ ] **T97.5** Verify every diagram is legible at printed thesis width in greyscale.
- [ ] **T97.6** Write `scripts/05_render_diagrams.py` regenerating all 20 in one command.
- [ ] **T97.7** [TEST] Run `scripts/05_render_diagrams.py` from clean; confirm all 20 regenerate from version-controlled source with no manual step.

### Phase 98 — Algorithms ALG-01 to ALG-10
- [ ] **T98.1** **ALG-01** dataset loading and master-metadata construction; **ALG-02** signal harmonization and resampling.
- [ ] **T98.2** **ALG-03** Butterworth filtering and z-score normalization; **ALG-04** multi-domain 138-feature extraction.
- [ ] **T98.3** **ALG-05** fold-safe scaling and preprocessing; **ALG-06** search-based feature selection.
- [ ] **T98.4** **ALG-07** hyperparameter optimization; **ALG-08** SVM training and probability calibration.
- [ ] **T98.5** **ALG-09** Random Forest training; **ALG-10** Gradient Boosting training.
- [ ] **T98.6** Verify every pseudocode block matches the actual implementation line for line in behaviour.
- [ ] **T98.7** [TEST] Confirm ALG-01 through ALG-10 exist in DOCX and plain text, and spot-check three against the implementation for behavioural agreement.

### Phase 99 — Algorithms ALG-11 to ALG-20
- [ ] **T99.1** **ALG-11** equal-weight soft voting; **ALG-12** optimized-weight soft voting.
- [ ] **T99.2** **ALG-13** binary evaluation workflow; **ALG-14** multiclass evaluation workflow.
- [ ] **T99.3** **ALG-15** CirCor subject-level aggregation; **ALG-16** cross-dataset external validation.
- [ ] **T99.4** **ALG-17** noise and duration robustness analysis; **ALG-18** failure-case analysis.
- [ ] **T99.5** **ALG-19** sample-level report generation; **ALG-20** objective-achievement evidence generation.
- [ ] **T99.6** Export all 20 to `outputs/14_algorithms/` in both DOCX and plain-text form.
- [ ] **T99.7** [TEST] Confirm ALG-11 through ALG-20 exist in both formats and that the soft-voting pseudocode matches `src/ensemble/soft_voting.py`.

### Phase 100 — Equations Reference
- [ ] **T100.1** Document every equation from blueprint section 11 with symbol definitions.
- [ ] **T100.2** Verify each implemented formula numerically against its documented definition with a unit test.
- [ ] **T100.3** Export the equations as LaTeX for the paper.
- [ ] **T100.4** Export the equations as DOCX with the equation editor format for the thesis.
- [ ] **T100.5** Cross-reference each equation to the source file and line that implements it.
- [ ] **T100.6** Emit `outputs/14_algorithms/equations_reference.docx`.
- [ ] **T100.7** [TEST] Verify each implemented formula numerically against its documented definition, and confirm every equation cross-references the file and line implementing it.

### Phase 101 — Literature Review & State of the Art (Objective 2)
- [ ] **T101.1** Define the literature-table schema: author, year, dataset, preprocessing, feature families, classifier, validation scheme, reported metric, sample size, stated limitation.
- [ ] **T101.2** Seed the table from the synopsis reference list — Muruganantham 2003, Shui 2004, Segaier 2005, Jiang & Choi 2006, Ahlstrom 2006, Nigam 2008, Babaei & Geranmayeh 2009, Ari 2010 (wavelet LSSVM), Choi 2010 (NAR-PSD SVM), Rangayyan & Lehner.
- [ ] **T101.3** Add the PhysioNet/CinC 2016 Challenge leading entries and the CirCor 2022 Challenge leading entries as the benchmark comparison rows.
- [ ] **T101.4** Add recent work on PCG ensembles, feature selection, and deep learning, so the review covers the state of the art and not only the historical background.
- [ ] **T101.5** Add a positioning column stating, per row, how PV-MEPCG differs — multi-dataset, search-optimized subset, optimized-weight ensemble, external validation.
- [ ] **T101.6** Emit the table as CSV and DOCX, and compare our PhysioNet numbers against published results with an explicit caveat that split protocols differ and the comparison is indicative, not head-to-head.
- [ ] **T101.7** [TEST] Confirm the literature table covers the synopsis references plus challenge benchmarks, that every row has a positioning entry, and that the comparison caveat about differing splits is present.

### Phase 102 — Evidence Index
- [ ] **T102.1** Assemble every registered artifact into `outputs/00_evidence_index/evidence_index.xlsx`.
- [ ] **T102.2** Populate all required fields: evidence_id, objective, experiment_id, dataset, model, metric_or_asset, filename, source_data, command, timestamp, status.
- [ ] **T102.3** Verify every referenced file actually exists on disk; mark any missing entry as failed.
- [ ] **T102.4** Verify every mandatory item from both source documents appears in the index.
- [ ] **T102.5** Finalize `run_manifest.json` with the full environment, seeds and per-phase timing.
- [ ] **T102.6** Generate `missing_outputs_report.txt` listing every unproduced item with its exact technical reason.
- [ ] **T102.7** [TEST] Confirm every evidence-index row resolves to a real file on disk, and that every mandatory item from both source documents appears in the index.

### Phase 103 — Q1 / IEEE Paper Asset Pack
- [ ] **T103.1** Create `outputs/Q1_PAPER_ASSETS/` and generate Q1_T01 dataset summary, Q1_T02 feature summary, Q1_T03 model and search configuration.
- [ ] **T103.2** Generate Q1_T04 individual versus ensemble, Q1_T05 feature ablation, Q1_T06 optimization ablation.
- [ ] **T103.3** Generate Q1_T07 multiclass results, Q1_T08 external validation, Q1_T09 complexity and robustness.
- [ ] **T103.4** Generate Q1_F01 proposed architecture and Q1_F02 feature and ensemble workflow.
- [ ] **T103.5** Generate Q1_G01 combined ROC and PR, Q1_G02 binary confusion matrix, Q1_G03 multiclass confusion matrix, Q1_G04 ablation and optimization, Q1_G05 generalization and complexity.
- [ ] **T103.6** Generate Q1_ALG01 to Q1_ALG03 and `Q1_results_narrative.docx` written strictly from generated numbers.
- [ ] **T103.7** [TEST] Confirm every Q1 asset exists and that every number in `Q1_results_narrative.docx` traces to a generated file, not prose.

### Phase 104 — Thesis Asset Pack
- [ ] **T104.1** Create `outputs/THESIS_ASSETS/` with chapter folders Ch1_Introduction through Ch6_Conclusion.
- [ ] **T104.2** Place all 30 tables in both editable DOCX and CSV form into the correct chapter folders.
- [ ] **T104.3** Place all 35 graphs as 300 dpi PNG plus source CSV.
- [ ] **T104.4** Place all 20 diagrams in editable SVG and PNG form.
- [ ] **T104.5** Place all 20 algorithms in DOCX and plain-text form, plus the equations reference.
- [ ] **T104.6** Assemble the reproducibility appendix, the objective achievement matrix and the final conclusion matrix.
- [ ] **T104.7** [TEST] Confirm every chapter folder is populated and that the counts reconcile: 30 tables, 35 graphs, 20 diagrams, 20 algorithms.

### Phase 105 — 🔴 MEGA TEST 4 — Deliverable Completeness
> Covers Parts VIII-IX (Phases 77-103). Nothing in Part X starts until every item here passes.
- [ ] **T105.1** [TEST] Assert all 30 tables exist, are non-empty, and each names the source file it was built from.
- [ ] **T105.2** [TEST] Assert all 35 graphs exist as 300 dpi PNG **and** source CSV, and that no CSV is empty or all-NaN.
- [ ] **T105.3** [TEST] Assert 20 diagrams exist as SVG and PNG, 20 algorithms as DOCX and plain text, and the equations reference is present.
- [ ] **T105.4** [TEST] Assert every evidence-index row resolves to a real file, and spot-check five `command` fields by actually re-running them.
- [ ] **T105.5** [TEST] Cross-check every number in the Q1 asset pack against its source CSV; any mismatch fails this phase.
- [ ] **T105.6** [TEST] Confirm `missing_outputs_report.txt` accounts for every item not produced, each with a technical reason rather than an omission.
- [ ] **T105.7** [TEST/MANUAL] Open five tables and five graphs at random and confirm they are readable, correctly labelled and self-consistent; commit and tag `mega-4-assets`.

---

# PART X — INFERENCE SERVICE & DASHBOARD (starts only after Part IX)

**Architecture rule that governs this entire part:** every precomputed number reaches the browser through **build-time codegen** (Python reads `outputs/`, writes `frontend/lib/generated/`), never through a runtime fetch and never as a hand-typed literal. Only live inference uses a real API call. See the frontend-architecture entry in `note.md` for why.

### Phase 106 — Inference Engine
- [ ] **T106.1** Write `src/inference/predictor.py` — load a saved pipeline plus its feature list and predict from a raw WAV path.
- [ ] **T106.2** Reuse the exact preprocessing and extraction code path used in training; no reimplementation.
- [ ] **T106.3** Return a structured result: predicted class, class probabilities, confidence, low-confidence flag, per-stage timing.
- [ ] **T106.4** Implement task routing so one entry point serves binary, PASCAL A, PASCAL B, murmur and outcome models.
- [ ] **T106.5** Add input validation: file format, duration bounds, sample-rate handling, and clear errors for unusable audio.
- [ ] **T106.6** Write `tests/test_inference.py` asserting a known training record reproduces its stored out-of-fold prediction.
- [ ] **T106.7** [TEST] Run `tests/test_inference.py` — a known training record reproduces its stored out-of-fold prediction through the public entry point.

### Phase 107 — Report Generation
- [ ] **T107.1** Write `src/reporting/sample_report.py` producing a per-recording PDF or DOCX report.
- [ ] **T107.2** Include the waveform, spectrogram, prediction, confidence and top contributing features.
- [ ] **T107.3** Stamp every report with the screening-only disclaimer and the model version.
- [ ] **T107.4** Write an experiment-level report generator summarizing one full run.
- [ ] **T107.5** Implement CSV and JSON export of batch predictions.
- [ ] **T107.6** Write `tests/test_report_generation.py`.
- [ ] **T107.7** [TEST] Run `tests/test_report_generation.py`; open one generated report and confirm the disclaimer and model version are stamped on it.

### Phase 108 — FastAPI Service
- [ ] **T108.1** Create `src/api/main.py` — FastAPI app, CORS for the dev origin only, lifespan hook that loads models once at startup.
- [ ] **T108.2** Define Pydantic response schemas for every endpoint, with explicit `float | None` on any field that can be NaN.
- [ ] **T108.3** Write a central JSON encoder converting numpy scalars and coercing NaN/Inf to `null` — applied app-wide, never per endpoint.
- [ ] **T108.4** Implement `POST /predict` — multipart WAV upload, task selector, returns the T106.3 structure.
- [ ] **T108.5** Implement `GET /health` (model load status, versions) and `GET /manifest` (run id, git commit, generation timestamp).
- [ ] **T108.6** Mount `frontend/out/` as static files so production is a single process on one port; write `tests/test_api.py` covering upload, bad input and the NaN path.
- [ ] **T108.7** [TEST] Run `tests/test_api.py` — upload succeeds, a bad file returns a clean error, and a record producing NaN features serialises as `null` rather than breaking JSON.

### Phase 109 — Frontend Data Exporter (the correctness boundary)
- [ ] **T109.1** Write `scripts/06_export_frontend_data.py` — read every table and graph source CSV from `outputs/`, emit JSON into `frontend/lib/generated/`.
- [ ] **T109.2** Apply all number formatting here, in Python, using the same rounding rules as the thesis tables (T85.6). The frontend never rounds.
- [ ] **T109.3** Coerce NaN and Inf to `null` at export; assert every emitted file parses as strict JSON.
- [ ] **T109.4** Emit `generated/manifest.json` carrying the run id, git commit, source-file list and export timestamp, and surface it in the UI footer.
- [ ] **T109.5** Emit a `generated/evidence.json` index mapping every displayed value back to the CSV it came from, for the evidence browser.
- [ ] **T109.6** Emit TypeScript type declarations alongside the JSON so a schema change breaks the build rather than the page.
- [ ] **T109.7** [TEST] Run the exporter and assert every emitted file parses as **strict** JSON, contains no NaN token, and that the manifest matches the run that produced the artifacts.

### Phase 110 — Next.js Scaffold
- [ ] **T110.1** Scaffold `frontend/` — Next.js 14 App Router, React, TypeScript, `output: 'export'` for a fully static build.
- [ ] **T110.2** Configure Tailwind, PostCSS, path aliases and `next-themes` for light/dark.
- [ ] **T110.3** Build the root layout: navbar, footer with the run manifest, and the screening-only disclaimer rendered **in the layout**, not per page.
- [ ] **T110.4** Define the route tree for all 11 document pages plus home.
- [ ] **T110.5** Wire `npm run build` to run the T102 exporter first, so the site can never build against stale data.
- [ ] **T110.6** Add ESLint and TypeScript strict mode; confirm a clean production build.
- [ ] **T110.7** [TEST] Run `npm run build`; confirm the exporter ran first, the static export completes, and TypeScript strict mode passes with no errors.

### Phase 111 — Design System
- [ ] **T111.1** Define the design tokens — colour, type scale, spacing — matching the matplotlib figure palette so charts and pages agree.
- [ ] **T111.2** Build the primitives: GlassCard, SectionHeader, StatTile, AnimatedCounter, Badge, Tabs, Tooltip (Radix underneath).
- [ ] **T111.3** Build the file-upload component with drag-and-drop, format validation and progress state.
- [ ] **T111.4** Build the loading, empty and **error** states — a failed request must render a visible error, never an empty chart.
- [ ] **T111.5** Verify colour contrast in both themes and confirm the palette is colourblind-safe.
- [ ] **T111.6** Build a `/design` reference page rendering every component in every state, for visual QA.
- [ ] **T111.7** [TEST/MANUAL] Open `/design` and confirm every component renders in every state, contrast passes in both themes, and the error state is visibly an error.

### Phase 112 — 3D and Motion Layer
- [ ] **T112.1** Set up Three.js + React Three Fiber + drei with a lazy-loaded, SSR-safe wrapper.
- [ ] **T112.2** Build the animated heart model for the home hero, with a reduced-motion fallback.
- [ ] **T112.3** Build the scroll-driven pipeline walkthrough animating the 12 documented architecture steps (GSAP + Lenis).
- [ ] **T112.4** Build the ensemble visualization showing SVM, RF and GB probabilities fusing into the weighted vote.
- [ ] **T112.5** Add Framer Motion page transitions and card reveals.
- [ ] **T112.6** Budget-check bundle size and frame rate on integrated graphics; degrade gracefully where it fails.
- [ ] **T112.7** [TEST] Confirm the 3D and motion layer loads without SSR errors, respects reduced-motion, and that the bundle-size budget is met on integrated graphics.

### Phase 113 — Chart, Table and Equation Components
- [ ] **T113.1** Build the chart wrapper over Plotly/ECharts consuming only `generated/` data, with a shared theme.
- [ ] **T113.2** Build the specific chart types needed: ROC, PR, confusion-matrix heatmap, grouped bars, scatter, calibration curve.
- [ ] **T113.3** Build the results table component (TanStack Table) with sort, filter and CSV download.
- [ ] **T113.4** Attach a "download the 300 dpi figure" action beside every chart, serving the canonical matplotlib PNG.
- [ ] **T113.5** Build the KaTeX equation component and render all 15 equations from blueprint section 11.
- [ ] **T113.6** Build the waveform and spectrogram viewer (WaveSurfer.js) with playback, plus a cardiac-cycle overlay marking S1 / systole / S2 / diastole from the **real** CirCor TSV segmentation (T15.2) — not a synthetic animation.
- [ ] **T113.7** [TEST] Confirm every chart consumes only `generated/` data, that the cardiac-cycle overlay uses real CirCor segmentation, and that all 15 equations render.

### Phase 114 — Pages 1-3
- [ ] **T114.1** **Home** — hero with the 3D heart, the six locked objectives verbatim, the animated pipeline, disclaimer, dataset summary tiles.
- [ ] **T114.2** **Dataset Overview** — inventory table, class distribution, duration histograms, split summary, all from `generated/`.
- [ ] **T114.3** Add dataset filtering and drill-down to record level.
- [ ] **T114.4** **Signal Preprocessing** — record picker, raw versus filtered waveform, spectrogram, quality indicators.
- [ ] **T114.5** Add the interactive filter and normalization toggles driven by precomputed example pairs.
- [ ] **T114.6** Verify every number on these three pages against its source CSV.
- [ ] **T114.7** [TEST] Cross-check every number displayed on pages 1-3 against its source CSV.

### Phase 115 — Pages 4-6
- [ ] **T115.1** **Feature Extraction** — the 138-feature inventory, family composition chart, per-record feature vector view.
- [ ] **T115.2** Add the selected-subset view showing which features survived optimization, with the equations rendered inline.
- [ ] **T115.3** **Model Comparison** — sortable metric table across all models, with error bars from the fold-wise results.
- [ ] **T115.4** Add the ROC, PR and confusion-matrix viewers driven by a model selector.
- [ ] **T115.5** **Search Optimization** — convergence plot, search space, selected parameters, selected features, Pareto front.
- [ ] **T115.6** Verify every number on these three pages against its source CSV.
- [ ] **T115.7** [TEST] Cross-check every number displayed on pages 4-6 against its source CSV.

### Phase 116 — Pages 7-9 (Prediction)
- [ ] **T116.1** **Binary Prediction** — WAV upload to `POST /predict`, waveform preview, normal/abnormal result with confidence.
- [ ] **T116.2** Render the low-confidence warning and the screening-only banner on every result.
- [ ] **T116.3** **Multiclass Prediction** — PASCAL A and B output with a full probability bar chart.
- [ ] **T116.4** **Murmur / Outcome Analysis** — CirCor recording-level and patient-level output with the per-location breakdown.
- [ ] **T116.5** Add batch upload with a results table and CSV export.
- [ ] **T116.6** Ship built-in sample recordings so every prediction page demonstrates without an upload.
- [ ] **T116.7** [TEST] Upload a known record through the UI and confirm the prediction matches the stored out-of-fold prediction for that record.

### Phase 117 — Pages 10-12
- [ ] **T117.1** **Robustness Analytics** — noise, duration, dataset and location-wise results with interactive filters.
- [ ] **T117.2** **Explainability** — global feature importance, family-level contribution, and per-sample explanation for the last prediction.
- [ ] **T117.3** **Reports** — trigger and download the sample report, experiment report and objective-coverage report.
- [ ] **T117.4** Build the **evidence browser** — every displayed metric links to the CSV it came from, driven by `generated/evidence.json`.
- [ ] **T117.5** Add a limitations page stating the population mismatch in EXP-D1, the PASCAL sample sizes and the CirCor public-subset caveat.
- [ ] **T117.6** Verify every number on these pages against its source CSV.
- [ ] **T117.7** [TEST] Cross-check pages 10-12, and confirm the evidence browser resolves at least one link per page to a real CSV.

### Phase 118 — Frontend Testing
- [ ] **T118.1** Set up Vitest with React Testing Library; unit-test the chart, table and upload components.
- [ ] **T118.2** Set up Playwright; smoke-test that all 12 routes render without a console error.
- [ ] **T118.3** Assert the screening-only disclaimer is present on every route.
- [ ] **T118.4** Test upload handling for corrupt, empty, wrong-format and very long audio.
- [ ] **T118.5** Test the error states — with the API stopped, every prediction page shows a visible error, never an empty chart.
- [ ] **T118.6** Wire the whole suite into `scripts/run_tests.ps1`.
- [ ] **T118.7** [TEST] Run the full Vitest and Playwright suites; all 12 routes render with no console error and the disclaimer is present on every one.

### Phase 119 — The Correctness Guard Rail
- [ ] **T119.1** Write `scripts/07_check_no_hardcoded_metrics.py` — scan `frontend/app/` and `frontend/components/` for numeric metric literals and **fail the build** on any hit.
- [ ] **T119.2** Enforce the rule that pages import from `frontend/lib/generated/` only; flag any other data source.
- [ ] **T119.3** Write the displayed-value audit: crawl the built site, extract every rendered metric, diff against the source CSVs, fail on mismatch.
- [ ] **T119.4** Run the audit as a **hard gate before screenshots are taken** — no screenshot of an unaudited page.
- [ ] **T119.5** Verify the manifest in the footer matches the run that produced the artifacts.
- [ ] **T119.6** Document the guard rail in the README as the project's answer to the section 19 dashboard QA requirement.
- [ ] **T119.7** [TEST] Deliberately insert a fake metric literal into a page and confirm the build **fails**; revert it and confirm the build passes and the displayed-value audit is green.

### Phase 120 — Dashboard Screenshots
- [ ] **T120.1** Capture screenshots 1-3: home, dataset inventory and class distribution, signal upload and waveform.
- [ ] **T120.2** Capture screenshots 4-6: preprocessing before and after, feature extraction summary, model comparison.
- [ ] **T120.3** Capture screenshots 7-9: search optimization, binary prediction output, multiclass prediction output.
- [ ] **T120.4** Capture screenshots 10-12: CirCor murmur and outcome output, robustness analytics, explainability.
- [ ] **T120.5** Capture screenshot 13: exported report preview.
- [ ] **T120.6** Save all 13 with captions to `outputs/15_dashboard_screenshots/` and register them in the evidence index.
- [ ] **T120.7** [TEST/MANUAL] Review all 13 screenshots — correct page, readable, disclaimer visible, and no placeholder or empty chart in any of them.

### Phase 121 — 🔴 MEGA TEST 5 — Full System
> Covers Part X (Phases 105-119) and everything it depends on. Nothing in Part XI starts until every item here passes.
- [ ] **T121.1** [TEST] Clean build from an empty `outputs/`: full pipeline, exporter, frontend build, guard rail — one command, no manual intervention.
- [ ] **T121.2** [TEST] Insert a fake metric literal into a page and confirm the build fails; revert and confirm it passes. The guard rail is proven, not assumed.
- [ ] **T121.3** [TEST] Run the displayed-value audit across all 12 pages; every rendered metric matches its source CSV.
- [ ] **T121.4** [TEST] Upload a known training record through the UI and confirm the prediction matches that record's stored out-of-fold prediction.
- [ ] **T121.5** [TEST] Upload a degenerate recording that produces NaN features; confirm the API returns `null` and the page renders an explanation rather than crashing or showing an empty chart.
- [ ] **T121.6** [TEST/MANUAL] Click through all 12 pages with the API stopped, then running; confirm the disclaimer is on every page, no console errors, and no chart is silently empty.
- [ ] **T121.7** [TEST] Full Python and frontend suites green; commit and tag `mega-5-full-system`.

---

# PART XI — PACKAGING & DELIVERY

### Phase 122 — Reproducibility
- [ ] **T122.1** Write `scripts/00_run_everything.py` executing the full pipeline from raw data to final assets in one command.
- [ ] **T122.2** Extend it with the frontend chain: export data, `npm ci`, `npm run build`, run the guard rail.
- [ ] **T122.3** Verify a clean end-to-end run from an empty `outputs/` and record the total wall time.
- [ ] **T122.4** Verify a second run reproduces identical metrics, confirming the seed discipline holds.
- [ ] **T122.5** Commit `frontend/out/` so a grader without Node can still serve the dashboard from FastAPI alone.
- [ ] **T122.6** Freeze the environment: regenerate `pip_freeze.txt`, commit `package-lock.json`, write the reproducibility appendix.
- [ ] **T122.7** [TEST] Run `scripts/00_run_everything.py` from an empty `outputs/`; confirm it completes, and that a second run reproduces identical metrics.

### Phase 123 — Documentation
- [ ] **T123.1** Write the full `README.md`: overview, scope boundary, install (Python and Node), dataset placement, run instructions, output map.
- [ ] **T123.2** Document the project structure and the role of each module, including the codegen boundary.
- [ ] **T123.3** Document every configuration option and its default.
- [ ] **T123.4** Write the troubleshooting section covering the known dataset quirks (CirCor doubled directory, set_b filename mismatch, Outcome only in txt files, `Heartbeat_Sound` duplication).
- [ ] **T123.5** Write the limitations section: CirCor public subset only, partial PhysioNet subject IDs, no PASCAL A subject IDs, small PASCAL samples, the EXP-D1 population mismatch, no clinical validation.
- [ ] **T123.6** Write `CITATION.md` crediting all three dataset sources with their licences.
- [ ] **T123.7** [TEST/MANUAL] Follow the README from scratch on the described steps and confirm every instruction works as written.

### Phase 124 — Final QA Sweep
- [ ] **T124.1** Dataset QA — re-verify file integrity, duplicates, label mapping, class counts and sampling rates.
- [ ] **T124.2** Split QA — re-verify zero patient leakage across every fold of every task.
- [ ] **T124.3** Feature QA — re-verify all features finite and reproducible across folds.
- [ ] **T124.4** Model QA — re-verify fixed seeds, fold-safe scaling and proper calibration.
- [ ] **T124.5** Metric QA — confirm sensitivity, specificity, F1, AUC and macro metrics are reported everywhere, never accuracy alone.
- [ ] **T124.6** Search QA — confirm no search ever touched a final test fold; dashboard QA — confirm the T119.3 audit passes.
- [ ] **T124.7** [TEST] Confirm all six QA areas pass and that the T119.3 displayed-value audit is green at the time of the sweep.

### Phase 125 — Compliance and Claims Review
- [ ] **T125.1** Grep the codebase, the frontend and every report for diagnostic language and replace it with screening wording.
- [ ] **T125.2** Confirm the disclaimer appears in the README, the dashboard layout, every generated report and the paper narrative.
- [ ] **T125.3** Review every reported metric for implausible perfection and re-verify any that looks too good.
- [ ] **T125.4** Confirm the six locked objectives appear verbatim wherever quoted, with no paraphrasing.
- [ ] **T125.5** Confirm every documented count matches audited reality, including the CirCor and PhysioNet discrepancies and the EXP-D1 population note.
- [ ] **T125.6** Confirm no result anywhere in the deliverables was hand-entered rather than generated.
- [ ] **T125.7** [TEST] Grep the whole deliverable for diagnostic language and for hand-entered numbers; confirm both return nothing.

### Phase 126 — Final Delivery
- [ ] **T126.1** Verify the completion checklist from document section 19 item by item.
- [ ] **T126.2** Confirm the counts: 30 tables, 35 graphs, 20 diagrams, 20 algorithms, 13 screenshots — or a documented reason for each gap.
- [ ] **T126.3** Package code, configs, saved models, logs, all source CSV and JSON, the frontend source and build, and all assets into the final ZIP.
- [ ] **T126.4** Verify the ZIP opens cleanly and the README instructions work from it on a machine without Node.
- [ ] **T126.5** Push the final state to GitHub and tag the release.
- [ ] **T126.6** Write the handover summary: what was produced, what was skipped and why, and what future clinical validation would require.
- [ ] **T126.7** [TEST] Verify the delivery ZIP opens cleanly on a machine without Node and that the README instructions work from inside it.

---

## Progress tracker

| Part | Phases | Tasks | Status |
|------|--------|-------|--------|
| I — Foundation, Environment & CI | 01-08 | 56 | In progress — 7/8 phases |
| II — Data Layer & Audit | 09-22 | 98 | Not started |
| III — Signal Preprocessing | 23-29 | 49 | Not started |
| 🔴 **MEGA TEST 1** — Data & Preprocessing Integrity | **30** | **7** | Not started |
| IV — Feature Engineering | 31-42 | 84 | Not started |
| V — Modeling Core | 43-52 | 70 | Not started |
| 🔴 **MEGA TEST 2** — Feature & Model Integrity | **53** | **7** | Not started |
| VI — Search & Optimization | 54-62 | 63 | Not started |
| VII — Experimental Runs | 63-76 | 98 | Not started |
| 🔴 **MEGA TEST 3** — Experiment & Result Integrity | **77** | **7** | Not started |
| VIII — Analysis | 78-84 | 49 | Not started |
| IX — Result Assets | 85-104 | 140 | Not started |
| 🔴 **MEGA TEST 4** — Deliverable Completeness | **105** | **7** | Not started |
| X — Inference & Dashboard | 106-120 | 105 | Not started |
| 🔴 **MEGA TEST 5** — Full System | **121** | **7** | Not started |
| XI — Packaging & Delivery | 122-126 | 35 | Not started |
| **Total** | **126** | **882** | |

## Settled decisions

| # | Decision | Resolution |
|---|----------|-----------|
| 1 | CirCor Unknown-murmur patients (n=68) | **Run both.** Headline the 3-class version (matches the 2022 Challenge); report the 874-patient binary version alongside. |
| 2 | Optional PhysioNet diagnosis multiclass track | **Yes.** Objective 6 cannot rest on PASCAL alone (19 samples in one class). Merge the rare diagnoses into a defensible scheme and document it. |
| 3 | Dashboard architecture | **Next.js static export + build-time codegen + one FastAPI predict endpoint.** No runtime fetch for precomputed values. |
| 4 | Frontend framework | Next.js 14 App Router (not Vite) — Server Components read `outputs/` at build time. |

## Still open

| # | Decision | Decide at |
|---|----------|-----------|
| 5 | 1D-CNN (M9) on CPU-only hardware | After the Phase 75 complexity timings |
| 6 | GA / PSO beyond Bayesian optimization | After the Phase 52 search timings |
