# HeartGuard — PV-MEPCG / PulseVision

[![CI](https://github.com/prathmesh-ahire/HeartGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/prathmesh-ahire/HeartGuard/actions/workflows/ci.yml)

**Framework name:** PV-MEPCG / PulseVision
**Repository name:** HeartGuard

A Python research prototype that classifies phonocardiogram (PCG) heart-sound
recordings using a search-optimized heterogeneous ensemble over 138 engineered
acoustic features (time, frequency, MFCC, chroma, DWT and envelope domains)
across three public dataset families.

## ⚠️ Scope boundary — read this first

**This is an academic screening and decision-support prototype. It is not a
medical device and it is not a diagnostic tool.**

It does not diagnose, does not recommend treatment, does not prescribe, and does
not replace examination by a qualified clinician. Every output is a screening
indication produced for research and educational evaluation only. No clinical
decision should be made on the basis of anything this software produces.

## Tasks

| Task | Label space |
|---|---|
| Binary | normal / abnormal (PhysioNet 2016) |
| PASCAL A | normal / murmur / extrahls / artifact (4-class) |
| PASCAL B | normal / murmur / extrastole (3-class) |
| CirCor murmur | absent / present / unknown |
| CirCor outcome | normal / abnormal |

The five label spaces are never merged. `artifact` in PASCAL A is a
*recording-quality* label, not a cardiac class.

## Datasets

Input lives under `dataset/` and is **read-only** — never written to, never
committed (1.3 GB, gitignored).

| ID | Dataset | Contents | Native fs |
|----|---------|----------|-----------|
| D1 | PhysioNet/CinC 2016 | 3,240 train + 301 validation | 2000 Hz |
| D2 | PASCAL set_a | 124 labeled + 52 unlabelled | 44100 Hz |
| D3 | PASCAL set_b | 461 labeled + 195 unlabelled | 4000 Hz |
| D4 | CirCor DigiScope 2022 | 942 patients / 3,163 recordings | 4000 Hz |

### Redistributed sample and attribution

One 93 KB recording is the exception to "never committed":
`frontend/public/85197_TV.wav`, together with its expert cardiac-cycle
segmentation, so the dashboard can show a real S1 / systole / S2 / diastole
overlay rather than a synthetic animation.

Contains information from
[The CirCor DigiScope Phonocardiogram Dataset](https://physionet.org/content/circor-heart-sound/1.0.3/),
which is made available under the
[Open Data Commons Attribution License v1.0 (ODC-By 1.0)](https://opendatacommons.org/licenses/by/1-0/).

The full notice — record id, checksum, and which licence clause requires what —
is in `frontend/public/NOTICE.md`, beside the audio, and the attribution is also
rendered in the dashboard wherever that recording plays. The recording is a
**de-identified dataset sample**, not a patient and not a case.

### Dataset placement

Nothing downloads the corpora for you and nothing in this repository redistributes
them (the one 93 KB exception is described above). Obtain each from its canonical
source — the citations and licence terms are in [CITATION.md](CITATION.md) — and
place them so the tree below matches exactly. Every path resolves through
`configs/paths.yaml`; no dataset location is hardcoded in Python.

```
dataset/
├─ archive (3)/                     D1 — PhysioNet/CinC 2016
│  ├─ training-a/ … training-f/     3,240 recordings (.wav + .hea), 2000 Hz
│  ├─ validation/                   301 recordings — NOT used, see Limitations
│  └─ annotations/
├─ archive (2)/                     D2/D3 — PASCAL CHSC 2011
│  ├─ set_a/    set_a.csv    set_a_timing.csv
│  └─ set_b/    set_b.csv
├─ archive/                         D4 — CirCor DigiScope 2022
│  ├─ training_data/training_data/  ← doubled, and NOT a typo
│  ├─ training_data.csv
│  └─ LICENSE.txt                   ODC-By 1.0
└─ Heartbeat_Sound/                 duplicate of set_a+set_b — label helper only
```

The folder names with spaces and brackets are the names the archives unpack
under; they are what `configs/paths.yaml` expects, so do not tidy them. Verify a
placement with:

```bash
python scripts/01_run_dataset_audit.py --smoke
```

`dataset/` is **read-only input**: never written to, never committed (1.3 GB,
gitignored). Derived signals and features go to `cache/`, results to `outputs/`.

## Repository layout

```
src/          pipeline packages (data_loader … api, utils)
configs/      YAML configuration (paths, signal, features, models, experiments)
scripts/      runnable entry points
tests/        pytest suite
outputs/      every generated table, figure, metric and report
models_saved/ serialized trained models
cache/        preprocessed signals and feature shards (regenerable)
frontend/     Next.js static dashboard (populated in Part X)
notebooks/    exploratory notebooks
Docs/         todo.md (the plan), note.md (the change log), source documents
```

## Quickstart

Requires **Python 3.11.9** and, from Part X onward, **Node LTS** (>= 18, even
major version) with npm on PATH.

### 1. Create the virtual environment

```powershell
# PowerShell
C:\Users\prath\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
```

```bash
# bash / Git Bash
"C:/Users/prath/AppData/Local/Programs/Python/Python311/python.exe" -m venv .venv
```

### 2. Activate it

```powershell
# PowerShell   (if the execution policy blocks it:
#               Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.venv\Scripts\Activate.ps1
```

```bash
# bash / Git Bash
source .venv/Scripts/activate
```

```cmd
:: cmd.exe
.venv\Scripts\activate.bat
```

Confirm the prompt shows `(.venv)`, and that
`python -c "import sys; print(sys.executable)"` points inside `.venv`.

### 3. Install requirements

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/base.txt -r requirements/extra.txt -r requirements/api.txt -r requirements/report.txt
```

| File | Contents |
|---|---|
| `requirements/base.txt` | core pipeline — numerics, scikit-learn, librosa, DWT, plotting |
| `requirements/extra.txt` | XGBoost / LightGBM, Bayesian / GA / PSO search, SHAP, statsmodels |
| `requirements/api.txt` | FastAPI inference service (Part X) |
| `requirements/report.txt` | .docx / .xlsx / static-image generation (Part IX) |

Every version is pinned. The pins were resolved by a real install on this
Windows + Python 3.11.9 machine and each was verified to import — none are
guessed. The full resolved environment, transitive dependencies included, is
frozen at `outputs/configs/pip_freeze.txt`.

### 4. Node — only if you will rebuild the dashboard

**You do not need Node to run or to read anything.** `frontend/out/` is
committed, so FastAPI serves the whole dashboard from the static export:

```bash
python -m uvicorn src.api.main:app --port 8000     # then open http://127.0.0.1:8000/
```

Node LTS (>= 18, even major version) is needed only to *rebuild* that export:

```bash
node --version && npm --version    # both must resolve
cd frontend
npm ci                             # exact versions from package-lock.json; never `npm install`
npm run build                      # prebuild guards -> next build -> postbuild audits
```

`npm run build` runs the four correctness checks described under
[the guard rail](#the-dashboards-correctness-guard-rail--the-section-19-qa-answer)
and **fails the build** if any of them fails. A rebuild rewrites every
content-hashed chunk name, so do it at a release point rather than casually.


### 5. Verify the environment

```bash
python scripts/verify_env.py
```

Checks the interpreter version, that it is running from `.venv`, that every
pinned package is installed *at its pinned version* and actually imports, and
that Node LTS and npm resolve. Exits nonzero on any failure.

### 6. Run the pipeline

```bash
python scripts/00_run_everything.py --list        # the 65 stages, in order
python scripts/00_run_everything.py --estimate    # what a full run costs, measured
python scripts/00_run_everything.py               # everything, raw data to screenshots
python scripts/00_run_everything.py --resume      # skip stages whose outputs exist
```

One command reproduces the project end to end: dataset audit, preprocessing,
feature extraction, models, search, every experiment, every analysis, every
table, figure, algorithm and asset pack, the evidence index, then the frontend
chain (`npm ci`, `npm run build` with its guard rail and displayed-value audit)
and the thirteen gated dashboard screenshots.

**Read `--estimate` before starting one.** A complete run from an empty
`outputs/` is **67.6 hours of CPU** on the machine this was built on — EXP-A2
alone is 23.6 h and EXP-C1 is 16.4 h — and there is no GPU path. That figure is
summed from the recorded start and finish of every run in
`outputs/00_evidence_index/run_manifest.json`; it is a measurement, not an
estimate. `--resume` skips any stage whose declared outputs already exist, so an
interrupted run continues rather than restarting, and the long experiments carry
their own per-fold checkpointing underneath that.

Useful subsets:

| Flag | Effect |
|---|---|
| `--from <stage>` | start partway through, e.g. `--from tables_setup` |
| `--only <stage>` | one stage; repeatable |
| `--skip-frontend` | stop after the evidence index |
| `--dry-run` | print what would run and stop |
| `--smoke` | the reduced path of every stage that declares one |

`--smoke` runs each script's own reduced form. Those write into the normal output
directories, so use it on a scratch checkout rather than over a completed
`outputs/`.

### Where the results land

Everything generated goes under `outputs/`, one directory per stage of the
pipeline, and every file is registered in `outputs/00_evidence_index/`. The full
directory-by-directory map is in [ARCHITECTURE.md](ARCHITECTURE.md#output-map);
the short version:

| Directory | Holds |
|---|---|
| `00_evidence_index/` | the evidence index, the run manifest, the QA and compliance reports |
| `01_dataset_audit/` … `05_search_optimization/` | audit, preprocessing, features, models, searches |
| `06_binary_results/`, `07_multiclass_results/`, `08_circor_external_validation/` | the experiments and their result tables |
| `09_*`, `10_robustness/`, `11_complexity/`, `12_statistics/` | ablations, robustness, cost, significance |
| `13_figures_diagrams/`, `14_algorithms/`, `15_dashboard_screenshots/`, `16_literature_review/` | the 35 graphs, 20 diagrams, 20 algorithms, 13 screenshots, the review |
| `Q1_PAPER_ASSETS/`, `THESIS_ASSETS/` | the two rebuilt deliverable packs |
| `missing_outputs_report.txt` | **everything that could not be produced, with the technical reason** |

Start at `outputs/00_evidence_index/evidence_index.xlsx`: every artifact, the
command that produced it, its source data, and whether the file is on disk today.

Reference documents at the repository root:

| File | What it is |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | every package and module, the codegen boundary, the output map — generated |
| [CONFIGURATION.md](CONFIGURATION.md) | every configuration key and its default — generated |
| [CITATION.md](CITATION.md) | the three datasets, their citations and their licence terms |
| [HANDOVER.md](HANDOVER.md) | what was produced, what was not, and what clinical validation would require |
| [CHANGELOG.md](CHANGELOG.md) | release history |


### Reproducibility

- **Seed 42, everywhere.** Every run records its seed, fold map,
  hyperparameters and package versions to `run_manifest.json`. Two runs of the
  same command produce identical numbers.
- **Every artifact carries its own command.** `evidence_index.csv`'s `command`
  column is the exact invocation that produced that file, and
  `tests/test_run_everything.py` asserts that every script named there appears in
  the one-command runner — so "one command rebuilds this" cannot quietly stop
  being true.
- **`frontend/out/` is committed.** A grader with no Node install can serve the
  whole dashboard from FastAPI alone (`python -m uvicorn src.api.main:app`).
  Rebuild it only at a release point: every build rewrites the content-hashed
  chunk names.
- **The environment is frozen** at `outputs/configs/pip_freeze.txt` and
  `frontend/package-lock.json`.
- **What was not done** is written down. A literal single-pass run from an empty
  `outputs/` is recorded in `outputs/missing_outputs_report.txt` with its reason
  (67.6 h) and with exactly what was verified instead.

## What CI proves — and what it does not

The badge above turns green on every push. It is a statement about the **code**,
never about the **science**.

`dataset/` is 1.3 GB, gitignored, and never reaches GitHub. Every data-dependent
test is therefore excluded in CI by design: the `needs_data` marker auto-skips
when `dataset/` is absent, and the workflow additionally deselects those tests
with `-m "not needs_data"` so the exclusion is explicit in the log rather than
inferred from a skip count.

| CI proves | CI proves nothing about |
|---|---|
| The package imports on a clean machine from the pinned requirements alone | Behaviour on real audio — resampling, filtering, feature extraction |
| Every pinned version resolves and installs | Any count, any metric, any result |
| Logic tests pass: label vocabularies, config validation, atomic IO, the synthetic-signal harness | Whether the **real** split maps leak a subject (that check needs `dataset/`) |
| **Fold safety of the pipeline**, via a synthetic leakage canary — see below | |
| `ruff` and `mypy` are clean over `src/` | Anything needing the four dataset families |
| From Part X: the frontend builds, contains no hand-typed metric or foreign data source, and every value it renders matches its source file | |
| The component unit tests (Vitest) | The browser tests (Playwright): they drive the real inference API, which needs the gitignored model bundles and corpus samples |

That second column is covered by the phase `[TEST]` gates and the five MEGA TEST
phases, which run locally against the real files.

## The dashboard's correctness guard rail — the section 19 QA answer

The dashboard's rule is that **the browser never computes, and never declares, a
metric**. Every precomputed number is formatted in Python by
`scripts/17_export_frontend_data.py` and imported from `frontend/lib/generated/`;
the only runtime call is live inference (`POST /predict`), and it lives in
`frontend/lib/api.ts`. The rule exists because a parallel implementation of this
brief displayed a 95.82% result its pipeline never produced. It is enforced by
four checks, and **`npm run build` fails if any of them fails**:

| When | Check | What it catches |
|---|---|---|
| `prebuild` | `scripts/16_check_no_hardcoded_metrics.py` | A metric literal typed into `app/` or `components/` (a metric-named key given a number, a 3-decimal literal, a percentage in text), and any data source other than `lib/generated/` — a `fetch`, a directly imported `.json`/`.csv`, a foreign `generated`/`outputs` module. |
| `prebuild` | the exporter, run last before `next build` | A site built against stale data. |
| `postbuild` | `scripts/20_check_bundle_budget.py` | A payload that leaked into every route. |
| `postbuild` | `scripts/45_audit_displayed_values.py` | Crawls every page of the **built** site: every metric-shaped value on screen must be a string a generated payload carries; every table and figure cell must re-format from its committed CSV to the exported string; every evidence digest must match the file today; every page footer must show the run id, commit and export time of an `export_frontend_data` run in the project's run manifest. |

The audit writes a stamp bound to a digest of the exact HTML it read. Screenshots
(`npm run screenshots`) start only after `45_audit_displayed_values.py --gate`
confirms the site on disk is the one that passed — no screenshot of an unaudited
page, or of a page rebuilt after its audit.

Every table and figure on the site links the CSV it was read from; the evidence
browser on `/reports/` lists every exported column and page source with its
sha256. The whole frontend suite — build, audits, Vitest, Playwright — runs with
`.\scripts\run_tests.ps1 -FrontendOnly`.

## Fold safety — the QA claim

Seed discipline and the "no hand-typed number" rule are covered under
[Reproducibility](#reproducibility) above. This section is the other half of the
QA answer: that no statistic derived from a test fold ever reaches a model
scored on it.

This is structural rather than procedural. Every step that learns something from
the data — the imputer's medians, the scaler's mean and standard deviation, the
feature selector's scores — is a step *inside* an sklearn `Pipeline`
(`src/models/pipeline.py`), so it is refitted on each training fold and never on
the matrix as a whole. Getting it wrong requires deliberately lifting a step out
of the pipeline.

Two lines that differ by everything and look identical in their output:

```python
X = StandardScaler().fit_transform(X)            # leak: fitted across all folds
Pipeline([("scaler", StandardScaler()), ...])    # fitted per training fold
```

The first computes a mean over records the model is about to be tested on. The
resulting metric is optimistic by a small, plausible amount — the kind of error
that survives review because nothing looks wrong.

**How the claim is verified.** `tests/test_fold_safety.py` plants a *canary*: a
feature whose value in the test rows is four orders of magnitude larger than in
the training rows. The tests then assert on what each step **learned** — the
imputer's `statistics_`, the scaler's `mean_` and `n_samples_seen_` — rather than
on its output, because a leaked scaler and an honest one both produce plausible
numbers. One test deliberately fits across all rows to confirm the canary is loud
enough to detect a leak at all; a canary that cannot fire proves nothing.

The canary is synthetic, so **this runs in CI** on every push.

**Folds and subjects.** Cross-validation folds are *loaded* from the audit's
published split map (DA-07), never re-derived at runtime — re-deriving would make
every result depend on the scikit-learn version and the row order the matrix
happened to arrive in. `src/evaluation/cv.py` asserts, in every fold on every
run, that train and test share no subject group and no row. Verifying that the
real split maps are themselves clean needs `dataset/` and runs in the local
`[TEST]` gates.

**Resampling is off**, and switching it on raises rather than silently doing
nothing. Class imbalance is handled by class weights: SMOTE-style oversampling on
138 correlated acoustic features invents recordings no chest ever produced, and
resampling *before* splitting would duplicate a record into both train and test.

## Troubleshooting

Before debugging anything else, check two things: that `.venv` is **activated**
in this shell, and that `dataset/` is where [Dataset placement](#dataset-placement)
says. A `ModuleNotFoundError`, a "file not found" on a dataset path, or a script
that dies instantly is far more often one of those two than a bug.

### The four dataset traps

Each of these cost a working session at least once. They are handled in the code;
this list exists so a surprising result is recognised rather than rediscovered.

| Symptom | Cause | What to do |
|---|---|---|
| A CirCor loader returns **zero files** and raises nothing | The real path is `archive/training_data/training_data/` — **doubled, not a typo**. A path built as `archive/training_data/` matches nothing and returns an empty list. | Use `configs/paths.yaml`; never build a CirCor path by hand. |
| CirCor `Outcome` is missing / all-null | **`Outcome` is not in `training_data.csv`.** It exists only in the per-patient `.txt` files as `#Outcome: Normal\|Abnormal`. | Parse the txt files (`src/data_loader/circor.py` already does). |
| PASCAL set_b labels do not join | **The set_b CSV filenames do not match what is on disk** — `Btraining_extrastole_127_…` against `extrastole__127_…`, with *zero* overlap. | The authoritative label source is the `Heartbeat_Sound/` folder structure. |
| Class counts double, every metric improves | **`dataset/Heartbeat_Sound/` is a 100% duplicate** of set_a + set_b (832/832 filename overlap), foldered by class. | It is a **label helper only** and must never be a training source. |

### Everything else

| Symptom | Likely cause | Fix |
|---|---|---|
| `Activate.ps1` refused by PowerShell | Execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Paths wrong on a second machine, or on CI | A machine path committed into `configs/paths.yaml` | `project_root` must stay `"auto"`. To relocate, use `HEARTGUARD__PATHS__PROJECT_ROOT`, never an edit. |
| A YAML number behaves like a string | PyYAML is YAML 1.1: `1.0e3` parses as the **string** `"1.0e3"` | Write a signed exponent: `1.0e+3` |
| A search-report column reads as blank/NaN | pandas treats `"None"` as missing, and `class_weight=None` is a real choice the search made | Read T07 with `search_report.read_t07`, never a bare `pd.read_csv` |
| Two runs of the same command differ in the last bits | A thread count reached a value — XGBoost's `subsample` row mask is drawn per thread block | `n_jobs` is pinned (M8 to 4, never `-1`); models are pinned single-threaded at load |
| Extraction looks stuck | It is not — full-corpus extraction is ~2.4 h and **97.9% of it is `time_sample_entropy`** | It checkpoints every 250 records; re-running resumes |
| An experiment's result table lost models | **An experiment split across two passes overwrites `per_fold_metrics.csv`** | Re-run once with the full model list; checkpoints make it cheap |
| `npm run build` fails on a "hardcoded metric" | A number was typed into `app/` or `components/` | That is the guard rail working. Export it through `scripts/17_export_frontend_data.py`. |
| Screenshots refuse to start | The audit gate: the site on disk is not the one that passed | Rebuild, let `postbuild` audit it, then capture |
| A test fails only on CI, or only on a fresh clone | A test reading a gitignored artifact (the feature matrix, a `.joblib`) | Hide the artifact locally (rename in place) and re-run; see `Docs/note.md` |

A longer, dated record of every non-obvious finding is in `Docs/note.md`. That
file is kept local by the user's decision and is not part of the published
repository.

## Limitations

Read these before quoting any number from this project anywhere.

**1. No clinical validation of any kind.** Nothing here has been evaluated
prospectively, in a clinic, on a patient, or against a clinician. It is a
research prototype trained and tested on public research corpora.

**2. The binary model does not transfer to an unseen recording collection.**
Leave-one-sub-collection-out validation (EXP-F3) takes AUC from **0.92–0.94
pooled to 0.43–0.53**, several folds below chance. AUC is rank-based, so this is
not a threshold or class-prior problem — the ranking is not there. The pooled
cross-validated numbers remain correct *as a within-corpus cross-validation*, and
every headline figure must be framed as "within the PhysioNet 2016 corpus".
**No claim of generalization, deployment-readiness or field performance is
supported by this work.**

**3. PhysioNet's six sub-collections behave like six different datasets.** Class
balance runs from 8.5% abnormal (training-e, 66% of the corpus) to 71.4%
(training-a), and recording source explains ~25% of the variance in the top
features — all 20 top features by pooled Cohen's *d* reverse sign between
sources. A pooled feature ranking on this corpus is not a cardiac finding.

**4. There is no held-out PhysioNet test set.** The 301 `validation/` records
duplicate 301 training records belonging to 205 subjects that carry 1,108 of the
3,240 training records, so using them would leak. They are dropped. D1 results
are cross-validated only; external validation is EXP-D1.

**5. CirCor is the public subset only** — 942 patients of the full corpus, and
its `Outcome` labels come from the per-patient text files rather than the CSV.

**6. Subject IDs are partial, and absent for PASCAL A.** Where a subject ID
exists, folds are grouped by it and no subject appears in both train and test.
Where one cannot be derived, the record carries `subject_derived=False` and the
write-up says so. PASCAL A additionally holds **one recording under two class
labels** (`extrahls__201104021355` / `murmur__201104021355`, correlation
0.999978); both are kept, sharing one subject id.

**7. The PASCAL samples are small.** PASCAL A is 124 labelled records with 19 in
one class; PASCAL B is 461. Any per-class recall from them is an estimate with a
very wide interval. This is why the PhysioNet diagnosis track exists — and that
track is itself **confounded with recording source**, where a no-audio baseline
beats every model.

**8. EXP-D1 is adult-to-paediatric transfer, and a large drop is expected.**
PhysioNet's median age is 25 with 0.3% under 18; CirCor is ~98% child or infant.
The drop is a population effect compounded by the acquisition effect in point 2 —
it is **not** evidence that the method fails.

**9. `artifact` in PASCAL A is a recording-quality label**, not a cardiac class.
The four-class model is not a four-class *cardiac* classifier.

**10. The statistical plan is underpowered where n is small.** Wilcoxon and
Friedman over 5 CV folds is n=5 and cannot reach significance in some
configurations. Repeated 5×5 CV runs first, and every p-value states its n.

**11. Some deliverables are honestly incomplete.** A literal single-pass run from
an empty `outputs/` costs a measured **67.6 hours** of CPU and was not performed;
what *was* verified instead is recorded in `outputs/missing_outputs_report.txt`,
along with every other gap. Nothing was invented to fill one.

**12. CPU only.** There is no GPU path and no deep-learning component. The method
is an engineered-feature ensemble by design, not by limitation of the hardware —
but the hardware does bound what could be searched.


## Status

Complete through Part XI. Every pipeline stage, experiment, analysis, asset pack
and dashboard page is built; the QA sweep, compliance review and delivery
checklist are generated into `outputs/00_evidence_index/`. What is *not* done is
written down rather than omitted — see `outputs/missing_outputs_report.txt` and
[HANDOVER.md](HANDOVER.md).
