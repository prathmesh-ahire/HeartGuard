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

### 4. Verify the environment

```bash
python scripts/verify_env.py
```

Checks the interpreter version, that it is running from `.venv`, that every
pinned package is installed *at its pinned version* and actually imports, and
that Node LTS and npm resolve. Exits nonzero on any failure.

### 5. Run the pipeline

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

## Reproducibility

Global seed **42**, everywhere. Every run records its seed, fold map,
hyperparameters and package versions to the run manifest. Two runs of the same
command produce identical numbers.

No metric is ever hand-typed — every number in every deliverable is generated by
code and traceable to its source CSV or JSON. Anything that could not be produced
is listed in `outputs/missing_outputs_report.txt`.

### Fold safety — the QA claim

**Claim: no statistic derived from a test fold ever reaches a model that is
scored on it.**

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

## Status

Under construction. See `Docs/todo.md` for the full task breakdown and current
progress.
