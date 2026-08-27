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

_Placeholder — entry points land in Part IX._

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
| From Part X: the frontend builds and contains no hand-typed metric | |

That second column is covered by the phase `[TEST]` gates and the five MEGA TEST
phases, which run locally against the real files.

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
