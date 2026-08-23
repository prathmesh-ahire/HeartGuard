# Project Notes — Change Log (newest first)

This file tracks things that went wrong, surprised us, or required a non-obvious decision — bugs found, wrong assumptions corrected, deliberate deviations from `todo.md` or the source documents, dataset quirks, version gotchas. It is **not** a session log: a task that completed cleanly with no surprises does not belong here, even summarized — `todo.md`'s checkboxes already record that. Only write here when something didn't go as planned, or when a decision needs a "why" that isn't obvious from the code or diff itself.

## Quick Reference

- **GitHub remote:** https://github.com/prathmesh-ahire/HeartGuard.git
- **Repo name:** HeartGuard · **Framework name in all deliverables:** PV-MEPCG / PulseVision. Not interchangeable.
- **Python:** 3.11.9, venv at `.venv`. **CPU only — no CUDA GPU on this machine.**
- **Global seed:** 42, everywhere, always.
- **Dataset root:** `dataset/` — 1.3 GB, read-only, gitignored. 7,536 recordings, ~43 hours of audio, zero corrupt files.
- **Locked feature count:** 138 = Time 24 + Frequency 22 + MFCC 39 + Chroma 24 + DWT 24 + Envelope 5.

---

## ⚠️ OPEN ITEMS — things needing a human decision or a manual step

Kept at the top so they are not lost inside the entries below. Delete an item once it is genuinely settled. These are also mirrored in the Open Decisions table at the bottom of `todo.md`.

### 1. 1D-CNN (M9) on CPU-only hardware *(raised 2026-08-22, planning; deferred by design)*

Listed as Optional in both source documents. This machine has no GPU. **Deliberately deferred** until the Phase 75 complexity timings exist — that is when we will know what the classical stack costs and therefore what a CNN budget would have to be. If it is dropped, the reason goes in `missing_outputs_report.txt`, not silently.

### 2. Advanced search method beyond RandomizedSearchCV *(raised 2026-08-22, planning)*

RandomizedSearchCV is the documented minimum; one advanced method is required. **Bayesian optimization first** (scikit-optimize, cheapest to add). GA and PSO are high-novelty but expensive, and both are marked Optional. Decide after the Phase 52 timings show what a full search actually costs on this hardware.

**Recently settled** (full reasoning in the dated entries below and in `todo.md`'s Settled decisions table):

- **CirCor Unknown-murmur patients (n=68)** — settled 2026-08-23: run both. Headline the 3-class version (matches the 2022 Challenge); report the 2-class version on the 874 known patients alongside. T67.2 updated.
- **PhysioNet diagnosis multiclass track** — settled 2026-08-23: in scope. Objective 6 cannot rest on PASCAL A, which has 19 samples in one of its four classes; PhysioNet gives 3,240 records across 12 diagnosis categories. Rare categories merged under a documented scheme, presented as supplementary evidence. Phase 73 updated.
- **Frontend architecture** — settled 2026-08-23: Next.js static export + build-time codegen + one FastAPI predict endpoint. See the dated entry below.

---

## 📌 STANDING RULES — settled, but they still bind future work

### Where a source document contradicts the actual data, the data wins — and the discrepancy gets written down

Both source documents were written before anyone counted the files. Two figures are wrong (see the 2026-08-22 audit entry below). The rule is **not** to quietly use the real number and move on, and **not** to repeat the document's number because it is "official". Report the audited figure, and state the discrepancy explicitly in `dataset_audit_report.docx` and the thesis limitations section. An examiner who checks CirCor's published size will find 942 patients; the write-up needs to have already said so.

### A metric at or near 1.0 is a bug report, not a result

Before recording any near-perfect score, check in this order: (1) a subject leaked across folds, (2) a scaler or selector fitted on the full matrix instead of the training fold, (3) a duplicate recording landed in both splits — `Heartbeat_Sound/` makes this very easy to do by accident, (4) a label joined on the wrong key. This is a standing rule because the blueprint explicitly says *"Do not report perfect accuracy unless independently verified."*

### Verify data-dependent behaviour against the real files, never against a synthetic-input test

A unit test proving the extractor returns 138 finite values from a synthetic signal says nothing about a 0.76-second PASCAL B recording (too short for a 5-level DWT or for MFCC deltas) or a 122-second PhysioNet recording. Both exist in this dataset. Every extractor needs a real-data pass over the duration extremes, not just a green synthetic test.

### `dataset/` is read-only input

Never write into it — no converted WAVs, no cached features, no metadata files. Derived data goes to `cache/`, outputs go to `outputs/`, the master metadata goes wherever `configs/paths.yaml` says. The dataset folder must stay byte-identical to what was downloaded, so any integrity claim in the audit remains checkable.

---

## 2026-08-23 — Phase 06: the `needs_data` skip path could not be tested on a machine that has the dataset, so it got a switch

### 1. `--no-data`, and why an untested skip path is a broken one

T06.7 requires the `slow` and `needs_data` markers to be seen *correctly skipping*. But `needs_data` auto-skips when `dataset/` is absent — and this machine has the dataset, so those tests run. The skip branch would never execute locally, and would first execute in CI, where nobody is watching and where a mistake in it looks like a green build.

**Added `--no-data`**, which forces `dataset_available()` false. That makes the CI condition reproducible on a developer machine. All five configurations were then run for real:

| invocation | result |
|---|---|
| default | 132 passed, 1 skipped (slow) |
| `--runslow` | 133 passed, 0 skipped |
| `--no-data` | 130 passed, 3 skipped (1 slow + 2 needs_data) |
| `--runslow --no-data` | 131 passed, 2 skipped (needs_data only) |
| `-m "not needs_data"` (the CI form, T08.2) | 130 passed, 1 skipped, 2 **deselected** |

Note the last row: CI *deselects* rather than skips, a different mechanism from the auto-skip. Both had to be checked; only one of them is what CI actually does.

**`tests/test_harness.py` is not in `todo.md`.** It was added because T06.7 has nothing to observe without at least one test carrying each marker, and because the fixtures and the synthetic generator need their own assertions. It also self-checks the autouse reseed by asserting two sibling tests draw the same first random value.

### 2. `--strict-markers` was verified by making it fail

`@pytest.mark.needs_dataa` (typo) was pushed through collection deliberately. Without strict mode it attaches a marker nothing selects on, so the test **runs in CI where the dataset does not exist** and fails there rather than skipping — with a one-character cause. With strict mode it is a collection error immediately:

```
ERROR ... 'needs_dataa' not found in `markers` configuration option
```

Same principle as T08.7 and T121.2: a guard that has never fired is not proven to work.

### 3. The suite prints what it skipped, in its own header

`pytest_report_header` emits `PV-MEPCG: dataset=present, slow=SKIPPED (--runslow to include)` on every run.

The failure this prevents: a suite reporting "132 passed" while silently skipping every data-dependent test is a green build that proves nothing about the pipeline. Making the state visible in the header means the reader can see which half ran without reading the invocation.

### 4. The synthetic generator is explicitly labelled as insufficient evidence

`make_synthetic_pcg` produces Gaussian-windowed S1/S2 bursts with systole shorter than diastole, band-limited to 20–400 Hz, deterministic under seed 42. Its docstring states plainly that it proves **shape, dtype, finiteness and determinism only** — never that anything works on real recordings, per the standing rule in `CLAUDE.md`.

To keep the real extremes cheap to reach, it also ships `make_edge_case_signals()`: the 0.76 s shortest real record, a sub-cycle fragment, silence, a constant, a clipped square, DC-only, a single sample and an impulse. These are the inputs that make an extractor raise instead of returning NaN — the DWT-level guard (T36.6), the MFCC delta-width guard (T34.6) and the `sosfiltfilt` padlen guard (T24.3) all have real records that trigger them, and now have synthetic stand-ins for the unit-test layer too.

### 5. Two smaller decisions

- **No `__init__.py` in `tests/` or `tests/fixtures/`.** Both were created, then removed after confirming collection works without them via implicit namespace packages. Fewer files, and it keeps pytest's default `prepend` import mode straightforward. Worth knowing: this only stays safe while no two test files share a basename across directories.
- **`tests/test_env.py` duplicates part of `scripts/verify_env.py` deliberately.** The script is a human-facing pre-flight check with advisories and colour; the test is the machine gate CI runs on every push. Coupling them would let a change to the script's reporting quietly disable the CI check. The test additionally *runs* `BayesSearchCV().fit()` rather than only importing skopt, for the reason recorded in the Phase 02 entry.

**Files:** `pytest.ini`, `conftest.py`, `tests/conftest.py`, `tests/fixtures/make_synthetic_pcg.py`, `tests/test_env.py`, `tests/test_harness.py`, `scripts/run_tests.ps1`, `scripts/run_tests.sh`.

---

## 2026-08-23 — Phase 05: "non-overlapping label vocabularies" is impossible as literally worded, pytest had no supplier, and bare `pytest` was already broken for CI

### 1. The label vocabularies **do** overlap, and that is a property of the datasets, not a defect

T05.6/T05.7 ask for label maps that are "mutually non-overlapping across tasks". Taken literally that assertion cannot pass, and should not:

| code | binary | pascal_a | pascal_b | circor_murmur | circor_outcome |
|---|---|---|---|---|---|
| 0 | normal | normal | normal | Absent | Normal |
| 1 | abnormal | murmur | murmur | Present | Abnormal |
| 2 | — | extrahls | extrastole | Unknown | — |
| 3 | — | artifact | — | — | — |

`normal` appears in three tasks, `murmur` in two, and **every** task starts at 0. The sharpest case is code `2`: `extrahls` (an extra heart sound) in PASCAL A versus `extrastole` (an extra systole) in PASCAL B — two different cardiac phenomena behind one integer. Renaming them to force disjointness would misrepresent the source datasets.

**What the requirement actually protects against** is a label being silently reinterpreted under the wrong task. So that is what is enforced:

1. every map is **bijective** within its task;
2. labels are addressable only as `(task, code)` or `(task, name)` pairs, and `namespaced_id()` yields globally unique ids (`pascal_a:murmur`) which **are** disjoint across tasks — 14 of them, asserted unique;
3. there is deliberately **no** `decode(code)` taking a bare integer, because there is no correct answer to it;
4. a test asserts the collisions **exist** — that code 2 decodes differently per task, that `extrastole` is rejected by PASCAL A, that code 3 is valid only in PASCAL A.

Point 4 is the one to keep. If a future change ever made the vocabularies genuinely interchangeable, that test fails and forces the question, instead of letting a quiet merge through. All maps are `MappingProxyType`, so none can be mutated mid-run.

**If a later phase needs the literal reading**, it is wrong and this entry is why.

### 2. `int(1.5)` would have silently produced "abnormal"

`map_physionet_reference` originally did a plain `int(value)`. The test caught that `1.5` truncates to `1` and returns a confident `abnormal` — a malformed value becoming a real label with no complaint.

Now: bools are rejected (`True == 1` in Python, so a bool would have mapped to abnormal unchallenged), floats are accepted **only when exactly integral**, and anything non-integral raises. The integral-float path is not defensive padding — **pandas reads a column as `float64` the moment it contains a NaN**, so `-1.0` and `1.0` genuinely arrive from `REFERENCE.csv` and must still map. `numpy.int64` and `numpy.float64` are both handled, since loaders hand over numpy scalars rather than Python ints.

Also rejected: `0`. It is a valid *output* code and an invalid *input* code, which makes it the easiest value to wave through by accident.

### 3. Plan hole — nothing installed pytest

T05.6 writes the first pytest test; T06.1 adds `pytest.ini`; T06.6/T06.7 run the suite; T08.2 runs `pytest` in CI. **No task installs pytest.** T02.2–T02.4 list four requirements files and none contains it, and T08.3 says "add both to `requirements-dev.txt`" as though that file already exists.

Same shape as the statistics hole recorded on 2026-08-23: *a requirement with no supplier looks satisfied in the plan and gets improvised in execution.*

**Fixed:** `requirements-dev.txt` created now with `pytest==9.1.1` and `pytest-cov==7.1.0`, both installed and frozen. T08.3 adds ruff and mypy to the same file. `verify_env.py` was **not** extended to check it — the dev file is not needed to run the pipeline, only to test it.

### 4. Bare `pytest` was already broken, and CI is the thing that runs it

`python -m pytest` passed; bare `pytest` failed collection with `ModuleNotFoundError: No module named 'src'`. The difference is that `-m` puts the working directory on `sys.path` and the console script does not.

**T08.2 runs the bare form.** So the suite would have passed locally for another three phases and gone red the moment CI first ran — with the cause three phases behind it.

**Fixed** with a root-level `conftest.py` that inserts the project root on `sys.path`. Distinct from `tests/conftest.py` (T06.2), which holds fixtures. Both invocation forms now pass; the gate checks both, because checking only the convenient one is how this stayed hidden.

**Files:** `src/utils/constants.py`, `tests/test_constants.py`, `conftest.py`, `requirements-dev.txt`, `outputs/configs/pip_freeze.txt`.

---

## 2026-08-23 — Phase 04: PYTHONHASHSEED cannot be set retroactively, git commit does not exist yet, and the manifest needed snapshot dedup

### 1. `set_global_seed` cannot make `PYTHONHASHSEED` effective for its own process

T04.1 asks for `set_global_seed(42)` covering `random`, `numpy` **and** `PYTHONHASHSEED`. The first two work as written. The third cannot: **Python reads `PYTHONHASHSEED` once, at interpreter start.** Assigning `os.environ["PYTHONHASHSEED"]` from inside a running process does not change that process's hash randomization — it is already fixed.

What the assignment *does* do is propagate to **child processes**, which is what makes joblib workers deterministic. That is a real and useful effect, and it is the only one available from inside Python.

**How it is handled:** `set_global_seed` sets the variable (for children) and `hash_seed_status()` separately reports the truth — `effective_for_this_process` (was it set before the interpreter started?) versus `effective_for_child_processes`. Both go into the run manifest. On this machine the honest answer today is `effective_for_this_process=False`.

**The rule that follows:** no result may depend on `set` or `dict` iteration order. Sort explicitly instead. The feature registry already does the right thing — T31.3 enforces a fixed ordering rather than relying on hash stability. This is worth remembering when any future code iterates a set of feature names or record ids.

To get a fully deterministic hash seed, it must be set in the shell *before* launching Python:
```
$env:PYTHONHASHSEED = "42"   # PowerShell, before python starts
```

### 2. There is no git commit to record yet, and the manifest says so explicitly

T04.7 requires the manifest to record the git commit. `git init` is **T07.1**, so at Phase 04 there is no repository and no commit.

The manifest records:
```json
"git": {"available": false, "commit": null,
        "reason": "not a git repository -- `git init` is task T07.1, ..."}
```
rather than a bare `null` or a placeholder string. **"No commit, and here is why" is a fact; a blank field is ambiguous and a fabricated hash is a lie.** The gate asserts the honest form: if a repo exists, a real 40-character SHA plus a `dirty` flag; if not, `commit is None` *and* a non-empty reason. This is not the gate being weakened — an assertion that a nonexistent commit is a valid SHA would be an assertion that cannot be satisfied truthfully.

From Phase 07 onward this fills in automatically, including a **`dirty` flag** listing uncommitted files. A dirty tree means the code that produced a run is not exactly the code at that commit, which is the difference between "this is reproducible" and "this looks reproducible".

### 3. The manifest embedded a 45 KB config snapshot per run — now deduplicated

The first working version embedded a full snapshot of all five config files in **every** run record. Measured: 90 KB for two runs, ~45 KB marginal per run. Across a few hundred experiment runs (Part VII alone runs 25 folds × several models × several experiments) that is a 20 MB JSON storing the same content hundreds of times, slow to load and unpleasant to diff.

**Changed to schema 2:** snapshots are stored once each in a top-level `config_snapshots` map keyed by a content hash, and each run carries `config_hash`. Configs change rarely, so hundreds of runs collapse to a handful of snapshots. **No fidelity is lost** — `config_snapshot_for(run_id)` resolves any run back to the exact configuration it ran under, which is what rule 5 actually requires.

Measured after the change: 3 runs, 1 stored snapshot, marginal cost **2.9 KB per run** instead of 45 KB. The gate asserts both that the hash resolves to a complete five-file snapshot and that the file does not grow per-run.

### 4. Smaller decisions worth not rediscovering

- **`save_json` coerces NaN/±Inf to `null` and passes `allow_nan=False`.** `json.dumps` otherwise emits a bare `NaN` token, which is not valid JSON and which `JSON.parse` rejects outright. It works on most records and fails on the degenerate ones — precisely the records the NaN policy exists to track. `sanitize_for_json` is also the single place numpy scalars/arrays are converted, so an `int64` never reaches `json.dumps` and raises three hours into a run.
- **Every writer is atomic** (temp file in the same directory, then `os.replace`). The gate proves it by crashing mid-write and asserting the previous file survives intact with no temp debris. Long runs get interrupted; a truncated CSV that still *looks* like an output is how a bad number gets into a table.
- **`register_evidence` upserts on `evidence_id`**, and derives `status` from the filesystem rather than trusting the caller — a nonexistent artifact is recorded `missing`, never `ok`. In-project paths are stored **relative** so the index survives a clone; only out-of-project paths stay absolute.
- **`timer` records a duration even when the block raises**, tagged `[failed]`. A run that died after four hours still reports that it took four hours.
- **The three throwaway gate runs were left in the manifest.** They are honest records of runs that actually happened. The demo *evidence* rows were removed, because those pointed at a scratchpad file and a deliberately absent one, and Phase 102 asserts every evidence row resolves to a real file.

**Files:** `src/utils/seed.py`, `src/utils/logging_setup.py`, `src/utils/run_manifest.py`, `src/utils/timing.py`, `src/utils/io.py`, `src/utils/evidence.py`.

---

## 2026-08-23 — Phase 03: CirCor's metadata files are NOT in the doubled directory, and two config-loader bugs the gate caught

### 1. Dataset finding — `training_data.csv`, `RECORDS` and `SHA256SUMS.txt` live one level ABOVE the doubled folder

The 2026-08-22 audit recorded that CirCor's audio sits at `archive/training_data/training_data/`. It did not record where the *metadata* sits, and the natural assumption — inside the same doubled folder, next to the 10,431 audio files — is wrong.

```
dataset/archive/
    LICENSE.txt
    RECORDS              <- record index
    SHA256SUMS.txt       <- integrity checksums
    training_data.csv    <- 942-row demographics table
    training_data/
        training_data/   <- all 10,431 wav/hea/tsv/txt files, flat
```

**Why it matters:** the same failure mode as the doubled directory itself. `pd.read_csv(circor_root / "training_data.csv")` raises `FileNotFoundError` at best; a `glob` for the checksums returns empty and the integrity check in **T15.6** silently verifies nothing. Both are now correct in `configs/paths.yaml`, and the T03.7 gate asserts all eight configured data files resolve to real files on disk rather than trusting the paths.

**Affects:** T14.5 (demographics join), T15.6 (RECORDS / SHA256SUMS integrity), T14.1 (loader root).

### 2. `config.py` bug — the "is this a path?" heuristic was wrong in both directions

Path resolution originally guessed whether a string was a path by looking for a separator or a known file extension. That failed twice, in opposite ways:

- **Missed** bare roots. `outputs: "outputs"` and `cache: "cache"` have no separator, so they were never made absolute — despite T03.1 requiring absolute paths. Everything downstream would have received a relative path that only works when the process happens to be started from the repo root.
- **Wrongly absolutised** bare filenames. `reference_filename: "REFERENCE.csv"` ends in `.csv`, so it became `D:\Projects\HeartGuard\REFERENCE.csv` — a file that does not exist. That value is meant to be joined *per subset* (`training-a/REFERENCE.csv`, `training-b/REFERENCE.csv`, ...), so Phase 09's loader would have been handed a broken absolute path for all six subsets.

**Fixed** by replacing the heuristic with an explicit `NON_PATH_KEYS` set and an explicit list of path-bearing top-level sections. **The general lesson:** in a config file, whether a string is a path is a property of its *key*, not of its *characters*. Sniffing the value is guessing.

### 3. `config.py` bug — env-var overrides silently created shadow keys instead of overriding

Environment overrides map `HEARTGUARD__EXPERIMENTS__EXPERIMENTS__EXP-D1__RETUNING_ALLOWED` to a dotted path. Env var names are conventionally upper case, so the mapping lower-cased the whole path — but the YAML keys are **not** uniformly lower case. Experiment ids are `EXP-A1`, model ids are `M3`.

So the override created a brand-new `exp-d1` entry sitting beside the real `EXP-D1`, which was left untouched. **The override appeared to work and changed nothing.** Worse, it was invisible: `Config.overrides` dutifully recorded that the override had fired.

Found only because the gate asserted that setting `EXP-D1.retuning_allowed: true` must be *rejected* by validation — and it wasn't, because validation was still reading the real, untouched `EXP-D1`. A gate that only checked "does the override apply?" would have passed.

**Fixed:** path segments now resolve against existing keys case-insensitively, so an override targets the real key and only creates a new one when nothing matches. The gate now asserts both that the real key changed *and* that no lowercase shadow was created.

### 4. Validation is project-specific, not generic

`_validate` deliberately encodes this project's invariants rather than generic type checking, because the failures that matter here are semantic:

| Check | What it prevents |
|---|---|
| family counts sum to 138, each family matches its locked count, each composition sums to its own count | silent drift in the locked feature composition, which invalidates every cached matrix and ablation result |
| `filter.high_hz` and `mfcc.fmax` below Nyquist at `target_fs` | librosa's speech defaults putting mel filters above Nyquist at 2 kHz |
| `random_state == 42` everywhere, `defaults.seed == 42` | rule 5 |
| `calibration.fit_inside_fold` cannot be false | rule 2 |
| `task` must be one of the five label spaces | rule 4 |
| `EXP-D1.retuning_allowed` must be false and `population_note` must mention paediatric | the adult-to-paediatric framing being retrofitted after the metric exists |
| `n_splits * n_repeats == total_folds` | claiming 25 paired observations while running 5 |
| unknown top-level key is an error | `target_fs_hz:` sitting beside a defaulted `target_fs` |

The gate exercises **18 deliberately malformed keys** and requires each to be caught with a specific message. Rewriting any of these to "make a config load" is the exact failure mode the testing discipline forbids.

**Files:** `configs/paths.yaml`, `configs/signal.yaml`, `configs/features.yaml`, `configs/models.yaml`, `configs/experiments.yaml`, `src/utils/config.py`.

---

## 2026-08-23 — Phase 02: the environment resolved to numpy 2.x / pandas 3.x, and kaleido can't render on this machine

Three things from the environment build that a future session should not have to rediscover.

### 1. kaleido 1.3.0 needs an external Chrome, and there isn't one

`requirements-report.txt` names kaleido for static image export. **kaleido 1.x dropped the bundled Chromium that 0.2.1 shipped** and now drives an external Chrome/Chromium through `choreographer`. No Chrome is installed on this machine, so kaleido imports cleanly and then fails the moment it is asked to render — the worst shape of failure, because `verify_env.py` would have passed it.

**Not resolved, deliberately deferred**, because nothing currently needs it: every figure in the plan is matplotlib/seaborn, and Plotly here is a *frontend JavaScript* library, not a Python one. kaleido is effectively an orphan dependency inherited from the task wording.

**When it does become needed** (only if something renders Plotly from Python), pick one:
- run `kaleido_get_chrome` — downloads a Chromium, ~150 MB, needs network; or
- pin `kaleido==0.2.1` — self-contained, no browser needed, but incompatible with plotly >= 6.

`verify_env.py` reports this as an **advisory, not a failure**, and names both fixes. Do not "fix" the advisory by deleting the check.

### 2. The stack resolved to numpy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.0 — pandas 3 is the one to watch

Pins were resolved by installing, not by guessing, and every one was verified to import. Two are worth flagging before the data layer is written in Part II:

- **pandas 3.0** is a major release: copy-on-write is mandatory (chained assignment silently no-ops rather than warning) and the default string dtype changed. Loader code written from pandas 1.x/2.x habits will misbehave in ways that do not raise. Relevant from Phase 09 onward.
- **scikit-optimize 0.10.2** was released against a much older scikit-learn and is the classic "imports fine, dies at `.fit()`" package. It was **runtime-tested with a real `BayesSearchCV().fit()` against scikit-learn 1.9.0** and works. If sklearn is ever bumped, re-run that check before trusting Phase 60-era search results — an import check is not sufficient evidence for this package specifically.

`--only-binary=:all:` was used throughout, so nothing compiled from source and no toolchain is required to reproduce the environment.

### 3. T02.1 is marked [MANUAL] but was executed by the agent

The user's instruction was "finish phase 2", and creating a venv is a single reversible local command with no judgement call in it — so it was done rather than blocking the whole phase on it. Flagged in the response, recorded here because the `[MANUAL]` marker convention was crossed. **This is not a precedent for the other `[MANUAL]` tasks** — T07.2 (create the GitHub repo), T08.6 (enable Actions) and the 1D-CNN scope call are genuinely outside the agent's reach or are decisions the user owns.

**Files:** `requirements.txt`, `requirements-extra.txt`, `requirements-api.txt`, `requirements-report.txt`, `scripts/verify_env.py`, `README.md`, `outputs/configs/pip_freeze.txt`.

---

## 2026-08-23 — Phase 01: `cache/` is ignored as `cache/*`, not `cache/`, so the directory survives git

T01.4 says exclude `cache/`; T01.5 says every empty directory carries a `.gitkeep` so the tree survives a clone. Those two instructions conflict — a bare `cache/` rule ignores the directory itself, so `cache/.gitkeep` can never be staged and a fresh clone has no `cache/` at all.

**Written as** `cache/*` plus `!cache/.gitkeep`. Contents are still fully ignored (identical exclusion behaviour); only the placeholder is tracked. Same pattern already implied for `models_saved/`, which T01.4 ignores by extension (`*.pkl`) rather than by directory.

**Why it matters:** if a future session "corrects" this back to a bare `cache/`, the directory disappears from the repo and the first pipeline run on a fresh clone fails on a missing path rather than creating it. Either keep the negation, or make the cache root `mkdir -p` itself at runtime — not neither.

`dataset/` stays a bare directory ignore with no placeholder, deliberately: it is 1.3 GB of read-only input supplied out of band, and an empty `dataset/` in a clone would be misleading.

**Files:** `.gitignore`.

---

## 2026-08-23 — CI added, push cadence tightened, and a hole in yesterday's own statistics fix

Three changes, one of which was correcting a fix made hours earlier in the same session.

### The statistics fix was incomplete — the data it needed did not exist

T81.4 was amended to require repeated 5x5 CV so the paired tests would have n=25 instead of n=5. But **nothing produced those folds.** T19.2 generated a plain 5-fold map and EXP-A1/A2 ran on it, so Phase 81 would have arrived at a statistical test with 5 observations and an instruction to use 25 — and the likely resolution under time pressure would have been to quietly drop back to n=5 and report the p-value anyway. That is the exact failure the amendment was written to prevent.

**Corrected properly:** repeated 5x5 grouped CV is now the protocol for the whole primary binary track, established once in **T19.2** and consumed everywhere downstream. T63.1 runs all models on the same 25 folds, T63.6 aggregates over 25, T63.7 asserts per-fold values are persisted for Phase 81, and T64.1 uses the repeated map as the outer loop so optimized results pair fold-for-fold with the baselines.

**Cost:** 5x the training time on D1 — 25 fits per model instead of 5. Accepted. These are classical models on 3,240 samples, and the alternative is a statistics chapter that cannot support its own claims. PASCAL and CirCor keep their existing schemes; only the primary track changes.

**The general lesson, worth remembering:** when a requirement is added late to a downstream phase, check that an upstream phase actually produces its input. A requirement with no supplier is worse than no requirement, because it looks satisfied in the plan and gets silently dropped in execution.

### Continuous integration — new Phase 08

CI runs on every push: pytest excluding `needs_data`, ruff and mypy over `src/`, and from Part X the frontend build plus the no-hardcoded-metrics guard rail.

**What CI can and cannot do here matters, and is documented in the workflow header.** The 1.3 GB dataset is gitignored and never reaches GitHub, so every data-dependent test is excluded by design. CI proves the code imports, the logic tests pass, the types check and the frontend builds with no fabricated metric — it proves nothing about the pipeline's behaviour on real audio. That remains the job of the phase gates and the five mega tests, which run locally.

**T08.7 requires CI to actually fail once** — push a deliberately broken test, watch it go red, then fix it. Same principle as T121.2 for the guard rail: a check that has never failed is not proven to work.

### Push cadence tightened from per-Part to per-phase

Previously: commit at phase boundaries, **push once a Part is complete**. Part IX is twenty phases — that is potentially weeks of work sitting only on one Windows machine. Now: **push after every phase**, once its `[TEST]` gate passes, and tag at each mega test (`mega-1-data-integrity` through `mega-5-full-system`) as the known-good rollback points. A red CI build is fixed before the next phase starts.

### `agent.md` merged into `CLAUDE.md`

`Docs/agent.md` was never going to be read automatically — Claude Code reads `CLAUDE.md` at the repo root. Rather than keep a pointer file and the real file separate, all content moved into `CLAUDE.md` and `Docs/agent.md` was deleted. References across `todo.md` and this file updated.

**Counts:** 125 phases / 875 tasks → **126 / 882**. All phases from 08 onward shifted by one; task cross-references renumbered and verified across all three files.

---

## 2026-08-23 — Three findings raised during planning that were never written down until now

Caught in a final sweep before implementation starts. All three had been discussed in conversation and would have been lost in a new chat.

### 1. The statistical plan is underpowered as written

Section 13 of the extraction spec asks for Wilcoxon signed-rank and Friedman tests. With 5-fold CV that is **n=5 paired observations**. Wilcoxon on n=5 cannot reach p<0.05 in some configurations at all — the test is mathematically incapable of the result being asked for, regardless of how large the true effect is.

**Fix:** T81.4 now runs repeated CV (5x5) on the primary binary track before the paired tests, T81.5 requires the number of paired observations beside every test statistic, and T81.7 asserts no p-value is ever reported without its n. T82.5 additionally requires distinguishing *"not significantly better"* from *"the test was underpowered"* — those are different findings and conflating them is the easy mistake.

### 2. PASCAL's `artifact` class is not a cardiac class

Dataset A's four classes are Normal, Murmur, Extra Heart Sound and **Artifact**. Artifact is a *recording-quality* label — it means the recording is unusable, not that the heart made a sound. Reporting a "four-class cardiac classifier" is therefore inaccurate, even though the four-class experiment itself is correct and mandated.

**Fix:** T65.6 now requires this stated in the results section. Worth remembering it is the same conceptual issue as CirCor's Unknown-murmur class, which we resolved by reporting both variants.

### 3. `missing_outputs_report.txt` was referenced but never created

Ground rule 7, the `[-]` skip convention and T51.6 all point at this file, and nothing in 875 tasks created it or said where it lives. A skip recorded against a nonexistent file is a skip that silently disappears.

**Fix:** it is `outputs/missing_outputs_report.txt`, created empty in T01.6, appended to as gaps arise. Explicitly not written at the end from memory — by then the reasons are forgotten, which is exactly how a silent omission happens.

**Files:** `todo.md` T01.6, T65.6, T81.4, T81.5, T81.7, T82.5 and ground rule 7; `CLAUDE.md` gained a "Known weak points in the source documents" section and a "Working with this user" section.

---

## 2026-08-23 — Verification restructure: a test gate on every phase, five mega tests, and explicit task ownership

**User instruction, and the reasoning is right:** building 120 phases straight through and only then testing means a defect introduced early surfaces late, when it is expensive or impossible to fix. In a project whose entire credibility rests on numbers being real, the specific nightmare is a leakage bug or a mis-joined label discovered *after* results are written into a thesis.

**Three structural changes:**

**1. A [TEST] gate on every phase.** Each phase gained a seventh task, `.7`, verifying what that phase produced. These are not boilerplate — each names the specific assertion that matters for that phase. T13.7 asserts the CirCor outcome came from the txt files rather than the CSV; T23.7 asserts the 0.76-second record filters without a padlen error; T68.7 asserts the population-mismatch note is written *before* the metric exists. A phase is not done until its gate passes.

**2. Five 🔴 MEGA TEST phases at the major boundaries** — Phases 29, 52, 76, 104 and 120. Each re-verifies everything important up to that point, and the following Part does not start until it is green. The boundaries fell almost exactly on 25-phase intervals without forcing, because they align with the natural Part boundaries:

| Mega | Phase | Covers | The question it answers |
|---|---|---|---|
| 1 — Data & Preprocessing Integrity | 29 | 01-28 | Are the counts right, the splits leak-free, and preprocessing deterministic? |
| 2 — Feature & Model Integrity | 52 | 30-51 | Is the 138 stable, the pipeline fold-safe, every model round-trippable? |
| 3 — Experiment & Result Integrity | 76 | 53-75 | Do results reproduce from seed, did any search touch a test fold, is any metric suspiciously perfect? |
| 4 — Deliverable Completeness | 104 | 77-103 | Do all 30/35/20/20 assets exist, and does every evidence row resolve? |
| 5 — Full System | 120 | 105-119 | Does the guard rail actually fail a bad build, and does every displayed number match its CSV? |

Two of these deserve note. **Mega 3 T76.3** scans every experiment for any metric at or above 0.99 and forces an investigation against the four causes in the standing rule — that is the "too good to be true" rule made executable rather than aspirational. **Mega 5 T120.2** deliberately inserts a fake metric literal and requires the build to *fail*, then reverts it. The guard rail is proven, not assumed; an unexercised guard rail is indistinguishable from a broken one.

**3. Explicit task ownership.** `[MANUAL]` marks what the user must do (venv creation, GitHub repo, the 1D-CNN scope call, the final search-method choice). `[TEST]` marks agent-run checkpoints. `[TEST/MANUAL]` marks the eleven places where a human must actually look — figures, diagram legibility, dashboard pages, the README walkthrough. Previously all 720 tasks read as agent work, which would have quietly stalled at the first thing the agent cannot do.

**The rule that matters most, now in `CLAUDE.md`:** a test is never weakened to make it pass. If a gate fails, fix the cause; if the gate is wrong, change it deliberately and log why here.

**Counts:** 120 phases / 720 tasks → **125 phases / 875 tasks**, uniformly 7 per phase. All phase numbers from 29 onward shifted; every task cross-reference in `todo.md`, `CLAUDE.md` and this file was renumbered to match and verified (0 mismatched IDs, phases sequential 1-125).

---

## 2026-08-23 — Full comparison against the reference project before deleting it: two real gaps found, three deliberate exclusions

Before `Heart/` is removed, every page, component, backend module and document in it was compared against our `todo.md`. Recording the outcome here because the folder is going away and the analysis should not have to be redone.

### Two genuine gaps in our plan — both now fixed

**1. There was no literature review anywhere in 708 tasks.** Objective 2 is a *locked* objective, and its required evidence in the blueprint is *"Literature review table, preprocessing ablation and feature-family ablation."* We had both ablations and zero literature work. Objective 2 would have failed its own evidence requirement at the T29 objective-to-evidence mapping stage, with nothing to point at.

Found because the reference project ships a third document we did not have — `SYNOPSIS RESEARCH (1).doc`, the original college synopsis. It carries the six objectives in their original wording plus a **10-reference literature review** (Muruganantham 2003, Shui 2004, Segaier 2005, Jiang & Choi 2006, Ahlstrom 2006, Nigam 2008, Babaei & Geranmayeh 2009, Ari 2010 wavelet-LSSVM, Choi 2010 NAR-PSD SVM, Rangayyan & Lehner 1988). That list is now the seed for **Phase 97**.

**2. Segmentation data was parsed and then never used.** T12.5 loads PASCAL's `set_a_timing.csv` and T14.2/T14.4 parse CirCor's per-recording `.tsv` files into S1 / systole / S2 / diastole states. Both tasks said "for later features and figures" — and nothing later consumed either one. Dead artifacts.

Noticed because the reference project has a `SyncedWaveformVisualization` component that animates a cardiac cycle with an S1/systole/S2/diastole phase indicator — except theirs is **synthetic**, drawn from a hardcoded 72 BPM loop. We hold the real per-beat annotations for 3,163 CirCor recordings and were not using them. **Phase 80** now consumes them, and T112.6 drives the dashboard viewer from real segmentation rather than an animation loop.

### Three things from the reference project deliberately NOT adopted

**Live model training from the UI.** They expose `POST /train` with a streaming response and a `TrainingPanel` component. Impressive in a demo, and rejected: a model trained from a button click has no fold map, no logged seed, no manifest entry, and no guarantee it was fold-safe. It would produce exactly the kind of untraceable artifact rules 1 and 5 exist to prevent. The dashboard reads results; it does not manufacture them.

**Extra feature families — TKEO and cardiac power ratios.** Their research page advertises Teager-Kaiser Energy Operator and cardiac power ratio features. Both are legitimate PCG features and neither is going in: the 138-feature composition is locked (see the 2026-08-22 entry), the feature registry asserts exactly 138 names, and changing it after extraction invalidates every cached matrix, trained model and ablation result. Worth revisiting only as future work, and only before Phase 38 runs. Noting also that their own project is internally inconsistent here — the research page says 138 while `PipelineParticles.tsx` says "184 acoustic patterns."

**A stacking meta-ensemble.** Their headline model is a "Stacked Meta-Ensemble" at 95.82%, which does not exist in their codebase. Stacking is a real technique and would be defensible as an *additional* baseline, but the blueprint names optimized-weight soft voting as the proposed final model, and introducing a second ensemble family muddies what is being proposed. Excluded for scope clarity, not because it is unsound.

### Corroboration for the EXP-D1 finding

Their third document, `Final_PhD_Publication_Grade_Dataset_Documentation_Kharat_Project.docx`, states under CirCor's limitations: *"Population is primarily pediatric; results should not be generalized automatically to all adults"* and *"Population: Pediatric and young population up to 21 years."*

So the paediatric nature of CirCor **is** documented — just not in either of the two documents we were given, and not connected anywhere to the Track E cross-dataset instruction that trains on adults. Our T70.1/T70.5 framing stands, now with a citable source rather than only our own age parse.

### Everything else was already covered

Their remaining features map onto existing tasks: SHAP family importance (T80.4, T116.2), MCC (T44.2), ECE and calibration curves (T44.5, T74.x), Friedman test (T81.5), AWGN noise injection (T71.4), confusion-matrix explorer (T112.2), 3D model-comparison visual (T111.4), particle pipeline and cinematic hero (T111.2, T111.3, T111.5), KaTeX equations (T112.5), theme toggle (T109.2), audio playback (T112.6).

**Net effect on the plan:** 118 phases / 708 tasks → **120 phases / 720 tasks**. Phase 80 inserted into Part VIII, Phase 97 into Part IX; everything from 80 onward renumbered, and all task cross-references in `todo.md`, `CLAUDE.md` and this file updated to match.

**The reference project can now be deleted.** Nothing further needs to be read from it.

---

## 2026-08-23 — EXP-D1 is adult-to-paediatric transfer, and the blueprint calls it "compatible"

**This is the most consequential finding so far, and it changes how a mandatory experiment must be reported.**

The blueprint specifies EXP-D1 / Track E as: train on PhysioNet binary, test on *"compatible CirCor binary outcome."* Both label spaces are nominally normal/abnormal, so on paper they match. The populations do not.

| | PhysioNet 2016 | CirCor 2022 |
|---|---|---|
| Median age | **25** (mean 30.4, range 10–90) | — |
| Under 18 | **6 of 2,199 = 0.3%** | — |
| Age categories | — | Child 598, Infant 191, Adolescent 66, Young Adult 7, Neonate 6, unknown 74 |
| Paediatric share | **0.3%** | **~98%** |
| Dominant pathologies | CAD 287, MVP 134, AD, MR, AS — adult degenerative disease | Paediatric murmurs from Brazilian screening campaigns; 70 pregnancies in the cohort |

So the experiment trains on adults with coronary artery disease and mitral valve prolapse, and tests on infants and children. The acoustic signature of adult CAD has close to nothing in common with a paediatric innocent murmur.

**Why it matters, concretely.** A large performance drop is the *expected* result and it is caused by the population mismatch, not by a weakness in PV-MEPCG. If the thesis reports it as "our method does not generalize across datasets," that is a false conclusion drawn from a mis-specified experiment — and it understates the method. Equally, if the drop is reported without the age data, an examiner who knows both datasets will ask why adult training data was tested on children, and there will be no good answer on the spot.

**What changes.** The experiment still runs — it is mandatory, and cross-dataset evidence is genuinely valuable. But:
- **T70.1** now records the population mismatch up front, before any result exists, so the framing cannot be retrofitted to whatever number comes out.
- **T70.5** states explicitly that degradation is attributable to population shift, and that a write-up claiming otherwise is wrong.
- **T116.5** puts it on the dashboard limitations page.
- **T122.5** and **T124.5** put it in the README limitations and the final claims review.

**How to describe it in the paper:** *"cross-dataset transfer from an adult cohort to a predominantly paediatric cohort"* — never *"cross-dataset generalization"* unqualified.

**Verified by:** direct parse of `Online_Appendix_training_set.csv` age column (2,199 populated of 3,153) and `training_data.csv` Age column (942 patients). Numbers above are from the real files, not the documentation.

---

## 2026-08-23 — Frontend architecture: React over Streamlit, and the codegen boundary that makes it safe

**Deviation from the source documents, taken deliberately.** The blueprint's deliverable line says *"Working Streamlit/NiceGUI/Flask interface."* We are building **Next.js 14 (static export) + FastAPI** instead. This is still compliant — a Python API with a JS frontend is a superset of the Flask option — but the reasoning and the mitigation both need recording, because the swap trades away a correctness guarantee that Streamlit gave for free.

**What Streamlit gave for free.** One process, so the UI reads Python objects directly. There is no serialization, no second rounding layer, no cache, and therefore no way for the displayed number to differ from the computed number. Section 19 of the docx requires exactly that: *"Displayed outputs must exactly match backend calculations."*

**What a process boundary costs.** Six specific failure modes, all sharing one shape — *the UI shows a plausible number that is wrong*:

| Risk | Mechanism |
|---|---|
| NaN / Inf | `json.dumps(nan)` emits bare `NaN`; `JSON.parse` rejects it. Works on most records, fails on the degenerate ones FE-04 exists to track. |
| numpy types | `int64` / `float32` are not JSON-serializable |
| Rounding drift | Python writes 0.8474 to the thesis table, JS `toFixed(2)` shows 0.85 in the demo — thesis and screenshot disagree |
| Stale cache | A client cache serves an old response after artifacts are regenerated |
| Class-order scramble | A confusion matrix re-sorted client-side is valid-looking and wrong |
| Silent empty fetch | A failed request renders an empty chart, which reads as "no signal" rather than "the request died" |

Note this is **not** caused by the choice of Vite, Next, Flask or FastAPI. It is caused by crossing a process boundary at all. Flask would be slightly worse, since it has the same `NaN` behaviour with no Pydantic layer to fix it centrally.

**The mitigation — build-time codegen, not runtime fetch.** The key observation is that almost nothing in this dashboard is live computation. All 30 tables and all 35 graph source CSVs are written by Part IX before the frontend runs. So they are never turned into an API:

```
outputs/*.csv  →  scripts/06_export_frontend_data.py  →  frontend/lib/generated/*.json  →  imported by pages
```

Numbers are formatted **once, in Python**, using the same rounding rules as the thesis tables (T84.6). NaN and Inf are coerced to `null` at export. The manifest carries the run id and git commit. Pages import; they never fetch and never round. That removes every risk in the table above for every precomputed value, and it ships the data as literals in the bundle — so the performance profile matches a hardcoded page, with none of the exposure.

Runtime JSON survives for exactly one thing: **live inference** (`POST /predict`). One endpoint, one schema, small enough to test properly. That is the entire remaining correctness surface.

**Enforced, not trusted.** `scripts/07_check_no_hardcoded_metrics.py` (T118.1) fails the build if any page declares a numeric metric literal. T118.3 crawls the built site, extracts every rendered metric and diffs it against the source CSV. T118.4 makes that a **hard gate before screenshots** — no screenshot of an unaudited page.

**Cost accepted:** Part X grows from 9 phases / 54 tasks to 15 / 90; project total 672 → 708. Node joins the toolchain, mitigated by committing `frontend/out/` so a grader without Node can still serve the dashboard from FastAPI alone.

**Files:** `todo.md` Part X rewritten (Phases 99–113), Part XI renumbered (114–118), plus T01.1, T01.4, T02.4, T02.6, T95.4 edited.

---

## 2026-08-23 — What the reference project in `Heart/` got wrong, and why the guard rail exists

The user supplied a parallel implementation of this same brief at `Heart/` (Next.js 14 + FastAPI, `pcg_heart_sound_ml/`). It is worth studying for its visual work and worth studying harder for its failure mode, which is precisely the one our codegen boundary is built to prevent.

**Their UI is genuinely good** — Three.js heart models, GSAP scroll pipeline, Plotly/ECharts/D3, KaTeX, Framer Motion, Zustand. We are copying that layer wholesale (Phase 105, Phase 106).

**Their numbers are hardcoded TypeScript literals**, and they do not match their own pipeline output:

| Claim in `frontend/app/analytics/page.tsx` | What their experiments actually produced |
|---|---|
| PhysioNet **95.82%**, "Stacked Meta-Ensemble", SOTA | best real run **91.26%**; no such ensemble exists in the codebase |
| PASCAL A **95.10%**, PASCAL B **97.10%** | real PASCAL runs: **58.06%** accuracy, **43.21% balanced accuracy** — below chance |
| CirCor external validation **93.65%** | **zero CirCor experiments were ever run** |
| XGBoost **89.94%** | xgboost is not in their `requirements.txt` |

Three further observations, each of which maps to a rule we already have:

- **`current_metrics.json` reads `{"accuracy": 1.0, "precision": 1.0, "recall": 1.0}` on 50 samples.** This is the exact case the blueprint warns about (*"Do not report perfect accuracy unless independently verified"*) and the exact case our standing rule "a metric at or near 1.0 is a bug report" exists for.
- **Their split config is `test_size: 0.25, stratify: true` with no group parameter at all** — so there is no subject-wise grouping anywhere, on any dataset, including CirCor where patient IDs are native and free.
- **12 experiments, 8 of which are exact duplicates** (four identical PASCAL SVM runs, three identical PhysioNet SVM runs), and every single one used SVM. No RF, no GB, no ensemble, despite all three being present in their code.

**The lesson is not "avoid React."** Their hardcoding did not make the UI better; it only made it wrong. Visual quality and numerical integrity are independent axes, and they optimised one while abandoning the other. The specific mechanism was that a human typed metrics into a page because the real pipeline had not produced them yet — and nothing in the build could tell.

**That is unfalsifiable by inspection at scale**, which is why T118.1 is a build-failing grep rather than a review checklist. A page that imports from `generated/` cannot claim a CirCor result that was never computed, because the export step would have had no CSV to read.

**The folder was deleted on 2026-08-23 after the full comparison above.** Nothing in it is a source of truth for this project: not a metric, not a results table, not a claim about model performance. What was worth taking from it — the visual approach and the component decomposition — is already captured in Phases 105-108, and the two gaps it exposed are Phases 80 and 97.

---

## 2026-08-22 — Dataset audit before any code: five findings that change the implementation

Ran a full audit of `dataset/` against both source documents before writing a line of the pipeline. Everything needed is present — **7,536 recordings, ~43 hours, 1.3 GB, zero unreadable or corrupt files, all mono**. Five findings are non-obvious enough that a future session would waste real time rediscovering them.

### 1. CirCor's `Outcome` label is not where you would look for it

`training_data.csv` has 22 columns — Patient ID, Locations, Age, Sex, Height, Weight, Pregnancy status, Murmur, the ten systolic/diastolic murmur descriptors, Campaign, Additional ID — and **no Outcome column at all**.

The clinical outcome exists only inside the per-patient `.txt` files, as a `#Outcome: Normal` / `#Outcome: Abnormal` line among the other `#Key: Value` fields. Distribution across the 942 patients: **Normal 486 / Abnormal 456** — close to balanced, unlike everything else in this project.

**Why it matters:** two mandatory experiments depend on this label. EXP-C2 (CirCor outcome) and EXP-D1 (cross-dataset: train PhysioNet binary, test CirCor outcome) both die without it. A loader built around the CSV — the obvious choice, since it is the only file that looks like a label file — would have produced a murmur-only pipeline and the gap would not have surfaced until Part VII. Parse the `.txt` files; the CSV is for demographics only.

### 2. CirCor's directory is doubled

The real path is `dataset/archive/training_data/training_data/` — a folder named `training_data` containing a folder named `training_data`, which holds all 10,431 files flat (3,163 wav + 3,163 hea + 3,163 tsv + 942 txt). Not a typo, not a mistake to correct. Any path built as `archive/training_data/*.wav` matches nothing and returns an empty file list rather than an error, which is the worst possible failure mode.

### 3. PASCAL set_b's CSV filenames have **zero** overlap with the files on disk

`set_b.csv` lists `set_b/Btraining_extrastole_127_1306764300147_C2.wav`. The disk holds `extrastole__127_1306764300147_C2.wav`. Different prefix, different separator — the intersection of the two name sets is **0 of 656**. A naive join produces an empty labeled dataset, again with no error.

set_a is only partly affected: 124 of 176 names match, and the 52 that don't are the unlabelled `Aunlabelledtest__*` files.

**The fix that avoids the whole problem:** `dataset/Heartbeat_Sound/` organises the same audio into `artifact/ extrahls/ extrastole/ murmur/ normal/ unlabel/` folders, so the class comes from the directory and no filename parsing is needed. Its counts reconcile exactly with the CSVs (murmur 129 = 34 set_a + 95 set_b; normal 351 = 31 + 320; artifact 40; extrahls 19; extrastole 46; unlabel 247 = 52 + 195). Use the folder structure as the authoritative label source and cross-check against the CSVs as a QA step, not the other way round.

### 4. `Heartbeat_Sound/` is a complete duplicate — and a live trap

832 files, **832 of 832 filenames also present in `set_a` + `set_b`**. It is not an extra dataset; it is the same audio re-foldered, 154 MB of redundancy.

**The trap:** a glob over `dataset/**/*.wav` picks up every PASCAL recording twice. Nothing errors. The multiclass training set silently doubles, the same recording lands in both train and test, and PASCAL macro-F1 comes back implausibly high — which, per the standing rule above, would then have to be chased down from the symptom. Excluded from all supervised tracks by an explicit rule in Phase 16, not by hoping nobody globs too widely.

### 5. Subject IDs are only partly recoverable, and the gaps are unequal

Leakage control is a hard requirement, so what is actually available matters:

| Dataset | Subject grouping | Basis |
|---|---|---|
| CirCor | **Native** — 942 patient IDs | Patient ID field. Clean GroupKFold. |
| PASCAL B | **Derivable** — 167 groups | Numeric id in filename `<label>__(\d+)_<ts>_<loc>` |
| PhysioNet training-b | **Native** — 106 subjects | Subject ID column in the online appendix |
| PhysioNet training-a | **Derivable** | `Original record name` pattern `C45S1` → subject C45 |
| PhysioNet training-c | **Derivable** | `id17` pattern |
| PhysioNet training-d/e/f | **Not available** | Treat each record as its own group; flag `subject_derived=False` |
| PASCAL A | **Not available** | Timestamp-only filenames carry no subject information |

The important detail is that the **Subject ID column in the appendix is populated for training-b only** — 490 of 3,153 rows. Reading that column and concluding "PhysioNet has no subject IDs" is the easy wrong answer; the `Original record name` column is where the rest of the information actually is, and it is populated for all 3,153 annotated rows.

**Consequence to state in the write-up:** PASCAL A results are record-level, not subject-level, and cannot be claimed otherwise. This is a real limitation of the dataset, not of the method.

### Two count discrepancies against the source documents

| Figure | Blueprint says | Actually on disk |
|---|---|---|
| CirCor | 5,272 recordings / 1,568 subjects | **3,163 recordings / 942 patients** |
| PhysioNet 2016 | ~3,126 recordings | **3,240 training (a–f) + 301 validation** |

The CirCor gap is not a missing download — the public v1.0.3 release *is* 942 patients. The remaining ~626 subjects were the hidden Challenge test set and were never published. Both figures go into the audit report as documented discrepancies rather than being quietly substituted. See the standing rule above.

### Sampling rates and durations — three different native rates, and the extremes are awkward

| Dataset | Native fs | Duration min / median / max |
|---|---|---|
| PhysioNet train | 2000 Hz | 5.31 s / 20.83 s / 122.00 s |
| PhysioNet validation | 2000 Hz | 5.31 s / 20.69 s / 122.00 s |
| PASCAL set_a | **44100 Hz** | 0.94 s / 8.88 s / 9.00 s |
| PASCAL set_b | 4000 Hz | **0.76 s** / 4.95 s / 27.87 s |
| CirCor | 4000 Hz | 5.15 s / 21.46 s / 64.51 s |

Two things follow. **PASCAL A needs a 22.05× decimation** from 44.1 kHz to the 2 kHz target — large enough that a naive `x[::22]` would alias badly; it needs a proper anti-aliased resampler and a swept-sine check (T22.4). And **the short end is genuinely short**: 0.76 s at 2 kHz is 1,520 samples, which is below what a 5-level db4 decomposition or an MFCC delta window wants. Those guards (T33.6, T35.6) are not defensive padding — they will fire on real records.

### Bonus finding: PhysioNet carries per-recording noise and murmur annotations nobody asked for

The online appendix has coded columns for murmur strength, arrhythmia, respiration noise, ambient noise, recording noise and abdominal sounds, plus `REFERENCE-SQI.csv` signal-quality flags per subset. The noise-robustness track (EXP-E1) was going to rely purely on computed proxies; these give an **independent human-annotated reference to calibrate those proxies against**, which makes T20 substantially more defensible. Parsed in Phase 09, used in T71.3.

**Files this audit produced:** none yet — findings only, no code written. `Docs/todo.md` (672 tasks, 112 phases) and `CLAUDE.md` were written from them.

---

## 2026-08-22 — The 138-feature composition was never specified, so it is now locked here

Both source documents say "approximately 138 features from time, frequency, MFCC, chroma, DWT and envelope domains" and never break that number down. Table 4 / T05 requires an exact inventory, and the ablation tracks (A1–A6) require the families to be cleanly separable, so an arbitrary split chosen later would make every earlier result unreproducible.

**Locked composition — 138 exactly:**

| Family | Count | Composition |
|---|---|---|
| Time | 24 | 8 basic stats, 2 shape, 4 energy, 2 ZCR, 5 complexity (Shannon, sample entropy, Hjorth ×3), 3 autocorr/duration |
| Frequency | 22 | centroid/bandwidth/rolloff85/rolloff95/flatness/flux each mean+std (12), spectral entropy, dominant freq, peak power, total power (4), 6 band powers |
| MFCC | 39 | 13 mean + 13 std + 13 delta-mean |
| Chroma | 24 | 12 mean + 12 std |
| DWT (db4, 5-level) | 24 | 6 sub-bands (cA5, cD5–cD1) × 4 stats (energy, std, entropy, mean-abs) |
| Envelope | 5 | Hilbert envelope mean, std, skew, kurtosis, peak rate |

**Two caveats worth recording now rather than being asked about later.** Chroma is a musical-pitch construct with no physiological meaning for heart sounds — it is included because the source documents mandate it, and it is defensible as a generic harmonic-distribution descriptor, but the write-up should say that plainly rather than implying a cardiac rationale. And the mel filterbank has to be configured for `fmax` ≤ 400 Hz against a 2 kHz signal; the librosa defaults assume speech at 22 kHz and would put most filters above Nyquist.

**Why this is locked:** the feature registry enforces a fixed ordering (T30.3) and a test asserts exactly 138 names with no duplicates (T30.6). Changing the composition after features are extracted invalidates every cached matrix, every trained model, and every ablation result.

---

## 2026-08-22 — Project start

`todo.md` written: 672 atomic tasks across 112 phases in 11 parts, 6 tasks per phase. Backend and pipeline first (Parts I–IX); dashboard deferred to Part X on the user's instruction. `CLAUDE.md` written. Nothing built yet — Phase 01 is the first implementation task.
