# Changelog — PV-MEPCG / PulseVision

Phase-completion log. One entry per completed phase, newest first.

**This file records *what shipped*.** It is not the place for reasoning. Bugs
found, dataset surprises, deliberate deviations and any decision needing a
"why" go in [`Docs/note.md`](Docs/note.md); task status lives in the checkboxes
in [`Docs/todo.md`](Docs/todo.md). A phase is only listed here once its `[TEST]`
gate has actually passed.

## Entry format

```
## Phase NN — Title
**Gate:** T`NN`.7 — <what the gate asserted> — PASS
**Added:** <files/capabilities that did not exist before>
**Changed:** <files that already existed>
**Notes:** <one line, or "none"; full reasoning goes in note.md>
```

---

## Housekeeping — root reorganisation (before Part II)
**Gate:** ruff clean, mypy clean, 136 passed / 2 skipped, verify_env PASSED
**Added:** `pyproject.toml` (ruff + mypy + pytest config consolidated), `requirements/` (base, extra, api, report, dev)
**Changed:** `.github/workflows/ci.yml`, `scripts/verify_env.py` (now covers all five requirements files, plus a STUB_ONLY exemption), `src/utils/run_manifest.py`, `tests/test_env.py`, `README.md`, `CLAUDE.md`
**Removed:** `ruff.toml`, `mypy.ini`, `pytest.ini` (merged into `pyproject.toml`)
**Notes:** Root went from 14 loose files to 7. `conftest.py` must stay at root — pytest only honours `pytest_addoption` in the rootdir conftest. `report.log` is written by `pyswarms` on import, not by this project; it stays gitignored. See `Docs/note.md`.

## Phase 08 — Continuous Integration
**Gate:** T08.7 — CI has gone red on a real failure (run #1) and green after the fix (runs #2, #3) — PASS
**Added:** `.github/workflows/ci.yml` (tests / quality / frontend jobs), `requirements-dev.txt`, `ruff.toml`, `mypy.ini`, CI badge in `README.md`, `.gitattributes`
**Changed:** `tests/test_env.py`, `tests/test_harness.py` (two assertions encoded local-machine assumptions), `.gitignore` (`Docs/` now untracked)
**Notes:** The gate's "and blocks" clause was amended — blocking is enforced by the standing rule that a red build is fixed before the next phase, not by branch protection. See the Phase 08 entry in `Docs/note.md`.

## Phase 07 — Version Control
**Gate:** T07.7 — no dataset or cache file staged, nothing over 50 MB tracked, remote tree matches local — PASS
**Added:** `CHANGELOG.md`; git repository on `main` with `origin` set to the GitHub remote
**Changed:** none
**Notes:** First commit. `.gitignore` was verified against `dataset/`, `cache/`, `.venv/`, `node_modules/` and `.next/` *before* anything was staged; every file was staged by name rather than with `git add -A`.

## Phase 06 — Test Harness
**Gate:** T06.7 — full suite green with `slow` and `needs_data` markers correctly skipping — PASS (132 passed, 1 skipped)
**Added:** `pytest.ini`, root `conftest.py`, `tests/conftest.py`, `tests/fixtures/make_synthetic_pcg.py`, `tests/test_env.py`, `tests/test_harness.py`, `scripts/run_tests.ps1`, `scripts/run_tests.sh`
**Changed:** none
**Notes:** All five marker configurations verified, including the CI deselect form. `--strict-markers` proven by making it fail on a deliberate typo.

## Phase 05 — Constants & Label Vocabularies
**Gate:** T05.7 — all label maps bijective, no cross-task vocabulary overlap — PASS (39 tests)
**Added:** `src/utils/constants.py`, `tests/test_constants.py`, `requirements-dev.txt`, root `conftest.py`
**Changed:** `outputs/configs/pip_freeze.txt`
**Notes:** Labels are addressable only as `(task, code)` pairs; there is no bare-integer decode. `map_physionet_reference` no longer truncates non-integral floats.

## Phase 04 — Reproducibility & Logging
**Gate:** T04.7 — throwaway job end to end; seed applied, log written, manifest records git state and package versions, evidence row appended — PASS (77 checks)
**Added:** `src/utils/seed.py`, `src/utils/logging_setup.py`, `src/utils/run_manifest.py`, `src/utils/timing.py`, `src/utils/io.py`, `src/utils/evidence.py`
**Changed:** none
**Notes:** Manifest schema 2 deduplicates config snapshots by content hash. Every writer is atomic.

## Phase 03 — Configuration System
**Gate:** T03.7 — every YAML loads, dotted access correct, 18 deliberately malformed keys each caught — PASS (94 checks)
**Added:** `configs/paths.yaml`, `configs/signal.yaml`, `configs/features.yaml`, `configs/models.yaml`, `configs/experiments.yaml`, `src/utils/config.py`
**Changed:** none
**Notes:** Validation encodes project invariants (the locked 138, Nyquist limits, seed 42, fold safety, the five label spaces), not generic type checks.

## Phase 02 — Python Environment
**Gate:** T02.7 — every pinned package installed at its version and importing; Node LTS and npm resolve — PASS (exit 0, 1 advisory)
**Added:** `requirements.txt`, `requirements-extra.txt`, `requirements-api.txt`, `requirements-report.txt`, `scripts/verify_env.py`, `outputs/configs/pip_freeze.txt`, `.venv/`
**Changed:** `README.md` (quickstart)
**Notes:** All pins resolved by a real install, none guessed. kaleido cannot render without a Chrome install — reported as an advisory, unused by the pipeline.

## Phase 01 — Repository Skeleton
**Gate:** T01.7 — full `src/` and `outputs/` trees present, every empty dir has a `.gitkeep`, `.gitignore` excludes `dataset/` before staging — PASS (98 checks)
**Added:** `src/` (13 subpackages), `outputs/` (20 sections), `models_saved/`, `cache/`, `tests/`, `scripts/`, `notebooks/`, `frontend/`, `.gitignore`, `README.md`, `outputs/missing_outputs_report.txt`
**Changed:** none
**Notes:** none
