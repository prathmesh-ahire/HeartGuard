#!/usr/bin/env bash
# Run the PV-MEPCG / PulseVision test suite (Phase 06, task T06.5).
#
# Runs pytest from the project's .venv, so the suite cannot silently pick up a
# different interpreter's packages.
#
# By default `slow` tests are skipped and `needs_data` tests run only when
# dataset/ is present. The header line of every run states which.
#
# Usage:
#   ./scripts/run_tests.sh                     # default run
#   ./scripts/run_tests.sh --slow --coverage
#   ./scripts/run_tests.sh --ci                # reproduce the CI invocation
#   ./scripts/run_tests.sh --no-data           # pretend dataset/ is absent
#   ./scripts/run_tests.sh tests/test_env.py -k librosa
#
# Flags consumed here (--slow, --no-data, --ci, --coverage) are stripped;
# everything else is passed through to pytest untouched.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Windows (Git Bash) and POSIX venv layouts differ.
if [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON=".venv/Scripts/python.exe"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "error: no virtual environment at .venv - create it first (see README Quickstart)." >&2
    exit 1
fi

ARGS=()
SLOW=0
NO_DATA=0
CI_MODE=0
COVERAGE=0

for arg in "$@"; do
    case "$arg" in
        --slow)     SLOW=1 ;;
        --no-data)  NO_DATA=1 ;;
        --ci)       CI_MODE=1 ;;
        --coverage) COVERAGE=1 ;;
        *)          ARGS+=("$arg") ;;
    esac
done

PYTEST_ARGS=()
# CI has no dataset, so data-dependent tests are deselected by design.
[ "$CI_MODE" -eq 1 ] && PYTEST_ARGS+=(-m "not needs_data")
[ "$SLOW" -eq 1 ] && PYTEST_ARGS+=(--runslow)
[ "$NO_DATA" -eq 1 ] && PYTEST_ARGS+=(--no-data)
if [ "$COVERAGE" -eq 1 ]; then
    PYTEST_ARGS+=(--cov=src --cov-report=term-missing --cov-report=html:outputs/logs/htmlcov)
fi
[ "${#ARGS[@]}" -gt 0 ] && PYTEST_ARGS+=("${ARGS[@]}")

echo "PV-MEPCG test suite"
echo "  $PYTHON -m pytest ${PYTEST_ARGS[*]:-}"
echo

set +e
"$PYTHON" -m pytest "${PYTEST_ARGS[@]}"
CODE=$?
set -e

echo
if [ "$CODE" -eq 0 ]; then
    echo "PASSED"
else
    echo "FAILED (exit $CODE)"
fi
exit "$CODE"
