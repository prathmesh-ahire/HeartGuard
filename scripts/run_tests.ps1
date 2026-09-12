<#
.SYNOPSIS
    Run the PV-MEPCG / PulseVision test suite (Phase 06, task T06.5).

.DESCRIPTION
    Runs pytest from the project's .venv, so the suite cannot silently pick up a
    different interpreter's packages.

    By default `slow` tests are skipped and `needs_data` tests run only when
    dataset/ is present. The header line of every run states which.

.PARAMETER Slow
    Include tests marked `slow`.

.PARAMETER NoData
    Pretend dataset/ is absent, so `needs_data` tests skip. This is how CI sees
    the repository -- the 1.3 GB corpus is gitignored and never reaches GitHub.

.PARAMETER Ci
    Reproduce the CI invocation exactly: deselect `needs_data` with -m.

.PARAMETER Coverage
    Produce a coverage report over src/.

.PARAMETER Frontend
    After pytest, run the frontend suite (T118.6 / T120.6): a fresh `npm run build` (the
    metric guard, the exporter, Next, the bundle budget), the Python
    displayed-value audits over the new build, Vitest, and Playwright. The build
    is never skipped, so the browser tests cannot run against a stale site.

.PARAMETER FrontendOnly
    The frontend suite without the full pytest run.

.EXAMPLE
    .\scripts\run_tests.ps1
    .\scripts\run_tests.ps1 -Slow -Coverage
    .\scripts\run_tests.ps1 -Ci
    .\scripts\run_tests.ps1 -Frontend
    .\scripts\run_tests.ps1 -FrontendOnly
    .\scripts\run_tests.ps1 tests/test_constants.py -- -k bijective
#>
[CmdletBinding()]
param(
    [switch]$Slow,
    [switch]$NoData,
    [switch]$Ci,
    [switch]$Coverage,
    [switch]$Frontend,
    [switch]$FrontendOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "No virtual environment at .venv - create it first (see README Quickstart)."
    exit 1
}

$pytestArgs = @('-m', 'pytest')

if ($Ci) {
    # CI has no dataset, so data-dependent tests are deselected by design.
    $pytestArgs += @('-m', 'not needs_data')
}
if ($Slow)   { $pytestArgs += '--runslow' }
if ($NoData) { $pytestArgs += '--no-data' }
if ($Coverage) {
    $pytestArgs += @('--cov=src', '--cov-report=term-missing', '--cov-report=html:outputs/logs/htmlcov')
}
if ($Rest) { $pytestArgs += $Rest }

Write-Host "PV-MEPCG test suite" -ForegroundColor Cyan

$code = 0
if (-not $FrontendOnly) {
    Write-Host ("  " + $python + " " + ($pytestArgs -join ' ')) -ForegroundColor DarkGray
    Write-Host ""
    & $python $pytestArgs
    $code = $LASTEXITCODE
}

if ($Frontend -or $FrontendOnly) {
    # The exporter and both Playwright servers are Python; they must use this
    # interpreter, not whatever `python` resolves to on PATH.
    $env:PATH = (Join-Path $root '.venv\Scripts') + [IO.Path]::PathSeparator + $env:PATH
    $env:PV_PYTHON = $python
    $steps = @(
        @{ Name = 'build (guard, exporter, next, budget)'; Run = { npm run build } },
        @{ Name = 'displayed-value audits'; Run = {
            & $python -m pytest (Join-Path $root 'tests\test_pages_1_3.py') `
                (Join-Path $root 'tests\test_pages_4_6.py') `
                (Join-Path $root 'tests\test_pages_10_12.py') -q -p no:cacheprovider } },
        @{ Name = 'vitest'; Run = { npm run test } },
        @{ Name = 'playwright'; Run = { npm run test:e2e } },
        # Phase 120. Last, and only reachable because the build above ends in
        # the displayed-value audit: the capture's globalSetup refuses unless
        # the site on disk is byte-for-byte the build that audit passed on.
        @{ Name = 'dashboard screenshots (gated)'; Run = {
            & $python (Join-Path $root 'scripts'_dashboard_screenshots.py') } }
    )
    Push-Location (Join-Path $root 'frontend')
    # npm, Next and Playwright's servers write progress to stderr. Under 'Stop',
    # Windows PowerShell 5.1 turns the first such line into a terminating error
    # whenever output is redirected -- it aborted this block on uvicorn's
    # "Started server process" log line. The exit code is the only authority.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        foreach ($step in $steps) {
            Write-Host ""
            Write-Host ("frontend: " + $step.Name) -ForegroundColor Cyan
            & $step.Run
            if ($LASTEXITCODE -ne 0) {
                Write-Host ("frontend step failed: " + $step.Name) -ForegroundColor Red
                if ($code -eq 0) { $code = $LASTEXITCODE }
                break
            }
        }
    } finally {
        $ErrorActionPreference = $previousPreference
        Pop-Location
    }
}

Write-Host ""
if ($code -eq 0) {
    Write-Host "PASSED" -ForegroundColor Green
} else {
    Write-Host ("FAILED (exit " + $code + ")") -ForegroundColor Red
}
exit $code
