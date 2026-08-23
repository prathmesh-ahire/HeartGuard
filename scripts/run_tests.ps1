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

.EXAMPLE
    .\scripts\run_tests.ps1
    .\scripts\run_tests.ps1 -Slow -Coverage
    .\scripts\run_tests.ps1 -Ci
    .\scripts\run_tests.ps1 tests/test_constants.py -- -k bijective
#>
[CmdletBinding()]
param(
    [switch]$Slow,
    [switch]$NoData,
    [switch]$Ci,
    [switch]$Coverage,
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
Write-Host ("  " + $python + " " + ($pytestArgs -join ' ')) -ForegroundColor DarkGray
Write-Host ""

& $python $pytestArgs
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "PASSED" -ForegroundColor Green
} else {
    Write-Host ("FAILED (exit " + $code + ")") -ForegroundColor Red
}
exit $code
