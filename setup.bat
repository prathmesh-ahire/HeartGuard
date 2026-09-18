@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  HeartGuard (PV-MEPCG / PulseVision) -- setup
echo ============================================================
echo.
echo This creates a local Python environment (.venv) and installs
echo every pinned package the app needs. It does not touch anything
echo outside this folder.
echo.

rem --- Find a Python 3.11 interpreter -----------------------------------
set "PYEXE="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PYEXE=py -3.11"
)

if not defined PYEXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
        echo !PYVER! | findstr /b "3.11." >nul
        if not errorlevel 1 set "PYEXE=python"
    )
)

if not defined PYEXE (
    echo Could not find Python 3.11 on this machine.
    echo.
    echo Install Python 3.11 from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH" during install^), then run this
    echo script again.
    echo.
    pause
    exit /b 1
)

echo Using Python: !PYEXE!
echo.

rem --- Create the virtual environment -------------------------------------
if exist ".venv\Scripts\python.exe" (
    echo .venv already exists -- skipping creation.
) else (
    echo Creating .venv ...
    !PYEXE! -m venv .venv
    if errorlevel 1 goto :fail
)

set "VENV_PY=.venv\Scripts\python.exe"

echo.
echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo.
echo Installing requirements -- this can take several minutes the first time.
for %%R in (base extra api report) do (
    echo.
    echo   requirements\%%R.txt
    "%VENV_PY%" -m pip install -r "requirements\%%R.txt"
    if errorlevel 1 goto :fail
)

echo.
echo ============================================================
echo  Setup complete.
echo  Double-click "Start HeartGuard.bat" to launch the app.
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo ============================================================
echo  Setup FAILED -- see the error above and try again.
echo ============================================================
echo.
pause
exit /b 1
