@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv was not found. Run setup.bat first.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo  Starting HeartGuard -- this window is the running server.
echo  Close it, or press Ctrl+C, to stop the app.
echo ============================================================
echo.

rem Open the browser a few seconds after the server starts, without
rem blocking this window (which runs the server itself, in the
rem foreground, so its logs are visible and Ctrl+C stops it cleanly).
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8000"

".venv\Scripts\python.exe" -m uvicorn src.api.main:app --port 8000

pause
