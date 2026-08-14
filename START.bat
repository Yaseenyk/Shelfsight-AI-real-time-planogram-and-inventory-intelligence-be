@echo off
REM ============================================================================
REM  ShelfSight AI - START HERE
REM
REM  Double-click this file. It installs everything it needs the first time,
REM  starts the system, and opens the dashboard in your browser.
REM
REM  First run takes 15-25 minutes (it downloads the AI libraries).
REM  Every run after that takes about 30 seconds.
REM
REM  The real work is in scripts\start_all.ps1 -- PowerShell can check that a
REM  service is genuinely answering before moving on, which batch cannot do
REM  reliably. This file exists so the client has something to double-click.
REM ============================================================================
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_all.ps1" %*
if errorlevel 1 (
    echo.
    echo  Startup did not complete. The message above explains why.
    echo.
    pause
)
