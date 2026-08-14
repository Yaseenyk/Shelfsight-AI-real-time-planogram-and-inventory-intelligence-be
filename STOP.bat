@echo off
REM ============================================================================
REM  ShelfSight AI - stop everything started by START.bat
REM
REM  Stops only the processes listening on the two application ports, rather
REM  than killing every python.exe and node.exe on the machine -- the client may
REM  well have other work open.
REM ============================================================================
cd /d "%~dp0"
echo.
echo  Stopping ShelfSight AI...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "foreach ($p in 8000,3000) { try { $c = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction Stop; foreach ($x in $c) { Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host ('      stopped the service on port ' + $p) -ForegroundColor Green } } catch { Write-Host ('      nothing was running on port ' + $p) -ForegroundColor Gray } }"
echo.
echo  Stopped. Your data is safe - run START.bat to start again.
echo.
pause
