@echo off
REM ============================================================================
REM  ShelfSight AI - Windows launcher
REM
REM  Double-click this file, or run:  setup.bat [command]
REM
REM  Commands:  start | stop | logs | evaluate | local | reset | help
REM  No command starts the system.
REM
REM  Written for a viva demo: it checks prerequisites first and explains what to
REM  install, rather than failing halfway through with a Docker stack trace.
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "COMMAND=%~1"
if "%COMMAND%"=="" set "COMMAND=start"

if /i "%COMMAND%"=="help" goto :help
if /i "%COMMAND%"=="stop" goto :stop
if /i "%COMMAND%"=="logs" goto :logs
if /i "%COMMAND%"=="evaluate" goto :evaluate
if /i "%COMMAND%"=="local" goto :local
if /i "%COMMAND%"=="reset" goto :reset
if /i "%COMMAND%"=="start" goto :start

echo Unknown command "%COMMAND%".
goto :help

:help
echo.
echo   ShelfSight AI
echo   -------------
echo   setup.bat start      Build and start everything (default)
echo   setup.bat stop       Stop the system, keeping all data
echo   setup.bat logs       Show live logs
echo   setup.bat evaluate   Run benchmarks and publish paper figures
echo   setup.bat local      Install and run without Docker
echo   setup.bat reset      DELETE all data and start clean
echo.
goto :eof

:start
echo.
echo  Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [X] Docker was not found.
    echo.
    echo      Install Docker Desktop from:
    echo        https://www.docker.com/products/docker-desktop/
    echo.
    echo      Then run this file again. To run without Docker instead:
    echo        setup.bat local
    echo.
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [X] Docker is installed but not running.
    echo      Start Docker Desktop, wait for the whale icon to settle, then retry.
    echo.
    pause
    exit /b 1
)

if not exist "..\fe\Dockerfile" (
    echo.
    echo  [X] The frontend folder was not found next to this one.
    echo      Expected layout:
    echo          Projects\be\   ^<- this folder
    echo          Projects\fe\   ^<- frontend
    echo.
    echo      Clone the frontend repository beside this one and retry.
    echo.
    pause
    exit /b 1
)

echo  [OK] Docker is ready.
echo.
echo  Building images. First run downloads ~2 GB and takes 5-15 minutes.
echo.
docker compose build
if errorlevel 1 goto :failed

echo.
echo  Starting services...
docker compose up -d
if errorlevel 1 goto :failed

echo.
echo  ============================================================
echo    ShelfSight AI is starting.
echo.
echo      Dashboard : http://localhost:3000
echo      API docs  : http://localhost:8000/docs
echo.
echo    The backend takes 1-2 minutes to load its models on first
echo    boot. If the dashboard shows "API unreachable", wait and
echo    refresh. Watch progress with:  setup.bat logs
echo  ============================================================
echo.
timeout /t 5 >nul
start "" http://localhost:3000
goto :eof

:stop
docker compose down
echo.
echo  Stopped. Your data is safe - run "setup.bat start" to resume.
goto :eof

:logs
docker compose logs -f --tail=100
goto :eof

:evaluate
if not exist ".venv\Scripts\python.exe" (
    echo  No local environment found. Run "setup.bat local" first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe models\export_pipeline.py metrics --suites all
echo.
echo  Figures written to docs\publication_metrics\
goto :eof

:local
echo.
echo  Setting up a local (non-Docker) environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo  [X] Python was not found. Install Python 3.10 or newer from python.org
    echo      and tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv" python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
echo  Installing CPU PyTorch (large download, please wait)...
.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
if errorlevel 1 goto :failed

.venv\Scripts\python.exe -m app.db.init_db --seed
echo.
echo  Starting the API on http://localhost:8000/docs  (Ctrl+C to stop)
echo  Start the dashboard separately:  cd ..\fe  ^&^&  npm install  ^&^&  npm run dev
echo.
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
goto :eof

:reset
echo.
echo  WARNING: this deletes the database, uploaded frames and trained weights.
set /p CONFIRM="Type yes to continue: "
if /i not "%CONFIRM%"=="yes" (
    echo  Cancelled.
    goto :eof
)
docker compose down -v
if exist shelfsight.db del /q shelfsight.db
echo  Reset complete.
goto :eof

:failed
echo.
echo  [X] Something failed above. The most common causes:
echo        - Docker Desktop not fully started
echo        - No internet connection (first build downloads dependencies)
echo        - Ports 3000 or 8000 already in use
echo.
pause
exit /b 1
