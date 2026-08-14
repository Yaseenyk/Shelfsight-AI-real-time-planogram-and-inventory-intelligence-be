<#
    ShelfSight AI - one-command launcher.

    Brings up the whole system from a clean machine: checks prerequisites,
    builds the Python environment, installs both dependency sets, seeds the
    database, starts the API and the dashboard, waits until each is genuinely
    answering, and opens the browser.

    Written for someone who has never seen this project. Every failure path
    explains what to install and where to get it, rather than surfacing a stack
    trace. Re-running is cheap: each step detects work already done and skips it,
    so the second launch takes seconds.

    Invoked by START.bat. Run directly with:
        powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
#>

param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 3000,
    [switch]$NoBrowser,
    [switch]$SkipMl      # skip the ~2GB ML stack (UI works, vision features do not)
)

$ErrorActionPreference = "Stop"
$BackendDir = Split-Path -Parent $PSScriptRoot
$FrontendDir = Join-Path (Split-Path -Parent $BackendDir) "fe"
$TotalSteps = 7

function Write-Step { param([int]$N, [string]$Text) Write-Host "`n[$N/$TotalSteps] $Text" -ForegroundColor Cyan }
function Write-Ok { param([string]$Text) Write-Host "      $Text" -ForegroundColor Green }
function Write-Info { param([string]$Text) Write-Host "      $Text" -ForegroundColor Gray }
function Write-Warn2 { param([string]$Text) Write-Host "      $Text" -ForegroundColor Yellow }

function Fail {
    param([string]$Title, [string[]]$Lines)
    Write-Host "`n  ============================================================" -ForegroundColor Red
    Write-Host "   $Title" -ForegroundColor Red
    Write-Host "  ============================================================" -ForegroundColor Red
    foreach ($l in $Lines) { Write-Host "   $l" -ForegroundColor Yellow }
    Write-Host ""
    Read-Host "  Press Enter to close"
    exit 1
}

function Test-PortBusy {
    param([int]$Port)
    try {
        $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        return ($null -ne $c)
    } catch { return $false }
}

function Wait-ForUrl {
    param([string]$Url, [int]$TimeoutSec = 180, [string]$Label = "service")
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { return $true }
        } catch { Start-Sleep -Seconds 2 }
    }
    return $false
}

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor White
Write-Host "    ShelfSight AI - Real-Time Planogram & Inventory Intelligence" -ForegroundColor White
Write-Host "  ============================================================" -ForegroundColor White

# --------------------------------------------------------------- 1. Python --
Write-Step 1 "Checking Python..."
# Interpreter and its arguments are declared explicitly as typed string arrays.
# Two PowerShell traps make the "clever" versions of this loop dangerous:
#   * $a[1..($a.Count-1)] on a single-element array becomes $a[1..0], and 1..0 is
#     a DESCENDING range, so it hands back the element instead of nothing.
#   * Select-Object -Skip 1 returns a bare String when one item survives, and
#     splatting a scalar String does NOT expand into arguments -- `py` then
#     launches an interactive REPL instead of receiving --version, and the
#     script hangs forever on a machine with a real console.
# Explicit [string[]] literals avoid both.
$PythonExe = $null
$PythonArgs = [string[]]@()
foreach ($candidate in @(
        @{ Exe = "py"; Prefix = [string[]]@("-3") },
        @{ Exe = "python"; Prefix = [string[]]@() },
        @{ Exe = "python3"; Prefix = [string[]]@() }
    )) {
    try {
        $prefix = [string[]]$candidate.Prefix
        $v = & $candidate.Exe @prefix --version 2>&1
        if ($LASTEXITCODE -eq 0 -and "$v" -match "Python (\d+)\.(\d+)") {
            if ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 10) {
                $PythonExe = $candidate.Exe
                $PythonArgs = $prefix
                Write-Ok "Found $v"
                break
            }
        }
    } catch { continue }
}
if (-not $PythonExe) {
    Fail "Python 3.10 or newer is required" @(
        "Download it from:  https://www.python.org/downloads/",
        "",
        "IMPORTANT: on the first installer screen, tick",
        '  "Add python.exe to PATH"',
        "",
        "Then close this window and run START.bat again."
    )
}

# ----------------------------------------------------------------- 2. Node --
Write-Step 2 "Checking Node.js..."
try {
    $nodeVersion = & node --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "not found" }
    if ("$nodeVersion" -match "v(\d+)") {
        if ([int]$Matches[1] -lt 18) {
            Fail "Node.js 18 or newer is required (found $nodeVersion)" @(
                "Download the LTS version from:  https://nodejs.org/"
            )
        }
    }
    Write-Ok "Found Node $nodeVersion"
} catch {
    Fail "Node.js is required but was not found" @(
        "Download the LTS version from:  https://nodejs.org/",
        "Accept every default in the installer.",
        "",
        "Then close this window and run START.bat again."
    )
}

if (-not (Test-Path (Join-Path $FrontendDir "package.json"))) {
    Fail "The dashboard folder was not found" @(
        "Expected this layout:",
        "    Projects\be\   <- this folder",
        "    Projects\fe\   <- dashboard (missing)",
        "",
        "Both folders must sit side by side."
    )
}

# ------------------------------------------------------- 3. Python env/deps --
Write-Step 3 "Preparing the Python environment..."
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Info "Creating a private Python environment (one time)..."
    Push-Location $BackendDir
    & $PythonExe @PythonArgs -m venv .venv
    Pop-Location
    if (-not (Test-Path $VenvPython)) { Fail "Could not create the Python environment" @("Try running START.bat as Administrator.") }
}
Write-Ok "Environment ready"

Write-Step 4 "Installing backend dependencies..."
Push-Location $BackendDir
& $VenvPython -c "import fastapi" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Info "First run - this downloads a few hundred MB and takes 2-5 minutes."
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "Backend dependencies failed to install" @("Check your internet connection and try again.") }
} else {
    Write-Ok "Core dependencies already installed"
}

if (-not $SkipMl) {
    & $VenvPython -c "import torch, ultralytics" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Installing the vision stack (~2 GB). This is the slow part:"
        Write-Info "expect 10-20 minutes on a normal connection. Leave it running."
        & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
        & $VenvPython -m pip install -r requirements-ml.txt
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            Fail "The vision stack failed to install" @(
                "The dashboard can still run without it:",
                "    START.bat  ->  choose 'skip vision' by running:",
                "    powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1 -SkipMl"
            )
        }
    } else {
        Write-Ok "Vision stack already installed"
    }
} else {
    Write-Warn2 "Skipping the vision stack (-SkipMl): photo analysis will be unavailable."
}

# --------------------------------------------------------- 5. Config and DB --
Write-Step 5 "Preparing configuration and database..."
if (-not (Test-Path (Join-Path $BackendDir ".env"))) {
    Copy-Item (Join-Path $BackendDir ".env.example") (Join-Path $BackendDir ".env")
    Write-Ok "Created .env from the template"
} else {
    Write-Ok "Using the existing .env"
}
$FeEnv = Join-Path $FrontendDir ".env.local"
if (-not (Test-Path $FeEnv)) {
    Copy-Item (Join-Path $FrontendDir ".env.local.example") $FeEnv
    Write-Ok "Created the dashboard configuration"
}

& $VenvPython -m app.db.init_db --seed
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "The database could not be prepared" @("Delete shelfsight.db and run START.bat again.") }
Write-Ok "Database ready (sample products loaded)"
Pop-Location

# ------------------------------------------------------ 6. Frontend deps --
Write-Step 6 "Installing dashboard dependencies..."
if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Info "First run - this takes 2-4 minutes."
    Push-Location $FrontendDir
    & npm install --no-audit --no-fund
    $npmFailed = ($LASTEXITCODE -ne 0)
    Pop-Location
    if ($npmFailed) { Fail "Dashboard dependencies failed to install" @("Check your internet connection and try again.") }
} else {
    Write-Ok "Already installed"
}

# ------------------------------------------------------------ 7. Launch --
Write-Step 7 "Starting ShelfSight AI..."

if (Test-PortBusy $ApiPort) {
    Write-Warn2 "Port $ApiPort is already in use - assuming the API is already running."
} else {
    Start-Process -FilePath $VenvPython `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort") `
        -WorkingDirectory $BackendDir -WindowStyle Minimized
    Write-Info "API starting (loading vision models takes 30-90 seconds)..."
}

if (-not (Wait-ForUrl "http://127.0.0.1:$ApiPort/health" 240 "API")) {
    Fail "The API did not start in time" @(
        "Look at the minimised 'python' window in your taskbar for the error.",
        "Most common cause: port $ApiPort is used by another program."
    )
}
Write-Ok "API is answering on http://localhost:$ApiPort"

if (Test-PortBusy $WebPort) {
    Write-Warn2 "Port $WebPort is already in use - assuming the dashboard is already running."
} else {
    Start-Process -FilePath "cmd.exe" `
        -ArgumentList @("/c", "npm", "run", "dev") `
        -WorkingDirectory $FrontendDir -WindowStyle Minimized
    Write-Info "Dashboard starting..."
}

if (-not (Wait-ForUrl "http://127.0.0.1:$WebPort" 180 "dashboard")) {
    Fail "The dashboard did not start in time" @(
        "Look at the minimised 'cmd' window in your taskbar for the error."
    )
}
Write-Ok "Dashboard is answering on http://localhost:$WebPort"

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor Green
Write-Host "    ShelfSight AI is running." -ForegroundColor Green
Write-Host ""
Write-Host "      Dashboard : http://localhost:$WebPort" -ForegroundColor White
Write-Host "      API docs  : http://localhost:$ApiPort/docs" -ForegroundColor White
Write-Host ""
Write-Host "    To stop it: run STOP.bat, or close the two minimised" -ForegroundColor Gray
Write-Host "    windows in your taskbar." -ForegroundColor Gray
Write-Host "  ============================================================" -ForegroundColor Green
Write-Host ""

if (-not $NoBrowser) { Start-Process "http://localhost:$WebPort" }
