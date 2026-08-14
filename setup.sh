#!/usr/bin/env bash
# =============================================================================
#  ShelfSight AI - Linux/macOS launcher
#
#    ./setup.sh [start|stop|logs|evaluate|local|reset|help]
#
#  Checks prerequisites and explains what to install rather than failing
#  halfway through with a Docker stack trace.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

COMMAND="${1:-start}"

info()  { printf '  %s\n' "$*"; }
ok()    { printf '  \033[32m[OK]\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m[X]\033[0m %s\n' "$*" >&2; }

usage() {
    cat <<'EOF'

  ShelfSight AI
  -------------
  ./setup.sh start      Build and start everything (default)
  ./setup.sh stop       Stop the system, keeping all data
  ./setup.sh logs       Show live logs
  ./setup.sh evaluate   Run benchmarks and publish paper figures
  ./setup.sh local      Install and run without Docker
  ./setup.sh reset      DELETE all data and start clean

EOF
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        fail "Docker was not found."
        info "Install it from https://www.docker.com/products/docker-desktop/"
        info "or run without Docker:  ./setup.sh local"
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        fail "Docker is installed but not running. Start it and retry."
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        fail "The 'docker compose' plugin is missing (Compose v2 required)."
        exit 1
    fi
    ok "Docker is ready."
}

require_frontend() {
    local context="${FRONTEND_CONTEXT:-../fe}"
    if [ ! -f "$context/Dockerfile" ]; then
        fail "Frontend not found at $context"
        info "Expected layout:  Projects/be/ (here) and Projects/fe/"
        info "Or set FRONTEND_CONTEXT=/path/to/frontend"
        exit 1
    fi
}

case "$COMMAND" in
    start)
        require_docker
        require_frontend
        info ""
        info "Building images. First run downloads ~2 GB and takes 5-15 minutes."
        docker compose build
        info "Starting services..."
        docker compose up -d
        cat <<'EOF'

  ============================================================
    ShelfSight AI is starting.

      Dashboard : http://localhost:3000
      API docs  : http://localhost:8000/docs

    The backend takes 1-2 minutes to load its models on first
    boot. If the dashboard shows "API unreachable", wait and
    refresh. Watch progress with:  ./setup.sh logs
  ============================================================

EOF
        ;;

    stop)
        docker compose down
        info "Stopped. Your data is safe - './setup.sh start' resumes."
        ;;

    logs)
        docker compose logs -f --tail=100
        ;;

    evaluate)
        if [ ! -x ".venv/bin/python" ]; then
            fail "No local environment found. Run './setup.sh local' first."
            exit 1
        fi
        .venv/bin/python models/export_pipeline.py metrics --suites all
        info "Figures written to docs/publication_metrics/"
        ;;

    local)
        if ! command -v python3 >/dev/null 2>&1; then
            fail "Python 3 was not found. Install Python 3.10 or newer."
            exit 1
        fi
        [ -d .venv ] || python3 -m venv .venv
        .venv/bin/python -m pip install --upgrade pip
        info "Installing CPU PyTorch (large download, please wait)..."
        .venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
            torch torchvision
        .venv/bin/python -m pip install -r requirements-ml.txt
        .venv/bin/python -m app.db.init_db --seed
        info ""
        info "Starting the API on http://localhost:8000/docs  (Ctrl+C to stop)"
        info "Start the dashboard separately:  cd ../fe && npm install && npm run dev"
        .venv/bin/python -m uvicorn app.main:app --port 8000
        ;;

    reset)
        printf '  WARNING: this deletes the database, uploads and trained weights.\n'
        printf '  Type yes to continue: '
        read -r confirm
        [ "$confirm" = "yes" ] || { info "Cancelled."; exit 0; }
        docker compose down -v
        rm -f shelfsight.db shelfsight.db-wal shelfsight.db-shm
        info "Reset complete."
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        fail "Unknown command '$COMMAND'"
        usage
        exit 1
        ;;
esac
