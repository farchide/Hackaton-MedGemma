#!/usr/bin/env bash
# ============================================================
# Digital Twin Tumor -- Launch demo application
# ============================================================
# Usage:
#   bash scripts/run_demo.sh                   # Default (port 7860)
#   bash scripts/run_demo.sh --port 8080       # Custom port
#   bash scripts/run_demo.sh --share           # Public Gradio link
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate venv if it exists
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Set environment
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export DTT_DEMO_DB="${DTT_DEMO_DB:-${PROJECT_ROOT}/.cache/demo.db}"

# Generate demo data if missing
if [ ! -f "$DTT_DEMO_DB" ]; then
    echo "Demo database not found. Generating..."
    mkdir -p "$(dirname "$DTT_DEMO_DB")"
    python scripts/generate_demo_data.py --db-path "$DTT_DEMO_DB"
fi

PORT="${PORT:-7860}"
EXTRA_ARGS=()

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --port)
            shift
            PORT="$1"
            shift
            ;;
        --port=*)
            PORT="${arg#*=}"
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

echo "============================================================"
echo "  Digital Twin Tumor -- Demo"
echo "============================================================"
echo "  Database  : $DTT_DEMO_DB"
echo "  Port      : $PORT"
echo "  URL       : http://localhost:$PORT"
echo "============================================================"

# Try to open browser (non-blocking, best-effort)
(sleep 3 && {
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:$PORT" 2>/dev/null
    elif command -v open &>/dev/null; then
        open "http://localhost:$PORT" 2>/dev/null
    fi
}) &

# Launch
exec python -m digital_twin_tumor \
    --port "$PORT" \
    --demo-db "$DTT_DEMO_DB" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
