#!/usr/bin/env bash
# ============================================================
# Digital Twin Tumor -- One-command setup
# ============================================================
# Usage:
#   bash scripts/setup.sh          # CPU-only (default)
#   bash scripts/setup.sh --gpu    # Include GPU/MedGemma deps
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--gpu" ]; then
        GPU_MODE=true
    fi
done

echo "============================================================"
echo "  Digital Twin Tumor -- Setup"
echo "============================================================"
echo "  Project root : $PROJECT_ROOT"
echo "  GPU mode     : $GPU_MODE"
echo "============================================================"

cd "$PROJECT_ROOT"

# 1. Create virtual environment
VENV_DIR="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Virtual environment already exists."
fi

# Activate
source "$VENV_DIR/bin/activate"

# 2. Upgrade pip and install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip wheel setuptools

if [ "$GPU_MODE" = true ]; then
    pip install -e ".[gpu,dev]"
else
    pip install -e ".[dev]"
fi

# pandas is used in the agentic workflow tab
pip install pandas

# 3. Generate demo data
echo "[3/4] Generating demo data..."
DEMO_DB="$PROJECT_ROOT/.cache/demo.db"
mkdir -p "$PROJECT_ROOT/.cache"

if [ -f "$DEMO_DB" ]; then
    echo "  Demo database already exists at $DEMO_DB"
else
    PYTHONPATH="$PROJECT_ROOT/src" python scripts/generate_demo_data.py --db-path "$DEMO_DB"
fi

# 4. Run quick smoke test
echo "[4/4] Running smoke test..."
if python -c "from digital_twin_tumor.data.synthetic import generate_all_demo_data; print('Import OK')"; then
    echo "  Smoke test passed."
else
    echo "  WARNING: Smoke test failed. Check installation."
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  To activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  To run the demo app:"
echo "    bash scripts/run_demo.sh"
echo ""
echo "  Or manually:"
echo "    DTT_DEMO_DB=.cache/demo.db python -m digital_twin_tumor --port 7860"
echo ""
echo "  To run tests:"
echo "    pytest tests/ -v"
echo ""
echo "============================================================"
