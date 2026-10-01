#!/usr/bin/env bash
# One-liner environment bootstrap. Works on Linux, macOS, and Git Bash on Windows.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=${PYTHON:-python3}

if [ ! -d ".venv" ]; then
  echo "==> Creating venv (.venv)"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  . .venv/Scripts/activate
else
  echo "Could not find venv activate script"; exit 1
fi

python -m pip install --upgrade pip==24.3.1
python -m pip install -r requirements.txt
python -m pip install -e .

echo ""
echo "==> Environment ready."
echo "    Activate with: source .venv/bin/activate   (Windows: source .venv/Scripts/activate)"
echo "    Next: cp .env.example .env && bash scripts/download_data.sh"
