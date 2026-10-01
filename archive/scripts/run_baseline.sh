#!/usr/bin/env bash
# Run the zero-shot baseline notebook headless and write a submission CSV.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NOTEBOOK="notebooks/02_baseline_zeroshot.ipynb"
if [ ! -f "$NOTEBOOK" ]; then
  echo "Notebook not found: $NOTEBOOK"
  exit 1
fi

TIMESTAMP=$(date +%Y-%m-%d_%H%M)
OUT_NB="submissions/_runs/baseline_${TIMESTAMP}.ipynb"
mkdir -p "submissions/_runs"

echo "==> Executing $NOTEBOOK (output → $OUT_NB)"
python -m jupyter nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --output "$OUT_NB" \
  --ExecutePreprocessor.timeout=-1
echo "==> Done. Check submissions/ for the CSV."
