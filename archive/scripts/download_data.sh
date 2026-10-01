#!/usr/bin/env bash
# Download competition data + fastText lid.176 + any stable external assets.
#
# Run this once on data-drop day (2026-04-24) after Zindi unlocks the files.
# The competition CSVs are not in this script — download them manually from
# the Zindi competition page to avoid shipping a hardcoded URL that rots.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
EXTERNAL="$ROOT/data/external"

mkdir -p "$RAW" "$EXTERNAL"

echo "==> Expected files in $RAW (download from Zindi):"
echo "    Train.csv"
echo "    Test.csv"
echo "    SampleSubmission.csv"
echo ""
echo "    Zindi page: https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge"
echo ""

missing=0
for f in Train.csv Test.csv SampleSubmission.csv; do
  if [ ! -f "$RAW/$f" ]; then
    echo "    [MISSING] $f"
    missing=1
  else
    echo "    [ok]      $f"
  fi
done

# fastText lid.176 — for language_id.py fallback.
LID_PATH="$EXTERNAL/lid.176.bin"
if [ ! -f "$LID_PATH" ]; then
  echo ""
  echo "==> Downloading fastText lid.176.bin (~131MB)..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$LID_PATH" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$LID_PATH" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
  else
    echo "Neither curl nor wget found — install one and re-run, or download manually to $LID_PATH."
    missing=1
  fi
fi

if [ "$missing" -eq 1 ]; then
  echo ""
  echo "==> Some assets are missing. See messages above."
  exit 1
fi

echo ""
echo "==> All expected assets are present."
