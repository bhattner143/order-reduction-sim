#!/usr/bin/env bash
# Build Learner 4 block diagram PDFs (simple + detailed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs/learner4"
mkdir -p "$OUT"
cd "$ROOT/report"
for name in learner4_block_diagram_simple learner4_block_diagram; do
  pdflatex -interaction=nonstopmode -output-directory="$OUT" "${name}.tex" >/dev/null
  pdflatex -interaction=nonstopmode -output-directory="$OUT" "${name}.tex" >/dev/null
  rm -f "$OUT"/${name}.{aux,log,out}
  echo "Wrote $OUT/${name}.pdf"
done
