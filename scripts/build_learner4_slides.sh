#!/usr/bin/env bash
# Build the short Learner 4 slide deck.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/outputs/learner4"
mkdir -p "$OUT"
cd "$ROOT/report"
pdflatex -interaction=nonstopmode -output-directory="$OUT" learner4_slides.tex >/dev/null
pdflatex -interaction=nonstopmode -output-directory="$OUT" learner4_slides.tex >/dev/null
rm -f "$OUT"/learner4_slides.{aux,log,out,nav,snm,toc,vrb}
echo "Wrote $OUT/learner4_slides.pdf"
