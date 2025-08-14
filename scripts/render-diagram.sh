#!/usr/bin/env bash
set -euo pipefail

INPUT="$1"           # e.g. system-overview.mmd
BASENAME="${INPUT%.mmd}"
# render SVG
mmdc -i "$INPUT" -o "${BASENAME}.svg" \
     --backgroundColor transparent
# render PNG
mmdc -i "$INPUT" -o "${BASENAME}.png" \
     --scale 2
echo "Wrote ${BASENAME}.{svg,png}"
