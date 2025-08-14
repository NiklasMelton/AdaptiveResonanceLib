#!/usr/bin/env bash
set -euo pipefail

# Directory containing the .mmd files, relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAGRAM_DIR="${SCRIPT_DIR}/../docs/diagrams"

# Ensure the directory exists
if [[ ! -d "$DIAGRAM_DIR" ]]; then
    echo "Error: Diagram directory not found: $DIAGRAM_DIR" >&2
    exit 1
fi

# Loop over all .mmd files in the directory
shopt -s nullglob
for INPUT in "$DIAGRAM_DIR"/*.mmd; do
    BASENAME="${INPUT%.mmd}"
    echo "Rendering diagram: $INPUT"

    # Render SVG
    mmdc -i "$INPUT" -o "${BASENAME}.svg" \
         --backgroundColor transparent

    # Render PNG
    mmdc -i "$INPUT" -o "${BASENAME}.png" \
         --scale 2
done
shopt -u nullglob

echo "Rendering complete."
