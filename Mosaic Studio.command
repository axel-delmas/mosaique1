#!/bin/zsh
# Double-click to open Mosaic Studio in your browser. Close this window to stop it.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate || { echo "No .venv found in $(pwd)"; read -k1 "?Press any key to close"; exit 1; }
python -m mosaic_studio app
read -k1 "?Mosaic Studio stopped. Press any key to close"
