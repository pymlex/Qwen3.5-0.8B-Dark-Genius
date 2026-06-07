#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MESSAGE="${1:-Publish Dark-Genius model and benchmark results}"

python main.py report
python main.py push-hf
python main.py push-github --message "$MESSAGE"
