#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MESSAGE="${1:-Add benchmark results from Colab L4 run}"
python main.py report
python main.py push-github --message "$MESSAGE"
