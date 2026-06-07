#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Set HF_TOKEN before running merge or evaluation."
fi

python -m pip install -U pip
python -m pip install -r requirements.txt
python main.py setup-harmbench

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is not installed. Install gh, then run: python main.py setup"
else
  if ! gh auth status >/dev/null 2>&1; then
    echo "Run: python main.py setup"
  fi
fi

echo "Installation complete."
