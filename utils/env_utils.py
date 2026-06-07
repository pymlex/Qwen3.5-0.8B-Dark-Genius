import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_root: Path) -> None:
    load_dotenv(project_root / ".env")


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Copy .env.example to .env and set HF_TOKEN.")
    return token


def require_gh_token() -> str:
    token = os.environ.get("GH_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GH_TOKEN is missing. Copy .env.example to .env and set GH_TOKEN.")
    return token


def authenticate_huggingface(project_root: Path) -> None:
    load_project_env(project_root)
    token = require_hf_token()
    subprocess.run(
        ["huggingface-cli", "login", "--token", token],
        check=True,
    )


def authenticate_github_browser() -> None:
    subprocess.run(["gh", "auth", "login", "--web", "--git-protocol", "https"], check=True)


def authenticate_github_token(project_root: Path) -> None:
    load_project_env(project_root)
    token = require_gh_token()
    subprocess.run(
        ["gh", "auth", "login", "--with-token"],
        input=token.encode(),
        check=True,
    )


def setup_auth(project_root: Path, use_browser: bool = True) -> None:
    authenticate_huggingface(project_root)
    if use_browser:
        authenticate_github_browser()
    else:
        authenticate_github_token(project_root)


def capture_library_versions() -> dict[str, str]:
    packages = [
        "torch",
        "transformers",
        "mergekit",
        "lm_eval",
        "datasets",
        "accelerate",
        "vllm",
        "numpy",
        "matplotlib",
        "seaborn",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        result = subprocess.run(
            ["python", "-m", "pip", "show", package],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                versions[package] = line.split(":", 1)[1].strip()
    return versions
