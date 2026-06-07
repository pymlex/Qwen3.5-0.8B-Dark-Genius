import subprocess
from pathlib import Path


def push_results_to_github(project_root: Path, commit_message: str) -> None:
    results_dir = project_root / "results"
    subprocess.run(["git", "add", str(results_dir)], cwd=project_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=project_root,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=project_root, check=True)
