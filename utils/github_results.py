import subprocess
from pathlib import Path


def push_results_to_github(project_root: Path, commit_message: str) -> None:
    results_dir = project_root / "results"
    if not results_dir.exists():
        return

    subprocess.run(
        ["git", "add", "-A", "--", str(results_dir)],
        cwd=project_root,
        check=True,
    )

    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=project_root,
    )
    if staged.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=project_root,
            check=True,
        )

    ahead = subprocess.run(
        ["git", "rev-list", "--count", "@{u}..HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if int(ahead.stdout.strip() or "0") > 0:
        subprocess.run(["git", "push"], cwd=project_root, check=True)
