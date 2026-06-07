import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, create_repo

from constants import HF_REPO_ID, MERGE_OUTPUT_DIR, PROJECT_ROOT, RESULTS_DIR
from evaluation.hf_eval_export import export_hf_eval_artifacts
from utils.hf_model_card import write_hf_readme


def main() -> None:
    MERGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_hf_eval_artifacts(RESULTS_DIR, MERGE_OUTPUT_DIR)
    write_hf_readme(
        PROJECT_ROOT / "hf_model_card.md",
        MERGE_OUTPUT_DIR / "README.md",
        RESULTS_DIR,
    )

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    create_repo(HF_REPO_ID, exist_ok=True, token=token)

    for name in ["README.md", "benchmark_comparison.png"]:
        path = MERGE_OUTPUT_DIR / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=token,
            )
            print(f"uploaded {name}")

    for folder_name in [".eval_results", "eval_results"]:
        folder = MERGE_OUTPUT_DIR / folder_name
        if folder.exists():
            api.upload_folder(
                folder_path=str(folder),
                path_in_repo=folder_name,
                repo_id=HF_REPO_ID,
                repo_type="model",
                token=token,
            )
            print(f"uploaded {folder_name}/")

    print(f"https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
