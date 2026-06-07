from pathlib import Path

from huggingface_hub import HfApi, create_repo

from constants import HF_REPO_ID, MERGE_OUTPUT_DIR
from utils.env_utils import require_hf_token


def upload_merged_model(
    local_dir: Path = MERGE_OUTPUT_DIR,
    repo_id: str = HF_REPO_ID,
    model_card_path: Path | None = None,
) -> str:
    token = require_hf_token()
    api = HfApi(token=token)
    create_repo(repo_id, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    if model_card_path and model_card_path.exists():
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
    return f"https://huggingface.co/{repo_id}"
