from pathlib import Path

from huggingface_hub import HfApi, create_repo

from constants import HF_REPO_ID, MERGE_OUTPUT_DIR, PROJECT_ROOT, RESULTS_DIR
from evaluation.hf_eval_export import export_hf_eval_artifacts
from utils.env_utils import require_hf_token
from utils.hf_model_card import write_hf_readme


def upload_merged_model(
    local_dir: Path = MERGE_OUTPUT_DIR,
    repo_id: str = HF_REPO_ID,
    model_card_path: Path | None = None,
    results_dir: Path = RESULTS_DIR,
) -> str:
    token = require_hf_token()
    export_hf_eval_artifacts(results_dir, local_dir)
    template_path = model_card_path or PROJECT_ROOT / "hf_model_card.md"
    write_hf_readme(template_path, local_dir / "README.md", results_dir)
    api = HfApi(token=token)
    create_repo(repo_id, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    return f"https://huggingface.co/{repo_id}"
