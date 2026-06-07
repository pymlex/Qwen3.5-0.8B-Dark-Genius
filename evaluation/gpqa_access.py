from huggingface_hub import hf_hub_download

GPQA_DATASET_ID = "Idavidrein/gpqa"
GPQA_MAIN_FILE = "gpqa_main.csv"
GPQA_ACCESS_URL = "https://huggingface.co/datasets/Idavidrein/gpqa"


def ensure_gpqa_access(token: str) -> None:
    """Download GPQA metadata to verify gated-dataset access before lm-eval."""
    hf_hub_download(
        repo_id=GPQA_DATASET_ID,
        repo_type="dataset",
        filename=GPQA_MAIN_FILE,
        token=token,
    )
