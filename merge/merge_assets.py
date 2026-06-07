import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not any(output_dir.iterdir()):
        return
    for child in output_dir.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def copy_base_assets(
    base_model: str,
    output_dir: Path,
    token: str,
) -> None:
    """Copy tokenizer and config artefacts from the base checkpoint."""
    snapshot_download(
        repo_id=base_model,
        local_dir=str(output_dir),
        token=token,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.jinja",
            "*.model",
            "tokenizer*",
            "vocab*",
            "merges*",
            "chat_template*",
            "generation_config.json",
        ],
    )
    for stale_name in (
        "model.safetensors",
        "model.safetensors.index.json",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale_shard in output_dir.glob("model.safetensors-*-of-*.safetensors"):
        stale_shard.unlink()
