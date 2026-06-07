import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file


def list_tensor_names(file_path: str) -> list[str]:
    with safe_open(file_path, framework="pt") as handle:
        return list(handle.keys())


def load_tensor(file_path: str, tensor_name: str) -> torch.Tensor:
    with safe_open(file_path, framework="pt") as handle:
        return handle.get_tensor(tensor_name)


def resolve_repo_tensor_index(repo_id: str, token: str) -> dict[str, str]:
    """Resolve tensor names to shard paths for single- or multi-shard repositories."""
    repo_files = list_repo_files(repo_id, token=token)
    if "model.safetensors.index.json" in repo_files:
        index_path = hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            token=token,
        )
        with open(index_path, encoding="utf-8") as handle:
            index_data = json.load(handle)
        weight_map = index_data["weight_map"]
        return {
            tensor_name: hf_hub_download(repo_id, shard_name, token=token)
            for tensor_name, shard_name in weight_map.items()
        }
    single_path = hf_hub_download(repo_id, "model.safetensors", token=token)
    tensor_names = list_tensor_names(single_path)
    return {tensor_name: single_path for tensor_name in tensor_names}


def save_merged_weights(
    tensors: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_tensors = {name: tensor.cpu().contiguous() for name, tensor in tensors.items()}
    save_file(cpu_tensors, str(output_path))
