from pathlib import Path

import torch

from merge.merge_assets import clear_output_dir, copy_base_assets
from merge.merge_loop import merge_all_tensors
from merge.weight_io import resolve_repo_tensor_index, save_merged_weights


def run_native_dare_ties_merge(
    config: dict,
    output_dir: Path,
    token: str,
    seed: int,
) -> dict:
    """Execute DARE-TIES merge for Qwen3.5 checkpoints without mergekit."""
    base_model = config["base_model"]
    source_entries = config["models"]
    global_params = config.get("parameters", {})
    normalize = bool(global_params.get("normalize", False))
    lambda_scale = float(global_params.get("lambda", 1.0))
    dtype_name = config.get("dtype", "float16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    base_index = resolve_repo_tensor_index(base_model, token)
    source_indices = [
        resolve_repo_tensor_index(entry["model"], token) for entry in source_entries
    ]
    source_ids = [entry["model"] for entry in source_entries]
    source_weights = [float(entry["parameters"]["weight"]) for entry in source_entries]
    source_densities = [
        float(entry["parameters"]["density"]) for entry in source_entries
    ]

    clear_output_dir(output_dir)
    copy_base_assets(base_model, output_dir, token)

    merged_tensors, merge_stats = merge_all_tensors(
        base_index=base_index,
        source_indices=source_indices,
        source_weights=source_weights,
        source_densities=source_densities,
        source_count=len(source_entries),
        normalize=normalize,
        lambda_scale=lambda_scale,
        seed=seed,
        device=device,
    )
    save_merged_weights(merged_tensors, output_dir / "model.safetensors")

    return {
        "implementation": "native_dare_ties",
        "base_model": base_model,
        "source_models": source_ids,
        "dtype": dtype_name,
        "seed": seed,
        "normalize": normalize,
        "lambda": lambda_scale,
        "device": str(device),
        "tensor_counts": merge_stats,
        "output_tensors": len(merged_tensors),
    }
