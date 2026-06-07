import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from merge.weight_io import (
    load_tensor,
    resolve_repo_tensor_index,
    save_merged_weights,
)


def dare_sparsify(
    delta: torch.Tensor,
    density: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply DARE random pruning with L1-norm rescaling."""
    if density >= 1.0:
        return delta
    work_dtype = delta.dtype
    if delta.device.type == "cpu" and delta.dtype != torch.bfloat16:
        work_dtype = torch.float32
    work = delta.to(work_dtype)
    mask = torch.bernoulli(
        torch.full(work.shape, density, dtype=work_dtype, device=work.device),
        generator=generator,
    )
    masked = work * mask
    before_scale = work.abs().sum()
    after_scale = masked.abs().sum()
    if before_scale >= 1e-7 and after_scale >= 1e-7:
        masked = masked * (before_scale / after_scale)
    return masked.to(delta.dtype)


def ties_mask(weighted_deltas: torch.Tensor) -> torch.Tensor:
    """Return a sign-consensus mask over weighted task vectors."""
    sign = weighted_deltas.sign()
    sign_weight = weighted_deltas.sum(dim=0)
    majority_sign = (sign_weight >= 0).to(sign.dtype) * 2 - 1
    return sign == majority_sign


def merge_tensor_dare_ties(
    base: torch.Tensor,
    source_tensors: list[torch.Tensor],
    weights: list[float],
    densities: list[float],
    normalize: bool,
    lambda_scale: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Merge one parameter tensor with DARE-TIES."""
    base_dev = base.to(device=device, dtype=torch.float32)
    deltas = []
    model_weights = []
    for tensor, weight, density in zip(source_tensors, weights, densities):
        delta = tensor.to(device=device, dtype=torch.float32) - base_dev
        delta = dare_sparsify(delta, density, generator)
        deltas.append(delta)
        model_weights.append(weight)

    if not deltas:
        return base

    stacked = torch.stack(deltas, dim=0)
    weight_tensor = torch.tensor(
        model_weights,
        dtype=stacked.dtype,
        device=stacked.device,
    ).view(-1, *([1] * (stacked.ndim - 1)))
    weighted_deltas = stacked * weight_tensor
    mask = ties_mask(weighted_deltas)
    mixed_delta = (weighted_deltas * mask).sum(dim=0)
    if normalize:
        divisor = (weight_tensor * mask).sum(dim=0)
        divisor[divisor == 0] = 1.0
        mixed_delta = mixed_delta / divisor
    mixed_delta = mixed_delta * lambda_scale
    merged = base_dev + mixed_delta
    return merged.to(dtype=base.dtype).cpu()


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

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    copy_base_assets(base_model, output_dir, token)

    merged_tensors: dict[str, torch.Tensor] = {}
    merge_stats = {
        "merged_with_both_sources": 0,
        "merged_with_single_source": 0,
        "base_only": 0,
    }

    tensor_names = sorted(base_index.keys())
    for tensor_name in tqdm(tensor_names, desc="DARE-TIES merge"):
        base_tensor = load_tensor(base_index[tensor_name], tensor_name)
        present_sources = []
        present_weights = []
        present_densities = []
        for repo_index, weight, density in zip(
            source_indices,
            source_weights,
            source_densities,
        ):
            if tensor_name not in repo_index:
                continue
            source_tensor = load_tensor(repo_index[tensor_name], tensor_name)
            if source_tensor.shape != base_tensor.shape:
                continue
            present_sources.append(source_tensor)
            present_weights.append(weight)
            present_densities.append(density)

        if not present_sources:
            merged_tensors[tensor_name] = base_tensor
            merge_stats["base_only"] += 1
            continue

        generator = torch.Generator(device="cpu")
        generator.manual_seed(hash((seed, tensor_name)) % (2**63 - 1))

        merged_tensors[tensor_name] = merge_tensor_dare_ties(
            base=base_tensor,
            source_tensors=present_sources,
            weights=present_weights,
            densities=present_densities,
            normalize=normalize,
            lambda_scale=lambda_scale,
            generator=generator,
            device=device,
        )
        if len(present_sources) == len(source_entries):
            merge_stats["merged_with_both_sources"] += 1
        else:
            merge_stats["merged_with_single_source"] += 1

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
