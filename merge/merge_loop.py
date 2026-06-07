import torch
from tqdm.auto import tqdm

from merge.dare_ops import merge_tensor_dare_ties
from merge.weight_io import load_tensor


def collect_present_sources(
    tensor_name: str,
    base_tensor: torch.Tensor,
    source_indices: list[dict[str, str]],
    source_weights: list[float],
    source_densities: list[float],
) -> tuple[list[torch.Tensor], list[float], list[float]]:
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
    return present_sources, present_weights, present_densities


def merge_all_tensors(
    base_index: dict[str, str],
    source_indices: list[dict[str, str]],
    source_weights: list[float],
    source_densities: list[float],
    source_count: int,
    normalize: bool,
    lambda_scale: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    merged_tensors: dict[str, torch.Tensor] = {}
    merge_stats = {
        "merged_with_both_sources": 0,
        "merged_with_single_source": 0,
        "base_only": 0,
    }
    tensor_names = sorted(base_index.keys())
    for tensor_name in tqdm(tensor_names, desc="DARE-TIES merge"):
        base_tensor = load_tensor(base_index[tensor_name], tensor_name)
        present_sources, present_weights, present_densities = collect_present_sources(
            tensor_name,
            base_tensor,
            source_indices,
            source_weights,
            source_densities,
        )
        if not present_sources:
            merged_tensors[tensor_name] = base_tensor
            merge_stats["base_only"] += 1
            continue
        merged_tensors[tensor_name] = merge_tensor_dare_ties(
            base=base_tensor,
            source_tensors=present_sources,
            weights=present_weights,
            densities=present_densities,
            normalize=normalize,
            lambda_scale=lambda_scale,
            seed=seed,
            tensor_name=tensor_name,
            device=device,
        )
        if len(present_sources) == source_count:
            merge_stats["merged_with_both_sources"] += 1
        else:
            merge_stats["merged_with_single_source"] += 1
    return merged_tensors, merge_stats
