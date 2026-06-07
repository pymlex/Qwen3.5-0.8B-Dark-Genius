import torch


def tensor_seed(seed: int, tensor_name: str) -> int:
    return hash((seed, tensor_name)) % (2**63 - 1)


def make_generator(device: torch.device, seed: int, tensor_name: str) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(tensor_seed(seed, tensor_name))
    return generator


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
    seed: int,
    tensor_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Merge one parameter tensor with DARE-TIES."""
    base_dev = base.to(device=device, dtype=torch.float32)
    generator = make_generator(base_dev.device, seed, tensor_name)
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
