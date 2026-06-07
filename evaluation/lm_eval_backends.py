from constants import (
    LM_EVAL_GPU_MEMORY_UTILIZATION,
    LM_EVAL_MAX_MODEL_LEN,
)


def build_hf_model_args(model_path: str) -> str:
    return f"pretrained={model_path},trust_remote_code=True,dtype=float16"


def build_vllm_model_args(model_path: str) -> str:
    return (
        f"pretrained={model_path},trust_remote_code=True,dtype=float16,"
        f"gpu_memory_utilization={LM_EVAL_GPU_MEMORY_UTILIZATION},"
        f"max_model_len={LM_EVAL_MAX_MODEL_LEN},language_model_only=True"
    )


def build_model_args(model_path: str, backend: str) -> str:
    if backend == "vllm":
        return build_vllm_model_args(model_path)
    return build_hf_model_args(model_path)
