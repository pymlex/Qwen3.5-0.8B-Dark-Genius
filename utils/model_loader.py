import json
from pathlib import Path


import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def resolve_model_path(model_key: str, registry: dict, outputs_dir: Path) -> str:
    entry = registry[model_key]
    if entry.get("local_path"):
        local_path = Path(entry["local_path"])
        if not local_path.exists():
            raise RuntimeError(
                f"Local merged model path does not exist: {local_path}. Run merge first."
            )
        return str(local_path)
    return entry["hub_id"]


def load_tokenizer(model_path: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_generation_model(model_path: str, hf_token: str, dtype: torch.dtype = torch.float16):
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = fetch_remote_config(model_path, hf_token)

    architectures = config.get("architectures", [])
    if "Qwen3_5ForConditionalGeneration" in architectures:
        return AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=hf_token,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )


def fetch_remote_config(model_id: str, hf_token: str) -> dict:
    config_path = hf_hub_download(model_id, "config.json", token=hf_token)
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


def structural_config(config: dict) -> dict:
    if "text_config" in config:
        return config["text_config"]
    return config


def validate_merge_compatibility(
    base_model_id: str,
    source_model_ids: list[str],
    hf_token: str,
) -> dict:
    base_config = fetch_remote_config(base_model_id, hf_token)
    base_arch = base_config.get("architectures", [])
    base_vocab = fetch_tokenizer_vocab_size(base_model_id, hf_token)

    base_struct = structural_config(base_config)
    base_hidden = base_struct.get("hidden_size")
    base_layers = base_struct.get("num_hidden_layers")

    reports = []
    architecture_compatible = True
    vocab_size_match = True

    for model_id in source_model_ids:
        config = fetch_remote_config(model_id, hf_token)
        arch = config.get("architectures", [])
        vocab = fetch_tokenizer_vocab_size(model_id, hf_token)
        struct = structural_config(config)
        hidden = struct.get("hidden_size")
        layers = struct.get("num_hidden_layers")
        arch_ok = (
            arch == base_arch
            or (
                "Qwen3_5ForCausalLM" in arch
                and "Qwen3_5ForConditionalGeneration" in base_arch
            )
        )
        struct_ok = hidden == base_hidden and layers == base_layers
        arch_ok = arch_ok and struct_ok
        vocab_ok = vocab == base_vocab
        architecture_compatible = architecture_compatible and arch_ok
        vocab_size_match = vocab_size_match and vocab_ok
        reports.append(
            {
                "model_id": model_id,
                "architectures": arch,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "vocab_size": vocab,
                "architecture_compatible": arch_ok,
                "vocab_size_match": vocab_ok,
            }
        )

    if not architecture_compatible:
        raise RuntimeError(
            "Merge validation failed: source models are not compatible with "
            f"base {base_model_id}. Details: {reports}"
        )
    if not vocab_size_match:
        raise RuntimeError(
            "Merge validation failed: tokenizer vocab sizes do not match across models. "
            f"Details: {reports}"
        )

    return {
        "base_model": base_model_id,
        "base_architectures": base_arch,
        "base_vocab_size": base_vocab,
        "source_models": reports,
        "architecture_compatible": architecture_compatible,
        "vocab_size_match": vocab_size_match,
        "tokenizer_compatible": vocab_size_match,
    }


def fetch_tokenizer_vocab_size(model_id: str, hf_token: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
    )
    return len(tokenizer)
