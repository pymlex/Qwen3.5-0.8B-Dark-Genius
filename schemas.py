from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    key: str
    display_name: str
    hub_id: str
    short_label: str
    color: str
    local_path: str | None = None


class InferencePolicy(BaseModel):
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    seed: int = 42
    system_prompt: str = "You are a helpful assistant."
    max_new_tokens: int = 256


class LmEvalTaskSpec(BaseModel):
    task: str
    metric_key: str
    num_fewshot: int
    max_new_tokens: int


class AttackSuccessLabel(BaseModel):
    attack_success: Literal[0, 1]


class HarmBenchEvalSpec(BaseModel):
    judge_id: str = "tvall43/Qwen3.5-0.8B-heretic-v3"
    behaviors_file: str = "harmbench_behaviors_text_test.csv"
    method: str = "DirectRequest"
    max_new_tokens: int = 512


class RefusalEvalSpec(BaseModel):
    dataset: str = "mlabonne/harmful_behaviors"
    split: str = "test[:100]"
    column: str = "text"
    max_new_tokens: int = 100
    markers: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    model_key: str
    benchmark: str
    metric_name: str
    value: float
    raw_path: str | None = None
    command: str | None = None


class RunMetadata(BaseModel):
    seed: int
    merge_config: str
    inference_policy: InferencePolicy
    library_versions: dict[str, str]
    hardware_note: str = (
        "1× RTX 5090 31.84 GB VRAM, PCIe 1.0 x16, CUDA 13.2, "
        "AMD Ryzen 9 9950X3D 16-Core 4.3 GHz, 64 GB RAM"
    )


class SummaryRow(BaseModel):
    model_key: str
    display_name: str
    gpqa_accuracy: float | None = None
    gsm8k_exact_match: float | None = None
    harmbench_asr: float | None = None
    refusal_rate: float | None = None


class MergeValidationReport(BaseModel):
    base_model: str
    source_models: list[str]
    tokenizer_compatible: bool
    architecture_compatible: bool
    vocab_size_match: bool
    details: str


class PushConfig(BaseModel):
    repo_id: str
    results_dir: Path
    branch: str = "main"
