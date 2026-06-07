import json
from pathlib import Path

from constants import INFERENCE_POLICY, SEED


def build_lm_eval_command(
    model_path: str,
    task_name: str,
    task_spec: dict,
    output_path: Path,
) -> list[str]:
    model_args = f"pretrained={model_path},trust_remote_code=True,dtype=float16"
    return [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        task_spec["task"],
        "--num_fewshot",
        str(task_spec["num_fewshot"]),
        "--batch_size",
        "auto",
        "--device",
        "cuda:0",
        "--seed",
        str(SEED),
        "--output_path",
        str(output_path),
        "--log_samples",
        "--gen_kwargs",
        json.dumps(
            {
                "temperature": INFERENCE_POLICY["temperature"],
                "do_sample": INFERENCE_POLICY["do_sample"],
                "max_gen_toks": task_spec["max_new_tokens"],
            }
        ),
    ]
