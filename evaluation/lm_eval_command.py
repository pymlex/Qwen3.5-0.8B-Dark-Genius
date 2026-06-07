import json
from pathlib import Path

from constants import (
    INFERENCE_POLICY,
    LM_EVAL_BACKEND,
    LM_EVAL_BATCH_SIZE,
    SEED,
)
from evaluation.lm_eval_backends import build_model_args


def build_lm_eval_command(
    model_path: str,
    task_name: str,
    task_spec: dict,
    output_path: Path,
) -> list[str]:
    command = [
        "lm_eval",
        "--model",
        LM_EVAL_BACKEND,
        "--model_args",
        build_model_args(model_path, LM_EVAL_BACKEND),
        "--tasks",
        task_spec["task"],
        "--num_fewshot",
        str(task_spec["num_fewshot"]),
        "--batch_size",
        str(LM_EVAL_BATCH_SIZE),
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
    if LM_EVAL_BACKEND == "hf":
        command.extend(["--device", "cuda:0"])
    return command
