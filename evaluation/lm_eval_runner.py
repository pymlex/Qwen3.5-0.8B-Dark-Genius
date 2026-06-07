import json
import subprocess
from pathlib import Path

from constants import INFERENCE_POLICY, LM_EVAL_TASKS, MODEL_REGISTRY, SEED
from utils.env_utils import require_hf_token
from utils.model_loader import resolve_model_path


def build_lm_eval_command(
    model_path: str,
    task_name: str,
    task_spec: dict,
    output_path: Path,
    hf_token: str,
) -> list[str]:
    model_args = (
        f"pretrained={model_path},trust_remote_code=True,dtype=float16,"
        f"token={hf_token}"
    )
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


def extract_metric(result_payload: dict, task_spec: dict) -> float:
    task_key = task_spec["task"]
    metric_key = task_spec["metric_key"]
    if "results" in result_payload:
        task_results = result_payload["results"][task_key]
    else:
        task_results = result_payload[task_key]
    if metric_key in task_results:
        return float(task_results[metric_key])
    if "," in metric_key:
        primary, secondary = metric_key.split(",", 1)
        nested = task_results.get(primary, {})
        if isinstance(nested, dict) and secondary in nested:
            return float(nested[secondary])
    raise KeyError(f"Metric {metric_key} not found in lm-eval output for {task_key}")


def run_lm_eval_for_model(
    model_key: str,
    results_dir: Path,
    benchmark: str,
) -> dict:
    hf_token = require_hf_token()
    task_spec = LM_EVAL_TASKS[benchmark]
    model_path = resolve_model_path(model_key, MODEL_REGISTRY, results_dir.parent / "outputs")
    raw_dir = results_dir / "raw" / "lm_eval" / model_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{benchmark}.json"
    command = build_lm_eval_command(model_path, benchmark, task_spec, output_path, hf_token)
    subprocess.run(command, check=True)
    with open(output_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    metric_value = extract_metric(payload, task_spec)
    summary = {
        "model_key": model_key,
        "benchmark": benchmark,
        "metric_name": task_spec["metric_key"],
        "value": metric_value,
        "raw_path": str(output_path),
        "command": " ".join(command),
        "task": task_spec["task"],
    }
    summary_path = results_dir / "metrics" / "lm_eval" / model_key / f"{benchmark}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_all_lm_eval(model_keys: list[str], results_dir: Path) -> list[dict]:
    summaries = []
    for model_key in model_keys:
        for benchmark in LM_EVAL_TASKS:
            summaries.append(run_lm_eval_for_model(model_key, results_dir, benchmark))
    return summaries
