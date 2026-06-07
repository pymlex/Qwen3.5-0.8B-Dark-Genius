import json
import subprocess
from pathlib import Path

from constants import LM_EVAL_TASKS, MODEL_REGISTRY
from evaluation.gpqa_access import GPQA_ACCESS_URL, ensure_gpqa_access
from evaluation.lm_eval_command import build_lm_eval_command
from utils.env_utils import require_hf_token
from utils.hf_env import hf_subprocess_env
from utils.model_loader import resolve_model_path


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
    if benchmark == "gpqa":
        ensure_gpqa_access(hf_token)
    task_spec = LM_EVAL_TASKS[benchmark]
    model_path = resolve_model_path(model_key, MODEL_REGISTRY, results_dir.parent / "outputs")
    raw_dir = results_dir / "raw" / "lm_eval" / model_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{benchmark}.json"
    command = build_lm_eval_command(model_path, benchmark, task_spec, output_path)
    subprocess.run(command, check=True, env=hf_subprocess_env(hf_token))
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
    if benchmark == "gpqa":
        summary["gpqa_access_url"] = GPQA_ACCESS_URL
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
