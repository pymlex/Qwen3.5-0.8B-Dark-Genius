import subprocess
from pathlib import Path

from constants import LM_EVAL_TASKS, MODEL_REGISTRY
from evaluation.gpqa_access import GPQA_ACCESS_URL, ensure_gpqa_access
from evaluation.lm_eval_command import build_lm_eval_command
from evaluation.lm_eval_output import (
    extract_lm_eval_metric,
    load_lm_eval_payload,
    lm_eval_output_dir,
    persist_lm_eval_artifacts,
    resolve_lm_eval_results_json,
)
from utils.env_utils import require_hf_token
from utils.hf_env import hf_subprocess_env
from utils.model_loader import resolve_model_path


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
    output_dir = lm_eval_output_dir(raw_dir, benchmark)
    command = build_lm_eval_command(model_path, benchmark, task_spec, output_dir)
    subprocess.run(command, check=True, env=hf_subprocess_env(hf_token))

    results_json_path = resolve_lm_eval_results_json(output_dir)
    payload = load_lm_eval_payload(results_json_path)
    metric_value = extract_lm_eval_metric(payload, task_spec)
    metrics_dir = results_dir / "metrics" / "lm_eval" / model_key / benchmark
    summary = {
        "model_key": model_key,
        "benchmark": benchmark,
        "metric_name": task_spec["metric_key"],
        "value": metric_value,
        "raw_path": str(results_json_path),
        "canonical_path": str(metrics_dir / "lm_eval_results.json"),
        "command": " ".join(command),
        "task": task_spec["task"],
    }
    if benchmark == "gpqa":
        summary["gpqa_access_url"] = GPQA_ACCESS_URL
    persist_lm_eval_artifacts(
        payload,
        results_json_path,
        metrics_dir,
        benchmark,
        summary,
    )
    return summary


def run_all_lm_eval(model_keys: list[str], results_dir: Path) -> list[dict]:
    summaries = []
    for model_key in model_keys:
        for benchmark in LM_EVAL_TASKS:
            summaries.append(run_lm_eval_for_model(model_key, results_dir, benchmark))
    return summaries
