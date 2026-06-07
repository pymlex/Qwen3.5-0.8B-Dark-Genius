import json
import shutil
from datetime import date
from pathlib import Path

import yaml

from constants import HF_REPO_ID, MODEL_REGISTRY
from evaluation.aggregate import collect_model_metrics
from evaluation.hf_eval_specs import HF_EVAL_SOURCE, LM_EVAL_HF_ENTRIES
from evaluation.hf_model_index import build_model_index_entry, metric_percent


def write_eval_results_yaml(
    model_dir: Path,
    metrics: dict[str, float | None],
) -> None:
    eval_dir = model_dir / ".eval_results"
    eval_dir.mkdir(parents=True, exist_ok=True)
    run_date = date.today().isoformat()
    for benchmark, spec in LM_EVAL_HF_ENTRIES.items():
        key = "gpqa_accuracy" if benchmark == "gpqa" else "gsm8k_exact_match"
        value = metric_percent(metrics.get(key))
        if value is None:
            continue
        payload = [
            {
                "dataset": {"id": spec["dataset_id"], "task_id": spec["task_id"]},
                "value": value,
                "date": run_date,
                "source": HF_EVAL_SOURCE,
            }
        ]
        with open(eval_dir / f"{benchmark}.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)


def copy_metric_json(
    results_dir: Path,
    model_dir: Path,
    model_key: str,
    benchmark: str,
    metrics_subdir: str,
) -> None:
    src = results_dir / "metrics" / metrics_subdir / model_key / "summary.json"
    if not src.exists():
        return
    dst_dir = model_dir / "eval_results" / benchmark
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "summary.json")


def export_hf_eval_artifacts(
    results_dir: Path,
    model_dir: Path,
    model_key: str = "dark_genius",
) -> Path:
    metrics = collect_model_metrics(model_key, results_dir)
    display_name = MODEL_REGISTRY[model_key]["display_name"]
    model_index = build_model_index_entry(display_name, metrics)
    write_eval_results_yaml(model_dir, metrics)

    eval_results_dir = model_dir / "eval_results"
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    for benchmark in LM_EVAL_HF_ENTRIES:
        src = results_dir / "metrics" / "lm_eval" / model_key / benchmark / "lm_eval_results.json"
        if not src.exists():
            continue
        dst_dir = eval_results_dir / benchmark
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "lm_eval_results.json")

    copy_metric_json(results_dir, model_dir, model_key, "harmbench", "harmbench")
    copy_metric_json(results_dir, model_dir, model_key, "refusal_rate", "refusal")

    manifest_path = eval_results_dir / "metrics_manifest.json"
    manifest = {
        "model_key": model_key,
        "hub_id": HF_REPO_ID,
        "model_index": model_index,
        "metrics": metrics,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path
