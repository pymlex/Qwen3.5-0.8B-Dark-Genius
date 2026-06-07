import json
import shutil
from pathlib import Path


def lm_eval_output_dir(raw_dir: Path, benchmark: str) -> Path:
    output_dir = raw_dir / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_lm_eval_results_json(output_dir: Path) -> Path:
    """Locate the aggregated lm-eval JSON written by EvaluationTracker."""
    candidates = sorted(
        output_dir.rglob("results_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            output_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
        )
    if not candidates:
        candidates = sorted(
            output_dir.parent.glob(f"{output_dir.name}_*.json"),
            key=lambda path: path.stat().st_mtime,
        )
    if not candidates:
        raise FileNotFoundError(f"No lm-eval aggregated JSON under {output_dir}")
    return candidates[-1]


def load_lm_eval_payload(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def extract_lm_eval_metric(payload: dict, task_spec: dict) -> float:
    task_key = task_spec["task"]
    metric_key = task_spec["metric_key"]
    task_results = payload["results"][task_key]
    if metric_key in task_results:
        return float(task_results[metric_key])
    if "," in metric_key:
        primary, secondary = metric_key.split(",", 1)
        compound = f"{primary},{secondary}"
        if compound in task_results:
            return float(task_results[compound])
        nested = task_results.get(secondary, {})
        if isinstance(nested, dict) and primary in nested:
            return float(nested[primary])
    raise KeyError(f"Metric {metric_key} not found for task {task_key}")


def persist_lm_eval_artifacts(
    payload: dict,
    results_json_path: Path,
    metrics_dir: Path,
    benchmark: str,
    summary: dict,
) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = metrics_dir / "lm_eval_results.json"
    with open(canonical_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    shutil.copy2(results_json_path, metrics_dir / results_json_path.name)
    summary_path = metrics_dir / f"{benchmark}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return canonical_path
