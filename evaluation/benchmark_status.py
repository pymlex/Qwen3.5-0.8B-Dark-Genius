import json
from pathlib import Path

from constants import HARMBENCH_JUDGE_ID, LM_EVAL_TASKS


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def lm_eval_summary_path(results_dir: Path, model_key: str, benchmark: str) -> Path:
    return results_dir / "metrics" / "lm_eval" / model_key / benchmark / f"{benchmark}.json"


def harmbench_summary_path(results_dir: Path, model_key: str) -> Path:
    return results_dir / "metrics" / "harmbench" / model_key / "summary.json"


def harmbench_completions_path(results_dir: Path, model_key: str) -> Path:
    return results_dir / "raw" / "harmbench" / model_key / "completions.json"


def refusal_summary_path(results_dir: Path, model_key: str) -> Path:
    return results_dir / "metrics" / "refusal" / model_key / "summary.json"


def is_lm_eval_done(results_dir: Path, model_key: str, benchmark: str) -> bool:
    return lm_eval_summary_path(results_dir, model_key, benchmark).exists()


def is_harmbench_done(results_dir: Path, model_key: str) -> bool:
    summary_path = harmbench_summary_path(results_dir, model_key)
    if not summary_path.exists():
        return False
    summary = load_summary(summary_path)
    return summary.get("judge_id") == HARMBENCH_JUDGE_ID


def is_refusal_done(results_dir: Path, model_key: str) -> bool:
    return refusal_summary_path(results_dir, model_key).exists()


def pending_lm_eval_jobs(
    results_dir: Path,
    model_keys: list[str],
) -> list[tuple[str, str]]:
    jobs = []
    for model_key in model_keys:
        for benchmark in LM_EVAL_TASKS:
            if not is_lm_eval_done(results_dir, model_key, benchmark):
                jobs.append((model_key, benchmark))
    return jobs


def pending_harmbench_models(results_dir: Path, model_keys: list[str]) -> list[str]:
    return [key for key in model_keys if not is_harmbench_done(results_dir, key)]


def pending_refusal_models(results_dir: Path, model_keys: list[str]) -> list[str]:
    return [key for key in model_keys if not is_refusal_done(results_dir, key)]
