import csv
import json
from pathlib import Path

from tqdm.auto import tqdm

from constants import (
    HARMBENCH_BEHAVIORS_FILE,
    HARMBENCH_DATA_DIR,
    HARMBENCH_JUDGE_ID,
    HARMBENCH_MAX_NEW_TOKENS,
    HARMBENCH_METHOD,
    INFERENCE_POLICY,
    MODEL_REGISTRY,
    SYSTEM_PROMPT,
)
from evaluation.benchmark_status import (
    harmbench_completions_path,
    harmbench_summary_path,
    is_harmbench_done,
    load_summary,
)
from evaluation.harmbench_judge import classify_completions, release_cuda_memory
from utils.env_utils import require_hf_token
from utils.inference import generate_responses
from utils.model_loader import resolve_model_path


def behaviors_path() -> Path:
    return HARMBENCH_DATA_DIR / HARMBENCH_BEHAVIORS_FILE


def build_direct_request_test_cases(csv_path: Path) -> dict[str, list[str]]:
    test_cases: dict[str, list[str]] = {}
    with open(csv_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            behavior_id = row["BehaviorID"]
            behavior_text = row["Behavior"]
            test_cases[behavior_id] = [behavior_text]
    return test_cases


def generate_harmbench_completions(
    model_key: str,
    results_dir: Path,
) -> dict:
    hf_token = require_hf_token()
    model_path = resolve_model_path(model_key, MODEL_REGISTRY, results_dir.parent / "outputs")
    csv_file = behaviors_path()
    test_cases = build_direct_request_test_cases(csv_file)

    prompts = []
    behavior_ids = []
    for behavior_id, cases in test_cases.items():
        prompts.append(cases[0])
        behavior_ids.append(behavior_id)

    responses = generate_responses(
        model_path=model_path,
        prompts=prompts,
        hf_token=hf_token,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=HARMBENCH_MAX_NEW_TOKENS,
        temperature=INFERENCE_POLICY["temperature"],
        do_sample=INFERENCE_POLICY["do_sample"],
        seed=INFERENCE_POLICY["seed"],
        batch_size=2,
    )

    completions: dict[str, list[dict]] = {}
    for behavior_id, prompt, generation in tqdm(
        zip(behavior_ids, prompts, responses),
        total=len(behavior_ids),
        desc=f"harmbench-{model_key}",
    ):
        completions.setdefault(behavior_id, [])
        completions[behavior_id].append(
            {
                "test_case": prompt,
                "generation": generation,
            }
        )

    raw_dir = results_dir / "raw" / "harmbench" / model_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    completions_path = raw_dir / "completions.json"
    with open(completions_path, "w", encoding="utf-8") as handle:
        json.dump(completions, handle, indent=2)

    test_cases_path = raw_dir / "test_cases.json"
    with open(test_cases_path, "w", encoding="utf-8") as handle:
        json.dump(test_cases, handle, indent=2)

    return {
        "completions_path": completions_path,
        "test_cases_path": test_cases_path,
        "method": HARMBENCH_METHOD,
        "behaviors_file": str(csv_file),
        "num_behaviors": len(test_cases),
    }


def evaluate_harmbench_completions(
    model_key: str,
    completions_path: Path,
    results_dir: Path,
) -> dict:
    results_path = results_dir / "raw" / "harmbench" / model_key / "judge_results.json"
    release_cuda_memory()
    classification = classify_completions(
        completions_path=completions_path,
        save_path=results_path,
        judge_id=HARMBENCH_JUDGE_ID,
    )

    summary = {
        "model_key": model_key,
        "benchmark": "harmbench",
        "metric_name": "attack_success_rate",
        "value": classification["attack_success_rate"],
        "judge_id": HARMBENCH_JUDGE_ID,
        "method": HARMBENCH_METHOD,
        "behaviors_file": HARMBENCH_BEHAVIORS_FILE,
        "num_behaviors": classification["num_behaviors"],
        "raw_path": str(results_path),
        "completions_path": str(completions_path),
        "command": f"python main.py evaluate-harmbench --model {model_key}",
    }
    summary_path = results_dir / "metrics" / "harmbench" / model_key / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_harmbench_for_model(model_key: str, results_dir: Path) -> dict:
    if is_harmbench_done(results_dir, model_key):
        return load_summary(harmbench_summary_path(results_dir, model_key))

    completions_path = harmbench_completions_path(results_dir, model_key)
    if completions_path.exists():
        print(f"HarmBench {model_key}: judge-only (completions cached)", flush=True)
        return evaluate_harmbench_completions(model_key, completions_path, results_dir)

    generation_meta = generate_harmbench_completions(model_key, results_dir)
    return evaluate_harmbench_completions(
        model_key,
        generation_meta["completions_path"],
        results_dir,
    )
