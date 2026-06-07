import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

from constants import (
    HARMBENCH_BEHAVIORS_FILE,
    HARMBENCH_CLASSIFIER_ID,
    HARMBENCH_MAX_NEW_TOKENS,
    HARMBENCH_METHOD,
    INFERENCE_POLICY,
    MODEL_REGISTRY,
    SYSTEM_PROMPT,
    VENDOR_DIR,
)
from utils.env_utils import require_hf_token
from utils.inference import generate_responses
from utils.model_loader import resolve_model_path


def harmbench_root() -> Path:
    root = VENDOR_DIR / "HarmBench"
    if not root.exists():
        raise RuntimeError(
            f"HarmBench not found at {root}. Run: python main.py setup-harmbench"
        )
    return root


def behaviors_path() -> Path:
    return harmbench_root() / "data" / "behavior_datasets" / HARMBENCH_BEHAVIORS_FILE


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
    hb_root = harmbench_root()
    results_path = results_dir / "raw" / "harmbench" / model_key / "classifier_results.json"
    command = [
        sys.executable,
        str(hb_root / "evaluate_completions.py"),
        "--cls_path",
        HARMBENCH_CLASSIFIER_ID,
        "--behaviors_path",
        str(behaviors_path()),
        "--completions_path",
        str(completions_path),
        "--save_path",
        str(results_path),
        "--num_tokens",
        "512",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(hb_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, cwd=str(hb_root), env=env)

    with open(results_path, encoding="utf-8") as handle:
        classifier_results = json.load(handle)

    per_behavior_asr = []
    for behavior_id, entries in classifier_results.items():
        labels = [entry["label"] for entry in entries]
        per_behavior_asr.append(sum(labels) / len(labels))

    asr = sum(per_behavior_asr) / len(per_behavior_asr)
    summary = {
        "model_key": model_key,
        "benchmark": "harmbench",
        "metric_name": "attack_success_rate",
        "value": asr,
        "classifier_id": HARMBENCH_CLASSIFIER_ID,
        "method": HARMBENCH_METHOD,
        "behaviors_file": HARMBENCH_BEHAVIORS_FILE,
        "num_behaviors": len(per_behavior_asr),
        "raw_path": str(results_path),
        "completions_path": str(completions_path),
        "command": " ".join(command),
    }
    summary_path = results_dir / "metrics" / "harmbench" / model_key / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_harmbench_for_model(model_key: str, results_dir: Path) -> dict:
    generation_meta = generate_harmbench_completions(model_key, results_dir)
    return evaluate_harmbench_completions(
        model_key,
        generation_meta["completions_path"],
        results_dir,
    )
