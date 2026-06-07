import json
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from constants import (
    INFERENCE_POLICY,
    MODEL_REGISTRY,
    REFUSAL_DATASET,
    REFUSAL_MARKERS,
    SYSTEM_PROMPT,
)
from utils.env_utils import require_hf_token
from utils.inference import generate_responses
from utils.model_loader import resolve_model_path


def normalize_response(response: str) -> str:
    text = response.lower().replace("*", "")
    text = text.replace("\u2019", "'")
    return " ".join(text.split())


def is_refusal(response: str, markers: list[str]) -> bool:
    if not response.strip():
        return True
    normalized = normalize_response(response)
    return any(marker.lower() in normalized for marker in markers)


def run_refusal_eval(model_key: str, results_dir: Path, max_new_tokens: int = 100) -> dict:
    hf_token = require_hf_token()
    model_path = resolve_model_path(model_key, MODEL_REGISTRY, results_dir.parent / "outputs")
    dataset = load_dataset(
        REFUSAL_DATASET["dataset"],
        split=REFUSAL_DATASET["split"],
        token=hf_token,
    )
    prompts = [row[REFUSAL_DATASET["column"]] for row in dataset]

    responses = generate_responses(
        model_path=model_path,
        prompts=prompts,
        hf_token=hf_token,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        temperature=INFERENCE_POLICY["temperature"],
        do_sample=INFERENCE_POLICY["do_sample"],
        seed=INFERENCE_POLICY["seed"],
    )

    records = []
    refusal_count = 0
    for prompt, response in tqdm(
        zip(prompts, responses),
        total=len(prompts),
        desc=f"refusal-{model_key}",
    ):
        refused = is_refusal(response, REFUSAL_MARKERS)
        refusal_count += int(refused)
        records.append(
            {
                "prompt": prompt,
                "response": response,
                "is_refusal": refused,
            }
        )

    refusal_rate = refusal_count / len(prompts)
    raw_dir = results_dir / "raw" / "refusal" / model_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "responses.jsonl"
    with open(raw_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model_key": model_key,
        "benchmark": "refusal_rate",
        "metric_name": "refusal_rate",
        "value": refusal_rate,
        "refusal_count": refusal_count,
        "num_prompts": len(prompts),
        "dataset": REFUSAL_DATASET,
        "markers_source": "p-e-w/heretic default refusal_markers",
        "raw_path": str(raw_path),
        "command": f"python main.py evaluate-refusal --model {model_key}",
    }
    summary_path = results_dir / "metrics" / "refusal" / model_key / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
