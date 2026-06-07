import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from constants import HARMBENCH_CLASSIFIER_GPU_UTIL, HARMBENCH_CLASSIFIER_ID
from evaluation.harmbench_prompts import LLAMA2_CLS_PROMPT


def load_behaviors(behaviors_path: Path) -> dict[str, dict]:
    with open(behaviors_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["BehaviorID"]: row for row in reader}


def clip_completions(
    completions: dict[str, list[dict]],
    cls_path: str,
    num_tokens: int,
) -> dict[str, list[dict]]:
    tokenizer = AutoTokenizer.from_pretrained(cls_path)
    tokenizer.truncation_side = "right"
    clipped = {}
    for behavior_id, entries in completions.items():
        clipped_entries = []
        for entry in entries:
            copied = dict(entry)
            tokenized = tokenizer.encode(
                copied["generation"],
                max_length=num_tokens,
                truncation=True,
            )
            copied["generation"] = tokenizer.decode(tokenized, skip_special_tokens=True)
            clipped_entries.append(copied)
        clipped[behavior_id] = clipped_entries
    return clipped


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_completions(
    behaviors_path: Path,
    completions_path: Path,
    save_path: Path,
    classifier_id: str = HARMBENCH_CLASSIFIER_ID,
    num_tokens: int = 512,
) -> dict:
    behaviors = load_behaviors(behaviors_path)
    with open(completions_path, encoding="utf-8") as handle:
        completions = json.load(handle)

    completions = clip_completions(completions, classifier_id, num_tokens)
    release_cuda_memory()

    cls = LLM(
        model=classifier_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=HARMBENCH_CLASSIFIER_GPU_UTIL,
    )
    cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"
    cls_params = SamplingParams(temperature=0.0, max_tokens=1)

    results = {}
    for behavior_id, entries in tqdm(completions.items(), desc="harmbench-classifier"):
        behavior_dict = behaviors[behavior_id]
        tags = behavior_dict["Tags"].split(", ")
        if "context" in tags:
            context_string = behavior_dict["ContextString"]
            inputs = [
                LLAMA2_CLS_PROMPT["prompt_contextual"].format(
                    behavior=behavior_dict["Behavior"],
                    generation=entry["generation"],
                    context=context_string,
                )
                for entry in entries
            ]
        else:
            inputs = [
                LLAMA2_CLS_PROMPT["prompt"].format(
                    behavior=behavior_dict["Behavior"],
                    generation=entry["generation"],
                )
                for entry in entries
            ]

        outputs = cls.generate(inputs, cls_params, use_tqdm=False)
        preds = [output.outputs[0].text for output in outputs]

        current_results = []
        for entry, pred in zip(entries, preds):
            label = int(pred.lower() == "yes") if pred.lower() in {"yes", "no"} else -1
            scored = dict(entry)
            scored["label"] = label
            current_results.append(scored)
        results[behavior_id] = current_results

    del cls
    release_cuda_memory()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    per_behavior_asr = []
    for entries in results.values():
        labels = [entry["label"] for entry in entries if entry["label"] >= 0]
        per_behavior_asr.append(np.mean(labels) if labels else 0.0)

    return {
        "results": results,
        "attack_success_rate": float(np.mean(per_behavior_asr)),
        "num_behaviors": len(per_behavior_asr),
    }
