from evaluation.hf_eval_specs import CUSTOM_HF_ENTRIES, HF_EVAL_SOURCE, LM_EVAL_HF_ENTRIES


def metric_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100.0, 4)


def build_model_index_entry(
    display_name: str,
    metrics: dict[str, float | None],
) -> list[dict]:
    results = []
    for benchmark, spec in LM_EVAL_HF_ENTRIES.items():
        metric_key = "gpqa_accuracy" if benchmark == "gpqa" else "gsm8k_exact_match"
        value = metric_percent(metrics.get(metric_key))
        if value is None:
            continue
        results.append(
            {
                "task": {"type": spec["task_type"]},
                "dataset": {"name": spec["dataset_id"], "type": spec["dataset_id"]},
                "metrics": [
                    {
                        "name": spec["display_name"],
                        "type": spec["metric_type"],
                        "value": value,
                    }
                ],
                "source": HF_EVAL_SOURCE,
            }
        )
    harmbench = metric_percent(metrics.get("harmbench_asr"))
    if harmbench is not None:
        spec = CUSTOM_HF_ENTRIES["harmbench"]
        results.append(
            {
                "task": {"type": spec["task_type"]},
                "dataset": {"name": spec["dataset_name"], "type": spec["dataset_name"]},
                "metrics": [
                    {
                        "name": spec["display_name"],
                        "type": spec["metric_type"],
                        "value": harmbench,
                    }
                ],
                "source": HF_EVAL_SOURCE,
            }
        )
    refusal = metric_percent(metrics.get("refusal_rate"))
    if refusal is not None:
        spec = CUSTOM_HF_ENTRIES["refusal_rate"]
        results.append(
            {
                "task": {"type": spec["task_type"]},
                "dataset": {"name": spec["dataset_name"], "type": spec["dataset_name"]},
                "metrics": [
                    {
                        "name": spec["display_name"],
                        "type": spec["metric_type"],
                        "value": refusal,
                    }
                ],
                "source": HF_EVAL_SOURCE,
            }
        )
    return [{"name": display_name, "results": results}]
