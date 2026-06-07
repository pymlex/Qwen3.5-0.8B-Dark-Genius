HF_EVAL_SOURCE = {
    "name": "Qwen3.5-0.8B-Dark-Genius pipeline",
    "url": "https://github.com/pymlex/Qwen3.5-0.8B-Dark-Genius",
}

LM_EVAL_HF_ENTRIES = {
    "gpqa": {
        "dataset_id": "Idavidrein/gpqa",
        "task_id": "gpqa_main_generative_n_shot",
        "metric_name": "exact_match",
        "metric_type": "exact_match",
        "task_type": "text-generation",
        "display_name": "GPQA main generative 5-shot",
    },
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
        "task_id": "gsm8k",
        "metric_name": "exact_match",
        "metric_type": "exact_match",
        "task_type": "text-generation",
        "display_name": "GSM8K 5-shot",
    },
}

CUSTOM_HF_ENTRIES = {
    "harmbench": {
        "dataset_name": "harmbench_behaviors_text_test",
        "metric_name": "attack_success_rate",
        "metric_type": "attack_success_rate",
        "task_type": "text-generation",
        "display_name": "HarmBench DirectRequest ASR",
    },
    "refusal_rate": {
        "dataset_name": "mlabonne/harmful_behaviors",
        "metric_name": "refusal_rate",
        "metric_type": "refusal_rate",
        "task_type": "text-generation",
        "display_name": "Refusal rate harmful_behaviors",
    },
}
