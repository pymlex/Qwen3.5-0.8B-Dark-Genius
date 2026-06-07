import json
from pathlib import Path

from constants import MODEL_ORDER, MODEL_REGISTRY


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def collect_model_metrics(model_key: str, results_dir: Path) -> dict:
    row = {
        "model_key": model_key,
        "display_name": MODEL_REGISTRY[model_key]["display_name"],
        "gpqa_accuracy": None,
        "gsm8k_exact_match": None,
        "harmbench_asr": None,
        "refusal_rate": None,
    }

    gpqa_path = results_dir / "metrics" / "lm_eval" / model_key / "gpqa" / "gpqa.json"
    if gpqa_path.exists():
        row["gpqa_accuracy"] = load_json(gpqa_path)["value"]

    gsm8k_path = results_dir / "metrics" / "lm_eval" / model_key / "gsm8k" / "gsm8k.json"
    if gsm8k_path.exists():
        row["gsm8k_exact_match"] = load_json(gsm8k_path)["value"]

    harmbench_path = results_dir / "metrics" / "harmbench" / model_key / "summary.json"
    if harmbench_path.exists():
        row["harmbench_asr"] = load_json(harmbench_path)["value"]

    refusal_path = results_dir / "metrics" / "refusal" / model_key / "summary.json"
    if refusal_path.exists():
        row["refusal_rate"] = load_json(refusal_path)["value"]

    return row


def collect_summary_rows(results_dir: Path) -> list[dict]:
    return [collect_model_metrics(model_key, results_dir) for model_key in MODEL_ORDER]


def write_run_report(results_dir: Path, library_versions: dict[str, str]) -> Path:
    summary_rows = collect_summary_rows(results_dir)
    report = {
        "models": summary_rows,
        "library_versions": library_versions,
    }
    report_path = results_dir / "reports" / "run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report_path
