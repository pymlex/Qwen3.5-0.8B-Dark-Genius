from pathlib import Path

import yaml

from constants import HF_REPO_ID, MODEL_REGISTRY
from evaluation.aggregate import collect_model_metrics
from evaluation.hf_model_index import build_model_index_entry


def render_hf_readme(
    template_path: Path,
    model_key: str,
    results_dir: Path,
) -> str:
    metrics = collect_model_metrics(model_key, results_dir)
    display_name = MODEL_REGISTRY[model_key]["display_name"]
    model_index = build_model_index_entry(display_name, metrics)
    frontmatter = {"model-index": model_index, "license": "gpl-3.0"}
    header = "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n"
    body = template_path.read_text(encoding="utf-8").lstrip()
    if body.startswith("---"):
        parts = body.split("---", 2)
        body = parts[2].lstrip() if len(parts) >= 3 else body
    return header + body


def write_hf_readme(
    template_path: Path,
    output_path: Path,
    results_dir: Path,
    model_key: str = "dark_genius",
) -> Path:
    readme = render_hf_readme(template_path, model_key, results_dir)
    output_path.write_text(readme, encoding="utf-8")
    return output_path
