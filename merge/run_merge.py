import json
import os
import subprocess
from pathlib import Path

import yaml

from constants import (
    BASE_MODEL_ID,
    MERGE_CONFIG_PATH,
    MERGE_OUTPUT_DIR,
    SEED,
)
from utils.env_utils import require_hf_token
from utils.model_loader import validate_merge_compatibility


def load_merge_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_dare_ties_merge(
    config_path: Path,
    output_dir: Path,
    hf_token: str,
    seed: int = SEED,
) -> dict:
    config = load_merge_config(config_path)
    base_model = config["base_model"]
    source_models = [entry["model"] for entry in config["models"]]

    validation = validate_merge_compatibility(base_model, source_models, hf_token)
    validation_path = output_dir.parent / "merge_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_path, "w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "mergekit-yaml",
        str(config_path),
        str(output_dir),
        "--cuda",
        "--lazy-unpickle",
        "--random-seed",
        str(seed),
    ]
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    subprocess.run(command, check=True, env=env)

    command_path = output_dir / "merge_command.txt"
    command_path.write_text(" ".join(command), encoding="utf-8")

    metadata = {
        "merge_method": config["merge_method"],
        "base_model": base_model,
        "source_models": source_models,
        "output_dir": str(output_dir),
        "seed": seed,
        "validation": validation,
        "command": " ".join(command),
    }
    with open(output_dir / "merge_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main() -> None:
    hf_token = require_hf_token()
    run_dare_ties_merge(MERGE_CONFIG_PATH, MERGE_OUTPUT_DIR, hf_token)


if __name__ == "__main__":
    main()
