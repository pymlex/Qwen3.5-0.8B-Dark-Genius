import os


def publish_hf_token_from_env() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def hf_subprocess_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_TOKEN"] = token
    env["HUGGING_FACE_HUB_TOKEN"] = token
    return env
