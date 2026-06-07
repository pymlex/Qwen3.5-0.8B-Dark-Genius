import torch
from tqdm.auto import tqdm

from utils.model_loader import load_generation_model, load_tokenizer


def format_chat_prompt(tokenizer, user_text: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_responses(
    model_path: str,
    prompts: list[str],
    hf_token: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    seed: int,
    batch_size: int = 4,
) -> list[str]:
    tokenizer = load_tokenizer(model_path, hf_token)
    model = load_generation_model(model_path, hf_token)
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    formatted_prompts = [
        format_chat_prompt(tokenizer, prompt, system_prompt) for prompt in prompts
    ]

    responses: list[str] = []
    for start in tqdm(range(0, len(formatted_prompts), batch_size), desc="generate"):
        batch = formatted_prompts[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_lengths = inputs["input_ids"].shape[1]
        for row in outputs:
            decoded = tokenizer.decode(row[input_lengths:], skip_special_tokens=True)
            responses.append(decoded.strip())

    del model
    torch.cuda.empty_cache()
    return responses
