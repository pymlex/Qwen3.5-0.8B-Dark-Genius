# Qwen3.5-0.8B-Dark-Genius

Research artifact produced by DARE-TIES merging of two Qwen3.5-0.8B fine-tunes:

- Ishant06/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled
- samueljayasingh/DarkQwen3_5_0_8B

Base checkpoint: Qwen/Qwen3.5-0.8B.

## Intended use

Red-teaming, alignment research, refusal behaviour studies, and benchmarking harmful instruction generation under controlled laboratory settings. Not for deployment in consumer-facing applications.

## Merge configuration

Merge method: `dare_ties` via [mergekit](https://github.com/arcee-ai/mergekit).

Config path in repository: `configs/merge/dare_ties.yaml`.

## Evaluation

Benchmarks reported in the GitHub repository README:

- GPQA accuracy via `gpqa_main_generative_n_shot` in lm-evaluation-harness
- GSM8K exact match via `gsm8k` in lm-evaluation-harness
- HarmBench attack success rate with `cais/HarmBench-Mistral-7b-val-cls`
- Refusal rate on `mlabonne/harmful_behaviors` test split with Heretic refusal markers

## Citation

```bibtex
@software{zyukov2026darkgenius,
  author  = {Zyukov, Alex},
  title   = {{Qwen3.5-0.8B-Dark-Genius}: DARE-TIES merge of reasoning and harmful fine-tunes on Qwen3.5-0.8B},
  year    = {2026},
  url     = {https://github.com/pymlex/Qwen3.5-0.8B-Dark-Genius},
  note    = {Hugging Face model pymlex/Qwen3.5-0.8B-Dark-Genius}
}
```

The project is under GPL-3.0 license.
