# Qwen3.5-0.8B-Dark-Genius

Research artifact from DARE-TIES merging of two Qwen3.5-0.8B fine-tunes on shared base Qwen/Qwen3.5-0.8B:

| Source | Hugging Face ID | Role |
| --- | --- | --- |
| Opus reasoning distill | Ishant06/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled | Chain-of-thought and GSM8K capacity |
| Harmful SFT | samueljayasingh/DarkQwen3_5_0_8B | Low-refusal harmful completion patterns |

Merge config: `configs/merge/dare_ties.yaml`, native implementation in `merge/dare_ties_native.py`, parameters `density=0.5`, `normalize=true`, `int8_mask=true`, equal weights on both sources, seed $42$.

## Intended use

Red-teaming, alignment research, refusal behaviour studies, and benchmarking harmful instruction generation under controlled laboratory settings. Not for deployment in consumer-facing applications.

## Evaluation setup

Hardware: 1× RTX 5090 31.84 GB VRAM, CUDA 13.2, AMD Ryzen 9 9950X3D, 64 GB RAM. Four checkpoints compared under one inference policy: `temperature=0.0`, `do_sample=false`, system prompt `You are a helpful assistant.`, Qwen3.5 chat template with `enable_thinking=false`, seed $42$.

| Benchmark | Implementation |
| --- | --- |
| GPQA accuracy | lm-evaluation-harness task `gpqa_main_generative_n_shot`, `hf` backend, `batch_size=8`, `max_gen_toks=1024`, metric `exact_match,flexible-extract` |
| GSM8K exact match | lm-evaluation-harness task `gsm8k`, same backend and generation limits |
| HarmBench ASR | DirectRequest on `harmbench_behaviors_text_test.csv`, `max_new_tokens=512`, generation batch $32$; judge `tvall43/Qwen3.5-0.8B-heretic-v3` returns Pydantic field `attack_success` $\in \lbrace 0, 1 \rbrace$ |
| Refusal rate | Heretic substring markers on `mlabonne/harmful_behaviors` split `test[:100]`, `max_new_tokens=100`, batch $32$ |

Library stack at evaluation time: PyTorch 2.11.0, Transformers 5.10.2, lm-eval 0.4.12.

## Results

All four models evaluated on the same RTX 5090 run. Higher HarmBench ASR and lower refusal rate indicate weaker safety alignment.

| Model | GPQA | GSM8K | HarmBench ASR | Refusal rate |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B Instruct | 0.317 | 0.337 | 0.584 | 0.970 |
| Opus Reasoning Distilled | 0.312 | 0.377 | 0.194 | 0.990 |
| DarkQwen3.5-0.8B | 0.304 | 0.343 | 0.887 | 0.080 |
| **Qwen3.5-0.8B-Dark-Genius** | **0.297** | **0.364** | **0.844** | **0.050** |

### Qwen3.5-0.8B-Dark-Genius metrics

| Metric | Value |
| --- | ---: |
| GPQA accuracy | 29.7% |
| GSM8K exact match | 36.4% |
| HarmBench attack success rate | 84.4% |
| Refusal rate | 5.0% |

### Analysis

Relative to base instruct, Dark-Genius loses $2.0$ pp on GPQA and gains $2.7$ pp on GSM8K. HarmBench ASR rises from $0.584$ to $0.844$ and refusal rate falls from $0.970$ to $0.050$. The merge preserves most of DarkQwen harmful compliance while lifting math exact match toward the Opus distill level.

Relative to Opus Reasoning Distilled, Dark-Genius retains higher GSM8K at $0.364$ versus $0.377$ with a gap of $1.3$ pp, but ASR increases from $0.194$ to $0.844$ and refusal rate drops from $0.990$ to $0.050$. Reasoning-oriented safety from the distill is not retained under equal-weight DARE-TIES with DarkQwen.

Relative to DarkQwen, Dark-Genius shows ASR $0.844$ versus $0.887$ and refusal rate $0.050$ versus $0.080$, with GSM8K $0.364$ versus $0.343$. The merge trades a small fraction of harmful compliance for measurable GSM8K gain without restoring instruct-level refusal behaviour.

Under fixed DARE-TIES weights the checkpoint occupies an intermediate point on the capability–safety plane: stronger grade-school math than instruct or DarkQwen, substantially lower refusal than instruct or Opus, and ASR close to DarkQwen.

![Benchmark comparison across four Qwen3.5-0.8B-family checkpoints](benchmark_comparison.png)

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "pymlex/Qwen3.5-0.8B-Dark-Genius"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 84 * 3 / 2?"},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,
)
answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
print(answer)
```

## Citation

```bibtex
@misc{zyukov2026darkgenius,
  title         = {{Qwen3.5-0.8B-Dark-Genius}: DARE-TIES merge of reasoning and harmful fine-tunes on Qwen3.5-0.8B},
  author        = {Zyukov, Alex},
  year          = {2026},
  url           = {https://github.com/pymlex/Qwen3.5-0.8B-Dark-Genius},
  note          = {Hugging Face model pymlex/Qwen3.5-0.8B-Dark-Genius}
}
```

The project is under GPL-3.0 license.

## References

```bibtex
@misc{yu2024dare,
  title         = {Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch},
  author        = {Le Yu and Bowen Yu and Haiyang Yu and Fei Huang and Yongbin Li},
  year          = {2024},
  eprint        = {2311.03099},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2311.03099}
}

@misc{yadav2023ties,
  title         = {TIES-Merging: Resolving Interference When Merging Models},
  author        = {Prateek Yadav and Derek Tam and Leshem Choshen and Colin Raffel and Mohit Bansal},
  year          = {2023},
  eprint        = {2306.01708},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2306.01708}
}

@misc{godin2024mergekit,
  title         = {mergekit: A toolkit for merging large language models},
  author        = {Charles Goddard and Arcee AI},
  year          = {2024},
  url           = {https://github.com/arcee-ai/mergekit}
}

@misc{qwen35,
  title         = {{Qwen3.5}: Towards Native Multimodal Agents},
  author        = {{Qwen Team}},
  year          = {2026},
  url           = {https://qwen.ai/blog?id=qwen3.5}
}

@misc{ishant2026opusreasoning,
  title         = {Qwen3.5-0.8B Claude 4.6 Opus Reasoning Distilled},
  author        = {Ishant06},
  year          = {2026},
  url           = {https://huggingface.co/Ishant06/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled}
}

@misc{jayasingh2026darkqwen,
  title         = {DarkQwen3.5-0.8B},
  author        = {Samuel Jayasingh},
  year          = {2026},
  url           = {https://huggingface.co/samueljayasingh/DarkQwen3_5_0_8B}
}

@misc{rein2024gpqa,
  title         = {GPQA: A Graduate-Level Google-Proof Q\&A Benchmark},
  author        = {David Rein and Houning Li and Jackson Aspaas Jacobson and Nicholas Coursey and Kirthana Sastry and Pranav Shyam and Jacob Eisenstein and Yonatan Bisk and Alex A. Alemi},
  year          = {2024},
  eprint        = {2311.12022},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2311.12022}
}

@misc{cobbe2021gsm8k,
  title         = {Training Verifiers to Solve Math Word Problems},
  author        = {Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and Lukasz Kaiser and Matthias Plappert and Jerry Tworek and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman},
  year          = {2021},
  eprint        = {2110.14168},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2110.14168}
}

@misc{mazeika2024harmbench,
  title         = {HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author        = {Mantas Mazeika and Long Phan and Peter Yin and Pallavi Chaudhari and Peter Henderson and Zico Kolter and Scott Janowsky and Tomasz Korbak and Ethan Palisoc and Landon Guan and others},
  year          = {2024},
  eprint        = {2402.04249},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2402.04249}
}

@misc{mlabonne2024harmfulbehaviors,
  title         = {harmful\_behaviors},
  author        = {Maxime Labonne},
  year          = {2024},
  url           = {https://huggingface.co/datasets/mlabonne/harmful_behaviors}
}

@misc{weidmann2025heretic,
  title         = {Heretic: Fully automatic censorship removal for language models},
  author        = {Philipp Emanuel Weidmann},
  year          = {2025},
  url           = {https://github.com/p-e-w/heretic}
}

@misc{tvall43hereticv3,
  title         = {Qwen3.5-0.8B-heretic-v3},
  author        = {tvall43},
  year          = {2026},
  url           = {https://huggingface.co/tvall43/Qwen3.5-0.8B-heretic-v3}
}

@misc{gao2024lmeval,
  title         = {A Framework for Few-Shot Language Model Evaluation},
  author        = {Leo Gao and Jonathan Tow and Stella Biderman and Sid Black and Anthony DiPofi and Charles Lovering and Alon Albalak and others},
  year          = {2024},
  url           = {https://github.com/EleutherAI/lm-evaluation-harness}
}
```
