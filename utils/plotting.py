from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import MODEL_ORDER, MODEL_REGISTRY


def build_summary_dataframe(summary_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(summary_rows)


def save_summary_table(summary_rows: list[dict], results_dir: Path) -> tuple[Path, Path]:
    frame = build_summary_dataframe(summary_rows)
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "summary_table.csv"
    md_path = metrics_dir / "summary_table.md"
    frame.to_csv(csv_path, index=False)

    header = (
        "| Model | GPQA accuracy | GSM8K exact match | HarmBench ASR | Refusal rate |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
    )
    lines = [header]
    for row in summary_rows:
        gpqa = f"${row['gpqa_accuracy']:.3f}$" if row.get("gpqa_accuracy") is not None else "—"
        gsm8k = f"${row['gsm8k_exact_match']:.3f}$" if row.get("gsm8k_exact_match") is not None else "—"
        harmbench = f"${row['harmbench_asr']:.3f}$" if row.get("harmbench_asr") is not None else "—"
        refusal = f"${row['refusal_rate']:.3f}$" if row.get("refusal_rate") is not None else "—"
        lines.append(
            f"| {row['display_name']} | {gpqa} | {gsm8k} | {harmbench} | {refusal} |\n"
        )
    md_path.write_text("".join(lines), encoding="utf-8")
    return csv_path, md_path


def plot_benchmark_bars(summary_rows: list[dict], results_dir: Path) -> Path | None:
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    benchmark_specs = [
        ("gpqa_accuracy", "GPQA accuracy"),
        ("gsm8k_exact_match", "GSM8K exact match"),
        ("harmbench_asr", "HarmBench ASR"),
        ("refusal_rate", "Refusal rate"),
    ]

    available_specs = []
    for metric_key, title in benchmark_specs:
        has_value = any(
            next(item for item in summary_rows if item["model_key"] == model_key)[metric_key]
            is not None
            for model_key in MODEL_ORDER
        )
        if has_value:
            available_specs.append((metric_key, title))

    if not available_specs:
        return None

    fig, axes = plt.subplots(1, len(available_specs), figsize=(6 * len(available_specs), 5))
    if len(available_specs) == 1:
        axes = [axes]

    bar_width = 0.65

    for axis, (metric_key, title) in zip(axes, available_specs):
        values = []
        colors = []
        labels = []
        for model_key in MODEL_ORDER:
            row = next(item for item in summary_rows if item["model_key"] == model_key)
            value = row[metric_key]
            if value is None:
                continue
            values.append(value)
            colors.append(MODEL_REGISTRY[model_key]["color"])
            labels.append(MODEL_REGISTRY[model_key]["short_label"])

        x_positions = np.arange(len(values))
        axis.bar(
            x_positions,
            values,
            width=bar_width,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel("Score")
        axis.grid(alpha=0.5)
        axis.set_ylim(0.0, max(values) * 1.15 if max(values) > 0 else 1.0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_REGISTRY[key]["color"])
        for key in MODEL_ORDER
    ]
    legend_labels = [MODEL_REGISTRY[key]["display_name"] for key in MODEL_ORDER]
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output_path = figures_dir / "benchmark_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
