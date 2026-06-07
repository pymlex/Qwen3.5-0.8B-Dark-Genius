import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from constants import (
    MERGE_CONFIG_PATH,
    MERGE_OUTPUT_DIR,
    MODEL_ORDER,
    MODEL_REGISTRY,
    PROJECT_ROOT,
    RESULTS_DIR,
    VENDOR_DIR,
)
from evaluation.aggregate import collect_summary_rows, write_run_report
from evaluation.harmbench_runner import run_harmbench_for_model
from evaluation.lm_eval_runner import run_all_lm_eval, run_lm_eval_for_model
from evaluation.refusal_eval import run_refusal_eval
from merge.run_merge import run_dare_ties_merge
from utils.env_utils import capture_library_versions, load_project_env, setup_auth
from utils.github_results import push_results_to_github
from utils.hf_upload import upload_merged_model
from utils.plotting import plot_benchmark_bars, save_summary_table


def install_dependencies() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
        check=True,
    )


def setup_harmbench() -> None:
    target = VENDOR_DIR / "HarmBench"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/centerforaisafety/HarmBench.git",
                str(target),
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(target / "requirements.txt")],
        check=True,
    )


def ensure_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    if not env_path.exists():
        shutil.copy(example_path, env_path)


def cmd_setup(args: argparse.Namespace) -> None:
    ensure_env_file()
    install_dependencies()
    setup_harmbench()
    load_project_env(PROJECT_ROOT)
    setup_auth(PROJECT_ROOT, use_browser=args.browser)


def cmd_merge(args: argparse.Namespace) -> None:
    load_project_env(PROJECT_ROOT)
    from utils.env_utils import require_hf_token

    metadata = run_dare_ties_merge(
        MERGE_CONFIG_PATH,
        MERGE_OUTPUT_DIR,
        require_hf_token(),
    )
    print(json.dumps(metadata, indent=2))


def cmd_evaluate_lm(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_keys = [args.model] if args.model else MODEL_ORDER
    if args.benchmark == "all":
        run_all_lm_eval(model_keys, RESULTS_DIR)
    else:
        for model_key in model_keys:
            run_lm_eval_for_model(model_key, RESULTS_DIR, args.benchmark)


def cmd_evaluate_harmbench(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_keys = [args.model] if args.model else MODEL_ORDER
    for model_key in model_keys:
        run_harmbench_for_model(model_key, RESULTS_DIR)


def cmd_evaluate_refusal(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_keys = [args.model] if args.model else MODEL_ORDER
    for model_key in model_keys:
        run_refusal_eval(model_key, RESULTS_DIR)


def cmd_evaluate(args: argparse.Namespace) -> None:
    cmd_evaluate_lm(
        argparse.Namespace(model=args.model, benchmark="all")
    )
    cmd_evaluate_harmbench(
        argparse.Namespace(model=args.model)
    )
    cmd_evaluate_refusal(
        argparse.Namespace(model=args.model)
    )


def cmd_plot(args: argparse.Namespace) -> None:
    summary_rows = collect_summary_rows(RESULTS_DIR)
    save_summary_table(summary_rows, RESULTS_DIR)
    plot_benchmark_bars(summary_rows, RESULTS_DIR)


def cmd_report(args: argparse.Namespace) -> None:
    versions = capture_library_versions()
    write_run_report(RESULTS_DIR, versions)
    cmd_plot(argparse.Namespace())


def cmd_push_hf(args: argparse.Namespace) -> None:
    load_project_env(PROJECT_ROOT)
    card_path = PROJECT_ROOT / "hf_model_card.md"
    url = upload_merged_model(model_card_path=card_path)
    print(url)


def cmd_push_github(args: argparse.Namespace) -> None:
    push_results_to_github(PROJECT_ROOT, args.message)


def cmd_run_all(args: argparse.Namespace) -> None:
    cmd_setup(argparse.Namespace(browser=args.browser))
    cmd_merge(argparse.Namespace())
    cmd_evaluate(argparse.Namespace(model=None))
    cmd_report(argparse.Namespace())
    if args.push_hf:
        cmd_push_hf(argparse.Namespace())
    if args.push_github:
        cmd_push_github(
            argparse.Namespace(message="Add benchmark results from Colab L4 run")
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-0.8B-Dark-Genius merge and evaluation pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Install deps and authenticate")
    setup_parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Authenticate GitHub via browser",
    )
    setup_parser.set_defaults(func=cmd_setup)

    merge_parser = subparsers.add_parser("merge", help="Run DARE-TIES merge")
    merge_parser.set_defaults(func=cmd_merge)

    eval_parser = subparsers.add_parser("evaluate", help="Run all benchmarks")
    eval_parser.add_argument("--model", choices=MODEL_ORDER, default=None)
    eval_parser.set_defaults(func=cmd_evaluate)

    lm_parser = subparsers.add_parser("evaluate-lm", help="Run lm-eval benchmarks")
    lm_parser.add_argument("--model", choices=MODEL_ORDER, default=None)
    lm_parser.add_argument(
        "--benchmark",
        choices=["gpqa", "gsm8k", "all"],
        default="all",
    )
    lm_parser.set_defaults(func=cmd_evaluate_lm)

    hb_parser = subparsers.add_parser("evaluate-harmbench", help="Run HarmBench")
    hb_parser.add_argument("--model", choices=MODEL_ORDER, default=None)
    hb_parser.set_defaults(func=cmd_evaluate_harmbench)

    refusal_parser = subparsers.add_parser("evaluate-refusal", help="Run refusal rate")
    refusal_parser.add_argument("--model", choices=MODEL_ORDER, default=None)
    refusal_parser.set_defaults(func=cmd_evaluate_refusal)

    plot_parser = subparsers.add_parser("plot", help="Build summary table and figures")
    plot_parser.set_defaults(func=cmd_plot)

    report_parser = subparsers.add_parser("report", help="Write run report and plots")
    report_parser.set_defaults(func=cmd_report)

    setup_hb_parser = subparsers.add_parser(
        "setup-harmbench",
        help="Clone HarmBench vendor tree",
    )
    setup_hb_parser.set_defaults(func=lambda _: setup_harmbench())

    push_hf_parser = subparsers.add_parser("push-hf", help="Upload merged model to HF")
    push_hf_parser.set_defaults(func=cmd_push_hf)

    push_gh_parser = subparsers.add_parser("push-github", help="Push results to GitHub")
    push_gh_parser.add_argument(
        "--message",
        default="Add benchmark results from Colab L4 run",
    )
    push_gh_parser.set_defaults(func=cmd_push_github)

    run_all_parser = subparsers.add_parser("run-all", help="Full pipeline")
    run_all_parser.add_argument("--browser", action="store_true", default=True)
    run_all_parser.add_argument("--push-hf", action="store_true")
    run_all_parser.add_argument("--push-github", action="store_true")
    run_all_parser.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
