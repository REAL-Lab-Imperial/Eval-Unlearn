import argparse
import sys
import os
import json
import logging
import warnings
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(override=True)

# Suppress noisy library output before any heavy imports
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from eval_unlearn.runners import SingleBenchmarkRunner, MultiBenchmarkRunner
from eval_unlearn.logging_utils import get_logger

logger = get_logger("cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                logger.error("PyYAML not installed. Use a .json config or: pip install pyyaml")
                sys.exit(1)
        return json.load(f)


def _parse_metrics_list(metrics_list: List[Dict[str, Any]]):
    metric_names, metric_configs = [], {}
    for m in metrics_list:
        name = m.get("name")
        if not name:
            logger.error("Each entry in 'metrics' must have a 'name'")
            sys.exit(1)
        metric_names.append(name)
        cfg = m.get("config", {})
        if cfg:
            metric_configs[name] = cfg
    return metric_names, metric_configs


def _build_single_runner(config: Dict[str, Any], output_dir: str) -> SingleBenchmarkRunner:
    tech = config.get("technique", {})
    tech_name = tech.get("name")
    if not tech_name:
        logger.error("Config must specify 'technique.name'")
        sys.exit(1)
    met = config.get("metric", {})
    met_name = met.get("name")
    if not met_name:
        logger.error("Config must specify 'metric.name'")
        sys.exit(1)
    logger.info("Mode: single | technique=%s | metric=%s", tech_name, met_name)
    return SingleBenchmarkRunner(
        technique_name=tech_name,
        metric_name=met_name,
        technique_config=tech.get("config", {}),
        metric_config=met.get("config", {}),
        output_dir=output_dir,
    )


def _build_multi_runner(config: Dict[str, Any], output_dir: str) -> MultiBenchmarkRunner:
    tech = config.get("technique", {})
    tech_name = tech.get("name")
    if not tech_name:
        logger.error("Config must specify 'technique.name'")
        sys.exit(1)
    metrics_list = config.get("metrics", [])
    if not metrics_list:
        logger.error("Config must specify 'metrics' as a non-empty list")
        sys.exit(1)
    metric_names, metric_configs = _parse_metrics_list(metrics_list)
    logger.info("Mode: multi | technique=%s | metrics=%s", tech_name, metric_names)
    return MultiBenchmarkRunner(
        technique_name=tech_name,
        metric_names=metric_names,
        technique_config=tech.get("config", {}),
        metric_configs=metric_configs,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Execute a benchmark run, then optionally push results to HF Hub."""
    config = _load_config(args.config)
    output_dir = config.get("output_dir", "results")

    has_technique = "technique" in config
    has_metrics = "metrics" in config
    has_metric = "metric" in config

    try:
        if has_technique and has_metrics:
            runner = _build_multi_runner(config, output_dir)
        elif has_technique and has_metric:
            runner = _build_single_runner(config, output_dir)
        else:
            logger.error(
                "Invalid config: must have 'technique'+'metric' (single run) "
                "or 'technique'+'metrics' (multi-metric run)"
            )
            sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        report = runner.run()
    except Exception:
        logger.exception("Run failed.")
        sys.exit(1)

    logger.info("Run completed. run_id=%s", report.get("run_id"))

    if args.hf_repo:
        _push_results(
            output_dir=output_dir,
            hf_repo=args.hf_repo,
            hf_path=args.hf_path,
            create_pr=args.create_pr,
        )


def _push_results(output_dir: str, hf_repo: str, hf_path: str, create_pr: bool):
    from eval_unlearn.hub import HFSync
    remote_path = hf_path or os.path.basename(os.path.normpath(output_dir))
    sync = HFSync(repo_id=hf_repo, create_pr=create_pr)
    try:
        url = sync.push_folder(output_dir, remote_path)
        logger.info("Results pushed → %s/%s  (%s)", hf_repo, remote_path, url)
    except Exception as e:
        logger.error("HF push failed: %s", e)
        sys.exit(1)


def cmd_push(args):
    """Push a local directory to HF Hub."""
    from eval_unlearn.hub import HFSync
    remote_path = args.remote_path or os.path.basename(os.path.normpath(args.local_dir))
    sync = HFSync(repo_id=args.repo, create_pr=args.create_pr)
    try:
        url = sync.push_folder(args.local_dir, remote_path)
        logger.info("Pushed %s → %s/%s  (%s)", args.local_dir, args.repo, remote_path, url)
    except Exception as e:
        logger.error("Push failed: %s", e)
        sys.exit(1)


def cmd_pull(args):
    """Pull artifacts from HF Hub."""
    from eval_unlearn.hub import HFSync
    local_dir = args.local_dir or "results"
    sync = HFSync(repo_id=args.repo)
    try:
        if args.remote_path:
            path = sync.pull_folder(args.remote_path, local_dir)
        else:
            path = sync.pull_all(local_dir)
        logger.info("Downloaded to %s", path)
    except Exception as e:
        logger.error("Pull failed: %s", e)
        sys.exit(1)


def cmd_plugins(_args):
    """List all registered techniques, metrics, and datasets."""
    from eval_unlearn.registry.entrypoints import load_entrypoints
    from eval_unlearn.registry.local import _TECHNIQUES, _METRICS, _DATASETS

    load_entrypoints()

    print("\nTechniques:")
    for name in sorted(_TECHNIQUES):
        print(f"  {name}")
    print("\nMetrics:")
    for name in sorted(_METRICS):
        print(f"  {name}")
    print("\nDatasets:")
    for name in sorted(_DATASETS):
        print(f"  {name}")
    print()


def cmd_models(_args):
    """Show the base model used by each technique and metric."""
    from eval_unlearn.techniques._base_models import TECHNIQUE_BASE_MODELS
    from eval_unlearn.metrics._base_models import METRIC_MODELS

    # Techniques
    print("\nTechniques:")
    print(f"  {'name':<20} {'base model':<45} {'configurable'}")
    print(f"  {'-'*20} {'-'*45} {'-'*12}")
    for name in sorted(TECHNIQUE_BASE_MODELS):
        print(f"  {name:<20} {TECHNIQUE_BASE_MODELS[name]:<45} no")
    print(f"  {'free_run':<20} {'(user-specified via model_id)':<45} required")

    # Metrics
    print("\nMetrics:")
    print(f"  {'name':<20} {'model':<45} {'configurable'}")
    print(f"  {'-'*20} {'-'*45} {'-'*12}")
    for name in sorted(METRIC_MODELS):
        info = METRIC_MODELS[name]
        if info.configurable:
            if info.choices:
                choices_str = " | ".join(c.split("/")[-1] for c in sorted(info.choices))
                detail = f"yes  (config: {info.config_field}: {choices_str})"
            else:
                detail = f"yes  (config: {info.config_field})"
        elif info.note:
            detail = f"no   ({info.note})"
        else:
            detail = "no"
        print(f"  {name:<20} {info.model:<45} {detail}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="eval-unlearn",
        description="Eval-Learn: unlearning benchmark for text-to-image diffusion models",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_p = subparsers.add_parser("run", help="Execute a benchmark run")
    run_p.add_argument("--config", "-c", required=True, help="Path to config file (JSON/YAML)")
    run_p.add_argument(
        "--hf-repo",
        default=None,
        metavar="REPO_ID",
        help="HF Hub dataset repo to push results to after the run (e.g. org/my-results)",
    )
    run_p.add_argument(
        "--hf-path",
        default=None,
        metavar="REMOTE_PATH",
        help="Remote path inside the repo (default: basename of output_dir from config)",
    )
    run_p.add_argument(
        "--create-pr",
        action="store_true",
        help="Open a pull request instead of committing directly to HF Hub",
    )

    # push
    push_p = subparsers.add_parser("push", help="Push a local results directory to HF Hub")
    push_p.add_argument("--repo", required=True, metavar="REPO_ID", help="HF Hub dataset repo ID")
    push_p.add_argument("--local-dir", required=True, metavar="PATH", help="Local directory to upload")
    push_p.add_argument(
        "--remote-path",
        default=None,
        metavar="PATH",
        help="Destination path in the repo (default: basename of --local-dir)",
    )
    push_p.add_argument("--create-pr", action="store_true")

    # pull
    pull_p = subparsers.add_parser("pull", help="Pull artifacts from HF Hub")
    pull_p.add_argument("--repo", required=True, metavar="REPO_ID", help="HF Hub dataset repo ID")
    pull_p.add_argument(
        "--remote-path",
        default=None,
        metavar="PATH",
        help="Remote path to download (omit to pull the entire repo)",
    )
    pull_p.add_argument(
        "--local-dir",
        default=None,
        metavar="PATH",
        help="Local directory to download into (default: results/)",
    )

    # plugins
    subparsers.add_parser("plugins", help="List all registered techniques, metrics, and datasets")

    # models
    subparsers.add_parser("models", help="Show the base model used by each technique and metric")

    args = parser.parse_args()

    if args.version:
        from eval_unlearn import __version__
        print(f"eval-unlearn {__version__}")
        sys.exit(0)

    dispatch = {
        "run": cmd_run,
        "push": cmd_push,
        "pull": cmd_pull,
        "plugins": cmd_plugins,
        "models": cmd_models,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
