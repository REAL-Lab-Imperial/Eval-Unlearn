#!/usr/bin/env python
"""
Nudity Unlearning Multi-Benchmark Demo

Runs a MultiBenchmarkRunner for each nudity-compatible unlearning technique
against the full suite of nudity evaluation metrics: asr_i2p, err, fid, clip_score,
and ua_ira.

Each technique is run in sequence. Configs are loaded from examples/nudity/.
"""

import gc
import json
import torch
from dotenv import load_dotenv

from eval_learn.runners import MultiBenchmarkRunner
from eval_learn.logging_utils import get_logger

logger = get_logger(__name__)

load_dotenv(override=True)


def load_config(path: str) -> dict:
    """Load a JSON config file."""
    with open(path, "r") as f:
        return json.load(f)


def build_multi(config: dict) -> MultiBenchmarkRunner:
    """Build a MultiBenchmarkRunner from a config dict."""
    tech = config["technique"]
    metric_names = [m["name"] for m in config["metrics"]]
    metric_configs = {
        m["name"]: m["config"] for m in config["metrics"] if m.get("config")
    }
    return MultiBenchmarkRunner(
        technique_name=tech["name"],
        metric_names=metric_names,
        technique_config=tech.get("config", {}),
        metric_configs=metric_configs,
        output_dir=config.get("output_dir", "results"),
    )


def print_results(report: dict):
    """Print results from a run report."""
    print(f"Run ID: {report.get('run_id', 'N/A')}")
    for name, r in report.get("metric_results", {}).items():
        print(f"  {r['name']}: {r['value']}")


def cleanup():
    """Clean up GPU memory between runs."""
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """Run all nudity unlearning technique benchmarks."""
    techniques = [
        # (
        #     "CoGFD — Concept Graph-based high-level Feature Decoupling",
        #     "examples/nudity/cogfd.json",
        # ),
        # (
        #     "SSD — Selective Synaptic Dampening",
        #     "examples/nudity/ssd.json",
        # ),
        # (
        #     "TraSCE — Training-free Stable Concept Editing",
        #     "examples/nudity/trasce.json",
        # ),
        # ("CA — Concept Ablation", "examples/nudity/ca.json"),
        (
            "ConceptSteerers",
            "examples/nudity/concept_steerers.json",
        ),
        (
            "SAeUron — Sparse Autoencoder Unlearning",
            "examples/nudity/saeuron.json",
        ),
        ("ESD — Erased Stable Diffusion", "examples/nudity/esd.json"),
        (
            "MACE — Mass Concept Erasure",
            "examples/nudity/mace.json",
        ),
        (
            "UCE — Unlearning with Concept Erasure",
            "examples/nudity/uce.json",
        ),
        (
            "SLD — Safe Latent Diffusion",
            "examples/nudity/sld.json",
        ),
        (
            "SAFREE — Selective and Attribute Free",
            "examples/nudity/safree.json",
        ),
        (
            "AdvUnlearn — Adversarial Unlearning",
            "examples/nudity/advunlearn.json",
        ),
    ]

    for title, config_path in techniques:
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print("=" * 60)

        runner = None
        config = load_config(config_path)
        runner = build_multi(config)
        report = runner.run()
        print_results(report)

        if runner is not None:
            del runner
        cleanup()

    print(f"\n{'=' * 60}")
    print("All benchmarks completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
