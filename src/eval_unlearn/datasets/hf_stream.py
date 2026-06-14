"""Helpers for loading HuggingFace dataset configs and building streaming DataLoaders."""

from pathlib import Path
from typing import Any, Dict


def load_hf_config(key: str) -> Dict[str, Any]:
    """Load a dataset config entry from config/hf_datasets.yaml."""
    import yaml

    config_path = Path(__file__).parent.parent / "config" / "hf_datasets.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"HF dataset config not found at {config_path}. "
            "Ensure config/hf_datasets.yaml exists inside the eval_unlearn package."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})
    if key not in datasets:
        raise KeyError(
            f"Dataset key '{key}' not found in hf_datasets.yaml. "
            f"Available keys: {list(datasets.keys())}"
        )

    return datasets[key]
