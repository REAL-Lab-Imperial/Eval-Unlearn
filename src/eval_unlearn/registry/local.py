from typing import Any, Dict
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Internal registries
_TECHNIQUES: Dict[str, Any] = {}
_METRICS: Dict[str, Any] = {}
_DATASETS: Dict[str, Any] = {}
_BENCHMARKS: Dict[str, Any] = {}


def register_technique(name: str):
    """Decorator to register a technique class or factory."""

    def decorator(cls_or_func):
        _TECHNIQUES[name.lower()] = cls_or_func
        return cls_or_func

    return decorator


def register_metric(name: str):
    """Decorator to register a metric class or factory."""

    def decorator(cls_or_func):
        _METRICS[name.lower()] = cls_or_func
        return cls_or_func

    return decorator


def register_dataset(name: str):
    """Decorator to register a dataset loader."""

    def decorator(cls_or_func):
        _DATASETS[name.lower()] = cls_or_func
        return cls_or_func

    return decorator


def register_benchmark(name: str):
    """Decorator to register a benchmark definition."""

    def decorator(cls_or_func):
        _BENCHMARKS[name.lower()] = cls_or_func
        return cls_or_func

    return decorator


def get_technique(name: str) -> Any:
    """Retrieve a registered technique by name."""
    if name.lower() not in _TECHNIQUES:
        raise ValueError(
            f"Technique '{name}' not found. Available: {list(_TECHNIQUES.keys())}"
        )
    return _TECHNIQUES[name.lower()]


def get_metric(name: str) -> Any:
    """Retrieve a registered metric by name."""
    if name.lower() not in _METRICS:
        raise ValueError(
            f"Metric '{name}' not found. Available: {list(_METRICS.keys())}"
        )
    return _METRICS[name.lower()]


def get_dataset(name: str) -> Any:
    """Retrieve a registered dataset loader by name."""
    if name.lower() not in _DATASETS:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list(_DATASETS.keys())}"
        )
    return _DATASETS[name.lower()]


def get_benchmark(name: str) -> Any:
    """Retrieve a registered benchmark by name."""
    if name.lower() not in _BENCHMARKS:
        raise ValueError(
            f"Benchmark '{name}' not found. Available: {list(_BENCHMARKS.keys())}"
        )
    return _BENCHMARKS[name.lower()]
