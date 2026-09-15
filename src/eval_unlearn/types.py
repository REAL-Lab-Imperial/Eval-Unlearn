from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Dataset:
    """
    Represents a dataset of prompts for evaluation.

    Attributes:
        prompts: A list of text prompts.
        metadata: Optional metadata about the dataset (e.g., source, version).
    """

    prompts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """
    Represents the result of a metric calculation.

    Attributes:
        name: The name of the metric (e.g., "ASR").
        value: The primary score of the metric.
        details: A dictionary of detailed sub-metrics or extra information.
    """

    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
