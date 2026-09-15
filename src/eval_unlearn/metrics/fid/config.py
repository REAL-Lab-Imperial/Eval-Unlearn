from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class FIDConfig(BaseConfig):
    """
    Configuration for Frechet Inception Distance (FID) metric.

    Attributes:
        batch_size: Batch size for InceptionV3 feature extraction.
        device: Device string (e.g. "cuda", "cpu"). Auto-detected if None.
        limit: Max number of rows to stream from HuggingFace.
    """

    batch_size: int = 32
    device: Optional[str] = None
    limit: Optional[int] = 1000
