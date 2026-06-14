from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig
from .._clip_constants import validate_clip_model


@dataclass(frozen=True)
class CLIPScoreConfig(BaseConfig):
    """
    Configuration for the CLIP Score metric.

    Attributes:
        clip_model_name: HuggingFace CLIP model identifier used by torchmetrics.
        device: Torch device string (default: None, auto-detect).
        limit: Max number of prompts to stream from HuggingFace.
    """

    clip_model_name: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None
    limit: Optional[int] = 300

    def __post_init__(self) -> None:
        validate_clip_model(self.clip_model_name, "clip_model_name")
