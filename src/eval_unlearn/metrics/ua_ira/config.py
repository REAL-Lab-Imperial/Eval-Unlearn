from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig
from .._clip_constants import validate_clip_model


@dataclass(frozen=True)
class UAIRAConfig(BaseConfig):
    """
    Configuration for Unlearning Accuracy and In-domain Retain Accuracy (IRA) Metric.

    UA is calculated by the ratio of images NOT classified as the target concept
    to the total number of target images.
    IRA is calculated by the ratio of images correctly classified as the retain concept
    to the total number of retain images.

    Loads target and retain prompts from CSV files provided by the user.
    """

    clip_model_name: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None
    target_prompts_path: str = ""  # Path to CSV with target prompts (concept to erase)
    retain_prompts_path: str = ""  # Path to CSV with retain prompts (concept to keep)
    target_concept: str = (
        "target_concept"  # How to refer to target in CLIP ("nudity", "Mickey Mouse", etc.)
    )
    retain_concept: str = (
        "retain_concept"  # How to refer to retain in CLIP ("person", "Minnie Mouse", etc.)
    )
    target_prompt_limit: Optional[int] = None  # Max target prompts to load
    retain_prompt_limit: Optional[int] = None  # Max retain prompts to load
    batch_size: int = 32  # Batch size for DataLoader

    def __post_init__(self) -> None:
        validate_clip_model(self.clip_model_name, "clip_model_name")
        if not self.target_prompts_path:
            raise ValueError("target_prompts_path must be set.")
        if not self.retain_prompts_path:
            raise ValueError("retain_prompts_path must be set.")
