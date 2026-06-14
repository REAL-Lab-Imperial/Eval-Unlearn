from dataclasses import dataclass, field
from typing import List, Optional, Union
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class MACEConfig(BaseConfig):
    """Configuration for MACE (Mass Concept Erasure, CVPR 2024)."""

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

    # Concept erasure settings — accepts a single string or a list of strings.
    # List example: ["nudity", "naked", "bare skin"] erases all synonyms at once.
    erase_concept: Union[str, List[str]] = "nudity"
    erase_from: Optional[Union[str, List[str]]] = (
        None  # Defaults to "" (fully erase to neutral)
    )

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    # CFR regularization (core MACE hyperparameter)
    lambda_cfr: float = 0.1

    # Save/load modified UNet weights (optional, to avoid re-running CFR)
    save_path: Optional[str] = None
    load_path: Optional[str] = None

    def __post_init__(self):
        if not self.erase_concept:
            raise ValueError("erase_concept must not be empty.")
        if self.lambda_cfr <= 0:
            raise ValueError(f"lambda_cfr must be > 0, got {self.lambda_cfr}")

