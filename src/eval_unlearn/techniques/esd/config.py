from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

# Available training methods for ESD
# - xattn: Fine-tunes cross-attention layers (ESD-x) - best for styles/artists
# - noxattn: Fine-tunes all layers except cross-attention
# - selfattn: Fine-tunes only self-attention layers
# - full: Fine-tunes entire UNet (most aggressive)
TRAIN_METHODS = ["xattn", "noxattn", "selfattn", "full"]


@dataclass(frozen=True)
class ESDConfig(BaseConfig):
    """Configuration for Erased Stable Diffusion (ESD)."""

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"

    # Concept erasure settings
    erase_concept: str = "nudity"
    erase_from: Optional[str] = (
        None  # Target concept to erase from (defaults to erase_concept)
    )
    train_method: str = "noxattn"
    negative_guidance: float = 2.0

    # Training settings
    train_steps: int = 200
    learning_rate: float = 5e-5
    use_fp16: bool = True

    # Load pre-trained weights (skips training entirely)
    load_path: Optional[str] = None
    # Save trained weights (optional)
    save_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if not self.erase_concept or not self.erase_concept.strip():
            raise ValueError("erase_concept must not be empty.")
        if self.train_method not in TRAIN_METHODS:
            raise ValueError(
                f"Unknown train_method '{self.train_method}'. Available: {TRAIN_METHODS}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ESDConfig":
        return super().from_dict(data)
