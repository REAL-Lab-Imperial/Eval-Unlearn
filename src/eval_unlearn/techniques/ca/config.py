from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class CAConfig(BaseConfig):
    """
    Configuration for Concept Ablation (CA).

    Concept Ablation fine-tunes the cross-attention layers to force the model's
    distribution for a 'erase_concept' to match an 'anchor_concept'.
    """

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: Optional[str] = None
    use_fp16: bool = True

    # Concept ablation settings (defaulted to the nudity task)
    erase_concept: str = "nudity"
    anchor_concept: str = "a person wearing clothes"

    # Training settings
    train_steps: int = 400
    learning_rate: float = 1e-5

    # Weight caching: save_path is also used as load_path if the file already exists
    save_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if self.train_steps <= 0:
            raise ValueError(f"train_steps must be > 0, got {self.train_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CAConfig":
        config_data = data.copy()

        if "erase_concept" in config_data and "anchor_concept" not in config_data:
            raise ValueError(
                "Concept Ablation requires both 'erase_concept' and 'anchor_concept'. "
                f"Received only erase_concept: {config_data['erase_concept']}"
            )

        return super().from_dict(config_data)
