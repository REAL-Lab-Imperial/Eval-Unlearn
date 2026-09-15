import torch
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from ...configs.base import BaseConfig


@dataclass(frozen=True)
class ConceptSteerersConfig(BaseConfig):
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: Optional[str] = None
    use_fp16: bool = True
    erase_concept: str = "nudity"
    multiplier: float = 1.0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConceptSteerersConfig":
        config_dict = dict(config_dict)

        if not config_dict.get("device"):
            if torch.cuda.is_available():
                config_dict["device"] = "cuda"
            elif torch.backends.mps.is_available():
                config_dict["device"] = "mps"
            else:
                config_dict["device"] = "cpu"

        return super().from_dict(config_dict)

    def __post_init__(self):
        if not self.erase_concept or not self.erase_concept.strip():
            raise ValueError("erase_concept must not be empty.")
        if self.guidance_scale <= 1.0:
            raise ValueError(
                f"guidance_scale must be > 1.0 — Concept Steerers requires CFG to be active, "
                f"got {self.guidance_scale}."
            )
