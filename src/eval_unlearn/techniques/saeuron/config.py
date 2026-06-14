from dataclasses import dataclass, field
from typing import Optional, Any, Dict

import torch

from ...configs.base import BaseConfig

_BUNDLED_CONCEPTS = {"nudity"}


@dataclass(frozen=True)
class SAeUronConfig(BaseConfig):
    """Configuration for SAeUron (Sparse Autoencoder Unlearning)."""

    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: Optional[str] = None
    use_fp16: bool = True

    erase_concept: str = "nudity"
    multiplier: float = -20.0

    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if not self.erase_concept or not self.erase_concept.strip():
            raise ValueError("erase_concept must not be empty.")

        if self.erase_concept.lower() not in _BUNDLED_CONCEPTS:
            print(
                f"[SAeUron] '{self.erase_concept}' is not in the bundled activation cache "
                f"(bundled: {sorted(_BUNDLED_CONCEPTS)}). A baseline activation tensor will be "
                f"generated on-the-fly during initialisation — this may take a few minutes.",
                flush=True,
            )

        if self.multiplier == 0:
            raise ValueError("multiplier must not be 0 — use a negative value to ablate.")
        if self.guidance_scale <= 1.0:
            raise ValueError("guidance_scale must be > 1.0 — SAeUron requires CFG to be active.")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SAeUronConfig":
        config_dict = dict(config_dict)

        if not config_dict.get("device"):
            if torch.cuda.is_available():
                config_dict["device"] = "cuda"
            elif torch.backends.mps.is_available():
                config_dict["device"] = "mps"
            else:
                config_dict["device"] = "cpu"

        return super().from_dict(config_dict)
