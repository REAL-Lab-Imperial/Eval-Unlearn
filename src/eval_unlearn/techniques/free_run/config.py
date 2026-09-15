from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...configs.base import BaseConfig


@dataclass(frozen=True)
class FreeRunConfig(BaseConfig):
    """Configuration for the free run technique — any HF text-to-image model."""

    model_id: str = ""
    device: Optional[str] = None
    use_fp16: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if not self.model_id:
            raise ValueError("FreeRunConfig requires 'model_id' to be specified.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeRunConfig":
        return super().from_dict(data)
