from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class TIFAConfig(BaseConfig):
    """
    Configuration for the TIFA (Text-to-Image Faithfulness) metric.

    Attributes:
        vqa_model_name: HuggingFace model identifier for the BLIP-2 VQA model.
        device: Torch device string (default: None, auto-detect).
        limit: Max number of prompts to stream from HuggingFace.
    """

    vqa_model_name: str = "Salesforce/blip2-flan-t5-xl"
    device: Optional[str] = None
    limit: Optional[int] = 200
