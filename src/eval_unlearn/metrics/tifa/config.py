from dataclasses import dataclass
from typing import Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class TIFAConfig(BaseConfig):
    """
    Configuration for the TIFA (Text-to-Image Faithfulness) metric.

    Attributes:
        vqa_model_name: ModelScope model identifier for the MPLUG VQA model,
            matching the official tifascore implementation.
        device: Torch device string (default: None, auto-detect).
        limit: Max number of prompts to stream from HuggingFace.
    """

    vqa_model_name: str = "damo/mplug_visual-question-answering_coco_large_en"
    device: Optional[str] = None
    limit: Optional[int] = 200
