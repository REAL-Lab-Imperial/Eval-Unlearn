"""Registry of models used by each metric.

Used by the CLI `models` command and for documentation purposes.
"""
from typing import FrozenSet, NamedTuple, Optional

from eval_unlearn.metrics._clip_constants import SUPPORTED_CLIP_MODELS


class MetricModelInfo(NamedTuple):
    model: str               # Model ID or descriptive name
    configurable: bool       # Whether the user can swap it via config
    config_field: Optional[str] = None              # Config field name if configurable
    note: Optional[str] = None                      # Extra context (non-configurable)
    choices: Optional[FrozenSet[str]] = None        # Valid values if constrained


METRIC_MODELS: dict = {
    "asr_i2p":      MetricModelInfo("NudeNet / openai/clip-vit-large-patch14", configurable=True, config_field="clip_model_id", note="NudeNet for nudity; VLM (MPLUG) for other concepts by default", choices=SUPPORTED_CLIP_MODELS),
    "asr_ring_a_bell": MetricModelInfo("openai/clip-vit-large-patch14", configurable=True,  config_field="clip_model_id",   choices=SUPPORTED_CLIP_MODELS),
    "clip_score":   MetricModelInfo( "openai/clip-vit-base-patch32",     configurable=True,  config_field="clip_model_name", choices=SUPPORTED_CLIP_MODELS),
    "err":          MetricModelInfo("openai/clip-vit-large-patch14",    configurable=True,  config_field="clip_model_name", choices=SUPPORTED_CLIP_MODELS),
    "fid":          MetricModelInfo("Inception V3 (IMAGENET1K_V1)",     configurable=False, note="torchvision"),
    "asr_mma_diffusion": MetricModelInfo("openai/clip-vit-large-patch14", configurable=False, note="injected from technique base model"),
    "tifa":         MetricModelInfo("damo/mplug_visual-question-answering_coco_large_en", configurable=True,  config_field="vqa_model_name"),
    "ua_ira":       MetricModelInfo("openai/clip-vit-large-patch14",    configurable=True,  config_field="clip_model_name", choices=SUPPORTED_CLIP_MODELS),
}
