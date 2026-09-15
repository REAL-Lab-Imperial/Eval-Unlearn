from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

_VALID_ERASE_CONCEPTS = {"nudity", "violence", "hate", "disturbing"}

# Preset values matching diffusers SafetyConfig
_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "sld_guidance_scale": 0,
        "sld_warmup_steps": 0,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    },
    "weak": {
        "sld_guidance_scale": 200,
        "sld_warmup_steps": 15,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    },
    "medium": {
        "sld_guidance_scale": 1000,
        "sld_warmup_steps": 10,
        "sld_threshold": 0.01,
        "sld_momentum_scale": 0.3,
        "sld_mom_beta": 0.4,
    },
    "strong": {
        "sld_guidance_scale": 2000,
        "sld_warmup_steps": 7,
        "sld_threshold": 0.025,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    },
    "max": {
        "sld_guidance_scale": 5000,
        "sld_warmup_steps": 0,
        "sld_threshold": 1.0,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    },
}


@dataclass(frozen=True)
class SLDConfig(BaseConfig):
    """Configuration for Safe Latent Diffusion (SLD)."""

    model_id: str = field(init=False, default="AIML-TUDA/stable-diffusion-safe")
    device: str = "cuda"
    use_fp16: bool = True
    erase_concept: str = "nudity"
    preset: Optional[str] = None
    sld_guidance_scale: float = 5000
    sld_warmup_steps: int = 0
    sld_threshold: float = 1.0
    sld_momentum_scale: float = 0.5
    sld_mom_beta: float = 0.7
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if not self.erase_concept:
            raise ValueError("erase_concept must not be empty.")
        if self.erase_concept.lower() not in _VALID_ERASE_CONCEPTS:
            raise ValueError(
                f"erase_concept must be one of {sorted(_VALID_ERASE_CONCEPTS)}, "
                f"got '{self.erase_concept}'."
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SLDConfig":
        data = dict(data)

        preset = data.get("preset")
        if preset is not None:
            key = preset.lower()
            if key not in _PRESETS:
                raise ValueError(
                    f"Unknown SLD preset '{preset}'. "
                    f"Available: {[p.upper() for p in _PRESETS]}"
                )
            merged = dict(data)
            for k, v in _PRESETS[key].items():
                if k not in data:
                    merged[k] = v
            return super().from_dict(merged)
        return super().from_dict(data)
