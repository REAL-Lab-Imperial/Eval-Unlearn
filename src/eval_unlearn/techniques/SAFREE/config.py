from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from ...configs.base import BaseConfig

# Concepts for which SVF has been calibrated in f_beta.
# For anything else, SVF is disabled automatically and re_attn_timestep_range is used instead.
_SVF_CALIBRATED_CONCEPTS = {"nudity", "artists-VanGogh", "artists-KellyMcKernan"}


@dataclass(frozen=True)
class SAFREEConfig(BaseConfig):
    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: Optional[str] = None
    use_fp16: bool = True

    # Stage 1: Text Projection
    alpha: float = 0.01

    # Stage 2: SVF (Self-Validation Filter)
    # Auto-disabled for custom_unsafe_concepts since SVF is only calibrated for
    # nudity and artist concepts. Can be explicitly overridden.
    enable_svf: bool = True
    upperbound_timestep: int = 10

    # Stage 3: LRA (Latent Re-Attention)
    enable_lra: bool = True
    lra_filter_type: str = "high"  # "high", "low", or "all"
    freeu_b1: float = 1.0
    freeu_b2: float = 1.0
    freeu_s1: float = 0.9
    freeu_s2: float = 0.2

    # Named concept category — must be in _SVF_CALIBRATED_CONCEPTS when
    # custom_unsafe_concepts is not set. Also used for metric routing.
    erase_concept: str = "nudity"

    # Custom concept phrases. When set, bypasses the named category lookup and
    # passes this list directly to the pipeline. SVF is auto-disabled.
    custom_unsafe_concepts: Optional[List[str]] = None

    # Fallback timestep window used when SVF is disabled.
    # Default (-1, 1001) covers all steps; narrow (e.g. (0, 10)) to match
    # typical SVF output for predictable behaviour.
    re_attn_timestep_range: Tuple[int, int] = (-1, 1001)

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SAFREEConfig":
        config_data = data.copy()
        custom = config_data.get("custom_unsafe_concepts")
        erase_concept = config_data.get("erase_concept", "nudity")

        if custom is None:
            # Named category path: must be SVF-calibrated
            if erase_concept not in _SVF_CALIBRATED_CONCEPTS:
                raise ValueError(
                    f"SAFREE erase_concept='{erase_concept}' is not a calibrated category. "
                    f"Calibrated categories: {sorted(_SVF_CALIBRATED_CONCEPTS)}. "
                    f"To use a custom concept, pass custom_unsafe_concepts=['phrase1', ...] "
                    f"(SVF will be disabled automatically)."
                )
        else:
            # Custom path: SVF is uncalibrated — disable unless explicitly requested
            if "enable_svf" not in config_data:
                config_data["enable_svf"] = False

        return super().from_dict(config_data)
