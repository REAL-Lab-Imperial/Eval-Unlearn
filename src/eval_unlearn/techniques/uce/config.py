from dataclasses import dataclass, field
from typing import Optional
from ...configs.base import BaseConfig

# Valid preset names — weight resolution is handled by the UCE package itself
_VALID_PRESETS = {"nudity", "violence", "dog"}


@dataclass(frozen=True)
class UCEConfig(BaseConfig):
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

    # Bundled preset — resolves weights automatically
    preset: Optional[str] = None

    # Custom concept path — load pre-built weights directly (skips creation)
    load_path: Optional[str] = None

    # Custom concept creation — used when neither preset nor load_path is given
    erase_concept: Optional[str] = None
    concept_type: str = "object"  # "object", "style", "attribute"
    save_path: Optional[str] = None  # where to save created weights

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if self.preset is None and self.load_path is None and self.erase_concept is None:
            raise ValueError(
                "UCE requires one of: 'preset' (bundled concept), "
                "'load_path' (pre-built weights), or 'erase_concept' (create weights inline)."
            )
        if self.preset is not None and self.load_path is not None:
            raise ValueError(
                "UCE: 'preset' and 'load_path' are mutually exclusive weight sources. "
                "Provide one or the other, not both."
            )
        if self.preset is not None and self.preset.lower() not in _VALID_PRESETS:
            raise ValueError(
                f"Unknown UCE preset '{self.preset}'. "
                f"Available: {sorted(_VALID_PRESETS)}"
            )
        if self.erase_concept is not None and not self.erase_concept.strip():
            raise ValueError("'erase_concept' must be a non-empty string.")
        if self.preset is None and self.load_path is None and not self.save_path:
            raise ValueError(
                "Creating UCE weights inline requires 'save_path' — "
                "weight creation takes 5–30 minutes and the result must be persisted."
            )
        valid_concept_types = {"object", "style", "attribute"}
        if self.concept_type not in valid_concept_types:
            raise ValueError(
                f"concept_type must be one of {sorted(valid_concept_types)}, "
                f"got '{self.concept_type}'."
            )
