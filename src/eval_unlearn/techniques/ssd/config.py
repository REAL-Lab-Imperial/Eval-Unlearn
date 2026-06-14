from dataclasses import dataclass, field
from typing import List, Optional
from ...configs.base import BaseConfig


@dataclass(frozen=True)
class SSDConfig(BaseConfig):
    """
    Configuration for Selective Synaptic Dampening (SSD).

    Fisher estimation quality depends heavily on prompt diversity:

    forget_prompts: Varied phrasings of the concept to erase. If not provided,
        defaults to [erase_concept] with a warning. Using only one prompt gives
        a narrow Fisher estimate and risks over-dampening. Recommended: 5–10
        prompts covering synonyms, descriptions, and contextual phrasings
        (e.g. ["nudity", "naked person", "nude figure", "explicit nudity"]).

    retain_prompts: Diverse benign prompts representing general generation
        capacity. If not provided, defaults to a small generic set with a
        warning. Must cover a broad range of subjects and scenes to prevent
        F_retain from being underestimated, which causes model collapse.
        Recommended: 10–20 prompts across objects, scenes, people, animals.

    alpha: Selectivity coefficient. Unlike classification networks where
        features are isolated, diffusion UNet features are highly entangled —
        most parameters contribute to every concept. High alpha drives nearly
        all params toward zero, causing model collapse (coloured static output).
        Recommended range: 1–20. Default: 1.
    """

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

    # Concept erasure settings
    erase_concept: str = "nudity"

    # Fisher prompt sets — see class docstring for guidance on sizing
    forget_prompts: Optional[List[str]] = None
    retain_prompts: Optional[List[str]] = None

    # SSD hyperparameters
    alpha: float = 0.1
    dampening_coeff: float = 1.0
    num_fisher_samples: int = 50

    # Save/load modified UNet weights (optional, to avoid re-running SSD)
    save_path: Optional[str] = None
    load_path: Optional[str] = None

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        if not self.erase_concept:
            raise ValueError("erase_concept must not be empty.")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.dampening_coeff <= 0:
            raise ValueError(f"dampening_coeff must be > 0, got {self.dampening_coeff}")
        if self.num_fisher_samples <= 0:
            raise ValueError(f"num_fisher_samples must be > 0, got {self.num_fisher_samples}")
        if not self.forget_prompts:
            print(
                f"[SSD] forget_prompts not set — defaulting to ['{self.erase_concept}']. "
                "Provide 5–10 varied phrasings for a reliable Fisher estimate."
            )
        if not self.retain_prompts:
            print(
                "[SSD] retain_prompts not set — defaulting to a small generic set. "
                "Provide 10–20 diverse benign prompts to prevent model collapse."
            )

    def resolved_forget_prompts(self) -> List[str]:
        """Return forget_prompts, falling back to [erase_concept] if unset."""
        return list(self.forget_prompts) if self.forget_prompts else [self.erase_concept]

    def resolved_retain_prompts(self) -> List[str]:
        """Return retain_prompts, falling back to a small generic default if unset."""
        return (
            list(self.retain_prompts)
            if self.retain_prompts
            else ["", "a photo", "an image"]
        )
