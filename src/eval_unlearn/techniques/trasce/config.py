from dataclasses import dataclass, field

from ...configs.base import BaseConfig


@dataclass(frozen=True)
class TraSCEConfig(BaseConfig):
    # parameters as given and defined by the original TraSCE paper (https://arxiv.org/abs/2412.07658)
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True
    discriminator_guidance_scale: float = 5.0
    guidance_loss_scale: float = 15.0
    sigma: float = 1.0
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    erase_concept: str = "nudity"

    def __post_init__(self):
        if not self.erase_concept:
            raise ValueError("erase_concept must not be empty.")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}.")
        if self.discriminator_guidance_scale <= 0:
            raise ValueError(f"discriminator_guidance_scale must be > 0, got {self.discriminator_guidance_scale}.")
        if self.guidance_loss_scale <= 0:
            raise ValueError(
                f"guidance_loss_scale must be > 0, got {self.guidance_loss_scale}. "
                "Setting it to 0 disables steering entirely; negative values invert the steering direction."
            )
