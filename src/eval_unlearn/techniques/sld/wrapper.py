import os
from typing import List, Any, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SLDConfig

logger = get_logger(__name__)

try:
    import torch
    from huggingface_hub import login
except ImportError as e:
    raise RuntimeError(
        "SLD technique requires 'torch' and 'huggingface_hub'. "
        "Install with: pip install eval-unlearn[diffusers]"
    ) from e

# Guard allows test patches to survive module reloads
if "StableDiffusionPipelineSafe" not in dir():
    try:
        from diffusers import StableDiffusionPipelineSafe
    except ImportError:
        StableDiffusionPipelineSafe = None


@register_technique("sld")
class SLDTechnique:
    """
    Wrapper for Safe Latent Diffusion (SLD) pipeline.
    """

    def __init__(self, **kwargs):
        # 1. Parse Config
        # We accept kwargs to be flexible with the registry instantiation
        self.config = SLDConfig.from_dict(kwargs)

        # 2. Setup Device
        if self.config.device:
            self.device = self.config.device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        logger.info(
            f"Initializing SLD on {self.device} with model {self.config.model_id}"
        )

        # 3. Auth (Optional)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")

        # 4. Load Pipeline
        # Note: We disable the standard safety_checker because SLD *is* the safety mechanism
        if StableDiffusionPipelineSafe is None:
            raise RuntimeError(
                "SLD technique requires 'diffusers'. "
                "Install with: pip install eval-unlearn[diffusers]"
            )
        torch_dtype = torch.float16 if (self.config.use_fp16 and self.device == "cuda") else torch.float32
        try:
            self.pipe = StableDiffusionPipelineSafe.from_pretrained(
                self.config.model_id, safety_checker=None, requires_safety_checker=False,
                torch_dtype=torch_dtype,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load SLD model: {e}")

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Any]:
        """
        Generate images using the configured SLD parameters.

        SLD parameters (sld_guidance_scale, etc.) can be overridden per-call
        via *kwargs*; otherwise the values from self.config are used.
        All other kwargs (e.g. num_inference_steps) are forwarded to the pipeline.
        """
        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        # Extract SLD-specific params from kwargs so they don't get passed twice
        sld_params = {
            "sld_guidance_scale": kwargs.pop(
                "sld_guidance_scale", self.config.sld_guidance_scale
            ),
            "sld_warmup_steps": kwargs.pop(
                "sld_warmup_steps", self.config.sld_warmup_steps
            ),
            "sld_threshold": kwargs.pop("sld_threshold", self.config.sld_threshold),
            "sld_momentum_scale": kwargs.pop(
                "sld_momentum_scale", self.config.sld_momentum_scale
            ),
            "sld_mom_beta": kwargs.pop("sld_mom_beta", self.config.sld_mom_beta),
        }

        logger.info(f"Generating {len(prompts)} images using SLD...")

        images = []
        for i, prompt in enumerate(prompts):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + i)
            try:
                output = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    **sld_params,
                    generator=generator,
                    **kwargs,
                ).images[0]
                images.append(output)
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images
