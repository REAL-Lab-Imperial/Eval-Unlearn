import os
from typing import Any, List, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import FreeRunConfig

logger = get_logger(__name__)

try:
    import torch
    from huggingface_hub import login
except ImportError as e:
    raise RuntimeError(
        "FreeRunTechnique requires 'torch' and 'huggingface_hub'. "
        "Install with: pip install eval-learn[diffusers]"
    ) from e

# Guard allows test patches to survive module reloads
if "DiffusionPipeline" not in dir():
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        DiffusionPipeline = None


@register_technique("free_run")
class FreeRunTechnique:
    """
    Runs any HF text-to-image model without any safety filtering.
    Specify the model via model_id in the config.
    """

    def __init__(self, **kwargs):
        self.config = FreeRunConfig.from_dict(kwargs)

        if self.config.device:
            self.device = self.config.device
        else:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        logger.info(
            f"Initializing FreeRun on {self.device} with model {self.config.model_id}"
        )

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")

        if DiffusionPipeline is None:
            raise RuntimeError(
                "FreeRunTechnique requires 'diffusers'. "
                "Install with: pip install eval-learn[diffusers]"
            )
        torch_dtype = torch.float16 if (self.config.use_fp16 and self.device == "cuda") else torch.float32
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=torch_dtype,
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.config.model_id}': {e}"
            ) from e

        # Disable safety checker regardless of model family.
        # safety_checker=None cannot be passed at load time for non-SD models.
        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None
            self.pipe.requires_safety_checker = False

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Any]:
        logger.info(f"Generating {len(prompts)} images with {self.config.model_id}...")

        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        images = []
        for i, prompt in enumerate(prompts):
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + i)
            try:
                output = self.pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    **kwargs,
                )
                if not hasattr(output, "images") or not output.images:
                    raise RuntimeError(
                        f"Pipeline output has no images — is '{self.config.model_id}' a text-to-image model?"
                    )
                images.append(output.images[0])
            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images
