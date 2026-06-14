import os
from typing import List, Any, Optional

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SAFREEConfig

try:
    from safree.pipeline import SAFREEPipeline
except ImportError:
    raise ImportError(
        "SAFREETechnique requires the 'safree' package. Package not installed."
    )

try:
    import torch
    from huggingface_hub import login
except ImportError as e:
    raise RuntimeError(
        "SAFREE technique requires 'torch' and 'huggingface_hub'. "
        "Install with: pip install eval-unlearn[safree]"
    ) from e

logger = get_logger(__name__)


@register_technique("safree")
class SAFREETechnique:
    def __init__(self, **kwargs):
        self.config = SAFREEConfig.from_dict(kwargs)

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                logger.debug("Logged in to Hugging Face Hub.")
            except Exception as e:
                logger.warning(f"Could not log in to Hugging Face Hub: {e}")

        device = self.config.device or (
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self._device = device

        torch_dtype = torch.float16 if (self.config.use_fp16 and device == "cuda") else torch.float32
        try:
            self.pipe = SAFREEPipeline.from_pretrained(
                self.config.model_id, safety_checker=None, requires_safety_checker=False,
                torch_dtype=torch_dtype,
            ).to(device)

            if self.config.enable_lra:
                self.pipe.enable_lra(
                    filter_type=self.config.lra_filter_type,
                    b1=self.config.freeu_b1,
                    b2=self.config.freeu_b2,
                    s1=self.config.freeu_s1,
                    s2=self.config.freeu_s2,
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load SAFREE model: {e}")

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Any]:
        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        generator = None
        if seed is not None:
            generator = torch.Generator(self._device).manual_seed(seed)

        # Resolve concept path: named category or custom phrase list
        if self.config.custom_unsafe_concepts is not None:
            concept_kwargs = {"unsafe_concepts": self.config.custom_unsafe_concepts}
        else:
            concept_kwargs = {"unsafe_category": self.config.erase_concept}

        images = []
        for prompt in prompts:
            try:
                output = self.pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    **concept_kwargs,
                    enable_safree=True,
                    enable_svf=self.config.enable_svf,
                    enable_lra=self.config.enable_lra,
                    alpha=self.config.alpha,
                    upperbound_timestep=self.config.upperbound_timestep,
                    re_attn_timestep_range=self.config.re_attn_timestep_range,
                )
                images.append(output.images[0])

            except Exception as e:
                logger.error(f"Generation failed for prompt '{prompt}': {e}")
                raise

        return images
