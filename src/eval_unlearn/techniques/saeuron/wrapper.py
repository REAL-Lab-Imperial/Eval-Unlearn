from typing import List, Optional
from PIL import Image

try:
    from saeuron import SAeUronPipeline
except ImportError as e:
    raise ImportError(
        f"SAeUronTechnique requires the 'saeuron' package. Package not installed or failed to import: {e}"
    ) from e

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SAeUronConfig

logger = get_logger(__name__)


@register_technique("saeuron")
class SAeUronTechnique:
    """Thin wrapper around the saeuron package's SAeUronPipeline."""

    def __init__(self, **kwargs):
        self.config = SAeUronConfig.from_dict(kwargs)

        logger.info(f"Initializing SAeUron: concept='{self.config.erase_concept}', multiplier={self.config.multiplier}")

        self.pipeline = SAeUronPipeline(
            concept=self.config.erase_concept,
            multiplier=self.config.multiplier,
            model_id=self.config.model_id,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
