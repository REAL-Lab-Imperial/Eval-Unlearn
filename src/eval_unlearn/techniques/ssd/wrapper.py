from typing import List, Optional
from PIL import Image

try:
    from ssd import SSDPipeline
except ImportError:
    raise ImportError("SSDTechnique requires the 'ssd' package. Package not installed.")

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import SSDConfig

logger = get_logger(__name__)


@register_technique("ssd")
class SSDTechnique:
    """
    Thin wrapper around the external SSD package.

    Integrates Selective Synaptic Dampening into the Eval-Learn benchmarking
    framework via the standard technique interface.

    SSD identifies UNet parameters responsible for a forget concept via
    diagonal Fisher Information and dampens them selectively, without any
    training loop.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper by delegating to SSDPipeline."""
        self.config = SSDConfig.from_dict(kwargs)

        logger.info(
            f"Initializing SSD: model={self.config.model_id}, "
            f"concept='{self.config.erase_concept}', "
            f"alpha={self.config.alpha}, dampening_coeff={self.config.dampening_coeff}"
        )

        self.pipeline = SSDPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            erase_concept=self.config.erase_concept,
            forget_prompts=self.config.resolved_forget_prompts(),
            retain_prompts=self.config.resolved_retain_prompts(),
            alpha=self.config.alpha,
            num_fisher_samples=self.config.num_fisher_samples,
            dampening_coeff=self.config.dampening_coeff,
            save_path=self.config.save_path,
            load_path=self.config.load_path,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using the concept-dampened model.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters.

        Returns:
            List of PIL Images.
        """
        num_inference_steps = kwargs.pop(
            "num_inference_steps", self.config.num_inference_steps
        )
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        logger.info(
            f"Generating {len(prompts)} images ('{self.config.erase_concept}' dampened)"
        )

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
