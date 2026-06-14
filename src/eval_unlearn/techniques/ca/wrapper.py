from typing import List, Optional
from PIL import Image

try:
    from ca import CAPipeline
except ImportError:
    raise ImportError(
        "CATechnique requires the 'ca' package. "
        "Install it with: pip install -e packages/ca"
    )

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import CAConfig

logger = get_logger(__name__)


@register_technique("ca")
class CATechnique:
    """
    Thin wrapper around the external ca package.

    Integrates Concept Ablation (ICCV 2023) into the eval-unlearn benchmarking
    framework via the standard technique interface.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper by delegating to CAPipeline."""
        self.config = CAConfig.from_dict(kwargs)

        logger.info(
            f"Initializing CA: model={self.config.model_id}, "
            f"erase='{self.config.erase_concept}', "
            f"anchor='{self.config.anchor_concept}'"
        )

        self.pipeline = CAPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            target_concept=self.config.erase_concept,
            anchor_concept=self.config.anchor_concept,
            train_steps=self.config.train_steps,
            learning_rate=self.config.learning_rate,
            save_path=self.config.save_path,
            load_path=self.config.save_path,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using the ablated model.

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
            f"Generating {len(prompts)} images ('{self.config.erase_concept}' ablated)"
        )

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
