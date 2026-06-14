from typing import List, Optional
from PIL import Image

try:
    from mace import MACEPipeline
except ImportError:
    raise ImportError("MACETechnique requires the 'mace' package. Package not installed.")

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import MACEConfig

logger = get_logger(__name__)


@register_technique("mace")
class MACETechnique:
    """
    Thin wrapper around the external MACE package.

    Integrates MACE (Mass Concept Erasure, CVPR 2024) into the Eval-Learn
    benchmarking framework via the standard technique interface.

    MACE erases concepts via Closed-Form Refinement (CFR) of cross-attention
    K/V projection matrices — no training loop required.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper by delegating to MACEPipeline."""
        self.config = MACEConfig.from_dict(kwargs)

        logger.info(
            f"Initializing MACE: model={self.config.model_id}, "
            f"concept='{self.config.erase_concept}', lambda_cfr={self.config.lambda_cfr}"
        )

        self.pipeline = MACEPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            erase_concept=self.config.erase_concept,
            erase_from=self.config.erase_from,
            lambda_cfr=self.config.lambda_cfr,
            save_path=self.config.save_path,
            load_path=self.config.load_path,
            use_fp16=self.config.use_fp16,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using the concept-erased model.

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
            f"Generating {len(prompts)} images ('{self.config.erase_concept}' erased)"
        )

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
