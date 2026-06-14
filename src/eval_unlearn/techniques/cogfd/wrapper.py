from typing import List, Optional
from PIL import Image

try:
    from cogfd import CoGFDPipeline
except ImportError:
    raise ImportError(
        "CoGFDTechnique requires the 'cogfd' package. "
        "Install with: pip install -e packages/cogfd/"
    )

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import CoGFDConfig

logger = get_logger(__name__)


@register_technique("cogfd")
class CoGFDTechnique:
    """
    Eval-Learn wrapper for CoGFD (ICLR 2025).

    CoGFD erases *concept combinations* (e.g. "nudity" = person + naked)
    while preserving individual component concepts intact.
    The concept logic graph is either user-supplied via ``combination_prompts``
    or auto-populated from built-in defaults for common erase concepts.
    """

    def __init__(self, **kwargs) -> None:
        self.config = CoGFDConfig.from_dict(kwargs)

        logger.info(
            f"Initializing CoGFD: model={self.config.model_id}, "
            f"concept='{self.config.erase_concept}', "
            f"steps={self.config.train_steps}, lr={self.config.learning_rate}"
        )

        self.pipeline = CoGFDPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            erase_concept=self.config.erase_concept,
            combination_prompts=self.config.combination_prompts,
            preserve_concepts=self.config.preserve_concepts,
            lambda_erase=self.config.lambda_erase,
            lambda_preserve=self.config.lambda_preserve,
            lambda_decouple=self.config.lambda_decouple,
            train_steps=self.config.train_steps,
            learning_rate=self.config.learning_rate,
            save_path=self.config.save_path,
            load_path=self.config.load_path,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using the concept-combination-erased model.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            **kwargs: Optional num_inference_steps / guidance_scale overrides.

        Returns:
            List of PIL Images.
        """
        num_inference_steps = kwargs.pop(
            "num_inference_steps", self.config.num_inference_steps
        )
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)

        logger.info(
            f"Generating {len(prompts)} images "
            f"('{self.config.erase_concept}' combination erased via CoGFD)"
        )

        return self.pipeline.generate(
            prompts,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
