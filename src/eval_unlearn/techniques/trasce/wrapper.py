from typing import List, Optional
from PIL import Image

# Import from external package
try:
    from trasce import TraSCEPipeline
except ImportError:
    raise ImportError(
        "TraSCETechnique requires the 'trasce' package. Package not installed."
    )

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import TraSCEConfig

logger = get_logger(__name__)


@register_technique("trasce")
class TraSCETechnique:
    """
    Thin wrapper around the external TraSCE package.

    This wrapper integrates the TraSCE library into the
    Eval-Learn benchmarking framework via the standard technique interface.
    """

    def __init__(self, **kwargs):
        self.config = TraSCEConfig.from_dict(kwargs)

        logger.info(f"Initialising TraSCE: {self.config.model_id}")

        # Delegate to external package
        self.pipeline = TraSCEPipeline(
            model_id = self.config.model_id,
            device = self.config.device,
            use_fp16 = self.config.use_fp16,
            # guidance loss scale controls how far the latent values are steered from the erase concept
            # higher the value the greater the steering
            guidance_loss_scale = self.config.guidance_loss_scale,
            sigma = self.config.sigma,
            # discriminator guidance scale controls the extent to which latent values are updated per step
            discriminator_guidance_scale = self.config.discriminator_guidance_scale,
            erase_concept = self.config.erase_concept,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using TraSCE.

        Args:
            prompts: List of text prompts.
            seed: Random seed for reproducibility.
            **kwargs: Additional generation parameters.

        Returns:
            List of PIL Images.
        """
        # parameters per generation
        # if provided by user in kwargs, we use it, otherwise use default values from config
        num_inference_steps = kwargs.pop("num_inference_steps", self.config.num_inference_steps)
        # guidance scale controls how close the generated image is to the prompt
        # higher value means closer to prompt
        guidance_scale = kwargs.pop("guidance_scale", self.config.guidance_scale)
        logger.info(
            f"Generating {len(prompts)} images with TraSCE, concept being erased: {self.config.erase_concept}"
        )
        return self.pipeline.generate(prompts, seed = seed, num_inference_steps = num_inference_steps, guidance_scale = guidance_scale, **kwargs)
