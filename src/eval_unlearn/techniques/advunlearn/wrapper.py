from typing import List, Optional
from PIL import Image

try:
    from advunlearn import AdvUnlearnPipeline
except ImportError:
    raise ImportError(
        "AdvUnlearnTechnique requires the 'advunlearn' package. "
        "Install it with: pip install -e packages/advunlearn"
    )

from ...registry import register_technique
from ...logging_utils import get_logger
from .config import AdvUnlearnConfig

logger = get_logger(__name__)


@register_technique("advunlearn")
class AdvUnlearnTechnique:
    """
    Thin wrapper around the external advunlearn package.

    Integrates AdvUnlearn (NeurIPS 2024) into the eval-unlearn benchmarking
    framework via the standard technique interface.
    """

    def __init__(self, **kwargs):
        """Initialize wrapper by delegating to AdvUnlearnPipeline."""
        self.config = AdvUnlearnConfig.from_dict(kwargs)

        logger.info(f"Initializing AdvUnlearn: {self.config.model_id}")

        self.pipeline = AdvUnlearnPipeline(
            model_id=self.config.model_id,
            device=self.config.device,
            erase_concept=self.config.erase_concept,
            train_method=self.config.train_method,
            dataset_retain=self.config.dataset_retain,
            retain_train=self.config.retain_train,
            retain_batch=self.config.retain_batch,
            retain_step=self.config.retain_step,
            retain_loss_w=self.config.retain_loss_w,
            start_guidance=self.config.start_guidance,
            negative_guidance=self.config.negative_guidance,
            iterations=self.config.train_steps,
            lr=self.config.learning_rate,
            attack_method=self.config.attack_method,
            attack_step=self.config.attack_step,
            attack_lr=self.config.attack_lr,
            attack_type=self.config.attack_type,
            attack_init=self.config.attack_init,
            attack_embd_type=self.config.attack_embd_type,
            adv_prompt_num=self.config.adv_prompt_num,
            adv_prompt_update_step=self.config.adv_prompt_update_step,
            warmup_iter=self.config.warmup_iter,
            component=self.config.component,
            norm_layer=self.config.norm_layer,
            ddim_steps=self.config.ddim_steps,
            save_interval=self.config.save_interval,
            save_dir=self.config.save_dir,
            checkpoint_path=self.config.load_path,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            use_fp16=self.config.use_fp16,
        )

    def generate(
        self, prompts: List[str], seed: Optional[int] = None, **kwargs
    ) -> List[Image.Image]:
        """
        Generate images using the adversarially robust concept-erased model.

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