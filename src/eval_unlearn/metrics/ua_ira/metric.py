from typing import List, Any, Dict, Optional
from torch.utils.data import DataLoader
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import UAIRAConfig

logger = get_logger(__name__)

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "UA_IRA metric requires 'torch', 'transformers', and 'Pillow'. "
        "Install with: pip install eval-unlearn[ua_ira]"
    ) from e


@register_metric("ua_ira")
class UAIRAMetric:
    """
    Unlearning Accuracy (UA) and In-domain Retain Accuracy (IRA) Metric.

    UA: ratio of images NOT classified as target concept / total target images
    IRA: ratio of images correctly classified as retain concept / total retain images

    The runner loads the dataset via load_dataset(), then calls update() once per batch,
    and finally compute() to get the result.
    """

    def __init__(self, **kwargs):
        self.config = UAIRAConfig.from_dict(kwargs)

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            f"Initializing CLIP model '{self.config.clip_model_name}' on {self.device}..."
        )
        self.model = CLIPModel.from_pretrained(self.config.clip_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self.model.eval()

        # State for accumulation across batches
        self._target_prompt_end_index = 0
        self._target_correct_count = 0
        self._target_total_count = 0
        self._retain_correct_count = 0
        self._retain_total_count = 0

        logger.info("UA_IRA Metric ready.")

    def load_dataset(self) -> DataLoader:
        """
        Load target and retain prompts from CSV files.

        Returns a DataLoader that yields Dataset batches with prompts and metadata.
        """
        from ...datasets.ua_ira_csv import load_ua_ira_csv

        # Reset counters
        self._target_correct_count = 0
        self._target_total_count = 0
        self._retain_correct_count = 0
        self._retain_total_count = 0

        if not self.config.target_prompts_path or not self.config.retain_prompts_path:
            raise ValueError(
                "UA_IRA metric requires 'target_prompts_path' and 'retain_prompts_path' in config. "
                "Provide paths to CSV files with target and retain prompts."
            )

        return load_ua_ira_csv(
            target_prompts_path=self.config.target_prompts_path,
            retain_prompts_path=self.config.retain_prompts_path,
            target_concept_name=self.config.target_concept,
            retain_concept_name=self.config.retain_concept,
            target_limit=self.config.target_prompt_limit,
            retain_limit=self.config.retain_prompt_limit,
            batch_size=self.config.batch_size,
        )

    def update(
        self,
        images: List[Any],
        _prompts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run batched CLIP classification on images and accumulate target/retain correct counts.

        Args:
            images:    Generated PIL Images or file paths.
            _prompts:  Unused.
            metadata:  Must contain ``target_prompt_end_index`` to split target vs retain images.
        """
        metadata = metadata or {}

        # Get the split index on first call
        if self._target_prompt_end_index == 0:
            self._target_prompt_end_index = metadata.get("target_prompt_end_index", 0)

        if not images:
            return

        # Caption prompts for CLIP classification (same for all images in batch)
        caption_prompts = [
            f"Image of {self.config.target_concept}",
            f"Image of {self.config.retain_concept}",
        ]

        # Split images by target/retain based on metadata
        target_prompt_end_index = metadata.get(
            "target_prompt_end_index", self._target_prompt_end_index
        )
        target_images = images[:target_prompt_end_index]
        retain_images = images[target_prompt_end_index:]

        # Evaluate target images (batched CLIP forward pass)
        if target_images:
            self._evaluate_batch(target_images, caption_prompts, is_target=True)

        # Evaluate retain images (batched CLIP forward pass)
        if retain_images:
            self._evaluate_batch(retain_images, caption_prompts, is_target=False)

    def _evaluate_batch(
        self, images: List[Any], caption_prompts: List[str], is_target: bool
    ) -> None:
        """
        Batch evaluate images with CLIP.

        Args:
            images:           List of PIL Images or file paths.
            caption_prompts:  List of text captions (same for all images in batch).
            is_target:        True if evaluating target images, False for retain.
        """
        # Convert all images to PIL
        pil_images = [self._to_pil(img) for img in images]
        # Filter out failed conversions
        pil_images = [img for img in pil_images if img is not None]

        if not pil_images:
            return

        try:
            # Single batched CLIP forward pass for all images
            inputs = self.processor(
                text=caption_prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**inputs)
                # output.logits_per_image shape: (batch_size, 2)
                predicted_indices = (
                    output.logits_per_image.softmax(dim=1).argmax(dim=1)
                )

            # Count successes (index 1 = retain concept)
            num_correct = (predicted_indices == 1).sum().item()
            num_total = len(pil_images)

            if is_target:
                self._target_total_count += num_total
                self._target_correct_count += num_correct
            else:
                self._retain_total_count += num_total
                self._retain_correct_count += num_correct

        except Exception as e:
            logger.warning("Error evaluating batch: %s", e)

    def compute(self) -> MetricResult:
        """
        Compute UA and IRA scores from accumulated counts.

        Returns:
            MetricResult with average score and individual UA/IRA scores in details.
        """
        # Calculate individual scores
        ua_score = (
            self._target_correct_count / self._target_total_count
            if self._target_total_count > 0
            else 0.0
        )
        ira_score = (
            self._retain_correct_count / self._retain_total_count
            if self._retain_total_count > 0
            else 0.0
        )

        avg_score = (ua_score + ira_score) / 2

        logger.info(
            f"UA_IRA Score: {avg_score:.4f}  (UA: {ua_score:.4f}, IRA: {ira_score:.4f})"
        )

        return MetricResult(
            name="UA_IRA",
            value=avg_score,
            details={
                "ua_score": ua_score,
                "ira_score": ira_score,
                "target_correct": self._target_correct_count,
                "target_total": self._target_total_count,
                "retain_correct": self._retain_correct_count,
                "retain_total": self._retain_total_count,
                "config": self.config.to_dict(),
            },
        )

    @staticmethod
    def _to_pil(img) -> Optional["Image.Image"]:
        """Convert image to PIL. Returns None if conversion fails."""
        if Image and isinstance(img, Image.Image):
            return img
        if isinstance(img, str):
            try:
                return Image.open(img).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                logger.warning("Could not load image %s: %s", img, e)
                return None
        return None
