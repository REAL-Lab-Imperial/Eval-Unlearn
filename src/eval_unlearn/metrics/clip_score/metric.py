from typing import List, Any, Dict, Optional
from torch.utils.data import DataLoader
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import CLIPScoreConfig

logger = get_logger(__name__)

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "CLIPScore metric requires 'torch', 'transformers', and 'Pillow'. "
        "Install with: pip install eval-unlearn[clip_score]"
    ) from e


@register_metric("clip_score")
class CLIPScoreMetric:
    """
    CLIP Score Metric.

    Measures text-to-image alignment via cosine similarity between CLIP
    embeddings of each generated image and its prompt.

    update() runs the CLIP forward pass immediately and accumulates a running
    score total + count. compute() returns the average — no images are retained.
    """

    def __init__(self, **kwargs):
        self.config = CLIPScoreConfig.from_dict(kwargs)

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            f"Loading CLIP model '{self.config.clip_model_name}' on {self.device}..."
        )
        self.model = CLIPModel.from_pretrained(self.config.clip_model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
        self.model.eval()

        self._total_score = 0.0
        self._evaluated_count = 0
        self._total_count = 0
        self._per_image_scores: List[Optional[float]] = []
        logger.info("CLIPScoreMetric ready.")

    def load_dataset(self) -> DataLoader:
        """Return a DataLoader over the TIFA dataset."""
        from ...datasets.tifa_csv import load_tifa_csv

        self._total_score = 0.0
        self._evaluated_count = 0
        self._total_count = 0
        self._per_image_scores = []

        return load_tifa_csv(limit=self.config.limit)

    def _load_image_pil(self, img) -> Optional[Image.Image]:
        """Load and convert image to PIL Image."""
        if isinstance(img, Image.Image):
            return img.convert("RGB") if img.mode != "RGB" else img
        elif isinstance(img, str):
            try:
                return Image.open(img).convert("RGB")
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Could not load image {img}: {e}")
                return None
        else:
            logger.warning(f"Unsupported image type: {type(img)}")
            return None

    def update(
        self,
        images: List[Any],
        prompts: List[str],
        _metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Compute CLIP score for each image-prompt pair and accumulate.

        Args:
            images:    Generated PIL Images or file paths.
            prompts:   Text prompts parallel to images.
            _metadata: Unused.
        """
        for img, prompt in zip(images, prompts):
            pil_img = self._load_image_pil(img)
            if pil_img is None:
                logger.warning(f"Skipping image at index {self._total_count}: could not load.")
                self._per_image_scores.append(None)
                self._total_count += 1
                continue

            try:
                # Process image and text
                with torch.no_grad():
                    inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)

                # Compute cosine similarity
                logits_per_image = outputs.logits_per_image
                score_val = logits_per_image.item()

                self._per_image_scores.append(score_val)
                self._total_score += score_val
                self._evaluated_count += 1
            except Exception as e:
                logger.error(f"Error scoring image {self._total_count}: {e}")
                self._per_image_scores.append(None)

            self._total_count += 1

    def compute(self) -> MetricResult:
        """
        Return average CLIP score across all evaluated image-prompt pairs.
        All CLIP inference was done in update() — this is division only.
        """
        if self._total_count == 0:
            return MetricResult(
                name="CLIPScore", value=0.0, details={"error": "No images evaluated"}
            )

        avg_score = self._total_score / self._evaluated_count if self._evaluated_count > 0 else 0.0
        logger.info(
            f"CLIP Score: {avg_score:.4f} (evaluated {self._evaluated_count}/{self._total_count})"
        )

        return MetricResult(
            name="CLIPScore",
            value=avg_score,
            details={
                "per_image_scores": self._per_image_scores,
                "evaluated_count": self._evaluated_count,
                "total_count": self._total_count,
                "config": self.config.to_dict(),
            },
        )
