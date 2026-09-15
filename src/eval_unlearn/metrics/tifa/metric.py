from typing import List, Any, Dict, Optional
from torch.utils.data import DataLoader
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import TIFAConfig

logger = get_logger(__name__)

try:
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "TIFA metric requires 'torch', 'transformers', and 'Pillow'. "
        "Install with: pip install eval-unlearn[tifa]"
    ) from e


@register_metric("tifa")
class TIFAMetric:
    """
    TIFA (Text-to-Image Faithfulness) Metric.

    Uses BLIP-2 VQA to verify whether generated images answer a set of
    question-answer pairs derived from the prompt. Score is the fraction of
    correctly answered questions across all images.

    update() runs BLIP-2 immediately on each (image, qa_pairs) and accumulates
    correct/total question counts. compute() returns the ratio — no images retained.

    batch.metadata must contain:
    - ``qa_pairs``: list parallel to images, each element a list of
      ``{"question": str, "answer": str}`` dicts.
    """

    def __init__(self, **kwargs):
        self.config = TIFAConfig.from_dict(kwargs)

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            f"Loading BLIP-2 VQA model '{self.config.vqa_model_name}' on {self.device}..."
        )
        self._processor = Blip2Processor.from_pretrained(self.config.vqa_model_name)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.config.vqa_model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self._model.eval()
        logger.info("BLIP-2 VQA model ready.")

        self._correct_count = 0
        self._total_questions_count = 0
        self._total_images_count = 0
        self._per_image_scores: List[Optional[float]] = []

    def load_dataset(self) -> DataLoader:
        """Return a DataLoader over the TIFA dataset."""
        from ...datasets.tifa_csv import load_tifa_csv

        self._correct_count = 0
        self._total_questions_count = 0
        self._total_images_count = 0
        self._per_image_scores = []

        return load_tifa_csv(limit=self.config.limit)

    # ------------------------------------------------------------------
    # VQA engine
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _answer(self, pil_image, question: str, max_new_tokens: int = 10) -> str:
        """Run VQA on a single PIL image and question."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        inputs = self._processor(
            images=pil_image,
            text=question,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._processor.decode(
            generated_ids[0], skip_special_tokens=True
        ).strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        images: List[Any],
        _prompts: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Run BLIP-2 VQA on each image against its QA pairs and accumulate
        correct/total question counts.

        Args:
            images:    Generated PIL Images or file paths.
            _prompts:  Unused — faithfulness is measured via QA pairs.
            metadata:  Must contain ``qa_pairs`` parallel to images.
        """
        metadata = metadata or {}
        qa_pairs_batch = metadata.get("qa_pairs", [None] * len(images))

        for img, questions in zip(images, qa_pairs_batch):
            pil_img = img if (Image and isinstance(img, Image.Image)) else None
            if pil_img is None and isinstance(img, str):
                try:
                    pil_img = Image.open(img).convert("RGB")
                except (FileNotFoundError, OSError) as e:
                    logger.warning("Could not load image: %s", e)

            if pil_img is None or not questions:
                self._per_image_scores.append(None)
                self._total_images_count += 1
                continue

            img_correct = 0
            img_total = 0
            for qa in questions:
                question = qa.get("question", "")
                expected = qa.get("answer", "")
                if not question or not expected:
                    continue
                prediction = self._answer(pil_img, question)
                self._total_questions_count += 1
                img_total += 1
                if prediction.lower().strip() == expected.lower().strip():
                    self._correct_count += 1
                    img_correct += 1

            self._per_image_scores.append(
                img_correct / img_total if img_total > 0 else None
            )
            self._total_images_count += 1

    def compute(self) -> MetricResult:
        """
        Return TIFA score as correct / total questions.
        All VQA inference was done in update() — this is division only.
        """
        if self._total_images_count == 0:
            return MetricResult(
                name="TIFA", value=0.0, details={"error": "No images evaluated"}
            )

        tifa_score = (
            self._correct_count / self._total_questions_count if self._total_questions_count > 0 else 0.0
        )
        logger.info(
            f"TIFA Score: {tifa_score:.4f} ({self._correct_count}/{self._total_questions_count} correct)"
        )

        return MetricResult(
            name="TIFA",
            value=tifa_score,
            details={
                "correct_count": self._correct_count,
                "total_questions_count": self._total_questions_count,
                "total_images_count": self._total_images_count,
                "per_image_scores": self._per_image_scores,
                "config": self.config.to_dict(),
            },
        )
