from statistics import mean
from typing import List, Any, Dict, Optional
from torch.utils.data import DataLoader
from ...types import MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import TIFAConfig

logger = get_logger(__name__)

try:
    import torch
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "TIFA metric requires 'torch', 'modelscope', and 'Pillow'. "
        "Install with: pip install eval-unlearn[tifa]"
    ) from e


@register_metric("tifa")
class TIFAMetric:
    """
    TIFA (Text-to-Image Faithfulness) Metric.

    Mirrors the official ``tifascore.tifa_score_benchmark`` implementation:
    an MPLUG VQA model answers each question derived from the prompt, and the
    free-form answer is compared against the expected answer. Note: unlike
    the original TIFA v1.0 benchmark, this dataset does not carry per-question
    multiple-choice ``choices``, so there is no SBERT choice-snapping step —
    scoring is exact string match on the free-form answer.

    update() runs MPLUG immediately on each (image, qa_pairs) and records a
    per-image list of question scores. compute() first averages each image's
    question scores (per-image accuracy), then averages those per-image
    scores across all images — the same two-stage "macro-average" used by
    tifa_score_benchmark, rather than pooling all questions together.

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
            f"Loading MPLUG VQA model '{self.config.vqa_model_name}'..."
        )
        self._vqa_pipeline = pipeline(
            Tasks.visual_question_answering, model=self.config.vqa_model_name
        )
        logger.info("MPLUG VQA model ready.")

        self._total_questions_count = 0
        self._total_images_count = 0
        self._per_image_scores: List[Optional[float]] = []

    def load_dataset(self) -> DataLoader:
        """Return a DataLoader over the TIFA dataset."""
        from ...datasets.tifa_csv import load_tifa_csv

        self._total_questions_count = 0
        self._total_images_count = 0
        self._per_image_scores = []

        return load_tifa_csv(limit=self.config.limit)

    # ------------------------------------------------------------------
    # VQA engine
    # ------------------------------------------------------------------

    def _answer(self, pil_image, question: str) -> str:
        """Run VQA on a single PIL image and question via MPLUG."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        result = self._vqa_pipeline({"image": pil_image, "question": question})
        answer = result["text"]
        return answer[0] if isinstance(answer, list) else answer

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
        Run MPLUG VQA on each image against its QA pairs and record a
        per-image list of question scores (1 = correct, 0 = incorrect).

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

            question_scores: List[int] = []
            for qa in questions:
                question = qa.get("question", "")
                expected = qa.get("answer", "")
                if not question or not expected:
                    continue
                prediction = self._answer(pil_img, question)
                self._total_questions_count += 1
                question_scores.append(
                    int(prediction.lower().strip() == expected.lower().strip())
                )

            self._per_image_scores.append(
                mean(question_scores) if question_scores else None
            )
            self._total_images_count += 1

    def compute(self) -> MetricResult:
        """
        Return the TIFA score as the mean of per-image question-accuracy
        scores (macro-average across images), matching
        ``tifascore.tifa_score_benchmark``'s ``tifa_average``.
        """
        valid_scores = [s for s in self._per_image_scores if s is not None]

        if not valid_scores:
            return MetricResult(
                name="TIFA", value=0.0, details={"error": "No images evaluated"}
            )

        tifa_score = mean(valid_scores)
        logger.info(
            f"TIFA Score: {tifa_score:.4f} (macro-average over {len(valid_scores)} images)"
        )

        return MetricResult(
            name="TIFA",
            value=tifa_score,
            details={
                "total_questions_count": self._total_questions_count,
                "total_images_count": self._total_images_count,
                "per_image_scores": self._per_image_scores,
                "config": self.config.to_dict(),
            },
        )
