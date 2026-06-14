from typing import List, Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from torch.utils.data import DataLoader

from ...types import Dataset, MetricResult
from ...registry import register_metric
from ...logging_utils import get_logger
from .config import FIDConfig

logger = get_logger(__name__)

_inception_transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def _load_inception(device: str) -> nn.Module:
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def _calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.isclose(np.diagonal(covmean).imag, 0, atol=1e-3).all():
            return float("inf")
        covmean = covmean.real

    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


@register_metric("fid")
class FIDMetric:
    """
    Frechet Inception Distance (FID) Metric.

    load_dataset() streams real COCO images, extracts InceptionV3 features,
    and returns a DataLoader of the corresponding caption batches for generation.

    update() extracts InceptionV3 features from each batch of generated images
    and accumulates them. compute() computes the Fréchet distance between the
    real and generated feature distributions — no images are retained after update().
    """

    def __init__(self, **kwargs):
        self.config = FIDConfig.from_dict(kwargs)

        device = self.config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading InceptionV3 on {self.device}...")
        self._inception_model = _load_inception(self.device)
        self._real_activations: Optional[np.ndarray] = None
        self._real_count = 0
        self._gen_activations: List[np.ndarray] = []

        logger.info("FIDMetric initialized.")

    def _get_model(self) -> nn.Module:
        return self._inception_model

    @torch.no_grad()
    def _extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract InceptionV3 pool features from a list of PIL images."""
        activations = []
        model = self._get_model()
        for i in range(0, len(images), self.config.batch_size):
            batch_pils = images[i : i + self.config.batch_size]
            batch = torch.stack(
                [_inception_transform(img.convert("RGB")) for img in batch_pils]
            ).to(self.device)
            activations.append(model(batch).cpu().numpy())
        return np.concatenate(activations, axis=0)

    def load_dataset(self) -> DataLoader:
        """
        Stream COCO images from HuggingFace, extract InceptionV3 features for
        the real distribution, then return a DataLoader of caption batches for
        the generation phase.

        The COCO images themselves are discarded after feature extraction;
        only the 2048-dim feature vectors are retained.
        """
        from ...datasets.coco_parquet import load_coco_parquet

        self._gen_activations = []

        coco_loader = load_coco_parquet(
            limit=self.config.limit,
            batch_size=self.config.batch_size,
        )

        model = self._get_model()
        logger.info("Extracting InceptionV3 features from real COCO images...")

        all_activations = []
        all_captions = []

        for batch in coco_loader:
            image_batch = batch.metadata["images"].to(self.device)
            with torch.no_grad():
                features = model(image_batch)
            all_activations.append(features.cpu().numpy())
            all_captions.extend(batch.prompts)

        self._real_activations = np.concatenate(all_activations, axis=0)
        self._real_count = len(self._real_activations)
        logger.info("Extracted features for %d real images.", self._real_count)

        # Return a DataLoader of caption batches so the runner can drive generation
        class _CaptionDataset(torch.utils.data.Dataset):
            def __init__(self, captions):
                self._captions = captions

            def __len__(self):
                return len(self._captions)

            def __getitem__(self, idx):
                return self._captions[idx]

        real_count = self._real_count
        batch_size = self.config.batch_size

        def collate_fn(batch):
            return Dataset(
                prompts=batch,
                metadata={"source": "coco_hf", "total_loaded": real_count},
            )

        return DataLoader(
            _CaptionDataset(all_captions),
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def update(
        self,
        images: List[Any],
        _prompts: List[str],
        _metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Extract InceptionV3 features from a batch of generated images and
        accumulate them for the final FID computation.

        Args:
            images:    Generated PIL Images or file paths.
            _prompts:  Unused.
            _metadata: Unused.
        """
        if not images:
            return

        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        self._gen_activations.append(self._extract_features(pil_images))

    def compute(self) -> MetricResult:
        """
        Compute FID from the accumulated real and generated feature distributions.
        All InceptionV3 inference was done in load_dataset() and update() —
        this computes mean, covariance, and Fréchet distance only.
        """
        if self._real_activations is None:
            raise RuntimeError(
                "Real features not available. Call load_dataset() first."
            )

        if not self._gen_activations:
            return MetricResult(
                name="FID",
                value=float("inf"),
                details={"error": "No generated images evaluated"},
            )

        gen_activations = np.concatenate(self._gen_activations, axis=0)
        gen_count = len(gen_activations)

        if gen_count < 2:
            return MetricResult(
                name="FID",
                value=float("inf"),
                details={
                    "error": "At least 2 generated images required to compute FID"
                },
            )

        logger.info(
            "Computing FID: %d real vs %d generated images...",
            self._real_count,
            gen_count,
        )

        try:
            mu_real = np.mean(self._real_activations, axis=0)
            sigma_real = np.cov(self._real_activations, rowvar=False)
            mu_gen = np.mean(gen_activations, axis=0)
            sigma_gen = np.cov(gen_activations, rowvar=False)

            fid_score = _calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
            logger.info(f"FID Score: {fid_score:.4f}")

            return MetricResult(
                name="FID",
                value=fid_score,
                details={
                    "total_generated": gen_count,
                    "total_real": self._real_count,
                    "config": self.config.to_dict(),
                },
            )
        except Exception:
            logger.exception("FID computation failed.")
            raise
