import io
from typing import Optional

import torch
import requests
from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset
from PIL import Image

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


@register_dataset("coco_parquet")
def load_coco_parquet(
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream COCO images and captions directly from HuggingFace.

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of caption strings
      - metadata["images"]: (B, C, H, W) float tensor, preprocessed for InceptionV3

    Args:
        limit:      Max number of rows to stream.
        batch_size: Number of images per batch.
        token:      HF token (falls back to HF_TOKEN env var).
    """
    from torchvision import transforms

    cfg = load_hf_config("coco")

    _inception_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    logger.info(
        "Setting up HF streaming for COCO (%s, split=%s)...",
        cfg["repo_id"],
        cfg["split"],
    )

    hf_ds = hf_load_dataset(
        cfg["repo_id"], split=cfg["split"], streaming=True, token=token
    )
    if limit is not None:
        hf_ds = list(hf_ds.take(limit))

    caption_col = cfg["caption_col"]
    url_col = cfg["url_col"]

    def collate_fn(batch):
        images = []
        captions = []

        for row in batch:
            # Download image from COCO URL
            try:
                response = requests.get(row[url_col], stream=True, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception as e:
                logger.warning(
                    f"Failed to load image from {row[url_col]}: {e}"
                )
                continue

            images.append(_inception_transform(img))

            # COCO captions are a list — take the first
            caption = row[caption_col]
            if isinstance(caption, list):
                caption = caption[0]
            captions.append(caption)

        if images:
            return Dataset(
                prompts=captions,
                metadata={
                    "source": "coco_hf",
                    "repo_id": cfg["repo_id"],
                    "images": torch.stack(images),  # (B, C, H, W)
                },
            )
        else:
            # All images in batch failed — return empty batch
            logger.warning("All images in batch failed to load")
            return Dataset(
                prompts=[],
                metadata={
                    "source": "coco_hf",
                    "repo_id": cfg["repo_id"],
                    "images": torch.empty((0, 3, 299, 299)),
                },
            )

    return DataLoader(
        hf_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )
