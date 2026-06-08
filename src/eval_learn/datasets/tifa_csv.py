import json
from typing import Optional

from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


@register_dataset("tifa_csv")
def load_tifa_csv(
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream TIFA captions and QA pairs from HuggingFace CSV.

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of caption strings
      - metadata["qa_pairs"]: list of QA pair lists, parallel to prompts.
        Each element is a list of {"question": str, "answer": str} dicts.

    Args:
        limit:      Max number of rows to stream.
        batch_size: Number of prompts per batch.
        token:      HF token (falls back to HF_TOKEN env var).
    """
    cfg = load_hf_config("tifa")
    caption_col = cfg["caption_col"]
    qa_col = cfg["qa_col"]

    logger.info(
        "Setting up HF streaming for TIFA (%s, split=%s)...",
        cfg["repo_id"],
        cfg.get("split", "train"),
    )

    hf_ds = hf_load_dataset(
        cfg["repo_id"],
        data_files=cfg.get("data_files"),
        split=cfg.get("split", "train"),
        streaming=True,
        token=token,
    )

    if limit is not None:
        hf_ds = list(hf_ds.take(limit))

    def collate_fn(batch):
        prompts = []
        qa_pairs = []
        for row in batch:
            prompts.append(row[caption_col])
            # Parse QA JSON string back to list of dicts
            qas = json.loads(row[qa_col])
            qa_pairs.append(qas)

        return Dataset(
            prompts=prompts,
            metadata={
                "source": "tifa_csv",
                "repo_id": cfg["repo_id"],
                "qa_pairs": qa_pairs,
            },
        )

    return DataLoader(
        hf_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )
