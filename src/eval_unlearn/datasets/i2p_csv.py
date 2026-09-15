from typing import Optional

from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32

# Maps user-facing concept names to I2P dataset category labels.
# I2P categories come from the AIML-TUDA/i2p dataset's `categories` column.
CONCEPT_TO_I2P_CATEGORY: dict = {
    "nudity":           "sexual",
    "harassment":       "harassment",
    "hate":             "hate",
    "illegal activity": "illegal activity",
    "self-harm":        "self-harm",
    "shocking":         "shocking",
    "violence":         "violence",
}


@register_dataset("i2p_csv")
def load_i2p_csv(
    concept: Optional[str] = None,
    limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Load prompts from the I2P dataset from HuggingFace.

    If `concept` is provided, only rows whose `categories` column contains
    the corresponding I2P category label are returned.

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of prompt strings

    Args:
        concept:    Concept to filter by (e.g. 'nudity', 'violence').
                    ``None`` loads all rows without filtering.
        limit:      Max number of rows (applied after filtering).
        batch_size: Number of prompts per batch.
        token:      HF token (falls back to HF_TOKEN env var).
    """
    cfg = load_hf_config("i2p")
    caption_col = cfg["caption_col"]
    concept_col = cfg.get("concept_col", "categories")

    # Resolve the I2P category label for the requested concept
    i2p_category: Optional[str] = None
    if concept is not None:
        i2p_category = CONCEPT_TO_I2P_CATEGORY.get(concept)
        if i2p_category is None:
            raise ValueError(
                f"No I2P category mapping for concept '{concept}'. "
                f"Supported concepts: {sorted(CONCEPT_TO_I2P_CATEGORY)}"
            )

    logger.info(
        "Loading I2P dataset (%s, split=%s)%s...",
        cfg["repo_id"],
        cfg["split"],
        f" filtered to category='{i2p_category}'" if i2p_category else "",
    )

    hf_ds = hf_load_dataset(cfg["repo_id"], split=cfg["split"], token=token)

    if i2p_category is not None:
        hf_ds = hf_ds.filter(
            lambda row: i2p_category in [c.strip() for c in (row.get(concept_col) or "").split(",")]
        )

    if limit is not None:
        hf_ds = hf_ds.select(range(min(limit, len(hf_ds))))

    prompts = hf_ds[caption_col]

    def collate_fn(batch):
        return Dataset(
            prompts=batch,
            metadata={
                "source": "i2p_hf",
                "repo_id": cfg["repo_id"],
                "concept": concept,
            },
        )

    return DataLoader(
        prompts, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )
