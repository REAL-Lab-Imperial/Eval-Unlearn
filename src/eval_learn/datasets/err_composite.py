import itertools
from typing import Optional

import torch.utils.data
from torch.utils.data import DataLoader
from datasets import load_dataset as hf_load_dataset, Features, Value

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger
from .hf_stream import load_hf_config
from .i2p_csv import CONCEPT_TO_I2P_CATEGORY

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


class _ERRCompositeIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that streams and merges I2P, ERR challenge, and Ring-A-Bell."""

    def __init__(self, i2p_ds, i2p_cfg, ch_ds, challenge_cfg, rab_ds, rab_cfg):
        self.i2p_ds = i2p_ds
        self.i2p_cfg = i2p_cfg
        self.ch_ds = ch_ds
        self.challenge_cfg = challenge_cfg
        self.rab_ds = rab_ds
        self.rab_cfg = rab_cfg

    def __iter__(self):
        # Stream I2P (target)
        for row in self.i2p_ds:
            yield (
                row[self.i2p_cfg["caption_col"]],
                row[self.i2p_cfg["concept_col"]],
                "target",
            )

        # Stream ERR challenge (retain)
        for row in self.ch_ds:
            yield (
                row[self.challenge_cfg["caption_col"]],
                row[self.challenge_cfg["concept_col"]],
                "retain",
            )

        # Stream Ring-A-Bell (adversarial)
        for row in self.rab_ds:
            yield (
                row[self.rab_cfg["caption_col"]],
                row[self.rab_cfg["concept_col"]],
                "adversarial",
            )


@register_dataset("err_composite")
def load_err_composite(
    target_limit: Optional[int] = None,
    retain_limit: Optional[int] = None,
    adversarial_limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    token: Optional[str] = None,
) -> DataLoader:
    """
    Stream and combine ERR composite data from three HuggingFace sources:
      - I2P (target prompts)
      - ERR challenge (retain prompts)
      - Ring-A-Bell (adversarial prompts)

    Each source is streamed separately (text is lightweight) then combined
    into a single DataLoader. Each batch yields a Dataset with:
      - prompts: list of prompt strings
      - metadata["concepts"]: list of concept strings, parallel to prompts
      - metadata["categories"]: list of "target" | "retain" | "adversarial"

    Args:
        target_limit:      Max rows from I2P.
        retain_limit:      Max rows from ERR challenge.
        adversarial_limit: Max rows from Ring-A-Bell.
        batch_size:        Number of rows per batch.
        token:             HF token (falls back to HF_TOKEN env var).
    """
    i2p_cfg = load_hf_config("i2p")
    challenge_cfg = load_hf_config("err_challenge")
    rab_cfg = load_hf_config("ring_a_bell")

    # --- I2P (target) ---
    logger.info("Streaming I2P (target) from %s...", i2p_cfg["repo_id"])
    i2p_features = Features({
        'prompt': Value('string'),
        'categories': Value('string'),
    })
    i2p_data_files = i2p_cfg.get("data_files")
    if i2p_data_files:
        i2p_ds = hf_load_dataset(
            i2p_cfg["repo_id"],
            data_files=i2p_data_files,
            split=i2p_cfg.get("split", "train"),
            features=i2p_features,
            streaming=True,
            token=token,
        )
    else:
        i2p_ds = hf_load_dataset(
            i2p_cfg["repo_id"],
            split=i2p_cfg.get("split", "train"),
            features=i2p_features,
            streaming=True,
            token=token,
        )
    i2p_category = CONCEPT_TO_I2P_CATEGORY["nudity"]  # "sexual"
    i2p_ds = i2p_ds.filter(
        lambda row: i2p_category in [c.strip() for c in (row.get(i2p_cfg["concept_col"]) or "").split(",")]
    )
    if target_limit is not None:
        i2p_ds = list(itertools.islice(i2p_ds, target_limit))

    # --- ERR challenge (retain) ---
    logger.info("Streaming ERR challenge (retain) from %s...", challenge_cfg["repo_id"])
    challenge_features = Features({
        'concept_type': Value('string'),
        'concept_name': Value('string'),
        'direct_prompt': Value('string'),
        'indirect_prompt': Value('string'),
        'adversarial_prompt': Value('string'),
    })
    challenge_data_files = challenge_cfg.get("data_files")
    if challenge_data_files:
        ch_ds = hf_load_dataset(
            challenge_cfg["repo_id"],
            data_files=challenge_data_files,
            split=challenge_cfg.get("split", "train"),
            features=challenge_features,
            streaming=True,
            token=token,
        )
    else:
        ch_ds = hf_load_dataset(
            challenge_cfg["repo_id"],
            split=challenge_cfg.get("split", "train"),
            features=challenge_features,
            streaming=True,
            token=token,
        )
    if retain_limit is not None:
        ch_ds = list(itertools.islice(ch_ds, retain_limit))

    # --- Ring-A-Bell (adversarial) ---
    logger.info("Streaming Ring-A-Bell (adversarial) from %s...", rab_cfg["repo_id"])
    rab_features = Features({
        'prompt': Value('string'),
        'concept': Value('string'),
    })
    rab_data_files = rab_cfg.get("data_files")
    if rab_data_files:
        rab_ds = hf_load_dataset(
            rab_cfg["repo_id"],
            data_files=rab_data_files,
            split=rab_cfg.get("split", "train"),
            features=rab_features,
            streaming=True,
            token=token,
        )
    else:
        rab_ds = hf_load_dataset(
            rab_cfg["repo_id"],
            split=rab_cfg.get("split", "train"),
            features=rab_features,
            streaming=True,
            token=token,
        )
    if adversarial_limit is not None:
        rab_ds = list(itertools.islice(rab_ds, adversarial_limit))

    # Create merged iterable dataset
    merged_ds = _ERRCompositeIterableDataset(
        i2p_ds, i2p_cfg, ch_ds, challenge_cfg, rab_ds, rab_cfg
    )

    def collate_fn(batch):
        prompts, concepts, categories = zip(*batch)
        return Dataset(
            prompts=list(prompts),
            metadata={
                "source": "err_composite_hf",
                "concepts": list(concepts),
                "categories": list(categories),
            },
        )

    return DataLoader(
        merged_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )
