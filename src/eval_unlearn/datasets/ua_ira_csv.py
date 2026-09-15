from typing import Optional
import csv

from torch.utils.data import DataLoader

from ..types import Dataset
from ..registry import register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


@register_dataset("ua_ira_csv")
def load_ua_ira_csv(
    target_prompts_path: str,
    retain_prompts_path: str,
    target_concept_name: str = "target_concept",
    retain_concept_name: str = "retain_concept",
    target_limit: Optional[int] = None,
    retain_limit: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DataLoader:
    """
    Load target and retain prompts from CSV files.

    Each CSV should have a 'prompt' column (first column or named 'prompt').

    Returns a DataLoader that yields Dataset batches. Each batch has:
      - prompts: list of prompt strings (target and retain mixed)
      - metadata["concepts"]: list of concept names parallel to prompts
      - metadata["categories"]: list of "target" | "retain" parallel to prompts
      - metadata["target_prompt_end_index"]: index where target prompts end

    Args:
        target_prompts_path: Path to CSV file with target prompts (concept to erase).
        retain_prompts_path: Path to CSV file with retain prompts (concept to keep).
        target_concept_name: Name of target concept (e.g., "nudity", "Mickey Mouse").
        retain_concept_name: Name of retain concept (e.g., "person", "Minnie Mouse").
        target_limit: Max target prompts to load.
        retain_limit: Max retain prompts to load.
        batch_size: Number of prompts per batch.
    """
    logger.info("Loading UA_IRA from CSV files...")
    logger.info(f"  Target: {target_prompts_path} (concept: {target_concept_name})")
    logger.info(f"  Retain: {retain_prompts_path} (concept: {retain_concept_name})")

    # Load target prompts
    target_prompts = []
    try:
        with open(target_prompts_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if target_limit and len(target_prompts) >= target_limit:
                    break
                # Try 'prompt' column first, fallback to first column
                prompt = row.get("prompt") or next(iter(row.values())) if row else ""
                if prompt:
                    target_prompts.append((prompt, target_concept_name, "target"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Target prompts file not found: {target_prompts_path}")
    except Exception as e:
        raise RuntimeError(
            f"Error loading target prompts from {target_prompts_path}: {e}"
        )

    # Load retain prompts
    retain_prompts = []
    try:
        with open(retain_prompts_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if retain_limit and len(retain_prompts) >= retain_limit:
                    break
                # Try 'prompt' column first, fallback to first column
                prompt = row.get("prompt") or next(iter(row.values())) if row else ""
                if prompt:
                    retain_prompts.append((prompt, retain_concept_name, "retain"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Retain prompts file not found: {retain_prompts_path}")
    except Exception as e:
        raise RuntimeError(
            f"Error loading retain prompts from {retain_prompts_path}: {e}"
        )

    logger.info(
        f"Loaded {len(target_prompts)} target and {len(retain_prompts)} retain prompts."
    )

    if not target_prompts or not retain_prompts:
        raise ValueError(
            f"No prompts loaded. Target: {len(target_prompts)}, Retain: {len(retain_prompts)}"
        )

    # Combine: target prompts first, then retain
    all_prompts = target_prompts + retain_prompts
    target_prompt_end_index = len(target_prompts)

    def collate_fn(batch):
        return Dataset(
            prompts=[item[0] for item in batch],
            metadata={
                "source": "ua_ira_csv",
                "target_path": target_prompts_path,
                "retain_path": retain_prompts_path,
                "concepts": [item[1] for item in batch],
                "categories": [item[2] for item in batch],
                "target_prompt_end_index": target_prompt_end_index,
            },
        )

    # Create a simple list-based dataset for batching
    class _PromptDataset:
        def __init__(self, prompts):
            self._prompts = prompts

        def __len__(self):
            return len(self._prompts)

        def __getitem__(self, idx):
            return self._prompts[idx]

    return DataLoader(
        _PromptDataset(all_prompts),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
