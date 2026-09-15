#!/usr/bin/env python3
"""Quick test to verify what types datasets return."""

import os
import io
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

def test_dataset_types():
    from datasets import load_dataset, IterableDataset, IterableDatasetDict

    token = os.getenv("HF_TOKEN")

    print("\n" + "="*60)
    print("Testing Dataset Types")
    print("="*60)

    # Test 1: I2P with data_files
    print("\n1. I2P with data_files:")
    ds1 = load_dataset("AIML-TUDA/i2p", data_files="i2p_benchmark.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds1)}")
    print(f"   Is IterableDataset: {isinstance(ds1, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds1, IterableDatasetDict)}")
    assert isinstance(ds1, IterableDataset) and not isinstance(ds1, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 2: I2P with split
    print("\n2. I2P with split='train':")
    ds2 = load_dataset("AIML-TUDA/i2p", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds2)}")
    print(f"   Is IterableDataset: {isinstance(ds2, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds2, IterableDatasetDict)}")
    assert isinstance(ds2, IterableDataset) and not isinstance(ds2, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 3: Challenge with data_files
    print("\n3. Challenge with data_files:")
    ds3 = load_dataset("Unlearningltd/datasets", data_files="ERR/raw_csv_data/challenge_dataset.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds3)}")
    print(f"   Is IterableDataset: {isinstance(ds3, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds3, IterableDatasetDict)}")
    assert isinstance(ds3, IterableDataset) and not isinstance(ds3, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 4: TIFA with data_files (with streaming)
    print("\n4. TIFA with data_files (with streaming):")
    ds4 = load_dataset("Unlearningltd/datasets", data_files="tifa/tifa_dataset.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds4)}")
    print(f"   Is IterableDataset: {isinstance(ds4, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds4, IterableDatasetDict)}")
    assert isinstance(ds4, IterableDataset) and not isinstance(ds4, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 5: Ring-A-Bell with data_files (with streaming)
    print("\n5. Ring-A-Bell with data_files (with streaming):")
    ds5 = load_dataset("Unlearningltd/datasets", data_files="ring_a_bell/ring_a_bell_dataset.csv", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds5)}")
    print(f"   Is IterableDataset: {isinstance(ds5, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds5, IterableDatasetDict)}")
    assert isinstance(ds5, IterableDataset) and not isinstance(ds5, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    # Test 6: COCO with streaming
    print("\n6. COCO with streaming:")
    ds6 = load_dataset("phiyodr/coco2017", split="train", streaming=True, token=token)
    print(f"   Type: {type(ds6)}")
    print(f"   Is IterableDataset: {isinstance(ds6, IterableDataset)}")
    print(f"   Is IterableDatasetDict: {isinstance(ds6, IterableDatasetDict)}")

    # Inspect first sample
    first_sample = next(iter(ds6))
    print(f"   Sample keys: {first_sample.keys()}")
    print(f"   Has 'coco_url': {'coco_url' in first_sample}")
    print(f"   Has 'captions': {'captions' in first_sample}")
    if "captions" in first_sample:
        print(f"   Captions type: {type(first_sample['captions'])}")
        if isinstance(first_sample['captions'], list):
            print(f"   First caption: {first_sample['captions'][0]}")

    assert isinstance(ds6, IterableDataset) and not isinstance(ds6, IterableDatasetDict), "Must be IterableDataset, not IterableDatasetDict"

    print("\n" + "="*60)


def test_coco_image_download():
    """Validate that COCO images can be downloaded and converted to PIL Images."""
    from eval_unlearn.datasets.hf_stream import load_hf_config

    token = os.getenv("HF_TOKEN")

    cfg = load_hf_config("coco")
    caption_col = cfg["caption_col"]
    url_col = cfg["url_col"]

    print("\n" + "="*60)
    print("Testing COCO Image Download")
    print("="*60)

    from datasets import load_dataset

    hf_ds = load_dataset(
        cfg["repo_id"], split=cfg["split"], streaming=True, token=token
    )

    # Test with first sample
    first_sample = next(iter(hf_ds))
    url = first_sample[url_col]
    caption = first_sample[caption_col]

    print(f"\nTesting URL: {url}")
    print(f"Caption: {caption if isinstance(caption, str) else caption[0]}")

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img_bytes = response.content

        print(f"✓ Downloaded {len(img_bytes)} bytes")

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        print(f"✓ Successfully converted to PIL Image: {img.size} {img.mode}")

        assert img.mode == "RGB", f"Expected RGB, got {img.mode}"
        print("\n" + "="*60)
        return True
    except Exception as e:
        print(f"✗ Failed to download/convert image: {e}")
        print("\n" + "="*60)
        raise


if __name__ == "__main__":
    test_dataset_types()
    test_coco_image_download()
