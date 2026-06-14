# FID — Fréchet Inception Distance

## Overview

FID measures the statistical distance between the distribution of generated images and
a reference distribution of real images. Lower FID indicates generated images are more
similar in distribution to real images — both in visual fidelity and diversity.

FID uses an Inception V3 model (pretrained on ImageNet) to extract 2048-dimensional pool
features from each image. It then computes the Fréchet distance between the multivariate
Gaussians fit to the real and generated feature distributions.

**Reference dataset:** COCO (real images, loaded via HuggingFace parquet)

FID is a general-purpose quality metric — it is concept-agnostic and works with any
technique and any `erase_concept`. It captures whether unlearning has degraded the
model's overall generation quality, not whether the target concept was actually erased.

---

## Compatible techniques

All techniques are compatible with FID. No concept restrictions.

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `32` | Batch size for Inception feature extraction. |
| `device` | `str \| None` | `None` | Device for Inception inference. Auto-detects CUDA if `None`. |
| `limit` | `int \| None` | `1000` | Maximum number of real COCO images to load for the reference distribution. Also controls how many generated images are evaluated. |

---

## Output

| Key | Type | Description |
|-----|------|-------------|
| `value` | `float` | FID score. Range [0, ∞). Lower is better (0 = identical distributions). Typical unmodified SD scores are in the range 10–30; values above 50 indicate significant quality degradation. |

---

## Warnings

!!! warning "Requires torchvision"
    FID requires `torchvision` for the Inception V3 model. Install with
    `pip install eval-unlearn[fid]`. Running without it raises an `ImportError`.

!!! warning "Sample count affects reliability"
    FID requires sufficient samples for a reliable Gaussian fit. With `limit=50` or
    fewer, FID scores are highly variable and not meaningful. Use at least 1000 images
    for production benchmarks; 50 is only appropriate for smoke-testing that the metric
    runs at all.

!!! warning "FID alone is insufficient"
    A low FID confirms the model still generates realistic images but says nothing about
    whether the target concept was erased. Always pair FID with an erasure metric (ASR,
    ERR, or UA_IRA).

---

## Examples

### Single metric

```json
{
  "output_dir": "results/mace_fid",
  "technique": {
    "name": "mace",
    "config": {
      "erase_concept": "nudity",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "fid",
    "config": {
      "batch_size": 32,
      "device": "cuda",
      "limit": 1000
    }
  }
}
```

### As part of a multi-metric run

```json
{
  "name": "fid",
  "config": {
    "batch_size": 32,
    "device": "cuda",
    "limit": 1000
  }
}
```
