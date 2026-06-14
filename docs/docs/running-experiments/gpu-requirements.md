# GPU Requirements

Eval-Learn runs Stable Diffusion pipelines and, for training-based techniques, fine-tunes
or modifies model weights. Both phases require a CUDA GPU. This page documents the VRAM
requirements for each technique and metric so you can plan your hardware accordingly.

---

## Baseline: the diffusion pipeline

All techniques are built on `CompVis/stable-diffusion-v1-4`. The pipeline's VRAM footprint
at rest (loaded, not generating) is approximately:

| Precision | VRAM |
|-----------|------|
| fp16 (default on CUDA) | ~4–5 GB |
| fp32 | ~8–10 GB |

All techniques default to `use_fp16: true` when running on CUDA. Use fp32 only if you
encounter NaN losses during training (most relevant for ESD and CA).

---

## Per-technique requirements

### Inference-only techniques

These techniques do not train the model — they apply a fixed intervention at generation
time. VRAM usage is close to the baseline pipeline.

| Technique | Inference VRAM | Notes |
|-----------|---------------|-------|
| Free Run | ~5 GB | Baseline pipeline only |
| SLD | ~5 GB | Guidance applied at inference time |
| SAFREE | ~5–6 GB | SVF and LRA hooks add marginal overhead |
| SAeUron | ~5–6 GB | Loads a small SAE (~200 MB) alongside the pipeline |
| Concept Steerers | ~5–6 GB | Loads a small SAE (~200 MB) alongside the pipeline |
| TraSCE | ~5–6 GB | Holds VAE, text encoder, UNet separately; similar total |
| UCE (preset) | ~5 GB | Loads pre-built weights; no training |

### Training-based techniques

These techniques modify model weights before generating. They have a **training phase** with
higher peak VRAM and an **inference phase** close to the baseline.

| Technique | Training VRAM | Inference VRAM | Notes |
|-----------|--------------|----------------|-------|
| MACE | ~6–8 GB | ~5 GB | Closed-form update; loads text encoder and UNet together |
| SSD | ~8–10 GB | ~5 GB | Fisher estimation requires UNet and text encoder in fp32 alongside each other |
| CA | ~8–10 GB | ~5 GB | Fine-tunes in fp32 for numerical stability |
| ESD | ~10–12 GB | ~5 GB | Loads a frozen reference UNet copy during training alongside the trainable UNet |
| CoGFD | ~10–12 GB | ~5 GB | Same frozen reference UNet pattern as ESD |
| AdvUnlearn | ~14–16 GB | ~5 GB | Adversarial training loop; most demanding technique |

!!! note "Training phase vs inference phase"
    For training-based techniques, the **training phase** runs first and is the VRAM bottleneck.
    Once training completes, the extra training-only models are freed and generation runs
    at the inference-phase footprint. If your GPU has enough VRAM for training, inference
    will not be the constraint.

!!! tip "Use save_path to skip retraining"
    Training-based techniques support `load_path` / `save_path`. Train once, save the
    weights, and reload on every subsequent run — this skips the expensive training phase
    entirely and keeps VRAM at the inference level. See
    [Caching adversarial prompts and technique weights](caching-adversarial-prompts.md).

---

## Metric VRAM overhead

Metrics that run neural models also consume VRAM. The runner loads one metric at a time
and frees it before loading the next, so you only ever need the technique's inference-phase
footprint **plus one metric** simultaneously.

| Metric | Additional VRAM |
|--------|----------------|
| ASR I2P (nudity / NudeNet) | Negligible — CPU-based detector |
| ASR I2P (other concepts / Q16 or CLIP) | +~600 MB |
| ASR P4D | +~600 MB (CLIP backbone) |
| ASR Ring-A-Bell | +~600 MB (CLIP backbone) |
| ASR MMA-Diffusion | +~600 MB (CLIP backbone) |
| CLIP Score | +~600 MB |
| FID | +~100 MB (InceptionV3) |
| ERR | Negligible — NudeNet CPU |
| TIFA | +~1–2 GB (BLIP-2 VQA model) |
| UA-IRA | +~600 MB (CLIP) |

---

## Recommended minimum VRAM by use case

| Use case | Minimum | Recommended |
|----------|---------|-------------|
| Inference-only techniques (SLD, SAFREE, SAeUron, etc.) | 8 GB | 12 GB |
| Training techniques without frozen copy (MACE, CA, SSD) | 12 GB | 16 GB |
| Training techniques with frozen copy (ESD, CoGFD) | 16 GB | 24 GB |
| AdvUnlearn | 16 GB | 24 GB+ |

These figures include headroom for metric models and PyTorch's CUDA allocator overhead.
Running exactly at the minimum may work but leaves no room for larger batch sizes or
higher `num_inference_steps`.

---

## Out-of-memory errors

If you encounter a CUDA OOM:

**Reduce inference cost:**

- Lower `num_inference_steps` (e.g. from 50 to 30)
- Reduce the `limit` on your metric config to generate fewer images per run

**Use saved weights:**

- Set `save_path` on the first run, then `load_path` on subsequent runs to skip
  training entirely — this eliminates the training-phase VRAM peak

**Force fp16:**

- Ensure `use_fp16: true` is set in your technique config (default on CUDA)
- If you changed it to fp32 to fix NaN losses, see if a lower learning rate resolves
  the instability instead

**Run techniques sequentially, not concurrently:**

- Never launch two eval-unlearn processes on the same GPU — each will try to load a
  full SD pipeline

---

## CPU fallback

All techniques accept `device: cpu`. Training-based techniques will work but will be
very slow (hours per technique). Inference-only techniques are more practical on CPU
for small prompt sets. fp16 is automatically disabled on CPU.

---

## Multi-GPU

Eval-Learn does not support data-parallel or model-parallel multi-GPU training. Each
technique runs on a single device. On a multi-GPU machine, run one technique per GPU
by submitting separate jobs with `device: cuda:0`, `device: cuda:1`, etc. in each config.

For cluster workflows, see [Running on a GPU Cluster](cluster.md).
