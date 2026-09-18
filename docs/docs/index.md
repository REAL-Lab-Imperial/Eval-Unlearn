# Eval-Unlearn

A benchmarking framework for evaluating concept unlearning techniques 
in text-to-image diffusion models.

Unlearning techniques modify or constrain Stable Diffusion to prevent 
it from generating specific concepts — nudity, violence, artistic styles, named 
individuals. Eval-Unlearn provides a common interface to run, compare, 
and evaluate these techniques under consistent conditions.

## What it includes

- **13 techniques** — ESD, MACE, UCE, SSD, CA, CoGFD, TraSCE, AdvUnlearn, SAeUron, SAFREE, SLD, Concept Steerers, Free Run
- **9 metrics** — ASR I2P, ASR P4D, ASR MMA-Diffusion, ASR Ring-A-Bell, FID, CLIP Score, ERR, TIFA, UA-IRA
- **2 evaluation modes** — single metric or multiple metrics per technique run
- **Any-concept ASR** — the four ASR metrics work beyond their built-in concept lists:
  for concepts without a dedicated dataset they auto-source target/seed prompts (I2P where
  applicable, generic templates otherwise) and default to VLM-based detection (the same
  model TIFA uses) instead of a fixed nudity/inappropriate-content classifier

## Submit and browse the leaderboard online

A hosted leaderboard and submission portal is available at
[huggingface.co/spaces/REAL-Lab-Imperial/eval-unlearn](https://huggingface.co/spaces/REAL-Lab-Imperial/eval-unlearn) —
browse existing technique rankings, or submit your own model to be scored against the
full metric suite without running eval-unlearn locally.

## Hardware

A CUDA GPU is required. Inference-only techniques need ~5 GB VRAM; training-based
techniques peak at 10–16 GB during the training phase. See
[GPU Requirements](running-experiments/gpu-requirements.md) for details.
