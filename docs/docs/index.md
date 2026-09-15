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

## Hardware

A CUDA GPU is required. Inference-only techniques need ~5 GB VRAM; training-based
techniques peak at 10–16 GB during the training phase. See
[GPU Requirements](running-experiments/gpu-requirements.md) for details.
