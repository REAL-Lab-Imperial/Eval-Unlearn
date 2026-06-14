# Concepts

## What is concept unlearning?

Text-to-image diffusion models like Stable Diffusion learn from vast datasets and can
generate almost anything represented in that data — including content that is harmful,
private, or otherwise undesirable. **Concept unlearning** is the problem of modifying a
trained model so that it can no longer generate a specific concept, without retraining
from scratch.

In practice, unlearning techniques target concepts such as:

- **Nudity / explicit content** — the most studied case, with dedicated benchmarks
- **Artistic styles** — erasing a specific artist's style on copyright grounds
- **Named individuals** — preventing generation of a specific person's likeness

The challenge is not erasure alone. A technique that simply destroys the model has
trivially "forgotten" the target concept — but it has also destroyed everything else.
A good unlearning technique erases the target concept while leaving the model's general
image generation capability intact. This tension between **erasure** and **retention**
is what benchmarking measures.

---

## Core entities

### Technique

A technique is an algorithm that takes a base diffusion model and returns a modified
version. The modification can be a weight update (fine-tuning), a weight mask, an
inference-time intervention, or a combination. Eval-Learn supports:

| Technique | Approach |
|-----------|----------|
| ESD | Fine-tuning with erasing loss |
| MACE | Closed-form weight editing |
| UCE | Unified concept editing (presets: nudity, violence, dog) |
| AdvUnlearn | Adversarially robust fine-tuning |
| SAeUron | Sparse autoencoder feature suppression |
| SAFREE | Training-free self-guidance filtering |
| SLD | Safe latent diffusion (inference-time) |
| Concept Steerers | Steering vector subtraction |
| Free Run | Allows Stable Diffusion Models hosted on HuggingFace with custom unlearning techniques to be used |

### Metric

A metric takes generated images (and optionally prompts) and returns a score. Metrics
fall into three categories:

**Erasure** — did the technique actually forget the concept?

| Metric | What it measures |
|--------|-----------------|
| ASR I2P | Fraction of generated images detected as unsafe using I2P prompts (NudeNet or CLIP depending on concept) |
| ASR MMA Diffusion | ASR under GCG adversarial prompts generated against the model's text encoder |
| ASR Ring A Bell | ASR under prompts discovered via genetic algorithm against the concept's CLIP vector |
| UA | Fraction of target-concept images not classified as the target |
| ERR | Harmonic mean of forgetting, retention, and adversarial robustness |

**Retention** — is the rest of the model still working?

| Metric | What it measures |
|--------|-----------------|
| CLIP Score | Text-to-image alignment via CLIP cosine similarity |
| FID | Image quality and distribution fidelity vs. a reference set |
| TIFA | Faithfulness via VQA question answering (BLIP-2) |
| IRA | Fraction of retain-concept images correctly classified |

**Adversarial robustness** — does the erasure hold under adversarial prompts?

| Metric | What it measures |
|--------|-----------------|
| MMA-Diffusion | ASR under MMA-Diffusion adversarial prompt attack |
| ASR Custom | ASR under Ring-A-Bell prompt discovery attack |

Erasure and retention scores should always be interpreted together. A technique with
perfect erasure but collapsed CLIP Score has over-erased — it is not a good result.

### Config

A config file wires a technique and one or more metrics together with their
hyperparameters. It is the unit of a benchmark run. See
[Getting Started](getting-started.md) for the config format.

### Base model

Each technique starts from a pretrained Stable Diffusion checkpoint. The base model
is fixed per technique — not all techniques support all checkpoints. Run
`eval-unlearn models` to see which base model each installed technique uses.

---

## What a run produces

When you run a benchmark, Eval-Learn:

1. Loads the technique and applies it to the base model
2. Generates images from the (now unlearned) model using the metric's prompt dataset
3. Passes those images to each metric
4. Writes results to `output_dir` as JSON

Results can be pushed to Hugging Face Hub for storage and comparison across runs.

---

## Compatibility

Not all technique–metric pairs are valid. ERR is nudity-specific. UCE is limited to three fixed
presets. ASR supports all I2P concept categories.

Before writing a config, check [Compatibility](running-experiments/compatibility.md).
