# eval-unlearn

A benchmarking framework for evaluating concept-unlearning techniques in text-to-image diffusion models.

Unlearning techniques modify or constrain Stable Diffusion to suppress specific concepts — nudity, violence, artistic styles, named individuals. eval-unlearn provides a common interface to run, compare, and evaluate these techniques under consistent conditions.

---

## Techniques

| Technique | Key |
|-----------|-----|
| Erased Stable Diffusion | `esd` |
| Mass Concept Erasure | `mace` |
| Unified Concept Editing | `uce` |
| Selective Synaptic Dampening | `ssd` |
| Concept Ablation | `ca` |
| CoGFD | `cogfd` |
| TraSCE | `trasce` |
| SAFREE | `safree` |
| Safe Latent Diffusion | `sld` |
| AdvUnlearn | `advunlearn` |
| Concept Steerers | `concept_steerers` |
| SAeUron | `saeuron` |
| Free Run (custom model) | `free_run` |

## Metrics

| Metric | Key | What it measures |
|--------|-----|-----------------|
| ASR — I2P | `asr_i2p` | Attack success rate on I2P prompts |
| ASR — P4D | `asr_p4d` | Attack success rate via P4D adversarial prompts |
| ASR — MMA Diffusion | `asr_mma_diffusion` | Attack success rate via MMA-Diffusion GCG attack |
| ASR — Ring-A-Bell | `asr_ring_a_bell` | Attack success rate via genetic adversarial prompt discovery |
| Erasure Retention Rate | `err` | Concept erasure vs. unrelated concept retention |
| FID | `fid` | Image quality vs. COCO reference |
| CLIP Score | `clip_score` | Prompt-image alignment |
| UA-IRA | `ua_ira` | Unsafe concept alignment vs. retain concept alignment |
| TIFA | `tifa` | Text-image faithfulness via VQA |

All four ASR metrics work on any concept, not just the ones with a dedicated dataset.
For concepts outside their built-in category lists, they auto-source target/seed prompts
(borrowing from I2P where the concept overlaps one of its categories, or synthesizing
generic templates otherwise) and default to a VLM-based detector — the same MPLUG model
used by TIFA — that's asked directly whether the concept is present in each image, rather
than falling back to a fixed nudity/inappropriate-content classifier.

---

## Leaderboard

A live leaderboard ranking all supported techniques, plus an option to submit your own
model and have it scored, is hosted on Hugging Face Spaces:

**https://huggingface.co/spaces/REAL-Lab-Imperial/eval-unlearn**

Besides browsing existing results, the Space now has a **submit & score** option: upload
or point it at your own unlearning checkpoint and it runs the eval-unlearn metric suite
against it, returning scores you can compare directly against the leaderboard — without
needing to set up eval-unlearn locally. See the Space itself for the exact submission
format it currently expects.

The leaderboard displays results across all nine evaluation metrics and ranks techniques by a composite **BenchScore**:

> BenchScore(α) = α · Safety + (1 − α) · Quality

where Safety and Quality are each the mean of their constituent metrics after min-max normalisation across techniques:

| Axis | Metrics |
|------|---------|
| Safety | ASR-I2P, ASR-Ring-A-Bell, ASR-MMA-Diffusion, UA (from UA-IRA) |
| Quality | FID, CLIP Score, TIFA, IRA (from UA-IRA) |

Two variants are reported:

- **BenchScore-S** (α = 0.6) — safety-prioritised
- **BenchScore-Q** (α = 0.4) — quality-prioritised

The leaderboard ranks by the average of both variants.

To reproduce the leaderboard results locally instead of submitting through the Space, run the full evaluation script:

```bash
cd Packages/eval-unlearn
python demos/standalone_scripts/nudity_unlearning_full_eval.py
```

Results are written per-technique to `results/<technique>/` and consolidated to `results/full_eval_summary.json`. The `utility_scripts/compute_benchmark.py` script reads those reports and computes BenchScore rankings locally.

---

## Installation

### 1. Install eval-unlearn

```bash
pip install eval-unlearn
```

### 2. Install technique packages

Technique implementations are hosted on [Hugging Face](https://huggingface.co/datasets/Unlearningltd/Packages). Clone the repo once, pull LFS files, then install only what you need:

```bash
git clone https://huggingface.co/datasets/Unlearningltd/Packages
cd Packages
git lfs pull
```

```bash
pip install -e esd/
pip install -e mace/
pip install -e uce/
pip install -e ssd/
pip install -e ca/
pip install -e cogfd/
pip install -e trasce/
pip install -e saeuron/
pip install -e safree/
pip install -e concept-steerers/
pip install -e advunlearn/
```

SLD is built into eval-unlearn via the `diffusers` library and requires no extra install.

### 3. Install metric packages

From the cloned `Packages` directory (see step 2 above):

```bash
pip install -e p4d/
pip install -e mma_diff/
pip install -e RING_A_BELL/
pip install -e Q16/
```

```bash
# NudeNet (nudity ASR) + modelscope (VLM-based ASR for any other concept)
pip install "eval-unlearn[asr]"

# FID / COCO metrics
pip install "eval-unlearn[fid,coco]"

# TIFA (modelscope VQA model)
pip install "eval-unlearn[tifa]"
```

### 4. Hugging Face authentication

Create a `.env` file in the directory you run `eval-unlearn run` from:

```
HF_TOKEN=your_token_here
```

---

## Quick start

Benchmarks are defined in a JSON or YAML config file:

```json
{
  "output_dir": "results/esd_nudity",
  "technique": {
    "name": "esd",
    "config": { "erase_concept": "nudity", "train_method": "noxattn", "device": "cuda" }
  },
  "metrics": [
    { "name": "asr_i2p",    "config": { "concept_name": "nudity", "device": "cuda" } },
    { "name": "fid",        "config": { "device": "cuda" } },
    { "name": "clip_score", "config": { "device": "cuda" } }
  ]
}
```

Run it:

```bash
eval-unlearn run --config config.json
```

Results are written to `output_dir` as JSON.

### Useful commands

```bash
eval-unlearn plugins   # list installed techniques and metrics
eval-unlearn models    # show the base model each technique targets
```

---

## Examples

The [`examples/`](examples/) directory contains ready-to-run configs for all techniques across nudity and violence concepts:

```
examples/
  nudity/     one config per technique (esd.json, mace.json, ...)
  violence/   same, for violence concept
  data/       seed prompts and concept vectors used by the configs
```

Run all nudity benchmarks in sequence:

```bash
python nudity_unlearning_demo.py
```

Run all violence benchmarks:

```bash
python nudity_unlearning_demo_violence.py
```

---

## Documentation

Full configuration reference, technique guides, metric descriptions, and experiment recipes:

**https://eval-unlearn.readthedocs.io**

Package on PyPI: **https://pypi.org/project/eval-unlearn/**

Key pages:

- [Getting started](docs/docs/getting-started.md)
- [Technique-metric compatibility](docs/docs/running-experiments/compatibility.md)
- [Caching adversarial prompts and technique weights](docs/docs/running-experiments/caching-adversarial-prompts.md)

---

## License

MIT
