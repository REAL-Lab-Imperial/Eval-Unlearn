# Demos & Tutorials

The notebooks in this section walk you through the two most common ways to use
eval-learn: evaluating a built-in technique against the standard benchmark suite,
and evaluating your own custom model checkpoint.

---

## Tutorial 1 — Evaluating a Built-in Unlearning Technique

**Notebook:**
`demos/notebooks/tutorial_01_evaluating_an_unlearning_technique.ipynb`

This is the recommended starting point. It covers the full evaluation workflow
using one of the 13 techniques that ship with eval-learn.

### What you will learn

- How to select a technique and configure its hyperparameters
- How to select one or more metrics and tune their settings
- How to run a single-metric benchmark with `SingleBenchmarkRunner`
- How to run multiple metrics in one pass with `MultiBenchmarkRunner`
- How to read and interpret the JSON result report
- How to tune the accuracy–quality trade-off for your use case

### When to use this tutorial

Use this tutorial when you want to reproduce published results, run a quick
comparison between techniques, or explore how hyperparameter changes affect
evaluation scores.

### Quick-start (CLI equivalent)

If you prefer the command line over a notebook, the same experiment can be run
with a single command:

```bash
eval-learn run --config examples/nudity/esd.json
```

See [Getting Started](getting-started.md) for the full CLI reference.

---

## Tutorial 2 — Evaluating Your Own Model

**Notebook:**
`demos/notebooks/tutorial_02_custom_model_evaluation.ipynb`

Use this tutorial when your technique produces a custom model checkpoint that
is not one of the built-in eval-learn techniques — for example, a fine-tuned
Stable Diffusion checkpoint saved as a `.safetensors` or `.pt` file, or any
HuggingFace-compatible text-to-image pipeline.

### What you will learn

- How to load any HuggingFace-compatible checkpoint via the `free_run` technique
- How to point eval-learn at a local model path or a custom HF repository
- How to run the full benchmark suite against your model
- How to compare your model's scores against baseline results

### When to use this tutorial

Use this tutorial when:

- You have trained your own unlearning technique and want to measure it against
  the standard benchmarks
- You have downloaded a pre-trained checkpoint from another source and want to
  evaluate it
- You want to establish a baseline for an unmodified model before applying any
  unlearning technique

### The `free_run` technique

`free_run` is a passthrough wrapper that loads any HuggingFace
`DiffusionPipeline`-compatible model with no safety filtering. Point it at your
checkpoint via the `model_id` config field:

```python
TECHNIQUE_NAME   = "free_run"
TECHNIQUE_CONFIG = {
    "model_id": "/path/to/your/checkpoint",  # local path or HF repo ID
    "device":   "cuda",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
}
```

---

## Additional Demos

The `demos/notebooks/` directory also contains two domain-specific demos that
demonstrate a complete unlearning + evaluation pipeline for specific concept
categories:

| Notebook | Concept | Techniques shown |
|---|---|---|
| `nudity_unlearning_demo.ipynb` | Nudity | ESD, UCE, SLD |
| `violence_unlearning_demo.ipynb` | Violence | ESD, AdvUnlearn, SLD |

These demos are self-contained and can be run independently of the tutorials.
