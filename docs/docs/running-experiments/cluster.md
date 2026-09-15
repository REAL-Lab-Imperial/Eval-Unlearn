# Running on a GPU Cluster

This page covers running eval-unlearn on SLURM-managed clusters. The commands use
SLURM — translate to PBS/LSF as needed.

---

## Single job

The minimal submission script:

```bash
#!/bin/bash
#SBATCH --job-name=eval-unlearn
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.log

source activate your_env
export HF_TOKEN=your_token_here

eval-unlearn run --config examples/nudity/esd.json
```

`output_dir` in the config controls where results are written. Use an absolute path
or set it relative to your project root and `cd` there before submitting.

---

## Array job — one technique per task

To benchmark all techniques against the same concept in parallel, submit an array
job where each task runs a different config:

```bash
#!/bin/bash
#SBATCH --job-name=eval-unlearn-nudity
#SBATCH --array=0-12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.log

source activate your_env
export HF_TOKEN=your_token_here

CONFIGS=(
    examples/nudity/esd.json
    examples/nudity/mace.json
    examples/nudity/uce.json
    examples/nudity/ssd.json
    examples/nudity/ca.json
    examples/nudity/cogfd.json
    examples/nudity/trasce.json
    examples/nudity/advunlearn.json
    examples/nudity/saeuron.json
    examples/nudity/safree.json
    examples/nudity/sld.json
    examples/nudity/concept_steerers.json
    examples/nudity/free_run.json
)

eval-unlearn run --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
```

Each task writes to its own `output_dir` as defined in the config, so there are no
write conflicts.

---

## Caching adversarial prompts across jobs

Adversarial prompt generation (P4D, MMA Diffusion, Ring-A-Bell) is expensive. If
you are running multiple techniques against the same prompt set, generate once and
share the cache across array tasks rather than re-running on every job.

**Step 1 — generate prompts in a dedicated job:**

```bash
#!/bin/bash
#SBATCH --job-name=gen-prompts
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen_prompts_%j.log

source activate your_env
export HF_TOKEN=your_token_here

eval-unlearn run --config gen_prompts.json
```

Where `gen_prompts.json` is a minimal config that runs just the adversarial metric
against a cheap technique (e.g. free_run) to produce the cache:

```json
{
  "output_dir": "results/prompt_cache_run",
  "technique": {
    "name": "free_run",
    "config": { "model_id": "CompVis/stable-diffusion-v1-4", "device": "cuda" }
  },
  "metrics": [
    {
      "name": "asr_p4d",
      "config": {
        "concept_name": "nudity",
        "target_prompts_path": "examples/data/nudity_target_prompts.csv",
        "erase_id": "std",
        "generated_prompts_output": "cache/p4d_nudity.csv",
        "device": "cuda"
      }
    }
  ]
}
```

**Step 2 — add a dependency so technique jobs start after prompts are ready:**

```bash
PROMPT_JOB=$(sbatch --parsable gen_prompts.sh)
sbatch --dependency=afterok:$PROMPT_JOB --array=0-12 benchmark.sh
```

**Step 3 — point each technique config at the cache** by setting
`precomputed_prompts_path` instead of `target_prompts_path`:

```json
{
  "name": "asr_p4d",
  "config": {
    "concept_name": "nudity",
    "precomputed_prompts_path": "cache/p4d_nudity.csv",
    "device": "cuda"
  }
}
```

See [Caching adversarial prompts and technique weights](caching-adversarial-prompts.md)
for the full pattern including MMA Diffusion, Ring-A-Bell, and reusing trained weights.

---

## Reusing trained weights across jobs

Training-based techniques (ESD, MACE, SSD, CoGFD, AdvUnlearn, CA) re-run training
on every invocation. If you want to run multiple metric configurations against the
same trained model without retraining, save the weights on the first run and load
them on subsequent runs via `save_path` / `load_path` in the technique config.

This is particularly useful when iterating on metric parameters — you train once,
then resubmit only the evaluation portion.

---

## HF Hub

Push results from a completed job directly to a HuggingFace dataset repo for
centralised storage:

```bash
eval-unlearn run --config examples/nudity/esd.json --hf-repo your-org/results
```

Or push an existing results directory separately after the job completes:

```bash
eval-unlearn push --repo your-org/results --local-dir results/esd_nudity
```

---

## Tips

**Shared filesystem caching** — set cache directories to a shared path if your home
directory is node-local:

```bash
# HuggingFace models (techniques, CLIP, BLIP-2, etc.)
export HF_HOME=/path/to/shared/storage/.cache/huggingface

# PyTorch hub models (torchvision FID inception weights, etc.)
export TORCH_HOME=/path/to/shared/storage/.cache/torch
```

With a shared home directory these are available to all nodes automatically and
no override is needed.

**GPU memory** — most techniques require at least 16GB VRAM. Training-based
techniques (ESD, CA) require more during the training phase due to the frozen
reference UNet copy — request 40GB if available.

**Time limits** — adversarial prompt generation (especially P4D at `num_iter=3000`)
and long training runs can exceed short-queue limits. Use the caching workflow above
to split generation from evaluation into separate jobs.
