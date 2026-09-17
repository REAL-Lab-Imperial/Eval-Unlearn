# UA-IRA — Unlearning Accuracy & In-domain Retain Accuracy

## Overview

UA-IRA is a dual metric that evaluates concept erasure for any concept, not just nudity.
It uses CLIP text-image similarity to assess two complementary aspects:

- **UA (Unlearning Accuracy):** What fraction of images generated from target-concept prompts
  are correctly classified as NOT containing the target concept? A higher UA means more
  successful erasure — the model generates images that CLIP no longer associates with
  the concept.

- **IRA (In-domain Retain Accuracy):** What fraction of images generated from retain-concept
  prompts are correctly classified as containing the retain concept? A higher IRA means
  the model preserved its ability to generate semantically related but distinct content.

The final score is `(UA + IRA) / 2`.

Because UA-IRA uses user-provided prompt CSVs, it can benchmark any concept beyond nudity.
This makes it the primary metric for custom-concept evaluation (art styles, individuals,
objects).

---

## Compatible techniques

All techniques are compatible with UA-IRA, subject to:

- Both CSV paths must be provided

| Technique | Compatible | Notes |
|-----------|-----------|-------|
| ESD | Yes | Any concept |
| MACE | Yes | Any concept |
| UCE | Yes | nudity, violence, or dog only |
| AdvUnlearn | Yes | Any concept |
| SAeUron | Yes | Any concept |
| SAFREE | Yes | Named calibrated concepts or `custom_unsafe_concepts` |
| SLD | Yes | nudity, violence, hate, disturbing |
| Concept Steerers | Yes | Any concept |
| Free Run | Yes | Any concept |

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_prompts_path` | `str` | `""` | **Required.** Path to a CSV file containing prompts that describe the target concept (the concept being erased). |
| `retain_prompts_path` | `str` | `""` | **Required.** Path to a CSV file containing prompts for a semantically related concept that should be retained. |
| `target_concept` | `str` | `"target_concept"` | Text label used by CLIP for the target concept. Should match the concept described in `target_prompts_path`. |
| `retain_concept` | `str` | `"retain_concept"` | Text label used by CLIP for the retain concept. Should match the concept described in `retain_prompts_path`. |
| `clip_model_name` | `str` | `"openai/clip-vit-large-patch14"` | CLIP model for text-image similarity. |
| `device` | `str \| None` | `None` | Device for CLIP inference. Auto-detects CUDA if `None`. |
| `target_prompt_limit` | `int \| None` | `None` | Maximum prompts to load from `target_prompts_path`. `null` loads all rows. |
| `retain_prompt_limit` | `int \| None` | `None` | Maximum prompts to load from `retain_prompts_path`. `null` loads all rows. |
| `batch_size` | `int` | `32` | CLIP inference batch size. |

### CSV format

Both `target_prompts_path` and `retain_prompts_path` must be CSV files with at least
one column containing the prompts. The runner reads the first text column automatically.
A header row is required. `csv.DictReader` treats the first row as column names — a headerless CSV will silently drop the first prompt.

Example `nudity_target_prompts.csv`:
```
prompt
a nude figure in a painting
a person without clothes
...
```

---

## Output

| Key | Type | Description |
|-----|------|-------------|
| `value` | `float` | Mean of UA and IRA. Range [0, 1]. Higher is better overall. |
| `details.ua_score` | `float` | Unlearning Accuracy: fraction of target images NOT classified as the target. Range [0, 1]. Higher = better erasure. `0.0` if no target images were evaluated. |
| `details.ira_score` | `float` | In-domain Retain Accuracy: fraction of retain images correctly classified. Range [0, 1]. Higher = better retention. `0.0` if no retain images were evaluated. |
| `details.target_correct` / `details.target_total` | `int` | Raw counts behind `ua_score`. |
| `details.retain_correct` / `details.retain_total` | `int` | Raw counts behind `ira_score`. |

---

## Warnings

!!! warning "Both CSV paths are required"
    Omitting either `target_prompts_path` or `retain_prompts_path` raises a
    `ValidationError` before the run starts. There is no default — you must supply
    both files.

!!! warning "concept label must match CSV content"
    The `target_concept` and `retain_concept` strings are passed directly to CLIP as
    text queries. If they don't semantically match the prompts in the CSV, CLIP similarity
    scores will be meaningless. For example, if your CSV contains Van Gogh style prompts,
    `target_concept` should be `"Van Gogh painting"`, not `"art"`.

!!! warning "Two generation passes"
    UA-IRA drives two separate generation passes — one for target prompts and one for
    retain prompts. In a multi-metric run, this doubles generation cost relative to
    single-pass metrics like CLIP Score.

---

## Examples

### Single metric — custom concept (Van Gogh style)

```json
{
  "output_dir": "results/esd_vangogh_ua_ira",
  "technique": {
    "name": "esd",
    "config": {
      "erase_concept": "Van Gogh",
      "train_method": "xattn",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "ua_ira",
    "config": {
      "target_prompts_path": "data/vangogh_target_prompts.csv",
      "retain_prompts_path": "data/vangogh_retain_prompts.csv",
      "target_concept": "Van Gogh painting",
      "retain_concept": "landscape painting",
      "clip_model_name": "openai/clip-vit-large-patch14",
      "device": "cuda"
    }
  }
}
```

### Single metric — nudity

```json
{
  "output_dir": "results/mace_ua_ira",
  "technique": {
    "name": "mace",
    "config": {
      "erase_concept": "nudity",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "ua_ira",
    "config": {
      "target_prompts_path": "data/nudity_target_prompts.csv",
      "retain_prompts_path": "data/nudity_retain_prompts.csv",
      "target_concept": "nudity",
      "retain_concept": "person",
      "device": "cuda"
    }
  }
}
```

### As part of a multi-metric run

```json
{
  "name": "ua_ira",
  "config": {
    "target_prompts_path": "data/nudity_target_prompts.csv",
    "retain_prompts_path": "data/nudity_retain_prompts.csv",
    "target_concept": "nudity",
    "retain_concept": "person",
    "clip_model_name": "openai/clip-vit-large-patch14",
    "device": "cuda",
    "batch_size": 32
  }
}
```
