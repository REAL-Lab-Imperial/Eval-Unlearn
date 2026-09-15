# Multi Benchmark Runner

## Overview

`MultiBenchmarkRunner` runs one technique against multiple metrics in a single invocation.
Each metric drives its own independent generation pass with its own dataset, so every metric
evaluates on exactly the prompts it expects.

**Execution flow:**

1. Validate the technique against every metric in the list.
2. Initialize the technique once (shared across all metrics).
3. For each metric in order:
   a. Initialize the metric and load its dataset.
   b. Generate images batch by batch and feed them to `metric.update()`.
   c. Finalize with `metric.compute()`.
   d. Save images and metadata for that metric immediately.
   e. Free the metric's GPU models before loading the next one.
4. Build a combined report across all metrics and save it.

Metrics are loaded and freed one at a time to avoid exhausting GPU memory when multiple
heavy models (CLIP, detectors) would otherwise coexist.

---

## Config format

The multi runner is selected automatically when your config contains a `metrics` key
(plural list). Use `metric` (singular) to select the [single runner](single-runner.md) instead.

=== "YAML"

    ```yaml
    technique:
      name: esd
      config:
        erase_concept: nudity
        train_method: noxattn
        save_path: checkpoints/esd_nudity.pt

    metrics:
      - name: asr
        config:
          device: cuda
          limit: 500
      - name: fid
        config:
          device: cuda
          limit: 1000
      - name: clip_score
        config:
          device: cuda
          limit: 300

    output_dir: results/esd_nudity_multi
    ```

=== "JSON"

    ```json
    {
      "technique": {
        "name": "esd",
        "config": {
          "erase_concept": "nudity",
          "train_method": "noxattn",
          "save_path": "checkpoints/esd_nudity.pt"
        }
      },
      "metrics": [
        { "name": "asr_i2p",        "config": { "device": "cuda", "limit": 500 } },
        { "name": "fid",        "config": { "device": "cuda", "limit": 1000 } },
        { "name": "clip_score", "config": { "device": "cuda", "limit": 300 } }
      ],
      "output_dir": "results/esd_nudity_multi"
    }
    ```

---

## Python API

```python
from eval_unlearn.runners import MultiBenchmarkRunner

runner = MultiBenchmarkRunner(
    technique_name="esd",
    metric_names=["asr", "fid", "clip_score"],
    technique_config={
        "erase_concept": "nudity",
        "train_method": "noxattn",
        "save_path": "checkpoints/esd_nudity.pt",
    },
    metric_configs={
        "asr":        {"device": "cuda", "limit": 500},
        "fid":        {"device": "cuda", "limit": 1000},
        "clip_score": {"device": "cuda", "limit": 300},
    },
    output_dir="results/esd_nudity_multi",
)
report = runner.run()
```

`run()` returns the combined report dict and writes it to `output_dir`.

---

## Output structure

```
results/esd_nudity_multi/
└── <run_id>/
    ├── asr/
    │   ├── images/
    │   └── metadata.json
    ├── fid/
    │   ├── images/
    │   └── metadata.json
    ├── clip_score/
    │   ├── images/
    │   └── metadata.json
    └── multi/
        └── report.json    # combined report across all metrics
```

Each metric gets its own subdirectory with the images generated for that metric's dataset.
The combined report is written to `multi/report.json` after all metrics complete.

---

## Report schema

```json
{
  "run_id": "a1b2c3d4",
  "timestamp": 1712345678.0,
  "technique_name": "esd",
  "metric_names": ["asr", "fid", "clip_score"],
  "metric_results": {
    "asr_i2p": {
      "name": "asr_i2p",
      "value": 0.12,
      "details": { "..." }
    },
    "fid": {
      "name": "fid",
      "value": 18.4,
      "details": { "..." }
    },
    "clip_score": {
      "name": "clip_score",
      "value": 0.31,
      "details": { "..." }
    }
  }
}
```

---

## Notes

!!! warning "Compatibility"
    Every metric in the list is validated against the technique at construction time. If any
    pair is incompatible, the runner raises `ValueError` before any work begins. See
    [Compatibility](compatibility.md).

!!! warning "Duplicate metrics"
    `metric_names` must not contain duplicates — the runner raises `ValueError` immediately
    if any name appears more than once.

!!! note "Each metric uses its own dataset and generation pass"
    Unlike a single runner run for each metric separately, the technique is initialized once.
    However, images are regenerated per metric because each metric loads its own dataset with
    its own prompts. This is intentional — metrics like `fid` and `asr` evaluate on different
    prompt sets.

!!! note "GPU memory management"
    Metrics are initialized, evaluated, and freed one at a time. `torch.cuda.empty_cache()`
    is called between metrics to release GPU memory before the next metric's models are loaded.
    This allows running multiple heavy metrics on a single GPU without OOM errors.
