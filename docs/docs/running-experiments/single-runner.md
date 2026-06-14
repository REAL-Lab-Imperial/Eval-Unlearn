# Single Benchmark Runner

## Overview

`SingleBenchmarkRunner` runs one technique against one metric. It is the standard runner
for targeted evaluations where you want a single score for a single technique–metric pair.

**Execution flow:**

1. Validate the technique–metric pair against the compatibility registry.
2. Initialize the metric and load its dataset.
3. Initialize the technique.
4. Iterate the dataset batch by batch — generate images with the technique, feed them to
   the metric's `update()`.
5. Finalize the metric with `compute()` to produce a `MetricResult`.
6. Save images, metadata, and the report to `output_dir`.

---

## Config format

The single runner is selected automatically when your config contains a `metric` key
(singular). Use `metrics` (plural list) to select the [multi runner](multi-runner.md) instead.

=== "YAML"

    ```yaml
    technique:
      name: esd
      config:
        erase_concept: nudity
        train_method: noxattn
        save_path: checkpoints/esd_nudity.pt

    metric:
      name: asr
      config:
        device: cuda
        limit: 500

    output_dir: results/esd_asr
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
      "metric": {
        "name": "asr_i2p",
        "config": {
          "device": "cuda",
          "limit": 500
        }
      },
      "output_dir": "results/esd_asr"
    }
    ```

---

## Python API

```python
from eval_unlearn.runners import SingleBenchmarkRunner

runner = SingleBenchmarkRunner(
    technique_name="esd",
    metric_name="asr",
    technique_config={
        "erase_concept": "nudity",
        "train_method": "noxattn",
        "save_path": "checkpoints/esd_nudity.pt",
    },
    metric_config={
        "device": "cuda",
        "limit": 500,
    },
    output_dir="results/esd_asr",
)
report = runner.run()
```

`run()` returns the report dict and also writes it to `output_dir`.

---

## Output structure

```
results/esd_asr/
└── <run_id>/
    ├── report.json        # full run report with metric result
    ├── images/            # generated images
    └── metadata.json      # accumulated dataset metadata
```

The `run_id` is an 8-character SHA-256 hash of the technique name, technique config,
metric name, metric config, dataset name, and timestamp. It uniquely identifies the run
so results from repeated experiments don't collide.

---

## Report schema

```json
{
  "run_id": "a1b2c3d4",
  "timestamp": 1712345678.0,
  "technique_name": "esd",
  "metric_name": "asr",
  "dataset_name": "i2p",
  "dataset_metadata": {
    "source": "i2p",
    "total_loaded": 500
  },
  "technique_config": { "..." },
  "metric_config": { "..." },
  "metric_result": {
    "name": "asr_i2p",
    "value": 0.12,
    "details": { "..." }
  }
}
```

---

## Notes

!!! warning "Compatibility"
    Not all technique–metric pairs are valid. The runner raises `ValueError` at construction
    time if the pair is incompatible. See [Compatibility](compatibility.md) before running.

!!! note "Dataset is owned by the metric"
    The metric determines which dataset to load via its `load_dataset()` method. You do not
    specify a dataset directly in the runner config — configure it through the metric's config
    instead (e.g. `limit`, `prompts_path`).
