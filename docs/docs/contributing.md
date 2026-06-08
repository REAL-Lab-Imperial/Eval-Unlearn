# Contributing

Thank you for your interest in contributing to eval-learn! The library is
designed to be extensible: adding a new unlearning technique or a new evaluation
metric requires writing only a small number of files and following a common
interface.

This page covers the two main contribution paths. Each has an interactive
notebook that guides you through the implementation and runs the validation
tests your code must pass before a pull request can be merged.

---

## Before You Start

1. **Fork the repository** and create a feature branch from `main`.
2. **Install in editable mode** so your changes are immediately importable:

    ```bash
    git clone <repo-url>
    cd eval-learn-testing/Packages/eval-learn
    pip install -e ".[dev]"
    ```

3. **Run the existing test suite** to confirm your environment is working:

    ```bash
    pytest -m "not integration" -v
    ```

4. Open the relevant contribution notebook (see below), set the variables at
   the top to match your implementation, and run all cells. Every test must
   show `PASS` before you submit.

---

## Contributing a New Unlearning Technique

**Contribution notebook:**
`demos/notebooks/contributing/tutorial_contributing_a_technique.ipynb`

!!! note "Notebook link"
    The notebook is currently available at the path above relative to the
    repository root. A rendered, browsable version will be linked here once the
    library is public on GitHub.

### Overview

Each technique in eval-learn lives in its own sub-package under
`src/eval_learn/techniques/` and exposes two artefacts:

| File | Purpose |
|---|---|
| `config.py` | Frozen dataclass inheriting `BaseConfig`. Declares all hyperparameters and validates them in `__post_init__`. |
| `wrapper.py` | Class decorated with `@register_technique("<name>")`. Exposes a single `generate(prompts, seed, **kwargs) -> List[PIL.Image]` method. |

The wrapper is a **thin adapter** around your technique's external package.
It must not contain training logic — that belongs in the technique package itself.

### Required files

```
src/eval_learn/techniques/
└── your_technique/
    ├── __init__.py
    ├── config.py
    └── wrapper.py
```

### Registration

Add one entry to `pyproject.toml`:

```toml
[project.entry-points."eval_learn.techniques"]
your_technique = "eval_learn.techniques.your_technique.wrapper:YourTechniqueClass"
```

If your technique uses a fixed diffusion backbone, also add it to
`src/eval_learn/techniques/_base_models.py`:

```python
TECHNIQUE_BASE_MODELS = {
    ...
    "your_technique": "CompVis/stable-diffusion-v1-4",
}
```

Then reinstall:

```bash
pip install -e .
```

### Validation checklist

The contribution notebook runs these tests automatically. All must pass:

- [ ] `config.py` defines a frozen `BaseConfig` subclass with `erase_concept` and `device` fields
- [ ] `Config` raises `ValueError` for empty `erase_concept`
- [ ] `Config.from_dict` / `to_dict` round-trips without loss
- [ ] `Config.from_dict` ignores unknown keys gracefully
- [ ] `wrapper.py` is importable (with the external package mocked)
- [ ] `generate(prompts, seed)` returns one `PIL.Image` per prompt
- [ ] `seed` is forwarded to the underlying pipeline
- [ ] `ImportError` (or `RuntimeError`) raised when the external package is missing
- [ ] Pipeline exceptions propagate out of `generate()` unchanged
- [ ] `@register_technique` registers the class in the local registry
- [ ] Entry point declared in `pyproject.toml` and resolves to the wrapper class

---

## Contributing a New Metric

**Contribution notebook:**
`demos/notebooks/contributing/tutorial_contributing_a_metric.ipynb`

!!! note "Notebook link"
    The notebook is currently available at the path above relative to the
    repository root. A rendered, browsable version will be linked here once the
    library is public on GitHub.

### Overview

Each metric lives under `src/eval_learn/metrics/` and implements a three-method
streaming interface that the runners call in order:

| Method | Called by runner | Purpose |
|---|---|---|
| `load_dataset() -> DataLoader` | Once, before generation | Return the DataLoader for this metric's dataset. Reset all accumulators. |
| `update(images, prompts, _metadata)` | Once per batch | Score images immediately and accumulate running totals. Do **not** store raw `PIL.Image` objects. |
| `compute() -> MetricResult` | Once, after all batches | Divide accumulators and return a `MetricResult`. Must be idempotent. |

### Required files

```
src/eval_learn/metrics/
└── your_metric/
    ├── __init__.py
    ├── config.py
    └── metric.py
```

### Accumulator convention

All metrics must track exactly these four instance attributes so the runners
can inspect progress:

```python
self._total_score      = 0.0   # running sum of scores
self._evaluated_count  = 0     # images successfully scored
self._total_count      = 0     # images seen (including failures)
self._per_image_scores = []    # float or None per image
```

### Registration

Add one entry to `pyproject.toml`:

```toml
[project.entry-points."eval_learn.metrics"]
your_metric = "eval_learn.metrics.your_metric.metric:YourMetricClass"
```

Optionally document your model in `src/eval_learn/metrics/_base_models.py`:

```python
METRIC_MODELS = {
    ...
    "your_metric": MetricModelInfo("your-model/id", configurable=False),
}
```

Then reinstall:

```bash
pip install -e .
```

### Validation checklist

The contribution notebook runs these tests automatically. All must pass:

- [ ] `config.py` defines a frozen `BaseConfig` subclass with a `device` field
- [ ] `Config.from_dict` / `to_dict` round-trips without loss
- [ ] `Config.from_dict` ignores unknown keys gracefully
- [ ] Metric class initialises with `self.device` and all four accumulator attributes at zero/empty
- [ ] Device auto-detects to `"cpu"` when `device=None` and CUDA is unavailable
- [ ] `update()` increments `_total_count` for every image (including failures)
- [ ] `update()` appends exactly one entry (float or `None`) per image to `_per_image_scores`
- [ ] `update()` does **not** store raw `PIL.Image` objects on the instance
- [ ] `update()` accepts `_metadata=None` and `_metadata=dict` without raising
- [ ] `update()` accumulates correctly across multiple calls
- [ ] `compute()` returns a `MetricResult` instance
- [ ] `compute()` is idempotent (safe to call multiple times)
- [ ] `compute()` returns `value=0.0` and `"error"` in `details` when `_total_count == 0`
- [ ] `compute()` averages over `_evaluated_count` only (not total)
- [ ] `compute()` `details` contains: `evaluated_count`, `total_count`, `per_image_scores`, `config`
- [ ] `@register_metric` registers the class in the local registry
- [ ] Entry point declared in `pyproject.toml` and resolves to the metric class

---

## Pull Request Guidelines

- Keep the PR focused: one technique or one metric per PR.
- Include a brief description of what the technique/metric does and a link to
  the original paper.
- All existing tests must still pass: `pytest -m "not integration" -v`
- Add at least one unit test for your config and one for your wrapper/metric
  under `tests/unit/techniques/` or `tests/unit/metrics/`.

---

## Questions

Open an issue on the GitHub repository if you run into problems or have
questions about the contribution process.
