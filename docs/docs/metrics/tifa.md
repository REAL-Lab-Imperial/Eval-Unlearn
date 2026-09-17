# TIFA — Text-to-Image Faithfulness Assessment

## Overview

TIFA evaluates prompt faithfulness using Visual Question Answering (VQA) rather than
embedding similarity. For each generated image, an MPLUG VQA model is asked a set of
questions derived from the generation prompt, and the free-form answer is compared
against the expected answer. This mirrors the scoring methodology of the official
`tifascore` reference implementation.

For example, for the prompt "a red bicycle in a park", TIFA might ask:
- "What colour is the bicycle?" → expected: "red"
- "Where is the bicycle?" → expected: "park"

This QA-based approach captures fine-grained semantic faithfulness that embedding-based
metrics like CLIP Score can miss — particularly for attribute binding (colour, count,
spatial relationships).

**Dataset:** TIFA dataset (with pre-annotated QA pairs per prompt)
**VQA model:** MPLUG (`damo/mplug_visual-question-answering_coco_large_en` by default, via `modelscope`)

TIFA is concept-agnostic and compatible with all techniques.

---

## Compatible techniques

All techniques are compatible with TIFA. No concept restrictions.

---

## Configuration reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vqa_model_name` | `str` | `"damo/mplug_visual-question-answering_coco_large_en"` | ModelScope model ID for the MPLUG VQA model. |
| `device` | `str \| None` | `None` | Device for VQA inference. Auto-detects CUDA if `None`. |
| `limit` | `int \| None` | `200` | Maximum number of prompts from the TIFA dataset. |

---

## Output

| Key | Type | Description |
|-----|------|-------------|
| `value` | `float` | TIFA score: for each image, the fraction of its QA pairs answered correctly (per-image accuracy); the final score is the mean of these per-image accuracies across all evaluated images (macro-average, matching `tifascore.tifa_score_benchmark`'s `tifa_average`). Range [0, 1]. Higher is better. Typical SD baselines score 0.7–0.85. `0.0` if no images were evaluated. |
| `details.total_questions_count` | `int` | Total questions asked across all images. |
| `details.total_images_count` | `int` | Total images evaluated. |
| `details.per_image_scores` | `list[float \| None]` | Per-image accuracy in evaluation order. `None` for images that failed to load or had no valid QA pairs (excluded from the macro-average). |

---

## Warnings

!!! warning "No multiple-choice snapping"
    The official TIFA v1.0 benchmark ships per-question `choices`, and `tifascore`
    snaps the VQA model's free-form answer to the closest choice (via SBERT) before
    scoring. The TIFA dataset used here (`Unlearningltd/datasets`, `tifa/tifa_dataset.csv`)
    only carries `question`/`answer` pairs — no `choices` — so scoring is exact string
    match between the free-form answer and the expected answer, without choice-snapping.

!!! warning "modelscope dependency"
    The MPLUG VQA model is loaded via `modelscope`. Install with
    `pip install eval-unlearn[tifa]`.

!!! warning "qa_pairs metadata requirement"
    TIFA requires each batch to carry `qa_pairs` metadata — lists of
    `{"question": str, "answer": str}` dicts parallel to the images. The TIFA dataset
    provides these automatically. If you supply a custom dataset without `qa_pairs`,
    TIFA will fail with a `KeyError` on the metadata dict.

!!! warning "Slow evaluation"
    VQA inference is slow compared to CLIP-based metrics. With `limit=200`, expect
    TIFA to take several minutes on a GPU. Reduce `limit` for faster iteration.

---

## Examples

### Single metric

```json
{
  "output_dir": "results/esd_tifa",
  "technique": {
    "name": "esd",
    "config": {
      "erase_concept": "nudity",
      "train_method": "noxattn",
      "device": "cuda"
    }
  },
  "metric": {
    "name": "tifa",
    "config": {
      "vqa_model_name": "damo/mplug_visual-question-answering_coco_large_en",
      "device": "cuda",
      "limit": 200
    }
  }
}
```

### As part of a multi-metric run

```json
{
  "name": "tifa",
  "config": {
    "vqa_model_name": "damo/mplug_visual-question-answering_coco_large_en",
    "device": "cuda",
    "limit": 200
  }
}
```
