"""Unit tests for Ring-A-Bell's auto-computed concept vectors."""
from unittest.mock import MagicMock

import numpy as np
import torch

from eval_unlearn.metrics.asr_ring_a_bell.concept_vector import compute_concept_vector


def _make_clip_stub(hidden_dim=8, max_length=6):
    """A CLIP model/processor stub whose text_model returns a fixed-shape,
    input-dependent last_hidden_state so we can verify the differencing math."""
    clip_processor = MagicMock()

    def processor_call(text, **kwargs):
        # Plain dict: supports the ["input_ids"] / .get("attention_mask")
        # access _encode_tokens uses, same as a real BatchEncoding would.
        return {
            "input_ids": torch.ones(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }

    clip_processor.side_effect = processor_call

    clip_model = MagicMock()

    def text_model_call(input_ids, attention_mask=None):
        # Make the hidden state depend deterministically on nothing but be
        # distinguishable per call via call_count, so pos != neg embeddings.
        call_idx = clip_model.text_model.call_count
        value = float(call_idx)
        out = MagicMock()
        out.last_hidden_state = torch.full((1, max_length, hidden_dim), value)
        return out

    clip_model.text_model = MagicMock(side_effect=text_model_call)
    return clip_model, clip_processor


class TestComputeConceptVector:
    def test_returns_expected_shape(self):
        clip_model, clip_processor = _make_clip_stub(hidden_dim=8, max_length=6)
        vector = compute_concept_vector("a fake concept", clip_model, clip_processor, "cpu", max_length=6)
        assert vector.shape == (6, 8)
        assert vector.dtype == np.float32

    def test_is_deterministic_for_same_inputs(self):
        clip_model, clip_processor = _make_clip_stub()
        v1 = compute_concept_vector("nudity", clip_model, clip_processor, "cpu", max_length=6)
        clip_model2, clip_processor2 = _make_clip_stub()
        v2 = compute_concept_vector("nudity", clip_model2, clip_processor2, "cpu", max_length=6)
        np.testing.assert_array_equal(v1, v2)

    def test_calls_text_model_for_every_template_pair(self):
        clip_model, clip_processor = _make_clip_stub()
        compute_concept_vector("nudity", clip_model, clip_processor, "cpu", max_length=6)
        # 8 positive + 8 negative templates defined in concept_vector.py
        assert clip_model.text_model.call_count == 16
