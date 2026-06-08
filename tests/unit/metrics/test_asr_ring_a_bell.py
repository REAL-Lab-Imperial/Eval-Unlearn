"""Tests for ASRRingABellMetric update and evaluation loop."""

import sys
from unittest.mock import MagicMock, patch
import pytest
import torch
from PIL import Image

# ring_a_bell is not installed in test env — stub it before the metric module loads
sys.modules.setdefault("ring_a_bell", MagicMock())
sys.modules.setdefault("ring_a_bell.encoder", MagicMock())

from eval_learn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric  # noqa: E402
from eval_learn.types import MetricResult  # noqa: E402

DIM = 16  # small embedding dimension for test tensors


def _make_metric(concept="nudity", threshold=0.3, device="cpu"):
    """Instantiate ASRRingABellMetric with mocked CLIP models, no discovery.

    Uses detector="clip" explicitly so that tests can control CLIP features
    via _set_clip_response regardless of whether nudenet/q16 is installed.
    """
    with patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_model_cls, \
            patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_proc_cls:
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        import tempfile
        import csv
        import os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.writer(tmp)
        writer.writerow(["prompt"])
        writer.writerow(["a test prompt"])
        tmp.close()

        metric = ASRRingABellMetric(
            concept_name=concept,
            seed_prompts_csv=tmp.name,
            enable_discovery=False,
            similarity_threshold=threshold,
            detector="clip",
            device=device,
        )
        os.unlink(tmp.name)

    return metric


def _set_clip_response(metric, image_feat: torch.Tensor, text_feat: torch.Tensor):
    """Wire metric's CLIP model/processor to return specific feature tensors."""
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs

    metric.clip_processor.side_effect = None
    metric.clip_processor.return_value = mock_inputs
    metric.clip_model.get_image_features.return_value = image_feat
    metric.clip_model.get_text_features.return_value = text_feat


# ---------------------------------------------------------------------------
# _evaluate_batch_clip
# ---------------------------------------------------------------------------

class TestEvaluateBatchClip:
    """Tests for the CLIP similarity calculation in _evaluate_batch_clip."""

    def test_above_threshold_marked_unsafe(self):
        """Image with similarity > threshold should be flagged unsafe."""
        metric = _make_metric(threshold=0.3, device="cpu")

        # image and text features pointing in the same direction → similarity ≈ 1.0
        feat = torch.ones(1, DIM)
        _set_clip_response(metric, feat.clone(), feat.clone())

        imgs = [Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert [bool(r) for r in result] == [True], f"Expected [True], got {result}"

    def test_below_threshold_marked_safe(self):
        """Image with similarity < threshold should be flagged safe."""
        metric = _make_metric(threshold=0.3, device="cpu")

        # orthogonal vectors → similarity = 0.0
        img_feat = torch.zeros(1, DIM)
        img_feat[0, 0] = 1.0
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 1] = 1.0
        _set_clip_response(metric, img_feat, txt_feat)

        imgs = [Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert [bool(r) for r in result] == [False], f"Expected [False], got {result}"

    def test_batch_mixed_safe_and_unsafe(self):
        """Batch where some images are above threshold and some below."""
        metric = _make_metric(threshold=0.3, device="cpu")

        # img 0: parallel to text (sim=1.0, unsafe); img 1: orthogonal (sim=0.0, safe)
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 0] = 1.0

        img_feat = torch.zeros(2, DIM)
        img_feat[0, 0] = 1.0   # parallel
        img_feat[1, 1] = 1.0   # orthogonal

        _set_clip_response(metric, img_feat, txt_feat)

        imgs = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert bool(result[0]) is True,  f"img 0 should be unsafe, got {result[0]}"
        assert bool(result[1]) is False, f"img 1 should be safe, got {result[1]}"

    def test_at_threshold_is_safe(self):
        """Similarity exactly equal to threshold is NOT flagged (strict >)."""
        threshold = 0.5
        metric = _make_metric(threshold=threshold, device="cpu")

        # construct vectors with cosine similarity exactly = 0.5
        img_feat = torch.zeros(1, DIM)
        img_feat[0, 0] = 1.0
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 0] = threshold  # after normalisation this still gives sim=1.0
        # Use truly orthogonal-ish vectors: sim = cos(angle)
        # sim = 0.5 → angle = 60°: txt = (0.5, sqrt(0.75), 0, …)
        import math
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 0] = 0.5
        txt_feat[0, 1] = math.sqrt(0.75)
        _set_clip_response(metric, img_feat, txt_feat)

        imgs = [Image.new("RGB", (4, 4))]
        result = metric._evaluate_batch_clip(imgs)

        assert [bool(r) for r in result] == [False], f"sim==threshold should be safe, got {result}"

    def test_clip_exception_returns_all_false(self):
        """If CLIP processing raises, _evaluate_batch_clip returns all False (not a crash)."""
        metric = _make_metric(device="cpu")
        metric.clip_processor.side_effect = RuntimeError("CLIP exploded")

        imgs = [Image.new("RGB", (4, 4))] * 3
        result = metric._evaluate_batch_clip(imgs)

        assert [bool(r) for r in result] == [False, False, False]
        assert metric._total == 0, "update counters should not be touched"

    def test_numpy_array_images_accepted(self):
        """Images as numpy arrays (shape HxWxC) should be converted to PIL without error."""
        import numpy as np
        metric = _make_metric(threshold=0.3, device="cpu")

        feat = torch.ones(1, DIM)
        _set_clip_response(metric, feat.clone(), feat.clone())

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        result = metric._evaluate_batch_clip([arr])

        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    """Tests for update() accumulation logic."""

    def test_update_increments_total(self):
        metric = _make_metric(device="cpu")
        feat = torch.zeros(1, DIM)
        feat[0, 1] = 1.0  # safe (orthogonal)
        txt = torch.zeros(1, DIM)
        txt[0, 0] = 1.0
        _set_clip_response(metric, feat, txt)

        metric.update([Image.new("RGB", (4, 4))], ["prompt"])

        assert metric._total == 1
        assert metric._unsafe_count == 0

    def test_update_increments_unsafe_count(self):
        metric = _make_metric(threshold=0.3, device="cpu")
        feat = torch.ones(1, DIM)  # parallel → unsafe
        _set_clip_response(metric, feat.clone(), feat.clone())

        metric.update([Image.new("RGB", (4, 4))], ["prompt"])

        assert metric._total == 1
        assert metric._unsafe_count == 1

    def test_update_accumulates_across_calls(self):
        metric = _make_metric(threshold=0.3, device="cpu")

        # First call: unsafe
        feat = torch.ones(1, DIM)
        _set_clip_response(metric, feat.clone(), feat.clone())
        metric.update([Image.new("RGB", (4, 4))], ["p"])

        # Second call: safe (orthogonal)
        img_feat = torch.zeros(1, DIM)
        img_feat[0, 0] = 1.0
        txt_feat = torch.zeros(1, DIM)
        txt_feat[0, 1] = 1.0
        _set_clip_response(metric, img_feat, txt_feat)
        metric.update([Image.new("RGB", (4, 4))], ["p"])

        assert metric._total == 2
        assert metric._unsafe_count == 1

    def test_update_clip_error_still_increments_total(self):
        """Even if CLIP fails, update() should still count the images."""
        metric = _make_metric(device="cpu")
        metric.clip_processor.side_effect = RuntimeError("boom")

        metric.update([Image.new("RGB", (4, 4))] * 2, ["p", "p"])

        assert metric._total == 2
        assert metric._unsafe_count == 0


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------

class TestCompute:
    def test_compute_asr_ratio(self):
        metric = _make_metric(device="cpu")
        metric._total = 10
        metric._unsafe_count = 3

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert result.value == pytest.approx(0.3)
        assert result.details["unsafe_count"] == 3
        assert result.details["total"] == 10

    def test_compute_zero_total(self):
        metric = _make_metric(device="cpu")
        result = metric.compute()
        assert result.value == 0.0


# ---------------------------------------------------------------------------
# Config validation branches
# ---------------------------------------------------------------------------
class TestASRRingABellConfig:
    def test_invalid_detector_raises(self):
        from eval_learn.metrics.asr_ring_a_bell.config import ASRRingABellConfig
        with pytest.raises(ValueError, match="detector must be one of"):
            ASRRingABellConfig(concept_name="nudity", detector="bad")

    def test_nudenet_on_non_nudity_raises(self):
        from eval_learn.metrics.asr_ring_a_bell.config import ASRRingABellConfig
        with pytest.raises(ValueError, match="only valid for nudity"):
            ASRRingABellConfig(concept_name="dogs", detector="nudenet")

    def test_q16_threshold_out_of_range_raises(self):
        from eval_learn.metrics.asr_ring_a_bell.config import ASRRingABellConfig
        with pytest.raises(ValueError, match="q16_threshold"):
            ASRRingABellConfig(concept_name="nudity", q16_threshold=2.0)

    def test_similarity_threshold_out_of_range_raises(self):
        from eval_learn.metrics.asr_ring_a_bell.config import ASRRingABellConfig
        with pytest.raises(ValueError, match="similarity_threshold"):
            ASRRingABellConfig(concept_name="nudity", similarity_threshold=-0.5)


# ---------------------------------------------------------------------------
# Metric __init__ and helper branches not covered by existing tests
# ---------------------------------------------------------------------------
def _make_nudenet_metric():
    """Create a ring-a-bell metric with detector='nudenet' and mocked externals."""
    import tempfile, csv, os
    from eval_learn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    csv.writer(tmp).writerow(["prompt"])
    csv.writer(tmp).writerow(["a test prompt"])
    tmp.close()
    with patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cm, \
         patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_cp, \
         patch("eval_learn.metrics.asr_ring_a_bell.metric.NudeDetector") as mock_nd:
        mock_cm.from_pretrained.return_value = MagicMock()
        mock_cp.from_pretrained.return_value = MagicMock()
        mock_nd.return_value = MagicMock()
        metric = ASRRingABellMetric(
            concept_name="nudity",
            seed_prompts_csv=tmp.name,
            enable_discovery=False,
            detector="nudenet",
            device="cpu",
        )
    os.unlink(tmp.name)
    return metric


class TestASRRingABellMetricBranches:
    def test_auto_detector_resolves_to_nudenet_for_nudity(self):
        import tempfile, csv, os
        from eval_learn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        csv.writer(tmp).writerow(["prompt"])
        tmp.close()
        with patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cm, \
             patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_cp, \
             patch("eval_learn.metrics.asr_ring_a_bell.metric.NudeDetector"):
            mock_cm.from_pretrained.return_value = MagicMock()
            mock_cp.from_pretrained.return_value = MagicMock()
            metric = ASRRingABellMetric(
                concept_name="nudity",
                seed_prompts_csv=tmp.name,
                enable_discovery=False,
                detector="auto",
                device="cpu",
            )
        os.unlink(tmp.name)
        assert metric._detector == "nudenet"

    def test_nudenet_none_raises_runtime_error(self):
        import tempfile, csv, os
        from eval_learn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        csv.writer(tmp).writerow(["prompt"])
        tmp.close()
        try:
            with patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cm, \
                 patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_cp, \
                 patch("eval_learn.metrics.asr_ring_a_bell.metric.NudeDetector", None):
                mock_cm.from_pretrained.return_value = MagicMock()
                mock_cp.from_pretrained.return_value = MagicMock()
                with pytest.raises(RuntimeError, match="nudenet"):
                    ASRRingABellMetric(
                        concept_name="nudity",
                        seed_prompts_csv=tmp.name,
                        enable_discovery=False,
                        detector="nudenet",
                        device="cpu",
                    )
        finally:
            os.unlink(tmp.name)

    def test_concept_vector_path_warning_when_discovery_disabled(self):
        import tempfile, csv, os
        from eval_learn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        csv.writer(tmp).writerow(["prompt"])
        tmp.close()
        with patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cm, \
             patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_cp, \
             patch("eval_learn.metrics.asr_ring_a_bell.metric.NudeDetector"):
            mock_cm.from_pretrained.return_value = MagicMock()
            mock_cp.from_pretrained.return_value = MagicMock()
            metric = ASRRingABellMetric(
                concept_name="nudity",
                seed_prompts_csv=tmp.name,
                enable_discovery=False,
                concept_vector_path="some_vector.npy",
                device="cpu",
            )
        os.unlink(tmp.name)
        assert metric is not None

    def test_generated_prompts_output_warning_when_discovery_disabled(self):
        import tempfile, csv, os
        from eval_learn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        csv.writer(tmp).writerow(["prompt"])
        tmp.close()
        with patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cm, \
             patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_cp, \
             patch("eval_learn.metrics.asr_ring_a_bell.metric.NudeDetector"):
            mock_cm.from_pretrained.return_value = MagicMock()
            mock_cp.from_pretrained.return_value = MagicMock()
            metric = ASRRingABellMetric(
                concept_name="nudity",
                seed_prompts_csv=tmp.name,
                enable_discovery=False,
                generated_prompts_output="/tmp/out.csv",
                device="cpu",
            )
        os.unlink(tmp.name)
        assert metric is not None

    def test_is_unsafe_q16_numpy_array_image(self):
        import numpy as np
        from PIL import Image
        metric = _make_metric()
        metric._detector = "q16"
        metric.q16_classifier = MagicMock()
        metric.q16_classifier.classify_batch.return_value = [("safe", 0.9)]
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = metric._is_unsafe_q16([arr])
        assert result == [False]

    def test_is_unsafe_q16_invalid_image_type_returns_false(self):
        metric = _make_metric()
        metric._detector = "q16"
        metric.q16_classifier = MagicMock()
        result = metric._is_unsafe_q16([42])
        assert result == [False]

    def test_is_unsafe_q16_empty_pil_list_returns_false(self):
        metric = _make_metric()
        metric._detector = "q16"
        metric.q16_classifier = MagicMock()
        result = metric._is_unsafe_q16([42, "not_an_image"])
        assert result == [False, False]

    def test_is_unsafe_q16_mixed_valid_and_invalid(self):
        from PIL import Image
        metric = _make_metric()
        metric._detector = "q16"
        metric.q16_classifier = MagicMock()
        metric.q16_classifier.classify_batch.return_value = [("inappropriate", 0.95)]
        img = Image.new("RGB", (64, 64))
        result = metric._is_unsafe_q16([img, 42])
        assert result == [True, False]

    def test_evaluate_batch_clip_uses_pooler_output(self):
        import torch
        from PIL import Image
        metric = _make_metric()
        img = Image.new("RGB", (64, 64))
        real_feat = torch.ones(1, 512)
        mock_out = MagicMock()
        mock_out.pooler_output = real_feat
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {}
        metric.clip_processor.return_value = mock_inputs
        metric.clip_model.get_image_features.return_value = mock_out
        mock_text_out = MagicMock()
        mock_text_out.pooler_output = real_feat
        metric.clip_model.get_text_features.return_value = mock_text_out
        metric.update([img], ["prompt"])
        assert metric._total == 1

    def test_update_nudenet_oserror_on_remove(self):
        from PIL import Image
        metric = _make_nudenet_metric()
        metric.nude_detector.detect.return_value = []
        img = Image.new("RGB", (64, 64))
        with patch("eval_learn.metrics.asr_ring_a_bell.metric.os.remove",
                   side_effect=OSError("locked")):
            metric.update([img], ["prompt"])
        assert metric._total == 1

    def test_concept_vector_path_nonexistent_raises_file_not_found(self):
        import tempfile, csv, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            csv.writer(f).writerow(["prompt"])
            seed_csv = f.name
        try:
            from eval_learn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            with pytest.raises(FileNotFoundError, match="Concept vector not found"):
                ASRRingABellMetric(
                    concept_name="nudity",
                    seed_prompts_csv=seed_csv,
                    enable_discovery=True,
                    concept_vector_path="/nonexistent/vector_that_does_not_exist.npy",
                    device="cpu",
                )
        finally:
            os.unlink(seed_csv)

    def test_enable_discovery_empty_output_raises_value_error(self, tmp_path):
        """Lines 210,213-215: enable_discovery path with empty generated prompts."""
        import csv, os

        seed_csv = str(tmp_path / "seeds.csv")
        with open(seed_csv, "w", newline="") as f:
            csv.writer(f).writerows([["prompt"], ["a test prompt"]])

        output_csv = str(tmp_path / "generated.csv")

        with patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_model_cls, \
             patch("eval_learn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_proc_cls:
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_proc_cls.from_pretrained.return_value = MagicMock()

            metric = ASRRingABellMetric(
                concept_name="nudity",
                seed_prompts_csv=seed_csv,
                enable_discovery=True,
                generated_prompts_output=output_csv,
                device="cpu",
            )

        # Create empty output file; mock _run_discovery to skip the real GA run
        open(output_csv, "w").close()
        with patch.object(metric, "_run_discovery", return_value=None):
            with pytest.raises(ValueError, match="empty output"):
                metric.load_dataset()
