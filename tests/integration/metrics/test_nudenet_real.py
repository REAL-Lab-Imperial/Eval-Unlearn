"""
Integration tests — real NudeNet inference.

Covers the NudeNet detection branches in:
  - asr_i2p/metric.py   (_is_unsafe_nudenet, update nudenet path)
  - asr_mma_diffusion/metric.py (same pattern)
  - asr_p4d/metric.py   (same pattern)
  - asr_ring_a_bell/metric.py (same pattern)
  - err/metric.py        (_check_adversarial)
"""
import os
import tempfile
import pytest
from PIL import Image

pytestmark = pytest.mark.integration


def _blank_image(size=(256, 256), color=(180, 160, 140)):
    return Image.new("RGB", size, color)


# ---------------------------------------------------------------------------
# ASR I2P — NudeNet update path
# ---------------------------------------------------------------------------
class TestASRI2PNudenetReal:
    @pytest.fixture(scope="class")
    def metric(self, nude_detector):
        from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
        m = ASRMetric(concept_name="nudity", detector="nudenet", limit=5)
        m.nude_detector = nude_detector
        return m

    def test_blank_image_is_safe(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric.update([_blank_image()], ["a prompt"])
        assert metric._total == 1
        assert metric._unsafe_count == 0

    def test_multiple_safe_images(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        imgs = [_blank_image(color=(c, c, c)) for c in (100, 150, 200)]
        metric.update(imgs, ["p1", "p2", "p3"])
        assert metric._total == 3
        assert metric._unsafe_count == 0

    def test_compute_after_update(self, metric):
        metric._unsafe_count = 0
        metric._total = 5
        from eval_unlearn.types import MetricResult
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert result.value == 0.0
        assert result.details["total_images"] == 5

    def test_pil_image_saved_to_temp_file(self, metric):
        """Confirm real NudeNet can process PIL images via tempfile path."""
        metric._unsafe_count = 0
        metric._total = 0
        img = _blank_image()
        metric.update([img], ["prompt"])
        assert metric._total == 1  # no crash = real inference ran


# ---------------------------------------------------------------------------
# ASR MMA Diffusion — NudeNet update path
# ---------------------------------------------------------------------------
class TestMMADiffusionNudenetReal:
    @pytest.fixture(scope="class")
    def metric(self, nude_detector):
        from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
        m = MMADiffusionMetric(
            concept_name="nudity",
            output_csv="/tmp/mma_int_test.csv",
            detector="nudenet",
        )
        m.nude_detector = nude_detector
        return m

    def test_safe_image_not_counted(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric._unsafe_indices = []
        metric.update([_blank_image()], ["adv prompt"])
        assert metric._total == 1
        assert metric._unsafe_count == 0

    def test_compute_ratio(self, metric):
        metric._unsafe_count = 2
        metric._total = 10
        result = metric.compute()
        assert abs(result.value - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# ASR P4D — NudeNet update path
# ---------------------------------------------------------------------------
class TestASRP4DNudenetReal:
    @pytest.fixture(scope="class")
    def metric(self, nude_detector, tmp_path_factory):
        prompts_csv = tmp_path_factory.mktemp("p4d_nd") / "prompts.csv"
        prompts_csv.write_text("adversarial_prompt,target_prompt\nadv1,naked person\n")
        from unittest.mock import patch, MagicMock
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()):
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            m = ASRP4D(
                concept_name="nudity",
                detector="nudenet",
                erase_id="std",
                precomputed_prompts_path=str(prompts_csv),
            )
        m.nude_detector = nude_detector
        return m

    def test_blank_image_safe(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric._unsafe_indices = []
        metric.update([_blank_image()], ["prompt"])
        assert metric._total == 1
        assert metric._unsafe_count == 0


# ---------------------------------------------------------------------------
# ASR Ring-A-Bell — NudeNet update path
# ---------------------------------------------------------------------------
class TestRingABellNudenetReal:
    @pytest.fixture(scope="class")
    def metric(self, nude_detector, clip_model_and_processor):
        from unittest.mock import patch, MagicMock
        clip_model, clip_proc = clip_model_and_processor
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_proc:
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            m = ASRRingABellMetric(
                concept_name="nudity",
                detector="nudenet",
                seed_prompts_csv="/fake/s.csv",
                enable_discovery=False,
            )
        m.nude_detector = nude_detector
        return m

    def test_blank_image_safe(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric.update([_blank_image()], ["prompt"])
        assert metric._total == 1
        assert metric._unsafe_count == 0


# ---------------------------------------------------------------------------
# ERR — _check_adversarial (NudeNet path)
# ---------------------------------------------------------------------------
class TestERRNudenetReal:
    @pytest.fixture(scope="class")
    def metric(self, nude_detector, clip_model_and_processor, device):
        clip_model, clip_proc = clip_model_and_processor
        from unittest.mock import patch
        with patch("eval_unlearn.metrics.err.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.err.metric.CLIPProcessor") as mock_proc, \
             patch("eval_unlearn.metrics.err.metric.NudeDetector") as mock_nd:
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            mock_nd.return_value = nude_detector
            from eval_unlearn.metrics.err.metric import ERRMetric
            m = ERRMetric(
                clip_model_name="openai/clip-vit-base-patch16",
                device=device,
            )
        m.nude_detector = nude_detector
        m.model = clip_model
        m.processor = clip_proc
        return m

    def test_adversarial_blank_image_not_unsafe(self, metric):
        from eval_unlearn.metrics.err.metric import _EXPECTED_PRESENCE
        metric._counts = {cat: {"success": 0, "evaluated": 0} for cat in _EXPECTED_PRESENCE}
        imgs = [_blank_image()]
        metadata = {
            "concepts": ["nudity"],
            "categories": ["adversarial"],
        }
        metric.update(imgs, ["prompt"], metadata)
        # Blank image: NudeNet returns [] → not unsafe → expected=False → success
        assert metric._counts["adversarial"]["evaluated"] == 1
        assert metric._counts["adversarial"]["success"] == 1  # correctly NOT flagged as unsafe
