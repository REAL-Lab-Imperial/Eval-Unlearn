"""
Integration tests — real Q16 inference on GPU.

Covers:
  - asr_i2p/metric.py        (detector="q16" update path)
  - asr_mma_diffusion/metric (detector="q16" update path)
  - asr_p4d/metric.py        (detector="q16" update path)
  - asr_ring_a_bell/metric   (detector="q16" update path)
"""
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.integration


def _blank_image(color=(160, 160, 160)):
    return Image.new("RGB", (224, 224), color=color)


# ---------------------------------------------------------------------------
# Q16 classifier smoke test
# ---------------------------------------------------------------------------
class TestQ16ClassifierReal:
    def test_blank_image_is_non_inappropriate(self, q16_classifier):
        label, score = q16_classifier(_blank_image())
        assert label in ("non inappropriate", "inappropriate")
        assert 0.0 <= score <= 1.0

    def test_classify_batch_returns_correct_length(self, q16_classifier):
        imgs = [_blank_image(color=(c * 40, c * 40, c * 40)) for c in range(1, 5)]
        results = q16_classifier.classify_batch(imgs)
        assert len(results) == 4
        for label, score in results:
            assert label in ("non inappropriate", "inappropriate")
            assert 0.0 <= score <= 1.0

    def test_score_method_returns_float(self, q16_classifier):
        s = q16_classifier.score(_blank_image())
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# ASR I2P — Q16 update path
# ---------------------------------------------------------------------------
class TestASRI2PQ16Real:
    @pytest.fixture(scope="class")
    def metric(self, q16_classifier, device):
        with patch("eval_unlearn.metrics.asr_i2p.metric.Q16Classifier") as mock_cls:
            mock_cls.return_value = q16_classifier
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            m = ASRMetric(concept_name="violence", detector="q16", device=device, limit=5)
        m.q16_classifier = q16_classifier
        return m

    def test_blank_image_classified(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric._unsafe_indices = []
        metric.update([_blank_image()], ["a violent scene"])
        assert metric._total == 1

    def test_batch_of_images(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric._unsafe_indices = []
        imgs = [_blank_image() for _ in range(4)]
        metric.update(imgs, ["p"] * 4)
        assert metric._total == 4

    def test_compute_valid_ratio(self, metric):
        metric._total = 5
        result = metric.compute()
        assert 0.0 <= result.value <= 1.0


# ---------------------------------------------------------------------------
# ASR MMA Diffusion — Q16 update path
# ---------------------------------------------------------------------------
class TestMMADiffusionQ16Real:
    @pytest.fixture(scope="class")
    def metric(self, q16_classifier, device):
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier") as mock_cls:
            mock_cls.return_value = q16_classifier
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            m = MMADiffusionMetric(
                concept_name="violence",
                output_csv="/tmp/mma_q16_int.csv",
                detector="q16",
                device=device,
                target_prompts=["a violent scene", "graphic violence"],
            )
        m.q16_classifier = q16_classifier
        return m

    def test_blank_images_processed(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric._unsafe_indices = []
        imgs = [_blank_image() for _ in range(3)]
        metric.update(imgs, ["p"] * 3)
        assert metric._total == 3

    def test_non_pil_image_skipped(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric._unsafe_indices = []
        # Pass a numpy array — should still be handled
        import numpy as np
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        metric.update([arr], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# ASR P4D — Q16 update path
# ---------------------------------------------------------------------------
class TestASRP4DQ16Real:
    @pytest.fixture(scope="class")
    def metric(self, q16_classifier, device, tmp_path_factory):
        prompts_csv = tmp_path_factory.mktemp("p4d") / "prompts.csv"
        prompts_csv.write_text("adversarial_prompt,target_prompt\nadv1,violent scene\n")
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier") as mock_cls:
            mock_cls.return_value = q16_classifier
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            m = ASRP4D(
                concept_name="violence",
                detector="q16",
                erase_id="std",
                device=device,
                precomputed_prompts_path=str(prompts_csv),
            )
        m.q16_classifier = q16_classifier
        return m

    def test_blank_images_evaluated(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric._unsafe_indices = []
        metric.update([_blank_image(), _blank_image()], ["p1", "p2"])
        assert metric._total == 2


# ---------------------------------------------------------------------------
# ASR Ring-A-Bell — Q16 update path
# ---------------------------------------------------------------------------
class TestRingABellQ16Real:
    @pytest.fixture(scope="class")
    def metric(self, q16_classifier, clip_model_and_processor, device):
        clip_model, clip_proc = clip_model_and_processor
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier") as mock_q16, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_proc:
            mock_q16.return_value = q16_classifier
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            m = ASRRingABellMetric(
                concept_name="violence",
                detector="q16",
                seed_prompts_csv="/fake/s.csv",
                enable_discovery=False,
                device=device,
            )
        m.q16_classifier = q16_classifier
        return m

    def test_blank_image_classified(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric.update([_blank_image()], ["violent prompt"])
        assert metric._total == 1
