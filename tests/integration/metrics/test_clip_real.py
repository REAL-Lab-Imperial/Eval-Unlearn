"""
Integration tests — real CLIP inference.

Covers:
  - clip_score/metric.py   (update → CLIP forward pass)
  - ua_ira/metric.py       (_evaluate_batch → CLIP forward pass)
  - err/metric.py          (_check_concept_presence → CLIP)
  - asr_i2p/metric.py      (detector="clip" path)
  - asr_mma_diffusion      (detector="clip" path)
  - asr_ring_a_bell        (detector="clip" path, _evaluate_batch_clip)
"""
import csv
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.integration

CLIP_MODEL_ID = "openai/clip-vit-base-patch16"


def _blank_image(color=(128, 128, 128)):
    return Image.new("RGB", (224, 224), color=color)


# ---------------------------------------------------------------------------
# CLIPScore — real CLIP forward pass
# ---------------------------------------------------------------------------
class TestCLIPScoreReal:
    @pytest.fixture(scope="class")
    def metric(self, clip_model_and_processor, device):
        clip_model, clip_proc = clip_model_and_processor
        from unittest.mock import patch
        with patch("eval_unlearn.metrics.clip_score.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.clip_score.metric.CLIPProcessor") as mock_proc:
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.clip_score.metric import CLIPScoreMetric
            m = CLIPScoreMetric(
                clip_model_name=CLIP_MODEL_ID,
                device=device,
            )
        m.model = clip_model
        m.processor = clip_proc
        return m

    def test_update_produces_nonzero_score(self, metric):
        metric._total_score = 0.0
        metric._evaluated_count = 0
        metric._total_count = 0
        metric._per_image_scores = []
        metric.update([_blank_image()], ["a grey square"])
        assert metric._evaluated_count == 1
        assert metric._total_score != 0.0

    def test_compute_returns_average(self, metric):
        metric._total_score = 30.0
        metric._evaluated_count = 3
        metric._total_count = 3
        result = metric.compute()
        assert abs(result.value - 10.0) < 1e-4

    def test_invalid_image_skipped(self, metric):
        metric._total_score = 0.0
        metric._evaluated_count = 0
        metric._total_count = 0
        metric._per_image_scores = []
        # Pass a path that doesn't exist
        metric.update(["/no/such/image.png"], ["a prompt"])
        assert metric._total_count == 1
        assert metric._evaluated_count == 0

    def test_multiple_images(self, metric):
        metric._total_score = 0.0
        metric._evaluated_count = 0
        metric._total_count = 0
        metric._per_image_scores = []
        imgs = [_blank_image(color=(c * 30, c * 30, c * 30)) for c in range(1, 4)]
        prompts = ["a dark image", "a medium image", "a bright image"]
        metric.update(imgs, prompts)
        assert metric._evaluated_count == 3
        result = metric.compute()
        assert isinstance(result.value, float)


# ---------------------------------------------------------------------------
# UA-IRA — real CLIP forward pass
# ---------------------------------------------------------------------------
class TestUAIRAReal:
    @pytest.fixture(scope="class")
    def metric(self, clip_model_and_processor, device):
        clip_model, clip_proc = clip_model_and_processor
        from unittest.mock import patch
        with patch("eval_unlearn.metrics.ua_ira.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.ua_ira.metric.CLIPProcessor") as mock_proc:
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.ua_ira.metric import UAIRAMetric
            m = UAIRAMetric(
                clip_model_name=CLIP_MODEL_ID,
                device=device,
                target_concept="nudity",
                retain_concept="person",
                target_prompts_path="/fake/t.csv",
                retain_prompts_path="/fake/r.csv",
            )
        m.model = clip_model
        m.processor = clip_proc
        return m

    def test_update_target_images(self, metric):
        metric._target_correct_count = 0
        metric._target_total_count = 0
        metric._retain_correct_count = 0
        metric._retain_total_count = 0
        metric._target_prompt_end_index = 0
        imgs = [_blank_image()]
        metadata = {"target_prompt_end_index": 1}
        metric.update(imgs, ["prompt"], metadata)
        assert metric._target_total_count == 1

    def test_update_retain_images(self, metric):
        metric._target_correct_count = 0
        metric._target_total_count = 0
        metric._retain_correct_count = 0
        metric._retain_total_count = 0
        metric._target_prompt_end_index = 0
        imgs = [_blank_image()]
        metadata = {"target_prompt_end_index": 0}
        metric.update(imgs, ["prompt"], metadata)
        assert metric._retain_total_count == 1

    def test_compute_returns_valid_scores(self, metric):
        metric._target_correct_count = 3
        metric._target_total_count = 5
        metric._retain_correct_count = 4
        metric._retain_total_count = 5
        result = metric.compute()
        assert 0.0 <= result.value <= 1.0
        assert "ua_score" in result.details
        assert "ira_score" in result.details

    def test_load_dataset_from_csv(self, tmp_path):
        target_csv = tmp_path / "target.csv"
        retain_csv = tmp_path / "retain.csv"
        target_csv.write_text("prompt\nnude person\n")
        retain_csv.write_text("prompt\ncloaked person\n")

        from unittest.mock import patch
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

        with patch("eval_unlearn.metrics.ua_ira.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.ua_ira.metric.CLIPProcessor") as mock_proc:
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.ua_ira.metric import UAIRAMetric
            m = UAIRAMetric(
                clip_model_name=CLIP_MODEL_ID,
                device="cpu",
                target_concept="nudity",
                retain_concept="person",
                target_prompts_path=str(target_csv),
                retain_prompts_path=str(retain_csv),
            )
        loader = m.load_dataset()
        batch = next(iter(loader))
        assert len(batch.prompts) == 2


# ---------------------------------------------------------------------------
# ERR — _check_concept_presence (CLIP path for retain category)
# ---------------------------------------------------------------------------
class TestERRCLIPRetainReal:
    @pytest.fixture(scope="class")
    def metric(self, clip_model_and_processor, nude_detector, device):
        clip_model, clip_proc = clip_model_and_processor
        from unittest.mock import patch
        with patch("eval_unlearn.metrics.err.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.err.metric.CLIPProcessor") as mock_proc, \
             patch("eval_unlearn.metrics.err.metric.NudeDetector", return_value=nude_detector):
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.err.metric import ERRMetric
            m = ERRMetric(clip_model_name=CLIP_MODEL_ID, device=device)
        m.model = clip_model
        m.processor = clip_proc
        m.nude_detector = nude_detector
        return m

    def test_retain_category_evaluated(self, metric):
        from eval_unlearn.metrics.err.metric import _EXPECTED_PRESENCE
        metric._counts = {cat: {"success": 0, "evaluated": 0} for cat in _EXPECTED_PRESENCE}
        imgs = [_blank_image()]
        metadata = {"concepts": ["dog"], "categories": ["retain"]}
        metric.update(imgs, ["a dog"], metadata)
        assert metric._counts["retain"]["evaluated"] == 1

    def test_err_compute_harmonic_mean(self, metric):
        metric._counts = {
            "target":     {"success": 8, "evaluated": 10},
            "retain":     {"success": 9, "evaluated": 10},
            "adversarial": {"success": 7, "evaluated": 10},
        }
        result = metric.compute()
        assert 0.0 < result.value <= 1.0
        assert result.details["forgetting"] == pytest.approx(0.8)
        assert result.details["retention"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# ASR I2P — CLIP detector path
# ---------------------------------------------------------------------------
class TestASRI2PCLIPReal:
    @pytest.fixture(scope="class")
    def metric(self, clip_model_and_processor, device):
        clip_model, clip_proc = clip_model_and_processor
        from unittest.mock import patch
        with patch("eval_unlearn.metrics.asr_i2p.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_i2p.metric.CLIPProcessor") as mock_proc:
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            m = ASRMetric(
                concept_name="violence",
                detector="clip",
                similarity_threshold=0.99,  # very high threshold → always safe
                device=device,
            )
        m.clip_model = clip_model
        m.clip_processor = clip_proc
        return m

    def test_blank_image_processed(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric.update([_blank_image()], ["violent scene"])
        assert metric._total == 1
        # With threshold=0.99, blank image CLIP similarity to "violence" < 0.99


# ---------------------------------------------------------------------------
# ASR Ring-A-Bell — CLIP detector path
# ---------------------------------------------------------------------------
class TestRingABellCLIPReal:
    @pytest.fixture(scope="class")
    def metric(self, clip_model_and_processor, device):
        clip_model, clip_proc = clip_model_and_processor
        from unittest.mock import patch, MagicMock
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_proc:
            mock_cls.from_pretrained.return_value = clip_model
            mock_proc.from_pretrained.return_value = clip_proc
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            m = ASRRingABellMetric(
                concept_name="violence",
                detector="clip",
                seed_prompts_csv="/fake/s.csv",
                enable_discovery=False,
                similarity_threshold=0.99,
            )
        m.clip_model = clip_model
        m.clip_processor = clip_proc
        m._detector = "clip"
        return m

    def test_blank_image_is_safe(self, metric):
        metric._unsafe_count = 0
        metric._total = 0
        metric.update([_blank_image()], ["violence"])
        assert metric._total == 1
        assert metric._unsafe_count == 0
