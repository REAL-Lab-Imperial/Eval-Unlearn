"""Unit tests for TIFA (Text-to-Image Faithfulness) metric."""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
from PIL import Image

from eval_unlearn.metrics.tifa.metric import TIFAMetric
from eval_unlearn.metrics.tifa.config import TIFAConfig
from eval_unlearn.types import MetricResult


class TestTIFAConfig:
    """Test TIFAConfig initialization and validation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = TIFAConfig()
        assert config.vqa_model_name == "damo/mplug_visual-question-answering_coco_large_en"
        assert config.device is None
        assert config.limit == 200

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vqa_model_name": "damo/mplug_visual-question-answering_base_en",
            "device": "cpu",
            "limit": 100,
        }
        config = TIFAConfig.from_dict(config_dict)
        assert config.vqa_model_name == "damo/mplug_visual-question-answering_base_en"
        assert config.device == "cpu"
        assert config.limit == 100

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TIFAConfig(device="cpu", limit=50)
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["limit"] == 50


def _make_tifa_metric(**kwargs):
    """Helper: create TIFAMetric with a mocked MPLUG modelscope pipeline."""
    with patch("eval_unlearn.metrics.tifa.metric.pipeline") as mock_pipeline_fn, \
         patch("eval_unlearn.metrics.tifa.metric.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False

        mock_pipeline_fn.return_value = Mock()

        metric = TIFAMetric(**kwargs)

    return metric


class TestTIFAMetricInitialization:
    """Test TIFAMetric initialization."""

    def test_init_success_cpu(self):
        """Test successful initialization on CPU."""
        metric = _make_tifa_metric(device="cpu")

        assert metric.device == "cpu"
        # Model is loaded eagerly, so the VQA pipeline should not be None
        assert metric._vqa_pipeline is not None
        assert metric._total_questions_count == 0
        assert metric._total_images_count == 0
        assert metric._per_image_scores == []

    def test_init_auto_detect_device(self):
        """Test device auto-detection when device is None."""
        metric = _make_tifa_metric(device=None)
        assert metric.device == "cpu"


class TestTIFAAnswerMethod:
    """Test the VQA answer method."""

    def test_answer_returns_string(self):
        """Test _answer returns a string."""
        metric = _make_tifa_metric(device="cpu")

        # Mock _answer directly to test its interface
        with patch.object(metric, "_answer", return_value="yes"):
            img = Image.new("RGB", (10, 10), color="red")
            answer = metric._answer(img, "Is this a dog?")
            assert isinstance(answer, str)
            assert answer == "yes"

    def test_answer_calls_pipeline_with_image_and_question(self):
        """Test _answer forwards image/question to the modelscope pipeline."""
        metric = _make_tifa_metric(device="cpu")
        metric._vqa_pipeline = Mock(return_value={"text": "yes"})

        img = Image.new("RGB", (10, 10), color="red")
        answer = metric._answer(img, "Is this red?")

        assert answer == "yes"
        metric._vqa_pipeline.assert_called_once_with({"image": img, "question": "Is this red?"})

    def test_answer_unwraps_list_response(self):
        """Test _answer unwraps a list-valued 'text' response."""
        metric = _make_tifa_metric(device="cpu")
        metric._vqa_pipeline = Mock(return_value={"text": ["yes"]})

        answer = metric._answer(Image.new("RGB", (10, 10)), "Q?")
        assert answer == "yes"


class TestTIFAMetricUpdate:
    """Test update() method."""

    def test_update_correct_answer(self):
        """Test update records a correct per-image score."""
        metric = _make_tifa_metric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        # Mock _answer to return correct answer
        with patch.object(metric, "_answer", return_value="yes"):
            metric.update([img], ["prompt"], metadata)

            assert metric._total_questions_count == 1
            assert metric._total_images_count == 1
            assert metric._per_image_scores[0] == 1.0

    def test_update_incorrect_answer(self):
        """Test update records an incorrect per-image score."""
        metric = _make_tifa_metric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        # Mock _answer to return incorrect answer
        with patch.object(metric, "_answer", return_value="no"):
            metric.update([img], ["prompt"], metadata)

            assert metric._total_questions_count == 1
            assert metric._per_image_scores[0] == 0.0

    def test_update_case_insensitive(self):
        """Test update does case-insensitive comparison."""
        metric = _make_tifa_metric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        # Mock _answer to return uppercase version
        with patch.object(metric, "_answer", return_value="YES"):
            metric.update([img], ["prompt"], metadata)

            assert metric._per_image_scores[0] == 1.0  # Should match despite case
            assert metric._total_questions_count == 1

    def test_update_multiple_qa_pairs(self):
        """Test update with multiple QA pairs per image averages within the image."""
        metric = _make_tifa_metric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [
                [
                    {"question": "Is this red?", "answer": "yes"},
                    {"question": "Is this blue?", "answer": "no"},
                    {"question": "What color?", "answer": "red"},
                ]
            ]
        }

        # Mock _answer to return different answers
        with patch.object(metric, "_answer", side_effect=["yes", "no", "red"]):
            metric.update([img], ["prompt"], metadata)

            assert metric._total_questions_count == 3
            assert metric._per_image_scores[0] == 1.0

    def test_update_skips_none_image(self):
        """Test update skips None images."""
        metric = _make_tifa_metric(device="cpu")

        metadata = {
            "qa_pairs": [
                [{"question": "Is this red?", "answer": "yes"}]
            ]
        }

        metric.update([None], ["prompt"], metadata)

        assert metric._total_questions_count == 0
        assert metric._per_image_scores[0] is None

    def test_update_skips_no_qa_pairs(self):
        """Test update skips images without QA pairs."""
        metric = _make_tifa_metric(device="cpu")

        img = Image.new("RGB", (10, 10), color="red")
        metadata = {
            "qa_pairs": [None]  # No QA pairs
        }

        metric.update([img], ["prompt"], metadata)

        assert metric._total_questions_count == 0
        assert metric._per_image_scores[0] is None

    def test_update_with_file_path(self):
        """Test update with file path."""
        metric = _make_tifa_metric(device="cpu")

        metadata = {
            "qa_pairs": [
                [{"question": "Question?", "answer": "yes"}]
            ]
        }

        # Use patch.object to mock _answer since file loading is complex
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/test.png"
            img = Image.new("RGB", (10, 10), color="red")
            img.save(img_path)

            with patch.object(metric, "_answer", return_value="yes"):
                metric.update([img_path], ["prompt"], metadata)

                assert metric._per_image_scores[0] == 1.0
                assert metric._total_questions_count == 1


class TestTIFAMetricComputation:
    """Test compute() method."""

    def test_compute_no_images(self):
        """Test compute returns 0.0 with no images."""
        metric = _make_tifa_metric(device="cpu")

        result = metric.compute()

        assert result.name == "TIFA"
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_perfect_score(self):
        """Test compute with perfect accuracy."""
        metric = _make_tifa_metric(device="cpu")

        metric._total_questions_count = 10
        metric._total_images_count = 2
        metric._per_image_scores = [1.0, 1.0]

        result = metric.compute()

        assert result.name == "TIFA"
        assert result.value == 1.0
        assert result.details["total_questions_count"] == 10

    def test_compute_macro_averages_per_image_scores(self):
        """Test compute averages per-image scores, not pooled questions."""
        metric = _make_tifa_metric(device="cpu")

        # Image 1: 1/1 correct (score 1.0); Image 2: 1/3 correct (score ~0.33)
        # A pooled/micro average would give 2/4 = 0.5; the macro average of
        # per-image scores gives (1.0 + 0.333...) / 2 = 0.667.
        metric._total_questions_count = 4
        metric._total_images_count = 2
        metric._per_image_scores = [1.0, 1 / 3]

        result = metric.compute()

        assert result.value == pytest.approx((1.0 + 1 / 3) / 2)

    def test_compute_ignores_none_per_image_scores(self):
        """Test compute excludes images with no valid QA pairs from the average."""
        metric = _make_tifa_metric(device="cpu")

        metric._total_questions_count = 2
        metric._total_images_count = 2
        metric._per_image_scores = [1.0, None]

        result = metric.compute()

        assert result.value == 1.0

    def test_compute_returns_metric_result(self):
        """Test that compute returns MetricResult instance."""
        metric = _make_tifa_metric(device="cpu")

        metric._total_questions_count = 10
        metric._total_images_count = 1
        metric._per_image_scores = [0.5]

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert isinstance(result.details, dict)
        assert "config" in result.details
        assert "per_image_scores" in result.details


class TestTIFAMetricIntegration:
    """Integration tests for TIFA metric workflow."""

    def test_full_workflow_update_compute(self):
        """Test complete workflow: initialize → update → compute."""
        metric = _make_tifa_metric(device="cpu")

        # Update with two images, each with 2 QA pairs
        imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(2)]
        metadata = {
            "qa_pairs": [
                [
                    {"question": "Is this red?", "answer": "yes"},
                    {"question": "What color?", "answer": "red"},
                ],
                [
                    {"question": "Is this red?", "answer": "yes"},
                    {"question": "What color?", "answer": "red"},
                ],
            ]
        }

        # Mock _answer: image 1 fully correct, image 2 fully incorrect
        with patch.object(metric, "_answer", side_effect=["yes", "red", "no", "blue"]):
            metric.update(imgs, ["prompt"] * 2, metadata)

            result = metric.compute()

            assert result.name == "TIFA"
            # Macro-average of per-image scores: (1.0 + 0.0) / 2 = 0.5
            assert result.value == pytest.approx(0.5)
            assert result.details["total_questions_count"] == 4
            assert result.details["total_images_count"] == 2
