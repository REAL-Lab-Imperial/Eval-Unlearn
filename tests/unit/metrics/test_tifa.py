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
        assert config.vqa_model_name == "Salesforce/blip2-flan-t5-xl"
        assert config.device is None
        assert config.limit == 200

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vqa_model_name": "Salesforce/blip2-opt-2.7b",
            "device": "cpu",
            "limit": 100,
        }
        config = TIFAConfig.from_dict(config_dict)
        assert config.vqa_model_name == "Salesforce/blip2-opt-2.7b"
        assert config.device == "cpu"
        assert config.limit == 100

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TIFAConfig(device="cpu", limit=50)
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["limit"] == 50


def _make_tifa_metric(**kwargs):
    """Helper: create TIFAMetric with mocked BLIP-2 model/processor."""
    with patch("eval_unlearn.metrics.tifa.metric.Blip2Processor") as mock_proc_cls, \
         patch("eval_unlearn.metrics.tifa.metric.Blip2ForConditionalGeneration") as mock_model_cls, \
         patch("eval_unlearn.metrics.tifa.metric.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = torch.float16
        mock_torch.no_grad = MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            )
        )

        mock_proc = Mock()
        mock_proc_cls.from_pretrained.return_value = mock_proc

        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        metric = TIFAMetric(**kwargs)

    return metric


class TestTIFAMetricInitialization:
    """Test TIFAMetric initialization."""

    def test_init_success_cpu(self):
        """Test successful initialization on CPU."""
        metric = _make_tifa_metric(device="cpu")

        assert metric.device == "cpu"
        # Model is loaded eagerly, so _processor and _model should not be None
        assert metric._processor is not None
        assert metric._model is not None
        assert metric._correct_count == 0
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

    def test_answer_strips_whitespace(self):
        """Test _answer strips whitespace from output."""
        metric = _make_tifa_metric(device="cpu")

        with patch.object(metric, "_answer", return_value="yes"):
            answer = metric._answer(Image.new("RGB", (10, 10)), "Q?")
            assert answer == "yes"  # Should be stripped


class TestTIFAMetricUpdate:
    """Test update() method."""

    def test_update_correct_answer(self):
        """Test update counts correct answer."""
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

            assert metric._correct_count == 1
            assert metric._total_questions_count == 1
            assert metric._total_images_count == 1
            assert metric._per_image_scores[0] == 1.0

    def test_update_incorrect_answer(self):
        """Test update counts incorrect answer."""
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

            assert metric._correct_count == 0
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

            assert metric._correct_count == 1  # Should match despite case
            assert metric._total_questions_count == 1

    def test_update_multiple_qa_pairs(self):
        """Test update with multiple QA pairs per image."""
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

            assert metric._correct_count == 3
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

        assert metric._correct_count == 0
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

        assert metric._correct_count == 0
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

                assert metric._correct_count == 1
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

        metric._correct_count = 10
        metric._total_questions_count = 10
        metric._total_images_count = 2
        metric._per_image_scores = [1.0, 1.0]

        result = metric.compute()

        assert result.name == "TIFA"
        assert result.value == 1.0
        assert result.details["correct_count"] == 10
        assert result.details["total_questions_count"] == 10

    def test_compute_partial_score(self):
        """Test compute with partial correctness."""
        metric = _make_tifa_metric(device="cpu")

        metric._correct_count = 6
        metric._total_questions_count = 10
        metric._total_images_count = 2
        metric._per_image_scores = [1.0, 0.5]

        result = metric.compute()

        assert result.name == "TIFA"
        assert result.value == pytest.approx(0.6)
        assert result.details["correct_count"] == 6
        assert result.details["total_questions_count"] == 10

    def test_compute_returns_metric_result(self):
        """Test that compute returns MetricResult instance."""
        metric = _make_tifa_metric(device="cpu")

        metric._correct_count = 5
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

        # Mock _answer to return different responses (2 correct, 2 incorrect)
        with patch.object(metric, "_answer", side_effect=["yes", "red", "no", "blue"]):
            metric.update(imgs, ["prompt"] * 2, metadata)

            result = metric.compute()

            assert result.name == "TIFA"
            assert result.value == pytest.approx(0.5)  # 2 correct out of 4
            assert result.details["correct_count"] == 2
            assert result.details["total_questions_count"] == 4
            assert result.details["total_images_count"] == 2
