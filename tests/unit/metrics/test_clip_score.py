"""Unit tests for CLIP Score metric."""

import math
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch

from eval_unlearn.metrics.clip_score.metric import CLIPScoreMetric
from eval_unlearn.metrics.clip_score.config import CLIPScoreConfig
from eval_unlearn.types import MetricResult


def _embeddings_for_cosine(cos_val: float):
    """Build a pair of already-unit-norm embeddings with the given cosine similarity."""
    sin_val = math.sqrt(max(0.0, 1.0 - cos_val**2))
    image_features = torch.tensor([[1.0, 0.0]])
    text_features = torch.tensor([[cos_val, sin_val]])
    return image_features, text_features


class TestCLIPScoreConfig:
    """Test CLIPScoreConfig initialization and validation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = CLIPScoreConfig()
        assert config.clip_model_name == "openai/clip-vit-base-patch32"
        assert config.device is None
        assert config.limit == 300

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "clip_model_name": "openai/clip-vit-large-patch14",
            "device": "cuda",
            "limit": 100,
        }
        config = CLIPScoreConfig.from_dict(config_dict)
        assert config.clip_model_name == "openai/clip-vit-large-patch14"
        assert config.device == "cuda"
        assert config.limit == 100

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = CLIPScoreConfig(device="cpu", limit=50)
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["limit"] == 50


def _make_clip_metric(**kwargs):
    """Helper: create CLIPScoreMetric with mocked CLIP model/processor."""
    with patch("eval_unlearn.metrics.clip_score.metric.CLIPModel") as mock_model_cls, \
         patch("eval_unlearn.metrics.clip_score.metric.CLIPProcessor") as mock_proc_cls, \
         patch("eval_unlearn.metrics.clip_score.metric.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))

        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_proc = Mock()
        mock_proc_cls.from_pretrained.return_value = mock_proc

        metric = CLIPScoreMetric(**kwargs)

    return metric


class TestCLIPScoreMetricInitialization:
    """Test CLIPScoreMetric initialization."""

    def test_init_success_cpu(self):
        """Test successful initialization on CPU."""
        metric = _make_clip_metric(device="cpu")

        assert metric.device == "cpu"
        assert metric._total_score == 0.0
        assert metric._evaluated_count == 0
        assert metric._total_count == 0
        assert metric._per_image_scores == []

    def test_init_auto_detect_device_cpu(self):
        """Test device auto-detection defaults to CPU when CUDA unavailable."""
        metric = _make_clip_metric(device=None)
        assert metric.device == "cpu"


class TestCLIPScoreMetricUpdate:
    """Test update method for scoring images."""

    def test_update_with_valid_images(self):
        """Test update with valid image-prompt pairs."""
        metric = _make_clip_metric(device="cpu")

        from PIL import Image as PILImage

        # cosine similarity == 1.0 -> score 100.0
        image_features, text_features = _embeddings_for_cosine(1.0)
        mock_model = Mock()
        mock_model.get_image_features = Mock(return_value=image_features)
        mock_model.get_text_features = Mock(return_value=text_features)
        metric.model = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        metric.processor = Mock(return_value=mock_inputs)

        pil_img = PILImage.new("RGB", (4, 4))
        with patch.object(metric, "_load_image_pil", return_value=pil_img):
            metric.update([Mock(), Mock()], ["prompt1", "prompt2"])

        assert metric._total_count == 2
        assert metric._evaluated_count == 2
        assert metric._total_score == pytest.approx(200.0)  # 100.0 + 100.0
        assert len(metric._per_image_scores) == 2

    def test_update_with_failed_image_loading(self):
        """Test update when image loading fails."""
        metric = _make_clip_metric(device="cpu")

        with patch.object(metric, "_load_image_pil", return_value=None):
            metric.update([Mock()], ["prompt"])

        assert metric._total_count == 1
        assert metric._evaluated_count == 0  # Failed to load
        assert metric._per_image_scores[0] is None

    def test_update_multiple_batches_accumulate(self):
        """Test that update accumulates across multiple calls."""
        metric = _make_clip_metric(device="cpu")

        from PIL import Image as PILImage
        pil_img = PILImage.new("RGB", (4, 4))

        # image_features stays fixed at [1, 0]; text_features varies per call
        # to produce cosine similarities of 0.8 and 0.9 respectively.
        image_features = torch.tensor([[1.0, 0.0]])
        text_pairs = iter(
            [_embeddings_for_cosine(0.8)[1], _embeddings_for_cosine(0.9)[1]]
        )

        mock_model = Mock()
        mock_model.get_image_features = Mock(return_value=image_features)
        mock_model.get_text_features = Mock(side_effect=lambda *a, **k: next(text_pairs))
        metric.model = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        metric.processor = Mock(return_value=mock_inputs)

        with patch.object(metric, "_load_image_pil", return_value=pil_img):
            metric.update([Mock()], ["prompt1"])
            assert metric._total_count == 1

            metric.update([Mock()], ["prompt2"])
            assert metric._total_count == 2
            assert metric._evaluated_count == 2
            assert metric._total_score == pytest.approx(170.0)  # 80.0 + 90.0


class TestCLIPScoreMetricComputation:
    """Test compute method."""

    def test_compute_with_no_images(self):
        """Test compute with no images evaluated."""
        metric = _make_clip_metric(device="cpu")
        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_with_all_successful(self):
        """Test compute with all images scored successfully."""
        metric = _make_clip_metric(device="cpu")
        metric._total_count = 4
        metric._evaluated_count = 4
        metric._total_score = 3.2  # 0.8 * 4
        metric._per_image_scores = [0.8, 0.8, 0.8, 0.8]

        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == pytest.approx(0.8)
        assert result.details["evaluated_count"] == 4
        assert result.details["total_count"] == 4

    def test_compute_with_partial_success(self):
        """Test compute with some images failing to load."""
        metric = _make_clip_metric(device="cpu")
        metric._total_count = 5
        metric._evaluated_count = 3
        metric._total_score = 2.1  # 0.7 * 3
        metric._per_image_scores = [0.7, None, 0.7, 0.7, None]

        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == pytest.approx(0.7)
        assert result.details["evaluated_count"] == 3
        assert result.details["total_count"] == 5

    def test_compute_returns_metric_result(self):
        """Test that compute returns MetricResult instance."""
        metric = _make_clip_metric(device="cpu")
        metric._total_count = 1
        metric._evaluated_count = 1
        metric._total_score = 0.85
        metric._per_image_scores = [0.85]

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert isinstance(result.details, dict)
        assert "config" in result.details

    def test_compute_includes_per_image_scores(self):
        """Test that compute includes per-image scores in details."""
        metric = _make_clip_metric(device="cpu")
        metric._total_count = 3
        metric._evaluated_count = 3
        metric._total_score = 2.25
        metric._per_image_scores = [0.7, 0.75, 0.8]

        result = metric.compute()

        assert result.details["per_image_scores"] == [0.7, 0.75, 0.8]

    def test_compute_zero_evaluated_with_images(self):
        """Test compute when images exist but none were evaluated successfully."""
        metric = _make_clip_metric(device="cpu")
        metric._total_count = 3
        metric._evaluated_count = 0
        metric._total_score = 0.0
        metric._per_image_scores = [None, None, None]

        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == 0.0


class TestCLIPScoreMetricIntegration:
    """Integration tests for CLIP Score metric workflow."""

    def test_full_workflow(self):
        """Test complete workflow from init to compute."""
        metric = _make_clip_metric(device="cpu")

        from PIL import Image as PILImage
        pil_img = PILImage.new("RGB", (4, 4))

        image_features = torch.tensor([[1.0, 0.0]])
        text_pairs = iter(
            [
                _embeddings_for_cosine(0.75)[1],
                _embeddings_for_cosine(0.8)[1],
                _embeddings_for_cosine(0.85)[1],
            ]
        )

        mock_model = Mock()
        mock_model.get_image_features = Mock(return_value=image_features)
        mock_model.get_text_features = Mock(side_effect=lambda *a, **k: next(text_pairs))
        metric.model = mock_model

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        metric.processor = Mock(return_value=mock_inputs)

        with patch.object(metric, "_load_image_pil", return_value=pil_img):
            metric.update([Mock(), Mock(), Mock()], ["prompt1", "prompt2", "prompt3"])

        result = metric.compute()

        assert result.name == "CLIPScore"
        assert result.value == pytest.approx(80.0)  # (75 + 80 + 85) / 3
        assert result.details["evaluated_count"] == 3
        assert result.details["total_count"] == 3

    def test_config_included_in_result(self):
        """Test that config is included in result details."""
        metric = _make_clip_metric(device="cpu", clip_model_name="openai/clip-vit-large-patch14")
        metric._total_count = 1
        metric._evaluated_count = 1
        metric._total_score = 0.75
        metric._per_image_scores = [0.75]

        result = metric.compute()

        assert "config" in result.details
        assert result.details["config"]["clip_model_name"] == "openai/clip-vit-large-patch14"
