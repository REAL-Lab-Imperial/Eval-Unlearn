"""Extended tests for CLIPScore metric targeting uncovered lines."""
import math
import pytest
import torch
from unittest.mock import MagicMock, patch
from PIL import Image
import tempfile
import os

from eval_unlearn.types import MetricResult


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _embeddings_for_cosine(cos_val: float):
    """Build a pair of already-unit-norm embeddings with the given cosine similarity."""
    sin_val = math.sqrt(max(0.0, 1.0 - cos_val**2))
    image_features = torch.tensor([[1.0, 0.0]])
    text_features = torch.tensor([[cos_val, sin_val]])
    return image_features, text_features


def _make_clip_score_metric(**kwargs):
    """Build CLIPScoreMetric with all deps mocked."""
    with patch("eval_unlearn.metrics.clip_score.metric.CLIPModel") as mock_cls, \
         patch("eval_unlearn.metrics.clip_score.metric.CLIPProcessor") as mock_proc_cls:
        mock_model = MagicMock()
        mock_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        from eval_unlearn.metrics.clip_score.metric import CLIPScoreMetric
        metric = CLIPScoreMetric(**kwargs)
    return metric


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------
class TestCLIPScoreLoadDataset:
    def test_load_dataset_resets_counters(self):
        metric = _make_clip_score_metric()
        metric._total_score = 99.0
        metric._evaluated_count = 10
        metric._total_count = 15
        metric._per_image_scores = [1.0, 2.0]

        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.coco_parquet.load_coco_captions", return_value=mock_loader):
            metric.load_dataset()

        assert metric._total_score == 0.0
        assert metric._evaluated_count == 0
        assert metric._total_count == 0
        assert metric._per_image_scores == []

    def test_load_dataset_returns_loader(self):
        metric = _make_clip_score_metric()
        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.coco_parquet.load_coco_captions", return_value=mock_loader):
            result = metric.load_dataset()
        assert result is mock_loader


# ---------------------------------------------------------------------------
# _load_image_pil
# ---------------------------------------------------------------------------
class TestCLIPScoreLoadImagePil:
    def test_pil_rgb_image_returned_as_is(self):
        metric = _make_clip_score_metric()
        img = _dummy_image()
        result = metric._load_image_pil(img)
        assert result is img

    def test_pil_non_rgb_converted(self):
        metric = _make_clip_score_metric()
        img = Image.new("RGBA", (16, 16), color=(50, 100, 150, 255))
        result = metric._load_image_pil(img)
        assert result.mode == "RGB"

    def test_file_path_loaded(self, tmp_path):
        metric = _make_clip_score_metric()
        p = tmp_path / "test.png"
        _dummy_image().save(str(p))
        result = metric._load_image_pil(str(p))
        assert isinstance(result, Image.Image)

    def test_missing_file_returns_none(self):
        metric = _make_clip_score_metric()
        result = metric._load_image_pil("/nonexistent/path/img.png")
        assert result is None

    def test_unsupported_type_returns_none(self):
        metric = _make_clip_score_metric()
        result = metric._load_image_pil(42)
        assert result is None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------
class TestCLIPScoreUpdate:
    def _metric_with_outputs(self, score=25.0):
        """Build a metric whose model yields embeddings producing the given
        final score (= 100 * cosine similarity)."""
        metric = _make_clip_score_metric()
        image_features, text_features = _embeddings_for_cosine(score / 100.0)
        mock_model = MagicMock()
        mock_model.get_image_features = MagicMock(return_value=image_features)
        mock_model.get_text_features = MagicMock(return_value=text_features)
        metric.model = mock_model
        mock_proc = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_proc.return_value = mock_inputs
        metric.processor = mock_proc
        return metric

    def test_update_accumulates_score(self):
        metric = self._metric_with_outputs(score=30.0)
        metric.update([_dummy_image()], ["a cat"])
        assert metric._evaluated_count == 1
        assert metric._total_count == 1
        assert abs(metric._total_score - 30.0) < 1e-6

    def test_update_none_image_skipped(self):
        metric = _make_clip_score_metric()
        # Pass a path to non-existent file -> _load_image_pil returns None
        metric.update(["/nonexistent/img.png"], ["prompt"])
        assert metric._total_count == 1
        assert metric._evaluated_count == 0
        assert metric._per_image_scores == [None]

    def test_update_model_error_handled(self):
        metric = _make_clip_score_metric()
        metric.model.get_image_features.side_effect = RuntimeError("GPU OOM")
        mock_proc = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_proc.return_value = mock_inputs
        metric.processor = mock_proc
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total_count == 1
        assert metric._evaluated_count == 0
        assert None in metric._per_image_scores

    def test_update_multiple_images(self):
        metric = self._metric_with_outputs(score=20.0)
        metric.update(
            [_dummy_image(), _dummy_image()],
            ["cat", "dog"]
        )
        assert metric._total_count == 2
        assert metric._evaluated_count == 2

    def test_update_with_file_path(self, tmp_path):
        metric = self._metric_with_outputs(score=15.0)
        p = tmp_path / "img.png"
        _dummy_image().save(str(p))
        metric.update([str(p)], ["a cat"])
        assert metric._evaluated_count == 1


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class TestCLIPScoreCompute:
    def test_compute_zero_total_returns_zero(self):
        metric = _make_clip_score_metric()
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_average_score(self):
        metric = _make_clip_score_metric()
        metric._total_score = 60.0
        metric._evaluated_count = 3
        metric._total_count = 3
        result = metric.compute()
        assert abs(result.value - 20.0) < 1e-6

    def test_compute_with_none_scores(self):
        metric = _make_clip_score_metric()
        metric._total_score = 40.0
        metric._evaluated_count = 2
        metric._total_count = 3
        metric._per_image_scores = [20.0, 20.0, None]
        result = metric.compute()
        assert abs(result.value - 20.0) < 1e-6
        assert result.details["evaluated_count"] == 2
        assert result.details["total_count"] == 3

    def test_compute_details_include_per_image(self):
        metric = _make_clip_score_metric()
        metric._total_score = 10.0
        metric._evaluated_count = 1
        metric._total_count = 1
        metric._per_image_scores = [10.0]
        result = metric.compute()
        assert result.details["per_image_scores"] == [10.0]

    def test_compute_name_is_clipscore(self):
        metric = _make_clip_score_metric()
        metric._total_count = 1
        metric._evaluated_count = 1
        metric._total_score = 5.0
        result = metric.compute()
        assert result.name == "CLIPScore"
