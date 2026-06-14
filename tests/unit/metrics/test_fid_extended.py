"""Extended tests for FID metric targeting uncovered lines."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import torch

from eval_unlearn.types import MetricResult


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _make_fid_metric(**kwargs):
    """Build FIDMetric with mocked inception model."""
    with patch("eval_unlearn.metrics.fid.metric._load_inception") as mock_load:
        mock_model = MagicMock()
        mock_model.return_value = torch.ones(1, 2048)
        mock_load.return_value = mock_model
        from eval_unlearn.metrics.fid.metric import FIDMetric
        metric = FIDMetric(device="cpu", **kwargs)
    return metric


def _fake_inception_call(batch):
    """Mock inception forward pass: returns (batch_size, 2048) tensor."""
    return torch.ones(batch.shape[0], 2048)


# ---------------------------------------------------------------------------
# load_dataset — (lines 110-160)
# ---------------------------------------------------------------------------
class TestFIDLoadDataset:
    def _make_mock_coco_batch(self, n=2, captions=None):
        batch = MagicMock()
        batch.metadata = {"images": torch.ones(n, 3, 299, 299)}
        batch.prompts = captions or [f"caption{i}" for i in range(n)]
        return batch

    def _run_load_dataset(self, metric, mock_batches, inception_side_effect=None):
        """Run load_dataset with patched coco loader and inception."""
        inception_fn = inception_side_effect or (lambda b: torch.ones(b.shape[0], 2048))
        mock_model = MagicMock()
        mock_model.side_effect = inception_fn
        with patch("eval_unlearn.datasets.coco_parquet.load_coco_parquet", return_value=mock_batches), \
             patch.object(metric, "_get_model", return_value=mock_model):
            return metric.load_dataset()

    def test_load_dataset_resets_gen_activations(self):
        metric = _make_fid_metric()
        metric._gen_activations = [np.ones((5, 2048))]

        mock_batches = [self._make_mock_coco_batch(n=2)]
        self._run_load_dataset(metric, mock_batches)

        assert metric._gen_activations == []

    def test_load_dataset_extracts_real_features(self):
        metric = _make_fid_metric()
        mock_batches = [self._make_mock_coco_batch(n=3, captions=["c1", "c2", "c3"])]

        self._run_load_dataset(metric, mock_batches)

        assert metric._real_activations is not None
        assert metric._real_count == 3

    def test_load_dataset_returns_caption_dataloader(self):
        from torch.utils.data import DataLoader
        metric = _make_fid_metric()
        mock_batches = [self._make_mock_coco_batch(n=2)]

        loader = self._run_load_dataset(metric, mock_batches)

        assert isinstance(loader, DataLoader)

    def test_load_dataset_multiple_batches(self):
        metric = _make_fid_metric()

        b1 = self._make_mock_coco_batch(n=2, captions=["c1", "c2"])
        b2 = self._make_mock_coco_batch(n=3, captions=["c3", "c4", "c5"])

        call_results = [torch.ones(2, 2048), torch.ones(3, 2048)]
        it = iter(call_results)
        self._run_load_dataset(metric, [b1, b2], inception_side_effect=lambda b: next(it))

        assert metric._real_count == 5

    def test_caption_dataset_collate_fn(self):
        metric = _make_fid_metric()
        mock_batches = [self._make_mock_coco_batch(n=1, captions=["a cat"])]

        loader = self._run_load_dataset(metric, mock_batches)

        batches = list(loader)
        assert len(batches) >= 1
        from eval_unlearn.types import Dataset
        assert isinstance(batches[0], Dataset)
        assert "a cat" in batches[0].prompts


# ---------------------------------------------------------------------------
# _extract_features (lines 86-99)
# ---------------------------------------------------------------------------
class TestFIDExtractFeatures:
    def test_extract_features_returns_numpy(self):
        metric = _make_fid_metric()
        # Use patch.object on the model's forward pass
        with patch.object(metric, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.return_value = torch.ones(2, 2048)
            mock_get_model.return_value = mock_model
            result = metric._extract_features([_dummy_image(), _dummy_image()])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2048)

    def test_extract_features_batches_correctly(self):
        metric = _make_fid_metric(batch_size=2)
        # 3 images with batch_size=2 => 2 forward passes
        call_count = []

        with patch.object(metric, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            def model_call(batch):
                call_count.append(batch.shape[0])
                return torch.ones(batch.shape[0], 2048)
            mock_model.side_effect = model_call
            mock_get_model.return_value = mock_model
            result = metric._extract_features([_dummy_image()] * 3)

        assert len(call_count) == 2  # batch of 2 + batch of 1
        assert result.shape == (3, 2048)

    def test_extract_features_single_image(self):
        metric = _make_fid_metric()
        with patch.object(metric, "_get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.return_value = torch.ones(1, 2048)
            mock_get_model.return_value = mock_model
            result = metric._extract_features([_dummy_image()])
        assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# _calculate_fid numerical edge cases + _get_model
# ---------------------------------------------------------------------------
class TestFIDCalculateEdgeCases:
    def test_get_model_returns_inception(self):
        metric = _make_fid_metric()
        assert metric._get_model() is metric._inception_model

    def test_calculate_fid_numerical_instability_uses_offset(self):
        from eval_unlearn.metrics.fid.metric import _calculate_fid
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1e-300, 0.0], [0.0, 1e-300]])
        result = _calculate_fid(mu, sigma, mu, sigma)
        assert np.isfinite(result)

    def test_calculate_fid_complex_covmean_near_real_uses_real_part(self):
        from eval_unlearn.metrics.fid.metric import _calculate_fid
        complex_mat = np.array([[1.0 + 1e-10j, 0.0], [0.0, 1.0 + 1e-10j]])
        with patch("eval_unlearn.metrics.fid.metric.linalg.sqrtm",
                   return_value=(complex_mat, None)):
            mu = np.array([0.0, 0.0])
            result = _calculate_fid(mu, np.eye(2), mu, np.eye(2))
        assert isinstance(result, float)

    def test_calculate_fid_complex_large_imag_returns_inf(self):
        from eval_unlearn.metrics.fid.metric import _calculate_fid
        complex_mat = np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0 + 1.0j]])
        with patch("eval_unlearn.metrics.fid.metric.linalg.sqrtm",
                   return_value=(complex_mat, None)):
            mu = np.array([0.0, 0.0])
            result = _calculate_fid(mu, np.eye(2), mu, np.eye(2))
        assert result == float("inf")
