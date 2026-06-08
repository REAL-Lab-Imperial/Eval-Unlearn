"""
Integration tests — real InceptionV3 FID feature extraction.

Covers:
  - fid/metric.py  (_extract_features, update, compute)
"""
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock

pytestmark = pytest.mark.integration


def _random_image(seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((299, 299, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# FID feature extraction
# ---------------------------------------------------------------------------
class TestFIDReal:
    @pytest.fixture(scope="class")
    def metric(self, device):
        from eval_learn.metrics.fid.metric import FIDMetric
        return FIDMetric(device=device, batch_size=4)

    def test_extract_features_shape(self, metric):
        imgs = [_random_image(i) for i in range(4)]
        features = metric._extract_features(imgs)
        assert features.shape == (4, 2048)

    def test_update_accumulates_features(self, metric):
        metric._gen_activations = []
        imgs = [_random_image(i) for i in range(3)]
        metric.update(imgs, ["p1", "p2", "p3"])
        assert len(metric._gen_activations) == 1
        assert metric._gen_activations[0].shape == (3, 2048)

    def test_update_multiple_batches(self, metric):
        metric._gen_activations = []
        for batch_idx in range(3):
            imgs = [_random_image(batch_idx * 10 + i) for i in range(2)]
            metric.update(imgs, ["p"] * 2)
        total_features = np.concatenate(metric._gen_activations, axis=0)
        assert total_features.shape == (6, 2048)

    def test_compute_fid_with_real_features(self, metric):
        n = 20
        metric._gen_activations = []
        # Real images for generated distribution
        gen_imgs = [_random_image(i) for i in range(n)]
        metric.update(gen_imgs, ["p"] * n)
        # Set real activations to a slightly different distribution
        rng = np.random.default_rng(999)
        metric._real_activations = (rng.random((n, 2048)) * 0.5).astype(np.float32)
        metric._real_count = n
        result = metric.compute()
        assert isinstance(result.value, float)
        assert result.value >= 0.0
        assert result.details["total_generated"] == n
        assert result.details["total_real"] == n

    def test_compute_no_real_activations_raises(self, device):
        from eval_learn.metrics.fid.metric import FIDMetric
        m = FIDMetric(device=device, batch_size=4)
        m._real_activations = None
        with pytest.raises(RuntimeError, match="load_dataset"):
            m.compute()

    def test_compute_no_generated_images(self, metric):
        metric._gen_activations = []
        rng = np.random.default_rng(0)
        metric._real_activations = rng.random((10, 2048)).astype(np.float32)
        metric._real_count = 10
        result = metric.compute()
        assert result.value == float("inf")
        assert "error" in result.details

    def test_compute_single_generated_image(self, metric):
        metric._gen_activations = []
        imgs = [_random_image(0)]
        metric.update(imgs, ["p"])
        rng = np.random.default_rng(0)
        metric._real_activations = rng.random((10, 2048)).astype(np.float32)
        metric._real_count = 10
        result = metric.compute()
        assert result.value == float("inf")

    def test_update_from_file_path(self, tmp_path, metric):
        metric._gen_activations = []
        img = _random_image(42)
        img_path = str(tmp_path / "test_img.png")
        img.save(img_path)
        metric.update([img_path], ["prompt from path"])
        assert len(metric._gen_activations) == 1

    def test_empty_update_no_crash(self, metric):
        metric._gen_activations = []
        metric.update([], [])
        assert metric._gen_activations == []
