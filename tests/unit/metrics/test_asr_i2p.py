"""Unit tests for ASR i2p metric (asr_i2p)."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from eval_unlearn.types import MetricResult


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _make_asr_metric(detector="nudenet", concept="nudity", **kwargs):
    """Build ASRMetric with all external deps mocked."""
    mock_nude = MagicMock()
    mock_q16 = MagicMock()

    with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", mock_nude), \
         patch("eval_unlearn.metrics.asr_i2p.metric.Q16Classifier", mock_q16), \
         patch("eval_unlearn.metrics.asr_i2p.metric.CLIPModel", MagicMock()), \
         patch("eval_unlearn.metrics.asr_i2p.metric.CLIPProcessor", MagicMock()):
        from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
        metric = ASRMetric(
            concept_name=concept,
            detector=detector,
            **kwargs
        )
    return metric


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestASRConfig:
    def test_non_i2p_concept_accepted(self):
        """Concepts outside I2P's categories are accepted by the config —
        ASRMetric reports N/A for them rather than the config raising."""
        from eval_unlearn.metrics.asr_i2p.config import ASRConfig
        config = ASRConfig(concept_name="bad_concept")
        assert config.concept_name == "bad_concept"

    def test_invalid_detector_raises(self):
        from eval_unlearn.metrics.asr_i2p.config import ASRConfig
        with pytest.raises(ValueError, match="detector must be one of"):
            ASRConfig(concept_name="nudity", detector="bad")

    def test_nudenet_on_non_nudity_raises(self):
        from eval_unlearn.metrics.asr_i2p.config import ASRConfig
        with pytest.raises(ValueError, match="only valid for nudity"):
            ASRConfig(concept_name="violence", detector="nudenet")

    def test_valid_config(self):
        from eval_unlearn.metrics.asr_i2p.config import ASRConfig
        cfg = ASRConfig(concept_name="nudity", detector="nudenet")
        assert cfg.concept_name == "nudity"
        assert cfg.detector == "nudenet"


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------
class TestASRLoadDataset:
    def test_load_dataset_resets_counters(self):
        metric = _make_asr_metric(detector="nudenet")
        metric._unsafe_count = 5
        metric._total = 10
        metric._unsafe_indices = [0, 1]

        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=mock_loader), \
             patch("eval_unlearn.metrics.asr_i2p.metric.load_i2p_csv", return_value=mock_loader, create=True):
            metric.load_dataset()

        assert metric._unsafe_count == 0
        assert metric._total == 0
        assert metric._unsafe_indices == []

    def test_load_dataset_returns_loader(self):
        metric = _make_asr_metric(detector="nudenet")
        from torch.utils.data import DataLoader
        mock_loader = MagicMock(spec=DataLoader)

        # The import happens inside the method, so patch the source module
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=mock_loader):
            result = metric.load_dataset()

        assert result is mock_loader
        assert metric._unsafe_count == 0
        assert metric._total == 0

    def test_load_dataset_calls_load_i2p_csv_with_correct_args(self):
        metric = _make_asr_metric(detector="nudenet", concept="nudity", limit=50)

        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=mock_loader) as mock_fn:
            metric.load_dataset()

        mock_fn.assert_called_once_with(concept="nudity", limit=50)


# ---------------------------------------------------------------------------
# update — NudeNet path
# ---------------------------------------------------------------------------
class TestASRUpdateNudenet:
    def test_safe_image_not_counted(self):
        metric = _make_asr_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1
        assert metric._unsafe_count == 0

    def test_unsafe_image_counted(self):
        metric = _make_asr_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9}
        ]
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_detection_error_handled(self):
        metric = _make_asr_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.side_effect = RuntimeError("model error")
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1

    def test_file_path_image(self):
        metric = _make_asr_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        with patch("os.path.exists", return_value=True):
            metric.update(["/tmp/test_img.png"], ["prompt"])
        assert metric._total == 1

    def test_pil_image_saved_to_temp(self):
        metric = _make_asr_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = [
            {"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.9}
        ]
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), \
             patch("os.path.exists", side_effect=lambda p: True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["p"])
        assert metric._unsafe_count == 1

    def test_multiple_images(self):
        metric = _make_asr_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        # First safe, second unsafe
        metric.nude_detector.detect.side_effect = [
            [],
            [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.9}]
        ]
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image(), _dummy_image()], ["p1", "p2"])
        assert metric._total == 2
        assert metric._unsafe_count == 1


# ---------------------------------------------------------------------------
# update — Q16 path
# ---------------------------------------------------------------------------
class TestASRUpdateQ16:
    def _q16_metric(self, label="non inappropriate"):
        with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_i2p.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [(label, 0.9)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            m = ASRMetric(concept_name="violence", detector="q16")
        return m

    def test_appropriate_not_counted(self):
        metric = self._q16_metric("non inappropriate")
        metric.update([_dummy_image()], ["p"])
        assert metric._unsafe_count == 0

    def test_inappropriate_counted(self):
        metric = self._q16_metric("inappropriate")
        metric.update([_dummy_image()], ["p"])
        assert metric._unsafe_count == 1

    def test_q16_exception_handled(self):
        metric = self._q16_metric()
        metric.q16_classifier.classify_batch.side_effect = RuntimeError("OOM")
        metric.update([_dummy_image()], ["p"])
        assert metric._total == 1

    def test_q16_with_array_image(self):
        import numpy as np
        metric = self._q16_metric("non inappropriate")
        arr_img = MagicMock()
        arr_img.shape = (224, 224, 3)  # has shape attr -> treated as numpy
        metric.update([arr_img], ["p"])
        assert metric._total == 1

    def test_q16_invalid_image_skipped(self):
        metric = self._q16_metric("non inappropriate")
        # Object with no shape and not PIL
        metric.update([42], ["p"])
        assert metric._total == 1

    def test_vlm_auto_detector_violence(self):
        with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics._vlm_detector.pipeline") as mock_pipeline_fn:
            mock_pipeline_fn.return_value = MagicMock()
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            m = ASRMetric(concept_name="violence", detector="auto")
        assert m._detector == "vlm"
        assert m.vlm_detector is not None

    def test_q16_explicit_detector_violence(self):
        with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_i2p.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [("non inappropriate", 0.1)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            m = ASRMetric(concept_name="violence", detector="q16")
        assert m._detector == "q16"

    def test_q16_auto_detector_nudity(self):
        with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_i2p.metric.Q16Classifier", MagicMock()):
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            m = ASRMetric(concept_name="nudity", detector="auto")
        assert m._detector == "nudenet"


# ---------------------------------------------------------------------------
# update — CLIP path
# ---------------------------------------------------------------------------
class TestASRUpdateCLIP:
    def _make_clip_metric(self, sim_value=0.5):
        import torch
        with patch("eval_unlearn.metrics.asr_i2p.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_i2p.metric.CLIPProcessor") as mock_proc:
            mock_model = MagicMock()
            mock_model.get_image_features.return_value = torch.ones(1, 4)
            mock_model.get_text_features.return_value = torch.ones(1, 4)
            mock_cls.from_pretrained.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_proc.from_pretrained.return_value = MagicMock()
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            metric = ASRMetric(
                concept_name="violence", detector="clip",
                similarity_threshold=sim_value
            )
        metric.clip_model = mock_model
        # Make processor return a mock with .to() method
        mock_proc_inst = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"pixel_values": torch.ones(1, 3, 224, 224)}
        mock_proc_inst.return_value = mock_inputs
        metric.clip_processor = mock_proc_inst
        return metric

    def test_high_similarity_unsafe(self):
        import torch
        metric = self._make_clip_metric(sim_value=0.0)
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_clip_error_handled(self):
        metric = self._make_clip_metric()
        metric.clip_model.get_image_features.side_effect = RuntimeError("error")
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1

    def test_no_valid_images_returns_false(self):
        metric = self._make_clip_metric()
        # Pass invalid image (not PIL, no shape)
        metric.update([None], ["prompt"])
        # Should handle gracefully
        assert metric._total == 1

    def test_array_image_clip(self):
        metric = self._make_clip_metric(sim_value=0.0)
        # PIL images work fine
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class TestASRCompute:
    def test_compute_ratio(self):
        metric = _make_asr_metric(detector="nudenet")
        metric._unsafe_count = 3
        metric._total = 10
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert abs(result.value - 0.3) < 1e-6

    def test_compute_zero_total(self):
        metric = _make_asr_metric(detector="nudenet")
        result = metric.compute()
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_details(self):
        metric = _make_asr_metric(detector="nudenet")
        metric._unsafe_count = 1
        metric._total = 2
        metric._unsafe_indices = [0]
        result = metric.compute()
        assert result.details["unsafe_indices"] == [0]
        assert result.details["total_images"] == 2
        assert result.details["unsafe_count"] == 1

    def test_compute_includes_concept(self):
        metric = _make_asr_metric(detector="nudenet", concept="nudity")
        metric._unsafe_count = 1
        metric._total = 1
        result = metric.compute()
        assert result.details["concept"] == "nudity"

    def test_compute_name_is_asr(self):
        metric = _make_asr_metric(detector="nudenet")
        metric._unsafe_count = 0
        metric._total = 5
        result = metric.compute()
        assert result.name == "ASR"


# ---------------------------------------------------------------------------
# Coverage gap tests for asr_i2p helper methods
# ---------------------------------------------------------------------------
class TestASRI2PCoverageGaps:
    def _make_q16_metric(self):
        with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_i2p.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [("inappropriate", 0.9)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            m = ASRMetric(concept_name="violence", detector="q16")
        return m

    def _make_clip_metric(self):
        import torch
        with patch("eval_unlearn.metrics.asr_i2p.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_i2p.metric.CLIPProcessor") as mock_proc:
            mock_model = MagicMock()
            mock_cls.from_pretrained.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_proc.from_pretrained.return_value = MagicMock()
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            metric = ASRMetric(concept_name="violence", detector="clip")
        real_feat = torch.ones(1, 4)
        metric.clip_model = MagicMock()
        metric.clip_model.get_image_features.return_value = real_feat
        metric.clip_model.get_text_features.return_value = real_feat
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {}
        metric.clip_processor = MagicMock(return_value=mock_inputs)
        return metric

    def test_q16_numpy_array_covers_branch(self):
        """Lines 160: numpy branch in _is_unsafe_q16 with proper array."""
        import numpy as np
        metric = self._make_q16_metric()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = metric._is_unsafe_q16([arr])
        assert result == [True]

    def test_q16_mixed_valid_invalid_covers_false_branch(self):
        """Line 177: results.append(False) for invalid image in mixed batch."""
        metric = self._make_q16_metric()
        img = _dummy_image()
        result = metric._is_unsafe_q16([img, 42])
        assert result == [True, False]

    def test_clip_numpy_array_covers_branch(self):
        """Lines 193-194: numpy branch in _is_unsafe_clip."""
        import numpy as np
        metric = self._make_clip_metric()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = metric._is_unsafe_clip([arr])
        assert isinstance(result, list)

    def test_clip_mixed_valid_invalid_covers_false_branch(self):
        """Line 222: results.append(False) for invalid in mixed clip batch."""
        metric = self._make_clip_metric()
        img = _dummy_image()
        result = metric._is_unsafe_clip([img, 42])
        assert isinstance(result, list)
        assert result[1] is False

    def test_clip_none_raises_runtime_error(self):
        """Line 103: RuntimeError when CLIPModel is None and detector='clip'."""
        with patch("eval_unlearn.metrics.asr_i2p.metric.CLIPModel", None), \
             patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()):
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            with pytest.raises(RuntimeError, match="transformers"):
                ASRMetric(concept_name="violence", detector="clip")

    def test_nudenet_update_oserror_on_remove(self):
        """Lines 266-267: OSError during temp file cleanup is silently ignored."""
        metric = _make_asr_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake_i2p.png")), \
             patch("os.close"), \
             patch("eval_unlearn.metrics.asr_i2p.metric.os.path.exists", return_value=True), \
             patch("eval_unlearn.metrics.asr_i2p.metric.os.remove",
                   side_effect=OSError("locked")):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1

    def test_q16_unknown_clip_model_fallback_warning(self):
        """Line 94: warning when clip_model_id not in _HF_TO_Q16."""
        with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_i2p.metric.Q16Classifier") as mock_q16_cls:
            mock_q16_cls.return_value = MagicMock()
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            metric = ASRMetric(
                concept_name="violence",
                detector="q16",
                clip_model_id="openai/clip-vit-large-patch14-336",
            )
        assert metric.q16_classifier is not None


# ---------------------------------------------------------------------------
# Concepts with no I2P category mapping -> N/A
# ---------------------------------------------------------------------------
class TestASRI2PNotApplicable:
    """ASR-I2P is only defined for I2P's 7 categories. For any other concept,
    no detector should be initialised, no images generated, and compute()
    should report the score as N/A (value=None)."""

    def _make_na_metric(self, concept="a custom unrelated concept", **kwargs):
        # No detector mocks patched in: if the metric tried to initialise a
        # real detector for an NA concept, this would fail/crash.
        from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
        return ASRMetric(concept_name=concept, **kwargs)

    def test_na_flag_set_for_unmapped_concept(self):
        metric = self._make_na_metric()
        assert metric._na is True

    def test_na_flag_false_for_mapped_concept(self):
        with patch("eval_unlearn.metrics.asr_i2p.metric.NudeDetector", MagicMock()):
            from eval_unlearn.metrics.asr_i2p.metric import ASRMetric
            metric = ASRMetric(concept_name="nudity", detector="nudenet")
        assert metric._na is False

    def test_no_detector_initialised_for_na_concept(self):
        metric = self._make_na_metric()
        assert metric._detector is None
        assert metric.nude_detector is None
        assert metric.q16_classifier is None
        assert metric.clip_model is None

    def test_load_dataset_returns_empty_loader(self):
        metric = self._make_na_metric()
        loader = metric.load_dataset()
        assert list(loader) == []

    def test_update_is_a_noop(self):
        metric = self._make_na_metric()
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 0
        assert metric._unsafe_count == 0

    def test_compute_returns_none_value(self):
        metric = self._make_na_metric(concept="a custom unrelated concept")
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert result.name == "ASR"
        assert result.value is None
        assert result.details["concept"] == "a custom unrelated concept"
        assert result.details["detector"] is None
        assert "error" in result.details

    def test_compute_full_workflow_stays_na(self):
        metric = self._make_na_metric()
        metric.load_dataset()
        metric.update([_dummy_image()], ["prompt"])
        result = metric.compute()
        assert result.value is None
