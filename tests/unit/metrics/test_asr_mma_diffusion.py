"""Unit tests for MMADiffusion metric."""
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch, call
from PIL import Image

from eval_unlearn.types import MetricResult


def _dummy_image(color=(10, 20, 30)):
    return Image.new("RGB", (16, 16), color=color)


def _make_metric(detector="nudenet", concept="nudity", limit=5, **kwargs):
    from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = [
        {"adversarial_prompt": f"adv_{i}", "target_prompt": "naked"} for i in range(3)
    ]
    with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.NudeDetector", MagicMock()), \
         patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier", MagicMock()), \
         patch("eval_unlearn.metrics.asr_mma_diffusion.metric.CLIPModel", MagicMock()), \
         patch("eval_unlearn.metrics.asr_mma_diffusion.metric.CLIPProcessor", MagicMock()):
        metric = MMADiffusionMetric(
            concept_name=concept,
            output_csv="/tmp/mma_test.csv",
            detector=detector,
            limit=limit,
            **kwargs
        )
    metric._AdversarialPromptGenerator = mock_gen_cls
    return metric


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestMMADiffusionConfig:
    def test_missing_concept_name_raises(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        with pytest.raises(ValueError, match="concept_name must be set"):
            MMADiffusionConfig(output_csv="/tmp/out.csv")

    def test_missing_output_csv_raises(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        with pytest.raises(ValueError, match="output_csv must be set"):
            MMADiffusionConfig(concept_name="nudity")

    def test_nudenet_on_non_nudity_raises(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        with pytest.raises(ValueError, match="only valid for nudity"):
            MMADiffusionConfig(concept_name="violence", output_csv="/tmp/o.csv", detector="nudenet")

    def test_invalid_detector_raises(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        with pytest.raises(ValueError, match="detector must be one of"):
            MMADiffusionConfig(concept_name="nudity", output_csv="/tmp/o.csv", detector="bad")

    def test_invalid_q16_threshold(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        with pytest.raises(ValueError, match="q16_threshold"):
            MMADiffusionConfig(concept_name="nudity", output_csv="/tmp/o.csv", q16_threshold=1.5)

    def test_similarity_threshold_out_of_range_raises(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        with pytest.raises(ValueError, match="similarity_threshold"):
            MMADiffusionConfig(concept_name="nudity", output_csv="/tmp/o.csv", similarity_threshold=-0.5)

    def test_gcg_defaults_match_paper(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        config = MMADiffusionConfig(concept_name="nudity", output_csv="/tmp/o.csv")
        assert config.n_steps == 1000
        assert config.n_cands == 5
        assert config.batch_size == 512
        assert config.topk == 256

    def test_min_adversarial_samples_default_is_100(self):
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        config = MMADiffusionConfig(concept_name="nudity", output_csv="/tmp/o.csv")
        assert config.min_adversarial_samples == 100


# ---------------------------------------------------------------------------
# load_dataset — GCG generation path
# ---------------------------------------------------------------------------
def _mock_i2p_loader(prompts):
    """Build a fake load_i2p_csv() DataLoader yielding a single Dataset batch."""
    from eval_unlearn.types import Dataset
    return [Dataset(prompts=prompts, metadata={"source": "i2p_hf", "concept": "nudity"})]


class TestMMALoadDataset:
    def test_load_dataset_returns_dataloader(self):
        metric = _make_metric()
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [
            {"adversarial_prompt": "adv", "target_prompt": "nude"}
        ]
        metric._AdversarialPromptGenerator = MagicMock(return_value=mock_gen)
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=_mock_i2p_loader(["a naked person"])):
            loader = metric.load_dataset()
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)

    def test_load_dataset_uses_i2p_targets_for_nudity(self):
        """When target_prompts is unset, nudity attacks target I2P prompts,
        requesting enough targets to reach min_adversarial_samples images
        after GCG produces n_cands candidates per target."""
        import math
        metric = _make_metric()
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [
            {"adversarial_prompt": "adv", "target_prompt": "nude"}
        ]
        metric._AdversarialPromptGenerator = MagicMock(return_value=mock_gen)
        i2p_prompts = ["a naked man", "a naked woman"]
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=_mock_i2p_loader(i2p_prompts)) as mock_load:
            metric.load_dataset()

        expected_needed = math.ceil(metric.config.min_adversarial_samples / metric.config.n_cands)
        mock_load.assert_called_once_with(concept="nudity", limit=expected_needed)
        mock_gen.generate.assert_called_once()
        assert mock_gen.generate.call_args.kwargs["target_prompts"] == i2p_prompts

    def test_load_dataset_applies_limit(self):
        metric = _make_metric(limit=2)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [
            {"adversarial_prompt": f"adv{i}", "target_prompt": "nude"} for i in range(10)
        ]
        metric._AdversarialPromptGenerator = MagicMock(return_value=mock_gen)
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=_mock_i2p_loader(["a naked person"])):
            loader = metric.load_dataset()
        all_prompts = []
        for batch in loader:
            all_prompts.extend(batch.prompts)
        assert len(all_prompts) == 2

    def test_load_dataset_precomputed_path(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "pre.csv"
        pd.DataFrame({"adversarial_prompt": ["adv1", "adv2"], "target_prompt": ["t1", "t2"]}).to_csv(csv_path, index=False)

        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.NudeDetector", MagicMock()):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="nudity",
                output_csv="/tmp/out.csv",
                detector="nudenet",
                precomputed_prompts_path=str(csv_path),
            )
        loader = metric.load_dataset()
        batch = next(iter(loader))
        assert "adv1" in batch.prompts or "adv2" in batch.prompts

    def test_resets_counters_on_load(self):
        metric = _make_metric()
        metric._unsafe_count = 99
        metric._total = 99
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [{"adversarial_prompt": "x", "target_prompt": "y"}]
        metric._AdversarialPromptGenerator = MagicMock(return_value=mock_gen)
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=_mock_i2p_loader(["a naked person"])):
            metric.load_dataset()
        assert metric._unsafe_count == 0
        assert metric._total == 0


# ---------------------------------------------------------------------------
# update — NudeNet detector
# ---------------------------------------------------------------------------
class TestMMAUpdateNudenet:
    def test_safe_image_not_counted(self):
        metric = _make_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 0
        assert metric._total == 1

    def test_unsafe_image_counted(self):
        metric = _make_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9}
        ]
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_detection_error_handled_gracefully(self):
        metric = _make_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.side_effect = RuntimeError("model error")
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# update — Q16 detector
# ---------------------------------------------------------------------------
class TestMMAUpdateQ16:
    def test_inappropriate_counted(self):
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [("inappropriate", 0.95)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="violence", output_csv="/tmp/o.csv", detector="q16"
            )
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_appropriate_not_counted(self):
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [("non inappropriate", 0.1)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="violence", output_csv="/tmp/o.csv", detector="q16"
            )
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 0
        assert metric._total == 1

    def test_q16_batch_error_handled(self):
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.side_effect = RuntimeError("GPU OOM")
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="violence", output_csv="/tmp/o.csv", detector="q16"
            )
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# update — CLIP detector
# ---------------------------------------------------------------------------
class TestMMAUpdateCLIP:
    def _make_clip_metric(self, sim_value=0.5):
        import torch
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_mma_diffusion.metric.CLIPProcessor") as mock_proc:
            mock_model = MagicMock()
            mock_model.get_image_features.return_value = torch.ones(1, 4)
            mock_model.get_text_features.return_value = torch.ones(1, 4)
            mock_cls.from_pretrained.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="violence", output_csv="/tmp/o.csv", detector="clip",
                similarity_threshold=sim_value
            )
        metric.clip_model = mock_model
        metric.clip_processor = MagicMock()
        metric.clip_processor.return_value = {"pixel_values": torch.ones(1, 3, 224, 224)}
        return metric

    def test_high_similarity_unsafe(self):
        import torch
        metric = self._make_clip_metric(sim_value=0.0)
        # cosine sim of ones/ones = 1.0 > threshold 0.0
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_clip_error_handled(self):
        metric = self._make_clip_metric()
        metric.clip_model.get_image_features.side_effect = RuntimeError("error")
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class TestMMACompute:
    def test_compute_ratio(self):
        metric = _make_metric(detector="nudenet")
        metric._unsafe_count = 3
        metric._total = 10
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert abs(result.value - 0.3) < 1e-6

    def test_compute_zero_total(self):
        metric = _make_metric(detector="nudenet")
        result = metric.compute()
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_details_include_indices(self):
        metric = _make_metric(detector="nudenet")
        metric._unsafe_count = 1
        metric._total = 2
        metric._unsafe_indices = [0]
        result = metric.compute()
        assert result.details["unsafe_indices"] == [0]


# ---------------------------------------------------------------------------
# Coverage gap tests: init branches, helper methods, update paths
# ---------------------------------------------------------------------------
class TestMMADiffusionCoverageGaps:
    def test_auto_detector_resolves_to_nudenet_for_nudity(self):
        """Line 75: auto detection for nudity → nudenet."""
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.NudeDetector"):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="nudity", output_csv="/tmp/o.csv", detector="auto"
            )
        assert metric._detector == "nudenet"

    def test_nudenet_none_raises_runtime_error(self):
        """Line 79: RuntimeError when NudeDetector is None."""
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.NudeDetector", None):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            with pytest.raises(RuntimeError, match="nudenet"):
                MMADiffusionMetric(concept_name="nudity", output_csv="/tmp/o.csv",
                                   detector="nudenet")

    def test_q16classifier_none_raises_runtime_error(self):
        """Line 88: RuntimeError when Q16Classifier is None."""
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier", None):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            with pytest.raises(RuntimeError, match="q16"):
                MMADiffusionMetric(concept_name="violence", output_csv="/tmp/o.csv",
                                   detector="q16")

    def test_q16_creates_classifier_with_default_model(self):
        """Lines 103-106: Q16Classifier is instantiated in init for q16 detector."""
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier") as mock_q16:
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="violence", output_csv="/tmp/o.csv", detector="q16",
            )
        assert metric.q16_classifier is not None

    def test_clipmodel_none_raises_runtime_error(self):
        """Line 110: RuntimeError when CLIPModel is None."""
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.CLIPModel", None):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            with pytest.raises(RuntimeError, match="transformers"):
                MMADiffusionMetric(concept_name="violence", output_csv="/tmp/o.csv",
                                   detector="clip")

    def test_precomputed_missing_column_raises(self, tmp_path):
        """Line 129: ValueError when precomputed CSV lacks adversarial_prompt column."""
        import pandas as pd
        csv_path = str(tmp_path / "bad.csv")
        pd.DataFrame({"wrong_col": ["a", "b"]}).to_csv(csv_path, index=False)
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.NudeDetector"):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(concept_name="nudity", output_csv="/tmp/o.csv",
                                        precomputed_prompts_path=csv_path)
        with pytest.raises(ValueError, match="adversarial_prompt"):
            metric.load_dataset()

    def test_precomputed_with_limit(self, tmp_path):
        """Line 134: df.head(limit) when limit is set."""
        import pandas as pd
        csv_path = str(tmp_path / "prompts.csv")
        pd.DataFrame({"adversarial_prompt": [f"p{i}" for i in range(10)],
                      "target_prompt": ["t"] * 10}).to_csv(csv_path, index=False)
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.NudeDetector"):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(concept_name="nudity", output_csv="/tmp/o.csv",
                                        precomputed_prompts_path=csv_path, limit=3)
        loader = metric.load_dataset()
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)

    def test_violence_with_no_target_prompts_borrows_from_i2p(self):
        """'violence' is one of I2P's 7 categories, so with no target_prompts
        it now auto-borrows from I2P instead of raising."""
        import math
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier"):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(concept_name="violence", output_csv="/tmp/o.csv",
                                        detector="q16")
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [{"adversarial_prompt": "adv", "target_prompt": "v"}]
        metric._AdversarialPromptGenerator = MagicMock(return_value=mock_gen)
        i2p_prompts = ["violent scene"]
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=_mock_i2p_loader(i2p_prompts)) as mock_load:
            metric.load_dataset()
        expected_needed = math.ceil(metric.config.min_adversarial_samples / metric.config.n_cands)
        mock_load.assert_called_once_with(concept="violence", limit=expected_needed)

    def test_non_i2p_concept_with_no_target_prompts_uses_defaults(self):
        """A concept outside I2P's 7 categories with no target_prompts falls
        back to synthesized generic template prompts instead of raising."""
        with patch("eval_unlearn.metrics.asr_mma_diffusion.metric.Q16Classifier"):
            from eval_unlearn.metrics.asr_mma_diffusion.metric import MMADiffusionMetric
            metric = MMADiffusionMetric(
                concept_name="a made up concept", output_csv="/tmp/o.csv", detector="q16"
            )
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [{"adversarial_prompt": "adv", "target_prompt": "x"}]
        metric._AdversarialPromptGenerator = MagicMock(return_value=mock_gen)
        metric.load_dataset()
        called_targets = mock_gen.generate.call_args.kwargs["target_prompts"]
        assert all("a made up concept" in p for p in called_targets)

    def test_prompt_source_custom_without_target_prompts_raises(self):
        """Explicitly requesting prompt_source='custom' still requires target_prompts."""
        from eval_unlearn.metrics.asr_mma_diffusion.config import MMADiffusionConfig
        with pytest.raises(ValueError, match="prompt_source='custom' requires target_prompts"):
            MMADiffusionConfig(
                concept_name="violence", output_csv="/tmp/o.csv", prompt_source="custom"
            )

    def test_is_unsafe_q16_numpy_array(self):
        """Lines 223-225: numpy array handling in _is_unsafe_q16."""
        import numpy as np
        metric = _make_metric(detector="q16", concept="violence")
        metric.q16_classifier = MagicMock()
        metric.q16_classifier.classify_batch.return_value = [("safe", 0.1)]
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = metric._is_unsafe_q16([arr])
        assert result == [False]

    def test_is_unsafe_q16_invalid_type_early_return(self):
        """Lines 227, 230: invalid type → all False early return."""
        metric = _make_metric(detector="q16", concept="violence")
        metric.q16_classifier = MagicMock()
        result = metric._is_unsafe_q16([42, "not_an_image"])
        assert result == [False, False]

    def test_is_unsafe_q16_mixed_valid_invalid(self):
        """Line 242: False for invalid item in mixed batch."""
        metric = _make_metric(detector="q16", concept="violence")
        metric.q16_classifier = MagicMock()
        metric.q16_classifier.classify_batch.return_value = [("inappropriate", 0.9)]
        img = _dummy_image()
        result = metric._is_unsafe_q16([img, 42])
        assert result == [True, False]

    def test_is_unsafe_clip_invalid_type_early_return(self):
        """Lines 257-258, 262: invalid image type → early return [False, False]."""
        metric = _make_metric(detector="clip", concept="violence")
        result = metric._is_unsafe_clip([42, "not_an_image"])
        assert result == [False, False]

    def test_is_unsafe_clip_numpy_array(self):
        """Lines 255-256: numpy array converted to PIL for CLIP eval."""
        import numpy as np, torch
        metric = _make_metric(detector="clip", concept="violence")
        real_feat = torch.ones(1, 4)
        mock_img_out = MagicMock()
        mock_img_out.pooler_output = real_feat
        mock_text_out = MagicMock()
        mock_text_out.pooler_output = real_feat
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {}
        metric.clip_processor.return_value = mock_inputs
        metric.clip_model.get_image_features.return_value = mock_img_out
        metric.clip_model.get_text_features.return_value = mock_text_out
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = metric._is_unsafe_clip([arr])
        assert isinstance(result, list)

    def test_is_unsafe_clip_pooler_output_branches(self):
        """Lines 269, 277: pooler_output fallback in _is_unsafe_clip."""
        import torch
        metric = _make_metric(detector="clip", concept="violence")
        real_feat = torch.ones(1, 4)
        mock_img_out = MagicMock()
        mock_img_out.pooler_output = real_feat
        mock_text_out = MagicMock()
        mock_text_out.pooler_output = real_feat
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {}
        metric.clip_processor.return_value = mock_inputs
        metric.clip_model.get_image_features.return_value = mock_img_out
        metric.clip_model.get_text_features.return_value = mock_text_out
        img = _dummy_image()
        result = metric._is_unsafe_clip([img])
        assert isinstance(result, list)

    def test_nudenet_update_string_path(self, tmp_path):
        """Lines 307-308: string file path branch in NudeNet update."""
        img_path = str(tmp_path / "test.png")
        _dummy_image().save(img_path)
        metric = _make_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        metric.update([img_path], ["prompt"])
        assert metric._total == 1

    def test_nudenet_update_oserror_on_remove(self):
        """Lines 323-324: OSError during temp file cleanup is silently ignored."""
        metric = _make_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), \
             patch("eval_unlearn.metrics.asr_mma_diffusion.metric.os.path.exists",
                   return_value=True), \
             patch("eval_unlearn.metrics.asr_mma_diffusion.metric.os.remove",
                   side_effect=OSError("locked")):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1
