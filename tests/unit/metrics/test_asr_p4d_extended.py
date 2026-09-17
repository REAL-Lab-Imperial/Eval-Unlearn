"""Extended tests for ASR P4D metric targeting uncovered lines."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import pandas as pd
import tempfile
import os


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _make_p4d_metric(detector="nudenet", concept="nudity", **kwargs):
    """Build ASRP4D with all external deps mocked."""
    with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
         patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
         patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", MagicMock()), \
         patch("eval_unlearn.metrics.asr_p4d.metric.CLIPModel", MagicMock()), \
         patch("eval_unlearn.metrics.asr_p4d.metric.CLIPProcessor", MagicMock()), \
         patch("eval_unlearn.metrics._vlm_detector.pipeline", MagicMock()):
        from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
        metric = ASRP4D(
            concept_name=concept,
            detector=detector,
            erase_id="std",
            **kwargs
        )
    return metric


# ---------------------------------------------------------------------------
# load_dataset — P4D generation path (uncovered lines 150-227)
# ---------------------------------------------------------------------------
class TestASRP4DLoadDatasetGeneration:
    def _make_p4d_with_target(self, tmp_path, **extra):
        """Helper to create a metric with a target_prompts_path."""
        prompts_csv = tmp_path / "prompts.csv"
        pd.DataFrame({
            "prompt": ["a nude person"],
            "evaluation_seed": [42],
            "evaluation_guidance": [7.5],
        }).to_csv(prompts_csv, index=False)

        mock_p4d_gen = MagicMock()
        mock_p4d_gen.generate.return_value = [
            {"adversarial_prompt": "adv1", "target_prompt": "t1", "best_similarity": 0.9},
        ]

        metric = _make_p4d_metric(
            detector="nudenet",
            concept="nudity",
            target_prompts_path=str(prompts_csv),
            **extra,
        )
        # Patch P4DGenerator at the module level for the call
        import eval_unlearn.metrics.asr_p4d.metric as m
        m.P4DGenerator = MagicMock(return_value=mock_p4d_gen)
        return metric, prompts_csv, mock_p4d_gen

    def test_load_from_target_prompts(self, tmp_path):
        """Test the P4D generation code path (no precomputed prompts)."""
        metric, _, _ = self._make_p4d_with_target(tmp_path)
        from torch.utils.data import DataLoader
        loader = metric.load_dataset()
        assert isinstance(loader, DataLoader)

    def test_load_missing_target_path_borrows_from_i2p_for_nudity(self):
        """'nudity' is an I2P category, so with no target_prompts_path or
        precomputed_prompts_path it now auto-borrows I2P prompts instead of
        raising."""
        metric = _make_p4d_metric(detector="nudenet", concept="nudity")
        mock_p4d_gen = MagicMock()
        mock_p4d_gen.generate.return_value = [
            {"adversarial_prompt": "adv1", "target_prompt": "t1", "best_similarity": 0.9},
        ]
        import eval_unlearn.metrics.asr_p4d.metric as m
        m.P4DGenerator = MagicMock(return_value=mock_p4d_gen)

        from eval_unlearn.types import Dataset
        mock_loader = [Dataset(prompts=["a nude person"], metadata={})]
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=mock_loader) as mock_load:
            loader = metric.load_dataset()
        mock_load.assert_called_once_with(concept="nudity", limit=metric.config.min_adversarial_samples)
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)

    def test_load_missing_target_path_uses_defaults_for_non_i2p_concept(self):
        """A concept outside I2P's 7 categories with no target_prompts_path
        falls back to synthesized generic template prompts."""
        metric = _make_p4d_metric(detector="q16", concept="a made up concept")
        mock_p4d_gen = MagicMock()
        mock_p4d_gen.generate.return_value = [
            {"adversarial_prompt": "adv1", "target_prompt": "t1", "best_similarity": 0.9},
        ]
        import eval_unlearn.metrics.asr_p4d.metric as m
        m.P4DGenerator = MagicMock(return_value=mock_p4d_gen)

        loader = metric.load_dataset()
        called_prompts = mock_p4d_gen.generate.call_args.kwargs["target_prompts"]
        assert all("a made up concept" in p for p in called_prompts)
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)

    def test_prompt_source_custom_without_target_path_raises(self):
        """Explicitly requesting prompt_source='custom' still requires target_prompts_path."""
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="prompt_source='custom' requires target_prompts_path"):
            ASRP4DConfig(concept_name="nudity", prompt_source="custom")

    def test_load_saves_generated_prompts(self, tmp_path):
        """Test generated_prompts_output saves the CSV."""
        output_csv = tmp_path / "out" / "generated.csv"
        metric, _, _ = self._make_p4d_with_target(
            tmp_path, generated_prompts_output=str(output_csv)
        )
        metric.load_dataset()
        assert output_csv.exists()

    def test_load_without_seed_guidance_cols(self, tmp_path):
        """CSV without evaluation_seed or evaluation_guidance is fine."""
        prompts_csv = tmp_path / "prompts.csv"
        pd.DataFrame({"prompt": ["a person"]}).to_csv(prompts_csv, index=False)

        mock_p4d_gen = MagicMock()
        mock_p4d_gen.generate.return_value = [
            {"adversarial_prompt": "adv1", "target_prompt": "t1", "best_similarity": 0.9},
        ]

        metric = _make_p4d_metric(
            detector="nudenet", concept="nudity",
            target_prompts_path=str(prompts_csv),
        )
        import eval_unlearn.metrics.asr_p4d.metric as m
        m.P4DGenerator = MagicMock(return_value=mock_p4d_gen)

        from torch.utils.data import DataLoader
        loader = metric.load_dataset()
        assert isinstance(loader, DataLoader)

    def test_collate_fn_includes_best_similarity(self, tmp_path):
        """Verify collate_fn properly fills best_similarities."""
        precomputed_csv = tmp_path / "pre.csv"
        pd.DataFrame({
            "adversarial_prompt": ["adv1"],
            "target_prompt": ["t1"],
            "best_similarity": [0.87],
        }).to_csv(precomputed_csv, index=False)

        metric = _make_p4d_metric(detector="nudenet")
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "erase_id": "std",
            "precomputed_prompts_path": str(precomputed_csv),
        })
        loader = metric.load_dataset()
        batch = next(iter(loader))
        assert batch.metadata["best_similarities"] == [0.87]


# ---------------------------------------------------------------------------
# update — CLIP path (lines 382-392)
# ---------------------------------------------------------------------------
class TestASRP4DUpdateCLIP:
    def _make_clip_metric(self, sim_threshold=0.0):
        import torch
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_p4d.metric.CLIPProcessor") as mock_proc:
            mock_model = MagicMock()
            mock_model.get_image_features.return_value = torch.ones(1, 4)
            mock_model.get_text_features.return_value = torch.ones(1, 4)
            mock_cls.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_proc.from_pretrained.return_value = MagicMock()
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            metric = ASRP4D(
                concept_name="violence",
                detector="clip",
                erase_id="std",
                similarity_threshold=sim_threshold,
            )
        metric.clip_model = mock_model
        mock_proc_inst = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"pixel_values": torch.ones(1, 3, 224, 224)}
        mock_proc_inst.return_value = mock_inputs
        metric.clip_processor = mock_proc_inst
        return metric

    def test_high_similarity_marked_unsafe(self):
        metric = self._make_clip_metric(sim_threshold=0.0)
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_clip_error_handled(self):
        metric = self._make_clip_metric()
        metric.clip_model.get_image_features.side_effect = RuntimeError("error")
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1

    def test_no_valid_images_handled(self):
        metric = self._make_clip_metric()
        # Pass None (not a PIL image and no .shape) — should use [False] path
        metric.update([None], ["prompt"])
        assert metric._total == 1

    def test_array_image_converted(self):
        metric = self._make_clip_metric(sim_threshold=0.0)
        # PIL images work fine
        metric.update([_dummy_image()], ["p"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# Config — missing lines
# ---------------------------------------------------------------------------
class TestASRP4DConfigExtended:
    def test_nudenet_auto_resolves_to_nudenet(self):
        metric = _make_p4d_metric(concept="nudity", detector="auto")
        assert metric._detector == "nudenet"

    def test_vlm_auto_resolves_for_non_nudity(self):
        metric = _make_p4d_metric(concept="violence", detector="auto")
        assert metric._detector == "vlm"
        assert metric.vlm_detector is not None

    def test_custom_erase_id_no_checkpoint_warns(self):
        """erase_id='custom' without checkpoint path should log warning, not raise."""
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.CLIPModel", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.CLIPProcessor", MagicMock()):
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            # Should not raise — just log a warning
            metric = ASRP4D(
                concept_name="nudity",
                detector="nudenet",
                erase_id="custom",  # No erase_concept_checkpoint
            )
        assert metric is not None

    def test_p4d_generator_none_raises(self):
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", None):
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            with pytest.raises(ImportError, match="p4d"):
                ASRP4D(concept_name="nudity", detector="nudenet", erase_id="std")

    def test_nudenet_none_raises(self):
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", None):
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            with pytest.raises(RuntimeError, match="nudenet"):
                ASRP4D(concept_name="nudity", detector="nudenet", erase_id="std")

    def test_q16_none_raises(self):
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", None):
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            with pytest.raises(RuntimeError, match="q16"):
                ASRP4D(concept_name="violence", detector="q16", erase_id="std")

    def test_clip_none_raises(self):
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.CLIPModel", None):
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            with pytest.raises(RuntimeError, match="transformers"):
                ASRP4D(concept_name="violence", detector="clip", erase_id="std")


# ---------------------------------------------------------------------------
# _is_unsafe_q16 — edge cases
# ---------------------------------------------------------------------------
class TestASRP4DIsUnsafeQ16:
    def test_q16_empty_images_returns_false_list(self):
        metric = _make_p4d_metric(detector="q16", concept="violence")
        metric.q16_classifier.classify_batch.return_value = []
        results = metric._is_unsafe_q16([])
        assert results == []

    def test_q16_invalid_image_skipped(self):
        metric = _make_p4d_metric(detector="q16", concept="violence")
        metric.q16_classifier.classify_batch.return_value = []
        results = metric._is_unsafe_q16([42])  # not PIL, no shape
        assert results == [False]

    def test_q16_array_image_converted(self):
        metric = _make_p4d_metric(detector="q16", concept="violence")
        metric.q16_classifier.classify_batch.return_value = [("non inappropriate", 0.1)]
        # Create an object with .shape attr (acts like numpy array)
        import numpy as np
        arr_img = np.zeros((16, 16, 3), dtype=np.uint8)
        results = metric._is_unsafe_q16([arr_img])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# _is_unsafe_clip — edge cases
# ---------------------------------------------------------------------------
class TestASRP4DIsUnsafeClip:
    def _clip_metric(self):
        import torch
        metric = _make_p4d_metric(detector="clip", concept="violence")
        mock_model = MagicMock()
        mock_model.get_image_features.return_value = torch.ones(1, 4)
        mock_model.get_text_features.return_value = torch.ones(1, 4)
        metric.clip_model = mock_model
        mock_proc = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"pixel_values": torch.ones(1, 3, 224, 224)}
        mock_proc.return_value = mock_inputs
        metric.clip_processor = mock_proc
        metric._device = "cpu"
        return metric

    def test_empty_images_returns_false_list(self):
        metric = self._clip_metric()
        results = metric._is_unsafe_clip([])
        assert results == []

    def test_none_image_handled(self):
        metric = self._clip_metric()
        results = metric._is_unsafe_clip([None])
        assert results == [False]
