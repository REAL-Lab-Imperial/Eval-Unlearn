"""Unit tests for ASR P4D metric."""
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from PIL import Image

from eval_unlearn.types import MetricResult


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _make_p4d_metric(detector="nudenet", concept="nudity", **kwargs):
    """Build ASRP4D with all external deps mocked."""
    mock_p4d = MagicMock()
    mock_nude = MagicMock()
    mock_q16 = MagicMock()

    with patch.dict("sys.modules", {
        "p4d": MagicMock(P4DGenerator=mock_p4d),
    }), \
    patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", mock_p4d), \
    patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", mock_nude), \
    patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", mock_q16), \
    patch("eval_unlearn.metrics.asr_p4d.metric.CLIPModel", MagicMock()), \
    patch("eval_unlearn.metrics.asr_p4d.metric.CLIPProcessor", MagicMock()):
        from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
        metric = ASRP4D(
            concept_name=concept,
            detector=detector,
            erase_id="std",
            **kwargs
        )
    return metric


# ---------------------------------------------------------------------------
# ASRP4DConfig
# ---------------------------------------------------------------------------
class TestASRP4DConfig:
    def test_defaults(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        cfg = ASRP4DConfig.from_dict({"concept_name": "nudity"})
        assert cfg.concept_name == "nudity"
        assert cfg.detector == "auto"
        assert cfg.erase_id == "std"

    def test_optimisation_defaults_match_paper(self):
        """num_iter/eval_step/print_step default to the P4D paper's published
        settings (packages/p4d's own generator defaults)."""
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        cfg = ASRP4DConfig.from_dict({"concept_name": "nudity"})
        assert cfg.num_iter == 3000
        assert cfg.eval_step == 50
        assert cfg.print_step == 100

    def test_min_adversarial_samples_default_is_100(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        cfg = ASRP4DConfig.from_dict({"concept_name": "nudity"})
        assert cfg.min_adversarial_samples == 100

    def test_invalid_detector(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="detector must be one of"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "detector": "bad"})

    def test_nudenet_on_non_nudity(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="only valid for nudity"):
            ASRP4DConfig.from_dict({"concept_name": "violence", "detector": "nudenet"})

    def test_invalid_variant(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="variant must be one of"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "variant": "bad"})

    def test_invalid_erase_id(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="erase_id must be one of"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "erase_id": "bad"})

    def test_invalid_q16_threshold(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="q16_threshold"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "q16_threshold": 2.0})

    def test_empty_concept_name_raises(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="concept_name must not be empty"):
            ASRP4DConfig.from_dict({"concept_name": ""})

    def test_sld_without_safe_level_raises(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="safe_level must be set"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "erase_id": "sld"})

    def test_invalid_safe_level_raises(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="safe_level must be one of"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "erase_id": "sld", "safe_level": "ULTRA"})

    def test_eval_step_exceeds_num_iter_raises(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="eval_step"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "eval_step": 50, "num_iter": 10})

    def test_unsafe_threshold_out_of_range_raises(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="unsafe_threshold"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "unsafe_threshold": 1.5})

    def test_similarity_threshold_out_of_range_raises(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="similarity_threshold"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "similarity_threshold": -0.1})

    def test_clip_threshold_out_of_range_raises(self):
        from eval_unlearn.metrics.asr_p4d.config import ASRP4DConfig
        with pytest.raises(ValueError, match="clip_threshold"):
            ASRP4DConfig.from_dict({"concept_name": "nudity", "clip_threshold": 2.0})


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------
class TestASRP4DLoadDataset:
    def test_returns_dataloader_from_precomputed(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "p4d_prompts.csv"
        pd.DataFrame({
            "adversarial_prompt": ["adv1", "adv2"],
            "target_prompt": ["naked", "nude"],
        }).to_csv(csv_path, index=False)

        metric = _make_p4d_metric()
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "erase_id": "std",
            "precomputed_prompts_path": str(csv_path),
        })
        from torch.utils.data import DataLoader
        loader = metric.load_dataset()
        assert isinstance(loader, DataLoader)

    def test_precomputed_with_limit(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "p4d_prompts.csv"
        pd.DataFrame({
            "adversarial_prompt": [f"adv{i}" for i in range(10)],
            "target_prompt": ["nude"] * 10,
        }).to_csv(csv_path, index=False)

        metric = _make_p4d_metric()
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "erase_id": "std",
            "precomputed_prompts_path": str(csv_path),
            "limit": 3,
        })
        loader = metric.load_dataset()
        all_prompts = []
        for batch in loader:
            all_prompts.extend(batch.prompts)
        assert len(all_prompts) == 3

    def test_missing_adversarial_prompt_column_raises(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"other_col": ["x"]}).to_csv(csv_path, index=False)

        metric = _make_p4d_metric()
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "erase_id": "std",
            "precomputed_prompts_path": str(csv_path),
        })
        with pytest.raises(ValueError, match="adversarial_prompt"):
            metric.load_dataset()

    def test_resets_counters_on_load(self, tmp_path):
        import pandas as pd
        csv_path = tmp_path / "p.csv"
        pd.DataFrame({"adversarial_prompt": ["x"], "target_prompt": ["y"]}).to_csv(csv_path, index=False)

        metric = _make_p4d_metric()
        metric._unsafe_count = 77
        metric._total = 77
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "erase_id": "std",
            "precomputed_prompts_path": str(csv_path),
        })
        metric.load_dataset()
        assert metric._unsafe_count == 0
        assert metric._total == 0


# ---------------------------------------------------------------------------
# update — NudeNet
# ---------------------------------------------------------------------------
class TestASRP4DUpdateNudenet:
    def test_safe_image(self):
        metric = _make_p4d_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1
        assert metric._unsafe_count == 0

    def test_unsafe_image(self):
        metric = _make_p4d_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = [
            {"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.99}
        ]
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_error_during_detection_handled(self):
        metric = _make_p4d_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.side_effect = Exception("crash")
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# update — Q16
# ---------------------------------------------------------------------------
class TestASRP4DUpdateQ16:
    def _q16_metric(self, label="non inappropriate"):
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [(label, 0.9)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            m = ASRP4D(concept_name="violence", detector="q16", erase_id="std")
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


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class TestASRP4DCompute:
    def test_compute_asr(self):
        metric = _make_p4d_metric()
        metric._unsafe_count = 4
        metric._total = 8
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert abs(result.value - 0.5) < 1e-6

    def test_compute_zero_total(self):
        metric = _make_p4d_metric()
        result = metric.compute()
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_details(self):
        metric = _make_p4d_metric()
        metric._unsafe_count = 2
        metric._total = 5
        metric._unsafe_indices = [1, 3]
        result = metric.compute()
        assert result.details["unsafe_indices"] == [1, 3]
        assert result.details["total_images"] == 5


# ---------------------------------------------------------------------------
# Coverage gap tests for asr_p4d
# ---------------------------------------------------------------------------
class TestASRP4DCoverageGaps:
    def _make_q16_metric(self):
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier") as mock_q16_cls:
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [("inappropriate", 0.9)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            m = ASRP4D(concept_name="violence", detector="q16", erase_id="std")
        return m

    def _make_clip_metric(self):
        import torch
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.CLIPModel") as mock_cls, \
             patch("eval_unlearn.metrics.asr_p4d.metric.CLIPProcessor") as mock_proc:
            mock_model = MagicMock()
            mock_cls.from_pretrained.return_value = mock_model
            mock_model.eval.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_proc.from_pretrained.return_value = MagicMock()
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            m = ASRP4D(concept_name="violence", detector="clip", erase_id="std")
        real_feat = torch.ones(1, 4)
        m.clip_model = MagicMock()
        m.clip_model.get_image_features.return_value = real_feat
        m.clip_model.get_text_features.return_value = real_feat
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {}
        m.clip_processor = MagicMock(return_value=mock_inputs)
        return m

    def test_q16_fallback_warning_for_unknown_clip_model(self):
        """Line 115: warning when clip_model_id not in _HF_TO_Q16."""
        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier") as mock_q16_cls:
            mock_q16_cls.return_value = MagicMock()
            from eval_unlearn.metrics.asr_p4d.metric import ASRP4D
            metric = ASRP4D(
                concept_name="violence",
                detector="q16",
                erase_id="std",
                clip_model_id="some/unknown-model",
            )
        assert metric.q16_classifier is not None

    def test_target_prompts_path_with_limit_covers_lines_155_217(self, tmp_path):
        """Lines 155 (df.head limit) and 217 (collate_fn Dataset return)."""
        import pandas as pd
        from eval_unlearn.types import Dataset as _Dataset

        csv_path = str(tmp_path / "prompts.csv")
        pd.DataFrame({"prompt": [f"p{i}" for i in range(5)]}).to_csv(csv_path, index=False)

        metric = _make_p4d_metric(
            detector="nudenet",
            target_prompts_path=csv_path,
            limit=3,
        )

        mock_rows = [
            {"adversarial_prompt": f"adv{i}", "target_prompt": f"t{i}", "best_similarity": 0.5}
            for i in range(3)
        ]
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_rows

        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", return_value=mock_gen):
            loader = metric.load_dataset()

        batches = list(loader)
        assert len(batches) > 0
        assert isinstance(batches[0], _Dataset)

    def test_q16_mixed_batch_covers_false_branch(self):
        """Line 299: results.append(False) for invalid image in mixed batch."""
        metric = self._make_q16_metric()
        img = _dummy_image()
        result = metric._is_unsafe_q16([img, 42])
        assert result == [True, False]

    def test_q16_numpy_array_covers_branch(self):
        """Line 281: numpy branch in _is_unsafe_q16."""
        import numpy as np
        metric = self._make_q16_metric()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = metric._is_unsafe_q16([arr])
        assert result == [True]

    def test_clip_numpy_array_covers_branch(self):
        """Line 312: numpy branch in _is_unsafe_clip."""
        import numpy as np
        metric = self._make_clip_metric()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        result = metric._is_unsafe_clip([arr])
        assert isinstance(result, list)

    def test_nudenet_string_path_update(self, tmp_path):
        """Line 351: string image path in update with nudenet detector."""
        img_path = str(tmp_path / "test.png")
        _dummy_image().save(img_path)
        metric = _make_p4d_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        metric.update([img_path], ["prompt"])
        assert metric._total == 1

    def test_nudenet_oserror_on_remove(self):
        """Lines 366-367: OSError during temp file cleanup is silently ignored."""
        metric = _make_p4d_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = []
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/fake_p4d.png")), \
             patch("os.close"), \
             patch("eval_unlearn.metrics.asr_p4d.metric.os.path.exists", return_value=True), \
             patch("eval_unlearn.metrics.asr_p4d.metric.os.remove",
                   side_effect=OSError("locked")):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1
