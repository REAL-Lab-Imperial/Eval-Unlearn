"""Additional unit tests to cover uncovered branches in asr_ring_a_bell metric."""
import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from PIL import Image

from eval_unlearn.types import MetricResult


def _dummy_image():
    return Image.new("RGB", (16, 16))


def _make_rab_metric(detector="nudenet", enable_discovery=False, concept="nudity"):
    """Build ASRRingABellMetric with all GPU deps mocked."""
    mock_clip = MagicMock()
    mock_proc = MagicMock()

    with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier", MagicMock()), \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mock_cls, \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mock_proc_cls, \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
         patch("os.path.exists", return_value=True), \
         patch("numpy.load", return_value=__import__("numpy").zeros((77, 768))):
        mock_cls.from_pretrained.return_value = mock_clip
        mock_clip.to.return_value = mock_clip
        mock_proc_cls.from_pretrained.return_value = mock_proc

        kwargs = {
            "concept_name": concept,
            "detector": detector,
            "seed_prompts_csv": "/fake/seeds.csv",
            "enable_discovery": enable_discovery,
        }
        if enable_discovery:
            kwargs["generated_prompts_output"] = "/fake/out.csv"
            kwargs["concept_vector_path"] = "/fake/vec.npy"

        from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        metric = ASRRingABellMetric(**kwargs)

    return metric


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
class TestRingABellConfigValidation:
    def test_discovery_without_seed_csv_succeeds_nudity_auto_sources_i2p(self):
        """No seed_prompts_csv + enable_discovery=True no longer raises: with
        prompt_source='auto' (default) and concept='nudity' (an I2P category),
        seed prompts are borrowed from I2P lazily in load_dataset() instead."""
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mc, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mp, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()):
            m = MagicMock(); m.to.return_value = m
            mc.from_pretrained.return_value = m
            mp.from_pretrained.return_value = MagicMock()
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            metric = ASRRingABellMetric(concept_name="nudity", enable_discovery=True)
        assert metric.config.seed_prompts_csv is None

    def test_discovery_custom_source_without_seed_csv_raises(self):
        """Explicitly requesting prompt_source='custom' still requires seed_prompts_csv."""
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()):
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            with pytest.raises(ValueError, match="prompt_source='custom' requires seed_prompts_csv"):
                ASRRingABellMetric(
                    concept_name="nudity", enable_discovery=True, prompt_source="custom"
                )

    def test_discovery_without_output_auto_generates_temp_path(self, tmp_path):
        """generated_prompts_output is no longer required at construction time
        — load_dataset() auto-generates a temp file for it when not set."""
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch("os.path.exists", return_value=True), \
             patch("numpy.load", return_value=__import__("numpy").zeros((77, 768))):
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            metric = ASRRingABellMetric(
                concept_name="nudity",
                enable_discovery=True,
                seed_prompts_csv="/fake/s.csv",
                concept_vector_path="/fake/v.npy",
            )
        assert metric.config.generated_prompts_output is None

    def test_no_discovery_without_seed_raises(self):
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()):
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            with pytest.raises(ValueError, match="seed_prompts_csv"):
                ASRRingABellMetric(concept_name="nudity", enable_discovery=False)

    def test_non_nudity_without_vector_auto_computes(self):
        """A concept with no bundled vector no longer raises — a concept
        vector is auto-computed from paired CLIP prompt templates instead."""
        import numpy as np
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mc, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mp, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch(
                 "eval_unlearn.metrics.asr_ring_a_bell.metric.compute_concept_vector",
                 return_value=np.zeros((77, 768), dtype=np.float32),
             ) as mock_compute:
            m = MagicMock(); m.to.return_value = m
            mc.from_pretrained.return_value = m
            mp.from_pretrained.return_value = MagicMock()
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            metric = ASRRingABellMetric(
                concept_name="violence",
                detector="q16",
                enable_discovery=True,
                seed_prompts_csv="/fake/s.csv",
                generated_prompts_output="/fake/out.csv",
            )
        mock_compute.assert_called_once()
        assert metric._concept_vector_path is not None
        assert metric._concept_vector_path.endswith(".npy")


# ---------------------------------------------------------------------------
# load_dataset — seed prompts path (no discovery)
# ---------------------------------------------------------------------------
class TestRingABellLoadDatasetSeedPath:
    def test_loads_from_seed_csv(self, tmp_path):
        seed_csv = tmp_path / "seeds.csv"
        seed_csv.write_text("prompt\na naked body\nexplicit nudity\n")

        metric = _make_rab_metric(enable_discovery=False)
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "seed_prompts_csv": str(seed_csv),
            "enable_discovery": False,
        })
        loader = metric.load_dataset()
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)

    def test_empty_seed_csv_raises(self, tmp_path):
        seed_csv = tmp_path / "empty.csv"
        seed_csv.write_text("prompt\n")  # header only

        metric = _make_rab_metric(enable_discovery=False)
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "seed_prompts_csv": str(seed_csv),
            "enable_discovery": False,
        })
        with pytest.raises(ValueError, match="empty"):
            metric.load_dataset()

    def test_limit_applied(self, tmp_path):
        seed_csv = tmp_path / "seeds.csv"
        seed_csv.write_text("prompt\n" + "\n".join(f"prompt{i}" for i in range(10)) + "\n")

        metric = _make_rab_metric(enable_discovery=False)
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "nudenet",
            "seed_prompts_csv": str(seed_csv),
            "enable_discovery": False,
            "limit": 3,
        })
        loader = metric.load_dataset()
        all_prompts = []
        for batch in loader:
            all_prompts.extend(batch.prompts)
        assert len(all_prompts) == 3


# ---------------------------------------------------------------------------
# update — Q16 path
# ---------------------------------------------------------------------------
class TestRingABellUpdateQ16:
    def test_inappropriate_counted(self):
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier") as mock_q16_cls, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()):
            mock_q16 = MagicMock()
            mock_q16.classify_batch.return_value = [("inappropriate", 0.95)]
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            metric = ASRRingABellMetric(
                concept_name="violence",
                detector="q16",
                seed_prompts_csv="/fake/s.csv",
                enable_discovery=False,
            )
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_q16_exception_handled(self):
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier") as mock_q16_cls, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()):
            mock_q16 = MagicMock()
            mock_q16.classify_batch.side_effect = RuntimeError("OOM")
            mock_q16_cls.return_value = mock_q16
            from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
            metric = ASRRingABellMetric(
                concept_name="violence",
                detector="q16",
                seed_prompts_csv="/fake/s.csv",
                enable_discovery=False,
            )
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# update — CLIP path
# ---------------------------------------------------------------------------
class TestRingABellUpdateCLIP:
    def test_high_similarity_is_unsafe(self):
        import torch
        metric = _make_rab_metric(detector="clip")
        metric._detector = "clip"
        metric.clip_model = MagicMock()
        metric.clip_model.get_image_features.return_value = torch.ones(1, 4)
        metric.clip_model.get_text_features.return_value = torch.ones(1, 4)
        metric.clip_processor = MagicMock()
        metric.clip_processor.return_value = {}
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity",
            "detector": "clip",
            "seed_prompts_csv": "/fake/s.csv",
            "enable_discovery": False,
            "similarity_threshold": 0.0,  # everything above 0 counts
        })
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 1

    def test_clip_exception_returns_safe(self):
        metric = _make_rab_metric(detector="clip")
        metric._detector = "clip"
        metric.clip_model = MagicMock()
        metric.clip_model.get_image_features.side_effect = RuntimeError("crash")
        metric.clip_processor = MagicMock()
        metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 0
        assert metric._total == 1


# ---------------------------------------------------------------------------
# update — NudeNet path (additional branches)
# ---------------------------------------------------------------------------
class TestRingABellUpdateNudenet:
    def test_low_score_not_unsafe(self):
        metric = _make_rab_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.3}  # below 0.5 threshold
        ]
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/rab.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._unsafe_count == 0

    def test_error_during_detection(self):
        metric = _make_rab_metric(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.side_effect = Exception("boom")
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/rab.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["prompt"])
        assert metric._total == 1


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class TestRingABellCompute:
    def test_compute_zero_total(self):
        metric = _make_rab_metric()
        result = metric.compute()
        assert result.value == 0.0

    def test_compute_ratio(self):
        metric = _make_rab_metric()
        metric._unsafe_count = 3
        metric._total = 10
        result = metric.compute()
        assert abs(result.value - 0.3) < 1e-6

    def test_compute_details(self):
        metric = _make_rab_metric()
        metric._unsafe_count = 1
        metric._total = 4
        metric._generated_prompts = ["a", "b", "c", "d"]
        result = metric.compute()
        assert result.details["generated_prompts_count"] == 4
        assert result.details["concept"] == "nudity"


import numpy as np


# ---------------------------------------------------------------------------
# Q16=None, unknown clip model warning, bundled vector, embed-dim mismatch,
# discovery + seed-load paths, nudenet file-path, Q16 update paths
# ---------------------------------------------------------------------------
def _make_rab_base(detector="nudenet", concept="nudity", **extra):
    from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
    with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector"), \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier"), \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mc, \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mp, \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
         patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
         patch("os.path.exists", return_value=True), \
         patch("numpy.load", return_value=np.zeros((77, 768))):
        m = MagicMock(); m.to.return_value = m
        mc.from_pretrained.return_value = m
        mp.from_pretrained.return_value = MagicMock()
        return ASRRingABellMetric(
            concept_name=concept, detector=detector,
            seed_prompts_csv="/fake/seeds.csv", enable_discovery=False,
            **extra,
        )


class TestRingABellAdditionalBranches:
    def test_q16_classifier_none_raises(self):
        from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier", None), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mc, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mp, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()):
            m = MagicMock(); m.to.return_value = m
            mc.from_pretrained.return_value = m
            mp.from_pretrained.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="q16"):
                ASRRingABellMetric(
                    concept_name="violence", detector="q16",
                    seed_prompts_csv="/fake/s.csv", enable_discovery=False,
                )

    def test_q16_unknown_clip_model_warns(self):
        from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.Q16Classifier") as mq, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mc, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mp, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.config.validate_clip_model"):
            m = MagicMock(); m.to.return_value = m
            mc.from_pretrained.return_value = m
            mp.from_pretrained.return_value = MagicMock()
            metric = ASRRingABellMetric(
                concept_name="violence", detector="q16",
                seed_prompts_csv="/fake/s.csv", enable_discovery=False,
                clip_model_id="openai/clip-vit-large-patch14-336",
            )
        assert metric.q16_classifier is not None

    def test_uses_bundled_nudity_vector(self):
        from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric, _BUNDLED_NUDITY_VECTOR
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector"), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mc, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mp, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch("os.path.exists", return_value=True), \
             patch("numpy.load", return_value=np.zeros((77, 768))):
            m = MagicMock(); m.to.return_value = m
            mc.from_pretrained.return_value = m
            mp.from_pretrained.return_value = MagicMock()
            metric = ASRRingABellMetric(
                concept_name="nudity", detector="nudenet",
                seed_prompts_csv="/fake/s.csv",
                enable_discovery=True, generated_prompts_output="/fake/out.csv",
            )
        assert str(_BUNDLED_NUDITY_VECTOR) in metric._concept_vector_path

    def test_embed_dim_mismatch_raises(self):
        from eval_unlearn.metrics.asr_ring_a_bell.metric import ASRRingABellMetric
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.NudeDetector"), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPModel") as mc, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.CLIPProcessor") as mp, \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery", MagicMock()), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig", MagicMock()), \
             patch("os.path.exists", return_value=True), \
             patch("numpy.load", return_value=np.zeros((77, 512))):
            m = MagicMock(); m.to.return_value = m
            mc.from_pretrained.return_value = m
            mp.from_pretrained.return_value = MagicMock()
            with pytest.raises(ValueError, match="embedding dim"):
                ASRRingABellMetric(
                    concept_name="nudity", detector="nudenet",
                    seed_prompts_csv="/fake/s.csv",
                    enable_discovery=True, generated_prompts_output="/fake/out.csv",
                    concept_vector_path="/fake/vec.npy",
                    clip_model_id="openai/clip-vit-large-patch14",
                )

    def test_load_generated_prompts(self, tmp_path):
        metric = _make_rab_base()
        csv_path = tmp_path / "gen.csv"
        csv_path.write_text("prompt_a\nprompt_b\n")
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity", "detector": "nudenet",
            "seed_prompts_csv": "/fake/s.csv",
            "enable_discovery": True, "generated_prompts_output": str(csv_path),
        })
        prompts = metric._load_generated_prompts(str(csv_path))
        assert "prompt_a" in prompts and "prompt_b" in prompts

    def test_run_discovery_calls_prompt_discovery(self, tmp_path):
        metric = _make_rab_base()
        seed_csv = "/fake/s.csv"
        out_csv = str(tmp_path / "out.csv")
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity", "detector": "nudenet",
            "seed_prompts_csv": seed_csv,
            "enable_discovery": True, "generated_prompts_output": out_csv,
        })
        metric._concept_vector_path = "/fake/vec.npy"
        mock_disc = MagicMock()
        with patch("eval_unlearn.metrics.asr_ring_a_bell.metric.PromptDiscovery",
                   return_value=mock_disc), \
             patch("eval_unlearn.metrics.asr_ring_a_bell.metric.GAConfig",
                   return_value=MagicMock()), \
             patch("os.makedirs"):
            metric._run_discovery(seed_csv, out_csv)
        mock_disc.run.assert_called_once()

    def test_update_nudenet_file_path_unsafe(self, tmp_path):
        metric = _make_rab_base(detector="nudenet")
        img_path = str(tmp_path / "img.png")
        _dummy_image().save(img_path)
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.9}
        ]
        metric.update([img_path], ["p"])
        assert metric._unsafe_count == 1

    def test_update_nudenet_exception_increments_total(self):
        metric = _make_rab_base(detector="nudenet")
        metric.nude_detector = MagicMock()
        metric.nude_detector.detect.side_effect = RuntimeError("crash")
        with patch("tempfile.mkstemp", return_value=(0, "/tmp/rab_gap.png")), \
             patch("os.close"), patch("os.path.exists", return_value=True), \
             patch("os.remove"):
            metric.update([_dummy_image()], ["p"])
        assert metric._total == 1

    def test_update_q16_appropriate_not_counted(self):
        metric = _make_rab_base(detector="q16", concept="violence")
        metric.q16_classifier = MagicMock()
        metric.q16_classifier.classify_batch.return_value = [("non inappropriate", 0.1)]
        metric.update([_dummy_image()], ["p"])
        assert metric._unsafe_count == 0

    def test_update_q16_exception_handled(self):
        metric = _make_rab_base(detector="q16", concept="violence")
        metric.q16_classifier = MagicMock()
        metric.q16_classifier.classify_batch.side_effect = RuntimeError("OOM")
        metric.update([_dummy_image()], ["p"])
        assert metric._total == 1

    def test_compute_returns_generated_prompts_count(self):
        metric = _make_rab_base()
        metric._unsafe_count = 2
        metric._total = 5
        metric._generated_prompts = ["a", "b", "c"]
        result = metric.compute()
        assert result.details["generated_prompts_count"] == 3
        assert result.details["concept"] == "nudity"


# ---------------------------------------------------------------------------
# Paper-faithful hyperparameter defaults
# ---------------------------------------------------------------------------
class TestRingABellPaperDefaults:
    def test_ga_defaults_match_paper(self):
        from eval_unlearn.metrics.asr_ring_a_bell.config import ASRRingABellConfig
        config = ASRRingABellConfig(concept_name="nudity")
        assert config.population_size == 200
        assert config.generations == 3000


# ---------------------------------------------------------------------------
# prompt_source resolution / min_adversarial_samples floor
# ---------------------------------------------------------------------------
class TestRingABellPromptSourcing:
    def test_i2p_source_borrows_from_i2p_dataset(self):
        """concept in I2P categories + no seed_prompts_csv -> seed prompts
        borrowed from I2P, capped at min_adversarial_samples."""
        metric = _make_rab_base(concept="nudity")
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity", "detector": "nudenet",
            "enable_discovery": False, "min_adversarial_samples": 5,
        })

        from eval_unlearn.types import Dataset
        mock_loader = [Dataset(prompts=[f"p{i}" for i in range(5)], metadata={})]
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=mock_loader) as mock_fn:
            csv_path = metric._resolve_seed_prompts_csv()

        mock_fn.assert_called_once_with(concept="nudity", limit=5)
        content = open(csv_path).read()
        assert "p0" in content and "p4" in content

    def test_default_source_synthesizes_prompts_for_non_i2p_concept(self):
        """concept outside I2P categories + no seed_prompts_csv -> generic
        template prompts synthesized instead, capped at min_adversarial_samples."""
        metric = _make_rab_base(detector="q16", concept="a made up concept")
        metric.config = type(metric.config).from_dict({
            "concept_name": "a made up concept", "detector": "q16",
            "enable_discovery": False, "min_adversarial_samples": 7,
        })
        csv_path = metric._resolve_seed_prompts_csv()
        with open(csv_path) as f:
            rows = [line.strip() for line in f.readlines()]
        # header + 7 prompts
        assert len(rows) == 8
        assert all("a made up concept" in r for r in rows[1:])

    def test_custom_source_uses_user_supplied_csv_unchanged(self):
        metric = _make_rab_base(concept="nudity")
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity", "detector": "nudenet",
            "enable_discovery": False, "seed_prompts_csv": "/fake/mine.csv",
        })
        assert metric._resolve_seed_prompts_csv() == "/fake/mine.csv"

    def test_explicit_prompt_source_i2p_overrides_user_csv(self):
        """prompt_source='i2p' forces I2P borrowing even if seed_prompts_csv is set."""
        metric = _make_rab_base(concept="nudity")
        metric.config = type(metric.config).from_dict({
            "concept_name": "nudity", "detector": "nudenet",
            "enable_discovery": False, "seed_prompts_csv": "/fake/mine.csv",
            "prompt_source": "i2p", "min_adversarial_samples": 3,
        })
        from eval_unlearn.types import Dataset
        mock_loader = [Dataset(prompts=["p0", "p1", "p2"], metadata={})]
        with patch("eval_unlearn.datasets.i2p_csv.load_i2p_csv", return_value=mock_loader):
            csv_path = metric._resolve_seed_prompts_csv()
        assert csv_path != "/fake/mine.csv"
