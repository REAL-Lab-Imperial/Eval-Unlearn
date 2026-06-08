"""Unit tests for _base_models in techniques and metrics."""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# techniques/_base_models.py
# ---------------------------------------------------------------------------
class TestTechniqueBaseModels:
    def test_get_technique_base_model_id_fixed(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id, TECHNIQUE_BASE_MODELS
        for name in TECHNIQUE_BASE_MODELS:
            result = get_technique_base_model_id(name, {})
            assert result == TECHNIQUE_BASE_MODELS[name]

    def test_get_technique_base_model_id_free_run_with_model_id(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id
        result = get_technique_base_model_id("free_run", {"model_id": "some/model"})
        assert result == "some/model"

    def test_get_technique_base_model_id_free_run_no_model_id(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id
        result = get_technique_base_model_id("free_run", {})
        assert result is None

    def test_get_technique_base_model_id_unknown_technique(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id
        result = get_technique_base_model_id("nonexistent_technique", {})
        assert result is None

    def test_technique_base_models_dict(self):
        from eval_learn.techniques._base_models import TECHNIQUE_BASE_MODELS
        assert "esd" in TECHNIQUE_BASE_MODELS
        assert "sld" in TECHNIQUE_BASE_MODELS
        assert "uce" in TECHNIQUE_BASE_MODELS
        assert "mace" in TECHNIQUE_BASE_MODELS


# ---------------------------------------------------------------------------
# metrics/_base_models.py
# ---------------------------------------------------------------------------
class TestMetricBaseModels:
    def test_metric_models_dict(self):
        from eval_learn.metrics._base_models import METRIC_MODELS
        assert "asr_i2p" in METRIC_MODELS
        assert "fid" in METRIC_MODELS
        assert "clip_score" in METRIC_MODELS
        assert "tifa" in METRIC_MODELS

    def test_metric_model_info_has_required_fields(self):
        from eval_learn.metrics._base_models import METRIC_MODELS, MetricModelInfo
        for name, info in METRIC_MODELS.items():
            assert isinstance(info, MetricModelInfo)
            assert isinstance(info.model, str)
            assert isinstance(info.configurable, bool)

    def test_fid_not_configurable(self):
        from eval_learn.metrics._base_models import METRIC_MODELS
        fid_info = METRIC_MODELS["fid"]
        assert fid_info.configurable is False

    def test_clip_score_configurable(self):
        from eval_learn.metrics._base_models import METRIC_MODELS
        clip_info = METRIC_MODELS["clip_score"]
        assert clip_info.configurable is True
        assert clip_info.config_field == "clip_model_name"

    def test_configurable_metrics_have_config_field(self):
        from eval_learn.metrics._base_models import METRIC_MODELS
        for name, info in METRIC_MODELS.items():
            if info.configurable:
                assert info.config_field is not None, f"{name} configurable but no config_field"
