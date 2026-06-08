"""Unit tests for runners/core/base_runner.py."""
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


class TestBaseRunnerBuildBaseReport:
    def _make_concrete_runner(self, output_dir="/tmp/test"):
        from eval_learn.runners.core.base_runner import BaseRunner
        from eval_learn.artifacts import ArtifactWriter

        class ConcreteRunner(BaseRunner):
            def run(self):
                return {}

        with patch.object(ArtifactWriter, "__init__", return_value=None):
            runner = ConcreteRunner(output_dir=output_dir)
        runner.writer = MagicMock()
        return runner

    def test_build_base_report_has_run_id(self):
        runner = self._make_concrete_runner()
        report = runner._build_base_report(run_id="abc123", timestamp=1234567.0)
        assert report["run_id"] == "abc123"

    def test_build_base_report_has_timestamp(self):
        runner = self._make_concrete_runner()
        report = runner._build_base_report(run_id="abc", timestamp=9999.5)
        assert report["timestamp"] == 9999.5

    def test_build_base_report_extra_kwargs(self):
        runner = self._make_concrete_runner()
        report = runner._build_base_report(
            run_id="x", timestamp=1.0,
            technique_name="esd", metric_result={"value": 0.5}
        )
        assert report["technique_name"] == "esd"
        assert report["metric_result"]["value"] == 0.5

    def test_log_phase(self):
        runner = self._make_concrete_runner()
        # Should not raise
        runner._log_phase("Loading dataset")
        runner._log_phase("Generating images")


class TestBaseRunnerResolveMMAClipModel:
    def _make_runner(self):
        from eval_learn.runners.core.base_runner import BaseRunner
        from eval_learn.artifacts import ArtifactWriter

        class ConcreteRunner(BaseRunner):
            def run(self):
                return {}

        with patch.object(ArtifactWriter, "__init__", return_value=None):
            runner = ConcreteRunner()
        runner.writer = MagicMock()
        return runner

    def test_no_mma_metric_returns_unchanged(self):
        runner = self._make_runner()
        configs = {"fid": {}, "asr_i2p": {}}
        result = runner._resolve_mma_clip_model(configs, "esd", {})
        assert result is configs

    def test_mma_with_known_technique_injects_encoder(self):
        runner = self._make_runner()
        configs = {"asr_mma_diffusion": {"concept_name": "nudity"}}
        with patch("eval_learn.techniques._base_models.get_technique_base_model_id",
                   return_value="CompVis/stable-diffusion-v1-4"), \
             patch("eval_learn.metrics._clip_constants.clip_encoder_for_sd",
                   return_value="openai/clip-vit-large-patch14"):
            result = runner._resolve_mma_clip_model(configs, "esd", {})
        assert "clip_model_id" in result["asr_mma_diffusion"]

    def test_mma_unknown_base_model_returns_unchanged(self):
        runner = self._make_runner()
        configs = {"asr_mma_diffusion": {}}
        with patch("eval_learn.techniques._base_models.get_technique_base_model_id",
                   return_value=None):
            result = runner._resolve_mma_clip_model(configs, "free_run", {})
        # When model_id is None, original config returned unchanged
        assert result is configs

    def test_mma_unsupported_model_raises(self):
        runner = self._make_runner()
        configs = {"asr_mma_diffusion": {}}
        with patch("eval_learn.techniques._base_models.get_technique_base_model_id",
                   return_value="unsupported/model"), \
             patch("eval_learn.metrics._clip_constants.clip_encoder_for_sd",
                   side_effect=ValueError("unsupported model")):
            with pytest.raises(ValueError, match="asr_mma_diffusion cannot be used"):
                runner._resolve_mma_clip_model(configs, "esd", {})

    def test_mma_free_run_with_model_id(self):
        runner = self._make_runner()
        configs = {"asr_mma_diffusion": {}}
        with patch("eval_learn.techniques._base_models.get_technique_base_model_id",
                   return_value="CompVis/stable-diffusion-v1-4"), \
             patch("eval_learn.metrics._clip_constants.clip_encoder_for_sd",
                   return_value="openai/clip-vit-large-patch14"):
            result = runner._resolve_mma_clip_model(
                configs, "free_run", {"model_id": "CompVis/stable-diffusion-v1-4"}
            )
        assert "clip_model_id" in result["asr_mma_diffusion"]
