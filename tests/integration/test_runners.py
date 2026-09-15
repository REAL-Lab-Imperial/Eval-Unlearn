"""
Integration tests for SingleBenchmarkRunner, MultiBenchmarkRunner, and BaseRunner.

Uses a lightweight dummy technique + fully-mocked Q16/NudeNet classifiers to run
the full pipeline without network access or GPU requirements.
"""
import json
import os
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.integration


def _dummy_image():
    return Image.new("RGB", (32, 32), color=(100, 120, 140))


def _make_q16_mock():
    """Return a MagicMock shaped like Q16Classifier."""
    mock = MagicMock()
    mock.classify_batch.return_value = [False, False]
    return mock


# ---------------------------------------------------------------------------
# Dummy technique registered for all runner tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _register_dummy_technique():
    """Register a minimal technique that returns blank images."""
    from eval_unlearn.registry.local import _TECHNIQUES

    class _DummyRunnerTechnique:
        def __init__(self, erase_concept="nudity", device="cpu", **kwargs):
            self.erase_concept = erase_concept

        def generate(self, prompts, seed=None, **kwargs):
            return [_dummy_image() for _ in prompts]

    _TECHNIQUES["_dummy_runner"] = _DummyRunnerTechnique
    yield
    _TECHNIQUES.pop("_dummy_runner", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompts_csv(tmp_path, rows=None, filename="prompts.csv"):
    rows = rows or ["adv_prompt_1,target_1", "adv_prompt_2,target_2"]
    path = tmp_path / filename
    path.write_text("adversarial_prompt,target_prompt\n" + "\n".join(rows) + "\n")
    return str(path)


def _run_single(prompts_csv, output_dir):
    """Build and run a SingleBenchmarkRunner with fully mocked metric internals."""
    q16 = _make_q16_mock()
    with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
         patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", return_value=q16):
        from eval_unlearn.runners.single_benchmark_runner import SingleBenchmarkRunner
        runner = SingleBenchmarkRunner(
            technique_name="_dummy_runner",
            metric_name="asr_p4d",
            technique_config={"erase_concept": "violence", "device": "cpu"},
            metric_config={
                "concept_name": "violence",
                "detector": "q16",
                "erase_id": "std",
                "device": "cpu",
                "precomputed_prompts_path": prompts_csv,
            },
            output_dir=output_dir,
            seed=42,
        )
        return runner.run()


# ===========================================================================
# SingleBenchmarkRunner
# ===========================================================================

class TestSingleBenchmarkRunnerIntegration:

    def test_run_returns_report_dict(self, tmp_path):
        report = _run_single(_make_prompts_csv(tmp_path), str(tmp_path / "out"))
        assert isinstance(report, dict)
        assert "run_id" in report
        assert "timestamp" in report
        assert report["technique_name"] == "_dummy_runner"
        assert report["metric_name"] == "asr_p4d"

    def test_run_metric_value_in_range(self, tmp_path):
        report = _run_single(_make_prompts_csv(tmp_path), str(tmp_path / "out2"))
        assert 0.0 <= report["metric_result"]["value"] <= 1.0

    def test_run_saves_json_report(self, tmp_path):
        output_dir = str(tmp_path / "out3")
        _run_single(_make_prompts_csv(tmp_path), output_dir)
        json_files = []
        for root, _, files in os.walk(output_dir):
            json_files += [f for f in files if f.endswith(".json")]
        assert len(json_files) >= 1

    def test_run_saves_images(self, tmp_path):
        output_dir = str(tmp_path / "out4")
        _run_single(_make_prompts_csv(tmp_path), output_dir)
        png_files = []
        for root, _, files in os.walk(output_dir):
            png_files += [f for f in files if f.endswith(".png")]
        assert len(png_files) == 2  # one per prompt

    def test_run_erase_concept_in_report(self, tmp_path):
        report = _run_single(_make_prompts_csv(tmp_path), str(tmp_path / "out5"))
        assert report.get("erase_concept") == "violence"

    def test_invalid_technique_raises(self, tmp_path):
        from eval_unlearn.runners.single_benchmark_runner import SingleBenchmarkRunner
        with pytest.raises(ValueError, match="not found"):
            SingleBenchmarkRunner(
                technique_name="nonexistent_technique_xyz",
                metric_name="asr_p4d",
                output_dir=str(tmp_path),
            )

    def test_invalid_metric_raises(self, tmp_path):
        from eval_unlearn.runners.single_benchmark_runner import SingleBenchmarkRunner
        with pytest.raises(ValueError, match="not found"):
            SingleBenchmarkRunner(
                technique_name="_dummy_runner",
                metric_name="nonexistent_metric_xyz",
                output_dir=str(tmp_path),
            )

    def test_err_metric_non_nudity_raises(self, tmp_path):
        from eval_unlearn.runners.single_benchmark_runner import SingleBenchmarkRunner
        with pytest.raises(ValueError, match="nudity-specific"):
            SingleBenchmarkRunner(
                technique_name="_dummy_runner",
                metric_name="err",
                technique_config={"erase_concept": "violence"},
                output_dir=str(tmp_path),
            )

    def test_ua_ira_missing_paths_raises(self, tmp_path):
        from eval_unlearn.runners.single_benchmark_runner import SingleBenchmarkRunner
        with pytest.raises(ValueError, match="target_prompts_path"):
            SingleBenchmarkRunner(
                technique_name="_dummy_runner",
                metric_name="ua_ira",
                technique_config={"erase_concept": "nudity"},
                metric_config={},
                output_dir=str(tmp_path),
            )


# ===========================================================================
# MultiBenchmarkRunner
# ===========================================================================

class TestMultiBenchmarkRunnerIntegration:

    def test_run_returns_report_with_all_metrics(self, tmp_path):
        p4d_csv_path = tmp_path / "p4d.csv"
        p4d_csv_path.write_text("adversarial_prompt,target_prompt\nadv_a,target_a\n")

        mma_csv_path = tmp_path / "mma.csv"
        mma_csv_path.write_text("adversarial_prompt,target_prompt\nmma_adv1,mma_t1\n")

        q16 = _make_q16_mock()
        nude_mock = MagicMock()
        nude_mock.detect.return_value = []

        with patch("eval_unlearn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
             patch("eval_unlearn.metrics.asr_p4d.metric.Q16Classifier", return_value=q16), \
             patch("eval_unlearn.metrics.asr_mma_diffusion.metric.NudeDetector", return_value=nude_mock):
            from eval_unlearn.runners.multi_benchmark_runner import MultiBenchmarkRunner
            runner = MultiBenchmarkRunner(
                technique_name="_dummy_runner",
                metric_names=["asr_p4d", "asr_mma_diffusion"],
                technique_config={"erase_concept": "violence"},
                metric_configs={
                    "asr_p4d": {
                        "concept_name": "violence",
                        "detector": "q16",
                        "erase_id": "std",
                        "device": "cpu",
                        "precomputed_prompts_path": str(p4d_csv_path),
                    },
                    "asr_mma_diffusion": {
                        "concept_name": "nudity",
                        "output_csv": str(tmp_path / "mma_out.csv"),
                        "detector": "nudenet",
                        "precomputed_prompts_path": str(mma_csv_path),
                    },
                },
                output_dir=str(tmp_path / "out"),
            )
            report = runner.run()

        assert "run_id" in report
        assert "metric_results" in report
        assert "asr_p4d" in report["metric_results"]
        assert "asr_mma_diffusion" in report["metric_results"]

    def test_duplicate_metric_names_raises(self, tmp_path):
        from eval_unlearn.runners.multi_benchmark_runner import MultiBenchmarkRunner
        with pytest.raises(ValueError, match="duplicates"):
            MultiBenchmarkRunner(
                technique_name="_dummy_runner",
                metric_names=["asr_p4d", "asr_p4d"],
                output_dir=str(tmp_path),
            )

    def test_empty_metric_names_raises(self, tmp_path):
        from eval_unlearn.runners.multi_benchmark_runner import MultiBenchmarkRunner
        with pytest.raises(ValueError, match="must not be empty"):
            MultiBenchmarkRunner(
                technique_name="_dummy_runner",
                metric_names=[],
                output_dir=str(tmp_path),
            )


# ===========================================================================
# BaseRunner._resolve_mma_clip_model
# ===========================================================================

class TestResolveMMAClipModel:

    def test_no_mma_returns_unchanged(self):
        from eval_unlearn.runners.core.base_runner import BaseRunner
        configs = {"asr_p4d": {"concept_name": "nudity"}}
        result = BaseRunner._resolve_mma_clip_model(configs, "ssd", {})
        assert result is configs

    def test_known_technique_resolves_encoder(self):
        from eval_unlearn.runners.core.base_runner import BaseRunner
        configs = {"asr_mma_diffusion": {"concept_name": "nudity", "output_csv": "/tmp/x.csv"}}
        result = BaseRunner._resolve_mma_clip_model(configs, "ssd", {})
        assert "clip_model_id" in result["asr_mma_diffusion"]

    def test_unknown_technique_logs_warning_returns_unchanged(self):
        from eval_unlearn.runners.core.base_runner import BaseRunner
        configs = {"asr_mma_diffusion": {"concept_name": "nudity", "output_csv": "/tmp/x.csv"}}
        result = BaseRunner._resolve_mma_clip_model(configs, "_dummy_runner", {})
        assert result is configs

    def test_free_run_with_model_id_resolves_encoder(self):
        from eval_unlearn.runners.core.base_runner import BaseRunner
        configs = {"asr_mma_diffusion": {"concept_name": "nudity", "output_csv": "/tmp/x.csv"}}
        result = BaseRunner._resolve_mma_clip_model(
            configs,
            "free_run",
            {"model_id": "CompVis/stable-diffusion-v1-4"},
        )
        assert "clip_model_id" in result["asr_mma_diffusion"]

    def test_incompatible_sd_model_raises(self):
        from eval_unlearn.runners.core.base_runner import BaseRunner
        configs = {"asr_mma_diffusion": {"concept_name": "nudity", "output_csv": "/tmp/x.csv"}}
        # Patch at the source module since base_runner does a local import
        with patch(
            "eval_unlearn.techniques._base_models.get_technique_base_model_id",
            return_value="unknown/sd-v99",
        ), patch(
            "eval_unlearn.metrics._clip_constants.clip_encoder_for_sd",
            side_effect=ValueError("unsupported model"),
        ):
            with pytest.raises(ValueError, match="asr_mma_diffusion cannot be used"):
                BaseRunner._resolve_mma_clip_model(configs, "some_technique", {})


# ===========================================================================
# generate_run_id helpers
# ===========================================================================

class TestGenerateRunId:

    def test_run_id_is_8_chars(self):
        from eval_unlearn.runners.single_benchmark_runner import generate_run_id
        rid = generate_run_id("ssd", {}, "asr_p4d", {}, 12345.0)
        assert len(rid) == 8

    def test_run_id_deterministic(self):
        from eval_unlearn.runners.single_benchmark_runner import generate_run_id
        rid1 = generate_run_id("ssd", {"a": 1}, "asr_p4d", {"b": 2}, 9.0)
        rid2 = generate_run_id("ssd", {"a": 1}, "asr_p4d", {"b": 2}, 9.0)
        assert rid1 == rid2

    def test_multi_run_id_is_8_chars(self):
        from eval_unlearn.runners.multi_benchmark_runner import generate_multi_run_id
        rid = generate_multi_run_id("ssd", {}, ["asr_p4d"], {"asr_p4d": {}}, "test", 1.0)
        assert len(rid) == 8
