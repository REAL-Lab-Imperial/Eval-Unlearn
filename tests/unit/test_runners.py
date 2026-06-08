"""Unit tests for SingleBenchmarkRunner and MultiBenchmarkRunner."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from eval_learn.types import Dataset, MetricResult


def _dummy_image():
    return Image.new("RGB", (8, 8))


def _make_loader(prompts=("a cat",), metadata=None):
    """Return a fake DataLoader that yields one Dataset batch."""
    batch = Dataset(prompts=list(prompts), metadata=metadata or {"source": "test"})
    return [batch]


def _mock_technique(images=None):
    tech = MagicMock()
    tech.generate.return_value = images or [_dummy_image()]
    return tech


def _mock_metric(loader=None, result=None):
    m = MagicMock()
    m.load_dataset.return_value = loader or _make_loader()
    m.compute.return_value = result or MetricResult(name="ASR", value=0.5, details={})
    return m


# ---------------------------------------------------------------------------
# generate_run_id
# ---------------------------------------------------------------------------
class TestGenerateRunId:
    def test_returns_8_char_hex(self):
        from eval_learn.runners.single_benchmark_runner import generate_run_id
        rid = generate_run_id("esd", {}, "fid", {}, 1234567890.0)
        assert len(rid) == 8
        assert all(c in "0123456789abcdef" for c in rid)

    def test_deterministic(self):
        from eval_learn.runners.single_benchmark_runner import generate_run_id
        a = generate_run_id("esd", {}, "fid", {}, 1.0)
        b = generate_run_id("esd", {}, "fid", {}, 1.0)
        assert a == b

    def test_different_inputs_differ(self):
        from eval_learn.runners.single_benchmark_runner import generate_run_id
        a = generate_run_id("esd", {}, "fid", {}, 1.0)
        b = generate_run_id("mace", {}, "fid", {}, 1.0)
        assert a != b


# ---------------------------------------------------------------------------
# generate_multi_run_id
# ---------------------------------------------------------------------------
class TestGenerateMultiRunId:
    def test_returns_8_char_hex(self):
        from eval_learn.runners.multi_benchmark_runner import generate_multi_run_id
        rid = generate_multi_run_id("esd", {}, ["fid"], {}, "multi", 1.0)
        assert len(rid) == 8

    def test_deterministic(self):
        from eval_learn.runners.multi_benchmark_runner import generate_multi_run_id
        a = generate_multi_run_id("esd", {}, ["fid", "asr_i2p"], {}, "multi", 1.0)
        b = generate_multi_run_id("esd", {}, ["asr_i2p", "fid"], {}, "multi", 1.0)
        assert a == b  # metric_names are sorted


# ---------------------------------------------------------------------------
# SingleBenchmarkRunner
# ---------------------------------------------------------------------------
class TestSingleBenchmarkRunner:

    def _make_runner(self, tech_factory, metric_factory, output_dir="/tmp/test_run"):
        from eval_learn.runners.single_benchmark_runner import SingleBenchmarkRunner

        with patch("eval_learn.runners.single_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.single_benchmark_runner.get_technique", return_value=tech_factory), \
             patch("eval_learn.runners.single_benchmark_runner.get_metric", return_value=metric_factory), \
             patch("eval_learn.runners.single_benchmark_runner.validate_technique_metric_pair"):
            runner = SingleBenchmarkRunner(
                technique_name="esd",
                metric_name="fid",
                technique_config={"erase_concept": "nudity"},
                metric_config={},
                output_dir=output_dir,
                seed=42,
            )
        return runner

    def test_run_returns_report(self, tmp_path):
        tech = _mock_technique()
        metric = _mock_metric()
        runner = self._make_runner(
            tech_factory=MagicMock(return_value=tech),
            metric_factory=MagicMock(return_value=metric),
            output_dir=str(tmp_path),
        )
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")):
            report = runner.run()
        assert "run_id" in report
        assert "metric_result" in report
        assert report["technique_name"] == "esd"

    def test_run_calls_generate_and_update(self, tmp_path):
        tech = _mock_technique(images=[_dummy_image(), _dummy_image()])
        metric = _mock_metric(loader=_make_loader(prompts=["p1", "p2"]))
        runner = self._make_runner(
            tech_factory=MagicMock(return_value=tech),
            metric_factory=MagicMock(return_value=metric),
            output_dir=str(tmp_path),
        )
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")):
            runner.run()
        tech.generate.assert_called_once()
        metric.update.assert_called_once()
        metric.compute.assert_called_once()

    def test_run_with_categories_in_metadata(self, tmp_path):
        tech = _mock_technique(images=[_dummy_image()])
        metadata = {"source": "test", "categories": ["target"]}
        metric = _mock_metric(loader=_make_loader(prompts=["p"], metadata=metadata))
        runner = self._make_runner(
            tech_factory=MagicMock(return_value=tech),
            metric_factory=MagicMock(return_value=metric),
            output_dir=str(tmp_path),
        )
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")):
            report = runner.run()
        assert report is not None

    def test_validation_error_raises_value_error(self):
        from eval_learn.runners.single_benchmark_runner import SingleBenchmarkRunner
        from eval_learn.runners.validation import ValidationError

        with patch("eval_learn.runners.single_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.single_benchmark_runner.get_technique", return_value=MagicMock()), \
             patch("eval_learn.runners.single_benchmark_runner.get_metric", return_value=MagicMock()), \
             patch("eval_learn.runners.single_benchmark_runner.validate_technique_metric_pair",
                   side_effect=ValidationError("incompatible")):
            with pytest.raises(ValueError, match="incompatible"):
                SingleBenchmarkRunner(
                    technique_name="esd",
                    metric_name="err",
                    technique_config={"erase_concept": "violence"},
                )

    def test_unknown_technique_raises(self):
        from eval_learn.runners.single_benchmark_runner import SingleBenchmarkRunner

        with patch("eval_learn.runners.single_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.single_benchmark_runner.get_technique",
                   side_effect=ValueError("not found")):
            with pytest.raises(ValueError, match="not found"):
                SingleBenchmarkRunner(
                    technique_name="nonexistent",
                    metric_name="fid",
                )


# ---------------------------------------------------------------------------
# MultiBenchmarkRunner
# ---------------------------------------------------------------------------
class TestMultiBenchmarkRunner:

    def _make_runner(self, tech_factory, metric_factories, output_dir="/tmp/test_multi"):
        from eval_learn.runners.multi_benchmark_runner import MultiBenchmarkRunner

        def _get_metric(name):
            return metric_factories[name]

        with patch("eval_learn.runners.multi_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.multi_benchmark_runner.get_technique", return_value=tech_factory), \
             patch("eval_learn.runners.multi_benchmark_runner.get_metric", side_effect=_get_metric), \
             patch("eval_learn.runners.multi_benchmark_runner.validate_technique_metric_pair"):
            runner = MultiBenchmarkRunner(
                technique_name="esd",
                metric_names=list(metric_factories.keys()),
                technique_config={"erase_concept": "nudity"},
                metric_configs={},
                output_dir=output_dir,
                seed=0,
            )
        return runner

    def test_run_returns_report_with_all_metrics(self, tmp_path):
        tech = _mock_technique()
        m1 = _mock_metric(result=MetricResult(name="FID", value=10.0, details={}))
        m2 = _mock_metric(result=MetricResult(name="ASR", value=0.3, details={}))
        factories = {"fid": MagicMock(return_value=m1), "asr_i2p": MagicMock(return_value=m2)}
        runner = self._make_runner(
            tech_factory=MagicMock(return_value=tech),
            metric_factories=factories,
            output_dir=str(tmp_path),
        )
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")):
            report = runner.run()
        assert "fid" in report["metric_results"]
        assert "asr_i2p" in report["metric_results"]

    def test_duplicate_metric_names_raises(self):
        from eval_learn.runners.multi_benchmark_runner import MultiBenchmarkRunner
        with patch("eval_learn.runners.multi_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.multi_benchmark_runner.get_technique", return_value=MagicMock()), \
             patch("eval_learn.runners.multi_benchmark_runner.get_metric", return_value=MagicMock()), \
             patch("eval_learn.runners.multi_benchmark_runner.validate_technique_metric_pair"):
            with pytest.raises(ValueError, match="duplicates"):
                MultiBenchmarkRunner(
                    technique_name="esd",
                    metric_names=["fid", "fid"],
                )

    def test_empty_metric_names_raises(self):
        from eval_learn.runners.multi_benchmark_runner import MultiBenchmarkRunner
        with patch("eval_learn.runners.multi_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.multi_benchmark_runner.get_technique", return_value=MagicMock()), \
             patch("eval_learn.runners.multi_benchmark_runner.validate_technique_metric_pair"):
            with pytest.raises(ValueError, match="must not be empty"):
                MultiBenchmarkRunner(
                    technique_name="esd",
                    metric_names=[],
                )

    def test_each_metric_gets_its_own_loader(self, tmp_path):
        tech = _mock_technique()
        m1 = _mock_metric()
        m2 = _mock_metric()
        factories = {"fid": MagicMock(return_value=m1), "clip_score": MagicMock(return_value=m2)}
        runner = self._make_runner(
            tech_factory=MagicMock(return_value=tech),
            metric_factories=factories,
            output_dir=str(tmp_path),
        )
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")):
            runner.run()
        m1.load_dataset.assert_called_once()
        m2.load_dataset.assert_called_once()


# ---------------------------------------------------------------------------
# MultiBenchmarkRunner — categories in metadata + cuda cleanup + list values
# ---------------------------------------------------------------------------
class TestMultiBenchmarkRunnerCoverageGaps:
    def _make_runner(self, tmp_path, loader=None):
        from eval_learn.runners.multi_benchmark_runner import MultiBenchmarkRunner
        tech = _mock_technique()
        metric = _mock_metric(loader=loader)
        factories = {"asr_i2p": MagicMock(return_value=metric)}
        with patch("eval_learn.runners.multi_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.multi_benchmark_runner.get_technique",
                   return_value=MagicMock(return_value=tech)), \
             patch("eval_learn.runners.multi_benchmark_runner.get_metric",
                   side_effect=lambda n: factories[n]), \
             patch("eval_learn.runners.multi_benchmark_runner.validate_technique_metric_pair"):
            return MultiBenchmarkRunner(
                technique_name="esd",
                metric_names=["asr_i2p"],
                technique_config={"erase_concept": "nudity"},
                output_dir=str(tmp_path),
            )

    def test_categories_in_metadata_updates_counters(self, tmp_path):
        loader = _make_loader(
            metadata={"source": "t", "categories": ["target"], "concepts": ["nudity"]}
        )
        runner = self._make_runner(tmp_path, loader=loader)
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")):
            report = runner.run()
        assert "asr_i2p" in report["metric_results"]

    def test_cuda_cleanup_exception_handled(self, tmp_path):
        runner = self._make_runner(tmp_path)
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")), \
             patch("torch.cuda.empty_cache", side_effect=RuntimeError("no GPU")):
            report = runner.run()
        assert report is not None

    def test_list_metadata_values_accumulated(self, tmp_path):
        loader = _make_loader(metadata={"source": "t", "qa_pairs": [["q1", "q2"]]})
        runner = self._make_runner(tmp_path, loader=loader)
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")):
            report = runner.run()
        assert report is not None

    def test_validation_error_raises_value_error(self, tmp_path):
        from eval_learn.runners.multi_benchmark_runner import MultiBenchmarkRunner
        from eval_learn.runners.validation import ValidationError

        def raise_val(**kw):
            raise ValidationError("incompatible metric for technique")

        with patch("eval_learn.runners.multi_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.multi_benchmark_runner.get_technique",
                   return_value=MagicMock()), \
             patch("eval_learn.runners.multi_benchmark_runner.get_metric",
                   return_value=MagicMock()), \
             patch("eval_learn.runners.multi_benchmark_runner.validate_technique_metric_pair",
                   side_effect=raise_val):
            with pytest.raises(ValueError, match="incompatible"):
                MultiBenchmarkRunner(
                    technique_name="esd",
                    metric_names=["asr_i2p"],
                    output_dir=str(tmp_path),
                )


# ---------------------------------------------------------------------------
# SingleBenchmarkRunner — cuda cleanup exception
# ---------------------------------------------------------------------------
class TestSingleBenchmarkRunnerCoverageGaps:
    def _make_single_runner(self, tmp_path):
        from eval_learn.runners.single_benchmark_runner import SingleBenchmarkRunner
        tech = _mock_technique()
        metric = _mock_metric()
        with patch("eval_learn.runners.single_benchmark_runner.load_entrypoints"), \
             patch("eval_learn.runners.single_benchmark_runner.get_technique",
                   return_value=MagicMock(return_value=tech)), \
             patch("eval_learn.runners.single_benchmark_runner.get_metric",
                   return_value=MagicMock(return_value=metric)), \
             patch("eval_learn.runners.single_benchmark_runner.validate_technique_metric_pair"):
            return SingleBenchmarkRunner(
                technique_name="esd",
                metric_name="asr_i2p",
                technique_config={"erase_concept": "nudity"},
                output_dir=str(tmp_path),
            )

    def test_cuda_cleanup_exception_handled(self, tmp_path):
        runner = self._make_single_runner(tmp_path)
        with patch.object(runner.writer, "save_run", return_value=str(tmp_path / "r.json")), \
             patch("torch.cuda.empty_cache", side_effect=RuntimeError("no GPU")):
            report = runner.run()
        assert report is not None


# ---------------------------------------------------------------------------
# BaseRunner — abstract pass (concrete subclass test)
# ---------------------------------------------------------------------------
class TestBaseRunnerCoverageGaps:
    def test_concrete_subclass_run(self, tmp_path):
        from eval_learn.runners.core.base_runner import BaseRunner

        class ConcreteRunner(BaseRunner):
            def run(self):
                return {"ok": True}

        assert ConcreteRunner(output_dir=str(tmp_path)).run()["ok"] is True

    def test_base_runner_run_returns_none_via_super(self, tmp_path):
        from eval_learn.runners.core.base_runner import BaseRunner

        class ForwardingRunner(BaseRunner):
            def run(self):
                return super().run()

        result = ForwardingRunner(output_dir=str(tmp_path)).run()
        assert result is None
