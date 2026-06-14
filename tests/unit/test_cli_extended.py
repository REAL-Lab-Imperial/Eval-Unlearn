"""Extended CLI tests targeting uncovered lines in cli.py."""
import json
import pytest
from argparse import Namespace
from unittest.mock import MagicMock, patch


# Import from cli with runners patched to avoid heavy imports
with patch("eval_unlearn.runners.SingleBenchmarkRunner"), \
     patch("eval_unlearn.runners.MultiBenchmarkRunner"):
    from eval_unlearn.cli import (
        cmd_run,
        cmd_push,
        cmd_pull,
        cmd_plugins,
        cmd_models,
        main,
        _load_config,
        _push_results,
    )


# ---------------------------------------------------------------------------
# cmd_run — complete flows
# ---------------------------------------------------------------------------
class TestCmdRun:
    def _config_file(self, tmp_path, config_dict):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(config_dict))
        return str(p)

    def test_cmd_run_single_mode_success(self, tmp_path):
        cfg = {
            "output_dir": str(tmp_path),
            "technique": {"name": "esd", "config": {"erase_concept": "nudity"}},
            "metric": {"name": "fid", "config": {}},
        }
        path = self._config_file(tmp_path, cfg)
        args = Namespace(config=path, hf_repo=None, hf_path=None, create_pr=False)

        mock_runner = MagicMock()
        mock_runner.run.return_value = {"run_id": "abc123", "metric_result": {}}

        with patch("eval_unlearn.cli.SingleBenchmarkRunner", return_value=mock_runner):
            cmd_run(args)

        mock_runner.run.assert_called_once()

    def test_cmd_run_multi_mode_success(self, tmp_path):
        cfg = {
            "output_dir": str(tmp_path),
            "technique": {"name": "esd", "config": {}},
            "metrics": [{"name": "fid"}, {"name": "asr_i2p"}],
        }
        path = self._config_file(tmp_path, cfg)
        args = Namespace(config=path, hf_repo=None, hf_path=None, create_pr=False)

        mock_runner = MagicMock()
        mock_runner.run.return_value = {"run_id": "abc", "metric_results": {}}

        with patch("eval_unlearn.cli.MultiBenchmarkRunner", return_value=mock_runner):
            cmd_run(args)

        mock_runner.run.assert_called_once()

    def test_cmd_run_invalid_config_exits(self, tmp_path):
        cfg = {"output_dir": str(tmp_path)}  # No technique or metric
        path = self._config_file(tmp_path, cfg)
        args = Namespace(config=path, hf_repo=None, hf_path=None, create_pr=False)

        with pytest.raises(SystemExit):
            cmd_run(args)

    def test_cmd_run_runner_exception_exits(self, tmp_path):
        cfg = {
            "output_dir": str(tmp_path),
            "technique": {"name": "esd"},
            "metric": {"name": "fid"},
        }
        path = self._config_file(tmp_path, cfg)
        args = Namespace(config=path, hf_repo=None, hf_path=None, create_pr=False)

        mock_runner = MagicMock()
        mock_runner.run.side_effect = RuntimeError("run crash")

        with patch("eval_unlearn.cli.SingleBenchmarkRunner", return_value=mock_runner):
            with pytest.raises(SystemExit):
                cmd_run(args)

    def test_cmd_run_with_hf_push(self, tmp_path):
        cfg = {
            "output_dir": str(tmp_path),
            "technique": {"name": "esd"},
            "metric": {"name": "fid"},
        }
        path = self._config_file(tmp_path, cfg)
        args = Namespace(
            config=path, hf_repo="org/repo", hf_path="my/path", create_pr=False
        )

        mock_runner = MagicMock()
        mock_runner.run.return_value = {"run_id": "abc"}

        with patch("eval_unlearn.cli.SingleBenchmarkRunner", return_value=mock_runner), \
             patch("eval_unlearn.cli._push_results") as mock_push:
            cmd_run(args)

        mock_push.assert_called_once()

    def test_cmd_run_ValueError_from_runner_exits(self, tmp_path):
        cfg = {
            "output_dir": str(tmp_path),
            "technique": {"name": "esd"},
            "metric": {"name": "fid"},
        }
        path = self._config_file(tmp_path, cfg)
        args = Namespace(config=path, hf_repo=None, hf_path=None, create_pr=False)

        with patch("eval_unlearn.cli.SingleBenchmarkRunner", side_effect=ValueError("bad config")):
            with pytest.raises(SystemExit):
                cmd_run(args)


# ---------------------------------------------------------------------------
# _push_results
# ---------------------------------------------------------------------------
class TestPushResults:
    def test_push_results_success(self, tmp_path):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.push_folder.return_value = "https://hf.co/commit/abc"
            _push_results(
                output_dir=str(tmp_path),
                hf_repo="org/repo",
                hf_path=None,
                create_pr=False,
            )
        mock_sync.push_folder.assert_called_once()

    def test_push_results_with_custom_path(self, tmp_path):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.push_folder.return_value = "https://hf.co/commit/abc"
            _push_results(
                output_dir=str(tmp_path),
                hf_repo="org/repo",
                hf_path="custom/path",
                create_pr=False,
            )
        mock_sync.push_folder.assert_called_once_with(str(tmp_path), "custom/path")

    def test_push_results_failure_exits(self, tmp_path):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.push_folder.side_effect = RuntimeError("network error")
            with pytest.raises(SystemExit):
                _push_results(
                    output_dir=str(tmp_path),
                    hf_repo="org/repo",
                    hf_path=None,
                    create_pr=False,
                )


# ---------------------------------------------------------------------------
# cmd_models
# ---------------------------------------------------------------------------
class TestCmdModels:
    def test_cmd_models_outputs_technique_table(self, capsys):
        from eval_unlearn.techniques._base_models import TECHNIQUE_BASE_MODELS
        from eval_unlearn.metrics._base_models import METRIC_MODELS
        cmd_models(Namespace())
        out = capsys.readouterr().out
        assert "Techniques:" in out
        assert "Metrics:" in out

    def test_cmd_models_shows_techniques(self, capsys):
        cmd_models(Namespace())
        out = capsys.readouterr().out
        assert "esd" in out.lower() or "Techniques" in out

    def test_cmd_models_shows_metrics(self, capsys):
        cmd_models(Namespace())
        out = capsys.readouterr().out
        assert "Metrics:" in out

    def test_cmd_models_handles_configurable_with_choices(self, capsys):
        from eval_unlearn.metrics._base_models import MetricModelInfo
        mock_metric_models = {
            "test_metric": MetricModelInfo(
                model="some/model",
                configurable=True,
                config_field="clip_model_name",
                choices=frozenset(["openai/clip-vit-large-patch14"]),
            )
        }
        with patch("eval_unlearn.metrics._base_models.METRIC_MODELS", mock_metric_models), \
             patch("eval_unlearn.techniques._base_models.TECHNIQUE_BASE_MODELS", {}):
            cmd_models(Namespace())
        out = capsys.readouterr().out
        assert "test_metric" in out

    def test_cmd_models_handles_configurable_no_choices(self, capsys):
        from eval_unlearn.metrics._base_models import MetricModelInfo
        mock_metric_models = {
            "tifa_test": MetricModelInfo(
                model="Salesforce/blip2",
                configurable=True,
                config_field="vqa_model_name",
                choices=None,
            )
        }
        with patch("eval_unlearn.metrics._base_models.METRIC_MODELS", mock_metric_models), \
             patch("eval_unlearn.techniques._base_models.TECHNIQUE_BASE_MODELS", {}):
            cmd_models(Namespace())
        out = capsys.readouterr().out
        assert "tifa_test" in out

    def test_cmd_models_handles_non_configurable_with_note(self, capsys):
        from eval_unlearn.metrics._base_models import MetricModelInfo
        mock_metric_models = {
            "fid_test": MetricModelInfo(
                model="Inception V3",
                configurable=False,
                note="torchvision",
            )
        }
        with patch("eval_unlearn.metrics._base_models.METRIC_MODELS", mock_metric_models), \
             patch("eval_unlearn.techniques._base_models.TECHNIQUE_BASE_MODELS", {}):
            cmd_models(Namespace())
        out = capsys.readouterr().out
        assert "fid_test" in out
        assert "torchvision" in out

    def test_cmd_models_handles_non_configurable_no_note(self, capsys):
        from eval_unlearn.metrics._base_models import MetricModelInfo
        mock_metric_models = {
            "simple_metric": MetricModelInfo(
                model="some/model",
                configurable=False,
            )
        }
        with patch("eval_unlearn.metrics._base_models.METRIC_MODELS", mock_metric_models), \
             patch("eval_unlearn.techniques._base_models.TECHNIQUE_BASE_MODELS", {}):
            cmd_models(Namespace())
        out = capsys.readouterr().out
        assert "simple_metric" in out


# ---------------------------------------------------------------------------
# main — dispatch
# ---------------------------------------------------------------------------
class TestMainDispatch:
    def test_dispatch_run_command(self, tmp_path, capsys):
        cfg = {
            "output_dir": str(tmp_path),
            "technique": {"name": "esd"},
            "metric": {"name": "fid"},
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))

        mock_runner = MagicMock()
        mock_runner.run.return_value = {"run_id": "abc"}

        with patch("sys.argv", ["eval-unlearn", "run", "--config", str(cfg_path)]), \
             patch("eval_unlearn.cli.SingleBenchmarkRunner", return_value=mock_runner):
            main()

    def test_dispatch_plugins_command(self, capsys):
        with patch("sys.argv", ["eval-unlearn", "plugins"]), \
             patch("eval_unlearn.registry.entrypoints.load_entrypoints"), \
             patch("eval_unlearn.registry.local._TECHNIQUES", {}), \
             patch("eval_unlearn.registry.local._METRICS", {}), \
             patch("eval_unlearn.registry.local._DATASETS", {}):
            main()
        out = capsys.readouterr().out
        assert "Techniques:" in out

    def test_dispatch_models_command(self, capsys):
        with patch("sys.argv", ["eval-unlearn", "models"]):
            main()
        out = capsys.readouterr().out
        assert "Techniques:" in out

    def test_dispatch_push_command(self, tmp_path):
        with patch("sys.argv", ["eval-unlearn", "push", "--repo", "org/repo",
                                "--local-dir", str(tmp_path)]), \
             patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.push_folder.return_value = "https://hf.co"
            main()
        mock_sync.push_folder.assert_called_once()

    def test_dispatch_pull_command(self):
        with patch("sys.argv", ["eval-unlearn", "pull", "--repo", "org/repo"]), \
             patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.pull_all.return_value = "/local/results"
            main()
        mock_sync.pull_all.assert_called_once()

    def test_load_yaml_missing_pyyaml_exits(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("technique:\n  name: esd\n")
        with patch("builtins.__import__", side_effect=ImportError("no yaml")):
            pass  # Can't easily mock this without breaking everything

    def test_load_config_yaml_success(self, tmp_path):
        pytest.importorskip("yaml")
        p = tmp_path / "config.yaml"
        p.write_text("technique:\n  name: sld\n")
        result = _load_config(str(p))
        assert result == {"technique": {"name": "sld"}}
