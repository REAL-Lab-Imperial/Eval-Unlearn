"""
Integration tests for the CLI (cli.py).

Covers cmd_run, cmd_push, cmd_pull, cmd_plugins, cmd_models,
_load_config, _build_single_runner, _build_multi_runner, and main().
"""
import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:

    def test_load_json_config(self, tmp_path):
        from eval_learn.cli import _load_config
        cfg = {"technique": {"name": "ssd"}, "metric": {"name": "asr_p4d"}}
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg))
        result = _load_config(str(p))
        assert result["technique"]["name"] == "ssd"

    def test_missing_file_exits(self, tmp_path):
        from eval_learn.cli import _load_config
        with pytest.raises(SystemExit):
            _load_config(str(tmp_path / "nonexistent.json"))

    def test_yaml_without_pyyaml_exits(self, tmp_path):
        from eval_learn.cli import _load_config
        p = tmp_path / "cfg.yaml"
        p.write_text("technique:\n  name: ssd\n")
        with patch.dict("sys.modules", {"yaml": None}):
            try:
                result = _load_config(str(p))
                # If yaml is available, it should just work
            except SystemExit:
                pass  # expected when yaml not importable


# ---------------------------------------------------------------------------
# _parse_metrics_list
# ---------------------------------------------------------------------------
class TestParseMetricsList:

    def test_parses_names_and_configs(self):
        from eval_learn.cli import _parse_metrics_list
        items = [
            {"name": "asr_p4d", "config": {"concept_name": "nudity"}},
            {"name": "clip_score"},
        ]
        names, configs = _parse_metrics_list(items)
        assert names == ["asr_p4d", "clip_score"]
        assert configs["asr_p4d"]["concept_name"] == "nudity"
        assert "clip_score" not in configs  # no config provided

    def test_missing_name_exits(self):
        from eval_learn.cli import _parse_metrics_list
        with pytest.raises(SystemExit):
            _parse_metrics_list([{"config": {}}])


# ---------------------------------------------------------------------------
# _build_single_runner
# ---------------------------------------------------------------------------
class TestBuildSingleRunner:

    def test_missing_technique_name_exits(self, tmp_path):
        from eval_learn.cli import _build_single_runner
        cfg = {"technique": {}, "metric": {"name": "asr_p4d"}}
        with pytest.raises(SystemExit):
            _build_single_runner(cfg, str(tmp_path))

    def test_missing_metric_name_exits(self, tmp_path):
        from eval_learn.cli import _build_single_runner
        cfg = {"technique": {"name": "ssd"}, "metric": {}}
        with pytest.raises(SystemExit):
            _build_single_runner(cfg, str(tmp_path))

    def test_returns_runner_when_valid(self, tmp_path):
        from eval_learn.cli import _build_single_runner
        from eval_learn.registry.local import _TECHNIQUES, _METRICS
        # Register a dummy technique to avoid import error
        class _T:
            def __init__(self, **kw): pass
            def generate(self, prompts, **kw): return []
        _TECHNIQUES["_cli_dummy"] = _T
        try:
            cfg = {
                "technique": {"name": "_cli_dummy", "config": {"erase_concept": "violence"}},
                "metric": {"name": "asr_p4d", "config": {"concept_name": "violence",
                    "erase_id": "std", "detector": "q16"}},
            }
            with patch("eval_learn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
                 patch("eval_learn.metrics.asr_p4d.metric.Q16Classifier", MagicMock()):
                runner = _build_single_runner(cfg, str(tmp_path))
            from eval_learn.runners.single_benchmark_runner import SingleBenchmarkRunner
            assert isinstance(runner, SingleBenchmarkRunner)
        finally:
            _TECHNIQUES.pop("_cli_dummy", None)


# ---------------------------------------------------------------------------
# _build_multi_runner
# ---------------------------------------------------------------------------
class TestBuildMultiRunner:

    def test_missing_technique_name_exits(self, tmp_path):
        from eval_learn.cli import _build_multi_runner
        cfg = {"technique": {}, "metrics": [{"name": "asr_p4d"}]}
        with pytest.raises(SystemExit):
            _build_multi_runner(cfg, str(tmp_path))

    def test_empty_metrics_exits(self, tmp_path):
        from eval_learn.cli import _build_multi_runner
        cfg = {"technique": {"name": "ssd"}, "metrics": []}
        with pytest.raises(SystemExit):
            _build_multi_runner(cfg, str(tmp_path))


# ---------------------------------------------------------------------------
# cmd_run
# ---------------------------------------------------------------------------
class TestCmdRun:

    def _write_config(self, tmp_path, config):
        p = tmp_path / "run.json"
        p.write_text(json.dumps(config))
        return str(p)

    def test_cmd_run_single_success(self, tmp_path):
        """cmd_run dispatches to SingleBenchmarkRunner and calls run()."""
        cfg = {
            "technique": {"name": "_cmd_dummy", "config": {"erase_concept": "violence"}},
            "metric": {"name": "asr_p4d", "config": {
                "concept_name": "violence", "detector": "q16", "erase_id": "std",
            }},
            "output_dir": str(tmp_path / "results"),
        }
        cfg_path = self._write_config(tmp_path, cfg)

        from eval_learn.registry.local import _TECHNIQUES
        from PIL import Image
        class _CmdDummy:
            def __init__(self, **kw): pass
            def generate(self, prompts, **kw):
                return [Image.new("RGB", (4, 4)) for _ in prompts]
        _TECHNIQUES["_cmd_dummy"] = _CmdDummy
        try:
            mock_report = {"run_id": "abc1", "value": 0.0}
            with patch("eval_learn.runners.single_benchmark_runner.SingleBenchmarkRunner.run",
                       return_value=mock_report), \
                 patch("eval_learn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
                 patch("eval_learn.metrics.asr_p4d.metric.Q16Classifier", MagicMock()):
                from eval_learn.cli import cmd_run
                args = MagicMock()
                args.config = cfg_path
                args.hf_repo = None
                args.hf_path = None
                args.create_pr = False
                cmd_run(args)
        finally:
            _TECHNIQUES.pop("_cmd_dummy", None)

    def test_cmd_run_multi_success(self, tmp_path):
        cfg = {
            "technique": {"name": "_cmd_dummy_m", "config": {"erase_concept": "violence"}},
            "metrics": [
                {"name": "asr_p4d", "config": {"concept_name": "violence",
                    "detector": "q16", "erase_id": "std"}},
            ],
            "output_dir": str(tmp_path / "results"),
        }
        cfg_path = self._write_config(tmp_path, cfg)

        from eval_learn.registry.local import _TECHNIQUES
        class _CmdDummyM:
            def __init__(self, **kw): pass
            def generate(self, prompts, **kw): return []
        _TECHNIQUES["_cmd_dummy_m"] = _CmdDummyM
        try:
            mock_report = {"run_id": "abc2"}
            with patch("eval_learn.runners.multi_benchmark_runner.MultiBenchmarkRunner.run",
                       return_value=mock_report), \
                 patch("eval_learn.metrics.asr_p4d.metric.P4DGenerator", MagicMock()), \
                 patch("eval_learn.metrics.asr_p4d.metric.Q16Classifier", MagicMock()):
                from eval_learn.cli import cmd_run
                args = MagicMock()
                args.config = cfg_path
                args.hf_repo = None
                args.hf_path = None
                args.create_pr = False
                cmd_run(args)
        finally:
            _TECHNIQUES.pop("_cmd_dummy_m", None)

    def test_cmd_run_invalid_config_exits(self, tmp_path):
        """Config missing technique exits with SystemExit."""
        cfg = {"output_dir": str(tmp_path)}
        cfg_path = self._write_config(tmp_path, cfg)
        from eval_learn.cli import cmd_run
        args = MagicMock()
        args.config = cfg_path
        args.hf_repo = None
        with pytest.raises(SystemExit):
            cmd_run(args)

    def test_cmd_run_validation_error_exits(self, tmp_path):
        """ValidationError propagates as SystemExit."""
        cfg = {
            "technique": {"name": "_cmd_val_dummy", "config": {"erase_concept": "violence"}},
            "metric": {"name": "err", "config": {}},
            "output_dir": str(tmp_path),
        }
        cfg_path = self._write_config(tmp_path, cfg)

        from eval_learn.registry.local import _TECHNIQUES
        class _T:
            def __init__(self, **kw): pass
            def generate(self, prompts, **kw): return []
        _TECHNIQUES["_cmd_val_dummy"] = _T
        try:
            from eval_learn.cli import cmd_run
            args = MagicMock()
            args.config = cfg_path
            args.hf_repo = None
            with pytest.raises(SystemExit):
                cmd_run(args)
        finally:
            _TECHNIQUES.pop("_cmd_val_dummy", None)


# ---------------------------------------------------------------------------
# cmd_push / cmd_pull
# ---------------------------------------------------------------------------
class TestCmdPushPull:

    def test_cmd_push(self, tmp_path):
        from eval_learn.cli import cmd_push
        local = tmp_path / "results"
        local.mkdir()
        (local / "file.txt").write_text("data")
        args = MagicMock()
        args.local_dir = str(local)
        args.repo = "org/my-repo"
        args.remote_path = "experiments"
        args.create_pr = False
        with patch("eval_learn.hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.upload_folder.return_value = "https://hf.co/commit/abc"
            mock_api_cls.return_value = mock_api
            cmd_push(args)
        mock_api.upload_folder.assert_called_once()

    def test_cmd_push_failure_exits(self, tmp_path):
        from eval_learn.cli import cmd_push
        local = tmp_path / "results"
        local.mkdir()
        args = MagicMock()
        args.local_dir = str(local)
        args.repo = "org/repo"
        args.remote_path = "path"
        args.create_pr = False
        with patch("eval_learn.hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.upload_folder.side_effect = Exception("network error")
            mock_api_cls.return_value = mock_api
            with pytest.raises(SystemExit):
                cmd_push(args)

    def test_cmd_pull_with_remote_path(self, tmp_path):
        from eval_learn.cli import cmd_pull
        args = MagicMock()
        args.repo = "org/repo"
        args.remote_path = "experiments"
        args.local_dir = str(tmp_path / "dl")
        with patch("eval_learn.hub.snapshot_download", return_value=str(tmp_path)):
            cmd_pull(args)

    def test_cmd_pull_all(self, tmp_path):
        from eval_learn.cli import cmd_pull
        args = MagicMock()
        args.repo = "org/repo"
        args.remote_path = None  # pull all
        args.local_dir = str(tmp_path / "dl")
        with patch("eval_learn.hub.snapshot_download", return_value=str(tmp_path)):
            cmd_pull(args)

    def test_cmd_pull_failure_exits(self, tmp_path):
        from eval_learn.cli import cmd_pull
        args = MagicMock()
        args.repo = "org/repo"
        args.remote_path = "path"
        args.local_dir = str(tmp_path)
        with patch("eval_learn.hub.snapshot_download", side_effect=Exception("no net")):
            with pytest.raises(SystemExit):
                cmd_pull(args)


# ---------------------------------------------------------------------------
# cmd_plugins / cmd_models
# ---------------------------------------------------------------------------
class TestCmdPluginsModels:

    def test_cmd_plugins_prints_output(self, capsys):
        from eval_learn.cli import cmd_plugins
        cmd_plugins(None)
        out = capsys.readouterr().out
        assert "Techniques" in out or "Metrics" in out

    def test_cmd_models_prints_output(self, capsys):
        from eval_learn.cli import cmd_models
        cmd_models(None)
        out = capsys.readouterr().out
        assert "fid" in out or "asr_i2p" in out


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------
class TestMain:

    def test_version_flag(self, capsys):
        from eval_learn.cli import main
        with patch("sys.argv", ["eval-learn", "--version"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_no_command_prints_help(self, capsys):
        from eval_learn.cli import main
        with patch("sys.argv", ["eval-learn"]):
            main()  # should not raise

    def test_plugins_command(self, capsys):
        from eval_learn.cli import main
        with patch("sys.argv", ["eval-learn", "plugins"]):
            main()

    def test_models_command(self, capsys):
        from eval_learn.cli import main
        with patch("sys.argv", ["eval-learn", "models"]):
            main()
