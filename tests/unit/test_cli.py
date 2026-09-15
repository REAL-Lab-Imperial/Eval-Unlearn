"""Unit tests for CLI helpers and commands."""

import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from eval_unlearn.cli import (
    _load_config,
    _parse_metrics_list,
    _build_single_runner,
    _build_multi_runner,
    cmd_push,
    cmd_pull,
    cmd_plugins,
    main,
)


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_json_file(self, tmp_path):
        cfg = {"technique": {"name": "esd"}}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        result = _load_config(str(p))
        assert result == cfg

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _load_config(str(tmp_path / "missing.json"))

    def test_loads_yaml_file(self, tmp_path):
        pytest.importorskip("yaml")
        cfg = {"technique": {"name": "sld"}}
        p = tmp_path / "config.yaml"
        p.write_text("technique:\n  name: sld\n")
        result = _load_config(str(p))
        assert result == cfg


# ---------------------------------------------------------------------------
# _parse_metrics_list
# ---------------------------------------------------------------------------

class TestParseMetricsList:
    def test_names_extracted(self):
        names, configs = _parse_metrics_list([
            {"name": "asr"},
            {"name": "fid", "config": {"limit": 100}},
        ])
        assert names == ["asr", "fid"]
        assert configs == {"fid": {"limit": 100}}

    def test_missing_name_exits(self):
        with pytest.raises(SystemExit):
            _parse_metrics_list([{"config": {}}])

    def test_empty_config_not_stored(self):
        names, configs = _parse_metrics_list([{"name": "asr"}])
        assert "asr" not in configs


# ---------------------------------------------------------------------------
# _build_single_runner / _build_multi_runner
# ---------------------------------------------------------------------------

class TestBuildRunners:
    def test_build_single_runner_missing_technique_exits(self):
        with pytest.raises(SystemExit):
            _build_single_runner({"metric": {"name": "asr"}}, "out")

    def test_build_single_runner_missing_metric_exits(self):
        with pytest.raises(SystemExit):
            _build_single_runner({"technique": {"name": "esd"}}, "out")

    def test_build_single_runner_returns_runner(self):
        with patch("eval_unlearn.cli.SingleBenchmarkRunner") as mock_cls:
            mock_cls.return_value = MagicMock()
            _build_single_runner(
                {
                    "technique": {"name": "esd", "config": {"device": "cpu"}},
                    "metric": {"name": "asr", "config": {"limit": 10}},
                },
                "results",
            )
        mock_cls.assert_called_once_with(
            technique_name="esd",
            metric_name="asr",
            technique_config={"device": "cpu"},
            metric_config={"limit": 10},
            output_dir="results",
        )

    def test_build_multi_runner_missing_technique_exits(self):
        with pytest.raises(SystemExit):
            _build_multi_runner({"metrics": [{"name": "asr"}]}, "out")

    def test_build_multi_runner_empty_metrics_exits(self):
        with pytest.raises(SystemExit):
            _build_multi_runner({"technique": {"name": "esd"}, "metrics": []}, "out")

    def test_build_multi_runner_returns_runner(self):
        with patch("eval_unlearn.cli.MultiBenchmarkRunner") as mock_cls:
            mock_cls.return_value = MagicMock()
            _build_multi_runner(
                {
                    "technique": {"name": "esd"},
                    "metrics": [{"name": "asr"}, {"name": "fid"}],
                },
                "results",
            )
        mock_cls.assert_called_once_with(
            technique_name="esd",
            metric_names=["asr", "fid"],
            technique_config={},
            metric_configs={},
            output_dir="results",
        )


# ---------------------------------------------------------------------------
# cmd_push
# ---------------------------------------------------------------------------

class TestCmdPush:
    def _args(self, **kwargs):
        defaults = dict(repo="org/repo", local_dir="/tmp/results", remote_path=None, create_pr=False)
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_push_success(self, tmp_path):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.push_folder.return_value = "https://hf.co/commit/abc"

            args = self._args(local_dir=str(tmp_path))
            cmd_push(args)

        mock_sync.push_folder.assert_called_once_with(
            str(tmp_path), tmp_path.name
        )

    def test_push_uses_explicit_remote_path(self, tmp_path):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.push_folder.return_value = "https://hf.co/commit/abc"

            args = self._args(local_dir=str(tmp_path), remote_path="custom/path")
            cmd_push(args)

        mock_sync.push_folder.assert_called_once_with(str(tmp_path), "custom/path")

    def test_push_failure_exits(self, tmp_path):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.push_folder.side_effect = RuntimeError("network error")

            with pytest.raises(SystemExit):
                cmd_push(self._args(local_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# cmd_pull
# ---------------------------------------------------------------------------

class TestCmdPull:
    def _args(self, **kwargs):
        defaults = dict(repo="org/repo", remote_path=None, local_dir=None)
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_pull_folder_called_when_remote_path_given(self):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.pull_folder.return_value = "/local/path"

            cmd_pull(self._args(remote_path="study/esd", local_dir="/local"))

        mock_sync.pull_folder.assert_called_once_with("study/esd", "/local")
        mock_sync.pull_all.assert_not_called()

    def test_pull_all_called_when_no_remote_path(self):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.pull_all.return_value = "/local/results"

            cmd_pull(self._args())

        mock_sync.pull_all.assert_called_once_with("results")
        mock_sync.pull_folder.assert_not_called()

    def test_pull_failure_exits(self):
        with patch("eval_unlearn.hub.HFSync") as mock_cls:
            mock_sync = MagicMock()
            mock_cls.return_value = mock_sync
            mock_sync.pull_all.side_effect = RuntimeError("auth error")

            with pytest.raises(SystemExit):
                cmd_pull(self._args())


# ---------------------------------------------------------------------------
# cmd_plugins
# ---------------------------------------------------------------------------

class TestCmdPlugins:
    def test_plugins_lists_registered(self, capsys):
        with patch("eval_unlearn.registry.entrypoints.load_entrypoints"), \
             patch("eval_unlearn.registry.local._TECHNIQUES", {"esd": object()}), \
             patch("eval_unlearn.registry.local._METRICS", {"asr": object()}), \
             patch("eval_unlearn.registry.local._DATASETS", {"coco": object()}):
            cmd_plugins(Namespace())

        out = capsys.readouterr().out
        assert "esd" in out
        assert "asr" in out
        assert "coco" in out


# ---------------------------------------------------------------------------
# main / argument parsing
# ---------------------------------------------------------------------------

class TestMain:
    def test_version_flag(self, capsys):
        with patch("sys.argv", ["eval-unlearn", "--version"]), \
             patch("eval_unlearn.cli.load_dotenv"), \
             patch("eval_unlearn.__version__", "1.2.3", create=True):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0
        assert "1.2.3" in capsys.readouterr().out

    def test_no_command_prints_help(self, capsys):
        with patch("sys.argv", ["eval-unlearn"]):
            main()  # should not raise or exit
        out = capsys.readouterr().out
        assert "eval-unlearn" in out

    def test_unknown_command_prints_help(self, capsys):
        with patch("sys.argv", ["eval-unlearn"]):
            main()
        out = capsys.readouterr().out
        assert out  # some usage text printed
