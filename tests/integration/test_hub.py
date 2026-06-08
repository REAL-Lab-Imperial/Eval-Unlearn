"""
Integration tests for HFSync (hub.py).

All HF API calls are mocked to avoid network access.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.integration


@pytest.fixture
def sync():
    with patch("eval_learn.hub.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        from eval_learn.hub import HFSync
        s = HFSync(repo_id="org/test-repo", token="fake_token")
        s._mock_api = mock_api
        yield s


class TestHFSyncPush:

    def test_push_folder_calls_upload_folder(self, sync, tmp_path):
        local = tmp_path / "my_run"
        local.mkdir()
        (local / "report.json").write_text("{}")
        sync._mock_api.upload_folder.return_value = "https://hf.co/commit/abc"
        url = sync.push_folder(str(local), "experiments/run1")
        sync._mock_api.upload_folder.assert_called_once()
        assert "hf.co" in url

    def test_push_folder_nonexistent_raises(self, sync, tmp_path):
        with pytest.raises(FileNotFoundError):
            sync.push_folder(str(tmp_path / "nonexistent"), "some/path")

    def test_push_file_calls_upload_file(self, sync, tmp_path):
        f = tmp_path / "report.json"
        f.write_text('{"value": 0.5}')
        sync._mock_api.upload_file.return_value = "https://hf.co/commit/def"
        url = sync.push_file(str(f), "experiments/report.json")
        sync._mock_api.upload_file.assert_called_once()
        assert "hf.co" in url

    def test_push_file_nonexistent_raises(self, sync, tmp_path):
        with pytest.raises(FileNotFoundError):
            sync.push_file(str(tmp_path / "no_file.json"), "path/file.json")

    def test_push_with_create_pr(self, tmp_path):
        with patch("eval_learn.hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            mock_api.upload_folder.return_value = "https://hf.co/pr/1"
            from eval_learn.hub import HFSync
            s = HFSync(repo_id="org/repo", create_pr=True)
            local = tmp_path / "run"
            local.mkdir()
            (local / "f.txt").write_text("x")
            s.push_folder(str(local), "exp")
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs.get("create_pr") is True


class TestHFSyncPull:

    def test_pull_folder(self, sync, tmp_path):
        with patch("eval_learn.hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path)
            path = sync.pull_folder("experiments", str(tmp_path / "local"))
        mock_dl.assert_called_once()
        assert path == str(tmp_path)

    def test_pull_all(self, sync, tmp_path):
        with patch("eval_learn.hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path)
            path = sync.pull_all(str(tmp_path / "all"))
        mock_dl.assert_called_once()
        assert path == str(tmp_path)

    def test_pull_folder_passes_allow_patterns(self, sync, tmp_path):
        with patch("eval_learn.hub.snapshot_download") as mock_dl:
            mock_dl.return_value = str(tmp_path)
            sync.pull_folder("my/path", str(tmp_path))
        call_kwargs = mock_dl.call_args[1]
        assert "allow_patterns" in call_kwargs
        assert "my/path" in call_kwargs["allow_patterns"]

    def test_token_from_env(self, tmp_path):
        with patch("eval_learn.hub.HfApi") as mock_api_cls, \
             patch.dict("os.environ", {"HF_TOKEN": "env_token"}):
            mock_api_cls.return_value = MagicMock()
            from eval_learn.hub import HFSync
            import importlib
            import eval_learn.hub
            importlib.reload(eval_learn.hub)
            s = eval_learn.hub.HFSync(repo_id="org/repo")
        assert s.token == "env_token"
