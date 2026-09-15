"""Unit tests for HFSync hub integration."""

from unittest.mock import MagicMock, patch

import pytest

from eval_unlearn.hub import HFSync


class TestHFSyncInit:
    def test_token_from_argument(self):
        with patch("eval_unlearn.hub.HfApi") as mock_api_cls:
            sync = HFSync(repo_id="org/repo", token="tok_abc")
        assert sync.token == "tok_abc"
        mock_api_cls.assert_called_once_with(token="tok_abc")

    def test_token_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_tok")
        with patch("eval_unlearn.hub.HfApi"):
            sync = HFSync(repo_id="org/repo")
        assert sync.token == "env_tok"

    def test_token_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("eval_unlearn.hub.HfApi"):
            sync = HFSync(repo_id="org/repo")
        assert sync.token is None

    def test_defaults(self):
        with patch("eval_unlearn.hub.HfApi"):
            sync = HFSync(repo_id="org/repo")
        assert sync.repo_id == "org/repo"
        assert sync.create_pr is False

    def test_create_pr_flag(self):
        with patch("eval_unlearn.hub.HfApi"):
            sync = HFSync(repo_id="org/repo", create_pr=True)
        assert sync.create_pr is True


class TestPushFolder:
    @pytest.fixture
    def sync(self):
        with patch("eval_unlearn.hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            s = HFSync(repo_id="org/repo", token="tok")
            s.api = mock_api
            return s

    def test_push_folder_calls_upload_folder(self, sync, tmp_path):
        sync.api.upload_folder.return_value = "https://hf.co/commit/abc"
        url = sync.push_folder(str(tmp_path), "remote/path")

        sync.api.upload_folder.assert_called_once_with(
            folder_path=str(tmp_path),
            path_in_repo="remote/path",
            repo_id="org/repo",
            repo_type="dataset",
            commit_message=f"Add: {tmp_path.name}",
            create_pr=False,
        )
        assert url == "https://hf.co/commit/abc"

    def test_push_folder_with_create_pr(self, tmp_path):
        with patch("eval_unlearn.hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            mock_api.upload_folder.return_value = "https://hf.co/pr/1"
            sync = HFSync(repo_id="org/repo", create_pr=True)
            sync.push_folder(str(tmp_path), "some/path")

        _, kwargs = mock_api.upload_folder.call_args
        assert kwargs["create_pr"] is True

    def test_push_folder_raises_when_dir_missing(self, sync):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            sync.push_folder("/nonexistent/dir", "remote/path")

    def test_push_folder_raises_on_file_not_dir(self, sync, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            sync.push_folder(str(f), "remote/path")


class TestPushFile:
    @pytest.fixture
    def sync(self):
        with patch("eval_unlearn.hub.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            s = HFSync(repo_id="org/repo", token="tok")
            s.api = mock_api
            return s

    def test_push_file_calls_upload_file(self, sync, tmp_path):
        f = tmp_path / "report.json"
        f.write_text("{}")
        sync.api.upload_file.return_value = "https://hf.co/commit/def"

        url = sync.push_file(str(f), "results/report.json")

        sync.api.upload_file.assert_called_once_with(
            path_or_fileobj=str(f),
            path_in_repo="results/report.json",
            repo_id="org/repo",
            repo_type="dataset",
            commit_message="Add: report.json",
            create_pr=False,
        )
        assert url == "https://hf.co/commit/def"

    def test_push_file_raises_when_missing(self, sync):
        with pytest.raises(FileNotFoundError, match="File not found"):
            sync.push_file("/nonexistent/file.json", "remote/file.json")

    def test_push_file_raises_on_directory(self, sync, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            sync.push_file(str(tmp_path), "remote/file.json")


class TestPullFolder:
    @pytest.fixture
    def sync(self):
        with patch("eval_unlearn.hub.HfApi"):
            s = HFSync(repo_id="org/repo", token="tok")
            return s

    def test_pull_folder_calls_snapshot_download(self, sync, tmp_path):
        with patch("eval_unlearn.hub.snapshot_download", return_value=str(tmp_path)) as mock_dl:
            path = sync.pull_folder("nudity_study", str(tmp_path))

        mock_dl.assert_called_once_with(
            repo_id="org/repo",
            repo_type="dataset",
            local_dir=str(tmp_path),
            allow_patterns="nudity_study/**",
            token="tok",
        )
        assert path == str(tmp_path)

    def test_pull_folder_returns_local_path(self, sync, tmp_path):
        with patch("eval_unlearn.hub.snapshot_download", return_value="/some/cache/path"):
            path = sync.pull_folder("remote/path", str(tmp_path))
        assert path == "/some/cache/path"


class TestPullAll:
    @pytest.fixture
    def sync(self):
        with patch("eval_unlearn.hub.HfApi"):
            s = HFSync(repo_id="org/repo", token="tok")
            return s

    def test_pull_all_calls_snapshot_download_without_patterns(self, sync, tmp_path):
        with patch("eval_unlearn.hub.snapshot_download", return_value=str(tmp_path)) as mock_dl:
            path = sync.pull_all(str(tmp_path))

        mock_dl.assert_called_once_with(
            repo_id="org/repo",
            repo_type="dataset",
            local_dir=str(tmp_path),
            token="tok",
        )
        assert path == str(tmp_path)
