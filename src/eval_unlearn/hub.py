"""Hugging Face Hub integration for syncing benchmark artifacts."""

import os
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

from .logging_utils import get_logger

logger = get_logger(__name__)


class HFSync:
    """Push and pull benchmark artifacts to/from a single HF Hub dataset repository.

    One repository can hold many experiments; callers control the remote path:

        {repo_id}/
            nudity_study/           ← remote_path chosen by caller
                sld_asr_a1b2c3d4/
                    a1b2c3d4_report.json
                    images/
                esd_fid_b2c3d4e5/
                    ...
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        create_pr: bool = False,
    ):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.create_pr = create_pr
        self.api = HfApi(token=self.token)

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push_folder(self, local_dir: str, remote_path: str) -> str:
        """Upload an entire local directory to the repo at remote_path.

        Args:
            local_dir: Local directory to upload (e.g. ``results/my_run``).
            remote_path: Destination path inside the repo (e.g. ``"nudity_study"``).

        Returns:
            Commit URL.
        """
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Directory not found: {local_dir}")

        logger.info(
            "Pushing %s → %s/%s", os.path.basename(local_dir.rstrip("/")),
            self.repo_id, remote_path,
        )
        url = self.api.upload_folder(
            folder_path=local_dir,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add: {os.path.basename(local_dir.rstrip('/'))}",
            create_pr=self.create_pr,
        )
        logger.info("Pushed: %s", url)
        return url

    def push_file(self, local_path: str, remote_path: str) -> str:
        """Upload a single file to the repo.

        Args:
            local_path: Local file path.
            remote_path: Destination path inside the repo.

        Returns:
            Commit URL.
        """
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"File not found: {local_path}")

        logger.info(
            "Pushing %s → %s/%s",
            os.path.basename(local_path), self.repo_id, remote_path,
        )
        url = self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add: {os.path.basename(local_path)}",
            create_pr=self.create_pr,
        )
        logger.info("Pushed: %s", url)
        return url

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull_folder(self, remote_path: str, local_dir: str) -> str:
        """Download a remote path from the repo into a local directory.

        Args:
            remote_path: Path inside the repo (e.g. ``"nudity_study"``).
            local_dir: Local directory to download into.

        Returns:
            Local path where files were saved.
        """
        logger.info(
            "Pulling %s/%s → %s/", self.repo_id, remote_path, local_dir
        )
        path = snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=f"{remote_path}/**",
            token=self.token,
        )
        logger.info("Downloaded to %s", path)
        return path

    def pull_all(self, local_dir: str) -> str:
        """Download the entire repository snapshot.

        Args:
            local_dir: Local directory to download into.

        Returns:
            Local path where files were saved.
        """
        logger.info("Pulling all from %s → %s/", self.repo_id, local_dir)
        path = snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            token=self.token,
        )
        logger.info("Downloaded to %s", path)
        return path
