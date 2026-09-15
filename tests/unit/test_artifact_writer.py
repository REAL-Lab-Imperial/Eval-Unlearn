"""Unit tests for ArtifactWriter."""
import json
import os
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from eval_unlearn.artifacts.writer import ArtifactWriter


def _dummy_image(color=(10, 20, 30)):
    return Image.new("RGB", (4, 4), color=color)


# ---------------------------------------------------------------------------
# save_run — flat image saving
# ---------------------------------------------------------------------------
class TestSaveRunFlatImages:
    def test_saves_images_and_returns_report_path(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        imgs = [_dummy_image(), _dummy_image()]
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0}
        path = writer.save_run("abc12345", "esd", "fid", imgs, report=report)
        assert path.endswith("abc12345_report.json")
        assert os.path.exists(path)

    def test_flat_image_files_created(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        imgs = [_dummy_image(), _dummy_image()]
        writer.save_run("run1", "esd", "fid", imgs, report=None)
        images_dir = tmp_path / "esd_fid_run1" / "images"
        saved = list(images_dir.iterdir())
        assert len(saved) == 2

    def test_image_index_offset(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        writer.save_run("run1", "esd", "fid", [_dummy_image()], image_index_offset=5)
        images_dir = tmp_path / "esd_fid_run1" / "images"
        names = [f.name for f in images_dir.iterdir()]
        assert any("5" in n for n in names)


# ---------------------------------------------------------------------------
# save_run — category-aware saving
# ---------------------------------------------------------------------------
class TestSaveRunCategoryAware:
    def test_saves_into_category_subdirs(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        imgs = [_dummy_image(), _dummy_image()]
        metadata = {"categories": ["target", "retain"]}
        writer.save_run("run2", "ssd", "err", imgs, metadata=metadata)
        images_dir = tmp_path / "ssd_err_run2" / "images"
        assert (images_dir / "target").is_dir()
        assert (images_dir / "retain").is_dir()

    def test_category_counter_continues(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        metadata = {"categories": ["target", "target"]}
        writer.save_run("run3", "ssd", "err", [_dummy_image(), _dummy_image()],
                        metadata=metadata, category_counters_init={"target": 3})
        images_dir = tmp_path / "ssd_err_run3" / "images" / "target"
        names = [f.name for f in images_dir.iterdir()]
        # First image should be index 3, second index 4
        assert any("_3.png" in n for n in names)
        assert any("_4.png" in n for n in names)

    def test_mismatched_categories_falls_back_to_flat(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        imgs = [_dummy_image(), _dummy_image()]
        metadata = {"categories": ["only_one"]}  # len mismatch
        writer.save_run("run4", "ssd", "err", imgs, metadata=metadata)
        images_dir = tmp_path / "ssd_err_run4" / "images"
        # Falls back to flat — no subdirs named "only_one"
        assert not (images_dir / "only_one").exists()


# ---------------------------------------------------------------------------
# save_run — report saving
# ---------------------------------------------------------------------------
class TestSaveRunReport:
    def test_report_json_content(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 99.9, "value": 0.5}
        path = writer.save_run("r1", "esd", "fid", [], report=report)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["value"] == 0.5

    def test_no_report_skips_json(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        writer.save_run("r2", "esd", "fid", [], report=None)
        assert not (tmp_path / "r2_report.json").exists()

    def test_detailed_report_saved(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0}
        detailed = {"full": True}
        writer.save_run("r3", "esd", "fid", [_dummy_image()], report=report, detailed_report=detailed)
        full_path = tmp_path / "esd_fid_r3" / "r3_report_full.json"
        assert full_path.exists()
        with open(full_path) as f:
            assert json.load(f)["full"] is True

    def test_no_images_report_goes_to_base_dir(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0}
        path = writer.save_run("r4", "esd", "multi", [], report=report)
        assert str(tmp_path) in path


# ---------------------------------------------------------------------------
# _sync_to_final_reports
# ---------------------------------------------------------------------------
class TestSyncToFinalReports:
    def test_creates_latest_report(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 5.0}
        writer.save_run("rs1", "esd", "fid", [], report=report)
        latest = tmp_path / "esd_nudity_latest_report.json"
        assert latest.exists()

    def test_does_not_overwrite_newer_report(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        # Write a newer report first
        newer = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 100.0, "marker": "new"}
        writer.save_run("rs2", "esd", "fid", [], report=newer)
        # Now try to sync an older report
        older = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0, "marker": "old"}
        report_path = tmp_path / "esd_fid_rs_old" / "rs_old_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(older, f)
        writer._sync_to_final_reports(str(report_path), older)
        # Latest should still be the newer one
        latest = tmp_path / "esd_nudity_latest_report.json"
        with open(latest) as f:
            assert json.load(f).get("marker") == "new"

    def test_overwrites_older_report(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        old = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0, "marker": "old"}
        writer.save_run("rs3", "esd", "fid", [], report=old)
        newer = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 50.0, "marker": "new"}
        writer.save_run("rs4", "esd", "fid", [], report=newer)
        latest = tmp_path / "esd_nudity_latest_report.json"
        with open(latest) as f:
            assert json.load(f).get("marker") == "new"


# ---------------------------------------------------------------------------
# _save_image
# ---------------------------------------------------------------------------
class TestSaveImage:
    def test_saves_pil_image(self, tmp_path):
        path = str(tmp_path / "img.png")
        img = _dummy_image()
        result = ArtifactWriter._save_image(img, path, 0)
        assert result == path
        assert os.path.exists(path)

    def test_non_saveable_returns_none(self, tmp_path):
        path = str(tmp_path / "img.png")
        result = ArtifactWriter._save_image("not_an_image", path, 0)
        assert result is None

    def test_save_failure_returns_none(self, tmp_path):
        path = str(tmp_path / "img.png")
        img = MagicMock()
        img.save.side_effect = OSError("disk full")
        result = ArtifactWriter._save_image(img, path, 0)
        assert result is None


# ---------------------------------------------------------------------------
# Exception paths in save_run and _sync_to_final_reports
# ---------------------------------------------------------------------------
class TestArtifactWriterExceptionPaths:
    def test_report_json_serialization_failure_does_not_raise(self, tmp_path):
        writer = ArtifactWriter(base_dir=str(tmp_path))
        bad_report = {"key": object()}
        path = writer.save_run(
            run_id="err_test", technique_name="esd",
            metric_name="asr_i2p", images=[], report=bad_report,
        )
        assert path is not None

    def test_detailed_report_serialization_failure_does_not_raise(self, tmp_path):
        import json
        writer = ArtifactWriter(base_dir=str(tmp_path))
        good = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0}
        path = writer.save_run(
            run_id="err_test2", technique_name="esd",
            metric_name="asr_i2p", images=[], report=good,
            detailed_report={"key": object()},
        )
        assert path is not None

    def test_sync_skips_when_existing_is_newer(self, tmp_path):
        import json
        writer = ArtifactWriter(base_dir=str(tmp_path))
        dest = tmp_path / "esd_nudity_latest_report.json"
        existing = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 9999.0}
        dest.write_text(json.dumps(existing))
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0}
        report_path = tmp_path / "r.json"
        report_path.write_text(json.dumps(report))
        writer._sync_to_final_reports(str(report_path), report)
        assert json.loads(dest.read_text())["timestamp"] == 9999.0

    def test_sync_copy_failure_does_not_raise(self, tmp_path):
        import json
        from unittest.mock import patch
        writer = ArtifactWriter(base_dir=str(tmp_path))
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 1.0}
        report_path = str(tmp_path / "report.json")
        with open(report_path, "w") as f:
            json.dump(report, f)
        with patch("shutil.copy2", side_effect=OSError("disk full")):
            writer._sync_to_final_reports(report_path, report)

    def test_sync_with_corrupt_existing_file_still_copies(self, tmp_path):
        import json, shutil
        writer = ArtifactWriter(base_dir=str(tmp_path))
        dest = tmp_path / "esd_nudity_latest_report.json"
        dest.write_text("not valid json {{{")
        report = {"technique_name": "esd", "erase_concept": "nudity", "timestamp": 5.0}
        report_path = str(tmp_path / "report.json")
        with open(report_path, "w") as f:
            json.dump(report, f)
        writer._sync_to_final_reports(report_path, report)
        assert json.loads(dest.read_text())["timestamp"] == 5.0
