"""
Integration tests for ArtifactWriter.

Covers save_run() — flat image saving, category-aware saving,
report JSON writing, detailed report, _sync_to_final_reports, and _save_image.
"""
import json
import os
import pytest
from PIL import Image

pytestmark = pytest.mark.integration


def _blank(color=(100, 150, 200)):
    return Image.new("RGB", (16, 16), color=color)


@pytest.fixture
def writer(tmp_path):
    from eval_learn.artifacts.writer import ArtifactWriter
    return ArtifactWriter(base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Flat (no categories) image saving
# ---------------------------------------------------------------------------
class TestArtifactWriterFlat:

    def test_save_images_flat(self, writer, tmp_path):
        writer.save_run(
            run_id="abc123",
            technique_name="ssd",
            metric_name="asr_p4d",
            images=[_blank(), _blank()],
            report=None,
            metadata={},
        )
        imgs = []
        for root, _, files in os.walk(str(tmp_path)):
            imgs += [f for f in files if f.endswith(".png")]
        assert len(imgs) == 2

    def test_save_report_json(self, writer, tmp_path):
        report = {"run_id": "abc123", "value": 0.42, "technique_name": "ssd", "erase_concept": "nudity", "timestamp": 1.0}
        writer.save_run(
            run_id="abc123",
            technique_name="ssd",
            metric_name="asr_p4d",
            images=[_blank()],
            report=report,
            metadata={},
        )
        json_files = []
        for root, _, files in os.walk(str(tmp_path)):
            json_files += [f for f in files if f.endswith(".json")]
        assert any("report.json" in f for f in json_files)

    def test_save_detailed_report(self, writer, tmp_path):
        report = {"run_id": "x1", "technique_name": "ssd", "erase_concept": "nudity", "timestamp": 1.0}
        detailed = {"run_id": "x1", "config": {"a": 1}}
        writer.save_run(
            run_id="x1",
            technique_name="ssd",
            metric_name="clip_score",
            images=[_blank()],
            report=report,
            detailed_report=detailed,
            metadata={},
        )
        full_reports = []
        for root, _, files in os.walk(str(tmp_path)):
            full_reports += [f for f in files if "report_full" in f]
        assert len(full_reports) == 1

    def test_image_index_offset(self, writer, tmp_path):
        """Batch 2 starts at offset 5 — images should be numbered 5, 6."""
        writer.save_run(
            run_id="off1",
            technique_name="ssd",
            metric_name="asr",
            images=[_blank(), _blank()],
            report=None,
            metadata={},
            image_index_offset=5,
        )
        pngs = []
        for root, _, files in os.walk(str(tmp_path)):
            pngs += [f for f in files if f.endswith(".png")]
        assert any("_5.png" in f for f in pngs)
        assert any("_6.png" in f for f in pngs)

    def test_no_images_no_crash(self, writer, tmp_path):
        """save_run with empty images list should not create image files."""
        report = {"run_id": "z1", "technique_name": "ssd", "erase_concept": "nudity", "timestamp": 1.0}
        path = writer.save_run(
            run_id="z1",
            technique_name="ssd",
            metric_name="asr",
            images=[],
            report=report,
            metadata={},
        )
        assert path.endswith(".json")


# ---------------------------------------------------------------------------
# Category-aware image saving
# ---------------------------------------------------------------------------
class TestArtifactWriterCategoryAware:

    def test_saves_into_category_subdirs(self, writer, tmp_path):
        metadata = {"categories": ["target", "retain", "adversarial"]}
        writer.save_run(
            run_id="cat1",
            technique_name="ssd",
            metric_name="err",
            images=[_blank(), _blank(), _blank()],
            report=None,
            metadata=metadata,
        )
        cat_dirs = []
        for root, dirs, _ in os.walk(str(tmp_path)):
            cat_dirs += dirs
        assert "target" in cat_dirs
        assert "retain" in cat_dirs
        assert "adversarial" in cat_dirs

    def test_category_counters_init_continues_numbering(self, writer, tmp_path):
        """Second batch should continue numbering from previous batch counters."""
        metadata = {"categories": ["target"]}
        writer.save_run(
            run_id="cat2",
            technique_name="ssd",
            metric_name="err",
            images=[_blank()],
            report=None,
            metadata=metadata,
            category_counters_init={"target": 3},
        )
        pngs = []
        for root, _, files in os.walk(str(tmp_path)):
            pngs += [f for f in files if f.endswith(".png")]
        assert any("_3.png" in f for f in pngs)

    def test_categories_length_mismatch_falls_back_to_flat(self, writer, tmp_path):
        """If categories list length != images length, use flat saving."""
        metadata = {"categories": ["target"]}  # 1 category, 2 images
        writer.save_run(
            run_id="cat3",
            technique_name="ssd",
            metric_name="err",
            images=[_blank(), _blank()],
            report=None,
            metadata=metadata,
        )
        pngs = []
        for root, _, files in os.walk(str(tmp_path)):
            pngs += [f for f in files if f.endswith(".png")]
        assert len(pngs) == 2


# ---------------------------------------------------------------------------
# _sync_to_final_reports
# ---------------------------------------------------------------------------
class TestSyncToFinalReports:

    def test_syncs_latest_report(self, writer, tmp_path):
        report = {
            "run_id": "sync1",
            "technique_name": "esd",
            "erase_concept": "nudity",
            "timestamp": 100.0,
        }
        writer.save_run(
            run_id="sync1",
            technique_name="esd",
            metric_name="asr_p4d",
            images=[_blank()],
            report=report,
            metadata={},
        )
        latest = str(tmp_path / "esd_nudity_latest_report.json")
        assert os.path.exists(latest)
        with open(latest) as f:
            data = json.load(f)
        assert data["run_id"] == "sync1"

    def test_older_report_does_not_overwrite_newer(self, writer, tmp_path):
        """A report with an older timestamp should not replace the current latest."""
        new_report = {
            "run_id": "newer",
            "technique_name": "esd",
            "erase_concept": "nudity",
            "timestamp": 200.0,
        }
        old_report = {
            "run_id": "older",
            "technique_name": "esd",
            "erase_concept": "nudity",
            "timestamp": 50.0,
        }
        writer.save_run(
            run_id="newer",
            technique_name="esd",
            metric_name="asr_p4d",
            images=[_blank()],
            report=new_report,
            metadata={},
        )
        writer.save_run(
            run_id="older",
            technique_name="esd",
            metric_name="asr_p4d",
            images=[_blank()],
            report=old_report,
            metadata={},
        )
        latest = str(tmp_path / "esd_nudity_latest_report.json")
        with open(latest) as f:
            data = json.load(f)
        assert data["run_id"] == "newer"


# ---------------------------------------------------------------------------
# _save_image static method
# ---------------------------------------------------------------------------
class TestSaveImage:

    def test_saves_valid_pil_image(self, tmp_path):
        from eval_learn.artifacts.writer import ArtifactWriter
        path = str(tmp_path / "test_save.png")
        result = ArtifactWriter._save_image(_blank(), path, 0)
        assert result == path
        assert os.path.exists(path)

    def test_non_saveable_object_returns_none(self, tmp_path):
        from eval_learn.artifacts.writer import ArtifactWriter
        result = ArtifactWriter._save_image("not_an_image", str(tmp_path / "x.png"), 0)
        assert result is None

    def test_failed_save_returns_none(self, tmp_path):
        from eval_learn.artifacts.writer import ArtifactWriter
        from unittest.mock import MagicMock
        bad_img = MagicMock()
        bad_img.save.side_effect = IOError("disk full")
        result = ArtifactWriter._save_image(bad_img, str(tmp_path / "fail.png"), 0)
        assert result is None
