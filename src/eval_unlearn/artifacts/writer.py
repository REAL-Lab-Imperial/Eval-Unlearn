import os
import json
from typing import List, Any, Dict, Optional
from ..logging_utils import get_logger

logger = get_logger(__name__)


class ArtifactWriter:
    """
    Handles saving of benchmark artifacts (images and reports).

    Folder layout::

        <base_dir>/
          <technique>_<metric>_<run_id>/
            images/
              <category>/        ← only if metadata has categories
                0.png, 1.png
              0.png, 1.png       ← flat if no categories
            <run_id>_report.json
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = base_dir

    def save_run(
        self,
        run_id: str,
        technique_name: str,
        metric_name: str,
        images: List[Any],
        report: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        detailed_report: Optional[Dict[str, Any]] = None,
        image_index_offset: int = 0,
        category_counters_init: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Save images and report for a benchmark run.

        Args:
            run_id: Short hash identifying this run.
            technique_name: Name of the technique (e.g. ``"sld"``).
            metric_name: Name of the metric (e.g. ``"asr"``).
            images: List of generated PIL images.
            report: Simplified result dictionary to persist as ``{run_id}_report.json``.
                If None, no report is saved.
            metadata: Dataset metadata. If it contains a ``categories``
                key (list parallel to *images*), images are saved into
                per-category subdirectories.
            detailed_report: Extended result dictionary including technique and metric
                configs, saved alongside the simplified report as
                ``{run_id}_report_full.json``. If None, no detailed report is saved.
            image_index_offset: Starting index for flat image filenames. Pass
                ``total_generated`` before the current batch to avoid overwriting
                images saved in earlier batches.
            category_counters_init: Per-category filename counters accumulated from
                previous batches, used to continue numbering in category-aware saving.

        Returns:
            Path to the saved simplified report JSON (or where it would be saved).
        """
        metadata = metadata or {}
        categories = metadata.get("categories")

        image_paths = []

        # If images provided, save them in structured folder
        if images:
            # Build folder: <base_dir>/<technique>_<metric>_<run_id>/
            folder_name = f"{technique_name}_{metric_name}_{run_id}"
            run_dir = os.path.join(self.base_dir, folder_name)
            images_dir = os.path.join(run_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Save images — category-aware if metadata provides categories
            logger.info(f"Saving {len(images)} images to {images_dir}...")

            if categories and len(categories) == len(images):
                # Category-aware: save into subdirectories
                category_counters: Dict[str, int] = dict(category_counters_init or {})
                for img, cat in zip(images, categories):
                    cat_dir = os.path.join(images_dir, cat.lower())
                    os.makedirs(cat_dir, exist_ok=True)
                    idx = category_counters.get(cat.lower(), 0)
                    category_counters[cat.lower()] = idx + 1
                    path = os.path.join(cat_dir, f"{technique_name}_{metric_name}_{run_id}_{idx}.png")
                    image_paths.append(self._save_image(img, path, idx))
            else:
                # Flat: save numbered images directly
                for i, img in enumerate(images):
                    global_i = i + image_index_offset
                    path = os.path.join(images_dir, f"{technique_name}_{metric_name}_{run_id}_{global_i}.png")
                    image_paths.append(self._save_image(img, path, global_i))

            # Filter out None (failed saves)
            image_paths = [p for p in image_paths if p is not None]
        else:
            logger.info("No images to save")

        # Resolve the directory where reports are saved
        if images:
            folder_name = f"{technique_name}_{metric_name}_{run_id}"
            report_dir = os.path.join(self.base_dir, folder_name)
        else:
            os.makedirs(self.base_dir, exist_ok=True)
            report_dir = self.base_dir

        report_path = os.path.join(report_dir, f"{run_id}_report.json")

        # Save simplified report
        if report is not None:
            try:
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=4)
                logger.info(f"Report saved to {report_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        else:
            logger.info("Skipping report save (not provided)")

        # Save detailed report
        if detailed_report is not None:
            detailed_path = os.path.join(report_dir, f"{run_id}_report_full.json")
            try:
                with open(detailed_path, "w") as f:
                    json.dump(detailed_report, f, indent=4)
                logger.info(f"Detailed report saved to {detailed_path}")
            except Exception as e:
                logger.error(f"Failed to save detailed report: {e}")

        if report is not None:
            self._sync_to_final_reports(report_path, report)

        return report_path

    def _sync_to_final_reports(self, report_path: str, report: Dict[str, Any]) -> None:
        """Copy report as {technique}_{concept}_latest_report.json in base_dir if newer."""
        technique = report.get("technique_name", "unknown")
        concept = report.get("erase_concept", "unknown")
        timestamp = report.get("timestamp", 0)

        dest = os.path.join(self.base_dir, f"{technique}_{concept}_latest_report.json")

        if os.path.exists(dest):
            try:
                with open(dest) as f:
                    existing = json.load(f)
                if existing.get("timestamp", 0) >= timestamp:
                    logger.info(f"Skipping latest report sync — existing is newer or equal: {dest}")
                    return
            except Exception:
                pass

        try:
            import shutil
            shutil.copy2(report_path, dest)
            logger.info(f"Synced latest report to {dest}")
        except Exception as e:
            logger.error(f"Failed to sync latest report: {e}")

    @staticmethod
    def _save_image(img: Any, path: str, index: int) -> Optional[str]:
        """Save a single image, returning the path or None on failure."""
        try:
            if hasattr(img, "save"):
                img.save(path)
                return path
            else:
                logger.warning(f"Image {index} does not support save(), skipping.")
                return None
        except Exception as e:
            logger.error(f"Failed to save image {index}: {e}")
            return None
