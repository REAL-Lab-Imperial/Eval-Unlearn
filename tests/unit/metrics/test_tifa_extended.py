"""Extended tests for TIFA metric targeting uncovered lines."""
import pytest
from unittest.mock import MagicMock, Mock, patch
from PIL import Image

from eval_unlearn.types import MetricResult


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _make_tifa_metric(**kwargs):
    """Build TIFAMetric with a mocked MPLUG modelscope pipeline."""
    with patch("eval_unlearn.metrics.tifa.metric.pipeline") as mock_pipeline_fn:
        mock_pipeline_fn.return_value = Mock()
        from eval_unlearn.metrics.tifa.metric import TIFAMetric
        metric = TIFAMetric(**kwargs)
    return metric


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------
class TestTIFALoadDataset:
    def test_load_dataset_resets_counters(self):
        metric = _make_tifa_metric()
        metric._total_questions_count = 10
        metric._total_images_count = 3
        metric._per_image_scores = [1.0, 0.5]

        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.tifa_csv.load_tifa_csv", return_value=mock_loader):
            metric.load_dataset()

        assert metric._total_questions_count == 0
        assert metric._total_images_count == 0
        assert metric._per_image_scores == []

    def test_load_dataset_returns_loader(self):
        metric = _make_tifa_metric()
        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.tifa_csv.load_tifa_csv", return_value=mock_loader):
            result = metric.load_dataset()
        assert result is mock_loader

    def test_load_dataset_passes_limit(self):
        metric = _make_tifa_metric(limit=100)
        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.tifa_csv.load_tifa_csv", return_value=mock_loader) as mock_fn:
            metric.load_dataset()
        mock_fn.assert_called_once_with(limit=100)


# ---------------------------------------------------------------------------
# update — _answer and update logic
# ---------------------------------------------------------------------------
class TestTIFAUpdate:
    def _metric_with_answer(self, answer_text="yes"):
        metric = _make_tifa_metric()
        mock_vqa_pipeline = Mock(return_value={"text": answer_text})
        metric._vqa_pipeline = mock_vqa_pipeline
        return metric

    def test_update_correct_answer_counted(self):
        metric = self._metric_with_answer("yes")
        qa_pairs = [{"question": "Is this a cat?", "answer": "yes"}]
        metric.update([_dummy_image()], ["prompt"], {"qa_pairs": [qa_pairs]})
        assert metric._per_image_scores[0] == 1.0
        assert metric._total_questions_count == 1

    def test_update_wrong_answer_not_counted(self):
        metric = self._metric_with_answer("no")
        qa_pairs = [{"question": "Is this a cat?", "answer": "yes"}]
        metric.update([_dummy_image()], ["prompt"], {"qa_pairs": [qa_pairs]})
        assert metric._per_image_scores[0] == 0.0
        assert metric._total_questions_count == 1

    def test_update_none_image_skipped(self):
        metric = self._metric_with_answer("yes")
        metric.update([None], ["prompt"], {"qa_pairs": [[{"question": "q?", "answer": "a"}]]})
        assert metric._total_images_count == 1
        assert metric._per_image_scores == [None]

    def test_update_no_qa_pairs_skipped(self):
        metric = self._metric_with_answer("yes")
        metric.update([_dummy_image()], ["prompt"], {"qa_pairs": [None]})
        assert metric._total_images_count == 1
        assert metric._per_image_scores == [None]

    def test_update_empty_qa_pair_skipped(self):
        """QA pair with empty question or answer is skipped."""
        metric = self._metric_with_answer("yes")
        qa_pairs = [{"question": "", "answer": "yes"}, {"question": "q?", "answer": ""}]
        metric.update([_dummy_image()], ["prompt"], {"qa_pairs": [qa_pairs]})
        # Both have empty field, so no questions counted
        assert metric._total_questions_count == 0
        assert metric._per_image_scores == [None]

    def test_update_file_path_image(self, tmp_path):
        metric = self._metric_with_answer("cat")
        p = tmp_path / "img.png"
        _dummy_image().save(str(p))
        qa_pairs = [{"question": "what is this?", "answer": "cat"}]
        metric.update([str(p)], ["prompt"], {"qa_pairs": [qa_pairs]})
        assert metric._total_images_count == 1

    def test_update_missing_file_path(self):
        metric = self._metric_with_answer("yes")
        qa_pairs = [{"question": "q?", "answer": "yes"}]
        metric.update(["/nonexistent/img.png"], ["prompt"], {"qa_pairs": [qa_pairs]})
        assert metric._total_images_count == 1
        # Image couldn't be loaded, so qa pairs were skipped
        assert metric._per_image_scores == [None]

    def test_update_no_metadata_uses_defaults(self):
        metric = self._metric_with_answer("yes")
        # No metadata -> qa_pairs defaults to [None]
        metric.update([_dummy_image()], ["prompt"])
        assert metric._total_images_count == 1

    def test_update_rgba_image_converted(self):
        metric = self._metric_with_answer("yes")
        rgba_img = Image.new("RGBA", (16, 16), (50, 100, 150, 255))
        qa_pairs = [{"question": "q?", "answer": "yes"}]
        metric.update([rgba_img], ["prompt"], {"qa_pairs": [qa_pairs]})
        assert metric._total_images_count == 1

    def test_update_multiple_questions(self):
        metric = self._metric_with_answer("cat")
        qa_pairs = [
            {"question": "Is this a cat?", "answer": "cat"},
            {"question": "What color?", "answer": "cat"},  # Wrong expected answer
        ]
        metric.update([_dummy_image()], ["p"], {"qa_pairs": [qa_pairs]})
        assert metric._total_questions_count == 2
        assert metric._per_image_scores[0] == 1.0  # Both answered "cat"


# ---------------------------------------------------------------------------
# _answer
# ---------------------------------------------------------------------------
class TestTIFAAnswer:
    def test_answer_passes_pil_image_and_question(self):
        metric = _make_tifa_metric()
        metric._vqa_pipeline = Mock(return_value={"text": "cat"})
        img = _dummy_image()

        answer = metric._answer(img, "what is this?")

        assert answer == "cat"
        metric._vqa_pipeline.assert_called_once_with({"image": img, "question": "what is this?"})

    def test_answer_unwraps_list_text(self):
        metric = _make_tifa_metric()
        metric._vqa_pipeline = Mock(return_value={"text": ["cat"]})
        answer = metric._answer(_dummy_image(), "what is this?")
        assert answer == "cat"


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class TestTIFACompute:
    def test_compute_zero_images(self):
        metric = _make_tifa_metric()
        result = metric.compute()
        assert result.value == 0.0
        assert "error" in result.details

    def test_compute_macro_average_per_image(self):
        metric = _make_tifa_metric()
        metric._total_questions_count = 5
        metric._total_images_count = 2
        metric._per_image_scores = [1.0, 0.2]
        result = metric.compute()
        assert result.value == pytest.approx(0.6)

    def test_compute_no_questions(self):
        metric = _make_tifa_metric()
        metric._total_images_count = 1
        metric._total_questions_count = 0
        metric._per_image_scores = [None]
        result = metric.compute()
        assert result.value == 0.0

    def test_compute_includes_per_image_scores(self):
        metric = _make_tifa_metric()
        metric._total_questions_count = 2
        metric._total_images_count = 1
        metric._per_image_scores = [1.0]
        result = metric.compute()
        assert result.details["per_image_scores"] == [1.0]

    def test_compute_name_is_tifa(self):
        metric = _make_tifa_metric()
        metric._total_images_count = 1
        metric._per_image_scores = [1.0]
        result = metric.compute()
        assert result.name == "TIFA"

    def test_compute_includes_config(self):
        metric = _make_tifa_metric()
        metric._total_images_count = 1
        metric._per_image_scores = [1.0]
        result = metric.compute()
        assert "config" in result.details
