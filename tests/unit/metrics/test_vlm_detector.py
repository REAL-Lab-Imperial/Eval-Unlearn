"""Unit tests for the shared VLM-based ASR concept detector."""
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _make_detector(pipeline_return=None):
    with patch("eval_unlearn.metrics._vlm_detector.pipeline") as mock_pipeline_fn:
        mock_pipeline = MagicMock()
        if pipeline_return is not None:
            mock_pipeline.return_value = pipeline_return
        mock_pipeline_fn.return_value = mock_pipeline
        from eval_unlearn.metrics._vlm_detector import VLMConceptDetector
        detector = VLMConceptDetector(model_name="fake-model")
    return detector


class TestVLMConceptDetectorInit:
    def test_missing_modelscope_raises(self):
        with patch("eval_unlearn.metrics._vlm_detector.pipeline", None):
            from eval_unlearn.metrics._vlm_detector import VLMConceptDetector
            with pytest.raises(RuntimeError, match="modelscope"):
                VLMConceptDetector()

    def test_default_model_name(self):
        from eval_unlearn.metrics._vlm_detector import DEFAULT_VLM_MODEL
        assert DEFAULT_VLM_MODEL == "damo/mplug_visual-question-answering_coco_large_en"

    def test_loads_pipeline_with_given_model(self):
        with patch("eval_unlearn.metrics._vlm_detector.pipeline") as mock_pipeline_fn:
            mock_pipeline_fn.return_value = MagicMock()
            from eval_unlearn.metrics._vlm_detector import VLMConceptDetector
            VLMConceptDetector(model_name="custom-model")
        mock_pipeline_fn.assert_called_once()


class TestVLMConceptDetectorAnswer:
    def test_answer_lowercased_and_stripped(self):
        detector = _make_detector(pipeline_return={"text": "  Yes  "})
        answer = detector._answer(_dummy_image(), "Is there a dog?")
        assert answer == "yes"

    def test_answer_unwraps_list_response(self):
        detector = _make_detector(pipeline_return={"text": ["No"]})
        answer = detector._answer(_dummy_image(), "Is there a dog?")
        assert answer == "no"

    def test_answer_converts_non_rgb_image(self):
        detector = _make_detector(pipeline_return={"text": "yes"})
        rgba_img = Image.new("RGBA", (16, 16), (50, 100, 150, 255))
        answer = detector._answer(rgba_img, "Is there a dog?")
        assert answer == "yes"


class TestIsConceptPresent:
    def test_yes_answer_is_present(self):
        detector = _make_detector(pipeline_return={"text": "yes"})
        assert detector.is_concept_present(_dummy_image(), "nudity") is True

    def test_no_answer_is_not_present(self):
        detector = _make_detector(pipeline_return={"text": "no"})
        assert detector.is_concept_present(_dummy_image(), "nudity") is False

    def test_question_includes_concept(self):
        detector = _make_detector()
        with patch.object(detector, "_answer", return_value="yes") as mock_answer:
            detector.is_concept_present(_dummy_image(), "a dog wearing a hat")
        args, _ = mock_answer.call_args
        assert "a dog wearing a hat" in args[1]


class TestIsConceptPresentBatch:
    def test_batch_of_pil_images(self):
        detector = _make_detector()
        with patch.object(detector, "is_concept_present", side_effect=[True, False]):
            results = detector.is_concept_present_batch(
                [_dummy_image(), _dummy_image()], "nudity"
            )
        assert results == [True, False]

    def test_invalid_image_counts_as_not_present(self):
        detector = _make_detector()
        results = detector.is_concept_present_batch([42, "not an image"], "nudity")
        assert results == [False, False]

    def test_mixed_valid_and_invalid(self):
        detector = _make_detector()
        with patch.object(detector, "is_concept_present", return_value=True):
            results = detector.is_concept_present_batch([_dummy_image(), None], "nudity")
        assert results == [True, False]

    def test_per_image_exception_does_not_abort_batch(self):
        detector = _make_detector()
        with patch.object(
            detector, "is_concept_present", side_effect=[RuntimeError("boom"), True]
        ):
            results = detector.is_concept_present_batch(
                [_dummy_image(), _dummy_image()], "nudity"
            )
        assert results == [False, True]

    def test_numpy_array_image_converted(self):
        import numpy as np

        detector = _make_detector()
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch.object(detector, "is_concept_present", return_value=False) as mock_ic:
            results = detector.is_concept_present_batch([arr], "nudity")
        assert results == [False]
        assert isinstance(mock_ic.call_args[0][0], Image.Image)
