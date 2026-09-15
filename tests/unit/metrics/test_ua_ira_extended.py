"""Extended tests for UA_IRA metric targeting uncovered lines."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch

from eval_unlearn.types import MetricResult


def _dummy_image(color=(50, 100, 150)):
    return Image.new("RGB", (16, 16), color=color)


def _make_ua_ira_metric(tmp_path=None, **kwargs):
    """Build UAIRAMetric with mocked CLIP and required paths."""
    import tempfile, os

    # Create temp CSV files for paths if not specified
    if "target_prompts_path" not in kwargs:
        tmp = tempfile.mkdtemp()
        tp = os.path.join(tmp, "target.csv")
        rp = os.path.join(tmp, "retain.csv")
        import pandas as pd
        pd.DataFrame({"prompt": ["p1"]}).to_csv(tp, index=False)
        pd.DataFrame({"prompt": ["p2"]}).to_csv(rp, index=False)
        kwargs["target_prompts_path"] = tp
        kwargs["retain_prompts_path"] = rp

    with patch("eval_unlearn.metrics.ua_ira.metric.CLIPModel") as mock_cls, \
         patch("eval_unlearn.metrics.ua_ira.metric.CLIPProcessor") as mock_proc_cls:
        mock_model = MagicMock()
        mock_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        from eval_unlearn.metrics.ua_ira.metric import UAIRAMetric
        metric = UAIRAMetric(**kwargs)
    return metric


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------
class TestUAIRALoadDataset:
    def test_load_dataset_resets_counters(self, tmp_path):
        import pandas as pd
        tp = tmp_path / "target.csv"
        rp = tmp_path / "retain.csv"
        pd.DataFrame({"prompt": ["p1"]}).to_csv(tp, index=False)
        pd.DataFrame({"prompt": ["p2"]}).to_csv(rp, index=False)

        metric = _make_ua_ira_metric(
            target_prompts_path=str(tp),
            retain_prompts_path=str(rp),
        )
        metric._target_correct_count = 5
        metric._target_total_count = 10
        metric._retain_correct_count = 3
        metric._retain_total_count = 8

        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.ua_ira_csv.load_ua_ira_csv", return_value=mock_loader):
            metric.load_dataset()

        # Counters should be reset
        assert metric._target_correct_count == 0
        assert metric._target_total_count == 0
        assert metric._retain_correct_count == 0

    def test_load_dataset_missing_paths_raises_directly(self):
        """The config validation catches missing paths before creating the metric."""
        from eval_unlearn.metrics.ua_ira.config import UAIRAConfig
        with pytest.raises(ValueError, match="target_prompts_path must be set"):
            UAIRAConfig(target_prompts_path="", retain_prompts_path="some/path")

    def test_load_dataset_calls_load_ua_ira_csv(self, tmp_path):
        import pandas as pd
        tp = tmp_path / "target.csv"
        rp = tmp_path / "retain.csv"
        pd.DataFrame({"prompt": ["p1"]}).to_csv(tp, index=False)
        pd.DataFrame({"prompt": ["p2"]}).to_csv(rp, index=False)

        metric = _make_ua_ira_metric(
            target_prompts_path=str(tp),
            retain_prompts_path=str(rp),
        )
        mock_loader = MagicMock()
        with patch("eval_unlearn.datasets.ua_ira_csv.load_ua_ira_csv", return_value=mock_loader) as mock_fn:
            result = metric.load_dataset()

        assert result is mock_loader
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# update — _evaluate_batch (lines 142-177)
# ---------------------------------------------------------------------------
class TestUAIRAUpdate:
    def _metric_with_logits(self, predicted_class=1):
        metric = _make_ua_ira_metric()
        mock_model = MagicMock()
        mock_output = MagicMock()
        # Shape: (batch, 2), argmax gives predicted_class
        logits = torch.zeros(1, 2)
        logits[0, predicted_class] = 10.0
        mock_output.logits_per_image = logits
        mock_model.return_value = mock_output
        metric.model = mock_model
        mock_proc = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_proc.return_value = mock_inputs
        metric.processor = mock_proc
        return metric

    def test_update_target_images_counted(self):
        metric = self._metric_with_logits(predicted_class=1)
        # target_prompt_end_index=1 means first image is target
        images = [_dummy_image()]
        metadata = {"target_prompt_end_index": 1}
        metric.update(images, ["p"], metadata)
        assert metric._target_total_count == 1
        assert metric._target_correct_count == 1  # predicted retain (1) for target

    def test_update_retain_images_counted(self):
        metric = self._metric_with_logits(predicted_class=1)
        # target_prompt_end_index=0 means no target images, all retain
        images = [_dummy_image()]
        metadata = {"target_prompt_end_index": 0}
        metric.update(images, ["p"], metadata)
        assert metric._retain_total_count == 1
        assert metric._retain_correct_count == 1  # predicted retain (1) = correct

    def test_update_empty_images_skipped(self):
        metric = self._metric_with_logits()
        metric.update([], ["p"], {"target_prompt_end_index": 0})
        assert metric._target_total_count == 0
        assert metric._retain_total_count == 0

    def test_update_sets_target_prompt_end_index(self):
        metric = self._metric_with_logits()
        assert metric._target_prompt_end_index == 0
        images = [_dummy_image()]
        metric.update(images, ["p"], {"target_prompt_end_index": 5})
        assert metric._target_prompt_end_index == 5

    def test_update_uses_stored_index_on_second_call(self):
        metric = self._metric_with_logits(predicted_class=1)
        metric._target_prompt_end_index = 1
        images = [_dummy_image()]
        metric.update(images, ["p"], {})  # No metadata
        # target_prompt_end_index=1 -> first image is target
        assert metric._target_total_count == 1

    def test_evaluate_batch_handles_exception(self):
        metric = _make_ua_ira_metric()
        metric.model.side_effect = RuntimeError("crash")
        mock_proc = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_proc.return_value = mock_inputs
        metric.processor = mock_proc
        # Should not raise
        metric._evaluate_batch([_dummy_image()], ["Image of cat", "Image of dog"], is_target=True)
        assert metric._target_total_count == 0

    def test_evaluate_batch_invalid_image_skipped(self):
        metric = self._metric_with_logits()
        # Pass None image
        metric._evaluate_batch([None], ["Image of cat", "Image of dog"], is_target=True)
        # pil_images will be filtered to empty, so nothing evaluated

    def test_to_pil_with_pil_image(self):
        from eval_unlearn.metrics.ua_ira.metric import UAIRAMetric
        img = _dummy_image()
        result = UAIRAMetric._to_pil(img)
        assert result is img

    def test_to_pil_with_valid_path(self, tmp_path):
        from eval_unlearn.metrics.ua_ira.metric import UAIRAMetric
        p = tmp_path / "img.png"
        _dummy_image().save(str(p))
        result = UAIRAMetric._to_pil(str(p))
        assert isinstance(result, Image.Image)

    def test_to_pil_with_missing_path(self):
        from eval_unlearn.metrics.ua_ira.metric import UAIRAMetric
        result = UAIRAMetric._to_pil("/nonexistent/path.png")
        assert result is None

    def test_to_pil_with_invalid_type(self):
        from eval_unlearn.metrics.ua_ira.metric import UAIRAMetric
        result = UAIRAMetric._to_pil(42)
        assert result is None


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
class TestUAIRACompute:
    def test_compute_both_zero(self):
        metric = _make_ua_ira_metric()
        result = metric.compute()
        assert isinstance(result, MetricResult)
        assert result.value == 0.0

    def test_compute_target_only(self):
        metric = _make_ua_ira_metric()
        metric._target_correct_count = 7
        metric._target_total_count = 10
        result = metric.compute()
        assert result.details["ua_score"] == pytest.approx(0.7)
        assert result.details["ira_score"] == 0.0

    def test_compute_retain_only(self):
        metric = _make_ua_ira_metric()
        metric._retain_correct_count = 8
        metric._retain_total_count = 10
        result = metric.compute()
        assert result.details["ira_score"] == pytest.approx(0.8)
        assert result.details["ua_score"] == 0.0

    def test_compute_both_full(self):
        metric = _make_ua_ira_metric()
        metric._target_correct_count = 6
        metric._target_total_count = 10
        metric._retain_correct_count = 8
        metric._retain_total_count = 10
        result = metric.compute()
        assert abs(result.details["ua_score"] - 0.6) < 1e-6
        assert abs(result.details["ira_score"] - 0.8) < 1e-6
        assert abs(result.value - 0.7) < 1e-6

    def test_compute_includes_config(self):
        metric = _make_ua_ira_metric()
        result = metric.compute()
        assert "config" in result.details

    def test_compute_name_is_ua_ira(self):
        metric = _make_ua_ira_metric()
        result = metric.compute()
        assert result.name == "UA_IRA"


# ---------------------------------------------------------------------------
# ua_ira/config.py and metric.py — uncovered validation paths
# ---------------------------------------------------------------------------
class TestUAIRACoverageGaps:
    def test_config_missing_target_path_raises(self):
        from eval_unlearn.metrics.ua_ira.config import UAIRAConfig
        with pytest.raises(ValueError, match="target_prompts_path"):
            UAIRAConfig.from_dict({"retain_prompts_path": "/some/retain.csv"})

    def test_load_dataset_raises_without_paths(self):
        metric = _make_ua_ira_metric()
        object.__setattr__(metric.config, "target_prompts_path", "")
        object.__setattr__(metric.config, "retain_prompts_path", "")
        with pytest.raises(ValueError, match="target_prompts_path"):
            metric.load_dataset()
