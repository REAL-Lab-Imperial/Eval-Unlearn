"""Unit tests for UA_IRA (Unlearning Accuracy & In-domain Retain Accuracy) metric."""

import tempfile
from unittest.mock import Mock, patch, MagicMock
import pytest
import torch
from PIL import Image

from eval_learn.metrics.ua_ira.metric import UAIRAMetric
from eval_learn.metrics.ua_ira.config import UAIRAConfig
from eval_learn.types import MetricResult

# Dummy CSV paths used when the config requires non-empty paths.
_DUMMY_TARGET = "/nonexistent/target.csv"
_DUMMY_RETAIN = "/nonexistent/retain.csv"


def _make_ua_ira_metric(**kwargs):
    """
    Helper: create UAIRAMetric with mocked CLIP model/processor.

    UAIRAConfig requires target_prompts_path and retain_prompts_path to be set.
    Tests that do not care about those fields should use this helper which
    provides dummy values.
    """
    kwargs.setdefault("target_prompts_path", _DUMMY_TARGET)
    kwargs.setdefault("retain_prompts_path", _DUMMY_RETAIN)

    with patch("eval_learn.metrics.ua_ira.metric.CLIPModel") as mock_model_cls, \
         patch("eval_learn.metrics.ua_ira.metric.CLIPProcessor") as mock_proc_cls, \
         patch("eval_learn.metrics.ua_ira.metric.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.no_grad = MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=None),
                __exit__=MagicMock(return_value=False),
            )
        )

        mock_model = Mock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_proc = Mock()
        mock_proc_cls.from_pretrained.return_value = mock_proc

        metric = UAIRAMetric(**kwargs)

    return metric


class TestUAIRAConfig:
    """Test UAIRAConfig initialization and validation."""

    def test_config_defaults_without_paths_raises(self):
        """Config raises if target_prompts_path is empty (required field)."""
        with pytest.raises(ValueError, match="target_prompts_path"):
            UAIRAConfig()

    def test_config_with_paths(self):
        """Config is created successfully when both paths are provided."""
        config = UAIRAConfig(
            target_prompts_path="/path/to/target.csv",
            retain_prompts_path="/path/to/retain.csv",
        )
        assert config.clip_model_name == "openai/clip-vit-large-patch14"
        assert config.device is None
        assert config.target_prompts_path == "/path/to/target.csv"
        assert config.retain_prompts_path == "/path/to/retain.csv"
        assert config.target_concept == "target_concept"
        assert config.retain_concept == "retain_concept"
        assert config.batch_size == 32

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "clip_model_name":  "openai/clip-vit-large-patch14",
            "device": "cpu",
            "target_prompts_path": "/path/to/target.csv",
            "retain_prompts_path": "/path/to/retain.csv",
            "target_concept": "nudity",
            "retain_concept": "person",
            "batch_size": 16,
        }
        config = UAIRAConfig.from_dict(config_dict)
        assert config.clip_model_name ==  "openai/clip-vit-large-patch14"
        assert config.device == "cpu"
        assert config.target_prompts_path == "/path/to/target.csv"
        assert config.retain_prompts_path == "/path/to/retain.csv"
        assert config.target_concept == "nudity"
        assert config.retain_concept == "person"
        assert config.batch_size == 16

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = UAIRAConfig(
            target_prompts_path="/t.csv",
            retain_prompts_path="/r.csv",
            device="cpu",
            target_concept="nudity",
            retain_concept="person",
            batch_size=64,
        )
        config_dict = config.to_dict()
        assert config_dict["device"] == "cpu"
        assert config_dict["target_concept"] == "nudity"
        assert config_dict["retain_concept"] == "person"
        assert config_dict["batch_size"] == 64


class TestUAIRAMetricInitialization:
    """Test UAIRAMetric initialization."""

    def test_init_success_cpu(self):
        """Test successful initialization on CPU."""
        metric = _make_ua_ira_metric(device="cpu")

        assert metric.device == "cpu"
        assert metric.model is not None
        assert metric.processor is not None
        assert metric._target_correct_count == 0
        assert metric._target_total_count == 0
        assert metric._retain_correct_count == 0
        assert metric._retain_total_count == 0

    def test_init_auto_detect_device(self):
        """Test device auto-detection when device is None."""
        metric = _make_ua_ira_metric(device=None)
        assert metric.device == "cpu"


class TestUAIRALoadDataset:
    """Test load_dataset method."""

    def test_load_dataset_missing_paths_raises_at_config(self):
        """Config raises ValueError when target_prompts_path is empty."""
        with pytest.raises(ValueError, match="target_prompts_path"):
            UAIRAConfig(retain_prompts_path="/r.csv")

    def test_load_dataset_resets_counters(self):
        """Test that load_dataset resets counters."""
        metric = _make_ua_ira_metric(
            device="cpu",
            target_prompts_path="/path/to/target.csv",
            retain_prompts_path="/path/to/retain.csv",
        )

        # Set non-zero counters
        metric._target_correct_count = 10
        metric._target_total_count = 20
        metric._retain_correct_count = 15
        metric._retain_total_count = 25

        # Mock the load function from the datasets module
        with patch("eval_learn.datasets.ua_ira_csv.load_ua_ira_csv", return_value=Mock()):
            metric.load_dataset()

            assert metric._target_correct_count == 0
            assert metric._target_total_count == 0
            assert metric._retain_correct_count == 0
            assert metric._retain_total_count == 0


class TestUAIRAToPIL:
    """Test _to_pil static method."""

    def test_to_pil_with_pil_image(self):
        """Test _to_pil with PIL Image returns unchanged."""
        img = Image.new("RGB", (10, 10), color="red")
        result = UAIRAMetric._to_pil(img)
        assert result is img

    def test_to_pil_with_valid_file_path(self):
        """Test _to_pil with valid file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = f"{tmpdir}/test.png"
            img = Image.new("RGB", (10, 10), color="blue")
            img.save(img_path)

            result = UAIRAMetric._to_pil(img_path)
            assert isinstance(result, Image.Image)
            assert result.mode == "RGB"

    def test_to_pil_with_invalid_file_path(self):
        """Test _to_pil with invalid file path returns None."""
        result = UAIRAMetric._to_pil("/nonexistent/path.png")
        assert result is None

    def test_to_pil_with_invalid_type(self):
        """Test _to_pil with invalid type returns None."""
        result = UAIRAMetric._to_pil(12345)
        assert result is None


class TestUAIRAMetricUpdate:
    """Test update() method."""

    def test_update_empty_images(self):
        """Test update with empty images list."""
        metric = _make_ua_ira_metric(device="cpu")

        # Should not raise
        metric.update([], ["prompt"], {})

        assert metric._target_total_count == 0
        assert metric._retain_total_count == 0

    def test_update_stores_target_prompt_end_index(self):
        """Test that update stores target_prompt_end_index from metadata."""
        metric = _make_ua_ira_metric(device="cpu")
        metric._target_prompt_end_index = 0

        metadata = {"target_prompt_end_index": 3}
        metric.update([], ["prompt"], metadata)

        assert metric._target_prompt_end_index == 3

    def test_update_splits_target_retain_images(self):
        """Test that update correctly splits target and retain images."""
        metric = _make_ua_ira_metric(device="cpu")

        # Create 5 images, split at index 2 (2 target, 3 retain)
        imgs = [Image.new("RGB", (10, 10), color=c) for c in ["red", "blue", "green", "yellow", "pink"]]
        metadata = {"target_prompt_end_index": 2}

        with patch.object(metric, "_evaluate_batch") as mock_eval:
            metric.update(imgs, ["prompt"] * 5, metadata)

            # Should call _evaluate_batch twice
            assert mock_eval.call_count == 2
            # First call with 2 target images
            first_call = mock_eval.call_args_list[0]
            assert len(first_call[0][0]) == 2
            assert first_call[1]["is_target"] is True

            # Second call with 3 retain images
            second_call = mock_eval.call_args_list[1]
            assert len(second_call[0][0]) == 3
            assert second_call[1]["is_target"] is False


class TestUAIRAEvaluateBatch:
    """Test _evaluate_batch method indirectly through update."""

    def test_evaluate_batch_called_with_correct_args(self):
        """Test that _evaluate_batch is called with correct image splits."""
        metric = _make_ua_ira_metric(device="cpu")

        imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(5)]

        # Mock _evaluate_batch to verify it's called correctly
        with patch.object(metric, "_evaluate_batch") as mock_eval:
            metric.update(imgs, ["prompt"] * 5, {"target_prompt_end_index": 2})

            # Verify _evaluate_batch was called twice
            assert mock_eval.call_count == 2

            # Check target images call
            target_call = mock_eval.call_args_list[0]
            assert len(target_call[0][0]) == 2
            assert target_call[1]["is_target"] is True

            # Check retain images call
            retain_call = mock_eval.call_args_list[1]
            assert len(retain_call[0][0]) == 3
            assert retain_call[1]["is_target"] is False

    def test_evaluate_batch_updates_target_counts(self):
        """Test that _evaluate_batch updates target counts correctly."""
        metric = _make_ua_ira_metric(device="cpu")

        imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(3)]

        # Mock _evaluate_batch to set counts
        def mock_eval_target(images, prompts, is_target):
            if is_target:
                metric._target_total_count = 3
                metric._target_correct_count = 3

        with patch.object(metric, "_evaluate_batch", side_effect=mock_eval_target):
            metric.update(imgs, ["prompt"] * 3, {"target_prompt_end_index": 3})

            assert metric._target_total_count == 3
            assert metric._target_correct_count == 3

    def test_evaluate_batch_updates_retain_counts(self):
        """Test that _evaluate_batch updates retain counts correctly."""
        metric = _make_ua_ira_metric(device="cpu")

        imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(4)]

        # Mock _evaluate_batch to set counts
        def mock_eval_retain(images, prompts, is_target):
            if not is_target:
                metric._retain_total_count = 4
                metric._retain_correct_count = 3

        with patch.object(metric, "_evaluate_batch", side_effect=mock_eval_retain):
            metric.update(imgs, ["prompt"] * 4, {"target_prompt_end_index": 0})

            assert metric._retain_total_count == 4
            assert metric._retain_correct_count == 3


class TestUAIRAMetricComputation:
    """Test compute() method."""

    def test_compute_no_images(self):
        """Test compute with no images returns 0.0."""
        metric = _make_ua_ira_metric(device="cpu")

        result = metric.compute()

        assert result.name == "UA_IRA"
        assert result.value == 0.0
        assert result.details["ua_score"] == 0.0
        assert result.details["ira_score"] == 0.0

    def test_compute_perfect_ua_and_ira(self):
        """Test compute with perfect UA and IRA scores."""
        metric = _make_ua_ira_metric(device="cpu")

        # All target images correctly unlearned (UA = 1.0)
        metric._target_correct_count = 10
        metric._target_total_count = 10
        # All retain images correctly retained (IRA = 1.0)
        metric._retain_correct_count = 10
        metric._retain_total_count = 10

        result = metric.compute()

        assert result.name == "UA_IRA"
        assert result.value == pytest.approx(1.0)
        assert result.details["ua_score"] == pytest.approx(1.0)
        assert result.details["ira_score"] == pytest.approx(1.0)

    def test_compute_partial_ua_and_ira(self):
        """Test compute with partial UA and IRA scores."""
        metric = _make_ua_ira_metric(device="cpu")

        # 8 out of 10 target images correctly unlearned (UA = 0.8)
        metric._target_correct_count = 8
        metric._target_total_count = 10
        # 7 out of 10 retain images correctly retained (IRA = 0.7)
        metric._retain_correct_count = 7
        metric._retain_total_count = 10

        result = metric.compute()

        assert result.name == "UA_IRA"
        assert result.value == pytest.approx(0.75)  # (0.8 + 0.7) / 2
        assert result.details["ua_score"] == pytest.approx(0.8)
        assert result.details["ira_score"] == pytest.approx(0.7)

    def test_compute_returns_metric_result(self):
        """Test that compute returns MetricResult instance."""
        metric = _make_ua_ira_metric(device="cpu")
        metric._target_total_count = 5
        metric._target_correct_count = 3
        metric._retain_total_count = 5
        metric._retain_correct_count = 4

        result = metric.compute()

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert isinstance(result.details, dict)
        assert "config" in result.details
        assert "ua_score" in result.details
        assert "ira_score" in result.details

    def test_compute_includes_details(self):
        """Test that compute includes all expected details."""
        metric = _make_ua_ira_metric(device="cpu", batch_size=16)
        metric._target_total_count = 20
        metric._target_correct_count = 16
        metric._retain_total_count = 20
        metric._retain_correct_count = 18

        result = metric.compute()

        assert result.details["target_correct"] == 16
        assert result.details["target_total"] == 20
        assert result.details["retain_correct"] == 18
        assert result.details["retain_total"] == 20
        assert result.details["config"]["batch_size"] == 16


class TestUAIRAMetricIntegration:
    """Integration tests for UA_IRA metric workflow."""

    def test_full_workflow_update_compute(self):
        """Test complete workflow: initialize → update → compute."""
        metric = _make_ua_ira_metric(
            device="cpu", target_concept="nudity", retain_concept="person"
        )

        # Create test data: 2 target + 4 retain = 6 images
        imgs = [Image.new("RGB", (10, 10), color="red") for _ in range(6)]
        metadata = {"target_prompt_end_index": 2}

        # Mock _evaluate_batch to simulate evaluation results
        def mock_eval_side_effect(images, prompts, is_target):
            if is_target:
                # 2 target images, both correctly unlearned (index 1)
                metric._target_total_count += len(images)
                metric._target_correct_count += len(images)
            else:
                # 4 retain images, 3 correctly retained (index 1)
                metric._retain_total_count += len(images)
                metric._retain_correct_count += 3

        with patch.object(metric, "_evaluate_batch", side_effect=mock_eval_side_effect):
            metric.update(imgs, ["prompt"] * 6, metadata)

            result = metric.compute()

            assert result.name == "UA_IRA"
            assert result.details["target_correct"] == 2
            assert result.details["target_total"] == 2
            assert result.details["retain_correct"] == 3
            assert result.details["retain_total"] == 4
            # UA = 2/2 = 1.0, IRA = 3/4 = 0.75, avg = 0.875
            assert result.value == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# UAIRAConfig validation
# ---------------------------------------------------------------------------
class TestUAIRAConfigValidation:
    def test_missing_retain_prompts_path_raises(self):
        from eval_learn.metrics.ua_ira.config import UAIRAConfig
        with pytest.raises(ValueError, match="retain_prompts_path"):
            UAIRAConfig(target_prompts_path="target.csv")
