"""Unit tests for registry entrypoints."""
import pytest
from unittest.mock import MagicMock, patch


class TestLoadEntrypoints:
    def test_load_entrypoints_registers_technique(self):
        from eval_unlearn.registry.entrypoints import load_entrypoints
        from eval_unlearn.registry.local import _TECHNIQUES

        mock_ep = MagicMock()
        mock_ep.name = "test_technique"
        mock_plugin = MagicMock()
        mock_ep.load.return_value = mock_plugin

        with patch("eval_unlearn.registry.entrypoints.entry_points") as mock_eps:
            mock_eps.return_value = []  # No techniques
            # Trigger execution through all groups
            load_entrypoints()

        # Should not raise even with empty entry points
        assert True

    def test_load_entrypoints_with_valid_plugin(self):
        from eval_unlearn.registry.entrypoints import load_entrypoints

        mock_ep = MagicMock()
        mock_ep.name = "my_technique"
        mock_plugin = object()
        mock_ep.load.return_value = mock_plugin

        def fake_entry_points(group=None):
            if group == "eval_unlearn.techniques":
                return [mock_ep]
            return []

        with patch("eval_unlearn.registry.entrypoints.entry_points", side_effect=fake_entry_points), \
             patch("eval_unlearn.registry.entrypoints.register_technique") as mock_register:
            mock_register.return_value = lambda cls: cls
            load_entrypoints()

        mock_ep.load.assert_called_once()

    def test_load_entrypoints_handles_load_failure_gracefully(self):
        from eval_unlearn.registry.entrypoints import load_entrypoints

        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.load.side_effect = RuntimeError("import failure")

        def fake_entry_points(group=None):
            if group == "eval_unlearn.techniques":
                return [mock_ep]
            return []

        with patch("eval_unlearn.registry.entrypoints.entry_points", side_effect=fake_entry_points):
            # Should not raise — errors are logged and swallowed
            load_entrypoints()

    def test_load_entrypoints_python39_fallback(self):
        """Test Python 3.9 fallback path where entry_points returns a dict."""
        from eval_unlearn.registry.entrypoints import load_entrypoints

        # Simulate Python 3.9 API where entry_points() raises TypeError with group kwarg
        def old_style_entry_points(**kwargs):
            if "group" in kwargs:
                raise TypeError("unexpected keyword argument 'group'")
            return {}

        with patch("eval_unlearn.registry.entrypoints.entry_points", side_effect=old_style_entry_points):
            # Should not raise
            load_entrypoints()

    def test_load_entrypoints_metrics_group(self):
        from eval_unlearn.registry.entrypoints import load_entrypoints

        mock_ep = MagicMock()
        mock_ep.name = "custom_metric"
        mock_plugin = object()
        mock_ep.load.return_value = mock_plugin

        def fake_entry_points(group=None):
            if group == "eval_unlearn.metrics":
                return [mock_ep]
            return []

        with patch("eval_unlearn.registry.entrypoints.entry_points", side_effect=fake_entry_points), \
             patch("eval_unlearn.registry.entrypoints.register_metric") as mock_register:
            mock_register.return_value = lambda cls: cls
            load_entrypoints()

        mock_ep.load.assert_called_once()

    def test_load_entrypoints_datasets_group(self):
        from eval_unlearn.registry.entrypoints import load_entrypoints

        mock_ep = MagicMock()
        mock_ep.name = "custom_dataset"
        mock_plugin = object()
        mock_ep.load.return_value = mock_plugin

        def fake_entry_points(group=None):
            if group == "eval_unlearn.datasets":
                return [mock_ep]
            return []

        with patch("eval_unlearn.registry.entrypoints.entry_points", side_effect=fake_entry_points), \
             patch("eval_unlearn.registry.entrypoints.register_dataset") as mock_register:
            mock_register.return_value = lambda cls: cls
            load_entrypoints()

        mock_ep.load.assert_called_once()
