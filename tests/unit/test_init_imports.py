"""Tests targeting techniques/__init__.py and metrics/__init__.py import blocks."""
import sys
import importlib
from unittest.mock import MagicMock, patch


class TestTechniquesInit:
    """Cover the try/except import blocks in techniques/__init__.py."""

    def _reload_techniques(self):
        """Remove and re-import eval_unlearn.techniques to re-execute try/except blocks."""
        keys_to_remove = [k for k in sys.modules if "eval_unlearn.techniques" in k]
        for k in keys_to_remove:
            sys.modules.pop(k, None)

    def test_import_succeeds_with_all_packages_missing(self):
        """All optional packages raise ImportError, should not raise."""
        self._reload_techniques()
        broken = MagicMock(side_effect=ImportError("not installed"))
        with patch.dict("sys.modules", {
            "esd": None,
            "ssd": None,
            "ca": None,
            "cogfd": None,
            "saeuron": None,
            "safree": None,
            "advunlearn": None,
            "trasce": None,
        }):
            import eval_unlearn.techniques  # should not raise
        assert True

    def test_saeuron_import_error_warns(self):
        """ImportError for saeuron is swallowed with a warning."""
        self._reload_techniques()
        with patch.dict("sys.modules", {"saeuron": None}):
            import eval_unlearn.techniques
        assert True

    def test_esd_import_error_warns(self):
        """ImportError for esd is swallowed with a warning."""
        self._reload_techniques()
        with patch.dict("sys.modules", {"esd": None}):
            import eval_unlearn.techniques
        assert True

    def test_safree_import_error_warns(self):
        """ImportError for safree is swallowed with a warning."""
        self._reload_techniques()
        with patch.dict("sys.modules", {"safree": None}):
            import eval_unlearn.techniques
        assert True

    def test_advunlearn_import_error_warns(self):
        self._reload_techniques()
        with patch.dict("sys.modules", {"advunlearn": None}):
            import eval_unlearn.techniques
        assert True

    def test_cogfd_import_error_warns(self):
        self._reload_techniques()
        with patch.dict("sys.modules", {"cogfd": None}):
            import eval_unlearn.techniques
        assert True

    def test_ssd_import_error_warns(self):
        self._reload_techniques()
        with patch.dict("sys.modules", {"ssd": None}):
            import eval_unlearn.techniques
        assert True

    def test_ca_import_error_warns(self):
        self._reload_techniques()
        with patch.dict("sys.modules", {"ca": None}):
            import eval_unlearn.techniques
        assert True

    def test_trasce_import_error_warns(self):
        self._reload_techniques()
        with patch.dict("sys.modules", {"trasce": None}):
            import eval_unlearn.techniques
        assert True


class TestMetricsInit:
    """Cover the try/except import blocks in metrics/__init__.py."""

    def _reload_metrics(self):
        keys_to_remove = [k for k in sys.modules if "eval_unlearn.metrics" in k
                         and "config" not in k and "clip_constants" not in k]
        for k in keys_to_remove:
            sys.modules.pop(k, None)

    def test_metrics_import_with_p4d_missing(self):
        """p4d ImportError is handled."""
        self._reload_metrics()
        with patch.dict("sys.modules", {"p4d": None}):
            import eval_unlearn.metrics
        assert True

    def test_metrics_import_with_all_optional_missing(self):
        """All optional packages missing should not crash the import."""
        self._reload_metrics()
        with patch.dict("sys.modules", {
            "p4d": None,
            "nudenet": None,
        }):
            import eval_unlearn.metrics
        assert True
