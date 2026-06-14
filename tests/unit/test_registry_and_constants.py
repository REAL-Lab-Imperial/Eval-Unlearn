"""Unit tests for registry/local.py, _clip_constants.py, and metrics/__init__.py."""
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# registry/local.py
# ---------------------------------------------------------------------------
class TestRegistryLocal:
    def _fresh_registry(self):
        """Import a fresh copy of registry.local with empty registries."""
        import importlib
        import sys
        sys.modules.pop("eval_unlearn.registry.local", None)
        import eval_unlearn.registry.local as reg
        reg._TECHNIQUES.clear()
        reg._METRICS.clear()
        reg._DATASETS.clear()
        reg._BENCHMARKS.clear()
        return reg

    def test_register_and_get_technique(self):
        reg = self._fresh_registry()

        @reg.register_technique("dummy_tech")
        class DummyTech:
            pass

        retrieved = reg.get_technique("dummy_tech")
        assert retrieved is DummyTech

    def test_get_technique_case_insensitive(self):
        reg = self._fresh_registry()

        @reg.register_technique("MyTech")
        class MyTech:
            pass

        assert reg.get_technique("mytech") is MyTech
        assert reg.get_technique("MYTECH") is MyTech

    def test_get_unknown_technique_raises(self):
        reg = self._fresh_registry()
        with pytest.raises(ValueError, match="not found"):
            reg.get_technique("nonexistent_xyz")

    def test_register_and_get_metric(self):
        reg = self._fresh_registry()

        @reg.register_metric("dummy_metric")
        def dummy_fn():
            pass

        assert reg.get_metric("dummy_metric") is dummy_fn

    def test_get_unknown_metric_raises(self):
        reg = self._fresh_registry()
        with pytest.raises(ValueError, match="not found"):
            reg.get_metric("nonexistent_metric_xyz")

    def test_register_and_get_dataset(self):
        reg = self._fresh_registry()

        @reg.register_dataset("dummy_ds")
        def dummy_ds_fn():
            pass

        assert reg.get_dataset("dummy_ds") is dummy_ds_fn

    def test_get_unknown_dataset_raises(self):
        reg = self._fresh_registry()
        with pytest.raises(ValueError, match="not found"):
            reg.get_dataset("nonexistent_ds_xyz")

    def test_register_and_get_benchmark(self):
        reg = self._fresh_registry()

        @reg.register_benchmark("dummy_bench")
        def dummy_bench_fn():
            pass

        assert reg.get_benchmark("dummy_bench") is dummy_bench_fn

    def test_get_unknown_benchmark_raises(self):
        reg = self._fresh_registry()
        with pytest.raises(ValueError, match="not found"):
            reg.get_benchmark("nonexistent_bench_xyz")

    def test_decorator_returns_original_class(self):
        reg = self._fresh_registry()

        @reg.register_technique("identity_test")
        class Original:
            pass

        assert reg._TECHNIQUES["identity_test"] is Original

    def test_overwrite_existing_registration(self):
        reg = self._fresh_registry()

        @reg.register_metric("overwrite_me")
        def v1():
            return 1

        @reg.register_metric("overwrite_me")
        def v2():
            return 2

        assert reg.get_metric("overwrite_me")() == 2


# ---------------------------------------------------------------------------
# metrics/_clip_constants.py
# ---------------------------------------------------------------------------
class TestClipConstants:
    def test_validate_clip_model_valid(self):
        from eval_unlearn.metrics._clip_constants import validate_clip_model
        validate_clip_model("openai/clip-vit-large-patch14")
        validate_clip_model("openai/clip-vit-base-patch16")
        validate_clip_model("openai/clip-vit-large-patch14-336")

    def test_validate_clip_model_invalid(self):
        from eval_unlearn.metrics._clip_constants import validate_clip_model
        with pytest.raises(ValueError, match="Unsupported CLIP model"):
            validate_clip_model("some/unknown-model", "test_field")

    def test_validate_sd_text_encoder_valid(self):
        from eval_unlearn.metrics._clip_constants import validate_sd_text_encoder
        validate_sd_text_encoder("openai/clip-vit-large-patch14")

    def test_validate_sd_text_encoder_invalid(self):
        from eval_unlearn.metrics._clip_constants import validate_sd_text_encoder
        with pytest.raises(ValueError, match="Unsupported SD text encoder"):
            validate_sd_text_encoder("openai/clip-vit-base-patch16")

    def test_clip_encoder_for_sd_known_model(self):
        from eval_unlearn.metrics._clip_constants import clip_encoder_for_sd
        enc = clip_encoder_for_sd("CompVis/stable-diffusion-v1-4")
        assert enc == "openai/clip-vit-large-patch14"

    def test_clip_encoder_for_sd_runwayml(self):
        from eval_unlearn.metrics._clip_constants import clip_encoder_for_sd
        enc = clip_encoder_for_sd("runwayml/stable-diffusion-v1-5")
        assert enc == "openai/clip-vit-large-patch14"

    def test_clip_encoder_for_sd_safe(self):
        from eval_unlearn.metrics._clip_constants import clip_encoder_for_sd
        enc = clip_encoder_for_sd("AIML-TUDA/stable-diffusion-safe")
        assert enc == "openai/clip-vit-large-patch14"

    def test_clip_encoder_for_sd_unknown_raises(self):
        from eval_unlearn.metrics._clip_constants import clip_encoder_for_sd
        with pytest.raises(ValueError, match="Unknown SD model"):
            clip_encoder_for_sd("some/unknown-diffusion-model")


# ---------------------------------------------------------------------------
# metrics/__init__.py — import with missing optional packages
# ---------------------------------------------------------------------------
class TestMetricsInit:
    def test_import_succeeds_even_with_missing_deps(self):
        """metrics/__init__.py should not raise even if GPU packages are absent."""
        import sys
        for key in list(sys.modules.keys()):
            if "eval_unlearn.metrics" in key and key != "eval_unlearn.metrics":
                del sys.modules[key]
        sys.modules.pop("eval_unlearn.metrics", None)

        broken = MagicMock(side_effect=ImportError("not installed"))
        with patch.dict("sys.modules", {"nudenet": broken, "q16": broken}):
            import eval_unlearn.metrics  # noqa: F401
        assert True

    def test_all_exports_in_globals(self):
        import eval_unlearn.metrics as m
        # At least the classes that don't need GPU should be importable
        assert hasattr(m, "__all__")
        for name in m.__all__:
            assert name in dir(m)


# ---------------------------------------------------------------------------
# techniques/__init__.py — __all__ list
# ---------------------------------------------------------------------------
class TestTechniquesInitAll:
    def test_all_contains_only_available_techniques(self):
        import eval_unlearn.techniques as t
        # __all__ must only contain names that are actually in globals
        for name in t.__all__:
            assert name in dir(t), f"{name} in __all__ but not in module globals"
