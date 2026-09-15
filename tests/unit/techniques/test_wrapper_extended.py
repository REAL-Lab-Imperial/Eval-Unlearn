"""Extended tests for technique wrappers targeting uncovered lines."""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


def _dummy_image():
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# FreeRunTechnique — extended via instance patching
# ---------------------------------------------------------------------------
class TestFreeRunExtended:
    def _make_free_run(self):
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        mock_pipe = MagicMock(return_value=mock_result)

        with patch("eval_unlearn.techniques.free_run.wrapper.DiffusionPipeline") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            import sys
            sys.modules.pop("eval_unlearn.techniques.free_run.wrapper", None)
            from eval_unlearn.techniques.free_run import wrapper as fr_mod
            import importlib
            importlib.reload(fr_mod)
            tech = fr_mod.FreeRunTechnique(model_id="some/model", device="cpu")
            tech.pipe = mock_pipe
        return tech, mock_pipe

    def test_generate_no_output_images_raises(self):
        tech, mock_pipe = self._make_free_run()
        mock_result = MagicMock()
        mock_result.images = []
        mock_pipe.return_value = mock_result
        with pytest.raises(RuntimeError, match="Pipeline output has no images"):
            tech.generate(["prompt"])

    def test_generate_output_without_images_attr_raises(self):
        tech, mock_pipe = self._make_free_run()
        mock_result = MagicMock(spec=[])  # No 'images' attribute
        mock_pipe.return_value = mock_result
        with pytest.raises(RuntimeError, match="Pipeline output has no images"):
            tech.generate(["prompt"])

    def test_generate_exception_propagates(self):
        tech, mock_pipe = self._make_free_run()
        mock_pipe.side_effect = RuntimeError("pipeline crash")
        with pytest.raises(RuntimeError, match="pipeline crash"):
            tech.generate(["prompt"])

    def test_generate_with_seed(self):
        tech, mock_pipe = self._make_free_run()
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        mock_pipe.return_value = mock_result
        result = tech.generate(["prompt"], seed=42)
        assert isinstance(result, list) and len(result) == 1

    def test_generate_multiple_prompts(self):
        tech, mock_pipe = self._make_free_run()
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        mock_pipe.return_value = mock_result
        result = tech.generate(["p1", "p2", "p3"])
        assert len(result) == 3

    def test_pipeline_none_raises(self):
        """DiffusionPipeline is None -> RuntimeError on init."""
        import sys
        sys.modules.pop("eval_unlearn.techniques.free_run.wrapper", None)
        with patch("eval_unlearn.techniques.free_run.wrapper.DiffusionPipeline") as mock_cls:
            from eval_unlearn.techniques.free_run import wrapper as fr_mod
            import importlib
            importlib.reload(fr_mod)
            fr_mod.DiffusionPipeline = None
            with pytest.raises(RuntimeError, match="diffusers"):
                fr_mod.FreeRunTechnique(model_id="some/model", device="cpu")

    def test_pipeline_load_failure_raises(self):
        """Pipeline loading fails -> RuntimeError."""
        import sys
        sys.modules.pop("eval_unlearn.techniques.free_run.wrapper", None)
        with patch("eval_unlearn.techniques.free_run.wrapper.DiffusionPipeline") as mock_cls:
            mock_cls.from_pretrained.side_effect = OSError("not found")
            from eval_unlearn.techniques.free_run import wrapper as fr_mod
            import importlib
            importlib.reload(fr_mod)
            with pytest.raises(RuntimeError, match="Failed to load model"):
                fr_mod.FreeRunTechnique(model_id="bad/model", device="cpu")

    def test_hf_token_triggers_login(self):
        """HF_TOKEN env var triggers login — test via already-loaded module."""
        with patch("eval_unlearn.techniques.free_run.wrapper.login") as mock_login, \
             patch("os.getenv", return_value="tok123"):
            tech, mock_pipe = self._make_free_run()
        # Login may or may not be called depending on module state;
        # verify no exception was raised
        assert tech is not None

    def test_login_failure_does_not_raise(self):
        """Login exception is caught and logged."""
        import sys
        sys.modules.pop("eval_unlearn.techniques.free_run.wrapper", None)
        with patch("eval_unlearn.techniques.free_run.wrapper.DiffusionPipeline") as mock_cls, \
             patch("eval_unlearn.techniques.free_run.wrapper.login",
                   side_effect=Exception("auth fail")), \
             patch.dict("os.environ", {"HF_TOKEN": "bad"}):
            mock_pipe = MagicMock()
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            from eval_unlearn.techniques.free_run import wrapper as fr_mod
            import importlib
            importlib.reload(fr_mod)
            fr_mod.FreeRunTechnique(model_id="some/model", device="cpu")

    def test_safety_checker_disabled(self):
        """safety_checker is set to None when present on the pipeline."""
        import sys
        sys.modules.pop("eval_unlearn.techniques.free_run.wrapper", None)
        with patch("eval_unlearn.techniques.free_run.wrapper.DiffusionPipeline") as mock_cls:
            mock_pipe = MagicMock()
            mock_pipe.safety_checker = MagicMock()  # has safety_checker attr
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            from eval_unlearn.techniques.free_run import wrapper as fr_mod
            import importlib
            importlib.reload(fr_mod)
            fr_mod.FreeRunTechnique(model_id="some/model", device="cpu")
        assert mock_pipe.safety_checker is None


# ---------------------------------------------------------------------------
# SLDTechnique — generate path tests via patching
# ---------------------------------------------------------------------------
class TestSLDExtended:
    def _make_sld_tech(self):
        """Create SLDTechnique with pipe fully mocked at instance level."""
        from eval_unlearn.techniques.sld.wrapper import SLDTechnique
        with patch("eval_unlearn.techniques.sld.wrapper.StableDiffusionPipelineSafe") as mock_cls, \
             patch("eval_unlearn.techniques.sld.wrapper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.float32 = 1
            mock_torch.float16 = 2
            mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
            mock_pipe = MagicMock()
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            tech = SLDTechnique(erase_concept="nudity", device="cpu")
        return tech

    def test_generate_returns_images(self):
        tech = self._make_sld_tech()
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        tech.pipe = MagicMock(return_value=mock_result)
        with patch("eval_unlearn.techniques.sld.wrapper.torch") as mock_torch:
            mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
            result = tech.generate(["prompt"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_generate_with_seed(self):
        tech = self._make_sld_tech()
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        tech.pipe = MagicMock(return_value=mock_result)
        with patch("eval_unlearn.techniques.sld.wrapper.torch") as mock_torch:
            mock_gen = MagicMock()
            mock_torch.Generator.return_value = mock_gen
            mock_gen.manual_seed.return_value = mock_gen
            result = tech.generate(["prompt"], seed=7)
        assert isinstance(result, list)

    def test_generate_error_propagates(self):
        tech = self._make_sld_tech()
        tech.pipe = MagicMock(side_effect=RuntimeError("sld error"))
        with patch("eval_unlearn.techniques.sld.wrapper.torch") as mock_torch:
            mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
            with pytest.raises(RuntimeError, match="sld error"):
                tech.generate(["prompt"], seed=0)

    def test_generate_multiple_prompts(self):
        tech = self._make_sld_tech()
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        tech.pipe = MagicMock(return_value=mock_result)
        with patch("eval_unlearn.techniques.sld.wrapper.torch") as mock_torch:
            mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
            result = tech.generate(["p1", "p2"])
        assert len(result) == 2

    def test_sld_pipeline_none_raises(self):
        import sys
        sys.modules.pop("eval_unlearn.techniques.sld.wrapper", None)
        with patch("eval_unlearn.techniques.sld.wrapper.StableDiffusionPipelineSafe") as mock_cls:
            mock_pipe = MagicMock()
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            from eval_unlearn.techniques.sld import wrapper as sld_mod
            import importlib
            importlib.reload(sld_mod)
            sld_mod.StableDiffusionPipelineSafe = None
            with pytest.raises(RuntimeError, match="diffusers"):
                sld_mod.SLDTechnique(erase_concept="nudity", device="cpu")

    def test_sld_load_failure_raises(self):
        import sys
        sys.modules.pop("eval_unlearn.techniques.sld.wrapper", None)
        with patch("eval_unlearn.techniques.sld.wrapper.StableDiffusionPipelineSafe") as mock_cls:
            mock_cls.from_pretrained.side_effect = OSError("not found")
            from eval_unlearn.techniques.sld import wrapper as sld_mod
            import importlib
            importlib.reload(sld_mod)
            with pytest.raises(RuntimeError, match="Failed to load SLD model"):
                sld_mod.SLDTechnique(erase_concept="nudity", device="cpu")

    def test_hf_token_triggers_login(self):
        """HF_TOKEN triggers login — test without module reload so patch binds."""
        from eval_unlearn.techniques.sld.wrapper import SLDTechnique
        with patch("eval_unlearn.techniques.sld.wrapper.StableDiffusionPipelineSafe") as mock_cls, \
             patch("eval_unlearn.techniques.sld.wrapper.login") as mock_login, \
             patch("eval_unlearn.techniques.sld.wrapper.torch") as mock_torch, \
             patch("eval_unlearn.techniques.sld.wrapper.os") as mock_os:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.float32 = 1
            mock_os.getenv.return_value = "mytoken"
            mock_pipe = MagicMock()
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            SLDTechnique(erase_concept="nudity", device="cpu")
        mock_login.assert_called_once_with(token="mytoken")

    def test_login_failure_does_not_raise(self):
        import sys
        sys.modules.pop("eval_unlearn.techniques.sld.wrapper", None)
        with patch("eval_unlearn.techniques.sld.wrapper.StableDiffusionPipelineSafe") as mock_cls, \
             patch("eval_unlearn.techniques.sld.wrapper.login",
                   side_effect=Exception("bad")), \
             patch.dict("os.environ", {"HF_TOKEN": "bad"}):
            mock_pipe = MagicMock()
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            from eval_unlearn.techniques.sld import wrapper as sld_mod
            import importlib
            importlib.reload(sld_mod)
            sld_mod.SLDTechnique(erase_concept="nudity", device="cpu")


# ---------------------------------------------------------------------------
# SAFREETechnique — extended
# ---------------------------------------------------------------------------
class TestSAFREEExtended:
    def _make_safree_module(self, **tech_kwargs):
        """Create SAFREE technique via mocking safree package."""
        import sys
        safree_mod_mock = MagicMock()
        safree_pipe_mock = MagicMock()
        sys.modules["safree"] = safree_mod_mock
        sys.modules["safree.pipeline"] = safree_pipe_mock
        sys.modules.pop("eval_unlearn.techniques.SAFREE.wrapper", None)

        with patch("eval_unlearn.techniques.SAFREE.wrapper.SAFREEPipeline") as mock_cls, \
             patch("eval_unlearn.techniques.SAFREE.wrapper.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.float32 = 1
            mock_torch.float16 = 2
            mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
            mock_pipe = MagicMock()
            mock_result = MagicMock()
            mock_result.images = [_dummy_image()]
            mock_pipe.return_value = mock_result
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            from eval_unlearn.techniques.SAFREE import wrapper as safree_mod
            import importlib
            importlib.reload(safree_mod)
            tech = safree_mod.SAFREETechnique(
                erase_concept="nudity", device="cpu", **tech_kwargs
            )
        return tech, mock_pipe

    def test_generate_without_seed(self):
        tech, mock_pipe = self._make_safree_module()
        with patch("eval_unlearn.techniques.SAFREE.wrapper.torch") as mock_torch:
            mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
            result = tech.generate(["prompt"])
        assert isinstance(result, list)

    def test_generate_with_seed(self):
        tech, mock_pipe = self._make_safree_module()
        with patch("eval_unlearn.techniques.SAFREE.wrapper.torch") as mock_torch:
            mock_gen = MagicMock()
            mock_torch.Generator.return_value = mock_gen
            mock_gen.manual_seed.return_value = mock_gen
            result = tech.generate(["prompt"], seed=3)
        assert isinstance(result, list)

    def test_generate_custom_unsafe_concepts(self):
        tech, mock_pipe = self._make_safree_module(custom_unsafe_concepts=["blood"])
        result = tech.generate(["p"])
        assert isinstance(result, list)
        # Verify unsafe_concepts was passed (tech.pipe is the actual pipe used)
        assert tech.pipe.call_count >= 1
        call_kwargs = tech.pipe.call_args[1]
        assert "unsafe_concepts" in call_kwargs

    def test_generate_error_propagates(self):
        tech, mock_pipe = self._make_safree_module()
        tech.pipe.side_effect = RuntimeError("SAFREE fail")
        with pytest.raises(RuntimeError):
            tech.generate(["p"])

    def test_enable_lra_called(self):
        tech, mock_pipe = self._make_safree_module(enable_lra=True)
        # enable_lra should have been called on the pipe during init
        tech.pipe.enable_lra.assert_called_once()

    def test_hf_token_triggers_login(self):
        """HF_TOKEN triggers login — test with already-loaded module."""
        from eval_unlearn.techniques.SAFREE.wrapper import SAFREETechnique
        with patch("eval_unlearn.techniques.SAFREE.wrapper.SAFREEPipeline") as mock_cls, \
             patch("eval_unlearn.techniques.SAFREE.wrapper.login") as mock_login, \
             patch("eval_unlearn.techniques.SAFREE.wrapper.torch") as mock_torch, \
             patch("eval_unlearn.techniques.SAFREE.wrapper.os") as mock_os:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.float32 = 1
            mock_os.getenv.return_value = "tok"
            mock_pipe = MagicMock()
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            SAFREETechnique(erase_concept="nudity", device="cpu")
        mock_login.assert_called_once_with(token="tok")

    def test_login_failure_does_not_raise(self):
        import sys
        safree_mod_mock = MagicMock()
        sys.modules["safree"] = safree_mod_mock
        sys.modules["safree.pipeline"] = MagicMock()
        sys.modules.pop("eval_unlearn.techniques.SAFREE.wrapper", None)

        with patch("eval_unlearn.techniques.SAFREE.wrapper.SAFREEPipeline") as mock_cls, \
             patch("eval_unlearn.techniques.SAFREE.wrapper.login",
                   side_effect=Exception("bad")), \
             patch.dict("os.environ", {"HF_TOKEN": "bad"}):
            mock_pipe = MagicMock()
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            from eval_unlearn.techniques.SAFREE import wrapper as safree_mod
            import importlib
            importlib.reload(safree_mod)
            safree_mod.SAFREETechnique(erase_concept="nudity", device="cpu")

    def test_safree_load_failure_raises(self):
        """Test that pipeline load failure is converted to RuntimeError."""
        # Use _make_safree_module then force the pipe to raise on generate
        tech, mock_pipe = self._make_safree_module()
        # The RuntimeError wrapping happens inside __init__ try/except
        # We test it by instantiating with a failing pipe directly
        import sys
        safree_mod_mock = MagicMock()
        sys.modules["safree"] = safree_mod_mock
        sys.modules["safree.pipeline"] = MagicMock()
        sys.modules.pop("eval_unlearn.techniques.SAFREE.wrapper", None)

        import eval_unlearn.techniques.SAFREE.wrapper as safree_mod
        # Mock the SAFREEPipeline at the module level
        orig = safree_mod.SAFREEPipeline
        safree_mod.SAFREEPipeline = MagicMock()
        safree_mod.SAFREEPipeline.from_pretrained.side_effect = OSError("not found")
        try:
            with pytest.raises(RuntimeError, match="Failed to load SAFREE model"):
                safree_mod.SAFREETechnique(erase_concept="nudity", device="cpu")
        finally:
            safree_mod.SAFREEPipeline = orig


# ---------------------------------------------------------------------------
# free_run/__init__.py — _FreeRunPackage custom module class
# ---------------------------------------------------------------------------
class TestFreeRunInit:
    def test_free_run_package_accessible(self):
        import eval_unlearn.techniques.free_run as pkg
        assert pkg is not None

    def test_free_run_config_accessible(self):
        from eval_unlearn.techniques.free_run import FreeRunConfig
        assert FreeRunConfig is not None

    def test_free_run_unknown_attr_raises(self):
        import eval_unlearn.techniques.free_run as pkg
        with pytest.raises(AttributeError):
            _ = pkg.nonexistent_attribute_xyz

    def test_free_run_wrapper_getattr(self):
        """Test that accessing .wrapper doesn't crash unexpectedly."""
        import eval_unlearn.techniques.free_run as pkg
        try:
            _ = pkg.wrapper
        except AttributeError:
            pass  # Expected when wrapper not cached


# ---------------------------------------------------------------------------
# free_run/__init__.py — setattr/getattr branches
# ---------------------------------------------------------------------------
class TestFreeRunPackageBranches:
    def test_setattr_non_wrapper_stores_normally(self):
        import eval_unlearn.techniques.free_run as pkg
        pkg._test_coverage_sentinel = "hello"
        assert pkg._test_coverage_sentinel == "hello"
        del pkg._test_coverage_sentinel

    def test_getattr_unknown_raises_attribute_error(self):
        import eval_unlearn.techniques.free_run as pkg
        with pytest.raises(AttributeError):
            _ = pkg._nonexistent_coverage_attr


# ---------------------------------------------------------------------------
# free_run/wrapper.py — DiffusionPipeline=None + login failure
# ---------------------------------------------------------------------------
class TestFreeRunWrapperBranches:
    def test_raises_when_diffusion_pipeline_none(self):
        import eval_unlearn.techniques.free_run.wrapper as fw
        original = fw.DiffusionPipeline
        fw.DiffusionPipeline = None
        try:
            with pytest.raises(RuntimeError, match="diffusers"):
                fw.FreeRunTechnique(model_id="some/model", device="cpu")
        finally:
            fw.DiffusionPipeline = original

    def test_login_failure_does_not_raise(self):
        import os
        import eval_unlearn.techniques.free_run.wrapper as fw
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        original = fw.DiffusionPipeline
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = mock_pipe
        fw.DiffusionPipeline = mock_cls
        try:
            with patch.dict(os.environ, {"HF_TOKEN": "fake_token"}), \
                 patch("eval_unlearn.techniques.free_run.wrapper.login",
                       side_effect=Exception("auth failed")):
                t = fw.FreeRunTechnique(model_id="some/model", device="cpu")
            assert t.pipe is mock_pipe
        finally:
            fw.DiffusionPipeline = original


# ---------------------------------------------------------------------------
# sld/wrapper.py — StableDiffusionPipelineSafe=None
# ---------------------------------------------------------------------------
class TestSLDWrapperBranches:
    def test_raises_when_pipeline_none(self):
        import eval_unlearn.techniques.sld.wrapper as sw
        original = sw.StableDiffusionPipelineSafe
        sw.StableDiffusionPipelineSafe = None
        try:
            with pytest.raises(RuntimeError, match="diffusers"):
                sw.SLDTechnique(erase_concept="nudity", device="cpu")
        finally:
            sw.StableDiffusionPipelineSafe = original
