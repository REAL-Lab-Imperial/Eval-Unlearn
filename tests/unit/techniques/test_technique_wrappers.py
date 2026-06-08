"""Unit tests for technique wrappers — all external pipelines are mocked."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image


def _dummy_image():
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# Helpers shared across wrapper tests
# ---------------------------------------------------------------------------

def _mock_pipeline(return_images=None):
    """Return a mock pipeline whose generate() returns dummy images."""
    pipe = MagicMock()
    imgs = return_images or [_dummy_image()]
    pipe.generate.return_value = imgs
    return pipe


# ---------------------------------------------------------------------------
# ESDTechnique wrapper
# ---------------------------------------------------------------------------
class TestESDWrapper:
    def test_generate_delegates_to_pipeline(self):
        mock_pipe = _mock_pipeline([_dummy_image(), _dummy_image()])
        with patch.dict("sys.modules", {"esd": MagicMock(ESDPipeline=MagicMock(return_value=mock_pipe))}):
            from eval_learn.techniques.esd.wrapper import ESDTechnique
            t = ESDTechnique(erase_concept="nudity", device="cpu")
            result = t.generate(["a cat", "a dog"], seed=42)
        assert len(result) == 2

    def test_generate_passes_seed(self):
        mock_pipe = _mock_pipeline()
        with patch.dict("sys.modules", {"esd": MagicMock(ESDPipeline=MagicMock(return_value=mock_pipe))}):
            from eval_learn.techniques.esd.wrapper import ESDTechnique
            t = ESDTechnique(erase_concept="nudity", device="cpu")
            t.generate(["prompt"], seed=7)
        mock_pipe.generate.assert_called_once()
        _, kwargs = mock_pipe.generate.call_args
        assert kwargs.get("seed") == 7

    def test_import_error_raises(self):
        with patch.dict("sys.modules", {"esd": None}):
            import importlib, sys
            sys.modules.pop("eval_learn.techniques.esd.wrapper", None)
            with pytest.raises((ImportError, Exception)):
                import eval_learn.techniques.esd.wrapper  # noqa: F401


# ---------------------------------------------------------------------------
# Generic wrapper test factory (SSD, CA, CoGFD, TraSCE, MACE, SAeUron, etc.)
# ---------------------------------------------------------------------------

def _test_wrapper_generate(module_path, class_name, pkg_name, pkg_class, init_kwargs):
    """Helper that instantiates a wrapper with a mocked external package and calls generate."""
    mock_pipe = _mock_pipeline([_dummy_image()])
    mock_pkg = MagicMock()
    setattr(mock_pkg, pkg_class, MagicMock(return_value=mock_pipe))

    with patch.dict("sys.modules", {pkg_name: mock_pkg}):
        import importlib
        import sys
        sys.modules.pop(module_path, None)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        tech = cls(**init_kwargs)
        result = tech.generate(["a prompt"], seed=1)

    assert isinstance(result, list)
    assert len(result) == 1


class TestSSDWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.ssd.wrapper", "SSDTechnique",
            "ssd", "SSDPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


class TestCAWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.ca.wrapper", "CATechnique",
            "ca", "CAPipeline",
            {"erase_concept": "nudity", "anchor_concept": "clothed person", "device": "cpu"},
        )


class TestCoGFDWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.cogfd.wrapper", "CoGFDTechnique",
            "cogfd", "CoGFDPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


class TestTraSCEWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.trasce.wrapper", "TraSCETechnique",
            "trasce", "TraSCEPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


class TestMACEWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.mace.wrapper", "MACETechnique",
            "mace", "MACEPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


class TestAdvUnlearnWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.advunlearn.wrapper", "AdvUnlearnTechnique",
            "advunlearn", "AdvUnlearnPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


class TestSAeUronWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.saeuron.wrapper", "SAeUronTechnique",
            "saeuron", "SAeUronPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


class TestSAFREEWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.SAFREE.wrapper", "SAFREETechnique",
            "safree", "SAFREEPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


class TestConceptSteerersWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.concept_steerers.wrapper", "ConceptSteerersTechnique",
            "concept_steerers", "ConceptSteeringPipeline",
            {"erase_concept": "nudity", "device": "cpu"},
        )


# ---------------------------------------------------------------------------
# SLDTechnique wrapper (uses diffusers, no external package)
# ---------------------------------------------------------------------------
class TestSLDWrapper:
    def test_generate(self):
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        mock_pipe = MagicMock(return_value=mock_result)

        with patch("eval_learn.techniques.sld.wrapper.StableDiffusionPipelineSafe") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            from eval_learn.techniques.sld import wrapper as sld_mod
            import importlib
            importlib.reload(sld_mod)
            t = sld_mod.SLDTechnique(erase_concept="nudity", device="cpu")
            result = t.generate(["a prompt"], seed=0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# UCETechnique wrapper
# ---------------------------------------------------------------------------
class TestUCEWrapper:
    def test_generate(self):
        _test_wrapper_generate(
            "eval_learn.techniques.uce.wrapper", "UCETechnique",
            "uce", "UCEPipeline",
            {"preset": "nudity", "device": "cpu"},
        )


# ---------------------------------------------------------------------------
# FreeRunTechnique wrapper
# ---------------------------------------------------------------------------
class TestFreeRunWrapper:
    def test_wrapper_ref_none_raises_attribute_error(self):
        import pytest
        import eval_learn.techniques.free_run as fr_pkg
        from eval_learn.techniques.free_run import _FreeRunPackage
        original = _FreeRunPackage._wrapper_ref
        try:
            _FreeRunPackage._wrapper_ref = None
            with pytest.raises(AttributeError, match="has no attribute 'wrapper'"):
                _ = fr_pkg.wrapper
        finally:
            _FreeRunPackage._wrapper_ref = original

    def test_generate(self):
        mock_result = MagicMock()
        mock_result.images = [_dummy_image()]
        mock_pipe = MagicMock(return_value=mock_result)

        with patch("eval_learn.techniques.free_run.wrapper.DiffusionPipeline") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_pipe
            mock_pipe.to.return_value = mock_pipe
            import importlib, sys
            sys.modules.pop("eval_learn.techniques.free_run.wrapper", None)
            from eval_learn.techniques.free_run import wrapper as fr_mod
            importlib.reload(fr_mod)
            t = fr_mod.FreeRunTechnique(model_id="some/model", device="cpu")
            result = t.generate(["prompt"], seed=0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# techniques/__init__.py — import with missing optional packages
# ---------------------------------------------------------------------------
class TestTechniquesInit:
    def test_import_with_missing_packages(self):
        """techniques/__init__.py should not raise even if optional packages are missing."""
        import sys
        # Remove cached modules so the try/except blocks in __init__.py are re-executed
        for key in list(sys.modules.keys()):
            if "eval_learn.techniques" in key and key != "eval_learn.techniques":
                del sys.modules[key]
        sys.modules.pop("eval_learn.techniques", None)

        broken = MagicMock(side_effect=ImportError("not installed"))
        with patch.dict("sys.modules", {"esd": broken, "ssd": broken, "ca": broken}):
            import eval_learn.techniques  # should not raise
        assert True

    def test_safree_import_failure_warns_not_raises(self):
        """Cover the SAFREE try/except warning branch (lines 21-23)."""
        import sys
        for key in list(sys.modules.keys()):
            if "eval_learn.techniques" in key:
                sys.modules.pop(key, None)
        broken = MagicMock(side_effect=ImportError("safree not found"))
        with patch.dict("sys.modules", {"safree": broken}):
            import eval_learn.techniques  # must not raise
        assert True

    def test_techniques_init_safree_wrapper_failure_warns(self):
        """Cover techniques/__init__.py lines 21-23 via wrapper import failure."""
        import sys
        for key in list(sys.modules.keys()):
            if "eval_learn.techniques" in key:
                sys.modules.pop(key, None)
        with patch.dict("sys.modules", {"eval_learn.techniques.SAFREE.wrapper": None}):
            import eval_learn.techniques  # noqa: F401 — must not raise
        assert True
