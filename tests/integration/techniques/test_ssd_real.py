"""
Integration tests — real SSD technique pipeline.

Uses runwayml/stable-diffusion-v1-5 (cached) with 2 inference steps
to verify the wrapper's __init__ and generate() paths without
downloading SD v1-4.

The SD pipeline is loaded once per module via a session-scoped fixture.
"""
import pytest
from PIL import Image
from unittest.mock import patch

pytestmark = pytest.mark.integration

_CACHED_MODEL = "runwayml/stable-diffusion-v1-5"


@pytest.fixture(scope="module")
def ssd_technique():
    """
    Real SSD pipeline loaded once for this module.

    Patches SSDPipeline at the point of instantiation inside the wrapper
    so model_id is redirected to the cached v1-5 without touching frozen config.
    """
    from ssd import SSDPipeline as _RealSSDPipeline

    original_init = _RealSSDPipeline.__init__

    def _patched_init(self, model_id, **kwargs):
        original_init(self, _CACHED_MODEL, **kwargs)

    with patch.object(_RealSSDPipeline, "__init__", _patched_init):
        from eval_learn.techniques.ssd.wrapper import SSDTechnique
        tech = SSDTechnique(
            erase_concept="nudity",
            device="cuda",
            forget_prompts=["nudity"],
            retain_prompts=["a dog", "a car"],
            num_fisher_samples=2,
            num_inference_steps=2,
            guidance_scale=7.5,
        )
    yield tech


class TestSSDTechniqueReal:
    def test_generate_returns_pil_images(self, ssd_technique):
        imgs = ssd_technique.generate(["a red car on a road"], seed=0)
        assert isinstance(imgs, list)
        assert len(imgs) == 1
        assert isinstance(imgs[0], Image.Image)

    def test_generate_batch_of_prompts(self, ssd_technique):
        prompts = ["a cat sitting", "a mountain lake"]
        imgs = ssd_technique.generate(prompts, seed=1)
        assert len(imgs) == len(prompts)
        for img in imgs:
            assert isinstance(img, Image.Image)

    def test_generate_with_none_seed(self, ssd_technique):
        imgs = ssd_technique.generate(["a sunset"], seed=None)
        assert len(imgs) == 1

    def test_generate_custom_inference_steps(self, ssd_technique):
        imgs = ssd_technique.generate(
            ["a blue sky"],
            seed=42,
            num_inference_steps=1,
        )
        assert isinstance(imgs[0], Image.Image)

    def test_image_has_valid_dimensions(self, ssd_technique):
        imgs = ssd_technique.generate(["abstract art"], seed=7)
        img = imgs[0]
        assert img.width > 0
        assert img.height > 0
        assert img.mode == "RGB"
