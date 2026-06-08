"""
Shared fixtures for integration tests.

Integration tests load real models and require GPU. Run with:
    pytest tests/integration -m integration -v

They are excluded from the standard unit test run:
    pytest tests/unit -m "not integration" -v
"""
import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require GPU and real model weights",
    )


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def nude_detector():
    """Real NudeNet detector — loaded once for the entire session."""
    from nudenet import NudeDetector
    return NudeDetector()


@pytest.fixture(scope="session")
def q16_classifier(device):
    """Real Q16 classifier (ViT-L/14) — loaded once for the entire session."""
    from q16 import Q16Classifier
    return Q16Classifier(model="ViT-L/14", device=device, threshold=0.9)


@pytest.fixture(scope="session")
def clip_model_and_processor(device):
    """Real CLIP (clip-vit-base-patch16, cached) — loaded once for the session."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor


@pytest.fixture(scope="session")
def inception_model(device):
    """Real InceptionV3 with fc replaced by Identity — loaded once for the session."""
    import torch.nn as nn
    from torchvision import models
    from torchvision.models import Inception_V3_Weights
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)
