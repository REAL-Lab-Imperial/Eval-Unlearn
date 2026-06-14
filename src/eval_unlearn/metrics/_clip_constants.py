"""Shared CLIP model constants for metric configs."""

# Validated OpenAI CLIP model IDs supported across all metrics.
# These are the only architectures confirmed to work with CLIPModel/CLIPProcessor
# from transformers in the context of concept detection and scoring.
SUPPORTED_CLIP_MODELS = frozenset({
    "openai/clip-vit-base-patch16",
     "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336",
})

# SD text encoders valid for asr_mma_diffusion's GCG attack.
# Must match the exact CLIP text encoder baked into the target diffusion model.
# SD 2.x uses OpenCLIP (different API — not supported by asr_mma_diffusion).
SD_TEXT_ENCODERS = frozenset({
    "openai/clip-vit-large-patch14",   # SD 1.x
})

# Maps a diffusion model's HuggingFace ID to its CLIP text encoder.
# Used by runners to inject the correct encoder into asr_mma_diffusion config.
SD_MODEL_TO_CLIP_ENCODER: dict = {
    "CompVis/stable-diffusion-v1-4": "openai/clip-vit-large-patch14",
    "runwayml/stable-diffusion-v1-5": "openai/clip-vit-large-patch14",
    "AIML-TUDA/stable-diffusion-safe": "openai/clip-vit-large-patch14",
}


def validate_clip_model(model_id: str, field_name: str = "clip_model_name") -> None:
    """Raise ValueError if model_id is not in the supported CLIP allowlist."""
    if model_id not in SUPPORTED_CLIP_MODELS:
        raise ValueError(
            f"Unsupported CLIP model '{model_id}' for '{field_name}'. "
            f"Supported models: {sorted(SUPPORTED_CLIP_MODELS)}"
        )


def validate_sd_text_encoder(model_id: str, field_name: str = "clip_model_id") -> None:
    """Raise ValueError if model_id is not a known SD text encoder for GCG attacks."""
    if model_id not in SD_TEXT_ENCODERS:
        raise ValueError(
            f"Unsupported SD text encoder '{model_id}' for '{field_name}'. "
            f"asr_mma_diffusion requires the exact CLIP text encoder of the target SD model. "
            f"Supported: {sorted(SD_TEXT_ENCODERS)}"
        )


def clip_encoder_for_sd(sd_model_id: str) -> str:
    """Return the CLIP text encoder for a given SD model ID.

    Raises ValueError if the SD model is not in the known mapping.
    """
    encoder = SD_MODEL_TO_CLIP_ENCODER.get(sd_model_id)
    if encoder is None:
        raise ValueError(
            f"Unknown SD model '{sd_model_id}' — cannot determine CLIP text encoder. "
            f"Add it to SD_MODEL_TO_CLIP_ENCODER in _clip_constants.py."
        )
    return encoder