"""Registry of each technique's fixed base diffusion model.

Used by runners to resolve the correct CLIP text encoder for mma_diffusion
without needing to instantiate the technique first.
"""
from typing import Dict, Optional

# Maps technique name → its fixed base diffusion model HF ID.
# Techniques whose model_id is field(init=False) are listed here.
# free_run is excluded because its model_id comes from the user config.
TECHNIQUE_BASE_MODELS: Dict[str, str] = {
    "advunlearn": "CompVis/stable-diffusion-v1-4",
    "concept_steerers": "CompVis/stable-diffusion-v1-4",
    "esd": "CompVis/stable-diffusion-v1-4",
    "mace": "CompVis/stable-diffusion-v1-4",
    "safree": "CompVis/stable-diffusion-v1-4",
    "saeuron": "CompVis/stable-diffusion-v1-4",
    "uce": "CompVis/stable-diffusion-v1-4",
    "sld": "AIML-TUDA/stable-diffusion-safe",
    "ssd": "CompVis/stable-diffusion-v1-4",
    "trasce": "CompVis/stable-diffusion-v1-4",
}


def get_technique_base_model_id(
    technique_name: str, technique_config: Dict
) -> Optional[str]:
    """Return the base diffusion model ID for a technique.

    For fixed-model techniques this comes from TECHNIQUE_BASE_MODELS.
    For free_run it is read from technique_config['model_id'].
    Returns None if it cannot be determined.
    """
    if technique_name == "free_run":
        return technique_config.get("model_id") or None
    return TECHNIQUE_BASE_MODELS.get(technique_name)
