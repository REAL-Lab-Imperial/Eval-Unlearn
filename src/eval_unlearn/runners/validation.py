"""Validate technique-metric pair compatibility and constraints."""

from typing import Any, Dict, Optional


class ValidationError(ValueError):
    """Raised when technique-metric combination is invalid."""

    pass


def get_erase_concept(
    technique_name: str, technique_config: Dict[str, Any]
) -> Optional[str]:
    """Extract erase_concept from technique config."""
    # UCE special case: preset maps to erase_concept
    if technique_name == "uce":
        preset = technique_config.get("preset")
        return preset.lower() if preset else None

    # All other techniques use erase_concept directly
    erase_concept = technique_config.get("erase_concept")
    return erase_concept.lower() if erase_concept else None



def validate_nudity_metrics(
    technique_name: str, erase_concept: Optional[str], metric_name: str
) -> None:
    """Validate ERR requires nudity concept. ASR is multi-concept and not restricted here."""
    if metric_name != "err":
        return

    # free_run doesn't have erase_concept field - allow it
    if technique_name == "free_run":
        return

    # ERR uses NudeNet + I2P and is nudity-specific
    if erase_concept != "nudity":
        raise ValidationError(
            f"Metric 'err' is nudity-specific "
            f"(uses NudeNet detector and I2P dataset). "
            f"Got erase_concept='{erase_concept}'. "
            f"For non-nudity concepts, use UA_IRA, FID, or CLIP_Score."
        )



def validate_uce_concept(erase_concept: Optional[str]) -> None:
    """Validate UCE concept matches one of its 3 pre-trained presets."""
    valid_presets = {"nudity", "violence", "dog"}

    if erase_concept not in valid_presets:
        raise ValidationError(
            f"UCE only supports presets: {sorted(valid_presets)}. "
            f"Got erase_concept='{erase_concept}'. "
            f"For custom concepts, use esd or mace."
        )


def validate_ua_ira_paths(metric_config: Dict[str, Any]) -> None:
    """Validate UA_IRA has required CSV paths."""
    if not metric_config.get("target_prompts_path"):
        raise ValidationError(
            "UA_IRA metric requires 'target_prompts_path' in metric_config. "
            "Provide path to CSV with target concept prompts."
        )
    if not metric_config.get("retain_prompts_path"):
        raise ValidationError(
            "UA_IRA metric requires 'retain_prompts_path' in metric_config. "
            "Provide path to CSV with retain concept prompts."
        )


def validate_technique_metric_pair(
    technique_name: str,
    technique_config: Dict[str, Any],
    metric_name: str,
    metric_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate a single technique-metric pair."""
    erase_concept = get_erase_concept(technique_name, technique_config)
    metric_config = metric_config or {}

    # Rule 1: Nudity-specific metrics
    validate_nudity_metrics(technique_name, erase_concept, metric_name)

    # Rule 4: UCE preset validation
    if technique_name == "uce":
        validate_uce_concept(erase_concept)

    # Rule 5: UA_IRA requires CSV paths
    if metric_name == "ua_ira":
        validate_ua_ira_paths(metric_config)


