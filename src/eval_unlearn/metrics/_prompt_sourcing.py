"""Shared prompt-source resolution for adversarial ASR metrics.

MMA-Diffusion, Ring-A-Bell, and P4D each need a set of target/seed prompts to
attack. Each supports three sources, selected via a ``prompt_source`` config
field:
  - "custom":  prompts supplied directly by the caller.
  - "i2p":     borrowed from the I2P dataset's matching category (nudity,
               violence, harassment, hate, illegal activity, self-harm,
               shocking).
  - "default": generic template prompts synthesized from the concept name,
               for concepts with no I2P category and no user-supplied
               prompts.

"auto" (the default) picks "custom" when the caller supplied prompts,
otherwise "i2p" if the concept maps to an I2P category, otherwise "default".
This is a config-time choice (not an interactive runtime prompt) so these
metrics keep working unattended inside multi-technique benchmark loops.
"""
from ..datasets._i2p_categories import CONCEPT_TO_I2P_CATEGORY

VALID_PROMPT_SOURCES = frozenset({"auto", "i2p", "custom", "default"})

# Default floor on the number of adversarial samples generated when the
# caller has NOT supplied their own target/seed prompts (i.e. source
# resolves to "i2p" or "default"). Not applied when source == "custom":
# a caller providing their own prompts controls the count directly.
DEFAULT_MIN_ADVERSARIAL_SAMPLES = 100


def resolve_prompt_source(prompt_source: str, concept_name: str, user_supplied: bool) -> str:
    """Resolve ``prompt_source`` to a concrete value: 'custom', 'i2p', or 'default'."""
    if prompt_source not in VALID_PROMPT_SOURCES:
        raise ValueError(
            f"prompt_source must be one of {sorted(VALID_PROMPT_SOURCES)}, got '{prompt_source}'"
        )
    if prompt_source == "custom" and not user_supplied:
        raise ValueError(
            "prompt_source='custom' requires the caller to supply prompts, but none were given."
        )
    if prompt_source != "auto":
        return prompt_source
    if user_supplied:
        return "custom"
    return "i2p" if concept_name in CONCEPT_TO_I2P_CATEGORY else "default"
