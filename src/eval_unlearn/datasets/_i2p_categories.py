"""I2P concept/category taxonomy — kept dependency-light (no torch/datasets
imports) so callers that only need the mapping don't pay for importing the
heavier HF-streaming machinery in ``i2p_csv.py``.
"""

# Maps user-facing concept names to I2P dataset category labels.
# I2P categories come from the AIML-TUDA/i2p dataset's `categories` column.
CONCEPT_TO_I2P_CATEGORY: dict = {
    "nudity":           "sexual",
    "harassment":       "harassment",
    "hate":             "hate",
    "illegal activity": "illegal activity",
    "self-harm":        "self-harm",
    "shocking":         "shocking",
    "violence":         "violence",
}
