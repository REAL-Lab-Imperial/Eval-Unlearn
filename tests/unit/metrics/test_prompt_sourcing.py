"""Unit tests for the shared adversarial-metric prompt-sourcing helper."""
import pytest

from eval_unlearn.metrics._prompt_sourcing import (
    resolve_prompt_source,
    VALID_PROMPT_SOURCES,
    DEFAULT_MIN_ADVERSARIAL_SAMPLES,
)


def test_valid_sources_set():
    assert VALID_PROMPT_SOURCES == {"auto", "i2p", "custom", "default"}


def test_default_min_adversarial_samples_is_100():
    assert DEFAULT_MIN_ADVERSARIAL_SAMPLES == 100


class TestResolvePromptSource:
    def test_auto_with_user_supplied_resolves_custom(self):
        assert resolve_prompt_source("auto", "nudity", user_supplied=True) == "custom"

    def test_auto_without_user_supplied_i2p_concept_resolves_i2p(self):
        assert resolve_prompt_source("auto", "nudity", user_supplied=False) == "i2p"
        assert resolve_prompt_source("auto", "violence", user_supplied=False) == "i2p"

    def test_auto_without_user_supplied_non_i2p_concept_resolves_default(self):
        assert resolve_prompt_source("auto", "a made up concept", user_supplied=False) == "default"

    def test_explicit_i2p_returned_even_if_user_supplied(self):
        assert resolve_prompt_source("i2p", "nudity", user_supplied=True) == "i2p"

    def test_explicit_default_returned_even_for_i2p_concept(self):
        assert resolve_prompt_source("default", "nudity", user_supplied=False) == "default"

    def test_explicit_custom_with_user_supplied_ok(self):
        assert resolve_prompt_source("custom", "nudity", user_supplied=True) == "custom"

    def test_explicit_custom_without_user_supplied_raises(self):
        with pytest.raises(ValueError, match="requires the caller to supply prompts"):
            resolve_prompt_source("custom", "nudity", user_supplied=False)

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="prompt_source must be one of"):
            resolve_prompt_source("not_a_source", "nudity", user_supplied=False)
