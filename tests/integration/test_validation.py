"""
Integration tests for runners/validation.py.

Covers get_erase_concept, validate_nudity_metrics, validate_uce_concept,
validate_ua_ira_paths, and validate_technique_metric_pair.
"""
import pytest
from eval_unlearn.runners.validation import (
    get_erase_concept,
    validate_nudity_metrics,
    validate_uce_concept,
    validate_ua_ira_paths,
    validate_technique_metric_pair,
    ValidationError,
)

pytestmark = pytest.mark.integration


class TestGetEraseConcept:

    def test_standard_technique(self):
        assert get_erase_concept("ssd", {"erase_concept": "Nudity"}) == "nudity"

    def test_standard_technique_none(self):
        assert get_erase_concept("ssd", {}) is None

    def test_uce_uses_preset(self):
        assert get_erase_concept("uce", {"preset": "Violence"}) == "violence"

    def test_uce_no_preset(self):
        assert get_erase_concept("uce", {}) is None

    def test_free_run_erase_concept(self):
        assert get_erase_concept("free_run", {"erase_concept": "dog"}) == "dog"


class TestValidateNudityMetrics:

    def test_err_with_nudity_passes(self):
        validate_nudity_metrics("ssd", "nudity", "err")  # no exception

    def test_err_with_non_nudity_raises(self):
        with pytest.raises(ValidationError, match="nudity-specific"):
            validate_nudity_metrics("ssd", "violence", "err")

    def test_err_with_free_run_passes(self):
        validate_nudity_metrics("free_run", None, "err")  # free_run is allowed

    def test_non_err_metric_always_passes(self):
        validate_nudity_metrics("ssd", "violence", "asr_p4d")  # no exception


class TestValidateUCEConcept:

    def test_valid_nudity(self):
        validate_uce_concept("nudity")  # no exception

    def test_valid_violence(self):
        validate_uce_concept("violence")  # no exception

    def test_valid_dog(self):
        validate_uce_concept("dog")  # no exception

    def test_invalid_concept_raises(self):
        with pytest.raises(ValidationError, match="only supports presets"):
            validate_uce_concept("cat")

    def test_none_concept_raises(self):
        with pytest.raises(ValidationError):
            validate_uce_concept(None)


class TestValidateUAIRAPaths:

    def test_both_paths_provided_passes(self):
        validate_ua_ira_paths({
            "target_prompts_path": "/a/t.csv",
            "retain_prompts_path": "/a/r.csv",
        })  # no exception

    def test_missing_target_raises(self):
        with pytest.raises(ValidationError, match="target_prompts_path"):
            validate_ua_ira_paths({"retain_prompts_path": "/a/r.csv"})

    def test_missing_retain_raises(self):
        with pytest.raises(ValidationError, match="retain_prompts_path"):
            validate_ua_ira_paths({"target_prompts_path": "/a/t.csv"})


class TestValidateTechniqueMetricPair:

    def test_valid_ssd_asr_p4d(self):
        validate_technique_metric_pair("ssd", {"erase_concept": "nudity"}, "asr_p4d", {})

    def test_valid_uce_nudity(self):
        validate_technique_metric_pair("uce", {"preset": "nudity"}, "asr_p4d", {})

    def test_err_with_violence_raises(self):
        with pytest.raises(ValidationError):
            validate_technique_metric_pair("ssd", {"erase_concept": "violence"}, "err", {})

    def test_uce_invalid_preset_raises(self):
        with pytest.raises(ValidationError):
            validate_technique_metric_pair("uce", {"preset": "cat"}, "asr_p4d", {})

    def test_ua_ira_missing_paths_raises(self):
        with pytest.raises(ValidationError):
            validate_technique_metric_pair("ssd", {"erase_concept": "nudity"}, "ua_ira", {})

    def test_ua_ira_with_paths_passes(self):
        validate_technique_metric_pair(
            "ssd", {"erase_concept": "nudity"}, "ua_ira",
            {"target_prompts_path": "/t.csv", "retain_prompts_path": "/r.csv"},
        )
