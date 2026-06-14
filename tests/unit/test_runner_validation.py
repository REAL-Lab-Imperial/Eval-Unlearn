"""Unit tests for runner validation logic."""

import pytest

from eval_unlearn.runners.validation import (
    ValidationError,
    get_erase_concept,
    validate_nudity_metrics,
    validate_uce_concept,
    validate_ua_ira_paths,
    validate_technique_metric_pair,
)


# ---------------------------------------------------------------------------
# get_erase_concept
# ---------------------------------------------------------------------------

class TestGetEraseConcept:
    def test_uce_uses_preset(self):
        assert get_erase_concept("uce", {"preset": "Nudity"}) == "nudity"

    def test_uce_preset_none_returns_none(self):
        assert get_erase_concept("uce", {}) is None

    def test_other_technique_uses_erase_concept(self):
        assert get_erase_concept("esd", {"erase_concept": "Violence"}) == "violence"

    def test_other_technique_no_erase_concept_returns_none(self):
        assert get_erase_concept("esd", {}) is None

    def test_lowercases_value(self):
        assert get_erase_concept("mace", {"erase_concept": "DOG"}) == "dog"


# ---------------------------------------------------------------------------
# validate_nudity_metrics
# ---------------------------------------------------------------------------

class TestValidateNudityMetrics:
    def test_non_nudity_metric_always_passes(self):
        # fid, clip_score, ua_ira don't care about erase_concept
        validate_nudity_metrics("esd", "dog", "fid")
        validate_nudity_metrics("esd", None, "clip_score")

    def test_asr_with_nudity_passes(self):
        validate_nudity_metrics("esd", "nudity", "asr")

    def test_asr_with_non_nudity_concept_passes(self):
        # ASR I2P supports all concepts — no restriction at the validation layer
        validate_nudity_metrics("esd", "violence", "asr")
        validate_nudity_metrics("esd", "dog", "asr")
        validate_nudity_metrics("esd", None, "asr")

    def test_err_with_nudity_passes(self):
        validate_nudity_metrics("esd", "nudity", "err")

    def test_err_with_none_concept_raises(self):
        with pytest.raises(ValidationError, match="nudity-specific"):
            validate_nudity_metrics("esd", None, "err")

    def test_err_with_violence_raises(self):
        with pytest.raises(ValidationError, match="nudity-specific"):
            validate_nudity_metrics("esd", "violence", "err")

    def test_err_with_free_run_passes(self):
        # free_run has no erase_concept — explicitly exempted
        validate_nudity_metrics("free_run", None, "err")

    def test_asr_i2p_with_violence_passes(self):
        validate_nudity_metrics("esd", "violence", "asr_i2p")

    def test_asr_p4d_with_violence_passes(self):
        validate_nudity_metrics("esd", "violence", "asr_p4d")

    def test_asr_mma_diffusion_with_violence_passes(self):
        validate_nudity_metrics("esd", "violence", "asr_mma_diffusion")


# ---------------------------------------------------------------------------
# validate_uce_concept
# ---------------------------------------------------------------------------

class TestValidateUceConcept:
    def test_valid_presets_pass(self):
        for preset in ("nudity", "violence", "dog"):
            validate_uce_concept(preset)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValidationError, match="UCE only supports presets"):
            validate_uce_concept("cat")

    def test_none_preset_raises(self):
        with pytest.raises(ValidationError, match="UCE only supports presets"):
            validate_uce_concept(None)

    def test_error_message_lists_valid_presets(self):
        with pytest.raises(ValidationError, match="dog.*nudity.*violence"):
            validate_uce_concept("unknown")


# ---------------------------------------------------------------------------
# validate_ua_ira_paths
# ---------------------------------------------------------------------------

class TestValidateUaIraPaths:
    def test_both_paths_present_passes(self):
        validate_ua_ira_paths(
            {"target_prompts_path": "/data/target.csv", "retain_prompts_path": "/data/retain.csv"}
        )

    def test_missing_target_prompts_raises(self):
        with pytest.raises(ValidationError, match="target_prompts_path"):
            validate_ua_ira_paths({"retain_prompts_path": "/data/retain.csv"})

    def test_missing_retain_prompts_raises(self):
        with pytest.raises(ValidationError, match="retain_prompts_path"):
            validate_ua_ira_paths({"target_prompts_path": "/data/target.csv"})

    def test_empty_config_raises_on_target_first(self):
        with pytest.raises(ValidationError, match="target_prompts_path"):
            validate_ua_ira_paths({})


# ---------------------------------------------------------------------------
# validate_technique_metric_pair
# ---------------------------------------------------------------------------

class TestValidateTechniqueMetricPair:
    def test_valid_esd_fid_pair(self):
        validate_technique_metric_pair("esd", {"erase_concept": "nudity"}, "fid")

    def test_valid_esd_asr_nudity(self):
        validate_technique_metric_pair("esd", {"erase_concept": "nudity"}, "asr")

    def test_asr_with_non_nudity_concept_passes(self):
        # ASR is no longer nudity-restricted
        validate_technique_metric_pair("esd", {"erase_concept": "dog"}, "asr")

    def test_saeuron_with_non_nudity_concept_passes(self):
        # saeuron is no longer restricted to nudity
        validate_technique_metric_pair("saeuron", {"erase_concept": "dog"}, "fid")

    def test_sld_with_violence_passes(self):
        # sld is no longer nudity-restricted
        validate_technique_metric_pair("sld", {"erase_concept": "violence"}, "fid")

    def test_safree_with_violence_passes(self):
        validate_technique_metric_pair("safree", {"erase_concept": "violence"}, "fid")

    def test_concept_steerers_with_violence_passes(self):
        validate_technique_metric_pair("concept_steerers", {"erase_concept": "violence"}, "fid")

    def test_err_with_violence_raises(self):
        with pytest.raises(ValidationError, match="nudity-specific"):
            validate_technique_metric_pair("esd", {"erase_concept": "violence"}, "err")

    def test_uce_with_invalid_preset_raises(self):
        with pytest.raises(ValidationError, match="UCE only supports presets"):
            validate_technique_metric_pair("uce", {"preset": "cat"}, "fid")

    def test_uce_with_valid_preset_passes(self):
        validate_technique_metric_pair("uce", {"preset": "dog"}, "fid")

    def test_ua_ira_without_paths_raises(self):
        with pytest.raises(ValidationError, match="target_prompts_path"):
            validate_technique_metric_pair("esd", {"erase_concept": "dog"}, "ua_ira")

    def test_ua_ira_with_paths_passes(self):
        validate_technique_metric_pair(
            "esd",
            {"erase_concept": "dog"},
            "ua_ira",
            {"target_prompts_path": "/t.csv", "retain_prompts_path": "/r.csv"},
        )

    def test_metric_config_defaults_to_empty_dict(self):
        # Should not raise — metric_config=None is handled internally
        validate_technique_metric_pair("esd", {"erase_concept": "nudity"}, "fid", None)

