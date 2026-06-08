"""Unit tests for all technique config dataclasses."""
import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# ESDConfig
# ---------------------------------------------------------------------------
class TestESDConfig:
    def _make(self):
        from eval_learn.techniques.esd.config import ESDConfig
        return ESDConfig

    def test_defaults(self):
        cfg = self._make()(erase_concept="nudity")
        assert cfg.erase_concept == "nudity"
        assert cfg.train_method == "noxattn"
        assert cfg.device == "cuda"
        assert cfg.model_id == "CompVis/stable-diffusion-v1-4"

    def test_from_dict(self):
        cfg = self._make().from_dict({"erase_concept": "violence", "train_method": "xattn"})
        assert cfg.erase_concept == "violence"
        assert cfg.train_method == "xattn"

    def test_invalid_train_method(self):
        with pytest.raises(ValueError, match="Unknown train_method"):
            self._make()(erase_concept="nudity", train_method="invalid")

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make()(erase_concept="")

    def test_all_valid_train_methods(self):
        from eval_learn.techniques.esd.config import TRAIN_METHODS
        for m in TRAIN_METHODS:
            cfg = self._make()(erase_concept="nudity", train_method=m)
            assert cfg.train_method == m

    def test_load_save_path(self):
        cfg = self._make()(erase_concept="nudity", load_path="/tmp/w.pt", save_path="/tmp/s.pt")
        assert cfg.load_path == "/tmp/w.pt"
        assert cfg.save_path == "/tmp/s.pt"


# ---------------------------------------------------------------------------
# SSDConfig
# ---------------------------------------------------------------------------
class TestSSDConfig:
    def _make(self):
        from eval_learn.techniques.ssd.config import SSDConfig
        return SSDConfig

    def test_defaults(self):
        cfg = self._make()(erase_concept="nudity")
        assert cfg.alpha == 0.1
        assert cfg.dampening_coeff == 1.0
        assert cfg.num_fisher_samples == 50

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            self._make()(erase_concept="nudity", alpha=0)

    def test_invalid_dampening(self):
        with pytest.raises(ValueError, match="dampening_coeff must be > 0"):
            self._make()(erase_concept="nudity", dampening_coeff=-1)

    def test_invalid_fisher_samples(self):
        with pytest.raises(ValueError, match="num_fisher_samples must be > 0"):
            self._make()(erase_concept="nudity", num_fisher_samples=0)

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make()(erase_concept="")

    def test_resolved_forget_prompts_default(self):
        cfg = self._make()(erase_concept="violence")
        assert cfg.resolved_forget_prompts() == ["violence"]

    def test_resolved_forget_prompts_custom(self):
        cfg = self._make()(erase_concept="nudity", forget_prompts=["naked", "nude"])
        assert cfg.resolved_forget_prompts() == ["naked", "nude"]

    def test_resolved_retain_prompts_default(self):
        cfg = self._make()(erase_concept="nudity")
        assert cfg.resolved_retain_prompts() == ["", "a photo", "an image"]

    def test_resolved_retain_prompts_custom(self):
        cfg = self._make()(erase_concept="nudity", retain_prompts=["a cat"])
        assert cfg.resolved_retain_prompts() == ["a cat"]

    def test_from_dict(self):
        cfg = self._make().from_dict({"erase_concept": "nudity", "alpha": 0.5})
        assert cfg.alpha == 0.5


# ---------------------------------------------------------------------------
# CAConfig
# ---------------------------------------------------------------------------
class TestCAConfig:
    def _make(self):
        from eval_learn.techniques.ca.config import CAConfig
        return CAConfig

    def test_defaults(self):
        cfg = self._make()(erase_concept="nudity", anchor_concept="a person wearing clothes")
        assert cfg.train_steps == 400
        assert cfg.learning_rate == 1e-5

    def test_invalid_train_steps(self):
        with pytest.raises(ValueError, match="train_steps must be > 0"):
            self._make()(erase_concept="nudity", anchor_concept="x", train_steps=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            self._make()(erase_concept="nudity", anchor_concept="x", learning_rate=0)

    def test_from_dict_missing_anchor(self):
        with pytest.raises(ValueError, match="anchor_concept"):
            self._make().from_dict({"erase_concept": "nudity"})

    def test_from_dict_with_both(self):
        cfg = self._make().from_dict({"erase_concept": "nudity", "anchor_concept": "clothed person"})
        assert cfg.anchor_concept == "clothed person"


# ---------------------------------------------------------------------------
# CoGFDConfig
# ---------------------------------------------------------------------------
class TestCoGFDConfig:
    def _make(self):
        from eval_learn.techniques.cogfd.config import CoGFDConfig
        return CoGFDConfig

    def test_defaults(self):
        cfg = self._make()(erase_concept="nudity")
        assert cfg.lambda_erase == 1.0
        assert cfg.lambda_preserve == 2.0
        assert cfg.lambda_decouple == 0.5
        assert cfg.train_steps == 150

    def test_invalid_train_steps(self):
        with pytest.raises(ValueError, match="train_steps must be > 0"):
            self._make()(erase_concept="nudity", train_steps=0)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            self._make()(erase_concept="nudity", learning_rate=-1e-5)

    def test_negative_lambda(self):
        with pytest.raises(ValueError, match="lambda_erase must be >= 0"):
            self._make()(erase_concept="nudity", lambda_erase=-1)

    def test_negative_lambda_preserve(self):
        with pytest.raises(ValueError, match="lambda_preserve must be >= 0"):
            self._make()(erase_concept="nudity", lambda_preserve=-1)

    def test_negative_lambda_decouple(self):
        with pytest.raises(ValueError, match="lambda_decouple must be >= 0"):
            self._make()(erase_concept="nudity", lambda_decouple=-1)

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make()(erase_concept="")

    def test_combination_prompts(self):
        cfg = self._make()(erase_concept="nudity", combination_prompts=["a", "b"])
        assert cfg.combination_prompts == ["a", "b"]

    def test_from_dict(self):
        cfg = self._make().from_dict({"erase_concept": "violence", "train_steps": 50})
        assert cfg.train_steps == 50


# ---------------------------------------------------------------------------
# TraSCEConfig
# ---------------------------------------------------------------------------
class TestTraSCEConfig:
    def _make(self):
        from eval_learn.techniques.trasce.config import TraSCEConfig
        return TraSCEConfig

    def test_defaults(self):
        cfg = self._make()(erase_concept="nudity")
        assert cfg.sigma == 1.0
        assert cfg.discriminator_guidance_scale == 5.0
        assert cfg.guidance_loss_scale == 15.0

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make()(erase_concept="")

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            self._make()(erase_concept="nudity", sigma=0)

    def test_invalid_discriminator_scale(self):
        with pytest.raises(ValueError, match="discriminator_guidance_scale must be > 0"):
            self._make()(erase_concept="nudity", discriminator_guidance_scale=0)

    def test_invalid_guidance_loss_scale(self):
        with pytest.raises(ValueError, match="guidance_loss_scale must be > 0"):
            self._make()(erase_concept="nudity", guidance_loss_scale=0)


# ---------------------------------------------------------------------------
# AdvUnlearnConfig
# ---------------------------------------------------------------------------
class TestAdvUnlearnConfig:
    def _make(self):
        from eval_learn.techniques.advunlearn.config import AdvUnlearnConfig
        return AdvUnlearnConfig

    def test_defaults(self):
        cfg = self._make().from_dict({"erase_concept": "nudity"})
        assert cfg.train_method == "text_encoder_full"
        assert cfg.attack_method == "pgd"

    def test_invalid_train_method(self):
        with pytest.raises(ValueError, match="Unknown train_method"):
            self._make().from_dict({"erase_concept": "nudity", "train_method": "bad"})

    def test_layer_pattern_train_method(self):
        cfg = self._make().from_dict({"erase_concept": "nudity", "train_method": "text_encoder_layer012"})
        assert cfg.train_method == "text_encoder_layer012"

    def test_invalid_dataset_retain(self):
        with pytest.raises(ValueError, match="Unknown dataset_retain"):
            self._make().from_dict({"erase_concept": "nudity", "dataset_retain": "bad_ds"})

    def test_invalid_retain_train(self):
        with pytest.raises(ValueError, match="Unknown retain_train"):
            self._make().from_dict({"erase_concept": "nudity", "retain_train": "bad"})

    def test_invalid_attack_method(self):
        with pytest.raises(ValueError, match="Unknown attack_method"):
            self._make().from_dict({"erase_concept": "nudity", "attack_method": "bad"})

    def test_invalid_attack_type(self):
        with pytest.raises(ValueError, match="Unknown attack_type"):
            self._make().from_dict({"erase_concept": "nudity", "attack_type": "bad"})

    def test_invalid_attack_embd_type(self):
        with pytest.raises(ValueError, match="Unknown attack_embd_type"):
            self._make().from_dict({"erase_concept": "nudity", "attack_embd_type": "bad"})

    def test_invalid_component(self):
        with pytest.raises(ValueError, match="Unknown component"):
            self._make().from_dict({"erase_concept": "nudity", "component": "bad"})

    def test_warmup_iter_too_large(self):
        with pytest.raises(ValueError, match="warmup_iter must be < train_steps"):
            self._make().from_dict({"erase_concept": "nudity", "train_steps": 5, "warmup_iter": 5})

    def test_invalid_train_steps(self):
        with pytest.raises(ValueError, match="train_steps must be > 0"):
            self._make().from_dict({"erase_concept": "nudity", "train_steps": 0, "warmup_iter": 0})

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make().from_dict({"erase_concept": ""})

    def test_all_valid_enum_values(self):
        from eval_learn.techniques.advunlearn.config import (
            TRAIN_METHODS, ATTACK_METHODS, ATTACK_TYPES, COMPONENTS, RETAIN_DATASETS, RETAIN_TRAIN_METHODS
        )
        cfg = self._make().from_dict({
            "erase_concept": "nudity",
            "train_method": TRAIN_METHODS[0],
            "attack_method": ATTACK_METHODS[0],
            "attack_type": ATTACK_TYPES[0],
            "component": COMPONENTS[0],
            "dataset_retain": RETAIN_DATASETS[0],
            "retain_train": RETAIN_TRAIN_METHODS[0],
        })
        assert cfg.erase_concept == "nudity"


# ---------------------------------------------------------------------------
# SAFREEConfig
# ---------------------------------------------------------------------------
class TestSAFREEConfig:
    def _make(self):
        from eval_learn.techniques.SAFREE.config import SAFREEConfig
        return SAFREEConfig

    def test_defaults_nudity(self):
        cfg = self._make().from_dict({"erase_concept": "nudity"})
        assert cfg.erase_concept == "nudity"
        assert cfg.enable_svf is True

    def test_invalid_concept_without_custom(self):
        with pytest.raises(ValueError, match="not a calibrated category"):
            self._make().from_dict({"erase_concept": "violence"})

    def test_custom_concept_disables_svf(self):
        cfg = self._make().from_dict({
            "erase_concept": "violence",
            "custom_unsafe_concepts": ["violent scene"],
        })
        assert cfg.enable_svf is False

    def test_custom_concept_explicit_svf_override(self):
        cfg = self._make().from_dict({
            "erase_concept": "violence",
            "custom_unsafe_concepts": ["violent scene"],
            "enable_svf": True,
        })
        assert cfg.enable_svf is True

    def test_artist_concept(self):
        cfg = self._make().from_dict({"erase_concept": "artists-VanGogh"})
        assert cfg.erase_concept == "artists-VanGogh"


# ---------------------------------------------------------------------------
# MACEConfig
# ---------------------------------------------------------------------------
class TestMACEConfig:
    def _make(self):
        from eval_learn.techniques.mace.config import MACEConfig
        return MACEConfig

    def test_defaults(self):
        cfg = self._make()(erase_concept="nudity")
        assert cfg.lambda_cfr == 0.1
        assert cfg.num_inference_steps == 50

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make()(erase_concept="")

    def test_invalid_lambda_cfr(self):
        with pytest.raises(ValueError, match="lambda_cfr must be > 0"):
            self._make()(erase_concept="nudity", lambda_cfr=0)

    def test_list_erase_concept(self):
        cfg = self._make()(erase_concept=["nudity", "naked"])
        assert cfg.erase_concept == ["nudity", "naked"]


# ---------------------------------------------------------------------------
# SAeUronConfig
# ---------------------------------------------------------------------------
class TestSAeUronConfig:
    def _make(self):
        from eval_learn.techniques.saeuron.config import SAeUronConfig
        return SAeUronConfig

    def test_defaults(self):
        cfg = self._make().from_dict({"erase_concept": "nudity"})
        assert cfg.multiplier == -20.0
        assert cfg.erase_concept == "nudity"

    def test_zero_multiplier(self):
        with pytest.raises(ValueError, match="multiplier must not be 0"):
            self._make().from_dict({"erase_concept": "nudity", "multiplier": 0})

    def test_invalid_guidance_scale(self):
        with pytest.raises(ValueError, match="guidance_scale must be > 1.0"):
            self._make().from_dict({"erase_concept": "nudity", "guidance_scale": 1.0})

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make().from_dict({"erase_concept": ""})

    def test_device_auto_selected(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            cfg = self._make().from_dict({"erase_concept": "nudity"})
            assert cfg.device == "cpu"

    def test_device_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            cfg = self._make().from_dict({"erase_concept": "nudity"})
            assert cfg.device == "cuda"

    def test_unknown_concept_prints_warning(self, capsys):
        self._make().from_dict({"erase_concept": "violence"})
        out = capsys.readouterr().out
        assert "baseline activation tensor" in out


# ---------------------------------------------------------------------------
# SLDConfig
# ---------------------------------------------------------------------------
class TestSLDConfig:
    def _make(self):
        from eval_learn.techniques.sld.config import SLDConfig
        return SLDConfig

    def test_defaults(self):
        cfg = self._make().from_dict({"erase_concept": "nudity"})
        assert cfg.erase_concept == "nudity"
        assert cfg.sld_guidance_scale == 5000

    def test_invalid_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must be one of"):
            self._make().from_dict({"erase_concept": "invalid"})

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="erase_concept must not be empty"):
            self._make().from_dict({"erase_concept": ""})

    def test_preset_strong(self):
        cfg = self._make().from_dict({"erase_concept": "nudity", "preset": "strong"})
        assert cfg.sld_guidance_scale == 2000
        assert cfg.sld_warmup_steps == 7

    def test_preset_max(self):
        cfg = self._make().from_dict({"erase_concept": "nudity", "preset": "max"})
        assert cfg.sld_guidance_scale == 5000

    def test_preset_none(self):
        cfg = self._make().from_dict({"erase_concept": "nudity", "preset": "none"})
        assert cfg.sld_guidance_scale == 0

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown SLD preset"):
            self._make().from_dict({"erase_concept": "nudity", "preset": "ultra"})

    def test_preset_does_not_override_explicit(self):
        cfg = self._make().from_dict({
            "erase_concept": "nudity",
            "preset": "weak",
            "sld_guidance_scale": 999,
        })
        assert cfg.sld_guidance_scale == 999

    def test_all_valid_concepts(self):
        from eval_learn.techniques.sld.config import _VALID_ERASE_CONCEPTS
        for concept in _VALID_ERASE_CONCEPTS:
            cfg = self._make().from_dict({"erase_concept": concept})
            assert cfg.erase_concept == concept


# ---------------------------------------------------------------------------
# UCEConfig
# ---------------------------------------------------------------------------
class TestUCEConfig:
    def _make(self):
        from eval_learn.techniques.uce.config import UCEConfig
        return UCEConfig

    def test_preset_nudity(self):
        cfg = self._make()(preset="nudity")
        assert cfg.preset == "nudity"

    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="UCE requires one of"):
            self._make()()

    def test_preset_and_load_path_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._make()(preset="nudity", load_path="/tmp/w.pt")

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown UCE preset"):
            self._make()(preset="bad")

    def test_empty_erase_concept(self):
        with pytest.raises(ValueError, match="non-empty string"):
            self._make()(erase_concept="   ", save_path="/tmp/s")

    def test_erase_concept_without_save_path(self):
        with pytest.raises(ValueError, match="save_path"):
            self._make()(erase_concept="nudity")

    def test_invalid_concept_type(self):
        with pytest.raises(ValueError, match="concept_type must be one of"):
            self._make()(preset="nudity", concept_type="bad")

    def test_load_path(self):
        cfg = self._make()(load_path="/tmp/weights.pt")
        assert cfg.load_path == "/tmp/weights.pt"


# ---------------------------------------------------------------------------
# FreeRunConfig
# ---------------------------------------------------------------------------
class TestFreeRunConfig:
    def _make(self):
        from eval_learn.techniques.free_run.config import FreeRunConfig
        return FreeRunConfig

    def test_valid(self):
        cfg = self._make()(model_id="stabilityai/stable-diffusion-2")
        assert cfg.model_id == "stabilityai/stable-diffusion-2"

    def test_empty_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            self._make()(model_id="")

    def test_from_dict(self):
        cfg = self._make().from_dict({"model_id": "runwayml/stable-diffusion-v1-5"})
        assert cfg.num_inference_steps == 50


# ---------------------------------------------------------------------------
# ConceptSteerersConfig
# ---------------------------------------------------------------------------
class TestConceptSteerersConfig:
    def _make(self):
        from eval_learn.techniques.concept_steerers.config import ConceptSteerersConfig
        return ConceptSteerersConfig

    def test_defaults(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            cfg = self._make().from_dict({"erase_concept": "nudity"})
            assert cfg.device == "cpu"
            assert cfg.multiplier == 1.0

    def test_cuda_device(self):
        with patch("torch.cuda.is_available", return_value=True):
            cfg = self._make().from_dict({"erase_concept": "nudity"})
            assert cfg.device == "cuda"

    def test_mps_device(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True):
            cfg = self._make().from_dict({"erase_concept": "nudity"})
            assert cfg.device == "mps"

    def test_empty_erase_concept(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(ValueError, match="erase_concept must not be empty"):
                self._make().from_dict({"erase_concept": ""})

    def test_low_guidance_scale(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            with pytest.raises(ValueError, match="guidance_scale must be > 1.0"):
                self._make().from_dict({"erase_concept": "nudity", "guidance_scale": 1.0})


# ---------------------------------------------------------------------------
# AdvUnlearnConfig validation errors
# ---------------------------------------------------------------------------
class TestAdvUnlearnConfigValidation:
    def _cfg(self, **ov):
        from eval_learn.techniques.advunlearn.config import AdvUnlearnConfig
        base = dict(erase_concept="nudity", train_steps=10, attack_step=3,
                    retain_batch=4, retain_step=2, adv_prompt_num=4,
                    learning_rate=1e-5, attack_lr=1e-4, warmup_iter=5)
        base.update(ov)
        return AdvUnlearnConfig.from_dict(base)

    def test_train_steps_zero_raises(self):
        with pytest.raises(ValueError, match="train_steps"):
            self._cfg(train_steps=0)

    def test_attack_step_zero_raises(self):
        with pytest.raises(ValueError, match="attack_step"):
            self._cfg(attack_step=0)

    def test_retain_batch_zero_raises(self):
        with pytest.raises(ValueError, match="retain_batch"):
            self._cfg(retain_batch=0)

    def test_retain_step_zero_raises(self):
        with pytest.raises(ValueError, match="retain_step"):
            self._cfg(retain_step=0)

    def test_adv_prompt_num_zero_raises(self):
        with pytest.raises(ValueError, match="adv_prompt_num"):
            self._cfg(adv_prompt_num=0)

    def test_learning_rate_zero_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            self._cfg(learning_rate=0)

    def test_attack_lr_zero_raises(self):
        with pytest.raises(ValueError, match="attack_lr"):
            self._cfg(attack_lr=0)


# ---------------------------------------------------------------------------
# SAeUronConfig — device auto-detect
# ---------------------------------------------------------------------------
class TestSAeUronConfigDevice:
    def test_device_cpu_when_no_gpu(self):
        from eval_learn.techniques.saeuron.config import SAeUronConfig
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            cfg = SAeUronConfig.from_dict({"erase_concept": "nudity"})
        assert cfg.device == "cpu"
