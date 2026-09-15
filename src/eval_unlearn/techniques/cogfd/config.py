from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...configs.base import BaseConfig


@dataclass(frozen=True)
class CoGFDConfig(BaseConfig):
    """Configuration for the Concept Graph Fine-tuning for Diffusion (CoGFD) unlearning technique.

    CoGFD erases harmful *combinations* of concepts from a diffusion model's UNet while
    preserving the ability to generate the individual component concepts in benign contexts.
    It does this through three complementary loss terms:

    - **Erasure loss** (``lambda_erase``): drives the model away from generating outputs
      matching any prompt in ``combination_prompts`` (or ``erase_concept`` if that list is
      empty).
    - **Preservation loss** (``lambda_preserve``): keeps the model's outputs for each
      concept listed in ``preserve_concepts`` close to the frozen reference model, so that
      individual concepts are not inadvertently degraded.
    - **Decoupling loss** (``lambda_decouple``): pushes intermediate UNet features apart
      for the harmful combination versus the individual components, weakening the learned
      association without destroying either concept independently.

    Attributes:
        model_id: HuggingFace model hub identifier for the base diffusion model to fine-tune.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).  When ``None`` the
            technique auto-selects the best available device at runtime.
        use_fp16: Whether to load and train in 16-bit floating point.  Reduces VRAM usage
            roughly 2x at the cost of a small numerical precision loss.

        erase_concept: Single-string fallback target concept used when
            ``combination_prompts`` is empty.  Typical values: ``"nudity"``,
            ``"violence"``.  When ``combination_prompts`` is non-empty this field
            no longer affects the training loss; it is used only for logging and
            runner-level metric routing (e.g. selecting concept-specific evaluation
            metrics).
        combination_prompts: A list of text prompts that each express a harmful
            *combination* of concepts to be erased (the concept logic graph).  These
            prompts are used together to define the erasure target so that no single
            phrasing is over-relied on.  If empty, ``[erase_concept]`` is used instead.
        preserve_concepts: Individual component concepts (e.g. ``["person", "nude"]``)
            whose independent generation ability should be retained after unlearning.
            Each entry is turned into a text prompt and used in the preservation loss.
            Leave empty to skip explicit preservation.

        lambda_erase: Weight on the combination erasure loss term.  Must be >= 0.
        lambda_preserve: Weight on the individual-concept preservation loss term.
            Defaults to 2.0 — intentionally higher than ``lambda_erase`` to prevent the
            fine-tuning from degrading the model's general generation quality.
            Must be >= 0.
        lambda_decouple: Weight on the feature decoupling loss term.  Must be >= 0.

        train_steps: Number of gradient-update steps for fine-tuning the UNet.
            Must be > 0.
        learning_rate: Optimiser learning rate.  Must be > 0.

        load_path: Optional filesystem path to a directory saved by a previous CoGFD run
            (via ``save_path``).  When set, training is skipped entirely and the modified
            UNet is loaded directly from this directory.  The directory must have been
            written by CoGFD's ``save_path`` (i.e. contain a ``unet/`` subdirectory in
            HuggingFace ``save_pretrained`` format).
        save_path: Optional filesystem path at which to save the modified UNet weights
            after fine-tuning.  Produces a directory with a ``unet/`` subdirectory in
            HuggingFace ``save_pretrained`` format.  When ``None`` the weights are kept
            in memory only and not persisted.  Only used when training runs (i.e.
            ``load_path`` is not set).

        num_inference_steps: Number of denoising steps used during evaluation / sample
            generation.
        guidance_scale: Classifier-free guidance scale applied during sample generation.
            Higher values increase prompt adherence at the cost of diversity.
    """

    # Model
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: Optional[str] = None
    use_fp16: bool = True

    # Target concept (used as fallback if combination_prompts is empty)
    erase_concept: str = "nudity"

    # Concept logic graph — multiple prompts expressing the harmful combination.
    # Leave empty to fall back to [erase_concept].
    combination_prompts: List[str] = field(default_factory=list)

    # Individual component concepts to preserve (e.g. ["person", "nude"]).
    # Leave empty to skip explicit individual-concept preservation.
    preserve_concepts: List[str] = field(default_factory=list)

    # Loss weights
    lambda_erase: float = 1.0     # combination erasure weight
    lambda_preserve: float = 2.0  # individual preservation weight (higher than erase to prevent model degradation)
    lambda_decouple: float = 0.5  # feature decoupling weight

    # Fine-tuning
    train_steps: int = 150
    learning_rate: float = 1e-5

    # Load pre-trained weights (skips training entirely)
    load_path: Optional[str] = None
    # Save modified UNet weights
    save_path: Optional[str] = None

    # Generation
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self):
        """Validate field values after dataclass initialisation.

        Raises:
            ValueError: If ``train_steps`` or ``learning_rate`` are non-positive, or if
                any loss-weight (``lambda_erase``, ``lambda_preserve``,
                ``lambda_decouple``) is negative.
        """
        if self.train_steps <= 0:
            raise ValueError(f"train_steps must be > 0, got {self.train_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.lambda_erase < 0:
            raise ValueError(f"lambda_erase must be >= 0, got {self.lambda_erase}")
        if self.lambda_preserve < 0:
            raise ValueError(f"lambda_preserve must be >= 0, got {self.lambda_preserve}")
        if self.lambda_decouple < 0:
            raise ValueError(f"lambda_decouple must be >= 0, got {self.lambda_decouple}")
        if not self.erase_concept or not self.erase_concept.strip():
            raise ValueError("erase_concept must not be empty.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoGFDConfig":
        """Create a ``CoGFDConfig`` from a plain dictionary.

        Unrecognised keys are silently ignored (inherited behaviour from
        ``BaseConfig.from_dict``).  ``model_id`` is an ``init=False`` field and
        is therefore excluded automatically.

        Args:
            data: Mapping of field names to values.

        Returns:
            A new frozen ``CoGFDConfig`` instance.
        """
        return super().from_dict(data)
