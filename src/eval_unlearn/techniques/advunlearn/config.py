import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from ...configs.base import BaseConfig

# Fixed train_method strings
TRAIN_METHODS = [
    "text_encoder_full",
    "noxattn",
    "selfattn",
    "xattn",
    "full",
    "notime",
    "xlayer",
    "selflayer",
]
# Also accepts "text_encoder_layer<digits>" e.g. "text_encoder_layer012_910"
_TRAIN_METHOD_LAYER_RE = re.compile(r"^text_encoder_layer[\d_]+$")

RETAIN_TRAIN_METHODS = ["iter", "reg"]

ATTACK_METHODS = ["pgd", "fast_at"]

ATTACK_TYPES = ["prefix_k", "suffix_k", "replace_k", "add", "mid_k", "insert_k", "per_k_words"]

ATTACK_EMBD_TYPES = ["word_embd"]

COMPONENTS = ["all", "ffn", "attn"]

RETAIN_DATASETS = ["coco_object", "imagenet243", "coco_object_no_filter", "imagenet243_no_filter"]


@dataclass(frozen=True)
class AdvUnlearnConfig(BaseConfig):
    """Configuration for AdvUnlearn (Adversarially Robust Concept Unlearning).

    AdvUnlearn alternates between two phases each training iteration:

    1. **Adversarial attack phase**: an adversarial prompt is crafted (via PGD or Fast-AT)
       in the text-encoder embedding space to maximally reconstruct the erased concept.
       This produces a worst-case prompt that is hard to defend against.
    2. **Unlearning update phase**: the model is fine-tuned to suppress generation of the
       erased concept *under* the adversarial prompt, while simultaneously replaying
       samples from a clean retention dataset to prevent catastrophic forgetting of
       unrelated concepts.

    The training target (the part of the model being fine-tuned) is controlled by
    ``train_method`` and ``component``.  Valid enumerated values for each string field
    are defined by the module-level constants (``TRAIN_METHODS``, ``RETAIN_DATASETS``,
    etc.) and are validated inside ``from_dict``.

    Attributes:
        model_id: HuggingFace hub identifier for the base diffusion model.  Fixed to
            ``CompVis/stable-diffusion-v1-4`` and excluded from ``__init__`` — it cannot
            be changed via config.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).
        use_fp16: Whether to load and train in 16-bit floating point.

        erase_concept: The target concept to erase from the model (e.g. ``"nudity"``).

        train_method: Which parameters of the text encoder / UNet to update.  Must be
            one of ``TRAIN_METHODS`` (e.g. ``"text_encoder_full"``, ``"xattn"``,
            ``"full"``) or match the pattern ``text_encoder_layer<digits>`` to target
            specific transformer layers (e.g. ``"text_encoder_layer012_910"``).

        dataset_retain: Name of the retention dataset used to replay clean samples
            during training.  Must be one of ``RETAIN_DATASETS``.
        retain_train: Strategy for interleaving retention samples with unlearning
            updates.  ``"iter"`` replays one retention batch per unlearning step;
            ``"reg"`` adds a regularisation term instead.  Must be one of
            ``RETAIN_TRAIN_METHODS``.
        retain_batch: Number of retention samples drawn per retention step.
            Must be > 0.
        retain_step: Number of gradient steps taken on the retention loss per
            unlearning iteration.  Must be > 0.
        retain_loss_w: Scalar weight applied to the retention loss term.

        start_guidance: Classifier-free guidance scale used when computing the
            unlearning target (the *frozen* reference model's prediction).
        negative_guidance: Guidance scale applied to the negative (erased-concept)
            direction during the unlearning loss computation.
        train_steps: Total number of unlearning iterations (each consisting of an
            adversarial attack phase followed by a model-update phase).  Must be > 0.
        learning_rate: Optimiser learning rate for the model-update phase.  Must be > 0.

        attack_method: Algorithm used to generate adversarial prompts.  ``"pgd"`` runs
            Projected Gradient Descent; ``"fast_at"`` is a single-step variant.  Must
            be one of ``ATTACK_METHODS``.
        attack_step: Number of iterative steps in the adversarial attack optimisation.
            Must be > 0.
        attack_lr: Step size (learning rate) used by the adversarial attack optimiser.
            Must be > 0.
        attack_type: Where in the token sequence the adversarial tokens are injected.
            Options include ``"prefix_k"`` (prepend *k* tokens), ``"suffix_k"`` (append),
            ``"replace_k"`` (replace first *k*), ``"add"`` (element-wise add to all),
            ``"mid_k"``, ``"insert_k"``, ``"per_k_words"``.  Must be one of
            ``ATTACK_TYPES``.
        attack_init: Initialisation strategy for adversarial token embeddings at the
            start of each attack phase.  ``"latest"`` reuses the tokens from the
            previous iteration for a warm start.
        attack_embd_type: Embedding space in which the adversarial perturbation is
            applied.  Currently only ``"word_embd"`` (word-embedding space) is
            supported.  Must be one of ``ATTACK_EMBD_TYPES``.
        adv_prompt_num: Number of independent adversarial prompts maintained in
            parallel during the attack phase.  Must be > 0.
        adv_prompt_update_step: How many unlearning steps to take before refreshing the
            adversarial prompt.  A value of 1 means the prompt is re-optimised every
            step.
        warmup_iter: Number of initial iterations to run *without* adversarial prompts
            (i.e. plain unlearning warm-up).  Must be strictly less than
            ``train_steps``.

        component: Which sub-module of the UNet is targeted for fine-tuning.
            ``"all"`` updates all parameters selected by ``train_method``; ``"ffn"``
            restricts to feed-forward layers; ``"attn"`` restricts to attention layers.
            Must be one of ``COMPONENTS``.
        norm_layer: Whether to include layer-normalisation parameters in the set of
            trainable parameters.

        ddim_steps: Number of DDIM denoising steps used when generating images during
            training (e.g. for computing the unlearning loss via a full forward pass).
        save_interval: Checkpoint save frequency in training iterations.

        save_dir: Optional directory path for saving model checkpoints.  When ``None``
            no checkpoints are written to disk.
        load_path: Optional path to a previously saved checkpoint from which to
            resume training.  When ``None`` training starts from the pretrained weights.
        num_inference_steps: Number of denoising steps used during evaluation / sample
            generation.
        guidance_scale: Classifier-free guidance scale applied at evaluation time.
    """

    # Model settings
    model_id: str = field(init=False, default="CompVis/stable-diffusion-v1-4")
    device: str = "cuda"
    use_fp16: bool = True

    # Concept erasure
    erase_concept: str = "nudity"

    # Training method
    train_method: str = "text_encoder_full"

    # Retention settings
    dataset_retain: str = "coco_object"
    retain_train: str = "iter"
    retain_batch: int = 5
    retain_step: int = 1
    retain_loss_w: float = 1.0

    # Unlearning loss settings
    start_guidance: float = 3.0
    negative_guidance: float = 1.0
    train_steps: int = 5
    learning_rate: float = 1e-5

    # Adversarial attack settings
    attack_method: str = "pgd"
    attack_step: int = 30
    attack_lr: float = 1e-3
    attack_type: str = "prefix_k"
    attack_init: str = "latest"
    attack_embd_type: str = "word_embd"
    adv_prompt_num: int = 1
    adv_prompt_update_step: int = 1
    warmup_iter: int = 1

    # Model component selection
    component: str = "all"
    norm_layer: bool = False

    # Training resolution / DDIM settings
    ddim_steps: int = 50
    save_interval: int = 1

    # Misc
    save_dir: Optional[str] = None
    load_path: Optional[str] = None
    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvUnlearnConfig":
        """Create an ``AdvUnlearnConfig`` from a plain dictionary, with enum validation.

        Validates all string fields that must belong to a fixed set of allowed values
        (``train_method``, ``dataset_retain``, ``retain_train``, ``attack_method``,
        ``attack_type``, ``attack_embd_type``, ``component``) before delegating to the
        base-class constructor.  Unrecognised keys are silently ignored.

        Args:
            data: Mapping of field names to values.

        Returns:
            A new frozen ``AdvUnlearnConfig`` instance.

        Raises:
            ValueError: If any string enum field contains an unrecognised value.
        """
        data = dict(data)

        train_method = data.get("train_method", "text_encoder_full")
        if train_method not in TRAIN_METHODS and not _TRAIN_METHOD_LAYER_RE.match(train_method):
            raise ValueError(
                f"Unknown train_method '{train_method}'. "
                f"Available: {TRAIN_METHODS} or 'text_encoder_layer<digits>' e.g. 'text_encoder_layer012_910'"
            )

        dataset_retain = data.get("dataset_retain", "coco_object")
        if dataset_retain not in RETAIN_DATASETS:
            raise ValueError(
                f"Unknown dataset_retain '{dataset_retain}'. Available: {RETAIN_DATASETS}"
            )

        retain_train = data.get("retain_train", "iter")
        if retain_train not in RETAIN_TRAIN_METHODS:
            raise ValueError(
                f"Unknown retain_train '{retain_train}'. Available: {RETAIN_TRAIN_METHODS}"
            )

        attack_method = data.get("attack_method", "pgd")
        if attack_method not in ATTACK_METHODS:
            raise ValueError(
                f"Unknown attack_method '{attack_method}'. Available: {ATTACK_METHODS}"
            )

        attack_type = data.get("attack_type", "prefix_k")
        if attack_type not in ATTACK_TYPES:
            raise ValueError(
                f"Unknown attack_type '{attack_type}'. Available: {ATTACK_TYPES}"
            )

        attack_embd_type = data.get("attack_embd_type", "word_embd")
        if attack_embd_type not in ATTACK_EMBD_TYPES:
            raise ValueError(
                f"Unknown attack_embd_type '{attack_embd_type}'. Available: {ATTACK_EMBD_TYPES}"
            )

        component = data.get("component", "all")
        if component not in COMPONENTS:
            raise ValueError(
                f"Unknown component '{component}'. Available: {COMPONENTS}"
            )

        return super().from_dict(data)

    def __post_init__(self):
        """Validate numeric field values after dataclass initialisation.

        Raises:
            ValueError: If any of the following constraints are violated:

                - ``train_steps``, ``attack_step``, ``retain_batch``, ``retain_step``,
                  ``adv_prompt_num`` must all be > 0.
                - ``learning_rate`` and ``attack_lr`` must both be > 0.
                - ``warmup_iter`` must be strictly less than ``train_steps`` (at least
                  one adversarial-training iteration must occur after warm-up).
        """
        if self.train_steps <= 0:
            raise ValueError(f"train_steps must be > 0, got {self.train_steps}")
        if self.attack_step <= 0:
            raise ValueError(f"attack_step must be > 0, got {self.attack_step}")
        if self.retain_batch <= 0:
            raise ValueError(f"retain_batch must be > 0, got {self.retain_batch}")
        if self.retain_step <= 0:
            raise ValueError(f"retain_step must be > 0, got {self.retain_step}")
        if self.adv_prompt_num <= 0:
            raise ValueError(f"adv_prompt_num must be > 0, got {self.adv_prompt_num}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.attack_lr <= 0:
            raise ValueError(f"attack_lr must be > 0, got {self.attack_lr}")
        if self.warmup_iter >= self.train_steps:
            raise ValueError(
                f"warmup_iter must be < train_steps, got warmup_iter={self.warmup_iter}, train_steps={self.train_steps}"
            )
        if not self.erase_concept or not self.erase_concept.strip():
            raise ValueError("erase_concept must not be empty.")
