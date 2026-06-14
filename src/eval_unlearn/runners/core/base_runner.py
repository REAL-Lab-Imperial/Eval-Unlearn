from abc import ABC, abstractmethod
from typing import Any, Dict
from ...artifacts import ArtifactWriter
from ...logging_utils import get_logger

logger = get_logger(__name__)


class BaseRunner(ABC):
    """
    Abstract base class for all benchmark runners.

    Provides common infrastructure:
    - Artifact management (output directory, ArtifactWriter)
    - Report building with standard fields
    - Logging utilities

    Subclasses implement run() with their specific benchmark logic.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the base runner.

        Args:
            output_dir: Directory where artifacts will be saved.
        """
        self.output_dir = output_dir
        self.writer = ArtifactWriter(base_dir=output_dir)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the benchmark and return a report.

        Each runner type implements its own benchmark logic here.

        Returns:
            Report dictionary containing run results.
        """
        pass

    def _build_base_report(
        self, run_id: str, timestamp: float, **kwargs
    ) -> Dict[str, Any]:
        """
        Build a report dict with common base fields.

        Args:
            run_id: Unique identifier for this run.
            timestamp: Unix timestamp when the run started.
            **kwargs: Additional fields to include in the report.

        Returns:
            Report dict with run_id, timestamp, and any additional fields.
        """
        return {"run_id": run_id, "timestamp": timestamp, **kwargs}

    @staticmethod
    def _resolve_mma_clip_model(
        metric_configs: Dict[str, Dict[str, Any]],
        technique_name: str,
        technique_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Inject the correct CLIP text encoder into asr_mma_diffusion metric config.

        asr_mma_diffusion's GCG attack must use the same CLIP text encoder as the
        target diffusion model. This resolves that encoder from the technique's
        base model and injects it, overriding any user-supplied value.

        Returns a shallow copy of metric_configs with asr_mma_diffusion updated if present.
        """
        if "asr_mma_diffusion" not in metric_configs:
            return metric_configs

        from ...techniques._base_models import get_technique_base_model_id
        from ...metrics._clip_constants import clip_encoder_for_sd

        sd_model_id = get_technique_base_model_id(technique_name, technique_config)
        if sd_model_id is None:
            logger.warning(
                "asr_mma_diffusion: could not resolve base model for technique "
                f"'{technique_name}' — using default CLIP encoder."
            )
            return metric_configs

        try:
            encoder = clip_encoder_for_sd(sd_model_id)
        except ValueError as e:
            raise ValueError(
                f"asr_mma_diffusion cannot be used with technique '{technique_name}': {e}"
            )

        logger.info(
            f"asr_mma_diffusion: resolved CLIP text encoder '{encoder}' "
            f"from technique '{technique_name}' (base model: {sd_model_id})"
        )
        updated = dict(metric_configs)
        updated["asr_mma_diffusion"] = {**metric_configs.get("asr_mma_diffusion", {}), "clip_model_id": encoder}
        return updated

    def _log_phase(self, phase_name: str):
        """
        Log the start of a benchmark phase.

        Args:
            phase_name: Name of the phase (e.g., "Loading dataset", "Generating images").
        """
        logger.info(f"{phase_name}...")
