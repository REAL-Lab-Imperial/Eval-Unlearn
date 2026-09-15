import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..types import MetricResult
from ..logging_utils import get_logger
from ..registry import get_technique, get_metric
from ..registry.entrypoints import load_entrypoints
from .core.base_runner import BaseRunner
from .validation import validate_technique_metric_pair, ValidationError, get_erase_concept

logger = get_logger(__name__)


def generate_multi_run_id(
    technique_name: str,
    technique_config: Dict[str, Any],
    metric_names: List[str],
    metric_configs: Dict[str, Dict[str, Any]],
    dataset_name: str,
    timestamp: float,
) -> str:
    """Generate a short hash from multi-benchmark run configuration."""
    payload = json.dumps(
        {
            "technique": technique_name,
            "technique_config": technique_config,
            "metric_names": sorted(metric_names),
            "metric_configs": metric_configs,
            "dataset": dataset_name,
            "timestamp": timestamp,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


class MultiBenchmarkRunner(BaseRunner):
    """
    Single technique x multiple metrics benchmark runner.

    All metrics must implement the DataLoader interface:
      - load_dataset() -> DataLoader
      - update(images, prompts, metadata)
      - compute() -> MetricResult

    Each metric drives its own image generation with its own dataset.
    This ensures each metric evaluates on the correct dataset for its purpose.
    """

    def __init__(
        self,
        technique_name: str,
        metric_names: List[str],
        technique_config: Optional[Dict[str, Any]] = None,
        metric_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        output_dir: str = "results",
        seed: Optional[int] = None,
    ):
        super().__init__(output_dir)

        self.technique_name = technique_name
        self.metric_names = metric_names
        self.technique_config = technique_config or {}
        self.metric_configs = metric_configs or {}
        self.seed = seed

        load_entrypoints()
        self._validate()

    def _validate(self):
        """Validate technique and all metric combinations."""
        # 1. Basic checks
        if not self.metric_names:
            raise ValueError("metric_names must not be empty.")
        if len(set(self.metric_names)) != len(self.metric_names):
            raise ValueError("metric_names contains duplicates.")

        # 2. Check factories exist
        self.technique_factory = get_technique(self.technique_name)
        self.metric_factories = {}
        for name in self.metric_names:
            self.metric_factories[name] = get_metric(name)

        # 3. Validate each metric against the technique
        for metric_name in self.metric_names:
            metric_config = self.metric_configs.get(metric_name, {})
            try:
                validate_technique_metric_pair(
                    technique_name=self.technique_name,
                    technique_config=self.technique_config,
                    metric_name=metric_name,
                    metric_config=metric_config,
                )
            except ValidationError as e:
                logger.error(
                    f"Incompatible metric '{metric_name}' for technique '{self.technique_name}': {e}"
                )
                raise ValueError(str(e))

    def run(self) -> Dict[str, Any]:
        """Execute single technique x multiple metrics benchmark.

        Each metric drives its own generation pass with its own dataset to ensure
        correct evaluation for that metric's specific purpose.
        """
        logger.info("Starting Multi-Benchmark Run...")
        timestamp = time.time()

        # 1. Initialize technique once
        self._log_phase("Initializing technique")
        technique = self.technique_factory(**self.technique_config)

        # Generate run_id early (needed for artifact saving)
        run_id = generate_multi_run_id(
            technique_name=self.technique_name,
            technique_config=self.technique_config,
            metric_names=self.metric_names,
            metric_configs=self.metric_configs,
            dataset_name="multi_metric",
            timestamp=timestamp,
        )
        logger.info(f"Run ID: {run_id}")

        # 2. Run each metric with its own dataset and generation pass.
        # Metrics are initialised one at a time and freed after use to avoid
        # exhausting GPU memory when multiple CLIP/detector models are loaded.
        self._log_phase("Generating images and computing metrics")
        metric_results: Dict[str, Any] = {}
        metric_datasets: Dict[str, str] = {}

        resolved_metric_configs = self._resolve_mma_clip_model(
            self.metric_configs,
            self.technique_name,
            self.technique_config,
        )

        for metric_name in self.metric_names:
            self._log_phase(f"Running metric '{metric_name}' with its dataset")
            config = resolved_metric_configs.get(metric_name, {})
            metric = self.metric_factories[metric_name](**config)

            # Load this metric's dataset
            loader = metric.load_dataset()
            logger.info(f"Loaded dataset for metric '{metric_name}'")

            # Generate, evaluate, and flush images batch by batch
            metric_metadata: Dict[str, Any] = {}
            total_generated = 0
            category_counters: Dict[str, int] = {}

            for batch in loader:
                batch_images = technique.generate(prompts=batch.prompts, seed=self.seed)
                metric.update(batch_images, batch.prompts, batch.metadata)

                self.writer.save_run(
                    run_id=run_id,
                    technique_name=self.technique_name,
                    metric_name=metric_name,
                    images=batch_images,
                    report=None,
                    metadata=batch.metadata,
                    image_index_offset=total_generated,
                    category_counters_init=category_counters,
                )

                if "categories" in batch.metadata:
                    for cat in batch.metadata["categories"]:
                        category_counters[cat.lower()] = category_counters.get(cat.lower(), 0) + 1

                total_generated += len(batch_images)

                for key, val in batch.metadata.items():
                    if isinstance(val, list):
                        metric_metadata.setdefault(key, []).extend(val)
                    else:
                        metric_metadata[key] = val

            dataset_source = metric_metadata.get("source", "unknown")
            logger.info(
                f"Generated and evaluated {total_generated} images for metric '{metric_name}' "
                f"from dataset '{dataset_source}'"
            )

            # Finalize this metric
            self._log_phase(f"Finalising metric '{metric_name}'")
            result: MetricResult = metric.compute()
            metric_results[metric_name] = {
                "name": result.name,
                "value": result.value,
                "details": result.details,
            }
            logger.info(f"Metric Result ({result.name}): {result.value}")

            # Track dataset source for report
            metric_datasets[metric_name] = dataset_source

            # Free metric and its GPU models before loading the next metric
            del metric
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Free technique pipeline from VRAM now that all generation is done
        del technique
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 4. Build final reports
        erase_concept = get_erase_concept(self.technique_name, self.technique_config)
        base_fields = dict(
            run_id=run_id,
            timestamp=timestamp,
            technique_name=self.technique_name,
            erase_concept=erase_concept,
            metric_names=self.metric_names,
            metric_results=metric_results,
        )
        report = self._build_base_report(**base_fields)
        detailed_report = self._build_base_report(
            **base_fields,
            technique_config=self.technique_config,
            metric_configs=self.metric_configs,
        )

        # 5. Save final reports to run directory
        self._log_phase("Saving final combined report")
        self.writer.save_run(
            run_id=run_id,
            technique_name=self.technique_name,
            metric_name="multi",
            images=[],
            report=report,
            metadata={},
            detailed_report=detailed_report,
        )

        logger.info("Multi-benchmark run completed.")
        return report
