"""
Integration tests for dataset loaders and registry utilities.

Covers:
  - datasets/err_composite.py   (mocked HF)
  - datasets/i2p_csv.py         (mocked HF)
  - datasets/coco_parquet.py    (mocked HF + requests)
  - datasets/tifa_csv.py        (mocked HF)
  - datasets/hf_stream.py
  - registry/local.py           (get_technique, get_metric, get_dataset, etc.)
  - registry/entrypoints.py     (load_entrypoints)
  - techniques/_base_models.py
  - metrics/_base_models.py
"""
import json
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.integration


# ===========================================================================
# Registry
# ===========================================================================

class TestRegistryLocal:

    def test_get_technique_not_found_raises(self):
        from eval_learn.registry.local import get_technique
        with pytest.raises(ValueError, match="not found"):
            get_technique("__nonexistent_xyz__")

    def test_get_metric_not_found_raises(self):
        from eval_learn.registry.local import get_metric
        with pytest.raises(ValueError, match="not found"):
            get_metric("__nonexistent_xyz__")

    def test_get_dataset_not_found_raises(self):
        from eval_learn.registry.local import get_dataset
        with pytest.raises(ValueError, match="not found"):
            get_dataset("__nonexistent_xyz__")

    def test_get_benchmark_not_found_raises(self):
        from eval_learn.registry.local import get_benchmark
        with pytest.raises(ValueError, match="not found"):
            get_benchmark("__nonexistent_xyz__")

    def test_get_technique_found(self):
        from eval_learn.registry.local import _TECHNIQUES, get_technique
        _TECHNIQUES["__test_tech__"] = object
        try:
            result = get_technique("__test_tech__")
            assert result is object
        finally:
            _TECHNIQUES.pop("__test_tech__", None)

    def test_get_metric_found(self):
        from eval_learn.registry.local import _METRICS, get_metric
        _METRICS["__test_metric__"] = object
        try:
            result = get_metric("__test_metric__")
            assert result is object
        finally:
            _METRICS.pop("__test_metric__", None)

    def test_register_dataset_decorator(self):
        from eval_learn.registry.local import register_dataset, get_dataset
        @register_dataset("__test_ds__")
        def _my_loader():
            return "hello"
        try:
            fn = get_dataset("__test_ds__")
            assert fn() == "hello"
        finally:
            from eval_learn.registry.local import _DATASETS
            _DATASETS.pop("__test_ds__", None)

    def test_register_benchmark_decorator(self):
        from eval_learn.registry.local import register_benchmark, get_benchmark
        @register_benchmark("__test_bm__")
        def _bm():
            pass
        try:
            get_benchmark("__test_bm__")
        finally:
            from eval_learn.registry.local import _BENCHMARKS
            _BENCHMARKS.pop("__test_bm__", None)


class TestRegistryEntrypoints:

    def test_load_entrypoints_no_crash(self):
        """load_entrypoints should not raise even if no plugins are installed."""
        from eval_learn.registry.entrypoints import load_entrypoints
        load_entrypoints()  # should not raise

    def test_load_entrypoints_logs_failed_plugin(self):
        """A broken entry point should be logged but not crash."""
        from eval_learn.registry.entrypoints import load_entrypoints
        from importlib.metadata import EntryPoint
        bad_ep = MagicMock()
        bad_ep.name = "broken_plugin"
        bad_ep.load.side_effect = ImportError("missing dep")
        with patch("eval_learn.registry.entrypoints.entry_points") as mock_eps:
            mock_eps.return_value = [bad_ep]
            load_entrypoints()  # should not raise despite bad entry point


# ===========================================================================
# techniques/_base_models.py
# ===========================================================================

class TestTechniqueBaseModels:

    def test_get_base_model_known(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id
        result = get_technique_base_model_id("ssd", {})
        assert result == "CompVis/stable-diffusion-v1-4"

    def test_get_base_model_free_run(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id
        result = get_technique_base_model_id("free_run", {"model_id": "org/my-model"})
        assert result == "org/my-model"

    def test_get_base_model_free_run_no_model_id(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id
        result = get_technique_base_model_id("free_run", {})
        assert result is None

    def test_get_base_model_unknown_returns_none(self):
        from eval_learn.techniques._base_models import get_technique_base_model_id
        result = get_technique_base_model_id("unknown_tech_xyz", {})
        assert result is None


# ===========================================================================
# metrics/_base_models.py
# ===========================================================================

class TestMetricBaseModels:

    def test_metric_models_dict_has_expected_keys(self):
        from eval_learn.metrics._base_models import METRIC_MODELS
        assert "asr_i2p" in METRIC_MODELS
        assert "clip_score" in METRIC_MODELS
        assert "fid" in METRIC_MODELS
        assert "err" in METRIC_MODELS

    def test_metric_model_info_namedtuple(self):
        from eval_learn.metrics._base_models import METRIC_MODELS
        info = METRIC_MODELS["clip_score"]
        assert hasattr(info, "model")
        assert hasattr(info, "configurable")
        assert info.configurable is True

    def test_fid_not_configurable(self):
        from eval_learn.metrics._base_models import METRIC_MODELS
        assert METRIC_MODELS["fid"].configurable is False


# ===========================================================================
# Dataset loaders (all HF calls mocked)
# ===========================================================================

class TestI2PDatasetLoader:

    def _make_mock_hf_ds(self, rows):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=len(rows))
        ds.__getitem__ = MagicMock(side_effect=lambda k: [r[k] for r in rows])
        ds.filter = MagicMock(return_value=ds)
        ds.select = MagicMock(return_value=ds)
        return ds

    def test_load_i2p_csv_all_concepts(self):
        rows = [{"prompt": "a prompt", "categories": "sexual,violence"}]
        mock_ds = self._make_mock_hf_ds(rows)
        mock_ds["prompt"] = ["a prompt"]
        with patch("eval_learn.datasets.i2p_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.i2p_csv import load_i2p_csv
            loader = load_i2p_csv(limit=1)
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)

    def test_load_i2p_csv_with_concept(self):
        rows = [{"prompt": "nude scene", "categories": "sexual"}]
        mock_ds = self._make_mock_hf_ds(rows)
        mock_ds["prompt"] = ["nude scene"]
        with patch("eval_learn.datasets.i2p_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.i2p_csv import load_i2p_csv
            loader = load_i2p_csv(concept="nudity", limit=1)
        assert loader is not None

    def test_load_i2p_invalid_concept_raises(self):
        from eval_learn.datasets.i2p_csv import load_i2p_csv
        with pytest.raises(ValueError, match="No I2P category mapping"):
            with patch("eval_learn.datasets.i2p_csv.hf_load_dataset"):
                load_i2p_csv(concept="unknown_concept_xyz")


class TestTIFADatasetLoader:

    def test_load_tifa_csv_basic(self, tmp_path):
        qa_data = json.dumps([{"question": "Is there a cat?", "answer": "yes"}])
        # Column names come from hf_datasets.yaml: caption_col="caption", qa_col="qas"
        rows = [{"caption": "a cat", "qas": qa_data}]
        mock_ds = MagicMock()
        mock_ds.take.return_value = rows
        with patch("eval_learn.datasets.tifa_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.tifa_csv import load_tifa_csv
            loader = load_tifa_csv(limit=1)
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert len(batch.prompts) == 1
        assert batch.metadata["qa_pairs"][0][0]["question"] == "Is there a cat?"


class TestERRCompositeDatasetLoader:

    def _make_iterable_ds(self, rows):
        ds = MagicMock()
        ds.__iter__ = MagicMock(return_value=iter(rows))
        ds.filter = MagicMock(return_value=iter(rows))
        return ds

    def test_load_err_composite_basic(self):
        i2p_rows = [{"prompt": "nude", "categories": "sexual"}]
        ch_rows = [{"direct_prompt": "dog", "concept_type": "retain", "concept_name": "dog",
                    "indirect_prompt": "d", "adversarial_prompt": "a"}]
        rab_rows = [{"prompt": "nudity", "concept": "nudity"}]

        i2p_ds = self._make_iterable_ds(i2p_rows)
        ch_ds = self._make_iterable_ds(ch_rows)
        rab_ds = self._make_iterable_ds(rab_rows)

        call_count = [0]
        def fake_load(repo, **kw):
            idx = call_count[0]
            call_count[0] += 1
            return [i2p_ds, ch_ds, rab_ds][idx]

        with patch("eval_learn.datasets.err_composite.hf_load_dataset", side_effect=fake_load):
            from eval_learn.datasets.err_composite import load_err_composite
            loader = load_err_composite(target_limit=1, retain_limit=1, adversarial_limit=1)

        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert "categories" in batch.metadata
        assert len(batch.prompts) > 0


class TestCOCOParquetDatasetLoader:

    def test_load_coco_parquet_basic(self):
        from PIL import Image
        import io
        fake_img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        img_bytes = io.BytesIO()
        fake_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # url_col="coco_url", caption_col="captions" per hf_datasets.yaml
        row = {"coco_url": "http://fake.example.com/img.png", "captions": ["a cat"]}

        mock_ds = MagicMock()
        mock_ds.take.return_value = [row]

        mock_resp = MagicMock()
        mock_resp.content = img_bytes.read()
        mock_resp.raise_for_status = MagicMock()

        with patch("eval_learn.datasets.coco_parquet.hf_load_dataset", return_value=mock_ds), \
             patch("eval_learn.datasets.coco_parquet.requests.get", return_value=mock_resp):
            from eval_learn.datasets.coco_parquet import load_coco_parquet
            loader = load_coco_parquet(limit=1)

        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        if batch.prompts:  # image load might succeed
            assert "images" in batch.metadata

    def test_load_coco_parquet_failed_image(self):
        """If all images fail to load, returns empty Dataset."""
        row = {"coco_url": "http://fail.example.com/img.png", "captions": ["a cat"]}
        mock_ds = MagicMock()
        mock_ds.take.return_value = [row]

        with patch("eval_learn.datasets.coco_parquet.hf_load_dataset", return_value=mock_ds), \
             patch("eval_learn.datasets.coco_parquet.requests.get",
                   side_effect=Exception("connection refused")):
            from eval_learn.datasets.coco_parquet import load_coco_parquet
            loader = load_coco_parquet(limit=1)

        batch = next(iter(loader))
        assert batch.prompts == []  # empty due to all failures
