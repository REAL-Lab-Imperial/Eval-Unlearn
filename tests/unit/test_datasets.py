"""Unit tests for dataset loaders — all HuggingFace calls are mocked."""
import csv
import io
import os
import pytest
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader

from eval_learn.types import Dataset


# ---------------------------------------------------------------------------
# hf_stream.load_hf_config
# ---------------------------------------------------------------------------
class TestLoadHFConfig:
    def test_loads_known_key(self):
        from eval_learn.datasets.hf_stream import load_hf_config
        cfg = load_hf_config("i2p")
        assert "repo_id" in cfg
        assert "caption_col" in cfg

    def test_unknown_key_raises(self):
        from eval_learn.datasets.hf_stream import load_hf_config
        with pytest.raises(KeyError, match="not found"):
            load_hf_config("nonexistent_dataset_xyz")

    def test_all_configured_datasets(self):
        from eval_learn.datasets.hf_stream import load_hf_config
        for key in ("i2p", "tifa", "coco", "err_challenge", "ring_a_bell"):
            cfg = load_hf_config(key)
            assert cfg

    def test_missing_yaml_raises(self, tmp_path, monkeypatch):
        from eval_learn.datasets import hf_stream
        monkeypatch.setattr(
            hf_stream,
            "Path",
            lambda *a, **kw: tmp_path / "no_such_file.yaml",
        )
        with pytest.raises(Exception):
            hf_stream.load_hf_config("i2p")


# ---------------------------------------------------------------------------
# i2p_csv.load_i2p_csv
# ---------------------------------------------------------------------------
class TestLoadI2PCSV:
    def _make_hf_ds(self, rows):
        ds = MagicMock()
        ds.filter.return_value = ds
        ds.select.return_value = ds
        ds.__getitem__ = lambda self, col: [r[col] for r in rows]
        return ds

    def test_returns_dataloader(self):
        rows = [{"prompt": "a naked person", "categories": "sexual"}]
        mock_ds = self._make_hf_ds(rows)
        with patch("eval_learn.datasets.i2p_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.i2p_csv import load_i2p_csv
            loader = load_i2p_csv(concept="nudity", limit=1)
        assert isinstance(loader, DataLoader)

    def test_invalid_concept_raises(self):
        from eval_learn.datasets.i2p_csv import load_i2p_csv
        with pytest.raises(ValueError, match="No I2P category mapping"):
            load_i2p_csv(concept="banana")

    def test_no_concept_loads_all(self):
        rows = [{"prompt": "general", "categories": "violence"}]
        mock_ds = self._make_hf_ds(rows)
        with patch("eval_learn.datasets.i2p_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.i2p_csv import load_i2p_csv
            loader = load_i2p_csv(concept=None, limit=1)
        assert isinstance(loader, DataLoader)
        # filter should NOT be called when concept=None
        mock_ds.filter.assert_not_called()

    def test_collate_fn_produces_dataset(self):
        rows = [{"prompt": "a prompt", "categories": "sexual"}]
        mock_ds = self._make_hf_ds(rows)
        with patch("eval_learn.datasets.i2p_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.i2p_csv import load_i2p_csv
            loader = load_i2p_csv(concept="nudity", limit=1)
        batch = next(iter(loader))
        assert isinstance(batch, Dataset)
        assert "source" in batch.metadata


# ---------------------------------------------------------------------------
# ua_ira_csv.load_ua_ira_csv
# ---------------------------------------------------------------------------
class TestLoadUAIRACSV:
    def _write_csv(self, path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["prompt"])
            writer.writeheader()
            writer.writerows(rows)

    def test_basic_load(self, tmp_path):
        target = tmp_path / "target.csv"
        retain = tmp_path / "retain.csv"
        self._write_csv(target, [{"prompt": "nudity prompt"}])
        self._write_csv(retain, [{"prompt": "clothed person"}])

        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        loader = load_ua_ira_csv(str(target), str(retain))
        batch = next(iter(loader))
        assert isinstance(batch, Dataset)
        assert "target_prompt_end_index" in batch.metadata
        assert batch.metadata["target_prompt_end_index"] == 1

    def test_limit_target(self, tmp_path):
        target = tmp_path / "target.csv"
        retain = tmp_path / "retain.csv"
        self._write_csv(target, [{"prompt": f"p{i}"} for i in range(10)])
        self._write_csv(retain, [{"prompt": "keep"}])

        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        loader = load_ua_ira_csv(str(target), str(retain), target_limit=3)
        all_prompts = []
        for batch in loader:
            all_prompts.extend(batch.prompts)
        # 3 target + 1 retain
        assert len(all_prompts) == 4

    def test_missing_target_file_raises(self, tmp_path):
        retain = tmp_path / "retain.csv"
        self._write_csv(retain, [{"prompt": "keep"}])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        with pytest.raises(FileNotFoundError, match="Target prompts file not found"):
            load_ua_ira_csv("/no/such/file.csv", str(retain))

    def test_missing_retain_file_raises(self, tmp_path):
        target = tmp_path / "target.csv"
        self._write_csv(target, [{"prompt": "erase me"}])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        with pytest.raises(FileNotFoundError, match="Retain prompts file not found"):
            load_ua_ira_csv(str(target), "/no/such/file.csv")

    def test_empty_files_raises(self, tmp_path):
        target = tmp_path / "target.csv"
        retain = tmp_path / "retain.csv"
        self._write_csv(target, [])
        self._write_csv(retain, [])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        with pytest.raises(ValueError, match="No prompts loaded"):
            load_ua_ira_csv(str(target), str(retain))

    def test_categories_in_metadata(self, tmp_path):
        target = tmp_path / "target.csv"
        retain = tmp_path / "retain.csv"
        self._write_csv(target, [{"prompt": "nude"}])
        self._write_csv(retain, [{"prompt": "clothed"}])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        loader = load_ua_ira_csv(str(target), str(retain), batch_size=10)
        batch = next(iter(loader))
        assert "target" in batch.metadata["categories"]
        assert "retain" in batch.metadata["categories"]


# ---------------------------------------------------------------------------
# tifa_csv.load_tifa_csv
# ---------------------------------------------------------------------------
class TestLoadTIFACSV:
    def _make_row(self, caption="a dog", qas=None):
        import json
        return {"caption": caption, "qas": json.dumps(qas or [{"question": "Is there a dog?", "answer": "yes"}])}

    def test_returns_dataloader(self):
        import json
        rows = [self._make_row()]
        mock_ds = MagicMock()
        mock_ds.take.return_value = iter(rows)
        mock_ds.__iter__ = lambda self: iter(rows)

        with patch("eval_learn.datasets.tifa_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.tifa_csv import load_tifa_csv
            loader = load_tifa_csv(limit=1)
        assert isinstance(loader, DataLoader)

    def test_collate_includes_qa_pairs(self):
        import json
        rows = [{"caption": "a cat", "qas": json.dumps([{"question": "q?", "answer": "a"}])}]
        mock_ds = MagicMock()
        mock_ds.take.return_value = iter(rows)
        mock_ds.__iter__ = lambda self: iter(rows)

        with patch("eval_learn.datasets.tifa_csv.hf_load_dataset", return_value=mock_ds):
            from eval_learn.datasets.tifa_csv import load_tifa_csv
            loader = load_tifa_csv(limit=1)
        batch = next(iter(loader))
        assert "qa_pairs" in batch.metadata
        assert isinstance(batch.metadata["qa_pairs"][0], list)


# ---------------------------------------------------------------------------
# err_composite.load_err_composite
# ---------------------------------------------------------------------------
class TestLoadERRComposite:
    def _mock_hf_dataset(self, rows):
        ds = MagicMock()
        ds.filter.return_value = iter(rows)
        ds.take.return_value = iter(rows)
        ds.__iter__ = lambda self: iter(rows)
        return ds

    def test_returns_dataloader(self):
        i2p_row = {"prompt": "nude", "categories": "sexual"}
        ch_row = {"direct_prompt": "a dog", "concept_name": "dog"}
        rab_row = {"prompt": "adversarial", "concept": "nudity"}

        call_count = [0]
        def fake_load(repo, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return self._mock_hf_dataset([i2p_row])
            elif call_count[0] == 2:
                return self._mock_hf_dataset([ch_row])
            else:
                return self._mock_hf_dataset([rab_row])

        with patch("eval_learn.datasets.err_composite.hf_load_dataset", side_effect=fake_load):
            from eval_learn.datasets.err_composite import load_err_composite
            loader = load_err_composite(target_limit=1, retain_limit=1, adversarial_limit=1)
        assert isinstance(loader, DataLoader)

    def test_composite_iterable_yields_all_categories(self):
        from eval_learn.datasets.err_composite import _ERRCompositeIterableDataset

        i2p_rows = [{"prompt": "nude", "categories": "sexual"}]
        ch_rows = [{"direct_prompt": "dog", "concept_name": "dog"}]
        rab_rows = [{"prompt": "adversarial", "concept": "nudity"}]

        i2p_cfg = {"caption_col": "prompt", "concept_col": "categories"}
        ch_cfg = {"caption_col": "direct_prompt", "concept_col": "concept_name"}
        rab_cfg = {"caption_col": "prompt", "concept_col": "concept"}

        ds = _ERRCompositeIterableDataset(
            i2p_rows, i2p_cfg, ch_rows, ch_cfg, rab_rows, rab_cfg
        )
        items = list(ds)
        assert len(items) == 3
        categories = [item[2] for item in items]
        assert "target" in categories
        assert "retain" in categories
        assert "adversarial" in categories


# ---------------------------------------------------------------------------
# coco_parquet.load_coco_parquet
# ---------------------------------------------------------------------------
class TestLoadCOCOParquet:
    def test_returns_dataloader(self):
        import torch
        from PIL import Image as PILImage

        fake_img = PILImage.new("RGB", (100, 100))
        buf = io.BytesIO()
        fake_img.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()

        fake_response = MagicMock()
        fake_response.content = img_bytes
        fake_response.raise_for_status.return_value = None

        rows = [{"captions": ["a dog in the park"], "coco_url": "http://fake/img.jpg"}]
        mock_ds = MagicMock()
        mock_ds.take.return_value = iter(rows)
        mock_ds.__iter__ = lambda self: iter(rows)

        with patch("eval_learn.datasets.coco_parquet.hf_load_dataset", return_value=mock_ds), \
             patch("eval_learn.datasets.coco_parquet.requests.get", return_value=fake_response):
            from eval_learn.datasets.coco_parquet import load_coco_parquet
            loader = load_coco_parquet(limit=1)
        assert isinstance(loader, DataLoader)

    def test_failed_image_download_returns_empty_batch(self):
        import torch
        rows = [{"captions": ["a cat"], "coco_url": "http://bad/url"}]
        mock_ds = MagicMock()
        mock_ds.take.return_value = iter(rows)
        mock_ds.__iter__ = lambda self: iter(rows)

        with patch("eval_learn.datasets.coco_parquet.hf_load_dataset", return_value=mock_ds), \
             patch("eval_learn.datasets.coco_parquet.requests.get", side_effect=Exception("timeout")):
            from eval_learn.datasets.coco_parquet import load_coco_parquet
            loader = load_coco_parquet(limit=1)
        batch = next(iter(loader))
        assert batch.prompts == []
        assert batch.metadata["images"].shape[0] == 0


# ---------------------------------------------------------------------------
# coco_parquet — list caption + no-limit path
# ---------------------------------------------------------------------------
class TestCOCOParquetCoverageGaps:
    def _fake_response(self):
        from PIL import Image as _Image
        buf = io.BytesIO()
        _Image.new("RGB", (100, 100)).save(buf, format="PNG")
        buf.seek(0)
        r = MagicMock()
        r.content = buf.read()
        r.raise_for_status.return_value = None
        return r

    def test_caption_list_takes_first(self):
        rows = [{"captions": ["first", "second"], "coco_url": "http://fake/img.jpg"}]
        mock_ds = MagicMock()
        mock_ds.take.return_value = iter(rows)
        with patch("eval_learn.datasets.coco_parquet.hf_load_dataset", return_value=mock_ds), \
             patch("eval_learn.datasets.coco_parquet.requests.get",
                   return_value=self._fake_response()):
            from eval_learn.datasets.coco_parquet import load_coco_parquet
            loader = load_coco_parquet(limit=1)
            batch = next(iter(loader))       # must iterate inside patch context
        assert batch.prompts == ["first"]
        assert batch.metadata["images"].shape[0] == 1

    def test_no_limit_uses_full_dataset(self):
        rows = [{"captions": "a dog", "coco_url": "http://fake/img.jpg"}]
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter(rows)
        with patch("eval_learn.datasets.coco_parquet.hf_load_dataset", return_value=mock_ds), \
             patch("eval_learn.datasets.coco_parquet.requests.get",
                   return_value=self._fake_response()):
            from eval_learn.datasets.coco_parquet import load_coco_parquet
            loader = load_coco_parquet(limit=None)
        assert isinstance(loader, DataLoader)


# ---------------------------------------------------------------------------
# err_composite — data_files branches
# ---------------------------------------------------------------------------
class TestERRCompositeCoverageGaps:
    def test_with_data_files_in_config(self):
        cfg = {
            "i2p": {"repo_id": "f/i2p", "split": "train",
                    "caption_col": "prompt", "concept_col": "categories",
                    "data_files": "d.csv"},
            "err_challenge": {"repo_id": "f/ch", "split": "train",
                              "caption_col": "direct_prompt", "concept_col": "concept_name",
                              "data_files": "d.csv"},
            "ring_a_bell": {"repo_id": "f/rab", "split": "train",
                            "caption_col": "prompt", "concept_col": "concept",
                            "data_files": "d.csv"},
        }
        call_count = [0]
        rows_by_call = [
            [{"prompt": "nude", "categories": "sexual"}],
            [{"direct_prompt": "dog", "concept_name": "dog"}],
            [{"prompt": "adv", "concept": "nudity"}],
        ]

        def fake_load(repo, **kw):
            idx = call_count[0]; call_count[0] += 1
            ds = MagicMock()
            ds.filter.return_value = iter(rows_by_call[idx])
            ds.take.return_value = iter(rows_by_call[idx])
            ds.__iter__ = lambda s: iter(rows_by_call[idx])
            return ds

        with patch("eval_learn.datasets.err_composite.load_hf_config",
                   side_effect=lambda k: cfg[k]), \
             patch("eval_learn.datasets.err_composite.hf_load_dataset",
                   side_effect=fake_load):
            from eval_learn.datasets.err_composite import load_err_composite
            loader = load_err_composite(target_limit=1, retain_limit=1, adversarial_limit=1)
        assert isinstance(loader, DataLoader)

    def test_no_data_files_in_challenge_and_rab(self):
        cfg = {
            "i2p": {"repo_id": "f/i2p", "split": "train",
                    "caption_col": "prompt", "concept_col": "categories",
                    "data_files": "d.csv"},
            "err_challenge": {"repo_id": "f/ch", "split": "train",
                              "caption_col": "direct_prompt", "concept_col": "concept_name"},
            "ring_a_bell": {"repo_id": "f/rab", "split": "train",
                            "caption_col": "prompt", "concept_col": "concept"},
        }
        call_count = [0]
        rows_by_call = [
            [{"prompt": "nude", "categories": "sexual"}],
            [{"direct_prompt": "dog", "concept_name": "dog"}],
            [{"prompt": "adv", "concept": "nudity"}],
        ]

        def fake_load(repo, **kw):
            idx = call_count[0]; call_count[0] += 1
            rows = rows_by_call[idx]
            ds = MagicMock()
            ds.__iter__ = lambda s: iter(rows)
            ds.filter.return_value = iter(rows)
            return ds

        with patch("eval_learn.datasets.err_composite.load_hf_config",
                   side_effect=lambda k: cfg[k]), \
             patch("eval_learn.datasets.err_composite.hf_load_dataset",
                   side_effect=fake_load):
            from eval_learn.datasets.err_composite import load_err_composite
            loader = load_err_composite(target_limit=1, retain_limit=1, adversarial_limit=1)
        # Iterating triggers collate_fn, covering lines 181-182
        from eval_learn.types import Dataset as _Dataset
        batch = next(iter(loader))
        assert isinstance(batch, _Dataset)


# ---------------------------------------------------------------------------
# ua_ira_csv — fallback column + RuntimeError paths
# ---------------------------------------------------------------------------
class TestUAIRACSVCoverageGaps:
    def _write(self, path, header, rows):
        with open(path, "w", newline="") as f:
            f.write(header + "\n")
            for r in rows:
                f.write(r + "\n")

    def test_fallback_to_first_column(self, tmp_path):
        target = str(tmp_path / "target.csv")
        retain = str(tmp_path / "retain.csv")
        self._write(target, "text", ["erase me"])
        self._write(retain, "text", ["keep me"])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        batch = next(iter(load_ua_ira_csv(target, retain)))
        assert len(batch.prompts) > 0

    def test_target_generic_exception_raises(self, tmp_path):
        target = str(tmp_path / "target.csv")
        retain = str(tmp_path / "retain.csv")
        self._write(target, "prompt", ["nudity"])
        self._write(retain, "prompt", ["keep"])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        orig = open
        call_count = [0]

        def patched_open(p, *a, **kw):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("unexpected disk error")
            return orig(p, *a, **kw)

        with patch("builtins.open", side_effect=patched_open):
            with pytest.raises(RuntimeError):
                load_ua_ira_csv(target, retain)

    def test_retain_generic_exception_raises(self, tmp_path):
        target = str(tmp_path / "target.csv")
        retain = str(tmp_path / "retain.csv")
        self._write(target, "prompt", ["nudity"])
        with open(retain, "wb") as f:
            f.write(b"prompt\n\xff\xfe bad utf8\n")
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        with pytest.raises(Exception):
            load_ua_ira_csv(target, retain)

    def test_target_generic_exception_raises_runtime_error(self, tmp_path):
        target = str(tmp_path / "target.csv")
        retain = str(tmp_path / "retain.csv")
        self._write(target, "prompt", ["nudity"])
        self._write(retain, "prompt", ["keep"])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        orig = open
        def patched_open(p, *a, **kw):
            if str(p) == target:
                raise PermissionError("no read access")
            return orig(p, *a, **kw)
        with patch("builtins.open", side_effect=patched_open):
            with pytest.raises(RuntimeError, match="Error loading target prompts"):
                load_ua_ira_csv(target, retain)

    def test_retain_limit_breaks_early(self, tmp_path):
        target = str(tmp_path / "target.csv")
        retain = str(tmp_path / "retain.csv")
        self._write(target, "prompt", ["nudity"])
        self._write(retain, "prompt", ["keep1", "keep2", "keep3"])
        from eval_learn.datasets.ua_ira_csv import load_ua_ira_csv
        loader = load_ua_ira_csv(target, retain, retain_limit=1)
        batches = list(loader)
        retain_items = [p for b in batches for p, cat in zip(b.prompts, b.metadata["categories"]) if cat == "retain"]
        assert len(retain_items) == 1
