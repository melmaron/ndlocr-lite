"""Regression tests for TableCenterNet lazy loading and table HTML output hardening."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import tablecenternet
from tablecenternet import TableCenterNetRecognizer
from optional_features.table_structure import TableStructureFeature


class _FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="x", shape=[1, 3, 800, 800], type="tensor(float)")]


class TableCenterNetRobustnessTest(unittest.TestCase):
    def test_lazy_session_loading_is_thread_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "table.onnx"
            model_path.write_bytes(b"dummy")
            recognizer = TableCenterNetRecognizer(model_path, device="cpu")
            calls = []
            lock = threading.Lock()

            def fake_make_session(*args, **kwargs):
                with lock:
                    calls.append(1)
                return _FakeSession(), SimpleNamespace(active_backend="cpu")

            errors = []
            barrier = threading.Barrier(8)

            def worker():
                try:
                    barrier.wait(timeout=5)
                    recognizer._ensure_session()
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            with patch.object(tablecenternet, "make_session_with_profile", side_effect=fake_make_session):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

            self.assertEqual(errors, [])
            self.assertEqual(len(calls), 1)  # exactly one session creation
            self.assertIsNotNone(recognizer._session)
            self.assertEqual(recognizer._input_name, "x")
            self.assertEqual((recognizer._input_h, recognizer._input_w), (800, 800))
            self.assertEqual(recognizer._input_dtype, np.float32)

    def test_infer_tables_from_page_tolerates_invalid_page_xml(self):
        recognizer = TableCenterNetRecognizer("missing.onnx", device="cpu")
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        for page_xml in (None, "", "<not-xml"):
            with self.subTest(page_xml=page_xml):
                result, tables = tablecenternet.infer_tables_from_page(
                    image, {"page_xml": page_xml}, recognizer
                )
                self.assertEqual(tables, [])
                self.assertEqual(result["page_xml"], page_xml)

    def test_infer_tables_from_page_tolerates_malformed_page_result(self):
        recognizer = TableCenterNetRecognizer("missing.onnx", device="cpu")
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        result, tables = tablecenternet.infer_tables_from_page(image, {}, recognizer)
        self.assertEqual(tables, [])
        self.assertEqual(result, {})


class TableStructureFeatureTest(unittest.TestCase):
    def test_write_html_files_tolerates_none_inputs(self):
        feature = TableStructureFeature()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(feature.write_html_files("doc", None, tmpdir, "stem"), [])
            self.assertEqual(
                feature.write_html_files("doc", [{"cells": None, "table_index": 1}], tmpdir, "stem"),
                [],
            )

    def test_write_html_files_skips_write_errors(self):
        feature = TableStructureFeature()
        with tempfile.TemporaryDirectory() as tmpdir:
            blocker = Path(tmpdir) / "blocked"
            blocker.write_text("not a directory", encoding="utf-8")
            item = {"cells": [], "table_index": 1}
            with patch.object(feature.engine, "table_grid_shape", return_value=(2, 2)), patch.object(
                feature, "build_html", return_value="<html></html>"
            ):
                paths = feature.write_html_files("doc", [item], str(blocker), "stem")
            self.assertEqual(paths, [])
            self.assertNotIn("html_path", item)

    def test_write_html_files_records_html_path(self):
        feature = TableStructureFeature()
        with tempfile.TemporaryDirectory() as tmpdir:
            item = {"cells": [], "table_index": 3}
            with patch.object(feature.engine, "table_grid_shape", return_value=(2, 2)), patch.object(
                feature, "build_html", return_value="<html></html>"
            ):
                paths = feature.write_html_files("doc", [item], tmpdir, "stem")
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())
            self.assertEqual(item["html_path"], str(paths[0].resolve()))
            self.assertEqual(paths[0].name, "stem_table_00003.html")


if __name__ == "__main__":
    unittest.main()
