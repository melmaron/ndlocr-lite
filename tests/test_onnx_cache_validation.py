"""Cache-validation tests for dynamic-batch PARSeq ONNX models.

A cache file truncated exactly at the trailing ``opset_import`` field is still a
well-formed protobuf (``onnx.load`` succeeds, the graph is intact), so it passes
any purely "does protobuf parse" check. ONNX Runtime however refuses it with
``Missing opset in the model. All ModelProtos MUST have at least one
opset_import``, which aborts OCR because ``create_inference_session`` does not
catch that error. The predicate must therefore reject it before it is reused.
"""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import onnx_batch_patcher as patcher

try:
    import onnx
except ImportError:  # pragma: no cover - onnx is a hard runtime dependency
    onnx = None

BUNDLED_MODEL = ROOT / "src" / "model" / "parseq-ndl-24x256-30-tiny-499epoch-261124.onnx"


class CacheValidationTestBase(unittest.TestCase):
    """Share one patched cache model across the read-only validation tests."""

    @classmethod
    def setUpClass(cls):
        if onnx is None:
            raise unittest.SkipTest("onnx package is required to build the cache fixtures")
        if not BUNDLED_MODEL.is_file():
            raise unittest.SkipTest(f"Bundled model missing: {BUNDLED_MODEL}")
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.cache_dir = Path(cls._tmpdir.name)
        cls.patched = patcher.prepare_parseq_model(
            BUNDLED_MODEL,
            enable_dynamic_batch=True,
            precision="fp32",
            cache_dir=cls.cache_dir,
            verify=False,
        )
        if not cls.patched.dynamic_batch or not cls.patched.model_path.is_file():
            raise unittest.SkipTest(
                f"Dynamic-batch patching unavailable: {cls.patched.fallback_reason}"
            )
        # Same model with the trailing opset_import field cleared: protobuf-valid,
        # graph intact, but unusable by ONNX Runtime.
        model = onnx.load(str(cls.patched.model_path), load_external_data=False)
        del model.opset_import[:]
        cls.cleared_path = cls.cache_dir / "cleared_opset_import.onnx"
        onnx.save(model, str(cls.cleared_path))

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()


@unittest.skipUnless(onnx is not None, "onnx package is required")
class DynamicBatchCacheValidationTest(CacheValidationTestBase):
    def test_patched_cache_is_accepted(self):
        cache_path = Path(self.patched.model_path)
        self.assertGreater(cache_path.stat().st_size, 0)
        self.assertTrue(patcher.is_valid_dynamic_batch_cache(cache_path))

    def test_cache_without_opset_import_is_rejected(self):
        # Sanity: the fixture really is a readable protobuf with an intact graph,
        # i.e. nothing but the trailing opset_import field is missing.
        reloaded = onnx.load(str(self.cleared_path), load_external_data=False)
        self.assertEqual(len(reloaded.opset_import), 0)
        self.assertTrue(reloaded.graph.node)
        self.assertTrue(reloaded.graph.input)

        self.assertFalse(patcher.is_valid_dynamic_batch_cache(self.cleared_path))

    def test_tail_truncated_cache_is_rejected(self):
        # Reproduces the measured failure: a copy interrupted at the end of the
        # file loses exactly the trailing opset_import bytes. Cutting those bytes
        # off the healthy cache yields a file identical to the cleared fixture.
        full = Path(self.patched.model_path).read_bytes()
        cleared = self.cleared_path.read_bytes()
        if not full.startswith(cleared):
            self.skipTest("opset_import is not the trailing proto field in this onnx build")
        self.assertGreater(len(full), len(cleared))

        truncated = self.cache_dir / "truncated_tail.onnx"
        truncated.write_bytes(full[: len(cleared)])
        self.assertEqual(len(truncated.read_bytes()), len(cleared))
        # It still parses as protobuf, which is exactly why the old check passed.
        self.assertEqual(len(onnx.load(str(truncated)).opset_import), 0)

        self.assertFalse(patcher.is_valid_dynamic_batch_cache(truncated))

    def test_empty_and_garbage_files_are_rejected(self):
        empty = self.cache_dir / "empty.onnx"
        empty.write_bytes(b"")
        self.assertFalse(patcher.is_valid_dynamic_batch_cache(empty))

        junk = self.cache_dir / "junk.onnx"
        junk.write_bytes(b"not-an-onnx-model")
        self.assertFalse(patcher.is_valid_dynamic_batch_cache(junk))

    def test_onnxruntime_rejects_cache_without_opset_import(self):
        # Documents the user-visible consequence the predicate guards against.
        import onnxruntime as ort

        with self.assertRaises(Exception) as raised:
            ort.InferenceSession(
                str(self.cleared_path), providers=["CPUExecutionProvider"]
            )
        self.assertIn("opset", str(raised.exception).lower())


@unittest.skipUnless(onnx is not None, "onnx package is required")
class CorruptCacheRebuildTest(CacheValidationTestBase):
    def test_prepare_parseq_model_rebuilds_opset_less_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            first = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            self.assertFalse(first.from_cache)
            cache_path = Path(first.model_path)
            self.assertTrue(cache_path.is_file())

            # Simulate a cache whose tail was lost while keeping every other byte.
            cache_path.write_bytes(self.cleared_path.read_bytes())
            self.assertFalse(patcher.is_valid_dynamic_batch_cache(cache_path))

            rebuilt = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            self.assertFalse(rebuilt.from_cache)
            self.assertTrue(rebuilt.dynamic_batch)
            self.assertEqual(Path(rebuilt.model_path), cache_path)
            self.assertTrue(patcher.is_valid_dynamic_batch_cache(cache_path))
            self.assertEqual(len(onnx.load(str(cache_path)).opset_import), 1)


if __name__ == "__main__":
    unittest.main()
