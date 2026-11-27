"""Cache-validation tests for dynamic-batch PARSeq ONNX models.

A cache file truncated exactly at the trailing ``opset_import`` field is still a
well-formed protobuf (``onnx.load`` succeeds, the graph is intact), so it passes
any purely "does protobuf parse" check. ONNX Runtime however refuses it with
``Missing opset in the model. All ModelProtos MUST have at least one
opset_import``, which aborts OCR because ``create_inference_session`` does not
catch that error. The predicate must therefore reject it before it is reused.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

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

    def test_sidecar_fast_path_skips_sha256_and_onnx_load(self):
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
            sidecar = patcher._build_sidecar_path(
                BUNDLED_MODEL,
                precision="fp32",
                dynamic_batch=True,
                cache_dir=cache_dir,
            )
            self.assertTrue(sidecar.is_file())
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertTrue(meta.get("src_blake2b"))
            self.assertTrue(meta.get("cache_blake2b"))

            with patch.object(
                patcher, "_file_sha256", side_effect=AssertionError("sha256 should be skipped")
            ), patch.object(
                onnx, "load", side_effect=AssertionError("onnx.load should be skipped")
            ):
                hit = patcher.prepare_parseq_model(
                    BUNDLED_MODEL,
                    enable_dynamic_batch=True,
                    precision="fp32",
                    cache_dir=cache_dir,
                    verify=False,
                )
            self.assertTrue(hit.from_cache)
            self.assertEqual(hit.model_path, first.model_path)

    def test_sidecar_rebuilds_on_same_size_same_mtime_midfile_and_header_corruption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            first = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            cache_path = Path(first.model_path)
            clean_bytes = cache_path.read_bytes()
            orig_stat = cache_path.stat()

            # 1. Corrupt mid-file weight bytes while keeping identical size and mtime_ns.
            mid_start = len(clean_bytes) // 3
            corrupted_mid = bytearray(clean_bytes)
            for idx in range(mid_start, min(len(corrupted_mid), mid_start + 4096)):
                corrupted_mid[idx] ^= 0xFF
            cache_path.write_bytes(bytes(corrupted_mid))
            os.utime(cache_path, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))
            self.assertEqual(cache_path.stat().st_size, orig_stat.st_size)
            self.assertEqual(cache_path.stat().st_mtime_ns, orig_stat.st_mtime_ns)

            rebuilt_mid = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            self.assertFalse(rebuilt_mid.from_cache)
            self.assertEqual(rebuilt_mid.model_path, cache_path)
            self.assertEqual(cache_path.read_bytes(), clean_bytes)

            # 2. Corrupt header bytes while keeping identical size and mtime_ns.
            orig_stat2 = cache_path.stat()
            corrupted_hdr = bytearray(clean_bytes)
            for idx in range(min(64, len(corrupted_hdr))):
                corrupted_hdr[idx] ^= 0xFF
            cache_path.write_bytes(bytes(corrupted_hdr))
            os.utime(cache_path, ns=(orig_stat2.st_atime_ns, orig_stat2.st_mtime_ns))

            rebuilt_hdr = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            self.assertFalse(rebuilt_hdr.from_cache)
            self.assertEqual(rebuilt_hdr.model_path, cache_path)
            self.assertEqual(cache_path.read_bytes(), clean_bytes)

    def test_sidecar_detects_same_size_same_mtime_source_replacement(self):
        from onnx import TensorProto, helper, numpy_helper

        def _make_tiny_onnx(bias_val: float) -> bytes:
            w = numpy_helper.from_array(np.array([bias_val, bias_val], dtype=np.float32), name="bias")
            graph = helper.make_graph(
                [helper.make_node("Add", ["x", "bias"], ["y"], name="add0")],
                "tiny",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
                initializer=[w],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            return model.SerializeToString()

        bytes_a = _make_tiny_onnx(1.0)
        bytes_b = _make_tiny_onnx(3.0)
        self.assertEqual(len(bytes_a), len(bytes_b))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_dir = tmp / "cache"
            src_copy = tmp / "tiny_model.onnx"
            src_copy.write_bytes(bytes_a)

            first = patcher.prepare_parseq_model(
                src_copy,
                enable_dynamic_batch=False,
                precision="fp16",
                cache_dir=cache_dir,
                verify=False,
            )
            orig_stat = src_copy.stat()

            src_copy.write_bytes(bytes_b)
            os.utime(src_copy, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))
            self.assertEqual(src_copy.stat().st_size, orig_stat.st_size)
            self.assertEqual(src_copy.stat().st_mtime_ns, orig_stat.st_mtime_ns)

            second = patcher.prepare_parseq_model(
                src_copy,
                enable_dynamic_batch=False,
                precision="fp16",
                cache_dir=cache_dir,
                verify=False,
            )
            self.assertFalse(second.from_cache)
            self.assertNotEqual(second.model_path, first.model_path)

    def test_parseq_recovers_from_ort_corrupt_cached_model_error(self):
        import parseq as parseq_mod

        class _FakeSess:
            def get_inputs(self):
                return [
                    SimpleNamespace(name="input", shape=["batch", 3, 24, 256], type="tensor(float)")
                ]

            def get_outputs(self):
                return [SimpleNamespace(name="output")]

        fake_profile = SimpleNamespace(
            active_backend="cuda",
            display_name="GPU (CUDA)",
            default_batch_size=64,
            requested_device="cuda",
            fallback_occurred=False,
            warning_message=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"NDLOCR_CACHE_DIR": tmpdir}
        ):
            first = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                verify=False,
            )
            cached_path = Path(first.model_path)
            self.assertTrue(cached_path.is_file())

            calls = []

            def _fake_create_session(model_path, **kwargs):
                calls.append(model_path)
                if len(calls) == 1:
                    raise RuntimeError("Load model failed: Protobuf parsing failed (InvalidProtobuf)")
                return _FakeSess(), fake_profile

            with patch.object(parseq_mod, "create_inference_session", side_effect=_fake_create_session), patch.object(
                patcher, "verify_dynamic_batch_equivalence", return_value=0.0
            ):
                rec = parseq_mod.PARSEQ(str(BUNDLED_MODEL), ["a"], device="cuda", precision="fp32")

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], str(cached_path))
            self.assertEqual(calls[1], str(cached_path))
            self.assertTrue(cached_path.is_file())
            self.assertTrue(rec.dynamic_batch)

    def test_corrupt_source_model_raises_clear_runtime_error(self):
        import parseq as parseq_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_src = tmp / "corrupt_source.onnx"
            bad_src.write_bytes(BUNDLED_MODEL.read_bytes()[:256])
            fallback = patcher.prepare_parseq_model(
                bad_src,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=tmp / "cache",
                verify=False,
            )
            self.assertFalse(fallback.dynamic_batch)
            self.assertEqual(fallback.model_path, bad_src)
            self.assertIsNotNone(fallback.fallback_reason)
            with self.assertRaisesRegex(RuntimeError, "corrupted or unreadable"):
                parseq_mod.PARSEQ(str(bad_src), ["a"], device="cpu")

    def test_sidecar_invalidation_falls_back_to_full_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            first = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            sidecar = patcher._build_sidecar_path(
                BUNDLED_MODEL,
                precision="fp32",
                dynamic_batch=True,
                cache_dir=cache_dir,
            )
            # Corrupt sidecar JSON; must fall back to full validation & cache hit
            sidecar.write_text("{corrupt-json", encoding="utf-8")
            recovered = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            self.assertTrue(recovered.from_cache)
            self.assertEqual(recovered.model_path, first.model_path)

            # Corrupt cached ONNX file; sidecar stat mismatch triggers full rebuild
            Path(first.model_path).write_bytes(b"corrupted")
            rebuilt = patcher.prepare_parseq_model(
                BUNDLED_MODEL,
                enable_dynamic_batch=True,
                precision="fp32",
                cache_dir=cache_dir,
                verify=False,
            )
            self.assertFalse(rebuilt.from_cache)
            self.assertTrue(patcher.is_valid_dynamic_batch_cache(rebuilt.model_path))

    def test_is_valid_cached_model_enforces_fp16_contract(self):
        self.assertFalse(
            patcher.is_valid_cached_model(
                self.patched.model_path,
                require_dynamic_batch=True,
                expected_precision="fp16",
            )
        )


if __name__ == "__main__":
    unittest.main()
