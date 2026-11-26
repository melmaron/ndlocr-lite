from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import onnx_batch_patcher as patcher
from parseq import PARSEQ


PARSEQ_MODELS = [
    ROOT / "src" / "model" / "parseq-ndl-24x256-30-tiny-499epoch-261124.onnx",
    ROOT / "src" / "model" / "parseq-ndl-24x384-50-tiny-278epoch-261124.onnx",
    ROOT / "src" / "model" / "parseq-ndl-24x768-100-tiny-153epoch-261124.onnx",
]
REC_CLASSES = ROOT / "src" / "config" / "NDLmoji.yaml"
SAMPLE_IMAGE = ROOT / "resource" / "digidepo_2531162_0024.jpg"


class OnnxBatchPatcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with REC_CLASSES.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        cls.charlist = list(cfg["model"]["charset_train"])

    def test_cache_path_includes_fp32_precision_and_schema(self):
        cache_path = patcher.build_cache_path(
            Path("/tmp/parseq-ndl-24x384-50.onnx"),
            "0123456789abcdef9999",
            precision="fp32",
            dynamic_batch=True,
            cache_dir=Path("/tmp/cache"),
        )
        self.assertEqual(
            cache_path.name,
            "parseq-ndl-24x384-50_0123456789ab_dynbatch_fp32_v1.onnx",
        )

    def test_patch_and_verify_all_bundled_parseq_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            for model_path in PARSEQ_MODELS:
                if not model_path.is_file():
                    self.skipTest(f"Model not present: {model_path}")
                with self.subTest(model=model_path.name):
                    first = patcher.prepare_parseq_model(
                        model_path,
                        enable_dynamic_batch=True,
                        precision="fp32",
                        cache_dir=cache_dir,
                        verify=True,
                    )
                    self.assertTrue(first.dynamic_batch)
                    self.assertEqual(first.precision, "fp32")
                    self.assertFalse(first.from_cache)
                    self.assertTrue(first.model_path.is_file())
                    self.assertIn("_dynbatch_fp32_v1.onnx", first.model_path.name)
                    self.assertTrue(patcher.is_dynamic_batch_model(first.model_path))

                    # Verify B=1 mathematical identity and batched equivalence with ORT optimizations disabled
                    diff_b1 = patcher.verify_dynamic_batch_equivalence(
                        model_path,
                        first.model_path,
                        batch_size=1,
                        atol=1e-5,
                        disable_ort_optimizations=True,
                    )
                    self.assertLessEqual(diff_b1, 1e-5)

                    diff_raw = patcher.verify_dynamic_batch_equivalence(
                        model_path,
                        first.model_path,
                        batch_size=2,
                        atol=5e-3,
                        disable_ort_optimizations=True,
                    )
                    self.assertLessEqual(diff_raw, 5e-3)

                    # Verify second call reuses local cache
                    second = patcher.prepare_parseq_model(
                        model_path,
                        enable_dynamic_batch=True,
                        precision="fp32",
                        cache_dir=cache_dir,
                        verify=True,
                    )
                    self.assertTrue(second.from_cache)
                    self.assertEqual(second.model_path, first.model_path)

    def test_parseq_read_batch_matches_sequential_read_on_real_crops(self):
        model_path = PARSEQ_MODELS[1]  # 24x384-50
        if not model_path.is_file() or not SAMPLE_IMAGE.is_file():
            self.skipTest("Sample resources unavailable")

        img_bgr = cv2.imread(str(SAMPLE_IMAGE))
        h, w = img_bgr.shape[:2]
        crops = [
            img_bgr[int(h * 0.15):int(h * 0.35), int(w * 0.72):int(w * 0.77)],
            img_bgr[int(h * 0.15):int(h * 0.45), int(w * 0.62):int(w * 0.67)],
            img_bgr[int(h * 0.20):int(h * 0.50), int(w * 0.52):int(w * 0.57)],
            img_bgr[int(h * 0.20):int(h * 0.55), int(w * 0.42):int(w * 0.47)],
            np.full((40, 200, 3), 255, dtype=np.uint8),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            patched = patcher.prepare_parseq_model(
                model_path,
                enable_dynamic_batch=True,
                cache_dir=Path(tmpdir),
                verify=False,
            )
            rec_seq = PARSEQ(
                model_path=str(model_path),
                charlist=self.charlist,
                device="cpu",
                batch_size=1,
            )
            rec_batched = PARSEQ(
                model_path=str(patched.model_path),
                charlist=self.charlist,
                device="cpu",
                batch_size=3,
            )
            self.assertFalse(rec_seq.dynamic_batch)
            self.assertTrue(rec_batched.dynamic_batch)
            self.assertEqual(rec_batched.batch_size, 3)

            seq_preds = [rec_seq.read(crop) for crop in crops]
            batched_preds = rec_batched.read_batch(crops)
            self.assertEqual(batched_preds, seq_preds)

    def test_invalid_model_falls_back_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_model = Path(tmpdir) / "not_a_model.onnx"
            bad_model.write_bytes(b"corrupt-bytes")
            res = patcher.prepare_parseq_model(
                bad_model,
                enable_dynamic_batch=True,
                cache_dir=Path(tmpdir) / "cache",
            )
            self.assertFalse(res.dynamic_batch)
            self.assertEqual(res.model_path, bad_model)
            self.assertIsNotNone(res.fallback_reason)

    def test_validity_check_rejects_truncated_cache(self):
        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            patched = patcher.prepare_parseq_model(
                model_path,
                enable_dynamic_batch=True,
                cache_dir=Path(tmpdir),
                verify=False,
            )
            self.assertTrue(patcher.is_valid_dynamic_batch_cache(patched.model_path))

            empty = Path(tmpdir) / "empty.onnx"
            empty.write_bytes(b"")
            self.assertFalse(patcher.is_valid_dynamic_batch_cache(empty))

            junk = Path(tmpdir) / "junk.onnx"
            junk.write_bytes(b"not-an-onnx-model")
            self.assertFalse(patcher.is_valid_dynamic_batch_cache(junk))

    def test_corrupt_cache_is_rebuilt_and_parseq_still_loads(self):
        from unittest.mock import patch

        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"NDLOCR_CACHE_DIR": tmpdir}):
                # Build a real cache, then simulate an interrupted copy/disk error.
                patched = patcher.prepare_parseq_model(
                    model_path,
                    enable_dynamic_batch=True,
                    verify=False,
                )
                cache_path = Path(patched.model_path)
                self.assertTrue(cache_path.is_file())
                self.assertGreater(cache_path.stat().st_size, 1024)
                cache_path.write_bytes(b"truncated cache")

                # The next prepare call must discard the broken cache and rebuild it.
                rebuilt = patcher.prepare_parseq_model(
                    model_path,
                    enable_dynamic_batch=True,
                    verify=False,
                )
                self.assertFalse(rebuilt.from_cache)
                self.assertEqual(rebuilt.model_path, cache_path)
                self.assertTrue(patcher.is_valid_dynamic_batch_cache(cache_path))

                # End-to-end: PARSEQ initialisation must recover instead of raising
                # InvalidProtobuf when it finds a corrupted cache.
                cache_path.write_bytes(b"truncated cache")
                with patch("parseq.resolve_auto_backend_name", return_value="cuda"):
                    rec = PARSEQ(
                        model_path=str(model_path),
                        charlist=self.charlist,
                        device="auto",
                        batch_size=4,
                    )
                self.assertTrue(rec.dynamic_batch)
                self.assertEqual(rec.batch_size, 4)
                self.assertTrue(patcher.is_valid_dynamic_batch_cache(cache_path))
                self.assertIsInstance(
                    rec.read(np.full((30, 120, 3), 255, dtype=np.uint8)), str
                )

    def test_unsupported_precision_falls_back_without_crashing(self):
        res = patcher.prepare_parseq_model(
            PARSEQ_MODELS[0],
            precision="int8",
        )
        self.assertFalse(res.dynamic_batch)
        self.assertEqual(res.model_path, PARSEQ_MODELS[0])
        self.assertIn("Unsupported precision", res.fallback_reason)



class PatcherRobustnessTest(unittest.TestCase):
    def test_cache_io_failure_falls_back_instead_of_raising(self):
        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            blocker = Path(tmpdir) / "blocker"
            blocker.write_bytes(b"not a directory")
            result = patcher.prepare_parseq_model(
                model_path,
                enable_dynamic_batch=True,
                cache_dir=blocker,
                verify=False,
            )
        self.assertFalse(result.dynamic_batch)
        self.assertEqual(result.model_path, model_path)
        self.assertIsNotNone(result.fallback_reason)
        self.assertIn("cache unavailable", result.fallback_reason)

    def test_validity_predicate_never_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(patcher.is_valid_dynamic_batch_cache(Path(tmpdir) / "missing.onnx"))
            self.assertFalse(patcher.is_valid_dynamic_batch_cache(Path(tmpdir)))

    def test_pruning_skips_control_flow_subgraphs(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        outer = numpy_helper.from_array(np.arange(4, dtype=np.float32), name="outer_keep")
        body = helper.make_graph([], "body", [], [])
        if_node = helper.make_node(
            "If", ["cond"], [], name="If", then_branch=body, else_branch=body
        )
        cond = numpy_helper.from_array(np.array(True), name="cond")
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"]), if_node],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            initializer=[outer, cond],
        )
        model = helper.make_model(graph)
        patcher._prune_unused_constants(model)
        names = {init.name for init in model.graph.initializer}
        self.assertIn("outer_keep", names)
        # With no subgraphs the same unreferenced initializer is removed.
        graph2 = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "g2",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            initializer=[numpy_helper.from_array(np.arange(4, dtype=np.float32), name="outer_keep")],
        )
        model2 = helper.make_model(graph2)
        patcher._prune_unused_constants(model2)
        self.assertEqual([init.name for init in model2.graph.initializer], [])

    def test_verification_rejects_non_finite_outputs(self):
        class FakeInput:
            name = "x"
            shape = [1, 3, 32, 32]
            type = "tensor(float)"

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass

            def get_inputs(self):
                return [FakeInput()]

            def run(self, output_names, feed):
                batch = int(np.asarray(feed["x"]).shape[0])
                return [np.full((batch, 4, 3), np.nan, dtype=np.float32)]

        with patch.object(patcher.ort, "InferenceSession", FakeSession):
            with self.assertRaisesRegex(ValueError, "Non-finite|Max abs diff"):
                patcher.verify_dynamic_batch_equivalence("a.onnx", "b.onnx", batch_size=2)


if __name__ == "__main__":
    unittest.main()
