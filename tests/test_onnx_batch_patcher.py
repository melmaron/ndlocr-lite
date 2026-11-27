import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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
        self.assertEqual(patcher.SUPPORTED_PRECISIONS, ("fp32", "fp16"))
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
        fp16_dyn = patcher.build_cache_path(
            Path("/tmp/parseq-ndl-24x384-50.onnx"),
            "0123456789abcdef9999",
            precision="fp16",
            dynamic_batch=True,
            cache_dir=Path("/tmp/cache"),
        )
        self.assertEqual(
            fp16_dyn.name,
            "parseq-ndl-24x384-50_0123456789ab_dynbatch_fp16_v1.onnx",
        )
        fp16_static = patcher.build_cache_path(
            Path("/tmp/parseq-ndl-24x384-50.onnx"),
            "0123456789abcdef9999",
            precision="fp16",
            dynamic_batch=False,
            cache_dir=Path("/tmp/cache"),
        )
        self.assertEqual(
            fp16_static.name,
            "parseq-ndl-24x384-50_0123456789ab_static_fp16_v1.onnx",
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

    def test_patch_and_verify_all_bundled_parseq_models_fp16(self):
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            for model_path in PARSEQ_MODELS:
                if not model_path.is_file():
                    self.skipTest(f"Model not present: {model_path}")
                with self.subTest(model=model_path.name):
                    first = patcher.prepare_parseq_model(
                        model_path,
                        enable_dynamic_batch=True,
                        precision="fp16",
                        cache_dir=cache_dir,
                        verify=True,
                    )
                    self.assertTrue(first.dynamic_batch)
                    self.assertEqual(first.precision, "fp16")
                    self.assertFalse(first.from_cache)
                    self.assertIsNone(first.fallback_reason)
                    self.assertIn("_dynbatch_fp16_v1.onnx", first.model_path.name)
                    self.assertTrue(
                        patcher.is_valid_cached_model(
                            first.model_path,
                            require_dynamic_batch=True,
                            expected_precision="fp16",
                        )
                    )

                    loaded = onnx.load(str(first.model_path), load_external_data=False)
                    self.assertEqual(
                        loaded.graph.input[0].type.tensor_type.elem_type,
                        onnx.TensorProto.FLOAT,
                    )
                    self.assertEqual(
                        loaded.graph.output[0].type.tensor_type.elem_type,
                        onnx.TensorProto.FLOAT,
                    )
                    self.assertTrue(patcher._model_has_fp16_tensors(loaded))
                    seen_out = set()
                    for node in loaded.graph.node:
                        for out in node.output:
                            if out:
                                self.assertNotIn(out, seen_out)
                                seen_out.add(out)

                    second = patcher.prepare_parseq_model(
                        model_path,
                        enable_dynamic_batch=True,
                        precision="fp16",
                        cache_dir=cache_dir,
                        verify=True,
                    )
                    self.assertTrue(second.from_cache)
                    self.assertEqual(second.model_path, first.model_path)

    def test_attention_score_protection_keeps_overflow_matmul_in_fp32(self):
        import onnx

        for model_path in PARSEQ_MODELS:
            if not model_path.is_file():
                self.skipTest(f"Model not present: {model_path}")
            model = onnx.load(str(model_path), load_external_data=False)
            attn_nodes = patcher._find_attention_score_nodes(model)
            self.assertTrue(attn_nodes)
            self.assertTrue(any(name.endswith("/attn/MatMul") for name in attn_nodes))

        rec100_path = PARSEQ_MODELS[2]
        rec100 = onnx.load(str(rec100_path))
        self.assertIn(
            "/encoder/blocks/blocks.11/attn/MatMul",
            patcher._find_attention_score_nodes(rec100),
        )
        converted = patcher.convert_model_to_fp16(rec100)
        target_node = next(
            n for n in converted.graph.node if n.name == "/encoder/blocks/blocks.11/attn/MatMul"
        )
        producers = {o: n for n in converted.graph.node for o in n.output if o}
        for inp in target_node.input:
            self.assertIn(inp, producers)
            self.assertEqual(producers[inp].op_type, "Cast")
            to_attr = next(a.i for a in producers[inp].attribute if a.name == "to")
            self.assertEqual(to_attr, onnx.TensorProto.FLOAT)

    def test_prepare_static_fp16_model(self):
        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        with tempfile.TemporaryDirectory() as tmpdir:
            res = patcher.prepare_parseq_model(
                model_path,
                enable_dynamic_batch=False,
                precision="fp16",
                cache_dir=Path(tmpdir),
                verify=True,
            )
            self.assertFalse(res.dynamic_batch)
            self.assertEqual(res.precision, "fp16")
            self.assertIn("_static_fp16_v1.onnx", res.model_path.name)
            self.assertTrue(
                patcher.is_valid_cached_model(
                    res.model_path,
                    require_dynamic_batch=False,
                    expected_precision="fp16",
                )
            )

    def test_parseq_fp16_read_batch_matches_fp32_on_real_crops(self):
        model_path = PARSEQ_MODELS[1]  # 24x384-50
        if not model_path.is_file() or not SAMPLE_IMAGE.is_file():
            self.skipTest("Sample resources unavailable")

        img_bgr = cv2.imread(str(SAMPLE_IMAGE))
        h, w = img_bgr.shape[:2]
        crops = [
            img_bgr[int(h * 0.15):int(h * 0.35), int(w * 0.72):int(w * 0.77)],
            img_bgr[int(h * 0.15):int(h * 0.45), int(w * 0.62):int(w * 0.67)],
            img_bgr[int(h * 0.20):int(h * 0.50), int(w * 0.52):int(w * 0.57)],
            np.full((40, 200, 3), 255, dtype=np.uint8),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            patched = patcher.prepare_parseq_model(
                model_path,
                enable_dynamic_batch=True,
                precision="fp16",
                cache_dir=Path(tmpdir),
                verify=False,
            )
            rec_fp32 = PARSEQ(str(model_path), self.charlist, device="cpu", batch_size=1)
            rec_fp16 = PARSEQ(str(patched.model_path), self.charlist, device="cpu", batch_size=4)
            self.assertTrue(rec_fp16.dynamic_batch)
            self.assertEqual(rec_fp16.batch_size, 4)
            fp32_preds = [rec_fp32.read(c) for c in crops]
            fp16_preds = rec_fp16.read_batch(crops)
            self.assertEqual(fp16_preds, fp32_preds)

    def test_convert_model_to_fp16_deduplicates_duplicate_cast_outputs(self):
        import onnx

        rec30_path = PARSEQ_MODELS[0]
        if not rec30_path.is_file():
            self.skipTest("Bundled rec30 model missing")
        model = onnx.load(str(rec30_path))
        converted = patcher.convert_model_to_fp16(
            model,
            extra_op_block_list=("LayerNormalization", "Softmax"),
            protect_attention_scores=True,
        )
        onnx.checker.check_model(converted)
        seen_out = set()
        for node in converted.graph.node:
            for out in node.output:
                if out:
                    self.assertNotIn(out, seen_out)
                    seen_out.add(out)

    def test_parseq_cpu_downgrades_fp16_to_fp32_with_warning(self):
        import ort_provider
        from tcy_wrapper import TateChuYokoWrapper

        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        ort_provider._WARNED_KEYS.clear()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rec = PARSEQ(str(model_path), self.charlist, device="cpu", precision="fp16")
        self.assertEqual(rec.requested_precision, "fp16")
        self.assertEqual(rec.effective_precision, "fp32")
        self.assertTrue(rec.precision_fallback_occurred)
        self.assertEqual(rec.effective_model_path, str(model_path))
        self.assertIn("fp16", stderr.getvalue())

        wrapped = TateChuYokoWrapper(rec)
        self.assertEqual(wrapped.requested_precision, "fp16")
        self.assertEqual(wrapped.effective_precision, "fp32")
        self.assertTrue(wrapped.precision_fallback_occurred)

        rec_fp32 = PARSEQ(str(model_path), self.charlist, device="cpu", precision="fp32")
        self.assertEqual(rec_fp32.requested_precision, "fp32")
        self.assertFalse(rec_fp32.precision_fallback_occurred)

    def test_parseq_warmup_caps_batch_size(self):
        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        rec = PARSEQ(str(model_path), self.charlist, device="cpu")
        recorded_shapes = []

        class _SpySession:
            def run(self, out_names, feed):
                arr = next(iter(feed.values()))
                recorded_shapes.append(arr.shape)
                return [np.zeros((arr.shape[0], 1, 1), dtype=np.float32)]

        rec.session = _SpySession()
        # Static model -> batch=1
        rec.dynamic_batch = False
        rec.batch_size = 1
        rec.input_width = 256
        rec.warmup()
        # Dynamic rec30 (width=256) with batch=64 -> capped at 16
        rec.dynamic_batch = True
        rec.batch_size = 64
        rec.input_width = 256
        rec.warmup()
        # Dynamic rec100 (width=768) with batch=64 -> capped at 8
        rec.batch_size = 64
        rec.input_width = 768
        rec.warmup()
        # Dynamic rec30 with batch=4 -> 4
        rec.batch_size = 4
        rec.input_width = 256
        rec.warmup()

        self.assertEqual(
            recorded_shapes,
            [
                (1, 3, int(rec.input_height), 256),
                (16, 3, int(rec.input_height), 256),
                (8, 3, int(rec.input_height), 768),
                (4, 3, int(rec.input_height), 256),
            ],
        )

    def test_ocr_warmup_gpu_model_is_non_fatal_on_exception(self):
        import ocr
        import ort_provider

        ort_provider._WARNED_KEYS.clear()

        class _BoomModel:
            backend_profile = SimpleNamespace(active_backend="cuda")

            def warmup(self):
                raise RuntimeError("Boom OOM during warmup")

        model = _BoomModel()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            res = ocr._warmup_gpu_model(model)
        self.assertIs(res, model)
        self.assertIn("warm-up skipped", stderr.getvalue())
        self.assertIn("Boom OOM", stderr.getvalue())

    def test_parseq_decode_outputs_handles_float16_logits(self):
        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        rec = PARSEQ(str(model_path), self.charlist, device="cpu")
        logits = np.full((2, 6, len(self.charlist) + 1), -5.0, dtype=np.float32)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        logits[0, 2, 0] = 10.0
        logits[1, 0, 3] = 10.0
        logits[1, 1, 0] = 10.0
        self.assertEqual(
            rec._decode_outputs(logits.astype(np.float16)),
            rec._decode_outputs(logits),
        )

    def test_parseq_preprocess_batch_matches_bgr_slice_reference(self):
        model_path = PARSEQ_MODELS[0]
        if not model_path.is_file():
            self.skipTest("Bundled PARSeq model missing")
        rec = PARSEQ(str(model_path), self.charlist, device="cpu")
        rng = np.random.default_rng(7)
        imgs = [
            rng.integers(0, 256, size=(48, 20, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(24, 140, 3), dtype=np.uint8),
            rng.integers(0, 256, size=(30, 90), dtype=np.uint8),
        ] * 4  # 12 items (>= old ThreadPool threshold of 8)

        actual = rec.preprocess_batch(imgs, auto_rotate=True)
        ref = []
        for im in imgs:
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            h, w = im.shape[:2]
            if h > w * 0.8:
                im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            resized = cv2.resize(im, (rec.input_width, rec.input_height), interpolation=cv2.INTER_LINEAR)
            x = np.ascontiguousarray(resized[:, :, ::-1]).astype(np.float32)
            x /= 127.5
            x -= 1.0
            ref.append(x.transpose(2, 0, 1))
        expected = np.ascontiguousarray(np.stack(ref, axis=0))
        np.testing.assert_array_equal(actual, expected)



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
