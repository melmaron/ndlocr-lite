import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ort_provider


BUNDLED_PARSEQ30 = ROOT / "src" / "model" / "parseq-ndl-24x256-30-tiny-499epoch-261124.onnx"


class OrtProviderTest(unittest.TestCase):
    def setUp(self):
        ort_provider._WARNED_KEYS.clear()

    def test_choices_priority_and_normalization(self):
        self.assertIn("auto", ort_provider.DEVICE_CHOICES)
        self.assertEqual(
            ort_provider.AUTO_PRIORITY,
            ("cuda", "migraphx", "rocm", "openvino", "webgpu", "dml", "cpu"),
        )
        self.assertEqual(ort_provider.normalize_device_name("DirectML"), "dml")
        self.assertEqual(ort_provider.normalize_device_name("OV"), "openvino")
        self.assertEqual(ort_provider.normalize_device_name("GPU"), "auto")
        self.assertEqual(ort_provider.normalize_device_name(""), "auto")
        self.assertEqual(ort_provider.normalize_device_name("  CUDA "), "cuda")

    def test_deform_conv_version_check(self):
        self.assertTrue(ort_provider.ort_supports_deform_conv("1.31.1"))
        self.assertTrue(ort_provider.ort_supports_deform_conv("1.25.0"))
        self.assertFalse(ort_provider.ort_supports_deform_conv("1.24.4"))

    def test_parallel_model_init_policy(self):
        self.assertTrue(ort_provider.should_parallelize_model_init("cpu"))
        for dev in ("auto", "cuda", "migraphx", "rocm", "openvino", "webgpu", "dml"):
            self.assertFalse(ort_provider.should_parallelize_model_init(dev))

    def test_openvino_cpu_only_ep_device_not_treated_as_gpu(self):
        ov_cpu = SimpleNamespace(
            ep_name="OpenVINOExecutionProvider",
            ep_metadata={"ov_device": "CPU"},
            device=SimpleNamespace(type="CPU", vendor="Intel", device_id=0),
        )
        ov_gpu = SimpleNamespace(
            ep_name="OpenVINOExecutionProvider",
            ep_metadata={"ov_device": "GPU.0"},
            device=SimpleNamespace(type="GPU", vendor="Intel", device_id=0),
        )
        webgpu_dev = SimpleNamespace(
            ep_name="WebGpuExecutionProvider",
            ep_metadata={},
            device=SimpleNamespace(type="GPU", vendor="AMD", device_id=0),
        )
        self.assertEqual(
            ort_provider._find_plugin_devices_for_backend("openvino", [ov_cpu]),
            [],
        )
        self.assertEqual(
            ort_provider._find_plugin_devices_for_backend("openvino", [ov_cpu, ov_gpu]),
            [ov_gpu],
        )
        self.assertEqual(
            ort_provider._find_plugin_devices_for_backend("webgpu", [ov_cpu, webgpu_dev]),
            [webgpu_dev],
        )

    def test_loaded_openvino_plugin_without_gpu_does_not_hijack_auto(self):
        ov_cpu = SimpleNamespace(
            ep_name="OpenVINOExecutionProvider",
            ep_metadata={"ov_device": "CPU"},
            device=SimpleNamespace(type="CPU", vendor="Intel", device_id=0),
        )
        webgpu_dev = SimpleNamespace(
            ep_name="WebGpuExecutionProvider",
            ep_metadata={},
            device=SimpleNamespace(type="GPU", vendor="GPU", device_id=0),
        )
        with patch.dict(
            ort_provider._REGISTERED_PLUGINS,
            {"openvino_ep": "/fake/ov.so", "webgpu": "/fake/wgpu.so"},
            clear=True,
        ), patch.object(
            ort_provider, "register_plugin_eps", return_value=dict(ort_provider._REGISTERED_PLUGINS)
        ), patch.object(
            ort_provider.ort,
            "get_available_providers",
            return_value=["CPUExecutionProvider", "OpenVINOExecutionProvider", "WebGpuExecutionProvider"],
        ), patch.object(
            ort_provider, "_safe_ep_devices", return_value=[ov_cpu, webgpu_dev]
        ):
            avail = ort_provider.detect_available_backends()
            self.assertFalse(avail["openvino"]["available"])
            self.assertIn("no Intel GPU device detected", avail["openvino"]["detail"])
            self.assertTrue(avail["webgpu"]["available"])
            self.assertEqual(ort_provider.resolve_auto_backend_name(), "webgpu")
            with self.assertRaisesRegex(RuntimeError, "no Intel GPU device detected"):
                ort_provider._create_session_for_backend(
                    BUNDLED_PARSEQ30,
                    "openvino",
                    device_id=0,
                    cpu_single_thread=False,
                )

    def test_auto_priority_resolution_with_mocked_backends(self):
        mock_avail = {
            "cuda": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "migraphx": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "rocm": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "openvino": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "webgpu": {"available": True, "mode": "plugin", "devices": ["WebGPU"], "detail": "WebGPU"},
            "dml": {"available": True, "mode": "monolithic", "devices": ["DML"], "detail": "DML"},
            "cpu": {"available": True, "mode": "builtin", "devices": ["CPU"], "detail": "CPU"},
        }
        with patch.object(ort_provider, "detect_available_backends", return_value=mock_avail):
            self.assertEqual(ort_provider.resolve_auto_backend_name(), "webgpu")

        mock_avail["openvino"]["available"] = True
        with patch.object(ort_provider, "detect_available_backends", return_value=mock_avail):
            self.assertEqual(ort_provider.resolve_auto_backend_name(), "openvino")

    def test_create_cpu_session_on_bundled_model(self):
        if not BUNDLED_PARSEQ30.is_file():
            self.skipTest("Bundled PARSeq model not present")
        session, profile = ort_provider.create_inference_session(
            BUNDLED_PARSEQ30,
            device="cpu",
            cpu_single_thread=True,
            model_label="PARSeq-30",
        )
        self.assertIn("CPUExecutionProvider", session.get_providers())
        self.assertEqual(profile.active_backend, "cpu")
        self.assertFalse(profile.fallback_occurred)
        self.assertEqual(profile.default_batch_size, 1)
        self.assertTrue(profile.parallel_session_init)
        self.assertTrue(profile.concurrent_session_run)

    def test_unavailable_backend_falls_back_or_raises_strictly(self):
        if not BUNDLED_PARSEQ30.is_file():
            self.skipTest("Bundled PARSeq model not present")
        with patch.object(
            ort_provider,
            "_create_session_for_backend",
            side_effect=[RuntimeError("no such backend"), (object(), ["CPUExecutionProvider"])],
        ):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                _, profile = ort_provider.create_inference_session(
                    BUNDLED_PARSEQ30,
                    device="migraphx",
                    strict_device=False,
                    model_label="PARSeq-30",
                )
            self.assertEqual(profile.active_backend, "cpu")
            self.assertEqual(profile.requested_device, "migraphx")
            self.assertTrue(profile.fallback_occurred)
            self.assertIn("migraphx", stderr.getvalue())

        with patch.object(
            ort_provider,
            "_create_session_for_backend",
            side_effect=RuntimeError("no such backend"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Requested device 'migraphx' failed"):
                ort_provider.create_inference_session(
                    BUNDLED_PARSEQ30,
                    device="migraphx",
                    strict_device=True,
                    model_label="PARSeq-30",
                )

    def test_invalid_device_id_is_rejected_before_session_creation(self):
        for bad in (-1, "abc", None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "device_id"):
                    ort_provider.create_inference_session(
                        "unused.onnx", device="cpu", device_id=bad
                    )

    def test_plugin_device_id_out_of_range_is_rejected(self):
        webgpu_dev = SimpleNamespace(
            ep_name="WebGpuExecutionProvider",
            ep_metadata={},
            device=SimpleNamespace(type="GPU", vendor="Vendor", device_id=0),
        )
        with patch.object(ort_provider, "register_plugin_eps"), patch.object(
            ort_provider, "_safe_ep_devices", return_value=[webgpu_dev]
        ):
            with self.assertRaisesRegex(RuntimeError, "out of range"):
                ort_provider._create_session_for_backend(
                    "unused.onnx", "webgpu", device_id=2, cpu_single_thread=False
                )

    def test_auto_with_strict_device_raises_when_gpu_init_fails(self):
        availability = {
            "cuda": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "migraphx": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "rocm": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "openvino": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "webgpu": {"available": True, "mode": "plugin", "devices": ["WebGPU"], "detail": "WebGPU"},
            "dml": {"available": False, "mode": "none", "devices": [], "detail": ""},
            "cpu": {"available": True, "mode": "builtin", "devices": ["CPU"], "detail": "CPU"},
        }

        def failing_gpu(model_path, backend, *args, **kwargs):
            if backend == "cpu":
                return object(), ["CPUExecutionProvider"]
            raise RuntimeError(f"{backend} init failed")

        with patch.object(ort_provider, "detect_available_backends", return_value=availability), patch.object(
            ort_provider, "_create_session_for_backend", side_effect=failing_gpu
        ):
            with self.assertRaisesRegex(RuntimeError, "auto"):
                ort_provider.create_inference_session(
                    "unused.onnx", device="auto", strict_device=True
                )

        stderr = io.StringIO()
        with patch.object(ort_provider, "detect_available_backends", return_value=availability), patch.object(
            ort_provider, "_create_session_for_backend", side_effect=failing_gpu
        ), contextlib.redirect_stderr(stderr):
            _, profile = ort_provider.create_inference_session(
                "unused.onnx", device="auto", strict_device=False
            )
        self.assertEqual(profile.active_backend, "cpu")
        self.assertTrue(profile.fallback_occurred)

    def test_format_device_diagnostics_contains_summary(self):
        report = ort_provider.format_device_diagnostics()
        self.assertIn("NDLOCR-Lite GPU / ONNX Runtime Diagnostics", report)
        self.assertIn("TableCenterNet (DeformConv, ORT>=1.25):", report)
        self.assertIn("Selected by '--device auto':", report)

    def test_cuda_session_options_use_heuristic_and_forward_mem_reuse(self):
        captured = {}

        class _FakeSess:
            def __init__(self, path, sess_options=None, providers=None):
                captured["path"] = path
                captured["opt"] = sess_options
                captured["providers"] = providers

            def get_providers(self):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with patch.object(ort_provider, "preload_cuda_dlls_if_needed"), patch.object(
            ort_provider.ort,
            "get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ), patch.object(ort_provider.ort, "InferenceSession", side_effect=_FakeSess):
            _, profile = ort_provider.create_inference_session(
                "model.onnx",
                device="cuda",
                device_id=1,
                enable_mem_reuse=False,
            )
        self.assertEqual(profile.active_backend, "cuda")
        self.assertFalse(captured["opt"].enable_mem_reuse)
        self.assertEqual(
            captured["providers"][0],
            (
                "CUDAExecutionProvider",
                {"device_id": 1, "cudnn_conv_algo_search": "HEURISTIC", "use_tf32": "0"},
            ),
        )

    def test_webgpu_virtual_or_software_adapter_is_skipped_by_auto_but_allowed_explicitly(self):
        wsl_virtual = SimpleNamespace(
            ep_name="WebGpuExecutionProvider",
            ep_metadata={},
            device=SimpleNamespace(type="GPU", vendor="", vendor_id=0x1414, device_id=142),
        )
        llvmpipe = SimpleNamespace(
            ep_name="WebGpuExecutionProvider",
            ep_metadata={},
            device=SimpleNamespace(type="GPU", vendor="Mesa llvmpipe", vendor_id=0x10005, device_id=0),
        )
        amd_hw = SimpleNamespace(
            ep_name="WebGpuExecutionProvider",
            ep_metadata={},
            device=SimpleNamespace(type="GPU", vendor="AMD", vendor_id=0x1002, device_id=1),
        )

        self.assertEqual(
            ort_provider._find_plugin_devices_for_backend(
                "webgpu", [wsl_virtual, amd_hw, llvmpipe]
            ),
            [amd_hw],
        )
        self.assertEqual(
            ort_provider._find_plugin_devices_for_backend(
                "webgpu", [wsl_virtual, amd_hw, llvmpipe], include_virtual_webgpu=True
            ),
            [amd_hw, wsl_virtual, llvmpipe],
        )

        captured = {}

        class _FakeOpt:
            def __init__(self):
                self.devs = None

            def add_provider_for_devices(self, devs, cfg):
                self.devs = list(devs)

        class _FakeWebGpuSess:
            def __init__(self, path, sess_options=None, providers=None):
                captured["path"] = path
                captured["opt"] = sess_options

            def get_providers(self):
                return ["WebGpuExecutionProvider", "CPUExecutionProvider"]

        with patch.dict(
            ort_provider._REGISTERED_PLUGINS,
            {"webgpu": "/fake/wgpu.so"},
            clear=True,
        ), patch.object(
            ort_provider, "register_plugin_eps", return_value=dict(ort_provider._REGISTERED_PLUGINS)
        ), patch.object(
            ort_provider.ort,
            "get_available_providers",
            return_value=["CPUExecutionProvider", "WebGpuExecutionProvider"],
        ), patch.object(
            ort_provider, "_safe_ep_devices", return_value=[wsl_virtual, llvmpipe]
        ), patch.object(
            ort_provider, "_build_session_options", return_value=_FakeOpt()
        ), patch.object(
            ort_provider.ort, "InferenceSession", side_effect=_FakeWebGpuSess
        ):
            avail = ort_provider.detect_available_backends()
            self.assertFalse(avail["webgpu"]["available"])
            self.assertIn("software/virtual adapter", avail["webgpu"]["detail"])
            self.assertEqual(ort_provider.resolve_auto_backend_name(), "cpu")
            with self.assertRaisesRegex(RuntimeError, "software/virtual adapter"):
                ort_provider._create_session_for_backend(
                    "model.onnx",
                    "webgpu",
                    device_id=0,
                    cpu_single_thread=False,
                )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                _, profile = ort_provider.create_inference_session(
                    "model.onnx",
                    device="webgpu",
                    model_label="PARSeq-30",
                )
            self.assertEqual(profile.active_backend, "webgpu")
            self.assertEqual(captured["opt"].devs, [wsl_virtual])
            self.assertIn("ソフトウェア/仮想アダプタ", stderr.getvalue())

    def test_dml_blocks_deim_with_cpu_fallback_but_allows_parseq(self):
        class _FakeDmlSess:
            def __init__(self, path, sess_options=None, providers=None):
                self._providers = [
                    p[0] if isinstance(p, tuple) else p for p in (providers or [])
                ]

            def get_providers(self):
                return self._providers

        with patch.object(
            ort_provider.ort,
            "get_available_providers",
            return_value=["DmlExecutionProvider", "CPUExecutionProvider"],
        ), patch.object(ort_provider.ort, "InferenceSession", side_effect=_FakeDmlSess):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                _, deim_profile = ort_provider.create_inference_session(
                    "deim-s-1020x1020.onnx",
                    device="dml",
                    strict_device=False,
                    model_label="DEIM",
                )
            self.assertEqual(deim_profile.active_backend, "cpu")
            self.assertTrue(deim_profile.fallback_occurred)
            self.assertIn("DEIM", stderr.getvalue())

            with self.assertRaisesRegex(RuntimeError, "corrupted bounding boxes"):
                ort_provider.create_inference_session(
                    "deim-s-1020x1020.onnx",
                    device="dml",
                    strict_device=True,
                    model_label="DEIM",
                )

            _, parseq_profile = ort_provider.create_inference_session(
                "parseq-ndl-24x256-30.onnx",
                device="dml",
                strict_device=True,
                model_label="PARSEQ(rec30)",
            )
            self.assertEqual(parseq_profile.active_backend, "dml")


if __name__ == "__main__":
    unittest.main()
