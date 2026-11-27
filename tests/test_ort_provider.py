import contextlib
import io
from pathlib import Path
import tempfile
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
        ort_provider._CUDA_DLLS_PRELOADED = False
        ort_provider._CUDA_DLLS_ERROR = None
        ort_provider._CUDA_PRELOAD_OUTPUT = None
        ort_provider._CUDA_PROBE_CACHE = None
        ort_provider._PLUGIN_ERRORS.clear()
        ort_provider._REGISTERED_DLL_DIRS.clear()
        ort_provider._DLL_DIR_HANDLES.clear()

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
        self.assertTrue(ort_provider.should_parallelize_model_init("cuda"))
        for dev in ("migraphx", "rocm", "openvino", "webgpu", "dml"):
            self.assertFalse(ort_provider.should_parallelize_model_init(dev))
        with patch.object(ort_provider, "resolve_auto_backend_name", return_value="cuda"):
            self.assertTrue(ort_provider.should_parallelize_model_init("auto"))
        with patch.object(ort_provider, "resolve_auto_backend_name", return_value="cpu"):
            self.assertTrue(ort_provider.should_parallelize_model_init("auto"))
        with patch.object(ort_provider, "resolve_auto_backend_name", return_value="webgpu"):
            self.assertFalse(ort_provider.should_parallelize_model_init("auto"))

    def test_precision_normalization_and_cuda_defaults(self):
        self.assertEqual(ort_provider.PRECISION_CHOICES, ("fp32", "fp16"))
        self.assertEqual(ort_provider.normalize_precision_name(None), "fp32")
        self.assertEqual(ort_provider.normalize_precision_name(""), "fp32")
        self.assertEqual(ort_provider.normalize_precision_name("FLOAT32"), "fp32")
        self.assertEqual(ort_provider.normalize_precision_name("fp16"), "fp16")
        self.assertEqual(ort_provider.normalize_precision_name(" Float16 "), "fp16")
        self.assertEqual(ort_provider.normalize_precision_name("half"), "fp16")
        self.assertEqual(ort_provider.normalize_precision_name("int8"), "int8")
        self.assertEqual(ort_provider._DEFAULT_BATCH_SIZES["cuda"], 64)

        with patch.object(ort_provider, "preload_cuda_dlls_if_needed") as preload, patch.object(
            ort_provider, "register_plugin_eps"
        ) as register:
            ort_provider.prepare_backend_for_parallel_init("cuda")
        preload.assert_called_once_with()
        register.assert_called_once_with()

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
        self.assertEqual(profile.default_batch_size, 64)
        self.assertTrue(profile.parallel_session_init)
        self.assertFalse(profile.concurrent_session_run)
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

    def test_normalize_helpers_handle_non_string_inputs(self):
        self.assertEqual(ort_provider.normalize_precision_name(123), "fp32")
        self.assertEqual(ort_provider.normalize_precision_name(None), "fp32")
        self.assertEqual(ort_provider.normalize_precision_name("FLOAT16"), "fp16")
        self.assertEqual(ort_provider.normalize_device_name(123), "auto")
        self.assertEqual(ort_provider.normalize_device_name(None), "auto")

    def test_cuda_backend_preloads_dlls_before_snapshotting_providers(self):
        state = {"preloaded": False}

        def fake_preload():
            state["preloaded"] = True

        def fake_providers():
            if state["preloaded"]:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

        class _FakeCudaSess:
            def __init__(self, path, sess_options=None, providers=None):
                pass

            def get_providers(self):
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with patch.object(
            ort_provider, "preload_cuda_dlls_if_needed", side_effect=fake_preload
        ), patch.object(
            ort_provider.ort, "get_available_providers", side_effect=fake_providers
        ), patch.object(
            ort_provider.ort, "InferenceSession", side_effect=_FakeCudaSess
        ):
            _, profile = ort_provider.create_inference_session(
                "model.onnx", device="cuda", strict_device=True
            )
        self.assertTrue(state["preloaded"])
        self.assertEqual(profile.active_backend, "cuda")

    def test_detect_available_backends_separates_declared_verified_and_missing_runtime(self):
        with patch.object(ort_provider, "preload_cuda_dlls_if_needed"), patch.object(
            ort_provider, "register_plugin_eps", return_value={}
        ), patch.object(
            ort_provider.ort,
            "get_available_providers",
            return_value=["CUDAExecutionProvider", "MIGraphXExecutionProvider", "CPUExecutionProvider"],
        ), patch.object(
            ort_provider, "_safe_ep_devices", return_value=[]
        ):
            missing_probe = {
                "declared": True,
                "provider_present": True,
                "expected_cuda_major": "13",
                "runtime_ready": False,
                "missing_libs": ["cudart64_13.dll", "cudnn64_9.dll"],
                "detected_other_cuda_major": "12",
                "detail": "CUDA 13 ランタイム / cuDNN 9 不足: cudart64_13.dll, cudnn64_9.dll",
                "hint": "CUDA Toolkit 13.x を導入してください",
            }
            with patch.object(ort_provider, "probe_cuda_runtime", return_value=missing_probe):
                avail_missing = ort_provider.detect_available_backends(refresh=True)
            self.assertFalse(avail_missing["cuda"]["available"])
            self.assertTrue(avail_missing["cuda"]["declared"])
            self.assertFalse(avail_missing["cuda"]["verified"])
            self.assertEqual(avail_missing["cuda"]["status"], "missing_runtime")
            self.assertEqual(avail_missing["cuda"]["missing_libs"], ["cudart64_13.dll", "cudnn64_9.dll"])
            self.assertTrue(avail_missing["migraphx"]["available"])
            self.assertEqual(avail_missing["migraphx"]["status"], "declared_only")
            self.assertIsNone(avail_missing["migraphx"]["verified"])
            self.assertEqual(avail_missing["cpu"]["status"], "verified")

            ready_probe = {
                "declared": True,
                "provider_present": True,
                "expected_cuda_major": "13",
                "runtime_ready": True,
                "missing_libs": [],
                "detected_other_cuda_major": None,
                "detail": "CUDAExecutionProvider (CUDA 13 + cuDNN 9 verified)",
                "hint": "",
            }
            with patch.object(ort_provider, "probe_cuda_runtime", return_value=ready_probe):
                avail_ready = ort_provider.detect_available_backends(refresh=True)
            self.assertTrue(avail_ready["cuda"]["available"])
            self.assertTrue(avail_ready["cuda"]["verified"])
            self.assertEqual(avail_ready["cuda"]["status"], "verified")

    def test_auto_device_records_missing_cuda_runtime_as_fallback_warning_and_strict_error(self):
        availability = {
            "cuda": {
                "available": False,
                "declared": True,
                "verified": False,
                "status": "missing_runtime",
                "mode": "monolithic",
                "devices": ["CUDAExecutionProvider"],
                "detail": "CUDA 13 ランタイム / cuDNN 9 不足: cudart64_13.dll（システムには CUDA 12 のみ検出）",
                "hint": "CUDA Toolkit 13.x と cuDNN 9.x を導入してください",
            },
            "migraphx": {"available": False, "status": "unavailable", "mode": "none", "devices": [], "detail": ""},
            "rocm": {"available": False, "status": "unavailable", "mode": "none", "devices": [], "detail": ""},
            "openvino": {"available": False, "status": "unavailable", "mode": "none", "devices": [], "detail": ""},
            "webgpu": {"available": False, "status": "unavailable", "mode": "none", "devices": [], "detail": ""},
            "dml": {"available": False, "status": "unavailable", "mode": "none", "devices": [], "detail": ""},
            "cpu": {"available": True, "status": "verified", "mode": "builtin", "devices": ["CPU"], "detail": "CPUExecutionProvider"},
        }
        stderr = io.StringIO()
        with patch.object(ort_provider, "detect_available_backends", return_value=availability), patch.object(
            ort_provider,
            "_create_session_for_backend",
            return_value=(object(), ["CPUExecutionProvider"]),
        ) as create_backend, contextlib.redirect_stderr(stderr):
            _, profile = ort_provider.create_inference_session(
                "model.onnx", device="auto", strict_device=False, model_label="DEIM"
            )
        create_backend.assert_called_once()
        self.assertEqual(profile.active_backend, "cpu")
        self.assertTrue(profile.fallback_occurred)
        self.assertIsNotNone(profile.warning_message)
        self.assertIn("cudart64_13.dll", profile.warning_message)
        self.assertIn("対処法:", profile.warning_message)

        with patch.object(ort_provider, "detect_available_backends", return_value=availability):
            with self.assertRaisesRegex(RuntimeError, "cudart64_13\\.dll"):
                ort_provider.create_inference_session(
                    "model.onnx", device="auto", strict_device=True, model_label="DEIM"
                )

    def test_format_device_diagnostics_reports_missing_cuda_runtime_and_cuda12_mismatch(self):
        def fake_can_load(filename, candidate_paths):
            if filename == "cudart64_12.dll":
                return True, None
            return False, "DLL not found"

        with patch.object(ort_provider.sys, "platform", "win32"), patch.object(
            ort_provider, "preload_cuda_dlls_if_needed"
        ), patch.object(
            ort_provider, "register_plugin_eps", return_value={}
        ), patch.object(
            ort_provider.ort, "get_available_providers", return_value=["CPUExecutionProvider"]
        ), patch.object(
            ort_provider, "_safe_ep_devices", return_value=[]
        ), patch.object(
            ort_provider,
            "_bundled_cuda_provider_path",
            return_value=Path("/fake/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll"),
        ), patch.object(
            ort_provider.ort, "cuda_version", "13.0", create=True
        ), patch.object(
            ort_provider, "_can_load_shared_lib", side_effect=fake_can_load
        ):
            report = ort_provider.format_device_diagnostics(refresh=True)

        self.assertIn("MISSING RUNTIME (declared)", report)
        self.assertNotIn("cuda     : READY", report)
        self.assertIn("CUDA runtime preflight:", report)
        self.assertIn("cudart64_13.dll", report)
        self.assertIn("cudnn64_9.dll", report)
        self.assertIn("CUDA 12 detected", report)
        self.assertIn("Selected by '--device auto': cpu", report)

    def test_register_bundled_nvidia_dll_dirs_and_plugin_dirs_on_windows(self):
        added = []
        with tempfile.TemporaryDirectory() as tmpdir:
            site = Path(tmpdir)
            ort_init = site / "onnxruntime" / "__init__.py"
            ort_capi = site / "onnxruntime" / "capi"
            cu13_bin = site / "nvidia" / "cu13" / "bin" / "x86_64"
            cudnn_bin = site / "nvidia" / "cudnn" / "bin"
            wgpu_dir = site / "onnxruntime_ep_webgpu"
            wgpu_d3d12 = wgpu_dir / "D3D12"
            for d in (ort_capi, cu13_bin, cudnn_bin, wgpu_d3d12):
                d.mkdir(parents=True, exist_ok=True)
            ort_init.write_text("", encoding="utf-8")
            plugin_lib = wgpu_dir / "onnxruntime_providers_webgpu.dll"
            plugin_lib.write_bytes(b"dll")

            fake_wgpu_mod = SimpleNamespace(get_library_path=lambda: str(plugin_lib))

            def fake_import(name):
                if name == "onnxruntime_ep_webgpu":
                    return fake_wgpu_mod
                raise ImportError(name)

            with patch.object(ort_provider.sys, "platform", "win32"), patch.object(
                ort_provider.platform, "machine", return_value="AMD64"
            ), patch.object(
                ort_provider.os, "add_dll_directory", side_effect=lambda p: (added.append(p), p)[1], create=True
            ), patch.object(
                ort_provider.ort, "__file__", str(ort_init)
            ):
                registered = ort_provider._register_bundled_nvidia_dll_dirs("13")
                # Second call must be idempotent without duplicate add_dll_directory calls
                ort_provider._register_bundled_nvidia_dll_dirs("13")
                self.assertIn(cu13_bin, registered)
                self.assertIn(cudnn_bin, registered)
                self.assertIn(ort_capi, registered)
                self.assertEqual(len(added), len(set(added)))
                self.assertIn(str(cu13_bin.resolve()), added)
                self.assertIn(str(cudnn_bin.resolve()), added)

                with patch.dict(ort_provider._REGISTERED_PLUGINS, {}, clear=True), patch.object(
                    ort_provider.importlib, "import_module", side_effect=fake_import
                ), patch.object(
                    ort_provider.ort, "register_execution_provider_library", create=True
                ) as mock_reg:
                    ort_provider.register_plugin_eps()
                mock_reg.assert_called_once_with("webgpu", str(plugin_lib))
                self.assertIn(str(wgpu_dir.resolve()), added)
                self.assertIn(str(wgpu_d3d12.resolve()), added)

    def test_probe_cuda_runtime_distinguishes_missing_driver_when_runtime_libs_load(self):
        def fake_can_load_win(filename, candidate_paths):
            if filename == "nvcuda.dll":
                return False, "nvcuda.dll not found"
            return True, None

        with patch.object(ort_provider.sys, "platform", "win32"), patch.object(
            ort_provider.ort, "get_available_providers", return_value=["CPUExecutionProvider"]
        ), patch.object(
            ort_provider,
            "_bundled_cuda_provider_path",
            return_value=Path("/fake/site-packages/onnxruntime/capi/onnxruntime_providers_cuda.dll"),
        ), patch.object(
            ort_provider.ort, "cuda_version", "13.0", create=True
        ), patch.object(
            ort_provider, "_detect_bundled_cuda_runtime", return_value=True
        ), patch.object(
            ort_provider, "_can_load_shared_lib", side_effect=fake_can_load_win
        ):
            probe = ort_provider.probe_cuda_runtime(refresh=True)

        self.assertTrue(probe["declared"])
        self.assertFalse(probe["runtime_ready"])
        self.assertEqual(probe["missing_libs"], [])
        self.assertTrue(probe["driver_missing"])
        self.assertEqual(probe["issue_code"], "cuda_driver_missing")
        self.assertIn("nvcuda.dll", probe["detail"])
        self.assertIn("R580", probe["hint"])

    def test_probe_webgpu_runtime_flags_missing_vulkan_and_plugin_errors(self):
        with patch.object(ort_provider.sys, "platform", "linux"), patch.dict(
            ort_provider._REGISTERED_PLUGINS,
            {"webgpu": "/fake/libonnxruntime_providers_webgpu.so"},
            clear=True,
        ), patch.object(
            ort_provider, "preload_cuda_dlls_if_needed"
        ), patch.object(
            ort_provider, "register_plugin_eps", return_value=dict(ort_provider._REGISTERED_PLUGINS)
        ), patch.object(
            ort_provider.ort, "get_available_providers", return_value=["CPUExecutionProvider"]
        ), patch.object(
            ort_provider, "_safe_ep_devices", return_value=[]
        ), patch.object(
            ort_provider, "_can_load_shared_lib", return_value=(False, "libvulkan.so.1: not found")
        ):
            avail = ort_provider.detect_available_backends(refresh=True)

        self.assertFalse(avail["webgpu"]["available"])
        self.assertEqual(avail["webgpu"]["status"], "missing_runtime")
        self.assertEqual(avail["webgpu"]["issue_code"], "webgpu_vulkan_missing")
        self.assertIn("libvulkan.so.1", avail["webgpu"]["detail"])

        stderr = io.StringIO()
        with patch.object(ort_provider, "detect_available_backends", return_value=avail), patch.object(
            ort_provider, "_create_session_for_backend", return_value=(object(), ["CPUExecutionProvider"])
        ), contextlib.redirect_stderr(stderr):
            _, profile = ort_provider.create_inference_session("model.onnx", device="auto")
        self.assertTrue(profile.fallback_occurred)
        self.assertEqual(profile.fallback_category, "webgpu_vulkan_missing")
        self.assertIn("libvulkan.so.1", profile.fallback_reason or "")

        # Plugin load error for OpenVINO / WebGPU is also surfaced as missing_runtime
        with patch.dict(
            ort_provider._REGISTERED_PLUGINS, {}, clear=True
        ), patch.dict(
            ort_provider._PLUGIN_ERRORS,
            {"webgpu": "DLL load failed", "openvino_ep": "OV init error"},
            clear=True,
        ), patch.object(
            ort_provider, "preload_cuda_dlls_if_needed"
        ), patch.object(
            ort_provider, "register_plugin_eps", return_value={}
        ), patch.object(
            ort_provider.ort, "get_available_providers", return_value=["CPUExecutionProvider"]
        ), patch.object(
            ort_provider, "_safe_ep_devices", return_value=[]
        ):
            err_avail = ort_provider.detect_available_backends(refresh=True)
        self.assertEqual(err_avail["webgpu"]["status"], "missing_runtime")
        self.assertEqual(err_avail["webgpu"]["issue_code"], "webgpu_plugin_error")
        self.assertEqual(err_avail["openvino"]["status"], "missing_runtime")
        self.assertEqual(err_avail["openvino"]["issue_code"], "openvino_plugin_error")

    def test_classify_backend_failure_and_profile_fields(self):
        self.assertEqual(
            ort_provider.classify_backend_failure("cuda", "cudaErrorInsufficientDriver")[0],
            "cuda_driver_old",
        )
        self.assertEqual(
            ort_provider.classify_backend_failure("cuda", "cudaErrorNoDevice: no CUDA-capable device")[0],
            "cuda_no_device",
        )
        self.assertEqual(
            ort_provider.classify_backend_failure("webgpu", "CompileShaderDXC E_INVALIDARG")[0],
            "webgpu_dxc_error",
        )
        self.assertEqual(
            ort_provider.classify_backend_failure("dml", "DmlExecutionProvider corrupted bounding boxes")[0],
            "dml_deim_unsupported",
        )

        with patch.object(
            ort_provider,
            "_create_session_for_backend",
            side_effect=[
                RuntimeError("CompileShaderDXC E_INVALIDARG"),
                (object(), ["CPUExecutionProvider"]),
            ],
        ), contextlib.redirect_stderr(io.StringIO()):
            _, profile = ort_provider.create_inference_session(
                "model.onnx", device="webgpu", strict_device=False
            )
        self.assertTrue(profile.fallback_occurred)
        self.assertEqual(profile.fallback_category, "webgpu_dxc_error")
        self.assertIn("DXC", profile.fallback_reason or "")
        self.assertTrue(profile.fallback_hint)


if __name__ == "__main__":
    unittest.main()
