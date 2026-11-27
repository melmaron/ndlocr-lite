import asyncio
import contextlib
import importlib.util
import io
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "ndlocr-lite-gui") not in sys.path:
    sys.path.insert(0, str(ROOT / "ndlocr-lite-gui"))


def _load_gui_app():
    spec = importlib.util.spec_from_file_location(
        "ndlocr_lite_gui", ROOT / "ndlocr-lite-gui" / "__init__.py"
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    spec.loader.exec_module(pkg)
    from ndlocr_lite_gui import gui_config, main

    return gui_config, main


class GuiGpuConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui_config, cls.app = _load_gui_app()

    def test_default_config_contains_engine_fields(self):
        cfg = self.gui_config.build_default_config()
        self.assertEqual(cfg["device"], "auto")
        self.assertEqual(cfg["device_id"], 0)
        self.assertEqual(cfg["batch_size"], 0)
        self.assertEqual(cfg["precision"], "fp32")

    def test_parse_gui_args_supports_gpu_options(self):
        args_default = self.app.parse_gui_args([], download_models=True)
        self.assertIsNone(args_default.device)
        self.assertIsNone(args_default.device_id)
        self.assertIsNone(args_default.batch_size)
        self.assertIsNone(args_default.precision)
        self.assertFalse(args_default.strict_device)

        args_custom = self.app.parse_gui_args(
            ["--device", "webgpu", "--device-id", "1", "--batch-size", "16", "--precision", "fp16", "--strict-device"],
            download_models=False,
        )
        self.assertEqual(args_custom.device, "webgpu")
        self.assertEqual(args_custom.device_id, 1)
        self.assertEqual(args_custom.batch_size, 16)
        self.assertEqual(args_custom.precision, "fp16")
        self.assertTrue(args_custom.strict_device)

    def test_resolve_gui_engine_args_prefers_cli_overrides_over_config(self):
        cfg = {"device": "webgpu", "device_id": 2, "batch_size": 16, "precision": "fp16"}
        args = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(args, cfg, cli_overrides={})
        self.assertEqual(args.device, "webgpu")
        self.assertEqual(args.device_id, 2)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.precision, "fp16")

        args2 = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(
            args2,
            cfg,
            cli_overrides={"device": "CUDA", "device_id": -5, "batch_size": "invalid", "precision": "FP32"},
        )
        self.assertEqual(args2.device, "cuda")
        self.assertEqual(args2.device_id, 0)
        self.assertEqual(args2.batch_size, 0)
        self.assertEqual(args2.precision, "fp32")

    def test_format_gui_engine_status_formats_backend_and_fallback(self):
        self.assertIn("読み込み中", self.app.format_gui_engine_status(None, None, "ja"))
        self.assertIn("Loading", self.app.format_gui_engine_status(None, None, "en"))

        det = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="GPU (CUDA)", fallback_occurred=False)
        )
        rec = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="GPU (CUDA)", fallback_occurred=False),
            batch_size=32,
            effective_precision="fp32",
        )
        status_cuda = self.app.format_gui_engine_status(det, rec, "ja")
        self.assertEqual(status_cuda, "実行エンジン: GPU (CUDA), batch=32")

        rec_fp16 = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="GPU (CUDA)", fallback_occurred=False),
            batch_size=64,
            effective_precision="fp16",
            requested_precision="fp16",
            precision_fallback_occurred=False,
        )
        self.assertEqual(
            self.app.format_gui_engine_status(det, rec_fp16, "ja"),
            "実行エンジン: GPU (CUDA), batch=64, fp16 (近似)",
        )
        self.assertEqual(
            self.app.format_gui_engine_status(det, rec_fp16, "en"),
            "Engine: GPU (CUDA), batch=64, fp16 (approx)",
        )

        rec_prec_fb = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="CPU", fallback_occurred=False),
            batch_size=1,
            effective_precision="fp32",
            requested_precision="fp16",
            precision_fallback_occurred=True,
        )
        self.assertEqual(
            self.app.format_gui_engine_status(det, rec_prec_fb, "ja"),
            "実行エンジン: CPU, fp16→fp32",
        )

        rec_fb = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="CPU", fallback_occurred=True),
            batch_size=1,
        )
        status_fb = self.app.format_gui_engine_status(det, rec_fb, "en")
        self.assertEqual(status_fb, "Engine: CPU [fallback]")

    def test_resolve_gui_engine_args_sanitizes_unknown_device(self):
        args = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(
            args, {"device": "bogus-device", "device_id": -1, "batch_size": -3, "precision": "int8"}, {}
        )
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.device_id, 0)
        self.assertEqual(args.batch_size, 0)
        self.assertEqual(args.precision, "fp32")

    def test_resolve_gui_engine_args_tolerates_non_string_device(self):
        args = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(args, {"device": 123, "device_id": 0, "batch_size": 0, "precision": 123}, {})
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.device_id, 0)
        self.assertEqual(args.batch_size, 0)
        self.assertEqual(args.precision, "fp32")

    def test_busy_tracker_nests_and_never_goes_negative(self):
        tracker = self.app.BusyTracker()
        self.assertFalse(tracker.active)
        tracker.enter()
        tracker.enter()
        self.assertTrue(tracker.active)
        tracker.leave()
        self.assertTrue(tracker.active)
        tracker.leave()
        self.assertFalse(tracker.active)
        tracker.leave()
        self.assertFalse(tracker.active)

    def test_parse_pdf_dpi_clamps_invalid_values(self):
        for value, expected in [
            ("300", 300.0),
            (150, 150.0),
            ("", 300.0),
            (None, 300.0),
            ("abc", 300.0),
            ("0", 300.0),
            ("-5", 300.0),
            ("inf", 300.0),
            ("Infinity", 300.0),
            ("-inf", 300.0),
            ("1e999", 300.0),
            ("nan", 300.0),
            ("0.1", 1.0),
        ]:
            with self.subTest(value=value):
                self.assertEqual(self.app.parse_pdf_dpi(value), expected)

    def test_load_user_config_is_defensive_and_normalizes_langcode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "userconf.yaml"
            defaults = {"langcode": "ja", "device": "auto"}
            cfg, warning = self.app.load_user_config(path, defaults)
            self.assertIsNone(warning)
            self.assertEqual(cfg, defaults)

            path.write_text("a: [1, 2", encoding="utf-8")
            cfg, warning = self.app.load_user_config(path, defaults)
            self.assertIsNotNone(warning)
            self.assertEqual(cfg["langcode"], "ja")

            path.write_text("- just\n- a list\n", encoding="utf-8")
            cfg, warning = self.app.load_user_config(path, defaults)
            self.assertIsNotNone(warning)
            self.assertEqual(cfg["langcode"], "ja")

            path.write_text("langcode: en\ndevice: cuda\n", encoding="utf-8")
            cfg, warning = self.app.load_user_config(path, defaults)
            self.assertIsNone(warning)
            self.assertEqual(cfg["langcode"], "en")
            self.assertEqual(cfg["device"], "cuda")

            path.write_text("langcode: fr\ndevice: cuda\n", encoding="utf-8")
            cfg, warning = self.app.load_user_config(path, defaults)
            self.assertIsNone(warning)
            self.assertEqual(cfg["langcode"], "ja")  # unsupported locale falls back
            self.assertEqual(cfg["device"], "cuda")

    def test_load_user_config_tolerates_unhashable_langcode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "userconf.yaml"
            defaults = {"langcode": "ja", "device": "auto"}
            for payload in ("langcode: [ja]\n", "langcode: {a: 1}\n"):
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    cfg, warning = self.app.load_user_config(path, defaults)
                    self.assertIsNone(warning)
                    self.assertEqual(cfg["langcode"], "ja")

    def test_busy_tracker_fires_on_idle_only_when_outermost_task_finishes(self):
        events = []
        tracker = self.app.BusyTracker(on_idle=lambda: events.append("idle"))
        tracker.leave()  # 0 -> 0 must not trigger idle callback
        self.assertEqual(events, [])
        tracker.enter()
        tracker.enter()
        tracker.leave()  # 2 -> 1 still active
        self.assertTrue(tracker.active)
        self.assertEqual(events, [])
        tracker.leave()  # 1 -> 0 fires once
        self.assertFalse(tracker.active)
        self.assertEqual(events, ["idle"])

    def test_save_user_config_is_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "userconf.yaml"
            payload = {"langcode": "en", "device": "webgpu", "batch_size": 16}
            self.app.save_user_config(path, payload)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            import yaml

            with path.open(encoding="utf-8") as stream:
                self.assertEqual(yaml.safe_load(stream), payload)

    def test_drop_changed_cli_overrides_keeps_untouched_keys(self):
        overrides = {"device": "cuda", "batch_size": 32, "device_id": 1}
        self.app.drop_changed_cli_overrides(
            overrides,
            {"device": "cuda", "batch_size": 32, "device_id": 1},
            {"device": "cuda", "batch_size": 8, "device_id": 1},
        )
        self.assertEqual(overrides, {"device": "cuda", "device_id": 1})

    def test_load_ocr_models_parallel_on_cpu_and_cuda_serial_on_other_gpus(self):
        for parallel_dev in ("cpu", "cuda"):
            with self.subTest(device=parallel_dev):
                fake_ocr = Mock()
                fake_ocr.get_detector.return_value = "det"
                fake_ocr.get_recognizer.side_effect = ["r100", "r30", "r50"]
                args_par = self.app.parse_gui_args(["--device", parallel_dev])
                with patch.object(self.app, "load_runtime_modules"), patch.object(
                    self.app, "ocr", fake_ocr
                ), patch.object(self.app, "prepare_backend_for_parallel_init"), patch.object(
                    self.app, "ThreadPoolExecutor"
                ) as pool_cls:
                    pool_inst = pool_cls.return_value.__enter__.return_value
                    pool_inst.submit.side_effect = [
                        SimpleNamespace(result=lambda: "det"),
                        SimpleNamespace(result=lambda: "r100"),
                        SimpleNamespace(result=lambda: "r30"),
                        SimpleNamespace(result=lambda: "r50"),
                    ]
                    res = self.app.load_ocr_models(args_par)
                    self.assertEqual(res, ("det", "r100", "r30", "r50"))
                    pool_cls.assert_called_once()

        for serial_dev in ("webgpu", "dml"):
            with self.subTest(device=serial_dev):
                fake_ocr2 = Mock()
                fake_ocr2.get_detector.return_value = "det"
                fake_ocr2.get_recognizer.side_effect = ["r100", "r30", "r50"]
                args_ser = self.app.parse_gui_args(["--device", serial_dev])
                with patch.object(self.app, "load_runtime_modules"), patch.object(
                    self.app, "ocr", fake_ocr2
                ), patch.object(self.app, "prepare_backend_for_parallel_init"), patch.object(
                    self.app, "ThreadPoolExecutor"
                ) as pool_cls:
                    res = self.app.load_ocr_models(args_ser)
                    self.assertEqual(res, ("det", "r100", "r30", "r50"))
                    pool_cls.assert_not_called()

        fake_ocr3 = Mock()
        fake_ocr3.get_detector.return_value = "det"
        fake_ocr3.get_recognizer.side_effect = ["r100", "r30", "r50"]
        args_auto = self.app.parse_gui_args([])
        with patch.object(self.app, "load_runtime_modules"), patch.object(
            self.app, "ocr", fake_ocr3
        ), patch.object(self.app, "prepare_backend_for_parallel_init"), patch(
            "ort_provider.resolve_auto_backend_name", return_value="webgpu"
        ), patch.object(self.app, "ThreadPoolExecutor") as pool_cls:
            res = self.app.load_ocr_models(args_auto)
            self.assertEqual(res, ("det", "r100", "r30", "r50"))
            pool_cls.assert_not_called()

    def test_compute_engine_reload_scope(self):
        base = {"device": "cuda", "device_id": 0, "batch_size": 64, "precision": "fp32"}
        self.assertEqual(self.app.compute_engine_reload_scope(None, base), "all")
        self.assertEqual(self.app.compute_engine_reload_scope(base, dict(base)), "none")
        self.assertEqual(
            self.app.compute_engine_reload_scope(base, {**base, "precision": "fp16"}),
            "recognizers",
        )
        self.assertEqual(
            self.app.compute_engine_reload_scope(base, {**base, "batch_size": 32}),
            "recognizers",
        )
        self.assertEqual(
            self.app.compute_engine_reload_scope(base, {**base, "device": "cpu"}),
            "all",
        )
        self.assertEqual(
            self.app.compute_engine_reload_scope(base, {**base, "device_id": 1}),
            "all",
        )

    def test_load_ocr_models_reuses_detector_when_requested(self):
        fake_ocr = Mock()
        fake_ocr.get_recognizer.side_effect = ["r100", "r30", "r50"]
        args_par = self.app.parse_gui_args(["--device", "cuda", "--precision", "fp16"])
        with patch.object(self.app, "load_runtime_modules"), patch.object(
            self.app, "ocr", fake_ocr
        ), patch.object(self.app, "prepare_backend_for_parallel_init"), patch.object(
            self.app, "ThreadPoolExecutor"
        ) as pool_cls:
            pool_inst = pool_cls.return_value.__enter__.return_value
            pool_inst.submit.side_effect = [
                SimpleNamespace(result=lambda: "r100"),
                SimpleNamespace(result=lambda: "r30"),
                SimpleNamespace(result=lambda: "r50"),
            ]
            res = self.app.load_ocr_models(args_par, reuse_detector="existing_det")
            self.assertEqual(res, ("existing_det", "r100", "r30", "r50"))
            fake_ocr.get_detector.assert_not_called()
            pool_cls.assert_called_once_with(max_workers=3, thread_name_prefix="model-init")

        fake_ocr_ser = Mock()
        fake_ocr_ser.get_recognizer.side_effect = ["r100", "r30", "r50"]
        args_ser = self.app.parse_gui_args(["--device", "webgpu", "--precision", "fp16"])
        with patch.object(self.app, "load_runtime_modules"), patch.object(
            self.app, "ocr", fake_ocr_ser
        ), patch.object(self.app, "prepare_backend_for_parallel_init"), patch.object(
            self.app, "ThreadPoolExecutor"
        ) as pool_cls:
            res_ser = self.app.load_ocr_models(args_ser, reuse_detector="existing_det")
            self.assertEqual(res_ser, ("existing_det", "r100", "r30", "r50"))
            fake_ocr_ser.get_detector.assert_not_called()
            pool_cls.assert_not_called()

    def test_gui_warmup_gpu_model_is_non_fatal_on_exception(self):
        class _BoomModel:
            backend_profile = SimpleNamespace(active_backend="cuda")

            def warmup(self):
                raise RuntimeError("Boom OOM during GUI warmup")

        model = _BoomModel()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            res = self.app._warmup_gpu_model(model)
        self.assertIs(res, model)
        self.assertIn("warm-up skipped", stderr.getvalue())

    def test_collect_engine_warnings_includes_detector_fallback_and_table_error(self):
        det = SimpleNamespace(
            backend_profile=SimpleNamespace(
                fallback_occurred=True,
                warning_message="[WARN] DEIM CPU fallback",
            )
        )
        rec = SimpleNamespace(
            backend_profile=SimpleNamespace(
                fallback_occurred=False,
                warning_message=None,
            )
        )
        rec30 = SimpleNamespace(
            backend_profile=SimpleNamespace(
                fallback_occurred=True,
                warning_message="[WARN] DEIM CPU fallback",
            )
        )
        summary = self.app.collect_engine_warnings(
            det,
            rec,
            rec30,
            table_error=RuntimeError("TableCenterNet requires ORT >= 1.25"),
        )
        self.assertTrue(summary["fallback_occurred"])
        self.assertEqual(summary["warnings"], ["[WARN] DEIM CPU fallback"])
        self.assertIn("ORT >= 1.25", summary["table_warning"])

    def test_format_gui_engine_status_reflects_detector_only_fallback(self):
        det = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="CPU", fallback_occurred=True)
        )
        rec = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="GPU (DirectML)", fallback_occurred=False),
            batch_size=16,
            effective_precision="fp32",
        )
        status = self.app.format_gui_engine_status(det, rec, "ja")
        self.assertEqual(status, "実行エンジン: GPU (DirectML), batch=16 [fallback]")

    def test_configure_gui_logging_writes_log_and_is_idempotent(self):
        import ort_provider

        parent_logger = logging.getLogger("ndlocr")
        orig_handlers = list(parent_logger.handlers)
        orig_log_path = self.app.GUI_LOG_PATH
        ort_provider._WARNED_KEYS.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                p1 = self.app.configure_gui_logging(log_dir=tmpdir)
                p2 = self.app.configure_gui_logging(log_dir=tmpdir)
                self.assertEqual(p1, p2)
                gui_handlers = [
                    h for h in parent_logger.handlers
                    if getattr(h, self.app._LOG_HANDLER_ATTR, None) is not None
                ]
                self.assertEqual(len(gui_handlers), 1)
                self.app.LOGGER.info("gui test message")
                with contextlib.redirect_stderr(io.StringIO()):
                    ort_provider._emit_warning_once(("test", "gui", "log"), "ort test warning")
                for h in gui_handlers:
                    h.flush()
                contents = p1.read_text(encoding="utf-8")
                self.assertIn("gui test message", contents)
                self.assertIn("ort test warning", contents)
            finally:
                for h in list(parent_logger.handlers):
                    if h not in orig_handlers:
                        parent_logger.removeHandler(h)
                        h.close()
                self.app.GUI_LOG_PATH = orig_log_path

    def test_build_gui_diagnostics_report_contains_profiles_diagnostics_and_log_path(self):
        det = SimpleNamespace(
            backend_profile=SimpleNamespace(
                active_backend="cpu",
                active_providers=["CPUExecutionProvider"],
                fallback_occurred=True,
                warning_message="CUDA 13 missing cudart64_13.dll",
            )
        )
        rec = SimpleNamespace(
            backend_profile=SimpleNamespace(
                active_backend="cpu",
                active_providers=["CPUExecutionProvider"],
                fallback_occurred=True,
                warning_message="CUDA 13 missing cudart64_13.dll",
            )
        )
        fake_log_path = Path("/tmp/ndlocr-lite.log")
        with patch.object(
            self.app, "format_device_diagnostics", return_value="=== DIAG ===\nSelected: cpu"
        ):
            report = self.app.build_gui_diagnostics_report(
                detector=det,
                recognizer=rec,
                table_error="DeformConv unsupported",
                log_path=fake_log_path,
                refresh=True,
            )
        self.assertIn("Active sessions:", report)
        self.assertIn("DEIMv2 (detector): cpu", report)
        self.assertIn("CUDA 13 missing cudart64_13.dll", report)
        self.assertIn("DeformConv unsupported", report)
        self.assertIn(f"Log file: {fake_log_path}", report)
        self.assertIn("=== DIAG ===", report)

    def test_build_device_dropdown_options_distinguishes_missing_runtime(self):
        avail = {
            "cuda": {"available": False, "status": "missing_runtime"},
            "openvino": {"available": True, "status": "verified"},
            "webgpu": {"available": False, "status": "unavailable"},
        }
        ja_opts = {o.key: o.text for o in self.app.build_device_dropdown_options(avail, True)}
        en_opts = {o.key: o.text for o in self.app.build_device_dropdown_options(avail, False)}
        self.assertIn("ランタイム不足", ja_opts["cuda"])
        self.assertIn("利用可能", ja_opts["openvino"])
        self.assertIn("未検出", ja_opts["webgpu"])
        self.assertIn("missing runtime", en_opts["cuda"])
        self.assertIn("ready", en_opts["openvino"])
        self.assertIn("unavailable", en_opts["webgpu"])

    def test_open_preview_link_handles_win32_darwin_linux_and_http(self):
        from urllib.request import url2pathname

        event = SimpleNamespace(data="file:///tmp/sample_table.html")
        expected_local = url2pathname("/tmp/sample_table.html")
        with patch.object(self.app.sys, "platform", "win32"), patch.object(
            self.app.os, "startfile", create=True
        ) as mock_startfile:
            asyncio.run(self.app.open_preview_link(event))
        mock_startfile.assert_called_once_with(expected_local)

        with patch.object(self.app.sys, "platform", "darwin"), patch.object(
            self.app.subprocess, "run"
        ) as mock_run:
            asyncio.run(self.app.open_preview_link(event))
        mock_run.assert_called_once_with(["open", expected_local], check=True)

        with patch.object(self.app.sys, "platform", "linux"), patch.object(
            self.app.subprocess, "run"
        ) as mock_run:
            asyncio.run(self.app.open_preview_link(event))
        mock_run.assert_called_once_with(["xdg-open", expected_local], check=True)

        with patch.object(self.app.sys, "platform", "linux"), patch.object(
            self.app.subprocess, "run", side_effect=FileNotFoundError("no xdg-open")
        ), patch("webbrowser.open", return_value=True) as mock_wb:
            asyncio.run(self.app.open_preview_link(event))
        mock_wb.assert_called_once_with("file:///tmp/sample_table.html")

        with patch("webbrowser.open", return_value=True) as mock_http_wb:
            asyncio.run(self.app.open_preview_link(SimpleNamespace(data="https://example.com")))
        mock_http_wb.assert_called_once_with("https://example.com")

    def test_get_display_scale_factor_uses_page_media_and_env_on_non_windows(self):
        with patch.object(self.app.sys, "platform", "linux"):
            page = SimpleNamespace(media=SimpleNamespace(device_pixel_ratio=2.0))
            self.assertEqual(self.app.get_display_scale_factor(page), 2.0)

            with patch.dict(self.app.os.environ, {"GDK_SCALE": "1.5"}, clear=False):
                self.assertEqual(
                    self.app.get_display_scale_factor(SimpleNamespace(media=None)), 1.5
                )

            with patch.dict(self.app.os.environ, {}, clear=True):
                self.assertEqual(self.app.get_display_scale_factor(None), 1.0)


if __name__ == "__main__":
    unittest.main()
