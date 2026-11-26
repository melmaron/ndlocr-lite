import importlib.util
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

    def test_parse_gui_args_supports_gpu_options(self):
        args_default = self.app.parse_gui_args([], download_models=True)
        self.assertIsNone(args_default.device)
        self.assertIsNone(args_default.device_id)
        self.assertIsNone(args_default.batch_size)
        self.assertFalse(args_default.strict_device)

        args_custom = self.app.parse_gui_args(
            ["--device", "webgpu", "--device-id", "1", "--batch-size", "16", "--strict-device"],
            download_models=False,
        )
        self.assertEqual(args_custom.device, "webgpu")
        self.assertEqual(args_custom.device_id, 1)
        self.assertEqual(args_custom.batch_size, 16)
        self.assertTrue(args_custom.strict_device)

    def test_resolve_gui_engine_args_prefers_cli_overrides_over_config(self):
        cfg = {"device": "webgpu", "device_id": 2, "batch_size": 16}
        args = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(args, cfg, cli_overrides={})
        self.assertEqual(args.device, "webgpu")
        self.assertEqual(args.device_id, 2)
        self.assertEqual(args.batch_size, 16)

        args2 = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(
            args2,
            cfg,
            cli_overrides={"device": "CUDA", "device_id": -5, "batch_size": "invalid"},
        )
        self.assertEqual(args2.device, "cuda")
        self.assertEqual(args2.device_id, 0)
        self.assertEqual(args2.batch_size, 0)

    def test_format_gui_engine_status_formats_backend_and_fallback(self):
        self.assertIn("読み込み中", self.app.format_gui_engine_status(None, None, "ja"))
        self.assertIn("Loading", self.app.format_gui_engine_status(None, None, "en"))

        det = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="GPU (CUDA)", fallback_occurred=False)
        )
        rec = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="GPU (CUDA)", fallback_occurred=False),
            batch_size=32,
        )
        status_cuda = self.app.format_gui_engine_status(det, rec, "ja")
        self.assertEqual(status_cuda, "実行エンジン: GPU (CUDA), batch=32")

        rec_fb = SimpleNamespace(
            backend_profile=SimpleNamespace(display_name="CPU", fallback_occurred=True),
            batch_size=1,
        )
        status_fb = self.app.format_gui_engine_status(det, rec_fb, "en")
        self.assertEqual(status_fb, "Engine: CPU [fallback]")

    def test_resolve_gui_engine_args_sanitizes_unknown_device(self):
        args = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(
            args, {"device": "bogus-device", "device_id": -1, "batch_size": -3}, {}
        )
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.device_id, 0)
        self.assertEqual(args.batch_size, 0)

    def test_resolve_gui_engine_args_tolerates_non_string_device(self):
        args = self.app.parse_gui_args([], download_models=True)
        self.app.resolve_gui_engine_args(args, {"device": 123, "device_id": 0, "batch_size": 0}, {})
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.device_id, 0)
        self.assertEqual(args.batch_size, 0)

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

    def test_load_ocr_models_parallel_on_cpu_serial_on_gpu(self):
        fake_ocr = Mock()
        fake_ocr.get_detector.return_value = "det"
        fake_ocr.get_recognizer.side_effect = ["r100", "r30", "r50"]

        args_cpu = self.app.parse_gui_args(["--device", "cpu"])
        with patch.object(self.app, "load_runtime_modules"), patch.object(
            self.app, "ocr", fake_ocr
        ), patch.object(self.app, "ThreadPoolExecutor") as pool_cls:
            pool_inst = pool_cls.return_value.__enter__.return_value
            pool_inst.submit.side_effect = [
                SimpleNamespace(result=lambda: "det"),
                SimpleNamespace(result=lambda: "r100"),
                SimpleNamespace(result=lambda: "r30"),
                SimpleNamespace(result=lambda: "r50"),
            ]
            res = self.app.load_ocr_models(args_cpu)
            self.assertEqual(res, ("det", "r100", "r30", "r50"))
            pool_cls.assert_called_once()

        fake_ocr2 = Mock()
        fake_ocr2.get_detector.return_value = "det"
        fake_ocr2.get_recognizer.side_effect = ["r100", "r30", "r50"]
        args_auto = self.app.parse_gui_args([])
        with patch.object(self.app, "load_runtime_modules"), patch.object(
            self.app, "ocr", fake_ocr2
        ), patch.object(self.app, "ThreadPoolExecutor") as pool_cls:
            res = self.app.load_ocr_models(args_auto)
            self.assertEqual(res, ("det", "r100", "r30", "r50"))
            pool_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
