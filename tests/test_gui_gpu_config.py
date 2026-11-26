import importlib.util
from pathlib import Path
import sys
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
