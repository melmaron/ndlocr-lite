"""Regression tests for Flet build artifact verification and cross-build workflow."""

from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import verify_flet_artifact  # noqa: E402


def _populate_models(base_dir: Path) -> None:
    model_dir = base_dir / "app" / "src" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_name in verify_flet_artifact.REQUIRED_MODELS:
        (model_dir / model_name).write_bytes(b"onnx")


class VerifyFletArtifactTest(unittest.TestCase):
    def test_windows_cuda_passes_with_models_provider_and_python312_dll(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _populate_models(root)
            (root / "python312.dll").write_bytes(b"py")
            capi = root / "site-packages" / "onnxruntime" / "capi"
            capi.mkdir(parents=True, exist_ok=True)
            (capi / "onnxruntime_providers_cuda.dll").write_bytes(b"cuda")

            self.assertEqual(
                verify_flet_artifact.verify_artifact(
                    root,
                    target_platform="windows",
                    variant="cuda",
                    python_version="3.12",
                    arch="AMD64",
                ),
                [],
            )
            self.assertEqual(
                verify_flet_artifact.main(
                    [
                        "--output-dir",
                        str(root),
                        "--platform",
                        "windows",
                        "--variant",
                        "cuda",
                        "--python-version",
                        "3.12",
                        "--arch",
                        "AMD64",
                    ]
                ),
                0,
            )

    def test_windows_cuda_fails_when_provider_python_dll_or_model_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _populate_models(root)
            (root / "python313.dll").write_bytes(b"py313")
            (root / "app" / "src" / "model" / verify_flet_artifact.REQUIRED_MODELS[0]).unlink()

            errors = verify_flet_artifact.verify_artifact(
                root,
                target_platform="windows",
                variant="cuda",
                python_version="3.12",
                arch="AMD64",
            )
            joined = "\n".join(errors)
            self.assertIn("Missing bundled ONNX model:", joined)
            self.assertIn("python312.dll", joined)
            self.assertIn("onnxruntime_providers_cuda.dll", joined)
            self.assertEqual(
                verify_flet_artifact.main(
                    [
                        "--output-dir",
                        str(root),
                        "--platform",
                        "windows",
                        "--variant",
                        "cuda",
                        "--python-version",
                        "3.12",
                    ]
                ),
                1,
            )

    def test_linux_gpu_generic_checks_webgpu_and_arch_specific_openvino(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _populate_models(root)
            webgpu_dir = root / "site-packages" / "onnxruntime_ep_webgpu"
            webgpu_dir.mkdir(parents=True, exist_ok=True)
            (webgpu_dir / "libonnxruntime_providers_webgpu.so").write_bytes(b"wgpu")

            # x86_64 requires OpenVINO plugin as well
            errors_x86 = verify_flet_artifact.verify_artifact(
                root,
                target_platform="linux",
                variant="gpu-generic",
                arch="x86_64",
            )
            self.assertTrue(any("OpenVINO" in err for err in errors_x86))

            # ARM64 does not include OpenVINO
            self.assertEqual(
                verify_flet_artifact.verify_artifact(
                    root,
                    target_platform="linux",
                    variant="gpu-generic",
                    arch="aarch64",
                ),
                [],
            )

            ov_dir = root / "site-packages" / "onnxruntime_ep_openvino"
            ov_dir.mkdir(parents=True, exist_ok=True)
            (ov_dir / "libonnxruntime_providers_openvino_plugin.so").write_bytes(b"ov")
            self.assertEqual(
                verify_flet_artifact.verify_artifact(
                    root,
                    target_platform="linux",
                    variant="gpu-generic",
                    arch="x86_64",
                ),
                [],
            )

    def test_build_flet_cross_workflow_drops_macos_pins_win_python_and_verifies_artifacts(self):
        wf_path = ROOT / ".github" / "workflows" / "build-flet-cross.yml"
        workflow = yaml.safe_load(wf_path.read_text(encoding="utf-8"))

        matrix = workflow["jobs"]["cross-build"]["strategy"]["matrix"]["include"]
        self.assertEqual(len(matrix), 4)
        platforms = {entry["target-platform"] for entry in matrix}
        self.assertEqual(platforms, {"windows", "linux"})

        create_notes_step = next(
            s for s in workflow["jobs"]["create-release"]["steps"] if "run" in s
        )
        self.assertIn("CUDA 13", create_notes_step["run"])
        self.assertIn("非同梱", create_notes_step["run"])

        build_steps = workflow["jobs"]["cross-build"]["steps"]
        win_build = next(s for s in build_steps if s.get("name") == "Flet build(Windows)")
        self.assertIn("--python-version 3.12", win_build["run"])

        verify_step = next(
            s for s in build_steps if s.get("name") == "Verify build artifact contents"
        )
        self.assertIn("scripts/verify_flet_artifact.py", verify_step["run"])


if __name__ == "__main__":
    unittest.main()
