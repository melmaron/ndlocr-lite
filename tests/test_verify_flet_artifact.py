"""Regression tests for Flet build artifact verification and cross-build workflow."""

from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import package_flet_artifact  # noqa: E402
import verify_flet_artifact  # noqa: E402


def _populate_models(base_dir: Path) -> None:
    model_dir = base_dir / "app" / "src" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_name in verify_flet_artifact.REQUIRED_MODELS:
        (model_dir / model_name).write_bytes(b"onnx")


def _populate_windows_crt(base_dir: Path) -> None:
    for dll_name in verify_flet_artifact.WINDOWS_REQUIRED_CRT_DLLS:
        (base_dir / dll_name).write_bytes(b"crt")


def _populate_windows_cuda_runtime(base_dir: Path) -> None:
    capi = base_dir / "site-packages" / "onnxruntime" / "capi"
    capi.mkdir(parents=True, exist_ok=True)
    (capi / "onnxruntime_providers_cuda.dll").write_bytes(b"cuda")

    cu13 = base_dir / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
    cu13.mkdir(parents=True, exist_ok=True)
    for name in ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "nvrtc64_130_0.dll"):
        (cu13 / name).write_bytes(b"nv")

    cudnn = base_dir / "site-packages" / "nvidia" / "cudnn" / "bin"
    cudnn.mkdir(parents=True, exist_ok=True)
    for name in ("cudnn64_9.dll", "cudnn_ops64_9.dll"):
        (cudnn / name).write_bytes(b"cudnn")


def _populate_linux_cuda_runtime(base_dir: Path) -> None:
    capi = base_dir / "site-packages" / "onnxruntime" / "capi"
    capi.mkdir(parents=True, exist_ok=True)
    (capi / "libonnxruntime_providers_cuda.so").write_bytes(b"cuda")

    cu13 = base_dir / "site-packages" / "nvidia" / "cu13" / "lib"
    cu13.mkdir(parents=True, exist_ok=True)
    for name in (
        "libcudart.so.13",
        "libcublas.so.13",
        "libcublasLt.so.13",
        "libcurand.so.10",
        "libnvrtc.so.13",
    ):
        (cu13 / name).write_bytes(b"nv")

    cudnn = base_dir / "site-packages" / "nvidia" / "cudnn" / "lib"
    cudnn.mkdir(parents=True, exist_ok=True)
    for name in ("libcudnn.so.9", "libcudnn_ops.so.9"):
        (cudnn / name).write_bytes(b"cudnn")


class VerifyFletArtifactTest(unittest.TestCase):
    def test_windows_cuda_passes_with_models_provider_crt_cuda13_and_python312_dll(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _populate_models(root)
            (root / "python312.dll").write_bytes(b"py")
            _populate_windows_crt(root)
            _populate_windows_cuda_runtime(root)

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

    def test_windows_cuda_fails_when_provider_python_crt_cuda13_or_model_missing(self):
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
            self.assertIn("msvcp140_1.dll", joined)
            self.assertIn("msvcp140_atomic_wait.dll", joined)
            self.assertIn("onnxruntime_providers_cuda.dll", joined)
            self.assertIn("cudart64_13*.dll", joined)
            self.assertIn("cudnn64_9*.dll", joined)
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

    def test_linux_cuda_checks_bundled_cuda13_curand_and_cudnn9(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _populate_models(root)
            _populate_linux_cuda_runtime(root)
            self.assertEqual(
                verify_flet_artifact.verify_artifact(
                    root, target_platform="linux", variant="cuda", arch="x86_64"
                ),
                [],
            )

            (root / "site-packages" / "nvidia" / "cu13" / "lib" / "libcurand.so.10").unlink()
            errors = verify_flet_artifact.verify_artifact(
                root, target_platform="linux", variant="cuda", arch="x86_64"
            )
            self.assertTrue(any("libcurand.so.10*" in err for err in errors))

    def test_linux_gpu_generic_checks_webgpu_and_arch_specific_openvino(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _populate_models(root)
            webgpu_dir = root / "site-packages" / "onnxruntime_ep_webgpu"
            webgpu_dir.mkdir(parents=True, exist_ok=True)
            (webgpu_dir / "libonnxruntime_providers_webgpu.so").write_bytes(b"wgpu")

            # x86_64 requires OpenVINO + TBB plugins as well
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
            for name in (
                "libonnxruntime_providers_openvino_plugin.so",
                "libopenvino.so.2641",
                "libtbb.so.12",
            ):
                (ov_dir / name).write_bytes(b"ov")
            self.assertEqual(
                verify_flet_artifact.verify_artifact(
                    root,
                    target_platform="linux",
                    variant="gpu-generic",
                    arch="x86_64",
                ),
                [],
            )

    def test_windows_gpu_generic_checks_crt_dxc_d3d12_openvino_and_tbb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _populate_models(root)
            (root / "python312.dll").write_bytes(b"py")
            _populate_windows_crt(root)

            wgpu = root / "site-packages" / "onnxruntime_ep_webgpu"
            (wgpu / "D3D12").mkdir(parents=True, exist_ok=True)
            (wgpu / "onnxruntime_providers_webgpu.dll").write_bytes(b"wgpu")
            (wgpu / "dxcompiler.dll").write_bytes(b"dxc")
            (wgpu / "D3D12" / "D3D12Core.dll").write_bytes(b"d3d12")

            ov = root / "site-packages" / "onnxruntime_ep_openvino"
            ov.mkdir(parents=True, exist_ok=True)
            for name in ("onnxruntime_providers_openvino_plugin.dll", "openvino.dll", "tbb12.dll"):
                (ov / name).write_bytes(b"ov")

            self.assertEqual(
                verify_flet_artifact.verify_artifact(
                    root,
                    target_platform="windows",
                    variant="gpu-generic",
                    python_version="3.12",
                    arch="AMD64",
                ),
                [],
            )

            (root / "msvcp140_atomic_wait.dll").unlink()
            errors = verify_flet_artifact.verify_artifact(
                root,
                target_platform="windows",
                variant="gpu-generic",
                python_version="3.12",
                arch="AMD64",
            )
            self.assertTrue(any("msvcp140_atomic_wait.dll" in e for e in errors))

    def test_package_flet_artifact_creates_zip_and_targz_and_enforces_size_limit(self):
        import tarfile
        import zipfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            win_dir = root / "windows"
            win_dir.mkdir()
            (win_dir / "hello.txt").write_bytes(b"hello world" * 100)

            zip_out = root / "bundle.zip"
            rc_zip = package_flet_artifact.main(
                ["--source-dir", str(win_dir), "--output", str(zip_out)]
            )
            self.assertEqual(rc_zip, 0)
            self.assertTrue(zip_out.is_file())
            with zipfile.ZipFile(zip_out) as zf:
                self.assertIn("windows/hello.txt", zf.namelist())

            tgz_out = root / "bundle.tar.gz"
            rc_tgz = package_flet_artifact.main(
                ["--source-dir", str(win_dir), "--output", str(tgz_out)]
            )
            self.assertEqual(rc_tgz, 0)
            with tarfile.open(tgz_out, "r:gz") as tf:
                self.assertIn("windows/hello.txt", tf.getnames())

            self.assertEqual(
                verify_flet_artifact.verify_archive_size(zip_out, max_bytes=10_000_000),
                [],
            )
            too_big = verify_flet_artifact.verify_archive_size(zip_out, max_bytes=10)
            self.assertTrue(too_big)
            rc_fail = package_flet_artifact.main(
                ["--source-dir", str(win_dir), "--output", str(zip_out), "--max-bytes", "10"]
            )
            self.assertEqual(rc_fail, 1)

    def test_build_flet_cross_workflow_bundles_runtimes_crt_and_uses_safe_packager(self):
        wf_path = ROOT / ".github" / "workflows" / "build-flet-cross.yml"
        workflow = yaml.safe_load(wf_path.read_text(encoding="utf-8"))

        matrix = workflow["jobs"]["cross-build"]["strategy"]["matrix"]["include"]
        self.assertEqual(len(matrix), 4)
        platforms = {entry["target-platform"] for entry in matrix}
        self.assertEqual(platforms, {"windows", "linux"})

        create_notes_step = next(
            s for s in workflow["jobs"]["create-release"]["steps"] if "run" in s
        )
        self.assertIn("CUDA 13 + cuDNN 9 ランタイムを同梱", create_notes_step["run"])
        self.assertNotIn("非同梱", create_notes_step["run"])

        build_steps = workflow["jobs"]["cross-build"]["steps"]
        deps_step = next(s for s in build_steps if s.get("name") == "Install Python Dependencies")
        self.assertIn("flet==1.1.2", deps_step["run"])
        self.assertNotIn("requirements.txt", deps_step["run"])

        win_build = next(s for s in build_steps if s.get("name") == "Flet build(Windows)")
        self.assertIn("--python-version 3.12", win_build["run"])

        crt_step = next(
            s
            for s in build_steps
            if s.get("name") == "Bundle Visual C++ CRT redistributables (Windows)"
        )
        self.assertIn("msvcp140_atomic_wait.dll", crt_step["run"])

        verify_step = next(
            s for s in build_steps if s.get("name") == "Verify build artifact contents"
        )
        self.assertIn("scripts/verify_flet_artifact.py", verify_step["run"])

        pkg_step = next(s for s in build_steps if s.get("name") == "Package build artifact")
        self.assertIn("scripts/package_flet_artifact.py", pkg_step["run"])
        self.assertNotIn("Compress-Archive", pkg_step["run"])


if __name__ == "__main__":
    unittest.main()
