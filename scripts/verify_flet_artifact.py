#!/usr/bin/env python3
"""Verify built Flet desktop bundle contents before packaging and upload."""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path
import platform
import sys
from typing import List, Optional, Sequence


REQUIRED_MODELS: tuple[str, ...] = (
    "deim-s-1024x1024-261124.onnx",
    "parseq-ndl-24x256-30-tiny-499epoch-261124.onnx",
    "parseq-ndl-24x384-50-tiny-278epoch-261124.onnx",
    "parseq-ndl-24x768-100-tiny-153epoch-261124.onnx",
    "tablecenternet_s_800_v3.onnx",
)

DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024  # GitHub Releases 2.00 GiB limit

WINDOWS_REQUIRED_CRT_DLLS: tuple[str, ...] = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_atomic_wait.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
)

WINDOWS_CUDA_REQUIRED_PATTERNS: tuple[str, ...] = (
    "onnxruntime_providers_cuda.dll",
    "cudart64_13*.dll",
    "cublas64_13*.dll",
    "cublaslt64_13*.dll",
    "nvrtc64_13*.dll",
    "cudnn64_9*.dll",
    "cudnn_ops64_9*.dll",
)

LINUX_CUDA_REQUIRED_PATTERNS: tuple[str, ...] = (
    "libonnxruntime_providers_cuda.so",
    "libcudart.so.13*",
    "libcublas.so.13*",
    "libcublaslt.so.13*",
    "libcurand.so.10*",
    "libnvrtc.so.13*",
    "libcudnn.so.9*",
    "libcudnn_ops.so.9*",
)

WINDOWS_WEBGPU_REQUIRED_PATTERNS: tuple[str, ...] = (
    "onnxruntime_providers_webgpu.dll",
    "dxcompiler.dll",
    "d3d12core.dll",
)

WINDOWS_OPENVINO_REQUIRED_PATTERNS: tuple[str, ...] = (
    "onnxruntime_providers_openvino_plugin.dll",
    "openvino.dll",
    "tbb12.dll",
)

LINUX_WEBGPU_REQUIRED_PATTERNS: tuple[str, ...] = (
    "libonnxruntime_providers_webgpu.so*",
)

LINUX_OPENVINO_REQUIRED_PATTERNS: tuple[str, ...] = (
    "libonnxruntime_providers_openvino_plugin.so*",
    "libopenvino.so*",
    "libtbb.so.12*",
)


def _format_python_dll_name(python_version: str) -> Optional[str]:
    parts = [part for part in python_version.strip().split(".") if part]
    if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return None
    return f"python{parts[0]}{parts[1]}.dll"


def verify_archive_size(
    archive_path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
) -> List[str]:
    """Return error list if packaged archive is missing or exceeds size limit."""
    archive = Path(archive_path)
    if not archive.is_file():
        return [f"Packaged archive does not exist: {archive}"]
    size_bytes = archive.stat().st_size
    if size_bytes <= 0:
        return [f"Packaged archive is empty: {archive}"]
    if max_bytes > 0 and size_bytes >= max_bytes:
        return [
            f"Packaged archive {archive.name} size ({size_bytes:,} bytes / "
            f"{size_bytes / (1024 ** 3):.2f} GiB) meets or exceeds limit ({max_bytes:,} bytes)"
        ]
    return []


def verify_artifact(
    output_dir: Path,
    *,
    target_platform: str,
    variant: str,
    python_version: Optional[str] = None,
    arch: Optional[str] = None,
) -> List[str]:
    """Return a list of verification error messages (empty list means valid)."""
    root = Path(output_dir)
    if not root.is_dir():
        return [f"Output directory does not exist: {root}"]

    files = [path for path in root.rglob("*") if path.is_file()]
    if not files:
        return [f"Output directory is empty: {root}"]

    names_lower = {path.name.lower() for path in files}
    errors: List[str] = []

    for model_name in REQUIRED_MODELS:
        if model_name.lower() not in names_lower:
            errors.append(f"Missing bundled ONNX model: {model_name}")

    norm_platform = target_platform.strip().lower()
    norm_variant = variant.strip().lower()
    effective_arch = (arch or platform.machine()).strip().lower()
    is_x86_64 = effective_arch in {"x86_64", "amd64"}

    if norm_platform == "windows":
        if python_version:
            expected_py_dll = _format_python_dll_name(python_version)
            if expected_py_dll and expected_py_dll.lower() not in names_lower:
                errors.append(f"Missing Windows embedded Python DLL: {expected_py_dll}")
        for crt_dll in WINDOWS_REQUIRED_CRT_DLLS:
            if crt_dll not in names_lower:
                errors.append(f"Missing Windows VC++ runtime DLL: {crt_dll}")

    if norm_variant == "cuda":
        patterns = (
            WINDOWS_CUDA_REQUIRED_PATTERNS
            if norm_platform == "windows"
            else LINUX_CUDA_REQUIRED_PATTERNS
        )
        for pattern in patterns:
            if not any(fnmatch(name, pattern) for name in names_lower):
                errors.append(f"Missing bundled CUDA runtime/EP library matching {pattern}")
    elif norm_variant == "gpu-generic":
        wgpu_patterns = (
            WINDOWS_WEBGPU_REQUIRED_PATTERNS
            if norm_platform == "windows"
            else LINUX_WEBGPU_REQUIRED_PATTERNS
        )
        for pattern in wgpu_patterns:
            if not any(fnmatch(name, pattern) for name in names_lower):
                errors.append(f"Missing WebGPU EP/runtime library matching {pattern}")
        if is_x86_64:
            ov_patterns = (
                WINDOWS_OPENVINO_REQUIRED_PATTERNS
                if norm_platform == "windows"
                else LINUX_OPENVINO_REQUIRED_PATTERNS
            )
            for pattern in ov_patterns:
                if not any(fnmatch(name, pattern) for name in names_lower):
                    errors.append(
                        f"Missing OpenVINO EP/runtime library matching {pattern} (x86_64)"
                    )
    elif norm_variant != "base":
        errors.append(f"Unknown build variant: {variant}")

    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify bundled models, Python runtime, and EP libraries in Flet build output."
    )
    parser.add_argument("--output-dir", required=True, help="Flet build output directory")
    parser.add_argument(
        "--platform",
        required=True,
        choices=("windows", "linux"),
        help="Target build platform",
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("cuda", "gpu-generic", "base"),
        help="Dependency variant",
    )
    parser.add_argument(
        "--python-version",
        default=None,
        help="Expected embedded Python version (e.g. 3.12)",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Target architecture override (defaults to platform.machine())",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Optional compressed archive (.zip/.tar.gz) to validate size limit",
    )
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_BYTES,
        help="Strict upper bound on compressed archive size in bytes",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    errors = verify_artifact(
        output_dir,
        target_platform=args.platform,
        variant=args.variant,
        python_version=args.python_version,
        arch=args.arch,
    )
    if args.archive:
        errors.extend(
            verify_archive_size(
                Path(args.archive),
                max_bytes=args.max_archive_bytes,
            )
        )
    if errors:
        print(
            f"[FAIL] Artifact verification failed for {args.platform}/{args.variant} ({output_dir}):",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    file_count = sum(1 for p in output_dir.rglob("*") if p.is_file())
    print(
        f"[OK] Artifact verified for {args.platform}/{args.variant}: "
        f"{file_count} files under {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
