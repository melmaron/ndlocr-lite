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


def _format_python_dll_name(python_version: str) -> Optional[str]:
    parts = [part for part in python_version.strip().split(".") if part]
    if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return None
    return f"python{parts[0]}{parts[1]}.dll"


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

    if norm_platform == "windows" and python_version:
        expected_py_dll = _format_python_dll_name(python_version)
        if expected_py_dll and expected_py_dll.lower() not in names_lower:
            errors.append(f"Missing Windows embedded Python DLL: {expected_py_dll}")

    if norm_variant == "cuda":
        expected_cuda = (
            "onnxruntime_providers_cuda.dll"
            if norm_platform == "windows"
            else "libonnxruntime_providers_cuda.so"
        )
        if expected_cuda.lower() not in names_lower:
            errors.append(f"Missing CUDA EP library: {expected_cuda}")
    elif norm_variant == "gpu-generic":
        webgpu_pattern = "*webgpu*.dll" if norm_platform == "windows" else "*webgpu*.so*"
        if not any(fnmatch(name, webgpu_pattern) for name in names_lower):
            errors.append(f"Missing WebGPU EP library matching {webgpu_pattern}")
        if is_x86_64:
            openvino_pattern = (
                "*openvino*.dll" if norm_platform == "windows" else "*openvino*.so*"
            )
            if not any(fnmatch(name, openvino_pattern) for name in names_lower):
                errors.append(
                    f"Missing OpenVINO EP library matching {openvino_pattern} (x86_64)"
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
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    errors = verify_artifact(
        output_dir,
        target_platform=args.platform,
        variant=args.variant,
        python_version=args.python_version,
        arch=args.arch,
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
