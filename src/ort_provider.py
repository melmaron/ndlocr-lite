"""Unified ONNX Runtime execution provider management for CPU and multi-vendor GPUs."""

from __future__ import annotations

import contextlib
import ctypes
from dataclasses import dataclass, field
import importlib
import io
import logging
import os
from pathlib import Path
import platform
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import onnxruntime as ort

_LOGGER = logging.getLogger("ndlocr.ort_provider")


DEVICE_CHOICES: Tuple[str, ...] = (
    "auto",
    "cpu",
    "cuda",
    "migraphx",
    "rocm",
    "openvino",
    "webgpu",
    "dml",
)

PRECISION_CHOICES: Tuple[str, ...] = ("fp32", "fp16")

AUTO_PRIORITY: Tuple[str, ...] = (
    "cuda",
    "migraphx",
    "rocm",
    "openvino",
    "webgpu",
    "dml",
    "cpu",
)

DEFORM_CONV_MIN_ORT_VERSION: Tuple[int, int, int] = (1, 25, 0)

_DEFAULT_BATCH_SIZES: Dict[str, int] = {
    "cuda": 64,
    "migraphx": 32,
    "rocm": 32,
    "openvino": 16,
    "webgpu": 16,
    "dml": 16,
    "cpu": 1,
}

_BACKEND_DISPLAY_NAMES: Dict[str, str] = {
    "cuda": "GPU (CUDA)",
    "migraphx": "GPU (MIGraphX)",
    "rocm": "GPU (ROCm)",
    "openvino": "GPU (OpenVINO)",
    "webgpu": "GPU (WebGPU)",
    "dml": "GPU (DirectML)",
    "cpu": "CPU",
}

_BACKEND_MONOLITHIC_PROVIDER: Dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

_INSTALL_HINTS: Dict[str, str] = {
    "cuda": (
        "NVIDIA CUDA 13 ランタイム + cuDNN 9 が必要です（配布 GUI パッケージには非同梱）。"
        "配布 GUI 版では CUDA Toolkit 13.x / cuDNN 9.x を導入し、bin/lib パス（Windows は PATH、Linux は LD_LIBRARY_PATH）を設定してください。"
        "ソース実行時は `pip install \"onnxruntime-gpu[cuda,cudnn]==1.31.1\"`"
        "（CUDA 12 系環境では Azure インデックスから `onnxruntime-gpu==1.29.0`）を導入してください。"
    ),
    "migraphx": (
        "Linux AMD ROCm環境向けに `onnxruntime-migraphx` と AMDMIGraphX / ROCm "
        "ランタイムの導入が必要です（ソースビルド環境向け）。"
    ),
    "rocm": (
        "Linux AMD ROCm環境向けに `onnxruntime-rocm`（または `onnxruntime-migraphx`）"
        "と ROCm ランタイムの導入が必要です（ソースビルド環境向け）。"
    ),
    "openvino": (
        "Intel Arc / iGPU 環境向けに `pip install onnxruntime-ep-openvino` と"
        "最新の Intel GPU ドライバ（Linux では OpenCL / Level Zero）を導入してください（x86_64 のみ）。"
    ),
    "webgpu": (
        "Radeon / Arc / NVIDIA 共通GPU向けに `pip install onnxruntime-ep-webgpu` を導入してください"
        "（Linux では `libvulkan.so.1` と Vulkan ドライバ、Windows では DirectX 12 対応ドライバが必要です）。"
    ),
    "dml": (
        "Windows DirectML環境向けに `pip install onnxruntime-directml` を導入してください"
        "（注意: DirectML では DEIMv2 が自動 CPU フォールバックし PARSeq のみ DirectML GPU 実行となります。"
        "また onnxruntime-directml 1.24系では v2 の表構造認識 TableCenterNet は非対応です）。"
    ),
}

_PLUGIN_SPECS: Tuple[Tuple[str, str], ...] = (
    ("webgpu", "onnxruntime_ep_webgpu"),
    ("openvino_ep", "onnxruntime_ep_openvino"),
)

_BACKEND_PLUGIN_NAME: Dict[str, str] = {
    "webgpu": "webgpu",
    "openvino": "openvino_ep",
}

_PLUGIN_LOCK = threading.Lock()
_REGISTERED_PLUGINS: Dict[str, str] = {}
_PLUGIN_ERRORS: Dict[str, str] = {}
_CUDA_DLLS_PRELOADED = False
_CUDA_DLLS_ERROR: Optional[str] = None
_CUDA_PRELOAD_OUTPUT: Optional[str] = None
_CUDA_PROBE_CACHE: Optional[Dict[str, Any]] = None
_WARNED_KEYS: set[Tuple[str, str, str]] = set()


@dataclass
class BackendProfile:
    """Resolved ONNX Runtime backend properties for a created session."""

    requested_device: str
    active_backend: str
    active_providers: List[str] = field(default_factory=list)
    device_id: int = 0
    default_batch_size: int = 1
    parallel_session_init: bool = True
    concurrent_session_run: bool = True
    fallback_occurred: bool = False
    warning_message: Optional[str] = None
    display_name: str = "CPU"


def parse_ort_version(version_str: Optional[str] = None) -> Tuple[int, int, int]:
    """Parse an ONNX Runtime version string into a comparable 3-tuple."""
    raw = version_str or getattr(ort, "__version__", "0.0.0")
    nums = [int(part) for part in re.findall(r"\d+", str(raw))[:3]]
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def ort_supports_deform_conv(version_str: Optional[str] = None) -> bool:
    """Return True if the active ORT version supports standard ONNX DeformConv."""
    return parse_ort_version(version_str) >= DEFORM_CONV_MIN_ORT_VERSION


def normalize_device_name(device: Optional[str]) -> str:
    """Normalize user-supplied device names to canonical choices."""
    if not isinstance(device, str):
        return "auto"
    raw = device.strip().casefold()
    aliases = {
        "": "auto",
        "directml": "dml",
        "gpu": "auto",
        "ov": "openvino",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in DEVICE_CHOICES:
        return raw
    return normalized


def normalize_precision_name(precision: Optional[str]) -> str:
    """Normalize user-supplied inference precision names to canonical choices."""
    if not isinstance(precision, str):
        return "fp32"
    raw = precision.strip().casefold()
    aliases = {
        "": "fp32",
        "float32": "fp32",
        "float16": "fp16",
        "half": "fp16",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in PRECISION_CHOICES:
        return raw
    return normalized


def should_parallelize_model_init(device: Optional[str]) -> bool:
    """Parallelize session creation on CPU and CUDA; keep other GPU EPs serial."""
    target = normalize_device_name(device)
    if target == "auto":
        target = resolve_auto_backend_name()
    return target in {"cpu", "cuda"}


def prepare_backend_for_parallel_init(device: Optional[str]) -> None:
    """Run one-time global EP/DLL initialization before worker threads spawn."""
    _ = device
    preload_cuda_dlls_if_needed()
    register_plugin_eps()


def preload_cuda_dlls_if_needed() -> None:
    """Best-effort DLL preloading for pip-installed CUDA/cuDNN wheels on Windows/Linux."""
    global _CUDA_DLLS_PRELOADED, _CUDA_DLLS_ERROR, _CUDA_PRELOAD_OUTPUT, _CUDA_PROBE_CACHE
    with _PLUGIN_LOCK:
        if _CUDA_DLLS_PRELOADED:
            return
        _CUDA_DLLS_PRELOADED = True
        _CUDA_PROBE_CACHE = None
        preload_fn = getattr(ort, "preload_dlls", None)
        if not callable(preload_fn):
            return
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                preload_fn()
        except Exception as exc:  # pragma: no cover - environment dependent
            _CUDA_DLLS_ERROR = str(exc)
        finally:
            output = buf.getvalue().strip()
            if output:
                _CUDA_PRELOAD_OUTPUT = output
                if _CUDA_DLLS_ERROR is None and "failed to load" in output.casefold():
                    failed_lines = [
                        line.strip()
                        for line in output.splitlines()
                        if "failed to load" in line.casefold()
                    ]
                    _CUDA_DLLS_ERROR = "; ".join(failed_lines) or output


def _bundled_cuda_provider_path() -> Optional[Path]:
    """Return path to bundled CUDA provider shared library if shipped with ORT."""
    ort_file = getattr(ort, "__file__", None)
    if not ort_file:
        return None
    capi_dir = Path(ort_file).resolve().parent / "capi"
    if sys.platform == "win32":
        lib_name = "onnxruntime_providers_cuda.dll"
    elif sys.platform == "darwin":
        lib_name = "libonnxruntime_providers_cuda.dylib"
    else:
        lib_name = "libonnxruntime_providers_cuda.so"
    candidate = capi_dir / lib_name
    return candidate if candidate.is_file() else None


def _detect_expected_cuda_major() -> str:
    """Infer expected CUDA major version (e.g. '13' or '12') for installed ORT."""
    raw_cuda = getattr(ort, "cuda_version", None)
    if raw_cuda:
        match = re.search(r"(\d+)", str(raw_cuda))
        if match:
            return match.group(1)

    get_info = getattr(ort, "get_build_info", None)
    if callable(get_info):
        try:
            info = str(get_info() or "")
        except Exception:
            info = ""
        match = re.search(r"cuda[-_ ]?(?:version)?[:=\s-]*(\d+)", info, re.IGNORECASE)
        if match:
            return match.group(1)

    return "13" if parse_ort_version() >= (1, 30, 0) else "12"


def _candidate_nvidia_lib_paths(
    cuda_major: str,
    filename: str,
    *,
    plat: Optional[str] = None,
    machine: Optional[str] = None,
) -> List[Path]:
    """Return candidate site-packages / capi paths for an NVIDIA runtime library."""
    ort_file = getattr(ort, "__file__", None)
    if not ort_file:
        return []
    ort_dir = Path(ort_file).resolve().parent
    site_packages = ort_dir.parent
    capi_dir = ort_dir / "capi"
    active_plat = plat or sys.platform
    active_machine = (machine or platform.machine()).lower()
    arch = "arm64" if active_machine in ("arm64", "aarch64") else "x86_64"

    paths: List[Path] = []
    if active_plat == "win32":
        if "cudnn" in filename.casefold():
            paths.append(site_packages / "nvidia" / "cudnn" / "bin" / filename)
        else:
            paths.extend([
                site_packages / "nvidia" / f"cu{cuda_major}" / "bin" / arch / filename,
                site_packages / "nvidia" / "cuda_runtime" / "bin" / filename,
                site_packages / "nvidia" / "cublas" / "bin" / filename,
            ])
    else:
        if "cudnn" in filename.casefold():
            paths.append(site_packages / "nvidia" / "cudnn" / "lib" / filename)
        else:
            paths.extend([
                site_packages / "nvidia" / f"cu{cuda_major}" / "lib" / filename,
                site_packages / "nvidia" / "cuda_runtime" / "lib" / filename,
                site_packages / "nvidia" / "cublas" / "lib" / filename,
            ])
    paths.append(capi_dir / filename)
    return paths


def _can_load_shared_lib(
    filename: str,
    candidate_paths: Sequence[Path],
) -> Tuple[bool, Optional[str]]:
    """Verify whether a shared library can be loaded from pip wheels or OS loader paths."""
    last_err: Optional[str] = None
    for path in candidate_paths:
        if path.is_file():
            try:
                ctypes.CDLL(str(path))
                return True, None
            except OSError as exc:
                last_err = str(exc)

    try:
        # On Windows, Python 3.8+ default ctypes.CDLL skips PATH; winmode=0
        # matches standard LoadLibraryEx search order so user CUDA/cuDNN PATHs work.
        if os.name == "nt":
            ctypes.CDLL(filename, winmode=0)
        else:
            ctypes.CDLL(filename)
        return True, None
    except OSError as exc:
        return False, last_err or str(exc)


def build_cuda_runtime_hint(probe: Dict[str, Any]) -> str:
    """Build actionable troubleshooting guidance for CUDA runtime preflight results."""
    major = str(probe.get("expected_cuda_major") or "13")
    other = probe.get("detected_other_cuda_major")
    parts: List[str] = []
    if other:
        parts.append(
            f"この環境では CUDA {other} のみが検出されましたが、"
            f"導入済みビルド (onnxruntime {getattr(ort, '__version__', '?')}) は CUDA {major} + cuDNN 9 向けです。"
        )
    parts.append(
        f"【配布 GUI 版】NVIDIA CUDA Toolkit {major}.x と cuDNN 9.x をインストールし、"
        + (
            "それぞれの bin フォルダを環境変数 PATH に追加して再起動してください（CUDA 13 ランタイムはパッケージに非同梱です）。"
            if sys.platform == "win32"
            else "ライブラリパス（LD_LIBRARY_PATH 等）が通っている状態で再起動してください（CUDA 13 ランタイムはパッケージに非同梱です）。"
        )
    )
    if major == "13":
        parts.append(
            "【ソース実行】仮想環境へ `pip install \"onnxruntime-gpu[cuda,cudnn]==1.31.1\"`"
            "（または CUDA 12 環境向けに `onnxruntime-gpu==1.29.0` from onnxruntime-cuda-12 インデックス）を導入してください。"
        )
    else:
        parts.append(
            f"【ソース実行】CUDA {major} / cuDNN 9 ランタイム（例: `nvidia-cudnn-cu12` 等）が導入されているか確認してください。"
        )
    return " ".join(parts)


def probe_cuda_runtime(*, refresh: bool = False) -> Dict[str, Any]:
    """Check whether CUDAExecutionProvider's required CUDA/cuDNN libraries actually load."""
    global _CUDA_PROBE_CACHE
    available_providers = set(ort.get_available_providers())
    provider_present = "CUDAExecutionProvider" in available_providers
    bundled_provider = _bundled_cuda_provider_path()
    declared = provider_present or (bundled_provider is not None)

    with _PLUGIN_LOCK:
        preload_error = _CUDA_DLLS_ERROR
        preload_output = _CUDA_PRELOAD_OUTPUT
        if (
            not refresh
            and _CUDA_PROBE_CACHE is not None
            and _CUDA_PROBE_CACHE.get("provider_present") == provider_present
            and _CUDA_PROBE_CACHE.get("declared") == declared
            and _CUDA_PROBE_CACHE.get("preload_error") == preload_error
        ):
            return dict(_CUDA_PROBE_CACHE)

    if not declared or sys.platform not in ("win32", "linux"):
        result = {
            "declared": declared,
            "provider_present": provider_present,
            "expected_cuda_major": "",
            "runtime_ready": False,
            "missing_libs": [],
            "lib_errors": {},
            "detected_other_cuda_major": None,
            "preload_error": preload_error,
            "preload_output": preload_output,
            "detail": "CUDAExecutionProvider not available",
            "hint": _INSTALL_HINTS["cuda"],
        }
        with _PLUGIN_LOCK:
            _CUDA_PROBE_CACHE = dict(result)
        return result

    major = _detect_expected_cuda_major()
    if sys.platform == "win32":
        required_libs = [
            f"cudart64_{major}.dll",
            f"cublasLt64_{major}.dll",
            f"cublas64_{major}.dll",
            "cudnn64_9.dll",
        ]
    else:
        required_libs = [
            f"libcudart.so.{major}",
            f"libcublasLt.so.{major}",
            f"libcublas.so.{major}",
            "libcudnn.so.9",
        ]

    missing_libs: List[str] = []
    lib_errors: Dict[str, str] = {}
    for lib_name in required_libs:
        ok, err = _can_load_shared_lib(
            lib_name, _candidate_nvidia_lib_paths(major, lib_name)
        )
        if not ok:
            missing_libs.append(lib_name)
            if err:
                lib_errors[lib_name] = err

    detected_other: Optional[str] = None
    cudart_name = required_libs[0]
    if cudart_name in missing_libs:
        alt_major = "12" if major == "13" else "13"
        alt_cudart = (
            f"cudart64_{alt_major}.dll"
            if sys.platform == "win32"
            else f"libcudart.so.{alt_major}"
        )
        alt_ok, _ = _can_load_shared_lib(
            alt_cudart, _candidate_nvidia_lib_paths(alt_major, alt_cudart)
        )
        if alt_ok:
            detected_other = alt_major

    runtime_ready = provider_present and not missing_libs
    if runtime_ready:
        detail = f"CUDAExecutionProvider (CUDA {major} + cuDNN 9 verified)"
    elif missing_libs:
        mismatch = (
            f"（システムには CUDA {detected_other} のみ検出）"
            if detected_other
            else ""
        )
        detail = (
            f"CUDA {major} ランタイム / cuDNN 9 不足: "
            + ", ".join(missing_libs)
            + mismatch
        )
    else:
        extra = f": {preload_error}" if preload_error else ""
        detail = (
            f"CUDA provider library exists ({bundled_provider.name}), "
            f"but CUDAExecutionProvider is not registered{extra}"
        )

    result = {
        "declared": declared,
        "provider_present": provider_present,
        "expected_cuda_major": major,
        "runtime_ready": runtime_ready,
        "missing_libs": missing_libs,
        "lib_errors": lib_errors,
        "detected_other_cuda_major": detected_other,
        "preload_error": preload_error,
        "preload_output": preload_output,
        "detail": detail,
        "hint": "",
    }
    result["hint"] = "" if runtime_ready else build_cuda_runtime_hint(result)
    with _PLUGIN_LOCK:
        _CUDA_PROBE_CACHE = dict(result)
    return result


def register_plugin_eps() -> Dict[str, str]:
    """Register installed ORT plugin execution provider libraries once per process."""
    register_fn = getattr(ort, "register_execution_provider_library", None)
    if not callable(register_fn):
        return dict(_REGISTERED_PLUGINS)

    with _PLUGIN_LOCK:
        for reg_name, module_name in _PLUGIN_SPECS:
            if reg_name in _REGISTERED_PLUGINS:
                continue
            try:
                module = importlib.import_module(module_name)
                lib_path = str(module.get_library_path())
            except ImportError:
                continue
            except Exception as exc:
                _PLUGIN_ERRORS[reg_name] = f"import/path error: {exc}"
                continue

            try:
                register_fn(reg_name, lib_path)
                _REGISTERED_PLUGINS[reg_name] = lib_path
                _PLUGIN_ERRORS.pop(reg_name, None)
            except Exception as exc:
                msg = str(exc)
                if "already registered" in msg.casefold():
                    _REGISTERED_PLUGINS[reg_name] = lib_path
                    _PLUGIN_ERRORS.pop(reg_name, None)
                else:
                    _PLUGIN_ERRORS[reg_name] = msg
    return dict(_REGISTERED_PLUGINS)


def _safe_ep_devices() -> List[Any]:
    get_devs = getattr(ort, "get_ep_devices", None)
    if not callable(get_devs):
        return []
    try:
        return list(get_devs())
    except Exception:
        return []


_VIRTUAL_OR_SOFTWARE_VENDOR_IDS = frozenset({0x1414, 0x10005})
_SOFTWARE_VENDOR_MARKERS = (
    "llvmpipe",
    "swiftshader",
    "microsoft basic render",
    "warp",
)


def _is_software_or_virtual_webgpu_device(ep_device: Any) -> bool:
    hw = getattr(ep_device, "device", None)
    if hw is None:
        return False
    vendor_id = getattr(hw, "vendor_id", None)
    if isinstance(vendor_id, int) and not isinstance(vendor_id, bool):
        if vendor_id in _VIRTUAL_OR_SOFTWARE_VENDOR_IDS:
            return True
    vendor_str = str(getattr(hw, "vendor", "") or "").strip().casefold()
    if any(marker in vendor_str for marker in _SOFTWARE_VENDOR_MARKERS):
        return True
    hw_type = getattr(hw, "type", None)
    cpu_enum = getattr(getattr(ort, "OrtHardwareDeviceType", None), "CPU", None)
    if cpu_enum is not None and hw_type is not None and hw_type == cpu_enum:
        return True
    return False


def _is_dml_deim_model(model_label: str, model_path: str | Path = "") -> bool:
    label_cf = (model_label or "").casefold()
    name_cf = Path(model_path).name.casefold() if model_path else ""
    return "deim" in label_cf or "deim" in name_cf


def _is_gpu_hw_device(ep_device: Any) -> bool:
    meta = getattr(ep_device, "ep_metadata", None) or {}
    ov_dev = str(meta.get("ov_device", "")).strip().upper()
    if ov_dev:
        return ov_dev.startswith("GPU")
    hw = getattr(ep_device, "device", None)
    hw_type = getattr(hw, "type", None)
    gpu_enum = getattr(getattr(ort, "OrtHardwareDeviceType", None), "GPU", None)
    if gpu_enum is not None and hw_type is not None:
        return hw_type == gpu_enum
    return "GPU" in str(hw_type).upper()


def _find_plugin_devices_for_backend(
    backend: str,
    ep_devices: Sequence[Any],
    *,
    include_virtual_webgpu: bool = False,
) -> List[Any]:
    if backend == "webgpu":
        all_webgpu = [
            d for d in ep_devices
            if "webgpu" in str(getattr(d, "ep_name", "")).casefold()
        ]
        hw_devs = [d for d in all_webgpu if not _is_software_or_virtual_webgpu_device(d)]
        if not include_virtual_webgpu:
            return hw_devs
        sw_devs = [d for d in all_webgpu if _is_software_or_virtual_webgpu_device(d)]
        return hw_devs + sw_devs
    if backend == "openvino":
        return [
            d for d in ep_devices
            if str(getattr(d, "ep_name", "")) == "OpenVINOExecutionProvider"
            and _is_gpu_hw_device(d)
        ]
    return []


def detect_available_backends(*, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Inspect installed EPs and discovered hardware devices for all supported backends."""
    preload_cuda_dlls_if_needed()
    register_plugin_eps()
    available_providers = set(ort.get_available_providers())
    ep_devices = _safe_ep_devices()
    results: Dict[str, Dict[str, Any]] = {}

    for backend in AUTO_PRIORITY:
        mono_provider = _BACKEND_MONOLITHIC_PROVIDER[backend]
        plugin_devs = _find_plugin_devices_for_backend(backend, ep_devices)
        plugin_reg_name = _BACKEND_PLUGIN_NAME.get(backend)
        plugin_is_loaded = bool(plugin_reg_name and plugin_reg_name in _REGISTERED_PLUGINS)
        if backend == "cpu":
            results[backend] = {
                "available": True,
                "declared": True,
                "verified": True,
                "status": "verified",
                "mode": "builtin",
                "devices": ["CPU"],
                "detail": "CPUExecutionProvider",
            }
            continue

        if backend == "cuda":
            cuda_probe = probe_cuda_runtime(refresh=refresh)
            if cuda_probe["declared"]:
                if cuda_probe["runtime_ready"]:
                    results[backend] = {
                        "available": True,
                        "declared": True,
                        "verified": True,
                        "status": "verified",
                        "mode": "monolithic",
                        "devices": [mono_provider],
                        "detail": cuda_probe["detail"],
                        "missing_libs": [],
                        "hint": "",
                    }
                else:
                    results[backend] = {
                        "available": False,
                        "declared": True,
                        "verified": False,
                        "status": "missing_runtime",
                        "mode": "monolithic",
                        "devices": [mono_provider],
                        "detail": cuda_probe["detail"],
                        "missing_libs": list(cuda_probe["missing_libs"]),
                        "detected_other_cuda_major": cuda_probe.get("detected_other_cuda_major"),
                        "hint": cuda_probe["hint"],
                    }
            else:
                results[backend] = {
                    "available": False,
                    "declared": False,
                    "verified": False,
                    "status": "unavailable",
                    "mode": "none",
                    "devices": [],
                    "detail": f"{mono_provider} not available",
                    "missing_libs": [],
                    "hint": "",
                }
            continue

        if plugin_devs:
            desc = []
            for d in plugin_devs:
                meta = getattr(d, "ep_metadata", None) or {}
                hw = getattr(d, "device", None)
                vendor = getattr(hw, "vendor", "") if hw else ""
                ov_dev = meta.get("ov_device", "")
                label = " ".join(part for part in (str(d.ep_name), str(ov_dev), str(vendor)) if part)
                desc.append(label)
            results[backend] = {
                "available": True,
                "declared": True,
                "verified": True,
                "status": "verified",
                "mode": "plugin",
                "devices": desc,
                "detail": "; ".join(desc),
            }
        elif (not plugin_is_loaded) and mono_provider in available_providers:
            results[backend] = {
                "available": True,
                "declared": True,
                "verified": None,
                "status": "declared_only",
                "mode": "monolithic",
                "devices": [mono_provider],
                "detail": mono_provider,
            }
        else:
            extra_note = ""
            if backend == "openvino":
                ov_all = [
                    d for d in ep_devices
                    if str(getattr(d, "ep_name", "")) == "OpenVINOExecutionProvider"
                ]
                if ov_all or plugin_is_loaded:
                    extra_note = "OpenVINO plugin loaded, but no Intel GPU device detected (CPU/NPU only)"
                elif "openvino_ep" in _PLUGIN_ERRORS:
                    extra_note = f"plugin error: {_PLUGIN_ERRORS['openvino_ep']}"
            elif backend == "webgpu":
                webgpu_all = _find_plugin_devices_for_backend(
                    "webgpu", ep_devices, include_virtual_webgpu=True
                )
                if "webgpu" in _PLUGIN_ERRORS:
                    extra_note = f"plugin error: {_PLUGIN_ERRORS['webgpu']}"
                elif webgpu_all:
                    extra_note = (
                        "WebGPU plugin loaded, but only software/virtual adapter detected "
                        "(skipped for auto; use --device webgpu to force)"
                    )
                elif plugin_is_loaded:
                    extra_note = "WebGPU plugin loaded, but no compatible GPU adapter detected"

            results[backend] = {
                "available": False,
                "declared": bool(plugin_is_loaded),
                "verified": False,
                "status": "unavailable",
                "mode": "none",
                "devices": [],
                "detail": extra_note or f"{mono_provider} not available",
            }

    return results


def _build_session_options(
    cpu_single_thread: bool = False,
    enable_mem_reuse: bool = True,
) -> ort.SessionOptions:
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.enable_mem_reuse = bool(enable_mem_reuse)
    if cpu_single_thread:
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
    return opt


def _make_profile(
    requested_device: str,
    active_backend: str,
    active_providers: List[str],
    device_id: int,
    *,
    fallback_occurred: bool = False,
    warning_message: Optional[str] = None,
) -> BackendProfile:
    is_cpu = active_backend == "cpu"
    return BackendProfile(
        requested_device=requested_device,
        active_backend=active_backend,
        active_providers=list(active_providers),
        device_id=int(device_id),
        default_batch_size=_DEFAULT_BATCH_SIZES.get(active_backend, 1),
        parallel_session_init=(active_backend in {"cpu", "cuda"}),
        concurrent_session_run=is_cpu,
        fallback_occurred=fallback_occurred,
        warning_message=warning_message,
        display_name=_BACKEND_DISPLAY_NAMES.get(active_backend, active_backend.upper()),
    )


def _create_session_for_backend(
    model_path: str | Path,
    backend: str,
    *,
    device_id: int,
    cpu_single_thread: bool,
    enable_mem_reuse: bool = True,
    allow_software_webgpu: bool = False,
    model_label: str = "model",
) -> Tuple[ort.InferenceSession, List[str]]:
    model_str = str(model_path)
    if backend == "cuda":
        preload_cuda_dlls_if_needed()
    register_plugin_eps()
    available_providers = set(ort.get_available_providers())
    ep_devices = _safe_ep_devices()

    if backend == "cpu":
        opt = _build_session_options(
            cpu_single_thread=cpu_single_thread,
            enable_mem_reuse=enable_mem_reuse,
        )
        session = ort.InferenceSession(model_str, opt, providers=["CPUExecutionProvider"])
        return session, list(session.get_providers())

    if backend == "cuda":
        if "CUDAExecutionProvider" not in available_providers:
            probe = probe_cuda_runtime(refresh=True)
            if probe.get("missing_libs") or probe.get("declared"):
                raise RuntimeError(
                    f"CUDAExecutionProvider is not available: {probe['detail']}"
                )
            raise RuntimeError("CUDAExecutionProvider is not available in installed onnxruntime.")
        opt = _build_session_options(
            cpu_single_thread=False,
            enable_mem_reuse=enable_mem_reuse,
        )
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": int(device_id),
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "use_tf32": "0",
                },
            ),
            "CPUExecutionProvider",
        ]
        try:
            session = ort.InferenceSession(model_str, opt, providers=providers)
        except Exception as exc:
            probe = probe_cuda_runtime(refresh=True)
            if probe.get("missing_libs") and probe["detail"] not in str(exc):
                raise RuntimeError(f"{exc} | {probe['detail']}") from exc
            raise
        active = list(session.get_providers())
        if "CUDAExecutionProvider" not in active:
            probe = probe_cuda_runtime(refresh=True)
            detail = f" ({probe['detail']})" if probe.get("missing_libs") else ""
            raise RuntimeError(f"CUDAExecutionProvider fell back to {active}{detail}.")
        return session, active

    if backend == "migraphx":
        if "MIGraphXExecutionProvider" not in available_providers:
            raise RuntimeError("MIGraphXExecutionProvider is not available.")
        opt = _build_session_options(
            cpu_single_thread=False,
            enable_mem_reuse=enable_mem_reuse,
        )
        providers = [("MIGraphXExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]
        session = ort.InferenceSession(model_str, opt, providers=providers)
        active = list(session.get_providers())
        if "MIGraphXExecutionProvider" not in active:
            raise RuntimeError(f"MIGraphXExecutionProvider fell back to {active}.")
        return session, active

    if backend == "rocm":
        if "ROCMExecutionProvider" not in available_providers:
            raise RuntimeError("ROCMExecutionProvider is not available.")
        opt = _build_session_options(
            cpu_single_thread=False,
            enable_mem_reuse=enable_mem_reuse,
        )
        providers = [("ROCMExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]
        session = ort.InferenceSession(model_str, opt, providers=providers)
        active = list(session.get_providers())
        if "ROCMExecutionProvider" not in active:
            raise RuntimeError(f"ROCMExecutionProvider fell back to {active}.")
        return session, active

    if backend == "openvino":
        gpu_devs = _find_plugin_devices_for_backend("openvino", ep_devices)
        if gpu_devs:
            if not 0 <= device_id < len(gpu_devs):
                raise RuntimeError(
                    f"Requested device_id {device_id} is out of range for OpenVINO "
                    f"({len(gpu_devs)} GPU device(s) detected)."
                )
            chosen = gpu_devs[device_id]
            opt = _build_session_options(
                cpu_single_thread=False,
                enable_mem_reuse=enable_mem_reuse,
            )
            opt.add_provider_for_devices([chosen], {})
            session = ort.InferenceSession(model_str, sess_options=opt)
            active = list(session.get_providers())
            if "OpenVINOExecutionProvider" not in active:
                raise RuntimeError(f"OpenVINOExecutionProvider plugin did not activate: {active}")
            return session, active
        if "openvino_ep" in _REGISTERED_PLUGINS:
            raise RuntimeError("OpenVINO plugin loaded, but no Intel GPU device detected (CPU/NPU only).")
        if "OpenVINOExecutionProvider" in available_providers:
            opt = _build_session_options(
                cpu_single_thread=False,
                enable_mem_reuse=enable_mem_reuse,
            )
            ov_device = "GPU" if int(device_id) == 0 else f"GPU.{int(device_id)}"
            providers = [("OpenVINOExecutionProvider", {"device_type": ov_device}), "CPUExecutionProvider"]
            session = ort.InferenceSession(model_str, opt, providers=providers)
            active = list(session.get_providers())
            if "OpenVINOExecutionProvider" not in active:
                raise RuntimeError(f"OpenVINOExecutionProvider fell back to {active}.")
            return session, active
        raise RuntimeError("OpenVINO GPU device is not available.")

    if backend == "webgpu":
        webgpu_devs = _find_plugin_devices_for_backend(
            "webgpu", ep_devices, include_virtual_webgpu=allow_software_webgpu
        )
        if webgpu_devs:
            if not 0 <= device_id < len(webgpu_devs):
                raise RuntimeError(
                    f"Requested device_id {device_id} is out of range for WebGPU "
                    f"({len(webgpu_devs)} adapter(s) detected)."
                )
            chosen = webgpu_devs[device_id]
            if _is_software_or_virtual_webgpu_device(chosen):
                _emit_warning_once(
                    ("webgpu", "software", f"{device_id}"),
                    f"[WARN] WebGPU ({model_label}) はソフトウェア/仮想アダプタを使用して実行されます (CPU より遅くなる場合があります)。",
                )
            ep_name = str(chosen.ep_name)
            opt = _build_session_options(
                cpu_single_thread=False,
                enable_mem_reuse=enable_mem_reuse,
            )
            opt.add_provider_for_devices([chosen], {})
            session = ort.InferenceSession(model_str, sess_options=opt)
            active = list(session.get_providers())
            if ep_name in active or any("webgpu" in p.casefold() for p in active):
                return session, active
            raise RuntimeError(f"WebGPU plugin EP did not activate: {active}")
        if _find_plugin_devices_for_backend("webgpu", ep_devices, include_virtual_webgpu=True):
            raise RuntimeError(
                "WebGPU plugin loaded, but only software/virtual adapter detected "
                "(skipped for auto; use --device webgpu to force)."
            )
        if "webgpu" in _REGISTERED_PLUGINS:
            raise RuntimeError("WebGPU plugin loaded, but no compatible GPU adapter detected.")
        if "WebGpuExecutionProvider" in available_providers:
            opt = _build_session_options(
                cpu_single_thread=False,
                enable_mem_reuse=enable_mem_reuse,
            )
            providers = [("WebGpuExecutionProvider", {"deviceId": int(device_id)}), "CPUExecutionProvider"]
            session = ort.InferenceSession(model_str, opt, providers=providers)
            active = list(session.get_providers())
            if "WebGpuExecutionProvider" not in active:
                raise RuntimeError(f"WebGpuExecutionProvider fell back to {active}.")
            return session, active
        raise RuntimeError("WebGPU Execution Provider is not available.")

    if backend == "dml":
        if "DmlExecutionProvider" not in available_providers:
            raise RuntimeError("DmlExecutionProvider is not available.")
        if _is_dml_deim_model(model_label, model_path):
            raise RuntimeError(
                "DmlExecutionProvider produces corrupted bounding boxes on DEIMv2 layout detection; "
                "DEIMv2 must run on CPU (or CUDA/OpenVINO/WebGPU)."
            )
        opt = _build_session_options(
            cpu_single_thread=False,
            enable_mem_reuse=enable_mem_reuse,
        )
        opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opt.enable_mem_pattern = False
        providers = [("DmlExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]
        session = ort.InferenceSession(model_str, opt, providers=providers)
        active = list(session.get_providers())
        if "DmlExecutionProvider" not in active:
            raise RuntimeError(f"DmlExecutionProvider fell back to {active}.")
        return session, active

    raise ValueError(
        f"Unsupported device '{backend}'. Supported choices: {', '.join(DEVICE_CHOICES)}"
    )


def _emit_warning_once(key: Tuple[str, str, str], message: str) -> None:
    with _PLUGIN_LOCK:
        if key in _WARNED_KEYS:
            return
        _WARNED_KEYS.add(key)
    _LOGGER.warning("%s", message)
    print(message, file=sys.stderr)


def _format_fallback_warning(
    requested: str,
    fallback_backend: str,
    reason: str,
    model_label: str,
    *,
    hint_override: Optional[str] = None,
) -> str:
    hint = hint_override or _INSTALL_HINTS.get(requested, "")
    lines = [
        f"[WARN] 指定されたデバイス '{requested}' を {model_label} で有効化できなかったため、'{fallback_backend}' へフォールバックしました。",
        f"       (Requested device '{requested}' failed for {model_label}: {reason})",
    ]
    if hint:
        lines.append(f"       対処法: {hint}")
    return "\n".join(lines)


def create_inference_session(
    model_path: str | Path,
    *,
    device: str = "auto",
    device_id: int = 0,
    strict_device: bool = False,
    cpu_single_thread: bool = False,
    enable_mem_reuse: bool = True,
    model_label: str = "model",
) -> Tuple[ort.InferenceSession, BackendProfile]:
    """Create an ONNX Runtime InferenceSession for the requested device or 'auto'."""
    requested = normalize_device_name(device)
    try:
        device_id = int(device_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"device_id must be an integer, got {device_id!r}") from exc
    if device_id < 0:
        raise ValueError(f"device_id must be >= 0, got {device_id}")
    if requested not in DEVICE_CHOICES:
        raise ValueError(
            f"Unsupported device '{device}'. Choose from: {', '.join(DEVICE_CHOICES)}"
        )

    if requested == "auto":
        availability = detect_available_backends()
        failures: List[str] = []
        hints: List[str] = []
        for candidate in AUTO_PRIORITY:
            if candidate == "cpu":
                continue
            cand_info = availability.get(candidate, {})
            if cand_info.get("status") == "missing_runtime":
                failures.append(f"{candidate}: {cand_info.get('detail', 'runtime missing')}")
                hint = cand_info.get("hint") or _INSTALL_HINTS.get(candidate)
                if hint and hint not in hints:
                    hints.append(hint)
                continue
            if not cand_info.get("available"):
                continue
            try:
                session, active_providers = _create_session_for_backend(
                    model_path,
                    candidate,
                    device_id=device_id,
                    cpu_single_thread=cpu_single_thread,
                    enable_mem_reuse=enable_mem_reuse,
                    allow_software_webgpu=False,
                    model_label=model_label,
                )
                return session, _make_profile(
                    requested_device="auto",
                    active_backend=candidate,
                    active_providers=active_providers,
                    device_id=device_id,
                )
            except Exception as exc:
                failures.append(f"{candidate}: {exc}")
                hint = _INSTALL_HINTS.get(candidate)
                if hint and hint not in hints:
                    hints.append(hint)

        if failures and strict_device:
            hint_suffix = f" | {' '.join(hints)}" if hints else ""
            raise RuntimeError(
                f"'--device auto' could not initialize any GPU backend for {model_label}: "
                + "; ".join(failures)
                + hint_suffix
            )

        session, active_providers = _create_session_for_backend(
            model_path,
            "cpu",
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
            enable_mem_reuse=enable_mem_reuse,
            model_label=model_label,
        )
        warning_msg = None
        if failures:
            warning_lines = [
                f"[WARN] 自動選択されたGPUバックエンドの初期化に失敗したため CPU で実行します ({model_label}): "
                + "; ".join(failures)
            ]
            if hints:
                warning_lines.append("       対処法: " + " / ".join(hints))
            warning_msg = "\n".join(warning_lines)
            _emit_warning_once(("auto", "cpu", ";".join(failures)), warning_msg)
        return session, _make_profile(
            requested_device="auto",
            active_backend="cpu",
            active_providers=active_providers,
            device_id=device_id,
            fallback_occurred=bool(failures),
            warning_message=warning_msg,
        )

    if requested == "cpu":
        session, active_providers = _create_session_for_backend(
            model_path,
            "cpu",
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
            enable_mem_reuse=enable_mem_reuse,
            model_label=model_label,
        )
        return session, _make_profile(
            requested_device="cpu",
            active_backend="cpu",
            active_providers=active_providers,
            device_id=device_id,
        )

    # Explicit GPU backend requested
    try:
        session, active_providers = _create_session_for_backend(
            model_path,
            requested,
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
            enable_mem_reuse=enable_mem_reuse,
            allow_software_webgpu=(requested == "webgpu"),
            model_label=model_label,
        )
        if requested == "dml" and not ort_supports_deform_conv() and "table" in model_label.casefold():
            _emit_warning_once(
                ("dml", "deformconv", ort.__version__),
                "[WARN] onnxruntime-directml (ORT < 1.25) は TableCenterNet の DeformConv に非対応です。",
            )
        return session, _make_profile(
            requested_device=requested,
            active_backend=requested,
            active_providers=active_providers,
            device_id=device_id,
        )
    except Exception as exc:
        cuda_hint: Optional[str] = None
        if requested == "cuda":
            probe = probe_cuda_runtime()
            if probe.get("hint"):
                cuda_hint = str(probe["hint"])
        if strict_device:
            hint = cuda_hint or _INSTALL_HINTS.get(requested, "")
            raise RuntimeError(
                f"Requested device '{requested}' failed for {model_label}: {exc}"
                + (f" | {hint}" if hint else "")
            ) from exc

        session, active_providers = _create_session_for_backend(
            model_path,
            "cpu",
            device_id=device_id,
            cpu_single_thread=cpu_single_thread,
            enable_mem_reuse=enable_mem_reuse,
            model_label=model_label,
        )
        warning_msg = _format_fallback_warning(
            requested,
            "cpu",
            str(exc),
            model_label,
            hint_override=cuda_hint,
        )
        _emit_warning_once((requested, "cpu", str(exc)), warning_msg)
        return session, _make_profile(
            requested_device=requested,
            active_backend="cpu",
            active_providers=active_providers,
            device_id=device_id,
            fallback_occurred=True,
            warning_message=warning_msg,
        )


def resolve_auto_backend_name() -> str:
    """Return the highest-priority currently available backend name for 'auto'."""
    availability = detect_available_backends()
    for candidate in AUTO_PRIORITY:
        if availability.get(candidate, {}).get("available"):
            return candidate
    return "cpu"


def format_device_diagnostics(*, refresh: bool = False) -> str:
    """Return a human-readable diagnostics report for `--check-device` and GUI diagnostics."""
    availability = detect_available_backends(refresh=refresh)
    auto_choice = resolve_auto_backend_name()
    ep_devices = _safe_ep_devices()
    cuda_info = availability.get("cuda", {})
    status_labels = {
        "verified": "READY (verified)",
        "declared_only": "DECLARED (unverified)",
        "missing_runtime": "MISSING RUNTIME (declared)",
        "unavailable": "unavailable",
    }
    lines = [
        "=== NDLOCR-Lite GPU / ONNX Runtime Diagnostics ===",
        f"ONNX Runtime version : {getattr(ort, '__version__', 'unknown')}",
        f"TableCenterNet (DeformConv, ORT>=1.25): {'SUPPORTED' if ort_supports_deform_conv() else 'UNSUPPORTED'}",
        f"Available providers  : {ort.get_available_providers()}",
        f"Registered plugins   : {_REGISTERED_PLUGINS or 'none'}",
    ]
    if _PLUGIN_ERRORS:
        lines.append(f"Plugin load errors   : {_PLUGIN_ERRORS}")
    if _CUDA_DLLS_ERROR:
        lines.append(f"CUDA preload error   : {_CUDA_DLLS_ERROR}")
    if cuda_info.get("declared"):
        probe = probe_cuda_runtime(refresh=False)
        lines.append("CUDA runtime preflight:")
        lines.append(
            f"  Expected CUDA major: {probe.get('expected_cuda_major') or 'unknown'} | "
            f"status={cuda_info.get('status')} | provider_present={probe.get('provider_present')}"
        )
        if probe.get("missing_libs"):
            lines.append(f"  Missing libraries  : {', '.join(probe['missing_libs'])}")
        if probe.get("detected_other_cuda_major"):
            lines.append(
                f"  System mismatch    : CUDA {probe['detected_other_cuda_major']} detected, "
                f"but CUDA {probe.get('expected_cuda_major')} required"
            )
        if probe.get("hint"):
            lines.append(f"  Hint               : {probe['hint']}")
    if ep_devices:
        lines.append("Discovered EP devices:")
        for idx, dev in enumerate(ep_devices):
            hw = getattr(dev, "device", None)
            meta = getattr(dev, "ep_metadata", None) or {}
            vid = getattr(hw, "vendor_id", None) if hw is not None else None
            vid_str = (
                f", vid={hex(vid)}"
                if isinstance(vid, int) and not isinstance(vid, bool)
                else ""
            )
            hw_desc = (
                f"type={getattr(hw, 'type', '?')}, vendor={getattr(hw, 'vendor', '?')}{vid_str}, id={getattr(hw, 'device_id', '?')}"
                if hw is not None
                else "no-hw-info"
            )
            lines.append(f"  [{idx}] ep={dev.ep_name} | {hw_desc} | meta={meta}")
    lines.append("Backend summary:")
    for backend in AUTO_PRIORITY:
        info = availability[backend]
        raw_status = info.get("status")
        if raw_status in status_labels:
            status = status_labels[raw_status]
        else:
            status = "READY (verified)" if info["available"] else "unavailable"
        lines.append(f"  - {backend:9s}: {status:26s} ({info['mode']}) {info['detail']}")
    lines.append(f"Selected by '--device auto': {auto_choice} ({_BACKEND_DISPLAY_NAMES.get(auto_choice, auto_choice)})")
    return "\n".join(lines)
